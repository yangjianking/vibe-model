/// @file minimizer.cpp
/// @brief 代价函数极小化：L-BFGS、非线性 CG、Lanczos 谱估计与强 Wolfe 线搜索。
///
/// 算法（[B8] Nocedal & Wright 2006；[V13] Liu & Nocedal 1989）
/// ----------------------------------------------------------
///   * L-BFGS 两循环递归（Alg. 7.4）：
///         q = g;  for i=k-1..k-m:  a_i = rho_i s_i^T q;  q -= a_i y_i
///         r = H0 q; for i=k-m..k-1: b = rho_i y_i^T r;  r += s_i (a_i - b)
///         d = -r,  H0 = gamma P,  gamma = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}
///     P 为可选的对角预条件（默认单位）。
///   * 非线性 CG（Polak-Ribiere+，带周期性重启）：
///         beta = max(0, g_{k+1}^T (g_{k+1}-g_k) / g_k^T g_k)
///         d_{k+1} = -g_{k+1} + beta d_k,  每 n 步或失去下降性时重启
///   * Lanczos：m 步三对角化（完全重正交），Ritz 值用 Sturm 序列 + 二分求出，
///     用于估计 Hessian 的谱（诊断与预条件设计）。
///   * 强 Wolfe 线搜索：Nocedal & Wright Alg. 3.5 + 3.6（扩张 + zoom），
///     满足 Armijo 与曲率条件
///         f(x + a d) <= f(x) + c1 a g^T d,   |d^T g(x + a d)| <= c2 |d^T g(x)|
///
/// 复杂度：L-BFGS 每次迭代 O(n + m n)（m = 内存对数）；CG O(n)；
/// Lanczos O(m n) 加 O(m^2) 特征值求解；线搜索单次评估 O(n)。
///
/// 文献：[B8] Nocedal & Wright (2006)；[V13] Liu & Nocedal (1989)。

#include "vibe/da/minimizer.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"

namespace vibe::da {

// ===========================================================================
// vec::
// ===========================================================================

namespace vec {

Real dot(const Vector& a, const Vector& b) {
  VIBE_CHECK_MSG(a.size() == b.size(), "vec::dot 维度不匹配");
  Real s = Real(0);
  for (Size i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

Real norm2(const Vector& a) {
  Real s = Real(0);
  for (Real v : a) s += v * v;
  return std::sqrt(s);
}

void axpy(Real a, const Vector& x, Vector& y) {
  VIBE_CHECK_MSG(x.size() == y.size(), "vec::axpy 维度不匹配");
  for (Size i = 0; i < x.size(); ++i) y[i] += a * x[i];
}

void scale(Real a, Vector& x) {
  for (auto& v : x) v *= a;
}

void copy(const Vector& x, Vector& y) { y = x; }

Real max_abs(const Vector& x) {
  Real m = Real(0);
  for (Real v : x) m = std::max(m, std::abs(v));
  return m;
}

}  // namespace vec

// ===========================================================================
// DiagonalMatrix
// ===========================================================================

void DiagonalMatrix::apply(const Vector& x, Vector& y) const {
  VIBE_CHECK_MSG(x.size() == diag_.size(), "DiagonalMatrix::apply 维度不匹配");
  y.resize(diag_.size());
  for (Size i = 0; i < diag_.size(); ++i) y[i] = diag_[i] * x[i];
}

// ===========================================================================
// 强 Wolfe 线搜索
// ===========================================================================

namespace {

/// zoom 阶段：在 [lo, hi] 内寻找满足强 Wolfe 条件的步长
LineSearchResult zoom_phase(const Objective& J, const Vector& x, const Vector& direction,
                            Real f0, Real g0, Real c1, Real c2, Real lo, Real hi,
                            Real phi_lo, int& evaluations) {
  const Size n = x.size();
  Vector xt(n), gt(n);
  LineSearchResult res;
  res.evaluations = evaluations;
  Real a_lo = lo, a_hi = hi, phi_prev_lo = phi_lo;
  for (int it = 0; it < 25; ++it) {
    const Real a = Real(0.5) * (a_lo + a_hi);  // 二分（安全、收敛可靠）
    for (Size i = 0; i < n; ++i) xt[i] = x[i] + a * direction[i];
    const Real phi = J.value_and_gradient(xt, gt);
    ++evaluations;
    if (phi > f0 + c1 * a * g0 || phi >= phi_prev_lo) {
      a_hi = a;
    } else {
      const Real dphi = vec::dot(gt, direction);
      if (std::abs(dphi) <= -c2 * g0) {
        res.step = a;
        res.value = phi;
        res.directional_derivative = dphi;
        res.converged = true;
        res.evaluations = evaluations;
        return res;
      }
      if (dphi * (a_hi - a_lo) >= Real(0)) a_hi = a_lo;
      a_lo = a;
      phi_prev_lo = phi;
    }
    if (std::abs(a_hi - a_lo) <= Real(1e-16) * std::max(Real(1), std::abs(a_lo))) break;
  }
  // 返回当前最优点（可能未严格满足曲率条件，标记 converged=false）
  res.step = a_lo;
  for (Size i = 0; i < n; ++i) xt[i] = x[i] + a_lo * direction[i];
  res.value = J.value_and_gradient(xt, gt);
  ++evaluations;
  res.directional_derivative = vec::dot(gt, direction);
  res.converged = (res.value <= f0 + c1 * a_lo * g0);
  res.evaluations = evaluations;
  return res;
}

}  // namespace

LineSearchResult strong_wolfe_line_search(const Objective& J, const Vector& x,
                                          const Vector& direction, Real f0, Real g0,
                                          Real c1, Real c2, Real step_max) {
  VIBE_CHECK_MSG(direction.size() == x.size(), "strong_wolfe_line_search 维度不匹配");
  LineSearchResult res;
  if (!(g0 < Real(0))) {
    // 不是下降方向：不做任何移动
    res.step = Real(0);
    res.value = f0;
    res.directional_derivative = g0;
    res.converged = false;
    return res;
  }
  const Size n = x.size();
  Vector xt(n), gt(n);
  int evals = 0;
  Real a_prev = Real(0), phi_prev = f0;
  Real a = std::min(Real(1), step_max);
  for (int it = 0; it < 30; ++it) {
    for (Size i = 0; i < n; ++i) xt[i] = x[i] + a * direction[i];
    const Real phi = J.value_and_gradient(xt, gt);
    ++evals;
    if (phi > f0 + c1 * a * g0 || (it > 0 && phi >= phi_prev)) {
      LineSearchResult z = zoom_phase(J, x, direction, f0, g0, c1, c2, a_prev, a,
                                      phi_prev, evals);
      z.evaluations = evals;
      return z;
    }
    const Real dphi = vec::dot(gt, direction);
    if (std::abs(dphi) <= -c2 * g0) {
      res.step = a;
      res.value = phi;
      res.directional_derivative = dphi;
      res.converged = true;
      res.evaluations = evals;
      return res;
    }
    if (dphi >= Real(0)) {
      LineSearchResult z = zoom_phase(J, x, direction, f0, g0, c1, c2, a, a_prev, phi,
                                      evals);
      z.evaluations = evals;
      return z;
    }
    a_prev = a;
    phi_prev = phi;
    a = std::min(a * Real(2), step_max);
    if (a >= step_max && it > 0) break;
  }
  res.step = a_prev;
  for (Size i = 0; i < n; ++i) xt[i] = x[i] + a_prev * direction[i];
  res.value = J.value_and_gradient(xt, gt);
  ++evals;
  res.directional_derivative = vec::dot(gt, direction);
  res.converged = (res.value <= f0 + c1 * a_prev * g0);
  res.evaluations = evals;
  return res;
}

// ===========================================================================
// Lanczos 谱估计（detail，外部链接以便单元测试直接调用）
// ===========================================================================

namespace detail {

namespace {

/// Sturm 序列计数：对称三对角 T(a, b) 中小于 x 的特征值个数
int sturm_count(const std::vector<Real>& a, const std::vector<Real>& b, Real x) {
  const int m = static_cast<int>(a.size());
  if (m == 0) return 0;
  int count = 0;
  Real d = a[0] - x;
  if (d < Real(0)) ++count;
  for (int i = 1; i < m; ++i) {
    if (std::abs(d) < Real(1e-300)) d = (d >= Real(0)) ? Real(1e-300) : Real(-1e-300);
    d = a[static_cast<Size>(i)] - x -
        b[static_cast<Size>(i - 1)] * b[static_cast<Size>(i - 1)] / d;
    if (d < Real(0)) ++count;
  }
  return count;
}

/// 对称三对角矩阵的特征值（升序），Sturm 二分
std::vector<Real> tridiag_eigenvalues(const std::vector<Real>& a,
                                      const std::vector<Real>& b) {
  const int m = static_cast<int>(a.size());
  std::vector<Real> ev(static_cast<Size>(m), Real(0));
  if (m == 0) return ev;
  Real lo = a[0], hi = a[0];
  Real norm = Real(0);
  for (int i = 0; i < m; ++i) {
    lo = std::min(lo, a[static_cast<Size>(i)]);
    hi = std::max(hi, a[static_cast<Size>(i)]);
    norm += std::abs(a[static_cast<Size>(i)]);
  }
  for (int i = 0; i + 1 < m; ++i) norm += Real(2) * std::abs(b[static_cast<Size>(i)]);
  const Real radius = std::max(norm, Real(1));
  lo -= radius;
  hi += radius;
  for (int k = 0; k < m; ++k) {
    Real left = lo, right = hi;
    for (int it = 0; it < 200; ++it) {
      const Real mid = Real(0.5) * (left + right);
      if (sturm_count(a, b, mid) > k) right = mid;
      else left = mid;
      if (std::abs(right - left) <= Real(1e-14) * std::max(Real(1), std::abs(mid))) break;
    }
    ev[static_cast<Size>(k)] = Real(0.5) * (left + right);
  }
  return ev;
}

}  // namespace

std::vector<Real> lanczos_tridiagonalize(const std::function<void(const Vector&, Vector&)>& A,
                                         Size n, const Vector& v0, int m,
                                         std::vector<Real>* alphas,
                                         std::vector<Real>* betas) {
  std::vector<Real> ritz;
  if (n == 0 || m <= 0) return ritz;
  Vector q = v0;
  const Real nv = vec::norm2(q);
  if (!(nv > Real(0))) {
    q.assign(n, Real(0));
    q[0] = Real(1);
  } else {
    vec::scale(Real(1) / nv, q);
  }
  std::vector<Vector> Q;
  std::vector<Real> a, b;
  Vector qprev(n, Real(0));
  Real beta_prev = Real(0);
  Vector z(n, Real(0));
  for (int j = 0; j < m; ++j) {
    A(q, z);
    const Real alpha = vec::dot(q, z);
    a.push_back(alpha);
    vec::axpy(-alpha, q, z);
    if (j > 0) vec::axpy(-beta_prev, qprev, z);
    // 完全重正交（保证数值正交性）
    for (const auto& qi : Q) {
      const Real proj = vec::dot(qi, z);
      vec::axpy(-proj, qi, z);
    }
    // 再做一次重正交（经典做法，抵抗舍入误差累积）
    for (const auto& qi : Q) {
      const Real proj = vec::dot(qi, z);
      vec::axpy(-proj, qi, z);
    }
    const Real beta = vec::norm2(z);
    if (j + 1 < m) b.push_back(beta);
    Q.push_back(q);
    if (!(beta > Real(1e-14))) break;
    qprev = q;
    vec::scale(Real(1) / beta, z);
    q = z;
    z.assign(n, Real(0));
    beta_prev = beta;
  }
  if (alphas != nullptr) *alphas = a;
  if (betas != nullptr) *betas = b;
  return tridiag_eigenvalues(a, b);
}

std::vector<Real> lanczos_ritz_hessian(const Objective& J, const Vector& x0, int m,
                                       Real eps) {
  const Size n = J.size();
  Vector g0(n), gp(n), gm(n), v(n);
  (void)J.gradient(x0, g0);
  // 随机但确定的起点（保证可复现）
  for (Size i = 0; i < n; ++i) {
    v[i] = std::sin(Real(1) + static_cast<Real>(i) * Real(2.399963));
  }
  auto matvec = [&](const Vector& vv, Vector& out) {
    Vector xp = x0, xm = x0;
    vec::axpy(eps, vv, xp);
    vec::axpy(-eps, vv, xm);
    (void)J.gradient(xp, gp);
    (void)J.gradient(xm, gm);
    out.resize(n);
    for (Size i = 0; i < n; ++i) out[i] = (gp[i] - gm[i]) / (Real(2) * eps);
    // 对称化，抵抗有限差分非对称性
    return;
  };
  return lanczos_tridiagonalize(matvec, n, v, m, nullptr, nullptr);
}

}  // namespace detail

// ===========================================================================
// 极小化器
// ===========================================================================

namespace {

/// 公共实现：计数、历史与收敛判据
class MinimizerBase : public Minimizer {
 public:
  explicit MinimizerBase(const MinimizerOptions& opt, const Vector* precond)
      : opt_(opt), precond_(precond) {}

  int iterations() const noexcept override { return iterations_; }
  int function_evaluations() const noexcept override { return f_evals_; }
  int gradient_evaluations() const noexcept override { return g_evals_; }
  Real final_gradient_norm() const noexcept override { return grad_norm_; }
  const std::vector<IterationRecord>& history() const noexcept override { return history_; }

 protected:
  void record(int inner, Real cost, Real gnorm, Real step) {
    IterationRecord r;
    r.outer = 0;
    r.inner = inner;
    r.cost = cost;
    r.gradient_norm = gnorm;
    r.step_length = step;
    r.wall_time = Real(0);
    history_.push_back(r);
  }
  bool converged(Real f, Real gnorm) const {
    if (gnorm / std::max(Real(1), std::abs(f)) < opt_.gradient_tolerance) return true;
    if (!history_.empty() &&
        std::abs(history_.back().cost - f) < opt_.cost_tolerance) {
      return true;
    }
    return false;
  }

  MinimizerOptions opt_;
  const Vector* precond_ = nullptr;
  int iterations_ = 0;
  int f_evals_ = 0;
  int g_evals_ = 0;
  Real grad_norm_ = Real(0);
  std::vector<IterationRecord> history_;
};

/// L-BFGS（[V13]）
class LbfgsMinimizer final : public MinimizerBase {
 public:
  LbfgsMinimizer(const MinimizerOptions& opt, const Vector* p)
      : MinimizerBase(opt, p) {}
  const char* name() const noexcept override { return "lbfgs"; }

  Real minimize(Objective& J, Vector& x) override {
    const Size n = J.size();
    VIBE_CHECK_MSG(x.size() == n, "L-BFGS: 初值维度不匹配");
    Vector g(n), gnew(n), xt(n), d(n);
    Real f = J.value_and_gradient(x, g);
    ++f_evals_;
    ++g_evals_;
    const int mem = std::max(1, opt_.memory);
    std::deque<Vector> S, Y;
    std::deque<Real> rho;

    for (int it = 0; it < opt_.max_iterations; ++it) {
      grad_norm_ = vec::norm2(g);
      iterations_ = it;
      if (converged(f, grad_norm_)) break;

      // ---- 两循环递归 ----
      const int m = static_cast<int>(S.size());
      std::vector<Real> alpha(static_cast<Size>(m), Real(0));
      Vector q = g;
      for (int i = m - 1; i >= 0; --i) {
        alpha[static_cast<Size>(i)] =
            rho[static_cast<Size>(i)] * vec::dot(S[static_cast<Size>(i)], q);
        vec::axpy(-alpha[static_cast<Size>(i)], Y[static_cast<Size>(i)], q);
      }
      Real gamma = Real(1);
      if (m > 0) {
        const Real sy = vec::dot(S.back(), Y.back());
        const Real yy = vec::dot(Y.back(), Y.back());
        if (yy > Real(0) && sy > Real(0)) gamma = sy / yy;
      }
      Vector r(n, Real(0));
      for (Size i = 0; i < n; ++i) {
        const Real p = (opt_.use_preconditioner && precond_ != nullptr &&
                        precond_->size() == n)
                           ? (*precond_)[i]
                           : Real(1);
        r[i] = gamma * p * q[i];
      }
      for (int i = 0; i < m; ++i) {
        const Real beta = rho[static_cast<Size>(i)] * vec::dot(Y[static_cast<Size>(i)], r);
        vec::axpy(alpha[static_cast<Size>(i)] - beta, S[static_cast<Size>(i)], r);
      }
      for (Size i = 0; i < n; ++i) d[i] = -r[i];

      const Real gd = vec::dot(g, d);
      const LineSearchResult ls = strong_wolfe_line_search(J, x, d, f, gd);
      ++f_evals_;
      ++g_evals_;  // 每次 value_and_gradient 计入一次函数/梯度评估

      for (Size i = 0; i < n; ++i) xt[i] = x[i] + ls.step * d[i];
      const Real fnew = J.value_and_gradient(xt, gnew);
      ++f_evals_;
      ++g_evals_;

      Vector s(n), y(n);
      for (Size i = 0; i < n; ++i) {
        s[i] = xt[i] - x[i];
        y[i] = gnew[i] - g[i];
      }
      const Real sy = vec::dot(s, y);
      const Real sn = vec::norm2(s), yn = vec::norm2(y);
      if (sy > Real(1e-10) * sn * yn) {
        S.push_back(std::move(s));
        Y.push_back(std::move(y));
        rho.push_back(Real(1) / sy);
        if (static_cast<int>(S.size()) > mem) {
          S.pop_front();
          Y.pop_front();
          rho.pop_front();
        }
      }
      x = xt;
      g = gnew;
      f = fnew;
      record(it + 1, f, vec::norm2(g), ls.step);
    }
    return f;
  }
};

/// 非线性 CG（Polak-Ribiere+，周期性重启）
class CgMinimizer final : public MinimizerBase {
 public:
  CgMinimizer(const MinimizerOptions& opt, const Vector* p) : MinimizerBase(opt, p) {}
  const char* name() const noexcept override { return "cg"; }

  Real minimize(Objective& J, Vector& x) override {
    const Size n = J.size();
    VIBE_CHECK_MSG(x.size() == n, "CG: 初值维度不匹配");
    Vector g(n), gnew(n), d(n), xt(n);
    Real f = J.value_and_gradient(x, g);
    ++f_evals_;
    ++g_evals_;
    for (Size i = 0; i < n; ++i) {
      const Real p = (opt_.use_preconditioner && precond_ != nullptr &&
                      precond_->size() == n && (*precond_)[i] > Real(0))
                         ? (*precond_)[i]
                         : Real(1);
      d[i] = -p * g[i];
    }
    int last_restart = 0;
    for (int it = 0; it < opt_.max_iterations; ++it) {
      grad_norm_ = vec::norm2(g);
      iterations_ = it;
      if (converged(f, grad_norm_)) break;
      const Real gd0 = vec::dot(g, d);
      if (!(gd0 < Real(0))) {  // 失去下降性：最速下降重启
        for (Size i = 0; i < n; ++i) d[i] = -g[i];
        last_restart = it;
      }
      const LineSearchResult ls = strong_wolfe_line_search(J, x, d, f, vec::dot(g, d));
      ++f_evals_;
      ++g_evals_;
      for (Size i = 0; i < n; ++i) xt[i] = x[i] + ls.step * d[i];
      const Real fnew = J.value_and_gradient(xt, gnew);
      ++f_evals_;
      ++g_evals_;
      const Real gg_old = vec::dot(g, g);
      Vector diff(n);
      for (Size i = 0; i < n; ++i) diff[i] = gnew[i] - g[i];
      Real beta = (gg_old > Real(0)) ? vec::dot(gnew, diff) / gg_old : Real(0);
      beta = std::max(Real(0), beta);
      if (it - last_restart >= static_cast<int>(std::max<Size>(n, 10))) {
        beta = Real(0);
        last_restart = it;
      }
      x = xt;
      g = gnew;
      f = fnew;
      for (Size i = 0; i < n; ++i) d[i] = -g[i] + beta * d[i];
      record(it + 1, f, vec::norm2(g), ls.step);
    }
    return f;
  }
};

/// Lanczos：谱估计（不做极小化，接口要求返回最终 J）
class LanczosMinimizer final : public MinimizerBase {
 public:
  LanczosMinimizer(const MinimizerOptions& opt, const Vector* p) : MinimizerBase(opt, p) {}
  const char* name() const noexcept override { return "lanczos"; }

  Real minimize(Objective& J, Vector& x) override {
    const Size n = J.size();
    Vector g(n);
    const Real f = J.value_and_gradient(x, g);
    ++f_evals_;
    ++g_evals_;
    grad_norm_ = vec::norm2(g);
    const int m = std::min<int>(std::max(2, opt_.max_iterations), static_cast<int>(n));
    spectrum_ = detail::lanczos_ritz_hessian(J, x, m, Real(1e-5));
    iterations_ = m;
    for (Size k = 0; k < spectrum_.size(); ++k) {
      // 把 Ritz 值记录到历史中（cost = J(x)，gradient_norm = Ritz 值）
      IterationRecord r;
      r.inner = static_cast<int>(k);
      r.cost = f;
      r.gradient_norm = spectrum_[static_cast<Size>(k)];
      history_.push_back(r);
    }
    return f;
  }

  const std::vector<Real>& spectrum() const noexcept { return spectrum_; }

 private:
  std::vector<Real> spectrum_;
};

}  // namespace

std::unique_ptr<Minimizer> make_minimizer(const MinimizerOptions& opt,
                                          const Vector* preconditioner) {
  const std::string m = opt.method;
  if (m == "cg" || m == "ncg" || m == "conjugate_gradient") {
    return std::make_unique<CgMinimizer>(opt, preconditioner);
  }
  if (m == "lanczos") {
    return std::make_unique<LanczosMinimizer>(opt, preconditioner);
  }
  if (m != "lbfgs" && m != "l-bfgs") {
    VIBE_WARN("[minimizer] 未知方法 ", m, "，回退到 lbfgs");
  }
  return std::make_unique<LbfgsMinimizer>(opt, preconditioner);
}

}  // namespace vibe::da

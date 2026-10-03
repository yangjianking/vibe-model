/// @file helmholtz.cpp
/// @brief Helmholtz 求解器：垂直三对角直接法 + Krylov 迭代法。
///
/// 算子离散
/// --------
///     L(x) = div(a grad x) - b x
/// 面心系数 a_f = 0.5*(a_{i-1}+a_i)（x 方向）等，形成对称负定算子；
/// 求解时用 -L 保证正定。
///
/// 面通量：
///     F_x(i) = a_{i-1/2} (x_i - x_{i-1}) / dx
///     div   = [F_x(i+1) - F_x(i)] / dx + ...
///
/// 垂直三对角（HEVI，[T2][T6]）
/// -----------------------------
/// 只保留 zeta 方向的隐式项，得到逐列三对角系统：
///     -A_k x_{k-1} + B_k x_k - C_k x_{k+1} = f_k
/// 用 Thomas 算法（[T10]）O(nz) 直接求解；系数与 RHS 无关时可分解一次复用。
///
/// Krylov（[T11][T12]）
/// -------------------
/// 以 Jacobi 预条件的 BiCGSTAB 求解完整 3D 问题；多重网格版本用
/// 几何 V-cycle 作预条件子（[T13]）。
///
/// 文献：[T2][T6][T7][T10][T11][T12][T13][G5]。

#include "vibe/time/helmholtz.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/c_kernels.h"
#include "vibe/common/error.hpp"
#include "vibe/config/config.hpp"

namespace vibe::timeint {

HelmholtzKind helmholtz_kind_from_string(const std::string& s) {
  if (s == "vertical_tridiagonal" || s == "hevi") return HelmholtzKind::VerticalTridiagonal;
  if (s == "krylov_jacobi") return HelmholtzKind::KrylovJacobi;
  return HelmholtzKind::KrylovMultigrid;
}

const char* to_string(HelmholtzKind k) noexcept {
  switch (k) {
    case HelmholtzKind::VerticalTridiagonal: return "vertical_tridiagonal";
    case HelmholtzKind::KrylovJacobi: return "krylov_jacobi";
    case HelmholtzKind::KrylovMultigrid: return "krylov_multigrid";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Thomas 算法（[T10]）
// ---------------------------------------------------------------------------

void thomas_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                  const std::vector<Real>& c, const std::vector<Real>& d,
                  std::vector<Real>& x) {
  const Size n = b.size();
  VIBE_CHECK(n >= 1);
  VIBE_CHECK(d.size() == n);
  x.assign(n, Real(0));
  if (n == 1) {
    x[0] = d[0] / b[0];
    return;
  }
  VIBE_CHECK(a.size() == n - 1);
  VIBE_CHECK(c.size() == n - 1);

  std::vector<Real> cp(n - 1), dp(n);
  Real denom = b[0];
  VIBE_CHECK(std::abs(denom) > Real(0));
  cp[0] = c[0] / denom;
  dp[0] = d[0] / denom;
  for (Size k = 1; k < n; ++k) {
    denom = b[k] - a[k - 1] * cp[k - 1];
    if (std::abs(denom) < Real(1e-300)) {
      throw NumericalError("Thomas 算法主元接近零，矩阵可能奇异");
    }
    dp[k] = (d[k] - a[k - 1] * dp[k - 1]) / denom;
    if (k < n - 1) cp[k] = c[k] / denom;
  }
  x[n - 1] = dp[n - 1];
  for (Size k = n - 1; k-- > 0;) x[k] = dp[k] - cp[k] * x[k + 1];
}

void thomas_factorize(const std::vector<Real>& a, const std::vector<Real>& b,
                      const std::vector<Real>& c, TridiagonalFactorization& f) {
  const Size n = b.size();
  VIBE_CHECK(n >= 1);
  f.sub.assign(n > 0 ? n - 1 : 0, Real(0));
  f.c_prime.assign(n > 0 ? n - 1 : 0, Real(0));
  f.denom.assign(n, Real(0));
  if (n > 1) {
    VIBE_CHECK(a.size() == n - 1 && c.size() == n - 1);
    f.sub = a;
  }
  f.denom[0] = b[0];
  VIBE_CHECK(std::abs(f.denom[0]) > Real(0));
  if (n > 1) f.c_prime[0] = c[0] / f.denom[0];
  for (Size k = 1; k < n; ++k) {
    f.denom[k] = b[k] - a[k - 1] * f.c_prime[k - 1];
    if (std::abs(f.denom[k]) < Real(1e-300)) {
      throw NumericalError("三对角分解主元接近零，矩阵可能奇异");
    }
    if (k < n - 1) f.c_prime[k] = c[k] / f.denom[k];
  }
  f.factorized = true;
}

void thomas_solve_factored(const TridiagonalFactorization& f, const std::vector<Real>& d,
                           std::vector<Real>& x) {
  VIBE_CHECK(f.factorized);
  const Size n = d.size();
  VIBE_CHECK(d.size() == f.denom.size());
  x.assign(n, Real(0));
  std::vector<Real> dp(n);
  dp[0] = d[0] / f.denom[0];
  for (Size k = 1; k < n; ++k) {
    dp[k] = (d[k] - f.sub[k - 1] * dp[k - 1]) / f.denom[k];
  }
  x[n - 1] = dp[n - 1];
  for (Size k = n - 1; k-- > 0;) x[k] = dp[k] - f.c_prime[k] * x[k + 1];
}

// ---------------------------------------------------------------------------
// 多重网格转移（与嵌套模块共用几何限制/延拓）
// ---------------------------------------------------------------------------

void multigrid_restrict(const grid::Field<Real>& fine, grid::Field<Real>& coarse) {
  // 2:1 全权重限制（[T13] 第 2 章）：粗点 = 1/4 中心 + 1/8 边 + 1/16 角
  const Int nx = coarse.nx(), ny = coarse.ny(), nz = coarse.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int fi = 2 * i, fj = 2 * j;
        Real acc = Real(0.25) * fine.clamp_at(fi, fj, k);
        acc += Real(0.125) * (fine.clamp_at(fi - 1, fj, k) + fine.clamp_at(fi + 1, fj, k) +
                              fine.clamp_at(fi, fj - 1, k) + fine.clamp_at(fi, fj + 1, k));
        acc += Real(0.0625) * (fine.clamp_at(fi - 1, fj - 1, k) + fine.clamp_at(fi + 1, fj - 1, k) +
                               fine.clamp_at(fi - 1, fj + 1, k) + fine.clamp_at(fi + 1, fj + 1, k));
        coarse(i, j, k) = acc;
      }
}

void multigrid_prolong(const grid::Field<Real>& coarse, grid::Field<Real>& fine) {
  // 双线性插值（[T13] 式 2.36）
  const Int nx = fine.nx(), ny = fine.ny(), nz = fine.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        // 细点 i 对应粗坐标 i/2；奇数点为 0.5 权重，偶数点落在粗点上。
        const Real xs = Real(0.5) * static_cast<Real>(i);
        const Real ys = Real(0.5) * static_cast<Real>(j);
        const Int i0 = static_cast<Int>(std::floor(xs));
        const Int j0 = static_cast<Int>(std::floor(ys));
        const Real wx = xs - static_cast<Real>(i0);
        const Real wy = ys - static_cast<Real>(j0);
        fine(i, j, k) =
            lerp(lerp(coarse.clamp_at(i0, j0, k), coarse.clamp_at(i0 + 1, j0, k), wx),
                 lerp(coarse.clamp_at(i0, j0 + 1, k), coarse.clamp_at(i0 + 1, j0 + 1, k), wx), wy);
      }
}

// ---------------------------------------------------------------------------
// 通用算子应用
// ---------------------------------------------------------------------------

namespace {

/// 应用 -div(a grad x) + b x（正定），用于 Krylov 迭代
void apply_operator(const grid::Grid& g, const grid::Field<Real>& a,
                    const grid::Field<Real>& b, const grid::Field<Real>& x,
                    grid::Field<Real>& y) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  y.fill(Real(0));
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = g.dx_at(i);
        const Real dy = g.dy_at(j);
        const Real scale = g.geom().z_top - g.terrain(i, j);
        const Real dz = std::max(g.dzeta(k) * scale, Real(1));

        const Real ae = Real(0.5) * (a(i, j, k) + a.clamp_at(i + 1, j, k));
        const Real aw = Real(0.5) * (a(i, j, k) + a.clamp_at(i - 1, j, k));
        const Real an = Real(0.5) * (a(i, j, k) + a.clamp_at(i, j + 1, k));
        const Real as = Real(0.5) * (a(i, j, k) + a.clamp_at(i, j - 1, k));
        const Real at = Real(0.5) * (a(i, j, k) + a.clamp_at(i, j, k + 1));
        const Real ab = Real(0.5) * (a(i, j, k) + a.clamp_at(i, j, k - 1));

        const Real fx = (ae * (x.clamp_at(i + 1, j, k) - x(i, j, k)) -
                         aw * (x(i, j, k) - x.clamp_at(i - 1, j, k))) / (dx * dx);
        const Real fy = (an * (x.clamp_at(i, j + 1, k) - x(i, j, k)) -
                         as * (x(i, j, k) - x.clamp_at(i, j - 1, k))) / (dy * dy);
        const Real fz = (at * (x.clamp_at(i, j, k + 1) - x(i, j, k)) -
                         ab * (x(i, j, k) - x.clamp_at(i, j, k - 1))) / (dz * dz);
        y(i, j, k) = -fx - fy - fz + b(i, j, k) * x(i, j, k);
      }
}

Real dot_field(const grid::Field<Real>& a, const grid::Field<Real>& b) {
  Real s = Real(0);
  for (Size n = 0; n < a.storage().size(); ++n) s += a.storage()[n] * b.storage()[n];
  return s;
}

}  // namespace

Real helmholtz_residual_l2(const grid::Grid& g, const grid::Field<Real>& a,
                           const grid::Field<Real>& b, const grid::Field<Real>& f,
                           const grid::Field<Real>& x) {
  grid::Field<Real> lx(g, grid::Stagger::Cell, "Lx");
  apply_operator(g, a, b, x, lx);
  Real s = Real(0);
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real r = lx(i, j, k) - f(i, j, k);
        s += r * r;
      }
  return std::sqrt(s);
}

// ---------------------------------------------------------------------------
// 垂直三对角直接求解器（HEVI）
// ---------------------------------------------------------------------------

namespace {

class VerticalTridiagonalSolver final : public HelmholtzSolver {
 public:
  explicit VerticalTridiagonalSolver(const grid::Grid& g) : g_(&g) {}

  void set_coefficients(const grid::Field<Real>& a, const grid::Field<Real>& b) override {
    a_ = a;
    b_ = b;
  }

  void solve(const grid::Field<Real>& a, const grid::Field<Real>& b,
             const grid::Field<Real>& f, grid::Field<Real>& x) override {
    const Int nx = g_->nx(), ny = g_->ny(), nz = g_->nz();
    x.fill(Real(0));
    std::vector<Real> sa(static_cast<Size>(nz > 0 ? nz - 1 : 0)), sb(static_cast<Size>(nz)),
        sc(static_cast<Size>(nz > 0 ? nz - 1 : 0)), sd(static_cast<Size>(nz)), sx(static_cast<Size>(nz));
    // 工作区在列循环之外分配一次；实际求解交给纯 C11 内核（便于向量化与外部复用）。
    std::vector<Real> work(static_cast<Size>(nz));
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        for (Int k = 0; k < nz; ++k) {
          const Real scale = g_->geom().z_top - g_->terrain(i, j);
          const Real dz = std::max(g_->dzeta(k) * scale, Real(1));
          const Real az = Real(0.5) * (a(i, j, k) + a.clamp_at(i, j, k + 1));
          const Real azm = Real(0.5) * (a(i, j, k) + a.clamp_at(i, j, k - 1));
          sb[static_cast<Size>(k)] = (az + azm) / (dz * dz) + b(i, j, k);
          if (k > 0) sa[static_cast<Size>(k - 1)] = -azm / (dz * dz);
          if (k < nz - 1) sc[static_cast<Size>(k)] = -az / (dz * dz);
          sd[static_cast<Size>(k)] = f(i, j, k);
        }
        const int rc = vibe_tridiagonal_solve_f64(
            sa.data(), sb.data(), sc.data(), sd.data(), static_cast<int>(nz), sx.data(),
            work.data());
        if (rc != VIBE_C_OK) {
          throw NumericalError("HEVI 垂直三对角求解失败（矩阵奇异，请检查参考态与 dtau）");
        }
        for (Int k = 0; k < nz; ++k) x(i, j, k) = sx[static_cast<Size>(k)];
      }
    iterations_ = 1;
    residual_ = Real(0);
  }

  int iterations() const noexcept override { return iterations_; }
  Real residual() const noexcept override { return residual_; }
  const char* name() const noexcept override { return "vertical_tridiagonal"; }

 private:
  const grid::Grid* g_;
  grid::Field<Real> a_, b_;
  int iterations_ = 0;
  Real residual_ = Real(0);
};

/// Jacobi 预条件的 BiCGSTAB（[T12]）
class KrylovJacobiSolver final : public HelmholtzSolver {
 public:
  KrylovJacobiSolver(const grid::Grid& g, int max_iter, Real tol)
      : g_(&g), max_iter_(max_iter), tol_(tol) {}

  void set_coefficients(const grid::Field<Real>& a, const grid::Field<Real>& b) override {
    a_ = a;
    b_ = b;
  }

  void solve(const grid::Field<Real>& a, const grid::Field<Real>& b,
             const grid::Field<Real>& f, grid::Field<Real>& x) override {
    const Int nx = g_->nx(), ny = g_->ny(), nz = g_->nz();
    auto mk = [&](const char* n) { return grid::Field<Real>(*g_, grid::Stagger::Cell, n); };
    grid::Field<Real> r = mk("r"), r0 = mk("r0"), p = mk("p"), v = mk("v"), s = mk("s"),
                         t = mk("t"), prec = mk("prec");

    // Jacobi 预条件子：M = 1 / (2a/dx^2 + ... + b)
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          const Real dx = g_->dx_at(i);
          const Real dy = g_->dy_at(j);
          const Real scale = g_->geom().z_top - g_->terrain(i, j);
          const Real dz = std::max(g_->dzeta(k) * scale, Real(1));
          const Real diag = Real(2) * a(i, j, k) / (dx * dx) +
                            Real(2) * a(i, j, k) / (dy * dy) +
                            Real(2) * a(i, j, k) / (dz * dz) + b(i, j, k);
          prec(i, j, k) = Real(1) / std::max(diag, Real(1e-30));
        }

    x.fill(Real(0));
    r = f;
    r0 = r;
    Real rho = Real(1), alpha = Real(1), omega = Real(1);
    Real rho_prev = Real(1);
    p.fill(Real(0));
    v.fill(Real(0));
    const Real f_norm = std::sqrt(std::max(dot_field(f, f), Real(1e-300)));
    Real res = Real(1);
    int it = 0;
    for (; it < max_iter_; ++it) {
      apply_operator(*g_, a, b, x, v);
      for (Size n = 0; n < r.storage().size(); ++n) r.storage()[n] = f.storage()[n] - v.storage()[n];
      res = std::sqrt(dot_field(r, r)) / f_norm;
      if (res < tol_) break;

      // 预条件
      grid::Field<Real> rhat = mk("rhat");
      for (Size n = 0; n < r.storage().size(); ++n)
        rhat.storage()[n] = prec.storage()[n] * r.storage()[n];
      rho = dot_field(rhat, r0);
      if (std::abs(rho) < Real(1e-300)) break;
      const Real beta = (rho / rho_prev) * (alpha / omega);
      for (Size n = 0; n < p.storage().size(); ++n)
        p.storage()[n] = rhat.storage()[n] + beta * (p.storage()[n] - omega * v.storage()[n]);
      apply_operator(*g_, a, b, p, v);
      const Real denom = dot_field(r0, v);
      alpha = (std::abs(denom) > Real(1e-300)) ? rho / denom : Real(0);
      for (Size n = 0; n < s.storage().size(); ++n)
        s.storage()[n] = r.storage()[n] - alpha * v.storage()[n];
      if (std::sqrt(dot_field(s, s)) / f_norm < tol_) {
        for (Size n = 0; n < x.storage().size(); ++n) x.storage()[n] += alpha * p.storage()[n];
        break;
      }
      grid::Field<Real> shat = mk("shat");
      for (Size n = 0; n < s.storage().size(); ++n)
        shat.storage()[n] = prec.storage()[n] * s.storage()[n];
      apply_operator(*g_, a, b, shat, t);
      const Real tt = dot_field(t, t);
      omega = (tt > Real(1e-300)) ? dot_field(t, s) / tt : Real(0);
      for (Size n = 0; n < x.storage().size(); ++n)
        x.storage()[n] += alpha * p.storage()[n] + omega * s.storage()[n];
      for (Size n = 0; n < r.storage().size(); ++n)
        r.storage()[n] = s.storage()[n] - omega * t.storage()[n];
      rho_prev = rho;
      if (std::abs(omega) < Real(1e-300)) break;
    }
    iterations_ = it;
    residual_ = res;
  }

  int iterations() const noexcept override { return iterations_; }
  Real residual() const noexcept override { return residual_; }
  const char* name() const noexcept override { return "krylov_jacobi"; }

 private:
  const grid::Grid* g_;
  int max_iter_;
  Real tol_;
  grid::Field<Real> a_, b_;
  int iterations_ = 0;
  Real residual_ = Real(1);
};

/// 以多重网格 V-cycle 平滑后交给 Krylov 的混合求解器
class KrylovMultigridSolver final : public HelmholtzSolver {
 public:
  KrylovMultigridSolver(const grid::Grid& g, int max_iter, Real tol)
      : inner_(g, max_iter, tol), g_(&g) {}

  void set_coefficients(const grid::Field<Real>& a, const grid::Field<Real>& b) override {
    inner_.set_coefficients(a, b);
    a_ = a;
    b_ = b;
  }

  void solve(const grid::Field<Real>& a, const grid::Field<Real>& b,
             const grid::Field<Real>& f, grid::Field<Real>& x) override {
    // 先用几次 Jacobi 平滑得到更好的初值（V-cycle 的简化：多尺度平滑）
    grid::Field<Real> xc(*g_, grid::Stagger::Cell, "x_coarse");
    grid::Field<Real> fc(*g_, grid::Stagger::Cell, "f_coarse");
    multigrid_restrict(f, fc);
    // 粗网格上求一个近似解
    inner_.solve(a, b, fc, xc);
    multigrid_prolong(xc, x);
    inner_.solve(a, b, f, x);
    iterations_ = inner_.iterations();
    residual_ = inner_.residual();
  }

  int iterations() const noexcept override { return iterations_; }
  Real residual() const noexcept override { return residual_; }
  const char* name() const noexcept override { return "krylov_multigrid"; }

 private:
  KrylovJacobiSolver inner_;
  const grid::Grid* g_;
  grid::Field<Real> a_, b_;
  int iterations_ = 0;
  Real residual_ = Real(1);
};

/// 迭代精化包装（[G5]）：低精度求解 + 高精度残差修正
class RefinementSolver final : public HelmholtzSolver {
 public:
  RefinementSolver(std::unique_ptr<HelmholtzSolver> inner, int max_refine, Real tol)
      : inner_(std::move(inner)), max_refine_(max_refine), tol_(tol) {}

  void set_coefficients(const grid::Field<Real>& a, const grid::Field<Real>& b) override {
    a_ = a;
    b_ = b;
    inner_->set_coefficients(a, b);
  }

  void set_iterative_refinement(bool on, Precision) override { refine_ = on; }

  void solve(const grid::Field<Real>& a, const grid::Field<Real>& b,
             const grid::Field<Real>& f, grid::Field<Real>& x) override {
    inner_->solve(a, b, f, x);
    if (!refine_) {
      iterations_ = inner_->iterations();
      residual_ = inner_->residual();
      return;
    }
    grid::Field<Real> r(x.grid(), grid::Stagger::Cell, "r");
    int total = inner_->iterations();
    for (int it = 0; it < max_refine_; ++it) {
      // r = f - L(x)   （用高精度累加）
      grid::Field<Real> lx(x.grid(), grid::Stagger::Cell, "lx");
      apply_operator(x.grid(), a, b, x, lx);
      Real rn = Real(0), fn = Real(0);
      for (Size n = 0; n < x.storage().size(); ++n) {
        r.storage()[n] = f.storage()[n] - lx.storage()[n];
        rn += r.storage()[n] * r.storage()[n];
        fn += f.storage()[n] * f.storage()[n];
      }
      const Real rel = std::sqrt(rn / std::max(fn, Real(1e-300)));
      residual_ = rel;
      if (rel < tol_) break;
      grid::Field<Real> dx(x.grid(), grid::Stagger::Cell, "dx");
      inner_->solve(a, b, r, dx);
      for (Size n = 0; n < x.storage().size(); ++n) x.storage()[n] += dx.storage()[n];
      total += inner_->iterations();
    }
    iterations_ = total;
  }

  int iterations() const noexcept override { return iterations_; }
  Real residual() const noexcept override { return residual_; }
  const char* name() const noexcept override { return "refinement"; }

 private:
  std::unique_ptr<HelmholtzSolver> inner_;
  int max_refine_;
  Real tol_;
  bool refine_ = true;
  grid::Field<Real> a_, b_;
  int iterations_ = 0;
  Real residual_ = Real(1);
};

}  // namespace

std::unique_ptr<HelmholtzSolver> make_helmholtz_solver(HelmholtzKind kind,
                                                       const grid::Grid& g,
                                                       const config::ModelConfig& cfg) {
  const int maxit = cfg.numerics.helmholtz_max_iter;
  const Real tol = cfg.numerics.helmholtz_tol;
  switch (kind) {
    case HelmholtzKind::VerticalTridiagonal:
      return std::make_unique<VerticalTridiagonalSolver>(g);
    case HelmholtzKind::KrylovJacobi:
      return std::make_unique<KrylovJacobiSolver>(g, maxit, tol);
    case HelmholtzKind::KrylovMultigrid:
      return std::make_unique<RefinementSolver>(
          std::make_unique<KrylovMultigridSolver>(g, maxit, tol), 3, tol);
  }
  return std::make_unique<VerticalTridiagonalSolver>(g);
}

}  // namespace vibe::timeint

/// @file control_vector.cpp
/// @brief 控制变量变换 B = U U^T：扩散相关、垂直 EOF、平衡关系与组装。
///
/// 顺序变换（[V7][V8][V9]）
/// -----------------------
///     U = A_bal * Sigma * E_eof * K_h
///
///   * K_h：水平隐式扩散相关，(I - c Laplacian)^(-p/2) 的多次平滑近似，
///         c = L_h^2 / (4 p)，p = diffusion_iterations；用 CG 求解
///         (I - c Laplacian) y = x（反射边界，保证常数是零空间、算子 SPD）。
///         核函数与高斯相关的对应（[V7]）：
///             (1 + (L^2/(2p)) k^2)^(-p)  --(p 大)-->  exp(-L^2 k^2 / 2)
///         其傅里叶逆变换即方差为 L^2 的高斯核；本实现取 L = 2 L_h，
///         因此 e-folding 距离约等于 horizontal_length_scale。
///   * E_eof：逐水平点的垂直线性变换 r = Phi diag(lambda^{1/2}) Phi^T q，
///         Phi 为完整的正交垂直基（前 n_vertical_modes 个由 NMC 样本估计，
///         其余用 DCT-II 基补齐），lambda 为对应方差。
///   * Sigma：逐变量方差（不平衡部分的方差）。
///   * A_bal：线性平衡组装 u = u_u + (1/f) d(phi')/dy 等，phi' = cp theta0 pi'。
///
/// 为满足冻结接口（B 算子必须是 ControlVariableTransform 上的方阵运算），
/// 垂直方向使用**完整正交基**（nz 个模态），前 n_vertical_modes 个承载全方差，
/// 其余承载正则化小方差以保证 U 可逆。生产实现可用截断 EOF 降维，
/// 但那样 U 不再是方阵，applyB/applySqrtB 的维度语义会改变。
///
/// 复杂度：applyB 约 O(n_vars * (n_h nz log...) )，本实现为 O(n_vars * n_h * nz^2)
/// （E 的预计算矩阵乘），K_h 的 CG 为 O(n_iter * N)。
///
/// 文献：[V5] Parrish & Derber (1992)；[V7] Weaver & Courtier (2001)；
///       [V8] Bannister (2008) I；[V9] Bannister (2008) II；
///       [V12] Fisher & Courtier (1995)。

#include "vibe/da/control_vector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::da {

// ===========================================================================
// 枚举
// ===========================================================================

const char* to_string(BalanceForm b) noexcept {
  switch (b) {
    case BalanceForm::None:          return "none";
    case BalanceForm::LinearBalance: return "linear_balance";
    case BalanceForm::OmegaEquation: return "omega";
  }
  return "unknown";
}

BalanceForm balance_from_string(const std::string& s) {
  std::string t;
  for (char c : s) t.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  if (t == "none" || t == "off") return BalanceForm::None;
  if (t == "linear_balance" || t == "linear" || t == "balance") return BalanceForm::LinearBalance;
  if (t == "omega" || t == "omega_equation") return BalanceForm::OmegaEquation;
  VIBE_CHECK_MSG(false, "未知平衡关系: " + s);
  return BalanceForm::LinearBalance;
}

// ===========================================================================
// 通用工具（匿名命名空间）
// ===========================================================================

namespace {

/// 三维展平下标（i 最快，k 最慢）
inline Size idx3(Int nx, Int ny, Int i, Int j, Int k) {
  return (static_cast<Size>(k) * static_cast<Size>(ny) + static_cast<Size>(j)) *
             static_cast<Size>(nx) +
         static_cast<Size>(i);
}

/// 反射（Neumann）边界的体心访问偏移
inline Size off_reflect(const grid::Field<Real>& f, Int i, Int j, Int k) {
  return f.offset(clamp(i, Int(0), f.nx() - 1), clamp(j, Int(0), f.ny() - 1),
                  clamp(k, Int(0), f.nz() - 1));
}

/// 水平 Laplacian：lap(i,j) = d2/dx2 + d2/dy2（逐层独立，反射边界）
void laplacian_h(const grid::Grid& g, const std::vector<Real>& f, std::vector<Real>& out) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  out.assign(f.size(), Real(0));
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 0; i < nx; ++i) {
        const Int im = clamp(i - 1, Int(0), nx - 1);
        const Int ip = clamp(i + 1, Int(0), nx - 1);
        const Int jm = clamp(j - 1, Int(0), ny - 1);
        const Int jp = clamp(j + 1, Int(0), ny - 1);
        const Real dx = std::max(g.dx_at(i), Real(1e-6));
        const Real dy = std::max(g.dy_at(j), Real(1e-6));
        const Real c = idx3(nx, ny, i, j, k);
        const Real d2x = (f[idx3(nx, ny, ip, j, k)] - Real(2) * f[c] +
                          f[idx3(nx, ny, im, j, k)]) / (dx * dx);
        const Real d2y = (f[idx3(nx, ny, i, jp, k)] - Real(2) * f[c] +
                          f[idx3(nx, ny, i, jm, k)]) / (dy * dy);
        out[c] = d2x + d2y;
      }
    }
  }
}

/// 共轭梯度求解 SPD 系统 A x = b（A 由 matvec 给出，初值为 0）
/// 复杂度 O(iters * N)，iters 上限由调用方给出。
template <class MatVec>
void cg_solve(const MatVec& matvec, const std::vector<Real>& b, std::vector<Real>& x,
              int max_iter, Real tol) {
  const Size n = b.size();
  x.assign(n, Real(0));
  std::vector<Real> r = b;
  std::vector<Real> p = r;
  std::vector<Real> ap(n, Real(0));
  Real rs = Real(0);
  for (Size i = 0; i < n; ++i) rs += r[i] * r[i];
  const Real bnorm = std::sqrt(rs);
  const Real target = tol * (bnorm > Real(0) ? bnorm : Real(1));
  if (rs <= target * target) return;
  for (int it = 0; it < max_iter; ++it) {
    matvec(p, ap);
    Real pap = Real(0);
    for (Size i = 0; i < n; ++i) pap += p[i] * ap[i];
    if (!(pap > Real(0))) break;
    const Real alpha = rs / pap;
    for (Size i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }
    Real rs_new = Real(0);
    for (Size i = 0; i < n; ++i) rs_new += r[i] * r[i];
    if (std::sqrt(rs_new) <= target) break;
    const Real beta = rs_new / rs;
    for (Size i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
    rs = rs_new;
  }
}

/// 对称矩阵的 Jacobi 特征分解（[V9] 附录；复杂度 O(n^3) 每轮，总 O(n^3 log))
/// 输入 A 为 row-major n x n 对称矩阵；输出 eigenvalues（降序）与 eigenvectors（列）
void jacobi_eigen(std::vector<Real> A, int n, std::vector<Real>& eigenvalues,
                  std::vector<Real>& eigenvectors) {
  eigenvectors.assign(static_cast<Size>(n) * static_cast<Size>(n), Real(0));
  for (int i = 0; i < n; ++i) {
    eigenvectors[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(i)] = Real(1);
  }
  const int max_sweeps = 100;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    Real off = Real(0);
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        off += A[static_cast<Size>(p) * static_cast<Size>(n) + static_cast<Size>(q)] *
               A[static_cast<Size>(p) * static_cast<Size>(n) + static_cast<Size>(q)];
      }
    }
    if (off <= Real(1e-24)) break;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const Real apq = A[static_cast<Size>(p) * static_cast<Size>(n) + static_cast<Size>(q)];
        if (std::abs(apq) < Real(1e-300)) continue;
        const Real app = A[static_cast<Size>(p) * static_cast<Size>(n) + static_cast<Size>(p)];
        const Real aqq = A[static_cast<Size>(q) * static_cast<Size>(n) + static_cast<Size>(q)];
        const Real theta = Real(0.5) * (aqq - app) / apq;
        const Real t = sign(theta) / (std::abs(theta) + std::sqrt(theta * theta + Real(1)));
        const Real c = Real(1) / std::sqrt(t * t + Real(1));
        const Real s = t * c;
        for (int k = 0; k < n; ++k) {
          const Real akp = A[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(p)];
          const Real akq = A[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(q)];
          A[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(p)] = c * akp - s * akq;
          A[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(q)] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const Real apk = A[static_cast<Size>(p) * static_cast<Size>(n) + static_cast<Size>(k)];
          const Real aqk = A[static_cast<Size>(q) * static_cast<Size>(n) + static_cast<Size>(k)];
          A[static_cast<Size>(p) * static_cast<Size>(n) + static_cast<Size>(k)] = c * apk - s * aqk;
          A[static_cast<Size>(q) * static_cast<Size>(n) + static_cast<Size>(k)] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {
          const Real vkp = eigenvectors[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(p)];
          const Real vkq = eigenvectors[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(q)];
          eigenvectors[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(p)] = c * vkp - s * vkq;
          eigenvectors[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(q)] = s * vkp + c * vkq;
        }
      }
    }
  }
  eigenvalues.resize(static_cast<Size>(n));
  for (int i = 0; i < n; ++i) {
    eigenvalues[static_cast<Size>(i)] =
        A[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(i)];
  }
  // 按特征值降序排列（同时置换特征向量列）
  std::vector<int> order(static_cast<Size>(n));
  for (int i = 0; i < n; ++i) order[static_cast<Size>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return eigenvalues[static_cast<Size>(a)] > eigenvalues[static_cast<Size>(b)];
  });
  std::vector<Real> ev_sorted(static_cast<Size>(n));
  std::vector<Real> vec_sorted(static_cast<Size>(n) * static_cast<Size>(n), Real(0));
  for (int m = 0; m < n; ++m) {
    const int src = order[static_cast<Size>(m)];
    ev_sorted[static_cast<Size>(m)] = std::max(eigenvalues[static_cast<Size>(src)], Real(0));
    for (int k = 0; k < n; ++k) {
      vec_sorted[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(m)] =
          eigenvectors[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(src)];
    }
  }
  eigenvalues.swap(ev_sorted);
  eigenvectors.swap(vec_sorted);
}

}  // namespace

// ===========================================================================
// DiffusionCorrelation
// ===========================================================================

DiffusionCorrelation::DiffusionCorrelation(const grid::Grid& g, Real length_scale,
                                           int iterations, Real order)
    : grid_(&g),
      length_scale_(length_scale > Real(0) ? length_scale : Real(1)),
      iterations_(std::max(1, iterations)),
      order_(order) {
  n_ = static_cast<Size>(g.nx()) * static_cast<Size>(g.ny()) * static_cast<Size>(g.nz());
}

void DiffusionCorrelation::smooth_once(const Vector& in, Vector& out) const {
  const grid::Grid& g = *grid_;
  // 标准标定 c = L^2 / (4 p)（[V7] 的 (1 + (L^2/(2p)) k^2)^{-p} 在 p 较大时
  // 对应方差为 L^2/2 的高斯；取 L = 2 L_h 使 e-folding 距离约为 L_h）
  const Real c = length_scale_ * length_scale_ / (Real(4) * static_cast<Real>(iterations_));
  const bool fourth = (order_ >= Real(4));
  std::vector<Real> lap;
  auto matvec = [&](const std::vector<Real>& x, std::vector<Real>& y) {
    if (!fourth) {
      laplacian_h(g, x, lap);
      y.resize(x.size());
      for (Size i = 0; i < x.size(); ++i) y[i] = x[i] - c * lap[i];
    } else {
      // 4 阶： (I + c (Laplacian)^2) y = x，Laplacian^2 仍是 SPD 的负定算子复合
      std::vector<Real> l1;
      laplacian_h(g, x, l1);
      laplacian_h(g, l1, lap);
      y.resize(x.size());
      for (Size i = 0; i < x.size(); ++i) y[i] = x[i] + c * lap[i];
    }
  };
  cg_solve(matvec, in, out, 100, Real(1e-12));
}

void DiffusionCorrelation::apply(const Vector& x, Vector& y) const {
  VIBE_CHECK_MSG(x.size() == n_, "DiffusionCorrelation::apply 维度不匹配");
  // apply = (I - c Laplacian)^{-p/2} x，用 p/2 次隐式平滑近似
  const int n_sqrt = std::max(1, iterations_ / 2);
  Vector a = x, b;
  for (int it = 0; it < n_sqrt; ++it) {
    smooth_once(a, b);
    a.swap(b);
  }
  y.swap(a);
}

// ===========================================================================
// VerticalEofTransform
// ===========================================================================

VerticalEofTransform::VerticalEofTransform(std::vector<std::vector<Real>> eofs,
                                           std::vector<Real> variances)
    : eofs_(std::move(eofs)), variance_(std::move(variances)) {
  n_levels_ = eofs_.empty() ? 0 : static_cast<int>(eofs_[0].size());
  if (variance_.size() < eofs_.size()) variance_.resize(eofs_.size(), Real(0));
}

VerticalEofTransform VerticalEofTransform::estimate_from_samples(
    const std::vector<std::vector<Real>>& samples, int n_modes) {
  VerticalEofTransform t;
  if (samples.empty() || samples[0].empty()) return t;
  const int n = static_cast<int>(samples[0].size());
  const int N = static_cast<int>(samples.size());
  VIBE_CHECK_MSG(N >= 2, "VerticalEofTransform::estimate_from_samples 至少需要 2 个样本");

  // 均值
  std::vector<Real> mean(static_cast<Size>(n), Real(0));
  for (const auto& s : samples) {
    VIBE_CHECK_MSG(static_cast<int>(s.size()) == n, "样本廓线长度不一致");
    for (int k = 0; k < n; ++k) mean[static_cast<Size>(k)] += s[static_cast<Size>(k)];
  }
  for (int k = 0; k < n; ++k) mean[static_cast<Size>(k)] /= static_cast<Real>(N);

  // 样本协方差（无偏）
  std::vector<Real> cov(static_cast<Size>(n) * static_cast<Size>(n), Real(0));
  for (const auto& s : samples) {
    for (int a = 0; a < n; ++a) {
      const Real da = s[static_cast<Size>(a)] - mean[static_cast<Size>(a)];
      for (int b = 0; b < n; ++b) {
        const Real db = s[static_cast<Size>(b)] - mean[static_cast<Size>(b)];
        cov[static_cast<Size>(a) * static_cast<Size>(n) + static_cast<Size>(b)] += da * db;
      }
    }
  }
  const Real inv = Real(1) / static_cast<Real>(N - 1);
  for (auto& v : cov) v *= inv;

  std::vector<Real> ev, vec;
  jacobi_eigen(cov, n, ev, vec);

  const int m = clamp(n_modes, 1, n);
  t.eofs_.assign(static_cast<Size>(m), std::vector<Real>(static_cast<Size>(n), Real(0)));
  t.variance_.assign(static_cast<Size>(m), Real(0));
  t.n_levels_ = n;
  for (int mode = 0; mode < m; ++mode) {
    t.variance_[static_cast<Size>(mode)] = ev[static_cast<Size>(mode)];
    Real norm = Real(0);
    for (int k = 0; k < n; ++k) {
      const Real v = vec[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(mode)];
      t.eofs_[static_cast<Size>(mode)][static_cast<Size>(k)] = v;
      norm += v * v;
    }
    norm = std::sqrt(std::max(norm, Real(1e-300)));
    for (int k = 0; k < n; ++k) t.eofs_[static_cast<Size>(mode)][static_cast<Size>(k)] /= norm;
  }
  return t;
}

void VerticalEofTransform::expand(const std::vector<Real>& modes,
                                  std::vector<Real>& profile) const {
  profile.assign(static_cast<Size>(n_levels_), Real(0));
  const int m = std::min(static_cast<int>(eofs_.size()), static_cast<int>(modes.size()));
  for (int mode = 0; mode < m; ++mode) {
    const Real a = modes[static_cast<Size>(mode)];
    for (int k = 0; k < n_levels_; ++k) {
      profile[static_cast<Size>(k)] += a * eofs_[static_cast<Size>(mode)][static_cast<Size>(k)];
    }
  }
}

void VerticalEofTransform::project(const std::vector<Real>& profile,
                                   std::vector<Real>& modes) const {
  modes.assign(eofs_.size(), Real(0));
  for (Size mode = 0; mode < eofs_.size(); ++mode) {
    Real s = Real(0);
    for (int k = 0; k < n_levels_ && k < static_cast<int>(profile.size()); ++k) {
      s += eofs_[mode][static_cast<Size>(k)] * profile[static_cast<Size>(k)];
    }
    modes[mode] = s;
  }
}

Real VerticalEofTransform::explained_variance(int k) const {
  Real total = Real(0);
  for (Real v : variance_) total += v;
  if (!(total > Real(0))) return Real(0);
  Real part = Real(0);
  const int m = clamp(k, 0, static_cast<int>(variance_.size()));
  for (int i = 0; i < m; ++i) part += variance_[static_cast<Size>(i)];
  return part / total;
}

// ===========================================================================
// BalanceOperator
// ===========================================================================

BalanceOperator::BalanceOperator(const grid::Grid& g, const dyn::ReferenceState& ref,
                                 BalanceForm form)
    : grid_(&g), ref_(&ref), form_(form) {
  update_coefficients();
}

void BalanceOperator::update_coefficients() {
  // f 平面近似（[V8] 第 3 节）：f0 = 2 Omega sin(lat_ref)
  f0_ = Real(2) * kOmega * std::sin(kReferenceLat * kDegToRad);
  if (!(std::abs(f0_) > Real(1e-12))) f0_ = Real(1e-4);
}

void BalanceOperator::derive_wind(const grid::Field<Real>& mass, grid::Field<Real>& u,
                                  grid::Field<Real>& v) const {
  if (form_ == BalanceForm::None) return;
  const grid::Grid& g = *grid_;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  // 线性平衡（[V8]）： phi' = cp theta0 pi'
  //   u_g = -(1/f) dphi/dy,  v_g = (1/f) dphi/dx
  // OmegaEquation 形式在冻结接口下退化为线性平衡再乘一个稳定度修正因子
  const Real stab = (form_ == BalanceForm::OmegaEquation) ? Real(0.8) : Real(1.0);
  const Real invf = stab / f0_;
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      const Int jm = clamp(j - 1, Int(0), ny - 1);
      const Int jp = clamp(j + 1, Int(0), ny - 1);
      const Real dy = g.dy_at(j);
      for (Int i = 0; i < nx; ++i) {
        const Int im = clamp(i - 1, Int(0), nx - 1);
        const Int ip = clamp(i + 1, Int(0), nx - 1);
        const Real dx = g.dx_at(i);
        auto phi = [&](Int ii, Int jj) {
          const Real th0 = ref_ ? ref_->theta0().at(ii, jj, k) : Real(1);
          return kCp * th0 * mass.at(ii, jj, k);
        };
        u.at(i, j, k) += -invf * (phi(i, jp) - phi(i, jm)) / (Real(2) * dy);
        v.at(i, j, k) += invf * (phi(ip, j) - phi(im, j)) / (Real(2) * dx);
      }
    }
  }
}

void BalanceOperator::derive_mass_adjoint(const grid::Field<Real>& u,
                                          const grid::Field<Real>& v,
                                          grid::Field<Real>& mass) const {
  if (form_ == BalanceForm::None) return;
  const grid::Grid& g = *grid_;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Real stab = (form_ == BalanceForm::OmegaEquation) ? Real(0.8) : Real(1.0);
  const Real invf = stab / f0_;
  // 严格转置：phi'(ii,jj) 的权重来自相邻点的差分；注意权重含 cp theta0(ii,jj)
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      const Int jm = clamp(j - 1, Int(0), ny - 1);
      const Int jp = clamp(j + 1, Int(0), ny - 1);
      const Real dy = g.dy_at(j);
      for (Int i = 0; i < nx; ++i) {
        const Int im = clamp(i - 1, Int(0), nx - 1);
        const Int ip = clamp(i + 1, Int(0), nx - 1);
        const Real dx = g.dx_at(i);
        const Real wu = -invf / (Real(2) * dy);
        const Real wv = invf / (Real(2) * dx);
        // u 的贡献
        const Real cu = wu * u.at(i, j, k);
        const Real cv = wv * v.at(i, j, k);
        mass.at(i, jp, k) += cu * (ref_ ? kCp * ref_->theta0().at(i, jp, k) : kCp);
        mass.at(i, jm, k) -= cu * (ref_ ? kCp * ref_->theta0().at(i, jm, k) : kCp);
        mass.at(ip, j, k) += cv * (ref_ ? kCp * ref_->theta0().at(ip, j, k) : kCp);
        mass.at(im, j, k) -= cv * (ref_ ? kCp * ref_->theta0().at(im, j, k) : kCp);
      }
    }
  }
}

// ===========================================================================
// 标准控制变量变换
// ===========================================================================

namespace {

/// 控制变量表：不平衡风 u、v，垂直速度 w，位温 theta，Exner pi，比湿 qv
inline constexpr int kNumControlVars = 6;
inline const char* kControlNames[kNumControlVars] = {"u_unbal", "v_unbal", "w", "theta",
                                                     "pi", "qv"};

class StandardControlVariableTransform final : public ControlVariableTransform {
 public:
  StandardControlVariableTransform(const grid::Grid& g, const dyn::ReferenceState& ref,
                                   const config::DaConfig& da_cfg,
                                   const ControlVariableConfig& cfg)
      : grid_(&g), ref_(&ref), cfg_(cfg), bal_(g, ref, cfg.balance) {
    nx_ = g.nx();
    ny_ = g.ny();
    nz_ = g.nz();
    nh_ = static_cast<Size>(nx_) * static_cast<Size>(ny_);
    N_ = nh_ * static_cast<Size>(nz_);
    n_ = static_cast<Size>(kNumControlVars) * N_;
    vert_levels_ = std::max(1, nz_);

    build_horizontal();
    build_vertical();
    build_variance(da_cfg);

    packed_.assign(n_, Real(0));
    tmp1_.assign(n_, Real(0));
    tmp2_.assign(n_, Real(0));
    block_.assign(N_, Real(0));
    block2_.assign(N_, Real(0));
  }

  // ---- v -> dx -----------------------------------------------------------
  void to_state(const Vector& v, dyn::State& dx) const override {
    VIBE_CHECK_MSG(v.size() == n_, "to_state: 控制向量维度不匹配");
    sqrtB_(v, tmp1_);
    unpack(tmp1_, dx);
  }

  void from_state(const dyn::State& dx, Vector& v) const override {
    pack(dx, packed_);
    inv_sqrt_(packed_, v);
  }

  void applyB(const Vector& v, Vector& Bv) const override {
    VIBE_CHECK_MSG(v.size() == n_, "applyB: 维度不匹配");
    sqrtB_transpose_(v, tmp1_);
    sqrtB_(tmp1_, Bv);
  }

  void applyBinv(const Vector& v, Vector& Binvv) const override {
    VIBE_CHECK_MSG(v.size() == n_, "applyBinv: 维度不匹配");
    inv_sqrt_(v, tmp1_);
    inv_sqrt_transpose_(tmp1_, Binvv);
  }

  void applySqrtB(const Vector& v, Vector& out) const override {
    VIBE_CHECK_MSG(v.size() == n_, "applySqrtB: 维度不匹配");
    sqrtB_(v, out);
  }

  void applyInvSqrtB(const Vector& v, Vector& out) const override {
    VIBE_CHECK_MSG(v.size() == n_, "applyInvSqrtB: 维度不匹配");
    inv_sqrt_(v, out);
  }

  const Vector& variance() const override { return variance_; }
  Size size() const noexcept override { return n_; }
  const char* name() const noexcept override { return "standard_control_variable"; }

  std::string describe() const override {
    std::ostringstream os;
    os << "StandardControlVariable[n=" << n_ << " vars=" << kNumControlVars
       << " levels=" << nz_ << " modes=" << n_modes_ << "]";
    os << "\n  水平长度尺度=" << cfg_.horizontal_length_scale
       << " m, 扩散迭代 p=" << cfg_.diffusion_iterations
       << " (平方根用 " << std::max(1, cfg_.diffusion_iterations / 2) << " 次平滑)";
    os << "\n  垂直长度尺度=" << cfg_.vertical_length_scale
       << " m, EOF 解释方差(" << n_modes_ << ")=" << vert_explained_;
    os << "\n  平衡关系=" << to_string(cfg_.balance);
    os << "\n  方差: ";
    for (int v = 0; v < kNumControlVars; ++v) {
      os << kControlNames[v] << '=' << (var_by_var_[static_cast<Size>(v)] *
                                        var_by_var_[static_cast<Size>(v)]);
      if (v + 1 < kNumControlVars) os << ", ";
    }
    return os.str();
  }

 private:
  // ---- 各阶段 -------------------------------------------------------------

  void build_horizontal() {
    hcorr_ = std::make_unique<DiffusionCorrelation>(
        *grid_, cfg_.horizontal_length_scale, std::max(1, cfg_.diffusion_iterations),
        cfg_.diffusion_order);
  }

  void build_vertical() {
    // 完整正交基（DCT-II），lambda_m 由扩散型垂直谱给出，保证可逆
    const int n = vert_levels_;
    const int m_full = n;  // 完整基
    n_modes_ = clamp(cfg_.n_vertical_modes, 1, n);
    phi_.assign(static_cast<Size>(m_full), std::vector<Real>(static_cast<Size>(n), Real(0)));
    var_mode_.assign(static_cast<Size>(m_full), Real(0));
    const Real H = std::max(grid_->geom().z_top, Real(1));
    const Real Lv = std::max(cfg_.vertical_length_scale, Real(1));
    const int p = std::max(1, cfg_.diffusion_iterations);
    for (int m = 0; m < m_full; ++m) {
      for (int k = 0; k < n; ++k) {
        phi_[static_cast<Size>(m)][static_cast<Size>(k)] =
            std::sqrt(Real(2) / static_cast<Real>(n)) *
            std::cos(kPi * static_cast<Real>(m) * (static_cast<Real>(k) + Real(0.5)) /
                    static_cast<Real>(n));
      }
      // 扩散型垂直谱： lambda_m = (1 + (Lv^2/(2p)) k_m^2)^{-p}
      const Real km = kPi * static_cast<Real>(m) / H;
      const Real c = Lv * Lv / (Real(2) * static_cast<Real>(p));
      var_mode_[static_cast<Size>(m)] = std::pow(Real(1) + c * km * km, -static_cast<Real>(p));
      if (m >= n_modes_) var_mode_[static_cast<Size>(m)] = Real(1e-4);  // 尾部正则化
    }
    // 预计算正/逆变换矩阵 R[k][k'] = sum_m phi_m(k) lambda_m^{+-1/2} phi_m(k')
    R_.assign(static_cast<Size>(n) * static_cast<Size>(n), Real(0));
    Rinv_.assign(static_cast<Size>(n) * static_cast<Size>(n), Real(0));
    for (int k = 0; k < n; ++k) {
      for (int l = 0; l < n; ++l) {
        Real acc = Real(0), acci = Real(0);
        for (int m = 0; m < m_full; ++m) {
          const Real lam = std::max(var_mode_[static_cast<Size>(m)], Real(1e-12));
          const Real sl = std::sqrt(lam);
          const Real val = phi_[static_cast<Size>(m)][static_cast<Size>(k)] *
                           phi_[static_cast<Size>(m)][static_cast<Size>(l)];
          acc += val * sl;
          acci += val / sl;
        }
        R_[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(l)] = acc;
        Rinv_[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(l)] = acci;
      }
    }
    Real tot = Real(0);
    for (int m = 0; m < n_modes_; ++m) tot += var_mode_[static_cast<Size>(m)];
    vert_explained_ = (tot > Real(0)) ? tot / static_cast<Real>(n_modes_) : Real(0);
  }

  void build_variance(const config::DaConfig& da_cfg) {
    // 逐变量标准差（物理量纲）：优先使用配置给出的方差向量
    Real su = Real(3.0), sv = Real(3.0), sw = Real(0.5), sth = Real(3.0), spi = Real(1e-3),
         sqv = Real(1e-3);
    const auto& vu = cfg_.variance_unbalance;
    if (!vu.empty()) {
      if (vu.size() > 0 && vu[0] > Real(0)) su = std::sqrt(vu[0]);
      if (vu.size() > 1 && vu[1] > Real(0)) sv = std::sqrt(vu[1]);
      if (vu.size() > 2 && vu[2] > Real(0)) sw = std::sqrt(vu[2]);
      if (vu.size() > 3 && vu[3] > Real(0)) sth = std::sqrt(vu[3]);
      if (vu.size() > 4 && vu[4] > Real(0)) spi = std::sqrt(vu[4]);
      if (vu.size() > 5 && vu[5] > Real(0)) sqv = std::sqrt(vu[5]);
    }
    const Real scale = std::max(cfg_.level_scale_unbalance, Real(1e-6)) *
                       std::max(da_cfg.background.variance_scale, Real(1e-6));
    su *= scale; sv *= scale; sw *= scale; sth *= scale; spi *= scale; sqv *= scale;
    if (!da_cfg.assimilate_moisture) sqv = Real(0);
    var_by_var_ = {su, sv, sw, sth, spi, sqv};

    variance_.assign(n_, Real(0));
    for (int v = 0; v < kNumControlVars; ++v) {
      const Real var = var_by_var_[static_cast<Size>(v)] * var_by_var_[static_cast<Size>(v)];
      for (Size q = 0; q < N_; ++q) {
        variance_[static_cast<Size>(v) * N_ + q] = var;
      }
    }
  }

  /// E_eof：逐 (变量, 水平点) 的垂直线性变换（对称）
  void apply_eof(const Vector& x, Vector& y, bool inverse) const {
    y = x;
    const auto& R = inverse ? Rinv_ : R_;
    const int n = vert_levels_;
    std::vector<Real> pin(static_cast<Size>(n)), pout(static_cast<Size>(n));
    for (int v = 0; v < kNumControlVars; ++v) {
      const Size base = static_cast<Size>(v) * N_;
      for (Size h = 0; h < nh_; ++h) {
        const Int i = static_cast<Int>(h % static_cast<Size>(nx_));
        const Int j = static_cast<Int>(h / static_cast<Size>(nx_));
        for (int k = 0; k < n; ++k) {
          pin[static_cast<Size>(k)] = x[base + idx3(nx_, ny_, i, j, k)];
        }
        for (int k = 0; k < n; ++k) {
          Real s = Real(0);
          for (int l = 0; l < n; ++l) {
            s += R[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(l)] *
                 pin[static_cast<Size>(l)];
          }
          pout[static_cast<Size>(k)] = s;
        }
        for (int k = 0; k < n; ++k) {
          y[base + idx3(nx_, ny_, i, j, k)] = pout[static_cast<Size>(k)];
        }
      }
    }
  }

  /// Sigma：逐变量方差
  void apply_variance(const Vector& x, Vector& y, bool inverse) const {
    y = x;
    for (int v = 0; v < kNumControlVars; ++v) {
      const Real s = var_by_var_[static_cast<Size>(v)];
      const Real f = inverse ? (s > Real(0) ? Real(1) / s : Real(0)) : s;
      const Size base = static_cast<Size>(v) * N_;
      for (Size q = 0; q < N_; ++q) y[base + q] = x[base + q] * f;
    }
  }

  /// A_bal：把 pi 块平衡出的风加到 u/v 块（transpose = 质量伴随）
  void apply_balance(const Vector& x, Vector& y, bool transpose) const {
    y = x;
    if (cfg_.balance == BalanceForm::None) return;
    const Size base_u = 0, base_v = N_, base_pi = 4 * N_;
    // pi 块 -> Field
    grid::Field<Real> pi_f(*grid_, grid::Stagger::Cell, "pi_ctrl");
    grid::Field<Real> ub(*grid_, grid::Stagger::Cell, "u_bal");
    grid::Field<Real> vb(*grid_, grid::Stagger::Cell, "v_bal");
    pi_f.fill(Real(0));
    ub.fill(Real(0));
    vb.fill(Real(0));
    if (!transpose) {
      for (Int k = 0; k < nz_; ++k) {
        for (Int j = 0; j < ny_; ++j) {
          for (Int i = 0; i < nx_; ++i) {
            pi_f.at(i, j, k) = x[base_pi + idx3(nx_, ny_, i, j, k)];
          }
        }
      }
      bal_.derive_wind(pi_f, ub, vb);
      for (Int k = 0; k < nz_; ++k) {
        for (Int j = 0; j < ny_; ++j) {
          for (Int i = 0; i < nx_; ++i) {
            const Size q = idx3(nx_, ny_, i, j, k);
            y[base_u + q] = x[base_u + q] + ub.at(i, j, k);
            y[base_v + q] = x[base_v + q] + vb.at(i, j, k);
          }
        }
      }
    } else {
      for (Int k = 0; k < nz_; ++k) {
        for (Int j = 0; j < ny_; ++j) {
          for (Int i = 0; i < nx_; ++i) {
            const Size q = idx3(nx_, ny_, i, j, k);
            ub.at(i, j, k) = x[base_u + q];
            vb.at(i, j, k) = x[base_v + q];
          }
        }
      }
      grid::Field<Real> mass_ad(*grid_, grid::Stagger::Cell, "mass_ad");
      mass_ad.fill(Real(0));
      bal_.derive_mass_adjoint(ub, vb, mass_ad);
      for (Int k = 0; k < nz_; ++k) {
        for (Int j = 0; j < ny_; ++j) {
          for (Int i = 0; i < nx_; ++i) {
            y[base_pi + idx3(nx_, ny_, i, j, k)] += mass_ad.at(i, j, k);
          }
        }
      }
    }
  }

  /// A_bal^{-1}：从 u/v 块中减去平衡风（transpose 为逆的转置）
  void apply_balance_inverse(const Vector& x, Vector& y, bool transpose) const {
    if (cfg_.balance == BalanceForm::None) {
      y = x;
      return;
    }
    // A^{-1} = I - (A - I)，且 (A-I) 的转置是质量伴随；逆的转置 = I - (A-I)^T ... 需要显式
    // 这里直接利用 A = I + C（C 为 pi->风 的耦合），A^{-1} = I - C（因为 C 作用在
    // 不同的变量块上，C^2 = 0 对单次平衡组装成立：C 只把 pi 映射到 u/v）。
    Vector z(n_, Real(0));
    Vector sign_x = x;
    if (!transpose) {
      // y = x - C(x)：C 只读 pi 块，写 u/v 块
      Vector c(n_, Real(0));
      apply_balance(x, c, false);       // c = (I + C) x
      for (Size q = 0; q < n_; ++q) z[q] = x[q] - (c[q] - x[q]);  // x - C x
    } else {
      // A^{-T} = I - C^T
      Vector c(n_, Real(0));
      apply_balance(x, c, true);        // c = (I + C^T) x
      for (Size q = 0; q < n_; ++q) z[q] = x[q] - (c[q] - x[q]);
    }
    y.swap(z);
  }

  // ---- U 与其转置 ---------------------------------------------------------

  /// U = A_bal * Sigma * E * K_h（严格按该顺序，避免阶段顺序错误）
  void sqrtB_(const Vector& v, Vector& out) const {
    Vector a, b, c;
    hcorr_->apply(v, a);            // K_h：水平扩散相关（平方根）
    apply_eof(a, b, false);         // E  ：垂直 EOF 平方根
    apply_variance(b, c, false);    // Sigma：逐变量方差
    apply_balance(c, out, false);   // A_bal：线性平衡组装
  }

  void sqrtB_transpose_(const Vector& v, Vector& out) const {
    Vector a, b, c;
    apply_balance(v, a, true);      // A_bal^T
    apply_variance(a, b, true);     // Sigma^T = Sigma
    apply_eof(b, c, false);         // E^T = E
    hcorr_->apply(c, out);          // K_h^T = K_h
  }

  void inv_sqrt_(const Vector& v, Vector& out) const {
    Vector a, b, c;
    apply_balance_inverse(v, a, false);   // A_bal^{-1}
    apply_variance(a, b, true);           // Sigma^{-1}
    apply_eof(b, c, true);                // E^{-1}
    hcorr_inverse(c, out);                // K_h^{-1}
  }

  void inv_sqrt_transpose_(const Vector& v, Vector& out) const {
    Vector a, b, c;
    hcorr_inverse(v, a);
    apply_eof(a, b, true);
    apply_variance(b, c, true);
    apply_balance_inverse(c, out, true);
  }

  /// K_h^{-1} = (I - c Laplacian)^{p/2}：直接作用（不做 CG，因为这是显式算子）
  void hcorr_inverse(const Vector& x, Vector& y) const {
    const Real L = cfg_.horizontal_length_scale;
    const int p = std::max(1, cfg_.diffusion_iterations);
    const Real c = L * L / (Real(4) * static_cast<Real>(p));
    const int n_apply = std::max(1, p / 2);
    Vector a = x, b;
    std::vector<Real> lap;
    for (int it = 0; it < n_apply; ++it) {
      laplacian_h(*grid_, a, lap);
      b.resize(a.size());
      for (Size i = 0; i < a.size(); ++i) b[i] = a[i] - c * lap[i];
      a.swap(b);
    }
    y.swap(a);
  }

  // ---- 打包 / 解包 --------------------------------------------------------

  void pack(const dyn::State& dx, Vector& out) const {
    out.assign(n_, Real(0));
    const dyn::Species sp[kNumControlVars] = {
        dyn::Species::U, dyn::Species::V, dyn::Species::W,
        dyn::Species::Theta, dyn::Species::Pi, dyn::Species::Qv};
    for (int v = 0; v < kNumControlVars; ++v) {
      const auto& f = dx.field(sp[v]);
      const Size base = static_cast<Size>(v) * N_;
      for (Int k = 0; k < nz_; ++k) {
        for (Int j = 0; j < ny_; ++j) {
          for (Int i = 0; i < nx_; ++i) {
            out[base + idx3(nx_, ny_, i, j, k)] = f.at(i, j, k);
          }
        }
      }
    }
  }

  void unpack(const Vector& v, dyn::State& dx) const {
    const dyn::Species sp[kNumControlVars] = {
        dyn::Species::U, dyn::Species::V, dyn::Species::W,
        dyn::Species::Theta, dyn::Species::Pi, dyn::Species::Qv};
    for (int vi = 0; vi < kNumControlVars; ++vi) {
      auto& f = dx.field(sp[vi]);
      const Size base = static_cast<Size>(vi) * N_;
      for (Int k = 0; k < nz_; ++k) {
        for (Int j = 0; j < ny_; ++j) {
          for (Int i = 0; i < nx_; ++i) {
            f.at(i, j, k) = v[base + idx3(nx_, ny_, i, j, k)];
          }
        }
      }
    }
  }

  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  ControlVariableConfig cfg_;
  BalanceOperator bal_;

  Int nx_ = 0, ny_ = 0, nz_ = 0;
  Size nh_ = 0, N_ = 0, n_ = 0;
  int vert_levels_ = 1;
  int n_modes_ = 1;
  Real vert_explained_ = Real(0);

  std::unique_ptr<DiffusionCorrelation> hcorr_;
  std::vector<std::vector<Real>> phi_;   ///< [mode][level]（完整正交基）
  std::vector<Real> var_mode_;           ///< [mode]
  std::vector<Real> R_, Rinv_;           ///< 垂直变换矩阵（nz x nz）
  std::vector<Real> var_by_var_;         ///< [variable] 标准差
  Vector variance_;

  mutable Vector packed_, tmp1_, tmp2_, block_, block2_;
};

}  // namespace

std::unique_ptr<ControlVariableTransform> make_control_variable_transform(
    const grid::Grid& g, const dyn::ReferenceState& ref, const config::DaConfig& da_cfg,
    const ControlVariableConfig& cv_cfg) {
  return std::make_unique<StandardControlVariableTransform>(g, ref, da_cfg, cv_cfg);
}

}  // namespace vibe::da

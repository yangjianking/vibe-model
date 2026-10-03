/// @file cost_function.cpp
/// @brief 增量 4D-Var 代价函数、梯度与弱约束惩罚项。
///
/// 定义（[V3] Courtier et al. 1994）
/// --------------------------------
/// 以标准化控制变量 v（dx = U v，U = B^{1/2}）：
///
///     J(v) = 1/2 v^T v
///          + 1/2 ( H U v - d )^T R^{-1} ( H U v - d )
///          + J_c(v)
///
///     d    = y - H(x^b)                    （创新向量）
///     J_c  = 1/2 gamma_df || F dx ||^2     （数字滤波，[V25]）
///          + 1/2 gamma_m  || dx ||^2       （量级惩罚，[V4]）
///          + 1/2 gamma_s  || L2 dx ||^2    （平滑惩罚，可选）
///
/// 梯度
/// ----
///     grad J = v + U^T H^T R^{-1} ( H U v - d ) + grad J_c
///
/// 其中 U^T 通过冻结接口的组合恒等式实现：
///     U^T w = U^{-1} (U U^T) w = applyInvSqrtB(applyB(w))
/// 不需要假设 U 对称。传入的 w 是"状态空间"的伴随量，按本模块统一的
/// DA 打包顺序（[u,v,w,theta,pi,qv]，k 最慢、i 最快）整理为 Vector。
///
/// 数字滤波的实现
/// --------------
/// 冻结接口的构造函数没有时间步长与增量轨迹，因此把数字滤波惩罚实现为
/// **空间方向的对称高通 FIR**（Dolph-Chebyshev 窗构造，零直流增益），沿垂直
/// 方向施加并采用反射边界，使惩罚算子是自伴的（F^T = F），从而
/// grad J_df = gamma * F^T F dx = gamma * F^2 dx。生产实现应在时间方向对
/// 增量轨迹做同样处理（[V25]），这部分由 Incremental4DVar 提供的轨迹承担。
///
/// 复杂度：一次评估 O(N + n_obs + nz log nz 的卷积)，其中 N 为状态自由度。
///
/// 文献：[V1][V2][V3][V4][V10][V22][V25]。

#include "vibe/da/cost_function.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::da {

namespace {

// ---------------------------------------------------------------------------
// DA 打包顺序（与 StandardControlVariableTransform 保持一致）
// ---------------------------------------------------------------------------

inline constexpr int kDa = 6;
const dyn::Species kDaSpecies[kDa] = {dyn::Species::U,   dyn::Species::V,
                                      dyn::Species::W,   dyn::Species::Theta,
                                      dyn::Species::Pi,  dyn::Species::Qv};

inline Size idx3(Int nx, Int ny, Int i, Int j, Int k) {
  return (static_cast<Size>(k) * static_cast<Size>(ny) + static_cast<Size>(j)) *
             static_cast<Size>(nx) +
         static_cast<Size>(i);
}

/// 把 DA 变量块打包为控制空间向量（大小 = 6 * nx*ny*nz）
void pack_da(const dyn::State& s, Vector& out) {
  const grid::Grid& g = s.grid();
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Size N = static_cast<Size>(nx) * static_cast<Size>(ny) * static_cast<Size>(nz);
  out.assign(static_cast<Size>(kDa) * N, Real(0));
  for (int v = 0; v < kDa; ++v) {
    const auto& f = s.field(kDaSpecies[v]);
    const Size base = static_cast<Size>(v) * N;
    for (Int k = 0; k < nz; ++k) {
      for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) {
          out[base + idx3(nx, ny, i, j, k)] = f.at(i, j, k);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 数字滤波 / 平滑惩罚的对称高通 FIR
// ---------------------------------------------------------------------------

/// 反射边界索引（对合，保证卷积算子自伴）
inline Int reflect_index(Int k, Int n) {
  if (n <= 1) return 0;
  const Int period = 2 * n - 2;
  Int m = k % period;
  if (m < 0) m += period;
  return (m < n) ? m : period - m;
}

/// Chebyshev 多项式 T_N(x)（|x| <= 1 用 cos，|x| > 1 用 cosh）
Real chebyshev(int N, Real x) {
  if (std::abs(x) <= Real(1)) return std::cos(static_cast<Real>(N) * std::acos(x));
  return std::cosh(static_cast<Real>(N) * std::acosh(x));
}

/// Dolph-Chebyshev 窗（奇长度，单位直流增益）；atten_db 为旁瓣衰减
std::vector<Real> dolph_chebyshev_window(int n, Real atten_db) {
  if (n % 2 == 0) ++n;  // 强制奇长度
  const int M = (n - 1) / 2;
  const Real r = std::pow(Real(10), -atten_db / Real(20));
  std::vector<Real> w(static_cast<Size>(n), Real(0));
  if (M == 0) {
    w[0] = Real(1);
    return w;
  }
  const Real x0 = std::cosh(std::acosh(Real(1) / r) / static_cast<Real>(n - 1));
  for (int k = 0; k < n; ++k) {
    Real s = Real(1);
    for (int m = 1; m <= M; ++m) {
      const Real xm = x0 * std::cos(kPi * static_cast<Real>(m) / static_cast<Real>(n));
      s += Real(2) * chebyshev(n - 1, xm) *
           std::cos(kTwoPi * static_cast<Real>(m) * static_cast<Real>(k) / static_cast<Real>(n));
    }
    w[static_cast<Size>(k)] = s / static_cast<Real>(n);
  }
  Real sum = Real(0);
  for (Real v : w) sum += v;
  if (sum != Real(0)) {
    for (auto& v : w) v /= sum;
  }
  return w;
}

/// 高通 FIR 核 h = delta - lowpass（对称、零直流增益）
std::vector<Real> highpass_kernel(int n) {
  std::vector<Real> w = dolph_chebyshev_window(std::max(n, 3), Real(40));
  const int mid = static_cast<int>(w.size()) / 2;
  std::vector<Real> h = w;
  for (auto& v : h) v = -v;
  h[static_cast<Size>(mid)] += Real(1);
  return h;
}

/// 二阶差分核 [-1, 2, -1]（对称，用于平滑惩罚）
std::vector<Real> second_difference_kernel() { return {Real(-1), Real(2), Real(-1)}; }

/// 沿垂直方向对状态做对称 FIR 卷积（反射边界）
void filter_state(const dyn::State& in, dyn::State& out, const std::vector<Real>& kernel) {
  const grid::Grid& g = in.grid();
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const int nh = static_cast<int>(kernel.size());
  const int mid = nh / 2;
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto& fi = in.field(static_cast<dyn::Species>(sp));
    auto& fo = out.field(static_cast<dyn::Species>(sp));
    for (Int k = 0; k < nz; ++k) {
      for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) {
          Real s = Real(0);
          for (int m = 0; m < nh; ++m) {
            const Int kk = reflect_index(k + (m - mid), nz);
            s += kernel[static_cast<Size>(m)] * fi.at(i, j, kk);
          }
          fo.at(i, j, k) = s;
        }
      }
    }
  }
}

/// 逐物种平方和（用于 ||dx||^2）
Real state_sqnorm(const dyn::State& s, const std::vector<dyn::Species>& which) {
  Real acc = Real(0);
  for (auto sp : which) {
    const auto& f = s.field(sp);
    for (Int k = 0; k < f.nz(); ++k) {
      for (Int j = 0; j < f.ny(); ++j) {
        for (Int i = 0; i < f.nx(); ++i) {
          const Real v = f.at(i, j, k);
          acc += v * v;
        }
      }
    }
  }
  return acc;
}

}  // namespace

// ===========================================================================
// CostFunction
// ===========================================================================

CostFunction::CostFunction(const ControlVariableTransform& b_transform,
                           const TangentLinearModel& tl,
                           const obs::ObservationOperator& h,
                           const obs::ObsSpace& obs, const dyn::State& x_background,
                           const std::vector<Real>& innovations)
    : b_(&b_transform),
      tl_(&tl),
      h_(&h),
      obs_(&obs),
      xb_(&x_background),
      innovations_(innovations) {
  VIBE_CHECK_MSG(innovations_.size() == obs.obs.size(),
                 "CostFunction: innovations 长度与观测数不一致");
  VIBE_CHECK_MSG(b_transform.size() > 0, "CostFunction: 控制变量维度必须为正");
}

void CostFunction::expand(const Vector& v, dyn::State& dx) const {
  b_->to_state(v, dx);
}

Real CostFunction::digital_filter_penalty(const dyn::State& dx) const {
  if (!penalty_.digital_filter || !(penalty_.digital_filter_weight > Real(0))) return Real(0);
  const std::vector<Real> h = highpass_kernel(static_cast<int>(std::max<Int>(dx.grid().nz(), 3)));
  dyn::State filtered(dx.grid());
  filter_state(dx, filtered, h);
  const std::vector<dyn::Species> all = {dyn::Species::U,   dyn::Species::V,
                                        dyn::Species::W,   dyn::Species::Theta,
                                        dyn::Species::Pi,  dyn::Species::Qv};
  return Real(0.5) * penalty_.digital_filter_weight * state_sqnorm(filtered, all);
}

Real CostFunction::value_and_gradient(const Vector& v, Vector& g) const {
  const Size n = b_->size();
  VIBE_CHECK_MSG(v.size() == n, "CostFunction: 控制向量维度不匹配");
  const grid::Grid& gr = xb_->grid();
  ++evaluations_;

  dyn::State dx(gr);
  b_->to_state(v, dx);  // dx = U v

  // ---- 背景项 -----------------------------------------------------------
  Real Jb = Real(0);
  for (Size i = 0; i < n; ++i) Jb += Real(0.5) * v[i] * v[i];

  obs::ModelStateView xv;
  xv.state = xb_;
  xv.ref = xb_->reference();
  xv.grid = &gr;
  xv.time = xb_->time;

  // ---- 观测项 -----------------------------------------------------------
  const std::vector<Real> rinv = obs_->inverse_variance();
  Real Jobs = Real(0);
  Vector w(n, Real(0));  // 状态空间伴随（打包后）
  std::vector<Real> w_obs(obs_->obs.size(), Real(0));
  if (h_ != nullptr && !obs_->obs.empty()) {
    std::vector<Real> ytl;
    h_->applyTL(xv, dx, *obs_, ytl);
    VIBE_CHECK_MSG(ytl.size() == obs_->obs.size(), "CostFunction: applyTL 长度不匹配");
    for (Size i = 0; i < ytl.size(); ++i) {
      const Real resid = ytl[i] - innovations_[i];
      Jobs += Real(0.5) * rinv[i] * resid * resid;
      w_obs[i] = rinv[i] * resid;
    }
    dyn::State dx_adj(gr);
    h_->applyAD(xv, w_obs, *obs_, dx_adj);
    pack_da(dx_adj, w);
  }

  // ---- 惩罚项 -----------------------------------------------------------
  Real Jpen = Real(0);
  dyn::State pen_grad(gr);   // 状态空间惩罚梯度（对 dx）
  const std::vector<dyn::Species> all = {dyn::Species::U,   dyn::Species::V,
                                        dyn::Species::W,   dyn::Species::Theta,
                                        dyn::Species::Pi,  dyn::Species::Qv};

  if (penalty_.digital_filter && penalty_.digital_filter_weight > Real(0)) {
    const std::vector<Real> h =
        highpass_kernel(static_cast<int>(std::max<Int>(gr.nz(), 3)));
    dyn::State f1(gr), f2(gr);
    filter_state(dx, f1, h);
    const Real nrm = state_sqnorm(f1, all);
    Jpen += Real(0.5) * penalty_.digital_filter_weight * nrm;
    // grad = gamma F^T F dx = gamma F^2 dx（F 自伴）
    filter_state(f1, f2, h);
    pen_grad.add_scaled(penalty_.digital_filter_weight, f2);
  }

  if (penalty_.magnitude_penalty && penalty_.magnitude_weight > Real(0)) {
    const Real nrm = state_sqnorm(dx, all);
    Jpen += Real(0.5) * penalty_.magnitude_weight * nrm;
    pen_grad.add_scaled(penalty_.magnitude_weight, dx);
  }

  if (penalty_.smoothness && penalty_.smoothness_weight > Real(0)) {
    const std::vector<Real> l2 = second_difference_kernel();
    dyn::State f1(gr), f2(gr);
    filter_state(dx, f1, l2);
    Jpen += Real(0.5) * penalty_.smoothness_weight * state_sqnorm(f1, all);
    filter_state(f1, f2, l2);
    pen_grad.add_scaled(penalty_.smoothness_weight, f2);
  }

  // ---- 组装梯度 ---------------------------------------------------------
  g.assign(n, v);  // 背景项
  if (h_ != nullptr && !obs_->obs.empty()) {
    // U^T w = U^{-1} (U U^T) w
    Vector t1, t2;
    b_->applyB(w, t1);
    b_->applyInvSqrtB(t1, t2);
    for (Size i = 0; i < n; ++i) g[i] += t2[i];
  }
  if (Jpen > Real(0)) {
    Vector pg, t1, t2;
    pack_da(pen_grad, pg);
    b_->applyB(pg, t1);
    b_->applyInvSqrtB(t1, t2);
    for (Size i = 0; i < n; ++i) g[i] += t2[i];
  }

  Real gn = Real(0);
  for (Size i = 0; i < n; ++i) gn += g[i] * g[i];
  stats_.background_term = Jb;
  stats_.observation_term = Jobs;
  stats_.penalty_term = Jpen;
  stats_.total = Jb + Jobs + Jpen;
  stats_.gradient_norm = std::sqrt(gn);
  stats_.n_observations = obs_->obs.size();
  stats_.evaluations = evaluations_;
  return stats_.total;
}

Real CostFunction::value(const Vector& v) const {
  Vector g;
  return value_and_gradient(v, g);
}

void CostFunction::gradient(const Vector& v, Vector& g) const {
  (void)value_and_gradient(v, g);
}

}  // namespace vibe::da

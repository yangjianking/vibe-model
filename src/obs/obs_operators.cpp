/// @file obs_operators.cpp
/// @brief Sounding / Surface / Wind / GnssRo / Radar / Radiance 观测算子。
///
/// 每个算子都实现 apply / applyTL / applyAD。共同的实现规范：
///   1. apply  先插值到观测点，再套物理诊断关系；
///   2. applyTL 的所有系数（插值权重、dT/dtheta、dT/dpi、dp/dpi 等）
///      **冻结在基础态 x 上**，只对扰动线性；
///   3. applyAD 是 applyTL 的严格转置，按相反顺序累加到 dx
///      （**累加**而不是覆盖，便于 CompositeOperator 串联）。
///
/// 基础态约定：State 中的 theta/pi 是**全场**（theta0 + theta' 等），
/// 因此 T = theta * pi 可直接使用；扰动 dx 的相应分量即全场的扰动。
///
/// 观测方程与线性化
/// ----------------
///   探空      H = (u, v, w, theta*pi, qv, p0 pi^{cp/Rd})
///   地面      H = (p0 pi_s^{cp/Rd}, (theta_1 + (g/cp)(z1-2)) pi_s, qv_1, u_1 f_log)
///   导风/散射 H = 单层风，散射计另乘对数律因子
///   掩星      N = 77.6 p_hPa / T + 3.73e5 qv p_hPa / T^2
///   雷达      dBZ = 10 log10( Z_rain + Z_snow + Z_graupel ),  Z = a (rho q)^1.75
///   辐射率    I = RT(theta, pi, qv, ...) 的逐层积分（见 radiative_transfer.cpp）
///
/// 复杂度：每个观测 O(1)（辐射率为 O(nz) 逐层）。
///
/// 文献：[O1][O3][O4][O5][O6][O7][O8][P3][P9][P10]。

#include "vibe/obs/obs_operators.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/dyn/diagnostics.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

// ---------------------------------------------------------------------------
// 整层快速辐射传输的内部接口
//   radiative_transfer.hpp 是冻结头文件，无法新增成员/类型；因此
//   radiative_transfer.cpp 在具名命名空间 detail 中提供"整层"积分接口，
//   RadianceOperator 复用同一份实现（定义见 radiative_transfer.cpp，
//   两处 ColumnProfile 的定义必须逐字一致，满足 ODR）。
// ---------------------------------------------------------------------------
namespace vibe::obs::detail {

inline constexpr int kNumPredictors = 13;

struct ColumnProfile {
  std::vector<Real> pressure;
  std::vector<Real> temperature;
  std::vector<Real> qv;
  std::vector<Real> ozone;
  std::vector<Real> cloud_water;
  std::vector<Real> cloud_ice;
  Real zenith_angle_deg = Real(0);
  Real surface_temperature = Real(0);
  Real surface_pressure = Real(0);
  Real surface_emissivity = Real(1);
  Real skin_temperature = Real(0);

  Size n_layers() const noexcept { return pressure.size(); }
};

RadianceResult forward_column(const std::vector<Real>& coefs, const Channel& ch,
                              const ColumnProfile& col);
RadianceJacobian jacobian_column(const std::vector<Real>& coefs, const Channel& ch,
                                 const ColumnProfile& col);

}  // namespace vibe::obs::detail

namespace vibe::obs {

namespace {

// ---------------------------------------------------------------------------
// 坐标映射与水平插值（与 obs_operator.cpp 保持同一约定）
// ---------------------------------------------------------------------------

Real cell_x_at(const grid::Grid& g, Int i) {
  const auto& geom = g.geom();
  if (!geom.variable_resolution || geom.dx_cell.empty()) {
    return geom.x0 + (static_cast<Real>(i) + Real(0.5)) * geom.dx;
  }
  Real x = geom.x0;
  for (Int m = 0; m < i; ++m) x += geom.dx_cell[static_cast<Size>(m)];
  return x + Real(0.5) * geom.dx_cell[static_cast<Size>(i)];
}

Real cell_y_at(const grid::Grid& g, Int j) {
  const auto& geom = g.geom();
  if (!geom.variable_resolution || geom.dy_cell.empty()) {
    return geom.y0 + (static_cast<Real>(j) + Real(0.5)) * geom.dy;
  }
  Real y = geom.y0;
  for (Int m = 0; m < j; ++m) y += geom.dy_cell[static_cast<Size>(m)];
  return y + Real(0.5) * geom.dy_cell[static_cast<Size>(j)];
}

struct Locate1D {
  Int  i0 = 0;
  Real w = Real(0);
};

template <class Xc>
Locate1D locate_1d(Int n, Xc xc, Real x) {
  if (n <= 1) return {0, Real(0)};
  const Real x0 = xc(0), xn = xc(n - 1);
  if (!(x > x0)) {
    const Real d = xc(1) - x0;
    return {0, d > Real(0) ? (x - x0) / d : Real(0)};
  }
  if (!(x < xn)) {
    const Real d = xn - xc(n - 2);
    return {n - 2, d > Real(0) ? (x - xc(n - 2)) / d : Real(0)};
  }
  Int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const Int mid = (lo + hi) / 2;
    if (xc(mid) <= x) lo = mid; else hi = mid;
  }
  const Real d = xc(lo + 1) - xc(lo);
  return {lo, d > Real(0) ? (x - xc(lo)) / d : Real(0)};
}

Locate1D locate_x(const grid::Grid& g, Real x) {
  return locate_1d(g.nx(), [&](Int i) { return cell_x_at(g, i); }, x);
}

Locate1D locate_y(const grid::Grid& g, Real y) {
  return locate_1d(g.ny(), [&](Int j) { return cell_y_at(g, j); }, y);
}

inline Size off_clamped(const grid::Field<Real>& f, Int i, Int j, Int k) {
  return f.offset(clamp(i, Int(0), f.nx() - 1), clamp(j, Int(0), f.ny() - 1),
                  clamp(k, Int(0), f.nz() - 1));
}

/// 单个层 k 上的水平双线性插值（标量体心场）
Real sample_level(const grid::Field<Real>& f, const grid::Grid& g, Real x, Real y, Int k) {
  const auto li = locate_x(g, x);
  const auto lj = locate_y(g, y);
  const Real* d = f.data();
  const Real v00 = d[off_clamped(f, li.i0, lj.i0, k)];
  const Real v10 = d[off_clamped(f, li.i0 + 1, lj.i0, k)];
  const Real v01 = d[off_clamped(f, li.i0, lj.i0 + 1, k)];
  const Real v11 = d[off_clamped(f, li.i0 + 1, lj.i0 + 1, k)];
  return lerp(lerp(v00, v10, li.w), lerp(v01, v11, li.w), lj.w);
}

/// sample_level 的转置：把体心量 v 按双线性权重散射回层 k
void scatter_level(grid::Field<Real>& f, const grid::Grid& g, Real x, Real y, Int k, Real v) {
  const auto li = locate_x(g, x);
  const auto lj = locate_y(g, y);
  Real* d = f.data();
  const Real wx[2] = {Real(1) - li.w, li.w};
  const Real wy[2] = {Real(1) - lj.w, lj.w};
  for (int dj = 0; dj < 2; ++dj) {
    for (int di = 0; di < 2; ++di) {
      const Real w = wx[di] * wy[dj];
      if (w == Real(0)) continue;
      d[off_clamped(f, li.i0 + di, lj.i0 + dj, k)] += v * w;
    }
  }
}

/// 单层错位场 -> 体心的水平插值（u/v 面心 -> 体心后双线性）
Real sample_level_staggered(const grid::Field<Real>& f, const grid::Grid& g,
                            grid::Stagger s, Real x, Real y, Int k) {
  const auto li = locate_x(g, x);
  const auto lj = locate_y(g, y);
  auto node = [&](Int i, Int j) -> Real {
    const Real* d = f.data();
    switch (s) {
      case grid::Stagger::Cell: return d[off_clamped(f, i, j, k)];
      case grid::Stagger::FaceX:
        return Real(0.5) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i + 1, j, k)]);
      case grid::Stagger::FaceY:
        return Real(0.5) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i, j + 1, k)]);
      case grid::Stagger::FaceZ:
        return Real(0.5) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i, j, k + 1)]);
      case grid::Stagger::Corner:
        return Real(0.25) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i + 1, j, k)] +
                             d[off_clamped(f, i, j + 1, k)] + d[off_clamped(f, i + 1, j + 1, k)]);
    }
    return Real(0);
  };
  return lerp(lerp(node(li.i0, lj.i0), node(li.i0 + 1, lj.i0), li.w),
              lerp(node(li.i0, lj.i0 + 1), node(li.i0 + 1, lj.i0 + 1), li.w), lj.w);
}

/// 单层错位场的伴随散射
void scatter_level_staggered(grid::Field<Real>& f, const grid::Grid& g, grid::Stagger s,
                             Real x, Real y, Int k, Real v) {
  const auto li = locate_x(g, x);
  const auto lj = locate_y(g, y);
  Real* d = f.data();
  auto node_add = [&](Int i, Int j, Real q) {
    switch (s) {
      case grid::Stagger::Cell:
        d[off_clamped(f, i, j, k)] += q;
        return;
      case grid::Stagger::FaceX:
        d[off_clamped(f, i, j, k)] += Real(0.5) * q;
        d[off_clamped(f, i + 1, j, k)] += Real(0.5) * q;
        return;
      case grid::Stagger::FaceY:
        d[off_clamped(f, i, j, k)] += Real(0.5) * q;
        d[off_clamped(f, i, j + 1, k)] += Real(0.5) * q;
        return;
      case grid::Stagger::FaceZ:
        d[off_clamped(f, i, j, k)] += Real(0.5) * q;
        d[off_clamped(f, i, j, k + 1)] += Real(0.5) * q;
        return;
      case grid::Stagger::Corner:
        d[off_clamped(f, i, j, k)] += Real(0.25) * q;
        d[off_clamped(f, i + 1, j, k)] += Real(0.25) * q;
        d[off_clamped(f, i, j + 1, k)] += Real(0.25) * q;
        d[off_clamped(f, i + 1, j + 1, k)] += Real(0.25) * q;
        return;
    }
  };
  const Real wx[2] = {Real(1) - li.w, li.w};
  const Real wy[2] = {Real(1) - lj.w, lj.w};
  for (int dj = 0; dj < 2; ++dj) {
    for (int di = 0; di < 2; ++di) {
      const Real w = wx[di] * wy[dj];
      if (w == Real(0)) continue;
      node_add(li.i0 + di, lj.i0 + dj, v * w);
    }
  }
}

// ---------------------------------------------------------------------------
// 热力学诊断
// ---------------------------------------------------------------------------

/// 由全 Exner 计算气压 p = p00 * pi^{cp/Rd}
inline Real pressure_of(Real pi) { return kP0 * std::pow(std::max(pi, Real(1e-6)), kCp / kRd); }

/// dp/dpi
inline Real dpressure_dpi(Real pi) {
  const Real e = kCp / kRd;
  return kP0 * e * std::pow(std::max(pi, Real(1e-6)), e - Real(1));
}

/// 默认臭氧气候廓线（若剖面未给出臭氧）：峰值位于 50 hPa 附近，[O1]
inline Real default_ozone(Real pressure_pa) {
  const Real x = std::log(std::max(pressure_pa, Real(1)) / Real(5000));
  return Real(8.0e-6) * std::exp(Real(-0.5) * x * x / Real(0.9));
}

/// 10 m 风的对数律因子 f = ln(10/z0)/ln(z1/z0)，z0 取陆地粗糙度
Real log_wind_factor(Real z1) {
  const Real z0 = kRoughnessLand;
  if (!(z1 > z0 * Real(1.001))) return Real(1);
  const Real denom = std::log(z1 / z0);
  if (!(denom > Real(1e-6))) return Real(1);
  return std::log(Real(10) / z0) / denom;
}

// ---------------------------------------------------------------------------
// 折射率与雷达反射率（在算子内部重写，便于解析求导）
// ---------------------------------------------------------------------------

/// GNSS 折射率 [O4][O5]：N = 77.6 p_hPa/T + 3.73e5 qv p_hPa/T^2
/// 输入气压为 Pa，内部换算为 hPa。
inline Real refractivity_terms(Real pressure_pa, Real temperature, Real qv, Real& dN_dp,
                              Real& dN_dT, Real& dN_dqv) {
  const Real t = std::max(temperature, Real(1));
  const Real p = pressure_pa / Real(100);
  const Real a = Real(77.6), b = Real(3.73e5);
  const Real N = a * p / t + b * qv * p / (t * t);
  dN_dp = (a / t + b * qv / (t * t)) / Real(100);          // 对 Pa
  dN_dT = -a * p / (t * t) - Real(2) * b * qv * p / (t * t * t);
  dN_dqv = b * p / (t * t);
  return N;
}

/// [P3] Thompson et al. (2008) 简化反射率因子；返回 Z 及对 qr/qs/qg/T 的解析导数。
/// 公式与 dyn::reflectivity_dbz **逐项一致**（简化形式 + 雪的融化因子）：
///   Z = 3.63e9 (rho qr)^1.75 + 1.0e10 (rho qs melt)^1.75 + 4.0e10 (rho qg)^1.75
///   melt = exp(-0.1 (T - 273.15))  (T > 273.15)，否则 1
struct RadarTerms {
  Real z = Real(0);
  Real dz_dqr = Real(0), dz_dqs = Real(0), dz_dqg = Real(0), dz_dT = Real(0);
};

inline RadarTerms radar_terms(Real qr, Real qs, Real qg, Real rho, Real temperature) {
  constexpr Real a_r = Real(3.63e9), a_s = Real(1.0e10), a_g = Real(4.0e10);
  constexpr Real b = Real(1.75);
  RadarTerms t;
  const Real melt = (temperature > kT0) ? std::exp(-Real(0.1) * (temperature - kT0)) : Real(1);
  const Real dmelt_dT = (temperature > kT0) ? -Real(0.1) * melt : Real(0);
  Real unused_dT = Real(0);
  // xf <= 0 的截断是冻结开关（导数为 0），与 dyn::reflectivity_dbz 的 max(.,0) 一致
  auto term = [&](Real q, Real a, Real factor, Real dfactor_dT, Real& dq, Real& dTm) -> Real {
    const Real x = rho * q;
    if (!(x > Real(0))) { dq = Real(0); dTm = Real(0); return Real(0); }
    const Real xf = x * factor;
    if (!(xf > Real(0))) { dq = Real(0); dTm = Real(0); return Real(0); }
    const Real z = a * std::pow(xf, b);
    const Real dz_dxf = a * b * std::pow(xf, b - Real(1));
    dq = dz_dxf * rho * factor;
    dTm = dz_dxf * x * dfactor_dT;
    return z;
  };
  t.z = term(qr, a_r, Real(1), Real(0), t.dz_dqr, unused_dT);
  t.z += term(qs, a_s, melt, dmelt_dT, t.dz_dqs, t.dz_dT);
  t.z += term(qg, a_g, Real(1), Real(0), t.dz_dqg, unused_dT);
  return t;
}

/// dBZ 与 d(dBZ)/dZ。与 dyn::reflectivity_dbz 相同：Z < 1e-2 时返回 -30（冻结开关）
inline Real dbz_of(Real z, Real& ddbz_dz) {
  if (!(z >= Real(1e-2))) {
    ddbz_dz = Real(0);
    return Real(-30);
  }
  ddbz_dz = Real(10) / (std::log(Real(10)) * z);
  return Real(10) * std::log10(z);
}

}  // namespace

// ===========================================================================
// SoundingOperator
// ===========================================================================

SoundingOperator::SoundingOperator(const grid::Grid& g) : grid_(&g) {}

void SoundingOperator::apply(const ModelStateView& x, const ObsSpace& obs,
                             std::vector<Real>& y) const {
  VIBE_CHECK_MSG(x.valid(), "SoundingOperator::apply 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  y.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    switch (o.variable) {
      case VarKind::U:
        y[m] = sample_staggered(s.u(), g, grid::Stagger::FaceX, o.x, o.y, o.z);
        break;
      case VarKind::V:
        y[m] = sample_staggered(s.v(), g, grid::Stagger::FaceY, o.x, o.y, o.z);
        break;
      case VarKind::W:
        y[m] = sample_staggered(s.w(), g, grid::Stagger::FaceZ, o.x, o.y, o.z);
        break;
      case VarKind::T: {
        // 全量场约定：State 的 theta/pi 已是全量，无需加参考态
        const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
        const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
        y[m] = th * pi;
        break;
      }
      case VarKind::Q:
        y[m] = sample_scalar(s.qv(), g, o.x, o.y, o.z);
        break;
      case VarKind::P: {
        const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
        y[m] = pressure_of(pi);
        break;
      }
      default:
        y[m] = Real(0);
        break;
    }
  }
}

void SoundingOperator::applyTL(const ModelStateView& x, const dyn::State& dx,
                               const ObsSpace& obs, std::vector<Real>& dy) const {
  VIBE_CHECK_MSG(x.valid(), "SoundingOperator::applyTL 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  dy.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    switch (o.variable) {
      case VarKind::U:
        dy[m] = sample_staggered(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, o.z);
        break;
      case VarKind::V:
        dy[m] = sample_staggered(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, o.z);
        break;
      case VarKind::W:
        dy[m] = sample_staggered(dx.w(), g, grid::Stagger::FaceZ, o.x, o.y, o.z);
        break;
      case VarKind::T: {
        // dT = pi * dtheta + theta * dpi，系数冻结在基础态插值值上
        // 全量场约定：State 的 theta/pi 已是全量，无需加参考态
        const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
        const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
        const Real dth = sample_scalar(dx.theta(), g, o.x, o.y, o.z);
        const Real dpi = sample_scalar(dx.pi(), g, o.x, o.y, o.z);
        dy[m] = pi * dth + th * dpi;
        break;
      }
      case VarKind::Q:
        dy[m] = sample_scalar(dx.qv(), g, o.x, o.y, o.z);
        break;
      case VarKind::P: {
        const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
        const Real dpi = sample_scalar(dx.pi(), g, o.x, o.y, o.z);
        dy[m] = dpressure_dpi(pi) * dpi;
        break;
      }
      default:
        dy[m] = Real(0);
        break;
    }
  }
}

void SoundingOperator::applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                               const ObsSpace& obs, dyn::State& dx) const {
  VIBE_CHECK_MSG(x.valid(), "SoundingOperator::applyAD 需要有效状态");
  VIBE_CHECK_MSG(dy.size() == obs.obs.size(), "SoundingOperator::applyAD dy 长度不匹配");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real v = dy[m];
    if (v == Real(0)) continue;
    switch (o.variable) {
      case VarKind::U:
        scatter_adjoint(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, o.z, v);
        break;
      case VarKind::V:
        scatter_adjoint(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, o.z, v);
        break;
      case VarKind::W:
        scatter_adjoint(dx.w(), g, grid::Stagger::FaceZ, o.x, o.y, o.z, v);
        break;
      case VarKind::T: {
        // 全量场约定：State 的 theta/pi 已是全量，无需加参考态
        const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
        const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
        scatter_adjoint(dx.theta(), g, grid::Stagger::Cell, o.x, o.y, o.z, pi * v);
        scatter_adjoint(dx.pi(), g, grid::Stagger::Cell, o.x, o.y, o.z, th * v);
        break;
      }
      case VarKind::Q:
        scatter_adjoint(dx.qv(), g, grid::Stagger::Cell, o.x, o.y, o.z, v);
        break;
      case VarKind::P: {
        const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
        scatter_adjoint(dx.pi(), g, grid::Stagger::Cell, o.x, o.y, o.z,
                        dpressure_dpi(pi) * v);
        break;
      }
      default:
        break;
    }
  }
}

// ===========================================================================
// SurfaceOperator
// ===========================================================================

SurfaceOperator::SurfaceOperator(const grid::Grid& g) : grid_(&g) {}

void SurfaceOperator::apply(const ModelStateView& x, const ObsSpace& obs,
                            std::vector<Real>& y) const {
  VIBE_CHECK_MSG(x.valid(), "SurfaceOperator::apply 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  y.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    // 地面观测固定在最低模式层
    const Real pi_s = sample_level(s.pi(), g, o.x, o.y, 0);
    const Real th_1 = sample_level(s.theta(), g, o.x, o.y, 0);
    switch (o.variable) {
      case VarKind::PS:
        y[m] = pressure_of(pi_s);
        break;
      case VarKind::T: {
        // theta_2m = theta_1 + (g/cp)(z1 - 2)，T_2m = theta_2m * pi_s
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real z1 = g.z_center(li.i0, lj.i0, 0);
        const Real th_2m = th_1 + (kGravity / kCp) * (z1 - Real(2));
        y[m] = th_2m * pi_s;
        break;
      }
      case VarKind::Q:
        y[m] = sample_level(s.qv(), g, o.x, o.y, 0);
        break;
      case VarKind::U: {
        // 10 m 风用对数律从最低层外推：u10 = u1 * ln(10/z0)/ln(z1/z0)
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        y[m] = f * sample_level_staggered(s.u(), g, grid::Stagger::FaceX, o.x, o.y, 0);
        break;
      }
      case VarKind::V: {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        y[m] = f * sample_level_staggered(s.v(), g, grid::Stagger::FaceY, o.x, o.y, 0);
        break;
      }
      default:
        y[m] = Real(0);
        break;
    }
  }
}

void SurfaceOperator::applyTL(const ModelStateView& x, const dyn::State& dx,
                              const ObsSpace& obs, std::vector<Real>& dy) const {
  VIBE_CHECK_MSG(x.valid(), "SurfaceOperator::applyTL 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  dy.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real pi_s = sample_level(s.pi(), g, o.x, o.y, 0);
    const Real th_1 = sample_level(s.theta(), g, o.x, o.y, 0);
    const Real dpi = sample_level(dx.pi(), g, o.x, o.y, 0);
    const Real dth = sample_level(dx.theta(), g, o.x, o.y, 0);
    switch (o.variable) {
      case VarKind::PS:
        dy[m] = dpressure_dpi(pi_s) * dpi;
        break;
      case VarKind::T: {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real z1 = g.z_center(li.i0, lj.i0, 0);
        const Real th_2m = th_1 + (kGravity / kCp) * (z1 - Real(2));
        // T_2m = (theta_1 + (g/cp)(z1-2)) * pi_s
        dy[m] = pi_s * dth + th_2m * dpi;
        break;
      }
      case VarKind::Q:
        dy[m] = sample_level(dx.qv(), g, o.x, o.y, 0);
        break;
      case VarKind::U: {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real z1 = g.z_center(li.i0, lj.i0, 0);
        const Real f = log_wind_factor(z1);
        dy[m] = f * sample_level_staggered(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, 0);
        break;
      }
      case VarKind::V: {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real z1 = g.z_center(li.i0, lj.i0, 0);
        const Real f = log_wind_factor(z1);
        dy[m] = f * sample_level_staggered(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, 0);
        break;
      }
      default:
        dy[m] = Real(0);
        break;
    }
  }
}

void SurfaceOperator::applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                              const ObsSpace& obs, dyn::State& dx) const {
  VIBE_CHECK_MSG(x.valid(), "SurfaceOperator::applyAD 需要有效状态");
  VIBE_CHECK_MSG(dy.size() == obs.obs.size(), "SurfaceOperator::applyAD dy 长度不匹配");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real v = dy[m];
    if (v == Real(0)) continue;
    const Real pi_s = sample_level(s.pi(), g, o.x, o.y, 0);
    switch (o.variable) {
      case VarKind::PS:
        scatter_level(dx.pi(), g, o.x, o.y, 0, dpressure_dpi(pi_s) * v);
        break;
      case VarKind::T: {
        const Real th_1 = sample_level(s.theta(), g, o.x, o.y, 0);
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real z1 = g.z_center(li.i0, lj.i0, 0);
        const Real th_2m = th_1 + (kGravity / kCp) * (z1 - Real(2));
        scatter_level(dx.theta(), g, o.x, o.y, 0, pi_s * v);
        scatter_level(dx.pi(), g, o.x, o.y, 0, th_2m * v);
        break;
      }
      case VarKind::Q:
        scatter_level(dx.qv(), g, o.x, o.y, 0, v);
        break;
      case VarKind::U: {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        scatter_level_staggered(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, 0, f * v);
        break;
      }
      case VarKind::V: {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        scatter_level_staggered(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, 0, f * v);
        break;
      }
      default:
        break;
    }
  }
}

// ===========================================================================
// WindOperator
// ===========================================================================

WindOperator::WindOperator(const grid::Grid& g, ObsType t) : grid_(&g), type_(t) {}

void WindOperator::apply(const ModelStateView& x, const ObsSpace& obs,
                         std::vector<Real>& y) const {
  VIBE_CHECK_MSG(x.valid(), "WindOperator::apply 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  y.assign(obs.obs.size(), Real(0));
  const bool scatterometer = (type_ == ObsType::Scatterometer);
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    if (o.variable == VarKind::U) {
      if (scatterometer) {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        y[m] = f * sample_level_staggered(s.u(), g, grid::Stagger::FaceX, o.x, o.y, 0);
      } else {
        y[m] = sample_staggered(s.u(), g, grid::Stagger::FaceX, o.x, o.y, o.z);
      }
    } else if (o.variable == VarKind::V) {
      if (scatterometer) {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        y[m] = f * sample_level_staggered(s.v(), g, grid::Stagger::FaceY, o.x, o.y, 0);
      } else {
        y[m] = sample_staggered(s.v(), g, grid::Stagger::FaceY, o.x, o.y, o.z);
      }
    }
  }
}

void WindOperator::applyTL(const ModelStateView& x, const dyn::State& dx,
                           const ObsSpace& obs, std::vector<Real>& dy) const {
  VIBE_CHECK_MSG(x.valid(), "WindOperator::applyTL 需要有效状态");
  const grid::Grid& g = *x.grid;
  dy.assign(obs.obs.size(), Real(0));
  const bool scatterometer = (type_ == ObsType::Scatterometer);
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    if (o.variable == VarKind::U) {
      if (scatterometer) {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        dy[m] = f * sample_level_staggered(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, 0);
      } else {
        dy[m] = sample_staggered(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, o.z);
      }
    } else if (o.variable == VarKind::V) {
      if (scatterometer) {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        dy[m] = f * sample_level_staggered(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, 0);
      } else {
        dy[m] = sample_staggered(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, o.z);
      }
    }
  }
}

void WindOperator::applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                           const ObsSpace& obs, dyn::State& dx) const {
  VIBE_CHECK_MSG(x.valid(), "WindOperator::applyAD 需要有效状态");
  VIBE_CHECK_MSG(dy.size() == obs.obs.size(), "WindOperator::applyAD dy 长度不匹配");
  const grid::Grid& g = *x.grid;
  const bool scatterometer = (type_ == ObsType::Scatterometer);
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real v = dy[m];
    if (v == Real(0)) continue;
    if (o.variable == VarKind::U) {
      if (scatterometer) {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        scatter_level_staggered(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, 0, f * v);
      } else {
        scatter_adjoint(dx.u(), g, grid::Stagger::FaceX, o.x, o.y, o.z, v);
      }
    } else if (o.variable == VarKind::V) {
      if (scatterometer) {
        const auto li = locate_x(g, o.x);
        const auto lj = locate_y(g, o.y);
        const Real f = log_wind_factor(g.z_center(li.i0, lj.i0, 0));
        scatter_level_staggered(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, 0, f * v);
      } else {
        scatter_adjoint(dx.v(), g, grid::Stagger::FaceY, o.x, o.y, o.z, v);
      }
    }
  }
}

// ===========================================================================
// GnssRoOperator
// ===========================================================================

GnssRoOperator::GnssRoOperator(const grid::Grid& g) : grid_(&g) {}

Real GnssRoOperator::refractivity(Real pressure, Real temperature, Real qv) {
  Real a = 0, b = 0, c = 0;
  return refractivity_terms(pressure, temperature, qv, a, b, c);
}

void GnssRoOperator::apply(const ModelStateView& x, const ObsSpace& obs,
                           std::vector<Real>& y) const {
  VIBE_CHECK_MSG(x.valid(), "GnssRoOperator::apply 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  y.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    // State 存储的是**全量**场（见 Equations::diagnose / initial_conditions：
    // theta = theta0 + theta'，pi 为全 Exner，rho 为全密度），因此无需再加参考态。
    const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
    const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
    const Real qv = sample_scalar(s.qv(), g, o.x, o.y, o.z);
    const Real p = pressure_of(pi);
    const Real t = th * pi;
    Real dp = 0, dt = 0, dq = 0;
    y[m] = refractivity_terms(p, t, qv, dp, dt, dq);
  }
}

void GnssRoOperator::applyTL(const ModelStateView& x, const dyn::State& dx,
                             const ObsSpace& obs, std::vector<Real>& dy) const {
  VIBE_CHECK_MSG(x.valid(), "GnssRoOperator::applyTL 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  dy.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    // State 存储的是**全量**场（见 Equations::diagnose / initial_conditions：
    // theta = theta0 + theta'，pi 为全 Exner，rho 为全密度），因此无需再加参考态。
    const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
    const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
    const Real qv = sample_scalar(s.qv(), g, o.x, o.y, o.z);
    const Real p = pressure_of(pi);
    const Real t = th * pi;
    Real dN_dp = 0, dN_dT = 0, dN_dqv = 0;
    refractivity_terms(p, t, qv, dN_dp, dN_dT, dN_dqv);

    const Real dpi = sample_scalar(dx.pi(), g, o.x, o.y, o.z);
    const Real dth = sample_scalar(dx.theta(), g, o.x, o.y, o.z);
    const Real dqv = sample_scalar(dx.qv(), g, o.x, o.y, o.z);
    // p = p(pi), T = theta * pi
    dy[m] = (dN_dp * dpressure_dpi(pi) + dN_dT * th) * dpi + (dN_dT * pi) * dth +
            dN_dqv * dqv;
  }
}

void GnssRoOperator::applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                             const ObsSpace& obs, dyn::State& dx) const {
  VIBE_CHECK_MSG(x.valid(), "GnssRoOperator::applyAD 需要有效状态");
  VIBE_CHECK_MSG(dy.size() == obs.obs.size(), "GnssRoOperator::applyAD dy 长度不匹配");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real v = dy[m];
    if (v == Real(0)) continue;
    // State 存储的是**全量**场（见 Equations::diagnose / initial_conditions：
    // theta = theta0 + theta'，pi 为全 Exner，rho 为全密度），因此无需再加参考态。
    const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
    const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
    const Real qv = sample_scalar(s.qv(), g, o.x, o.y, o.z);
    const Real p = pressure_of(pi);
    const Real t = th * pi;
    Real dN_dp = 0, dN_dT = 0, dN_dqv = 0;
    refractivity_terms(p, t, qv, dN_dp, dN_dT, dN_dqv);
    scatter_adjoint(dx.pi(), g, grid::Stagger::Cell, o.x, o.y, o.z,
                    (dN_dp * dpressure_dpi(pi) + dN_dT * th) * v);
    scatter_adjoint(dx.theta(), g, grid::Stagger::Cell, o.x, o.y, o.z, dN_dT * pi * v);
    scatter_adjoint(dx.qv(), g, grid::Stagger::Cell, o.x, o.y, o.z, dN_dqv * v);
  }
}

// ===========================================================================
// RadarOperator
// ===========================================================================

RadarOperator::RadarOperator(const grid::Grid& g) : grid_(&g) {}

void RadarOperator::apply(const ModelStateView& x, const ObsSpace& obs,
                          std::vector<Real>& y) const {
  VIBE_CHECK_MSG(x.valid(), "RadarOperator::apply 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  y.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real qr = sample_scalar(s.field(dyn::Species::Qr), g, o.x, o.y, o.z);
    const Real qs = sample_scalar(s.field(dyn::Species::Qs), g, o.x, o.y, o.z);
    const Real qg = sample_scalar(s.field(dyn::Species::Qg), g, o.x, o.y, o.z);
    const Real rho = sample_scalar(s.rho(), g, o.x, o.y, o.z);
    const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
    const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
    // 正演直接调用 dyn 层实现，保证与诊断量 reflectivity 完全一致（[P3]）；
    // 其解析导数在本文件内用同构公式复写（冻结接口不暴露 dyn 的导数）。
    y[m] = dyn::reflectivity_dbz(qr, qs, qg, rho, th * pi);
  }
}

void RadarOperator::applyTL(const ModelStateView& x, const dyn::State& dx,
                            const ObsSpace& obs, std::vector<Real>& dy) const {
  VIBE_CHECK_MSG(x.valid(), "RadarOperator::applyTL 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  dy.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real qr = sample_scalar(s.field(dyn::Species::Qr), g, o.x, o.y, o.z);
    const Real qs = sample_scalar(s.field(dyn::Species::Qs), g, o.x, o.y, o.z);
    const Real qg = sample_scalar(s.field(dyn::Species::Qg), g, o.x, o.y, o.z);
    const Real rho = sample_scalar(s.rho(), g, o.x, o.y, o.z);
    const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
    const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
    const RadarTerms zt = radar_terms(qr, qs, qg, rho, th * pi);
    Real ddbz = 0;
    dbz_of(zt.z, ddbz);
    const Real dqr = sample_scalar(dx.field(dyn::Species::Qr), g, o.x, o.y, o.z);
    const Real dqs = sample_scalar(dx.field(dyn::Species::Qs), g, o.x, o.y, o.z);
    const Real dqg = sample_scalar(dx.field(dyn::Species::Qg), g, o.x, o.y, o.z);
    // T = theta * pi（全量场），系数冻结在基础态
    const Real dT = pi * sample_scalar(dx.theta(), g, o.x, o.y, o.z) +
                    th * sample_scalar(dx.pi(), g, o.x, o.y, o.z);
    dy[m] = ddbz * (zt.dz_dqr * dqr + zt.dz_dqs * dqs + zt.dz_dqg * dqg + zt.dz_dT * dT);
  }
}

void RadarOperator::applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                            const ObsSpace& obs, dyn::State& dx) const {
  VIBE_CHECK_MSG(x.valid(), "RadarOperator::applyAD 需要有效状态");
  VIBE_CHECK_MSG(dy.size() == obs.obs.size(), "RadarOperator::applyAD dy 长度不匹配");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real v = dy[m];
    if (v == Real(0)) continue;
    const Real qr = sample_scalar(s.field(dyn::Species::Qr), g, o.x, o.y, o.z);
    const Real qs = sample_scalar(s.field(dyn::Species::Qs), g, o.x, o.y, o.z);
    const Real qg = sample_scalar(s.field(dyn::Species::Qg), g, o.x, o.y, o.z);
    const Real rho = sample_scalar(s.rho(), g, o.x, o.y, o.z);
    const Real th = sample_scalar(s.theta(), g, o.x, o.y, o.z);
    const Real pi = sample_scalar(s.pi(), g, o.x, o.y, o.z);
    const RadarTerms zt = radar_terms(qr, qs, qg, rho, th * pi);
    Real ddbz = 0;
    dbz_of(zt.z, ddbz);
    const Real w = ddbz * v;
    scatter_adjoint(dx.field(dyn::Species::Qr), g, grid::Stagger::Cell, o.x, o.y, o.z,
                    w * zt.dz_dqr);
    scatter_adjoint(dx.field(dyn::Species::Qs), g, grid::Stagger::Cell, o.x, o.y, o.z,
                    w * zt.dz_dqs);
    scatter_adjoint(dx.field(dyn::Species::Qg), g, grid::Stagger::Cell, o.x, o.y, o.z,
                    w * zt.dz_dqg);
    // dT/dtheta = pi, dT/dpi = theta
    scatter_adjoint(dx.theta(), g, grid::Stagger::Cell, o.x, o.y, o.z, w * zt.dz_dT * pi);
    scatter_adjoint(dx.pi(), g, grid::Stagger::Cell, o.x, o.y, o.z, w * zt.dz_dT * th);
  }
}

// ===========================================================================
// RadianceOperator
// ===========================================================================

namespace {

/// 通道下标解析：优先按 0 基下标，其次按 Channel::id 匹配
int resolve_channel(const RadiativeTransfer& rt, int c) {
  const Size n = rt.n_channels();
  if (n == 0) return -1;
  if (c >= 0 && static_cast<Size>(c) < n) return c;
  for (Size i = 0; i < n; ++i) {
    if (rt.channel(static_cast<int>(i)).id == c) return static_cast<int>(i);
  }
  return 0;
}

}  // namespace

RadianceOperator::RadianceOperator(const grid::Grid& g, const RadiativeTransfer& rt)
    : grid_(&g), rt_(rt) {}

void RadianceOperator::apply(const ModelStateView& x, const ObsSpace& obs,
                             std::vector<Real>& y) const {
  VIBE_CHECK_MSG(x.valid(), "RadianceOperator::apply 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  const Int nz = g.nz();
  y.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const int c = resolve_channel(rt_, o.channel);
    if (c < 0) continue;

    detail::ColumnProfile col;
    col.pressure.resize(static_cast<Size>(nz));
    col.temperature.resize(static_cast<Size>(nz));
    col.qv.resize(static_cast<Size>(nz));
    col.ozone.resize(static_cast<Size>(nz));
    col.cloud_water.resize(static_cast<Size>(nz));
    col.cloud_ice.resize(static_cast<Size>(nz));
    for (Int k = 0; k < nz; ++k) {
      const Real pi = sample_level(s.pi(), g, o.x, o.y, k);
      const Real th = sample_level(s.theta(), g, o.x, o.y, k);
      const Real p = pressure_of(pi);
      col.pressure[static_cast<Size>(k)] = p;
      col.temperature[static_cast<Size>(k)] = th * pi;
      col.qv[static_cast<Size>(k)] = sample_level(s.qv(), g, o.x, o.y, k);
      col.ozone[static_cast<Size>(k)] = default_ozone(p);
      col.cloud_water[static_cast<Size>(k)] =
          sample_level(s.field(dyn::Species::Qc), g, o.x, o.y, k);
      col.cloud_ice[static_cast<Size>(k)] =
          sample_level(s.field(dyn::Species::Qi), g, o.x, o.y, k);
    }
    col.surface_pressure = col.pressure[0];
    col.surface_temperature = col.temperature[0];
    col.skin_temperature = col.temperature[0];
    col.surface_emissivity = Real(0.95);
    col.zenith_angle_deg = Real(0);

    const auto& a = rt_.coefficients()[static_cast<Size>(c)];
    const auto& ch = rt_.channel(c);
    y[m] = detail::forward_column(a, ch, col).brightness_temperature;
  }
}

void RadianceOperator::applyTL(const ModelStateView& x, const dyn::State& dx,
                               const ObsSpace& obs, std::vector<Real>& dy) const {
  VIBE_CHECK_MSG(x.valid(), "RadianceOperator::applyTL 需要有效状态");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  const Int nz = g.nz();
  dy.assign(obs.obs.size(), Real(0));
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const int c = resolve_channel(rt_, o.channel);
    if (c < 0) continue;

    detail::ColumnProfile col;
    col.pressure.resize(static_cast<Size>(nz));
    col.temperature.resize(static_cast<Size>(nz));
    col.qv.resize(static_cast<Size>(nz));
    col.ozone.resize(static_cast<Size>(nz));
    col.cloud_water.assign(static_cast<Size>(nz), Real(0));
    col.cloud_ice.assign(static_cast<Size>(nz), Real(0));
    std::vector<Real> pi_b(static_cast<Size>(nz)), th_b(static_cast<Size>(nz));
    for (Int k = 0; k < nz; ++k) {
      const Real pi = sample_level(s.pi(), g, o.x, o.y, k);
      const Real th = sample_level(s.theta(), g, o.x, o.y, k);
      pi_b[static_cast<Size>(k)] = pi;
      th_b[static_cast<Size>(k)] = th;
      col.pressure[static_cast<Size>(k)] = pressure_of(pi);
      col.temperature[static_cast<Size>(k)] = th * pi;
      col.qv[static_cast<Size>(k)] = sample_level(s.qv(), g, o.x, o.y, k);
      col.ozone[static_cast<Size>(k)] = default_ozone(col.pressure[static_cast<Size>(k)]);
    }
    col.surface_pressure = col.pressure[0];
    col.surface_temperature = col.temperature[0];
    col.skin_temperature = col.temperature[0];
    col.surface_emissivity = Real(0.95);

    const auto& a = rt_.coefficients()[static_cast<Size>(c)];
    const auto& ch = rt_.channel(c);
    const RadianceJacobian jac = detail::jacobian_column(a, ch, col);

    Real acc = Real(0);
    for (Int k = 0; k < nz; ++k) {
      const Size kk = static_cast<Size>(k);
      const Real dpi = sample_level(dx.pi(), g, o.x, o.y, k);
      const Real dth = sample_level(dx.theta(), g, o.x, o.y, k);
      const Real dqv = sample_level(dx.qv(), g, o.x, o.y, k);
      // dBT = dB/dT (pi dtheta + theta dpi) + dB/dqv dqv + dB/dO3 dO3（O3 非状态量）
      acc += jac.temperature[kk] * (pi_b[kk] * dth + th_b[kk] * dpi);
      acc += jac.humidity[kk] * dqv;
    }
    // 地面温度取最低层温度：dTs = pi_0 dtheta_0 + theta_0 dpi_0
    {
      const Real dpi0 = sample_level(dx.pi(), g, o.x, o.y, 0);
      const Real dth0 = sample_level(dx.theta(), g, o.x, o.y, 0);
      acc += jac.surface_temperature * (pi_b[0] * dth0 + th_b[0] * dpi0);
    }
    dy[m] = acc;
  }
}

void RadianceOperator::applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                               const ObsSpace& obs, dyn::State& dx) const {
  VIBE_CHECK_MSG(x.valid(), "RadianceOperator::applyAD 需要有效状态");
  VIBE_CHECK_MSG(dy.size() == obs.obs.size(), "RadianceOperator::applyAD dy 长度不匹配");
  const grid::Grid& g = *x.grid;
  const dyn::State& s = *x.state;
  const Int nz = g.nz();
  for (Size m = 0; m < obs.obs.size(); ++m) {
    const auto& o = obs.obs[m];
    const Real v = dy[m];
    if (v == Real(0)) continue;
    const int c = resolve_channel(rt_, o.channel);
    if (c < 0) continue;

    detail::ColumnProfile col;
    col.pressure.resize(static_cast<Size>(nz));
    col.temperature.resize(static_cast<Size>(nz));
    col.qv.resize(static_cast<Size>(nz));
    col.ozone.resize(static_cast<Size>(nz));
    col.cloud_water.assign(static_cast<Size>(nz), Real(0));
    col.cloud_ice.assign(static_cast<Size>(nz), Real(0));
    std::vector<Real> pi_b(static_cast<Size>(nz)), th_b(static_cast<Size>(nz));
    for (Int k = 0; k < nz; ++k) {
      const Real pi = sample_level(s.pi(), g, o.x, o.y, k);
      const Real th = sample_level(s.theta(), g, o.x, o.y, k);
      pi_b[static_cast<Size>(k)] = pi;
      th_b[static_cast<Size>(k)] = th;
      col.pressure[static_cast<Size>(k)] = pressure_of(pi);
      col.temperature[static_cast<Size>(k)] = th * pi;
      col.qv[static_cast<Size>(k)] = sample_level(s.qv(), g, o.x, o.y, k);
      col.ozone[static_cast<Size>(k)] = default_ozone(col.pressure[static_cast<Size>(k)]);
    }
    col.surface_pressure = col.pressure[0];
    col.surface_temperature = col.temperature[0];
    col.skin_temperature = col.temperature[0];
    col.surface_emissivity = Real(0.95);

    const auto& a = rt_.coefficients()[static_cast<Size>(c)];
    const auto& ch = rt_.channel(c);
    const RadianceJacobian jac = detail::jacobian_column(a, ch, col);

    for (Int k = 0; k < nz; ++k) {
      const Size kk = static_cast<Size>(k);
      const Real wT = jac.temperature[kk] * v;
      scatter_level(dx.theta(), g, o.x, o.y, k, wT * pi_b[kk]);
      scatter_level(dx.pi(), g, o.x, o.y, k, wT * th_b[kk]);
      scatter_level(dx.qv(), g, o.x, o.y, k, jac.humidity[kk] * v);
    }
    {
      const Real wT = jac.surface_temperature * v;
      scatter_level(dx.theta(), g, o.x, o.y, 0, wT * pi_b[0]);
      scatter_level(dx.pi(), g, o.x, o.y, 0, wT * th_b[0]);
    }
  }
}

}  // namespace vibe::obs

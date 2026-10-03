/// @file microphysics_kessler.cpp
/// @brief Kessler 暖雨方案 [P1]，以及 microphysics.hpp 中的公共相变工具与方案工厂。
///
/// 说明：交付清单中没有独立的 microphysics.cpp，因此**公共工具与工厂**
/// （particle_mass_rate / ventilation_factor / slope_parameter /
/// saturation_adjust_* / MicrophysicsBase 的缺省实现 / make_microphysics）
/// 一并实现在本文件中，实现与声明一一对应。
///
/// 文献：[P1] Kessler (1969)；[P2] Lin et al. (1983)；[D16] WRF ARW 技术说明。

#include "vibe/physics/microphysics.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

#include "vibe/common/error.hpp"

namespace vibe::physics {

namespace {
constexpr Real kDynViscosityAir = Real(1.8e-5);  ///< 空气动力粘性 mu (Pa s)
constexpr Real kSchmidtNumber = Real(0.6);        ///< 水汽 Schmidt 数
}  // namespace

// ===========================================================================
// 公共相变与粒子谱工具
// ===========================================================================

// ---------------------------------------------------------------------------
// Maxwell-Mason 相变质量速率
//   dm/dt = 4 pi r f_v (rho_v - rho_vs) / [1 + L^2 D_v rho_vs/(K_a R_v T^2)]
// 离散化：解析式；rho_v = rho qv，rho_vs = rho qs。
// 文献：[P2] 式 (A1)-(A4)；[P3] 式 (A2)-(A6)。
// 复杂度：O(1)。
Real particle_mass_rate(Real t, Real p, Real rho, Real qv, Real qs, Real radius,
                        bool ice_phase, Real ventilation) noexcept {
  VIBE_UNUSED(p);
  if (!(radius > Real(0)) || !(rho > Real(0)) || !(t > Real(0))) return Real(0);
  const Real rho_v = qv * rho;
  const Real rho_vs = qs * rho;
  const Real l = ice_phase ? latent_heat_sublimation(t) : latent_heat_vaporization(t);
  const Real denom =
      Real(1) + l * l * kDvWater * rho_vs / (kThermCondAir * kRv * t * t);
  return Real(4) * kPi * radius * ventilation * (rho_v - rho_vs) / denom;
}

// ---------------------------------------------------------------------------
// 通风因子
//   f_v = 0.78 + 0.308 Sc^(1/3) Re^(1/2),  Re = 2 r V rho / mu
// 文献：[P2] 式 (A6)；[P3]。
// 复杂度：O(1)。
Real ventilation_factor(Real radius, Real fall_speed, Real rho) noexcept {
  if (!(radius > Real(0))) return Real(1);
  const Real re = Real(2) * radius * std::max(fall_speed, Real(0)) * rho / kDynViscosityAir;
  const Real fv = Real(0.78) + Real(0.308) * std::cbrt(kSchmidtNumber) * std::sqrt(std::max(re, Real(0)));
  return std::max(fv, Real(1));
}

// ---------------------------------------------------------------------------
// 指数谱斜率
//   N(D) = N0 exp(-lambda D), m(D) = a D^b
//   lambda = ( a N0 Gamma(1+b) / (rho q) )^(1/b)
// 文献：[P2] 式 (2)；[P3] 表 1。
// 复杂度：O(1)（一次 tgamma + pow）。
Real slope_parameter(Real rho, Real q, Real n0, Real a, Real b) noexcept {
  const Real qq = std::max(q, Real(1.0e-14));
  const Real num = a * n0 * std::tgamma(Real(1) + b);
  const Real den = std::max(rho * qq, Real(1.0e-20));
  return std::pow(num / den, Real(1) / b);
}

Real number_concentration(Real n0, Real lambda) noexcept {
  return n0 / std::max(lambda, Real(1.0e-8));
}

// ---------------------------------------------------------------------------
// 质量加权平均 D^d
//   <D^d>_m = Gamma(4+d)/Gamma(4) lambda^(-d) = Gamma(4+d)/6 * lambda^(-d)
// 文献：[P2] 式 (3)。
// 复杂度：O(1)。
Real mass_weighted_diameter_power(Real lambda, Real d) noexcept {
  const Real lam = std::max(lambda, Real(1.0e-8));
  return std::tgamma(Real(4) + d) / Real(6) * std::pow(lam, -d);
}

// ===========================================================================
// 饱和调整
// ===========================================================================

// ---------------------------------------------------------------------------
// 单点饱和调整（Newton，隐式耦合凝结潜热）
//   qv - delta = q_s( p, T0 + (L/cp) delta ),   delta ∈ [-qc, qv]
//   f(delta)  = qv - delta - qs(T)
//   f'(delta) = -1 - (L/cp) dqs/dT,   dqs/dT = L qs/(Rv T^2)
// 离散化：Newton 迭代，收敛判据 |f| <= tol * max(qs, 1e-8)。
// 文献：[P1]；[D16] 第 2 章。
// 复杂度：O(n_iter) ≈ O(3)。
SaturationAdjust saturation_adjust_point(Real& qv, Real& qc, Real& theta, Real p,
                                         bool ice_phase, int max_iter, Real tol) noexcept {
  SaturationAdjust out;
  qv = std::max(qv, Real(0));
  qc = std::max(qc, Real(0));
  const Real pi = exner_from_pressure(p);
  Real t = std::max(theta * pi, Real(150));
  Real qs = saturation_mixing_ratio(p, t, ice_phase);
  Real l = ice_phase ? latent_heat_sublimation(t) : latent_heat_vaporization(t);

  // 线性化初值： delta0 = (qv - qs) / (1 + L^2 qs/(cp Rv T^2))
  const Real denom0 = Real(1) + l * l * qs / (kCp * kRv * t * t);
  Real delta = clamp((qv - qs) / denom0, -qc, qv);
  if (std::abs(delta) < Real(1.0e-16)) {
    out.residual = std::abs(qv - qs) / std::max(qs, Real(1.0e-12));
    return out;
  }

  for (int it = 1; it <= std::max(max_iter, 1); ++it) {
    out.iterations = it;
    t = theta * pi + (l / kCp) * delta;
    t = std::max(t, Real(150));
    qs = saturation_mixing_ratio(p, t, ice_phase);
    l = ice_phase ? latent_heat_sublimation(t) : latent_heat_vaporization(t);
    const Real f = qv - delta - qs;
    const Real dqsdT = l * qs / (kRv * t * t);
    const Real df = Real(-1) - (l / kCp) * dqsdT;
    out.residual = std::abs(f) / std::max(qs, Real(1.0e-12));
    if (out.residual <= tol) break;
    const Real step = -f / df;
    const Real trial = clamp(delta + clamp(step, Real(-1.0), Real(1.0)), -qc, qv);
    if (trial == delta) break;  // 已顶到边界（层内水物质不足）
    delta = trial;
  }

  qv -= delta;
  qc += delta;
  theta += l * delta / (kCp * pi);
  qv = std::max(qv, Real(0));
  qc = std::max(qc, Real(0));
  out.condensed = delta;
  out.dtheta = l * delta / (kCp * pi);
  out.residual = std::abs(qv - qs) / std::max(qs, Real(1.0e-12));
  return out;
}

SaturationAdjust saturation_adjust_column(PhysicsColumn& col, Int k0, Int k1,
                                         bool ice_phase) noexcept {
  SaturationAdjust total;
  const Int lo = clamp(k0, Int(0), col.nz());
  const Int hi = clamp(k1, Int(0), col.nz());
  for (Int k = lo; k < hi; ++k) {
    const Size s = static_cast<Size>(k);
    const SaturationAdjust r =
        saturation_adjust_point(col.qv[s], col.qc[s], col.theta[s], col.p[s], ice_phase, 10, Real(1.0e-10));
    total.condensed += r.condensed;
    total.dtheta += r.dtheta;
    total.iterations = std::max(total.iterations, r.iterations);
    total.residual = std::max(total.residual, r.residual);
  }
  return total;
}

// ===========================================================================
// MicrophysicsBase 缺省实现
// ===========================================================================

Real MicrophysicsBase::max_fall_speed(const PhysicsColumn& col) const {
  VIBE_UNUSED(col);
  return Real(0);
}

std::string MicrophysicsBase::describe() const {
  std::ostringstream os;
  os << name() << "{ qv_min=" << opt_.qv_min << " qx_min=" << opt_.qx_min
     << " 正定保护=" << (opt_.positive_definite ? "开" : "关")
     << " 干大气=" << (opt_.dry_atmosphere ? "开" : "关") << " }";
  return os.str();
}

// ===========================================================================
// Kessler 暖雨 [P1]
// ===========================================================================

// ---------------------------------------------------------------------------
// 雨滴落速
//   V_t = a (rho q_r)^b,  a = 36.34, b = 0.1364  (SI, [D16] 第 2 章)
// 复杂度：O(1)。
Real KesslerMicrophysics::rain_fall_speed(Real rho, Real qr) const noexcept {
  const Real load = std::max(rho * std::max(qr, Real(0)), Real(0));
  return opt_.kessler_rain_a * std::pow(load, opt_.kessler_rain_b);
}

// 自动转换： P_auto = k1 max(qc - qc0, 0)   [P1]
Real KesslerMicrophysics::autoconversion(Real qc) const noexcept {
  return opt_.kessler_k1 * std::max(qc - opt_.kessler_qc0, Real(0));
}

// 收集： P_acc = k2 qc qr  [P1]（连续收集的简化形式）
Real KesslerMicrophysics::accretion(Real qc, Real qr) const noexcept {
  return opt_.kessler_k2 * std::max(qc, Real(0)) * std::max(qr, Real(0));
}

// ---------------------------------------------------------------------------
// 雨水蒸发（Maxwell-Mason 群体积分）
//   dqr/dt|_evap = - N_r (dm/dt)|_r / rho
//   取指数谱： lambda_r = (pi rho_w N0/(rho qr))^(1/3), N_r = N0/lambda_r,
//              r_mean = 1/(2 lambda_r)
//   dqr/dt = (4 pi N_r r_mean / rho) (rho_v - rho_vs) f_v /
//            [1 + Lv^2 Dv rho_vs/(Ka Rv T^2)]
// 正值 = 蒸发（qv < qs 时 rho_v - rho_vs < 0，故取负号返回）
// 文献：[P2] 式 (A1)-(A6)；[P1]。
// 复杂度：O(1)。
Real KesslerMicrophysics::rain_evaporation_rate(Real p, Real t, Real rho, Real qv, Real qs,
                                                Real qr) const noexcept {
  if (!(qr > Real(0)) || !(rho > Real(0))) return Real(0);
  const Real lambda = slope_parameter(rho, qr, kRainIntercept, rain_mass_a(), kRainMassB);
  const Real nr = number_concentration(kRainIntercept, lambda);
  const Real r_mean = Real(0.5) / std::max(lambda, Real(1.0e-8));
  const Real vt = rain_fall_speed(rho, qr);
  const Real fv = ventilation_factor(r_mean, vt, rho);
  const Real dm = particle_mass_rate(t, p, rho, qv, qs, r_mean, false, fv);  // < 0 为蒸发
  // 单位体积粒子数 N_r 乘以单粒子质量速率，再除以 rho 得到混合比速率
  const Real rate = (-dm) * nr / rho;
  return std::max(opt_.kessler_evap_coef * rate, Real(0));
}

// ---------------------------------------------------------------------------
// 雨水沉降：上游通量法 + 可用质量限制
//
// 离散化（自上而下扫描，k = 0 为最低层，故循环方向为 k = 0..nz-1 时
// 需要从**上层**开始，这里改用 k = nz-1 -> 0 的顺序把"上方入流"传入下层）：
//     F_in(0) = 0
//     available_k = rho_k qr_k dz_k + F_in,k dt
//     F_out,k      = min( rho_k qr_k Vt_k dt , available_k )
//     qr_k^{n+1}   = (available_k - F_out,k) / (rho_k dz_k)
// 性质：(a) 无条件正定；(b) 列内质量守恒（离开列底的通量即降水）；
//       (c) CFL<1 时退化为标准一阶上游格式。
// 文献：[P1]；[B6] 第 4 章（一阶上游通量法）。
// 复杂度：O(nz)。
Real KesslerMicrophysics::sediment_rain(PhysicsColumn& col, Real dt) const {
  const Int n = col.nz();
  Real flux_in = Real(0);  // 从上方进入当前层的质量通量 (kg/(m^2 s))，自下而上传递
  // 自下而上扫描：层 k 的入流来自层 k+1
  for (Int k = n - 1; k >= 0; --k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    const Real dz = std::max(col.dz[s], Real(1.0e-6));
    const Real qr = std::max(col.qr[s], Real(0));
    const Real mass = rho * qr * dz;                    // kg/m^2
    const Real vt = rain_fall_speed(rho, qr);
    const Real available = mass + flux_in * dt;
    const Real out = std::min(rho * qr * vt * dt, std::max(available, Real(0)));
    col.qr[s] = std::max(available - out, Real(0)) / (rho * dz);
    flux_in = out / dt;
  }
  return flux_in * dt;  // 离开列底的质量 (kg/m^2)
}

// ---------------------------------------------------------------------------
// Kessler 单列积分
//
// 步骤（全部显式欧拉，dt 一阶；饱和调整用 Newton 隐式）：
//   1. 饱和调整：    qv -> qc（凝结潜热回写 theta）
//   2. 云水->雨水：  P = k1 max(qc-qc0,0) + k2 qc qr
//   3. 雨水沉降：    上游通量法
//   4. 雨水蒸发：    Maxwell-Mason，受次饱和亏缺与可用雨水限制
//   5. 正定保护与守恒检查
// 复杂度：O(nz)。
Real KesslerMicrophysics::step_column(PhysicsColumn& col, Real dt, PhysicsDiagnostics& diag) {
  if (!(dt > Real(0))) return Real(0);
  const Int n = col.nz();
  if (n <= 0) return Real(0);
  if (opt_.dry_atmosphere) return Real(0);

  const Real mass_before = col.total_water_mass();  // kg/m^2

  // ---- 1. 饱和调整（云水蒸发/凝结；Kessler 为暖雨，不使用冰相） ----
  saturation_adjust_column(col, 0, n, /*ice_phase=*/false);

  // ---- 2. 云水 -> 雨水（自动转换 + 收集） ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    Real qc = std::max(col.qc[s], Real(0));
    Real qr = std::max(col.qr[s], Real(0));
    const Real rate = autoconversion(qc) + accretion(qc, qr);
    const Real dq = std::min(rate * dt, qc);
    col.qc[s] = qc - dq;
    col.qr[s] = qr + dq;
  }

  // ---- 3. 雨水沉降 ----
  const Real precip_mass = sediment_rain(col, dt);  // kg/m^2

  // ---- 4. 雨水蒸发 ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real qr = std::max(col.qr[s], Real(0));
    if (qr <= Real(0)) continue;
    const Real t = col.temperature(k);
    const Real pi = col.exner(k);
    const Real qs = saturation_mixing_ratio(col.p[s], t);
    const Real qv = std::max(col.qv[s], Real(0));
    if (qv >= qs) continue;
    Real evap = rain_evaporation_rate(col.p[s], t, col.rho[s], qv, qs, qr) * dt;
    // 限制 1：不超过可用雨水
    evap = std::min(evap, qr);
    // 限制 2：不使层内过饱和（给潜热反馈留出余量）
    const Real lv = latent_heat_vaporization(t);
    const Real cap = (qs - qv) / (Real(1) + lv * lv * qs / (kCp * kRv * t * t));
    evap = std::min(evap, std::max(cap, Real(0)));
    if (!(evap > Real(0))) continue;
    col.qr[s] = qr - evap;
    col.qv[s] = qv + evap;
    col.theta[s] = col.theta[s] - lv * evap / (kCp * pi);
  }

  // ---- 5. 正定保护与守恒检查 ----
  if (opt_.positive_definite) {
    for (Int k = 0; k < n; ++k) {
      const Size s = static_cast<Size>(k);
      col.qv[s] = std::max(col.qv[s], Real(0));
      col.qc[s] = std::max(col.qc[s], Real(0));
      col.qr[s] = std::max(col.qr[s], Real(0));
    }
  }
  const Real mass_after = col.total_water_mass() + precip_mass;
  diag.mass_conservation_residual =
      std::max(diag.mass_conservation_residual, relative_water_residual(mass_before, mass_after));

  // 1 mm/s == 1 kg/(m^2 s)
  return precip_mass / dt;
}

Real KesslerMicrophysics::max_fall_speed(const PhysicsColumn& col) const {
  Real vmax = Real(0);
  const Int n = col.nz();
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    vmax = std::max(vmax, rain_fall_speed(col.rho[s], col.qr[s]));
  }
  return vmax;
}

std::string KesslerMicrophysics::describe() const {
  std::ostringstream os;
  os << "KesslerMicrophysics{ qc0=" << opt_.kessler_qc0 << " kg/kg k1=" << opt_.kessler_k1
     << " 1/s k2=" << opt_.kessler_k2 << " Vt=" << opt_.kessler_rain_a << " (rho qr)^"
     << opt_.kessler_rain_b << " 蒸发系数=" << opt_.kessler_evap_coef
     << " N0r=" << KesslerMicrophysics::kRainIntercept << " m^-4 }";
  return os.str();
}

// ===========================================================================
// 工厂
// ===========================================================================

std::unique_ptr<MicrophysicsBase> make_microphysics(MicrophysicsScheme scheme,
                                                    const PhysicsOptions& opt) {
  switch (scheme) {
    case MicrophysicsScheme::None:
      return nullptr;
    case MicrophysicsScheme::Kessler:
      return std::make_unique<KesslerMicrophysics>(opt);
    case MicrophysicsScheme::Thompson:
      return std::make_unique<ThompsonMicrophysics>(opt);
    case MicrophysicsScheme::Morrison:
      throw NotImplemented("Morrison 双参数微物理 [P4] 仅保留接口，尚未实现");
    case MicrophysicsScheme::Wsm6:
      throw NotImplemented("WSM6 微物理 [P2] 仅保留接口，尚未实现");
    default:
      throw NotImplemented("未知的微物理方案编号");
  }
}

}  // namespace vibe::physics

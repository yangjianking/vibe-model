/// @file microphysics_thompson.cpp
/// @brief Thompson et al. (2008) 6 类微物理参数化 [P3]。
///
/// 水物质类别：水汽 qv、云水 qc、雨水 qr、云冰 qi、雪 qs、霰 qg。
///
/// 粒子谱约定（[P3] 表 1）
///     N(D) = N0 exp(-lambda D)                 指数谱
///     m(D) = a D^b        (雪 a=0.069 b=2；雨/冰/霰 b=3)
///     V(D) = c D^d        (雪 40 D^0.55；霰 442 D^0.89；雨 4854 D e^{-195 D})
///     斜率  lambda = (a N0 Gamma(1+b)/(rho q))^(1/b)
///     <D^d>_m = Gamma(4+d)/Gamma(4) lambda^(-d)
/// 落速含空气密度修正 (rho0/rho)^0.54（[P3] 式 (A14)）。
///
/// 时间离散：所有源汇项为显式向前欧拉（对 dt 一阶），饱和调整用 Newton
/// 隐式迭代，沉降用上游通量法 + 可用质量限制（无条件正定）。每个子步之后
/// 立即做非负截断，保证 qx >= 0。
///
/// 文献：[P3] Thompson et al. (2008)；[P2] Lin et al. (1983)；
///       [P4] Morrison et al. (2005)（双参数扩展的接口参考）。

#include "vibe/physics/microphysics.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"

namespace vibe::physics {

namespace {

constexpr Real kRho0Ref = Real(1.225);      ///< 参考空气密度 (kg/m^3)
constexpr Real kDensityExp = Real(0.54);    ///< 落速密度修正指数 [P3]
constexpr Real kQiSeed = Real(1.0e-6);      ///< 冰自动转换阈值 (kg/kg)

// ---------------------------------------------------------------------------
// 通用沉降：上游通量法 + 可用质量限制（无条件正定）
//     available_k = rho_k q_k dz_k + F_in,k dt
//     out_k       = min( rho_k q_k Vt_k dt , available_k )
//     q_k^{n+1}   = (available_k - out_k)/(rho_k dz_k)
// 自**顶**向下传递入流（k = n-1 -> 0）。
// 文献：[B6] 第 4 章（一阶上游通量法）。
// 复杂度：O(nz)。
Real sediment_species(std::vector<Real>& q, const std::vector<Real>& rho,
                      const std::vector<Real>& dz, const std::vector<Real>& vt,
                      Real dt) {
  const Int n = static_cast<Int>(q.size());
  Real flux_in = Real(0);  // 上层进入本层的通量 kg/(m^2 s)
  for (Int k = n - 1; k >= 0; --k) {
    const Size s = static_cast<Size>(k);
    const Real r = std::max(rho[s], Real(1.0e-6));
    const Real d = std::max(dz[s], Real(1.0e-6));
    const Real qq = std::max(q[s], Real(0));
    const Real available = r * qq * d + flux_in * dt;
    const Real out = std::min(r * qq * std::max(vt[s], Real(0)) * dt, std::max(available, Real(0)));
    q[s] = std::max(available - out, Real(0)) / (r * d);
    flux_in = out / dt;
  }
  return flux_in * dt;  // 离开列底的质量 (kg/m^2)
}

}  // namespace

// ===========================================================================
// 粒子谱与落速
// ===========================================================================

// lambda_s = (a_s N0s Gamma(1+b_s)/(rho qs))^(1/b_s)   [P3] 表 1
Real ThompsonMicrophysics::snow_lambda(Real rho, Real qs) const noexcept {
  return slope_parameter(rho, qs, kSnowIntercept, kSnowMassA, kSnowMassB);
}
// lambda_g = (a_g N0g Gamma(4)/(rho qg))^(1/3)
Real ThompsonMicrophysics::graupel_lambda(Real rho, Real qg) const noexcept {
  return slope_parameter(rho, qg, kGraupelIntercept, kGraupelMassA, kGraupelMassB);
}
// lambda_r = (pi rho_w N0r/(rho qr))^(1/3)
Real ThompsonMicrophysics::rain_lambda(Real rho, Real qr) const noexcept {
  return slope_parameter(rho, qr, KesslerMicrophysics::kRainIntercept,
                         KesslerMicrophysics::rain_mass_a(), KesslerMicrophysics::kRainMassB);
}

// ---------------------------------------------------------------------------
// 雪落速：V = c <D^d>_m (rho0/rho)^0.54,  c = 40, d = 0.55   [P3] 表 1
// 复杂度 O(1)（一次 tgamma/pow）。
Real ThompsonMicrophysics::snow_fall_speed(Real rho, Real qs) const noexcept {
  if (!(qs > Real(1.0e-12))) return Real(0);
  const Real lambda = snow_lambda(rho, qs);
  const Real v = mass_weighted_fall_speed(lambda, kSnowFallC, kSnowFallD);
  return v * std::pow(kRho0Ref / std::max(rho, Real(1.0e-3)), kDensityExp);
}

// 霰落速：V = 442 <D^0.89>_m (rho0/rho)^0.54   [P3]
Real ThompsonMicrophysics::graupel_fall_speed(Real rho, Real qg) const noexcept {
  if (!(qg > Real(1.0e-12))) return Real(0);
  const Real lambda = graupel_lambda(rho, qg);
  const Real v = mass_weighted_fall_speed(lambda, kGraupelFallC, kGraupelFallD);
  return v * std::pow(kRho0Ref / std::max(rho, Real(1.0e-3)), kDensityExp);
}

// ---------------------------------------------------------------------------
// 雨落速：对 V(D) = 4854 D exp(-195 D) 作质量加权平均
//     <V>_m = 4854 * Gamma(5)/Gamma(4) * lambda^4/(lambda+195)^5
//           = 4854 * 4 lambda^4/(lambda+195)^5
// 含密度修正 (rho0/rho)^0.54。
// 复杂度 O(1)。
Real ThompsonMicrophysics::rain_fall_speed(Real rho, Real qr) const noexcept {
  if (!(qr > Real(1.0e-12))) return Real(0);
  const Real lambda = rain_lambda(rho, qr);
  const Real den = std::pow(lambda + kRainFallB, Real(5));
  const Real v = kRainFallA * Real(4) * std::pow(lambda, Real(4)) / std::max(den, Real(1.0e-30));
  return v * std::pow(kRho0Ref / std::max(rho, Real(1.0e-3)), kDensityExp);
}

// ===========================================================================
// 源汇项参数化
// ===========================================================================

// ---------------------------------------------------------------------------
// 云水 -> 雨水自动转换（基于体积平均半径的松弛式，[P3] Berry-Reinhardt 简化）
//     r_vol  = (3 rho qc/(4 pi rho_w Nc))^(1/3)
//     r_crit = r_0 [1 + 0.02 (T - T0)]，裁剪到 [0.5 r0, 2 r0]
//     P_ra   = (qc/tau) max(0, 1 - r_crit/r_vol)
// 离散化：解析式；tau = thompson_autoconv_time（默认 1000 s）。
// 文献：[P3] 第 3 节；[P4] 第 2 节。
// 复杂度 O(1)。
Real ThompsonMicrophysics::thompson_autoconversion(Real rho, Real qc, Real t) const noexcept {
  if (!(qc > Real(0)) || !(rho > Real(0))) return Real(0);
  const Real nc = std::max(opt_.thompson_nc, Real(1.0));
  const Real r_vol = std::cbrt(Real(3) * rho * qc / (Real(4) * kPi * kRhoWater * nc));
  const Real r0 = opt_.thompson_autoconv_radius_um * Real(1.0e-6);
  const Real r_crit = clamp(r0 * (Real(1) + Real(0.02) * (t - kT0)), Real(0.5) * r0, Real(2) * r0);
  const Real drive = Real(1) - r_crit / std::max(r_vol, Real(1.0e-9));
  if (drive <= Real(0)) return Real(0);
  return qc / std::max(opt_.thompson_autoconv_time, Real(1)) * drive;
}

// ---------------------------------------------------------------------------
// Bergeron 过程（[P3] 式 (A9)）
//     beta = c max(qv/qsi - 1, 0),  c = thompson_bergeron_rate (0.0059 1/s)
//     云水 -> 云冰 转移比例 = 1 - exp(-beta dt)
// 复杂度 O(1)。
Real ThompsonMicrophysics::bergeron_rate(Real qv, Real qsi) const noexcept {
  const Real q = std::max(qsi, Real(1.0e-12));
  return opt_.thompson_bergeron_rate * std::max(qv / q - Real(1), Real(0));
}

// ---------------------------------------------------------------------------
// 云水冻结（均匀冻结 + 浸泡冻结）
//     T <= 233.15 K            : 均匀冻结，速率 = qc * 1 s^-1（准瞬时）
//     233.15 < T < 273.15 K    : P = (qc/tau_f) [exp(A'(T0-T)) - 1], tau_f = 1e4 s
// 系数 A' = thompson_bigg_a = 0.66 K^-1（Bigg 型温度依赖，[P3] 式 (A17)）。
// 复杂度 O(1)。
Real ThompsonMicrophysics::cloud_water_freezing_rate(Real t, Real qc) const noexcept {
  if (!(qc > Real(0)) || t >= kT0) return Real(0);
  if (t <= Real(233.15)) return qc;  // 均匀冻结：1 s^-1
  const Real tau_f = Real(1.0e4);
  const Real drive = std::exp(opt_.thompson_bigg_a * (kT0 - t)) - Real(1);
  return std::max(qc / tau_f * drive, Real(0));
}

// ---------------------------------------------------------------------------
// 雨水冻结（Bigg 1953，[P3] 式 (A17)）
//     P = (pi^2 rho_w B'/(12 rho)) [exp(A'(T0-T)) - 1] * Integral_0^inf D^6 N(D) dD
//     Integral D^6 N(D) dD = N0r Gamma(7) lambda_r^(-7) = N0r * 720 lambda_r^{-7}
// 系数：A' = thompson_bigg_a = 0.66 1/K，B' = thompson_bigg_b = 100 m^-3 s^-1。
// 复杂度 O(1)。
Real ThompsonMicrophysics::rain_freezing_rate(Real t, Real rho, Real qr) const noexcept {
  if (!(qr > Real(0)) || t >= kT0 || !(rho > Real(0))) return Real(0);
  const Real lambda = rain_lambda(rho, qr);
  const Real mom6 = KesslerMicrophysics::kRainIntercept * std::tgamma(Real(7)) *
                    std::pow(std::max(lambda, Real(1.0e-8)), Real(-7));
  const Real drive = std::exp(opt_.thompson_bigg_a * (kT0 - t)) - Real(1);
  const Real coef = kPi * kPi * kRhoWater * opt_.thompson_bigg_b / Real(12);
  return std::max(coef * drive * mom6 / rho, Real(0));
}

// ---------------------------------------------------------------------------
// 雪/霰融化（Maxwell-Mason 热量收支，[P2] 式 (A20)；[P3] 式 (A22)）
//     dm/dt = 4 pi r f_v [K_a (T - T0) + L_v D_v rho (q_v - q_{s,0})] / L_f
//     dq/dt = N <dm/dt>/rho
// 温度低于三相点时返回 0。q_{s,0} 为 T0 处的饱和混合比。
// 复杂度 O(1)。
namespace {
// 通用融化速率： dq/dt = N * 4 pi r f_v [Ka (T-T0) + Lv Dv rho (qv-qs0)] / (Lf rho)
// 复杂度 O(1)。
Real melting_rate_common(Real t, Real p, Real rho, Real qv, Real diameter, Real number,
                         Real ventilation) {
  if (!(t > kT0) || !(number > Real(0)) || !(rho > Real(0))) return Real(0);
  const Real qs0 = saturation_mixing_ratio(p, kT0);
  const Real lv = latent_heat_vaporization(kT0);
  const Real flux = kThermCondAir * (t - kT0) + lv * kDvWater * rho * (qv - qs0);
  const Real dm = Real(4) * kPi * Real(0.25) * diameter * ventilation * flux / kLf;
  return std::max(dm * number / rho, Real(0));
}
}  // namespace

Real ThompsonMicrophysics::snow_melting_rate(Real t, Real p, Real rho, Real qv,
                                             Real qsnow) const noexcept {
  if (!(qsnow > Real(0))) return Real(0);
  const Real lambda = snow_lambda(rho, qsnow);
  const Real d_mean = mass_weighted_diameter_power(lambda, kSnowMassB);  // <D>_m
  const Real number = number_concentration(kSnowIntercept, lambda);
  const Real v = snow_fall_speed(rho, qsnow);
  const Real fv = ventilation_factor(Real(0.5) * d_mean, v, rho);
  return melting_rate_common(t, p, rho, qv, d_mean, number, fv);
}

Real ThompsonMicrophysics::graupel_melting_rate(Real t, Real p, Real rho, Real qv,
                                                Real qgraupel) const noexcept {
  if (!(qgraupel > Real(0))) return Real(0);
  const Real lambda = graupel_lambda(rho, qgraupel);
  const Real d_mean = mass_weighted_diameter_power(lambda, kGraupelMassB);
  const Real number = number_concentration(kGraupelIntercept, lambda);
  const Real v = graupel_fall_speed(rho, qgraupel);
  const Real fv = ventilation_factor(Real(0.5) * d_mean, v, rho);
  return melting_rate_common(t, p, rho, qv, d_mean, number, fv);
}

// ---------------------------------------------------------------------------
// 沉积/凝华（Maxwell-Mason，冰面饱和比，[P2] 式 (A4)）
//     dq/dt = N * dm/dt|_{ice} / rho
//     dm/dt = 4 pi r f_v (rho qv - rho qsi)/[1 + Ls^2 Dv rho qsi/(Ka Rv T^2)]
// 复杂度 O(1)。
Real ThompsonMicrophysics::deposition_rate(Real t, Real p, Real rho, Real qv, Real qsat_ice,
                                           Real number, Real diameter, bool snow) const noexcept {
  if (!(number > Real(0)) || !(rho > Real(0))) return Real(0);
  const Real r = Real(0.5) * std::max(diameter, Real(1.0e-8));
  const Real fv = ventilation_factor(r, Real(0.1), rho);  // 沉积项的通风影响较弱
  const Real dm = particle_mass_rate(t, p, rho, qv, qsat_ice, r, /*ice_phase=*/true, fv);
  VIBE_UNUSED(snow);
  return dm * number / rho;
}

// ===========================================================================
// 单列积分
// ===========================================================================

// ---------------------------------------------------------------------------
// Thompson 6 类单列积分。
//
// 步骤（每步后立即正定截断）：
//   1. 混合相态饱和调整 + Bergeron（T < T0 时云水 -> 云冰，随后冰面饱和调整）
//   2. 云水 -> 雨水自动转换
//   3. 雨水收集云水
//   4. 云水冻结、雨水冻结（Bigg）
//   5. 云冰 -> 雪自动转换
//   6. 雪淞附 -> 霰（riming）
//   7. 雪/霰 沉积-凝华
//   8. 雪/霰 融化（T > T0）
//   9. 雨/雪/霰 沉降
//  10. 守恒检查（含地面降水累积）
//
// 复杂度：O(nz * (n_iter + 常数))。
Real ThompsonMicrophysics::step_column(PhysicsColumn& col, Real dt, PhysicsDiagnostics& diag) {
  if (!(dt > Real(0))) return Real(0);
  const Int n = col.nz();
  if (n <= 0) return Real(0);
  if (opt_.dry_atmosphere) return Real(0);

  const Real mass_before = col.total_water_mass();

  // ---- 1. 混合相态饱和调整 + Bergeron ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    Real t = col.temperature(k);
    const Real pi = col.exner(k);
    if (t >= kT0) {
      saturation_adjust_point(col.qv[s], col.qc[s], col.theta[s], col.p[s], /*ice_phase=*/false);
    } else {
      // Bergeron：冰面过饱和时云水向云冰转化（[P3] 式 (A9)）
      const Real qsi = saturation_mixing_ratio(col.p[s], t, /*ice_phase=*/true);
      const Real beta = bergeron_rate(col.qv[s], qsi);
      if (beta > Real(0) && col.qc[s] > Real(0)) {
        const Real frac = Real(1) - std::exp(-beta * dt);
        const Real dq = std::min(std::max(col.qc[s], Real(0)) * frac, std::max(col.qc[s], Real(0)));
        col.qc[s] -= dq;
        col.qi[s] += dq;
        col.theta[s] += latent_heat_sublimation(t) * dq / (kCp * pi);
      }
      // 冰面饱和调整（云冰的凝华/升华）
      saturation_adjust_point(col.qv[s], col.qi[s], col.theta[s], col.p[s], /*ice_phase=*/true);
    }
  }

  // ---- 2. 云水 -> 雨水自动转换；3. 雨水收集云水 ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    const Real t = col.temperature(k);
    Real qc = std::max(col.qc[s], Real(0));
    Real qr = std::max(col.qr[s], Real(0));
    const Real auto_rate = thompson_autoconversion(rho, qc, t);
    const Real acc_rate = opt_.kessler_k2 * qc * qr;  // 连续收集（雨-云落速差隐含其中）
    const Real dq = std::min((auto_rate + acc_rate) * dt, qc);
    col.qc[s] = qc - dq;
    col.qr[s] = qr + dq;
  }

  // ---- 4. 冻结：云水冻结 + 雨水冻结（Bigg） ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    const Real t = col.temperature(k);
    const Real pi = col.exner(k);
    if (t < kT0) {
      const Real lf = kLf;
      // 云水 -> 云冰
      const Real r_cw = cloud_water_freezing_rate(t, std::max(col.qc[s], Real(0)));
      const Real dqc = std::min(r_cw * dt, std::max(col.qc[s], Real(0)));
      if (dqc > Real(0)) {
        col.qc[s] -= dqc;
        col.qi[s] += dqc;
        col.theta[s] += lf * dqc / (kCp * pi);
      }
      // 雨水 -> 霰
      const Real r_rn = rain_freezing_rate(t, rho, std::max(col.qr[s], Real(0)));
      const Real dqr = std::min(r_rn * dt, std::max(col.qr[s], Real(0)));
      if (dqr > Real(0)) {
        col.qr[s] -= dqr;
        col.qg[s] += dqr;
        col.theta[s] += lf * dqr / (kCp * pi);
      }
    }
  }

  // ---- 5. 云冰 -> 雪自动转换；6. 雪淞附 -> 霰 ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    // 云冰自动转换：冰晶长大到落速显著（超过阈值部分）
    const Real qi = std::max(col.qi[s], Real(0));
    if (qi > kQiSeed) {
      const Real dqi = std::min(opt_.thompson_ice_autoconv * (qi - kQiSeed) * dt, qi);
      col.qi[s] = qi - dqi;
      col.qs[s] = std::max(col.qs[s], Real(0)) + dqi;
    }
    // 雪淞附：云水冻结到雪表面形成霰（riming），密度因子 (rho0/rho)^0.5
    const Real qc = std::max(col.qc[s], Real(0));
    const Real qs = std::max(col.qs[s], Real(0));
    if (qc > Real(0) && qs > Real(0)) {
      const Real rim = opt_.thompson_riming_coef * qc * qs * std::sqrt(kRho0Ref / rho);
      const Real dq = std::min(rim * dt, qc);
      col.qc[s] = qc - dq;
      col.qg[s] = std::max(col.qg[s], Real(0)) + dq;
      col.theta[s] += kLf * dq / (kCp * col.exner(k));
    }
  }

  // ---- 7. 沉积/凝华（雪、霰） ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    const Real t = col.temperature(k);
    const Real pi = col.exner(k);
    const Real qsi = saturation_mixing_ratio(col.p[s], t, /*ice_phase=*/true);
    const Real qv = std::max(col.qv[s], Real(0));

    const Real qsnow = std::max(col.qs[s], Real(0));
    if (qsnow > Real(1.0e-12)) {
      const Real lambda = snow_lambda(rho, qsnow);
      const Real number = number_concentration(kSnowIntercept, lambda);
      const Real d_mean = mass_weighted_diameter_power(lambda, kSnowMassB);
      const Real rate = deposition_rate(t, col.p[s], rho, qv, qsi, number, d_mean, true);
      const Real dq = rate * dt;
      // 限制：增长不超过可凝结的水汽（潜热反馈余量），减少不超过雪量
      const Real ls = latent_heat_sublimation(t);
      const Real grow_cap = std::max(qsi - qv, Real(0)) /
                            (Real(1) + ls * ls * qsi / (kCp * kRv * t * t));
      const Real dq_lim = (dq > Real(0)) ? std::min(dq, grow_cap)
                                         : -std::min(-dq, qsnow);
      col.qs[s] = qsnow + dq_lim;
      col.qv[s] = qv - dq_lim;
      col.theta[s] += latent_heat_sublimation(t) * dq_lim / (kCp * pi);
    }

    const Real qgr = std::max(col.qg[s], Real(0));
    if (qgr > Real(1.0e-12)) {
      const Real lambda = graupel_lambda(rho, qgr);
      const Real number = number_concentration(kGraupelIntercept, lambda);
      const Real d_mean = mass_weighted_diameter_power(lambda, kGraupelMassB);
      const Real rate = deposition_rate(t, col.p[s], rho, std::max(col.qv[s], Real(0)), qsi,
                                        number, d_mean, false);
      const Real dq = rate * dt;
      const Real ls = latent_heat_sublimation(t);
      const Real grow_cap = std::max(qsi - std::max(col.qv[s], Real(0)), Real(0)) /
                            (Real(1) + ls * ls * qsi / (kCp * kRv * t * t));
      const Real dq_lim = (dq > Real(0)) ? std::min(dq, grow_cap) : -std::min(-dq, qgr);
      col.qg[s] = qgr + dq_lim;
      col.qv[s] = std::max(col.qv[s] - dq_lim, Real(0));
      col.theta[s] += latent_heat_sublimation(t) * dq_lim / (kCp * pi);
    }
  }

  // ---- 8. 融化（雪、霰 -> 雨水） ----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    const Real t = col.temperature(k);
    const Real pi = col.exner(k);
    const Real qv = std::max(col.qv[s], Real(0));
    if (t <= kT0) continue;

    const Real qsnow = std::max(col.qs[s], Real(0));
    if (qsnow > Real(1.0e-12)) {
      const Real rate = snow_melting_rate(t, col.p[s], rho, qv, qsnow);
      const Real dq = std::min(rate * dt, qsnow);
      col.qs[s] = qsnow - dq;
      col.qr[s] = std::max(col.qr[s], Real(0)) + dq;
      col.theta[s] -= kLf * dq / (kCp * pi);
    }
    const Real qgr = std::max(col.qg[s], Real(0));
    if (qgr > Real(1.0e-12)) {
      const Real rate = graupel_melting_rate(t, col.p[s], rho, qv, qgr);
      const Real dq = std::min(rate * dt, qgr);
      col.qg[s] = qgr - dq;
      col.qr[s] = std::max(col.qr[s], Real(0)) + dq;
      col.theta[s] -= kLf * dq / (kCp * pi);
    }
  }

  // ---- 9. 沉降（雨、雪、霰；自下而上累加地面降水） ----
  std::vector<Real> vt(static_cast<Size>(n), Real(0));
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    vt[s] = rain_fall_speed(rho, col.qr[s]);
  }
  Real precip_mass = sediment_species(col.qr, col.rho, col.dz, vt, dt);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    vt[s] = snow_fall_speed(rho, col.qs[s]);
  }
  precip_mass += sediment_species(col.qs, col.rho, col.dz, vt, dt);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    vt[s] = graupel_fall_speed(rho, col.qg[s]);
  }
  precip_mass += sediment_species(col.qg, col.rho, col.dz, vt, dt);

  // ---- 10. 正定截断与守恒检查 ----
  // 截断产生的质量亏缺（kg/kg 求和后按层积分）全部归还给水汽，保持整列守恒。
  if (opt_.positive_definite) {
    const Real mass_before_clip = col.total_water_mass();
    col.clip_negatives();
    const Real deficit = mass_before_clip - col.total_water_mass();
    if (deficit > Real(0)) {
      Real denom = Real(0);
      for (Int k = 0; k < n; ++k) {
        denom += std::max(col.rho[static_cast<Size>(k)], Real(0)) *
                 std::max(col.dz[static_cast<Size>(k)], Real(1.0e-6));
      }
      if (denom > Real(0)) {
        for (Int k = 0; k < n; ++k) col.qv[static_cast<Size>(k)] += deficit / denom;
      }
    }
  }
  const Real mass_after = col.total_water_mass() + precip_mass;
  diag.mass_conservation_residual =
      std::max(diag.mass_conservation_residual, relative_water_residual(mass_before, mass_after));

  return precip_mass / dt;  // mm/s
}

Real ThompsonMicrophysics::max_fall_speed(const PhysicsColumn& col) const {
  Real vmax = Real(0);
  const Int n = col.nz();
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    vmax = std::max(vmax, rain_fall_speed(rho, col.qr[s]));
    vmax = std::max(vmax, snow_fall_speed(rho, col.qs[s]));
    vmax = std::max(vmax, graupel_fall_speed(rho, col.qg[s]));
  }
  return vmax;
}

std::string ThompsonMicrophysics::describe() const {
  std::ostringstream os;
  os << "ThompsonMicrophysics{ Nc=" << opt_.thompson_nc << " Nr=" << opt_.thompson_nr
     << " Ni=" << opt_.thompson_ni << " Ns=" << opt_.thompson_ns << " Ng=" << opt_.thompson_ng
     << " m^-3; 雪 m=0.069 D^2, V=40 D^0.55; 霰 V=442 D^0.89; 雨 V=4854 D e^-195D;"
     << " Bergeron c=" << opt_.thompson_bergeron_rate << " 1/s; Bigg A'="
     << opt_.thompson_bigg_a << " 1/K B'=" << opt_.thompson_bigg_b << " m^-3 s^-1;"
     << " 自动转换 tau=" << opt_.thompson_autoconv_time << " s r_crit="
     << opt_.thompson_autoconv_radius_um << " um }";
  return os.str();
}

}  // namespace vibe::physics

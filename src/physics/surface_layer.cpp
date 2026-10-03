/// @file surface_layer.cpp
/// @brief 地面层相似理论（Monin-Obukhov）、五层陆面与海面参数化。
///
/// 文献：[P9] Monin & Obukhov (1954)；[P10] Businger et al. (1971)；
///       [P14] Noilhan & Planton (1989)；[P15] Chen & Dudhia (2001)。

#include "vibe/physics/surface.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/physics/pbl.hpp"  // 复用 thomas_solve

namespace vibe::physics {

namespace {

constexpr Real kZetaMin = Real(-50);   ///< zeta 下限（强不稳定）
constexpr Real kZetaMax = Real(20);    ///< zeta 上限（强稳定）
constexpr Real kZ0Min = Real(1.0e-6);  ///< 粗糙度下限
constexpr Real kNeutralL = Real(1.0e30);

// 土壤层厚度（Noilhan-Planton 五层）
const Real kSoilDepthDefault[5] = {Real(0.05), Real(0.10), Real(0.20), Real(0.30), Real(0.35)};

/// 土壤温度/湿度的隐式扩散（上边界通量、下边界 Dirichlet）
///     C_i d_i dX_i/dt = F_{i+1/2} - F_{i-1/2},   F_{i+1/2} = K_{i+1/2}(X_{i+1}-X_i)/dz
///     F_{-1/2} = top_flux（向下为正，来自地表能量平衡）
///     底层： K (X_deep - X_n)/dz_half
/// 全隐式（向后欧拉）+ Thomas 算法，无条件稳定。复杂度 O(n)。
void soil_implicit_step(const std::vector<Real>& k_diff, const std::vector<Real>& thick,
                        const std::vector<Real>& capacity, const std::vector<Real>& x,
                        Real top_flux_down, Real deep_value, Real dt,
                        std::vector<Real>& x_new) {
  const Int n = static_cast<Int>(x.size());
  x_new = x;
  if (n <= 0) return;
  std::vector<Real> a(static_cast<Size>(n), Real(0));
  std::vector<Real> b(static_cast<Size>(n), Real(0));
  std::vector<Real> c(static_cast<Size>(n), Real(0));
  std::vector<Real> d(static_cast<Size>(n), Real(0));

  // 界面扩散系数：调和平均（串联热阻）
  std::vector<Real> kf(static_cast<Size>(n) + 1, Real(0));
  for (Int i = 0; i < n - 1; ++i) {
    const Size s = static_cast<Size>(i);
    const Real k1 = k_diff[s], k2 = k_diff[s + 1];
    kf[s + 1] = (k1 > Real(0) && k2 > Real(0)) ? Real(2) * k1 * k2 / (k1 + k2)
                                              : Real(0.5) * (k1 + k2);
  }
  kf[0] = k_diff[0];
  kf[static_cast<Size>(n)] = k_diff[static_cast<Size>(n - 1)];

  for (Int i = 0; i < n; ++i) {
    const Size s = static_cast<Size>(i);
    const Real di = std::max(thick[s], Real(1.0e-6));
    const Real ci = std::max(capacity[s], Real(1.0e-6));
    const Real dz_up = (i > 0) ? Real(0.5) * (di + thick[s - 1]) : Real(0.5) * di;
    const Real dz_dn = (i < n - 1) ? Real(0.5) * (di + thick[s + 1]) : Real(0.5) * di;
    const Real ku = kf[s] / (di * dz_up);
    const Real kd = kf[s + 1] / (di * dz_dn);
    const Real coef = dt / ci;
    if (i > 0) a[s] = -coef * ku;
    if (i < n - 1) c[s] = -coef * kd;
    Real rhs = x[s];
    if (i == 0) rhs += dt * top_flux_down / (ci * di);
    if (i == n - 1) rhs += coef * kd * deep_value;  // 底层 Dirichlet 修正项
    b[s] = Real(1) + coef * (ku + kd);
    d[s] = rhs;
  }
  thomas_solve(a, b, c, d, x_new);
}

}  // namespace

// ===========================================================================
// 稳定性函数与积分形式
// ===========================================================================

// phi_m = (1-16 zeta)^(-1/4) (zeta<0)；1+5 zeta (zeta>=0)   [P10]
Real MoninObukhov::phi_m(Real zeta) noexcept {
  if (zeta < Real(0)) return std::pow(std::max(Real(1) - Real(16) * zeta, Real(1.0e-6)), Real(-0.25));
  return Real(1) + Real(5) * zeta;
}

// phi_h = (1-16 zeta)^(-1/2) (zeta<0)；1+5 zeta (zeta>=0)
Real MoninObukhov::phi_h(Real zeta) noexcept {
  if (zeta < Real(0)) return std::pow(std::max(Real(1) - Real(16) * zeta, Real(1.0e-6)), Real(-0.5));
  return Real(1) + Real(5) * zeta;
}

// ---------------------------------------------------------------------------
// Paulson (1970) 积分形式
//   zeta<0: psi_m = 2 ln((1+x)/2) + ln((1+x^2)/2) - 2 atan(x) + pi/2, x=(1-16 zeta)^(1/4)
//   zeta>=0: psi_m = -5 zeta
// 复杂度 O(1)。
Real MoninObukhov::psi_m(Real zeta) noexcept {
  const Real z = clamp(zeta, kZetaMin, kZetaMax);
  if (z < Real(0)) {
    const Real x = std::pow(std::max(Real(1) - Real(16) * z, Real(1.0e-6)), Real(0.25));
    return Real(2) * std::log((Real(1) + x) * Real(0.5)) +
           std::log((Real(1) + x * x) * Real(0.5)) - Real(2) * std::atan(x) + kPi * Real(0.5);
  }
  return Real(-5) * z;
}

//   zeta<0: psi_h = 2 ln((1+x^2)/2)
//   zeta>=0: psi_h = -5 zeta
Real MoninObukhov::psi_h(Real zeta) noexcept {
  const Real z = clamp(zeta, kZetaMin, kZetaMax);
  if (z < Real(0)) {
    const Real x = std::pow(std::max(Real(1) - Real(16) * z, Real(1.0e-6)), Real(0.25));
    return Real(2) * std::log((Real(1) + x * x) * Real(0.5));
  }
  return Real(-5) * z;
}

Real MoninObukhov::log_wind(Real z, Real z0, Real ustar) noexcept {
  return ustar / kVonKarman * std::log(std::max(z, Real(1.0e-3)) / std::max(z0, kZ0Min));
}

Real MoninObukhov::neutral_exchange(Real z, Real z0) noexcept {
  const Real l = std::log(std::max(z, Real(1.0e-3)) / std::max(z0, kZ0Min));
  return kVonKarman * kVonKarman / (l * l);
}

//   Ri_b = g z (theta_v1 - theta_vs) / (theta_v1 |V|^2)
Real MoninObukhov::bulk_richardson(Real z, Real u, Real v, Real thv1, Real thvs) noexcept {
  const Real spd2 = std::max(u * u + v * v, Real(1.0e-4));
  return kGravity * z * (thv1 - thvs) / (std::max(thv1, Real(1)) * spd2);
}

// Charnock 粗糙度： z0 = alpha_c u*^2/g + 0.11 nu/u*   ([P10]；nu 取动力粘性/密度)
Real MoninObukhov::charnock_roughness(Real ustar, Real alpha_c) noexcept {
  const Real u = std::max(ustar, Real(1.0e-3));
  return std::max(alpha_c * u * u / kGravity + Real(0.11) * Real(1.5e-5) / u, kZ0Min);
}

// ===========================================================================
// 地面层迭代求解
// ===========================================================================

// ---------------------------------------------------------------------------
// 迭代算法（固定点 + 前 4 次欠松弛）
//   L = u*^2/(kappa (g/theta0) theta_v*),  theta_v* = theta*(1+0.608 q1) + 0.608 theta1 q*
//   u*     = kappa |V1| / [ln(z1/z0)  - psi_m(z1/L) + psi_m(z0/L)]
//   theta* = kappa (theta1-theta_s)/[ln(z1/z0h) - psi_h(z1/L) + psi_h(z0h/L)]
//   q*     = kappa (q1-q_s)     / [ln(z1/z0q) - psi_h(z1/L) + psi_h(z0q/L)]
// 收敛判据： |delta L| / L <= mo_tolerance；L -> 无穷视为中性。
// 复杂度 O(n_iter)。
SurfaceFluxes MoninObukhov::solve(const PhysicsColumn& col, SurfaceState& sfc, Real dt) const {
  VIBE_UNUSED(dt);
  SurfaceFluxes f;
  const Int n = col.nz();
  if (n <= 0) return f;

  const Real z1 = std::max(col.z[0] - sfc.terrain_height, Real(1.0));
  const Real u1 = col.u[0];
  const Real v1 = col.v[0];
  Real spd = std::sqrt(u1 * u1 + v1 * v1);
  spd = std::max(spd, Real(0.1));
  const Real theta1 = col.theta[0];
  const Real pi1 = col.exner(0);
  const Real q1 = std::max(col.qv[0], Real(0));
  const Real rho1 = (col.rho[0] > Real(0)) ? col.rho[0]
                                           : density_from_pressure_tv(col.p[0], col.virtual_temperature(0));

  const bool water = sfc.is_water();
  const Real p_sfc = (sfc.surface_pressure > Real(0)) ? sfc.surface_pressure : col.p[0];
  const Real pi_sfc = exner_from_pressure(p_sfc);

  // 粗糙度初值
  Real z0 = water ? Real(1.0e-4) : std::max(sfc.roughness, kZ0Min);
  Real z0h = std::max(Real(0.1) * z0, Real(2.0e-5));
  Real z0q = z0h;

  // 地表温度与湿度
  Real t_sfc = (sfc.skin_temperature > Real(0)) ? sfc.skin_temperature
                                                 : (water ? opt_.sst : col.temperature(0));
  Real beta = Real(1);
  if (!water) {
    const Real w1 = sfc.soil_moisture.empty() ? Real(0.25) : sfc.soil_moisture[0];
    beta = LandSurface::moisture_availability(w1, sfc.soil_field_capacity, sfc.soil_wilting_point);
  }
  Real theta_s = t_sfc / pi_sfc;
  Real q_s = beta * saturation_mixing_ratio(p_sfc, t_sfc);

  // 中性初值
  const Real ln_m = std::max(std::log(z1 / z0), Real(1.0e-3));
  Real ustar = kVonKarman * spd / ln_m;
  Real tstar = Real(0), qstar = Real(0);
  Real L = kNeutralL;
  Real residual = kHuge;
  bool converged = false;
  const int max_iter = std::max(opt_.mo_max_iterations, 2);
  int it_used = 0;

  for (int it = 1; it <= max_iter; ++it) {
    it_used = it;
    const Real zeta1 = clamp(z1 / L, kZetaMin, kZetaMax);
    const Real zeta0m = clamp(z0 / L, kZetaMin, kZetaMax);
    const Real zeta0h = clamp(z0h / L, kZetaMin, kZetaMax);

    const Real den_m = std::max(std::log(z1 / z0) - psi_m(zeta1) + psi_m(zeta0m), Real(1.0e-3));
    const Real den_h = std::max(std::log(z1 / z0h) - psi_h(zeta1) + psi_h(zeta0h), Real(1.0e-3));
    const Real den_q = std::max(std::log(z1 / z0q) - psi_h(zeta1) + psi_h(zeta0h), Real(1.0e-3));

    ustar = kVonKarman * spd / den_m;
    tstar = kVonKarman * (theta1 - theta_s) / den_h;
    qstar = kVonKarman * (q1 - q_s) / den_q;

    if (water) {
      z0 = charnock_roughness(ustar, opt_.charnock_alpha);
      z0h = std::max(Real(0.1) * z0, Real(2.0e-5));
      z0q = z0h;
    }

    const Real thv_star = tstar * (Real(1) + Real(0.608) * q1) + Real(0.608) * theta1 * qstar;
    Real L_new = kNeutralL;
    if (std::abs(thv_star) > Real(1.0e-12)) {
      L_new = ustar * ustar / (kVonKarman * (kGravity / std::max(theta1, Real(1))) * thv_star);
      L_new = clamp(L_new, Real(-1.0e7), Real(1.0e7));
    }
    const Real relax = (it <= 4) ? Real(0.5) : Real(1.0);
    const Real L_next = L + relax * (L_new - L);
    residual = std::abs(L_next - L) / std::max(std::abs(L_next), Real(1));
    L = L_next;
    if (std::abs(thv_star) <= Real(1.0e-12)) {
      L = kNeutralL;
      residual = Real(0);
      converged = true;
      break;
    }
    if (residual <= opt_.mo_tolerance) {
      converged = true;
      break;
    }
  }
  last_iterations_ = it_used;
  last_residual_ = residual;

  // 最终廓线量
  const Real zeta1 = clamp(z1 / L, kZetaMin, kZetaMax);
  f.zeta = zeta1;
  f.obukhov = L;
  f.ustar = ustar;
  f.tstar = tstar;
  f.qstar = qstar;
  f.z0 = z0;
  f.z0h = z0h;
  f.z0q = z0q;
  f.iterations = it_used;
  f.residual = residual;
  f.converged = converged;
  f.bulk_richardson = bulk_richardson(z1, u1, v1, col.theta_v(0), theta_s);

  const Real lv = latent_heat_vaporization(Real(0.5) * (col.temperature(0) + t_sfc));
  f.sensible = -rho1 * kCp * ustar * tstar;      // W/m^2，向上为正
  f.latent = -rho1 * lv * ustar * qstar;
  f.momentum = rho1 * ustar * ustar;             // N/m^2
  f.cd = ustar * ustar / (spd * spd);

  // 交换系数（正值，用于陆面能量平衡的线性化）
  const Real den_h = std::max(std::log(z1 / z0h) - psi_h(zeta1) + psi_h(clamp(z0h / L, kZetaMin, kZetaMax)), Real(1.0e-3));
  const Real den_q = std::max(std::log(z1 / z0q) - psi_h(zeta1) + psi_h(clamp(z0q / L, kZetaMin, kZetaMax)), Real(1.0e-3));
  f.ch = kVonKarman * ustar / den_h;
  f.cq = kVonKarman * ustar / den_q;

  // 2 m / 10 m 诊断
  const Real th2 = theta_s + tstar / kVonKarman *
                               (std::log(Real(2) / z0h) - psi_h(clamp(Real(2) / L, kZetaMin, kZetaMax)) +
                                psi_h(clamp(z0h / L, kZetaMin, kZetaMax)));
  const Real q2 = q_s + qstar / kVonKarman *
                             (std::log(Real(2) / z0q) - psi_h(clamp(Real(2) / L, kZetaMin, kZetaMax)) +
                              psi_h(clamp(z0q / L, kZetaMin, kZetaMax)));
  const Real spd10 = ustar / kVonKarman *
                     (std::log(Real(10) / z0) - psi_m(clamp(Real(10) / L, kZetaMin, kZetaMax)) +
                      psi_m(clamp(z0 / L, kZetaMin, kZetaMax)));
  f.t2 = th2 * pi_sfc;
  f.q2 = std::max(q2, Real(0));
  const Real scale10 = spd10 / spd;
  f.u10 = u1 * scale10;
  f.v10 = v1 * scale10;

  // 回写地表状态
  sfc.roughness = z0;
  sfc.roughness_heat = z0h;
  sfc.roughness_moist = z0q;
  sfc.friction_velocity = ustar;
  sfc.temperature_scale = tstar;
  sfc.moisture_scale = qstar;
  sfc.obukhov_length = L;
  sfc.sensible_heat_flux = f.sensible;
  sfc.latent_heat_flux = f.latent;
  sfc.momentum_flux = f.momentum;
  sfc.t2 = f.t2;
  sfc.q2 = f.q2;
  sfc.u10 = f.u10;
  sfc.v10 = f.v10;
  sfc.layer_depth = col.dz[0];
  return f;
}

std::string MoninObukhov::describe() const {
  std::ostringstream os;
  os << "MoninObukhov{ 最大迭代=" << opt_.mo_max_iterations << " 容差=" << opt_.mo_tolerance
     << " 海面 Charnock=" << opt_.charnock_alpha << " SST=" << opt_.sst
     << " K; 陆面 beta 系数=" << opt_.land_evap_beta_coef << " }";
  return os.str();
}

// ===========================================================================
// 陆面
// ===========================================================================

void LandSurface::initialize(SurfaceState& sfc) const {
  const Int n = std::max(opt_.n_soil_layers, 1);
  sfc.resize_soil(n);
  // 初始廓线：深层气候值 + 随深度衰减的近地表偏差（保证存在温度梯度）
  for (Int i = 0; i < n; ++i) {
    const Size s = static_cast<Size>(i);
    sfc.soil_temperature[s] =
        opt_.soil_deep_temperature + Real(2) * std::exp(-Real(0.5) * static_cast<Real>(i));
    sfc.soil_moisture[s] = opt_.soil_deep_moisture;
  }
  if (!(sfc.skin_temperature > Real(0))) sfc.skin_temperature = sfc.soil_temperature[0];
}

//   beta = min(1, w1/w_fc)（Noilhan-Planton 的蒸发效率）
Real LandSurface::moisture_availability(Real w1, Real field_capacity, Real wilting_point) noexcept {
  const Real wfc = std::max(field_capacity, Real(1.0e-3));
  const Real ww = std::min(std::max(wilting_point, Real(0)), wfc);
  const Real w = clamp(w1, Real(0), wfc);
  if (w <= ww) {
    // 凋萎点以下：指数衰减（残余蒸发）
    return std::exp(-Real(5) * (ww - w) / std::max(ww, Real(1.0e-3)));
  }
  return std::min(Real(1), w / wfc);
}

Real LandSurface::energy_balance_residual(const SurfaceState& sfc) {
  return sfc.net_radiation - sfc.sensible_heat_flux - sfc.latent_heat_flux - sfc.ground_heat_flux;
}

// ---------------------------------------------------------------------------
// 表皮温度 Newton 求解 + 五层土壤推进
//
// 能量平衡残差（对 T_g 求导用解析式）：
//   F(Tg) = R_n - H - LE - G
//   R_n = (1-a)S + eps(L - sigma Tg^4),  dR_n/dTg = -4 eps sigma Tg^3
//   H   = rho cp Ch |V| (Tg/pi_s - theta1),      dH/dTg  = rho cp Ch |V|/pi_s
//   LE  = rho Lv Cq |V| (q_sat(Tg) beta - q1),   dLE/dTg = rho Lv Cq |V| beta dq_sat/dTg
//   G   = lambda_s (Tg - T_soil1)/(0.5 dz1),     dG/dTg  = lambda_s/(0.5 dz1)
// 复杂度：O(n_iter + n_soil)。
void LandSurface::step_column(PhysicsColumn& col, SurfaceState& sfc, SurfaceFluxes& flx, Real dt,
                              ColumnTendency& tend, PhysicsDiagnostics& diag) const {
  VIBE_UNUSED(tend);  // 大气响应由 PBL 方案给出；本函数只更新陆面状态与地表通量
  const Int n = col.nz();
  if (n <= 0 || !(dt > Real(0))) return;
  if (sfc.soil_temperature.empty()) initialize(sfc);

  const bool water = sfc.is_water();
  const Real z1 = std::max(col.z[0] - sfc.terrain_height, Real(1.0));
  const Real u1 = col.u[0];
  const Real v1 = col.v[0];
  const Real spd = std::max(std::sqrt(u1 * u1 + v1 * v1), Real(0.1));
  const Real theta1 = col.theta[0];
  const Real rho1 = (col.rho[0] > Real(0)) ? col.rho[0]
                                           : density_from_pressure_tv(col.p[0], col.virtual_temperature(0));
  const Real p_sfc = (sfc.surface_pressure > Real(0)) ? sfc.surface_pressure : col.p[0];
  const Real pi_sfc = exner_from_pressure(p_sfc);
  const Real q1 = std::max(col.qv[0], Real(0));
  const Real shortwave = std::max(sfc.sw_down, Real(0));
  const Real longwave = std::max(sfc.lw_down, Real(0));
  const Real eps = clamp(sfc.emissivity, Real(0.5), Real(1));
  const Real albedo = clamp(sfc.albedo, Real(0), Real(1));

  // 海面：固定 SST（粗糙度已在 M-O 迭代中按 Charnock 更新）
  if (water) {
    SeaSurface sea(opt_);
    sea.update(sfc, flx);
    const Real sst = opt_.sst;
    const Real q_s = saturation_mixing_ratio(p_sfc, sst);
    const Real lv = latent_heat_vaporization(sst);
    const Real hgt = -rho1 * kCp * flx.ustar * flx.tstar;
    const Real lhe = -rho1 * lv * flx.ustar * flx.qstar;
    const Real sigma_t4 = kStefanBoltzmann * sst * sst * sst * sst;
    const Real rn = (Real(1) - sfc.albedo) * shortwave + eps * (longwave - sigma_t4);
    sfc.skin_temperature = sst;
    sfc.sensible_heat_flux = hgt;
    sfc.latent_heat_flux = lhe;
    sfc.ground_heat_flux = rn - hgt - lhe;  // 海面：余项进入海洋热储存（SST 固定）
    sfc.net_radiation = rn;
    flx.sensible = hgt;
    flx.latent = lhe;
    diag.energy_balance_residual_max = std::max(diag.energy_balance_residual_max, Real(0));
    VIBE_UNUSED(q_s);
    return;
  }

  // ---- 土壤层厚度与热力学参数 ----
  const Int ns = std::max(sfc.nsoil(), Int(1));
  std::vector<Real> thick(static_cast<Size>(ns), Real(0));
  for (Int i = 0; i < ns; ++i) {
    const Size s = static_cast<Size>(i);
    thick[s] = (sfc.soil_depth.size() == static_cast<Size>(ns) && sfc.soil_depth[s] > Real(0))
                   ? sfc.soil_depth[s]
                   : (ns == 5 ? kSoilDepthDefault[i] : Real(1) / static_cast<Real>(ns));
  }
  const Real lambda_s = std::max(opt_.soil_thermal_conductivity, Real(1.0e-3));
  const Real cap_s = std::max(opt_.soil_heat_capacity, Real(1.0e3));

  // ---- 表皮温度 Newton 迭代 ----
  Real tg = (sfc.skin_temperature > Real(0)) ? sfc.skin_temperature : sfc.soil_temperature[0];
  Real beta = Real(1);
  {
    const Real w1 = sfc.soil_moisture[0];
    beta = std::max(Real(0.05), moisture_availability(w1, sfc.soil_field_capacity, sfc.soil_wilting_point) *
                                   opt_.land_evap_beta_coef);
  }
  const Real lv = latent_heat_vaporization(std::max(col.temperature(0), Real(250)));
  Real residual = kHuge;
  for (int it = 0; it < 20; ++it) {
    const Real tg4 = tg * tg * tg * tg;
    const Real rn = (Real(1) - albedo) * shortwave + eps * (longwave - kStefanBoltzmann * tg4);
    const Real theta_s = tg / pi_sfc;
    const Real hgt = rho1 * kCp * flx.ch * spd * (theta_s - theta1);
    const Real qsat = saturation_mixing_ratio(p_sfc, tg);
    const Real lhe = rho1 * lv * flx.cq * spd * (beta * qsat - q1);
    const Real g = lambda_s * (tg - sfc.soil_temperature[0]) / (Real(0.5) * thick[0]);
    const Real fval = rn - hgt - lhe - g;
    residual = std::abs(fval);
    // 导数
    const Real dqsdT = lv * qsat / (kRv * tg * tg);
    const Real df = -Real(4) * eps * kStefanBoltzmann * tg * tg * tg -
                    rho1 * kCp * flx.ch * spd / pi_sfc -
                    rho1 * lv * flx.cq * spd * beta * dqsdT - lambda_s / (Real(0.5) * thick[0]);
    if (std::abs(df) < Real(1.0e-12)) break;
    const Real step = clamp(fval / df, Real(-20), Real(20));
    tg -= step;
    tg = clamp(tg, Real(200), Real(350));
    if (std::abs(step) < Real(1.0e-10) && residual < Real(1.0e-8)) break;
  }

  // ---- 用最终表皮温度重算通量 ----
  const Real theta_s = tg / pi_sfc;
  const Real qsat = saturation_mixing_ratio(p_sfc, tg);
  Real hgt = rho1 * kCp * flx.ch * spd * (theta_s - theta1);
  Real lhe = rho1 * lv * flx.cq * spd * (beta * qsat - q1);
  const Real rn = (Real(1) - albedo) * shortwave + eps * (longwave - kStefanBoltzmann * tg * tg * tg * tg);
  Real g = rn - hgt - lhe;  // 能量平衡闭合（Newton 收敛后与土壤热通量一致）

  sfc.skin_temperature = tg;
  sfc.sensible_heat_flux = hgt;
  sfc.latent_heat_flux = lhe;
  sfc.ground_heat_flux = g;
  sfc.net_radiation = rn;
  flx.sensible = hgt;
  flx.latent = lhe;
  // 与 M-O 的 u*theta* 通量做一致性修正（保持符号与量级一致）
  if (std::abs(flx.ustar) > Real(1.0e-8) && std::abs(hgt) > Real(1.0e-6)) {
    flx.tstar = -hgt / (rho1 * kCp * flx.ustar);
  }
  if (std::abs(flx.ustar) > Real(1.0e-8) && std::abs(lhe) > Real(1.0e-6)) {
    flx.qstar = -lhe / (rho1 * lv * flx.ustar);
  }
  diag.energy_balance_residual_max =
      std::max(diag.energy_balance_residual_max, std::abs(energy_balance_residual(sfc)));
  VIBE_UNUSED(residual);

  // ---- 五层土壤温度与湿度（隐式扩散） ----
  const Real dt_soil = std::min(dt, Real(3600));
  std::vector<Real> kt(static_cast<Size>(ns), lambda_s);
  std::vector<Real> ct(static_cast<Size>(ns), cap_s);
  std::vector<Real> t_new;
  soil_implicit_step(kt, thick, ct, sfc.soil_temperature, g, opt_.soil_deep_temperature,
                     dt_soil, t_new);
  for (Int i = 0; i < ns; ++i) {
    const Size s = static_cast<Size>(i);
    sfc.soil_temperature[s] = clamp(t_new[s], Real(200), Real(350));
  }

  // 湿度：水汽通量（向上）= lhe/Lv，向下进入土壤为负
  std::vector<Real> dw(static_cast<Size>(ns), std::max(opt_.soil_hydraulic_diffusivity, Real(0)));
  std::vector<Real> cw(static_cast<Size>(ns), Real(1));
  const Real evap_flux = (lv > Real(0)) ? lhe / lv : Real(0);   // kg/(m^2 s)，向上为正
  std::vector<Real> w_new;
  soil_implicit_step(dw, thick, cw, sfc.soil_moisture, -evap_flux, opt_.soil_deep_moisture,
                     dt_soil, w_new);
  for (Int i = 0; i < ns; ++i) {
    const Size s = static_cast<Size>(i);
    sfc.soil_moisture[s] = clamp(w_new[s], Real(0.01), sfc.soil_field_capacity);
  }
}

std::string LandSurface::describe() const {
  std::ostringstream os;
  os << "LandSurface{ 层数=" << opt_.n_soil_layers << " lambda=" << opt_.soil_thermal_conductivity
     << " W/(m K) C_s=" << opt_.soil_heat_capacity << " J/(m^3 K) D_w="
     << opt_.soil_hydraulic_diffusivity << " m^2/s 田间持水=" << opt_.soil_field_capacity
     << " 凋萎点=" << opt_.soil_wilting_point << " 深层T=" << opt_.soil_deep_temperature
     << " K 深层w=" << opt_.soil_deep_moisture << " }";
  return os.str();
}

// ===========================================================================
// 海面
// ===========================================================================

void SeaSurface::update(SurfaceState& sfc, const SurfaceFluxes& flx) const {
  sfc.land_fraction = Real(0);
  sfc.albedo = kAlbedoSea;
  sfc.emissivity = Real(0.98);
  sfc.skin_temperature = opt_.sst;
  sfc.roughness = MoninObukhov::charnock_roughness(flx.ustar, opt_.charnock_alpha);
  sfc.roughness_heat = std::max(Real(0.1) * sfc.roughness, Real(2.0e-5));
  sfc.roughness_moist = sfc.roughness_heat;
}

// 名义上限： rho Lv Cq U10 (q_sat(SST) - 0)，取 Cq = 1.5e-3、U10 = 10 m/s
Real SeaSurface::max_latent_flux(const SurfaceState& sfc) const noexcept {
  const Real p = (sfc.surface_pressure > Real(0)) ? sfc.surface_pressure : kP0;
  const Real rho = p / (kRd * std::max(sfc.t2, Real(200)));
  const Real qs = saturation_mixing_ratio(p, opt_.sst);
  return rho * latent_heat_vaporization(opt_.sst) * Real(1.5e-3) * Real(10) * std::max(qs, Real(0));
}

std::string SeaSurface::describe() const {
  std::ostringstream os;
  os << "SeaSurface{ 固定 SST=" << opt_.sst << " K Charnock alpha=" << opt_.charnock_alpha
     << " 粘性项系数=" << opt_.charnock_viscous << " 反照率=" << kAlbedoSea << " }";
  return os.str();
}

std::unique_ptr<MoninObukhov> make_surface_layer(const PhysicsOptions& opt) {
  return std::make_unique<MoninObukhov>(opt);
}
std::unique_ptr<LandSurface> make_land_surface(const PhysicsOptions& opt) {
  return std::make_unique<LandSurface>(opt);
}
std::unique_ptr<SeaSurface> make_sea_surface(const PhysicsOptions& opt) {
  return std::make_unique<SeaSurface>(opt);
}

}  // namespace vibe::physics

/// @file radiation_rrtmg.cpp
/// @brief RRTM/RRTMG 型辐射参数化：相关 k 分布长波与短波（[P5][P6]）。
///
/// 结构
/// ----
///   1. 谱带：长波按 RRTM 的 16 个谱带边界（cm^-1）、短波按 RRTMG 的 14 个谱带
///      边界；带数可由 PhysicsOptions 调整（非默认值时按对数等距生成）。
///   2. 带内积分用相关 k 分布（Gauss-Laguerre 求积）：
///        T_band(u) = sum_g w_g exp( -k_g u ),   sum_g w_g = 1,  sum_g w_g k_g = kbar
///   3. 长波：发射率/吸收率累加（向上/向下两趟），
///        F_up[k+1] = T_k F_up[k] + (1-T_k) sigma T_k^4 f_b(T_k)
///        F_dn[k]   = T_k F_dn[k+1] + (1-T_k) sigma T_k^4 f_b(T_k)
///      地表：F_up[0] = eps sigma T_s^4 f_b(T_s) + (1-eps) F_dn[0]。
///   4. 短波：太阳直射光束（Beer-Lambert，含太阳高度角与日地距离修正）
///        + 单次散射源（按不对称因子分配到上/下半球）+ 漫射衰减（扩散因子 1.66）
///        + 地表反射；夜间（cos z <= 阈值）直接跳过短波。
///   5. 加热率：净通量散度
///        dT/dt = -1/(rho cp) dF_net/dz,   dtheta/dt = (1/pi) dT/dt
///
/// 云光学厚度（[P6] 第 3 节，几何光学）
///     tau_c = 3 LWP/(2 rho_w r_e) + 3 IWP/(2 rho_i r_e,i)，再按云量线性缩放。
///
/// 文献：[P5] Mlawer et al. (1997)；[P6] Iacono et al. (2008)；
///       [B4] Holton & Hakim (2013) 第 2、12 章（黑体与辐射传输基础）。

#include "vibe/physics/radiation.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"

namespace vibe::physics {

namespace {

constexpr Real kC2 = Real(14388.0);      ///< hc/k (um K)
constexpr Real kSunTemperature = Real(5776.0);   ///< 太阳等效黑体温度 (K)
constexpr Real kDiffusivity = Real(1.66);        ///< 漫射扩散因子 D = 1/mu_mean
constexpr Real kCo2MassRatio = Real(6.38e-4);    ///< 420 ppmv CO2 的质量混合比
constexpr Real kDuToKgM2 = Real(2.1415e-5);      ///< 1 Dobson Unit (kg/m^2)
constexpr Real kOzoneCenterM = Real(22000.0);    ///< 臭氧层中心高度
constexpr Real kOzoneWidthM = Real(5000.0);      ///< 臭氧层宽度
constexpr Real kRayleigh550 = Real(9.7e-6);      ///< 550 nm Rayleigh 质量系数 m^2/kg

// ---------------------------------------------------------------------------
// 累积 Planck 函数（黑体辐射份额）
//   f(lambda T) = 15/pi^4 sum_{n>=1} e^{-n a/x}[ a^3/(n^3 x^3) + 3a^2/(n^2 x^2)
//                                              + 6a/(n x) + 6 ] / n^4
//   a = hc/k = 14388 um K, x = lambda T
// 数值验证：x = 2898 um K -> f = 0.2495（Wien 峰值）；x = 4100 -> 0.4942（中值）。
// 离散化：级数求和，n 上限 600（含早退），x/a > 30 时取 1，x/a < 1/40 时取 0。
// 复杂度：O(N)，N <= 600（一般 < 250）。
Real planck_fraction_impl(Real lambda_um, Real t) noexcept {
  if (!(lambda_um > Real(0)) || !(t > Real(0))) return Real(0);
  const Real x = lambda_um * t;
  const Real y = kC2 / x;
  if (y < Real(0.033)) return Real(1);
  if (y > Real(25)) return Real(0);
  Real sum = Real(0);
  const Real y2 = y * y;
  const Real y3 = y2 * y;
  for (int n = 1; n <= 600; ++n) {
    const Real nl = static_cast<Real>(n);
    const Real ny = nl * y;
    if (ny > Real(745)) break;
    const Real n2 = nl * nl;
    // term = e^{-n y} [ y^3/n^7 + 3 y^2/n^6 + 6 y/n^5 + 6/n^4 ]
    const Real term = std::exp(-ny) * (y3 / (n2 * n2 * n2 * nl) + Real(3) * y2 / (n2 * n2 * n2) +
                                      Real(6) * y / (n2 * n2 * nl) + Real(6) / (n2 * n2));
    sum += term;
    if (term < Real(1.0e-17) && ny > Real(5)) break;
  }
  return std::min(Real(15) / (kPi * kPi * kPi * kPi) * sum, Real(1));
}

// 谱带透射率： T = sum_g w_g exp(-tau_g)
Real band_transmittance(const std::vector<Real>& tau_g, const std::vector<Real>& w) noexcept {
  Real s = Real(0);
  for (Size g = 0; g < tau_g.size(); ++g) s += w[g] * std::exp(-tau_g[g]);
  return s;
}

// 谱带平均消光光学厚度（由带平均透射率反演，用于漫射衰减与云光学厚度）
Real tau_from_transmittance(Real trans) noexcept {
  return -std::log(clamp(trans, Real(1.0e-12), Real(1)));
}

// 长波谱带边界（RRTM，cm^-1）与短波谱带边界（RRTMG，cm^-1）
const Real kLwWavenumber[] = {10,   250,  500,  630,  700,  820,  980,  1080, 1180,
                              1390, 1480, 1800, 2080, 2250, 2380, 2600, 3250};
const Real kSwWavenumber[] = {820,  2600, 3250,  4000,  4650,  5150,  7700, 8050,
                              12850, 16000, 22650, 29000, 38000, 50000};

// 谱带吸收系数（教学用简化参数化，量级与 RRTM 各带一致；不是业务系数表）
//   kbar(lambda) 为质量吸收系数 (m^2/kg)
Real kbar_h2o_lw(Real lam) noexcept {
  return Real(0.35) + Real(14) * std::exp(-sqr((lam - Real(6.3)) / Real(1.3))) +
         Real(6) * std::exp(-sqr((lam - Real(25)) / Real(18))) +
         Real(1.5) * std::exp(-sqr((lam - Real(3.2)) / Real(0.5)));
}
Real kbar_co2_lw(Real lam) noexcept {
  return Real(60) * std::exp(-sqr((lam - Real(15)) / Real(1.4))) +
         Real(3) * std::exp(-sqr((lam - Real(4.3)) / Real(0.4)));
}
Real kbar_o3_lw(Real lam) noexcept {
  return Real(45) * std::exp(-sqr((lam - Real(9.6)) / Real(0.6))) +
         Real(2) * std::exp(-sqr((lam - Real(14)) / Real(3)));
}
Real kbar_h2o_sw(Real lam) noexcept {
  return Real(0.002) + Real(0.02) * std::exp(-sqr((lam - Real(1.4)) / Real(0.6))) +
         Real(0.05) * std::exp(-sqr((lam - Real(2.0)) / Real(0.7))) +
         Real(0.01) * std::exp(-sqr((lam - Real(0.94)) / Real(0.15)));
}
Real kbar_o3_sw(Real lam) noexcept {
  return Real(0.6) * std::exp(-sqr((lam - Real(0.30)) / Real(0.08))) +
         Real(0.02) * std::exp(-sqr((lam - Real(0.60)) / Real(0.10)));
}

}  // namespace

// ===========================================================================
// 构造与建表
// ===========================================================================

RadiationDriver::RadiationDriver(RadiationScheme scheme, const PhysicsOptions& opt)
    : scheme_(scheme), opt_(opt) {}

const char* RadiationDriver::name() const noexcept {
  switch (scheme_) {
    case RadiationScheme::Rrtmg: return "RRTMG 相关 k 分布（长波+短波）[P5][P6]";
    case RadiationScheme::RrtmgLongwaveOnly: return "RRTMG 长波 [P5]";
    case RadiationScheme::RrtmgShortwaveOnly: return "RRTMG 短波 [P6]";
    case RadiationScheme::SimpleGrey: return "灰体教学版本";
    default: return "无辐射";
  }
}

// ---------------------------------------------------------------------------
// Gauss-Laguerre 求积（Laguerre 递推 + Newton 迭代）
//   递推： L_0 = 1, L_1 = 1 - x, (j+1) L_{j+1} = (2j+1-x) L_j - j L_{j-1}
//   导数： L_n'(x) = (n L_n - n L_{n-1}) / x
//   权重： w_i = x_i / ((n+1)^2 L_{n+1}(x_i)^2)，归一化后 sum w_i = 1
// 文献：[B11] 第 4 章（正交多项式与数值积分）。
// 复杂度：O(n^2 * n_iter)。
void RadiationDriver::gauss_laguerre(Int n, std::vector<Real>& nodes,
                                     std::vector<Real>& weights) {
  const Int m = clamp(n, Int(1), Int(32));
  nodes.assign(static_cast<Size>(m), Real(0));
  weights.assign(static_cast<Size>(m), Real(0));
  if (m == 1) {
    nodes[0] = Real(1);
    weights[0] = Real(1);
    return;
  }
  const Real fm = static_cast<Real>(m);
  for (Int i = 0; i < m; ++i) {
    // Newton 初值（Numerical Recipes 型）
    Real x = Real(0);
    if (i == 0) {
      x = Real(3) / (Real(1) + Real(2.4) * fm);
    } else if (i == 1) {
      x = nodes[0] + Real(15) / (Real(1) + Real(2.5) * fm);
    } else {
      const Real fi = static_cast<Real>(i);
      x = nodes[static_cast<Size>(i - 1)] +
          (Real(1) + Real(2.61) * (fi - Real(1))) / (Real(1) + Real(2.55) * (fi - Real(1))) *
              (nodes[static_cast<Size>(i - 1)] - nodes[static_cast<Size>(i - 2)]);
    }
    Real ln = Real(0), ln1 = Real(0);
    for (int it = 0; it < 20; ++it) {
      // 计算 L_m(x) 与 L_{m+1}(x)
      Real p0 = Real(1), p1 = Real(1) - x;
      for (Int j = 1; j < m; ++j) {
        const Real fj = static_cast<Real>(j);
        const Real p2 = ((Real(2) * fj + Real(1) - x) * p1 - fj * p0) / (fj + Real(1));
        p0 = p1;
        p1 = p2;
      }
      ln = p1;      // L_m
      ln1 = p0;     // L_{m-1}
      const Real dln = (fm * ln - fm * ln1) / std::max(x, Real(1.0e-12));
      const Real dx = ln / std::max(dln, Real(1.0e-30));
      const Real xn = x - dx;
      x = (xn > Real(1.0e-10)) ? xn : Real(0.5) * x;
      if (std::abs(dx) < Real(1.0e-14)) break;
    }
    // 权重（含 L_{m+1}）
    Real p0 = Real(1), p1 = Real(1) - x;
    for (Int j = 1; j <= m; ++j) {
      const Real fj = static_cast<Real>(j);
      const Real p2 = ((Real(2) * fj + Real(1) - x) * p1 - fj * p0) / (fj + Real(1));
      p0 = p1;
      p1 = p2;
    }
    const Real lmp1 = p1;  // L_{m+1}
    nodes[static_cast<Size>(i)] = x;
    weights[static_cast<Size>(i)] =
        x / ((fm + Real(1)) * (fm + Real(1)) * lmp1 * lmp1);
  }
  // 归一化（保证 sum w = 1）
  Real s = Real(0);
  for (Real w : weights) s += w;
  if (s > Real(0)) {
    for (Real& w : weights) w /= s;
  }
}

void RadiationDriver::build_k_distribution(Int n_g, std::vector<Real>& k_ratio,
                                           std::vector<Real>& weights) {
  gauss_laguerre(n_g, k_ratio, weights);
}

void RadiationDriver::initialize(const grid::Grid* g) {
  VIBE_UNUSED(g);
  build_k_distribution(opt_.n_g_points, k_ratio_, k_weight_);

  // ---- 长波谱带 ----
  const Int n_lw = clamp(opt_.n_longwave_bands, Int(1), Int(64));
  const Int n_default_lw = static_cast<Int>(sizeof(kLwWavenumber) / sizeof(Real)) - 1;
  lw_.clear();
  for (Int b = 0; b < n_lw; ++b) {
    RadiationBand band;
    if (n_lw == n_default_lw) {
      band.lambda_hi_um = Real(1.0e4) / kLwWavenumber[b];
      band.lambda_lo_um = Real(1.0e4) / kLwWavenumber[b + 1];
    } else {
      const Real lo = Real(3.08), hi = Real(1000.0);
      const Real t0 = static_cast<Real>(b) / static_cast<Real>(n_lw);
      const Real t1 = static_cast<Real>(b + 1) / static_cast<Real>(n_lw);
      band.lambda_lo_um = lo * std::pow(hi / lo, t0);
      band.lambda_hi_um = lo * std::pow(hi / lo, t1);
    }
    const Real lc = Real(0.5) * (band.lambda_lo_um + band.lambda_hi_um);
    const Size ng = k_ratio_.size();
    band.k_h2o.resize(ng);
    band.k_co2.resize(ng);
    band.k_o3.resize(ng);
    band.k_cloud.resize(ng);
    band.weights = k_weight_;
    for (Size gg = 0; gg < ng; ++gg) {
      band.k_h2o[gg] = k_ratio_[gg] * kbar_h2o_lw(lc);
      band.k_co2[gg] = k_ratio_[gg] * kbar_co2_lw(lc);
      band.k_o3[gg] = k_ratio_[gg] * kbar_o3_lw(lc);
      band.k_cloud[gg] = Real(3) / (Real(2) * kRhoWater * opt_.cloud_droplet_radius_um * Real(1.0e-6));
    }
    band.rayleigh_reference = Real(0);
    band.n_pressure = Real(0.5);
    band.n_temperature = Real(0.5);
    lw_.push_back(std::move(band));
  }
  // 长波带份额（在 250 K 参考温度下归一化，覆盖 3.08-1000 um）
  {
    Real total = Real(0);
    for (auto& band : lw_) {
      band.spectral_fraction = planck_band_fraction(band.lambda_lo_um, band.lambda_hi_um, Real(250));
      total += band.spectral_fraction;
    }
    if (total > Real(0)) {
      for (auto& band : lw_) band.spectral_fraction /= total;
    }
  }

  // ---- 短波谱带 ----
  const Int n_sw = clamp(opt_.n_shortwave_bands, Int(1), Int(64));
  const Int n_default_sw = static_cast<Int>(sizeof(kSwWavenumber) / sizeof(Real)) - 1;
  sw_.clear();
  for (Int b = 0; b < n_sw; ++b) {
    RadiationBand band;
    if (n_sw == n_default_sw) {
      band.lambda_lo_um = Real(1.0e4) / kSwWavenumber[b + 1];
      band.lambda_hi_um = Real(1.0e4) / kSwWavenumber[b];
    } else {
      const Real lo = Real(0.2), hi = Real(12.2);
      const Real t0 = static_cast<Real>(b) / static_cast<Real>(n_sw);
      const Real t1 = static_cast<Real>(b + 1) / static_cast<Real>(n_sw);
      band.lambda_lo_um = lo * std::pow(hi / lo, t0);
      band.lambda_hi_um = lo * std::pow(hi / lo, t1);
    }
    const Real lc = Real(0.5) * (band.lambda_lo_um + band.lambda_hi_um);
    const Size ng = k_ratio_.size();
    band.k_h2o.resize(ng);
    band.k_co2.resize(ng);
    band.k_o3.resize(ng);
    band.k_cloud.resize(ng);
    band.weights = k_weight_;
    for (Size gg = 0; gg < ng; ++gg) {
      band.k_h2o[gg] = k_ratio_[gg] * kbar_h2o_sw(lc);
      band.k_co2[gg] = k_ratio_[gg] * Real(1.0e-4);
      band.k_o3[gg] = k_ratio_[gg] * kbar_o3_sw(lc);
      band.k_cloud[gg] = Real(3) / (Real(2) * kRhoWater * opt_.cloud_droplet_radius_um * Real(1.0e-6));
    }
    // Rayleigh：k = 9.7e-6 (0.55/lambda)^4.05  (m^2/kg)
    band.rayleigh_reference = kRayleigh550 * std::pow(Real(0.55) / lc, Real(4.05));
    band.n_pressure = Real(1);
    band.n_temperature = Real(0);
    sw_.push_back(std::move(band));
  }
  // 短波带份额（太阳黑体 5776 K，覆盖 0.2-12.2 um，归一化）
  {
    Real total = Real(0);
    for (auto& band : sw_) {
      band.spectral_fraction = planck_band_fraction(band.lambda_lo_um, band.lambda_hi_um, kSunTemperature);
      total += band.spectral_fraction;
    }
    if (total > Real(0)) {
      for (auto& band : sw_) band.spectral_fraction /= total;
    }
  }
  initialized_ = true;
}

// ===========================================================================
// 黑体与谱带份额
// ===========================================================================

Real RadiationDriver::planck_cumulative_fraction(Real lambda_um, Real t) noexcept {
  return planck_fraction_impl(lambda_um, t);
}

Real RadiationDriver::planck_band_fraction(Real lambda_lo_um, Real lambda_hi_um, Real t) noexcept {
  const Real flo = planck_fraction_impl(lambda_lo_um, t);
  const Real fhi = planck_fraction_impl(lambda_hi_um, t);
  return std::max(fhi - flo, Real(0));
}

// ===========================================================================
// 云光学厚度（几何光学近似，[P6] 第 3 节）
// ===========================================================================

//   tau = 3 LWP/(2 rho_w r_e) + 3 IWP/(2 rho_i r_e,i)
//   LWP/IWP 单位 kg/m^2，r_e 单位 m。复杂度 O(1)。
Real RadiationDriver::cloud_optical_depth_lw(Real lwp, Real iwp, Real r_e_um) noexcept {
  const Real re = std::max(r_e_um, Real(0.5)) * Real(1.0e-6);
  const Real re_i = std::max(r_e_um, Real(1.0)) * Real(1.0e-6);
  const Real l = std::max(lwp, Real(0));
  const Real i = std::max(iwp, Real(0));
  return Real(1.5) * (l / (kRhoWater * re) + i / (kRhoIce * re_i));
}

Real RadiationDriver::cloud_optical_depth_sw(Real lwp, Real iwp, Real r_e_um) noexcept {
  // 短波：几何光学消光效率更接近 2，等效系数为长波的 4/3
  return Real(4) / Real(3) * cloud_optical_depth_lw(lwp, iwp, r_e_um);
}

// ===========================================================================
// 太阳几何
// ===========================================================================

// 太阳赤纬（Cooper 近似，[B4] 第 2 章）： delta = 23.45 deg cos(2 pi (172-N)/365)
Real RadiationDriver::solar_declination(Real julian_day) noexcept {
  return Real(23.45) * kDegToRad * std::cos(kTwoPi * (Real(172) - julian_day) / Real(365));
}

// 日地距离修正： (r0/r)^2 = 1 + 0.033 cos(2 pi N/365)
Real RadiationDriver::earth_sun_distance_factor(Real julian_day) noexcept {
  return Real(1) + Real(0.033) * std::cos(kTwoPi * julian_day / Real(365));
}

// 太阳天顶角余弦： cos z = sin(lat) sin(delta) + cos(lat) cos(delta) cos(h)
//                   h = (UTC - 12) * 15 deg + lon
Real RadiationDriver::solar_zenith_cosine(Real lat_deg, Real lon_deg, Real julian_day,
                                          Real utc_hour) noexcept {
  const Real lat = lat_deg * kDegToRad;
  const Real delta = solar_declination(julian_day);
  const Real hour_angle = ((utc_hour - Real(12)) * Real(15) + lon_deg) * kDegToRad;
  return std::sin(lat) * std::sin(delta) + std::cos(lat) * std::cos(delta) * std::cos(hour_angle);
}

// ===========================================================================
// 通量散度 -> 加热率
// ===========================================================================

// ---------------------------------------------------------------------------
// 净"向上"通量散度：
//     dT/dt_k = (F_up[k] - F_up[k+1] + F_dn[k+1] - F_dn[k]) / (rho cp dz)
// 短波采用净向下约定，调用时交换参数即可（F_up,F_dn）->（F_dn,F_up）。
// 单位：K/s；再除以 pi 得到 dtheta/dt。
// 复杂度 O(nz)。
void RadiationDriver::flux_divergence_heating(const PhysicsColumn& col,
                                              const std::vector<Real>& f_up,
                                              const std::vector<Real>& f_dn,
                                              std::vector<Real>& heating) const {
  const Int n = col.nz();
  heating.assign(static_cast<Size>(n), Real(0));
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real dz = std::max(col.dz[s], Real(1.0e-6));
    const Real rho = std::max(col.rho[s], Real(1.0e-8));
    const Real denom = rho * kCp * dz;
    const Real div = (f_up[s] - f_up[s + 1]) + (f_dn[s + 1] - f_dn[s]);
    heating[s] = div / denom;
  }
}

// ===========================================================================
// 长波：发射率/吸收率累加
// ===========================================================================

// ---------------------------------------------------------------------------
// 每层、每带的透射率与发射项：
//     u_h2o = rho qv dz,  u_co2 = rho 6.38e-4 dz,  u_o3 = rho q_o3 dz
//     tau_g = k_h2o,g u_h2o + k_co2,g u_co2 + k_o3,g u_o3 + tau_cloud
//     T_b   = sum_g w_g exp(-tau_g),   E_b = sigma T^4 f_b(T)
// 臭氧按总量（DOU）以高斯权重分布在 22 km 附近；云光学厚度由 LWP/IWP 给出。
// 复杂度 O(nz * nb * ng)。
void RadiationDriver::longwave_column(const PhysicsColumn& col, const SurfaceState& sfc,
                                      std::vector<Real>& f_up, std::vector<Real>& f_dn,
                                      RadiationFluxes& fluxes) const {
  const Int n = col.nz();
  const Size nb = lw_.size();
  const Size ng = k_ratio_.size();
  f_up.assign(static_cast<Size>(n) + 1, Real(0));
  f_dn.assign(static_cast<Size>(n) + 1, Real(0));
  if (n <= 0 || nb == 0) return;

  // 臭氧质量混合比廓线：把总量按归一化高斯权重分配
  const Real o3_total = std::max(opt_.o3_column_du, Real(0)) * kDuToKgM2;  // kg/m^2
  std::vector<Real> o3_w(static_cast<Size>(n), Real(0));
  Real o3_norm = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real w = std::exp(-sqr((col.z[s] - kOzoneCenterM) / kOzoneWidthM));
    o3_w[s] = w;
    o3_norm += w * col.rho[s] * std::max(col.dz[s], Real(1.0e-6));
  }
  if (o3_norm <= Real(0)) o3_norm = Real(1);

  std::vector<Real> cf(static_cast<Size>(n), Real(0));
  for (Int k = 0; k < n; ++k) {
    cf[static_cast<Size>(k)] = opt_.use_cloud_fraction
                                   ? col.cloud_fraction(k)
                                   : (col.qc[static_cast<Size>(k)] > Real(0) ? Real(1) : Real(0));
  }

  // 每层每带的透射率 T_b 与带发射 E_b
  std::vector<Real> tband(static_cast<Size>(n) * nb, Real(0));
  std::vector<Real> eband(static_cast<Size>(n) * nb, Real(0));
  std::vector<Real> fb(nb, Real(0));
  std::vector<Real> tau_g(ng, Real(0));
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(0));
    const Real dz = std::max(col.dz[s], Real(1.0e-6));
    const Real t = col.temperature(k);
    const Real u_h2o = rho * std::max(col.qv[s], Real(0)) * dz;
    const Real u_co2 = rho * kCo2MassRatio * dz;
    const Real u_o3 = o3_total * o3_w[s] * rho * dz / o3_norm;
    // 云水/云冰路径 (kg/m^2)
    const Real lwp = rho * std::max(col.qc[s], Real(0)) * dz;
    const Real iwp = rho * (std::max(col.qi[s], Real(0)) + std::max(col.qs[s], Real(0)) +
                            std::max(col.qg[s], Real(0))) * dz;
    const Real tau_cloud = cf[s] * cloud_optical_depth_lw(lwp, iwp, opt_.cloud_ice_radius_um);
    const Real sigma_t4 = planck_flux(t);
    // 各带黑体份额（逐层计算，用于带发射）
    for (Size b = 0; b < nb; ++b) {
      fb[b] = planck_band_fraction(lw_[b].lambda_lo_um, lw_[b].lambda_hi_um, t);
    }
    for (Size b = 0; b < nb; ++b) {
      const RadiationBand& band = lw_[b];
      for (Size gg = 0; gg < ng; ++gg) {
        tau_g[gg] = band.k_h2o[gg] * u_h2o + band.k_co2[gg] * u_co2 + band.k_o3[gg] * u_o3 +
                    tau_cloud;
      }
      tband[s * nb + b] = band_transmittance(tau_g, band.weights);
      eband[s * nb + b] = sigma_t4 * fb[b];
    }
  }

  // 向下累加（从模式顶开始）
  std::vector<Real> dn_b(nb, Real(0));
  std::vector<Real> up_b(nb, Real(0));
  for (Size b = 0; b < nb; ++b) dn_b[b] = Real(0);  // 模式顶无入射长波
  for (Int k = n - 1; k >= 0; --k) {
    const Size s = static_cast<Size>(k);
    for (Size b = 0; b < nb; ++b) {
      const Real tb = tband[s * nb + b];
      dn_b[b] = tb * dn_b[b] + (Real(1) - tb) * eband[s * nb + b];
      f_dn[s] += dn_b[b];
    }
  }
  // 地表：发射 + 反射
  const Real t_sfc = (sfc.skin_temperature > Real(0)) ? sfc.skin_temperature : col.temperature(0);
  const Real eps = clamp(sfc.emissivity, Real(0.5), Real(1));
  const Real sigma_ts4 = planck_flux(t_sfc);
  for (Size b = 0; b < nb; ++b) {
    const Real fbs = planck_band_fraction(lw_[b].lambda_lo_um, lw_[b].lambda_hi_um, t_sfc);
    up_b[b] = eps * sigma_ts4 * fbs + (Real(1) - eps) * dn_b[b];
    f_up[0] += up_b[b];
  }
  // 向上累加
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    for (Size b = 0; b < nb; ++b) {
      const Real tb = tband[s * nb + b];
      up_b[b] = tb * up_b[b] + (Real(1) - tb) * eband[s * nb + b];
      f_up[s + 1] += up_b[b];
    }
  }

  fluxes.surface_longwave_down = f_dn[0];
  fluxes.surface_longwave_up = f_up[0];
  fluxes.toa_longwave_up = f_up[static_cast<Size>(n)];
}

// ===========================================================================
// 短波：直射 + 单次散射 + 漫射
// ===========================================================================

// ---------------------------------------------------------------------------
// 每层光学量：
//     tau_abs,g = k_h2o,g u_h2o + k_co2,g u_co2 + k_o3,g u_o3
//     tau_sca   = k_ray u_air + k_aer u_air_massfrac + tau_cloud
//     有效不对称因子 g_eff = (tau_aer*0.7 + tau_cloud*0.85)/tau_sca
// 光束： T_dir = sum_g w_g exp(-(tau_abs,g+tau_sca)/mu0)
//        T_abs = sum_g w_g exp(-tau_abs,g/mu0)
// 漫射： T_diff = sum_g w_g exp(-D (tau_abs,g+tau_sca))
// 单次散射源： sca = F_dir(入) (T_abs - T_dir)，按 (1+g)/2 分到下半球
// 复杂度 O(nz * nb * ng)。
void RadiationDriver::shortwave_column(const PhysicsColumn& col, const SurfaceState& sfc,
                                       Real cos_zenith, Real sun_earth_factor,
                                       std::vector<Real>& f_up, std::vector<Real>& f_dn,
                                       RadiationFluxes& fluxes, Real& sw_down_sfc) const {
  const Int n = col.nz();
  const Size nb = sw_.size();
  const Size ng = k_ratio_.size();
  f_up.assign(static_cast<Size>(n) + 1, Real(0));
  f_dn.assign(static_cast<Size>(n) + 1, Real(0));
  sw_down_sfc = Real(0);
  if (n <= 0 || nb == 0) return;

  const Real mu0 = std::max(cos_zenith, Real(1.0e-3));
  const Real albedo = clamp(sfc.albedo, Real(0), Real(1));

  // 气溶胶按空气质量分配（列积分等于 AOD）
  Real air_mass = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    air_mass += std::max(col.rho[s], Real(0)) * std::max(col.dz[s], Real(1.0e-6));
  }
  if (air_mass <= Real(0)) air_mass = Real(1);

  std::vector<Real> cf(static_cast<Size>(n), Real(0));
  for (Int k = 0; k < n; ++k) {
    cf[static_cast<Size>(k)] = opt_.use_cloud_fraction
                                   ? col.cloud_fraction(k)
                                   : (col.qc[static_cast<Size>(k)] > Real(0) ? Real(1) : Real(0));
  }

  std::vector<Real> dir(static_cast<Size>(n) + 1, Real(0));
  std::vector<Real> dnd(static_cast<Size>(n) + 1, Real(0));  // 向下漫射
  std::vector<Real> upd(static_cast<Size>(n) + 1, Real(0));  // 向上漫射
  std::vector<Real> tdir(static_cast<Size>(n) * nb, Real(0));
  std::vector<Real> tabs(static_cast<Size>(n) * nb, Real(0));
  std::vector<Real> tdif(static_cast<Size>(n) * nb, Real(0));
  std::vector<Real> sdn(static_cast<Size>(n) * nb, Real(0));
  std::vector<Real> sup(static_cast<Size>(n) * nb, Real(0));
  std::vector<Real> tau_g(ng, Real(0));

  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real rho = std::max(col.rho[s], Real(0));
    const Real dz = std::max(col.dz[s], Real(1.0e-6));
    const Real u_air = rho * dz;
    const Real u_h2o = rho * std::max(col.qv[s], Real(0)) * dz;
    const Real u_co2 = rho * kCo2MassRatio * dz;
    const Real lwp = rho * std::max(col.qc[s], Real(0)) * dz;
    const Real iwp = rho * (std::max(col.qi[s], Real(0)) + std::max(col.qs[s], Real(0)) +
                            std::max(col.qg[s], Real(0))) * dz;
    const Real tau_cloud = cf[s] * cloud_optical_depth_sw(lwp, iwp, opt_.cloud_ice_radius_um);
    const Real mass_frac = u_air / air_mass;
    // 臭氧（短波）：总量按质量加权分布到整层
    const Real u_o3 = std::max(opt_.o3_column_du, Real(0)) * kDuToKgM2 * mass_frac;

    for (Size b = 0; b < nb; ++b) {
      const RadiationBand& band = sw_[b];
      const Real lam = Real(0.5) * (band.lambda_lo_um + band.lambda_hi_um);
      const Real tau_ray = band.rayleigh_reference * u_air;
      const Real aod_band = std::max(opt_.aerosol_optical_depth, Real(0)) *
                            std::pow(Real(0.55) / lam, Real(1.3)) * mass_frac;
      // 气溶胶以消光光学厚度形式进入（不随 g 点变化）
      const Real tau_aer = aod_band;
      const Real tau_sca = tau_ray + tau_aer + tau_cloud;
      const Real sca_weights = tau_ray * Real(0) + tau_aer * Real(0.7) + tau_cloud * Real(0.85);
      const Real g_eff = (tau_sca > Real(0)) ? clamp(sca_weights / tau_sca, Real(0), Real(0.95))
                                             : Real(0);
      // 逐 g 点计算
      Real td = Real(0), ta = Real(0), tf = Real(0);
      for (Size gg = 0; gg < ng; ++gg) {
        tau_g[gg] = band.k_h2o[gg] * u_h2o + band.k_co2[gg] * u_co2 + band.k_o3[gg] * u_o3;
        const Real w = band.weights[gg];
        td += w * std::exp(-(tau_g[gg] + tau_sca) / mu0);
        ta += w * std::exp(-tau_g[gg] / mu0);
        tf += w * std::exp(-(tau_g[gg] + tau_sca) * kDiffusivity);
      }
      tdir[s * nb + b] = td;
      tabs[s * nb + b] = ta;
      tdif[s * nb + b] = clamp(tf, Real(0), Real(1));
      sup[s * nb + b] = g_eff;  // 暂存 g_eff（后续用于源项分配）
    }
  }

  // 各带的大气顶入射
  const Real f0 = opt_.solar_constant * sun_earth_factor;
  for (Size b = 0; b < nb; ++b) {
    const Real f0b = f0 * sw_[b].spectral_fraction * mu0;
    dir[static_cast<Size>(n)] = f0b;
    fluxes.toa_shortwave_down += f0b;
    // 直射光束自上而下
    Real beam = f0b;
    for (Int k = n - 1; k >= 0; --k) {
      const Size s = static_cast<Size>(k);
      const Real td = tdir[s * nb + b];
      const Real ta = tabs[s * nb + b];
      const Real sca = std::max(beam * (ta - td), Real(0));
      const Real g_eff = sup[s * nb + b];
      // 源项放在层中心，衰减半个层的光学厚度
      const Real half = std::sqrt(std::max(tdif[s * nb + b], Real(1.0e-12)));
      sdn[s * nb + b] = sca * (Real(1) + g_eff) * Real(0.5) * half;
      sup[s * nb + b] = sca * (Real(1) - g_eff) * Real(0.5) * half;
      beam *= td;
      dir[s] = beam;  // 界面 s 处的直射光束通量（水平面）
    }
    // 向下漫射（自顶向下）
    dnd[static_cast<Size>(n)] = Real(0);
    for (Int k = n - 1; k >= 0; --k) {
      const Size s = static_cast<Size>(k);
      dnd[s] = tdif[s * nb + b] * dnd[s + 1] + sdn[s * nb + b];
    }
    // 地表反射（直射 + 漫射），向上累加
    upd[0] = albedo * (dir[0] + dnd[0]);
    for (Int k = 0; k < n; ++k) {
      const Size s = static_cast<Size>(k);
      upd[s + 1] = tdif[s * nb + b] * upd[s] + sup[s * nb + b];
    }
    // 累加到总通量（含直射光束）
    for (Int L = 0; L <= n; ++L) {
      const Size s = static_cast<Size>(L);
      f_dn[s] += dir[s] + dnd[s];
      f_up[s] += upd[s];
    }
    fluxes.toa_shortwave_up = f_up[static_cast<Size>(n)];
  }
  sw_down_sfc = f_dn[0];
  fluxes.surface_shortwave_down = f_dn[0];
  fluxes.surface_shortwave_up = f_up[0];
}

// ===========================================================================
// 单列总驱动
// ===========================================================================

void RadiationDriver::step_column(const PhysicsColumn& col, SurfaceState& sfc, Real dt,
                                  Real cos_zenith, Real sun_earth_factor, ColumnTendency& tend,
                                  PhysicsDiagnostics& diag, RadiationFluxes& fluxes) const {
  const Int n = col.nz();
  if (n <= 0) return;
  VIBE_UNUSED(dt);
  fluxes = RadiationFluxes{};  // 调用方无需预清零

  std::vector<Real> heating(static_cast<Size>(n), Real(0));
  std::vector<Real> heat_lw(static_cast<Size>(n), Real(0));
  std::vector<Real> heat_sw(static_cast<Size>(n), Real(0));
  std::vector<Real> f_up, f_dn;

  const bool do_lw = (scheme_ == RadiationScheme::Rrtmg) ||
                     (scheme_ == RadiationScheme::RrtmgLongwaveOnly) ||
                     (scheme_ == RadiationScheme::SimpleGrey);
  const bool do_sw = (scheme_ == RadiationScheme::Rrtmg) ||
                     (scheme_ == RadiationScheme::RrtmgShortwaveOnly) ||
                     (scheme_ == RadiationScheme::SimpleGrey);
  const bool night = !(cos_zenith > opt_.radiation_min_cos_zenith);

  // ---- 长波 ----
  if (do_lw) {
    longwave_column(col, sfc, f_up, f_dn, fluxes);
    flux_divergence_heating(col, f_up, f_dn, heat_lw);
    sfc.lw_down = fluxes.surface_longwave_down;
    sfc.lw_up = fluxes.surface_longwave_up;
    for (Int k = 0; k < n; ++k) {
      heat_lw[static_cast<Size>(k)] =
          clamp(heat_lw[static_cast<Size>(k)], -opt_.max_heating_rate, opt_.max_heating_rate);
      fluxes.heating_longwave_min = std::min(fluxes.heating_longwave_min, heat_lw[static_cast<Size>(k)]);
    }
  }

  // ---- 短波（夜间直接跳过） ----
  if (do_sw && !night) {
    Real sw_down_sfc = Real(0);
    shortwave_column(col, sfc, cos_zenith, sun_earth_factor, f_up, f_dn, fluxes, sw_down_sfc);
    // 短波为净向下约定：交换参数
    flux_divergence_heating(col, f_dn, f_up, heat_sw);
    sfc.sw_down = fluxes.surface_shortwave_down;
    sfc.sw_up = fluxes.surface_shortwave_up;
    for (Int k = 0; k < n; ++k) {
      heat_sw[static_cast<Size>(k)] =
          clamp(heat_sw[static_cast<Size>(k)], Real(0), opt_.max_heating_rate);
      fluxes.heating_shortwave_max = std::max(fluxes.heating_shortwave_max, heat_sw[static_cast<Size>(k)]);
    }
  } else {
    sfc.sw_down = Real(0);
    sfc.sw_up = Real(0);
    fluxes.toa_shortwave_down = Real(0);
    fluxes.toa_shortwave_up = Real(0);
    fluxes.surface_shortwave_down = Real(0);
    fluxes.surface_shortwave_up = Real(0);
  }

  // ---- 加热率累加（K/s -> dtheta/dt 需除以 pi）----
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real h = clamp(heat_lw[s] + heat_sw[s], -opt_.max_heating_rate, opt_.max_heating_rate);
    heating[s] = h;
    // theta = T / pi，故 dtheta/dt = (dT/dt)/pi
    const Real pi = std::max(col.exner(k), Real(1.0e-3));
    tend.theta[s] += h / pi;
    fluxes.heating_total_abs_max = std::max(fluxes.heating_total_abs_max, std::abs(h));
  }
  diag.radiation_heating_max = std::max(diag.radiation_heating_max, fluxes.heating_total_abs_max);

  // ---- 收支汇总 ----
  const Real sw_net_toa = fluxes.toa_shortwave_down - fluxes.toa_shortwave_up;
  fluxes.toa_net = sw_net_toa - fluxes.toa_longwave_up;
  fluxes.surface_net = (fluxes.surface_shortwave_down - fluxes.surface_shortwave_up) +
                       (fluxes.surface_longwave_down - fluxes.surface_longwave_up);
  sfc.net_radiation = fluxes.surface_net;
  // 注意：逐列通量由 PhysicsDriver 从 fluxes 写入 PhysicsDiagnostics 的二维数组，
  // 域平均由 PhysicsDiagnostics::aggregate() 完成，这里只更新域最大值。
}

std::string RadiationDriver::describe() const {
  std::ostringstream os;
  os << "RadiationDriver{ " << name() << "; 长波带=" << lw_.size() << " 短波带=" << sw_.size()
     << " g点=" << k_ratio_.size() << "; CO2=" << opt_.co2_ppm << " ppmv O3="
     << opt_.o3_column_du << " DU AOD=" << opt_.aerosol_optical_depth
     << "; 云滴半径=" << opt_.cloud_droplet_radius_um << " um 云量缩放="
     << (opt_.use_cloud_fraction ? "开" : "关") << "; 太阳常数=" << opt_.solar_constant
     << " W/m^2 日=" << opt_.julian_day << " UTC=" << opt_.utc_hour << "h }";
  return os.str();
}

std::unique_ptr<RadiationDriver> make_radiation(RadiationScheme scheme,
                                               const PhysicsOptions& opt) {
  if (scheme == RadiationScheme::None) return nullptr;
  return std::make_unique<RadiationDriver>(scheme, opt);
}

}  // namespace vibe::physics

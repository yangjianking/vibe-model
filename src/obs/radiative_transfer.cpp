/// @file radiative_transfer.cpp
/// @brief 快速辐射传输：Planck、预报因子、光学厚度、逐层积分与解析雅可比。
///
/// 模型（教学级 RTTOV，[O1][O2]）
/// ------------------------------
/// 对每个通道 c，令波数 nu（cm^-1，微波通道用 GHz 的等效波数 nu = f_GHz/29.9792458）：
///
///   * 逐层光学厚度（无散射、局地热力学平衡）
///         Delta tau_k = sec(theta) * ( kappa_dry + w_k kappa_wv + o3_k kappa_o3 ) * m_k,
///         m_k = Delta p_k / g                      （静力质量路径，kg m^-2）
///   * 吸收系数用**线性化的幂律**（对 p、T 的依赖写成预报因子，见下）
///         kappa(T, p) = kappa0(nu) * [(1-a) + a p'] * [(1+b) - b T'],
///         p' = p / p_ref, T' = T / T_ref, p_ref = 1e5 Pa, T_ref = 250 K
///     该形式对 p'、T' 是双线性的，因此可以被 13 个预报因子的**线性**组合精确表示。
///   * 预报因子（每层 13 个，[O1] 的 predictor 体系）
///         X = [ m, m p', m T', m p' T',
///               m w, m w p', m w T', m w p' T', m w^2,
///               m o3, m o3 T', m o3 p', m o3 p' T' ]
///     a_c = [ kappa_dry0*(1-a_d)(1+b_d), kappa_dry0*a_d*(1+b_d),
///             -kappa_dry0*(1-a_d)*b_d,   -kappa_dry0*a_d*b_d,
///             kappa_wv0*(1-a_w)(1+b_w),  kappa_wv0*a_w*(1+b_w),
///             -kappa_wv0*(1-a_w)*b_w,    -kappa_wv0*a_w*b_w,  kappa_wv2,
///             kappa_o3*(1-a_o)(1+b_o),   -kappa_o3*(1-a_o)*b_o,
///             kappa_o3*a_o*(1+b_o),      -kappa_o3*a_o*b_o ]
///   * 透射率与辐射积分（上行、天底/倾斜观测；忽略散射与地表反射的下行项之外的
///     多次反射，地表反射用最低层 Planck 近似）
///         Theta_k = sum_{j>k} tau_j          （层 k 以上到太空的光学厚度）
///         t_k^u = exp(-Theta_k),  t_k^l = t_k^u exp(-tau_k),  W_k = t_k^u - t_k^l
///         t_s = exp(-sum_k tau_k)
///         I = eps B(T_s) t_s + sum_k B(T_k) W_k + (1-eps) t_s B(T_0)
///   * 亮温：T_b = B^{-1}(I)
///
/// 解析雅可比
/// ----------
/// 记 S_k = sum_{k'<k} B_{k'} W_{k'}，G_k = -eps B_s t_s - S_k + B_k t_k^l，则
///         dI/dT_k  = B'(T_k) W_k + (d tau_k/dT_k) G_k
///         dI/dqv_k = (d tau_k/dqv_k) G_k
///         dI/do3_k = (d tau_k/do3_k) G_k
///         dI/dT_s  = eps B'(T_s) t_s
///         dI/deps  = B_s t_s - t_s B(T_0)
/// 再乘以 dB^{-1}/dI = 1/B'(T_b) 得到 dT_b/d...。
///
/// 验证方法（有限差分，见 tests/unit/test_obs_operator.cpp）
/// --------------------------------------------------------
///   对 13 个状态的每个 T_k / qv_k / o3_k 做中心差分
///         (BT(x + eps e_i) - BT(x - eps e_i)) / (2 eps)
///   与 jacobian_column 的输出比较；eps 取 1e-3 K（温度）与 1e-6 kg/kg（水汽），
///   期望相对误差 < 1e-4（二阶截断误差 O(eps^2)）。
///
/// 复杂度：forward O(n_layers * 13)，jacobian 同阶（前缀和避免 O(n^2)）。
///
/// 文献：[O1] Saunders et al. (1999)；[O2] RTTOV v11 Users Guide；
///       [O3] Eyre (1990)；[O10] Rodgers (2000) 第 9 章。

#include "vibe/obs/radiative_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"

namespace vibe::obs {

// ===========================================================================
// 内部（detail）：Planck、预报因子、逐层量与积分
// 说明：这些函数放在具名命名空间 detail 中（外部链接），
//       以便 src/obs/obs_operators.cpp 的 RadianceOperator 复用**整层**积分，
//       而不需要修改冻结的 radiative_transfer.hpp。
// ===========================================================================
namespace detail {

/// 预报因子个数（见文件头注释）
inline constexpr int kNumPredictors = 13;
/// 参考气压（Pa）与参考温度（K）
inline constexpr Real kRefPressure = Real(1.0e5);
inline constexpr Real kRefTemperature = Real(250.0);
/// 最小/最大天顶角余弦（<= 84 度）
inline constexpr Real kMinCosZenith = Real(0.10);

/// 辐射常数（[O1]）：c1 = 2 h c^2（c 以 cm/s 计，输出 mW m^-2 sr^-1 cm^-1）
inline constexpr Real kC1 = Real(1.191042972e-5);
/// c2 = h c / k（K cm）
inline constexpr Real kC2 = Real(1.4387768775);
/// GHz -> 等效波数 cm^-1 的换算因子
inline constexpr Real kGhzToCm = Real(1.0 / 29.9792458);

/// 整层大气剖面（自下而上：k = 0 为最低层）
struct ColumnProfile {
  std::vector<Real> pressure;       ///< 层中心气压 (Pa)
  std::vector<Real> temperature;    ///< 层中心温度 (K)
  std::vector<Real> qv;             ///< 比湿 (kg/kg)
  std::vector<Real> ozone;          ///< 臭氧质量混合比 (kg/kg)
  std::vector<Real> cloud_water;    ///< 云水 (kg/kg)
  std::vector<Real> cloud_ice;      ///< 云冰 (kg/kg)
  Real zenith_angle_deg = Real(0);
  Real surface_temperature = Real(0);
  Real surface_pressure = Real(0);
  Real surface_emissivity = Real(1);
  Real skin_temperature = Real(0);

  Size n_layers() const noexcept { return pressure.size(); }
};

/// 由单层 LayerProfile 构造退化（n = 1）的整层剖面
ColumnProfile to_column(const LayerProfile& p);

/// Planck 函数（波数 cm^-1，辐射率 mW m^-2 sr^-1 cm^-1）
Real planck(Real wavenumber, Real temperature);

/// 逆 Planck：由辐射率反算亮温
Real inverse_planck(Real wavenumber, Real radiance);

/// dB/dT
Real planck_derivative(Real wavenumber, Real temperature);

/// 逐层预报因子（展平，长度 n_layers * kNumPredictors）
std::vector<Real> predictors_column(const ColumnProfile& col);

/// 逐层光学厚度（不含几何因子之外的处理；已含 sec(theta)）
std::vector<Real> optical_depth_column(const std::vector<Real>& coefs, const Channel& ch,
                                       const ColumnProfile& col);

/// 整层正演
RadianceResult forward_column(const std::vector<Real>& coefs, const Channel& ch,
                              const ColumnProfile& col);

/// 整层解析雅可比
RadianceJacobian jacobian_column(const std::vector<Real>& coefs, const Channel& ch,
                                 const ColumnProfile& col);

/// 逐层量（供正演与雅可比共享）
struct LayerQuantities {
  std::vector<Real> dp;        ///< Delta p_k (Pa)
  std::vector<Real> mass;      ///< Delta p_k / g
  std::vector<Real> p_rel;     ///< p / p_ref
  std::vector<Real> t_rel;     ///< T / T_ref
  std::vector<Real> w;         ///< qv
  std::vector<Real> o3;        ///< ozone
  std::vector<Real> tau;       ///< 光学厚度
  std::vector<Real> dtau_dT;   ///< d tau_k / d T_k
  std::vector<Real> dtau_dw;   ///< d tau_k / d qv_k
  std::vector<Real> dtau_do3;  ///< d tau_k / d o3_k
  Real secant = Real(1);
};

/// 计算逐层量与 d tau/d(...)（对 13 个预报因子逐项求导）
void compute_layers(const std::vector<Real>& a, const ColumnProfile& col,
                    LayerQuantities& q);

// ---------------------------------------------------------------------------
// 实现
// ---------------------------------------------------------------------------

Real planck(Real wavenumber, Real temperature) {
  const Real nu = wavenumber;
  const Real T = std::max(temperature, Real(1));
  const Real x = kC2 * nu / T;
  if (x > Real(700)) return Real(0);  // 防止 exp 上溢
  const Real ex = std::exp(x);
  return kC1 * nu * nu * nu / (ex - Real(1));
}

Real inverse_planck(Real wavenumber, Real radiance) {
  const Real nu = wavenumber;
  const Real b = std::max(radiance, Real(1e-30));
  const Real arg = Real(1) + kC1 * nu * nu * nu / b;
  if (arg <= Real(1) || !std::isfinite(arg)) {
    // 瑞利-金斯极限：T = c2 B / (c1 nu^2)
    return std::max(kC2 * b / (kC1 * nu * nu), Real(1));
  }
  return std::max(kC2 * nu / std::log(arg), Real(1));
}

Real planck_derivative(Real wavenumber, Real temperature) {
  const Real nu = wavenumber;
  const Real T = std::max(temperature, Real(1));
  const Real x = kC2 * nu / T;
  if (x > Real(700)) return Real(0);
  const Real ex = std::exp(x);
  const Real b = kC1 * nu * nu * nu / (ex - Real(1));
  return b * x / T * ex / (ex - Real(1));
}

ColumnProfile to_column(const LayerProfile& p) {
  ColumnProfile c;
  c.pressure.push_back(p.pressure);
  c.temperature.push_back(p.temperature);
  c.qv.push_back(p.qv);
  c.ozone.push_back(p.ozone);
  c.cloud_water.push_back(p.cloud_water);
  c.cloud_ice.push_back(p.cloud_ice);
  c.zenith_angle_deg = p.zenith_angle_deg;
  c.surface_temperature = p.surface_temperature;
  c.surface_pressure = p.surface_pressure;
  c.surface_emissivity = p.surface_emissivity;
  c.skin_temperature = p.skin_temperature;
  return c;
}

namespace {

/// 层压力厚度（自下而上、气压递减）：内部用相邻层中心差，边界用半层
std::vector<Real> layer_dp(const ColumnProfile& col) {
  const Size n = col.n_layers();
  std::vector<Real> dp(n, Real(0));
  if (n == 0) return dp;
  if (n == 1) {
    const Real ps = col.surface_pressure > col.pressure[0] ? col.surface_pressure
                                                           : col.pressure[0];
    dp[0] = std::max(ps - col.pressure[0], Real(0.25) * col.pressure[0]);
    return dp;
  }
  for (Size k = 0; k < n; ++k) {
    const Real p_up = (k + 1 < n) ? col.pressure[k + 1] : col.pressure[k] * Real(0.5);
    const Real p_dn = (k > 0) ? col.pressure[k - 1]
                              : std::max(col.surface_pressure, col.pressure[k]);
    dp[k] = std::max(Real(0.5) * (p_dn - p_up), Real(0));
  }
  return dp;
}

/// 天顶角余弦钳制后的 secant
Real secant_of(Real zenith_deg) {
  const Real c = std::cos(zenith_deg * kDegToRad);
  return Real(1) / std::max(c, kMinCosZenith);
}

/// 13 个预报因子在 (m, p', T', w, o3) 处的值
inline void predictor_values(Real m, Real pr, Real tr, Real w, Real o3, Real* x) {
  x[0]  = m;
  x[1]  = m * pr;
  x[2]  = m * tr;
  x[3]  = m * pr * tr;
  x[4]  = m * w;
  x[5]  = m * w * pr;
  x[6]  = m * w * tr;
  x[7]  = m * w * pr * tr;
  x[8]  = m * w * w;
  x[9]  = m * o3;
  x[10] = m * o3 * tr;
  x[11] = m * o3 * pr;
  x[12] = m * o3 * pr * tr;
}

/// dX_j/dT'（注意 dT'/dT = 1/T_ref）
inline void predictor_dT(Real m, Real pr, Real w, Real o3, Real* d) {
  d[0] = 0; d[1] = 0;
  d[2] = m;              d[3] = m * pr;
  d[4] = 0; d[5] = 0;
  d[6] = m * w;          d[7] = m * w * pr;
  d[8] = 0;
  d[9] = 0;              d[10] = m * o3;
  d[11] = 0;             d[12] = m * o3 * pr;
}

/// dX_j/dw
inline void predictor_dw(Real m, Real pr, Real tr, Real w, Real* d) {
  d[0] = 0; d[1] = 0; d[2] = 0; d[3] = 0;
  d[4] = m;              d[5] = m * pr;
  d[6] = m * tr;         d[7] = m * pr * tr;
  d[8] = Real(2) * m * w;
  d[9] = 0; d[10] = 0; d[11] = 0; d[12] = 0;
}

/// dX_j/do3
inline void predictor_do3(Real m, Real pr, Real tr, Real* d) {
  d[0] = 0; d[1] = 0; d[2] = 0; d[3] = 0;
  d[4] = 0; d[5] = 0; d[6] = 0; d[7] = 0; d[8] = 0;
  d[9] = m; d[10] = m * tr; d[11] = m * pr; d[12] = m * pr * tr;
}

}  // namespace

std::vector<Real> predictors_column(const ColumnProfile& col) {
  const Size n = col.n_layers();
  const auto dp = layer_dp(col);
  std::vector<Real> out(n * static_cast<Size>(kNumPredictors), Real(0));
  Real x[kNumPredictors];
  for (Size k = 0; k < n; ++k) {
    const Real m = dp[k] / kGravity;
    const Real pr = col.pressure[k] / kRefPressure;
    const Real tr = col.temperature[k] / kRefTemperature;
    predictor_values(m, pr, tr, col.qv[k], col.ozone[k], x);
    for (int j = 0; j < kNumPredictors; ++j) {
      out[k * static_cast<Size>(kNumPredictors) + static_cast<Size>(j)] = x[j];
    }
  }
  return out;
}

void compute_layers(const std::vector<Real>& a, const ColumnProfile& col,
                    LayerQuantities& q) {
  const Size n = col.n_layers();
  q.secant = secant_of(col.zenith_angle_deg);
  q.dp = layer_dp(col);
  q.mass.resize(n);
  q.p_rel.resize(n);
  q.t_rel.resize(n);
  q.w.resize(n);
  q.o3.resize(n);
  q.tau.assign(n, Real(0));
  q.dtau_dT.assign(n, Real(0));
  q.dtau_dw.assign(n, Real(0));
  q.dtau_do3.assign(n, Real(0));

  Real x[kNumPredictors], dt[kNumPredictors], dw[kNumPredictors], do3[kNumPredictors];
  for (Size k = 0; k < n; ++k) {
    q.mass[k] = q.dp[k] / kGravity;
    q.p_rel[k] = col.pressure[k] / kRefPressure;
    q.t_rel[k] = col.temperature[k] / kRefTemperature;
    q.w[k] = col.qv[k];
    q.o3[k] = col.ozone[k];
    predictor_values(q.mass[k], q.p_rel[k], q.t_rel[k], q.w[k], q.o3[k], x);
    predictor_dT(q.mass[k], q.p_rel[k], q.w[k], q.o3[k], dt);
    predictor_dw(q.mass[k], q.p_rel[k], q.t_rel[k], q.w[k], dw);
    predictor_do3(q.mass[k], q.p_rel[k], q.t_rel[k], do3);

    Real tau = Real(0), dT = Real(0), dW = Real(0), dO = Real(0);
    for (int j = 0; j < kNumPredictors; ++j) {
      tau += a[static_cast<Size>(j)] * x[j];
      dT += a[static_cast<Size>(j)] * dt[j];
      dW += a[static_cast<Size>(j)] * dw[j];
      dO += a[static_cast<Size>(j)] * do3[j];
    }
    // dT' / dT = 1 / T_ref
    q.tau[k] = q.secant * tau;
    q.dtau_dT[k] = q.secant * dT / kRefTemperature;
    q.dtau_dw[k] = q.secant * dW;
    q.dtau_do3[k] = q.secant * dO;
  }
}

std::vector<Real> optical_depth_column(const std::vector<Real>& coefs, const Channel& ch,
                                       const ColumnProfile& col) {
  VIBE_UNUSED(ch);
  LayerQuantities q;
  compute_layers(coefs, col, q);
  return q.tau;
}

RadianceResult forward_column(const std::vector<Real>& coefs, const Channel& ch,
                              const ColumnProfile& col) {
  const Size n = col.n_layers();
  RadianceResult r;
  r.layer_optical_depth.resize(n);
  r.weighting_function.resize(n);
  if (n == 0) return r;

  LayerQuantities q;
  compute_layers(coefs, col, q);
  const Real nu = static_cast<Real>(ch.wavenumber);

  // 地表温度与发射率
  Real t_surf = col.surface_temperature;
  if (!(t_surf > Real(0))) t_surf = col.skin_temperature;
  if (!(t_surf > Real(0))) t_surf = col.temperature[0];
  const Real emissivity = clamp(col.surface_emissivity, Real(0), Real(1));

  // Theta_k = sum_{j>k} tau_j（层 k 以上到太空的光学厚度）
  std::vector<Real> theta(n, Real(0));
  Real accum = Real(0);
  for (Size k = n; k-- > 0;) {
    theta[k] = accum;
    accum += q.tau[k];
  }
  const Real tau_total = accum;
  const Real t_surface = std::exp(-tau_total);

  Real I = emissivity * planck(nu, t_surf) * t_surface;
  I += (Real(1) - emissivity) * t_surface * planck(nu, col.temperature[0]);
  for (Size k = 0; k < n; ++k) {
    const Real tu = std::exp(-theta[k]);
    const Real tl = tu * std::exp(-q.tau[k]);
    const Real W = tu - tl;
    r.layer_optical_depth[k] = q.tau[k];
    I += planck(nu, col.temperature[k]) * W;

    // 权重函数 d t / d ln p：用层上下界的气压对数差归一
    const Real p_up = (k + 1 < n) ? col.pressure[k + 1]
                                  : std::max(col.pressure[k] - Real(0.5) * q.dp[k], Real(1));
    const Real p_dn = (k > 0) ? col.pressure[k - 1]
                              : std::max(col.pressure[k] + Real(0.5) * q.dp[k], Real(1));
    const Real dlnp = std::abs(std::log(std::max(p_dn, Real(1)) / std::max(p_up, Real(1))));
    r.weighting_function[k] = (dlnp > Real(0)) ? W / dlnp : W;
  }

  r.radiance = I;
  r.brightness_temperature = inverse_planck(nu, I);
  r.transmittance_to_surface = t_surface;
  return r;
}

RadianceJacobian jacobian_column(const std::vector<Real>& coefs, const Channel& ch,
                                 const ColumnProfile& col) {
  const Size n = col.n_layers();
  RadianceJacobian j;
  j.temperature.assign(n, Real(0));
  j.humidity.assign(n, Real(0));
  j.ozone.assign(n, Real(0));
  if (n == 0) return j;

  LayerQuantities q;
  compute_layers(coefs, col, q);
  const Real nu = static_cast<Real>(ch.wavenumber);

  Real t_surf = col.surface_temperature;
  if (!(t_surf > Real(0))) t_surf = col.skin_temperature;
  if (!(t_surf > Real(0))) t_surf = col.temperature[0];
  const Real emissivity = clamp(col.surface_emissivity, Real(0), Real(1));

  std::vector<Real> theta(n, Real(0)), tu(n, Real(0)), tl(n, Real(0)), W(n, Real(0)),
      Bk(n, Real(0)), dW(n, Real(0));
  Real accum = Real(0);
  for (Size k = n; k-- > 0;) {
    theta[k] = accum;
    accum += q.tau[k];
  }
  const Real tau_total = accum;
  const Real t_surface = std::exp(-tau_total);
  const Real Bs = planck(nu, t_surf);

  Real I = emissivity * Bs * t_surface;
  I += (Real(1) - emissivity) * t_surface * planck(nu, col.temperature[0]);
  for (Size k = 0; k < n; ++k) {
    tu[k] = std::exp(-theta[k]);
    tl[k] = tu[k] * std::exp(-q.tau[k]);
    W[k] = tu[k] - tl[k];
    Bk[k] = planck(nu, col.temperature[k]);
    dW[k] = tl[k];  // dW_k / d tau_k
    I += Bk[k] * W[k];
  }

  const Real bt = inverse_planck(nu, I);
  const Real dBdT_bt = planck_derivative(nu, bt);
  const Real scale = (dBdT_bt > Real(0)) ? Real(1) / dBdT_bt : Real(0);

  // 前缀和 S_k = sum_{k'<k} B_{k'} W_{k'}
  Real S = Real(0);
  for (Size k = 0; k < n; ++k) {
    const Real G = -emissivity * Bs * t_surface - S + Bk[k] * tl[k];
    j.temperature[k] = scale * (planck_derivative(nu, col.temperature[k]) * W[k] +
                                q.dtau_dT[k] * G);
    j.humidity[k] = scale * (q.dtau_dw[k] * G);
    j.ozone[k] = scale * (q.dtau_do3[k] * G);
    S += Bk[k] * W[k];
  }
  j.surface_temperature = scale * emissivity * planck_derivative(nu, t_surf) * t_surface;
  j.surface_emissivity = scale * (Bs * t_surface - t_surface * planck(nu, col.temperature[0]));
  return j;
}

}  // namespace detail

// ===========================================================================
// 教学系数生成器
// ===========================================================================

namespace {

/// 洛伦兹线型（微波用）
inline Real lorentz(Real nu, Real nu0, Real gamma) {
  const Real d = (nu - nu0) / gamma;
  return Real(1) / (Real(1) + d * d);
}

/// 高斯带型（红外用）
inline Real gauss_band(Real nu, Real nu0, Real sigma) {
  const Real d = (nu - nu0) / sigma;
  return std::exp(Real(-0.5) * d * d);
}

/// 逐通道的质量吸收系数标定（解析线强度模型）
struct AbsorptionScale {
  Real kappa_dry = Real(0);   ///< 干空气（CO2/O2）吸收
  Real kappa_wv = Real(0);    ///< 水汽吸收
  Real kappa_wv2 = Real(0);   ///< 水汽自加宽（w^2 项）
  Real kappa_o3 = Real(0);    ///< 臭氧吸收
  Real a_dry = Real(0.8), b_dry = Real(0.5);
  Real a_wv = Real(1.0), b_wv = Real(1.5);
  Real a_o3 = Real(0.5), b_o3 = Real(1.0);
};

/// 由通道波数（cm^-1）与谱域给出解析吸收尺度
AbsorptionScale absorption_scale(const Channel& ch) {
  const Real nu = static_cast<Real>(ch.wavenumber);
  AbsorptionScale s;
  if (ch.band == SpectralBand::Microwave) {
    // O2 60 GHz 复相带（等效波数 ~1.99 cm^-1）与 118 GHz 线（~3.95 cm^-1）
    s.kappa_dry = Real(1.2e-4) * gauss_band(nu, Real(1.99), Real(0.15)) +
                  Real(2.0e-5) * gauss_band(nu, Real(3.95), Real(0.25)) + Real(3.0e-6);
    // H2O 183.31 GHz（6.1146 cm^-1）、22.235 GHz（0.7417 cm^-1）与宽带连续统
    s.kappa_wv = Real(1.0e-2) * lorentz(nu, Real(6.1146), Real(0.12)) +
                 Real(3.5e-3) * lorentz(nu, Real(6.1146), Real(1.20)) +
                 Real(2.2e-2) * lorentz(nu, Real(0.7417), Real(0.05)) + Real(1.0e-4);
    s.kappa_wv2 = Real(0.15) * s.kappa_wv;
    s.kappa_o3 = Real(5.0e-4) * lorentz(nu, Real(0.735), Real(0.05));
    s.a_dry = Real(0.9); s.b_dry = Real(0.7);
    s.a_wv = Real(1.1);  s.b_wv = Real(1.8);
    s.a_o3 = Real(0.5);  s.b_o3 = Real(0.5);
  } else {
    // CO2 15 um 带（667.4 cm^-1）与红外连续统
    s.kappa_dry = Real(1.4e-4) * gauss_band(nu, Real(667.4), Real(22.0)) + Real(3.0e-6);
    // H2O 6.3 um 带（1595 cm^-1）与窗区连续统
    s.kappa_wv = Real(1.4e-2) * gauss_band(nu, Real(1595.0), Real(200.0)) + Real(4.0e-5);
    s.kappa_wv2 = Real(0.20) * s.kappa_wv;
    // O3 9.6 um 带（1042 cm^-1）
    s.kappa_o3 = Real(20.0) * gauss_band(nu, Real(1042.0), Real(28.0));
    s.a_dry = Real(0.8); s.b_dry = Real(0.5);
    s.a_wv = Real(1.0);  s.b_wv = Real(1.5);
    s.a_o3 = Real(0.5);  s.b_o3 = Real(1.0);
  }
  return s;
}

/// 由吸收尺度生成 13 个回归系数
std::vector<Real> coefficients_from_scale(const AbsorptionScale& s) {
  std::vector<Real> a(static_cast<Size>(detail::kNumPredictors), Real(0));
  // 干空气：kappa = kappa_dry [(1-a)+a p'][(1+b)-b T']
  a[0] = s.kappa_dry * (Real(1) - s.a_dry) * (Real(1) + s.b_dry);
  a[1] = s.kappa_dry * s.a_dry * (Real(1) + s.b_dry);
  a[2] = -s.kappa_dry * (Real(1) - s.a_dry) * s.b_dry;
  a[3] = -s.kappa_dry * s.a_dry * s.b_dry;
  // 水汽：kappa = kappa_wv w [(1-a)+a p'][(1+b)-b T'] + kappa_wv2 w^2
  a[4] = s.kappa_wv * (Real(1) - s.a_wv) * (Real(1) + s.b_wv);
  a[5] = s.kappa_wv * s.a_wv * (Real(1) + s.b_wv);
  a[6] = -s.kappa_wv * (Real(1) - s.a_wv) * s.b_wv;
  a[7] = -s.kappa_wv * s.a_wv * s.b_wv;
  a[8] = s.kappa_wv2;
  // 臭氧：kappa = kappa_o3 o3 [(1-a)+a p'][(1+b)-b T']
  a[9] = s.kappa_o3 * (Real(1) - s.a_o3) * (Real(1) + s.b_o3);
  a[10] = -s.kappa_o3 * (Real(1) - s.a_o3) * s.b_o3;
  a[11] = s.kappa_o3 * s.a_o3 * (Real(1) + s.b_o3);
  a[12] = -s.kappa_o3 * s.a_o3 * s.b_o3;
  return a;
}

/// 由 GHz 频率构造微波通道
Channel mw_channel(int id, double ghz, bool surface_sensitive, int pe) {
  Channel c;
  c.id = id;
  c.wavenumber = ghz * static_cast<double>(detail::kGhzToCm);
  c.band = SpectralBand::Microwave;
  c.solar_contribution = Real(0);
  c.surface_sensitive = surface_sensitive;
  c.pe = pe;
  return c;
}

/// 由波长（微米）构造红外通道
Channel ir_channel(int id, double wavelength_um, bool surface_sensitive, int pe) {
  Channel c;
  c.id = id;
  c.wavenumber = 10000.0 / wavelength_um;
  c.band = SpectralBand::Infrared;
  c.solar_contribution = Real(0);
  c.surface_sensitive = surface_sensitive;
  c.pe = pe;
  return c;
}

}  // namespace

// ===========================================================================
// RadiativeTransfer
// ===========================================================================

RadiativeTransfer::RadiativeTransfer(const std::vector<Channel>& channels)
    : channels_(channels) {
  coefs_.resize(channels_.size());
  for (Size c = 0; c < channels_.size(); ++c) {
    coefs_[c] = coefficients_from_scale(absorption_scale(channels_[c]));
  }
}

RadiativeTransfer RadiativeTransfer::builtin_channels(const std::string& set_name) {
  const std::string s = set_name.empty() ? std::string("amsua") : set_name;
  std::vector<Channel> ch;

  if (s == "mhs") {
    // MHS：89 / 157 / 183.31+-1 / +-3 / +-7 GHz（[O1] 表 1）
    ch.push_back(mw_channel(1, 89.0, true, 0));
    ch.push_back(mw_channel(2, 157.0, true, 1));
    ch.push_back(mw_channel(3, 182.311, false, 10));
    ch.push_back(mw_channel(4, 186.311, false, 8));
    ch.push_back(mw_channel(5, 190.311, false, 6));
  } else if (s == "hirs") {
    // HIRS 采样（波长微米）；pe 为峰能量层序号（0 近地面）
    const double wl[19] = {14.95, 14.71, 14.49, 14.22, 13.97, 13.64, 13.35,
                           12.66, 12.02, 11.11, 9.71,  8.30,  7.33,  6.52,
                           4.57,  4.52,  4.47,  4.45,  4.24};
    for (int i = 0; i < 19; ++i) {
      ch.push_back(ir_channel(i + 1, wl[i], wl[i] > 10.0, 14 - i / 2));
    }
  } else if (s == "seviri") {
    const double wl[8] = {0.635, 0.810, 1.640, 3.900, 6.250, 7.350, 8.700, 10.800};
    for (int i = 0; i < 8; ++i) {
      Channel c = ir_channel(i + 1, wl[i], wl[i] > 3.0, 2 + i / 2);
      if (wl[i] < 3.0) {
        c.band = SpectralBand::Visible;
        c.solar_contribution = Real(0.05 * (i + 1));
      }
      ch.push_back(c);
    }
  } else if (s == "ir" || s == "window") {
    ch.push_back(ir_channel(1, 11.0, true, 0));
    ch.push_back(ir_channel(2, 12.0, true, 0));
    ch.push_back(ir_channel(3, 8.7, true, 0));
    ch.push_back(ir_channel(4, 9.7, false, 8));
    ch.push_back(ir_channel(5, 7.3, false, 4));
    ch.push_back(ir_channel(6, 6.25, false, 6));
    ch.push_back(ir_channel(7, 4.57, false, 12));
    ch.push_back(ir_channel(8, 15.0, false, 14));
  } else {
    if (s != "amsua") {
      VIBE_WARN("[rt] 未知内置通道组 ", s, "，回退到 amsua");
    }
    // AMSU-A 15 通道（GHz 标称频率，[O1] 表 1）
    const double f[15] = {23.800,  31.400,  50.300,  52.800,  53.596,  54.400,
                          54.940,  55.500,  57.290344, 57.290344 + 0.217,
                          57.290344 + 0.3222, 57.290344 + 0.3222 + 0.048,
                          57.290344 + 0.3222 + 0.022, 57.290344 + 0.3222 + 0.010,
                          57.290344 + 0.3222 + 0.0045};
    const bool sfc[15] = {true, true, true, false, false, false, false, false,
                          false, false, false, false, false, false, true};
    for (int i = 0; i < 15; ++i) {
      ch.push_back(mw_channel(i + 1, f[i], sfc[i], i));
    }
  }
  return RadiativeTransfer(ch);
}

RadianceResult RadiativeTransfer::forward(int channel, const LayerProfile& profile) const {
  VIBE_CHECK(channel >= 0 && static_cast<Size>(channel) < channels_.size());
  const auto& ch = channels_[static_cast<Size>(channel)];
  const auto& a = coefs_[static_cast<Size>(channel)];
  return detail::forward_column(a, ch, detail::to_column(profile));
}

RadianceJacobian RadiativeTransfer::jacobian(int channel, const LayerProfile& profile) const {
  VIBE_CHECK(channel >= 0 && static_cast<Size>(channel) < channels_.size());
  const auto& ch = channels_[static_cast<Size>(channel)];
  const auto& a = coefs_[static_cast<Size>(channel)];
  return detail::jacobian_column(a, ch, detail::to_column(profile));
}

std::vector<Real> RadiativeTransfer::optical_depth(int channel,
                                                   const LayerProfile& profile) const {
  VIBE_CHECK(channel >= 0 && static_cast<Size>(channel) < channels_.size());
  const auto& ch = channels_[static_cast<Size>(channel)];
  const auto& a = coefs_[static_cast<Size>(channel)];
  return detail::optical_depth_column(a, ch, detail::to_column(profile));
}

std::vector<Real> RadiativeTransfer::predictors(const LayerProfile& profile,
                                                int channel) const {
  VIBE_UNUSED(channel);
  return detail::predictors_column(detail::to_column(profile));
}

Real RadiativeTransfer::planck(double wavenumber, Real temperature) const {
  return detail::planck(static_cast<Real>(wavenumber), temperature);
}

Real RadiativeTransfer::inverse_planck(double wavenumber, Real radiance) const {
  return detail::inverse_planck(static_cast<Real>(wavenumber), radiance);
}

void RadiativeTransfer::set_coefficients(const std::vector<std::vector<Real>>& coefs) {
  VIBE_CHECK_MSG(coefs.size() == channels_.size(),
                 "set_coefficients: 通道数与系数行数不一致");
  for (Size c = 0; c < coefs.size(); ++c) {
    VIBE_CHECK_MSG(coefs[c].size() == static_cast<Size>(detail::kNumPredictors),
                   "set_coefficients: 每行必须恰好 " +
                       std::to_string(detail::kNumPredictors) + " 个预报因子系数");
  }
  coefs_ = coefs;
}

std::string RadiativeTransfer::describe() const {
  std::ostringstream os;
  os << "RadiativeTransfer[channels=" << channels_.size()
     << " predictors=" << detail::kNumPredictors << "]";
  const char* band_name[] = {"infrared", "microwave", "visible"};
  for (Size c = 0; c < channels_.size(); ++c) {
    const auto& ch = channels_[c];
    const int bi = static_cast<int>(ch.band);
    os << "\n  ch" << ch.id << " nu=" << ch.wavenumber << " cm^-1 ("
       << (bi >= 0 && bi < 3 ? band_name[bi] : "?") << ")"
       << (ch.surface_sensitive ? " surface-sensitive" : "") << " pe=" << ch.pe;
  }
  // 系数矩阵的若干统计（诊断系数量级）
  if (!coefs_.empty()) {
    Real lo = kHuge, hi = -kHuge, s = 0;
    Size n = 0;
    for (const auto& row : coefs_) {
      for (Real v : row) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
        s += std::abs(v);
        ++n;
      }
    }
    os << "\n  系数: |a| 均值=" << (n ? s / static_cast<Real>(n) : Real(0))
       << " 范围=[" << lo << ", " << hi << "]";
  }
  return os.str();
}

}  // namespace vibe::obs

#pragma once
/// @file radiation.hpp
/// @brief 辐射参数化：相关 k 分布长波/短波接口与 RRTMG 型实现。
///
/// 结构（[P5] RRTM；[P6] RRTMG）
/// ------------------------------
///   * 谱带划分：长波默认 16 带（3.08-1000 um），短波默认 14 带（0.2-12.2 um），
///     可通过 PhysicsOptions 调整；带内的谱积分用**相关 k 分布**近似
///         T_band = sum_g w_g exp( -sum_s k_{g,s}(p,T) u_s )
///     其中 u_s 为吸收剂质量路径 (kg/m^2)，g 点为 k 分布的求积点。
///   * 长波：发射率/吸收率累加法（向上、向下两趟累加），
///     层发射率 eps = 1 - T_band，黑体发射用 pi B = sigma T^4 的带份额。
///   * 短波：太阳直射光束（Beer-Lambert）+ 单次散射源 + 漫射衰减，
///     含太阳高度角、日地距离修正、Rayleigh/气溶胶/云消光与表面反射。
///   * 云：光学厚度用几何光学近似 tau = 3 LWP/(2 rho_w r_e)（[P6] 第 3 节）。
///
/// 与动力学的耦合
/// --------------
///   step_column 只写**加热率** (K/s) 到 ColumnTendency::theta，
///   辐射通量写入 SurfaceState 与 PhysicsDiagnostics。
///   辐射按 radiation_cadence 调用（见 PhysicsDriver），两次调用之间加热率保持不变
///   （即"加热率的时间平流"，[D16] 第 8 章）。
///
/// 文献：[P5] Mlawer et al. (1997)；[P6] Iacono et al. (2008)。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/types.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/physics/physics_types.hpp"

namespace vibe::physics {

// ---------------------------------------------------------------------------
/// 单列辐射通量与加热率诊断（向下为正；大气顶向上长波 = OLR）
struct RadiationFluxes {
  Real toa_shortwave_down = Real(0);   ///< 大气顶入射短波 (W/m^2)
  Real toa_shortwave_up = Real(0);     ///< 大气顶反射短波
  Real toa_longwave_up = Real(0);      ///< 出射长波 OLR
  Real toa_net = Real(0);              ///< 净入射（短波净 - OLR）
  Real surface_shortwave_down = Real(0);
  Real surface_shortwave_up = Real(0);
  Real surface_longwave_down = Real(0);
  Real surface_longwave_up = Real(0);
  Real surface_net = Real(0);          ///< 地表净辐射（向下为正）
  Real heating_longwave_min = Real(0); ///< 长波最小加热率 (K/s)
  Real heating_shortwave_max = Real(0);///< 短波最大加热率 (K/s)
  Real heating_total_abs_max = Real(0);
};

// ---------------------------------------------------------------------------
/// 谱带定义 + 相关 k 分布求积信息
struct RadiationBand {
  Real lambda_lo_um = Real(0);
  Real lambda_hi_um = Real(0);
  Real spectral_fraction = Real(0);        ///< 该带占总光谱的份额（黑体积分）
  std::vector<Real> k_h2o;                 ///< H2O 质量吸收系数 (m^2/kg)
  std::vector<Real> k_co2;
  std::vector<Real> k_o3;
  std::vector<Real> k_cloud;               ///< 云的质量消光系数 (m^2/kg)
  std::vector<Real> weights;               ///< g 点权重（和为 1）
  Real rayleigh_reference = Real(0);       ///< 550 nm 处 Rayleigh 质量系数 (m^2/kg)
  Real n_pressure = Real(0.5);             ///< 系数压力标度指数
  Real n_temperature = Real(0.5);          ///< 系数温度标度指数
};

// ===========================================================================
/// 辐射驱动
class RadiationDriver {
 public:
  RadiationDriver(RadiationScheme scheme, const PhysicsOptions& opt);

  RadiationScheme scheme() const noexcept { return scheme_; }
  const char* name() const noexcept;
  const PhysicsOptions& options() const noexcept { return opt_; }

  /// 建立谱带表与 k 分布求积点（幂等）。
  /// g 仅用于（将来）按网格度量建表；可为 nullptr（单元测试）。
  void initialize(const grid::Grid* g = nullptr);
  bool initialized() const noexcept { return initialized_; }

  // -------------------------------------------------------------------------
  /// 单列辐射积分。
  /// @param col        输入列（不被修改）
  /// @param sfc        输入/输出：读取 albedo/emissivity/地表温度，写回向下/向上通量
  /// @param dt         物理步长 (s)，仅用于加热率限幅
  /// @param cos_zenith 太阳天顶角余弦（<= 0 视为夜间）
  /// @param sun_earth_factor 日地距离修正因子 (r0/r)^2
  /// @param tend       输出：加热率 (K/s) 累加到 tend.theta
  /// @param diag       输出：通量诊断
  /// @param fluxes     输出：本列通量收支
  void step_column(const PhysicsColumn& col, SurfaceState& sfc, Real dt, Real cos_zenith,
                   Real sun_earth_factor, ColumnTendency& tend, PhysicsDiagnostics& diag,
                   RadiationFluxes& fluxes) const;

  // ---- 太阳几何 ----
  /// 太阳赤纬（Cooper 近似，[B4] 第 2 章）
  ///     delta = 23.45 deg * cos( 2 pi (172 - N) / 365 )
  static Real solar_declination(Real julian_day) noexcept;
  /// 太阳天顶角余弦（含时角）
  ///     cos z = sin(lat) sin(delta) + cos(lat) cos(delta) cos(h)
  ///     h = (UTC - 12) * 15 deg + lon
  static Real solar_zenith_cosine(Real lat_deg, Real lon_deg, Real julian_day,
                                  Real utc_hour) noexcept;
  /// 日地距离修正 (r0/r)^2 = 1 + 0.033 cos(2 pi N / 365)
  static Real earth_sun_distance_factor(Real julian_day) noexcept;

  // ---- 黑体与谱带份额 ----
  /// 黑体辐射通量 pi B = sigma T^4 (W/m^2)
  static Real planck_flux(Real t) noexcept { return kStefanBoltzmann * t * t * t * t; }
  /// 累积 Planck 函数 f(lambda T)：0 至 lambda 的黑体辐射份额（无量纲）
  ///     f(x) = 15/pi^4 * sum_{n=1}^inf e^{-n a/x} (a^3/(n^3 x^3) + 3a^2/(n^2 x^2)
  ///                                            + 6a/(n x) + 6) / n^4
  ///     a = h c / k = 14388.0 um K
  static Real planck_cumulative_fraction(Real lambda_um, Real t) noexcept;
  /// 谱带 [lo, hi] 内黑体份额
  static Real planck_band_fraction(Real lambda_lo_um, Real lambda_hi_um, Real t) noexcept;

  // ---- 云光学厚度（几何光学近似，[P6]） ----
  /// 长波： tau = 3 LWP/(2 rho_w r_e) + 3 IWP/(2 rho_i r_e,i)
  static Real cloud_optical_depth_lw(Real lwp, Real iwp, Real r_e_um) noexcept;
  /// 短波：同式，另含短波下多次散射的等效因子 2
  static Real cloud_optical_depth_sw(Real lwp, Real iwp, Real r_e_um) noexcept;

  // ---- 相关 k 分布 ----
  /// Gauss-Laguerre 求积（权重归一化到 1，节点为 e^{-x} 权下的高斯点）
  /// 用于把带内 k 分布积分
  ///     Integral_0^inf exp(-k) exp(-k u) dk = 1/(1+u)  ~= sum_i w_i exp(-k_i u)
  /// 离散化：Laguerre 递推 + Newton 求根，复杂度 O(n^2)。
  static void gauss_laguerre(Int n, std::vector<Real>& nodes, std::vector<Real>& weights);
  /// 构造相关 k 分布求积点：k_ratio_i = 节点（均值 1），权重和为 1
  static void build_k_distribution(Int n_g, std::vector<Real>& k_ratio,
                                   std::vector<Real>& weights);

  Int longwave_bands() const noexcept { return static_cast<Int>(lw_.size()); }
  Int shortwave_bands() const noexcept { return static_cast<Int>(sw_.size()); }
  Int g_points() const noexcept { return static_cast<Int>(opt_.n_g_points); }
  const std::vector<RadiationBand>& longwave_table() const noexcept { return lw_; }
  const std::vector<RadiationBand>& shortwave_table() const noexcept { return sw_; }

  std::string describe() const;

 private:
  /// 长波单列积分（向上/向下累加）
  void longwave_column(const PhysicsColumn& col, const SurfaceState& sfc,
                       std::vector<Real>& f_up, std::vector<Real>& f_dn,
                       RadiationFluxes& fluxes) const;
  /// 短波单列积分（直射 + 单次散射 + 漫射）
  void shortwave_column(const PhysicsColumn& col, const SurfaceState& sfc, Real cos_zenith,
                        Real sun_earth_factor, std::vector<Real>& f_up,
                        std::vector<Real>& f_dn, RadiationFluxes& fluxes, Real& sw_down_sfc) const;
  /// 把界面净通量散度转为加热率 (K/s)
  void flux_divergence_heating(const PhysicsColumn& col, const std::vector<Real>& f_up,
                               const std::vector<Real>& f_dn, std::vector<Real>& heating) const;

  RadiationScheme scheme_;
  PhysicsOptions opt_;
  bool initialized_ = false;
  std::vector<RadiationBand> lw_;
  std::vector<RadiationBand> sw_;
  std::vector<Real> k_ratio_;        ///< g 点相对 k（均值 1）
  std::vector<Real> k_weight_;       ///< g 点权重（和为 1）
};

/// 辐射工厂：None -> nullptr
std::unique_ptr<RadiationDriver> make_radiation(RadiationScheme scheme,
                                                const PhysicsOptions& opt);

}  // namespace vibe::physics

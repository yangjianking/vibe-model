#pragma once
/// @file physics_types.hpp
/// @brief 物理参数化模块的公共数据类型：单列视图、地表状态、方案枚举、选项与诊断。
///
/// 设计约定（见 docs/design/00_architecture.md 第 3、6、12 节与
/// docs/design/08_physics.md）：
///   * 物理量一律以 **单列视图** (PhysicsColumn) 在内存中连续排列，
///     水平方向由 driver 循环，便于向量化与 GPU 化；
///   * 物理过程只输出 **倾向**（dyn::PhysicsTendency），不直接改写 dyn::State；
///     唯一的例外是陆面/海面状态（土壤温度湿度、表皮温度），它们由
///     PhysicsDriver 内部持有并跨时间步保持；
///   * 所有水物质混合比为正定守恒量，任何参数化都必须做正定保护。
///
/// 坐标系约定
/// ----------
/// z 为几何高度（m，向上为正，z[0] 为最低的模式层中心）；p 为气压（Pa）；
/// theta 为位温（K）；pi = (p/p0)^{Rd/cp} 为 Exner 函数；混合比单位为 kg/kg。
///
/// 文献：[P1] Kessler 暖雨、[P8] Mellor-Yamada 闭合、[P9] Monin-Obukhov、
///       [P14] Noilhan-Planton 陆面、[P18] Janjic MYJ。

#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"

namespace vibe::physics {

// ===========================================================================
// 1. 方案枚举与字符串转换
// ===========================================================================

/// 微物理方案
enum class MicrophysicsScheme : int {
  None = 0,      ///< 关闭
  Kessler = 1,   ///< 暖雨（饱和调整 + 自动转换 + 收集 + 雨蒸发）[P1]
  Thompson = 2,  ///< 6 类双参数（云水/雨/云冰/雪/霰 + 水汽）[P3]
  Morrison = 3,  ///< 双参数 5 类 [P4]（本模块只提供接口与缺省转发）
  Wsm6 = 4,      ///< WSM6 单参数 6 类 [P2]（接口保留）
  Count = 5
};

/// 辐射方案
enum class RadiationScheme : int {
  None = 0,
  Rrtmg = 1,              ///< 长波 + 短波，相关 k 分布 [P5][P6]
  RrtmgLongwaveOnly = 2,  ///< 只做长波（夜间/冷启动）
  RrtmgShortwaveOnly = 3,
  SimpleGrey = 4,         ///< 灰体教学版本（单带）
  Count = 5
};

/// 边界层方案
enum class PblScheme : int {
  None = 0,
  Ysu = 1,          ///< YSU：K 廓线 + 逆梯度（非局地）项 [P7]
  Myj = 2,          ///< Mellor-Yamada 2.5 阶 TKE 闭合 [P8][P18]
  Mynn = 3,         ///< MYNN 接口保留
  Smagorinsky = 4,  ///< 局地 Smagorinsky 型 K [P16]
  Count = 5
};

/// 陆面/海面方案
enum class SurfaceScheme : int {
  None = 0,
  MoninObukhov = 1,  ///< 仅相似理论通量（无土壤预报）
  Noah = 2,          ///< 五层土壤温度/湿度（Noilhan-Planton 简化）[P14]
  Slab = 3,          ///< 单层平板（热容量 + 固定湿度）
  Count = 4
};

/// 积云对流方案
enum class CumulusScheme : int {
  None = 0,
  KainFritsch = 1,   ///< KF 质量通量 + CAPE 消耗闭合 [P11]
  GrellDevenyi = 2,  ///< GD 集合框架 [P12]
  Tiedtke = 3,       ///< 接口保留 [P13]
  Count = 4
};

const char* to_string(MicrophysicsScheme s) noexcept;
const char* to_string(RadiationScheme s) noexcept;
const char* to_string(PblScheme s) noexcept;
const char* to_string(SurfaceScheme s) noexcept;
const char* to_string(CumulusScheme s) noexcept;

MicrophysicsScheme microphysics_from_string(const std::string& name);
RadiationScheme radiation_from_string(const std::string& name);
PblScheme pbl_from_string(const std::string& name);
SurfaceScheme surface_from_string(const std::string& name);
CumulusScheme cumulus_from_string(const std::string& name);

// ===========================================================================
// 2. 热力学工具（Bolton 饱和水汽压、Exner、虚温）
// ===========================================================================

// ---------------------------------------------------------------------------
// 饱和水汽压（液面）
// ---------------------------------------------------------------------------
/// 公式（Bolton 1980，基于 Clausius-Clapeyron 的拟合式，[B4] 第 2 章）
///     e_s(T) = 611.2 * exp( 17.67 * T_c / (T_c + 243.5) )   [Pa]
///     其中 T_c = T - 273.15
/// 离散化：无（逐点解析式）。
/// 文献：[B4] Holton & Hakim 热力学；[D16] WRF ARW 技术说明（第 2 章）。
/// 复杂度：O(1)，含一次 exp。
Real saturation_vapor_pressure(Real t_k) noexcept;

/// 饱和水汽压（冰面，Buck 1981 拟合，[D16]）
///     e_i(T) = 611.15 * exp( 22.452 * T_c / (T_c + 272.55) )
Real saturation_vapor_pressure_ice(Real t_k) noexcept;

/// 饱和水汽压（按相态自动选择：T < T0 且 ice_phase 时用冰面）
inline Real saturation_vapor_pressure_phase(Real t_k, bool ice_phase) noexcept {
  return (ice_phase && t_k < kT0) ? saturation_vapor_pressure_ice(t_k)
                                  : saturation_vapor_pressure(t_k);
}

/// 饱和混合比（kg/kg）
///     q_s = eps * e_s / (p - e_s),   eps = Rd/Rv = 0.62197
/// 复杂度：O(1)。
Real saturation_mixing_ratio(Real p, Real t_k, bool ice_phase = false) noexcept;

/// 由混合比反算水汽压： e = q p / (eps + q)
Real vapor_pressure_from_mixing_ratio(Real p, Real qv) noexcept;

/// 相对湿度（相对水面，0..1.2 之间不截断；调用方自行限制）
Real relative_humidity_from_qv(Real p, Real t_k, Real qv) noexcept;

/// 露点温度（由 Bolton 反函数）
///     T_d = 243.5 * ln(e/611.2) / (17.67 - ln(e/611.2)) + 273.15
Real dewpoint_temperature(Real p, Real qv) noexcept;

/// Exner 函数 pi = (p/p0)^(Rd/cp)
inline Real exner_from_pressure(Real p) noexcept {
  return std::pow(p / kP0, kKappa);
}

/// 温度 T = theta * pi
inline Real temperature_from_theta_pi(Real theta, Real pi) noexcept { return theta * pi; }

/// 位温 theta = T / pi
inline Real theta_from_temperature_pressure(Real t_k, Real p) noexcept {
  return t_k / exner_from_pressure(p);
}

/// 虚温（含凝结态水物质的质量负载，[B4] 式 (2.?) 的标准近似）
///     T_v = T * (1 + 0.608 qv - qc - qi - qs - qg)
inline Real virtual_temperature(Real t_k, Real qv, Real qc = Real(0), Real qi = Real(0),
                                Real qs = Real(0), Real qg = Real(0)) noexcept {
  return t_k * (Real(1) + Real(0.608) * qv - qc - qi - qs - qg);
}

/// 湿空气密度（含虚温）： rho = p / (Rd * Tv)
inline Real density_from_pressure_tv(Real p, Real tv) noexcept { return p / (kRd * tv); }

/// 汽化潜热（温度相关）： Lv(T) = (2.501 - 0.00237 * T_c) * 1e6
inline Real latent_heat_vaporization(Real t_k) noexcept {
  return (Real(2.501) - Real(0.00237) * (t_k - kT0)) * Real(1.0e6);
}

/// 升华潜热： Ls = Lv(T) + Lf
inline Real latent_heat_sublimation(Real t_k) noexcept {
  return latent_heat_vaporization(t_k) + kLf;
}

/// 融化潜热： Lf（常数近似）
inline Real latent_heat_fusion(Real /*t_k*/) noexcept { return kLf; }

// ===========================================================================
// 3. 单列大气视图
// ===========================================================================

// ---------------------------------------------------------------------------
/// 一列大气的输入/输出视图。
///
/// 数组长度全部为 nz（不变量由 resize() 保证）。u/v 位于标量层中心
/// （物理参数化使用层中心风，不需要 C-grid 错位信息）；tke 为
/// 湍流动能 e = q^2/2 (m^2/s^2)，由 PBL 方案预报/诊断。
///
/// 复杂度：所有成员函数对单列均为 O(nz)。
struct PhysicsColumn {
  std::vector<Real> z;      ///< 层中心几何高度 (m)，z[0] 最低
  std::vector<Real> p;      ///< 气压 (Pa)
  std::vector<Real> rho;    ///< 干空气密度 (kg/m^3)
  std::vector<Real> theta;  ///< 位温 (K)
  std::vector<Real> qv;     ///< 水汽混合比 (kg/kg)
  std::vector<Real> qc;     ///< 云水
  std::vector<Real> qr;     ///< 雨水
  std::vector<Real> qi;     ///< 云冰
  std::vector<Real> qs;     ///< 雪
  std::vector<Real> qg;     ///< 霰
  std::vector<Real> u;      ///< 纬向风 (m/s)
  std::vector<Real> v;      ///< 经向风 (m/s)
  std::vector<Real> tke;    ///< 湍流动能 (m^2/s^2)
  std::vector<Real> dz;     ///< 层厚度 (m)，界面间距

  /// 分配/重置为 nz 层全零列
  void resize(Int nz);
  /// 置零（保留长度）
  void zero();
  /// 层数
  Int nz() const noexcept { return static_cast<Int>(z.size()); }
  /// 是否为空
  bool empty() const noexcept { return z.empty(); }
  /// 全部数组长度一致
  bool consistent() const noexcept;

  // ---- 逐层热力学诊断 -----------------------------------------------------
  /// Exner 函数 (p/p0)^kappa
  Real exner(Int k) const noexcept { return exner_from_pressure(p[static_cast<Size>(k)]); }
  /// 温度 T = theta * pi
  Real temperature(Int k) const noexcept {
    return temperature_from_theta_pi(theta[static_cast<Size>(k)], exner(k));
  }
  /// 饱和混合比（按温度与相位自动选择）
  Real saturation_mixing_ratio(Int k, bool ice_phase = false) const noexcept;
  /// 相对湿度
  Real relative_humidity(Int k) const noexcept {
    return relative_humidity_from_qv(p[static_cast<Size>(k)], temperature(k),
                                     qv[static_cast<Size>(k)]);
  }
  /// 虚温
  Real virtual_temperature(Int k) const noexcept;
  /// 位温虚温 theta_v = theta (1 + 0.608 qv - qc - qi - qs - qg)
  Real theta_v(Int k) const noexcept;
  /// 由状态方程重算密度
  void diagnose_density();
  /// 由 z 重算层厚 dz（内部层中心差分，顶底层单边）
  void diagnose_layer_depth();
  /// 层界面高度（nz+1，出口）
  void interface_heights(std::vector<Real>& zf) const;

  // ---- 水物质与守恒 -------------------------------------------------------
  /// 该层总水物质 qv+qc+qr+qi+qs+qg
  Real total_water(Int k) const noexcept;
  /// 整列水路径 (kg/m^2)： sum_k qx_k * rho_k * dz_k
  Real water_path() const noexcept;
  /// 整列水汽路径 (kg/m^2)
  Real vapor_path() const noexcept;
  /// 整列凝结态水路径 (kg/m^2)
  Real condensate_path() const noexcept;
  /// 云水路径 LWP、云冰路径 IWP (g/m^2)
  Real liquid_water_path() const noexcept;
  Real ice_water_path() const noexcept;
  /// 整列水物质质量 (kg/m^2)，用于守恒检查
  Real total_water_mass() const noexcept;
  /// 所有预报量有限
  bool finite() const noexcept;
  /// 水物质正定性：所有 qx >= 0
  bool positive_definite() const noexcept;
  /// 把负值截断到 0，返回被截断的总量 (kg/kg)
  Real clip_negatives() noexcept;
  /// 单层温度廓线（出口）
  void temperatures(std::vector<Real>& t) const;
  /// 湿度加权云量诊断（Xu-Randall 型简化，[P6]）
  Real cloud_fraction(Int k) const noexcept;

  /// 文字摘要
  std::string describe() const;
};

/// 理想探空列（测试与理想试验用）
///   * 高度线性分层 z_k = (k+0.5) * z_top / nz
///   * 温度按固定递减率 gamma 从 t_sfc 递减，再转位温
///   * 湿度按 qv = qv_sfc * exp(-z/H) 递减（H = 2 km）
///   * 风按 u = u_sfc + shear * z 线性切变
/// 复杂度：O(nz)。
PhysicsColumn ideal_sounding(Int nz, Real z_top = Real(20000), Real p_sfc = kP0,
                             Real t_sfc = Real(288.0), Real qv_sfc = Real(0.012),
                             Real lapse = Real(6.5e-3), Real shear = Real(2.0e-3),
                             Real u_sfc = Real(2.0));

// ===========================================================================
// 4. 单列物理倾向
// ===========================================================================

// ---------------------------------------------------------------------------
/// 单列倾向容器：所有物理方案写这里，driver 再散射到 dyn::PhysicsTendency。
///
/// 单位：theta (K/s)、水物质 (kg/kg/s)、u/v (m/s^2)、tke (m^2/s^3)。
struct ColumnTendency {
  std::vector<Real> theta, qv, qc, qr, qi, qs, qg, u, v, tke;

  void resize(Int nz);
  void zero();
  Int nz() const noexcept { return static_cast<Int>(theta.size()); }
  /// *this += a * other
  void add_scaled(Real a, const ColumnTendency& other);
  /// *this = a * other + b * (*this)
  void axpy(Real a, const ColumnTendency& other, Real b);
  /// 最大绝对值（用于稳定诊断）
  Real max_abs() const noexcept;
  bool finite() const noexcept;
};

// ===========================================================================
// 5. 地表状态
// ===========================================================================

// ---------------------------------------------------------------------------
/// 地表状态：跨时间步保持（土壤温度/湿度、表皮温度、粗糙度）。
///
/// 能量平衡（[P14] 式 (1)，[P15]）
///     R_n = H + LE + G
///     R_n = (1-a) S^down + L^down - sigma T_g^4
///     H   = -rho cp u* theta*      (感热，向上为正)
///     LE  = -rho Lv u* q*          (潜热)
///     G   = -lambda dT/dz|_sfc     (土壤热通量)
///
/// 通量符号约定：向下辐射为正；H、LE、G 以**向上**为正（与
/// dyn::PhysicsTendency 的 surface_flux_heat 符号一致）。
struct SurfaceState {
  // ---- 几何与类型 ----
  Real terrain_height = Real(0);          ///< 地形高度 (m)
  Real land_fraction = Real(1);           ///< 陆面比例 0..1（1 = 纯陆）
  Real roughness = kRoughnessLand;        ///< 动量粗糙度 z0 (m)
  Real roughness_heat = Real(1.0e-3);     ///< 热量粗糙度 z0h (m)
  Real roughness_moist = Real(1.0e-3);    ///< 水汽粗糙度 z0q (m)
  Real albedo = kAlbedoLand;              ///< 反照率
  Real emissivity = Real(0.98);           ///< 长波发射率

  // ---- 五层土壤 ----
  std::vector<Real> soil_temperature;     ///< 五层土壤温度 (K)
  std::vector<Real> soil_moisture;        ///< 五层体积含水量 (m^3/m^3)
  std::vector<Real> soil_depth;           ///< 五层厚度 (m)，长度 = nsoil
  Real soil_field_capacity = Real(0.3);   ///< 田间持水量
  Real soil_wilting_point = Real(0.1);    ///< 凋萎点

  // ---- 大气变量 ----
  Real surface_pressure = kP0;            ///< 地面气压 (Pa)
  Real t2 = Real(288.0);                  ///< 2 m 温度 (K)
  Real q2 = Real(0.010);                  ///< 2 m 比湿/混合比 (kg/kg)
  Real u10 = Real(0);                     ///< 10 m 纬向风 (m/s)
  Real v10 = Real(0);                     ///< 10 m 经向风 (m/s)
  Real skin_temperature = Real(288.0);    ///< 表皮温度 (K)

  // ---- 通量与辐射 ----
  Real sensible_heat_flux = Real(0);      ///< H (W/m^2，向上为正)
  Real latent_heat_flux = Real(0);        ///< LE (W/m^2，向上为正)
  Real ground_heat_flux = Real(0);        ///< G (W/m^2，向上为正)
  Real momentum_flux = Real(0);           ///< |tau| (N/m^2)
  Real friction_velocity = Real(0);       ///< u* (m/s)
  Real temperature_scale = Real(0);       ///< theta* (K)
  Real moisture_scale = Real(0);          ///< q* (kg/kg)
  Real obukhov_length = Real(kHuge);      ///< L (m)，中性时为大数
  Real sw_down = Real(0);                 ///< 向下短波 (W/m^2)
  Real lw_down = Real(0);                 ///< 向下长波 (W/m^2)
  Real sw_up = Real(0);                   ///< 向上短波
  Real lw_up = Real(0);                   ///< 向上长波
  Real net_radiation = Real(0);           ///< 净辐射 (W/m^2，向下为正)
  Real layer_depth = Real(0);             ///< 最底层大气厚度 (m)，用于交换系数

  /// 分配土壤层（nsoil = 5 时按 0.05/0.10/0.20/0.30/0.35 分配厚度）
  void resize_soil(Int nsoil);
  /// 土壤层数
  Int nsoil() const noexcept { return static_cast<Int>(soil_temperature.size()); }
  /// 是否为水点（land_fraction < 0.5 且非冰）
  bool is_water() const noexcept { return land_fraction < Real(0.5); }
  /// 2 m 温度/湿度一致性检查
  bool finite() const noexcept;
  /// 文字摘要
  std::string describe() const;
};

// ===========================================================================
// 6. Mellor-Yamada 闭合常数
// ===========================================================================

/// MY 闭合常数。
///  * original_my82(): (A1,A2,B1,B2,C1) = (0.92, 0.74, 16.6, 10.1, 0.08) [P8]
///  * janjic_1994()  : 非奇异化后的等价值 [P18]，与 WRF MYJ 一致
struct MellorYamadaConstants {
  Real a1 = Real(0.659888514560862645);
  Real a2 = Real(0.6574209922667784586);
  Real b1 = Real(11.87799326209552761);
  Real b2 = Real(7.226971804046074028);
  Real c1 = Real(0.000830955950095854396);
  Real s_q = Real(0.2);   ///< TKE 扩散的湍流 Schmidt 数倒数 S_q [P8]
  Real btg = Real(9.80665) / Real(273.0);  ///< beta * g，beta = 1/273 [P18]

  static MellorYamadaConstants original_my82() noexcept;
  static MellorYamadaConstants janjic_1994() noexcept { return {}; }
};

// ===========================================================================
// 7. 物理选项
// ===========================================================================

// ---------------------------------------------------------------------------
/// 物理参数化总选项。所有可调常数集中在此，便于配置与参数表（见 08_physics.md）。
struct PhysicsOptions {
  // ---- 时间与网格 ----
  Real dt_physics = Real(10);      ///< 物理调用步长 (s)
  Real dx = Real(1000), dy = Real(1000);
  Real z_top = Real(20000);
  Int nx = 1, ny = 1, nz = 40;
  Real latitude = kReferenceLat;   ///< 域中心纬度 (deg)
  Real longitude = Real(0);        ///< 域中心经度 (deg)

  // ---- 调度 ----
  Int radiation_cadence = 1;       ///< 每多少步调用一次辐射
  Int pbl_cadence = 1;
  Int microphysics_cadence = 1;
  Int cumulus_cadence = 1;
  Real radiation_heating_relax = Real(1);  ///< 新加热率权重（其余取上一步，平滑"加热率平流"）

  // ---- 方案选择 ----
  MicrophysicsScheme microphysics = MicrophysicsScheme::Kessler;
  RadiationScheme radiation = RadiationScheme::Rrtmg;
  PblScheme pbl = PblScheme::Ysu;
  SurfaceScheme surface = SurfaceScheme::Noah;
  CumulusScheme cumulus = CumulusScheme::KainFritsch;

  // ---- 通用正定与阈值 ----
  Real qv_min = Real(1.0e-12);   ///< 水汽下限
  Real qx_min = Real(1.0e-12);   ///< 凝结态水物质下限
  Real qc_min = Real(1.0e-6);    ///< 云水阈值（低于此不参与转换）
  bool positive_definite = true; ///< 是否强制 qx >= 0
  bool dry_atmosphere = false;   ///< 关闭所有水物质过程（教学检验）

  // ---- Kessler 微物理 [P1] ----
  Real kessler_qc0 = Real(1.0e-3);        ///< 自动转换阈值 (kg/kg)
  Real kessler_k1 = Real(1.0e-3);         ///< 自动转换率 (1/s)
  Real kessler_k2 = Real(2.2);            ///< 雨收集云水系数 (无量纲)
  Real kessler_evap_coef = Real(1.0);     ///< 雨蒸发系数修正
  Real kessler_rain_a = Real(36.34);      ///< 雨落速 Vt = a (rho qr)^b
  Real kessler_rain_b = Real(0.1364);
  Real kessler_cloud_evap_rate = Real(1.0e-5);  ///< 云水蒸发率 (1/s)

  // ---- Thompson 微物理 [P3] ----
  Real thompson_nc = Real(1.0e8);        ///< 云滴数浓度 (m^-3)
  Real thompson_nr = Real(1.0e6);        ///< 雨滴数浓度 (m^-3)
  Real thompson_ni = Real(1.0e5);        ///< 云冰数浓度 (m^-3)
  Real thompson_ns = Real(3.0e4);        ///< 雪数浓度 (m^-3)
  Real thompson_ng = Real(4.0e4);        ///< 霰数浓度 (m^-3)
  Real thompson_bergeron_rate = Real(0.0059);  ///< Bergeron 转换系数 c (1/s)
  Real thompson_bigg_a = Real(0.66);     ///< Bigg 冻结温度系数 (1/K)
  Real thompson_bigg_b = Real(100.0);    ///< Bigg 冻结强度 (m^-3 s^-1)
  Real thompson_riming_coef = Real(0.5); ///< 霰淞附系数
  Real thompson_ice_autoconv = Real(1.0e-3);  ///< 冰自动转换率 (1/s)
  Real thompson_autoconv_time = Real(1000.0); ///< 云水->雨水松弛时间 (s)
  Real thompson_autoconv_radius_um = Real(10.0);  ///< 临界体积平均半径 (um)

  // ---- 辐射 [P5][P6] ----
  Int n_longwave_bands = 16;       ///< 长波谱带数（RRTM 为 16）
  Int n_shortwave_bands = 14;      ///< 短波谱带数（RRTM 为 14）
  Int n_g_points = 4;              ///< 每个谱带的相关 k 分布求积点数
  Real co2_ppm = kCO2ppm;
  Real o3_column_du = kOzoneColumnDU;
  Real aerosol_optical_depth = Real(0.10);  ///< 550 nm 气溶胶光学厚度
  Real aerosol_ssa = Real(0.95);
  Real aerosol_asymmetry = Real(0.70);
  Real solar_constant = kSolarConstant;
  Real cloud_droplet_radius_um = Real(10);
  Real cloud_ice_radius_um = Real(30);
  Int julian_day = 172;            ///< 年积日（1..366）
  Real utc_hour = Real(12);
  Real radiation_min_cos_zenith = Real(0.01);  ///< 低于此值视为夜间
  bool use_cloud_fraction = true;  ///< 云光学厚度是否按云量线性缩放
  bool enable_tendency_physics = true;  ///< 是否把物理倾向写入 dyn::PhysicsTendency

  // ---- PBL ----
  MellorYamadaConstants my;
  Real myj_el0max = Real(1000);    ///< Blackadar 渐近混合长上限 (m) [P18]
  Real myj_el0min = Real(1);       ///< 下限 (m)
  Real myj_alph = Real(0.30);      ///< 渐近混合长系数 (Blackadar)
  Real myj_e_min = Real(1.0e-4);   ///< 最小 TKE (m^2/s^2)
  Real ysu_ri_critical = Real(0.25);    ///< PBL 顶临界总体 Richardson 数 [P7]
  Real ysu_entrainment_coef = Real(0.15);
  Real ysu_prandtl_coef = Real(2.1);    ///< Pr = 1 + coef * z/h
  Real ysu_countergradient_coef = Real(15.0);  ///< 逆梯度项系数 C
  Real ysu_wstar_coef = Real(1.0);      ///< 对流速度尺度权重 (phi_m * kappa)

  // ---- 陆面/海面 [P14][P15] ----
  Int n_soil_layers = 5;
  Real soil_thermal_conductivity = Real(1.0);   ///< W/(m K)
  Real soil_heat_capacity = kSoilHeatCap;       ///< J/(m^3 K)
  Real soil_hydraulic_diffusivity = Real(2.0e-7);  ///< m^2/s
  Real soil_deep_temperature = Real(288.0);
  Real soil_deep_moisture = Real(0.25);
  Real skin_layer_capacity = Real(2.0e4);       ///< J/(m^2 K)
  Real sst = Real(288.15);                      ///< 固定海表温度 (K)
  Real charnock_alpha = Real(0.018);
  Real charnock_viscous = Real(0.11);
  Int  mo_max_iterations = 40;
  Real mo_tolerance = Real(1.0e-8);
  Real land_evap_beta_coef = Real(1.0);         ///< 土壤湿度对蒸发的限制系数

  // ---- 积云 [P11][P12] ----
  /// 卷入率 (1/m)：3e-4 对应每公里约 35% 的质量增加，适用于深对流（[P11] 的
  /// 无量纲 0.03 表述换算为 1/m 后为本值；数值受云厚与 CAPE 敏感性控制）
  Real kf_entrainment_rate = Real(3.0e-4);
  Real kf_detrainment_rate = Real(1.5e-4);  ///< 卷出率 (1/m)
  Real kf_min_cape = Real(1000.0);         ///< 触发所需最小 CAPE (J/kg)
  Real kf_min_cloud_depth = Real(4000.0);  ///< 最小云厚 (m)
  Real kf_max_cloud_depth = Real(16000.0);
  Real kf_closure_time = Real(1800.0);     ///< CAPE 消耗时间尺度 (s)
  Real kf_downdraft_rate = Real(0.10);     ///< 下沉气流质量通量比
  Real kf_precip_efficiency_land = Real(0.5);
  Real kf_precip_efficiency_sea = Real(0.7);
  Int  gd_ensemble_size = 12;              ///< Grell-Devenyi 成员数 [P12]
  Real gd_spread = Real(0.7);              ///< 成员参数扰动幅度

  // ---- 次网格动量与水平扩散 [P16] ----
  bool use_smagorinsky = true;
  Real smagorinsky_coef = Real(0.20);
  Real horizontal_diffusion_coef = Real(0.0);   ///< 额外常数扩散 (m^2/s)

  // ---- 数值稳定 ----
  Real cfl_limit = Real(0.5);              ///< 水平 CFL 上限（物理诊断）
  Real vertical_cfl_limit = Real(0.9);     ///< 垂直 CFL 上限（落速/扩散）
  Real max_heating_rate = Real(2.0e-2);    ///< 加热率上限 (K/s)，防止积分发散
  Real max_theta_increment = Real(20.0);   ///< 单步位温变化上限 (K)

  /// 默认选项
  static PhysicsOptions defaults() noexcept { return {}; }
  /// 一致性校验（失败抛 ConfigError）
  void validate() const;
  /// 参数表文字（写入日志/文档）
  std::string describe() const;
};

// ===========================================================================
// 8. 物理诊断
// ===========================================================================

// ---------------------------------------------------------------------------
/// 物理诊断容器。二维字段按行主序 (j*nx + i) 存放。
struct PhysicsDiagnostics {
  Int nx = 0, ny = 0;

  // 二维诊断（长度 nx*ny）
  std::vector<Real> precipitation_grid;       ///< 网格尺度降水率 (mm/s)
  std::vector<Real> precipitation_convective; ///< 对流降水率 (mm/s)
  std::vector<Real> pbl_height;               ///< PBL 高度 (m)
  std::vector<Real> cloud_fraction;           ///< 整层平均云量 (0..1)
  std::vector<Real> cape;                     ///< CAPE (J/kg)
  std::vector<Real> cin;                      ///< CIN (J/kg)
  std::vector<Real> cloud_base;               ///< 对流云底 (m)
  std::vector<Real> cloud_top;                ///< 对流云顶 (m)
  std::vector<Real> surface_latent_flux;      ///< 地表潜热通量 (W/m^2)
  std::vector<Real> surface_sensible_flux;    ///< 地表感热通量 (W/m^2)
  std::vector<Real> surface_net_radiation;    ///< 地表净辐射 (W/m^2)
  std::vector<Real> toa_outgoing_longwave;    ///< OLR (W/m^2)
  std::vector<Real> toa_net_shortwave;        ///< 大气顶净短波 (W/m^2)

  // 域平均标量
  Real precipitation_grid_mean = Real(0);
  Real precipitation_grid_max = Real(0);
  Real precipitation_convective_mean = Real(0);
  Real precipitation_total_mean = Real(0);
  Real cloud_fraction_mean = Real(0);
  Real liquid_water_path_mean = Real(0);   ///< g/m^2
  Real ice_water_path_mean = Real(0);      ///< g/m^2
  Real pbl_height_mean = Real(0);
  Real pbl_height_max = Real(0);
  Real tke_mean = Real(0);                 ///< m^2/s^2
  Real tke_max = Real(0);
  Real toa_net_shortwave_mean = Real(0);
  Real toa_outgoing_longwave_mean = Real(0);
  Real toa_net_flux_mean = Real(0);        ///< 净入射（短波 - OLR）
  Real surface_net_radiation_mean = Real(0);
  Real surface_sw_down_mean = Real(0);
  Real surface_lw_down_mean = Real(0);
  Real surface_flux_heat_mean = Real(0);   ///< W/m^2
  Real surface_flux_moist_mean = Real(0);  ///< W/m^2
  Real surface_flux_momentum_mean = Real(0);  ///< N/m^2
  Real radiation_heating_max = Real(0);    ///< K/s
  Real convective_heating_max = Real(0);   ///< K/s
  Real mass_conservation_residual = Real(0);  ///< 微物理水物质最大相对残差
  Real energy_balance_residual_max = Real(0); ///< 陆面能量平衡最大残差 (W/m^2)

  // 计数
  Int radiation_calls = 0, microphysics_calls = 0, pbl_calls = 0, cumulus_calls = 0;
  Int pbl_cadence = 1;

  void resize(Int nx_in, Int ny_in);
  /// 清空所有统计量（保留分配）
  void reset();
  /// 行主序下标
  Size index(Int i, Int j) const noexcept { return static_cast<Size>(j) * static_cast<Size>(nx) + static_cast<Size>(i); }
  /// 由二维场汇总域平均量
  void aggregate();
  std::string summarize() const;
};

// ===========================================================================
// 9. 气块抬升（CAPE/CIN）与云量诊断
// ===========================================================================

/// 气块抬升结果
struct ParcelAscent {
  Real cape = Real(0);   ///< J/kg
  Real cin = Real(0);    ///< J/kg（负值）
  Int lfc = -1;          ///< 自由对流高度所在层（无则为 -1）
  Int el = -1;           ///< 平衡高度所在层
  Real plcl = Real(0);   ///< 抬升凝结高度气压 (Pa)
  Real zlcl = Real(0);   ///< LCL 高度 (m)
  Real theta_e_parcel = Real(0);
  Real theta_e_env_0 = Real(0);
};

// ---------------------------------------------------------------------------
/// 气块法 CAPE/CIN（[B4] 第 3 章；[P11] Kain-Fritsch 气块法）
///
/// 公式：
///     CAPE = g * Integral_{z_LFC}^{z_EL} (Tv_p - Tv_e)/Tv_e dz
///     CIN  = g * Integral_{z_b}^{z_LFC} (Tv_p - Tv_e)/Tv_e dz   (负值)
/// 离散化：气块自 source_level 干绝热抬升至 LCL（theta 与 qv 守恒），
/// 之后沿假相当位温守恒的湿绝热线上升（Newton 迭代求饱和温度），
/// 逐层用梯形法累加浮力。若 entraining = true，气块按
///     d(ln m)/dz = eps
/// 卷入环境空气（[P11] 式 (1)）。
/// 文献：[P11] Kain & Fritsch (1990)；[B4] Holton & Hakim (2013)。
/// 复杂度：O(nz * n_iter)，n_iter ≈ 4。
ParcelAscent moist_parcel_ascent(const PhysicsColumn& col, Int source_level,
                                 bool entraining = false, Real entrainment_rate = Real(0));

/// 相当位温（Bolton 1980 的近似式，[B4]）
///     theta_e = theta * exp( Lv qv / (cp T) )  (T = theta * pi)
Real equivalent_potential_temperature(const PhysicsColumn& col, Int k) noexcept;

/// 饱和相当位温（用于气块抬升）
Real saturated_equivalent_potential_temperature(Real theta, Real qv, Real p, Real t) noexcept;

// ===========================================================================
// 10. 守恒与稳定诊断
// ===========================================================================

/// 水物质总量（整列积分，kg/m^2）；微物理守恒检查用
Real column_total_water(const PhysicsColumn& col) noexcept;

/// 相对残差 |after-before| / (|before| + eps)
Real relative_water_residual(Real before, Real after) noexcept;

/// 物理过程的 CFL 数集合
struct PhysicsCfl {
  Real horizontal = Real(0);   ///< max(|u|,|v|) dt / dx
  Real fall_speed = Real(0);   ///< max 落速 dt / min(dz)（微物理）
  Real diffusion = Real(0);    ///< 2 K dt / min(dz)^2（PBL 垂直扩散）
  Real vertical = Real(0);     ///< max(fall_speed, diffusion)
  Real total = Real(0);        ///< max(horizontal, vertical)
  Real min_dz = Real(0);       ///< 最小层厚 (m)
};

/// 计算 CFL（微物理落速与湍流扩散由调用方提供）
PhysicsCfl compute_cfl(const PhysicsColumn& col, Real dt, Real dx,
                       Real max_fall_speed, Real max_diffusivity) noexcept;

}  // namespace vibe::physics

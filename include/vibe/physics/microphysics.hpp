#pragma once
/// @file microphysics.hpp
/// @brief 微物理参数化：抽象接口、公共相变速率与两种方案（Kessler / Thompson）。
///
/// 收支约定
/// --------
///   * 所有水物质为混合比 q (kg/kg)，且必须保持正定；
///   * step_column() 就地把 PhysicsColumn 推进 dt，并返回**地面降水率** (mm/s)；
///     在本模块单位制下 1 mm/s 恰好对应 1 kg/(m^2 s)（rho_w = 1000 kg/m^3）；
///   * 潜热通过位温回写： dtheta = L * dq_cond/(cp * pi)，pi = (p/p0)^(Rd/cp)；
///   * 守恒检查：列内水物质增量 + 地面降水累积 = 0（机器精度量级），
///     残差写入 PhysicsDiagnostics::mass_conservation_residual。
///
/// 离散化
/// ------
/// 所有源汇项用显式向前欧拉（对 dt 一阶），但配合
///   (a) 饱和调整的 Newton 迭代（隐式处理凝结-潜热耦合），
///   (b) 沉降的上游通量法 + 可用质量限制（无条件正定）
/// 使方案在本模式允许的 dt（<= 30 s）下稳定。
///
/// 文献：[P1] Kessler (1969)、[P2] Lin et al. (1983)、[P3] Thompson et al. (2008)。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/types.hpp"
#include "vibe/physics/physics_types.hpp"

namespace vibe::physics {

// ===========================================================================
// 1. 公共相变与粒子谱工具
// ===========================================================================

// ---------------------------------------------------------------------------
/// 单个粒子（液滴/冰粒子）的相变质量速率 (kg/s)，正值为增长。
///
/// 公式（Maxwell-Mason，[P2] 式 (A1)-(A4)；[P3] 式 (A2)-(A6)）
///     dm/dt = 4 pi r f_v (rho_v - rho_vs) / [ 1 + L^2 D_v rho_vs / (K_a R_v T^2) ]
/// 其中 rho_v = rho qv, rho_vs = rho qs 为水汽密度与饱和水汽密度，
/// L 为相变潜热（液面 Lv、冰面 Ls），K_a = 2.4e-2 W/(m K) 空气导热率，
/// D_v = 2.21e-5 m^2/s 水汽扩散系数，f_v 为通风因子。
///
/// 离散化：解析式；分母的修正项表示潜热释放对扩散生长的抑制作用。
/// 复杂度：O(1)（一次 pow 计算 e_s 由 qs 传入，不重复计算）。
Real particle_mass_rate(Real t, Real p, Real rho, Real qv, Real qs, Real radius,
                        bool ice_phase = false, Real ventilation = Real(1)) noexcept;

/// 通风因子（[P2] 式 (A6)）
///     f_v = 0.78 + 0.308 * Sc^(1/3) * Re^(1/2),  Re = 2 r V rho / mu
/// Sc = 0.6，mu = 1.8e-5 Pa s（常数近似）。
Real ventilation_factor(Real radius, Real fall_speed, Real rho) noexcept;

/// 指数粒子谱斜率 lambda（[P2] 式 (2)）
///     N(D) = N0 exp(-lambda D),  m(D) = a D^b
///     lambda = ( a N0 Gamma(1+b) / (rho q) )^(1/b)
/// 复杂度：O(1)（一次 Gamma 与 pow）。
Real slope_parameter(Real rho, Real q, Real n0, Real a, Real b) noexcept;

/// 数浓度 N = N0 / lambda
Real number_concentration(Real n0, Real lambda) noexcept;

/// 指数谱下 D^d 的质量加权平均（[P2] 式 (3)）
///     <D^d>_m = Gamma(4+d)/Gamma(4) * lambda^(-d)
Real mass_weighted_diameter_power(Real lambda, Real d) noexcept;

/// 指数谱下的质量加权平均落速 V = c D^d
///     V_mean = c * Gamma(4+d)/Gamma(4) * lambda^(-d)
inline Real mass_weighted_fall_speed(Real lambda, Real c, Real d) noexcept {
  return c * mass_weighted_diameter_power(lambda, d);
}

// ===========================================================================
// 2. 饱和调整
// ===========================================================================

/// 饱和调整结果
struct SaturationAdjust {
  Real condensed = Real(0);  ///< 净凝结量 (kg/kg)，正 = 凝结（潜热加热）
  Real dtheta = Real(0);     ///< 位温变化 (K)
  int iterations = 0;
  Real residual = Real(0);   ///< |qv - qs| / max(qs, eps)
};

// ---------------------------------------------------------------------------
/// 单点饱和调整（Newton 迭代，隐式处理凝结与潜热）
///
/// 公式
///     qv - delta = q_s( p, T0 + (L/cp) delta )
///     delta 为凝结量；T0 = theta' * pi，气块在该步内保持在 mode 层
/// Newton：
///     f(delta)  = qv - delta - qs(T(delta))
///     f'(delta) = -1 - (L/cp) * dqs/dT,   dqs/dT = L qs/(Rv T^2)
///     约束： delta ∈ [-qc, qv]（不得蒸发出超过已有云水的量）
///
/// 文献：[P1] 饱和调整思想；[D16] 第 2 章实现细节。
/// 复杂度：O(n_iter)，n_iter <= max_iter（默认 10，收敛判据 1e-10）。
SaturationAdjust saturation_adjust_point(Real& qv, Real& qc, Real& theta, Real p,
                                         bool ice_phase = false, int max_iter = 10,
                                         Real tol = Real(1.0e-10)) noexcept;

/// 整列饱和调整（逐层调用 saturation_adjust_point）
SaturationAdjust saturation_adjust_column(PhysicsColumn& col, Int k0, Int k1,
                                         bool ice_phase = false) noexcept;

// ===========================================================================
// 3. 抽象接口
// ===========================================================================

// ---------------------------------------------------------------------------
/// 微物理方案抽象接口。
///
/// 接口契约：
///   * step_column 就地把 col 推进 dt，返回地面降水率 (mm/s)；
///   * 实现必须保证 col 中所有水物质非负，并把守恒残差写入 diag；
///   * 实现必须把潜热写回 col.theta（不允许只改水物质）。
class MicrophysicsBase {
 public:
  virtual ~MicrophysicsBase() = default;

  /// 方案标识
  virtual MicrophysicsScheme scheme() const noexcept = 0;
  /// 方案名称（日志/文档用）
  virtual const char* name() const noexcept = 0;

  // -------------------------------------------------------------------------
  /// 单列微物理积分
  /// @param col  输入/输出：整列大气（theta、水物质被就地更新）
  /// @param dt   时间步长 (s)
  /// @param diag 诊断容器（降水率、守恒残差）
  /// @return     地面降水率 (mm/s)
  virtual Real step_column(PhysicsColumn& col, Real dt, PhysicsDiagnostics& diag) = 0;

  /// 本列最大粒子落速 (m/s)，供 CFL 诊断
  virtual Real max_fall_speed(const PhysicsColumn& col) const;

  /// 方案参数摘要
  virtual std::string describe() const;

 protected:
  explicit MicrophysicsBase(const PhysicsOptions& opt) : opt_(opt) {}
  const PhysicsOptions& opt_;
};

// ===========================================================================
// 4. Kessler 暖雨方案 [P1]
// ===========================================================================

// ---------------------------------------------------------------------------
/// Kessler (1969) 暖雨参数化：水汽/云水/雨水三种水物质。
///
/// 过程清单
///   1. 饱和调整（云水凝结/蒸发，Newton 迭代，含潜热）；
///   2. 云水 -> 雨水自动转换  P_auto = k1 max(qc - qc0, 0)；
///   3. 雨水收集云水          P_acc  = k2 qc qr；
///   4. 雨水沉降（上游通量法，落速 V = a (rho qr)^b）；
///   5. 雨水蒸发（Maxwell-Mason，受次饱和与可用雨水限制）。
///
/// 文献：[P1]；[D16] 第 2 章（Kessler 方案的 WRF 实现）。
class KesslerMicrophysics final : public MicrophysicsBase {
 public:
  explicit KesslerMicrophysics(const PhysicsOptions& opt) : MicrophysicsBase(opt) {}

  MicrophysicsScheme scheme() const noexcept override { return MicrophysicsScheme::Kessler; }
  const char* name() const noexcept override { return "Kessler 暖雨 [P1]"; }

  Real step_column(PhysicsColumn& col, Real dt, PhysicsDiagnostics& diag) override;
  Real max_fall_speed(const PhysicsColumn& col) const override;
  std::string describe() const override;

  /// 雨滴落速经验式（[P1]；[D16] 式 (4)）
  ///     V_t = a (rho q_r)^b,  a = 36.34, b = 0.1364 (SI)
  /// 复杂度 O(1)。
  Real rain_fall_speed(Real rho, Real qr) const noexcept;

  /// 自动转换项 (kg/kg/s)： P_auto = k1 * max(qc - qc0, 0)
  Real autoconversion(Real qc) const noexcept;

  /// 收集项 (kg/kg/s)： P_acc = k2 qc qr
  Real accretion(Real qc, Real qr) const noexcept;

  /// 雨水蒸发速率 (kg/kg/s，正 = 蒸发)
  Real rain_evaporation_rate(Real p, Real t, Real rho, Real qv, Real qs, Real qr) const noexcept;

  /// 雨水沉降（上游通量法 + 可用质量限制）
  /// @return 本步离开列底的水质量 (kg/m^2)
  Real sediment_rain(PhysicsColumn& col, Real dt) const;

  /// 雨的粒子谱参数（Marshall-Palmer 截距 N0 = 8e6 m^-4）
  static constexpr Real kRainIntercept = Real(8.0e6);
  /// 雨的 m-D 关系参数： m = a D^b, a = pi/6 * rho_w, b = 3
  static Real rain_mass_a() noexcept { return kPi / Real(6) * kRhoWater; }
  static constexpr Real kRainMassB = Real(3.0);
};

// ===========================================================================
// 5. Thompson 6 类方案 [P3]
// ===========================================================================

// ---------------------------------------------------------------------------
/// Thompson et al. (2008) 6 类（水汽/云水/雨水/云冰/雪/霰）参数化。
///
/// 过程清单（[P3] 表 1、式 (A1)-(A22) 的结构）
///   * 饱和调整（混合相态：T < 0 C 时云水与云冰之间的 Bergeron 转换）；
///   * 云水 -> 雨水自动转换（基于体积平均半径的松弛式，见 thompson_autoconversion）；
///   * 雨水收集云水（连续收集，含落速差）；
///   * Bergeron 过程：冰晶在冰面过饱和环境中消耗云水
///       P_berg = qc [1 - exp(-beta dt)],  beta = c (qv/qsi - 1)；
///   * 云水/雨水冻结（均匀冻结 + Bigg 浸泡冻结）；
///   * 云冰 -> 雪自动转换（冰晶长大到落速显著）；
///   * 雪淞附（riming）-> 霰、雪 - 霰自动转换；
///   * 雪/霰融化（Maxwell-Mason 热量收支，[P2] 式 (A20)）；
///   * 沉积/凝华（Maxwell-Mason，冰面饱和比）；
///   * 雨/雪/霰沉降（指数谱质量加权落速，[P3] 表 1）。
///
/// 数值说明
///   * 各项均为显式欧拉 + 严格的正定限制（每一步后调用 clip_negatives）；
///   * 落速 V = c D^d，D 为质量加权平均直径（Gamma 函数解析积分）；
///   * 落速含空气密度修正 (rho0/rho)^0.54（[P3] 式 (A14)）。
class ThompsonMicrophysics final : public MicrophysicsBase {
 public:
  explicit ThompsonMicrophysics(const PhysicsOptions& opt) : MicrophysicsBase(opt) {}

  MicrophysicsScheme scheme() const noexcept override { return MicrophysicsScheme::Thompson; }
  const char* name() const noexcept override { return "Thompson 6 类 [P3]"; }

  Real step_column(PhysicsColumn& col, Real dt, PhysicsDiagnostics& diag) override;
  Real max_fall_speed(const PhysicsColumn& col) const override;
  std::string describe() const override;

  // ---- 各项参数化（公开以便单元测试与文档对照） ----
  /// 云水 -> 雨水：基于体积平均半径的松弛式（[P3] 的 Berry-Reinhardt 简化）
  ///     r_vol = (3 rho qc / (4 pi rho_w Nc))^(1/3)
  ///     P_ra  = qc/tau * max(0, 1 - r_crit/r_vol)
  Real thompson_autoconversion(Real rho, Real qc, Real t) const noexcept;
  /// Bergeron 过程（[P3] 式 (A9)）： beta = c max(qv/qsi - 1, 0)
  Real bergeron_rate(Real qv, Real qsi) const noexcept;
  /// 均匀冻结 + 云水冻结速率 (kg/kg/s)
  Real cloud_water_freezing_rate(Real t, Real qc) const noexcept;
  /// 雨水冻结（Bigg，[P3] 式 (A17)）
  ///   P = (pi^2 rho_w B'/(12 rho)) [exp(A'(T0-T)) - 1] Integral D^6 N(D) dD
  Real rain_freezing_rate(Real t, Real rho, Real qr) const noexcept;
  /// 雪融化速率 (kg/kg/s)，T > T0 时为正（[P2] 式 (A20)）
  Real snow_melting_rate(Real t, Real p, Real rho, Real qv, Real qsnow) const noexcept;
  /// 霰融化速率 (kg/kg/s)
  Real graupel_melting_rate(Real t, Real p, Real rho, Real qv, Real qgraupel) const noexcept;
  /// 沉积/凝华速率 (kg/kg/s)，正 = 增长（[P2] 式 (A4)）
  Real deposition_rate(Real t, Real p, Real rho, Real qv, Real qsat_ice, Real number,
                       Real diameter, bool snow = true) const noexcept;
  /// 雪的粒子谱斜率 lambda_s
  Real snow_lambda(Real rho, Real qs) const noexcept;
  /// 霰的粒子谱斜率 lambda_g
  Real graupel_lambda(Real rho, Real qg) const noexcept;
  /// 雨的粒子谱斜率 lambda_r
  Real rain_lambda(Real rho, Real qr) const noexcept;
  /// 雪落速（含密度修正）
  Real snow_fall_speed(Real rho, Real qs) const noexcept;
  /// 霰落速
  Real graupel_fall_speed(Real rho, Real qg) const noexcept;
  /// 雨落速
  Real rain_fall_speed(Real rho, Real qr) const noexcept;

  // ---- 粒子谱常数（[P3] 表 1） ----
  //   质量-直径关系   m = a D^b   (kg, m)
  //   落速关系        V = c D^d   (m/s)
  //   指数分布截距    N0          (m^-4)
  //   球体质量系数    a = pi/6 * rho_bulk
  static constexpr Real kSnowMassA = Real(0.069);           ///< 雪 m = 0.069 D^2
  static constexpr Real kSnowMassB = Real(2.0);
  static constexpr Real kSnowFallC = Real(40.0);            ///< 雪 V = 40 D^0.55
  static constexpr Real kSnowFallD = Real(0.55);
  static constexpr Real kGraupelMassA = Real(209.43951023931953);  ///< pi/6*400 kg/m^3
  static constexpr Real kGraupelMassB = Real(3.0);
  static constexpr Real kGraupelFallC = Real(442.0);        ///< 霰 V = 442 D^0.89
  static constexpr Real kGraupelFallD = Real(0.89);
  static constexpr Real kIceMassA = Real(465.9029102824860);  ///< pi/6*890 kg/m^3
  static constexpr Real kIceMassB = Real(3.0);
  static constexpr Real kSnowIntercept = Real(2.0e7);       ///< N0s (m^-4)
  static constexpr Real kGraupelIntercept = Real(4.0e6);    ///< N0g (m^-4)
  static constexpr Real kIceIntercept = Real(1.0e7);        ///< N0i (m^-4)
  /// 雨的落速与谱参数（Marshall-Palmer）
  static constexpr Real kRainFallA = Real(4854.0);          ///< V = 4854 D exp(-195 D)
  static constexpr Real kRainFallB = Real(195.0);
};

// ===========================================================================
// 6. 工厂
// ===========================================================================

// ---------------------------------------------------------------------------
/// 微物理方案工厂。
///   None            -> nullptr（关闭，driver 直接跳过）
///   Kessler         -> KesslerMicrophysics
///   Thompson        -> ThompsonMicrophysics
///   Morrison / Wsm6 -> 抛 NotImplemented（接口保留，见 08_physics.md 第 9 节）
std::unique_ptr<MicrophysicsBase> make_microphysics(MicrophysicsScheme scheme,
                                                    const PhysicsOptions& opt);

}  // namespace vibe::physics

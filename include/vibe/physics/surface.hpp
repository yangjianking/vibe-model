#pragma once
/// @file surface.hpp
/// @brief 地面层相似理论（Monin-Obukhov）、五层陆面（Noilhan-Planton 简化）与海面。
///
/// 物理框架
/// --------
///   1. 地面层通量-廓线关系（Businger-Dyer，[P10]）
///         dU/dz = (u*/kappa z) phi_m(zeta)
///         dtheta/dz = (theta*/kappa z) phi_h(zeta)
///         phi_m = (1-16 zeta)^(-1/4) (zeta<0),  1+5 zeta (zeta>=0)
///         phi_h = (1-16 zeta)^(-1/2) (zeta<0),  1+5 zeta (zeta>=0)
///      积分形式（Paulson 1970）给出对数 + 稳定度修正的廓线，
///      通量由 u*, theta*, q* 给出： H = -rho cp u* theta*, LE = -rho Lv u* q*。
///   2. 陆面：五层土壤温度/湿度 + 表皮温度能量平衡
///         R_n = H + LE + G,   R_n = (1-a) S_down + eps (L_down - sigma T_g^4)
///      土壤内用隐式（全隐式三对角）扩散，表层的通量边界条件由能量平衡给出。
///   3. 海面：固定海温 SST，粗糙度用 Charnock 关系
///         z0 = alpha_c u*^2/g + 0.11 nu/u*
///
/// 文献：[P9] Monin & Obukhov (1954)；[P10] Businger et al. (1971)；
///       [P14] Noilhan & Planton (1989)；[P15] Chen & Dudhia (2001)。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/types.hpp"
#include "vibe/physics/physics_types.hpp"

namespace vibe::physics {

// ---------------------------------------------------------------------------
/// 地面层通量与尺度参数
struct SurfaceFluxes {
  Real ustar = Real(0);         ///< 摩擦速度 (m/s)
  Real tstar = Real(0);         ///< theta* (K)
  Real qstar = Real(0);         ///< q* (kg/kg)
  Real obukhov = kHuge;         ///< Obukhov 长度 L (m)
  Real zeta = Real(0);          ///< z1/L
  Real sensible = Real(0);      ///< 感热通量 (W/m^2，向上为正)
  Real latent = Real(0);        ///< 潜热通量 (W/m^2，向上为正)
  Real momentum = Real(0);      ///< |tau| (N/m^2)
  Real z0 = Real(0);            ///< 动量粗糙度 (m)
  Real z0h = Real(0);           ///< 热量粗糙度 (m)
  Real z0q = Real(0);           ///< 水汽粗糙度 (m)
  Real bulk_richardson = Real(0);
  Real cd = Real(0);            ///< 动量交换系数
  Real ch = Real(0);            ///< 热量交换系数
  Real cq = Real(0);            ///< 水汽交换系数
  Real t2 = Real(0);            ///< 2 m 温度 (K)
  Real q2 = Real(0);            ///< 2 m 湿度 (kg/kg)
  Real u10 = Real(0), v10 = Real(0);
  int iterations = 0;
  Real residual = Real(0);
  bool converged = false;
};

// ===========================================================================
/// Monin-Obukhov 相似理论地面层
class MoninObukhov {
 public:
  explicit MoninObukhov(const PhysicsOptions& opt) : opt_(opt) {}

  // ---- 稳定性函数与积分形式 ----
  /// phi_m(zeta)：不稳定 (1-16 zeta)^(-1/4)，稳定 1+5 zeta
  static Real phi_m(Real zeta) noexcept;
  /// phi_h(zeta)：不稳定 (1-16 zeta)^(-1/2)，稳定 1+5 zeta
  static Real phi_h(Real zeta) noexcept;
  /// psi_m(zeta) = Integral_{zeta0}^{zeta} [1-phi_m]/z' dz'（Paulson 1970）
  ///   zeta<0: 2 ln((1+x)/2) + ln((1+x^2)/2) - 2 atan(x) + pi/2,  x=(1-16 zeta)^(1/4)
  ///   zeta>=0: -5 zeta
  static Real psi_m(Real zeta) noexcept;
  /// psi_h(zeta)
  ///   zeta<0: 2 ln((1+x^2)/2),  x=(1-16 zeta)^(1/4)
  ///   zeta>=0: -5 zeta
  static Real psi_h(Real zeta) noexcept;
  /// 中性对数风廓线 u(z) = (u*/kappa) ln(z/z0)
  static Real log_wind(Real z, Real z0, Real ustar) noexcept;
  /// 中性交换系数 Cd = kappa^2 / ln(z/z0)^2
  static Real neutral_exchange(Real z, Real z0) noexcept;
  /// 总体 Richardson 数
  ///   Ri_b = g z (theta_v1 - theta_vs) / (theta_v1 |V|^2)
  static Real bulk_richardson(Real z, Real u, Real v, Real thv1, Real thvs) noexcept;

  // -------------------------------------------------------------------------
  /// 迭代求解 u*, theta*, q*、L 与地表通量。
  ///
  /// 离散化/算法：固定点迭代
  ///     L^(n+1) = u*^2 / (kappa (g/theta0) theta_v*)
  ///     u*   = kappa |V1| / [ln(z1/z0) - psi_m(z1/L) + psi_m(z0/L)]
  ///     theta* = kappa (theta1-theta_s) / [ln(z1/z0h) - psi_h(z1/L) + psi_h(z0h/L)]
  ///     q*     = kappa (q1-q_s)      / [ln(z1/z0q) - psi_h(z1/L) + psi_h(z0q/L)]
  /// 初值取中性解；zeta 限幅 [-50, 20]；收敛判据 |dL|/|L| <= mo_tolerance。
  /// 复杂度：O(n_iter)，n_iter <= mo_max_iterations（默认 40）。
  SurfaceFluxes solve(const PhysicsColumn& col, SurfaceState& sfc, Real dt) const;

  /// 上一次求解的迭代信息（单元测试用）
  int last_iterations() const noexcept { return last_iterations_; }
  Real last_residual() const noexcept { return last_residual_; }
  void set_tolerance(Real tol, int max_iter) noexcept {
    opt_.mo_tolerance = tol;
    opt_.mo_max_iterations = max_iter;
  }
  /// 迭代更新海面 Charnock 粗糙度
  ///   z0 = alpha_c u*^2/g + 0.11 nu/u*
  static Real charnock_roughness(Real ustar, Real alpha_c) noexcept;
  std::string describe() const;

 private:
  PhysicsOptions opt_;
  mutable int last_iterations_ = 0;
  mutable Real last_residual_ = 0;
};

// ===========================================================================
/// 五层陆面（Noilhan-Planton 简化 [P14][P15]）
class LandSurface {
 public:
  explicit LandSurface(const PhysicsOptions& opt) : opt_(opt) {}

  /// 初始化土壤廓线（首次调用时把深层值填入各层）
  void initialize(SurfaceState& sfc) const;

  /// 表皮能量平衡 Newton 迭代 + 五层土壤温度/湿度推进 + 2 m/10 m 诊断
  ///
  /// 能量平衡： R_n(T_g) = H(T_g) + LE(T_g) + G(T_g)
  ///     R_n = (1-a) S_down + eps (L_down - sigma T_g^4)
  ///     H   = rho cp Ch |V| (T_g - theta_1 pi_1)    (线性化)
  ///     LE  = rho Lv Cq |V| (q_sat(T_g) beta - q_1)
  ///     G   = lambda_s (T_g - T_soil,1)/(0.5 dz_1)
  /// Newton： dR_n/dT_g = -4 eps sigma T_g^3，
  ///          dH/dT_g = rho cp Ch |V|，
  ///          dLE/dT_g = rho Lv Cq |V| beta dq_sat/dT，
  ///          dG/dT_g = lambda_s/(0.5 dz_1)
  ///
  /// 土壤温度/湿度：全隐式（Thomas 三对角）
  ///     C_s dT/dt = d/dz (lambda dT/dz),  dw/dt = d/dz (D_w dw/dz)
  /// 表层通量边界： -lambda dT/dz|_sfc = R_n - H - LE（= G）
  /// 底层边界： 固定深层温度/湿度（Dirichlet）
  /// 复杂度：O(n_soil) 每列。
  /// 注意：flx 为**非 const**，因为表皮温度求解后需用最终 T_g 更新 H/LE/G，
  /// 供随后的边界层方案作为下边界通量使用；大气倾向由 PBL 方案产生。
  void step_column(PhysicsColumn& col, SurfaceState& sfc, SurfaceFluxes& flx, Real dt,
                   ColumnTendency& tend, PhysicsDiagnostics& diag) const;

  /// 由状态量重算能量平衡残差 (W/m^2)： R_n - H - LE - G
  static Real energy_balance_residual(const SurfaceState& sfc);

  /// 土壤湿度对蒸发的限制 beta（Noilhan-Planton）
  ///     beta = min(1, w_1/w_fc)；w < w_wilt 时按指数衰减
  static Real moisture_availability(Real w1, Real field_capacity, Real wilting_point) noexcept;

  std::string describe() const;

 private:
  PhysicsOptions opt_;
};

// ===========================================================================
/// 海面：固定 SST + Charnock 粗糙度
class SeaSurface {
 public:
  explicit SeaSurface(const PhysicsOptions& opt) : opt_(opt) {}

  /// 设置海表状态（SST 固定、饱和湿度、Charnock 粗糙度）
  void update(SurfaceState& sfc, const SurfaceFluxes& flx) const;
  /// 海表潜热通量的上限（能量限制，可选）
  Real max_latent_flux(const SurfaceState& sfc) const noexcept;
  std::string describe() const;

 private:
  PhysicsOptions opt_;
};

/// 陆面/海面工厂
std::unique_ptr<MoninObukhov> make_surface_layer(const PhysicsOptions& opt);
std::unique_ptr<LandSurface> make_land_surface(const PhysicsOptions& opt);
std::unique_ptr<SeaSurface> make_sea_surface(const PhysicsOptions& opt);

}  // namespace vibe::physics

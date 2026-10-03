#pragma once
/// @file pbl.hpp
/// @brief 行星边界层参数化接口：YSU（非局地 K 廓线）与 MYJ（Mellor-Yamada 2.5 阶 TKE）。
///
/// 两个方案都写成 **垂直扩散方程** 的隐式求解：
///     dX/dt = d/dz ( K_X dX/dz ) + S
/// 差别在于 K 的构造（YSU：解析 K 廓线 + 逆梯度项；MYJ：由 TKE 预报方程闭合），
/// 以及表面通量如何进入下边界条件（两者都取地面层的 u*, theta*, q*）。
///
/// 文献：[P7] Hong et al. (2006) YSU；[P8] Mellor & Yamada (1982)；
///       [P18] Janjic (1994) MYJ 非奇异实现。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/types.hpp"
#include "vibe/physics/physics_types.hpp"
#include "vibe/physics/surface.hpp"

namespace vibe::physics {

// ===========================================================================
// 垂直扩散求解器（YSU 与 MYJ 共用）
// ===========================================================================

// ---------------------------------------------------------------------------
/// 一维垂直扩散的全隐式（向后欧拉）求解
///     dX/dt = d/dz ( K dX/dz ) - decay * X + source
///
/// 有限体积离散（层中心 k，界面 k+1/2）：
///     F_{k+1/2} = K_{k+1/2} (X_{k+1}-X_k)/dz_{k+1/2},  dz_{k+1/2}=0.5(dz_k+dz_{k+1})
///     [1 + dt (a_k + b_k + decay_k)] X_k^{n+1} - dt a_k X_{k-1}^{n+1}
///         - dt b_k X_{k+1}^{n+1} = X_k^n + dt source_k
///     a_k = K_{k-1/2}/(dz_k dz_{k-1/2}),  b_k = K_{k+1/2}/(dz_k dz_{k+1/2})
/// 边界：下边界为给定运动学通量 surface_flux_lower（正向上），
///       上边界为零通量；三对角用 Thomas 算法求解。
///
/// 稳定性：全隐式对任意 dt、任意 K >= 0 无条件稳定（放大因子 <= 1）。
/// 文献：[B2] 第 3 章（隐式扩散）；[P7][P8]。
/// 复杂度：O(nz)。
void solve_implicit_diffusion(const std::vector<Real>& k_diff, const std::vector<Real>& dz,
                              const std::vector<Real>& x, const std::vector<Real>& source,
                              const std::vector<Real>& decay, Real dt,
                              Real surface_flux_lower, std::vector<Real>& x_new);

/// Thomas（三对角）算法：a x_{i-1} + b x_i + c x_{i+1} = d
/// 复杂度 O(n)，回代前不破坏输入（内部拷贝）。
/// 文献：[B2] 附录；[B11] 第 3 章。
void thomas_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                  const std::vector<Real>& c, const std::vector<Real>& d,
                  std::vector<Real>& x);

// ===========================================================================
/// 边界层方案抽象接口
class PblBase {
 public:
  virtual ~PblBase() = default;
  virtual PblScheme scheme() const noexcept = 0;
  virtual const char* name() const noexcept = 0;

  // -------------------------------------------------------------------------
  /// 单列边界层输送：把 theta/qv/qc/qi/u/v/tke 的倾向累加到 tend（同时就地更新 col.tke）
  /// @return PBL 高度 (m)
  virtual Real step_column(PhysicsColumn& col, SurfaceState& sfc, const SurfaceFluxes& flx,
                           Real dt, ColumnTendency& tend) = 0;

  /// 只诊断 PBL 高度（不产生倾向）
  virtual Real diagnose_height(const PhysicsColumn& col, const SurfaceState& sfc,
                               const SurfaceFluxes& flx) const = 0;

  virtual std::string describe() const;
  /// 本方案本次调用使用的最大垂直交换系数 (m^2/s)，用于 CFL 诊断
  virtual Real max_diffusivity() const noexcept { return max_diffusivity_; }

 protected:
  explicit PblBase(const PhysicsOptions& opt) : opt_(opt) {}
  const PhysicsOptions& opt_;
  mutable Real max_diffusivity_ = Real(0);
};

// ===========================================================================
/// MYJ：Mellor-Yamada 2.5 阶 TKE 闭合 [P8][P18]
/// ===========================================================================
class MyjPbl final : public PblBase {
 public:
  explicit MyjPbl(const PhysicsOptions& opt);

  PblScheme scheme() const noexcept override { return PblScheme::Myj; }
  const char* name() const noexcept override { return "MYJ (MY2.5 TKE 闭合) [P8][P18]"; }

  Real step_column(PhysicsColumn& col, SurfaceState& sfc, const SurfaceFluxes& flx, Real dt,
                   ColumnTendency& tend) override;
  Real diagnose_height(const PhysicsColumn& col, const SurfaceState& sfc,
                       const SurfaceFluxes& flx) const override;

  // ---- MY2.5 闭合要素（公开以便测试与文档对照） ----
  /// 稳定性函数（[P8] 式 (35)；[P18] 的非奇异化形式）
  ///   GM = (du/dz)^2+(dv/dz)^2,  GH = -(g/theta0) dtheta_v/dz（稳定时为正）
  ///   ELOQ2 = (l/q)^2，q^2 = 2 e
  ///   D  = (ADNM GM + ADNH GH) GH (l/q)^4 + (BDNM GM + BDNH GH)(l/q)^2 + 1
  ///   S_M = (BSMH GH (l/q)^2 + CESM)/D
  ///   S_H = ((BSHM GM + BSHH GH)(l/q)^2 + CESH)/D
  void stability_functions(Real gm, Real gh, Real el_o_q2, Real& sm, Real& sh) const noexcept;

  /// 平衡混合长（由 TKE 平衡方程解 (l/q)^2 的二次方程）
  ///   稳定分支： GM/GH <= REQU 时湍流被禁止（返回极小值），
  ///              否则解 AUBR x^2 + BUBR x + CUBR = 0
  ///   不稳定分支：解 ADEN x^2 + BDEN x + 1 = 0
  Real equilibrium_length(Real gm, Real gh, Real q2) const noexcept;

  /// Blackadar 渐近混合长
  ///   EL0 = alph * Integral q z dz / Integral q dz（PBL 内质量加权），限幅 [EL0MIN, EL0MAX]
  Real asymptotic_length(const std::vector<Real>& q, const std::vector<Real>& zf,
                         Int pbl_top) const noexcept;

  /// 混合长廓线： PBL 内 EL = min(kappa (z-z_sfc)/(1+kappa(z-z_sfc)/EL0), ELM)，
  /// PBL 上 EL = min(0.23 dz, ELM)，随后做一次 1-2-1 平滑
  void mixing_length(const PhysicsColumn& col, Int pbl_top, std::vector<Real>& el) const;

  /// 诊断 PBL 顶所在层（自顶向下第一个 TKE <= e_min 的层）
  Int pbl_top_index(const PhysicsColumn& col) const noexcept;

  std::string describe() const override;

 private:
  /// 由 MY 常数派生的系数（构造时计算一次）
  struct Derived {
    Real adnh, adnm, bdnh, bdnm, bshh, bshm, bsmh, cesh, cesm;
    Real aeqh, aeqm, requ;
    Real aubh, aubm, bubh, bubm, cubr, rcubr;
    Real btg;
  } d_;
  void compute_derived();
};

// ===========================================================================
/// YSU：非局地 K 廓线 + 夹卷 + 逆梯度项 [P7]
/// ===========================================================================
class YsuPbl final : public PblBase {
 public:
  explicit YsuPbl(const PhysicsOptions& opt) : PblBase(opt) {}

  PblScheme scheme() const noexcept override { return PblScheme::Ysu; }
  const char* name() const noexcept override { return "YSU (非局地 K 廓线) [P7]"; }

  Real step_column(PhysicsColumn& col, SurfaceState& sfc, const SurfaceFluxes& flx, Real dt,
                   ColumnTendency& tend) override;
  Real diagnose_height(const PhysicsColumn& col, const SurfaceState& sfc,
                       const SurfaceFluxes& flx) const override;

  // ---- YSU 要素 ----
  /// 由总体 Richardson 数确定 PBL 高度：自下而上找 Ri_b >= Ri_c 的首层
  Real pbl_height_from_richardson(const PhysicsColumn& col, const SurfaceState& sfc,
                                  const SurfaceFluxes& flx, Int& top_index) const noexcept;
  /// 对流速度尺度： w*^3 = (g/theta0) h (w'theta')_0
  Real convective_velocity(Real h, Real surface_heat_flux_kinematic, Real theta0) const noexcept;
  /// 速度尺度： w_s = (u*^3 + phi_m kappa w*^3 z/h)^(1/3)
  Real velocity_scale(Real ustar, Real wstar, Real z, Real h) const noexcept;
  /// 动量扩散系数： K_m = kappa w_s z (1-z/h)^2  (z<=h)
  Real diffusivity_momentum(Real ws, Real z, Real h) const noexcept;
  /// 逆梯度（非局地）项： gamma = C (w'theta')_0/(w_s h)
  Real counter_gradient(Real surface_heat_flux_kinematic, Real ws, Real h) const noexcept;
  /// Prandtl 数： Pr = 1 + coef * z/h
  Real prandtl(Real z, Real h) const noexcept;
  std::string describe() const override;
};

/// 边界层方案工厂：None -> nullptr
std::unique_ptr<PblBase> make_pbl(PblScheme scheme, const PhysicsOptions& opt);

}  // namespace vibe::physics

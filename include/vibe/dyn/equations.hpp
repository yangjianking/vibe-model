#pragma once
/// @file equations.hpp
/// @brief 控制方程装配器：把各物理/数值过程组合成时间趋势。
///
/// 控制方程组（地形追随坐标，[D1][D2][D5][D13]）
/// ------------------------------------------------
/// 记 Exner 扰动 pi'，参考态 pi0, rho0, theta0，度量因子 G^{1/2} = dz/dzeta。
///
/// 动量方程（通量形式 + 曲线坐标度量项）：
///
///     du/dt = -u du/dx - v du/dy - w~ du/dzeta
///             - cp * theta * d pi'/dx + f v + D_u
///     dv/dt = -u dv/dx - v dv/dy - w~ dv/dzeta
///             - cp * theta * d pi'/dy - f u + D_v
///     dw/dt = -u dw/dx - v dw/dy - w~ dw/dzeta
///             - cp * theta * d pi'/dzeta + g * (theta'/theta0 - rho' / rho0 ... )
///             + D_w
///
/// 质量守恒（守恒形式）：
///
///     d rho'/dt = -div(rho u)  -  rho0 * div(u)
///
/// 位温方程（含非绝热加热）：
///
///     d theta/dt = -u.grad(theta) + (theta0 / (rho0 cp T0)) * Q_heat
///
/// 水物质：
///
///     d q_x/dt = -u.grad(q_x) + S_x
///
/// 状态方程（诊断 Exner）：
///
///     pi = ( p / p00 )^{Rd/cp},
///     p = rho Rd T (1 + qv/epsilon) / (1 + qv)
///
/// 分裂
/// ----
///   `tendencies()`        : 非线性（慢）项，含平流、科氏、物理、阻尼
///   `acoustic_tendencies()`: 线性声波-重力波项，供子步或半隐式使用
///   `diagnose()`          : 由 rho, theta, 水物质更新 pi' 与诊断量
///
/// 文献：[D1][D2][D5][D6][D13][D16]。

#include <memory>
#include <string>

#include "vibe/common/types.hpp"
#include "vibe/dyn/advection.hpp"
#include "vibe/dyn/coriolis.hpp"
#include "vibe/dyn/damping.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::config { struct ModelConfig; }

namespace vibe::dyn {

/// 方程装配器
class Equations {
 public:
  Equations(const grid::Grid& g, const ReferenceState& ref,
            const config::ModelConfig& cfg);

  /// 慢过程趋势 F_slow
  void tendencies(const State& s, Tendency& d) const;

  /// 线性声波-重力波趋势 F_acoustic（不含平流/物理）
  void acoustic_tendencies(const State& s, Tendency& d) const;

  /// 诊断：由状态更新 pi'，保证状态方程相容
  void diagnose(State& s) const;

  /// 计算暴露的线性化系数（供半隐式算子使用）：
  ///   alpha = 1/(rho0 c_s^2) 在体心；beta = 1/rho0 在体心
  void linearized_coefficients(const ReferenceState& ref,
                               grid::Field<Real>& alpha,
                               grid::Field<Real>& beta) const;

  /// 声波速度上限（用于 CFL 与子步数选择）
  Real max_sound_speed(const State& s) const;

  /// 诊断性倾向：仅声波项，用于线性稳定性测试
  const Advection& advection() const noexcept { return *adv_; }
  const Coriolis& coriolis() const noexcept { return *cor_; }
  const Damping& damping() const noexcept { return *damp_; }

 private:
  const grid::Grid* grid_;
  const ReferenceState* ref_;
  std::unique_ptr<Advection> adv_;
  std::unique_ptr<Coriolis> cor_;
  std::unique_ptr<Damping> damp_;
  bool moist_ = true;
  Real acoustic_damping_ = Real(0.0);
};

}  // namespace vibe::dyn

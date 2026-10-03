#pragma once
/// @file damping.hpp
/// @brief 数值阻尼、海绵层与散度阻尼。
///
/// 组成
/// ----
///   1. **上层海绵层（Rayleigh damping）**：在模式顶附近对 u,v,w,theta',pi' 施加
///      牛顿松弛，抑制上边界反射（[D1] 第 4 节；[D16] 第 3 章）。系数
///          alpha(z) = alpha_max * sin^2( pi/2 * (z - z_sponge)/(z_top - z_sponge) )
///   2. **水平散度阻尼**：对质量场施加 Del^2 或 Del^4 滤波，抑制声波与
///      短波噪声（[D6]）。
///   3. **垂直隐式阻尼**：在大时间步末端施加垂直扩散，用于吸收重力波。
///
/// 时间离散：海绵层用后向欧拉（无条件稳定）。
///
/// 文献：[D1][D6][D16]。

#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::dyn {

class Damping {
 public:
  Damping(const grid::Grid& g, Real z_sponge_start, Real z_top,
          Real alpha_max = Real(0.2), int divergence_order = 2,
          Real divergence_coeff = Real(0.0));

  /// 显式阻尼趋势（散度滤波），可并入 F_slow
  void tendencies(const State& s, Tendency& d) const;

  /// 隐式海绵层（后向欧拉）：s <- (s + dt * alpha * s_target) / (1 + dt * alpha)
  void apply_sponge(State& s, const ReferenceState& ref, Real dt) const;

  /// 顶层海绵层权重剖面（层中心，长度 nz）
  const std::vector<Real>& sponge_profile() const noexcept { return sponge_; }

 private:
  const grid::Grid* grid_;
  Real z_sponge_start_;
  Real z_top_;
  Real alpha_max_;
  int divergence_order_;
  Real divergence_coeff_;
  std::vector<Real> sponge_;
};

}  // namespace vibe::dyn

#pragma once
/// @file coriolis.hpp
/// @brief 科氏力项：f-plane 与 beta-plane。
///
///     du/dt = + f v
///     dv/dt = - f u
///     f = 2 Omega sin(phi),  beta = 2 Omega cos(phi) / a
///
/// C-grid 上的离散采用 v 点 -> u 点、u 点 -> v 点的四角平均，保证离散
/// 动能守恒与无虚假能量增长（[D3] 第 4 节；[B5] Vallis 第 2 章）。
///
/// 对于深对流/全球尺度试验可启用 beta 项；教学配置默认 f-plane。
///
/// 文献：[D3][B4][B5]。

#include <string>

#include "vibe/common/constants.hpp"
#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::dyn {

enum class CoriolisMode { None, FPlane, BetaPlane };

class Coriolis {
 public:
  Coriolis(const grid::Grid& g, CoriolisMode mode, Real latitude_deg,
           Real u_ref = Real(0));

  /// 写入 u、v 趋势中的科氏贡献
  void apply(const State& s, Tendency& d) const;

  Real f_at(Int j) const;
  Real beta() const noexcept { return beta_; }
  CoriolisMode mode() const noexcept { return mode_; }

 private:
  const grid::Grid* grid_;
  CoriolisMode mode_;
  Real f0_ = Real(0);
  Real beta_ = Real(0);
  Real lat0_ = Real(0);
};

}  // namespace vibe::dyn

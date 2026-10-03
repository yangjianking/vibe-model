#pragma once
/// @file advection.hpp
/// @brief 动量与标量的平流离散。
///
/// 离散方案
/// --------
///   1. 二阶中心（order=2）：`u dq/dx` 的 C-grid 交错形式，能量守恒错位（[D3][D4]）；
///   2. 四阶中心（order=4）：非交错 5 点模板；
///   3. 六阶中心（order=6）：7 点模板，适合理想试验；
///   4. 五阶 WENO（weno=true）：通量形式，抑制陡峭梯度处的振荡（[D7][D8]）。
///
/// 守恒形式（有限体积，[D6][D13]）
/// --------------------------------
///     d(rho q)/dt = - div(rho q u)
/// 离散采用面通量差分，配合错位网格可达二阶精度且严格守恒。
///
/// CFL 检查
/// --------
///     CFL = dt * max(|u|/dx + |v|/dy + |w|/dz)
/// 该函数由 `cfl_number` 提供，供时间推进器选择子步数。
///
/// 文献：[D3][D4][D6][D7][D8][D13][T17][T18]。

#include <string>

#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::dyn {

enum class AdvectionScheme { Central2, Central4, Central6, WENO5, Upwind3 };

inline AdvectionScheme advection_from_string(const std::string& s);

class Advection {
 public:
  Advection(const grid::Grid& g, AdvectionScheme scheme, bool flux_form = true);

  /// 标量（含密度加权的守恒形式）：dq/dt = -u.grad(q)
  void scalar(const grid::Field<Real>& q,
              const grid::Field<Real>& u, const grid::Field<Real>& v,
              const grid::Field<Real>& w,
              const grid::Field<Real>& rho,
              grid::Field<Real>& dqdt) const;

  /// 动量平流（含曲线坐标度量项），写入 u/v/w 趋势
  void momentum(const State& s, Tendency& d) const;

  /// 标量平流（不含密度加权，用于诊断量输运）
  void scalar_nonconservative(const grid::Field<Real>& q,
                              const grid::Field<Real>& u,
                              const grid::Field<Real>& v,
                              const grid::Field<Real>& w,
                              grid::Field<Real>& dqdt) const;

  /// 当前格式的 CFL 数
  Real cfl_number(const State& s, Real dt) const;

 private:
  /// WENO5 的重建（[D8] 式 (2.6)）
  static void weno5_reconstruct(Real qm2, Real qm1, Real q0, Real q1, Real q2,
                                Real& q_left, Real& q_right);

  const grid::Grid* grid_;
  AdvectionScheme scheme_;
  bool flux_form_;
  int order_;
};

}  // namespace vibe::dyn

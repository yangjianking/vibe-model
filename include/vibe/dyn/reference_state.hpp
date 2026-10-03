#pragma once
/// @file reference_state.hpp
/// @brief 静力平衡参考态（Klemp & Wilhelmson 1978 分离形式）。
///
/// 定义
/// ----
/// 参考态满足静力平衡与状态方程
///
///     d pi0 / dz = -g / (cp * theta0)
///     pi0 = (p0 / p00)^{Rd/cp}
///     rho0 = p00 * pi0^{cv/Rd} / (Rd * theta0)
///
/// 由 theta0(z) 剖面从地面气压 p_surf 向上积分得到 pi0。
/// 离散采用梯形/辛普森混合格式，保证与动力学的垂直差分相容
/// （同一套层界面 z 与层中心 z）。
///
/// 参考态是**时间不变**的，因此在半隐式与 4D-Var 的外层循环中可以
/// 预计算并缓存全部线性系数（[T2][T6][T7]）。
///
/// 文献：[D1] Klemp & Wilhelmson (1978)；[D14] Harris & Durran (2010)。

#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::dyn {

class ReferenceState {
 public:
  ReferenceState() = default;

  /// 等温大气（教学/理想试验）
  static ReferenceState isothermal(const grid::Grid& g, Real theta0,
                                   Real p_surf = kP0, Real qv0 = Real(0));

  /// 由探空剖面构造；z 与 theta 长度相同时按线性插值填充
  static ReferenceState from_profile(const grid::Grid& g,
                                     const std::vector<Real>& z,
                                     const std::vector<Real>& theta,
                                     Real p_surf = kP0, Real qv0 = Real(0));

  /// 标准大气（US Standard Atmosphere 1976 的简化 2 层版本）
  static ReferenceState standard_atmosphere(const grid::Grid& g,
                                            Real p_surf = kP0);

  const grid::Field<Real>& pi0() const noexcept { return pi0_; }
  const grid::Field<Real>& rho0() const noexcept { return rho0_; }
  const grid::Field<Real>& theta0() const noexcept { return theta0_; }
  const grid::Field<Real>& p0() const noexcept { return p0_; }
  const grid::Field<Real>& qv0() const noexcept { return qv0_; }

  /// 层中心的 Brunt-Vaisala 频率平方 N^2 = (g/theta0) dtheta0/dz
  Real brunt_vaisala2(Int i, Int j, Int k) const;

  /// 声波速度平方 c_s^2 = gamma * Rd * T0
  Real sound_speed2(Int i, Int j, Int k) const;

  /// 检验参考态是否静力平衡（用于测试）
  Real hydrostatic_residual() const;

  /// 是否已初始化
  bool valid() const noexcept { return !pi0_.empty(); }

  std::string describe() const;

  const grid::Grid& grid() const { VIBE_CHECK(grid_ != nullptr); return *grid_; }

 private:
  const grid::Grid* grid_ = nullptr;
  grid::Field<Real> pi0_, rho0_, theta0_, p0_, qv0_;
};

}  // namespace vibe::dyn

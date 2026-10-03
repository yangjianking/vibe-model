#pragma once
/// @file variable_resolution.hpp
/// @brief 平滑变分辨率网格：由目标分辨率函数构造单调映射。
///
/// 数学形式
/// --------
/// 给定目标密度 rho(x) >= 0（表示"每单位计算坐标需要多少物理长度"），
/// 定义单调映射
///
///     xi = F(x) = (1/L) * integral_0^x ds / h(s),     x = F^{-1}(xi)
///
/// 其中 h(s) 是目标分辨率场，介于 h_min 与 h_max 之间。等距 xi 网格
/// 映射回 x 后自动在 h 小的区域加密。为保证解与网格光滑，h 必须满足
///
///     |dh/dx| <= C * h / L      （缓变条件）
///
/// 实践中由 `smooth_refinement` 迭代平滑 + 限幅实现。
///
/// 度量项
/// ------
///     dx/dxi = h(x)           （x 方向雅可比）
///     G^{1/2} = (dx/dxi)(dy/deta) / (dx_ref * dy_ref)
///
/// 动力学代码通过 Geometry::dx_cell / dy_cell 读取这些量，因此与均匀网格
/// 共用同一套离散。CFL 由最小单元宽度决定，这是变分辨率网格的主要代价。
///
/// 文献：[N5] MPAS（Skamarock et al. 2012）；[N6] Ringler et al. (2013)；
///       [N7] Tomita & Satoh (2004)；[T19] ICON（Zängl et al. 2015）。

#include <functional>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::grid {

/// 目标分辨率函数
struct RefinementFunction {
  Real h_min = Real(1000);
  Real h_max = Real(15000);
  Real Lx = Real(0), Ly = Real(0);       ///< 物理域尺度（米）
  std::function<Real(Real x, Real y)> h; ///< 返回目标分辨率（米）

  /// 圆形加密区（中心 c，半径 R，内部 h_min，外部 h_max，光滑过渡）
  static RefinementFunction circular(Real h_min, Real h_max, Real Lx, Real Ly,
                                     Real cx, Real cy, Real radius,
                                     Real transition_width);

  /// 通道型加密（沿 x 方向成带）
  static RefinementFunction channel(Real h_min, Real h_max, Real Lx, Real Ly,
                                    Real y0, Real width, Real transition_width);
};

/// 一维单调映射 x <-> xi
class VarResMap1D {
 public:
  VarResMap1D() = default;
  VarResMap1D(Real L, Int n, Real h_min, Real h_max,
              const std::function<Real(Real)>& h);

  /// 计算坐标 xi in [0,1] -> 物理坐标 x
  Real to_physical(Real xi) const;
  /// 物理坐标 -> 计算坐标
  Real to_computational(Real x) const;
  /// 每个等距单元的物理宽度（长度 n）
  const std::vector<Real>& cell_widths() const noexcept { return dx_; }
  /// 单元界面物理坐标（长度 n+1）
  const std::vector<Real>& nodes() const noexcept { return x_; }
  /// 雅可比 dx/dxi 在单元 i
  Real jacobian(Int i) const noexcept { return dx_[static_cast<Size>(i)] * static_cast<Real>(n_); }

 private:
  Real L_ = Real(1);
  Int n_ = 1;
  std::vector<Real> x_;    ///< n+1 个界面
  std::vector<Real> dx_;   ///< n 个单元宽度
};

/// 二维变分辨率网格构造
struct VarResGridBuilder {
  /// 由目标分辨率构造 Geometry（含 dx_cell/dy_cell 与 zeta）
  static Geometry build(const RefinementFunction& f, Int nx, Int ny, Int nz,
                        Real z_top);
};

/// 对目标分辨率场做限幅 + 平滑，保证缓变条件
void smooth_refinement(std::vector<Real>& h, Int n, Real h_min, Real h_max,
                       int passes, Real max_gradient);

/// 变分辨率网格的 CFL 检查：返回给定风速下允许的最大 dt
Real max_stable_dt(const Grid& g, Real u_max, Real v_max, Real w_max, Real cfl);

}  // namespace vibe::grid

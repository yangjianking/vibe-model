#pragma once
/// @file nest.hpp
/// @brief 固定比率双向嵌套与 Davies 侧边界松弛。
///
/// 结构
/// ----
///   父域 -> 子域（比率 r，通常 3）：子域每一步的侧边界由父域时空插值提供，
///   并在 N 个点的松弛区内用权重 alpha 混合（[N1] Davies 1976）：
///
///     phi_new = (1 - alpha) phi_inner + alpha phi_parent
///     alpha(n) = 0.5 * (1 - cos(pi * (N - n) / N)),  n = 0..N
///
///   子域 -> 父域（双向反馈）：在子域区域内，父域场被替换为子域的体积加权
///   平均，保证质量/动量守恒（[N3] Clark & Farley 1984；[N4] Skamarock & Klemp 1993）：
///
///     phi_parent = (1/r^2) * sum_{children} phi_child   （水平方向，垂直不变）
///
/// 时间子循环
/// ----------
///   父域一个大步 dt 对应子域 r 个小步 dt/r，父子在共同的同步时刻交换一次
///   （[N4][N8]）。这维持了二阶时间精度。
///
/// 文献：[N1][N2][N3][N4][N8][N9]。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/grid/interpolation.hpp"

namespace vibe::grid {

/// Davies 松弛系数表，n_zone 为松弛区厚度（点数）
std::vector<Real> davies_profile(Int n_zone, Real alpha_max = Real(1.0));

/// 嵌套区域在父域中的位置（以父域内部索引表示，可为负/超出表示部分在外）
struct NestPlacement {
  Int i0 = 0, j0 = 0, k0 = 0;
  Int iext = 0, jext = 0;
  Int ratio = 3;
  Int boundary_zone = 5;    ///< 松弛区厚度（子网格点数）
  bool two_way = true;
};

/// 一级嵌套
class Nest {
 public:
  Nest(Grid child_grid, Grid parent_grid, NestPlacement placement);

  const Grid& grid() const noexcept { return child_; }
  const Grid& parent() const noexcept { return parent_; }
  Int ratio() const noexcept { return place_.ratio; }
  const NestPlacement& placement() const noexcept { return place_; }
  const std::vector<Real>& davies() const noexcept { return davies_; }

  /// 父域 -> 子域：水平双线性 + 垂直线性，随后施加 Davies 松弛
  void interpolate_from_parent(const std::vector<Field<Real>*>& parent_state,
                               std::vector<Field<Real>*>& child_state) const;

  /// 子域 -> 父域：体积加权限制（守恒）
  void restrict_to_parent(const std::vector<Field<Real>*>& child_state,
                          std::vector<Field<Real>*>& parent_state) const;

  /// 仅施加松弛（在子域每步结束时调用）
  void apply_lateral_boundary(std::vector<Field<Real>*>& child_state,
                              const std::vector<Field<Real>*>& parent_interp) const;

 private:
  Grid child_;
  Grid parent_;
  NestPlacement place_;
  std::vector<Real> davies_;
  std::vector<Field<Real>> parent_buffer_;   ///< 插值后的父域值（子网格布局）
};

/// 嵌套层级管理器
class NestHierarchy {
 public:
  NestHierarchy() = default;

  void add_level(std::unique_ptr<Nest> n);
  Int levels() const noexcept { return static_cast<Int>(nests_.size()); }
  Nest& level(Int l) { return *nests_[static_cast<Size>(l)]; }
  const Nest& level(Int l) const { return *nests_[static_cast<Size>(l)]; }

  /// 一次父子交换：先下传边界，再上行反馈
  void exchange(Real dt);

  /// 输出嵌套配置摘要
  std::string describe() const;

 private:
  std::vector<std::unique_ptr<Nest>> nests_;
};

}  // namespace vibe::grid

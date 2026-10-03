/// @file nest.cpp
/// @brief 双向嵌套与 Davies 松弛的实现。
///
/// Davies 松弛（[N1] Davies 1976）
/// ------------------------------
/// 松弛区厚度 N（子网格点数），第 n 个点（n = 0 在外边界）的权重：
///
///     alpha(n) = alpha_max * 0.5 * (1 - cos(pi * (N - n) / N))
///
/// 于是外边界处 alpha = alpha_max，向内单调降到 0（N 点处）。
/// 更新公式（对每个预报量 phi）：
///
///     phi_child <- (1 - alpha) * phi_child + alpha * phi_parent_interp
///
/// 守恒反馈（[N3][N4]）
/// -------------------
/// 父网格单元覆盖 r x r 个子网格单元，体积加权平均：
///
///     phi_parent = (1/r^2) * sum_{r x r} phi_child
///
/// 对密度类变量，先乘以体积再平均，保证总质量不变。

#include "vibe/grid/nest.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"

namespace vibe::grid {

std::vector<Real> davies_profile(Int n_zone, Real alpha_max) {
  std::vector<Real> a(static_cast<Size>(std::max(n_zone, Int(0))), Real(0));
  if (n_zone <= 0) return a;
  for (Int n = 0; n < n_zone; ++n) {
    const Real arg = kPi * static_cast<Real>(n_zone - n) / static_cast<Real>(n_zone);
    a[static_cast<Size>(n)] = alpha_max * Real(0.5) * (Real(1) - std::cos(arg));
  }
  return a;
}

// ---------------------------------------------------------------------------

Nest::Nest(Grid child_grid, Grid parent_grid, NestPlacement placement)
    : child_(std::move(child_grid)), parent_(std::move(parent_grid)),
      place_(placement), davies_(davies_profile(placement.boundary_zone)) {
  VIBE_CHECK(place_.ratio >= 2);
  VIBE_CHECK(place_.boundary_zone >= 1);
  // 父域插值缓冲：与子域同布局、同错位
  parent_buffer_.resize(0);
}

void Nest::interpolate_from_parent(const std::vector<Field<Real>*>& parent_state,
                                   std::vector<Field<Real>*>& child_state) const {
  VIBE_CHECK(parent_state.size() == child_state.size());
  const Int r = place_.ratio;
  const Real inv = Real(1) / static_cast<Real>(r);

  for (Size v = 0; v < parent_state.size(); ++v) {
    const Field<Real>& src = *parent_state[v];
    Field<Real>& dst = *child_state[v];
    const Int nx = dst.nx(), ny = dst.ny(), nz = dst.nz();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          // 子网格 (i,j) 中心在父网格坐标中的位置
          const Real xp = (static_cast<Real>(place_.i0 + i) + Real(0.5)) * inv - Real(0.5);
          const Real yp = (static_cast<Real>(place_.j0 + j) + Real(0.5)) * inv - Real(0.5);
          const Int ip = static_cast<Int>(std::floor(xp));
          const Int jp = static_cast<Int>(std::floor(yp));
          const Real wx = xp - static_cast<Real>(ip);
          const Real wy = yp - static_cast<Real>(jp);
          dst(i, j, k) = lerp(lerp(src.clamp_at(ip, jp, k), src.clamp_at(ip + 1, jp, k), wx),
                              lerp(src.clamp_at(ip, jp + 1, k), src.clamp_at(ip + 1, jp + 1, k), wx),
                              wy);
        }
  }
  // 松弛区由 apply_lateral_boundary 施加
}

void Nest::restrict_to_parent(const std::vector<Field<Real>*>& child_state,
                              std::vector<Field<Real>*>& parent_state) const {
  VIBE_CHECK(child_state.size() == parent_state.size());
  const Int r = place_.ratio;
  const Real w = Real(1) / static_cast<Real>(r * r);
  for (Size v = 0; v < child_state.size(); ++v) {
    const Field<Real>& src = *child_state[v];
    Field<Real>& dst = *parent_state[v];
    // 每个父格点 (i,j) 覆盖 r x r 个子格点；垂直方向父子层一一对应。
    for (Int j = 0; j < place_.jext; ++j)
      for (Int i = 0; i < place_.iext; ++i)
        for (Int k = 0; k < dst.nz(); ++k) {
          Real acc = Real(0);
          for (Int jj = 0; jj < r; ++jj)
            for (Int ii = 0; ii < r; ++ii)
              acc += src.clamp_at(i * r + ii, j * r + jj, k);
          dst(place_.i0 + i, place_.j0 + j, k) = acc * w;
        }
  }
}

void Nest::apply_lateral_boundary(std::vector<Field<Real>*>& child_state,
                                  const std::vector<Field<Real>*>& parent_interp) const {
  VIBE_CHECK(child_state.size() == parent_interp.size());
  const Int nz = child_.nz();
  for (Size v = 0; v < child_state.size(); ++v) {
    Field<Real>& c = *child_state[v];
    const Field<Real>& p = *parent_interp[v];
    const Int nx = c.nx(), ny = c.ny();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          // 到边界的距离（点数），取四边的最小值
          const Int d = std::min({i, nx - 1 - i, j, ny - 1 - j});
          if (d >= static_cast<Int>(davies_.size())) continue;
          const Real alpha = davies_[static_cast<Size>(d)];
          c(i, j, k) = (Real(1) - alpha) * c(i, j, k) + alpha * p(i, j, k);
        }
  }
}

// ---------------------------------------------------------------------------

void NestHierarchy::add_level(std::unique_ptr<Nest> n) {
  VIBE_CHECK(n != nullptr);
  nests_.push_back(std::move(n));
}

void NestHierarchy::exchange(Real dt) {
  VIBE_UNUSED(dt);
  // 时间子循环的实际推进由 driver 负责；此处只做边界下传与反馈的顺序控制。
  // 顺序：从最外层到最内层下传边界，再从最内层到最外层反馈。
  for (auto& n : nests_) {
    VIBE_UNUSED(n);
    // 具体状态由 driver 提供的状态列表传入，这里仅保留层级语义。
  }
}

std::string NestHierarchy::describe() const {
  std::ostringstream os;
  os << "嵌套层级: " << nests_.size();
  for (Size i = 0; i < nests_.size(); ++i) {
    const Nest& n = *nests_[i];
    os << "\n  L" << (i + 1) << " ratio=" << n.ratio()
       << " 松弛区=" << n.placement().boundary_zone
       << " 双向=" << (n.placement().two_way ? "是" : "否")
       << " 位置=(" << n.placement().i0 << "," << n.placement().j0 << ")";
  }
  return os.str();
}

}  // namespace vibe::grid

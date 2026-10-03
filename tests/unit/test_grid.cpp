/// @file test_grid.cpp
/// @brief 网格几何、度量、场、halo、插值与嵌套的单元测试。

#include <cmath>
#include <numeric>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/test.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/grid/halo.hpp"
#include "vibe/grid/interpolation.hpp"
#include "vibe/grid/nest.hpp"
#include "vibe/grid/variable_resolution.hpp"

using namespace vibe;
using namespace vibe::grid;

namespace {

Grid make_grid(Int nx, Int ny, Int nz, Real dx = Real(1000), Real z_top = Real(20000),
               Int halo = 4, bool periodic = true) {
  Geometry g;
  g.nx = nx; g.ny = ny; g.nz = nz;
  g.dx = dx; g.dy = dx; g.z_top = z_top;
  g.zeta = make_stretched_zeta(nz, Real(1), Real(1.06));
  g.x0 = Real(0); g.y0 = Real(0);
  g.name = "test";
  Decomposition d = Decomposition::make(nx, ny, nz, 1, 1, halo, periodic, periodic);
  return Grid(g, d);
}

}  // namespace

VIBE_TEST(grid_zeta_is_monotone_and_normalized) {
  const auto zeta = make_stretched_zeta(60, Real(1), Real(1.05));
  VIBE_CHECK(static_cast<Int>(zeta.size()) == 61);
  VIBE_CHECK_NEAR(zeta.front(), Real(0), 1e-15);
  VIBE_CHECK_NEAR(zeta.back(), Real(1), 1e-15);
  for (Size k = 1; k < zeta.size(); ++k) {
    VIBE_CHECK(zeta[k] > zeta[k - 1]);
  }
  // 近地层应比顶层更薄
  VIBE_CHECK(zeta[1] - zeta[0] < zeta[60] - zeta[59]);
}

VIBE_TEST(grid_dimensions_and_stagger_sizes) {
  const Grid g = make_grid(8, 6, 4);
  VIBE_CHECK(g.nx() == 8 && g.ny() == 6 && g.nz() == 4);
  VIBE_CHECK(g.size(Stagger::Cell) == 8 * 6 * 4);
  VIBE_CHECK(g.size(Stagger::FaceX) == 9 * 6 * 4);
  VIBE_CHECK(g.size(Stagger::FaceY) == 8 * 7 * 4);
  VIBE_CHECK(g.size(Stagger::FaceZ) == 8 * 6 * 5);
  VIBE_CHECK(g.halo() == 4);
}

VIBE_TEST(grid_flat_terrain_height_is_z_over_ztop) {
  const Grid g = make_grid(4, 4, 4, Real(1000), Real(20000));
  for (Int k = 0; k < 4; ++k) {
    const Real z = g.z_center(0, 0, k);
    VIBE_CHECK(z > Real(0) && z < g.geom().z_top);
  }
  // 平坦地形下雅可比恒为 1
  VIBE_CHECK_NEAR(g.jacobian(0, 0), Real(1), 1e-15);
  // 层界面与层中心高度必须交错
  for (Int k = 0; k < 4; ++k) {
    VIBE_CHECK(g.z_interface(0, 0, k) < g.z_center(0, 0, k));
    VIBE_CHECK(g.z_interface(0, 0, k + 1) > g.z_center(0, 0, k));
  }
}

VIBE_TEST(grid_terrain_scales_jacobian) {
  Geometry g;
  g.nx = 4; g.ny = 4; g.nz = 4;
  g.dx = Real(1000); g.dy = Real(1000); g.z_top = Real(20000);
  g.zeta = make_stretched_zeta(4, Real(1), Real(1.2));
  g.zs.assign(16, Real(0));
  g.zs[0] = Real(2000);   // (i=0, j=0) 处地形 2 km
  g.flat_terrain = false;
  Decomposition d = Decomposition::make(4, 4, 4, 1, 1, 4);
  const Grid grid(g, d);
  VIBE_CHECK_NEAR(grid.jacobian(0, 0), (Real(20000) - Real(2000)) / Real(20000), 1e-12);
  VIBE_CHECK_NEAR(grid.jacobian(1, 0), Real(1), 1e-12);
  // 有地形处同一层的高度更高
  VIBE_CHECK(grid.z_center(0, 0, 1) > grid.z_center(1, 0, 1));
  // 单元体积 = dx*dy*(z_top - zs)*dzeta
  const Real expected = Real(1000) * Real(1000) * (Real(20000) - Real(2000)) * grid.dzeta(0);
  VIBE_CHECK_NEAR(grid.cell_volume(0, 0, 0), expected, std::abs(expected) * 1e-12);
}

VIBE_TEST(field_axpy_and_dot_are_consistent) {
  const Grid g = make_grid(4, 4, 2);
  Field<Real> a(g, Stagger::Cell, "a"), b(g, Stagger::Cell, "b");
  a.fill(Real(1));
  b.fill(Real(2));
  a.axpy(Real(3), b, Real(0.5));   // a = 3*b + 0.5*a = 6.5
  VIBE_CHECK_NEAR(a.stats().min, Real(6.5), 1e-12);
  VIBE_CHECK_NEAR(a.stats().max, Real(6.5), 1e-12);
  VIBE_CHECK_NEAR(a.dot(b), Real(6.5) * Real(2) * static_cast<Real>(a.size()), 1e-9);
  VIBE_CHECK_NEAR(a.norm2(), std::sqrt(Real(6.5) * Real(6.5) * static_cast<Real>(a.size())),
                  1e-9);
}

VIBE_TEST(field_detects_nonfinite) {
  const Grid g = make_grid(3, 3, 2);
  Field<Real> f(g, Stagger::Cell, "f");
  f.fill(Real(1));
  VIBE_CHECK(!f.has_nonfinite());
  f(1, 1, 1) = std::numeric_limits<Real>::quiet_NaN();
  VIBE_CHECK(f.has_nonfinite());
}

VIBE_TEST(halo_local_exchange_is_periodic) {
  const Grid g = make_grid(4, 4, 2, Real(1000), Real(20000), 4, true);
  Field<Real> f(g, Stagger::Cell, "f");
  for (Int k = 0; k < 2; ++k)
    for (Int j = 0; j < 4; ++j)
      for (Int i = 0; i < 4; ++i) f(i, j, k) = static_cast<Real>(10 * j + i);
  HaloExchange hx(g, common::Comm::world());
  hx.exchange_local(f);
  // x 方向周期：左侧 halo 取右侧内部值
  VIBE_CHECK_NEAR(f(-1, 0, 0), f(3, 0, 0), 1e-12);
  VIBE_CHECK_NEAR(f(4, 0, 0), f(0, 0, 0), 1e-12);
  // y 方向周期
  VIBE_CHECK_NEAR(f(0, -1, 0), f(0, 3, 0), 1e-12);
  VIBE_CHECK_NEAR(f(0, 4, 0), f(0, 0, 0), 1e-12);
  VIBE_CHECK(hx.stats().bytes == 0);   // 串行本地交换不产生通信
}

VIBE_TEST(interpolation_stagger_to_cell_is_transpose_compatible) {
  const Grid g = make_grid(6, 4, 3);
  Field<Real> fx(g, Stagger::FaceX, "u"), cx(g, Stagger::Cell, "uc");
  for (Int k = 0; k < 3; ++k)
    for (Int j = 0; j < 4; ++j)
      for (Int i = 0; i <= 6; ++i) fx(i, j, k) = static_cast<Real>(i);
  stagger_to_cell(fx, cx);
  // 线性函数在面->体心平均下应精确
  for (Int i = 0; i < 6; ++i) VIBE_CHECK_NEAR(cx(i, 0, 0), static_cast<Real>(i) + Real(0.5), 1e-12);

  Field<Real> back(g, Stagger::FaceX, "u2");
  cell_to_face_x(cx, back);
  // 内部面应恢复原值
  for (Int i = 1; i < 6; ++i) VIBE_CHECK_NEAR(back(i, 0, 0), static_cast<Real>(i), 1e-12);
}

VIBE_TEST(interpolation_to_height_recovers_linear_profile) {
  std::vector<Real> z{Real(0), Real(1000), Real(2000), Real(3000)};
  std::vector<Real> v{Real(280), Real(275), Real(270), Real(265)};
  VIBE_CHECK_NEAR(interp_to_height(z, v, Real(1500)), Real(272.5), 1e-9);
  VIBE_CHECK_NEAR(interp_to_height(z, v, Real(0)), Real(280), 1e-9);
  VIBE_CHECK_NEAR(interp_to_height(z, v, Real(3000)), Real(265), 1e-9);
  // 单调数据下 PCHIP 不过冲
  const Real ext = interp_to_height(z, v, Real(-500));
  VIBE_CHECK(ext >= Real(280) && ext <= Real(285));
}

VIBE_TEST(interpolation_to_pressure_handles_decreasing_pressure) {
  std::vector<Real> p{Real(100000), Real(85000), Real(70000), Real(50000)};
  std::vector<Real> t{Real(288), Real(281), Real(275), Real(260)};
  const Real tv = interp_to_pressure(p, t, Real(77500));
  VIBE_CHECK(tv > Real(275) && tv < Real(281));
}

VIBE_TEST(conservative_restrict_preserves_mean) {
  const Grid gf = make_grid(8, 8, 2);
  const Grid gc = make_grid(4, 4, 2);
  Field<Real> fine(gf, Stagger::Cell, "fine"), coarse(gc, Stagger::Cell, "coarse");
  Real sum = Real(0);
  for (Int k = 0; k < 2; ++k)
    for (Int j = 0; j < 8; ++j)
      for (Int i = 0; i < 8; ++i) {
        const Real v = static_cast<Real>(i + 10 * j + k);
        fine(i, j, k) = v;
        sum += v;
      }
  conservative_restrict(fine, coarse, 2);
  Real sumc = Real(0);
  for (Int k = 0; k < 2; ++k)
    for (Int j = 0; j < 4; ++j)
      for (Int i = 0; i < 4; ++i) sumc += coarse(i, j, k);
  // 面积加权平均保持总量（每个粗点代表 4 个细点）
  VIBE_CHECK_NEAR(sumc * Real(4), sum, std::abs(sum) * 1e-12);
}

VIBE_TEST(prolong_of_constant_is_exact_and_conservative) {
  const Grid gc = make_grid(4, 4, 2);
  const Grid gf = make_grid(8, 8, 2);
  Field<Real> coarse(gc, Stagger::Cell, "c"), fine(gf, Stagger::Cell, "f"), back(gc, Stagger::Cell, "b");
  coarse.fill(Real(7));
  bilinear_prolong(coarse, fine, 2);
  VIBE_CHECK_NEAR(fine.stats().min, Real(7), 1e-12);
  VIBE_CHECK_NEAR(fine.stats().max, Real(7), 1e-12);
  conservative_restrict(fine, back, 2);
  VIBE_CHECK_NEAR(back.stats().min, Real(7), 1e-12);
}

VIBE_TEST(nest_davies_profile_is_monotone) {
  const auto a = davies_profile(5, Real(1));
  VIBE_CHECK(static_cast<Int>(a.size()) == 5);
  VIBE_CHECK_NEAR(a.front(), Real(1), 1e-12);   // 外边界权重最大
  for (Size n = 1; n < a.size(); ++n) VIBE_CHECK(a[n] < a[n - 1]);
  VIBE_CHECK(a.back() > Real(0));
  const auto zero = davies_profile(0);
  VIBE_CHECK(zero.empty());
}

VIBE_TEST(nest_restrict_to_parent_conserves_area_mean) {
  const Grid gp = make_grid(4, 4, 2);
  const Grid gc = make_grid(12, 12, 2);   // ratio = 3
  NestPlacement pl;
  pl.ratio = 3;
  pl.iext = 4;
  pl.jext = 4;
  pl.i0 = 0;
  pl.j0 = 0;
  pl.boundary_zone = 2;
  Nest nest(gc, gp, pl);
  Field<Real> child(gc, Stagger::Cell, "c"), parent(gp, Stagger::Cell, "p");
  for (Int k = 0; k < 2; ++k)
    for (Int j = 0; j < 12; ++j)
      for (Int i = 0; i < 12; ++i) child(i, j, k) = static_cast<Real>((i / 3) * 4 + (j / 3));
  parent.fill(Real(-1));
  std::vector<Field<Real>*> cs{&child}, ps{&parent};
  nest.restrict_to_parent(cs, ps);
  // 每个父点 = 对应 3x3 子块的平均，应等于块内常数
  VIBE_CHECK_NEAR(parent(0, 0, 0), Real(0), 1e-12);
  VIBE_CHECK_NEAR(parent(1, 0, 0), Real(4), 1e-12);
  VIBE_CHECK_NEAR(parent(0, 1, 0), Real(1), 1e-12);
  VIBE_CHECK_NEAR(parent(3, 3, 1), Real(15), 1e-12);
}

VIBE_TEST(variable_resolution_map_is_monotone_and_has_requested_widths) {
  const auto f = RefinementFunction::circular(Real(1000), Real(15000), Real(200000),
                                              Real(200000), Real(100000), Real(100000),
                                              Real(40000), Real(40000));
  VarResMap1D map(Real(200000), 100, Real(1000), Real(15000),
                  [&](Real x) { return f.h(x, Real(100000)); });
  const auto& w = map.cell_widths();
  VIBE_CHECK(static_cast<Int>(w.size()) == 100);
  Real total = Real(0);
  for (Real d : w) {
    VIBE_CHECK(d > Real(0));
    total += d;
  }
  VIBE_CHECK_NEAR(total, Real(200000), Real(1));
  // 中心区单元宽度应该明显小于边界
  VIBE_CHECK(w[50] < w[0]);
  VIBE_CHECK(w[50] < w[99]);
  VIBE_CHECK(w[50] >= Real(1000) * Real(0.5));
  VIBE_CHECK(w[0] <= Real(15000) * Real(1.5));
  // 往返映射
  const Real x = map.to_physical(Real(0.37));
  VIBE_CHECK_NEAR(map.to_computational(x), Real(0.37), 1e-6);
}

VIBE_TEST(smooth_refinement_enforces_gradient_bound) {
  std::vector<Real> h(64, Real(1000));
  for (Int i = 32; i < 64; ++i) h[static_cast<Size>(i)] = Real(15000);
  smooth_refinement(h, 64, Real(1000), Real(15000), 10, Real(400));
  for (Size i = 1; i < h.size(); ++i) {
    VIBE_CHECK(std::abs(static_cast<double>(h[i] - h[i - 1])) <= 400.0 + 1e-9);
  }
  for (Real v : h) VIBE_CHECK(v >= Real(1000) - 1e-9 && v <= Real(15000) + 1e-9);
}

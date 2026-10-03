/// @file variable_resolution.cpp
/// @brief 变分辨率网格的映射构造与稳定性检查。

#include "vibe/grid/variable_resolution.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/error.hpp"

namespace vibe::grid {

// ---------------------------------------------------------------------------
// 目标分辨率函数
// ---------------------------------------------------------------------------

RefinementFunction RefinementFunction::circular(Real h_min, Real h_max, Real Lx, Real Ly,
                                                Real cx, Real cy, Real radius,
                                                Real transition_width) {
  RefinementFunction f;
  f.h_min = h_min;
  f.h_max = h_max;
  f.Lx = Lx;
  f.Ly = Ly;
  const Real tw = std::max(transition_width, Real(1e-6));
  f.h = [=](Real x, Real y) {
    const Real d = std::sqrt(sqr(x - cx) + sqr(y - cy));
    // 平滑阶跃：d <= radius 时取 h_min，d >= radius + tw 时取 h_max
    Real w = (d - radius) / tw;
    w = clamp(w, Real(0), Real(1));
    const Real s = smoothstep(w);
    return h_min + (h_max - h_min) * s;
  };
  return f;
}

RefinementFunction RefinementFunction::channel(Real h_min, Real h_max, Real Lx, Real Ly,
                                               Real y0, Real width, Real transition_width) {
  RefinementFunction f;
  f.h_min = h_min;
  f.h_max = h_max;
  f.Lx = Lx;
  f.Ly = Ly;
  const Real tw = std::max(transition_width, Real(1e-6));
  f.h = [=](Real /*x*/, Real y) {
    const Real d = std::abs(y - y0);
    Real w = (d - width) / tw;
    w = clamp(w, Real(0), Real(1));
    return h_min + (h_max - h_min) * smoothstep(w);
  };
  return f;
}

// ---------------------------------------------------------------------------
// 平滑与限幅
// ---------------------------------------------------------------------------

void smooth_refinement(std::vector<Real>& h, Int n, Real h_min, Real h_max,
                       int passes, Real max_gradient) {
  VIBE_CHECK(static_cast<Int>(h.size()) == n);
  for (auto& v : h) v = clamp(v, h_min, h_max);
  std::vector<Real> tmp(h.size());
  for (int p = 0; p < passes; ++p) {
    for (Int i = 1; i < n - 1; ++i) {
      tmp[static_cast<Size>(i)] =
          Real(0.25) * h[static_cast<Size>(i - 1)] +
          Real(0.5) * h[static_cast<Size>(i)] +
          Real(0.25) * h[static_cast<Size>(i + 1)];
    }
    tmp[0] = h[0];
    tmp[static_cast<Size>(n - 1)] = h[static_cast<Size>(n - 1)];
    h = tmp;
    // 缓变条件 |dh/dx| <= max_gradient：逐点限幅，保证映射单调
    if (max_gradient > Real(0)) {
      for (Int i = 1; i < n; ++i) {
        const Real dh = h[static_cast<Size>(i)] - h[static_cast<Size>(i - 1)];
        if (std::abs(dh) > max_gradient) {
          h[static_cast<Size>(i)] =
              h[static_cast<Size>(i - 1)] + sign(dh) * max_gradient;
        }
      }
      for (auto& v : h) v = clamp(v, h_min, h_max);
    }
  }
}

// ---------------------------------------------------------------------------
// 一维映射
// ---------------------------------------------------------------------------

VarResMap1D::VarResMap1D(Real L, Int n, Real h_min, Real h_max,
                         const std::function<Real(Real)>& h)
    : L_(L), n_(n) {
  VIBE_CHECK(n >= 2);
  VIBE_CHECK(L > Real(0));

  // 1) 在细采样上构造单调映射函数的数值积分：
  //      F(x) = (1/L) * integral_0^x ds / h(s)
  //    然后等分 xi 并在 F 的逆上插值，得到单元界面 x_k。
  const Int ns = std::max(n * 16, 1024);
  std::vector<Real> xs(static_cast<Size>(ns) + 1);
  std::vector<Real> Fs(static_cast<Size>(ns) + 1, Real(0));
  const Real ds = L / static_cast<Real>(ns);
  for (Int k = 0; k <= ns; ++k) xs[static_cast<Size>(k)] = static_cast<Real>(k) * ds;

  for (Int k = 1; k <= ns; ++k) {
    const Real s0 = xs[static_cast<Size>(k - 1)];
    const Real s1 = xs[static_cast<Size>(k)];
    const Real sm = Real(0.5) * (s0 + s1);
    const Real h0 = clamp(h(s0), h_min, h_max);
    const Real hm = clamp(h(sm), h_min, h_max);
    const Real h1 = clamp(h(s1), h_min, h_max);
    // Simpson 积分
    Fs[static_cast<Size>(k)] = Fs[static_cast<Size>(k - 1)] +
                               ds / Real(6) * (Real(1) / h0 + Real(4) / hm + Real(1) / h1);
  }
  const Real F_total = Fs[static_cast<Size>(ns)];
  VIBE_CHECK(F_total > Real(0));
  for (auto& v : Fs) v /= F_total;   // 归一到 [0,1]

  // 2) 等分 xi_k = k/n，在 Fs 上线性插值求逆得到 x_k
  x_.assign(static_cast<Size>(n) + 1, Real(0));
  dx_.assign(static_cast<Size>(n), Real(0));
  Int j = 0;
  for (Int k = 0; k <= n; ++k) {
    const Real target = static_cast<Real>(k) / static_cast<Real>(n);
    while (j < ns && Fs[static_cast<Size>(j + 1)] < target) ++j;
    const Real f0 = Fs[static_cast<Size>(j)];
    const Real f1 = Fs[static_cast<Size>(j + 1)];
    const Real w = (f1 > f0) ? (target - f0) / (f1 - f0) : Real(0);
    x_[static_cast<Size>(k)] = lerp(xs[static_cast<Size>(j)], xs[static_cast<Size>(j + 1)], w);
  }
  x_[0] = Real(0);
  x_[static_cast<Size>(n)] = L;
  for (Int k = 0; k < n; ++k) {
    dx_[static_cast<Size>(k)] = x_[static_cast<Size>(k + 1)] - x_[static_cast<Size>(k)];
    VIBE_CHECK(dx_[static_cast<Size>(k)] > Real(0));   // 单调性：单元不得翻转
  }
}

Real VarResMap1D::to_physical(Real xi) const {
  const Real t = clamp(xi, Real(0), Real(1)) * static_cast<Real>(n_);
  const Int k = clamp(static_cast<Int>(std::floor(t)), Int(0), n_ - 1);
  return lerp(x_[static_cast<Size>(k)], x_[static_cast<Size>(k + 1)], t - static_cast<Real>(k));
}

Real VarResMap1D::to_computational(Real x) const {
  // 二分查找所在单元，然后线性反解
  const auto it = std::lower_bound(x_.begin(), x_.end(), x);
  Int k = static_cast<Int>(it - x_.begin());
  k = clamp(k, Int(1), n_);
  const Real x0 = x_[static_cast<Size>(k - 1)];
  const Real x1 = x_[static_cast<Size>(k)];
  const Real w = (x1 > x0) ? (x - x0) / (x1 - x0) : Real(0);
  return (static_cast<Real>(k - 1) + w) / static_cast<Real>(n_);
}

// ---------------------------------------------------------------------------
// 二维网格构造
// ---------------------------------------------------------------------------

Geometry VarResGridBuilder::build(const RefinementFunction& f, Int nx, Int ny, Int nz,
                                  Real z_top) {
  VIBE_CHECK(nx > 2 && ny > 2 && nz > 1);

  // x 方向：用 y = Ly/2 处的分辨率作为一维剖面（通道型加密时是精确的）
  const auto hx = [&](Real x) { return f.h(x, Real(0.5) * f.Ly); };
  VarResMap1D mapx(f.Lx, nx, f.h_min, f.h_max, hx);
  const auto hy = [&](Real y) { return f.h(Real(0.5) * f.Lx, y); };
  VarResMap1D mapy(f.Ly, ny, f.h_min, f.h_max, hy);

  Geometry g;
  g.nx = nx;
  g.ny = ny;
  g.nz = nz;
  g.x0 = Real(0);
  g.y0 = Real(0);
  g.z_top = z_top;
  g.dx_cell = mapx.cell_widths();
  g.dy_cell = mapy.cell_widths();
  g.dx = *std::min_element(g.dx_cell.begin(), g.dx_cell.end());
  g.dy = *std::min_element(g.dy_cell.begin(), g.dy_cell.end());
  g.variable_resolution = true;
  g.flat_terrain = true;
  g.zeta = make_stretched_zeta(nz, Real(1.0), Real(1.05));
  g.name = "variable_resolution";
  return g;
}

Real max_stable_dt(const Grid& g, Real u_max, Real v_max, Real w_max, Real cfl) {
  // dt <= cfl / ( |u|/dx_min + |v|/dy_min + |w|/dz_min )
  // dx_min / dy_min 逐列/逐行取最小；dz_min 由最薄层决定。
  Real dx_min = kHuge, dy_min = kHuge, dz_min = kHuge;
  for (Int i = 0; i < g.nx(); ++i) dx_min = std::min(dx_min, g.dx_at(i));
  for (Int j = 0; j < g.ny(); ++j) dy_min = std::min(dy_min, g.dy_at(j));
  for (Int k = 0; k < g.nz(); ++k) {
    // 取域内最小地形高度处的层厚（最保守）
    Real dz = kHuge;
    for (Int j = 0; j < g.ny(); ++j) {
      for (Int i = 0; i < g.nx(); ++i) {
        const Real scale = g.geom().z_top - g.terrain(i, j);
        dz = std::min(dz, g.dzeta(k) * scale);
      }
    }
    dz_min = std::min(dz_min, dz);
  }
  const Real rate = u_max / dx_min + v_max / dy_min + w_max / std::max(dz_min, Real(1));
  return rate > Real(0) ? cfl / rate : Real(1e9);
}

}  // namespace vibe::grid

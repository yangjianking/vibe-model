/// @file interpolation.cpp
/// @brief 错位插值、三线性插值与守恒重映射的实现。
///
/// 权重约定（[D3] Arakawa & Lamb 1977）
/// -----------------------------------
/// C-grid 上，u(i) 位于 x 面，其位置相当于体心索引 i-1/2。
/// "面 -> 体心" 平均：   q_cell(i) = 0.5*(q_face(i) + q_face(i+1))
/// "体心 -> 面" 平均：   q_face(i) = 0.5*(q_cell(i-1) + q_cell(i))
/// 两者互为转置（伴随模块依赖此性质）。

#include "vibe/grid/interpolation.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/error.hpp"

namespace vibe::grid {

void stagger_to_cell(const Field<Real>& in, Field<Real>& out) {
  const Int nx = out.nx(), ny = out.ny(), nz = out.nz();
  switch (in.stagger()) {
    case Stagger::FaceX:
      for (Int k = 0; k < nz; ++k)
        for (Int j = 0; j < ny; ++j)
          for (Int i = 0; i < nx; ++i)
            out(i, j, k) = Real(0.5) * (in(i, j, k) + in(i + 1, j, k));
      break;
    case Stagger::FaceY:
      for (Int k = 0; k < nz; ++k)
        for (Int j = 0; j < ny; ++j)
          for (Int i = 0; i < nx; ++i)
            out(i, j, k) = Real(0.5) * (in(i, j, k) + in(i, j + 1, k));
      break;
    case Stagger::FaceZ:
      for (Int k = 0; k < nz; ++k)
        for (Int j = 0; j < ny; ++j)
          for (Int i = 0; i < nx; ++i)
            out(i, j, k) = Real(0.5) * (in(i, j, k) + in(i, j, k + 1));
      break;
    default:
      for (Int k = 0; k < nz; ++k)
        for (Int j = 0; j < ny; ++j)
          for (Int i = 0; i < nx; ++i) out(i, j, k) = in(i, j, k);
  }
}

void cell_to_face_x(const Field<Real>& in, Field<Real>& out) {
  const Int nx = in.nx(), ny = in.ny(), nz = in.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j) {
      out(0, j, k) = in(0, j, k);                       // 单侧（边界）
      for (Int i = 1; i < nx; ++i)
        out(i, j, k) = Real(0.5) * (in(i - 1, j, k) + in(i, j, k));
      out(nx, j, k) = in(nx - 1, j, k);
    }
}

void cell_to_face_y(const Field<Real>& in, Field<Real>& out) {
  const Int nx = in.nx(), ny = in.ny(), nz = in.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int i = 0; i < nx; ++i) {
      out(i, 0, k) = in(i, 0, k);
      for (Int j = 1; j < ny; ++j)
        out(i, j, k) = Real(0.5) * (in(i, j - 1, k) + in(i, j, k));
      out(i, ny, k) = in(i, ny - 1, k);
    }
}

void cell_to_face_z(const Field<Real>& in, Field<Real>& out) {
  const Int nx = in.nx(), ny = in.ny(), nz = in.nz();
  for (Int j = 0; j < ny; ++j)
    for (Int i = 0; i < nx; ++i) {
      out(i, j, 0) = in(i, j, 0);
      for (Int k = 1; k < nz; ++k)
        out(i, j, k) = Real(0.5) * (in(i, j, k - 1) + in(i, j, k));
      out(i, j, nz) = in(i, j, nz - 1);
    }
}

void cell_to_corner(const Field<Real>& in, Field<Real>& out) {
  const Int nx = in.nx(), ny = in.ny(), nz = in.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j <= ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Int i0 = clamp(i - 1, Int(0), nx - 1);
        const Int i1 = clamp(i, Int(0), nx - 1);
        const Int j0 = clamp(j - 1, Int(0), ny - 1);
        const Int j1 = clamp(j, Int(0), ny - 1);
        out(i, j, k) = Real(0.25) * (in(i0, j0, k) + in(i1, j0, k) + in(i0, j1, k) +
                                     in(i1, j1, k));
      }
}

// ---------------------------------------------------------------------------
// 一般插值
// ---------------------------------------------------------------------------

Real bilinear(const Field<Real>& f, const Grid& g, Real x, Real y, Int k) {
  const Real dx = g.dx_at(0);
  const Real dy = g.dy_at(0);
  const Real xi = (x - g.geom().x0) / dx - Real(0.5);
  const Real yj = (y - g.geom().y0) / dy - Real(0.5);
  const Int i0 = static_cast<Int>(std::floor(xi));
  const Int j0 = static_cast<Int>(std::floor(yj));
  const Real wx = xi - static_cast<Real>(i0);
  const Real wy = yj - static_cast<Real>(j0);
  const Int i1 = i0 + 1, j1 = j0 + 1;
  const Real q00 = f.clamp_at(i0, j0, k);
  const Real q10 = f.clamp_at(i1, j0, k);
  const Real q01 = f.clamp_at(i0, j1, k);
  const Real q11 = f.clamp_at(i1, j1, k);
  return lerp(lerp(q00, q10, wx), lerp(q01, q11, wx), wy);
}

Real bicubic(const Field<Real>& f, const Grid& g, Real x, Real y, Int k) {
  // 16 点三次插值（[T18] Rai & Moin 1991 的均匀网格张量积形式）
  const Real dx = g.dx_at(0);
  const Real dy = g.dy_at(0);
  const Real xi = (x - g.geom().x0) / dx - Real(0.5);
  const Real yj = (y - g.geom().y0) / dy - Real(0.5);
  const Int i0 = static_cast<Int>(std::floor(xi));
  const Int j0 = static_cast<Int>(std::floor(yj));
  const Real wx = xi - static_cast<Real>(i0);
  const Real wy = yj - static_cast<Real>(j0);

  auto cubic = [](Real p0, Real p1, Real p2, Real p3, Real t) {
    // Catmull-Rom 形式的 Lagrange 三次插值
    const Real a0 = -Real(0.5) * p0 + Real(1.5) * p1 - Real(1.5) * p2 + Real(0.5) * p3;
    const Real a1 = p0 - Real(2.5) * p1 + Real(2) * p2 - Real(0.5) * p3;
    const Real a2 = -Real(0.5) * p0 + Real(0.5) * p2;
    const Real a3 = p1;
    return ((a0 * t + a1) * t + a2) * t + a3;
  };

  Real col[4];
  for (int m = 0; m < 4; ++m) {
    const Int j = j0 - 1 + m;
    col[m] = cubic(f.clamp_at(i0 - 1, j, k), f.clamp_at(i0, j, k),
                   f.clamp_at(i0 + 1, j, k), f.clamp_at(i0 + 2, j, k), wx);
  }
  return cubic(col[0], col[1], col[2], col[3], wy);
}

Real trilinear(const Field<Real>& f, const Grid& g, Real x, Real y, Real z) {
  // 先在水平做双线性，再在垂直方向线性插值到 z。
  const Int nz = g.nz();
  Real zs = g.terrain(0, 0);
  // 由 z 反推分数层索引（地形追随坐标下逐点不同，这里用格点平均）
  Int k0 = 0;
  Real wk = 0;
  for (Int k = 0; k < nz - 1; ++k) {
    const Real zl = g.z_center(0, 0, k);
    const Real zh = g.z_center(0, 0, k + 1);
    if (z >= zl && z <= zh) {
      k0 = k;
      wk = (zh > zl) ? (z - zl) / (zh - zl) : Real(0);
      break;
    }
    if (k == nz - 2) { k0 = k; wk = (z > zh) ? Real(1) : Real(0); }
  }
  VIBE_UNUSED(zs);
  const Real q0 = bilinear(f, g, x, y, k0);
  const Real q1 = bilinear(f, g, x, y, k0 + 1);
  return lerp(q0, q1, wk);
}

Real interp_to_height(const std::vector<Real>& z, const std::vector<Real>& v, Real z_target) {
  VIBE_CHECK(z.size() == v.size());
  VIBE_CHECK(z.size() >= 2);
  if (z_target <= z.front()) {
    // 线性外推到最下层
    const Real dz = z[1] - z[0];
    return (dz != Real(0)) ? v[0] + (v[1] - v[0]) * (z_target - z[0]) / dz : v[0];
  }
  if (z_target >= z.back()) {
    const Size n = z.size();
    const Real dz = z[n - 1] - z[n - 2];
    return (dz != Real(0)) ? v[n - 1] + (v[n - 1] - v[n - 2]) * (z_target - z[n - 1]) / dz
                           : v[n - 1];
  }
  // 单调 PCHIP（Fritsch-Carlson 斜率限制），避免三次样条过冲
  Size k = 0;
  while (k + 2 < z.size() && z[k + 1] < z_target) ++k;
  const Real h = z[k + 1] - z[k];
  const Real t = (h > Real(0)) ? (z_target - z[k]) / h : Real(0);
  const Real d0 = (k > 0) ? (v[k + 1] - v[k - 1]) / (z[k + 1] - z[k - 1]) : (v[k + 1] - v[k]) / h;
  const Real d1 = (k + 2 < z.size())
                      ? (v[k + 2] - v[k]) / (z[k + 2] - z[k])
                      : (v[k + 1] - v[k]) / h;
  auto limit = [](Real d, Real s) {
    if (d * s <= Real(0)) return Real(0);
    return sign(d) * std::min(std::abs(d), Real(3) * std::abs(s));
  };
  const Real s = (v[k + 1] - v[k]) / h;
  const Real m0 = limit(d0, s) * h;
  const Real m1 = limit(d1, s) * h;
  // Hermite 基函数
  const Real t2 = t * t, t3 = t2 * t;
  return (Real(2) * t3 - Real(3) * t2 + Real(1)) * v[k] +
         (t3 - Real(2) * t2 + t) * m0 +
         (-Real(2) * t3 + Real(3) * t2) * v[k + 1] +
         (t3 - t2) * m1;
}

Real interp_to_pressure(const std::vector<Real>& p, const std::vector<Real>& v,
                        Real p_target) {
  // 气压随高度单调递减；插值前先翻转为升序，使 interp_to_height 可复用。
  VIBE_CHECK(p.size() == v.size());
  std::vector<Real> pp(p.rbegin(), p.rend());
  std::vector<Real> vv(v.rbegin(), v.rend());
  return interp_to_height(pp, vv, p_target);
}

// ---------------------------------------------------------------------------
// 守恒重映射
// ---------------------------------------------------------------------------

void conservative_restrict(const Field<Real>& src, Field<Real>& dst, Int ratio) {
  VIBE_CHECK(ratio >= 1);
  const Int nx = dst.nx(), ny = dst.ny(), nz = dst.nz();
  const Real w = Real(1) / static_cast<Real>(ratio * ratio);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        Real acc = Real(0);
        for (Int jj = 0; jj < ratio; ++jj)
          for (Int ii = 0; ii < ratio; ++ii)
            acc += src.clamp_at(i * ratio + ii, j * ratio + jj, k);
        dst(i, j, k) = acc * w;
      }
}

void bilinear_prolong(const Field<Real>& src, Field<Real>& dst, Int ratio,
                      bool enforce_positive) {
  const Int nx = dst.nx(), ny = dst.ny(), nz = dst.nz();
  const Real inv = Real(1) / static_cast<Real>(ratio);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real xs = (static_cast<Real>(i) + Real(0.5)) * inv - Real(0.5);
        const Real ys = (static_cast<Real>(j) + Real(0.5)) * inv - Real(0.5);
        const Int i0 = static_cast<Int>(std::floor(xs));
        const Int j0 = static_cast<Int>(std::floor(ys));
        const Real wx = xs - static_cast<Real>(i0);
        const Real wy = ys - static_cast<Real>(j0);
        const Real q = lerp(lerp(src.clamp_at(i0, j0, k), src.clamp_at(i0 + 1, j0, k), wx),
                            lerp(src.clamp_at(i0, j0 + 1, k), src.clamp_at(i0 + 1, j0 + 1, k), wx),
                            wy);
        dst(i, j, k) = (enforce_positive && q < Real(0)) ? Real(0) : q;
      }
}

// ---------------------------------------------------------------------------
// 插值器
// ---------------------------------------------------------------------------

namespace {

class LinearInterpolator final : public Interpolator {
 public:
  Real operator()(const Field<Real>& f, const Grid& g, Real x, Real y, Real z) const override {
    return trilinear(f, g, x, y, z);
  }
  const char* name() const noexcept override { return "trilinear"; }
};

class CubicInterpolator final : public Interpolator {
 public:
  Real operator()(const Field<Real>& f, const Grid& g, Real x, Real y, Real z) const override {
    const Int nz = g.nz();
    Int k0 = 0;
    Real wk = 0;
    for (Int k = 0; k < nz - 1; ++k) {
      const Real zl = g.z_center(0, 0, k);
      const Real zh = g.z_center(0, 0, k + 1);
      if (z >= zl && z <= zh) { k0 = k; wk = (zh > zl) ? (z - zl) / (zh - zl) : Real(0); break; }
      if (k == nz - 2) k0 = k;
    }
    return lerp(bicubic(f, g, x, y, k0), bicubic(f, g, x, y, std::min(k0 + 1, nz - 1)), wk);
  }
  const char* name() const noexcept override { return "bicubic"; }
};

}  // namespace

std::unique_ptr<Interpolator> make_interpolator(InterpOrder order) {
  switch (order) {
    case InterpOrder::Linear: return std::make_unique<LinearInterpolator>();
    case InterpOrder::Cubic:
    case InterpOrder::Quintic: return std::make_unique<CubicInterpolator>();
  }
  return std::make_unique<LinearInterpolator>();
}

}  // namespace vibe::grid

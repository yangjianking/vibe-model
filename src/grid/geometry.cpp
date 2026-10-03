/// @file geometry.cpp
/// @brief 网格几何、度量项与理想地形生成的实现。

#include "vibe/grid/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"

namespace vibe::grid {

namespace {

/// 在分数层索引 kk 处取 zeta 坐标。
///   kk = k      -> 层界面 k
///   kk = k+0.5  -> 层中心 k（严格取上下界面的平均，即使层厚不均匀）
Real zeta_at(const Geometry& g, Real kk) {
  const Int nz = g.nz;
  if (kk <= Real(0)) return g.zeta[0];
  if (kk >= Real(nz)) return g.zeta[static_cast<Size>(nz)];
  const Int k0 = static_cast<Int>(std::floor(kk));
  const Real frac = kk - static_cast<Real>(k0);
  if (frac == Real(0)) return g.zeta[static_cast<Size>(k0)];
  if (std::abs(frac - Real(0.5)) < Real(1e-12)) {
    return Real(0.5) * (g.zeta[static_cast<Size>(k0)] + g.zeta[static_cast<Size>(k0 + 1)]);
  }
  return lerp(g.zeta[static_cast<Size>(k0)], g.zeta[static_cast<Size>(k0 + 1)], frac);
}

}  // namespace

// ---------------------------------------------------------------------------
// 垂直层生成
// ---------------------------------------------------------------------------

std::vector<Real> make_stretched_zeta(Int nz, Real first_thickness, Real stretch) {
  // 目标：层界面 zeta_k in [0,1]，单调递增；层厚按几何级数增长后归一化。
  // 这样在近地层有细网格（捕捉边界层），在平流层有粗网格（节省计算）。
  VIBE_CHECK(nz >= 1);
  std::vector<Real> zeta(static_cast<Size>(nz) + 1);
  std::vector<Real> dz(static_cast<Size>(nz));
  Real total = Real(0);
  Real h = first_thickness;
  for (Int k = 0; k < nz; ++k) {
    dz[static_cast<Size>(k)] = h;
    total += h;
    h *= stretch;
  }
  zeta[0] = Real(0);
  Real acc = Real(0);
  for (Int k = 0; k < nz; ++k) {
    acc += dz[static_cast<Size>(k)];
    zeta[static_cast<Size>(k + 1)] = acc / total;
  }
  zeta[static_cast<Size>(nz)] = Real(1);
  return zeta;
}

std::vector<Real> Geometry::default_zeta_levels(Int nz, Real stretch) {
  return make_stretched_zeta(nz, Real(1.0), stretch > Real(0) ? stretch : Real(1.05));
}

// ---------------------------------------------------------------------------
// 地形
// ---------------------------------------------------------------------------

void smooth_terrain(std::vector<Real>& zs, Int nx, Int ny, int passes, Real max_slope) {
  if (zs.empty() || passes <= 0) return;
  std::vector<Real> tmp(zs.size());
  for (int p = 0; p < passes; ++p) {
    // 1-2-1 平滑（相当于一次 Del^2 滤波），边界保持原值
    for (Int j = 1; j < ny - 1; ++j) {
      for (Int i = 1; i < nx - 1; ++i) {
        const Size c = static_cast<Size>(j * nx + i);
        tmp[c] = Real(0.25) * (zs[c - 1] + zs[c + 1] + zs[c - static_cast<Size>(nx)] +
                               zs[c + static_cast<Size>(nx)]) +
                 Real(0.5) * zs[c];
      }
    }
    for (Int j = 1; j < ny - 1; ++j) {
      for (Int i = 1; i < nx - 1; ++i) {
        const Size c = static_cast<Size>(j * nx + i);
        zs[c] = std::min(zs[c], tmp[c]);
      }
    }
    // 限幅：相邻点高差不超过 max_slope * dx（此处 dx 隐含在调用者提供的 max_slope 中）
    if (max_slope > Real(0)) {
      for (Int j = 1; j < ny - 1; ++j) {
        for (Int i = 1; i < nx - 1; ++i) {
          const Size c = static_cast<Size>(j * nx + i);
          const Real nb = std::max({zs[c - 1], zs[c + 1], zs[c - static_cast<Size>(nx)],
                                    zs[c + static_cast<Size>(nx)]});
          zs[c] = std::min(zs[c], nb + max_slope);
        }
      }
    }
  }
}

std::vector<Real> witch_of_agnesi(Int nx, Int ny, Real dx, Real dy,
                                  Real h0, Real a, Real xc, Real yc) {
  // h(x,y) = h0 * a^2 / (a^2 + d^2),  d^2 = (x-xc)^2 + (y-yc)^2
  // 文献 [D2] Gal-Chen & Somerville (1975)；经典二维山波试验。
  std::vector<Real> zs(static_cast<Size>(nx) * static_cast<Size>(ny), Real(0));
  for (Int j = 0; j < ny; ++j) {
    for (Int i = 0; i < nx; ++i) {
      const Real x = static_cast<Real>(i) * dx - xc;
      const Real y = static_cast<Real>(j) * dy - yc;
      const Real d2 = x * x + y * y;
      zs[static_cast<Size>(j * nx + i)] = h0 * a * a / (a * a + d2);
    }
  }
  return zs;
}

std::vector<Real> schar_mountain(Int nx, Int ny, Real dx, Real dy,
                                 Real h0, Real lambda, Real xc, Real yc) {
  // 光滑余弦山（[D9] Schär et al. 2002）：
  //   h = h0 * cos^2(pi d / (2 lambda))  for |d| <= lambda, else 0
  std::vector<Real> zs(static_cast<Size>(nx) * static_cast<Size>(ny), Real(0));
  for (Int j = 0; j < ny; ++j) {
    for (Int i = 0; i < nx; ++i) {
      const Real x = static_cast<Real>(i) * dx - xc;
      const Real y = static_cast<Real>(j) * dy - yc;
      const Real d = std::sqrt(x * x + y * y);
      if (d < lambda) {
        const Real c = std::cos(kPi * d / (Real(2) * lambda));
        zs[static_cast<Size>(j * nx + i)] = h0 * c * c;
      }
    }
  }
  return zs;
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

Grid::Grid(Geometry g, Decomposition d) : geom_(std::move(g)), dec_(std::move(d)) {
  VIBE_CHECK(geom_.valid());
  rebuild_metrics();
}

void Grid::rebuild_metrics() {
  const Int nxl = dec_.nx_local();
  const Int nyl = dec_.ny_local();
  const Int nz = geom_.nz;

  // 本地地形切片（若全局地形存在）
  zs_local_.assign(static_cast<Size>(nxl) * static_cast<Size>(nyl), Real(0));
  if (!geom_.zs.empty()) {
    for (Int j = 0; j < nyl; ++j) {
      const Int gj = dec_.js + j;
      for (Int i = 0; i < nxl; ++i) {
        const Int gi = dec_.is + i;
        zs_local_[static_cast<Size>(j * nxl + i)] =
            geom_.zs[static_cast<Size>(gj * geom_.nx + gi)];
      }
    }
    geom_.flat_terrain = false;
  }

  // 层中心 zeta 与 1/dz
  zc_.assign(static_cast<Size>(nz), Real(0));
  dzi_.assign(static_cast<Size>(nz), Real(0));
  for (Int k = 0; k < nz; ++k) {
    const Real zlo = geom_.zeta[static_cast<Size>(k)];
    const Real zhi = geom_.zeta[static_cast<Size>(k + 1)];
    zc_[static_cast<Size>(k)] = Real(0.5) * (zlo + zhi);
    const Real dz = (zhi - zlo) * geom_.z_top;
    dzi_[static_cast<Size>(k)] = dz > Real(0) ? Real(1) / dz : Real(0);
  }
}

Int Grid::size(Stagger s) const noexcept {
  const Int nxl = dec_.nx_local();
  const Int nyl = dec_.ny_local();
  const Int nz = geom_.nz;
  switch (s) {
    case Stagger::FaceX: return (nxl + 1) * nyl * nz;
    case Stagger::FaceY: return nxl * (nyl + 1) * nz;
    case Stagger::FaceZ: return nxl * nyl * (nz + 1);
    default: return nxl * nyl * nz;
  }
}

Int Grid::storage_size(Stagger s) const noexcept {
  const Int h = dec_.halo;
  Int nsx = dec_.nx_local() + 2 * h;
  Int nsy = dec_.ny_local() + 2 * h;
  Int nsz = geom_.nz + 2 * h;
  if (s == Stagger::FaceX) nsx += 1;
  if (s == Stagger::FaceY) nsy += 1;
  if (s == Stagger::FaceZ) nsz += 1;
  return nsx * nsy * nsz;
}

Index3 Grid::origin(Stagger s) const noexcept {
  Index3 o{dec_.halo, dec_.halo, dec_.halo};
  VIBE_UNUSED(s);
  return o;
}

Index3 Grid::store_index(Stagger s, Int i, Int j, Int k) const noexcept {
  VIBE_UNUSED(s);
  return {i + dec_.halo, j + dec_.halo, k + dec_.halo};
}

Index Grid::flatten(Stagger s, Int i, Int j, Int k) const noexcept {
  Int nsx = dec_.nx_local() + 2 * dec_.halo;
  Int nsy = dec_.ny_local() + 2 * dec_.halo;
  if (s == Stagger::FaceX) nsx += 1;
  if (s == Stagger::FaceY) nsy += 1;
  const Int ii = i + dec_.halo;
  const Int jj = j + dec_.halo;
  const Int kk = k + dec_.halo;
  return static_cast<Index>((kk * nsy + jj) * nsx + ii);
}

bool Grid::interior(Stagger s, Int i, Int j, Int k) const noexcept {
  const Int imax = (s == Stagger::FaceX) ? dec_.nx_local() : dec_.nx_local() - 1;
  const Int jmax = (s == Stagger::FaceY) ? dec_.ny_local() : dec_.ny_local() - 1;
  const Int kmax = (s == Stagger::FaceZ) ? geom_.nz : geom_.nz - 1;
  return i >= 0 && i <= imax && j >= 0 && j <= jmax && k >= 0 && k <= kmax;
}

Real Grid::dx_at(Int i) const noexcept {
  if (geom_.dx_cell.empty()) return geom_.dx;
  const Int gi = dec_.is + i;
  return geom_.dx_cell[static_cast<Size>(clamp(gi, Int(0), geom_.nx - 1))];
}

Real Grid::dy_at(Int j) const noexcept {
  if (geom_.dy_cell.empty()) return geom_.dy;
  const Int gj = dec_.js + j;
  return geom_.dy_cell[static_cast<Size>(clamp(gj, Int(0), geom_.ny - 1))];
}

Real Grid::dzeta(Int k) const noexcept {
  const Int kk = clamp(k, Int(0), geom_.nz - 1);
  return geom_.zeta[static_cast<Size>(kk + 1)] - geom_.zeta[static_cast<Size>(kk)];
}

Real Grid::terrain(Int i, Int j) const noexcept {
  const Int nxl = dec_.nx_local();
  const Int nyl = dec_.ny_local();
  const Int ii = clamp(i, Int(0), nxl - 1);
  const Int jj = clamp(j, Int(0), nyl - 1);
  return zs_local_[static_cast<Size>(jj * nxl + ii)];
}

Real Grid::jacobian(Int i, Int j) const noexcept {
  // G^{1/2}(x,y) = (z_top - z_s(x,y)) / z_top，无量纲（与 z_top 相除后用于体积元）
  return (geom_.z_top - terrain(i, j)) / geom_.z_top;
}

Real Grid::height(Int i, Int j, Real kk) const noexcept {
  const Real zs = terrain(i, j);
  return zs + zeta_at(geom_, kk) * (geom_.z_top - zs);
}

Real Grid::inv_dz(Int i, Int j, Int k) const noexcept {
  // 层中心的 1/dz：由上下界面高度得到 dz = (zeta_{k+1}-zeta_k)*(z_top - z_s)
  const Real scale = geom_.z_top - terrain(i, j);
  const Real dz = dzeta(k) * scale;
  return dz > Real(0) ? Real(1) / dz : Real(0);
}

std::string Grid::describe() const {
  std::ostringstream os;
  os << "格点 " << geom_.name << ": 全局 " << geom_.nx << "x" << geom_.ny << "x" << geom_.nz
     << ", 本地 " << dec_.nx_local() << "x" << dec_.ny_local() << "x" << geom_.nz
     << ", 分解 " << dec_.px << "x" << dec_.py << " (rank " << dec_.rank << ")";
  if (geom_.flat_terrain) {
    os << ", 平坦地形";
  } else {
    Real zmin = kHuge, zmax = -kHuge;
    for (Real z : zs_local_) {
      zmin = std::min(zmin, z);
      zmax = std::max(zmax, z);
    }
    os << ", 地形 " << zmin << ".." << zmax << " m";
  }
  if (geom_.variable_resolution) {
    os << ", 变分辨率 h in [" << geom_.dx << ", ...]";
  }
  os << ", dx=" << geom_.dx << " dy=" << geom_.dy << " z_top=" << geom_.z_top;
  return os.str();
}

}  // namespace vibe::grid

/// @file coriolis.cpp
/// @brief 科氏项的 C-grid 离散。
///
/// 离散（[D3] Arakawa & Lamb 1977 §4）
/// -----------------------------------
/// u 点上的 v 通过周围四个 v 点平均：
///
///     v|_u(i,j,k) = 1/4 [ v(i-1,j-1,k) + v(i,j-1,k) + v(i-1,j,k) + v(i,j,k) ]
///     (du/dt)_cor = + f * v|_u
///
/// v 点上的 u 同理：
///
///     u|_v(i,j,k) = 1/4 [ u(i-1,j,k) + u(i,j,k) + u(i-1,j+1,k) + u(i,j+1,k) ]
///     (dv/dt)_cor = - f * u|_v
///
/// 该离散在 f 为常数时严格保持离散动能（反对称性），且插值与其伴随互为转置。

#include "vibe/dyn/coriolis.hpp"

#include <algorithm>
#include <cmath>

namespace vibe::dyn {

Coriolis::Coriolis(const grid::Grid& g, CoriolisMode mode, Real latitude_deg,
                   Real u_ref)
    : grid_(&g), mode_(mode), lat0_(latitude_deg) {
  VIBE_UNUSED(u_ref);
  f0_ = Real(2) * kOmega * std::sin(latitude_deg * kDegToRad);
  // beta = 2 Omega cos(phi) / a
  beta_ = Real(2) * kOmega * std::cos(latitude_deg * kDegToRad) / kEarthRadius;
}

Real Coriolis::f_at(Int j) const {
  if (mode_ != CoriolisMode::BetaPlane) return f0_;
  // beta 平面上 f = f0 + beta * (y - y0)，y0 取全局域的 y 中心。
  const Real y = (static_cast<Real>(grid_->decomp().js) + static_cast<Real>(j) + Real(0.5)) *
                 grid_->dy_at(j);
  const Real y0 = Real(0.5) * static_cast<Real>(grid_->ny_global()) * grid_->geom().dy;
  return f0_ + beta_ * (y - y0);
}

void Coriolis::apply(const State& s, Tendency& d) const {
  if (mode_ == CoriolisMode::None) return;
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  grid::Field<Real>& du = d.u();
  grid::Field<Real>& dv = d.v();
  const grid::Field<Real>& u = s.u();
  const grid::Field<Real>& v = s.v();

  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Real v_at_u = Real(0.25) * (v(std::max(i - 1, 0), std::max(j - 1, 0), k) +
                                          v(std::min(i, nx - 1), std::max(j - 1, 0), k) +
                                          v(std::max(i - 1, 0), j, k) +
                                          v(std::min(i, nx - 1), j, k));
        du(i, j, k) += f_at(j) * v_at_u;
      }

  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j <= ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real u_at_v = Real(0.25) * (u(std::max(i - 1, 0), std::max(j - 1, 0), k) +
                                          u(i, std::max(j - 1, 0), k) +
                                          u(std::max(i - 1, 0), std::min(j, ny - 1), k) +
                                          u(i, std::min(j, ny - 1), k));
        dv(i, j, k) -= f_at(j) * u_at_v;
      }
}

}  // namespace vibe::dyn

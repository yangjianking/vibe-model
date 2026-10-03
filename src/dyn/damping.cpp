/// @file damping.cpp
/// @brief 海绵层与散度阻尼的实现。
///
/// 上层海绵层（[D1] 第 4 节）
/// -------------------------
///     alpha(z) = alpha_max * sin^2( pi/2 * (z - z_s)/(z_top - z_s) )
/// 后向欧拉更新（无条件稳定）：
///     phi <- (phi + dt * alpha * phi_ref) / (1 + dt * alpha)
/// 其中 u, v, w 的参考值取 0；theta'、pi' 取 0（即松弛到参考态）。
///
/// 散度阻尼（[D6]）
/// ---------------
///     F_damp = -nu * Del^{2p}(phi)
/// 用 2 阶中心差分的反复应用实现 Del^2 与 Del^4。
/// 只作用于质量场/Exner 场，抑制声波与短波噪声。

#include "vibe/dyn/damping.hpp"

#include <algorithm>
#include <cmath>

namespace vibe::dyn {

Damping::Damping(const grid::Grid& g, Real z_sponge_start, Real z_top, Real alpha_max,
                 int divergence_order, Real divergence_coeff)
    : grid_(&g), z_sponge_start_(z_sponge_start), z_top_(z_top),
      alpha_max_(alpha_max), divergence_order_(divergence_order),
      divergence_coeff_(divergence_coeff) {
  const Int nz = g.nz();
  sponge_.assign(static_cast<Size>(nz), Real(0));
  for (Int k = 0; k < nz; ++k) {
    const Real z = g.z_center(0, 0, k);
    if (z <= z_sponge_start_) {
      sponge_[static_cast<Size>(k)] = Real(0);
      continue;
    }
    const Real t = (z - z_sponge_start_) / std::max(z_top_ - z_sponge_start_, Real(1));
    const Real s = std::sin(Real(0.5) * kPi * clamp(t, Real(0), Real(1)));
    sponge_[static_cast<Size>(k)] = alpha_max_ * s * s;
  }
}

void Damping::tendencies(const State& s, Tendency& d) const {
  if (divergence_coeff_ <= Real(0)) return;
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();

  // Del^2 滤波（2 阶）或 Del^2(Del^2)（4 阶）
  auto laplacian2 = [&](const grid::Field<Real>& f, Int i, Int j, Int k) {
    const Real dx = grid_->dx_at(i);
    const Real dy = grid_->dy_at(j);
    return (f.clamp_at(i + 1, j, k) - Real(2) * f(i, j, k) + f.clamp_at(i - 1, j, k)) / (dx * dx) +
           (f.clamp_at(i, j + 1, k) - Real(2) * f(i, j, k) + f.clamp_at(i, j - 1, k)) / (dy * dy);
  };

  grid::Field<Real>& dpi = d.pi();
  const grid::Field<Real>& pi = s.pi();

  if (divergence_order_ <= 2) {
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i)
          dpi(i, j, k) -= divergence_coeff_ * laplacian2(pi, i, j, k);
  } else {
    // Del^4：对 Del^2 结果再做一次 Del^2
    grid::Field<Real> tmp(*grid_, grid::Stagger::Cell, "lap");
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) tmp(i, j, k) = laplacian2(pi, i, j, k);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i)
          dpi(i, j, k) -= divergence_coeff_ * laplacian2(tmp, i, j, k);
  }
}

void Damping::apply_sponge(State& s, const ReferenceState& ref, Real dt) const {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();

  // 找到海绵层起始层
  Int kmin = nz;
  for (Int k = 0; k < nz; ++k) {
    if (sponge_[static_cast<Size>(k)] > Real(0)) { kmin = k; break; }
  }
  if (kmin >= nz) return;

  // 通用后向欧拉松弛：phi <- (phi + dt*alpha*phi_ref) / (1 + dt*alpha)
  // 动量与 pi' 的目标值为 0；rho 与 theta 的目标值为参考态（扰动形式下也为 0）。
  auto alpha_at = [&](Int k) { return sponge_[static_cast<Size>(std::min(k, nz - 1))]; };

  auto relax_cell = [&](grid::Field<Real>& f, const grid::Field<Real>* target) {
    for (Int k = kmin; k < nz; ++k) {
      const Real a = alpha_at(k);
      if (a <= Real(0)) continue;
      const Real w = Real(1) / (Real(1) + dt * a);
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          const Real tgt = (target != nullptr) ? (*target)(i, j, k) : Real(0);
          f(i, j, k) = w * (f(i, j, k) + dt * a * tgt);
        }
    }
  };

  auto relax_facex = [&](grid::Field<Real>& f) {
    for (Int k = kmin; k < nz; ++k) {
      const Real a = alpha_at(k);
      if (a <= Real(0)) continue;
      const Real w = Real(1) / (Real(1) + dt * a);
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i <= nx; ++i) f(i, j, k) = w * f(i, j, k);
    }
  };
  auto relax_facey = [&](grid::Field<Real>& f) {
    for (Int k = kmin; k < nz; ++k) {
      const Real a = alpha_at(k);
      if (a <= Real(0)) continue;
      const Real w = Real(1) / (Real(1) + dt * a);
      for (Int j = 0; j <= ny; ++j)
        for (Int i = 0; i < nx; ++i) f(i, j, k) = w * f(i, j, k);
    }
  };
  auto relax_facez = [&](grid::Field<Real>& f) {
    for (Int k = kmin; k <= nz; ++k) {
      const Real a = alpha_at(k);
      if (a <= Real(0)) continue;
      const Real w = Real(1) / (Real(1) + dt * a);
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) f(i, j, k) = w * f(i, j, k);
    }
  };

  relax_facex(s.u());
  relax_facey(s.v());
  relax_facez(s.w());
  // 质量与热力学场松弛到参考态
  relax_cell(s.rho(), &ref.rho0());
  relax_cell(s.theta(), &ref.theta0());
  relax_cell(s.pi(), &ref.pi0());
}

}  // namespace vibe::dyn

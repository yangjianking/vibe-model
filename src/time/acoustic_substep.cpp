/// @file acoustic_substep.cpp
/// @brief 声波子步：前向-后向格式与 HEVI 垂直隐式。
///
/// 前向-后向（[D1] 第 3 节）
/// ------------------------
///     1) 由 u^n 更新 rho^{n+1}, pi'^{n+1}
///     2) 由 pi'^{n+1} 更新 u^{n+1}
/// 该格式对声波等价于蛙跳，稳定性条件为
///
///     dtau <= 2 / ( c_s sqrt(1/dx^2 + 1/dy^2 + 1/dz^2) )
///
/// 实践取 c_s 的域最大值，并把安全系数取 0.8。
///
/// HEVI（[T2] Tapp & White 1976；[T6] Cullen 1990）
/// -----------------------------------------------
/// 垂直声波项用 Crank-Nicolson，得到三对角系统
///
///     -A_k x_{k-1} + B_k x_k - C_k x_{k+1} = f_k
///     A_k = (dtau/2)^2 cp theta0_k rho0_k / dz_k^2 （k 与 k-1 之间取平均）
///     B_k = 1/(rho0 c_s^2)_k + A_k + A_{k+1}
///
/// 由于系数只依赖参考态与 dtau，可以在整个时间步中复用分解。

#include "vibe/time/acoustic_substep.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/time/helmholtz.hpp"

namespace vibe::timeint {

AcousticSubstepper::AcousticSubstepper(const grid::Grid& g, const AcousticOptions& opt)
    : grid_(&g), opt_(opt) {}

void AcousticSubstepper::update_coefficients(const dyn::ReferenceState& ref, Real dtau) {
  const Int nz = grid_->nz();
  tri_a_.assign(static_cast<Size>(std::max(nz - 1, 0)), Real(0));
  tri_b_.assign(static_cast<Size>(nz), Real(0));
  tri_c_.assign(static_cast<Size>(std::max(nz - 1, 0)), Real(0));
  tri_rhs_.assign(static_cast<Size>(nz), Real(0));
  tri_work_.assign(static_cast<Size>(nz), Real(0));

  const Real half = Real(0.5) * dtau;
  for (Int k = 0; k < nz; ++k) {
    // 用参考态在 (0,0) 列上的值构造（水平均匀参考态下各处相同）
    const Real scale = grid_->geom().z_top - grid_->terrain(0, 0);
    const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
    const Real rho0 = ref.rho0()(0, 0, k);
    const Real th0 = ref.theta0()(0, 0, k);
    const Real cs2 = ref.sound_speed2(0, 0, k);
    const Real a_k = half * half * kCp * th0 * rho0 / (dz * dz);
    const Real a_kp1 = (k < nz - 1)
                           ? half * half * kCp * ref.theta0()(0, 0, k + 1) *
                                 ref.rho0()(0, 0, k + 1) / (dz * dz)
                           : a_k;
    const Real b0 = Real(1) / std::max(rho0 * cs2, Real(1e-12));
    tri_b_[static_cast<Size>(k)] = b0 + a_k + a_kp1;
    if (k < nz - 1) tri_c_[static_cast<Size>(k)] = -a_kp1;
    if (k > 0) tri_a_[static_cast<Size>(k - 1)] = -a_k;
  }
  VIBE_UNUSED(opt_);
}

void AcousticSubstepper::update_mass_and_pressure(dyn::State& s, Real dtau,
                                                  const dyn::ReferenceState& ref) {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  grid::Field<Real> drho(s.grid(), grid::Stagger::Cell, "drho");
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        const Real r0 = ref.rho0()(i, j, k);
        const Real fx = s.u()(i + 1, j, k) * Real(0.5) * (r0 + ref.rho0()(std::min(i + 1, nx - 1), j, k)) -
                        s.u()(i, j, k) * Real(0.5) * (ref.rho0()(std::max(i - 1, 0), j, k) + r0);
        const Real fy = s.v()(i, j + 1, k) * Real(0.5) * (r0 + ref.rho0()(i, std::min(j + 1, ny - 1), k)) -
                        s.v()(i, j, k) * Real(0.5) * (ref.rho0()(i, std::max(j - 1, 0), k) + r0);
        const Real fz = s.w()(i, j, k + 1) * Real(0.5) * (r0 + ref.rho0()(i, j, std::min(k + 1, nz - 1))) -
                        s.w()(i, j, k) * Real(0.5) * (ref.rho0()(i, j, std::max(k - 1, 0)) + r0);
        drho(i, j, k) = -(fx / dx + fy / dy + fz / dz);
      }

  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real r0 = std::max(ref.rho0()(i, j, k), Real(1e-8));
        s.rho()(i, j, k) += dtau * drho(i, j, k);
        s.pi()(i, j, k) += dtau * (kRd / kCv) * (ref.pi0()(i, j, k) / r0) * drho(i, j, k);
      }
}

void AcousticSubstepper::update_momentum(dyn::State& s, Real dtau,
                                         const dyn::ReferenceState& ref) {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Int ip = std::min(i, nx - 1), im = std::max(i - 1, 0);
        const Real dx = grid_->dx_at(ip);
        const Real th0 = Real(0.5) * (ref.theta0()(im, j, k) + ref.theta0()(ip, j, k));
        s.u()(i, j, k) -= dtau * kCp * th0 * (s.pi()(ip, j, k) - s.pi()(im, j, k)) / dx;
      }
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j <= ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int jp = std::min(j, ny - 1), jm = std::max(j - 1, 0);
        const Real dy = grid_->dy_at(jp);
        const Real th0 = Real(0.5) * (ref.theta0()(i, jm, k) + ref.theta0()(i, jp, k));
        s.v()(i, j, k) -= dtau * kCp * th0 * (s.pi()(i, jp, k) - s.pi()(i, jm, k)) / dy;
      }
  for (Int k = 0; k <= nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int kc = std::min(std::max(k, 0), nz - 1), km = std::max(k - 1, 0);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(kc) * scale, Real(1));
        const Real th0 = Real(0.5) * (ref.theta0()(i, j, km) + ref.theta0()(i, j, kc));
        s.w()(i, j, k) -= dtau * kCp * th0 * (s.pi()(i, j, kc) - s.pi()(i, j, km)) / dz;
      }
}

void AcousticSubstepper::solve_vertical_implicit(dyn::State& s, Real dtau,
                                                  const dyn::ReferenceState& ref) {
  update_coefficients(ref, dtau);
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  std::vector<Real> rhs(static_cast<Size>(nz)), sol(static_cast<Size>(nz));
  for (Int j = 0; j < ny; ++j)
    for (Int i = 0; i < nx; ++i) {
      for (Int k = 0; k < nz; ++k) rhs[static_cast<Size>(k)] = s.pi()(i, j, k);
      thomas_solve(tri_a_, tri_b_, tri_c_, rhs, sol);
      for (Int k = 0; k < nz; ++k) s.pi()(i, j, k) = sol[static_cast<Size>(k)];
    }
}

void AcousticSubstepper::step(dyn::State& s, Real dtau, const dyn::ReferenceState& ref) {
  if (opt_.vertical_implicit) {
    // HEVI：质量/Exner 前向，随后垂直方向隐式平滑，最后动量后向
    update_mass_and_pressure(s, dtau, ref);
    solve_vertical_implicit(s, dtau, ref);
    update_momentum(s, dtau, ref);
  } else {
    update_mass_and_pressure(s, dtau, ref);
    update_momentum(s, dtau, ref);
  }
}

Real AcousticSubstepper::acoustic_cfl(const dyn::State& s, Real dtau) const {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  Real worst = Real(0);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        const Real t = s.theta()(i, j, k) * s.pi()(i, j, k);
        const Real cs = std::sqrt((kCp / kCv) * kRd * std::max(t, Real(1)));
        const Real uu = std::abs(Real(0.5) * (s.u()(i, j, k) + s.u()(i + 1, j, k)));
        const Real vv = std::abs(Real(0.5) * (s.v()(i, j, k) + s.v()(i, j + 1, k)));
        const Real ww = std::abs(Real(0.5) * (s.w()(i, j, k) + s.w()(i, j, k + 1)));
        worst = std::max(worst, dtau * ((cs + uu) / dx + (cs + vv) / dy + (cs + ww) / dz));
      }
  return worst;
}

int AcousticSubstepper::required_substeps(const dyn::State& s, Real dt, Real safety) const {
  const Real cfl = acoustic_cfl(s, dt);
  if (cfl <= safety || cfl <= Real(0)) return 1;
  return static_cast<int>(std::ceil(cfl / safety));
}

}  // namespace vibe::timeint

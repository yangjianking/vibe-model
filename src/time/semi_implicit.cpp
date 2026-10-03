/// @file semi_implicit.cpp
/// @brief 半隐式积分器：Crank–Nicolson 隐式声波-重力波 + 3D Helmholtz。
///
/// 推导（[T2][T6][T7]）
/// -------------------
/// 线性声波-重力波子系统（参考态水平均匀）：
///
///     du/dt = -cp theta0 grad(pi') + N_u
///     dpi'/dt = -c * div(rho0 u),     c = (Rd/cv) (pi0/rho0)
///     d rho'/dt = -div(rho0 u)
///
/// 对线性项用 Crank-Nicolson（时间中点），非线性项 N 用二阶外推
/// （第一步用前向欧拉，避免启动问题）：
///
///     u^{n+1} = u^n + dt N_u - (dt/2) cp theta0 grad(pi'^{n+1} + pi'^n)
///     pi'^{n+1} = pi'^n - dt c G + (dt/2)^2 c div( rho0 cp theta0 grad(pi'^{n+1}+pi'^n) )
///
/// 其中 G = div(rho0 u^n) + (dt/2) div(rho0 N_u)。
/// 整理后得到关于 pi'^{n+1} 的 Helmholtz 方程：
///
///     pi'^{n+1} - div( a grad pi'^{n+1} ) = pi'^n - dt c G + div( a grad pi'^n ),
///     a = (dt/2)^2 (Rd/cv) pi0 cp theta0
///
/// 用 -div(a grad x) + b x = f（b = 1）的形式交给 HelmholtzSolver。
/// 求解后按上式回写 u、rho，并对超出稳定域的短波做可选的时间滤波
/// （Robert-Asselin-Williams，[T3]）。
///
/// 文献：[T1][T2][T3][T6][T7][T8][T14]。

#include "vibe/time/semi_implicit.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/timer.hpp"
#include "vibe/config/config.hpp"

namespace vibe::timeint {

SemiImplicitIntegrator::SemiImplicitIntegrator(const grid::Grid& g,
                                               const dyn::ReferenceState& ref,
                                               const config::ModelConfig& cfg,
                                               dyn::Equations& eq,
                                               std::unique_ptr<HelmholtzSolver> solver)
    : grid_(&g), ref_(&ref), eq_(&eq), solver_(std::move(solver)),
      alpha_coef_(g, grid::Stagger::Cell, "a_coef"),
      beta_coef_(g, grid::Stagger::Cell, "b_coef"),
      rhs_(g, grid::Stagger::Cell, "rhs"),
      pi_new_(g, grid::Stagger::Cell, "pi_new"),
      pi_prev_(g, grid::Stagger::Cell, "pi_prev"),
      n_prev_(g), n_cur_(g), scratch_(g) {
  VIBE_CHECK(solver_ != nullptr);
  outer_ = std::max(cfg.numerics.helmholtz_max_iter > 0 ? 1 : 1, 1);
}

void SemiImplicitIntegrator::assemble_rhs(const dyn::State& s_prev, const dyn::State& s_cur,
                                          const dyn::Tendency& n_prev,
                                          const dyn::Tendency& n_cur, Real dt,
                                          grid::Field<Real>& rhs) {
  VIBE_UNUSED(n_prev);
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  const Real half_dt = Real(0.5) * dt;

  // a = (dt/2)^2 (Rd/cv) pi0 cp theta0 ；b = 1
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        alpha_coef_(i, j, k) = half_dt * half_dt * (kRd / kCv) * ref_->pi0()(i, j, k) * kCp *
                               ref_->theta0()(i, j, k);
        beta_coef_(i, j, k) = Real(1);
      }

  // G = div(rho0 u^n) + (dt/2) div(rho0 N_u)
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));

        auto flux_div = [&](const grid::Field<Real>& u, const grid::Field<Real>& v,
                            const grid::Field<Real>& w) {
          const Real fx = u(i + 1, j, k) * Real(0.5) * (ref_->rho0()(i, j, k) + ref_->rho0()(std::min(i + 1, nx - 1), j, k)) -
                          u(i, j, k) * Real(0.5) * (ref_->rho0()(std::max(i - 1, 0), j, k) + ref_->rho0()(i, j, k));
          const Real fy = v(i, j + 1, k) * Real(0.5) * (ref_->rho0()(i, j, k) + ref_->rho0()(i, std::min(j + 1, ny - 1), k)) -
                          v(i, j, k) * Real(0.5) * (ref_->rho0()(i, std::max(j - 1, 0), k) + ref_->rho0()(i, j, k));
          const Real fz = w(i, j, k + 1) * Real(0.5) * (ref_->rho0()(i, j, k) + ref_->rho0()(i, j, std::min(k + 1, nz - 1))) -
                          w(i, j, k) * Real(0.5) * (ref_->rho0()(i, j, std::max(k - 1, 0)) + ref_->rho0()(i, j, k));
          return fx / dx + fy / dy + fz / dz;
        };

        const Real div_un = flux_div(s_prev.u(), s_prev.v(), s_prev.w());
        const Real div_nn = flux_div(n_cur.u(), n_cur.v(), n_cur.w());
        const Real G = div_un + half_dt * div_nn;
        const Real c = (kRd / kCv) * (ref_->pi0()(i, j, k) / std::max(ref_->rho0()(i, j, k), Real(1e-8)));
        // RHS = pi'^n - dt c G + div(a grad pi'^n)
        const Real dxa = grid_->dx_at(i), dya = grid_->dy_at(j);
        const Real a0 = alpha_coef_(i, j, k);
        const Real div_a_grad =
            (a0 * (s_prev.pi()(std::min(i + 1, nx - 1), j, k) - s_prev.pi()(i, j, k)) -
             a0 * (s_prev.pi()(i, j, k) - s_prev.pi()(std::max(i - 1, 0), j, k))) / (dxa * dxa) +
            (a0 * (s_prev.pi()(i, std::min(j + 1, ny - 1), k) - s_prev.pi()(i, j, k)) -
             a0 * (s_prev.pi()(i, j, k) - s_prev.pi()(i, std::max(j - 1, 0), k))) / (dya * dya);
        rhs(i, j, k) = s_prev.pi()(i, j, k) - dt * c * G + div_a_grad;
      }
}

void SemiImplicitIntegrator::update_from_pressure(dyn::State& s, const grid::Field<Real>& pi_new,
                                                  Real dt) {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  const Real half_dt = Real(0.5) * dt;

  // pi' 的时间平均（Crank-Nicolson 的 (pi^{n+1}+pi^n)/2 部分由调用者通过 pi_prev_ 提供）
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dpi = Real(0.5) * (pi_new(i, j, k) - pi_prev_(i, j, k));
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        // u^{n+1} = u* - (dt/2) cp theta0 grad(pi'^{n+1} - pi'^n)
        s.u()(i, j, k) -= half_dt * kCp *
                          Real(0.5) * (ref_->theta0()(i, j, k) + ref_->theta0()(std::max(i - 1, 0), j, k)) *
                          (dpi) / dx;
        s.v()(i, j, k) -= half_dt * kCp *
                          Real(0.5) * (ref_->theta0()(i, j, k) + ref_->theta0()(i, std::max(j - 1, 0), k)) *
                          (dpi) / dy;
        s.w()(i, j, k) -= half_dt * kCp * ref_->theta0()(i, j, k) * (dpi) / dz;
      }
  // 完成 u 面右侧的更新（x 方向）
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 1; i <= nx; ++i) {
        const Real dx = grid_->dx_at(std::min(i, nx - 1));
        const Real dpi = Real(0.5) * (pi_new(i, j, k) - pi_prev_(i, j, k));
        const Real th0 = Real(0.5) * (ref_->theta0()(std::min(i, nx - 1), j, k) +
                                      ref_->theta0()(std::max(i - 1, 0), j, k));
        s.u()(i, j, k) -= half_dt * kCp * th0 * dpi / dx;
      }
  for (Int k = 0; k < nz; ++k)
    for (Int j = 1; j <= ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dy = grid_->dy_at(std::min(j, ny - 1));
        const Real dpi = Real(0.5) * (pi_new(i, j, k) - pi_prev_(i, j, k));
        const Real th0 = Real(0.5) * (ref_->theta0()(i, std::min(j, ny - 1), k) +
                                      ref_->theta0()(i, std::max(j - 1, 0), k));
        s.v()(i, j, k) -= half_dt * kCp * th0 * dpi / dy;
      }
  for (Int k = 1; k <= nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int kc = std::min(k, nz - 1);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(kc) * scale, Real(1));
        const Real dpi = Real(0.5) * (pi_new(i, j, k) - pi_prev_(i, j, k));
        s.w()(i, j, k) -= half_dt * kCp * ref_->theta0()(i, j, kc) * dpi / dz;
      }

  // 回写 pi'
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) s.pi()(i, j, k) = pi_new(i, j, k);
}

void SemiImplicitIntegrator::step(dyn::State& s, const StepContext& ctx) {
  common::ScopedTimer timer("semi_implicit_step");
  const Real dt = ctx.dt;

  pi_prev_ = s.pi();   // pi'^n

  // 非线性项：二阶 Adams-Bashforth（首步用前向欧拉）
  eq_->tendencies(s, n_cur_);
  if (physics_ != nullptr) {
    dyn::PhysicsTendency pt(*grid_);
    physics_->step(s, *grid_, *ref_, pt, dt);
    n_cur_.theta().add_scaled(Real(1), pt.theta);
    n_cur_.qv().add_scaled(Real(1), pt.qv);
    n_cur_.u().add_scaled(Real(1), pt.u);
    n_cur_.v().add_scaled(Real(1), pt.v);
  }

  // 显式预报量（用于构造 G 中的 div(rho0 N_u) 与最后的状态推进）
  dyn::State s_explicit = s.clone();
  s_explicit.add_scaled(dt, n_cur_);

  // ---- 组装并求解 Helmholtz ----
  assemble_rhs(s, s_explicit, n_prev_, n_cur_, dt, rhs_);
  solver_->set_coefficients(alpha_coef_, beta_coef_);
  solver_->solve(alpha_coef_, beta_coef_, rhs_, pi_new_);
  stats_.helmholtz_iterations += solver_->iterations();
  stats_.helmholtz_residual = solver_->residual();

  // ---- 由 pi'^{n+1} 回写状态 ----
  update_from_pressure(s_explicit, pi_new_, dt);

  // ---- 标量（theta、水物质、rho）用显式更新 ----
  s.theta().add_scaled(dt, n_cur_.theta());
  s.qv().add_scaled(dt, n_cur_.qv());
  s.rho().add_scaled(dt, n_cur_.rho());
  for (int sp = static_cast<int>(dyn::Species::Qc); sp <= static_cast<int>(dyn::Species::Qg); ++sp) {
    const dyn::Species species = static_cast<dyn::Species>(sp);
    s.field(species).add_scaled(dt, n_cur_.field(species));
  }

  // Robert-Asselin-Williams 时间滤波（可选）
  if (ra_filter_ > Real(0) && !first_step_) {
    const Real a = ra_filter_;
    s.pi().axpy(Real(1) - a, s.pi(), a);
  }

  eq_->damping().apply_sponge(s, *ref_, dt);
  if (boundary_) boundary_(s, ctx);

  n_prev_ = n_cur_;
  first_step_ = false;
  stats_.steps += 1;
  stats_.wall_time = timer.elapsed();
  s.time = ctx.time + dt;

  if (s.has_nonfinite()) {
    stats_.diverged = true;
    throw NumericalError("半隐式积分出现非有限值");
  }
}

Real SemiImplicitIntegrator::max_stable_dt(const dyn::State& s) const {
  const Real adv_cfl = eq_->advection().cfl_number(s, Real(1));
  if (adv_cfl <= Real(0)) return Real(600);
  // 半隐式只受平流 CFL 约束；四阶平流取 0.8，二阶取 0.4
  return Real(0.6) / adv_cfl;
}

}  // namespace vibe::timeint

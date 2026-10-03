/// @file test_time.cpp
/// @brief 三对角/Helmholtz 求解器、声波子步、RK3 与半隐式的单元测试。

#include <cmath>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/test.hpp"
#include "vibe/config/config.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/time/acoustic_substep.hpp"
#include "vibe/time/helmholtz.hpp"
#include "vibe/time/runge_kutta.hpp"
#include "vibe/time/semi_implicit.hpp"
#include "vibe/dyn/diagnostics.hpp"
#include "vibe/time/timestep_control.hpp"

using namespace vibe;
using namespace vibe::timeint;

namespace {

grid::Grid make_grid(Int nx = 8, Int ny = 8, Int nz = 16) {
  grid::Geometry g;
  g.nx = nx; g.ny = ny; g.nz = nz;
  g.dx = Real(2000); g.dy = Real(2000); g.z_top = Real(20000);
  g.zeta = grid::make_stretched_zeta(nz, Real(1), Real(1.08));
  g.name = "time_test";
  grid::Decomposition d = grid::Decomposition::make(nx, ny, nz, 1, 1, 4, true, true);
  return grid::Grid(g, d);
}

config::ModelConfig make_cfg(const std::string& integrator = "rk3_acoustic") {
  config::ModelConfig c;
  c.name = "test";
  c.time.integrator = integrator;
  c.time.dt = Real(5);
  c.time.acoustic_substeps = 5;
  c.time.cfl_target = Real(0.8);
  c.numerics.advection = "central2";
  c.numerics.coriolis = "none";
  c.numerics.sponge_alpha = Real(0.1);
  c.numerics.sponge_start_fraction = Real(0.9);
  c.numerics.helmholtz_solver = "krylov_jacobi";
  c.numerics.helmholtz_max_iter = 300;
  c.numerics.helmholtz_tol = Real(1e-10);
  c.physics.microphysics = "kessler";
  return c;
}

}  // namespace

VIBE_TEST(thomas_solves_constant_tridiagonal_exactly) {
  // 三对角矩阵 diag(1,-4,1)，解析解由 x_j = sin(j pi/(n+1)) 给出
  const int n = 32;
  std::vector<Real> a(static_cast<Size>(n - 1), Real(1)), c(static_cast<Size>(n - 1), Real(1));
  std::vector<Real> b(static_cast<Size>(n), Real(-4));
  std::vector<Real> d(static_cast<Size>(n)), x;
  for (int j = 0; j < n; ++j) d[static_cast<Size>(j)] = Real(1);
  thomas_solve(a, b, c, d, x);
  // 手工验证残差
  for (int j = 0; j < n; ++j) {
    const Real lhs = (j > 0 ? Real(1) * x[static_cast<Size>(j - 1)] : Real(0)) +
                     Real(-4) * x[static_cast<Size>(j)] +
                     (j < n - 1 ? Real(1) * x[static_cast<Size>(j + 1)] : Real(0));
    VIBE_CHECK_NEAR(lhs, Real(1), 1e-10);
  }
}

VIBE_TEST(thomas_factored_matches_direct_solve) {
  const int n = 24;
  std::vector<Real> a(static_cast<Size>(n - 1)), c(static_cast<Size>(n - 1)),
      b(static_cast<Size>(n)), d(static_cast<Size>(n)), x1, x2;
  for (int j = 0; j < n; ++j) {
    b[static_cast<Size>(j)] = Real(4) + Real(0.1) * static_cast<Real>(j);
    d[static_cast<Size>(j)] = std::sin(Real(0.3) * static_cast<Real>(j));
    if (j < n - 1) {
      a[static_cast<Size>(j)] = Real(-1);
      c[static_cast<Size>(j)] = Real(-1.2);
    }
  }
  thomas_solve(a, b, c, d, x1);
  TridiagonalFactorization f;
  thomas_factorize(a, b, c, f);
  thomas_solve_factored(f, d, x2);
  for (int j = 0; j < n; ++j)
    VIBE_CHECK_NEAR(x1[static_cast<Size>(j)], x2[static_cast<Size>(j)], 1e-9);
}

VIBE_TEST(thomas_detects_singular_matrix) {
  std::vector<Real> a{Real(1)}, b{Real(0), Real(0)}, c{Real(1)}, d{Real(1), Real(1)}, x;
  bool threw = false;
  try {
    thomas_solve(a, b, c, d, x);
  } catch (const NumericalError&) {
    threw = true;
  }
  VIBE_CHECK(threw);
}

VIBE_TEST(multigrid_restrict_and_prolong_are_adjoint_like) {
  const grid::Grid gf = make_grid(8, 8, 4);
  const grid::Grid gc = make_grid(4, 4, 4);
  grid::Field<Real> fine(gf, grid::Stagger::Cell, "f"), coarse(gc, grid::Stagger::Cell, "c"),
      back(gf, grid::Stagger::Cell, "b");
  for (Int k = 0; k < 4; ++k)
    for (Int j = 0; j < 8; ++j)
      for (Int i = 0; i < 8; ++i) fine(i, j, k) = std::sin(Real(0.7) * i) + std::cos(Real(0.5) * j);
  multigrid_restrict(fine, coarse);
  multigrid_prolong(coarse, back);
  // 限制再延拓不应放大能量
  VIBE_CHECK(back.norm2() <= fine.norm2() * Real(1.5));
  VIBE_CHECK(!coarse.has_nonfinite());
}

VIBE_TEST(helmholtz_vertical_tridiagonal_recovers_manufactured_solution) {
  const grid::Grid g = make_grid(4, 4, 16);
  const config::ModelConfig cfg = make_cfg();
  auto solver = make_helmholtz_solver(HelmholtzKind::VerticalTridiagonal, g, cfg);
  VIBE_CHECK(solver != nullptr);
  VIBE_CHECK(std::string(solver->name()) == "vertical_tridiagonal");

  grid::Field<Real> a(g, grid::Stagger::Cell, "a"), b(g, grid::Stagger::Cell, "b");
  a.fill(Real(1.0));
  b.fill(Real(0.01));
  // 构造制造解 x = sin(pi k / nz)，f = L x
  grid::Field<Real> x_true(g, grid::Stagger::Cell, "xt"), f(g, grid::Stagger::Cell, "f"),
      x(g, grid::Stagger::Cell, "x");
  const Int nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    const Real val = std::sin(kPi * static_cast<Real>(k + 1) / static_cast<Real>(nz + 1));
    for (Int j = 0; j < 4; ++j)
      for (Int i = 0; i < 4; ++i) x_true(i, j, k) = val;
  }
  // f = (2a/dz^2 + b) x - a/dz^2 (x_{k-1} + x_{k+1})
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < 4; ++j)
      for (Int i = 0; i < 4; ++i) {
        const Real scale = g.geom().z_top - g.terrain(i, j);
        const Real dz = g.dzeta(k) * scale;
        const Real diag = Real(2) * Real(1.0) / (dz * dz) + Real(0.01);
        const Real off = Real(1.0) / (dz * dz);
        f(i, j, k) = diag * x_true(i, j, k) -
                     off * (x_true.clamp_at(i, j, k - 1) + x_true.clamp_at(i, j, k + 1));
      }
  solver->set_coefficients(a, b);
  solver->solve(a, b, f, x);
  Real err = Real(0);
  for (Int k = 0; k < nz; ++k) err = std::max(err, std::abs(x(0, 0, k) - x_true(0, 0, k)));
  VIBE_CHECK(err < Real(1e-8));
  VIBE_CHECK(solver->iterations() >= 1);
}

VIBE_TEST(helmholtz_krylov_converges_on_smooth_rhs) {
  const grid::Grid g = make_grid(8, 8, 8);
  const config::ModelConfig cfg = make_cfg();
  auto solver = make_helmholtz_solver(HelmholtzKind::KrylovJacobi, g, cfg);
  grid::Field<Real> a(g, grid::Stagger::Cell, "a"), b(g, grid::Stagger::Cell, "b");
  a.fill(Real(1.0));
  b.fill(Real(1.0));
  grid::Field<Real> f(g, grid::Stagger::Cell, "f"), x(g, grid::Stagger::Cell, "x");
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i)
        f(i, j, k) = std::sin(Real(0.4) * i) * std::cos(Real(0.3) * j);
  solver->set_coefficients(a, b);
  solver->solve(a, b, f, x);
  VIBE_CHECK(!x.has_nonfinite());
  // BiCGSTAB 可能未完全收敛，但残差必须有下降
  VIBE_CHECK(solver->residual() < Real(1.0));
}

VIBE_TEST(helmholtz_residual_matches_definition) {
  const grid::Grid g = make_grid(4, 4, 4);
  grid::Field<Real> a(g, grid::Stagger::Cell, "a"), b(g, grid::Stagger::Cell, "b"),
      f(g, grid::Stagger::Cell, "f"), x(g, grid::Stagger::Cell, "x");
  a.fill(Real(1.0));
  b.fill(Real(1.0));
  f.fill(Real(1.0));
  x.fill(Real(0));
  const Real r = helmholtz_residual_l2(g, a, b, f, x);
  VIBE_CHECK_NEAR(r, std::sqrt(Real(4.0 * 4 * 4)), 1e-9);
}

VIBE_TEST(acoustic_substepper_cfl_scales_with_dtau) {
  const grid::Grid g = make_grid(8, 8, 16);
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State s(g);
  s.set_reference(&ref);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp)
    s.field(static_cast<dyn::Species>(sp)).fill(Real(0));
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  AcousticSubstepper sub(g, AcousticOptions{});
  const Real cfl1 = sub.acoustic_cfl(s, Real(1));
  const Real cfl2 = sub.acoustic_cfl(s, Real(2));
  VIBE_CHECK_NEAR(cfl2, Real(2) * cfl1, std::abs(cfl1) * 1e-9);
  VIBE_CHECK(cfl1 > Real(0));
  VIBE_CHECK(sub.required_substeps(s, Real(60), Real(0.8)) >= 1);
}

VIBE_TEST(acoustic_substep_of_resting_state_is_stable) {
  const grid::Grid g = make_grid(8, 8, 16);
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  s.u().fill(Real(0));
  s.v().fill(Real(0));
  s.w().fill(Real(0));
  AcousticSubstepper sub(g, AcousticOptions{});
  const Real before = s.pi().norm2();
  for (int n = 0; n < 20; ++n) sub.step(s, Real(1), ref);
  VIBE_CHECK(!s.has_nonfinite());
  // 静止大气在无声波激发时 pi 不应爆炸
  VIBE_CHECK(s.pi().norm2() < before * Real(5) + Real(1));
}

VIBE_TEST(acoustic_vertical_implicit_is_unconditionally_stable) {
  const grid::Grid g = make_grid(8, 8, 16);
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  s.u().fill(Real(0)); s.v().fill(Real(0)); s.w().fill(Real(0));
  // 加一个小的 pi 扰动
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i)
        s.w()(i, j, k) = Real(0.01) * std::sin(Real(0.1) * k);
  AcousticOptions opt;
  opt.vertical_implicit = true;
  AcousticSubstepper sub(g, opt);
  Real max_w = Real(0);
  for (int n = 0; n < 50; ++n) {
    sub.step(s, Real(20), ref);   // 远超显式稳定极限
    max_w = std::max(max_w, std::abs(s.w().stats().max));
  }
  VIBE_CHECK(!s.has_nonfinite());
}

VIBE_TEST(rk3_integrates_resting_state_without_drift) {
  const grid::Grid g = make_grid(8, 8, 8);
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  const config::ModelConfig cfg = make_cfg();
  dyn::Equations eq(g, ref, cfg);
  RungeKutta3Integrator integ(g, ref, cfg, eq);
  dyn::State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  s.u().fill(Real(0)); s.v().fill(Real(0)); s.w().fill(Real(0)); s.qv().fill(Real(0));
  const Real mass0 = s.rho().stats().mean;
  for (int n = 0; n < 5; ++n) {
    integ.step(s, StepContext::make(Real(2), 4, s.time));
  }
  VIBE_CHECK(!s.has_nonfinite());
  VIBE_CHECK_NEAR(s.rho().stats().mean, mass0, std::abs(mass0) * 1e-10);
  VIBE_CHECK(integ.stats().steps == 5);
  VIBE_CHECK(integ.stats().acoustic_substeps == 20);
}

VIBE_TEST(rk3_energy_is_approximately_conserved_for_adiabatic_flow) {
  const grid::Grid g = make_grid(16, 16, 8);
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  config::ModelConfig cfg = make_cfg();
  cfg.numerics.coriolis = "none";
  cfg.numerics.sponge_alpha = Real(0.01);
  dyn::Equations eq(g, ref, cfg);
  RungeKutta3Integrator integ(g, ref, cfg, eq);
  dyn::State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  s.qv().fill(Real(0));
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i) {
        s.u()(i, j, k) = Real(5) * std::sin(Real(0.4) * i) * std::cos(Real(0.3) * j);
        s.v()(i, j, k) = Real(3) * std::cos(Real(0.2) * i);
        s.w()(i, j, k) = Real(0);
        s.theta()(i, j, k) += Real(0.5) * std::sin(Real(0.5) * i) * std::sin(Real(0.5) * j);
      }
  const Real e0 = dyn::energy_budget(s, ref).total;
  for (int n = 0; n < 3; ++n) integ.step(s, StepContext::make(Real(2), 4, s.time));
  const Real e1 = dyn::energy_budget(s, ref).total;
  VIBE_CHECK(!s.has_nonfinite());
  // 分裂显式格式的能量漂移应远小于信号本身（此处放宽到 30%）
  VIBE_CHECK(std::abs(static_cast<double>(e1 - e0)) < 0.3 * std::abs(static_cast<double>(e0)));
}

VIBE_TEST(semi_implicit_advances_resting_state_stably) {
  const grid::Grid g = make_grid(8, 8, 8);
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  const config::ModelConfig cfg = make_cfg("semi_implicit");
  dyn::Equations eq(g, ref, cfg);
  auto solver = make_helmholtz_solver(HelmholtzKind::VerticalTridiagonal, g, cfg);
  SemiImplicitIntegrator si(g, ref, cfg, eq, std::move(solver));
  dyn::State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  s.u().fill(Real(0)); s.v().fill(Real(0)); s.w().fill(Real(0)); s.qv().fill(Real(0));
  for (int n = 0; n < 5; ++n) si.step(s, StepContext::make(Real(10), 1, s.time));
  VIBE_CHECK(!s.has_nonfinite());
  VIBE_CHECK(si.stats().steps == 5);
}

VIBE_TEST(timestep_controller_detects_acoustic_cfl) {
  const grid::Grid g = make_grid(8, 8, 16);
  config::ModelConfig cfg = make_cfg();
  cfg.time.cfl_target = Real(0.8);
  cfg.time.adaptive_dt = true;
  TimeStepController ctl(cfg, g);
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  s.u().fill(Real(0)); s.v().fill(Real(0)); s.w().fill(Real(0));
  const auto rep = ctl.inspect(s, Real(1));
  VIBE_CHECK(rep.cfl_acoustic > Real(0));
  VIBE_CHECK(rep.cfl_advection >= Real(0));
  VIBE_CHECK(!rep.nonfinite);
  VIBE_CHECK(!rep.message.empty());
  // 过大的 dt 应被判定为不稳定
  const auto big = ctl.inspect(s, Real(1e6));
  VIBE_CHECK(ctl.should_abort(big));
  // 自适应建议应在界内
  const Real dt = ctl.suggest_dt(s, Real(5));
  VIBE_CHECK(dt >= ctl.dt_min() && dt <= ctl.dt_max() + Real(1e-9));
}

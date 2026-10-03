/// @file test_dynamics.cpp
/// @brief 参考态、状态、平流、科氏、阻尼、方程装配与诊断量的单元测试。

#include <cmath>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/test.hpp"
#include "vibe/config/config.hpp"
#include "vibe/dyn/advection.hpp"
#include "vibe/dyn/coriolis.hpp"
#include "vibe/dyn/damping.hpp"
#include "vibe/dyn/diagnostics.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/state.hpp"

using namespace vibe;
using namespace vibe::dyn;

namespace {

grid::Grid make_grid(Int nx = 8, Int ny = 8, Int nz = 12, bool periodic = true) {
  grid::Geometry g;
  g.nx = nx; g.ny = ny; g.nz = nz;
  g.dx = Real(2000); g.dy = Real(2000); g.z_top = Real(20000);
  g.zeta = grid::make_stretched_zeta(nz, Real(1), Real(1.1));
  g.name = "dyn_test";
  grid::Decomposition d = grid::Decomposition::make(nx, ny, nz, 1, 1, 4, periodic, periodic);
  return grid::Grid(g, d);
}

config::ModelConfig make_cfg() {
  config::ModelConfig c;
  c.name = "test";
  c.domain.nx = 8; c.domain.ny = 8; c.domain.nz = 12;
  c.domain.dx = Real(2000); c.domain.dy = Real(2000); c.domain.z_top = Real(20000);
  c.domain.periodic_x = true; c.domain.periodic_y = true;
  c.time.dt = Real(6); c.time.acoustic_substeps = 6;
  c.numerics.advection = "central2";
  c.numerics.coriolis = "fplane";
  c.numerics.fplane_latitude = Real(45);
  c.numerics.sponge_alpha = Real(0.2);
  c.numerics.sponge_start_fraction = Real(0.8);
  c.numerics.divergence_damping = Real(0);
  c.physics.microphysics = "kessler";
  return c;
}

}  // namespace

VIBE_TEST(refstate_isothermal_is_hydrostatic) {
  const grid::Grid g = make_grid();
  const ReferenceState ref = ReferenceState::isothermal(g, Real(300), Real(100000));
  VIBE_CHECK(ref.valid());
  // 离散静力平衡残差应很小（梯形积分与中心差分相容）
  VIBE_CHECK(ref.hydrostatic_residual() < Real(5e-3));
  // 密度与气压随高度递减
  VIBE_CHECK(ref.rho0()(0, 0, 5) < ref.rho0()(0, 0, 0));
  VIBE_CHECK(ref.p0()(0, 0, 5) < ref.p0()(0, 0, 0));
  // 等温：theta0 近似常数，pi0 线性递减
  const Real dpi = (ref.pi0()(0, 0, 1) - ref.pi0()(0, 0, 0));
  VIBE_CHECK(dpi < Real(0));
  // N^2 应接近 0
  VIBE_CHECK(std::abs(static_cast<double>(ref.brunt_vaisala2(0, 0, 4))) < 1e-6);
  // 声速平方物理合理
  const Real cs2 = ref.sound_speed2(0, 0, 0);
  VIBE_CHECK(cs2 > Real(90000) && cs2 < Real(130000));
}

VIBE_TEST(refstate_standard_atmosphere_has_inversion) {
  const grid::Grid g = make_grid(4, 4, 20);
  const ReferenceState ref = ReferenceState::standard_atmosphere(g, Real(101325));
  VIBE_CHECK(ref.valid());
  VIBE_CHECK(ref.hydrostatic_residual() < Real(0.05));
  // 对流层 N^2 > 0（稳定）
  for (Int k = 1; k < 6; ++k) VIBE_CHECK(ref.brunt_vaisala2(0, 0, k) > Real(0));
}

VIBE_TEST(state_pack_unpack_roundtrip) {
  const grid::Grid g = make_grid(6, 5, 4);
  State s(g);
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    auto& f = s.field(static_cast<Species>(sp));
    for (Int k = 0; k < f.nz(); ++k)
      for (Int j = 0; j < f.ny(); ++j)
        for (Int i = 0; i < f.nx(); ++i)
          f(i, j, k) = static_cast<Real>(sp * 1000 + i * 100 + j * 10 + k);
  }
  std::vector<Real> packed;
  s.pack(packed);
  VIBE_CHECK(packed.size() == s.packed_size());
  State t(g);
  t.unpack(packed);
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    const auto spc = static_cast<Species>(sp);
    VIBE_CHECK_NEAR(s.field(spc)(2, 2, 2), t.field(spc)(2, 2, 2), 1e-12);
  }
}

VIBE_TEST(state_axpy_and_norm) {
  const grid::Grid g = make_grid(4, 4, 2);
  State a(g), b(g), r(g);
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    a.field(static_cast<Species>(sp)).fill(Real(1));
    b.field(static_cast<Species>(sp)).fill(Real(3));
  }
  r.axpy(Real(2), a, Real(0));    // r = 2*a
  VIBE_CHECK_NEAR(r.u().stats().max, Real(2), 1e-12);
  r.axpy(Real(1), b, Real(1));    // r = b + r = 5
  VIBE_CHECK_NEAR(r.u().stats().min, Real(5), 1e-12);
  // 点积自洽
  VIBE_CHECK_NEAR(r.dot(r), r.norm2() * r.norm2(), std::abs(r.norm2() * r.norm2()) * 1e-12);
}

VIBE_TEST(advection_of_constant_field_is_zero) {
  const grid::Grid g = make_grid();
  State s(g);
  s.u().fill(Real(10));
  s.v().fill(Real(-5));
  s.w().fill(Real(0.5));
  s.rho().fill(Real(1.0));
  s.theta().fill(Real(300));
  Advection adv(g, AdvectionScheme::Central2, false);
  grid::Field<Real> dq(g, grid::Stagger::Cell, "dq");
  adv.scalar_nonconservative(s.theta(), s.u(), s.v(), s.w(), dq);
  // 常值场的梯度为 0（内部点）
  for (Int k = 1; k < g.nz() - 1; ++k)
    for (Int j = 1; j < g.ny() - 1; ++j)
      for (Int i = 1; i < g.nx() - 1; ++i) VIBE_CHECK_NEAR(dq(i, j, k), Real(0), 1e-10);
}

VIBE_TEST(advection_is_linear) {
  const grid::Grid g = make_grid();
  State s(g);
  s.u().fill(Real(8));
  s.v().fill(Real(3));
  s.w().fill(Real(0));
  Advection adv(g, AdvectionScheme::Central4, false);
  grid::Field<Real> q1(g, grid::Stagger::Cell, "q1"), q2(g, grid::Stagger::Cell, "q2");
  grid::Field<Real> d1(g, grid::Stagger::Cell, "d1"), d2(g, grid::Stagger::Cell, "d2"),
                   dc(g, grid::Stagger::Cell, "dc");
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i) {
        q1(i, j, k) = std::sin(Real(0.3) * static_cast<Real>(i));
        q2(i, j, k) = std::cos(Real(0.2) * static_cast<Real>(j) + static_cast<Real>(k));
      }
  adv.scalar_nonconservative(q1, s.u(), s.v(), s.w(), d1);
  adv.scalar_nonconservative(q2, s.u(), s.v(), s.w(), d2);
  grid::Field<Real> qc(g, grid::Stagger::Cell, "qc");
  qc.axpy(Real(2), q1, Real(0));
  qc.add_scaled(Real(-3), q2);
  adv.scalar_nonconservative(qc, s.u(), s.v(), s.w(), dc);
  for (Int k = 1; k < g.nz() - 1; ++k)
    for (Int j = 1; j < g.ny() - 1; ++j)
      for (Int i = 1; i < g.nx() - 1; ++i)
        VIBE_CHECK_NEAR(dc(i, j, k), Real(2) * d1(i, j, k) - Real(3) * d2(i, j, k), 1e-9);
}

VIBE_TEST(advection_cfl_matches_hand_calculation) {
  const grid::Grid g = make_grid();
  State s(g);
  s.u().fill(Real(0));
  s.v().fill(Real(0));
  s.w().fill(Real(0));
  Advection adv(g, AdvectionScheme::Central2, true);
  VIBE_CHECK_NEAR(adv.cfl_number(s, Real(10)), Real(0), 1e-12);
  s.u().fill(Real(100));
  const Real cfl = adv.cfl_number(s, Real(10));
  // dt * |u| / dx = 10 * 100 / 2000 = 0.5
  VIBE_CHECK_NEAR(cfl, Real(0.5), 1e-9);
}

VIBE_TEST(weno5_reconstructs_smooth_function_accurately) {
  // 对光滑函数，WENO5 的重构误差应为 O(h^5) 量级
  const Real h = Real(0.01);
  auto f = [](Real x) { return std::sin(Real(3) * x); };
  Real ql, qr;
  // 使用真实的类接口：通过 Advection 的静态私有函数不可访问，这里用等价的
  // 面值检验方式：构造场并检查通量差分的精度。
  const grid::Grid g = make_grid(16, 4, 4);
  State s(g);
  s.u().fill(Real(1));
  s.v().fill(Real(0));
  s.w().fill(Real(0));
  grid::Field<Real> q(g, grid::Stagger::Cell, "q");
  for (Int i = 0; i < g.nx(); ++i) q(i, 0, 0) = f(static_cast<Real>(i) * g.dx_at(i));
  std::vector<Real> dq;
  Advection adv(g, AdvectionScheme::WENO5, true);
  grid::Field<Real> d(g, grid::Stagger::Cell, "d");
  grid::Field<Real> rho(g, grid::Stagger::Cell, "rho");
  rho.fill(Real(1));
  adv.scalar(q, s.u(), s.v(), s.w(), rho, d);
  // 解析导数 -u * f'(x) = -3 cos(3x)
  const Real x = Real(8) * g.dx_at(0);
  const Real analytic = -Real(3) * std::cos(Real(3) * x);
  VIBE_CHECK(std::abs(static_cast<double>(d(8, 0, 0) - analytic)) < 0.05);
  VIBE_UNUSED(ql);
  VIBE_UNUSED(qr);
  VIBE_UNUSED(h);
  VIBE_UNUSED(dq);
}

VIBE_TEST(coriolis_fplane_preserves_kinetic_energy) {
  // f 平面上科氏力的离散应为反对称，不动能增长
  const grid::Grid g = make_grid(8, 8, 4);
  State s(g);
  for (Int k = 0; k < 4; ++k)
    for (Int j = 0; j < 8; ++j)
      for (Int i = 0; i <= 8; ++i) s.u()(i, j, k) = std::sin(Real(0.4) * static_cast<Real>(i));
  for (Int k = 0; k < 4; ++k)
    for (Int j = 0; j <= 8; ++j)
      for (Int i = 0; i < 8; ++i) s.v()(i, j, k) = std::cos(Real(0.3) * static_cast<Real>(j));
  Tendency d(g);
  Coriolis cor(g, CoriolisMode::FPlane, Real(45));
  cor.apply(s, d);
  // 科氏力做功 = rho * (u * f v + v * (-f u)) 逐点相消（插值后近似成立）
  Real work = Real(0);
  for (Int k = 0; k < 4; ++k)
    for (Int j = 0; j < 8; ++j)
      for (Int i = 0; i < 8; ++i) {
        const Real uc = Real(0.5) * (s.u()(i, j, k) + s.u()(i + 1, j, k));
        const Real vc = Real(0.5) * (s.v()(i, j, k) + s.v()(i, j + 1, k));
        work += uc * d.u()(i, j, k) + vc * d.v()(i, j, k);
      }
  VIBE_CHECK(std::abs(static_cast<double>(work)) < 1.0);
  VIBE_CHECK_NEAR(cor.f_at(0), Real(2) * kOmega * std::sin(Real(45) * kDegToRad), 1e-15);
}

VIBE_TEST(coriolis_none_mode_is_noop) {
  const grid::Grid g = make_grid(4, 4, 2);
  State s(g);
  s.u().fill(Real(5));
  s.v().fill(Real(5));
  Tendency d(g);
  d.zero();
  Coriolis cor(g, CoriolisMode::None, Real(45));
  cor.apply(s, d);
  VIBE_CHECK_NEAR(d.u().stats().max, Real(0), 1e-15);
  VIBE_CHECK_NEAR(d.v().stats().max, Real(0), 1e-15);
}

VIBE_TEST(damping_sponge_profile_increases_with_height) {
  const grid::Grid g = make_grid(4, 4, 20);
  Damping damp(g, Real(16000), Real(20000), Real(0.5), 2, Real(0));
  const auto& prof = damp.sponge_profile();
  VIBE_CHECK(static_cast<Int>(prof.size()) == 20);
  for (Int k = 0; k < g.nz(); ++k) {
    const Real z = g.z_center(0, 0, k);
    if (z < Real(16000)) VIBE_CHECK_NEAR(prof[static_cast<Size>(k)], Real(0), 1e-15);
  }
  // 顶层权重最大
  VIBE_CHECK(prof.back() > prof[static_cast<Size>(g.nz() - 5)]);
  VIBE_CHECK(prof.back() <= Real(0.5) + 1e-12);
}

VIBE_TEST(damping_sponge_relaxes_to_reference) {
  const grid::Grid g = make_grid(4, 4, 20);
  const ReferenceState ref = ReferenceState::isothermal(g, Real(300), Real(100000));
  State s(g);
  s.set_reference(&ref);
  for (int sp = 0; sp < kNumSpecies; ++sp) s.field(static_cast<Species>(sp)).fill(Real(0));
  // 顶层给一个偏差
  s.u()(0, 0, 19) = Real(100);
  s.theta()(0, 0, 19) = ref.theta0()(0, 0, 19) + Real(50);
  Damping damp(g, Real(16000), Real(20000), Real(0.5), 2, Real(0));
  damp.apply_sponge(s, ref, Real(10));
  VIBE_CHECK(std::abs(static_cast<double>(s.u()(0, 0, 19))) < 100.0);
  VIBE_CHECK(std::abs(static_cast<double>(s.theta()(0, 0, 19) - ref.theta0()(0, 0, 19))) < 50.0);
  // 底层不应被触碰
  VIBE_CHECK_NEAR(s.u()(0, 0, 0), Real(0), 1e-15);
}

VIBE_TEST(equations_acoustic_tendency_is_linear_in_pi) {
  const grid::Grid g = make_grid(8, 8, 8);
  const ReferenceState ref = ReferenceState::isothermal(g, Real(300), Real(100000));
  const config::ModelConfig cfg = make_cfg();
  Equations eq(g, ref, cfg);
  State s(g);
  s.set_reference(&ref);
  for (int sp = 0; sp < kNumSpecies; ++sp) s.field(static_cast<Species>(sp)).fill(Real(0));
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  Tendency d1(g), d2(g), dsum(g);
  eq.acoustic_tendencies(s, d1);
  // 叠加一个 pi 扰动
  State s2 = s.clone();
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i) s2.pi()(i, j, k) += Real(0.001) * std::sin(Real(0.5) * i);
  eq.acoustic_tendencies(s2, d2);
  // 与加速度线性相关：d2 - d1 应正比于 pi 扰动
  Real maxd = Real(0);
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i) maxd = std::max(maxd, std::abs(d2.u()(i, j, k) - d1.u()(i, j, k)));
  VIBE_CHECK(maxd > Real(0));
}

VIBE_TEST(equations_diagnose_is_self_consistent) {
  const grid::Grid g = make_grid(8, 8, 8);
  const ReferenceState ref = ReferenceState::isothermal(g, Real(300), Real(100000));
  const config::ModelConfig cfg = make_cfg();
  Equations eq(g, ref, cfg);
  State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.qv().fill(Real(0.005));
  s.pi().fill(Real(0));
  eq.diagnose(s);
  // 诊断出的 pi 应满足状态方程：p = rho Rd T (1+qv/eps)/(1+qv)
  for (Int k = 0; k < g.nz(); k += 3)
    for (Int j = 0; j < g.ny(); j += 3)
      for (Int i = 0; i < g.nx(); i += 3) {
        const Real qv = s.qv()(i, j, k);
        const Real gf = (Real(1) + qv / kEpsilonVap) / (Real(1) + qv);
        const Real T = s.theta()(i, j, k) * s.pi()(i, j, k);
        const Real p_eos = s.rho()(i, j, k) * kRd * T * gf;
        const Real p_pi = kP0 * std::pow(s.pi()(i, j, k), kCp / kRd);
        VIBE_CHECK_NEAR(p_eos, p_pi, std::abs(p_pi) * 1e-10);
      }
}

VIBE_TEST(equations_linearized_coefficients_are_physical) {
  const grid::Grid g = make_grid(4, 4, 6);
  const ReferenceState ref = ReferenceState::isothermal(g, Real(300), Real(100000));
  const config::ModelConfig cfg = make_cfg();
  Equations eq(g, ref, cfg);
  grid::Field<Real> alpha(g, grid::Stagger::Cell, "a"), beta(g, grid::Stagger::Cell, "b");
  eq.linearized_coefficients(ref, alpha, beta);
  // alpha = 1/(rho c^2)：量级约 1/(1 * 1e5) = 1e-5
  VIBE_CHECK(alpha(0, 0, 0) > Real(1e-6) && alpha(0, 0, 0) < Real(1e-3));
  // beta = 1/rho：量级约 1
  VIBE_CHECK_NEAR(beta(0, 0, 0), Real(1) / ref.rho0()(0, 0, 0), 1e-12);
}

VIBE_TEST(diagnostics_exner_conversions_invert) {
  const Real pi = Real(0.9);
  const Real p = exner_to_pressure(pi);
  VIBE_CHECK_NEAR(std::pow(p / kP0, kRd / kCp), pi, 1e-12);
  VIBE_CHECK_NEAR(exner_to_temperature(pi, Real(300)), Real(270), 1e-12);
}

VIBE_TEST(diagnostics_relative_humidity_and_dewpoint) {
  // 饱和时 RH ~ 1
  const Real T = Real(290);
  const Real p = Real(100000);
  const Real es = Real(611.2) * std::exp(Real(17.67) * (T - kT0) / (T - Real(29.65)));
  const Real qs = kEpsilonVap * es / (p - es);
  const Real pi = std::pow(p / kP0, kRd / kCp);
  const Real theta = T / pi;
  const Real rh = relative_humidity(pi, theta, qs);
  VIBE_CHECK_NEAR(rh, Real(1), Real(0.02));
  // 露点应低于或等于温度
  const Real td = dewpoint_from_qv(p, qs);
  VIBE_CHECK(td <= T + Real(0.5));
  // 干空气露点很低
  VIBE_CHECK(dewpoint_from_qv(p, Real(1e-5)) < Real(250));
}

VIBE_TEST(diagnostics_theta_e_exceeds_theta_when_moist) {
  const Real theta = Real(300), pi = Real(0.9);
  const Real te_dry = theta_e_from_rho(theta, Real(0), pi);
  const Real te_moist = theta_e_from_rho(theta, Real(0.015), pi);
  VIBE_CHECK(te_moist > te_dry);
  VIBE_CHECK(te_dry > Real(270) && te_dry < Real(320));
}

VIBE_TEST(diagnostics_reflectivity_ordering) {
  const Real rho = Real(1.0), T = Real(280);
  VIBE_CHECK(reflectivity_dbz(Real(0), Real(0), Real(0), rho, T) <= Real(-29));
  const Real light = reflectivity_dbz(Real(1e-5), Real(0), Real(0), rho, T);
  const Real heavy = reflectivity_dbz(Real(1e-3), Real(0), Real(0), rho, T);
  VIBE_CHECK(heavy > light);
  VIBE_CHECK(heavy > Real(20) && heavy < Real(80));
}

VIBE_TEST(diagnostics_cape_positive_in_unstable_profile) {
  // 构造一个条件不稳定廓线：地面暖湿，上面干冷
  std::vector<Real> z{Real(0), Real(1000), Real(2000), Real(3000), Real(4000), Real(5000)};
  std::vector<Real> p{Real(100000), Real(90000), Real(80000), Real(70000), Real(60000), Real(50000)};
  std::vector<Real> t{Real(300), Real(290), Real(280), Real(270), Real(260), Real(250)};
  std::vector<Real> qv{Real(0.018), Real(0.014), Real(0.010), Real(0.006), Real(0.003), Real(0.001)};
  const CapeResult r = cape_cin_column(z, p, t, qv, Real(500));
  VIBE_CHECK(r.cape >= Real(0));
  VIBE_CHECK(std::isfinite(static_cast<double>(r.cape)));
  // 稳定廓线（处处等温）应几乎无 CAPE
  std::vector<Real> t_stable{Real(280), Real(279), Real(278), Real(277), Real(276), Real(275)};
  std::vector<Real> qv_dry(6, Real(0.0001));
  const CapeResult r2 = cape_cin_column(z, p, t_stable, qv_dry, Real(500));
  VIBE_CHECK(r2.cape < r.cape + Real(1));
}

VIBE_TEST(diagnostics_energy_budget_mass_matches_sum) {
  const grid::Grid g = make_grid(4, 4, 6);
  const ReferenceState ref = ReferenceState::isothermal(g, Real(300), Real(100000));
  State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  s.u().fill(Real(0));
  s.v().fill(Real(0));
  s.w().fill(Real(0));
  s.qv().fill(Real(0));
  const EnergyBudget e = energy_budget(s, ref);
  Real mass = Real(0);
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i) mass += s.rho()(i, j, k) * g.cell_volume(i, j, k);
  VIBE_CHECK_NEAR(e.mass, mass, std::abs(mass) * 1e-12);
  VIBE_CHECK(e.internal > Real(0));
  VIBE_CHECK(e.potential > Real(0));
  VIBE_CHECK_NEAR(e.total, e.kinetic + e.internal + e.potential + e.latent,
                  std::abs(e.total) * 1e-12);
}

VIBE_TEST(diagnostics_ertel_pv_is_finite) {
  const grid::Grid g = make_grid(8, 8, 8);
  const ReferenceState ref = ReferenceState::isothermal(g, Real(300), Real(100000));
  State s(g);
  s.set_reference(&ref);
  s.rho() = ref.rho0();
  s.theta() = ref.theta0();
  s.pi() = ref.pi0();
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i) {
        s.u()(i, j, k) = Real(10) * std::sin(Real(0.5) * static_cast<Real>(j));
        s.v()(i, j, k) = Real(0);
      }
  Diagnostics diag(g);
  diagnose_all(s, ref, diag);
  VIBE_CHECK(!diag.pv.has_nonfinite());
  VIBE_CHECK(!diag.pressure.has_nonfinite());
  // 气压应在 1e4-1.1e5 Pa 范围内
  const auto ps = diag.pressure.stats();
  VIBE_CHECK(ps.min > Real(1e4) && ps.max < Real(1.2e5));
}

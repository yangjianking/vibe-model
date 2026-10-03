/// @file test_obs_operator.cpp
/// @brief 观测算子、快速辐射传输与偏差订正的单元测试。
///
/// 覆盖：折射率公式与已知值、插值权重和 = 1、格点上的精确性、水平线性函数的
/// 精确再现、applyTL 线性性、点积检验（< 1e-8）、创新向量、2m 温度公式、
/// RT 亮温范围与雅可比对有限差分的一致性、内置通道数、VarBC 可逆性与更新、
/// 复合算子一致性、观测空间统计与 CSV 往返。
///
/// 入口由 tests/unit/test_main.cpp（目标 vibe_test_main）统一提供。

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/test.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/decomposition.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/obs/bias_correction.hpp"
#include "vibe/obs/obs_operator.hpp"
#include "vibe/obs/obs_operators.hpp"
#include "vibe/obs/observations.hpp"
#include "vibe/obs/radiative_transfer.hpp"

using vibe::Int;
using vibe::Real;
using vibe::Size;
using vibe::obs::ObsSpace;
using vibe::obs::ObsType;
using vibe::obs::Observation;
using vibe::obs::VarKind;

namespace {

grid::Grid make_grid(Int nx = 8, Int ny = 6, Int nz = 5) {
  grid::Geometry geom;
  geom.nx = nx;
  geom.ny = ny;
  geom.nz = nz;
  geom.dx = Real(1000);
  geom.dy = Real(1000);
  geom.z_top = Real(10000);
  geom.zeta = grid::Geometry::default_zeta_levels(nz, Real(1.0));
  const grid::Decomposition dec = grid::Decomposition::make(nx, ny, nz, 1, 1, 4);
  return grid::Grid(geom, dec);
}

dyn::ReferenceState make_ref(const grid::Grid& g) {
  return dyn::ReferenceState::isothermal(g, Real(300), Real(100000), Real(0.005));
}

/// 填一个平滑、物理上合理的状态（theta'/pi'/qv/u/v/w 都有取值）
void fill_state(dyn::State& s, const grid::ReferenceState& ref) {
  const grid::Grid& g = s.grid();
  s.set_reference(&ref);
  for (Int k = 0; k < g.nz(); ++k) {
    for (Int j = 0; j < g.ny(); ++j) {
      for (Int i = 0; i < g.nx(); ++i) {
        const Real x = Real(i) * g.geom().dx;
        const Real y = Real(j) * g.geom().dy;
        s.u().at(i, j, k) = Real(5) + Real(0.001) * x;
        s.v().at(i, j, k) = Real(2) - Real(0.0005) * y;
        s.w().at(i, j, k) = Real(0.1);
        // State 存全量场（参考态 + 小扰动），与 dyn 层一致
        s.theta().at(i, j, k) =
            ref.theta0().at(i, j, k) + Real(1) * std::sin(Real(0.001) * x);
        s.pi().at(i, j, k) = ref.pi0().at(i, j, k) + Real(1e-4) * std::cos(Real(0.001) * y);
        s.qv().at(i, j, k) = Real(0.005) + Real(1e-4) * std::sin(Real(0.001) * x);
        s.rho().at(i, j, k) = ref.rho0().at(i, j, k);
        s.field(dyn::Species::Qr).at(i, j, k) = Real(1e-4);
        s.field(dyn::Species::Qs).at(i, j, k) = Real(0);
        s.field(dyn::Species::Qg).at(i, j, k) = Real(0);
      }
    }
  }
}

Real node_x(const grid::Grid& g, Int i) {
  return g.geom().x0 + (static_cast<Real>(i) + Real(0.5)) * g.geom().dx;
}
Real node_y(const grid::Grid& g, Int j) {
  return g.geom().y0 + (static_cast<Real>(j) + Real(0.5)) * g.geom().dy;
}

Observation make_obs(ObsType t, VarKind v, Real x, Real y, Real z, Real value = Real(0),
                     Real sigma = Real(1)) {
  Observation o;
  o.type = t;
  o.variable = v;
  o.x = x;
  o.y = y;
  o.z = z;
  o.value = value;
  o.sigma = sigma;
  o.qc_flag = 0;
  return o;
}

ObsSpace make_space(std::vector<Observation> v) {
  ObsSpace s;
  s.obs = std::move(v);
  s.window_length = Real(3600);
  s.source = "unit-test";
  return s;
}

vibe::obs::ModelStateView view_of(const dyn::State& s, const dyn::ReferenceState& ref,
                                  const grid::Grid& g) {
  vibe::obs::ModelStateView v;
  v.state = &s;
  v.ref = &ref;
  v.grid = &g;
  return v;
}

}  // namespace

// ===========================================================================
// 1. 折射率公式与已知值
// ===========================================================================
VIBE_TEST(obs_gnss_refractivity_known_value) {
  // N = 77.6 p[Pa]/100 / T + 3.73e5 qv (p/100) / T^2
  const Real n_dry = vibe::obs::GnssRoOperator::refractivity(Real(100000), Real(300), Real(0));
  VIBE_CHECK_NEAR(n_dry, Real(77.6) * Real(1000) / Real(300), Real(1e-9));
  const Real n_wet = vibe::obs::GnssRoOperator::refractivity(Real(100000), Real(300), Real(0.01));
  VIBE_CHECK(n_wet > n_dry);
  const Real expect_wet =
      Real(77.6) * Real(1000) / Real(300) +
      Real(3.73e5) * Real(0.01) * Real(1000) / (Real(300) * Real(300));
  VIBE_CHECK_NEAR(n_wet, expect_wet, Real(1e-9));
}

// ===========================================================================
// 2. 常数场插值 = 常数（权重和为 1 的直接检验）
// ===========================================================================
VIBE_TEST(obs_sampling_constant_field) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  s.qv().fill(Real(0.007));
  vibe::obs::SoundingOperator op(g);
  const auto v = view_of(s, ref, g);
  const ObsSpace obs = make_space({make_obs(ObsType::Radiosonde, VarKind::Q, Real(1234),
                                            Real(2345), Real(3456))});
  std::vector<Real> y;
  op.apply(v, obs, y);
  VIBE_CHECK_NEAR(y[0], Real(0.007), Real(1e-14));
}

// ===========================================================================
// 3. 观测点落在格点上时插值精确
// ===========================================================================
VIBE_TEST(obs_sampling_exact_on_grid_point) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  const Int i0 = 3, j0 = 2, k0 = 2;
  const Real x = node_x(g, i0);
  const Real y = node_y(g, j0);
  const Real z = g.z_center(i0, j0, k0);
  const Real expected = s.qv().at(i0, j0, k0);
  vibe::obs::SoundingOperator op(g);
  const auto v = view_of(s, ref, g);
  const ObsSpace obs =
      make_space({make_obs(ObsType::Radiosonde, VarKind::Q, x, y, z)});
  std::vector<Real> y;
  op.apply(v, obs, y);
  VIBE_CHECK_NEAR(y[0], expected, Real(1e-14));
}

// ===========================================================================
// 4. 水平线性函数被双线性插值精确再现
// ===========================================================================
VIBE_TEST(obs_sampling_linear_horizontal) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  const Real A = Real(0.01), B = Real(1e-6);
  for (Int k = 0; k < g.nz(); ++k) {
    for (Int j = 0; j < g.ny(); ++j) {
      for (Int i = 0; i < g.nx(); ++i) {
        s.qv().at(i, j, k) = A + B * node_x(g, i);
      }
    }
  }
  const Real xt = node_x(g, 2) + Real(0.37) * g.geom().dx;
  const Real yt = node_y(g, 1);
  const Real zt = g.z_center(2, 1, 1);
  vibe::obs::SoundingOperator op(g);
  const auto v = view_of(s, ref, g);
  const ObsSpace obs = make_space({make_obs(ObsType::Radiosonde, VarKind::Q, xt, yt, zt)});
  std::vector<Real> y;
  op.apply(v, obs, y);
  VIBE_CHECK_NEAR(y[0], A + B * xt, Real(1e-14));
}

// ===========================================================================
// 5. applyTL 的线性性
// ===========================================================================
VIBE_TEST(obs_operator_tl_linearity) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  const auto v = view_of(s, ref, g);

  dyn::State d1(g), d2(g), d3(g);
  for (Int k = 0; k < g.nz(); ++k) {
    for (Int j = 0; j < g.ny(); ++j) {
      for (Int i = 0; i < g.nx(); ++i) {
        const Real x = Real(i), y = Real(j);
        d1.theta().at(i, j, k) = std::sin(Real(0.7) * x + Real(0.3) * y);
        d2.theta().at(i, j, k) = std::cos(Real(0.4) * x - Real(0.2) * y);
        d1.pi().at(i, j, k) = Real(1e-5) * std::sin(Real(0.5) * x);
        d2.pi().at(i, j, k) = Real(1e-5) * std::cos(Real(0.9) * y);
        d1.qv().at(i, j, k) = Real(1e-5) * std::sin(Real(1.1) * x);
        d2.qv().at(i, j, k) = Real(1e-5) * std::cos(Real(0.6) * y);
        d1.u().at(i, j, k) = std::sin(Real(0.2) * x + Real(0.8) * y);
        d2.u().at(i, j, k) = std::cos(Real(0.3) * x - Real(0.4) * y);
      }
    }
  }
  const Real a = Real(1.7), b = Real(-0.9);
  d3.axpy(a, d1, Real(0));
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    d3.field(static_cast<dyn::Species>(sp)).add_scaled(b, d2.field(static_cast<dyn::Species>(sp)));
  }

  const ObsSpace obs = make_space({
      make_obs(ObsType::Radiosonde, VarKind::T, node_x(g, 2) + Real(100), node_y(g, 1), g.z_center(2, 1, 1)),
      make_obs(ObsType::Radiosonde, VarKind::Q, node_x(g, 4), node_y(g, 3), g.z_center(4, 3, 2)),
      make_obs(ObsType::Radiosonde, VarKind::U, node_x(g, 1) + Real(200), node_y(g, 2), g.z_center(1, 2, 1)),
  });
  vibe::obs::SoundingOperator op(g);
  std::vector<Real> y1, y2, y3;
  op.applyTL(v, d1, obs, y1);
  op.applyTL(v, d2, obs, y2);
  op.applyTL(v, d3, obs, y3);
  for (Size m = 0; m < y3.size(); ++m) {
    VIBE_CHECK_NEAR(y3[m], a * y1[m] + b * y2[m], Real(1e-12));
  }
}

// ===========================================================================
// 6. 探空算子点积检验
// ===========================================================================
VIBE_TEST(obs_sounding_adjoint_check) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  vibe::obs::SoundingOperator op(g);
  const ObsSpace obs = make_space({
      make_obs(ObsType::Radiosonde, VarKind::T, node_x(g, 2) + Real(110), node_y(g, 1), g.z_center(2, 1, 1)),
      make_obs(ObsType::Radiosonde, VarKind::Q, node_x(g, 4), node_y(g, 3) + Real(90), g.z_center(4, 3, 2)),
      make_obs(ObsType::Radiosonde, VarKind::P, node_x(g, 1), node_y(g, 2), g.z_center(1, 2, 3)),
      make_obs(ObsType::Radiosonde, VarKind::U, node_x(g, 3), node_y(g, 4), g.z_center(3, 4, 1)),
      make_obs(ObsType::Radiosonde, VarKind::V, node_x(g, 5), node_y(g, 1), g.z_center(5, 1, 2)),
      make_obs(ObsType::Radiosonde, VarKind::W, node_x(g, 2), node_y(g, 4), g.z_center(2, 4, 2)),
  });
  const auto v = view_of(s, ref, g);
  const Real err = op.check_adjoint(v, obs, 7u);
  VIBE_CHECK(err < Real(1e-8));
  // 多组随机种子都应通过
  VIBE_CHECK(op.check_adjoint(v, obs, 11u) < Real(1e-8));
  VIBE_CHECK(op.check_adjoint(v, obs, 42u) < Real(1e-8));
}

// ===========================================================================
// 7. 切线性一致性（有限差分）
// ===========================================================================
VIBE_TEST(obs_sounding_tangent_check) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  vibe::obs::SoundingOperator op(g);
  const ObsSpace obs = make_space({
      make_obs(ObsType::Radiosonde, VarKind::T, node_x(g, 2) + Real(110), node_y(g, 1), g.z_center(2, 1, 1)),
      make_obs(ObsType::Radiosonde, VarKind::Q, node_x(g, 4), node_y(g, 3), g.z_center(4, 3, 2)),
  });
  const auto v = view_of(s, ref, g);
  const Real err = op.check_tangent(v, obs, Real(1e-6), 3u);
  VIBE_CHECK(err < Real(1e-5));
}

// ===========================================================================
// 8. 创新向量 d = y - H(x)
// ===========================================================================
VIBE_TEST(obs_innovations_definition) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  const Real x = node_x(g, 2), y = node_y(g, 2), z = g.z_center(2, 2, 1);
  Observation o = make_obs(ObsType::Radiosonde, VarKind::Q, x, y, z, Real(0.02));
  o.bias = Real(0.001);
  const ObsSpace obs = make_space({o});
  vibe::obs::SoundingOperator op(g);
  const auto v = view_of(s, ref, g);
  std::vector<Real> d;
  op.innovations(v, obs, d);
  const Real hx = s.qv().at(2, 2, 1);
  VIBE_CHECK_NEAR(d[0], Real(0.02) - Real(0.001) - hx, Real(1e-14));
}

// ===========================================================================
// 9. 地面 2m 温度公式
// ===========================================================================
VIBE_TEST(obs_surface_2m_temperature_formula) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  const Int i0 = 2, j0 = 3;
  const Real x = node_x(g, i0), y = node_y(g, j0);
  const Real th1 = s.theta().at(i0, j0, 0);   // 已含 theta0
  const Real pi_s = s.pi().at(i0, j0, 0);     // 已含 pi0
  const Real z1 = g.z_center(i0, j0, 0);
  const Real expected = (th1 + (vibe::kGravity / vibe::kCp) * (z1 - Real(2))) * pi_s;
  vibe::obs::SurfaceOperator op(g);
  const auto v = view_of(s, ref, g);
  const ObsSpace obs = make_space({make_obs(ObsType::Surface, VarKind::T, x, y, Real(2))});
  std::vector<Real> y;
  op.apply(v, obs, y);
  VIBE_CHECK_NEAR(y[0], expected, Real(1e-10));
  VIBE_CHECK(expected > Real(200) && expected < Real(400));
}

// ===========================================================================
// 10. 地面算子点积检验（含 10 m 对数律）
// ===========================================================================
VIBE_TEST(obs_surface_adjoint_check) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  vibe::obs::SurfaceOperator op(g);
  const ObsSpace obs = make_space({
      make_obs(ObsType::Surface, VarKind::T, node_x(g, 2), node_y(g, 2), Real(2)),
      make_obs(ObsType::Surface, VarKind::U, node_x(g, 3) + Real(120), node_y(g, 1), Real(10)),
      make_obs(ObsType::Surface, VarKind::V, node_x(g, 1), node_y(g, 3) + Real(80), Real(10)),
      make_obs(ObsType::Surface, VarKind::PS, node_x(g, 4), node_y(g, 2), Real(0)),
  });
  const auto v = view_of(s, ref, g);
  VIBE_CHECK(op.check_adjoint(v, obs, 5u) < Real(1e-8));
}

// ===========================================================================
// 11. GNSS 掩星与雷达算子点积检验
// ===========================================================================
VIBE_TEST(obs_gnss_and_radar_adjoint_check) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  const auto v = view_of(s, ref, g);
  vibe::obs::GnssRoOperator ro(g);
  const ObsSpace obs_ro = make_space({
      make_obs(ObsType::GnssRo, VarKind::Refractivity, node_x(g, 2) + Real(130), node_y(g, 1), g.z_center(2, 1, 2)),
      make_obs(ObsType::GnssRo, VarKind::Refractivity, node_x(g, 4), node_y(g, 3), g.z_center(4, 3, 1)),
  });
  VIBE_CHECK(ro.check_adjoint(v, obs_ro, 9u) < Real(1e-8));

  vibe::obs::RadarOperator radar(g);
  const ObsSpace obs_radar = make_space({
      make_obs(ObsType::RadarReflectivity, VarKind::Reflectivity, node_x(g, 1) + Real(60), node_y(g, 2), g.z_center(1, 2, 1)),
      make_obs(ObsType::RadarReflectivity, VarKind::Reflectivity, node_x(g, 5), node_y(g, 4), g.z_center(5, 4, 2)),
  });
  VIBE_CHECK(radar.check_adjoint(v, obs_radar, 13u) < Real(1e-8));
}

// ===========================================================================
// 12. 辐射传输：内置通道与亮温范围
// ===========================================================================
VIBE_TEST(obs_rt_builtin_channels_and_bt_range) {
  vibe::obs::RadiativeTransfer amsua = vibe::obs::RadiativeTransfer::builtin_channels("amsua");
  VIBE_CHECK(amsua.n_channels() == 15);
  vibe::obs::RadiativeTransfer mhs = vibe::obs::RadiativeTransfer::builtin_channels("mhs");
  VIBE_CHECK(mhs.n_channels() == 5);

  vibe::obs::LayerProfile p;
  p.pressure = Real(50000);
  p.temperature = Real(255);
  p.qv = Real(0.004);
  p.ozone = Real(2e-6);
  p.zenith_angle_deg = Real(0);
  p.surface_pressure = Real(100000);
  p.surface_temperature = Real(290);
  p.skin_temperature = Real(290);
  p.surface_emissivity = Real(0.95);

  for (int c = 0; c < static_cast<int>(amsua.n_channels()); ++c) {
    const auto r = amsua.forward(c, p);
    VIBE_CHECK(std::isfinite(r.brightness_temperature));
    VIBE_CHECK(r.brightness_temperature > Real(150));
    VIBE_CHECK(r.brightness_temperature < Real(350));
    VIBE_CHECK(r.transmittance_to_surface >= Real(0));
    VIBE_CHECK(r.transmittance_to_surface <= Real(1));
    VIBE_CHECK(r.layer_optical_depth.size() == 1);
  }
  for (int c = 0; c < static_cast<int>(mhs.n_channels()); ++c) {
    const auto r = mhs.forward(c, p);
    VIBE_CHECK(std::isfinite(r.brightness_temperature));
    VIBE_CHECK(r.brightness_temperature > Real(120));
    VIBE_CHECK(r.brightness_temperature < Real(360));
  }
}

// ===========================================================================
// 13. 辐射传输：解析雅可比对有限差分一致
// ===========================================================================
VIBE_TEST(obs_rt_jacobian_vs_finite_difference) {
  vibe::obs::RadiativeTransfer rt = vibe::obs::RadiativeTransfer::builtin_channels("amsua");
  for (int c : {0, 4, 9}) {
    vibe::obs::LayerProfile p;
    p.pressure = Real(50000);
    p.temperature = Real(255);
    p.qv = Real(0.004);
    p.ozone = Real(2e-6);
    p.surface_pressure = Real(100000);
    p.surface_temperature = Real(290);
    p.skin_temperature = Real(290);
    p.surface_emissivity = Real(0.95);

    const auto jac = rt.jacobian(c, p);

    // 温度雅可比（中心差分）
    const Real eps_t = Real(1e-2);
    vibe::obs::LayerProfile pp = p, pm = p;
    pp.temperature += eps_t;
    pm.temperature -= eps_t;
    const Real fd_t = (rt.forward(c, pp).brightness_temperature -
                       rt.forward(c, pm).brightness_temperature) /
                      (Real(2) * eps_t);
    VIBE_CHECK(std::abs(fd_t - jac.temperature[0]) <=
               Real(1e-3) * std::max(Real(1e-3), std::abs(jac.temperature[0])));

    // 水汽雅可比
    const Real eps_q = Real(1e-6);
    vibe::obs::LayerProfile qp = p, qm = p;
    qp.qv += eps_q;
    qm.qv -= eps_q;
    const Real fd_q = (rt.forward(c, qp).brightness_temperature -
                       rt.forward(c, qm).brightness_temperature) /
                      (Real(2) * eps_q);
    VIBE_CHECK(std::abs(fd_q - jac.humidity[0]) <=
               Real(1e-2) * std::max(Real(1e-4), std::abs(jac.humidity[0])));

    // 地面温度雅可比
    const Real eps_s = Real(1e-2);
    vibe::obs::LayerProfile sp = p, sm = p;
    sp.surface_temperature += eps_s;
    sp.skin_temperature += eps_s;
    sm.surface_temperature -= eps_s;
    sm.skin_temperature -= eps_s;
    const Real fd_s = (rt.forward(c, sp).brightness_temperature -
                       rt.forward(c, sm).brightness_temperature) /
                      (Real(2) * eps_s);
    VIBE_CHECK(std::abs(fd_s - jac.surface_temperature) <=
               Real(1e-3) * std::max(Real(1e-3), std::abs(jac.surface_temperature)));
  }
}

// ===========================================================================
// 14. 辐射传输：光学厚度随水汽单调增加
// ===========================================================================
VIBE_TEST(obs_rt_optical_depth_monotonic_in_humidity) {
  vibe::obs::RadiativeTransfer rt = vibe::obs::RadiativeTransfer::builtin_channels("amsua");
  vibe::obs::LayerProfile dry, wet;
  dry.pressure = wet.pressure = Real(50000);
  dry.temperature = wet.temperature = Real(260);
  dry.qv = Real(0.0005);
  wet.qv = Real(0.01);
  dry.ozone = wet.ozone = Real(0);
  dry.surface_pressure = wet.surface_pressure = Real(100000);
  dry.surface_temperature = wet.surface_temperature = Real(290);
  dry.skin_temperature = wet.skin_temperature = Real(290);
  const auto td = rt.optical_depth(0, dry);
  const auto tw = rt.optical_depth(0, wet);
  VIBE_CHECK(td.size() == 1 && tw.size() == 1);
  VIBE_CHECK(tw[0] > td[0]);
}

// ===========================================================================
// 15. 偏差订正：预报的可逆性
// ===========================================================================
VIBE_TEST(obs_varbc_predict_and_correct_reversible) {
  vibe::obs::BiasCorrectionConfig cfg;
  cfg.background_variance = Real(1e8);
  cfg.enabled = true;
  vibe::obs::VariationalBiasCorrection varbc(cfg);

  vibe::obs::BiasPredictor p;
  p.scan_angle = Real(0.3);
  p.layer_thickness = Real(500);
  p.surface_type = Real(1);
  p.surface_pressure = Real(95000);
  p.cloud_amount = Real(0.2);

  Observation o = make_obs(ObsType::Radiance, VarKind::Radiance, Real(0), Real(0), Real(0));
  o.channel = 0;
  o.value = Real(250);
  o.sigma = Real(1);

  const std::vector<Real> beta = {Real(2.0), Real(-1.0), Real(0.5), Real(0.01), Real(0.2),
                                  Real(1e-5), Real(-0.5)};
  varbc.set_coefficients(ObsType::Radiance, 0, beta);

  const std::vector<Real> x = p.to_vector();
  Real expected_bias = Real(0);
  for (Size j = 0; j < x.size(); ++j) expected_bias += beta[j] * x[j];
  VIBE_CHECK_NEAR(varbc.predict(o, p), expected_bias, Real(1e-12));
  VIBE_CHECK_NEAR(varbc.correct(o, p), Real(250) - expected_bias, Real(1e-12));

  // correct_inplace 把偏差累加进 o.bias，unbiased() 即为订正值
  Observation o2 = o;
  varbc.correct_inplace(o2, p);
  VIBE_CHECK_NEAR(o2.unbiased(), Real(250) - expected_bias, Real(1e-12));
  // 归零系数后可逆恢复
  varbc.set_coefficients(ObsType::Radiance, 0, std::vector<Real>(7, Real(0)));
  VIBE_CHECK_NEAR(varbc.predict(o2, p), Real(0), Real(1e-15));

  // 加入第二个分组互不影响
  varbc.add_group(ObsType::AMV, -1);
  VIBE_CHECK(varbc.n_groups() == 2);
  VIBE_CHECK(!varbc.describe().empty());
}

// ===========================================================================
// 16. 偏差订正：Gauss-Newton 更新恢复已知系数
// ===========================================================================
VIBE_TEST(obs_varbc_update_recovers_coefficients) {
  vibe::obs::BiasCorrectionConfig cfg;
  cfg.background_variance = Real(1e10);  // 先验极弱 -> 一步 GN 近似精确
  cfg.relaxation = Real(1);
  cfg.max_iterations = 2;
  vibe::obs::VariationalBiasCorrection varbc(cfg);

  const std::vector<Real> beta_true = {Real(1.5), Real(0.4), Real(-0.2), Real(0.01),
                                       Real(0.3), Real(2e-5), Real(-0.4)};
  std::vector<vibe::obs::BiasPredictor> preds;
  std::vector<Observation> obs;
  std::vector<Real> innov;
  for (int m = 0; m < 20; ++m) {
    vibe::obs::BiasPredictor p;
    p.scan_angle = Real(0.05) * static_cast<Real>(m);
    p.layer_thickness = Real(100) + Real(37) * static_cast<Real>(m);
    p.surface_type = Real(m % 3);
    p.surface_pressure = Real(90000) + Real(700) * static_cast<Real>(m);
    p.cloud_amount = Real(0.05) * static_cast<Real>(m % 10);
    preds.push_back(p);
    Observation o = make_obs(ObsType::Radiance, VarKind::Radiance, Real(0), Real(0), Real(0));
    o.channel = 3;
    o.sigma = Real(1);
    obs.push_back(o);
    const auto x = p.to_vector();
    Real d = Real(0);
    for (Size j = 0; j < x.size(); ++j) d += beta_true[j] * x[j];
    innov.push_back(d);
  }
  ObsSpace os = make_space(std::move(obs));
  varbc.add_group(ObsType::Radiance, 3);
  varbc.update(os, innov, &preds);

  const auto* set = varbc.find(ObsType::Radiance, 3);
  VIBE_CHECK(set != nullptr);
  if (set != nullptr) {
    for (Size j = 0; j < beta_true.size(); ++j) {
      VIBE_CHECK_NEAR(set->beta[j], beta_true[j],
                      Real(1e-3) * std::max(Real(1e-3), std::abs(beta_true[j])));
    }
  }
  VIBE_CHECK(!varbc.describe().empty());
}

// ===========================================================================
// 17. 复合算子与子算子一致，且通过点积检验
// ===========================================================================
VIBE_TEST(obs_composite_operator_consistency) {
  const grid::Grid g = make_grid();
  dyn::ReferenceState ref = make_ref(g);
  dyn::State s(g);
  fill_state(s, ref);
  const auto v = view_of(s, ref, g);

  vibe::obs::CompositeOperator comp;
  comp.add(std::make_unique<vibe::obs::SoundingOperator>(g));
  comp.add(std::make_unique<vibe::obs::SurfaceOperator>(g));
  VIBE_CHECK(comp.size() == 2);

  const ObsSpace obs = make_space({
      make_obs(ObsType::Radiosonde, VarKind::T, node_x(g, 2), node_y(g, 1), g.z_center(2, 1, 1)),
      make_obs(ObsType::Surface, VarKind::PS, node_x(g, 3), node_y(g, 2), Real(0)),
      make_obs(ObsType::Radiosonde, VarKind::Q, node_x(g, 4), node_y(g, 3), g.z_center(4, 3, 2)),
  });
  std::vector<Real> yc, ys, yf;
  comp.apply(v, obs, yc);
  vibe::obs::SoundingOperator so(g);
  vibe::obs::SurfaceOperator su(g);
  so.apply(v, obs.subset(ObsType::Radiosonde), ys);
  su.apply(v, obs.subset(ObsType::Surface), yf);
  VIBE_CHECK(yc.size() == 3);
  VIBE_CHECK_NEAR(yc[0], ys[0], Real(1e-14));
  VIBE_CHECK_NEAR(yc[2], ys[1], Real(1e-14));
  VIBE_CHECK_NEAR(yc[1], yf[0], Real(1e-14));
  VIBE_CHECK(comp.check_adjoint(v, obs, 21u) < Real(1e-8));
}

// ===========================================================================
// 18. 观测空间统计：计数、时隙与描述
// ===========================================================================
VIBE_TEST(obs_space_statistics_and_slots) {
  ObsSpace os;
  os.window_length = Real(3600);
  for (int m = 0; m < 6; ++m) {
    Observation o = make_obs(m % 2 == 0 ? ObsType::Radiosonde : ObsType::Surface,
                             m % 3 == 0 ? VarKind::T : VarKind::Q, Real(m), Real(m), Real(m));
    o.time = Real(m) * Real(600);
    if (m == 5) o.qc_flag = 1;
    os.obs.push_back(o);
  }
  VIBE_CHECK(os.size() == 6);
  VIBE_CHECK(os.usable_count() == 5);
  const auto bt = os.count_by_type();
  VIBE_CHECK(bt[static_cast<Size>(ObsType::Radiosonde)] == 3);
  VIBE_CHECK(bt[static_cast<Size>(ObsType::Surface)] == 3);
  const auto bv = os.count_by_variable();
  VIBE_CHECK(bv[static_cast<Size>(VarKind::T)] == 2);
  VIBE_CHECK(bv[static_cast<Size>(VarKind::Q)] == 4);
  const auto slots = os.by_slot(3);
  VIBE_CHECK(slots.size() == 3);
  Size total = 0;
  for (const auto& sl : slots) total += sl.size();
  VIBE_CHECK(total == 6);
  VIBE_CHECK(slots[0].size() == 2);
  VIBE_CHECK(!os.describe().empty());
  VIBE_CHECK(os.subset(ObsType::Surface).size() == 3);
  const auto rinv = os.inverse_variance();
  VIBE_CHECK_NEAR(rinv[5], Real(0), Real(1e-15));
}

// ===========================================================================
// 19. CSV 写入 / 读回往返（含表头与转义）
// ===========================================================================
VIBE_TEST(obs_csv_round_trip) {
  const std::string path = "vibe_obs_unit_test.csv";
  ObsSpace os;
  os.window_start = Real(0);
  os.window_length = Real(10800);
  os.source = "unit,with-comma";
  os.valid_time = "2000-01-01T00:00:00Z";
  Observation o = make_obs(ObsType::Radiance, VarKind::Radiance, Real(1234.5), Real(-678.25),
                           Real(0), Real(245.75), Real(0.9));
  o.channel = 7;
  o.time = Real(1800);
  os.obs.push_back(o);
  os.obs.push_back(make_obs(ObsType::GnssRo, VarKind::Refractivity, Real(1), Real(2), Real(3),
                            Real(280), Real(1.5)));

  vibe::obs::write_csv(path, os);
  const ObsSpace back = vibe::obs::ObsReader::read(path, Real(10800));
  VIBE_CHECK(back.size() == 2);
  VIBE_CHECK_NEAR(back.window_length, Real(10800), Real(1e-12));
  VIBE_CHECK(back.obs[0].type == ObsType::Radiance);
  VIBE_CHECK(back.obs[0].variable == VarKind::Radiance);
  VIBE_CHECK(back.obs[0].channel == 7);
  VIBE_CHECK_NEAR(back.obs[0].value, Real(245.75), Real(1e-12));
  VIBE_CHECK_NEAR(back.obs[0].time, Real(1800), Real(1e-12));
  VIBE_CHECK(back.obs[1].type == ObsType::GnssRo);
  VIBE_CHECK(back.obs[1].variable == VarKind::Refractivity);
  VIBE_CHECK_NEAR(back.obs[1].sigma, Real(1.5), Real(1e-12));
  std::remove(path.c_str());
  VIBE_CHECK(back.describe().find("radiance") != std::string::npos);
}

// ===========================================================================
// 20. 观测类型 / 变量字符串解析
// ===========================================================================
VIBE_TEST(obs_type_string_parsing) {
  VIBE_CHECK(vibe::obs::obs_type_from_string("radiosonde") == ObsType::Radiosonde);
  VIBE_CHECK(vibe::obs::obs_type_from_string("GNSS-RO") == ObsType::GnssRo);
  VIBE_CHECK(vibe::obs::obs_type_from_string("radar") == ObsType::RadarReflectivity);
  VIBE_CHECK(vibe::obs::var_kind_from_string("refractivity") == VarKind::Refractivity);
  VIBE_CHECK(vibe::obs::var_kind_from_string("PS") == VarKind::PS);
  VIBE_CHECK(std::string(vibe::obs::to_string(ObsType::Scatterometer)) == "scatterometer");
  VIBE_CHECK(std::string(vibe::obs::to_string(VarKind::Radiance)) == "radiance");
}

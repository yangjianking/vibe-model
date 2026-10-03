/// @file test_physics.cpp
/// @brief 物理参数化模块单元测试（内置 vibe::test 框架）。
///
/// 覆盖：
///   * Bolton 饱和水汽压在本征点/常温点的已知值
///   * 饱和调整的收敛性（RH = 1）、水物质守恒与正定
///   * Kessler 无云不变、自动转换阈值、质量守恒、降水正定、雨沉降
///   * Thompson 守恒、Bergeron 冰相生成、极端条件下的正定
///   * Monin-Obukhov 中性对数廓线、迭代收敛、稳定/不稳定分支符号
///   * 陆面能量平衡闭合（残差 < 1e-6 W/m^2）
///   * YSU / MYJ 边界层：PBL 高度、扩散系数、TKE 正定有界
///   * 积云：稳定层结不触发；条件性不稳定层结触发且水物质收支闭合
///   * 辐射：夜间短波为 0、Planck 累积函数已知值、太阳几何、白天短波非零
///   * 工厂与配置映射、CFL 诊断
///
/// 运行：VIBE_TEST 自动注册，main 中调用 vibe::test::run_all()。

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "vibe/common/test.hpp"
#include "vibe/common/types.hpp"
#include "vibe/physics/cumulus.hpp"
#include "vibe/physics/microphysics.hpp"
#include "vibe/physics/pbl.hpp"
#include "vibe/physics/physics_driver.hpp"
#include "vibe/physics/physics_types.hpp"
#include "vibe/physics/radiation.hpp"
#include "vibe/physics/surface.hpp"

using vibe::Real;
using namespace vibe::physics;

namespace {

/// 饱和列（qv = q_s，可选给定云水），用于微物理测试
PhysicsColumn make_saturated_column(Int nz = 40, Real z_top = Real(20000),
                                    Real t_sfc = Real(292), Real qc = Real(0),
                                    Real qr = Real(0)) {
  PhysicsColumn col = ideal_sounding(nz, z_top, kP0, t_sfc, Real(0.014), Real(6.5e-3), Real(2.0e-3));
  for (Int k = 0; k < nz; ++k) {
    const Size s = static_cast<Size>(k);
    const Real t = col.temperature(k);
    col.qv[s] = saturation_mixing_ratio(col.p[s], t);
    col.qc[s] = qc;
    col.qr[s] = qr;
    col.qi[s] = Real(0);
    col.qs[s] = Real(0);
    col.qg[s] = Real(0);
    col.rho[s] = density_from_pressure_tv(col.p[s], virtual_temperature(t, col.qv[s], qc));
  }
  return col;
}

/// 干列（远离饱和）
PhysicsColumn make_dry_column(Int nz = 40, Real qv_sfc = Real(0.001)) {
  PhysicsColumn col = ideal_sounding(nz, Real(20000), kP0, Real(292), qv_sfc, Real(6.5e-3), Real(2.0e-3));
  col.diagnose_density();
  return col;
}

/// 五层土壤 + 地表状态（陆面）
SurfaceState make_land_surface_state(Real t_soil = Real(290)) {
  SurfaceState sfc;
  sfc.resize_soil(5);
  for (Size i = 0; i < sfc.soil_temperature.size(); ++i) {
    sfc.soil_temperature[i] = t_soil - Real(0.5) * static_cast<Real>(i);
    sfc.soil_moisture[i] = Real(0.25);
  }
  sfc.land_fraction = Real(1);
  sfc.roughness = Real(0.1);
  sfc.albedo = Real(0.2);
  sfc.emissivity = Real(0.98);
  sfc.skin_temperature = t_soil;
  sfc.surface_pressure = kP0;
  sfc.soil_field_capacity = Real(0.3);
  sfc.soil_wilting_point = Real(0.1);
  return sfc;
}

}  // namespace

// ===========================================================================
// 1. 热力学与饱和水汽压
// ===========================================================================

VIBE_TEST(physics_saturation_vapor_pressure_bolton_known_values) {
  // Bolton (1980)： e_s(0 C) = 611.2 Pa；e_s(20 C) ≈ 2337 Pa
  const Real es0 = saturation_vapor_pressure(kT0);
  const Real es20 = saturation_vapor_pressure(kT0 + Real(20));
  VIBE_CHECK_NEAR(es0, 611.2, 0.5);
  VIBE_CHECK_NEAR(es20, 2337.0, 10.0);
  // 单调递增
  VIBE_CHECK(saturation_vapor_pressure(kT0 + Real(30)) > es20);
  VIBE_CHECK(es20 > es0);
  // 冰面饱和水汽压低于液面
  const Real ei = saturation_vapor_pressure_ice(kT0 - Real(10));
  const Real ew = saturation_vapor_pressure(kT0 - Real(10));
  VIBE_CHECK(ei > Real(0));
  VIBE_CHECK(ei < ew);
}

VIBE_TEST(physics_saturation_mixing_ratio_and_rh) {
  const Real p = Real(90000);
  const Real t = Real(285);
  const Real qs = saturation_mixing_ratio(p, t);
  VIBE_CHECK(qs > Real(0));
  VIBE_CHECK(qs < Real(0.05));
  VIBE_CHECK_NEAR(relative_humidity_from_qv(p, t, qs), Real(1), Real(1.0e-12));
  VIBE_CHECK_NEAR(relative_humidity_from_qv(p, t, Real(0.5) * qs), Real(0.5), Real(1.0e-12));
  // 露点：饱和时露点等于温度
  VIBE_CHECK_NEAR(dewpoint_temperature(p, qs), t, Real(0.05));
}

VIBE_TEST(physics_exner_and_virtual_temperature) {
  VIBE_CHECK_NEAR(exner_from_pressure(kP0), Real(1), Real(1.0e-12));
  const Real t = Real(280);
  const Real p = Real(85000);
  const Real theta = theta_from_temperature_pressure(t, p);
  VIBE_CHECK_NEAR(temperature_from_theta_pi(theta, exner_from_pressure(p)), t, Real(1.0e-10));
  // 虚温 > 温度（水汽），< 温度（凝结态水负载）
  VIBE_CHECK(virtual_temperature(t, Real(0.01)) > t);
  VIBE_CHECK(virtual_temperature(t, Real(0), Real(0.01)) < t);
}

VIBE_TEST(physics_ideal_sounding_hydrostatic_and_finite) {
  const PhysicsColumn col = ideal_sounding(40, Real(20000), kP0, Real(288), Real(0.012));
  VIBE_CHECK(col.consistent());
  VIBE_CHECK(col.finite());
  VIBE_CHECK(col.nz() == 40);
  // 气压单调递减、位温随高度增加
  for (Int k = 1; k < col.nz(); ++k) {
    VIBE_CHECK(col.p[static_cast<Size>(k)] < col.p[static_cast<Size>(k - 1)]);
    VIBE_CHECK(col.theta[static_cast<Size>(k)] > col.theta[static_cast<Size>(k - 1)]);
  }
  // 静力平衡： dp/dz = -rho g（离散相对误差 < 1%）
  const Size s = 10;
  const Real dpdz = (col.p[s + 1] - col.p[s - 1]) / (col.z[s + 1] - col.z[s - 1]);
  const Real hydro = -col.rho[s] * kGravity;
  VIBE_CHECK_REL(dpdz, hydro, Real(0.02));
}

VIBE_TEST(physics_column_water_paths_and_cloud_fraction) {
  PhysicsColumn col = make_saturated_column(40, Real(20000), Real(292), Real(5.0e-4));
  col.diagnose_layer_depth();
  VIBE_CHECK(col.water_path() > Real(0));
  VIBE_CHECK(col.liquid_water_path() > Real(0));
  VIBE_CHECK(col.ice_water_path() >= Real(0));
  for (Int k = 0; k < col.nz(); ++k) {
    const Real cf = col.cloud_fraction(k);
    VIBE_CHECK(cf >= Real(0) && cf <= Real(1));
  }
  VIBE_CHECK(col.positive_definite());
}

// ===========================================================================
// 2. 饱和调整
// ===========================================================================

VIBE_TEST(physics_saturation_adjust_condenses_to_rh_one) {
  PhysicsColumn col = make_dry_column(20);
  // 强制过饱和
  for (Int k = 0; k < 20; ++k) {
    const Size s = static_cast<Size>(k);
    col.qv[s] = Real(1.5) * saturation_mixing_ratio(col.p[s], col.temperature(k));
    col.qc[s] = Real(0);
  }
  const Real w_before = col.total_water_mass();
  const std::vector<Real> th_before = col.theta;
  const SaturationAdjust r = saturation_adjust_column(col, 0, col.nz(), false);
  VIBE_CHECK(r.condensed > Real(0));
  VIBE_CHECK(r.iterations >= 1);
  VIBE_CHECK(r.iterations <= 10);
  VIBE_CHECK(r.residual < Real(1.0e-6));
  VIBE_CHECK_REL(col.total_water_mass(), w_before, Real(1.0e-12));
  for (Int k = 0; k < 20; ++k) {
    const Size s = static_cast<Size>(k);
    VIBE_CHECK(col.qc[s] > Real(0));
    VIBE_CHECK(std::abs(col.relative_humidity(k) - Real(1)) < Real(1.0e-6));
    VIBE_CHECK(col.theta[s] > th_before[s]);  // 凝结释放潜热
  }
}

VIBE_TEST(physics_saturation_adjust_evaporates_cloud) {
  PhysicsColumn col = make_dry_column(20, Real(0.002));
  for (Int k = 0; k < 20; ++k) {
    const Size s = static_cast<Size>(k);
    col.qv[s] = Real(0.3) * saturation_mixing_ratio(col.p[s], col.temperature(k));
    col.qc[s] = Real(1.0e-3);
  }
  const Real w_before = col.total_water_mass();
  const SaturationAdjust r = saturation_adjust_column(col, 0, col.nz(), false);
  VIBE_CHECK(r.condensed < Real(0));  // 净蒸发
  VIBE_CHECK_REL(col.total_water_mass(), w_before, Real(1.0e-12));
  for (Int k = 0; k < 20; ++k) {
    const Real rh = col.relative_humidity(k);
    // 云水足够时达到饱和；不足时留有一部分云水
    VIBE_CHECK(rh <= Real(1) + Real(1.0e-6));
    VIBE_CHECK(col.qv[static_cast<Size>(k)] >= Real(0));
    VIBE_CHECK(col.qc[static_cast<Size>(k)] >= Real(0));
  }
}

// ===========================================================================
// 3. Kessler 暖雨
// ===========================================================================

VIBE_TEST(physics_kessler_no_cloud_no_change) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  KesslerMicrophysics mp(opt);
  PhysicsColumn col = make_dry_column(30);
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  const std::vector<Real> th0 = col.theta, qv0 = col.qv;
  const Real w0 = col.total_water_mass();
  const Real precip = mp.step_column(col, Real(60), diag);
  VIBE_CHECK_NEAR(precip, Real(0), Real(1.0e-14));
  for (Int k = 0; k < 30; ++k) {
    const Size s = static_cast<Size>(k);
    VIBE_CHECK_NEAR(col.theta[s], th0[s], Real(1.0e-12));
    VIBE_CHECK_NEAR(col.qv[s], qv0[s], Real(1.0e-14));
    VIBE_CHECK_NEAR(col.qc[s], Real(0), Real(1.0e-14));
    VIBE_CHECK_NEAR(col.qr[s], Real(0), Real(1.0e-14));
  }
  VIBE_CHECK_REL(col.total_water_mass(), w0, Real(1.0e-12));
  VIBE_CHECK(diag.mass_conservation_residual < Real(1.0e-10));
}

VIBE_TEST(physics_kessler_autoconversion_threshold) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  KesslerMicrophysics mp(opt);
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  // 云水低于阈值 qc0：不产生雨水
  PhysicsColumn low = make_saturated_column(30, Real(20000), Real(292), Real(0.5e-3));
  mp.step_column(low, Real(10), diag);
  Real qr_low = Real(0);
  for (Int k = 0; k < low.nz(); ++k) qr_low += low.qr[static_cast<Size>(k)];
  VIBE_CHECK_NEAR(qr_low, Real(0), Real(1.0e-16));
  // 云水高于阈值：产生雨水
  PhysicsColumn high = make_saturated_column(30, Real(20000), Real(292), Real(5.0e-3));
  mp.step_column(high, Real(10), diag);
  Real qr_high = Real(0);
  for (Int k = 0; k < high.nz(); ++k) qr_high += high.qr[static_cast<Size>(k)];
  VIBE_CHECK(qr_high > Real(0));
}

VIBE_TEST(physics_kessler_mass_conservation_and_positivity) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  KesslerMicrophysics mp(opt);
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  PhysicsColumn col = make_saturated_column(40, Real(20000), Real(295), Real(4.0e-3), Real(1.0e-3));
  const Real w0 = col.total_water_mass();
  const Real precip = mp.step_column(col, Real(60), diag);
  const Real w1 = col.total_water_mass();
  VIBE_CHECK(precip >= Real(0));
  // 整列水物质 + 降水 = 初始水物质（相对误差 < 1e-10）
  VIBE_CHECK_REL(w1 + precip * Real(60), w0, Real(1.0e-10));
  VIBE_CHECK(diag.mass_conservation_residual < Real(1.0e-10));
  VIBE_CHECK(col.positive_definite());
  VIBE_CHECK(col.finite());
}

VIBE_TEST(physics_kessler_rain_sedimentation_reaches_ground) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  KesslerMicrophysics mp(opt);
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  PhysicsColumn col = make_saturated_column(30, Real(15000), Real(292), Real(0), Real(2.0e-3));
  const Real w0 = col.total_water_mass();
  const Real precip = mp.step_column(col, Real(600), diag);
  VIBE_CHECK(precip > Real(0));
  VIBE_CHECK_REL(col.total_water_mass() + precip * Real(600), w0, Real(1.0e-10));
  // 落速随雨水量增大而增大
  VIBE_CHECK(mp.rain_fall_speed(Real(1.0), Real(1.0e-3)) > Real(1.0));
  VIBE_CHECK(mp.rain_fall_speed(Real(1.0), Real(1.0e-2)) >
             mp.rain_fall_speed(Real(1.0), Real(1.0e-3)));
  VIBE_CHECK_NEAR(mp.autoconversion(Real(0.5e-3)), Real(0), Real(1.0e-20));
  VIBE_CHECK(mp.autoconversion(Real(3.0e-3)) > Real(0));
}

// ===========================================================================
// 4. Thompson 6 类
// ===========================================================================

VIBE_TEST(physics_thompson_mass_conservation) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  opt.microphysics = MicrophysicsScheme::Thompson;
  ThompsonMicrophysics mp(opt);
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  PhysicsColumn col = make_saturated_column(40, Real(20000), Real(295), Real(3.0e-3), Real(1.0e-3));
  const Real w0 = col.total_water_mass();
  const Real precip = mp.step_column(col, Real(60), diag);
  const Real w1 = col.total_water_mass();
  VIBE_CHECK(precip >= Real(0));
  VIBE_CHECK_REL(w1 + precip * Real(60), w0, Real(1.0e-8));
  VIBE_CHECK(diag.mass_conservation_residual < Real(1.0e-8));
  VIBE_CHECK(col.positive_definite());
  VIBE_CHECK(col.finite());
}

VIBE_TEST(physics_thompson_bergeron_ice_production) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  ThompsonMicrophysics mp(opt);
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  // 冷云：T < 0 C，冰面过饱和，含云水
  PhysicsColumn col = make_saturated_column(30, Real(12000), Real(268), Real(0));
  for (Int k = 0; k < 8; ++k) {
    const Size s = static_cast<Size>(k);
    const Real t = col.temperature(k);
    VIBE_CHECK(t < kT0);
    // 相对冰面过饱和（水汽 > qsi）
    col.qv[s] = Real(1.2) * saturation_mixing_ratio(col.p[s], t, true);
    col.qc[s] = Real(1.0e-3);
  }
  Real qc0 = Real(0);
  for (Int k = 0; k < 8; ++k) qc0 += col.qc[static_cast<Size>(k)];
  mp.step_column(col, Real(60), diag);
  Real qi1 = Real(0);
  for (Int k = 0; k < 8; ++k) qi1 += col.qi[static_cast<Size>(k)];
  VIBE_CHECK(qi1 > Real(0));  // Bergeron/凝华生成云冰
  VIBE_CHECK(col.positive_definite());
  VIBE_CHECK(mp.bergeron_rate(Real(1.1) * Real(1.0e-3), Real(1.0e-3)) > Real(0));
  VIBE_CHECK_NEAR(mp.bergeron_rate(Real(0.5e-3), Real(1.0e-3)), Real(0), Real(1.0e-20));
  VIBE_UNUSED(qc0);
}

VIBE_TEST(physics_thompson_extreme_values_stay_positive) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  ThompsonMicrophysics mp(opt);
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  PhysicsColumn col = make_saturated_column(10, Real(6000), Real(300), Real(0.02), Real(0.01));
  for (Int k = 0; k < 10; ++k) {
    const Size s = static_cast<Size>(k);
    col.qi[s] = Real(5.0e-3);
    col.qs[s] = Real(5.0e-3);
    col.qg[s] = Real(5.0e-3);
  }
  const Real precip = mp.step_column(col, Real(120), diag);
  VIBE_CHECK(std::isfinite(precip));
  VIBE_CHECK(col.positive_definite());
  VIBE_CHECK(col.finite());
  for (Int k = 0; k < 10; ++k) {
    const Size s = static_cast<Size>(k);
    VIBE_CHECK(col.qv[s] >= Real(0));
    VIBE_CHECK(col.qc[s] >= Real(0));
    VIBE_CHECK(col.qr[s] >= Real(0));
    VIBE_CHECK(col.qi[s] >= Real(0));
    VIBE_CHECK(col.qs[s] >= Real(0));
    VIBE_CHECK(col.qg[s] >= Real(0));
  }
  // 落速单调性
  VIBE_CHECK(mp.graupel_fall_speed(Real(1.0), Real(5.0e-3)) >
             mp.graupel_fall_speed(Real(1.0), Real(1.0e-3)));
  VIBE_CHECK(mp.snow_fall_speed(Real(1.0), Real(1.0e-3)) > Real(0));
  VIBE_CHECK(mp.rain_fall_speed(Real(1.0), Real(1.0e-3)) > Real(0));
}

// ===========================================================================
// 5. 地面层与陆面
// ===========================================================================

VIBE_TEST(physics_monin_obukhov_neutral_log_profile) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  MoninObukhov mo(opt);
  // 中性：位温与湿度随高度不变，且地表值与最低层一致
  PhysicsColumn col = make_dry_column(20, Real(0.0));
  for (Int k = 0; k < col.nz(); ++k) {
    const Size s = static_cast<Size>(k);
    col.theta[s] = Real(300);
    col.qv[s] = Real(0.010);
    col.u[s] = Real(5);
    col.v[s] = Real(0);
  }
  SurfaceState sfc = make_land_surface_state(Real(290));
  sfc.surface_pressure = col.p[0];
  sfc.roughness = Real(0.1);
  sfc.skin_temperature = Real(300) * exner_from_pressure(col.p[0]);  // theta_s = theta_1
  sfc.soil_moisture[0] = sfc.soil_field_capacity;                     // beta = 1
  // 令 q1 = q_sat(p1, T1)：中性湿度廓线
  col.qv[0] = saturation_mixing_ratio(col.p[0], sfc.skin_temperature);
  for (Int k = 1; k < col.nz(); ++k) col.qv[static_cast<Size>(k)] = col.qv[0];
  const SurfaceFluxes f = mo.solve(col, sfc, Real(10));
  const Real z1 = col.z[0] - sfc.terrain_height;
  const Real ustar_expected = kVonKarman * Real(5) / std::log(z1 / sfc.roughness);
  VIBE_CHECK_NEAR(f.ustar, ustar_expected, Real(1.0e-6));
  VIBE_CHECK(std::abs(f.obukhov) > Real(1.0e4));  // 中性：L -> 无穷
  VIBE_CHECK_NEAR(f.tstar, Real(0), Real(1.0e-10));
  VIBE_CHECK_NEAR(f.sensible, Real(0), Real(1.0e-8));
  // 中性交换系数 Cd = kappa^2/ln(z/z0)^2
  VIBE_CHECK_REL(f.cd, MoninObukhov::neutral_exchange(z1, sfc.roughness), Real(1.0e-6));
  // 对数风廓线： u(z) = (u*/kappa) ln(z/z0)
  VIBE_CHECK_REL(MoninObukhov::log_wind(z1, sfc.roughness, f.ustar), Real(5), Real(1.0e-9));
}

VIBE_TEST(physics_monin_obukhov_iteration_converges) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  MoninObukhov mo(opt);
  PhysicsColumn col = make_dry_column(20, Real(0.001));
  for (Int k = 0; k < col.nz(); ++k) {
    col.u[static_cast<Size>(k)] = Real(3);
    col.v[static_cast<Size>(k)] = Real(1);
  }
  SurfaceState sfc = make_land_surface_state(Real(292));
  sfc.surface_pressure = col.p[0];
  // 不稳定：地表比空气暖
  sfc.skin_temperature = col.temperature(0) + Real(6);
  const SurfaceFluxes fu = mo.solve(col, sfc, Real(10));
  VIBE_CHECK(fu.converged);
  VIBE_CHECK(fu.iterations <= opt.mo_max_iterations);
  VIBE_CHECK(fu.residual < Real(1.0e-6));
  VIBE_CHECK(fu.obukhov < Real(0));
  VIBE_CHECK(fu.sensible > Real(0));  // 向上感热
  VIBE_CHECK(fu.ustar > Real(0));
  VIBE_CHECK(std::isfinite(fu.t2) && std::isfinite(fu.q2));
  VIBE_CHECK(fu.t2 > Real(200) && fu.t2 < Real(340));
  VIBE_CHECK(mo.last_iterations() == fu.iterations);

  // 稳定：地表比空气冷
  SurfaceState sfc2 = make_land_surface_state(Real(285));
  sfc2.surface_pressure = col.p[0];
  sfc2.skin_temperature = col.temperature(0) - Real(6);
  const SurfaceFluxes fs = mo.solve(col, sfc2, Real(10));
  VIBE_CHECK(fs.converged);
  VIBE_CHECK(fs.obukhov > Real(0));
  VIBE_CHECK(fs.sensible < Real(0));  // 向下感热
  VIBE_CHECK(fs.ustar < fu.ustar);    // 稳定层结抑制混合
  // 稳定度函数符号
  VIBE_CHECK(MoninObukhov::psi_m(Real(0.5)) < Real(0));
  VIBE_CHECK(MoninObukhov::psi_m(Real(-0.5)) > Real(0));
  VIBE_CHECK(MoninObukhov::phi_m(Real(-1)) < Real(1));
  VIBE_CHECK(MoninObukhov::phi_m(Real(1)) > Real(1));
  VIBE_CHECK_NEAR(MoninObukhov::phi_m(Real(0)), Real(1), Real(1.0e-12));
}

VIBE_TEST(physics_surface_energy_balance_closes) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  opt.surface = SurfaceScheme::Noah;
  MoninObukhov mo(opt);
  LandSurface land(opt);
  PhysicsColumn col = make_dry_column(20, Real(0.008));
  SurfaceState sfc = make_land_surface_state(Real(291));
  sfc.surface_pressure = col.p[0];
  sfc.skin_temperature = col.temperature(0) + Real(2);
  sfc.sw_down = Real(500);
  sfc.lw_down = Real(320);
  SurfaceFluxes flx = mo.solve(col, sfc, Real(10));
  ColumnTendency td;
  td.resize(col.nz());
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  land.step_column(col, sfc, flx, Real(10), td, diag);
  const Real residual = LandSurface::energy_balance_residual(sfc);
  VIBE_CHECK(std::abs(residual) < Real(1.0e-6));
  VIBE_CHECK(sfc.ground_heat_flux != Real(0));
  // 与土壤热传导的一致性（Newton 收敛质量）
  const Real g_soil = opt.soil_thermal_conductivity * (sfc.skin_temperature - sfc.soil_temperature[0]) /
                      (Real(0.5) * sfc.soil_depth[0]);
  VIBE_CHECK(std::abs(sfc.ground_heat_flux - g_soil) < Real(1.0e-3));
  // 土壤温度被更新并有合理范围
  for (Size i = 0; i < sfc.soil_temperature.size(); ++i) {
    VIBE_CHECK(sfc.soil_temperature[i] > Real(200) && sfc.soil_temperature[i] < Real(350));
    VIBE_CHECK(sfc.soil_moisture[i] >= Real(0) && sfc.soil_moisture[i] <= sfc.soil_field_capacity);
  }
  VIBE_CHECK(diag.energy_balance_residual_max < Real(1.0e-6));
  // 土壤湿度可用度函数
  VIBE_CHECK_NEAR(LandSurface::moisture_availability(Real(0.15), Real(0.3), Real(0.1)), Real(0.5),
                  Real(1.0e-12));
  VIBE_CHECK_NEAR(LandSurface::moisture_availability(Real(0.3), Real(0.3), Real(0.1)), Real(1),
                  Real(1.0e-12));
}

VIBE_TEST(physics_sea_surface_charnock) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  MoninObukhov mo(opt);
  PhysicsColumn col = make_dry_column(20, Real(0.012));
  SurfaceState sfc = make_land_surface_state(Real(291));
  sfc.land_fraction = Real(0);  // 海面
  sfc.surface_pressure = col.p[0];
  sfc.skin_temperature = opt.sst;
  const SurfaceFluxes f = mo.solve(col, sfc, Real(10));
  VIBE_CHECK(sfc.is_water());
  VIBE_CHECK(f.ustar > Real(0));
  VIBE_CHECK(sfc.roughness >= Real(0));
  const Real z0a = MoninObukhov::charnock_roughness(Real(0.2), opt.charnock_alpha);
  const Real z0b = MoninObukhov::charnock_roughness(Real(0.6), opt.charnock_alpha);
  VIBE_CHECK(z0b > z0a);  // Charnock：风越大粗糙度越大
}

// ===========================================================================
// 6. 边界层
// ===========================================================================

VIBE_TEST(physics_ysu_pbl_height_and_diffusivity) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  opt.pbl = PblScheme::Ysu;
  MoninObukhov mo(opt);
  YsuPbl pbl(opt);
  PhysicsColumn col = make_dry_column(40, Real(0.010));
  for (Int k = 0; k < col.nz(); ++k) col.u[static_cast<Size>(k)] = Real(4);
  SurfaceState sfc = make_land_surface_state(Real(294));
  sfc.surface_pressure = col.p[0];
  sfc.skin_temperature = col.temperature(0) + Real(4);
  const SurfaceFluxes f = mo.solve(col, sfc, Real(10));
  VIBE_CHECK(f.sensible > Real(0));
  ColumnTendency td;
  td.resize(col.nz());
  const Real h = pbl.step_column(col, sfc, f, Real(10), td);
  VIBE_CHECK(h >= Real(100));
  VIBE_CHECK(h < Real(6000));
  VIBE_CHECK(pbl.max_diffusivity() > Real(0));
  VIBE_CHECK(td.finite());
  VIBE_CHECK(col.finite());
  // 诊断高度落于合理量级（此处 col 已被扩散更新，允许与 step 返回值不同）
  const Real h_diag = pbl.diagnose_height(col, sfc, f);
  VIBE_CHECK(h_diag >= Real(50) && h_diag < Real(6000));
  // K 廓线：PBL 内为正值，顶部趋零
  VIBE_CHECK_NEAR(pbl.diffusivity_momentum(Real(1), h, h), Real(0), Real(1.0e-12));
  VIBE_CHECK(pbl.diffusivity_momentum(Real(1), Real(0.5) * h, h) > Real(0));
  VIBE_CHECK(pbl.counter_gradient(Real(0.1), Real(1), Real(1000)) > Real(0));
  VIBE_CHECK(pbl.prandtl(Real(0.5) * h, h) > Real(1));
}

VIBE_TEST(physics_myj_tke_positive_and_pbl_height) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  opt.pbl = PblScheme::Myj;
  MoninObukhov mo(opt);
  MyjPbl pbl(opt);
  PhysicsColumn col = make_dry_column(40, Real(0.010));
  for (Int k = 0; k < col.nz(); ++k) {
    col.u[static_cast<Size>(k)] = Real(4) + Real(0.05) * static_cast<Real>(k);
    col.tke[static_cast<Size>(k)] = Real(0.2);
  }
  SurfaceState sfc = make_land_surface_state(Real(294));
  sfc.surface_pressure = col.p[0];
  sfc.skin_temperature = col.temperature(0) + Real(4);
  const SurfaceFluxes f = mo.solve(col, sfc, Real(10));
  ColumnTendency td;
  td.resize(col.nz());
  const Real h = pbl.step_column(col, sfc, f, Real(10), td);
  VIBE_CHECK(h > Real(0));
  VIBE_CHECK(pbl.max_diffusivity() > Real(0));
  VIBE_CHECK(td.finite());
  // TKE 正定且有界
  for (Int k = 0; k < col.nz(); ++k) {
    VIBE_CHECK(col.tke[static_cast<Size>(k)] >= Real(0));
    VIBE_CHECK(col.tke[static_cast<Size>(k)] < Real(1000));
  }
  // 稳定性函数：中性附近 S_M > 0, S_H > 0；强稳定时 S_H 减小
  // 稳定性函数（取 l/q = 10、GM = 1e-4 的典型量级）
  Real sm0 = 0, sh0 = 0, sm1 = 0, sh1 = 0;
  pbl.stability_functions(Real(1.0e-4), Real(1.0e-9), Real(10), sm0, sh0);
  pbl.stability_functions(Real(1.0e-4), Real(1.0e-3), Real(10), sm1, sh1);
  VIBE_CHECK(sm0 > Real(0) && sh0 > Real(0));
  VIBE_CHECK(sm1 > Real(0) && sh1 > Real(0));
  VIBE_CHECK(sh1 < sh0);
  VIBE_CHECK(sm1 < sm0);
  // 平衡混合长有限且为正
  const Real elm = pbl.equilibrium_length(Real(1.0e-5), Real(0), Real(1.0));
  VIBE_CHECK(elm >= Real(0.3) && elm <= opt.myj_el0max + Real(1.0e-9));
  // 混合长廓线：PBL 内为正、随高度先增（Blackadar 型）后受 ELM 限制
  std::vector<Real> el;
  pbl.mixing_length(col, pbl.pbl_top_index(col), el);
  VIBE_CHECK(static_cast<Int>(el.size()) == col.nz());
  for (Real e : el) VIBE_CHECK(e >= Real(0.3));
  VIBE_CHECK(pbl.pbl_top_index(col) >= 0);
}

// ===========================================================================
// 7. 积云
// ===========================================================================

VIBE_TEST(physics_cumulus_kf_no_trigger_in_stable_profile) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  opt.cumulus = CumulusScheme::KainFritsch;
  KainFritschCumulus kf(opt);
  // 极稳定 + 干：未饱和气块无法到达 LFC
  PhysicsColumn col = ideal_sounding(40, Real(20000), kP0, Real(300), Real(0.002), Real(3.0e-3),
                                     Real(1.0e-3), Real(2));
  col.diagnose_density();
  SurfaceState sfc = make_land_surface_state(Real(290));
  sfc.surface_pressure = col.p[0];
  sfc.skin_temperature = col.temperature(0);
  ColumnTendency td;
  td.resize(col.nz());
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  const Real precip = kf.step_column(col, sfc, Real(600), td, diag);
  VIBE_CHECK_NEAR(precip, Real(0), Real(1.0e-14));
  VIBE_CHECK(!kf.last_diagnostics().triggered);
  VIBE_CHECK_NEAR(td.max_abs(), Real(0), Real(1.0e-14));
  // 气块抬升：稳定层结下没有 LFC
  ParcelAscent ascent;
  Int src = -1;
  VIBE_CHECK(!kf.trigger(col, sfc, ascent, src));
}

VIBE_TEST(physics_cumulus_kf_trigger_and_water_budget) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  opt.cumulus = CumulusScheme::KainFritsch;
  KainFritschCumulus kf(opt);
  // 条件性不稳定 + 湿：lapse 9 K/km，地面湿润
  PhysicsColumn col = ideal_sounding(40, Real(20000), kP0, Real(301), Real(0.017), Real(9.0e-3),
                                     Real(1.0e-3), Real(2));
  col.diagnose_density();
  SurfaceState sfc = make_land_surface_state(Real(295));
  sfc.surface_pressure = col.p[0];
  sfc.skin_temperature = col.temperature(0);
  sfc.land_fraction = Real(1);
  ParcelAscent ascent;
  Int src = -1;
  VIBE_CHECK(kf.trigger(col, sfc, ascent, src));
  VIBE_CHECK(ascent.cape > Real(0));
  ColumnTendency td;
  td.resize(col.nz());
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  const Real precip = kf.step_column(col, sfc, Real(1800), td, diag);
  const ConvectionDiagnostics& d = kf.last_diagnostics();
  VIBE_CHECK(d.triggered);
  VIBE_CHECK(d.cape > Real(0));
  VIBE_CHECK(d.mass_flux_base > Real(0));
  VIBE_CHECK(d.cloud_top_level > d.cloud_base_level);
  VIBE_CHECK(d.cloud_top_height > d.cloud_base_height);
  VIBE_CHECK(precip >= Real(0));
  VIBE_CHECK(std::isfinite(precip));
  VIBE_CHECK(td.finite());
  // 云层内应有净增温（补偿下沉 + 潜热）
  VIBE_CHECK(d.heating_max > Real(0));
  // 水物质收支闭合： 降水通量 == -整列水汽变率
  Real col_change = Real(0);
  for (Int k = 0; k < col.nz(); ++k) {
    const Size s = static_cast<Size>(k);
    col_change += td.qv[s] * col.rho[s] * col.dz[s];
  }
  VIBE_CHECK_REL(precip, -col_change, Real(1.0e-8));
  VIBE_CHECK(diag.mass_conservation_residual < Real(1.0e-8));
}

VIBE_TEST(physics_grell_devenyi_ensemble_framework) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  opt.cumulus = CumulusScheme::GrellDevenyi;
  opt.gd_ensemble_size = 4;
  GrellDevenyiEnsemble gd(opt);
  VIBE_CHECK(gd.member_count() == 4);
  // 成员扰动应彼此不同且保持正值
  PhysicsOptions a, b;
  gd.perturbed_options(0, a);
  gd.perturbed_options(3, b);
  VIBE_CHECK(a.kf_entrainment_rate != b.kf_entrainment_rate);
  VIBE_CHECK(a.kf_entrainment_rate > Real(0));
  VIBE_CHECK(b.kf_detrainment_rate > Real(0));
  VIBE_CHECK(a.kf_closure_time > Real(0) && b.kf_closure_time > Real(0));
  // 集合积分：结果有限、倾向有限
  PhysicsColumn col = ideal_sounding(30, Real(16000), kP0, Real(300), Real(0.016), Real(9.0e-3),
                                     Real(1.0e-3), Real(2));
  col.diagnose_density();
  SurfaceState sfc = make_land_surface_state(Real(295));
  sfc.surface_pressure = col.p[0];
  sfc.skin_temperature = col.temperature(0);
  ColumnTendency td;
  td.resize(col.nz());
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  const Real precip = gd.step_column(col, sfc, Real(1800), td, diag);
  VIBE_CHECK(std::isfinite(precip));
  VIBE_CHECK(precip >= Real(0));
  VIBE_CHECK(td.finite());
}

// ===========================================================================
// 8. 辐射
// ===========================================================================

VIBE_TEST(physics_radiation_planck_and_bands) {
  // 累积 Planck 函数：Wien 峰值处约 0.2495，中值约 0.494（x = lambda T）
  const Real t = Real(300);
  const Real f_peak = RadiationDriver::planck_cumulative_fraction(Real(2898) / t, t);
  const Real f_median = RadiationDriver::planck_cumulative_fraction(Real(4100) / t, t);
  VIBE_CHECK_NEAR(f_peak, Real(0.2495), Real(1.0e-3));
  VIBE_CHECK_NEAR(f_median, Real(0.4942), Real(2.0e-3));
  VIBE_CHECK(RadiationDriver::planck_cumulative_fraction(Real(1.0e6), t) > Real(0.999));
  VIBE_CHECK_NEAR(RadiationDriver::planck_cumulative_fraction(Real(1.0e-3), t), Real(0), Real(1.0e-6));
  VIBE_CHECK_NEAR(RadiationDriver::planck_flux(t), kStefanBoltzmann * t * t * t * t, Real(1.0e-9));

  PhysicsOptions opt = PhysicsOptions::defaults();
  RadiationDriver rad(RadiationScheme::Rrtmg, opt);
  rad.initialize(nullptr);
  VIBE_CHECK(rad.initialized());
  VIBE_CHECK(rad.longwave_bands() == opt.n_longwave_bands);
  VIBE_CHECK(rad.shortwave_bands() == opt.n_shortwave_bands);
  VIBE_CHECK(rad.g_points() == opt.n_g_points);
  // 谱带份额之和 ≈ 1（长波 3.08-1000 um，短波 0.2-12.2 um）
  Real lw_sum = Real(0), sw_sum = Real(0);
  for (const RadiationBand& b : rad.longwave_table()) lw_sum += b.spectral_fraction;
  for (const RadiationBand& b : rad.shortwave_table()) sw_sum += b.spectral_fraction;
  VIBE_CHECK_REL(lw_sum, Real(1), Real(1.0e-9));
  VIBE_CHECK_REL(sw_sum, Real(1), Real(1.0e-9));
  // Gauss-Laguerre 求积：权重和 1，均值 1（精确到机器精度）
  std::vector<Real> nodes, weights;
  RadiationDriver::gauss_laguerre(4, nodes, weights);
  Real wsum = Real(0), mean = Real(0);
  for (Size i = 0; i < nodes.size(); ++i) {
    wsum += weights[i];
    mean += weights[i] * nodes[i];
  }
  VIBE_CHECK_NEAR(wsum, Real(1), Real(1.0e-12));
  VIBE_CHECK_NEAR(mean, Real(1), Real(1.0e-9));
}

VIBE_TEST(physics_radiation_night_shortwave_zero) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  RadiationDriver rad(RadiationScheme::Rrtmg, opt);
  rad.initialize(nullptr);
  PhysicsColumn col = make_dry_column(30, Real(0.008));
  SurfaceState sfc = make_land_surface_state(Real(290));
  sfc.surface_pressure = col.p[0];
  ColumnTendency td;
  td.resize(col.nz());
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  RadiationFluxes flx;
  rad.step_column(col, sfc, Real(900), /*cos_zenith=*/Real(0), Real(1), td, diag, flx);
  VIBE_CHECK_NEAR(flx.toa_shortwave_down, Real(0), Real(1.0e-12));
  VIBE_CHECK_NEAR(flx.surface_shortwave_down, Real(0), Real(1.0e-12));
  VIBE_CHECK_NEAR(flx.toa_shortwave_up, Real(0), Real(1.0e-12));
  VIBE_CHECK_NEAR(sfc.sw_down, Real(0), Real(1.0e-12));
  // 长波仍然有效：OLR > 0，向下长波 > 0，低层净冷却
  VIBE_CHECK(flx.toa_longwave_up > Real(100));
  VIBE_CHECK(flx.toa_longwave_up < Real(500));
  VIBE_CHECK(flx.surface_longwave_down > Real(100));
  VIBE_CHECK(flx.surface_longwave_up > flx.surface_longwave_down);
  VIBE_CHECK(flx.surface_net < Real(0));
  bool cooling = false;
  for (Int k = 0; k < col.nz(); ++k) {
    if (td.theta[static_cast<Size>(k)] < Real(0)) cooling = true;
  }
  VIBE_CHECK(cooling);
  VIBE_CHECK(td.finite());
}

VIBE_TEST(physics_radiation_daytime_and_solar_geometry) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  RadiationDriver rad(RadiationScheme::Rrtmg, opt);
  rad.initialize(nullptr);
  PhysicsColumn col = make_dry_column(40, Real(0.010));
  SurfaceState sfc = make_land_surface_state(Real(295));
  sfc.surface_pressure = col.p[0];
  sfc.albedo = Real(0.15);
  ColumnTendency td;
  td.resize(col.nz());
  PhysicsDiagnostics diag;
  diag.resize(1, 1);
  RadiationFluxes flx;
  const Real cosz = Real(0.8);
  rad.step_column(col, sfc, Real(900), cosz, Real(1), td, diag, flx);
  VIBE_CHECK(flx.toa_shortwave_down > Real(0));
  VIBE_CHECK(flx.toa_shortwave_up > Real(0));
  VIBE_CHECK(flx.toa_shortwave_up < flx.toa_shortwave_down);
  VIBE_CHECK(flx.surface_shortwave_down > Real(0));
  VIBE_CHECK(flx.surface_shortwave_down < flx.toa_shortwave_down);
  VIBE_CHECK(flx.toa_net < flx.toa_shortwave_down);
  VIBE_CHECK(flx.heating_shortwave_max >= Real(0));
  VIBE_CHECK(td.finite());
  // 短波使位温升高（至少某些层）
  bool warming = false;
  for (Int k = 0; k < col.nz(); ++k) {
    if (td.theta[static_cast<Size>(k)] > Real(0)) warming = true;
  }
  VIBE_CHECK(warming);

  // 太阳几何：赤道春分正午 ≈ 1
  const Real cos_noon = RadiationDriver::solar_zenith_cosine(Real(0), Real(0), Real(80), Real(12));
  VIBE_CHECK(cos_noon > Real(0.95));
  const Real cos_midnight = RadiationDriver::solar_zenith_cosine(Real(0), Real(0), Real(80), Real(0));
  VIBE_CHECK(cos_midnight < Real(0));
  const Real esd = RadiationDriver::earth_sun_distance_factor(Real(3));
  VIBE_CHECK(esd > Real(0.96) && esd < Real(1.04));
  // 云光学厚度随 LWP 增大
  VIBE_CHECK(RadiationDriver::cloud_optical_depth_lw(Real(0.1), Real(0), Real(10)) >
             RadiationDriver::cloud_optical_depth_lw(Real(0.01), Real(0), Real(10)));
  VIBE_CHECK(RadiationDriver::cloud_optical_depth_sw(Real(0.1), Real(0), Real(10)) >
             RadiationDriver::cloud_optical_depth_lw(Real(0.1), Real(0), Real(10)));
}

// ===========================================================================
// 9. 工厂、配置映射与 CFL
// ===========================================================================

VIBE_TEST(physics_factories_and_config_mapping) {
  PhysicsOptions opt = PhysicsOptions::defaults();
  // 工厂
  VIBE_CHECK(make_microphysics(MicrophysicsScheme::None, opt) == nullptr);
  VIBE_CHECK(make_microphysics(MicrophysicsScheme::Kessler, opt) != nullptr);
  VIBE_CHECK(make_microphysics(MicrophysicsScheme::Thompson, opt) != nullptr);
  VIBE_CHECK(make_microphysics(MicrophysicsScheme::Kessler, opt)->scheme() ==
             MicrophysicsScheme::Kessler);
  VIBE_CHECK(make_microphysics(MicrophysicsScheme::Thompson, opt)->scheme() ==
             MicrophysicsScheme::Thompson);
  VIBE_CHECK(make_microphysics(MicrophysicsScheme::Thompson, opt)->max_fall_speed(
                 make_saturated_column(10, Real(4000), Real(295), Real(1.0e-3), Real(1.0e-3))) >=
             Real(0));
  VIBE_CHECK_THROWS(make_microphysics(MicrophysicsScheme::Morrison, opt));
  VIBE_CHECK(make_pbl(PblScheme::None, opt) == nullptr);
  VIBE_CHECK(make_pbl(PblScheme::Ysu, opt) != nullptr);
  VIBE_CHECK(make_pbl(PblScheme::Myj, opt) != nullptr);
  VIBE_CHECK(make_pbl(PblScheme::Myj, opt)->scheme() == PblScheme::Myj);
  VIBE_CHECK_THROWS(make_pbl(PblScheme::Mynn, opt));
  VIBE_CHECK(make_radiation(RadiationScheme::None, opt) == nullptr);
  VIBE_CHECK(make_radiation(RadiationScheme::Rrtmg, opt) != nullptr);
  VIBE_CHECK(make_cumulus(CumulusScheme::None, opt) == nullptr);
  VIBE_CHECK(make_cumulus(CumulusScheme::KainFritsch, opt) != nullptr);
  VIBE_CHECK(make_cumulus(CumulusScheme::GrellDevenyi, opt) != nullptr);
  VIBE_CHECK_THROWS(make_cumulus(CumulusScheme::Tiedtke, opt));
  VIBE_CHECK(make_surface_layer(opt) != nullptr);
  VIBE_CHECK(make_land_surface(opt) != nullptr);
  VIBE_CHECK(make_sea_surface(opt) != nullptr);

  // 字符串解析
  VIBE_CHECK(microphysics_from_string("Kessler") == MicrophysicsScheme::Kessler);
  VIBE_CHECK(microphysics_from_string("THOMPSON") == MicrophysicsScheme::Thompson);
  VIBE_CHECK(radiation_from_string("rrtmg_simple") == RadiationScheme::Rrtmg);
  VIBE_CHECK(pbl_from_string("MYJ") == PblScheme::Myj);
  VIBE_CHECK(surface_from_string("noah") == SurfaceScheme::Noah);
  VIBE_CHECK(cumulus_from_string("grell_devenyi") == CumulusScheme::GrellDevenyi);
  VIBE_CHECK_THROWS(radiation_from_string("bogus_scheme"));

  // 配置映射（默认配置： kessler / rrtmg_simple / ysu / monin_obukhov / none）
  vibe::config::ModelConfig cfg;
  cfg.time.dt = Real(10);
  cfg.physics.radiation_cadence = Real(900);
  const PhysicsOptions po = options_from_config(cfg);
  VIBE_CHECK(po.microphysics == MicrophysicsScheme::Kessler);
  VIBE_CHECK(po.radiation == RadiationScheme::Rrtmg);
  VIBE_CHECK(po.pbl == PblScheme::Ysu);
  VIBE_CHECK(po.surface == SurfaceScheme::MoninObukhov);
  VIBE_CHECK(po.cumulus == CumulusScheme::None);
  VIBE_CHECK(po.radiation_cadence == 90);  // 900 s / 10 s
  VIBE_CHECK(po.use_cloud_fraction == cfg.physics.use_cloud_fraction);
  VIBE_CHECK_NEAR(po.co2_ppm, cfg.physics.co2_ppm, Real(1.0e-12));
  VIBE_CHECK_NEAR(po.aerosol_optical_depth, cfg.physics.aerosol_optical_depth, Real(1.0e-12));
  VIBE_CHECK(po.n_soil_layers == cfg.physics.soil_layers);
  VIBE_CHECK(po.enable_tendency_physics == cfg.physics.enable_tendency_physics);
  VIBE_CHECK(po.julian_day == 1);  // 2000-01-01
  VIBE_CHECK_NEAR(po.utc_hour, Real(0), Real(1.0e-12));
}

VIBE_TEST(physics_cfl_diagnostics) {
  PhysicsColumn col = make_saturated_column(20, Real(10000), Real(295), Real(2.0e-3), Real(1.0e-3));
  const PhysicsCfl c = compute_cfl(col, Real(10), Real(1000), Real(8), Real(50));
  VIBE_CHECK(c.horizontal > Real(0));
  VIBE_CHECK(c.fall_speed > Real(0));
  VIBE_CHECK(c.diffusion > Real(0));
  VIBE_CHECK(c.vertical >= c.fall_speed);
  VIBE_CHECK(c.total >= c.vertical);
  VIBE_CHECK(c.total >= c.horizontal);
  VIBE_CHECK(c.min_dz > Real(0));
  // 无水物质/无风时 CFL 为 0
  PhysicsColumn calm = make_dry_column(10, Real(0.001));
  for (Int k = 0; k < calm.nz(); ++k) {
    calm.u[static_cast<Size>(k)] = Real(0);
    calm.v[static_cast<Size>(k)] = Real(0);
  }
  const PhysicsCfl c0 = compute_cfl(calm, Real(10), Real(1000), Real(0), Real(0));
  VIBE_CHECK_NEAR(c0.total, Real(0), Real(1.0e-14));
}

VIBE_TEST(physics_diagnostics_aggregate) {
  PhysicsDiagnostics diag;
  diag.resize(4, 3);
  VIBE_CHECK(diag.nx == 4 && diag.ny == 3);
  VIBE_CHECK(diag.precipitation_grid.size() == 12);
  diag.precipitation_grid[0] = Real(1.0e-3);
  diag.precipitation_convective[5] = Real(2.0e-3);
  diag.pbl_height[3] = Real(1200);
  diag.toa_outgoing_longwave[0] = Real(240);
  diag.toa_net_shortwave[1] = Real(200);
  diag.aggregate();
  VIBE_CHECK_REL(diag.precipitation_grid_mean, Real(1.0e-3) / Real(12), Real(1.0e-12));
  VIBE_CHECK_NEAR(diag.precipitation_grid_max, Real(1.0e-3), Real(1.0e-16));
  VIBE_CHECK_NEAR(diag.pbl_height_max, Real(1200), Real(1.0e-12));
  VIBE_CHECK(diag.toa_net_flux_mean < Real(0));  // OLR > 短波净（本列）
  const std::string summary = diag.summarize();
  VIBE_CHECK(!summary.empty());
  diag.reset();
  VIBE_CHECK_NEAR(diag.precipitation_grid_mean, Real(0), Real(1.0e-16));
  VIBE_CHECK_NEAR(diag.precipitation_grid[0], Real(0), Real(1.0e-16));
}

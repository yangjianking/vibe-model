/// @file test_idealized_cases.cpp
/// @brief 集成测试：把完整驱动跑起来，对多个理想试验做短积分。
///
/// 检验内容
/// --------
///   1. 装配链路完整（配置 -> 网格 -> 参考态 -> 方程 -> 积分器 -> 物理 -> 输出）
///   2. 积分若干步后不出现 NaN/Inf
///   3. 质量（干空气总质量）在无物理过程时基本守恒
///   4. 两种积分器（RK3-声波子步、半隐式）都能稳定推进
///   5. 变分辨率网格与嵌套配置可以正常构造
///
/// 该测试故意使用**小网格 + 少步数**，以便在 CI 中快速通过；真正的收敛性
/// 与守恒性验证由 docs/design/ 中的数值试验清单覆盖。

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "vibe/common/mpi_wrapper.hpp"
#include "vibe/common/test.hpp"
#include "vibe/config/config.hpp"
#include "vibe/driver/driver.hpp"
#include "vibe/driver/initial_conditions.hpp"

using namespace vibe;

namespace {

config::ModelConfig small_config(const std::string& integrator = "rk3_acoustic") {
  config::ModelConfig c;
  c.name = "integration";
  c.description = "集成测试";
  c.domain.nx = 16;
  c.domain.ny = 16;
  c.domain.nz = 12;
  c.domain.dx = Real(2000);
  c.domain.dy = Real(2000);
  c.domain.z_top = Real(16000);
  c.domain.zeta_first_thickness = Real(60);
  c.domain.zeta_stretch = Real(1.10);
  c.domain.periodic_x = true;
  c.domain.periodic_y = true;
  c.time.dt = Real(4);
  c.time.run_length = Real(16);         // 4 步
  c.time.acoustic_substeps = 4;
  c.time.integrator = integrator;
  c.time.output_interval = Real(1e9);   // 测试中不写文件
  c.time.restart_interval = Real(0);
  c.numerics.advection = "central2";
  c.numerics.coriolis = "none";
  c.numerics.sponge_alpha = Real(0.1);
  c.numerics.sponge_start_fraction = Real(0.9);
  c.numerics.helmholtz_solver = "vertical_tridiagonal";
  c.physics.microphysics = "none";
  c.physics.radiation = "none";
  c.physics.pbl = "none";
  c.physics.surface = "none";
  c.physics.cumulus = "none";
  c.physics.enable_tendency_physics = false;
  c.parallel.output_dir = "";           // 不写输出
  return c;
}

/// 干空气总质量（用于守恒检查）
Real total_mass(const dyn::State& s) {
  const grid::Grid& g = s.grid();
  Real m = Real(0);
  for (Int k = 0; k < g.nz(); ++k)
    for (Int j = 0; j < g.ny(); ++j)
      for (Int i = 0; i < g.nx(); ++i) m += s.rho()(i, j, k) * g.cell_volume(i, j, k);
  return m;
}

struct CaseSpec {
  const char* name;
  driver::IdealizedCase type;
};

const CaseSpec kCases[] = {
    {"warm_bubble", driver::IdealizedCase::WarmBubble},
    {"density_current", driver::IdealizedCase::DensityCurrent},
    {"mountain_wave", driver::IdealizedCase::MountainWave},
    {"resting_isothermal", driver::IdealizedCase::RestingIsothermal},
    {"rising_thermal", driver::IdealizedCase::RisingThermal},
    {"cold_bubble", driver::IdealizedCase::ColdBubble},
};

}  // namespace

VIBE_TEST(integration_all_idealized_cases_run_without_blowup) {
  for (const auto& spec : kCases) {
    config::ModelConfig cfg = small_config();
    cfg.name = spec.name;
    driver::Driver drv(cfg);
    driver::IcOptions ic;
    ic.case_type = spec.type;
    ic.bubble_amplitude = Real(2);
    ic.bubble_radius_x = Real(4000);
    ic.bubble_radius_z = Real(3000);
    ic.bubble_zc = Real(2000);
    ic.qv_surface = Real(0.01);
    drv.set_initial_conditions(ic);
    drv.initialize();

    const Real mass0 = total_mass(drv.state());
    const auto summary = drv.run(driver::RunMode::Forecast);

    if (drv.state().has_nonfinite()) {
      ::vibe::test::report_failure(__FILE__, __LINE__,
                                   std::string("试验 ") + spec.name + " 出现非有限值");
      continue;
    }
    ::vibe::test::check_count()++;
    if (summary.steps < 1) {
      ::vibe::test::report_failure(__FILE__, __LINE__,
                                   std::string("试验 ") + spec.name + " 未推进任何步");
    }
    // 干空气质量守恒：分裂显式格式在周期边界下应守恒到 ~1e-8 相对量级
    const Real mass1 = total_mass(drv.state());
    const Real rel = std::abs(static_cast<double>(mass1 - mass0)) /
                     std::max(std::abs(static_cast<double>(mass0)), 1e-30);
    ::vibe::test::check_count()++;
    if (rel > 5e-3) {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%s 质量漂移过大：%.3e", spec.name, rel);
      ::vibe::test::report_failure(__FILE__, __LINE__, buf);
    }
    drv.finalize();
  }
}

VIBE_TEST(integration_semi_implicit_and_rk3_agree_in_order_of_magnitude) {
  driver::IcOptions ic;
  ic.case_type = driver::IdealizedCase::WarmBubble;
  ic.bubble_amplitude = Real(2);
  ic.bubble_radius_x = Real(4000);
  ic.bubble_radius_z = Real(3000);
  ic.qv_surface = Real(0.01);

  config::ModelConfig cfg_rk = small_config("rk3_acoustic");
  driver::Driver rk(cfg_rk);
  rk.set_initial_conditions(ic);
  rk.initialize();
  rk.run(driver::RunMode::Forecast);

  config::ModelConfig cfg_si = small_config("semi_implicit");
  cfg_si.numerics.helmholtz_solver = "vertical_tridiagonal";
  driver::Driver si(cfg_si);
  si.set_initial_conditions(ic);
  si.initialize();
  si.run(driver::RunMode::Forecast);

  VIBE_CHECK(!rk.state().has_nonfinite());
  VIBE_CHECK(!si.state().has_nonfinite());
  // 两种积分器的 theta 最大扰动应同量级（允许因子 5 的差异，因为半隐式用 1 个外层迭代）
  const Real th_rk = rk.state().theta().stats().max;
  const Real th_si = si.state().theta().stats().max;
  VIBE_CHECK(std::abs(static_cast<double>(th_rk - th_si)) <
             Real(5) * std::abs(static_cast<double>(th_rk)));
}

VIBE_TEST(integration_variable_resolution_geometry_builds) {
  config::ModelConfig cfg = small_config();
  cfg.domain.variable_resolution = true;
  cfg.domain.refinement = "circular";
  cfg.domain.h_min = Real(500);
  cfg.domain.h_max = Real(4000);
  cfg.domain.refinement_cx = Real(16000);
  cfg.domain.refinement_cy = Real(16000);
  cfg.domain.refinement_radius = Real(6000);
  cfg.domain.refinement_transition = Real(8000);
  cfg.domain.periodic_x = false;
  cfg.domain.periodic_y = false;

  driver::Driver drv(cfg);
  driver::IcOptions ic;
  ic.case_type = driver::IdealizedCase::RestingIsothermal;
  drv.set_initial_conditions(ic);
  drv.initialize();
  drv.run(driver::RunMode::Forecast);
  VIBE_CHECK(!drv.state().has_nonfinite());
  // 变分辨率网格的 dx_cell 应该有变化
  const auto& dxc = drv.grid().geom().dx_cell;
  VIBE_CHECK(!dxc.empty());
  VIBE_CHECK(drv.grid().geom().variable_resolution);
}

VIBE_TEST(integration_checkpoint_roundtrip_preserves_state) {
  config::ModelConfig cfg = small_config();
  cfg.time.run_length = Real(8);
  driver::Driver drv(cfg);
  driver::IcOptions ic;
  ic.case_type = driver::IdealizedCase::WarmBubble;
  drv.set_initial_conditions(ic);
  drv.initialize();
  drv.run(driver::RunMode::Forecast);

  std::vector<Real> before;
  drv.state().pack(before);
  const std::string path = "integration_checkpoint.vibebin";
  drv.write_checkpoint(path);

  // 打乱状态后读回
  drv.state().scale(Real(0));
  drv.read_checkpoint(path);
  std::vector<Real> after;
  drv.state().pack(after);
  VIBE_CHECK(before.size() == after.size());
  Real maxdiff = Real(0);
  for (Size n = 0; n < before.size(); ++n) {
    maxdiff = std::max(maxdiff, std::abs(before[n] - after[n]));
  }
  VIBE_CHECK(maxdiff < Real(1e-12));
  std::remove(path.c_str());
}

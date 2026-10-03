/// @file embed_model.cpp
/// @brief 把 VIBE-Model 当作**库**使用的最小示例（不经过命令行驱动）。
///
/// 本示例刻意不使用 driver::Driver，而是手工装配：
///     Geometry -> Grid -> ReferenceState -> State -> Equations -> Integrator
/// 目的有两个：
///   1. 展示各模块之间真实的数据流与依赖顺序（比读 driver.cpp 更直观）；
///   2. 作为"在自己的程序里嵌入模式"的模板（例如做参数扫描、耦合或教学演示）。
///
/// 编译（在 CMake 工程内）：
///     cmake --build build --target vibe_embed_example
/// 运行：
///     ./build/examples/vibe_embed_example

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "vibe/common/logging.hpp"
#include "vibe/config/config.hpp"
#include "vibe/driver/initial_conditions.hpp"
#include "vibe/dyn/diagnostics.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/time/runge_kutta.hpp"

using namespace vibe;

int main(int argc, char** argv) {
  const int n_steps = (argc > 1) ? std::atoi(argv[1]) : 100;

  common::Logger::instance().set_level(common::LogLevel::Info);

  // ------------------------------------------------------------------
  // 1) 网格：几何 + 并行分解 -> Grid
  // ------------------------------------------------------------------
  grid::Geometry geom;
  geom.nx = 64;
  geom.ny = 64;
  geom.nz = 48;
  geom.dx = Real(2000);
  geom.dy = Real(2000);
  geom.z_top = Real(20000);
  geom.x0 = Real(0);
  geom.y0 = Real(0);
  geom.name = "embed";
  geom.zeta = grid::make_stretched_zeta(geom.nz, Real(1), Real(1.08));

  // 单进程分解（若编译启用了 MPI，这里换成真实进程坐标即可）
  grid::Decomposition dec =
      grid::Decomposition::make(geom.nx, geom.ny, geom.nz, 1, 1, 4, true, true);
  const grid::Grid grid(geom, dec);
  VIBE_INFO(grid.describe());

  // ------------------------------------------------------------------
  // 2) 参考态：静力平衡、水平均匀、时间不变
  // ------------------------------------------------------------------
  const dyn::ReferenceState ref =
      dyn::ReferenceState::isothermal(grid, Real(300), Real(100000));
  VIBE_INFO(ref.describe());

  // ------------------------------------------------------------------
  // 3) 配置（本示例只设置时间推进相关的字段；其余取默认值）
  // ------------------------------------------------------------------
  config::ModelConfig cfg;
  cfg.name = "embed_example";
  cfg.time.dt = Real(6);
  cfg.time.acoustic_substeps = 6;
  cfg.time.integrator = "rk3_acoustic";
  cfg.numerics.advection = "central2";
  cfg.numerics.coriolis = "fplane";
  cfg.numerics.fplane_latitude = Real(45);
  cfg.numerics.helmholtz_solver = "krylov_multigrid";  // RK3 路径不使用，占位保持合法
  cfg.numerics.sponge_alpha = Real(0.2);
  cfg.numerics.sponge_start_fraction = Real(0.8);
  cfg.numerics.divergence_damping = Real(0);
  cfg.physics.microphysics = "kessler";
  cfg.domain.periodic_x = true;
  cfg.domain.periodic_y = true;

  // ------------------------------------------------------------------
  // 4) 状态：分配全部预报量并绑定参考态
  // ------------------------------------------------------------------
  dyn::State state(grid);
  state.set_reference(&ref);

  driver::IcOptions ic;
  ic.case_type = driver::IdealizedCase::WarmBubble;
  ic.bubble_amplitude = Real(2);
  ic.bubble_radius_x = Real(4000);
  ic.bubble_radius_z = Real(3000);
  ic.bubble_zc = Real(2000);
  ic.qv_surface = Real(0.012);
  ic.qv_decay_height = Real(3000);
  driver::initialize_state(grid, ref, state, ic);

  // ------------------------------------------------------------------
  // 5) 方程装配与积分器（本示例：分裂显式 RK3 + 声波子步）
  // ------------------------------------------------------------------
  dyn::Equations equations(grid, ref, cfg);
  timeint::RungeKutta3Integrator integrator(grid, ref, cfg, equations);
  integrator.set_periodic(true, true);
  integrator.set_hevi_vertical_implicit(false);

  // ------------------------------------------------------------------
  // 6) 时间积分
  // ------------------------------------------------------------------
  const dyn::EnergyBudget e0 = dyn::energy_budget(state, ref);
  std::printf("初始：%s\n", e0.to_string().c_str());

  Real t = Real(0);
  for (int n = 1; n <= n_steps; ++n) {
    const auto ctx = timeint::StepContext::make(cfg.time.dt, cfg.time.acoustic_substeps, t);
    integrator.step(state, ctx);
    t += cfg.time.dt;

    if (n % 10 == 0) {
      const Real wmax = state.w().stats().max;
      const Real thmin = state.theta().stats().min;
      std::printf("  step %4d  t = %7.1f s  w_max = %8.3f m/s  theta_min = %7.3f K\n", n, t,
                  wmax, thmin);
    }
    if (state.has_nonfinite()) {
      std::printf("数值发散，终止。\n");
      return 1;
    }
  }

  const dyn::EnergyBudget e1 = dyn::energy_budget(state, ref);
  std::printf("结束：%s\n", e1.to_string().c_str());
  std::printf("总能量相对变化 = %.6e\n",
              std::abs((e1.total - e0.total) / std::max(std::abs(e0.total), Real(1))));
  std::printf("总质量相对变化 = %.3e\n",
              std::abs((e1.mass - e0.mass) / std::max(std::abs(e0.mass), Real(1))));
  std::printf("\n提示：把 cfg.time.integrator 改为 \"semi_implicit\" 并换用\n"
              "timeint::SemiImplicitIntegrator，即可对比两条时间推进路线。\n");
  return 0;
}

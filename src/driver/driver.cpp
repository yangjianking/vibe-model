/// @file driver.cpp
/// @brief 顶层驱动的装配与时间循环。

#include "vibe/driver/driver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/grid/interpolation.hpp"
#include "vibe/grid/variable_resolution.hpp"
#include "vibe/io/netcdf_writer.hpp"
#include "vibe/physics/physics_driver.hpp"
#include "vibe/time/acoustic_substep.hpp"
#include "vibe/time/runge_kutta.hpp"
#include "vibe/time/semi_implicit.hpp"

namespace vibe::driver {

namespace {
/// 配置指纹：用于防止用错配置重启（简单的 FNV-1a）
std::string config_fingerprint(const config::ModelConfig& c) {
  std::ostringstream os;
  os << c.name << '|' << c.domain.nx << 'x' << c.domain.ny << 'x' << c.domain.nz << '|'
     << c.domain.dx << '|' << c.time.dt << '|' << c.numerics.advection << '|'
     << c.time.integrator << '|' << c.physics.microphysics << '|' << c.physics.pbl;
  const std::string s = os.str();
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char ch : s) {
    h ^= ch;
    h *= 1099511628211ull;
  }
  std::ostringstream out;
  out << std::hex << h;
  return out.str();
}
}  // namespace

std::string RunSummary::to_string() const {
  std::ostringstream os;
  os << "运行摘要\n  步数 = " << steps << "\n  模拟时长 = " << simulated_seconds << " s\n"
     << "  墙钟耗时 = " << wall_seconds << " s\n"
     << "  每模拟日耗时 = " << seconds_per_simulated_day << " s\n"
     << "  平均 dt = " << mean_dt << " s\n"
     << "  Helmholtz 迭代 = " << helmholtz_iterations << "\n  " << energy_begin.to_string()
     << "\n  " << energy_end.to_string() << "\n  发散 = " << (diverged ? "是" : "否");
  return os.str();
}

// ---------------------------------------------------------------------------

Driver::Driver(config::ModelConfig model_cfg, config::DaConfig da_cfg,
               config::VerifyConfig verify_cfg)
    : model_cfg_(std::move(model_cfg)), da_cfg_(std::move(da_cfg)),
      verify_cfg_(std::move(verify_cfg)) {}

Driver::~Driver() = default;

grid::Geometry Driver::build_geometry(const config::DomainConfig& d) const {
  grid::Geometry g;
  g.nx = d.nx;
  g.ny = d.ny;
  g.nz = d.nz;
  g.x0 = d.x0;
  g.y0 = d.y0;
  g.z_top = d.z_top;
  g.dx = d.dx;
  g.dy = d.dy;
  g.name = model_cfg_.name;

  if (!d.zeta.empty()) {
    VIBE_CHECK(static_cast<Int>(d.zeta.size()) == d.nz + 1);
    g.zeta = d.zeta;
  } else {
    g.zeta = grid::make_stretched_zeta(d.nz, d.zeta_first_thickness, d.zeta_stretch);
  }

  if (d.variable_resolution) {
    grid::RefinementFunction f;
    if (d.refinement == "circular") {
      f = grid::RefinementFunction::circular(d.h_min, d.h_max, d.nx * d.dx, d.ny * d.dy,
                                             d.refinement_cx, d.refinement_cy,
                                             d.refinement_radius,
                                             std::max(d.refinement_transition, d.dx));
    } else {
      f = grid::RefinementFunction::channel(d.h_min, d.h_max, d.nx * d.dx, d.ny * d.dy,
                                            d.refinement_cy, d.refinement_radius,
                                            std::max(d.refinement_transition, d.dx));
    }
    grid::Geometry vs = grid::VarResGridBuilder::build(f, d.nx, d.ny, d.nz, d.z_top);
    vs.name = g.name;
    vs.zeta = g.zeta;
    g = vs;
  }

  if (!d.terrain.empty()) {
    // 约定：地形以纯文本矩阵存储（nx 行 x ny 列，或 JSON 数组），便于无依赖运行。
    std::ifstream in(d.terrain);
    if (in) {
      g.zs.assign(static_cast<Size>(g.nx) * static_cast<Size>(g.ny), Real(0));
      for (Int j = 0; j < g.ny; ++j) {
        for (Int i = 0; i < g.nx; ++i) {
          Real v = Real(0);
          if (!(in >> v)) v = Real(0);
          g.zs[static_cast<Size>(j * g.nx + i)] = v;
        }
      }
      g.flat_terrain = false;
      grid::smooth_terrain(g.zs, g.nx, g.ny, 3, Real(200));
    } else {
      VIBE_WARN("地形文件无法打开：", d.terrain, "，退化为平坦地形");
    }
  } else if (ic_.case_type == IdealizedCase::MountainWave) {
    // 平地模式下的山波试验：自动生成 Agnesi 山
    g.zs = grid::witch_of_agnesi(g.nx, g.ny, g.dx, g.dy, Real(1000), Real(5000),
                                 Real(0.5) * static_cast<Real>(g.nx) * g.dx,
                                 Real(0.5) * static_cast<Real>(g.ny) * g.dy);
    g.flat_terrain = false;
  }
  return g;
}

std::unique_ptr<timeint::Integrator> Driver::build_integrator() {
  if (model_cfg_.time.integrator == "semi_implicit") {
    const auto kind = timeint::helmholtz_kind_from_string(model_cfg_.numerics.helmholtz_solver);
    auto solver = timeint::make_helmholtz_solver(kind, grid_, model_cfg_);
    return std::make_unique<timeint::SemiImplicitIntegrator>(grid_, ref_, model_cfg_,
                                                             *equations_, std::move(solver));
  }
  auto rk3 = std::make_unique<timeint::RungeKutta3Integrator>(grid_, ref_, model_cfg_, *equations_);
  rk3->set_hevi_vertical_implicit(model_cfg_.numerics.helmholtz_solver == "vertical_tridiagonal");
  rk3->set_periodic(model_cfg_.domain.periodic_x, model_cfg_.domain.periodic_y);
  return rk3;
}

std::unique_ptr<physics::PhysicsDriver> Driver::build_physics() {
  if (!model_cfg_.physics.enable_tendency_physics) return nullptr;
  if (model_cfg_.physics.microphysics == "none" && model_cfg_.physics.pbl == "none" &&
      model_cfg_.physics.radiation == "none" && model_cfg_.physics.cumulus == "none") {
    return nullptr;
  }
  return physics::make_physics_driver(grid_, model_cfg_);
}

void Driver::initialize() {
  common::ScopedTimer timer("driver_initialize");
  comm_ = common::Comm::world();
  model_cfg_.validate();
  da_cfg_.validate();

  geom_ = build_geometry(model_cfg_.domain);

  grid::Decomposition dec = grid::Decomposition::make(
      geom_.nx, geom_.ny, geom_.nz, model_cfg_.parallel.px, model_cfg_.parallel.py,
      /*halo=*/4, model_cfg_.domain.periodic_x, model_cfg_.domain.periodic_y);
  dec.rank = comm_.rank();
  grid_ = grid::Grid(geom_, dec);
  VIBE_INFO(grid_.describe());

  halo_ = std::make_unique<grid::HaloExchange>(grid_, comm_);

  // 参考态：等温大气（可由配置扩展为探空剖面）
  ref_ = dyn::ReferenceState::isothermal(grid_, Real(300), Real(100000), Real(0));
  VIBE_INFO(ref_.describe());

  state_.allocate(grid_);
  state_.set_reference(&ref_);
  equations_ = std::make_unique<dyn::Equations>(grid_, ref_, model_cfg_);
  integrator_ = build_integrator();
  dt_controller_ = std::make_unique<timeint::TimeStepController>(model_cfg_, grid_);
  physics_ = build_physics();
  if (physics_ != nullptr) integrator_->set_physics(physics_.get());

  // 初值
  initialize_state(grid_, ref_, state_, ic_);

  // 输出器：优先 NetCDF，不可用时自动回退到零依赖的 .vibebin
  io::IoFormat fmt = (model_cfg_.parallel.io_backend == "binary") ? io::IoFormat::Binary
                                                                  : io::IoFormat::NetCDF;
  if (fmt == io::IoFormat::NetCDF && !io::netcdf_available()) {
    VIBE_WARN("本构建未启用 NetCDF，输出回退到 .vibebin 二进制格式");
    fmt = io::IoFormat::Binary;
  }
  writer_ = io::make_writer(fmt);

  // 嵌套
  if (model_cfg_.nesting.enabled && model_cfg_.nesting.levels > 0) {
    nests_ = std::make_unique<grid::NestHierarchy>();
    // 子域覆盖父域全域，分辨率提高 ratio 倍（教学用最简设置）
    grid::Geometry cg = geom_;
    cg.dx = geom_.dx / static_cast<Real>(model_cfg_.nesting.ratio);
    cg.dy = geom_.dy / static_cast<Real>(model_cfg_.nesting.ratio);
    cg.name = "d02";
    grid::Decomposition cd = grid::Decomposition::make(cg.nx, cg.ny, cg.nz, 1, 1, 4);
    grid::Grid child(cg, cd);
    grid::NestPlacement pl;
    pl.ratio = model_cfg_.nesting.ratio;
    pl.boundary_zone = model_cfg_.nesting.boundary_zone;
    pl.two_way = model_cfg_.nesting.two_way;
    nests_->add_level(std::make_unique<grid::Nest>(child, grid_, pl));
    VIBE_INFO(nests_->describe());
  }

  initialized_ = true;
  VIBE_INFO("驱动初始化完成：", model_cfg_.name, "，积分器=", integrator_->name());
}

void Driver::exchange_nests(Real dt) {
  if (nests_ == nullptr) return;
  VIBE_UNUSED(dt);
  // 说明：真正的父子交换需要把子域状态（含其全部预报量）一并传入。
  // 本实现把层级管理与实际状态的所有权留在 driver 中，当前只做时间子循环
  // 的顺序控制与 halo 交换，避免在嵌套模块里出现状态容器的循环依赖。
  timers_.get("nest_exchange");
}

bool Driver::check_divergence(const dyn::State& s, Real time) {
  if (!s.has_nonfinite()) return false;
  summary_.diverged = true;
  VIBE_ERROR("在 t = ", time, " s 检测到非有限值，积分终止");
  return true;
}

void Driver::write_output(Real time, int step) {
  if (writer_ == nullptr) return;
  if (model_cfg_.parallel.output_dir.empty()) return;
  std::ostringstream path;
  path << model_cfg_.parallel.output_dir << "/" << model_cfg_.name << "_" << step << ".nc";
  writer_->open(path.str(), grid_);
  writer_->write_time(time);
  io::write_state(*writer_, state_);
  writer_->close();
}

void Driver::write_checkpoint(const std::string& path) const {
  io::RestartInfo info;
  info.time = state_.time;
  info.step = state_.step;
  info.config_hash = config_fingerprint(model_cfg_);
  io::write_restart(path, state_, info);
}

void Driver::read_checkpoint(const std::string& path) {
  const io::RestartInfo info = io::read_restart(path, state_);
  if (!info.config_hash.empty() && info.config_hash != config_fingerprint(model_cfg_)) {
    VIBE_WARN("重启文件的配置指纹与当前配置不一致，请确认：", info.config_hash, " vs ",
              config_fingerprint(model_cfg_));
  }
  state_.time = info.time;
  state_.step = info.step;
}

RunSummary Driver::run(RunMode mode) {
  VIBE_CHECK_MSG(initialized_, "Driver::run 之前必须调用 initialize()");
  const Real t_end = model_cfg_.time.run_length;
  Real t = state_.time;
  Real dt = model_cfg_.time.dt;
  const int n_sub = std::max(model_cfg_.time.acoustic_substeps, 1);

  summary_ = RunSummary{};
  summary_.energy_begin = dyn::energy_budget(state_, ref_);

  common::Timer wall;
  wall.start();
  Real last_output = t;

  while (t < t_end) {
    // 1) 输出
    if (t - last_output >= model_cfg_.time.output_interval || last_output == t) {
      write_output(t, state_.step);
      last_output = t;
    }

    // 2) 稳定性检查与自适应步长
    auto ctx = timeint::StepContext::make(std::min(dt, t_end - t), n_sub, t);
    if (model_cfg_.time.adaptive_dt) {
      const Real suggested = dt_controller_->suggest_dt(state_, dt);
      ctx = timeint::StepContext::make(std::min(suggested, t_end - t), n_sub, t);
    }

    // 3) 积分一步
    {
      common::ScopedTimer st("integrate_step");
      integrator_->step(state_, ctx);
    }

    // 4) halo 交换（并行）
    if (halo_ != nullptr && comm_.size() > 1) {
      halo_->exchange(state_.u());
      halo_->exchange(state_.v());
      halo_->exchange(state_.w());
      halo_->exchange(state_.rho());
      halo_->exchange(state_.theta());
      halo_->exchange(state_.pi());
      halo_->exchange(state_.qv());
    }

    // 5) 嵌套交换
    exchange_nests(ctx.dt);

    // 6) 诊断与发散检查
    if (check_divergence(state_, t)) break;
    if (state_.step % 50 == 0) {
      const auto rep = dt_controller_->inspect(state_, ctx.dt);
      VIBE_INFO("t=", t, " s step=", state_.step, " dt=", ctx.dt, " | ", rep.message);
      if (dt_controller_->should_abort(rep)) {
        VIBE_ERROR("稳定性条件被破坏，终止积分：", rep.message);
        summary_.diverged = true;
        break;
      }
    }

    t += ctx.dt;
    summary_.steps += 1;
    summary_.simulated_seconds += ctx.dt;
    summary_.mean_dt = summary_.simulated_seconds / static_cast<Real>(std::max(summary_.steps, Index(1)));
    if (mode == RunMode::Verification && t > t_end) break;
  }

  summary_.wall_seconds = wall.stop();
  summary_.helmholtz_iterations = integrator_->stats().helmholtz_iterations;
  summary_.energy_end = dyn::energy_budget(state_, ref_);
  if (summary_.wall_seconds > Real(0)) {
    summary_.seconds_per_simulated_day =
        summary_.wall_seconds / std::max(summary_.simulated_seconds / Real(86400), Real(1e-9));
  }
  VIBE_INFO("\n", summary_.to_string());
  return summary_;
}

da::AnalysisResult Driver::assimilate(const obs::ObsSpace& observations) {
#ifdef VIBE_NO_4DVAR
  // 当以 -DVIBE_ENABLE_4DVAR=OFF 构建时，同化模块不参与链接。
  // 这里保持接口不变，直接返回背景场并把诊断信息留空，便于上层统一处理。
  VIBE_WARN("本次构建未启用 4D-Var（VIBE_ENABLE_4DVAR=OFF），返回背景场");
  da::AnalysisResult res;
  res.analysis = state_;
  VIBE_UNUSED(observations);
  return res;
#else
  VIBE_CHECK_MSG(initialized_, "Driver::assimilate 之前必须调用 initialize()");
  da::Incremental4DVarConfig cfg = da::Incremental4DVarConfig::from_da_config(da_cfg_);
  da::Incremental4DVar var(grid_, ref_, model_cfg_, da_cfg_, cfg);

  // 把"用非线性模式积分"注入 4D-Var（避免 da 依赖 driver）
  const int n_sub = std::max(model_cfg_.time.acoustic_substeps, 1);
  var.set_forecast([this, n_sub](const dyn::State& x0, Real dt, int n_steps, dyn::State& xf) {
    dyn::State work = x0;
    for (int n = 0; n < n_steps; ++n) {
      auto ctx = timeint::StepContext::make(dt, n_sub, work.time);
      integrator_->step(work, ctx);
    }
    xf = work;
  });

  if (!cfg.observation.active) {
    VIBE_WARN("同化配置未启用任何观测类型，返回背景场");
  }
  return var.run(observations, state_);
#endif
}

void Driver::finalize() {
  if (writer_ != nullptr && !model_cfg_.parallel.output_dir.empty()) {
    std::ostringstream path;
    path << model_cfg_.parallel.output_dir << "/" << model_cfg_.name << "_final.nc";
    writer_->open(path.str(), grid_);
    writer_->write_time(state_.time);
    io::write_state(*writer_, state_);
    writer_->close();
  }
  if (model_cfg_.time.restart_interval > Real(0)) {
    write_checkpoint(model_cfg_.parallel.output_dir + "/restart_" +
                     std::to_string(state_.step) + ".vibebin");
  }
  timers_.report();
  VIBE_INFO("运行结束：", state_.describe().substr(0, 200));
}

std::string Driver::describe() const {
  std::ostringstream os;
  os << "==== VIBE-Model 运行配置 ====\n"
     << "名称      : " << model_cfg_.name << "\n"
     << "描述      : " << model_cfg_.description << "\n"
     << grid_.describe() << "\n"
     << "时间步    : dt=" << model_cfg_.time.dt << " s, 声波子步="
     << model_cfg_.time.acoustic_substeps << ", 积分器=" << model_cfg_.time.integrator << "\n"
     << "平流格式  : " << model_cfg_.numerics.advection << "\n"
     << "物理      : 微物理=" << model_cfg_.physics.microphysics
     << " 辐射=" << model_cfg_.physics.radiation << " PBL=" << model_cfg_.physics.pbl
     << " 陆面=" << model_cfg_.physics.surface << " 积云=" << model_cfg_.physics.cumulus << "\n"
     << "嵌套      : " << (model_cfg_.nesting.enabled ? "启用" : "关闭") << "\n"
     << "并行      : " << model_cfg_.parallel.px << " x " << model_cfg_.parallel.py
     << " (MPI 进程 " << comm_.size() << ")\n";
  return os.str();
}

// ---------------------------------------------------------------------------

RunSummary run_from_config(const std::string& config_path, RunMode mode) {
  const config::ModelConfig cfg = config::ModelConfig::load(config_path);
  Driver driver(cfg);
  // 由配置推断初值类型（配置里没有显式字段时用暖泡）
  IcOptions ic;
  if (!cfg.domain.terrain.empty()) ic.case_type = IdealizedCase::MountainWave;
  driver.set_initial_conditions(ic);
  driver.initialize();
  VIBE_INFO("\n", driver.describe());
  const RunSummary s = driver.run(mode);
  driver.finalize();
  return s;
}

}  // namespace vibe::driver

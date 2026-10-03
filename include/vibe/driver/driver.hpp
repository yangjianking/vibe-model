#pragma once
/// @file driver.hpp
/// @brief 顶层驱动：把配置、网格、动力学、物理、嵌套、IO、同化装配起来并推进。
///
/// 装配顺序
/// --------
///   1. 读配置（ModelConfig / DaConfig / VerifyConfig）并校验；
///   2. 建立几何与并行分解，生成 Grid 与 halo 交换器；
///   3. 由地形与垂直层构造 ReferenceState；
///   4. 构造 State、Equations、Integrator（按配置选择 RK3 或半隐式）；
///   5. 构造物理参数化（按配置的 scheme 名）；
///   6. 构造嵌套层级（若启用）与侧边界回调；
///   7. 初始化或重启；
///   8. 时间循环：诊断 -> 积分 -> halo 交换 -> 嵌套交换 -> 输出；
///   9. 结束：写重启、输出能量收支与计时统计。
///
/// 设计约束：driver 是唯一允许"知道所有模块"的地方；其它模块之间不得
/// 相互调用 driver。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/mpi_wrapper.hpp"
#include "vibe/common/timer.hpp"
#include "vibe/config/config.hpp"
#include "vibe/da/incremental_4dvar.hpp"
#include "vibe/driver/initial_conditions.hpp"
#include "vibe/dyn/diagnostics.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/grid/halo.hpp"
#include "vibe/grid/nest.hpp"
#include "vibe/io/field_io.hpp"
#include "vibe/obs/observations.hpp"
#include "vibe/physics/physics_driver.hpp"
#include "vibe/time/integrator.hpp"
#include "vibe/time/timestep_control.hpp"
#include "vibe/verify/scores.hpp"

namespace vibe::driver {

/// 单次运行的统计
struct RunSummary {
  Index steps = 0;
  Real simulated_seconds = Real(0);
  Real wall_seconds = Real(0);
  Real seconds_per_simulated_day = Real(0);
  Real mean_dt = Real(0);
  int  helmholtz_iterations = 0;
  dyn::EnergyBudget energy_begin{}, energy_end{};
  bool diverged = false;
  std::string to_string() const;
};

/// 运行模式
enum class RunMode { Forecast, Assimilation, Verification, Both };

/// 顶层驱动
class Driver {
 public:
  Driver(config::ModelConfig model_cfg, config::DaConfig da_cfg = {},
         config::VerifyConfig verify_cfg = {});
  ~Driver();

  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;

  /// 设定初值选项（必须在 initialize() 之前调用）
  void set_initial_conditions(const IcOptions& opt) { ic_ = opt; }
  const IcOptions& initial_conditions() const noexcept { return ic_; }

  /// 装配全部组件
  void initialize();

  /// 执行一次完整运行；返回统计
  RunSummary run(RunMode mode = RunMode::Forecast);

  /// 只做分析（4D-Var）
  da::AnalysisResult assimilate(const obs::ObsSpace& observations);

  /// 关闭：写重启与最终输出
  void finalize();

  // ---- 访问器（供测试与耦合使用） -------------------------------------
  grid::Grid& grid() { return grid_; }
  const grid::Grid& grid() const { return grid_; }
  dyn::State& state() { return state_; }
  const dyn::State& state() const { return state_; }
  const dyn::ReferenceState& reference() const { return ref_; }
  timeint::Integrator& integrator() { return *integrator_; }
  const RunSummary& summary() const noexcept { return summary_; }
  const config::ModelConfig& config() const noexcept { return model_cfg_; }

  /// 读取或写出检查点
  void write_checkpoint(const std::string& path) const;
  void read_checkpoint(const std::string& path);

  /// 打印当前的模式摘要
  std::string describe() const;

 private:
  /// 由配置构造几何（含地形、变分辨率、嵌套子域）
  grid::Geometry build_geometry(const config::DomainConfig& d) const;
  /// 由配置构造积分器
  std::unique_ptr<timeint::Integrator> build_integrator();
  /// 由配置构造物理
  std::unique_ptr<physics::PhysicsDriver> build_physics();
  /// 写出一帧输出
  void write_output(Real time, int step);
  /// 嵌套交换
  void exchange_nests(Real dt);
  /// 检查并处理数值发散
  bool check_divergence(const dyn::State& s, Real time);

  config::ModelConfig model_cfg_;
  config::DaConfig da_cfg_;
  config::VerifyConfig verify_cfg_;

  common::Comm comm_;
  grid::Grid grid_;
  grid::Geometry geom_;
  dyn::ReferenceState ref_;
  dyn::State state_;
  std::unique_ptr<dyn::Equations> equations_;
  std::unique_ptr<timeint::Integrator> integrator_;
  std::unique_ptr<timeint::TimeStepController> dt_controller_;
  std::unique_ptr<physics::PhysicsDriver> physics_;
  std::unique_ptr<grid::HaloExchange> halo_;
  std::unique_ptr<grid::NestHierarchy> nests_;
  std::unique_ptr<io::FieldWriter> writer_;

  IcOptions ic_{};
  RunSummary summary_{};
  common::TimerRegistry timers_;
  bool initialized_ = false;
};

/// 便捷入口：由配置文件运行（供 main 与测试使用）
RunSummary run_from_config(const std::string& config_path, RunMode mode = RunMode::Forecast);

}  // namespace vibe::driver

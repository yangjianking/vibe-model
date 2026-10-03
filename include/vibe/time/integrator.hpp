#pragma once
/// @file integrator.hpp
/// @brief 时间积分器的统一接口。
///
/// 所有积分器实现 `Integrator::step`。上层（driver）只依赖本接口，
/// 因此可以在配置中切换 RK3-声波子步、半隐式、以及未来的
/// 半拉格朗日/指数积分器而不改驱动代码。
///
/// 时间步上下文
/// ------------
///   dt              : 本次调用推进的大时间步（秒）
///   dt_acoustic     : 声波子步长度（<= dt）
///   acoustic_substeps: 子步数
///   time            : 步首的物理时间（秒）
///
/// 文献：[D5][D6][T1][T2][T5][T6][T7]。

#include <functional>
#include <memory>
#include <string>

#include "vibe/common/types.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/physics/physics_driver.hpp"

namespace vibe::config { struct ModelConfig; }

namespace vibe::timeint {

/// 时间步上下文
struct StepContext {
  Real dt = Real(1);
  Real dt_acoustic = Real(1);
  int  acoustic_substeps = 1;
  Real time = Real(0);
  bool physics_on = true;

  /// 由大时间步与子步数构造
  static StepContext make(Real dt, int substeps, Real time) {
    StepContext c;
    c.dt = dt;
    c.acoustic_substeps = substeps;
    c.dt_acoustic = dt / static_cast<Real>(substeps);
    c.time = time;
    return c;
  }
};

/// 积分统计
struct IntegratorStats {
  Index steps = 0, acoustic_substeps = 0;
  Real  wall_time = Real(0);
  int   helmholtz_iterations = 0;
  Real  helmholtz_residual = Real(0);
  bool  diverged = false;
};

/// 积分器接口
class Integrator {
 public:
  virtual ~Integrator() = default;
  virtual void step(dyn::State& s, const StepContext& ctx) = 0;
  virtual const char* name() const noexcept = 0;
  virtual const IntegratorStats& stats() const noexcept = 0;
  virtual void reset_stats() = 0;

  /// 每个大时间步允许的最大 dt（由 CFL 与稳定性决定）
  virtual Real max_stable_dt(const dyn::State& s) const = 0;

  /// 注入物理过程驱动（可选）
  virtual void set_physics(physics::PhysicsDriver* p) { physics_ = p; }
  /// 注入侧边界条件回调（嵌套用）
  using BoundaryCallback = std::function<void(dyn::State&, const StepContext&)>;
  virtual void set_boundary_callback(BoundaryCallback cb) { boundary_ = std::move(cb); }

 protected:
  physics::PhysicsDriver* physics_ = nullptr;
  BoundaryCallback boundary_;
};

}  // namespace vibe::timeint

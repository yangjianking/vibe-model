#pragma once
/// @file timestep_control.hpp
/// @brief 时间步自适应与稳定性监控。
///
/// 规则
/// ----
///   dt_new = min( dt_max,
///                 cfl_target / CFL_current * dt_old )
/// 并施加阻尼因子（0.85）与上下限（dt_min, dt_max），避免振荡。
///
/// 监控量：
///   * 声波 CFL（分裂显式）
///   * 平流 CFL
///   * 垂直 CFL（层厚最小处）
///   * 物理过程稳定指标（PBL/微物理）
///   * 状态的非有限值出现（立即中止）
///
/// 文献：[D16] WRF ARW 第 3 章（CFL 与自适应步长）；[B2] Durran 第 3 章。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::config { struct ModelConfig; }

namespace vibe::timeint {

/// 稳定性诊断
struct StabilityReport {
  Real cfl_advection = Real(0);
  Real cfl_acoustic = Real(0);
  Real cfl_vertical = Real(0);
  Real cfl_physics = Real(0);
  Real max_wind = Real(0);
  Real min_dz = Real(0);
  bool nonfinite = false;
  bool stable = true;
  std::string message;
};

/// 时间步控制器
class TimeStepController {
 public:
  TimeStepController(const config::ModelConfig& cfg, const grid::Grid& g);

  /// 计算当前状态的稳定性报告
  StabilityReport inspect(const dyn::State& s, Real dt) const;

  /// 建议的下一个时间步
  Real suggest_dt(const dyn::State& s, Real dt_current) const;

  /// 强制上限/下限
  void set_bounds(Real dt_min, Real dt_max);

  /// 是否应该因为不稳定而中止
  bool should_abort(const StabilityReport& r) const;

  Real dt_min() const noexcept { return dt_min_; }
  Real dt_max() const noexcept { return dt_max_; }

 private:
  const grid::Grid* grid_;
  Real cfl_target_ = Real(0.8);
  Real dt_min_ = Real(0.1);
  Real dt_max_ = Real(600);
  Real damping_ = Real(0.85);
  bool adaptive_ = false;
  int  max_nonfinite_ = 1;
};

}  // namespace vibe::timeint

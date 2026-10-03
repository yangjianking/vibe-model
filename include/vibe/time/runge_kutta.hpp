#pragma once
/// @file runge_kutta.hpp
/// @brief 三阶 Runge-Kutta + 声波子步（分裂显式）积分器。
///
/// 算法（Wicker & Skamarock 2002，[T5]）
/// -----------------------------------
/// 记大时间步为 dt，声波子步为 dtau = dt / n。
/// RK3 使用低存储的 SSP 形式（[T5] 式 2.4）：
///
///     phi^(1) = phi^t + dt/3 * F(phi^t)
///     phi^(2) = phi^t + dt/2 * F(phi^(1))
///     phi^(t+dt) = phi^t + dt   * F(phi^(2))
///
/// 其中 F 为慢过程趋势。每一步之后用 n 个声波子步推进线性声波-重力波项，
/// 每个子步采用前向-后向格式（[D1] 第 3 节；[D6]）：
///
///     rho^(tau+dtau) = rho^tau - dtau * div(rho0 u^tau)          （前向）
///     pi'^(tau+dtau) 由 rho^(tau+dtau) 的状态方程诊断
///     u^(tau+dtau)   = u^tau - dtau * cp theta0 grad(pi'^(tau+dtau)) （后向）
///
/// 该格式对声波无条件稳定的子步长度约为 dtau <= dx / (2 c_s)，因此
/// n 的选择应满足 n >= 2 * c_s * dt / min(dx, dy)。垂直方向若采用
/// 隐式（HEVI），则垂直声波项被隐式处理，n 可进一步减小。
///
/// HEVI
/// ----
/// 当 `hevi_vertical_implicit = true` 时，垂直声波项由垂直三对角求解
/// 隐式处理（[T2] Tapp & White 1976；[T6] Cullen 1990），子步数只受水平
/// 声波 CFL 约束。这是本模式在陡峭地形与高分辨率下的推荐配置。
///
/// 文献：[D1][D5][D6][T2][T5][T6][T15]。

#include <memory>

#include "vibe/common/types.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/time/integrator.hpp"

namespace vibe::config { struct ModelConfig; }

namespace vibe::timeint {

/// RK3 + 声波子步
class RungeKutta3Integrator final : public Integrator {
 public:
  RungeKutta3Integrator(const grid::Grid& g, const dyn::ReferenceState& ref,
                        const config::ModelConfig& cfg,
                        dyn::Equations& eq);

  void step(dyn::State& s, const StepContext& ctx) override;
  const char* name() const noexcept override { return "rk3_acoustic"; }
  const IntegratorStats& stats() const noexcept override { return stats_; }
  void reset_stats() override { stats_ = {}; }
  Real max_stable_dt(const dyn::State& s) const override;

  /// 单个声波子步（供测试直接调用）
  void acoustic_substep(dyn::State& s, Real dtau, Real time);

  /// 是否启用垂直隐式（HEVI）
  void set_hevi_vertical_implicit(bool v) noexcept { hevi_ = v; }
  bool hevi_vertical_implicit() const noexcept { return hevi_; }

  /// 循环边界条件（理想试验用）
  void set_periodic(bool x, bool y) noexcept { periodic_x_ = x; periodic_y_ = y; }

 private:
  /// 慢过程趋势 + 可选物理（物理过程需要非 const 状态以写入诊断量）
  void slow_tendency(dyn::State& s, dyn::Tendency& d, Real dt);

  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  dyn::Equations* eq_;
  dyn::Tendency scratch_, slow_;
  IntegratorStats stats_{};
  bool hevi_ = false;
  bool periodic_x_ = true, periodic_y_ = true;
  Real divergence_damping_ = Real(0);
};

}  // namespace vibe::timeint

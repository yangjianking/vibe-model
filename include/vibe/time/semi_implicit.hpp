#pragma once
/// @file semi_implicit.hpp
/// @brief 半隐式积分器（线性声波-重力波全隐式 -> 3D Helmholtz）。
///
/// 与分裂显式的区别
/// ----------------
/// 分裂显式（RK3 + 声波子步）用许多小的显式子步来满足声波 CFL；
/// 半隐式则把线性声波-重力波项用 Crank-Nicolson 隐式处理，一个
/// Helmholtz 求解就覆盖整个 dt，因而 dt 只受**平流** CFL 与非线性
/// 稳定性限制，典型可放宽 5-20 倍（[T1][T2][T6][T7]）。
///
/// 时间离散（[T7] Bénard 2003 的中心隐式形式）
/// -------------------------------------------
///     phi^{n+1} = phi^n + dt * N(phi^{n+1/2}) + dt/2 * ( L phi^n + L phi^{n+1} )
///
/// 其中 N 为非线性项（二阶外推），L 为线性声波-重力波算子。
/// 消去 pi' 后得到关于 pi'^{n+1} 的 Helmholtz 方程：
///
///     div(a grad pi'^{n+1}) - b pi'^{n+1} = f(phi^n, N)
///
/// 非线性项用二阶 Adams-Bashforth 或 RK2 外推保证整体二阶精度。
///
/// 数值稳定性
/// ----------
/// 放大因子分析（[T7] 第 3 节）表明：对线性声波与重力波，
/// Crank-Nicolson 格式的放大因子模恒为 1（中性），而对流项由
/// AB2/RK2 的稳定域约束。实践中的经验法则：
///
///     dt <= CFL_adv * min(dx, dy) / |U|_max ,
///     CFL_adv ~ 0.8 （四阶平流），0.4（二阶中心）
///
/// 文献：[T1][T2][T3][T4][T6][T7][T8][T9][T14]。

#include <memory>

#include "vibe/common/types.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/time/helmholtz.hpp"
#include "vibe/time/integrator.hpp"

namespace vibe::config { struct ModelConfig; }

namespace vibe::timeint {

/// 半隐式积分器
class SemiImplicitIntegrator final : public Integrator {
 public:
  SemiImplicitIntegrator(const grid::Grid& g, const dyn::ReferenceState& ref,
                         const config::ModelConfig& cfg, dyn::Equations& eq,
                         std::unique_ptr<HelmholtzSolver> solver);

  void step(dyn::State& s, const StepContext& ctx) override;
  const char* name() const noexcept override { return "semi_implicit"; }
  const IntegratorStats& stats() const noexcept override { return stats_; }
  void reset_stats() override { stats_ = {}; }
  Real max_stable_dt(const dyn::State& s) const override;

  /// 外层（非线性）迭代次数；>1 时对非线性项做定点迭代以提升精度
  void set_outer_iterations(int n) noexcept { outer_ = n; }
  int  outer_iterations() const noexcept { return outer_; }

  /// 是否启用耗散时间滤波（Robert-Asselin-Williams，[T3]）
  void set_time_filter(Real alpha) noexcept { ra_filter_ = alpha; }

 private:
  /// 计算非线性趋势 + 线性算子右端
  void assemble_rhs(const dyn::State& s_prev, const dyn::State& s_cur,
                    const dyn::Tendency& n_prev, const dyn::Tendency& n_cur,
                    Real dt, grid::Field<Real>& rhs);

  /// 由 pi' 回写全部状态
  void update_from_pressure(dyn::State& s, const grid::Field<Real>& pi_new,
                            Real dt);

  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  dyn::Equations* eq_;
  std::unique_ptr<HelmholtzSolver> solver_;

  grid::Field<Real> alpha_coef_, beta_coef_;   ///< Helmholtz 系数
  grid::Field<Real> rhs_, pi_new_, pi_prev_;
  dyn::Tendency n_prev_, n_cur_;
  dyn::Tendency scratch_;

  IntegratorStats stats_{};
  int outer_ = 1;
  Real ra_filter_ = Real(0);
  bool first_step_ = true;
};

}  // namespace vibe::timeint

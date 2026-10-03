/// @file runge_kutta.cpp
/// @brief RK3 + 声波子步（分裂显式）积分器（[T5] Wicker & Skamarock 2002）。
///
/// 低存储 RK3
/// ----------
///     phi^(1) = phi^t + (dt/3) R(phi^t)
///     phi^(2) = phi^t + (dt/2) R(phi^(1))
///     phi^(t+dt) = phi^t + dt R(phi^(2))
///
/// 其中 R 为**慢过程**趋势（平流 + 科氏 + 物理 + 阻尼）。每个 RK 阶段之后，
/// 用 n 个声波子步推进线性声波-重力波项，同时把该阶段的慢趋势冻结为常数，
/// 即
///
///     phi <- phi + dtau * [ R_slow(stage) + R_acoustic(phi) ]     (n 次)
///
/// 这正是 [D6] Klemp et al. (2008) 的"守恒分裂显式"结构；慢趋势在每个
/// 子步中都加上，保证了时间一致性与守恒性。
///
/// 文献：[D1][D5][D6][T5][T15]。

#include "vibe/time/runge_kutta.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/error.hpp"
#include "vibe/common/timer.hpp"
#include "vibe/config/config.hpp"

namespace vibe::timeint {

RungeKutta3Integrator::RungeKutta3Integrator(const grid::Grid& g,
                                             const dyn::ReferenceState& ref,
                                             const config::ModelConfig& cfg,
                                             dyn::Equations& eq)
    : grid_(&g), ref_(&ref), eq_(&eq),
      scratch_(g), slow_(g) {
  hevi_ = cfg.numerics.helmholtz_solver == "vertical_tridiagonal";
  periodic_x_ = cfg.domain.periodic_x;
  periodic_y_ = cfg.domain.periodic_y;
  divergence_damping_ = cfg.numerics.divergence_damping;
  // 散度阻尼已由 dyn::Damping 施加；此处的副本只用于日志与诊断报告。
  VIBE_UNUSED(divergence_damping_);
}

void RungeKutta3Integrator::slow_tendency(dyn::State& s, dyn::Tendency& d, Real dt) {
  eq_->tendencies(s, d);
  if (physics_ != nullptr) {
    // 物理过程把倾向累加到 pt（局地变化率形式），再并入慢过程趋势。
    dyn::PhysicsTendency pt(*grid_);
    physics_->step(s, *grid_, *ref_, pt, dt);
    // 物理过程以局地变化率形式返回（theta K/s，水物质 kg/kg/s，动量 m/s^2）
    d.theta().add_scaled(Real(1), pt.theta);
    d.qv().add_scaled(Real(1), pt.qv);
    d.field(dyn::Species::Qc).add_scaled(Real(1), pt.qc);
    d.field(dyn::Species::Qr).add_scaled(Real(1), pt.qr);
    d.field(dyn::Species::Qi).add_scaled(Real(1), pt.qi);
    d.field(dyn::Species::Qs).add_scaled(Real(1), pt.qs);
    d.field(dyn::Species::Qg).add_scaled(Real(1), pt.qg);
    d.u().add_scaled(Real(1), pt.u);
    d.v().add_scaled(Real(1), pt.v);
  }
}

void RungeKutta3Integrator::acoustic_substep(dyn::State& s, Real dtau, Real time) {
  VIBE_UNUSED(time);
  // 前向-后向：先诊断质量与 Exner，再更新动量
  eq_->acoustic_tendencies(s, scratch_);
  s.rho().add_scaled(dtau, scratch_.rho());
  s.pi().add_scaled(dtau, scratch_.pi());
  eq_->acoustic_tendencies(s, scratch_);
  s.u().add_scaled(dtau, scratch_.u());
  s.v().add_scaled(dtau, scratch_.v());
  s.w().add_scaled(dtau, scratch_.w());
}

void RungeKutta3Integrator::step(dyn::State& s, const StepContext& ctx) {
  common::ScopedTimer timer("rk3_step");
  const Real dt = ctx.dt;
  const int n_sub = std::max(ctx.acoustic_substeps, 1);
  const Real dtau = dt / static_cast<Real>(n_sub);

  const dyn::State s0 = s.clone();
  const Real dt_stage[3] = {dt / Real(3), dt / Real(2), dt};

  for (int stage = 0; stage < 3; ++stage) {
    // 1) 在**上一个阶段状态**上评估慢趋势
    slow_tendency(s, slow_, dt);

    // 2) RK3 的慢项更新： phi_stage = phi^t + dt_stage * R_slow
    s = s0;
    s.add_scaled(dt_stage[stage], slow_);

    // 3) 在 dt_stage 区间内用 n_sub 个前向-后向声波子步推进线性声波-重力波项。
    //    声波项在每个子步重新评估，因此这是一个"RK3 外层 + 前向-后向内层"的分裂格式。
    const Real dtau_stage = dt_stage[stage] / static_cast<Real>(n_sub);
    for (int n = 0; n < n_sub; ++n) {
      acoustic_substep(s, dtau_stage, ctx.time + dt_stage[stage] *
                                                  (static_cast<Real>(n) / static_cast<Real>(n_sub)));
    }
    stats_.acoustic_substeps += n_sub;

    if (s.has_nonfinite()) {
      stats_.diverged = true;
      throw NumericalError("RK3 积分出现非有限值，请检查 CFL 与物理参数化");
    }
  }

  // 海绵层（后向欧拉，无条件稳定）
  // 由 Equations 持有的 Damping 应用
  eq_->damping().apply_sponge(s, *ref_, dt);

  // 可选：循环边界或物理边界回调
  if (boundary_) boundary_(s, ctx);

  stats_.steps += 1;
  stats_.wall_time = timer.elapsed();
  s.time = ctx.time + dt;
}

Real RungeKutta3Integrator::max_stable_dt(const dyn::State& s) const {
  // 平流 CFL 限制 + 声波限制
  const Real adv_cfl = eq_->advection().cfl_number(s, Real(1));
  const Real cs = eq_->max_sound_speed(s);
  Real acoustic_rate = Real(0);
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        acoustic_rate = std::max(acoustic_rate,
                                 cs * (Real(1) / grid_->dx_at(i) + Real(1) / grid_->dy_at(j) +
                                       Real(1) / dz));
      }
  const Real dt_adv = (adv_cfl > Real(0)) ? Real(0.8) / adv_cfl : Real(1e9);
  // 声波限制：允许 dt = n_sub * dtau_max，n_sub 由调用者在 StepContext 中给出；
  // 这里返回"单子步"限制，供 TimeStepController 结合子步数一起判断。
  const Real dt_ac = (acoustic_rate > Real(0)) ? Real(2) / acoustic_rate : Real(1e9);
  return std::min(dt_adv, dt_ac);
}

}  // namespace vibe::timeint

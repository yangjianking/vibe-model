/// @file timestep_control.cpp
/// @brief 时间步控制与稳定性诊断。

#include "vibe/time/timestep_control.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "vibe/common/constants.hpp"
#include "vibe/config/config.hpp"

namespace vibe::timeint {

TimeStepController::TimeStepController(const config::ModelConfig& cfg, const grid::Grid& g)
    : grid_(&g) {
  cfl_target_ = cfg.time.cfl_target;
  dt_min_ = std::max(cfg.time.dt * Real(0.01), Real(1e-3));
  dt_max_ = cfg.time.dt * Real(10);
  adaptive_ = cfg.time.adaptive_dt;
}

StabilityReport TimeStepController::inspect(const dyn::State& s, Real dt) const {
  StabilityReport r;
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        const Real uc = Real(0.5) * (s.u()(i, j, k) + s.u()(i + 1, j, k));
        const Real vc = Real(0.5) * (s.v()(i, j, k) + s.v()(i, j + 1, k));
        const Real wc = Real(0.5) * (s.w()(i, j, k) + s.w()(i, j, k + 1));
        const Real speed = std::sqrt(uc * uc + vc * vc + wc * wc);
        r.max_wind = std::max(r.max_wind, speed);
        r.min_dz = (r.min_dz == Real(0)) ? dz : std::min(r.min_dz, dz);
        r.cfl_advection =
            std::max(r.cfl_advection, dt * (std::abs(uc) / dx + std::abs(vc) / dy + std::abs(wc) / dz));
        const Real t = s.theta()(i, j, k) * s.pi()(i, j, k);
        const Real cs = std::sqrt((kCp / kCv) * kRd * std::max(t, Real(1)));
        r.cfl_acoustic = std::max(r.cfl_acoustic,
                                  dt * cs * (Real(1) / dx + Real(1) / dy + Real(1) / dz));
        r.cfl_vertical = std::max(r.cfl_vertical, dt * (std::abs(wc) + cs) / dz);
      }

  r.nonfinite = s.has_nonfinite();
  r.stable = !r.nonfinite && r.cfl_advection < Real(1.5) && r.cfl_vertical < Real(2.0);
  std::ostringstream os;
  os << "CFL 平流=" << r.cfl_advection << " 声波=" << r.cfl_acoustic
     << " 垂直=" << r.cfl_vertical << " 最大风速=" << r.max_wind
     << " 最小层厚=" << r.min_dz << " m";
  if (r.nonfinite) os << " [非有限值]";
  r.message = os.str();
  return r;
}

Real TimeStepController::suggest_dt(const dyn::State& s, Real dt_current) const {
  if (!adaptive_) return dt_current;
  const StabilityReport r = inspect(s, dt_current);
  if (r.cfl_advection <= Real(0)) return std::min(dt_max_, dt_current * Real(1.5));
  Real dt_new = damping_ * cfl_target_ / r.cfl_advection * dt_current;
  dt_new = clamp(dt_new, dt_min_, dt_max_);
  return dt_new;
}

void TimeStepController::set_bounds(Real dt_min, Real dt_max) {
  dt_min_ = dt_min;
  dt_max_ = dt_max;
}

bool TimeStepController::should_abort(const StabilityReport& r) const {
  VIBE_UNUSED(max_nonfinite_);
  return r.nonfinite || r.cfl_advection > Real(4.0);
}

}  // namespace vibe::timeint

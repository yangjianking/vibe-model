/// @file equations.cpp
/// @brief 控制方程装配（[D1][D2][D5][D6][D13]）。
///
/// 预报方程组（地形追随坐标，pi' 为 Exner 扰动）
/// --------------------------------------------
///     du/dt = -u.grad(u) - cp * theta * dpi'/dx + f v + D_u
///     dv/dt = -u.grad(v) - cp * theta * dpi'/dy - f u + D_v
///     dw/dt = -u.grad(w) - cp * theta * G^{-1} dpi'/dzeta + B + D_w
///     d rho/dt = -div(rho u)
///     d theta/dt = -u.grad(theta) + Q/(Pi cp T)
///     d q_x/dt = -u.grad(q_x) + S_x
///
/// 浮力项 B（含虚温效应与水凝物负重，[D1] 式 2.8）
///     B = g [ theta'/theta0 + 0.608 qv - (qc+qr+qi+qs+qg) ]
///
/// 状态方程（诊断 pi）
///     pi = [ rho Rd theta (1 + qv/eps) / (1 + qv) / p00 ]^{Rd/cv}
///
/// 线性声波子系统的 Exner 方程（由状态方程在定 theta 下的时间微分得到）
///     d pi'/dt = (Rd/cv) * (pi0/rho0) * d rho'/dt
///
/// 文献：[D1][D2][D5][D6][D13][D16]。

#include "vibe/dyn/equations.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/config/config.hpp"

namespace vibe::dyn {

namespace {
/// 虚温因子：gamma_f = (1 + qv/eps)/(1 + qv)
inline Real moist_factor(Real qv) {
  return (Real(1) + qv / kEpsilonVap) / (Real(1) + qv);
}
constexpr Real kBuoyancyMoist = Real(0.608);   ///< Rv/Rd - 1
}  // namespace

Equations::Equations(const grid::Grid& g, const ReferenceState& ref,
                     const config::ModelConfig& cfg)
    : grid_(&g), ref_(&ref) {
  adv_ = std::make_unique<Advection>(g,
                                     advection_from_string(cfg.numerics.advection),
                                     cfg.numerics.flux_form);
  CoriolisMode cm = CoriolisMode::FPlane;
  if (cfg.numerics.coriolis == "none") cm = CoriolisMode::None;
  else if (cfg.numerics.coriolis == "betaplane") cm = CoriolisMode::BetaPlane;
  cor_ = std::make_unique<Coriolis>(g, cm, cfg.numerics.fplane_latitude);
  damp_ = std::make_unique<Damping>(g, cfg.numerics.sponge_start_fraction * g.geom().z_top,
                                    g.geom().z_top, cfg.numerics.sponge_alpha,
                                    cfg.numerics.divergence_order,
                                    cfg.numerics.divergence_damping);
  moist_ = cfg.physics.microphysics != "none";
}

void Equations::tendencies(const State& s, Tendency& d) const {
  d.zero();
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();

  // ---- 动量平流 + 科氏 + 数值阻尼 ----
  adv_->momentum(s, d);
  cor_->apply(s, d);
  damp_->tendencies(s, d);

  // ---- 水平气压梯度力 ----
  grid::Field<Real>& du = d.u();
  grid::Field<Real>& dv = d.v();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Int ip = std::min(i, nx - 1);
        const Int im = std::max(i - 1, 0);
        const Real dx = grid_->dx_at(ip);
        const Real th = Real(0.5) * (s.theta()(im, j, k) + s.theta()(ip, j, k));
        du(i, j, k) -= kCp * th * (s.pi()(ip, j, k) - s.pi()(im, j, k)) / dx;
      }
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j <= ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int jp = std::min(j, ny - 1);
        const Int jm = std::max(j - 1, 0);
        const Real dy = grid_->dy_at(jp);
        const Real th = Real(0.5) * (s.theta()(i, jm, k) + s.theta()(i, jp, k));
        dv(i, j, k) -= kCp * th * (s.pi()(i, jp, k) - s.pi()(i, jm, k)) / dy;
      }

  // ---- 垂直气压梯度 + 浮力 ----
  grid::Field<Real>& dw = d.w();
  for (Int k = 0; k <= nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int kc = std::min(std::max(k, 0), nz - 1);
        const Int km = std::max(k - 1, 0);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(kc) * scale, Real(1));
        const Real th = Real(0.5) * (s.theta()(i, j, km) + s.theta()(i, j, kc));
        dw(i, j, k) -= kCp * th * (s.pi()(i, j, kc) - s.pi()(i, j, km)) / dz;

        const Real th0 = ref_->theta0()(i, j, kc);
        const Real thp = s.theta()(i, j, kc) - th0;
        const Real qcond = s.field(Species::Qc)(i, j, kc) + s.field(Species::Qr)(i, j, kc) +
                           s.field(Species::Qi)(i, j, kc) + s.field(Species::Qs)(i, j, kc) +
                           s.field(Species::Qg)(i, j, kc);
        dw(i, j, k) += kGravity * (thp / std::max(th0, Real(1)) +
                                   kBuoyancyMoist * s.qv()(i, j, kc) - qcond);
      }

  // ---- 质量守恒（守恒形式，全密度） ----
  grid::Field<Real>& drho = d.rho();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        const Real fx = s.u()(i + 1, j, k) * Real(0.5) * (s.rho()(i, j, k) + s.rho()(i + 1, j, k)) -
                        s.u()(i, j, k) * Real(0.5) * (s.rho()(std::max(i - 1, 0), j, k) + s.rho()(i, j, k));
        const Real fy = s.v()(i, j + 1, k) * Real(0.5) * (s.rho()(i, j, k) + s.rho()(i, j + 1, k)) -
                        s.v()(i, j, k) * Real(0.5) * (s.rho()(i, std::max(j - 1, 0), k) + s.rho()(i, j, k));
        const Real fz = s.w()(i, j, k + 1) * Real(0.5) * (s.rho()(i, j, k) + s.rho()(i, j, std::min(k + 1, nz - 1))) -
                        s.w()(i, j, k) * Real(0.5) * (s.rho()(i, j, std::max(k - 1, 0)) + s.rho()(i, j, k));
        drho(i, j, k) = -(fx / dx + fy / dy + fz / dz);
      }

  // ---- 位温与水的平流（非守恒形式，二阶/高阶由 Advection 决定） ----
  adv_->scalar_nonconservative(s.theta(), s.u(), s.v(), s.w(), d.theta());
  adv_->scalar_nonconservative(s.qv(), s.u(), s.v(), s.w(), d.qv());
  if (moist_) {
    for (int sp = static_cast<int>(Species::Qc); sp <= static_cast<int>(Species::Qg); ++sp) {
      const Species species = static_cast<Species>(sp);
      adv_->scalar_nonconservative(s.field(species), s.u(), s.v(), s.w(), d.field(species));
    }
  }

  // 水物质正定保护：若已是 0 且平流给出负趋势，则截断（避免负混合比）
  for (int sp = static_cast<int>(Species::Qv); sp <= static_cast<int>(Species::Qg); ++sp) {
    const Species species = static_cast<Species>(sp);
    const grid::Field<Real>& q = s.field(species);
    grid::Field<Real>& dq = d.field(species);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          if (q(i, j, k) <= Real(0) && dq(i, j, k) < Real(0)) dq(i, j, k) = Real(0);
        }
  }
}

void Equations::acoustic_tendencies(const State& s, Tendency& d) const {
  d.zero();
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();

  // 线性动量方程：du/dt = -cp * theta0 * grad(pi')
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Int ip = std::min(i, nx - 1);
        const Int im = std::max(i - 1, 0);
        const Real dx = grid_->dx_at(ip);
        const Real th0 = Real(0.5) * (ref_->theta0()(im, j, k) + ref_->theta0()(ip, j, k));
        d.u()(i, j, k) = -kCp * th0 * (s.pi()(ip, j, k) - s.pi()(im, j, k)) / dx;
      }
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j <= ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int jp = std::min(j, ny - 1);
        const Int jm = std::max(j - 1, 0);
        const Real dy = grid_->dy_at(jp);
        const Real th0 = Real(0.5) * (ref_->theta0()(i, jm, k) + ref_->theta0()(i, jp, k));
        d.v()(i, j, k) = -kCp * th0 * (s.pi()(i, jp, k) - s.pi()(i, jm, k)) / dy;
      }
  for (Int k = 0; k <= nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int kc = std::min(std::max(k, 0), nz - 1);
        const Int km = std::max(k - 1, 0);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(kc) * scale, Real(1));
        const Real th0 = Real(0.5) * (ref_->theta0()(i, j, km) + ref_->theta0()(i, j, kc));
        d.w()(i, j, k) = -kCp * th0 * (s.pi()(i, j, kc) - s.pi()(i, j, km)) / dz;
      }

  // 线性质量方程：d rho'/dt = -div(rho0 u)
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        const Real fx = s.u()(i + 1, j, k) * Real(0.5) * (ref_->rho0()(i, j, k) + ref_->rho0()(std::min(i + 1, nx - 1), j, k)) -
                        s.u()(i, j, k) * Real(0.5) * (ref_->rho0()(std::max(i - 1, 0), j, k) + ref_->rho0()(i, j, k));
        const Real fy = s.v()(i, j + 1, k) * Real(0.5) * (ref_->rho0()(i, j, k) + ref_->rho0()(i, std::min(j + 1, ny - 1), k)) -
                        s.v()(i, j, k) * Real(0.5) * (ref_->rho0()(i, std::max(j - 1, 0), k) + ref_->rho0()(i, j, k));
        const Real fz = s.w()(i, j, k + 1) * Real(0.5) * (ref_->rho0()(i, j, k) + ref_->rho0()(i, j, std::min(k + 1, nz - 1))) -
                        s.w()(i, j, k) * Real(0.5) * (ref_->rho0()(i, j, std::max(k - 1, 0)) + ref_->rho0()(i, j, k));
        d.rho()(i, j, k) = -(fx / dx + fy / dy + fz / dz);
      }

  // 线性化 Exner 方程：d pi'/dt = (Rd/cv) (pi0/rho0) d rho'/dt
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real r0 = std::max(ref_->rho0()(i, j, k), Real(1e-8));
        d.pi()(i, j, k) = (kRd / kCv) * (ref_->pi0()(i, j, k) / r0) * d.rho()(i, j, k);
      }
}

void Equations::diagnose(State& s) const {
  // 由状态方程反解 pi，使 (rho, theta, qv) 与 pi 相容：
  //   pi = [ rho Rd theta (1 + qv/eps)/(1 + qv) / p00 ]^{Rd/cv}
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real qv = std::max(s.qv()(i, j, k), Real(0));
        const Real num = s.rho()(i, j, k) * kRd * s.theta()(i, j, k) * moist_factor(qv);
        const Real val = num / kP0;
        s.pi()(i, j, k) = (val > Real(0)) ? std::pow(val, kRd / kCv) : s.pi()(i, j, k);
      }
}

void Equations::linearized_coefficients(const ReferenceState& ref,
                                        grid::Field<Real>& alpha,
                                        grid::Field<Real>& beta) const {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real cs2 = ref.sound_speed2(i, j, k);
        alpha(i, j, k) = Real(1) / std::max(ref.rho0()(i, j, k) * cs2, Real(1e-12));
        beta(i, j, k) = Real(1) / std::max(ref.rho0()(i, j, k), Real(1e-12));
      }
}

Real Equations::max_sound_speed(const State& s) const {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  Real best = Real(0);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real gamma = kCp / kCv;
        const Real t = s.theta()(i, j, k) * s.pi()(i, j, k);
        best = std::max(best, std::sqrt(gamma * kRd * std::max(t, Real(1))));
      }
  return best;
}

}  // namespace vibe::dyn

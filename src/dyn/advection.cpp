/// @file advection.cpp
/// @brief 平流离散的实现：2/4/6 阶中心与 5 阶 WENO。
///
/// 二阶中心（C-grid，[D3]）
/// -----------------------
/// 标量在体心，u 在 x 面，于是
///
///     (u dq/dx)_i  ~  [0.5 (u_i + u_{i+1})] * (q_{i+1} - q_{i-1}) / (2 dx)
///
/// 四阶与六阶把梯度换成 5 点 / 7 点中心差分：
///
///     dq/dx|_4 = ( q_{i-2} - 8 q_{i-1} + 8 q_{i+1} - q_{i+2} ) / (12 dx)
///     dq/dx|_6 = ( -q_{i-3} + 9 q_{i-2} - 45 q_{i-1} + 45 q_{i+1} - 9 q_{i+2} + q_{i+3} ) / (60 dx)
///
/// 守恒形式（有限体积，[D6][D13]）
/// ------------------------------
///     d(rho q)/dt = -div(rho q u)
/// 面上通量用 2 阶中心插值：
///     F_x(i) = u(i) * 0.5(rho_{i-1}+rho_i) * 0.5(q_{i-1}+q_i)
/// 该形式在周期边界下严格守恒（除浮点舍入）。
///
/// WENO5（[D7][D8]）
/// ----------------
/// 在面 i+1/2 上做左右偏重构，用非线性权重组合 3 个候选模板；
/// 迎风选择由面通量的特征速度符号决定（局部 Lax-Friedrichs 分裂）。

#include "vibe/dyn/advection.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/error.hpp"

namespace vibe::dyn {

AdvectionScheme advection_from_string(const std::string& s) {
  if (s == "central4") return AdvectionScheme::Central4;
  if (s == "central6") return AdvectionScheme::Central6;
  if (s == "weno5") return AdvectionScheme::WENO5;
  if (s == "upwind3") return AdvectionScheme::Upwind3;
  return AdvectionScheme::Central2;
}

Advection::Advection(const grid::Grid& g, AdvectionScheme scheme, bool flux_form)
    : grid_(&g), scheme_(scheme), flux_form_(flux_form) {
  switch (scheme) {
    case AdvectionScheme::Central2: order_ = 2; break;
    case AdvectionScheme::Central4: order_ = 4; break;
    case AdvectionScheme::Central6: order_ = 6; break;
    case AdvectionScheme::WENO5:    order_ = 5; break;
    case AdvectionScheme::Upwind3:  order_ = 3; break;
  }
  // 高阶与 WENO 需要更宽的 halo
  const Int need = (order_ + 1) / 2 + 1;
  VIBE_CHECK(g.halo() >= need);
}

// ---------------------------------------------------------------------------
// WENO5 重构
// ---------------------------------------------------------------------------

void Advection::weno5_reconstruct(Real qm2, Real qm1, Real q0, Real q1, Real q2,
                                  Real& q_left, Real& q_right) {
  constexpr Real eps = Real(1e-6);
  const Real d0 = Real(0.1), d1 = Real(0.6), d2 = Real(0.3);

  // 左偏重构（用于面右侧的值 q^-_{i+1/2}）
  const Real v0 = (Real(2) * qm2 - Real(7) * qm1 + Real(11) * q0) / Real(6);
  const Real v1 = (-qm1 + Real(5) * q0 + Real(2) * q1) / Real(6);
  const Real v2 = (Real(2) * q0 + Real(5) * q1 - q2) / Real(6);
  const Real b0 = Real(13.0 / 12.0) * sqr(qm2 - Real(2) * qm1 + q0) +
                  Real(0.25) * sqr(qm2 - Real(4) * qm1 + Real(3) * q0);
  const Real b1 = Real(13.0 / 12.0) * sqr(qm1 - Real(2) * q0 + q1) +
                  Real(0.25) * sqr(qm1 - q1);
  const Real b2 = Real(13.0 / 12.0) * sqr(q0 - Real(2) * q1 + q2) +
                  Real(0.25) * sqr(Real(3) * q0 - Real(4) * q1 + q2);
  const Real a0 = d0 / sqr(eps + b0);
  const Real a1 = d1 / sqr(eps + b1);
  const Real a2 = d2 / sqr(eps + b2);
  const Real wsum = a0 + a1 + a2;
  q_left = (a0 * v0 + a1 * v1 + a2 * v2) / wsum;

  // 右偏重构（镜像）：用于面左侧的值 q^+_{i+1/2}
  const Real u0 = (Real(2) * q2 - Real(7) * q1 + Real(11) * q0) / Real(6);
  const Real u1 = (-q1 + Real(5) * q0 + Real(2) * qm1) / Real(6);
  const Real u2 = (Real(2) * q0 + Real(5) * qm1 - qm2) / Real(6);
  const Real g0 = Real(13.0 / 12.0) * sqr(q2 - Real(2) * q1 + q0) +
                  Real(0.25) * sqr(q2 - Real(4) * q1 + Real(3) * q0);
  const Real g1 = Real(13.0 / 12.0) * sqr(q1 - Real(2) * q0 + qm1) +
                  Real(0.25) * sqr(q1 - qm1);
  const Real g2 = Real(13.0 / 12.0) * sqr(q0 - Real(2) * qm1 + qm2) +
                  Real(0.25) * sqr(Real(3) * q0 - Real(4) * qm1 + qm2);
  const Real c0 = d0 / sqr(eps + g0);
  const Real c1 = d1 / sqr(eps + g1);
  const Real c2 = d2 / sqr(eps + g2);
  const Real wsum2 = c0 + c1 + c2;
  q_right = (c0 * u0 + c1 * u1 + c2 * u2) / wsum2;
}

// ---------------------------------------------------------------------------
// 梯度（中心差分，按阶数选择模板）
// ---------------------------------------------------------------------------

namespace {

inline Real grad_x(const grid::Field<Real>& q, Int i, Int j, Int k, Real inv_dx, int order) {
  switch (order) {
    case 4:
      return (q.clamp_at(i - 2, j, k) - Real(8) * q.clamp_at(i - 1, j, k) +
              Real(8) * q.clamp_at(i + 1, j, k) - q.clamp_at(i + 2, j, k)) *
             (inv_dx / Real(12));
    case 6:
      return (-q.clamp_at(i - 3, j, k) + Real(9) * q.clamp_at(i - 2, j, k) -
              Real(45) * q.clamp_at(i - 1, j, k) + Real(45) * q.clamp_at(i + 1, j, k) -
              Real(9) * q.clamp_at(i + 2, j, k) + q.clamp_at(i + 3, j, k)) *
             (inv_dx / Real(60));
    default:
      return (q.clamp_at(i + 1, j, k) - q.clamp_at(i - 1, j, k)) * (Real(0.5) * inv_dx);
  }
}

inline Real grad_y(const grid::Field<Real>& q, Int i, Int j, Int k, Real inv_dy, int order) {
  switch (order) {
    case 4:
      return (q.clamp_at(i, j - 2, k) - Real(8) * q.clamp_at(i, j - 1, k) +
              Real(8) * q.clamp_at(i, j + 1, k) - q.clamp_at(i, j + 2, k)) *
             (inv_dy / Real(12));
    case 6:
      return (-q.clamp_at(i, j - 3, k) + Real(9) * q.clamp_at(i, j - 2, k) -
              Real(45) * q.clamp_at(i, j - 1, k) + Real(45) * q.clamp_at(i, j + 1, k) -
              Real(9) * q.clamp_at(i, j + 2, k) + q.clamp_at(i, j + 3, k)) *
             (inv_dy / Real(60));
    default:
      return (q.clamp_at(i, j + 1, k) - q.clamp_at(i, j - 1, k)) * (Real(0.5) * inv_dy);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 标量平流
// ---------------------------------------------------------------------------

void Advection::scalar(const grid::Field<Real>& q, const grid::Field<Real>& u,
                       const grid::Field<Real>& v, const grid::Field<Real>& w,
                       const grid::Field<Real>& rho, grid::Field<Real>& dqdt) const {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();

  if (scheme_ == AdvectionScheme::WENO5) {
    // 通量形式的 WENO5，配合局部 Lax-Friedrichs 分裂
    auto reconstruct_face = [&](Real qm2, Real qm1, Real q0, Real q1, Real q2, Real vel) {
      Real ql, qr;
      weno5_reconstruct(qm2, qm1, q0, q1, q2, ql, qr);
      const Real alpha = std::max(std::abs(vel), Real(1e-3));
      return Real(0.5) * (vel * (ql + qr) - alpha * (qr - ql));
    };
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
          const Real dz = grid_->dzeta(k) * (grid_->geom().z_top - grid_->terrain(i, j));
          const Real flux_e = reconstruct_face(q.clamp_at(i - 2, j, k), q.clamp_at(i - 1, j, k),
                                               q(i, j, k), q.clamp_at(i + 1, j, k),
                                               q.clamp_at(i + 2, j, k), u(i + 1, j, k));
          const Real flux_w = reconstruct_face(q.clamp_at(i - 3, j, k), q.clamp_at(i - 2, j, k),
                                               q.clamp_at(i - 1, j, k), q(i, j, k),
                                               q.clamp_at(i + 1, j, k), u(i, j, k));
          const Real flux_n = reconstruct_face(q.clamp_at(i, j - 2, k), q.clamp_at(i, j - 1, k),
                                               q(i, j, k), q.clamp_at(i, j + 1, k),
                                               q.clamp_at(i, j + 2, k), v(i, j + 1, k));
          const Real flux_s = reconstruct_face(q.clamp_at(i, j - 3, k), q.clamp_at(i, j - 2, k),
                                               q.clamp_at(i, j - 1, k), q(i, j, k),
                                               q.clamp_at(i, j + 1, k), v(i, j, k));
          Real flux_t = Real(0), flux_b = Real(0);
          if (k + 2 < nz || k >= 2) {
            flux_t = reconstruct_face(q.clamp_at(i, j, k - 2), q.clamp_at(i, j, k - 1),
                                      q(i, j, k), q.clamp_at(i, j, k + 1),
                                      q.clamp_at(i, j, k + 2), w(i, j, k + 1));
            flux_b = reconstruct_face(q.clamp_at(i, j, k - 3), q.clamp_at(i, j, k - 2),
                                      q.clamp_at(i, j, k - 1), q(i, j, k),
                                      q.clamp_at(i, j, k + 1), w(i, j, k));
          }
          dqdt(i, j, k) = -(flux_e - flux_w) / dx - (flux_n - flux_s) / dy -
                          (flux_t - flux_b) / dz;
        }
    return;
  }

  if (!flux_form_) {
    scalar_nonconservative(q, u, v, w, dqdt);
    return;
  }

  // 守恒形式（2 阶面通量）
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = grid_->dzeta(k) * scale;

        auto fx = [&](Int f) {
          const Real rface = Real(0.5) * (rho.clamp_at(f - 1, j, k) + rho.clamp_at(f, j, k));
          const Real qface = Real(0.5) * (q.clamp_at(f - 1, j, k) + q.clamp_at(f, j, k));
          return u(f, j, k) * rface * qface;
        };
        auto fy = [&](Int f) {
          const Real rface = Real(0.5) * (rho.clamp_at(i, f - 1, k) + rho.clamp_at(i, f, k));
          const Real qface = Real(0.5) * (q.clamp_at(i, f - 1, k) + q.clamp_at(i, f, k));
          return v(i, f, k) * rface * qface;
        };
        auto fz = [&](Int f) {
          const Real rface = Real(0.5) * (rho.clamp_at(i, j, f - 1) + rho.clamp_at(i, j, f));
          const Real qface = Real(0.5) * (q.clamp_at(i, j, f - 1) + q.clamp_at(i, j, f));
          return w(i, j, f) * rface * qface;
        };

        const Real div = (fx(i + 1) - fx(i)) / dx + (fy(j + 1) - fy(j)) / dy +
                         (fz(k + 1) - fz(k)) / dz;
        dqdt(i, j, k) = -div / std::max(rho(i, j, k), Real(1e-12));
      }
}

void Advection::scalar_nonconservative(const grid::Field<Real>& q,
                                       const grid::Field<Real>& u,
                                       const grid::Field<Real>& v,
                                       const grid::Field<Real>& w,
                                       grid::Field<Real>& dqdt) const {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = grid_->dzeta(k) * scale;
        const Real ua = Real(0.5) * (u(i, j, k) + u(i + 1, j, k));
        const Real va = Real(0.5) * (v(i, j, k) + v(i, j + 1, k));
        const Real wa = Real(0.5) * (w(i, j, k) + w(i, j, k + 1));
        dqdt(i, j, k) = -(ua * grad_x(q, i, j, k, Real(1) / dx, order_) +
                          va * grad_y(q, i, j, k, Real(1) / dy, order_) +
                          wa * (q.clamp_at(i, j, k + 1) - q.clamp_at(i, j, k - 1)) *
                               (Real(0.5) / dz));
      }
}

// ---------------------------------------------------------------------------
// 动量平流
// ---------------------------------------------------------------------------

void Advection::momentum(const State& s, Tendency& d) const {
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  const grid::Field<Real>& u = s.u();
  const grid::Field<Real>& v = s.v();
  const grid::Field<Real>& w = s.w();
  grid::Field<Real>& du = d.u();
  grid::Field<Real>& dv = d.v();
  grid::Field<Real>& dw = d.w();

  // u 方程：u 位于面 x=i，y/z 在体心
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Int ip = std::min(i, nx - 1);
        const Real dx = grid_->dx_at(ip), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(ip, j);
        const Real dz = grid_->dzeta(std::min(k, nz - 1)) * scale;
        const Real uc = Real(0.5) * (u(std::max(i - 1, 0), j, k) + u(ip, j, k));
        const Real ux = (u(ip, j, k) - u(std::max(i - 1, 0), j, k)) / dx;
        const Real uy = (u(ip, std::min(j + 1, ny - 1), k) - u(ip, std::max(j - 1, 0), k)) /
                        (Real(2) * dy);
        const Real uz = (u(ip, j, std::min(k + 1, nz - 1)) - u(ip, j, std::max(k - 1, 0))) /
                        (Real(2) * dz);
        const Real vc = Real(0.25) * (v(std::max(i - 1, 0), std::max(j - 1, 0), k) +
                                      v(ip, std::max(j - 1, 0), k) +
                                      v(std::max(i - 1, 0), j, k) + v(ip, j, k));
        const Real wc = Real(0.5) * (w(ip, j, k) + w(ip, j, std::min(k + 1, nz - 1)));
        du(i, j, k) -= uc * ux + vc * uy + wc * uz;
      }

  // v 方程
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j <= ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int jp = std::min(j, ny - 1);
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(jp);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, jp);
        const Real dz = grid_->dzeta(std::min(k, nz - 1)) * scale;
        const Real vc = Real(0.5) * (v(i, std::max(j - 1, 0), k) + v(i, jp, k));
        const Real vy = (v(i, jp, k) - v(i, std::max(j - 1, 0), k)) / dy;
        const Real vx = (v(std::min(i + 1, nx - 1), jp, k) - v(std::max(i - 1, 0), jp, k)) /
                        (Real(2) * dx);
        const Real vz = (v(i, jp, std::min(k + 1, nz - 1)) - v(i, jp, std::max(k - 1, 0))) /
                        (Real(2) * dz);
        const Real uc = Real(0.25) * (u(std::max(i - 1, 0), std::max(j - 1, 0), k) +
                                      u(i, std::max(j - 1, 0), k) +
                                      u(std::max(i - 1, 0), jp, k) + u(i, jp, k));
        const Real wc = Real(0.5) * (w(i, jp, k) + w(i, jp, std::min(k + 1, nz - 1)));
        dv(i, j, k) -= uc * vx + vc * vy + wc * vz;
      }

  // w 方程
  for (Int k = 0; k <= nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Int kc = std::min(std::max(k, 0), nz - 1);
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = grid_->dzeta(kc) * scale;
        const Real wc = Real(0.5) * (w(i, j, kc) + w(i, j, std::min(k, nz)));
        const Real wz = (w(i, j, std::min(k, nz)) - w(i, j, std::max(k - 1, 0))) / dz;
        const Real wx = (w(std::min(i + 1, nx - 1), j, std::min(k, nz)) -
                         w(std::max(i - 1, 0), j, std::min(k, nz))) / (Real(2) * dx);
        const Real wy = (w(i, std::min(j + 1, ny - 1), std::min(k, nz)) -
                         w(i, std::max(j - 1, 0), std::min(k, nz))) / (Real(2) * dy);
        const Real uc = Real(0.5) * (u(i, j, kc) + u(i + 1, j, kc));
        const Real vc = Real(0.5) * (v(i, j, kc) + v(i, j + 1, kc));
        dw(i, j, k) -= uc * wx + vc * wy + wc * wz;
      }
}

// ---------------------------------------------------------------------------
// CFL
// ---------------------------------------------------------------------------

Real Advection::cfl_number(const State& s, Real dt) const {
  Real worst = Real(0);
  const Int nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dx = grid_->dx_at(i), dy = grid_->dy_at(j);
        const Real scale = grid_->geom().z_top - grid_->terrain(i, j);
        const Real dz = std::max(grid_->dzeta(k) * scale, Real(1));
        const Real uu = Real(0.5) * (s.u()(i, j, k) + s.u()(i + 1, j, k));
        const Real vv = Real(0.5) * (s.v()(i, j, k) + s.v()(i, j + 1, k));
        const Real ww = Real(0.5) * (s.w()(i, j, k) + s.w()(i, j, k + 1));
        worst = std::max(worst, dt * (std::abs(uu) / dx + std::abs(vv) / dy + std::abs(ww) / dz));
      }
  return worst;
}

}  // namespace vibe::dyn

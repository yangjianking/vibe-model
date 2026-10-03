/// @file reference_state.cpp
/// @brief 静力参考态的构造与诊断（[D1] Klemp & Wilhelmson 1978）。
///
/// 离散
/// ----
/// 由层界面的 theta0 剖面从地面向上积分 Exner 函数：
///
///     pi0(z_{k+1}) = pi0(z_k) - g * (z_{k+1} - z_k) / (cp * theta0_mid)
///
/// 采用梯形平均 theta0_mid = 0.5*(theta0(z_k) + theta0(z_{k+1}))，
/// 与动力学的垂直差分格式相容（二阶）。地面值由 p_surf 给出：
///
///     pi0_surf = (p_surf / p00)^{Rd/cp}

#include "vibe/dyn/reference_state.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace vibe::dyn {

namespace {

/// 由 pi0 与 theta0 得到密度：rho0 = p00 pi0^{cv/Rd} / (Rd theta0)
Real density_from(Real pi0, Real theta0) {
  return kP0 * std::pow(pi0, kCv / kRd) / (kRd * theta0);
}

/// 由 pi0 得到气压：p = p00 pi0^{cp/Rd}
Real pressure_from(Real pi0) { return kP0 * std::pow(pi0, kCp / kRd); }

}  // namespace

ReferenceState ReferenceState::isothermal(const grid::Grid& g, Real theta0, Real p_surf,
                                          Real qv0) {
  ReferenceState rs;
  rs.grid_ = &g;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  rs.theta0_ = grid::Field<Real>(g, grid::Stagger::Cell, "theta0");
  rs.pi0_ = grid::Field<Real>(g, grid::Stagger::Cell, "pi0");
  rs.rho0_ = grid::Field<Real>(g, grid::Stagger::Cell, "rho0");
  rs.p0_ = grid::Field<Real>(g, grid::Stagger::Cell, "p0");
  rs.qv0_ = grid::Field<Real>(g, grid::Stagger::Cell, "qv0");

  const Real pi_surf = std::pow(p_surf / kP0, kRd / kCp);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real z = g.z_center(i, j, k);
        const Real pi = pi_surf - kGravity * z / (kCp * theta0);
        rs.theta0_(i, j, k) = theta0;
        rs.pi0_(i, j, k) = pi;
        rs.p0_(i, j, k) = pressure_from(pi);
        rs.rho0_(i, j, k) = density_from(pi, theta0);
        rs.qv0_(i, j, k) = qv0;
      }
  return rs;
}

ReferenceState ReferenceState::from_profile(const grid::Grid& g,
                                            const std::vector<Real>& z,
                                            const std::vector<Real>& theta,
                                            Real p_surf, Real qv0) {
  VIBE_CHECK(z.size() == theta.size());
  VIBE_CHECK(z.size() >= 2);
  ReferenceState rs;
  rs.grid_ = &g;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  rs.theta0_ = grid::Field<Real>(g, grid::Stagger::Cell, "theta0");
  rs.pi0_ = grid::Field<Real>(g, grid::Stagger::Cell, "pi0");
  rs.rho0_ = grid::Field<Real>(g, grid::Stagger::Cell, "rho0");
  rs.p0_ = grid::Field<Real>(g, grid::Stagger::Cell, "p0");
  rs.qv0_ = grid::Field<Real>(g, grid::Stagger::Cell, "qv0");

  auto theta_at = [&](Real zz) {
    if (zz <= z.front()) return theta.front();
    if (zz >= z.back()) return theta.back();
    Size k = 0;
    while (k + 2 < z.size() && z[k + 1] < zz) ++k;
    const Real w = (z[k + 1] - z[k] > Real(0)) ? (zz - z[k]) / (z[k + 1] - z[k]) : Real(0);
    return lerp(theta[k], theta[k + 1], w);
  };

  const Real pi_surf = std::pow(p_surf / kP0, kRd / kCp);
  for (Int j = 0; j < ny; ++j)
    for (Int i = 0; i < nx; ++i) {
      // 层界面高度
      std::vector<Real> zi(static_cast<Size>(nz) + 1);
      for (Int k = 0; k <= nz; ++k) zi[static_cast<Size>(k)] = g.z_interface(i, j, k);
      // 从地面向上积分 dpi/dz = -g/(cp theta0)
      Real pi = pi_surf;
      const Real th_surf = theta_at(zi[0]);
      rs.theta0_(i, j, 0) = th_surf;
      for (Int k = 0; k < nz; ++k) {
        const Real th_lo = theta_at(zi[static_cast<Size>(k)]);
        const Real th_hi = theta_at(zi[static_cast<Size>(k + 1)]);
        const Real th_mid = Real(0.5) * (th_lo + th_hi);
        const Real dz = zi[static_cast<Size>(k + 1)] - zi[static_cast<Size>(k)];
        pi -= kGravity * dz / (kCp * th_mid);
        const Real th_c = Real(0.5) * (th_lo + th_hi);
        rs.theta0_(i, j, k) = th_c;
        rs.pi0_(i, j, k) = pi;
        rs.p0_(i, j, k) = pressure_from(pi);
        rs.rho0_(i, j, k) = density_from(pi, th_c);
        rs.qv0_(i, j, k) = qv0;
      }
    }
  return rs;
}

ReferenceState ReferenceState::standard_atmosphere(const grid::Grid& g, Real p_surf) {
  // US Standard Atmosphere 1976 的简化两层：
  //   对流层 0-11 km: T = 288.15 - 6.5e-3 z (K)
  //   平流层 11-20 km: T = 216.65 (K)
  // 先给出温度廓线，再转为位温。
  std::vector<Real> z, theta;
  for (int k = 0; k <= 40; ++k) {
    const Real zz = static_cast<Real>(k) * Real(500);
    const Real T = (zz <= Real(11000)) ? Real(288.15) - Real(6.5e-3) * zz : Real(216.65);
    // 用标准气压公式得到位温
    Real p;
    if (zz <= Real(11000)) {
      p = Real(101325) * std::pow(T / Real(288.15), Real(9.80665) / (Real(287.05) * Real(6.5e-3)));
    } else {
      p = Real(22632) * std::exp(-Real(9.80665) * (zz - Real(11000)) / (Real(287.05) * Real(216.65)));
    }
    z.push_back(zz);
    theta.push_back(T * std::pow(kP0 / p, kRd / kCp));
  }
  return from_profile(g, z, theta, p_surf);
}

Real ReferenceState::brunt_vaisala2(Int i, Int j, Int k) const {
  const Real g = kGravity;
  const Real th = theta0_(i, j, k);
  Real dthdz;
  if (k == 0) {
    dthdz = (theta0_(i, j, 1) - theta0_(i, j, 0)) * grid_->inv_dz(i, j, 0);
  } else if (k == grid_->nz() - 1) {
    dthdz = (theta0_(i, j, k) - theta0_(i, j, k - 1)) * grid_->inv_dz(i, j, k);
  } else {
    const Real dz = grid_->z_center(i, j, k + 1) - grid_->z_center(i, j, k - 1);
    dthdz = (theta0_(i, j, k + 1) - theta0_(i, j, k - 1)) / std::max(dz, Real(1e-6));
  }
  return g / std::max(th, Real(1)) * dthdz;
}

Real ReferenceState::sound_speed2(Int i, Int j, Int k) const {
  // c_s^2 = gamma * Rd * T = (cp/cv) * Rd * theta0 * pi0
  const Real gamma = kCp / kCv;
  return gamma * kRd * theta0_(i, j, k) * pi0_(i, j, k);
}

Real ReferenceState::hydrostatic_residual() const {
  Real worst = Real(0);
  for (Int k = 0; k < grid_->nz(); ++k)
    for (Int j = 0; j < grid_->ny(); ++j)
      for (Int i = 0; i < grid_->nx(); ++i) {
        const Real dz = grid_->dzeta(k) * (grid_->geom().z_top - grid_->terrain(i, j));
        if (dz <= Real(0)) continue;
        Real dpi;
        if (k < grid_->nz() - 1) {
          dpi = (pi0_(i, j, k + 1) - pi0_(i, j, k)) / dz;
        } else {
          dpi = (pi0_(i, j, k) - pi0_(i, j, k - 1)) / dz;
        }
        const Real rhs = -kGravity / (kCp * theta0_(i, j, k));
        worst = std::max(worst, std::abs(dpi - rhs) / (std::abs(rhs) + Real(1e-12)));
      }
  return worst;
}

std::string ReferenceState::describe() const {
  std::ostringstream os;
  if (!valid()) return "参考态：未初始化";
  const Real ts = theta0_.stats().mean;
  const Real ps = p0_.at(0, 0, 0);
  const Real pt = p0_.at(0, 0, grid_->nz() - 1);
  os << "参考态：theta0 平均 " << ts << " K, 地面气压 " << ps << " Pa, 顶层气压 " << pt
     << " Pa, 静力残差(相对) " << hydrostatic_residual();
  return os.str();
}

}  // namespace vibe::dyn

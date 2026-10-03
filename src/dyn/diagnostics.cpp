/// @file diagnostics.cpp
/// @brief 诊断量的实现。
///
/// Ertel 位涡（[B4] Holton & Hakim 4.5 节）
///     PV = (1/rho) * omega_a . grad(theta)
///     omega_a = ( -dv/dz, du/dz, f + dv/dx - du/dy )
///
/// 相对涡度 / 散度（C-grid，二阶中心）
///     zeta = dv/dx - du/dy
///     div  = du/dx + dv/dy
///
/// CAPE（[P11] Kain & Fritsch 1990 气块法）
///     CAPE = g * sum_k (Tv_parcel - Tv_env)/Tv_env * dz
/// 气块从起始层干绝热上升至 LCL，之后沿湿绝热上升。
///
/// 雷达反射率（[P3] Thompson et al. 2008 附录 A 的简化形式）
///     Z = 3.63e9 (rho qr)^{1.75} + ... [mm^6 m^-3]，再转 dBZ
///
/// 海平面气压（[D16] 第 3 章）
///     p_sl = p_sfc * exp( g * z_sfc / (Rd * T_v,mean) )

#include "vibe/dyn/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "vibe/common/constants.hpp"

namespace vibe::dyn {

void Diagnostics::allocate(const grid::Grid& g) {
  auto mk = [&](const char* n) { return grid::Field<Real>(g, grid::Stagger::Cell, n); };
  vorticity = mk("vorticity");
  divergence = mk("divergence");
  pv = mk("pv");
  pressure = mk("pressure");
  temperature = mk("temperature");
  theta_e = mk("theta_e");
  rh = mk("rh");
  dewpoint = mk("dewpoint");
  height = mk("height");
  reflectivity = mk("reflectivity");
}

Real exner_to_pressure(Real pi) { return kP0 * std::pow(pi, kCp / kRd); }

Real exner_to_temperature(Real pi, Real theta) { return theta * pi; }

Real theta_e_from_rho(Real theta, Real qv, Real pi) {
  // 相当位温（Bolton 1980 近似）
  //   T_L = 2840/(3.5 ln T - ln p - 4.805) + 55   (T in K, p in hPa)
  //   theta_e = T * (1000/p)^{0.2854} * exp( (3.376/T_L - 0.00254) qv*1000 )
  const Real T = theta * pi;
  const Real p_hPa = exner_to_pressure(pi) * Real(0.01);
  if (qv <= Real(0) || T <= Real(0) || p_hPa <= Real(0)) {
    return T * std::pow(Real(1000) / std::max(p_hPa, Real(1e-6)), Real(0.2854));
  }
  const Real lnT = std::log(T);
  const Real T_L = Real(2840) / (Real(3.5) * lnT - std::log(p_hPa) - Real(4.805)) + Real(55);
  return T * std::pow(Real(1000) / p_hPa, Real(0.2854)) *
         std::exp((Real(3.376) / T_L - Real(0.00254)) * qv * Real(1000));
}

Real relative_humidity(Real pi, Real theta, Real qv, Real qc) {
  VIBE_UNUSED(qc);
  const Real T = theta * pi;
  const Real p = exner_to_pressure(pi);
  // 饱和水汽压（Bolton 1980）：es = 611.2 exp(17.67 (T-273.15)/(T-29.65))
  const Real tc = T - kT0;
  const Real es = Real(611.2) * std::exp(Real(17.67) * tc / (T - Real(29.65)));
  const Real qs = kEpsilonVap * es / std::max(p - es, Real(1));
  return (qs > Real(0)) ? clamp(qv / qs, Real(0), Real(1.2)) : Real(0);
}

Real dewpoint_from_qv(Real pressure, Real qv) {
  // 由混合比反解水汽压，再用 Bolton 反演露点
  const Real e = std::max(pressure * qv / (kEpsilonVap + qv), Real(1e-6));
  const Real ln = std::log(e / Real(611.2));
  return Real(243.5) * ln / (Real(17.67) - ln) + kT0;
}

Real relative_vorticity(const State& s, const grid::Grid& g, Int i, Int j, Int k) {
  const Real dx = g.dx_at(i), dy = g.dy_at(j);
  // dv/dx 在体心：v 在 y 面，先用 x 方向平均（保持涡度的 2 阶精度）
  const Real dvdx = (Real(0.5) * (s.v()(i + 1, j, k) + s.v()(i + 1, j + 1, k)) -
                     Real(0.5) * (s.v()(i - 1, j, k) + s.v()(i - 1, j + 1, k))) /
                    (Real(2) * dx);
  const Real dudy = (Real(0.5) * (s.u()(i, j + 1, k) + s.u()(i + 1, j + 1, k)) -
                     Real(0.5) * (s.u()(i, j - 1, k) + s.u()(i + 1, j - 1, k))) /
                    (Real(2) * dy);
  return dvdx - dudy;
}

Real divergence(const State& s, const grid::Grid& g, Int i, Int j, Int k) {
  const Real dx = g.dx_at(i), dy = g.dy_at(j);
  return (s.u()(i + 1, j, k) - s.u()(i, j, k)) / dx + (s.v()(i, j + 1, k) - s.v()(i, j, k)) / dy;
}

Real ertel_pv(const State& s, const ReferenceState& ref, Int i, Int j, Int k) {
  const grid::Grid& g = ref.grid();
  const Real dx = g.dx_at(i), dy = g.dy_at(j);
  const Real scale = g.geom().z_top - g.terrain(i, j);
  const Real dz = std::max(g.dzeta(std::min(k, g.nz() - 1)) * scale, Real(1));
  const Real f = Real(2) * kOmega * std::sin(kReferenceLat * kDegToRad);
  const Real zeta = relative_vorticity(s, g, i, j, k) + f;
  const Real dudz = (s.u()(i, j, std::min(k + 1, g.nz() - 1)) -
                     s.u()(i, j, std::max(k - 1, 0))) / (Real(2) * dz);
  const Real dvdz = (s.v()(i, j, std::min(k + 1, g.nz() - 1)) -
                     s.v()(i, j, std::max(k - 1, 0))) / (Real(2) * dz);
  const Real dthdx = (s.theta()(std::min(i + 1, g.nx() - 1), j, k) -
                      s.theta()(std::max(i - 1, 0), j, k)) / (Real(2) * dx);
  const Real dthdy = (s.theta()(i, std::min(j + 1, g.ny() - 1), k) -
                      s.theta()(i, std::max(j - 1, 0), k)) / (Real(2) * dy);
  const Real dthdz = (s.theta()(i, j, std::min(k + 1, g.nz() - 1)) -
                      s.theta()(i, j, std::max(k - 1, 0))) / (Real(2) * dz);
  const Real rho = std::max(s.rho()(i, j, k), Real(1e-8));
  return (-dvdz * dthdx + dudz * dthdy + zeta * dthdz) / rho;
}

Real reflectivity_dbz(Real qr, Real qs, Real qg, Real rho, Real t) {
  // 简化形式：Z = 3.63e9 (rho qr)^1.75 + 1e10 (rho qs)^1.75 + 4e10 (rho qg)^1.75
  // 雪在 0 摄氏度以上按融化因子衰减（[P3]）
  const Real rr = std::max(rho * qr, Real(0));
  const Real rs = std::max(rho * qs, Real(0));
  const Real rg = std::max(rho * qg, Real(0));
  const Real melt = (t > kT0) ? std::exp(-Real(0.1) * (t - kT0)) : Real(1);
  const Real z = Real(3.63e9) * std::pow(rr, Real(1.75)) +
                 Real(1.0e10) * std::pow(rs * melt, Real(1.75)) +
                 Real(4.0e10) * std::pow(rg, Real(1.75));
  if (z < Real(1e-2)) return Real(-30);
  return Real(10) * std::log10(z);
}

CapeResult cape_cin_column(const std::vector<Real>& z, const std::vector<Real>& p,
                           const std::vector<Real>& t, const std::vector<Real>& qv,
                           Real z_start) {
  CapeResult r;
  const Size n = z.size();
  if (n < 2) return r;
  // 找到起始层
  Size k0 = 0;
  while (k0 + 1 < n && z[k0 + 1] < z_start) ++k0;

  // 气块起始热力学量
  Real tp = t[k0];
  Real qp = qv[k0];
  const Real pp0 = p[k0];

  auto esat = [](Real T) {
    const Real tc = T - kT0;
    return Real(611.2) * std::exp(Real(17.67) * tc / (T - Real(29.65)));
  };
  auto qsat = [&](Real T, Real P) {
    const Real es = esat(T);
    return kEpsilonVap * es / std::max(P - es, Real(1));
  };
  // 抬升凝结高度（露点温度法）
  const Real td = dewpoint_from_qv(pp0, qp);
  Real T_lcl = td - Real(0.0015) * (z[k0] - Real(0));
  VIBE_UNUSED(T_lcl);

  const Real kappa = kRd / kCp;
  Real tv_env0 = t[k0] * (Real(1) + Real(0.608) * qv[k0]);

  Real cape = Real(0), cin = Real(0);
  bool saturated = false;
  for (Size k = k0 + 1; k < n; ++k) {
    const Real dz = z[k] - z[k - 1];
    if (dz <= Real(0)) continue;
    // 干绝热抬升
    tp *= std::pow(p[k] / p[k - 1], kappa);
    if (!saturated) {
      const Real qs = qsat(tp, p[k]);
      if (qp >= qs) {
        saturated = true;
        qp = qs;
      }
    } else {
      // 湿绝热：用饱和调整的潜热释放近似（每层按 dT = -L/cp dqs）
      const Real qs = qsat(tp, p[k]);
      const Real dqs = std::max(qs - qp, Real(0));
      tp += kLv * dqs / kCp;
      qp = qsat(tp, p[k]);
    }
    const Real tv_parcel = tp * (Real(1) + Real(0.608) * qp);
    const Real tv_env = t[k] * (Real(1) + Real(0.608) * qv[k]);
    const Real buoy = kGravity * (tv_parcel - tv_env) / std::max(tv_env, Real(1));
    if (buoy > Real(0)) {
      cape += buoy * dz;
      if (r.lfc == Real(0)) r.lfc = z[k];
      r.el = z[k];
    } else if (cape <= Real(0)) {
      cin += -buoy * dz;
    }
  }
  VIBE_UNUSED(tv_env0);
  r.cape = cape;
  r.cin = cin;
  return r;
}

CapeResult cape_cin(const State& s, const ReferenceState& ref, Int i, Int j) {
  const grid::Grid& g = ref.grid();
  const Int nz = g.nz();
  std::vector<Real> z(static_cast<Size>(nz)), p(static_cast<Size>(nz)),
      t(static_cast<Size>(nz)), qv(static_cast<Size>(nz));
  for (Int k = 0; k < nz; ++k) {
    z[static_cast<Size>(k)] = g.z_center(i, j, k);
    p[static_cast<Size>(k)] = exner_to_pressure(s.pi()(i, j, k));
    t[static_cast<Size>(k)] = exner_to_temperature(s.pi()(i, j, k), s.theta()(i, j, k));
    qv[static_cast<Size>(k)] = std::max(s.qv()(i, j, k), Real(0));
  }
  return cape_cin_column(z, p, t, qv, z[0] + Real(500));
}

Real sea_level_pressure(const State& s, const ReferenceState& ref, Int i, Int j) {
  const grid::Grid& g = ref.grid();
  const Real zs = g.terrain(i, j);
  const Real ps = exner_to_pressure(s.pi()(i, j, 0));
  const Real ts = exner_to_temperature(s.pi()(i, j, 0), s.theta()(i, j, 0));
  const Real qv = std::max(s.qv()(i, j, 0), Real(0));
  const Real tv = ts * (Real(1) + Real(0.608) * qv);
  const Real lapse = Real(0.0065);
  const Real t_sl = tv + lapse * zs;
  return ps * std::exp(kGravity * zs / (kRd * std::max(t_sl, Real(1))));
}

void diagnose_all(const State& s, const ReferenceState& ref, Diagnostics& out) {
  const grid::Grid& g = ref.grid();
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real pi = s.pi()(i, j, k);
        const Real th = s.theta()(i, j, k);
        out.pressure(i, j, k) = exner_to_pressure(pi);
        out.temperature(i, j, k) = exner_to_temperature(pi, th);
        out.theta_e(i, j, k) = theta_e_from_rho(th, std::max(s.qv()(i, j, k), Real(0)), pi);
        out.rh(i, j, k) = relative_humidity(pi, th, std::max(s.qv()(i, j, k), Real(0)));
        out.dewpoint(i, j, k) = dewpoint_from_qv(out.pressure(i, j, k),
                                                 std::max(s.qv()(i, j, k), Real(0)));
        out.height(i, j, k) = g.z_center(i, j, k);
        out.vorticity(i, j, k) = relative_vorticity(s, g, i, j, k);
        out.divergence(i, j, k) = divergence(s, g, i, j, k);
        out.pv(i, j, k) = ertel_pv(s, ref, i, j, k);
        out.reflectivity(i, j, k) = reflectivity_dbz(
            std::max(s.field(Species::Qr)(i, j, k), Real(0)),
            std::max(s.field(Species::Qs)(i, j, k), Real(0)),
            std::max(s.field(Species::Qg)(i, j, k), Real(0)), s.rho()(i, j, k),
            out.temperature(i, j, k));
      }
}

EnergyBudget energy_budget(const State& s, const ReferenceState& ref) {
  const grid::Grid& g = ref.grid();
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  EnergyBudget e;
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real dv = g.cell_volume(i, j, k);
        const Real rho = s.rho()(i, j, k);
        const Real uc = Real(0.5) * (s.u()(i, j, k) + s.u()(i + 1, j, k));
        const Real vc = Real(0.5) * (s.v()(i, j, k) + s.v()(i, j + 1, k));
        const Real wc = Real(0.5) * (s.w()(i, j, k) + s.w()(i, j, k + 1));
        const Real T = s.theta()(i, j, k) * s.pi()(i, j, k);
        const Real z = g.z_center(i, j, k);
        e.mass += rho * dv;
        e.kinetic += Real(0.5) * rho * (uc * uc + vc * vc + wc * wc) * dv;
        e.internal += rho * kCv * T * dv;
        e.potential += rho * kGravity * z * dv;
        e.latent += rho * kLv * std::max(s.qv()(i, j, k), Real(0)) * dv;
      }
  e.total = e.kinetic + e.internal + e.potential + e.latent;
  return e;
}

std::string EnergyBudget::to_string() const {
  std::ostringstream os;
  os << "能量收支: KE=" << kinetic << " IE=" << internal << " PE=" << potential
     << " LE=" << latent << " 总计=" << total << " J, 总质量=" << mass << " kg";
  return os.str();
}

}  // namespace vibe::dyn

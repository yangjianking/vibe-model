/// @file pbl_ysu.cpp
/// @brief YSU 边界层方案（非局地 K 廓线 + 夹卷 + 逆梯度项）[P7]，
///        以及 YSU/MYJ 共用的垂直扩散求解器（Thomas 三对角）。
///
/// 文献：[P7] Hong, Noh & Dudhia (2006)；[B2] 第 3 章（隐式扩散）；
///       [P8] Mellor & Yamada (1982)。

#include "vibe/physics/pbl.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"

namespace vibe::physics {

namespace {
constexpr Real kFreeAtmosphereK = Real(0.10);  ///< 自由大气背景扩散系数 (m^2/s)
constexpr Real kMinK = Real(1.0e-6);
}  // namespace

// ===========================================================================
// 三对角与垂直扩散求解器
// ===========================================================================

// ---------------------------------------------------------------------------
// Thomas 算法（追赶法）
//   a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i
// 离散化：先向前消元（计算 c'_i = c_i/(b_i - a_i c'_{i-1}) 等），再回代。
// 稳定性：对角占优时无条件稳定；分母加符号微扰避免除零。
// 文献：[B2] 附录 A；[B11] 第 3 章。
// 复杂度：O(n)，额外内存 O(n)。
void thomas_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                  const std::vector<Real>& c, const std::vector<Real>& d,
                  std::vector<Real>& x) {
  const Size n = b.size();
  x.assign(n, Real(0));
  if (n == 0) return;
  std::vector<Real> cp(n, Real(0));
  std::vector<Real> dp(n, Real(0));
  Real denom = b[0];
  if (std::abs(denom) < Real(1.0e-30)) denom = Real(1.0e-30);
  cp[0] = (n > 1) ? c[0] / denom : Real(0);
  dp[0] = d[0] / denom;
  for (Size i = 1; i < n; ++i) {
    denom = b[i] - a[i] * cp[i - 1];
    if (std::abs(denom) < Real(1.0e-30)) denom = (denom >= Real(0)) ? Real(1.0e-30) : Real(-1.0e-30);
    cp[i] = (i + 1 < n) ? c[i] / denom : Real(0);
    dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
  }
  x[n - 1] = dp[n - 1];
  for (Size i = n - 1; i-- > 0;) x[i] = dp[i] - cp[i] * x[i + 1];
}

// ---------------------------------------------------------------------------
// 垂直扩散全隐式求解
//   dX/dt = d/dz (K dX/dz) - decay X + source
// 界面 K 取算术平均，层厚取相邻层中心的距离。
// 下边界：给定运动学通量 surface_flux_lower（向上为正），
//   对第 0 层贡献 +F_sfc/dz_0（源项形式，物理上表示地面向上输送给空气）；
// 上边界：零通量。
// 稳定性：全隐式，任意 dt、K >= 0 无条件稳定。
// 复杂度 O(nz)。
void solve_implicit_diffusion(const std::vector<Real>& k_diff, const std::vector<Real>& dz,
                              const std::vector<Real>& x, const std::vector<Real>& source,
                              const std::vector<Real>& decay, Real dt,
                              Real surface_flux_lower, std::vector<Real>& x_new) {
  const Size n = x.size();
  x_new = x;
  if (n == 0) return;
  std::vector<Real> kf(n + 1, kMinK);
  kf[0] = std::max(k_diff[0], kMinK);
  for (Size i = 0; i + 1 < n; ++i) {
    kf[i + 1] = std::max(Real(0.5) * (k_diff[i] + k_diff[i + 1]), kMinK);
  }
  kf[n] = std::max(k_diff[n - 1], kMinK);

  std::vector<Real> a(n, Real(0)), b(n, Real(0)), c(n, Real(0)), d(n, Real(0));
  for (Size i = 0; i < n; ++i) {
    const Real dzc = std::max(dz[i], Real(1.0e-6));
    const Real dz_up = (i > 0) ? Real(0.5) * (dzc + std::max(dz[i - 1], Real(1.0e-6)))
                               : Real(0.5) * dzc;
    const Real dz_dn = (i + 1 < n) ? Real(0.5) * (dzc + std::max(dz[i + 1], Real(1.0e-6)))
                                   : Real(0.5) * dzc;
    const Real ku = kf[i] / (dzc * dz_up);
    const Real kd = kf[i + 1] / (dzc * dz_dn);
    if (i > 0) a[i] = -dt * ku;
    if (i + 1 < n) c[i] = -dt * kd;
    const Real dec = (i < decay.size()) ? std::max(decay[i], Real(0)) : Real(0);
    b[i] = Real(1) + dt * (ku + kd) + dt * dec;
    Real rhs = x[i] + dt * ((i < source.size()) ? source[i] : Real(0));
    if (i == 0) rhs += dt * surface_flux_lower / dzc;
    d[i] = rhs;
  }
  thomas_solve(a, b, c, d, x_new);
}

std::string PblBase::describe() const {
  std::ostringstream os;
  os << name();
  return os.str();
}

// ===========================================================================
// YSU
// ===========================================================================

// ---------------------------------------------------------------------------
// 由总体 Richardson 数确定 PBL 高度
//   Ri_b(z) = g z (theta_v(z) - theta_v,s) / (theta_v,1 |V(z)|^2)
//   自下而上找首个 Ri_b >= Ri_c 的层，h 取其下一层高度（>= 100 m）
// 文献：[P7] 第 3 节（Ric ≈ 0.25）
// 复杂度 O(nz)。
Real YsuPbl::pbl_height_from_richardson(const PhysicsColumn& col, const SurfaceState& sfc,
                                        const SurfaceFluxes& flx, Int& top_index) const noexcept {
  const Int n = col.nz();
  top_index = 0;
  if (n <= 1) return Real(100);
  const Real zs = sfc.terrain_height;
  const Real thv_s = col.theta_v(0);  // 用最低层位温虚温作参考（缺表皮位温虚温）
  VIBE_UNUSED(flx);
  Real h = std::max(col.z[1] - zs, Real(50));
  for (Int k = 1; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real z = std::max(col.z[s] - zs, Real(1));
    const Real thv = col.theta_v(k);
    const Real spd2 = std::max(col.u[s] * col.u[s] + col.v[s] * col.v[s], Real(0.25));
    const Real rib = kGravity * z * (thv - thv_s) / (std::max(thv_s, Real(1)) * spd2);
    if (rib >= opt_.ysu_ri_critical) break;
    h = z;
    top_index = k;
  }
  return std::max(h, Real(100));
}

Real YsuPbl::convective_velocity(Real h, Real surface_heat_flux_kinematic, Real theta0) const noexcept {
  const Real f = std::max(surface_heat_flux_kinematic, Real(0));
  return std::cbrt(std::max(kGravity / std::max(theta0, Real(1)) * h * f, Real(0)));
}

//   w_s = (u*^3 + phi_m kappa w*^3 z/h)^(1/3)
Real YsuPbl::velocity_scale(Real ustar, Real wstar, Real z, Real h) const noexcept {
  const Real hh = std::max(h, Real(1));
  const Real zi = clamp(z / hh, Real(0), Real(1));
  const Real t3 = ustar * ustar * ustar +
                  opt_.ysu_wstar_coef * kVonKarman * wstar * wstar * wstar * zi;
  return std::cbrt(std::max(t3, Real(1.0e-12)));
}

//   K_m = kappa w_s z (1-z/h)^2  (z <= h)
Real YsuPbl::diffusivity_momentum(Real ws, Real z, Real h) const noexcept {
  const Real hh = std::max(h, Real(1));
  const Real zi = clamp(z / hh, Real(0), Real(1));
  const Real shape = (Real(1) - zi) * (Real(1) - zi);
  return kVonKarman * ws * z * shape;
}

//   gamma = C (w'theta')_0/(w_s h)
Real YsuPbl::counter_gradient(Real surface_heat_flux_kinematic, Real ws, Real h) const noexcept {
  return opt_.ysu_countergradient_coef * surface_heat_flux_kinematic / (std::max(ws, Real(0.1)) * std::max(h, Real(1)));
}

//   Pr = 1 + coef * z/h
Real YsuPbl::prandtl(Real z, Real h) const noexcept {
  return Real(1) + opt_.ysu_prandtl_coef * clamp(z / std::max(h, Real(1)), Real(0), Real(1));
}

Real YsuPbl::diagnose_height(const PhysicsColumn& col, const SurfaceState& sfc,
                             const SurfaceFluxes& flx) const noexcept {
  Int top = 0;
  return pbl_height_from_richardson(col, sfc, flx, top);
}

// ---------------------------------------------------------------------------
// YSU 单列积分
//
// 步骤
//   1. PBL 高度 h（Richardson 数判据）
//   2. 速度尺度 w_s，K_m = kappa w_s z (1-z/h)^2，K_h = K_m/Pr
//   3. 逆梯度项 gamma = C (w'theta')_0/(w_s h)，作为源项
//        S_gamma = [K_{k+1/2} gamma - K_{k-1/2} gamma]/dz_k
//   4. 夹卷：PBL 顶通量 F_e = -coef (w'theta')_0，在 h 上下相邻层产生 ±F_e/dz
//   5. 隐式求解 u, v, theta, qv（下边界为地面层通量）
// 复杂度 O(nz)。
Real YsuPbl::step_column(PhysicsColumn& col, SurfaceState& sfc, const SurfaceFluxes& flx,
                         Real dt, ColumnTendency& tend) {
  const Int n = col.nz();
  if (n <= 0 || !(dt > Real(0))) return Real(0);

  Int top_index = 0;
  const Real h = pbl_height_from_richardson(col, sfc, flx, top_index);
  const Real theta0 = std::max(col.theta[0], Real(1));

  // 地面运动学通量（正向上）
  const Real f_theta_sfc = -flx.ustar * flx.tstar;                 // K m/s
  const Real f_q_sfc = -flx.ustar * flx.qstar;                     // kg/kg m/s
  const Real spd = std::max(std::sqrt(col.u[0] * col.u[0] + col.v[0] * col.v[0]), Real(0.1));
  const Real f_u_sfc = -flx.ustar * flx.ustar * col.u[0] / spd;    // m^2/s^2
  const Real f_v_sfc = -flx.ustar * flx.ustar * col.v[0] / spd;

  const Real wstar = convective_velocity(h, f_theta_sfc, theta0);
  const Real ws = velocity_scale(flx.ustar, wstar, std::max(h, Real(2)), std::max(h, Real(2)));
  const Real gamma_theta = (f_theta_sfc > Real(0)) ? counter_gradient(f_theta_sfc, ws, h) : Real(0);
  const Real gamma_q = (f_q_sfc > Real(0)) ? counter_gradient(f_q_sfc, ws, h) : Real(0);

  // 界面高度与 K（界面 k 位于层 k-1 与 k 之间）
  std::vector<Real> zf;
  col.interface_heights(zf);
  const Real zs = sfc.terrain_height;
  std::vector<Real> km(static_cast<Size>(n), kMinK);
  std::vector<Real> kh(static_cast<Size>(n), kMinK);
  Real kmax = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    // 层中心处的 K（用于该层的扩散系数）
    const Real z = std::max(col.z[s] - zs, Real(1));
    Real kk = Real(0);
    if (z <= h) {
      const Real wsl = velocity_scale(flx.ustar, wstar, z, h);
      kk = diffusivity_momentum(wsl, z, h);
    } else {
      const Real wsh = velocity_scale(flx.ustar, wstar, Real(0.99) * h, h);
      const Real kh_edge = diffusivity_momentum(wsh, Real(0.99) * h, h);
      kk = kFreeAtmosphereK + kh_edge * std::exp(-(z - h) / (Real(0.1) * std::max(h, Real(1))));
    }
    km[s] = std::max(kk, kMinK);
    kh[s] = std::max(km[s] / prandtl(z, h), kMinK);
    kmax = std::max(kmax, km[s]);
  }
  max_diffusivity_ = kmax;

  // 逆梯度源项
  std::vector<Real> src_theta(static_cast<Size>(n), Real(0));
  std::vector<Real> src_q(static_cast<Size>(n), Real(0));
  const std::vector<Real> zero_src(static_cast<Size>(n), Real(0));
  const std::vector<Real> zero_decay(static_cast<Size>(n), Real(0));
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real dz = std::max(col.dz[s], Real(1.0e-6));
    const Real z_lo = std::max(zf[s] - zs, Real(0));
    const Real z_hi = std::max(zf[s + 1] - zs, Real(0));
    const Real g_lo = (z_lo <= h) ? gamma_theta : Real(0);
    const Real g_hi = (z_hi <= h) ? gamma_theta : Real(0);
    const Real k_lo = km[s];
    const Real k_hi = (s + 1 < static_cast<Size>(n)) ? km[s + 1] : km[s];
    src_theta[s] = (k_hi * g_hi - k_lo * g_lo) / dz;
    const Real q_lo = (z_lo <= h) ? gamma_q : Real(0);
    const Real q_hi = (z_hi <= h) ? gamma_q : Real(0);
    src_q[s] = (k_hi * q_hi - k_lo * q_lo) / dz;
  }

  // 夹卷：PBL 顶通量 F_e = coef * F_sfc（向上为正，对流边界层取地面通量的 0.15 倍，
  // 即"夹卷通量与地面通量同号"的能量守恒形式：PBL 顶降温、自由大气增温）。
  // [P7] 的夹卷通量比是按 theta 定义的，故水汽夹卷由 K 廓线与逆梯度项承担。
  const Int ktop = clamp(top_index, Int(0), n - 1);
  {
    const Real f_e_theta = opt_.ysu_entrainment_coef * f_theta_sfc;
    const Real dz_lo = std::max(col.dz[static_cast<Size>(ktop)], Real(1.0e-6));
    src_theta[static_cast<Size>(ktop)] -= f_e_theta / dz_lo;
    if (ktop + 1 < n) {
      const Size su = static_cast<Size>(ktop + 1);
      const Real dz_up = std::max(col.dz[su], Real(1.0e-6));
      src_theta[su] += f_e_theta / dz_up;
    }
  }

  // 隐式扩散求解并回写倾向
  std::vector<Real> x_new;
  // 动量 u
  solve_implicit_diffusion(km, col.dz, col.u, zero_src, zero_decay, dt, f_u_sfc, x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.u[s] += (x_new[s] - col.u[s]) / dt;
    col.u[s] = x_new[s];
  }
  solve_implicit_diffusion(km, col.dz, col.v, zero_src, zero_decay, dt, f_v_sfc, x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.v[s] += (x_new[s] - col.v[s]) / dt;
    col.v[s] = x_new[s];
  }
  solve_implicit_diffusion(kh, col.dz, col.theta, src_theta, zero_decay, dt, f_theta_sfc, x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.theta[s] += (x_new[s] - col.theta[s]) / dt;
    col.theta[s] = x_new[s];
  }
  solve_implicit_diffusion(kh, col.dz, col.qv, src_q, zero_decay, dt, f_q_sfc, x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.qv[s] += (x_new[s] - col.qv[s]) / dt;
    col.qv[s] = std::max(x_new[s], Real(0));
  }
  // 云水/云冰随 θ 的交换系数扩散（不含源项与逆梯度，避免过冲）
  solve_implicit_diffusion(kh, col.dz, col.qc, zero_src, zero_decay, dt, Real(0), x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.qc[s] += (x_new[s] - col.qc[s]) / dt;
    col.qc[s] = std::max(x_new[s], Real(0));
  }
  solve_implicit_diffusion(kh, col.dz, col.qi, zero_src, zero_decay, dt, Real(0), x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.qi[s] += (x_new[s] - col.qi[s]) / dt;
    col.qi[s] = std::max(x_new[s], Real(0));
  }
  return h;
}

std::string YsuPbl::describe() const {
  std::ostringstream os;
  os << "YsuPbl{ Ri_c=" << opt_.ysu_ri_critical << " 夹卷系数=" << opt_.ysu_entrainment_coef
     << " Pr=1+" << opt_.ysu_prandtl_coef << " z/h 逆梯度系数 C="
     << opt_.ysu_countergradient_coef << " 自由大气 K=" << kFreeAtmosphereK << " m^2/s }";
  return os.str();
}

// ===========================================================================
// 工厂
// ===========================================================================

std::unique_ptr<PblBase> make_pbl(PblScheme scheme, const PhysicsOptions& opt) {
  switch (scheme) {
    case PblScheme::None:
      return nullptr;
    case PblScheme::Ysu:
      return std::make_unique<YsuPbl>(opt);
    case PblScheme::Myj:
      return std::make_unique<MyjPbl>(opt);
    case PblScheme::Mynn:
      throw NotImplemented("MYNN 边界层方案仅保留接口，尚未实现");
    case PblScheme::Smagorinsky:
      throw NotImplemented("Smagorinsky 局地 K 方案仅保留接口，尚未实现");
    default:
      throw NotImplemented("未知的边界层方案编号");
  }
}

}  // namespace vibe::physics

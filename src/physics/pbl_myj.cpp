/// @file pbl_myj.cpp
/// @brief Mellor-Yamada 2.5 阶 TKE 闭合边界层方案（MYJ）[P8][P18]。
///
/// 预报方程（[P8] 式 (1)、(28)；q^2 = 2 e）
///     d e/dt = d/dz ( K_e d e/dz ) + K_m S^2 - K_h (g/theta0) dtheta_v/dz
///              - 2 e^{3/2}/(B_1 l)
///     K_m = l sqrt(2e) S_M,  K_h = l sqrt(2e) S_H,  K_e = l sqrt(2e) S_q
///     S^2 = (du/dz)^2 + (dv/dz)^2
///
/// 稳定性函数（[P8] 式 (35)；本实现采用 [P18] 的非奇异化系数形式）
///     GM = S^2,  GH = (g/theta0) dtheta_v/dz  （稳定时 GH > 0）
///     ELOQ2 = (l/q)^2
///     D     = (ADNM GM + ADNH GH) GH ELOQ2^2 + (BDNM GM + BDNH GH) ELOQ2 + 1
///     S_M   = (BSMH GH ELOQ2 + CESM) / D
///     S_H   = ((BSHM GM + BSHH GH) ELOQ2 + CESH) / D
///   ADNH = 9 A1 A2^2 (12 A1 + 3 B2) BTG^2        ADNM = 18 A1^2 A2 (B2 - 3 A2) BTG
///   BDNH = 3 A2 (7 A1 + B2) BTG                  BDNM = 6 A1^2
///   BSHH = 9 A1 A2^2 BTG                         BSHM = 18 A1^2 A2 C1
///   BSMH = -3 A1 A2 (3 A2 + 3 B2 C1 + 12 A1 C1 - B2) BTG
///   CESH = A2,  CESM = A1 (1 - 3 C1),  BTG = g/273
///
/// 混合长（Blackadar 渐近 + 平衡长度上限）
///     ELM 由平衡方程解 (l/q)^2 的二次方程得到；EL0 = alph Integral q z dz/Integral q dz
///     PBL 内  EL = min( kappa z/(1 + kappa z/EL0), ELM )
///     PBL 上  EL = min( 0.23 dz, ELM )
///
/// 文献：[P8] Mellor & Yamada (1982)；[P18] Janjic (1994)。

#include "vibe/physics/pbl.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace vibe::physics {

namespace {
constexpr Real kEpsLength = Real(0.32);   ///< 最小混合长 (m) [P18]

// 单二次方程 z^2 + b z + c = 0 的较大根（数值稳定形式，避免 b>0 时的相消）
//   z = c / (-b/2 - sqrt(b^2/4 - c))    (b >= 0, 等价于 -b/2 + sqrt(...)，但更稳定)
//   z = -b/2 + sqrt(b^2/4 - c)          (b < 0)
// 判别式 disc = b^2/4 - c < 0 时返回 0（无实根）。
Real stable_monic_root(Real b, Real c, Real disc) noexcept {
  if (disc < Real(0)) return Real(0);
  const Real sq = std::sqrt(disc);
  if (b >= Real(0)) {
    const Real den = -Real(0.5) * b - sq;
    if (std::abs(den) > Real(1.0e-30)) return c / den;
  }
  return -Real(0.5) * b + sq;
}
constexpr Real kEpsGh = Real(1.0e-9);     ///< GH 的数值下限
constexpr Real kEpsGm = Real(1.0e-9);
constexpr Real kMinK = Real(1.0e-6);
constexpr Real kFh = Real(1.01);          ///< PBL 顶判据的 TKE 松弛因子 [P18]
}  // namespace

// ===========================================================================
// 构造与派生系数
// ===========================================================================

MyjPbl::MyjPbl(const PhysicsOptions& opt) : PblBase(opt) { compute_derived(); }

void MyjPbl::compute_derived() {
  const auto& c = opt_.my;
  const Real btg = c.btg;
  d_.btg = btg;
  d_.adnh = Real(9) * c.a1 * c.a2 * c.a2 * (Real(12) * c.a1 + Real(3) * c.b2) * btg * btg;
  d_.adnm = Real(18) * c.a1 * c.a1 * c.a2 * (c.b2 - Real(3) * c.a2) * btg;
  d_.bdnh = Real(3) * c.a2 * (Real(7) * c.a1 + c.b2) * btg;
  d_.bdnm = Real(6) * c.a1 * c.a1;
  d_.bshh = Real(9) * c.a1 * c.a2 * c.a2 * btg;
  d_.bshm = Real(18) * c.a1 * c.a1 * c.a2 * c.c1;
  d_.bsmh = -Real(3) * c.a1 * c.a2 *
            (Real(3) * c.a2 + Real(3) * c.b2 * c.c1 + Real(12) * c.a1 * c.c1 - c.b2) * btg;
  d_.cesh = c.a2;
  d_.cesm = c.a1 * (Real(1) - Real(3) * c.c1);
  // 平衡方程系数与"湍流禁区"
  d_.aeqh = Real(9) * c.a1 * c.a2 * c.a2 * c.b1 * btg * btg +
            Real(9) * c.a1 * c.a2 * c.a2 * (Real(12) * c.a1 + Real(3) * c.b2) * btg * btg;
  d_.aeqm = Real(3) * c.a1 * c.a2 * c.b1 *
                (Real(3) * c.a2 + Real(3) * c.b2 * c.c1 + Real(18) * c.a1 * c.c1 - c.b2) * btg +
            Real(18) * c.a1 * c.a1 * c.a2 * (c.b2 - Real(3) * c.a2) * btg;
  d_.requ = (std::abs(d_.aeqm) > Real(1.0e-30)) ? -d_.aeqh / d_.aeqm : Real(0);
  // 近各向同性（UBRY）分支系数
  const Real epsrs = Real(1.0e-7);
  const Real denom = d_.requ * d_.adnm + d_.adnh;
  const Real ubryl = (denom != Real(0))
                         ? (Real(18) * d_.requ * c.a1 * c.a1 * c.a2 * c.b2 * c.c1 * btg +
                            Real(9) * c.a1 * c.a2 * c.a2 * c.b2 * btg * btg) / denom
                         : Real(0);
  const Real ubry3 = Real(3) * (Real(1) + epsrs) * ubryl;
  d_.aubh = Real(27) * c.a1 * c.a2 * c.a2 * c.b2 * btg * btg - d_.adnh * ubry3;
  d_.aubm = Real(54) * c.a1 * c.a1 * c.a2 * c.b2 * c.c1 * btg - d_.adnm * ubry3;
  d_.bubh = (Real(9) * c.a1 * c.a2 + Real(3) * c.a2 * c.b2) * btg - d_.bdnh * ubry3;
  d_.bubm = Real(18) * c.a1 * c.a1 * c.c1 - d_.bdnm * ubry3;
  d_.cubr = Real(1) - ubry3;
  d_.rcubr = (std::abs(d_.cubr) > Real(1.0e-12)) ? Real(1) / d_.cubr : Real(0);
}

// ===========================================================================
// 稳定性函数与平衡混合长
// ===========================================================================

void MyjPbl::stability_functions(Real gm, Real gh, Real el_o_q2, Real& sm, Real& sh) const noexcept {
  const Real gml = std::max(gm, kEpsGm);
  Real ghl = gh;
  if (std::abs(ghl) <= kEpsGh) ghl = (ghl >= Real(0)) ? kEpsGh : -kEpsGh;
  const Real x = std::max(el_o_q2, Real(1.0e-12));
  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  const Real aden = (d_.adnm * gml + d_.adnh * ghl) * ghl;
  const Real bden = d_.bdnm * gml + d_.bdnh * ghl;
  const Real rden = Real(1) / (aden * x4 + bden * x2 + Real(1));
  const Real besm = d_.bsmh * ghl;
  const Real besh = d_.bshm * gml + d_.bshh * ghl;
  sm = (besm * x2 + d_.cesm) * rden;
  sh = (besh * x2 + d_.cesh) * rden;
}

// ---------------------------------------------------------------------------
// 平衡混合长 ELM：解 (l/q)^2 的二次方程
//   稳定（GH > 0）： GM/GH <= REQU 时湍流被禁止；否则 AUBR x^2 + BUBR x + CUBR = 0
//   不稳定（GH <= 0）：ADEN x^2 + BDEN x + 1 = 0
//   再取 ELM = sqrt(q^2/x)
// 文献：[P18] 第 2 节；[P8] 附录。
// 复杂度 O(1)。
Real MyjPbl::equilibrium_length(Real gm, Real gh, Real q2) const noexcept {
  const Real gml = std::max(gm, kEpsGm);
  const Real ghl = gh;
  Real x = Real(0);
  if (ghl >= kEpsGh) {
    // 稳定分支：湍流禁区判据 GM/GH <= REQU
    if (gml / ghl <= d_.requ) return kEpsLength;
    if (std::abs(d_.cubr) < Real(1.0e-12)) return opt_.myj_el0max;
    const Real aub = (d_.aubm * gml + d_.aubh * ghl) * ghl;
    const Real bub = d_.bubm * gml + d_.bubh * ghl;
    // z^2 + BUBR z + AUBR*CUBR = 0  ->  QOL2ST = z / CUBR
    const Real c_coef = aub * d_.cubr;
    const Real disc = Real(0.25) * bub * bub - c_coef;
    if (disc < Real(0)) return opt_.myj_el0max;  // 无实根：由 Blackadar 廓线限制
    x = stable_monic_root(bub, c_coef, disc) * d_.rcubr;
  } else {
    // 不稳定分支： z^2 + BDEN z + ADEN = 0  ->  QOL2UN = z
    const Real aden = (d_.adnm * gml + d_.adnh * ghl) * ghl;
    const Real bden = d_.bdnm * gml + d_.bdnh * ghl;
    const Real disc = Real(0.25) * bden * bden - aden;
    if (disc < Real(0)) return opt_.myj_el0max;
    x = stable_monic_root(bden, aden, disc);
  }
  // x = (l/q)^2；无正根（弱切变/近中性）时混合长不受平衡条件限制
  if (!(x > Real(1.0e-12))) return opt_.myj_el0max;
  return clamp(std::sqrt(std::max(q2, Real(0)) / x), kEpsLength, opt_.myj_el0max);
}

// Blackadar 渐近混合长： EL0 = alph Integral q z dz / Integral q dz
Real MyjPbl::asymptotic_length(const std::vector<Real>& q, const std::vector<Real>& z,
                               Int pbl_top) const noexcept {
  const Int n = static_cast<Int>(q.size());
  if (n <= 0) return opt_.myj_el0min;
  const Int kt = clamp(pbl_top, Int(0), n - 1);
  Real num = Real(0), den = Real(0);
  for (Int k = 0; k <= kt; ++k) {
    const Size s = static_cast<Size>(k);
    const Real qk = std::max(q[s], Real(0));
    num += qk * z[s];
    den += qk;
  }
  Real el0 = (den > Real(0)) ? opt_.myj_alph * num / den : opt_.myj_el0min;
  return clamp(el0, opt_.myj_el0min, opt_.myj_el0max);
}

void MyjPbl::mixing_length(const PhysicsColumn& col, Int pbl_top, std::vector<Real>& el) const {
  const Int n = col.nz();
  el.assign(static_cast<Size>(n), kEpsLength);
  if (n <= 0) return;
  std::vector<Real> q(static_cast<Size>(n), Real(0));
  for (Int k = 0; k < n; ++k) {
    q[static_cast<Size>(k)] = std::sqrt(std::max(Real(2) * col.tke[static_cast<Size>(k)], Real(0)));
  }
  const Real el0 = asymptotic_length(q, col.z, pbl_top);
  std::vector<Real> elm(static_cast<Size>(n), kEpsLength);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    // 中心差分求 GM 与 GH
    Real dudz = Real(0), dvdz = Real(0), dthvdz = Real(0);
    if (k == 0 && n > 1) {
      const Real h = std::max(col.z[1] - col.z[0], Real(1.0e-3));
      dudz = (col.u[1] - col.u[0]) / h;
      dvdz = (col.v[1] - col.v[0]) / h;
      dthvdz = (col.theta_v(1) - col.theta_v(0)) / h;
    } else if (k == n - 1 && n > 1) {
      const Real h = std::max(col.z[s] - col.z[s - 1], Real(1.0e-3));
      dudz = (col.u[s] - col.u[s - 1]) / h;
      dvdz = (col.v[s] - col.v[s - 1]) / h;
      dthvdz = (col.theta_v(k) - col.theta_v(k - 1)) / h;
    } else if (n > 2) {
      const Real h = std::max(col.z[s + 1] - col.z[s - 1], Real(1.0e-3));
      dudz = (col.u[s + 1] - col.u[s - 1]) / h;
      dvdz = (col.v[s + 1] - col.v[s - 1]) / h;
      dthvdz = (col.theta_v(k + 1) - col.theta_v(k - 1)) / h;
    }
    const Real gm = dudz * dudz + dvdz * dvdz;
    const Real th0 = std::max(col.theta_v(k), Real(1));
    const Real gh = (kGravity / th0) * dthvdz;  // 稳定 > 0
    const Real q2 = std::max(Real(2) * col.tke[s], Real(1.0e-8));
    elm[s] = equilibrium_length(gm, gh, q2);
  }
  // Blackadar 廓线
  const Real zsfc = col.z.empty() ? Real(0) : (col.z[0] - Real(0.5) * col.dz[0]);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real z = std::max(col.z[s] - zsfc, Real(1));
    Real e = Real(0);
    if (k <= pbl_top) {
      const Real kz = kVonKarman * z;
      e = std::min(kz / (Real(1) + kz / std::max(el0, Real(1e-3))), elm[s]);
    } else {
      e = std::min(Real(0.23) * std::max(col.dz[s], Real(1.0e-3)), elm[s]);
    }
    el[s] = std::max(e, kEpsLength);
  }
  // 一次 1-2-1 平滑，抑制相邻层混合长的跳变
  std::vector<Real> es = el;
  for (Int k = 1; k < n - 1; ++k) {
    const Size s = static_cast<Size>(k);
    const Real sm = Real(0.5) * (el[s - 1] + el[s + 1]);
    es[s] = std::max(Real(0.5) * (sm + el[s]), kEpsLength);
  }
  el = es;
}

// PBL 顶：自顶向下第一个 TKE <= e_min * FH 的层
Int MyjPbl::pbl_top_index(const PhysicsColumn& col) const noexcept {
  const Int n = col.nz();
  if (n <= 0) return 0;
  const Real thr = opt_.myj_e_min * kFh;
  for (Int k = n - 1; k >= 0; --k) {
    if (col.tke[static_cast<Size>(k)] <= thr) return k;
  }
  return 0;
}

Real MyjPbl::diagnose_height(const PhysicsColumn& col, const SurfaceState& sfc,
                             const SurfaceFluxes& flx) const noexcept {
  VIBE_UNUSED(flx);
  const Int kt = pbl_top_index(col);
  std::vector<Real> zf;
  col.interface_heights(zf);
  const Real h = zf[static_cast<Size>(kt + 1)] - zf[0];
  return std::max(h, Real(50));
}

// ===========================================================================
// MYJ 单列积分
// ===========================================================================

// ---------------------------------------------------------------------------
// 步骤
//   1. 混合长 EL（平衡长度 + Blackadar 廓线 + 平滑）
//   2. 稳定性函数 -> K_m, K_h, K_e
//   3. TKE 预报：隐式扩散 + 切变产生 + 浮力（稳定为汇）+ 线性化耗散
//        decay = 2 sqrt(e_old)/(B1 l)
//        下边界： e(0) >= 0.5 B1^(2/3) u*^2（[P8] 的 MY 下边界条件）
//   4. 用 K_m/K_h 隐式扩散 u, v, theta, qv, qc, qi（下边界为地面通量）
//   5. PBL 高度诊断
// 复杂度 O(nz)。
Real MyjPbl::step_column(PhysicsColumn& col, SurfaceState& sfc, const SurfaceFluxes& flx,
                         Real dt, ColumnTendency& tend) {
  const Int n = col.nz();
  if (n <= 0 || !(dt > Real(0))) return Real(0);

  // 初始 TKE 保护
  for (Int k = 0; k < n; ++k) {
    col.tke[static_cast<Size>(k)] = std::max(col.tke[static_cast<Size>(k)], opt_.myj_e_min);
  }
  const Int pbl_top = pbl_top_index(col);
  std::vector<Real> zf_pbl;
  col.interface_heights(zf_pbl);
  const Real h_pbl = std::max(zf_pbl[static_cast<Size>(pbl_top + 1)] - zf_pbl[0], Real(50));
  std::vector<Real> el;
  mixing_length(col, pbl_top, el);

  // 稳定性函数与扩散系数（层中心）
  std::vector<Real> km(static_cast<Size>(n), kMinK);
  std::vector<Real> kh(static_cast<Size>(n), kMinK);
  std::vector<Real> ke(static_cast<Size>(n), kMinK);
  std::vector<Real> shear2(static_cast<Size>(n), Real(0));
  std::vector<Real> buoy(static_cast<Size>(n), Real(0));
  Real kmax = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    Real dudz = Real(0), dvdz = Real(0), dthvdz = Real(0);
    if (k == 0 && n > 1) {
      const Real h = std::max(col.z[1] - col.z[0], Real(1.0e-3));
      dudz = (col.u[1] - col.u[0]) / h;
      dvdz = (col.v[1] - col.v[0]) / h;
      dthvdz = (col.theta_v(1) - col.theta_v(0)) / h;
    } else if (k == n - 1 && n > 1) {
      const Real h = std::max(col.z[s] - col.z[s - 1], Real(1.0e-3));
      dudz = (col.u[s] - col.u[s - 1]) / h;
      dvdz = (col.v[s] - col.v[s - 1]) / h;
      dthvdz = (col.theta_v(k) - col.theta_v(k - 1)) / h;
    } else if (n > 2) {
      const Real h = std::max(col.z[s + 1] - col.z[s - 1], Real(1.0e-3));
      dudz = (col.u[s + 1] - col.u[s - 1]) / h;
      dvdz = (col.v[s + 1] - col.v[s - 1]) / h;
      dthvdz = (col.theta_v(k + 1) - col.theta_v(k - 1)) / h;
    }
    const Real th0 = std::max(col.theta_v(k), Real(1));
    shear2[s] = dudz * dudz + dvdz * dvdz;
    const Real gh = (kGravity / th0) * dthvdz;
    const Real q = std::max(std::sqrt(Real(2) * col.tke[s]), Real(1.0e-6));
    const Real el_o_q2 = el[s] / q;
    Real sm = Real(0), sh = Real(0);
    stability_functions(shear2[s], gh, el_o_q2, sm, sh);
    const Real scale = el[s] * q;  // l * q
    km[s] = std::max(scale * sm, kMinK);
    kh[s] = std::max(scale * sh, kMinK);
    ke[s] = std::max(scale * opt_.my.s_q, kMinK);
    kmax = std::max(kmax, km[s]);
    buoy[s] = -kh[s] * gh;
  }
  max_diffusivity_ = kmax;

  // TKE 预报方程
  std::vector<Real> src(static_cast<Size>(n), Real(0));
  std::vector<Real> decay(static_cast<Size>(n), Real(0));
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real e_old = std::max(col.tke[s], opt_.myj_e_min);
    src[s] = km[s] * shear2[s] + buoy[s];
    decay[s] = Real(2) * std::sqrt(e_old) / (opt_.my.b1 * std::max(el[s], kEpsLength));
  }
  std::vector<Real> e_new;
  solve_implicit_diffusion(ke, col.dz, col.tke, src, decay, dt, Real(0), e_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real e = std::max(e_new[s], opt_.myj_e_min);
    tend.tke[s] += (e - col.tke[s]) / dt;
    col.tke[s] = e;
  }
  // MY 下边界条件：q^2(sfc) = B1^{2/3} u*^2  =>  e = 0.5 B1^{2/3} u*^2
  {
    const Real e_sfc = Real(0.5) * std::pow(opt_.my.b1, Real(2.0) / Real(3.0)) * flx.ustar * flx.ustar;
    col.tke[0] = std::max(col.tke[0], e_sfc);
  }

  // 动量与水物质/热量扩散（下边界为地面层通量，正向上）
  const Real f_theta_sfc = -flx.ustar * flx.tstar;
  const Real f_q_sfc = -flx.ustar * flx.qstar;
  const Real spd = std::max(std::sqrt(col.u[0] * col.u[0] + col.v[0] * col.v[0]), Real(0.1));
  const Real f_u_sfc = -flx.ustar * flx.ustar * col.u[0] / spd;
  const Real f_v_sfc = -flx.ustar * flx.ustar * col.v[0] / spd;
  const std::vector<Real> zero_src(static_cast<Size>(n), Real(0));
  const std::vector<Real> zero_decay(static_cast<Size>(n), Real(0));
  std::vector<Real> x_new;

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
  solve_implicit_diffusion(kh, col.dz, col.theta, zero_src, zero_decay, dt, f_theta_sfc, x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.theta[s] += (x_new[s] - col.theta[s]) / dt;
    col.theta[s] = x_new[s];
  }
  solve_implicit_diffusion(kh, col.dz, col.qv, zero_src, zero_decay, dt, f_q_sfc, x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.qv[s] += (x_new[s] - col.qv[s]) / dt;
    col.qv[s] = std::max(x_new[s], Real(0));
  }
  // 云水
  solve_implicit_diffusion(kh, col.dz, col.qc, zero_src, zero_decay, dt, Real(0), x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.qc[s] += (x_new[s] - col.qc[s]) / dt;
    col.qc[s] = std::max(x_new[s], Real(0));
  }
  // 云冰
  solve_implicit_diffusion(kh, col.dz, col.qi, zero_src, zero_decay, dt, Real(0), x_new);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.qi[s] += (x_new[s] - col.qi[s]) / dt;
    col.qi[s] = std::max(x_new[s], Real(0));
  }

  VIBE_UNUSED(sfc);
  return h_pbl;
}

std::string MyjPbl::describe() const {
  std::ostringstream os;
  os << "MyjPbl{ A1=" << opt_.my.a1 << " A2=" << opt_.my.a2 << " B1=" << opt_.my.b1
     << " B2=" << opt_.my.b2 << " C1=" << opt_.my.c1 << " S_q=" << opt_.my.s_q
     << " BTG=" << d_.btg << " REQU=" << d_.requ << " EL0=[" << opt_.myj_el0min << ","
     << opt_.myj_el0max << "] m alph=" << opt_.myj_alph << " e_min=" << opt_.myj_e_min
     << " m^2/s^2 }";
  return os.str();
}

}  // namespace vibe::physics

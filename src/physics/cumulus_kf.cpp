/// @file cumulus_kf.cpp
/// @brief Kain-Fritsch 质量通量对流参数化 [P11] 与 Grell-Devenyi 集合框架 [P12]。
///
/// 文献：[P11] Kain & Fritsch (1990), JAS 47, 2784-2802；
///       [P12] Grell & Devenyi (2002), GRL 29, 1693。

#include "vibe/physics/cumulus.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"

namespace vibe::physics {

namespace {

/// 源层搜索的最大深度（约 300 hPa，[P11] 第 2 节）
constexpr Real kSourceSearchDepth = Real(3000.0);

/// 给定 theta_e 守恒条件下的饱和抬升：Newton 反解温度
///     F(T) = (T/pi) exp( Ls qsat_i(T)/(cp T) ) - theta_e = 0
///     dF/dT = (1/pi)exp(a) + (T/pi)exp(a) da/dT,
///     da/dT = Ls dqsat/dT/(cp T) - a/T,  dqsat/dT = Ls qsat/(Rv T^2)
/// 迭代 6 次，步长限幅 30 K。复杂度 O(1)。
Real saturated_temperature_from_theta_e(Real p, Real theta_e, Real t_guess) noexcept {
  const Real pi = exner_from_pressure(p);
  Real t = clamp(t_guess, Real(180), Real(320));
  for (int it = 0; it < 6; ++it) {
    const Real lv = latent_heat_vaporization(t);
    const Real qs = saturation_mixing_ratio(p, t);
    const Real a = lv * qs / (kCp * t);
    const Real f = (t / pi) * std::exp(a) - theta_e;
    const Real dqsdT = lv * qs / (kRv * t * t);
    const Real da = lv * dqsdT / (kCp * t) - a / t;
    const Real df = (Real(1) / pi) * std::exp(a) + (t / pi) * std::exp(a) * da;
    if (std::abs(df) < Real(1.0e-30)) break;
    const Real dt = clamp(f / df, Real(-30), Real(30));
    t -= dt;
    t = clamp(t, Real(180), Real(320));
    if (std::abs(dt) < Real(1.0e-6)) break;
  }
  return t;
}

}  // namespace

std::string CumulusBase::describe() const { return name(); }

// ===========================================================================
// 触发
// ===========================================================================

// ---------------------------------------------------------------------------
// 触发判据
//   1. 在最低 min(3000 m) 内取 theta_e 最大的层作为源层
//   2. 气块抬升（moist_parcel_ascent）得到 CAPE/CIN/LFC
//   3. 要求 CAPE >= kf_min_cape 且存在 LFC
//   4. 云厚（云顶由 plume_ascent 确定）由调用方检查
// 复杂度 O(nz)。
bool KainFritschCumulus::trigger(const PhysicsColumn& col, const SurfaceState& sfc,
                                 ParcelAscent& ascent, Int& source_level) const {
  const Int n = col.nz();
  source_level = -1;
  if (n < 3) return false;
  const Real zs = sfc.terrain_height;
  Real best = -kHuge;
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    if (col.z[s] - zs > kSourceSearchDepth) break;
    const Real the = equivalent_potential_temperature(col, k);
    if (the > best) {
      best = the;
      source_level = k;
    }
  }
  if (source_level < 0) return false;
  ascent = moist_parcel_ascent(col, source_level, /*entraining=*/false, Real(0));
  if (ascent.lfc < 0) return false;
  if (ascent.cape < opt_.kf_min_cape) return false;
  return true;
}

// ===========================================================================
// plume 积分
// ===========================================================================

// ---------------------------------------------------------------------------
// 卷入/卷出 plume（归一化质量通量 m(z)/m_b）
//
//   干绝热段（源层 -> 云底）：theta、qv 守恒
//   湿绝热段（云底以上）：theta_e 守恒，Newton 反解 T；
//     卷入： m_{k} = m_{k-1} exp(eps dz)；theta_e 按质量加权混合
//        theta_e_p <- (m_{k-1} theta_e_p + (m_k - m_{k-1}) theta_e_env)/m_k
//     卷出： 当浮力为负时按 delta = kf_detrainment_rate 卷出；
//            浮力连续两层为负 -> 云顶
//   凝结量： condensate_k = m_k max(qsat_{k-1} - qsat_k, 0)
//
// 复杂度 O(nz * n_iter)。
void KainFritschCumulus::plume_ascent(const PhysicsColumn& col, Int source_level, Int cloud_base,
                                      PlumeProfile& plume) const {
  const Int n = col.nz();
  plume.resize(n);
  plume.source_level = source_level;
  plume.cloud_base_level = cloud_base;
  if (n <= 0 || source_level < 0 || cloud_base < 0) return;
  if (cloud_base <= source_level) plume.cloud_base_level = source_level;

  const Size ss = static_cast<Size>(source_level);
  Real theta_e_p = equivalent_potential_temperature(col, source_level);
  Real tv_p = col.virtual_temperature(source_level);
  Real mass = Real(1);
  plume.mass_flux[ss] = mass;
  plume.theta_e[ss] = theta_e_p;
  plume.theta[ss] = col.theta[ss];
  plume.qv[ss] = col.qv[ss];
  plume.tv[ss] = tv_p;
  plume.buoyancy[ss] = Real(0);

  Real qsat_prev = saturation_mixing_ratio(col.p[ss], col.temperature(source_level));
  Int neg_count = 0;
  Int top = n - 1;
  for (Int k = source_level + 1; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real dz = std::max(col.dz[s], Real(1.0e-3));
    const Real p = col.p[s];

    // 卷入：质量通量按 exp(eps dz) 增长
    const Real dm = mass * (std::exp(opt_.kf_entrainment_rate * dz) - Real(1));
    const Real mass_new = mass + dm;
    const Real the_env = equivalent_potential_temperature(col, k);
    theta_e_p = (mass * theta_e_p + dm * the_env) / std::max(mass_new, Real(1.0e-12));
    mass = mass_new;

    // 气块温度与湿度
    Real t_p = col.temperature(k);
    Real qv_p = Real(0);
    if (k >= plume.cloud_base_level) {
      t_p = saturated_temperature_from_theta_e(p, theta_e_p, col.temperature(k));
      qv_p = saturation_mixing_ratio(p, t_p);
    } else {
      // 干绝热段：位温守恒（用源层位温；这里直接用环境位温的抬升近似）
      t_p = col.theta[ss] * exner_from_pressure(p);
      qv_p = std::max(col.qv[ss], Real(0));
    }
    const Real tv = virtual_temperature(t_p, qv_p);
    const Real b = kGravity * (tv - col.virtual_temperature(k)) / std::max(col.virtual_temperature(k), Real(1));

    // 卷出：浮力为负时按卷出率减质量
    Real delta = Real(0);
    if (b < Real(0)) {
      delta = opt_.kf_detrainment_rate;
      mass *= std::exp(-delta * dz);
      ++neg_count;
    } else {
      neg_count = 0;
    }
    plume.mass_flux[s] = mass;
    plume.theta_e[s] = theta_e_p;
    plume.theta[s] = t_p / exner_from_pressure(p);
    plume.qv[s] = qv_p;
    plume.tv[s] = tv;
    plume.buoyancy[s] = b;
    plume.detrain[s] = delta;
    if (k >= plume.cloud_base_level) {
      const Real qsat_now = saturation_mixing_ratio(p, t_p);
      plume.condensate[s] = std::max(mass * (qsat_prev - qsat_now), Real(0));
      plume.qc[s] = std::max(qsat_prev - qsat_now, Real(0));
      qsat_prev = qsat_now;
    }
    if (neg_count >= 2 && k > plume.cloud_base_level) {
      top = k;
      break;
    }
    top = k;
  }
  plume.cloud_top_level = top;
  // 云内 CAPE（沿 plume 路径的浮力积分）
  Real cape = Real(0);
  for (Int k = plume.cloud_base_level; k < top; ++k) {
    const Size s = static_cast<Size>(k);
    const Real dz = std::max(col.dz[s], Real(1.0e-3));
    cape += Real(0.5) * (plume.buoyancy[s] + plume.buoyancy[static_cast<Size>(k + 1)]) * dz;
  }
  plume.cape = std::max(cape, Real(0));
  plume.valid = (top > plume.cloud_base_level);
}

// ===========================================================================
// 闭合与效率
// ===========================================================================

// ---------------------------------------------------------------------------
// CAPE 消耗闭合（[P11] 第 3 节）
//   S    = (1/(z_t-z_b)) Integral (g/(cp theta_ve)) |d s_v/dz| dz    [1/s^2]
//          （用有限差分的 s_v = cp T + g z + Lv qv）
//   m_b  = rho_b CAPE / (tau S)
// 量纲校验： rho[kg/m^3] * CAPE[m^2/s^2] / (tau[s] * S[1/s^2]) = kg/(m^2 s)。
// 复杂度 O(nz)。
Real KainFritschCumulus::closure_mass_flux(const PhysicsColumn& col,
                                           const PlumeProfile& plume) const noexcept {
  const Int kb = plume.cloud_base_level;
  const Int kt = plume.cloud_top_level;
  if (kb < 0 || kt <= kb) return Real(0);
  Real sum = Real(0);
  Real depth = Real(0);
  for (Int k = kb; k < kt; ++k) {
    const Size s = static_cast<Size>(k);
    const Real dz = std::max(col.dz[s], Real(1.0e-3));
    const Real t_lo = col.temperature(k);
    const Real t_hi = col.temperature(k + 1);
    const Real sv_lo = kCp * t_lo + kGravity * col.z[s] + latent_heat_vaporization(t_lo) * std::max(col.qv[s], Real(0));
    const Real sv_hi = kCp * t_hi + kGravity * col.z[s + 1] +
                       latent_heat_vaporization(t_hi) * std::max(col.qv[s + 1], Real(0));
    const Real thve = std::max(col.theta_v(k), Real(1));
    sum += (kGravity / (kCp * thve)) * std::abs(sv_hi - sv_lo) / dz * dz;
    depth += dz;
  }
  if (!(depth > Real(0)) || !(sum > Real(0))) return Real(0);
  const Real s_norm = sum / depth;  // 平均 (g/(cp theta)) |ds_v/dz| [1/s^2]
  const Real rho_b = std::max(col.rho[static_cast<Size>(kb)], Real(1.0e-3));
  const Real cape = std::max(plume.cape, Real(0));
  const Real mb = rho_b * cape / (std::max(opt_.kf_closure_time, Real(1)) * s_norm);
  return std::max(mb, Real(0));
}

//   浅对流（云厚 < 4 km）取 0.2；深对流按陆面/海面取值
Real KainFritschCumulus::precipitation_efficiency(Real cloud_depth, bool over_land) const noexcept {
  if (cloud_depth < opt_.kf_min_cloud_depth) return Real(0.2);
  return over_land ? opt_.kf_precip_efficiency_land : opt_.kf_precip_efficiency_sea;
}

Int KainFritschCumulus::downdraft_origin(const PhysicsColumn& col,
                                         const PlumeProfile& plume) const noexcept {
  const Int kb = plume.cloud_base_level;
  const Int kt = plume.cloud_top_level;
  if (kb < 0 || kt <= kb) return kb;
  Int kmin = kb;
  Real min_the = kHuge;
  for (Int k = kb; k <= kt; ++k) {
    const Real the = plume.theta_e[static_cast<Size>(k)];
    if (the < min_the) {
      min_the = the;
      kmin = k;
    }
  }
  return kmin;
}

// ===========================================================================
// 单列 KF 积分
// ===========================================================================

// ---------------------------------------------------------------------------
// 步骤
//   1. 触发与 plume 积分，确定云底/云顶
//   2. 闭合得到云底质量通量 m_b
//   3. 环境响应（补偿下沉 + 卷出）
//        dtheta/dt = -(1/rho) m_b dtheta_p/dz + D_k (theta_p - theta)/(rho dz)
//        dqv/dt    = 类似
//   4. 水物质收支： L = -Integral rho dq/dt dz；P = eps_p L；
//      未降水部分 (1-eps_p)L 作为下沉气流再蒸发返回云下层（并对应降温）
//   5. 对流动量输送： d u/dt = -(1/rho) m_b d u_p/dz（简化为按 plume 质量通量）
// 复杂度 O(nz)。
Real KainFritschCumulus::step_column(PhysicsColumn& col, const SurfaceState& sfc, Real dt,
                                     ColumnTendency& tend, PhysicsDiagnostics& diag) {
  last_ = ConvectionDiagnostics{};
  const Int n = col.nz();
  if (n < 3 || !(dt > Real(0))) return Real(0);

  ParcelAscent ascent;
  Int source_level = -1;
  if (!trigger(col, sfc, ascent, source_level)) {
    last_.cape = ascent.cape;
    last_.cin = ascent.cin;
    return Real(0);
  }
  // 云底：取 LFC（若无则取源层上一层的抬升凝结高度所在层）
  Int cloud_base = (ascent.lfc >= 0) ? ascent.lfc : source_level + 1;
  cloud_base = clamp(cloud_base, source_level + 1, n - 2);
  // 云底以下（源层到云底）为干绝热段，plume 从源层开始积分
  PlumeProfile plume;
  plume_ascent(col, source_level, cloud_base, plume);
  if (!plume.valid) return Real(0);
  const Int ktop = plume.cloud_top_level;
  const Real depth = col.z[static_cast<Size>(ktop)] - col.z[static_cast<Size>(cloud_base)];
  if (depth < opt_.kf_min_cloud_depth || depth > opt_.kf_max_cloud_depth) return Real(0);

  const Real mb = closure_mass_flux(col, plume);
  if (!(mb > Real(0))) return Real(0);

  // 环境响应：补偿下沉 + 卷出
  std::vector<Real> dtheta(static_cast<Size>(n), Real(0));
  std::vector<Real> dqv(static_cast<Size>(n), Real(0));
  std::vector<Real> du(static_cast<Size>(n), Real(0));
  std::vector<Real> dv(static_cast<Size>(n), Real(0));
  for (Int k = cloud_base; k < ktop; ++k) {
    const Size s = static_cast<Size>(k);
    const Real dz = std::max(col.dz[s], Real(1.0e-3));
    const Real rho = std::max(col.rho[s], Real(1.0e-6));
    const Real dthp = (plume.theta[s + 1] - plume.theta[s]) / dz;
    const Real dqvp = (plume.qv[s + 1] - plume.qv[s]) / dz;
    // 卷出质量（单位面积）
    const Real detrain_mass = mb * (Real(1) - std::exp(-plume.detrain[s] * dz));
    dtheta[s] = -mb * dthp / rho +
                detrain_mass * (plume.theta[s] - col.theta[s]) / (rho * dz);
    dqv[s] = -mb * dqvp / rho + detrain_mass * (plume.qv[s] - col.qv[s]) / (rho * dz);
    // 对流动量输送：补偿下沉引起的动量通量辐合  du/dt = -(1/rho) m_b du/dz
    du[s] = -mb * (col.u[s + 1] - col.u[s]) / (rho * dz);
    dv[s] = -mb * (col.v[s + 1] - col.v[s]) / (rho * dz);
  }
  // 云底以下：补偿下沉把云底以上的空气输送到云下层（增温、减湿）
  //   dtheta/dt = +0.2 m_b dtheta/dz / rho,   dqv/dt = +0.2 m_b dqv/dz / rho
  //   （系数 0.2 为云下层权重，见 08_physics.md 参数表）
  {
    const Real z_cb = col.z[static_cast<Size>(cloud_base)];
    const Real dz_cb = std::max(z_cb - col.z[0], Real(1));
    const Real grad_theta = (col.theta[static_cast<Size>(cloud_base)] - col.theta[0]) / dz_cb;
    const Real grad_qv = (col.qv[static_cast<Size>(cloud_base)] - col.qv[0]) / dz_cb;
    for (Int k = 0; k < cloud_base; ++k) {
      const Size s = static_cast<Size>(k);
      const Real rho = std::max(col.rho[s], Real(1.0e-6));
      dtheta[s] = Real(0.2) * mb * grad_theta / rho;
      dqv[s] = Real(0.2) * mb * grad_qv / rho;
    }
  }

  // 水物质净减少率 L (kg/(m^2 s))
  Real water_removed = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    water_removed -= dqv[s] * col.rho[s] * col.dz[s];
  }
  const Real l_rate = std::max(water_removed, Real(0));
  const bool over_land = !sfc.is_water();
  const Real eps_p = clamp(precipitation_efficiency(depth, over_land), Real(0), Real(1));
  const Real precip_flux = eps_p * l_rate;          // kg/(m^2 s)
  const Real reevap = (Real(1) - eps_p) * l_rate;   // kg/(m^2 s)

  // 再蒸发返回云下层（按质量权重分配），并对应降温潜热
  if (reevap > Real(0)) {
    Real mass_sub = Real(0);
    for (Int k = 0; k <= cloud_base && k < n; ++k) {
      mass_sub += col.rho[static_cast<Size>(k)] * col.dz[static_cast<Size>(k)];
    }
    if (mass_sub > Real(0)) {
      for (Int k = 0; k <= cloud_base && k < n; ++k) {
        const Size s = static_cast<Size>(k);
        const Real w = col.rho[s] * col.dz[s] / mass_sub;
        const Real dq = w * reevap / (col.rho[s] * col.dz[s]);
        dqv[s] += dq;
        const Real lv = latent_heat_vaporization(col.temperature(k));
        dtheta[s] -= lv * dq / (kCp * col.exner(k));
      }
    }
  }

  // 累加到倾向
  Real heating_max = Real(0), moist_max = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    tend.theta[s] += dtheta[s];
    tend.qv[s] += dqv[s];
    tend.u[s] += du[s];
    tend.v[s] += dv[s];
    heating_max = std::max(heating_max, std::abs(dtheta[s]));
    moist_max = std::max(moist_max, std::abs(dqv[s]));
  }
  // 水物质收支闭合检查： 列内减少 + 降水 = 0
  {
    Real col_change = Real(0);
    for (Int k = 0; k < n; ++k) {
      const Size s = static_cast<Size>(k);
      col_change += dqv[s] * col.rho[s] * col.dz[s];
    }
    const Real residual = relative_water_residual(precip_flux, -(col_change));
    diag.mass_conservation_residual = std::max(diag.mass_conservation_residual, residual);
    last_.water_residual = residual;
  }

  // 诊断
  last_.triggered = true;
  last_.cape = ascent.cape;
  last_.cin = ascent.cin;
  last_.lfc = ascent.lfc;
  last_.el = ascent.el;
  last_.source_level = source_level;
  last_.cloud_base_level = cloud_base;
  last_.cloud_top_level = ktop;
  last_.cloud_base_height = col.z[static_cast<Size>(cloud_base)] - sfc.terrain_height;
  last_.cloud_top_height = col.z[static_cast<Size>(ktop)] - sfc.terrain_height;
  last_.mass_flux_base = mb;
  last_.updraft_velocity = std::sqrt(std::max(Real(2) * plume.cape, Real(0)));
  last_.precipitation = precip_flux;  // 1 mm/s == 1 kg/(m^2 s)
  last_.downdraft_mass_flux = opt_.kf_downdraft_rate * mb;
  last_.heating_max = heating_max;
  last_.moistening_max = moist_max;
  diag.convective_heating_max = std::max(diag.convective_heating_max, heating_max);
  return last_.precipitation;
}

std::string KainFritschCumulus::describe() const {
  std::ostringstream os;
  os << "KainFritschCumulus{ 卷入率=" << opt_.kf_entrainment_rate << " 1/m 卷出率="
     << opt_.kf_detrainment_rate << " CAPE_min=" << opt_.kf_min_cape << " J/kg 云厚=["
     << opt_.kf_min_cloud_depth << "," << opt_.kf_max_cloud_depth << "] m 闭合时间 tau="
     << opt_.kf_closure_time << " s 下沉比=" << opt_.kf_downdraft_rate
     << " 降水效率(陆/海)=" << opt_.kf_precip_efficiency_land << "/"
     << opt_.kf_precip_efficiency_sea << " }";
  return os.str();
}

// ===========================================================================
// Grell-Devenyi 集合框架 [P12]
// ===========================================================================

Int GrellDevenyiEnsemble::member_count() const noexcept {
  return clamp(opt_.gd_ensemble_size, Int(1), kMaxMembers);
}

// ---------------------------------------------------------------------------
// 成员扰动（[P12] 表 1 的思想：对卷入/卷出/闭合时间/降水效率施加确定性分散）
//   扰动因子按 (i - (N-1)/2) 线性分布，幅度由 gd_spread 控制。
// 复杂度 O(1)。
void GrellDevenyiEnsemble::perturbed_options(Int member, PhysicsOptions& out) const {
  out = opt_;
  const Int nm = std::max(member_count(), Int(1));
  const Real center = static_cast<Real>(nm - 1) * Real(0.5);
  const Real t = (nm > 1) ? (static_cast<Real>(member) - center) / center : Real(0);
  const Real spread = clamp(opt_.gd_spread, Real(0), Real(0.95));
  out.kf_entrainment_rate = opt_.kf_entrainment_rate * (Real(1) + spread * t);
  out.kf_detrainment_rate = opt_.kf_detrainment_rate * (Real(1) - spread * t);
  out.kf_closure_time = opt_.kf_closure_time * (Real(1) + Real(0.5) * spread * t);
  out.kf_precip_efficiency_land =
      clamp(opt_.kf_precip_efficiency_land * (Real(1) + Real(0.5) * spread * t), Real(0.05), Real(1));
  out.kf_precip_efficiency_sea =
      clamp(opt_.kf_precip_efficiency_sea * (Real(1) + Real(0.5) * spread * t), Real(0.05), Real(1));
  out.kf_min_cape = opt_.kf_min_cape * (Real(1) + Real(0.3) * spread * t);
}

// ---------------------------------------------------------------------------
// 集合平均：对每个成员用扰动参数运行 KF，按等权重平均倾向与降水。
// 复杂度 O(n_member * nz)。
Real GrellDevenyiEnsemble::step_column(PhysicsColumn& col, const SurfaceState& sfc, Real dt,
                                       ColumnTendency& tend, PhysicsDiagnostics& diag) {
  last_ = ConvectionDiagnostics{};
  const Int nm = std::max(member_count(), Int(1));
  ColumnTendency acc;
  acc.resize(col.nz());
  Real precip_sum = Real(0);
  Int triggered_members = 0;
  Real cape_sum = Real(0), mb_sum = Real(0);
  Real cbh = Real(0), cth = Real(0);
  Int src = -1, cbl = -1, ctl = -1;
  for (Int m = 0; m < nm; ++m) {
    PhysicsOptions popt;
    perturbed_options(m, popt);
    KainFritschCumulus member(popt);
    ColumnTendency t_m;
    t_m.resize(col.nz());
    const Real p = member.step_column(col, sfc, dt, t_m, diag);
    const ConvectionDiagnostics& d = member.last_diagnostics();
    precip_sum += p;
    if (d.triggered) {
      ++triggered_members;
      cape_sum += d.cape;
      mb_sum += d.mass_flux_base;
      cbh += d.cloud_base_height;
      cth += d.cloud_top_height;
      if (src < 0) {
        src = d.source_level;
        cbl = d.cloud_base_level;
        ctl = d.cloud_top_level;
      }
    }
    acc.add_scaled(Real(1) / static_cast<Real>(nm), t_m);
  }
  tend.add_scaled(Real(1), acc);
  const Real precip_mean = precip_sum / static_cast<Real>(nm);
  if (triggered_members > 0) {
    const Real inv = Real(1) / static_cast<Real>(triggered_members);
    last_.triggered = true;
    last_.cape = cape_sum * inv;
    last_.mass_flux_base = mb_sum * inv;
    last_.cloud_base_height = cbh * inv;
    last_.cloud_top_height = cth * inv;
    last_.source_level = src;
    last_.cloud_base_level = cbl;
    last_.cloud_top_level = ctl;
    last_.precipitation = precip_mean;
    last_.updraft_velocity = std::sqrt(std::max(Real(2) * last_.cape, Real(0)));
  }
  return precip_mean;
}

std::string GrellDevenyiEnsemble::describe() const {
  std::ostringstream os;
  os << "GrellDevenyiEnsemble{ 成员数=" << member_count() << " 扰动幅度=" << opt_.gd_spread
     << " 成员方案=Kain-Fritsch（[P12] 的集合平均框架）}";
  return os.str();
}

// ===========================================================================
// 工厂
// ===========================================================================

std::unique_ptr<CumulusBase> make_cumulus(CumulusScheme scheme, const PhysicsOptions& opt) {
  switch (scheme) {
    case CumulusScheme::None:
      return nullptr;
    case CumulusScheme::KainFritsch:
      return std::make_unique<KainFritschCumulus>(opt);
    case CumulusScheme::GrellDevenyi:
      return std::make_unique<GrellDevenyiEnsemble>(opt);
    case CumulusScheme::Tiedtke:
      throw NotImplemented("Tiedtke 积云方案 [P13] 仅保留接口，尚未实现");
    default:
      throw NotImplemented("未知的积云方案编号");
  }
}

}  // namespace vibe::physics

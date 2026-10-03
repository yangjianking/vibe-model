/// @file physics_driver.cpp
/// @brief 物理过程总驱动与配置映射。
///
/// 文献：[D5][D16]（时间分裂与物理调度）；[P1]-[P18]（各参数化方案）。

#include "vibe/physics/physics_driver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"

namespace vibe::physics {

namespace {

/// 年积日（1..366）与小数日
void parse_start_time(const std::string& iso, int& julian_day, Real& utc_hour) {
  julian_day = 172;
  utc_hour = Real(12);
  if (iso.size() < 13) return;
  int year = 0, month = 0, day = 0, hour = 0, minute = 0;
  if (std::sscanf(iso.c_str(), "%d-%d-%dT%d:%d", &year, &month, &day, &hour, &minute) < 4) return;
  static const int mdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  const bool leap = (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
  int doy = day;
  for (int m = 1; m < month && m <= 12; ++m) {
    doy += mdays[m - 1];
    if (m == 2 && leap) ++doy;
  }
  julian_day = clamp(doy, 1, 366);
  utc_hour = static_cast<Real>(hour) + static_cast<Real>(minute) / Real(60);
}

}  // namespace

// ===========================================================================
// 配置 -> 物理选项
// ===========================================================================

// ---------------------------------------------------------------------------
// 映射规则
//   * 方案字符串 -> 枚举（未知方案抛 ConfigError）
//   * cadence： radiation_cadence 秒 -> 步数 max(1, round(秒/dt))
//   * use_cloud_fraction / co2_ppm / aerosol_optical_depth / soil_layers 直接映射
//   * enable_tendency_physics=false 时 physics 只做诊断，不写倾向
//   * 太阳几何由 time.start_time 解析（年积日 + UTC 小时）
// 复杂度 O(1)。
PhysicsOptions options_from_config(const config::ModelConfig& cfg) {
  PhysicsOptions o;
  const auto& pc = cfg.physics;

  o.microphysics = microphysics_from_string(pc.microphysics);
  o.radiation = radiation_from_string(pc.radiation);
  o.pbl = pbl_from_string(pc.pbl);
  o.surface = surface_from_string(pc.surface);
  o.cumulus = cumulus_from_string(pc.cumulus);

  o.dt_physics = (cfg.time.dt > Real(0)) ? cfg.time.dt : Real(10);
  o.dx = cfg.domain.dx;
  o.dy = cfg.domain.dy;
  o.z_top = cfg.domain.z_top;
  o.latitude = cfg.numerics.fplane_latitude;
  o.longitude = Real(0);

  const Real dt = o.dt_physics;
  const Real rad_seconds = (pc.radiation_cadence > Real(0)) ? pc.radiation_cadence : dt;
  o.radiation_cadence = std::max(Int(1), static_cast<Int>(std::lround(rad_seconds / dt)));
  o.use_cloud_fraction = pc.use_cloud_fraction;
  o.co2_ppm = pc.co2_ppm;
  o.aerosol_optical_depth = pc.aerosol_optical_depth;
  o.n_soil_layers = std::max(pc.soil_layers, Int(1));
  o.enable_tendency_physics = pc.enable_tendency_physics;

  parse_start_time(cfg.time.start_time, o.julian_day, o.utc_hour);
  o.validate();
  return o;
}

// ===========================================================================
// 状态 -> 单列
// ===========================================================================

// ---------------------------------------------------------------------------
//   z_k   = 网格层中心高度
//   pi    = pi0(z) + pi'（状态中的 Exner 扰动）
//   p     = p0 * pi^(1/kappa)
//   rho   = 状态中的干空气密度（缺省时由状态方程给出）
//   u,v   = 由面心场平均到单元中心（0.5*(u_i + u_{i-1})）
// 复杂度 O(nz)。
void state_to_column(const dyn::State& s, const dyn::ReferenceState& ref, const grid::Grid& g,
                     Int i, Int j, PhysicsColumn& col) {
  const Int nz = g.nz();
  col.resize(nz);
  const bool has_ref = ref.pi0().nx() > 0;
  for (Int k = 0; k < nz; ++k) {
    const Size sk = static_cast<Size>(k);
    col.z[sk] = g.z_center(i, j, k);
    const Real pi0 = has_ref ? ref.pi0().at(i, j, k) : exner_from_pressure(kP0);
    const Real pi = pi0 + s.pi().at(i, j, k);
    col.p[sk] = kP0 * std::pow(std::max(pi, Real(1.0e-3)), Real(1) / kKappa);
    col.theta[sk] = s.theta().at(i, j, k);
    col.qv[sk] = std::max(s.field(dyn::Species::Qv).at(i, j, k), Real(0));
    col.qc[sk] = std::max(s.field(dyn::Species::Qc).at(i, j, k), Real(0));
    col.qr[sk] = std::max(s.field(dyn::Species::Qr).at(i, j, k), Real(0));
    col.qi[sk] = std::max(s.field(dyn::Species::Qi).at(i, j, k), Real(0));
    col.qs[sk] = std::max(s.field(dyn::Species::Qs).at(i, j, k), Real(0));
    col.qg[sk] = std::max(s.field(dyn::Species::Qg).at(i, j, k), Real(0));
    const Real tv = virtual_temperature(col.temperature(k), col.qv[sk], col.qc[sk], col.qi[sk],
                                        col.qs[sk], col.qg[sk]);
    const Real rho_state = s.rho().at(i, j, k);
    col.rho[sk] = (rho_state > Real(0)) ? rho_state : density_from_pressure_tv(col.p[sk], tv);
    // 面心风 -> 单元中心
    const Real uw = s.u().at(i, j, k);
    const Real ue = s.u().at(i - 1, j, k);
    const Real vs = s.v().at(i, j, k);
    const Real vn = s.v().at(i, j - 1, k);
    col.u[sk] = Real(0.5) * (uw + ue);
    col.v[sk] = Real(0.5) * (vs + vn);
    col.tke[sk] = Real(0);
  }
  col.diagnose_layer_depth();
}

// ===========================================================================
// 构造
// ===========================================================================

PhysicsDriver::PhysicsDriver(const PhysicsOptions& opt, const grid::Grid& g)
    : opt_(opt), grid_(&g) {
  opt_.validate();
  nx_ = g.nx();
  ny_ = g.ny();
  nz_ = g.nz();
  microphysics_ = make_microphysics(opt_.microphysics, opt_);
  radiation_ = make_radiation(opt_.radiation, opt_);
  if (radiation_) radiation_->initialize(&g);
  pbl_ = make_pbl(opt_.pbl, opt_);
  surface_layer_ = make_surface_layer(opt_);
  land_ = make_land_surface(opt_);
  sea_ = make_sea_surface(opt_);
  cumulus_ = make_cumulus(opt_.cumulus, opt_);

  const Size nyx = static_cast<Size>(nx_) * static_cast<Size>(ny_);
  const Size nvol = nyx * static_cast<Size>(nz_);
  surface_.resize(nyx);
  tke_.assign(nvol, Real(0));
  rad_theta_tend_.assign(nvol, Real(0));
  diag_.resize(nx_, ny_);
  diag_.pbl_cadence = opt_.pbl_cadence;
  lwp_work_.assign(nyx, Real(0));
  iwp_work_.assign(nyx, Real(0));

  // 地表状态初始化：地形、土壤廓线
  for (Int j = 0; j < ny_; ++j) {
    for (Int i = 0; i < nx_; ++i) {
      SurfaceState& sfc = surface_[static_cast<Size>(j) * static_cast<Size>(nx_) + static_cast<Size>(i)];
      sfc.terrain_height = g.terrain(i, j);
      sfc.surface_pressure = kP0;
      sfc.land_fraction = Real(1);  // 无陆海掩膜输入时默认全陆（可用 surface_states() 修改）
      sfc.roughness = kRoughnessLand;
      sfc.albedo = kAlbedoLand;
      sfc.emissivity = Real(0.98);
      land_->initialize(sfc);
    }
  }
}

void PhysicsDriver::set_options(const PhysicsOptions& opt) {
  opt.validate();
  const PhysicsOptions keep = opt_;
  opt_ = opt;
  // 保持分辨率的本地值（配置层可能给全局值）
  opt_.nx = keep.nx;
  opt_.ny = keep.ny;
  opt_.nz = keep.nz;
}

void PhysicsDriver::ensure_allocated(const grid::Grid& g) {
  if (nx_ == g.nx() && ny_ == g.ny() && nz_ == g.nz()) return;
  nx_ = g.nx();
  ny_ = g.ny();
  nz_ = g.nz();
  const Size nyx = static_cast<Size>(nx_) * static_cast<Size>(ny_);
  surface_.resize(nyx);
  tke_.assign(nyx * static_cast<Size>(nz_), Real(0));
  rad_theta_tend_.assign(nyx * static_cast<Size>(nz_), Real(0));
  diag_.resize(nx_, ny_);
  diag_.pbl_cadence = opt_.pbl_cadence;
  lwp_work_.assign(nyx, Real(0));
  iwp_work_.assign(nyx, Real(0));
  for (Int j = 0; j < ny_; ++j) {
    for (Int i = 0; i < nx_; ++i) {
      SurfaceState& sfc = surface_[static_cast<Size>(j) * static_cast<Size>(nx_) + static_cast<Size>(i)];
      sfc.terrain_height = g.terrain(i, j);
      sfc.surface_pressure = kP0;
      sfc.land_fraction = Real(1);
      sfc.roughness = kRoughnessLand;
      sfc.albedo = kAlbedoLand;
      sfc.emissivity = Real(0.98);
      land_->initialize(sfc);
    }
  }
}

bool PhysicsDriver::call_radiation_step() const noexcept {
  if (!radiation_) return false;
  const Int cad = std::max(opt_.radiation_cadence, Int(1));
  return (static_cast<Int>(step_index_) % cad) == 0;
}

// ===========================================================================
// 单列物理
// ===========================================================================

void PhysicsDriver::column_physics(Int i, Int j, dyn::State& s, const grid::Grid& g,
                                   const dyn::ReferenceState& ref, Real dt, Real cos_zenith,
                                   Real sun_earth_factor, bool call_radiation, ColumnTendency& td,
                                   SurfaceFluxes& flx_out) {
  const Size idx = static_cast<Size>(j) * static_cast<Size>(nx_) + static_cast<Size>(i);
  PhysicsColumn col;
  state_to_column(s, ref, g, i, j, col);
  // 湍流动能从持久存储载入
  for (Int k = 0; k < nz_; ++k) {
    col.tke[static_cast<Size>(k)] = std::max(tke_[idx * static_cast<Size>(nz_) + static_cast<Size>(k)], Real(0));
  }
  td.resize(nz_);
  td.zero();
  SurfaceState& sfc = surface_[idx];
  sfc.terrain_height = g.terrain(i, j);
  sfc.surface_pressure = col.p[0];

  // ---- 1. 辐射 ----
  if (radiation_) {
    RadiationFluxes rflux;
    if (call_radiation) {
      ColumnTendency rad_td;
      rad_td.resize(nz_);
      rad_td.zero();
      radiation_->step_column(col, sfc, dt, cos_zenith, sun_earth_factor, rad_td, diag_, rflux);
      const Real relax = clamp(opt_.radiation_heating_relax, Real(0), Real(1));
      for (Int k = 0; k < nz_; ++k) {
        const Size sk = static_cast<Size>(k);
        const Size gidx = idx * static_cast<Size>(nz_) + sk;
        rad_theta_tend_[gidx] = relax * rad_td.theta[sk] + (Real(1) - relax) * rad_theta_tend_[gidx];
      }
      diag_.toa_outgoing_longwave[idx] = rflux.toa_longwave_up;
      diag_.toa_net_shortwave[idx] = rflux.toa_shortwave_down - rflux.toa_shortwave_up;
      diag_.surface_net_radiation[idx] = rflux.surface_net;
      diag_.radiation_calls++;
    } else {
      // 未调用辐射：沿用上一次加热率（加热率的时间平流），通量诊断保持地表状态值
      diag_.surface_net_radiation[idx] = sfc.net_radiation;
    }
    for (Int k = 0; k < nz_; ++k) {
      const Size sk = static_cast<Size>(k);
      td.theta[sk] += rad_theta_tend_[idx * static_cast<Size>(nz_) + sk];
    }
  }

  // ---- 2. 地面层 + 陆面/海面 ----
  SurfaceFluxes flx = surface_layer_->solve(col, sfc, dt);
  land_->step_column(col, sfc, flx, dt, td, diag_);
  flx_out = flx;
  diag_.surface_sensible_flux[idx] = sfc.sensible_heat_flux;
  diag_.surface_latent_flux[idx] = sfc.latent_heat_flux;

  // ---- 3. 边界层 ----
  if (pbl_) {
    const Real h = pbl_->step_column(col, sfc, flx, dt, td);
    diag_.pbl_height[idx] = h;
    diag_.pbl_calls++;
  } else {
    diag_.pbl_height[idx] = Real(0);
  }

  // ---- 4. 微物理 ----
  if (microphysics_) {
    std::vector<Real> th0 = col.theta, qv0 = col.qv, qc0 = col.qc, qr0 = col.qr, qi0 = col.qi,
                      qs0 = col.qs, qg0 = col.qg;
    const Real precip = microphysics_->step_column(col, dt, diag_);
    for (Int k = 0; k < nz_; ++k) {
      const Size sk = static_cast<Size>(k);
      td.theta[sk] += (col.theta[sk] - th0[sk]) / dt;
      td.qv[sk] += (col.qv[sk] - qv0[sk]) / dt;
      td.qc[sk] += (col.qc[sk] - qc0[sk]) / dt;
      td.qr[sk] += (col.qr[sk] - qr0[sk]) / dt;
      td.qi[sk] += (col.qi[sk] - qi0[sk]) / dt;
      td.qs[sk] += (col.qs[sk] - qs0[sk]) / dt;
      td.qg[sk] += (col.qg[sk] - qg0[sk]) / dt;
    }
    diag_.precipitation_grid[idx] += precip;
    diag_.microphysics_calls++;
    const PhysicsCfl c = compute_cfl(col, dt, opt_.dx, microphysics_->max_fall_speed(col), Real(0));
    if (c.total > cfl_last_.total) cfl_last_ = c;
  }

  // ---- 5. 积云 ----
  if (cumulus_) {
    const Real precip = cumulus_->step_column(col, sfc, dt, td, diag_);
    diag_.precipitation_convective[idx] += precip;
    const ConvectionDiagnostics& cd = cumulus_->last_diagnostics();
    diag_.cape[idx] = cd.cape;
    diag_.cin[idx] = cd.cin;
    diag_.cloud_base[idx] = cd.cloud_base_height;
    diag_.cloud_top[idx] = cd.cloud_top_height;
    diag_.cumulus_calls++;
  }

  // ---- 6. 次网格动量通量 ----
  subgrid_momentum_tendency(s, g, i, j, col, td);

  // ---- 诊断 ----
  Real cf_sum = Real(0);
  for (Int k = 0; k < nz_; ++k) {
    cf_sum += col.cloud_fraction(k);
    tke_[idx * static_cast<Size>(nz_) + static_cast<Size>(k)] =
        std::max(col.tke[static_cast<Size>(k)], Real(0));
  }
  diag_.cloud_fraction[idx] = (nz_ > 0) ? cf_sum / static_cast<Real>(nz_) : Real(0);
  lwp_work_[idx] = col.liquid_water_path();
  iwp_work_[idx] = col.ice_water_path();
  if (pbl_) {
    const PhysicsCfl c = compute_cfl(col, dt, opt_.dx, Real(0), pbl_->max_diffusivity());
    if (c.total > cfl_last_.total) cfl_last_ = c;
  }
}

void PhysicsDriver::subgrid_momentum_tendency(const dyn::State& s, const grid::Grid& g, Int i, Int j,
                                              const PhysicsColumn& col, ColumnTendency& td) const {
  // Smagorinsky 型水平次网格动量通量
  //   S_ij 由单元中心风场中心差分得到；K_m = (c_s Delta)^2 sqrt(2 S_ij S_ij)
  //   du/dt = d/dx(K du/dx) + d/dy(K du/dy) + K_const * Lap(u)
  // 文献：[P16] Smagorinsky (1963)。
  // 复杂度 O(nz)。
  if (!opt_.use_smagorinsky && !(opt_.horizontal_diffusion_coef > Real(0))) return;
  const Real dx = (g.geom().dx > Real(0)) ? g.geom().dx : Real(1000);
  const Real dy = (g.geom().dy > Real(0)) ? g.geom().dy : Real(1000);
  const Real delta = std::sqrt(dx * dy);
  const Real cs_delta2 = sqr(opt_.smagorinsky_coef * delta);
  auto ucell = [&](Int ii, Int jj, Int kk) {
    return Real(0.5) * (s.u().clamp_at(ii, jj, kk) + s.u().clamp_at(ii - 1, jj, kk));
  };
  auto vcell = [&](Int ii, Int jj, Int kk) {
    return Real(0.5) * (s.v().clamp_at(ii, jj, kk) + s.v().clamp_at(ii, jj - 1, kk));
  };
  const Int n = col.nz();
  for (Int k = 0; k < n; ++k) {
    const Real dudx = (ucell(i + 1, j, k) - ucell(i - 1, j, k)) / (Real(2) * dx);
    const Real dudy = (ucell(i, j + 1, k) - ucell(i, j - 1, k)) / (Real(2) * dy);
    const Real dvdx = (vcell(i + 1, j, k) - vcell(i - 1, j, k)) / (Real(2) * dx);
    const Real dvdy = (vcell(i, j + 1, k) - vcell(i, j - 1, k)) / (Real(2) * dy);
    const Real s2 = Real(2) * (dudx * dudx + dvdy * dvdy) + sqr(dudy + dvdx);
    const Real km = cs_delta2 * std::sqrt(std::max(s2, Real(0))) + opt_.horizontal_diffusion_coef;
    if (!(km > Real(0))) continue;
    const Real lap_u = (ucell(i + 1, j, k) + ucell(i - 1, j, k) - Real(2) * ucell(i, j, k)) / (dx * dx) +
                       (ucell(i, j + 1, k) + ucell(i, j - 1, k) - Real(2) * ucell(i, j, k)) / (dy * dy);
    const Real lap_v = (vcell(i + 1, j, k) + vcell(i - 1, j, k) - Real(2) * vcell(i, j, k)) / (dx * dx) +
                       (vcell(i, j + 1, k) + vcell(i, j - 1, k) - Real(2) * vcell(i, j, k)) / (dy * dy);
    td.u[static_cast<Size>(k)] += km * lap_u;
    td.v[static_cast<Size>(k)] += km * lap_v;
  }
}

void PhysicsDriver::scatter(Int i, Int j, const ColumnTendency& td, dyn::PhysicsTendency& out) const {
  for (Int k = 0; k < nz_; ++k) {
    const Size sk = static_cast<Size>(k);
    out.theta.at(i, j, k) += td.theta[sk];
    out.qv.at(i, j, k) += td.qv[sk];
    out.qc.at(i, j, k) += td.qc[sk];
    out.qr.at(i, j, k) += td.qr[sk];
    out.qi.at(i, j, k) += td.qi[sk];
    out.qs.at(i, j, k) += td.qs[sk];
    out.qg.at(i, j, k) += td.qg[sk];
    out.u.at(i, j, k) += td.u[sk];
    out.v.at(i, j, k) += td.v[sk];
    out.tke.at(i, j, k) += td.tke[sk];
  }
}

// ===========================================================================
// 主循环
// ===========================================================================

void PhysicsDriver::step(dyn::State& s, const grid::Grid& g, const dyn::ReferenceState& ref,
                         dyn::PhysicsTendency& out, Real dt) {
  ensure_allocated(g);
  if (!(dt > Real(0)) || nx_ <= 0 || ny_ <= 0 || nz_ <= 0) return;

  const bool call_rad = call_radiation_step();
  const Real cosz = RadiationDriver::solar_zenith_cosine(opt_.latitude, opt_.longitude,
                                                         static_cast<Real>(opt_.julian_day), opt_.utc_hour);
  const Real esd = RadiationDriver::earth_sun_distance_factor(static_cast<Real>(opt_.julian_day));

  // 每步重置域平均与二维诊断；调用计数跨步累计
  const Int n_rad = diag_.radiation_calls, n_mp = diag_.microphysics_calls,
            n_pbl = diag_.pbl_calls, n_cu = diag_.cumulus_calls;
  diag_.reset();
  diag_.radiation_calls = n_rad;
  diag_.microphysics_calls = n_mp;
  diag_.pbl_calls = n_pbl;
  diag_.cumulus_calls = n_cu;
  cfl_last_ = PhysicsCfl{};

  ColumnTendency td;
  SurfaceFluxes flx;
  Real heat_sum = Real(0), moist_sum = Real(0), mom_sum = Real(0);
  Real lwp_sum = Real(0), iwp_sum = Real(0), tke_sum = Real(0), tke_max = Real(0);
  const Size nyx = static_cast<Size>(nx_) * static_cast<Size>(ny_);

  for (Int j = 0; j < ny_; ++j) {
    for (Int i = 0; i < nx_; ++i) {
      column_physics(i, j, s, g, ref, dt, cosz, esd, call_rad, td, flx);
      const Size idx = static_cast<Size>(j) * static_cast<Size>(nx_) + static_cast<Size>(i);
      if (opt_.enable_tendency_physics) scatter(i, j, td, out);
      heat_sum += flx.sensible;
      moist_sum += flx.latent;
      mom_sum += flx.momentum;
      lwp_sum += lwp_work_[idx];
      iwp_sum += iwp_work_[idx];
      for (Int k = 0; k < nz_; ++k) {
        const Real e = tke_[idx * static_cast<Size>(nz_) + static_cast<Size>(k)];
        tke_sum += e;
        tke_max = std::max(tke_max, e);
      }
    }
  }

  const Real inv = Real(1) / static_cast<Real>(nyx);
  diag_.surface_flux_heat_mean = heat_sum * inv;
  diag_.surface_flux_moist_mean = moist_sum * inv;
  diag_.surface_flux_momentum_mean = mom_sum * inv;
  diag_.liquid_water_path_mean = lwp_sum * inv;
  diag_.ice_water_path_mean = iwp_sum * inv;
  diag_.tke_mean = tke_sum * inv / static_cast<Real>(std::max(nz_, Int(1)));
  diag_.tke_max = tke_max;
  // 域平均/最大
  diag_.aggregate();

  // 标量汇总写入 PhysicsTendency
  out.surface_flux_heat += diag_.surface_flux_heat_mean;
  out.surface_flux_moist += diag_.surface_flux_moist_mean;
  out.surface_flux_momentum += diag_.surface_flux_momentum_mean;
  out.precipitation_rate += diag_.precipitation_total_mean;
  out.top_of_atmosphere_flux += diag_.toa_net_flux_mean;

  ++step_index_;
}

void PhysicsDriver::step_column(Int i, Int j, dyn::State& s, const grid::Grid& g,
                                const dyn::ReferenceState& ref, dyn::PhysicsTendency& out,
                                Real dt) {
  ensure_allocated(g);
  if (!(dt > Real(0))) return;
  if (i < 0 || j < 0 || i >= nx_ || j >= ny_) throw DimensionError("PhysicsDriver::step_column 索引越界");
  const Real cosz = RadiationDriver::solar_zenith_cosine(opt_.latitude, opt_.longitude,
                                                         static_cast<Real>(opt_.julian_day), opt_.utc_hour);
  const Real esd = RadiationDriver::earth_sun_distance_factor(static_cast<Real>(opt_.julian_day));
  ColumnTendency td;
  SurfaceFluxes flx;
  column_physics(i, j, s, g, ref, dt, cosz, esd, call_radiation_step(), td, flx);
  if (opt_.enable_tendency_physics) scatter(i, j, td, out);
  diag_.aggregate();
}

// ===========================================================================
// 诊断
// ===========================================================================

// ---------------------------------------------------------------------------
// 采样若干列计算物理 CFL（水平平流、粒子落速、湍流扩散）
//   水平: max(|u|,|v|) dt/dx        落速: Vt dt/min(dz)
//   扩散: 2 K dt/min(dz)^2
// 复杂度 O(n_sample * nz)。
PhysicsCfl PhysicsDriver::cfl_physics(const dyn::State& s, const dyn::ReferenceState& ref,
                                      Real dt) const {
  PhysicsCfl worst;
  if (grid_ == nullptr || nx_ <= 0 || ny_ <= 0 || nz_ <= 0) return worst;
  PhysicsColumn col;
  const Int stride_i = std::max(Int(1), nx_ / 8);
  const Int stride_j = std::max(Int(1), ny_ / 8);
  for (Int j = 0; j < ny_; j += stride_j) {
    for (Int i = 0; i < nx_; i += stride_i) {
      state_to_column(s, ref, *grid_, i, j, col);
      const Real vfall = microphysics_ ? microphysics_->max_fall_speed(col) : Real(0);
      const Real kdif = pbl_ ? pbl_->max_diffusivity() : Real(0);
      const PhysicsCfl c = compute_cfl(col, dt, opt_.dx, vfall, kdif);
      if (c.total > worst.total) worst = c;
    }
  }
  return worst;
}

std::string PhysicsDriver::stability_report() const {
  std::ostringstream os;
  os.setf(std::ios::scientific);
  os.precision(3);
  os << "物理稳定诊断: CFL(水平)=" << cfl_last_.horizontal << " CFL(落速)=" << cfl_last_.fall_speed
     << " CFL(扩散)=" << cfl_last_.diffusion << " CFL(总)=" << cfl_last_.total
     << " 最小层厚=" << cfl_last_.min_dz << " m"
     << " 限值(水平/垂直)=" << opt_.cfl_limit << "/" << opt_.vertical_cfl_limit
     << (cfl_last_.total > opt_.vertical_cfl_limit ? "  [警告] 垂直 CFL 超限，建议减小 dt 或改用隐式沉降"
                                                   : "  [正常]");
  return os.str();
}

std::string PhysicsDriver::describe() const {
  std::ostringstream os;
  os << "PhysicsDriver{ 网格=" << nx_ << "x" << ny_ << "x" << nz_ << "\n"
     << "  " << (microphysics_ ? microphysics_->describe() : std::string("微物理: 关闭")) << "\n"
     << "  " << (radiation_ ? radiation_->describe() : std::string("辐射: 关闭")) << "\n"
     << "  " << (pbl_ ? pbl_->describe() : std::string("边界层: 关闭")) << "\n"
     << "  " << (surface_layer_ ? surface_layer_->describe() : std::string("地面层: 关闭")) << "\n"
     << "  " << (land_ ? land_->describe() : std::string("陆面: 关闭")) << "\n"
     << "  " << (sea_ ? sea_->describe() : std::string("海面: 关闭")) << "\n"
     << "  " << (cumulus_ ? cumulus_->describe() : std::string("积云: 关闭")) << "\n"
     << "  辐射间隔=" << opt_.radiation_cadence << " 步 (" << opt_.radiation_heating_relax
     << " 权重平滑) 倾向输出=" << (opt_.enable_tendency_physics ? "开" : "关") << " }";
  return os.str();
}

std::unique_ptr<PhysicsDriver> make_physics_driver(const grid::Grid& g,
                                                   const config::ModelConfig& cfg) {
  PhysicsOptions opt = options_from_config(cfg);
  opt.nx = g.nx();
  opt.ny = g.ny();
  opt.nz = g.nz();
  opt.dx = (g.geom().dx > Real(0)) ? g.geom().dx : opt.dx;
  opt.dy = (g.geom().dy > Real(0)) ? g.geom().dy : opt.dy;
  return std::make_unique<PhysicsDriver>(opt, g);
}

}  // namespace vibe::physics

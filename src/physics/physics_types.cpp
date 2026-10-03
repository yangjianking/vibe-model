/// @file physics_types.cpp
/// @brief physics_types.hpp 的实现：热力学工具、单列视图、选项校验与气块抬升。
///
/// 所有数值例程均给出公式、离散化、文献与复杂度（见各函数注释）。
/// 文献索引见 docs/design/references.md。

#include "vibe/physics/physics_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>

#include "vibe/common/error.hpp"

namespace vibe::physics {

namespace {
constexpr Real kDzEps = Real(1.0e-6);   ///< 层厚下限，避免除零
constexpr Real kQvFloor = Real(1.0e-12);
}  // namespace

// ===========================================================================
// 1. 方案名与字符串转换
// ===========================================================================

const char* to_string(MicrophysicsScheme s) noexcept {
  switch (s) {
    case MicrophysicsScheme::None: return "none";
    case MicrophysicsScheme::Kessler: return "kessler";
    case MicrophysicsScheme::Thompson: return "thompson";
    case MicrophysicsScheme::Morrison: return "morrison";
    case MicrophysicsScheme::Wsm6: return "wsm6";
    default: return "unknown";
  }
}
const char* to_string(RadiationScheme s) noexcept {
  switch (s) {
    case RadiationScheme::None: return "none";
    case RadiationScheme::Rrtmg: return "rrtmg";
    case RadiationScheme::RrtmgLongwaveOnly: return "rrtmg_lw";
    case RadiationScheme::RrtmgShortwaveOnly: return "rrtmg_sw";
    case RadiationScheme::SimpleGrey: return "simple_grey";
    default: return "unknown";
  }
}
const char* to_string(PblScheme s) noexcept {
  switch (s) {
    case PblScheme::None: return "none";
    case PblScheme::Ysu: return "ysu";
    case PblScheme::Myj: return "myj";
    case PblScheme::Mynn: return "mynn";
    case PblScheme::Smagorinsky: return "smagorinsky";
    default: return "unknown";
  }
}
const char* to_string(SurfaceScheme s) noexcept {
  switch (s) {
    case SurfaceScheme::None: return "none";
    case SurfaceScheme::MoninObukhov: return "monin_obukhov";
    case SurfaceScheme::Noah: return "noah";
    case SurfaceScheme::Slab: return "slab";
    default: return "unknown";
  }
}
const char* to_string(CumulusScheme s) noexcept {
  switch (s) {
    case CumulusScheme::None: return "none";
    case CumulusScheme::KainFritsch: return "kain_fritsch";
    case CumulusScheme::GrellDevenyi: return "grell_devenyi";
    case CumulusScheme::Tiedtke: return "tiedtke";
    default: return "unknown";
  }
}

namespace {
std::string lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}
}  // namespace

MicrophysicsScheme microphysics_from_string(const std::string& name) {
  const std::string s = lower(name);
  if (s == "none" || s == "off") return MicrophysicsScheme::None;
  if (s == "kessler") return MicrophysicsScheme::Kessler;
  if (s == "thompson") return MicrophysicsScheme::Thompson;
  if (s == "morrison") return MicrophysicsScheme::Morrison;
  if (s == "wsm6" || s == "wsmsix") return MicrophysicsScheme::Wsm6;
  throw ConfigError("未知的微物理方案: " + name);
}
RadiationScheme radiation_from_string(const std::string& name) {
  const std::string s = lower(name);
  if (s == "none" || s == "off") return RadiationScheme::None;
  if (s == "rrtmg" || s == "rrtm" || s == "rrtmg_simple" || s == "rrtm_simple") {
    return RadiationScheme::Rrtmg;
  }
  if (s == "rrtmg_lw" || s == "rrtm_lw") return RadiationScheme::RrtmgLongwaveOnly;
  if (s == "rrtmg_sw" || s == "rrtm_sw") return RadiationScheme::RrtmgShortwaveOnly;
  if (s == "simple_grey" || s == "grey") return RadiationScheme::SimpleGrey;
  throw ConfigError("未知的辐射方案: " + name);
}
PblScheme pbl_from_string(const std::string& name) {
  const std::string s = lower(name);
  if (s == "none" || s == "off") return PblScheme::None;
  if (s == "ysu") return PblScheme::Ysu;
  if (s == "myj" || s == "my") return PblScheme::Myj;
  if (s == "mynn") return PblScheme::Mynn;
  if (s == "smagorinsky") return PblScheme::Smagorinsky;
  throw ConfigError("未知的边界层方案: " + name);
}
SurfaceScheme surface_from_string(const std::string& name) {
  const std::string s = lower(name);
  if (s == "none" || s == "off") return SurfaceScheme::None;
  if (s == "monin_obukhov" || s == "mo") return SurfaceScheme::MoninObukhov;
  if (s == "noah") return SurfaceScheme::Noah;
  if (s == "slab") return SurfaceScheme::Slab;
  throw ConfigError("未知的陆面方案: " + name);
}
CumulusScheme cumulus_from_string(const std::string& name) {
  const std::string s = lower(name);
  if (s == "none" || s == "off") return CumulusScheme::None;
  if (s == "kain_fritsch" || s == "kf") return CumulusScheme::KainFritsch;
  if (s == "grell_devenyi" || s == "gd") return CumulusScheme::GrellDevenyi;
  if (s == "tiedtke") return CumulusScheme::Tiedtke;
  throw ConfigError("未知的积云方案: " + name);
}

// ===========================================================================
// 2. 热力学工具
// ===========================================================================

// ---------------------------------------------------------------------------
// 饱和水汽压（液面），Bolton (1980)
//   e_s(T) = 611.2 exp(17.67 T_c / (T_c + 243.5)),  T_c = T - 273.15
// 离散化：解析式，逐点计算；低温端用 T_c 下限裁剪避免指数溢出。
// 文献：[B4]
// 复杂度：O(1)（一次 exp）。
Real saturation_vapor_pressure(Real t_k) noexcept {
  const Real tc = std::max(t_k - kT0, Real(-80.0));
  return Real(611.2) * std::exp(Real(17.67) * tc / (tc + Real(243.5)));
}

// ---------------------------------------------------------------------------
// 饱和水汽压（冰面），Buck (1981) 拟合
//   e_i(T) = 611.15 exp(22.452 T_c / (T_c + 272.55))
// 文献：[B4][D16]
// 复杂度：O(1)。
Real saturation_vapor_pressure_ice(Real t_k) noexcept {
  const Real tc = std::max(t_k - kT0, Real(-80.0));
  return Real(611.15) * std::exp(Real(22.452) * tc / (tc + Real(272.55)));
}

// ---------------------------------------------------------------------------
// 饱和混合比
//   q_s = eps e_s / (p - e_s)
// 把 q 视为混合比（kg/kg）；p <= e_s 时返回一个上限值以保证正定。
// 复杂度：O(1)。
Real saturation_mixing_ratio(Real p, Real t_k, bool ice_phase) noexcept {
  const Real es = saturation_vapor_pressure_phase(t_k, ice_phase);
  const Real den = std::max(p - es, Real(1.0));
  return kEpsilonVap * es / den;
}

Real vapor_pressure_from_mixing_ratio(Real p, Real qv) noexcept {
  const Real q = std::max(qv, Real(0));
  return q * p / (kEpsilonVap + q);
}

Real relative_humidity_from_qv(Real p, Real t_k, Real qv) noexcept {
  const Real qs = std::max(saturation_mixing_ratio(p, t_k), kQvFloor);
  return qv / qs;
}

// ---------------------------------------------------------------------------
// 露点（Bolton 反函数）
//   ln(e/611.2) = 17.67 T_c/(T_c+243.5)  =>  T_d = 243.5 ln(e/611.2)/(17.67 - ln(e/611.2)) + T0
// 复杂度：O(1)（一次 log）。
Real dewpoint_temperature(Real p, Real qv) noexcept {
  const Real e = std::max(vapor_pressure_from_mixing_ratio(p, qv), Real(1.0e-3));
  const Real l = std::log(e / Real(611.2));
  return Real(243.5) * l / (Real(17.67) - l) + kT0;
}

// ===========================================================================
// 3. PhysicsColumn
// ===========================================================================

void PhysicsColumn::resize(Int nz) {
  const Size n = static_cast<Size>(std::max(nz, Int(0)));
  z.assign(n, Real(0)); p.assign(n, Real(0)); rho.assign(n, Real(0));
  theta.assign(n, Real(0)); qv.assign(n, Real(0)); qc.assign(n, Real(0));
  qr.assign(n, Real(0)); qi.assign(n, Real(0)); qs.assign(n, Real(0));
  qg.assign(n, Real(0)); u.assign(n, Real(0)); v.assign(n, Real(0));
  tke.assign(n, Real(0)); dz.assign(n, Real(0));
}

void PhysicsColumn::zero() {
  std::fill(z.begin(), z.end(), Real(0));
  std::fill(p.begin(), p.end(), Real(0));
  std::fill(rho.begin(), rho.end(), Real(0));
  std::fill(theta.begin(), theta.end(), Real(0));
  std::fill(qv.begin(), qv.end(), Real(0));
  std::fill(qc.begin(), qc.end(), Real(0));
  std::fill(qr.begin(), qr.end(), Real(0));
  std::fill(qi.begin(), qi.end(), Real(0));
  std::fill(qs.begin(), qs.end(), Real(0));
  std::fill(qg.begin(), qg.end(), Real(0));
  std::fill(u.begin(), u.end(), Real(0));
  std::fill(v.begin(), v.end(), Real(0));
  std::fill(tke.begin(), tke.end(), Real(0));
  std::fill(dz.begin(), dz.end(), Real(0));
}

bool PhysicsColumn::consistent() const noexcept {
  const Size n = z.size();
  return p.size() == n && rho.size() == n && theta.size() == n && qv.size() == n &&
         qc.size() == n && qr.size() == n && qi.size() == n && qs.size() == n &&
         qg.size() == n && u.size() == n && v.size() == n && tke.size() == n &&
         dz.size() == n;
}

Real PhysicsColumn::saturation_mixing_ratio(Int k, bool ice_phase) const noexcept {
  const Size s = static_cast<Size>(k);
  return physics::saturation_mixing_ratio(p[s], temperature(k), ice_phase);
}

Real PhysicsColumn::virtual_temperature(Int k) const noexcept {
  const Size s = static_cast<Size>(k);
  return physics::virtual_temperature(theta[s] * exner(k), qv[s], qc[s], qi[s], qs[s], qg[s]);
}

Real PhysicsColumn::theta_v(Int k) const noexcept {
  const Size s = static_cast<Size>(k);
  return theta[s] * (Real(1) + Real(0.608) * qv[s] - qc[s] - qi[s] - qs[s] - qg[s]);
}

void PhysicsColumn::diagnose_density() {
  const Int n = nz();
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    rho[s] = density_from_pressure_tv(p[s], virtual_temperature(k));
  }
}

// ---------------------------------------------------------------------------
// 层厚：界面高度用相邻层中心的中点，顶/底界面外推
//   zf[0]   = z[0] - 0.5 (z[1]-z[0])
//   zf[k]   = 0.5 (z[k-1] + z[k])
//   zf[nz]  = z[nz-1] + 0.5 (z[nz-1]-z[nz-2])
//   dz[k]   = zf[k+1] - zf[k]
// 复杂度 O(nz)。
void PhysicsColumn::diagnose_layer_depth() {
  const Int n = nz();
  if (n <= 0) return;
  std::vector<Real> zf;
  interface_heights(zf);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    dz[s] = std::max(zf[static_cast<Size>(k + 1)] - zf[static_cast<Size>(k)], kDzEps);
  }
}

void PhysicsColumn::interface_heights(std::vector<Real>& zf) const {
  const Int n = nz();
  zf.assign(static_cast<Size>(std::max(n, Int(0))) + 1, Real(0));
  if (n == 0) return;
  if (n == 1) {
    const Real d = (dz.empty() || dz[0] <= Real(0)) ? Real(100) : dz[0];
    zf[0] = z[0] - Real(0.5) * d;
    zf[1] = z[0] + Real(0.5) * d;
    return;
  }
  zf[0] = z[0] - Real(0.5) * (z[1] - z[0]);
  for (Int k = 1; k < n; ++k) {
    zf[static_cast<Size>(k)] = Real(0.5) * (z[static_cast<Size>(k - 1)] + z[static_cast<Size>(k)]);
  }
  zf[static_cast<Size>(n)] = z[static_cast<Size>(n - 1)] + Real(0.5) * (z[static_cast<Size>(n - 1)] - z[static_cast<Size>(n - 2)]);
}

Real PhysicsColumn::total_water(Int k) const noexcept {
  const Size s = static_cast<Size>(k);
  return qv[s] + qc[s] + qr[s] + qi[s] + qs[s] + qg[s];
}

Real PhysicsColumn::total_water_mass() const noexcept {
  const Int n = nz();
  Real sum = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    sum += total_water(k) * rho[s] * dz[s];
  }
  return sum;
}

Real PhysicsColumn::vapor_path() const noexcept {
  const Int n = nz();
  Real sum = Real(0);
  for (Int k = 0; k < n; ++k) sum += qv[static_cast<Size>(k)] * rho[static_cast<Size>(k)] * dz[static_cast<Size>(k)];
  return sum;
}

Real PhysicsColumn::condensate_path() const noexcept {
  const Int n = nz();
  Real sum = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    sum += (qc[s] + qr[s] + qi[s] + qs[s] + qg[s]) * rho[s] * dz[s];
  }
  return sum;
}

Real PhysicsColumn::water_path() const noexcept { return vapor_path() + condensate_path(); }

Real PhysicsColumn::liquid_water_path() const noexcept {
  const Int n = nz();
  Real sum = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    sum += (qc[s] + qr[s]) * rho[s] * dz[s];
  }
  return sum * Real(1.0e3);  // kg/m^2 -> g/m^2
}

Real PhysicsColumn::ice_water_path() const noexcept {
  const Int n = nz();
  Real sum = Real(0);
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    sum += (qi[s] + qs[s] + qg[s]) * rho[s] * dz[s];
  }
  return sum * Real(1.0e3);
}

bool PhysicsColumn::finite() const noexcept {
  auto ok = [](const std::vector<Real>& a) {
    for (Real x : a) {
      if (!std::isfinite(x)) return false;
    }
    return true;
  };
  return ok(z) && ok(p) && ok(rho) && ok(theta) && ok(qv) && ok(qc) && ok(qr) && ok(qi) &&
         ok(qs) && ok(qg) && ok(u) && ok(v) && ok(tke) && ok(dz);
}

bool PhysicsColumn::positive_definite() const noexcept {
  auto nonneg = [](const std::vector<Real>& a) {
    for (Real x : a) {
      if (x < Real(-1.0e-15)) return false;
    }
    return true;
  };
  return nonneg(qv) && nonneg(qc) && nonneg(qr) && nonneg(qi) && nonneg(qs) && nonneg(qg);
}

Real PhysicsColumn::clip_negatives() noexcept {
  Real removed = Real(0);
  for (std::vector<Real>* f : {&qv, &qc, &qr, &qi, &qs, &qg}) {
    for (Real& x : *f) {
      if (x < Real(0)) {
        removed -= x;
        x = Real(0);
      }
    }
  }
  if (!tke.empty()) {
    for (Real& e : tke) {
      if (e < Real(0)) e = Real(0);
    }
  }
  return removed;
}

void PhysicsColumn::temperatures(std::vector<Real>& t) const {
  const Int n = nz();
  t.resize(static_cast<Size>(n));
  for (Int k = 0; k < n; ++k) t[static_cast<Size>(k)] = temperature(k);
}

// ---------------------------------------------------------------------------
// 云量诊断（Sundqvist 型统计云量，[B1] 第 2 章；[D16] 第 2 章）
//   CF = 1 - exp( -alpha * qc / qc0 ),  alpha = 100, qc0 = 1e-3 kg/kg
//   并对低湿层施加 RH 权重：RH < 0.6 时按 (RH-0.6)/0.4 线性打折
// 离散化：逐层解析；结果裁剪到 [0,1]。
// 复杂度：O(1)。
Real PhysicsColumn::cloud_fraction(Int k) const noexcept {
  const Size s = static_cast<Size>(k);
  const Real qc_v = std::max(qc[s], Real(0));
  if (qc_v <= Real(0)) return Real(0);
  const Real cf_q = Real(1) - std::exp(-Real(100) * qc_v / Real(1.0e-3));
  const Real rh = clamp(relative_humidity(k), Real(0), Real(1.2));
  Real w = Real(1);
  if (rh < Real(0.6)) w = Real(0);
  else if (rh < Real(1.0)) w = (rh - Real(0.6)) / Real(0.4);
  return clamp(cf_q * w, Real(0), Real(1));
}

std::string PhysicsColumn::describe() const {
  std::ostringstream os;
  os << "PhysicsColumn(nz=" << nz() << ")";
  if (nz() > 0) {
    const auto zs = std::minmax_element(z.begin(), z.end());
    const auto ps = std::minmax_element(p.begin(), p.end());
    os << " z=[" << *zs.first << "," << *zs.second << "] m"
       << " p=[" << *ps.first << "," << *ps.second << "] Pa"
       << " 水路径=" << water_path() << " kg/m^2"
       << " 云水路径=" << liquid_water_path() << " g/m^2"
       << " 冰水路径=" << ice_water_path() << " g/m^2";
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// 理想探空列
//   温度：T(z) = T_sfc - gamma z（下限 180 K）
//   气压：静力积分 p(z+dz) = p(z) exp(-g dz/(Rd T_mean))（层平均温度，梯形）
//   湿度：q(z) = q_sfc exp(-z/H)，H = 2 km
//   密度：rho = p/(Rd Tv)
// 复杂度：O(nz)。
PhysicsColumn ideal_sounding(Int nz, Real z_top, Real p_sfc, Real t_sfc, Real qv_sfc,
                            Real lapse, Real shear, Real u_sfc) {
  VIBE_CHECK_MSG(nz > 0, "ideal_sounding 需要 nz > 0");
  PhysicsColumn col;
  col.resize(nz);
  const Real dz0 = z_top / static_cast<Real>(nz);
  const Real H = Real(2000);
  Real p_prev = p_sfc;
  for (Int k = 0; k < nz; ++k) {
    const Size s = static_cast<Size>(k);
    const Real zc = (static_cast<Real>(k) + Real(0.5)) * dz0;
    col.z[s] = zc;
    col.dz[s] = dz0;
    const Real t_lo = std::max(t_sfc - lapse * (zc - Real(0.5) * dz0), Real(180));
    const Real t_hi = std::max(t_sfc - lapse * (zc + Real(0.5) * dz0), Real(180));
    const Real t_mean = Real(0.5) * (t_lo + t_hi);
    p_prev = p_prev * std::exp(-kGravity * dz0 / (kRd * t_mean));
    col.p[s] = p_prev;
    const Real t = std::max(t_sfc - lapse * zc, Real(180));
    col.theta[s] = theta_from_temperature_pressure(t, p_prev);
    col.qv[s] = qv_sfc * std::exp(-zc / H);
    col.qc[s] = Real(0);
    col.u[s] = u_sfc + shear * zc;
    col.v[s] = Real(0);
    col.tke[s] = Real(0);
    col.rho[s] = density_from_pressure_tv(p_prev, virtual_temperature(t, col.qv[s]));
  }
  return col;
}

// ===========================================================================
// 4. ColumnTendency
// ===========================================================================

void ColumnTendency::resize(Int nz) {
  const Size n = static_cast<Size>(std::max(nz, Int(0)));
  theta.assign(n, Real(0)); qv.assign(n, Real(0)); qc.assign(n, Real(0));
  qr.assign(n, Real(0)); qi.assign(n, Real(0)); qs.assign(n, Real(0));
  qg.assign(n, Real(0)); u.assign(n, Real(0)); v.assign(n, Real(0)); tke.assign(n, Real(0));
}

void ColumnTendency::zero() {
  for (std::vector<Real>* f : {&theta, &qv, &qc, &qr, &qi, &qs, &qg, &u, &v, &tke}) {
    std::fill(f->begin(), f->end(), Real(0));
  }
}

void ColumnTendency::add_scaled(Real a, const ColumnTendency& o) {
  VIBE_CHECK(o.nz() == nz());
  const Size n = static_cast<Size>(nz());
  for (Size k = 0; k < n; ++k) {
    theta[k] += a * o.theta[k];
    qv[k] += a * o.qv[k];
    qc[k] += a * o.qc[k];
    qr[k] += a * o.qr[k];
    qi[k] += a * o.qi[k];
    qs[k] += a * o.qs[k];
    qg[k] += a * o.qg[k];
    u[k] += a * o.u[k];
    v[k] += a * o.v[k];
    tke[k] += a * o.tke[k];
  }
}

void ColumnTendency::axpy(Real a, const ColumnTendency& o, Real b) {
  VIBE_CHECK(o.nz() == nz());
  const Size n = static_cast<Size>(nz());
  auto mix = [&](std::vector<Real>& dst, const std::vector<Real>& src) {
    for (Size k = 0; k < n; ++k) dst[k] = a * src[k] + b * dst[k];
  };
  mix(theta, o.theta); mix(qv, o.qv); mix(qc, o.qc); mix(qr, o.qr); mix(qi, o.qi);
  mix(qs, o.qs); mix(qg, o.qg); mix(u, o.u); mix(v, o.v); mix(tke, o.tke);
}

Real ColumnTendency::max_abs() const noexcept {
  Real m = Real(0);
  for (const std::vector<Real>* f : {&theta, &qv, &qc, &qr, &qi, &qs, &qg, &u, &v, &tke}) {
    for (Real x : *f) m = std::max(m, std::abs(x));
  }
  return m;
}

bool ColumnTendency::finite() const noexcept {
  for (const std::vector<Real>* f : {&theta, &qv, &qc, &qr, &qi, &qs, &qg, &u, &v, &tke}) {
    for (Real x : *f) {
      if (!std::isfinite(x)) return false;
    }
  }
  return true;
}

// ===========================================================================
// 5. SurfaceState
// ===========================================================================

// ---------------------------------------------------------------------------
// 五层土壤分层（Noilhan-Planton 简化，[P14]）
//   厚度: 0.05, 0.10, 0.20, 0.30, 0.35 m（总深度 1.0 m）
//   nsoil != 5 时按等厚分配。
// 复杂度 O(nsoil)。
void SurfaceState::resize_soil(Int nsoil) {
  const Int n = std::max(nsoil, Int(1));
  soil_temperature.assign(static_cast<Size>(n), Real(288.0));
  soil_moisture.assign(static_cast<Size>(n), Real(0.25));
  soil_depth.assign(static_cast<Size>(n), Real(1.0) / static_cast<Real>(n));
  if (n == 5) {
    soil_depth = {Real(0.05), Real(0.10), Real(0.20), Real(0.30), Real(0.35)};
  }
}

bool SurfaceState::finite() const noexcept {
  auto okv = [](Real x) { return std::isfinite(x); };
  if (!okv(terrain_height) || !okv(land_fraction) || !okv(roughness) || !okv(albedo) ||
      !okv(surface_pressure) || !okv(t2) || !okv(q2) || !okv(u10) || !okv(v10) ||
      !okv(skin_temperature) || !okv(sensible_heat_flux) || !okv(latent_heat_flux) ||
      !okv(ground_heat_flux) || !okv(sw_down) || !okv(lw_down) || !okv(net_radiation)) {
    return false;
  }
  for (Real x : soil_temperature) {
    if (!std::isfinite(x)) return false;
  }
  for (Real x : soil_moisture) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

std::string SurfaceState::describe() const {
  std::ostringstream os;
  os << "SurfaceState(陆面比例=" << land_fraction << " z0=" << roughness << " m"
     << " T_skin=" << skin_temperature << " K"
     << " T2=" << t2 << " K q2=" << q2
     << " H=" << sensible_heat_flux << " W/m^2"
     << " LE=" << latent_heat_flux << " W/m^2"
     << " G=" << ground_heat_flux << " W/m^2"
     << " Rn=" << net_radiation << " W/m^2"
     << " u*=" << friction_velocity << " m/s L=" << obukhov_length << " m)";
  return os.str();
}

// ===========================================================================
// 6. MY 常数
// ===========================================================================

MellorYamadaConstants MellorYamadaConstants::original_my82() noexcept {
  MellorYamadaConstants c;
  c.a1 = Real(0.92);
  c.a2 = Real(0.74);
  c.b1 = Real(16.6);
  c.b2 = Real(10.1);
  c.c1 = Real(0.08);
  c.s_q = Real(0.2);
  c.btg = kGravity / Real(273.0);
  return c;
}

// ===========================================================================
// 7. PhysicsOptions
// ===========================================================================

void PhysicsOptions::validate() const {
  if (nz <= 0) throw ConfigError("PhysicsOptions::nz 必须 > 0");
  if (nx <= 0 || ny <= 0) throw ConfigError("PhysicsOptions::nx/ny 必须 > 0");
  if (!(dt_physics > Real(0))) throw ConfigError("PhysicsOptions::dt_physics 必须 > 0");
  if (!(dx > Real(0)) || !(dy > Real(0))) throw ConfigError("PhysicsOptions::dx/dy 必须 > 0");
  if (radiation_cadence < 1 || pbl_cadence < 1 || microphysics_cadence < 1 || cumulus_cadence < 1) {
    throw ConfigError("物理过程 cadence 必须 >= 1");
  }
  if (n_longwave_bands < 1 || n_longwave_bands > 64) throw ConfigError("n_longwave_bands 需在 1..64");
  if (n_shortwave_bands < 1 || n_shortwave_bands > 64) throw ConfigError("n_shortwave_bands 需在 1..64");
  if (n_g_points < 1 || n_g_points > 32) throw ConfigError("n_g_points 需在 1..32");
  if (n_soil_layers < 1 || n_soil_layers > 20) throw ConfigError("n_soil_layers 需在 1..20");
  if (!(kessler_k1 > Real(0)) || !(kessler_k2 > Real(0))) throw ConfigError("Kessler 系数必须 > 0");
  if (!(kessler_qc0 > Real(0))) throw ConfigError("kessler_qc0 必须 > 0");
  if (!(kf_closure_time > Real(0))) throw ConfigError("kf_closure_time 必须 > 0");
  if (!(kf_min_cloud_depth > Real(0))) throw ConfigError("kf_min_cloud_depth 必须 > 0");
  if (!(mo_max_iterations >= 2)) throw ConfigError("mo_max_iterations 必须 >= 2");
  if (!(mo_tolerance > Real(0))) throw ConfigError("mo_tolerance 必须 > 0");
  if (!(my.a1 > Real(0) && my.a2 > Real(0) && my.b1 > Real(0))) throw ConfigError("MY 常数必须 > 0");
  if (!(kf_precip_efficiency_land > Real(0) && kf_precip_efficiency_land <= Real(1))) {
    throw ConfigError("kf_precip_efficiency_land 需在 (0,1]");
  }
}

std::string PhysicsOptions::describe() const {
  std::ostringstream os;
  os.setf(std::ios::scientific);
  os.precision(4);
  os << "PhysicsOptions{"
     << " dt=" << dt_physics << "s 网格=" << nx << "x" << ny << "x" << nz
     << " dx=" << dx << "m dt_phys/dx="
     << (dt_physics / std::max(dx, Real(1))) << "\n"
     << "  方案: 微物理=" << to_string(microphysics) << " 辐射=" << to_string(radiation)
     << " PBL=" << to_string(pbl) << " 陆面=" << to_string(surface)
     << " 积云=" << to_string(cumulus) << "\n"
     << "  cadence: 辐射=" << radiation_cadence << " PBL=" << pbl_cadence
     << " 微物理=" << microphysics_cadence << " 积云=" << cumulus_cadence
     << " 加热率权重=" << radiation_heating_relax << "\n"
     << "  Kessler: qc0=" << kessler_qc0 << " k1=" << kessler_k1 << " k2=" << kessler_k2
     << " Vt=a(rho qr)^b a=" << kessler_rain_a << " b=" << kessler_rain_b << "\n"
     << "  Thompson: Nc=" << thompson_nc << " Nr=" << thompson_nr << " Ni=" << thompson_ni
     << " Ns=" << thompson_ns << " Ng=" << thompson_ng << "\n"
     << "  辐射: 长波带=" << n_longwave_bands << " 短波带=" << n_shortwave_bands
     << " g点=" << n_g_points << " CO2=" << co2_ppm << "ppm AOD=" << aerosol_optical_depth
     << " 日=" << julian_day << " UTC=" << utc_hour << "h\n"
     << "  PBL: ri_c=" << ysu_ri_critical << " entrain=" << ysu_entrainment_coef
     << " Pr系数=" << ysu_prandtl_coef << " MY(A1,A2,B1,B2,C1)=(" << my.a1 << "," << my.a2
     << "," << my.b1 << "," << my.b2 << "," << my.c1 << ")\n"
     << "  陆面: 层数=" << n_soil_layers << " lambda=" << soil_thermal_conductivity
     << " C_s=" << soil_heat_capacity << " SST=" << sst << " Charnock=" << charnock_alpha << "\n"
     << "  积云: 卷入=" << kf_entrainment_rate << " CAPE_min=" << kf_min_cape
     << " 云厚>=" << kf_min_cloud_depth << " 闭合时间=" << kf_closure_time << "s\n"
     << "  稳定: cfl=" << cfl_limit << " 垂直cfl=" << vertical_cfl_limit
     << " max_dtheta/dt=" << max_heating_rate << " K/s\n"
     << "}";
  return os.str();
}

// ===========================================================================
// 8. PhysicsDiagnostics
// ===========================================================================

void PhysicsDiagnostics::resize(Int nx_in, Int ny_in) {
  nx = nx_in;
  ny = ny_in;
  const Size n = static_cast<Size>(std::max(nx, Int(0))) * static_cast<Size>(std::max(ny, Int(0)));
  precipitation_grid.assign(n, Real(0));
  precipitation_convective.assign(n, Real(0));
  pbl_height.assign(n, Real(0));
  cloud_fraction.assign(n, Real(0));
  cape.assign(n, Real(0));
  cin.assign(n, Real(0));
  cloud_base.assign(n, Real(0));
  cloud_top.assign(n, Real(0));
  surface_latent_flux.assign(n, Real(0));
  surface_sensible_flux.assign(n, Real(0));
  surface_net_radiation.assign(n, Real(0));
  toa_outgoing_longwave.assign(n, Real(0));
  toa_net_shortwave.assign(n, Real(0));
  reset();
}

void PhysicsDiagnostics::reset() {
  for (std::vector<Real>* f : {&precipitation_grid, &precipitation_convective, &pbl_height,
                               &cloud_fraction, &cape, &cin, &cloud_base, &cloud_top,
                               &surface_latent_flux, &surface_sensible_flux,
                               &surface_net_radiation, &toa_outgoing_longwave,
                               &toa_net_shortwave}) {
    std::fill(f->begin(), f->end(), Real(0));
  }
  precipitation_grid_mean = Real(0);
  precipitation_grid_max = Real(0);
  precipitation_convective_mean = Real(0);
  precipitation_total_mean = Real(0);
  cloud_fraction_mean = Real(0);
  liquid_water_path_mean = Real(0);
  ice_water_path_mean = Real(0);
  pbl_height_mean = Real(0);
  pbl_height_max = Real(0);
  tke_mean = Real(0);
  tke_max = Real(0);
  toa_net_shortwave_mean = Real(0);
  toa_outgoing_longwave_mean = Real(0);
  toa_net_flux_mean = Real(0);
  surface_net_radiation_mean = Real(0);
  surface_sw_down_mean = Real(0);
  surface_lw_down_mean = Real(0);
  surface_flux_heat_mean = Real(0);
  surface_flux_moist_mean = Real(0);
  surface_flux_momentum_mean = Real(0);
  radiation_heating_max = Real(0);
  convective_heating_max = Real(0);
  mass_conservation_residual = Real(0);
  energy_balance_residual_max = Real(0);
  radiation_calls = 0;
  microphysics_calls = 0;
  pbl_calls = 0;
  cumulus_calls = 0;
}

// ---------------------------------------------------------------------------
// 域平均汇总（O(nx*ny)）
// ---------------------------------------------------------------------------
void PhysicsDiagnostics::aggregate() {
  const Size n = static_cast<Size>(std::max(nx, Int(0))) * static_cast<Size>(std::max(ny, Int(0)));
  if (n == 0) return;
  const Real inv = Real(1) / static_cast<Real>(n);
  auto mean_of = [&](const std::vector<Real>& f) {
    Real s = Real(0);
    for (Real x : f) s += x;
    return s * inv;
  };
  auto max_of = [&](const std::vector<Real>& f) {
    Real m = Real(0);
    for (Real x : f) m = std::max(m, x);
    return m;
  };
  precipitation_grid_mean = mean_of(precipitation_grid);
  precipitation_grid_max = max_of(precipitation_grid);
  precipitation_convective_mean = mean_of(precipitation_convective);
  precipitation_total_mean = precipitation_grid_mean + precipitation_convective_mean;
  cloud_fraction_mean = mean_of(cloud_fraction);
  pbl_height_mean = mean_of(pbl_height);
  pbl_height_max = max_of(pbl_height);
  toa_net_shortwave_mean = mean_of(toa_net_shortwave);
  toa_outgoing_longwave_mean = mean_of(toa_outgoing_longwave);
  toa_net_flux_mean = toa_net_shortwave_mean - toa_outgoing_longwave_mean;
  surface_net_radiation_mean = mean_of(surface_net_radiation);
  surface_flux_moist_mean = mean_of(surface_latent_flux);
  surface_flux_heat_mean = mean_of(surface_sensible_flux);
  surface_flux_momentum_mean = Real(0);  // 动量通量只保留逐点值
}

std::string PhysicsDiagnostics::summarize() const {
  std::ostringstream os;
  os.setf(std::ios::scientific);
  os.precision(3);
  os << "物理诊断{ 网格降水=" << precipitation_grid_mean * Real(3600)
     << " mm/h(平均) " << precipitation_grid_max * Real(3600) << " mm/h(最大)"
     << " 对流降水=" << precipitation_convective_mean * Real(3600) << " mm/h"
     << " 云量=" << cloud_fraction_mean
     << " LWP=" << liquid_water_path_mean << " g/m^2 IWP=" << ice_water_path_mean << " g/m^2"
     << " PBLH=" << pbl_height_mean << " m(最大 " << pbl_height_max << ")"
     << " TKE=" << tke_mean << " m^2/s^2(最大 " << tke_max << ")"
     << " 短波净=" << toa_net_shortwave_mean << " W/m^2 OLR=" << toa_outgoing_longwave_mean
     << " W/m^2 大气顶净=" << toa_net_flux_mean << " W/m^2"
     << " 地表净辐射=" << surface_net_radiation_mean << " W/m^2"
     << " H=" << surface_flux_heat_mean << " LE=" << surface_flux_moist_mean << " W/m^2"
     << " 水物质残差=" << mass_conservation_residual
     << " 能量平衡残差=" << energy_balance_residual_max << " W/m^2"
     << " 调用[辐射/PBL/微物理/积云]=" << radiation_calls << "/" << pbl_calls << "/"
     << microphysics_calls << "/" << cumulus_calls << " }";
  return os.str();
}

// ===========================================================================
// 9. 气块抬升与 CAPE/CIN
// ===========================================================================

// ---------------------------------------------------------------------------
// 相当位温（Bolton 1980 近似式）
//     theta_e = theta * exp( Lv(T) qv / (cp T) )
// 文献：[B4] 第 3 章
// 复杂度：O(1)。
Real equivalent_potential_temperature(const PhysicsColumn& col, Int k) noexcept {
  const Real t = col.temperature(k);
  const Real qv = std::max(col.qv[static_cast<Size>(k)], Real(0));
  const Real lv = latent_heat_vaporization(t);
  return col.theta[static_cast<Size>(k)] * std::exp(lv * qv / (kCp * t));
}

Real saturated_equivalent_potential_temperature(Real theta, Real qv, Real p, Real t) noexcept {
  const Real qs = saturation_mixing_ratio(p, t);
  const Real q = std::max(qv, qs);
  return theta * std::exp(latent_heat_vaporization(t) * q / (kCp * t));
}

// ---------------------------------------------------------------------------
// 气块抬升：干绝热段 + 湿绝热段（Newton 迭代求饱和温度）
//   干绝热: theta, qv 守恒
//   湿绝热: theta_e = theta exp(Lv qs(T)/(cp T)) 守恒，Newton 求 T
//           d f/dT = -Lv qs/(cp T) (1 + Lv qs/(cp T))  (用 Clausius-Clapeyron)
//   卷入:   X_new = X + (X_env - X)(1 - exp(-eps dz))
//   浮力:   b = g (Tv_p - Tv_e)/Tv_e，梯形积分得 CAPE/CIN
// 文献：[P11][B4]
// 复杂度：O(nz * n_iter)，n_iter <= 6。
ParcelAscent moist_parcel_ascent(const PhysicsColumn& col, Int source_level,
                                 bool entraining, Real entrainment_rate) {
  ParcelAscent out;
  const Int n = col.nz();
  if (n <= 1) return out;
  Int src = clamp(source_level, Int(0), n - 1);

  const Size ss = static_cast<Size>(src);
  Real theta_p = col.theta[ss];
  Real qv_p = std::max(col.qv[ss], Real(0));
  Real p_p = col.p[ss];
  Real t_p = col.temperature(src);
  out.theta_e_parcel = theta_p * std::exp(latent_heat_vaporization(t_p) * qv_p / (kCp * t_p));
  out.theta_e_env_0 = equivalent_potential_temperature(col, src);

  // 湿绝热段的天顶 theta_e 由 LCL 处确定
  Real theta_e_sat = Real(0);
  bool saturated = false;
  Int lcl_level = -1;
  bool found_lfc = false;
  Real b_prev = Real(0);
  for (Int k = src + 1; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    const Real p_k = col.p[s];
    const Real dz = std::max(col.dz[s], kDzEps);

    if (entraining && entrainment_rate > Real(0)) {
      const Real frac = Real(1) - std::exp(-entrainment_rate * dz);
      theta_p += frac * (col.theta[s] - theta_p);
      qv_p += frac * (std::max(col.qv[s], Real(0)) - qv_p);
      if (saturated && theta_e_sat > Real(0)) {
        theta_e_sat += frac * (equivalent_potential_temperature(col, k) - theta_e_sat);
      }
    }

    // 干绝热温度
    t_p = theta_p * exner_from_pressure(p_k);
    const Real qs_dry = saturation_mixing_ratio(p_k, t_p);
    if (!saturated && qv_p > qs_dry) {
      // 到达 LCL：用当前 theta_e 作为守恒量
      saturated = true;
      lcl_level = k;
      theta_e_sat = theta_p * std::exp(latent_heat_vaporization(t_p) * qs_dry / (kCp * t_p));
      out.plcl = p_k;
      out.zlcl = col.z[s];
    }

    if (saturated) {
      // Newton 迭代：求解 F(T) = theta(T) exp(Lv qs(T)/(cp T)) - theta_e_sat = 0
      // 其中 theta(T) = T / pi(p)（pi 由环境气压确定，气块与环境的 p 相同）
      const Real pi = exner_from_pressure(p_k);
      Real t_sat = std::max(t_p, Real(150));
      Real qs = Real(0);
      for (int it = 0; it < 6; ++it) {
        qs = saturation_mixing_ratio(p_k, t_sat);
        const Real lv = latent_heat_vaporization(t_sat);
        const Real a = lv * qs / (kCp * t_sat);
        const Real f = (t_sat / pi) * std::exp(a) - theta_e_sat;
        // dq_s/dT = Lv qs/(Rv T^2)
        const Real dqsdT = lv * qs / (kRv * t_sat * t_sat);
        const Real da = lv * dqsdT / (kCp * t_sat) - a / t_sat;
        const Real df = (Real(1) / pi) * std::exp(a) + (t_sat / pi) * std::exp(a) * da;
        if (std::abs(df) < Real(1.0e-30)) break;
        const Real dt_newton = f / df;
        t_sat -= clamp(dt_newton, Real(-30), Real(30));
        if (std::abs(dt_newton) < Real(1.0e-6)) break;
      }
      t_p = t_sat;
      qv_p = qs;
    }

    // 浮力与 CAPE/CIN 累加（梯形）
    const Real tve = col.virtual_temperature(k);
    const Real tvp = virtual_temperature(t_p, qv_p);
    const Real b = kGravity * (tvp - tve) / std::max(tve, Real(1));
    // 源层处气块与环境一致，故 b_prev 由 0 起步（梯形积分）
    const Real trap = Real(0.5) * (b + b_prev) * dz;
    if (!found_lfc) {
      if (b > Real(0) && b_prev <= Real(0)) {
        found_lfc = true;
        out.lfc = k;
      } else if (b < Real(0)) {
        out.cin += trap;
      }
    }
    if (found_lfc) {
      if (b <= Real(0)) {
        out.el = k;
        break;
      }
      out.cape += trap;
    }
    b_prev = b;
  }
  if (lcl_level < 0) out.lfc = -1;
  out.cin = std::min(out.cin, Real(0));
  out.cape = std::max(out.cape, Real(0));
  return out;
}

// ===========================================================================
// 10. 守恒与 CFL
// ===========================================================================

Real column_total_water(const PhysicsColumn& col) noexcept { return col.total_water_mass(); }

Real relative_water_residual(Real before, Real after) noexcept {
  const Real den = std::max(std::abs(before), Real(1.0e-10));
  return std::abs(after - before) / den;
}

// ---------------------------------------------------------------------------
// CFL 诊断
//   水平:   C_h = max(|u|,|v|) dt / dx
//   落速:   C_f = V_max dt / min(dz)
//   扩散:   C_d = 2 K dt / min(dz)^2
// 文献：[B2] 第 3 章；[D5]
// 复杂度：O(nz)。
PhysicsCfl compute_cfl(const PhysicsColumn& col, Real dt, Real dx,
                       Real max_fall_speed, Real max_diffusivity) noexcept {
  PhysicsCfl c;
  const Int n = col.nz();
  if (n <= 0) return c;
  Real vmax = Real(0);
  Real dzmin = kHuge;
  for (Int k = 0; k < n; ++k) {
    const Size s = static_cast<Size>(k);
    vmax = std::max(vmax, std::max(std::abs(col.u[s]), std::abs(col.v[s])));
    if (col.dz[s] > Real(0)) dzmin = std::min(dzmin, col.dz[s]);
  }
  if (dzmin >= kHuge) dzmin = Real(1);
  c.min_dz = dzmin;
  c.horizontal = (dx > Real(0)) ? vmax * dt / dx : Real(0);
  c.fall_speed = max_fall_speed * dt / dzmin;
  c.diffusion = Real(2) * max_diffusivity * dt / (dzmin * dzmin);
  c.vertical = std::max(c.fall_speed, c.diffusion);
  c.total = std::max(c.horizontal, c.vertical);
  return c;
}

}  // namespace vibe::physics

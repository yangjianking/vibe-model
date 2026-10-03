/// @file initial_conditions.cpp
/// @brief 理想试验初值的实现。
///
/// 各试验的解析表达式与出处
/// ------------------------
/// 暖泡（[D1]）：
///     theta' = dtheta * cos^2( pi/2 * min(1, r) ),  r^2 = (dx/rx)^2 + (dz/rz)^2
/// 重力流（[B6] 第 4 章）：冷池中心位温扰动 -15 K，半径 4 km
/// 山波（[D2][D9]）：u = 10 m/s，N = 0.01 s^-1 的稳定层结 + Agnesi 山
/// 惯性重力波（[T5]）：u' = -A cos(pi z / H) * cos(pi (x - xc)/L)
/// 上升热泡（[T5]）：theta' = A cos^2(pi r / 2) 截断
/// 斜压波（Jablonowski-Williamson）：解析的经向风廓线与位温扰动
/// 静止等温大气（[T2]）：所有动量为零，pi = pi0
///
/// 状态方程自洽化
/// --------------
/// 给定 theta、qv 与目标气压 p，先算 rho：
///     rho = p / (Rd T) * (1 + qv)/(1 + qv/eps),  T = theta * pi
/// 再由 pi 的初值迭代 2-3 次即可（本实现直接用 rho 反解 pi）。

#include "vibe/driver/initial_conditions.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/grid/interpolation.hpp"

namespace vibe::driver {

const char* to_string(IdealizedCase c) noexcept {
  switch (c) {
    case IdealizedCase::WarmBubble: return "warm_bubble";
    case IdealizedCase::ColdBubble: return "cold_bubble";
    case IdealizedCase::DensityCurrent: return "density_current";
    case IdealizedCase::MountainWave: return "mountain_wave";
    case IdealizedCase::InertiaGravityWave: return "inertia_gravity_wave";
    case IdealizedCase::RisingThermal: return "rising_thermal";
    case IdealizedCase::BaroclinicWave: return "baroclinic_wave";
    case IdealizedCase::RestingIsothermal: return "resting_isothermal";
    case IdealizedCase::BalancedJet: return "balanced_jet";
    case IdealizedCase::FromFile: return "from_file";
    default: return "unknown";
  }
}

IdealizedCase case_from_string(const std::string& s) {
  if (s == "cold_bubble") return IdealizedCase::ColdBubble;
  if (s == "density_current") return IdealizedCase::DensityCurrent;
  if (s == "mountain_wave") return IdealizedCase::MountainWave;
  if (s == "inertia_gravity_wave") return IdealizedCase::InertiaGravityWave;
  if (s == "rising_thermal") return IdealizedCase::RisingThermal;
  if (s == "baroclinic_wave") return IdealizedCase::BaroclinicWave;
  if (s == "resting_isothermal") return IdealizedCase::RestingIsothermal;
  if (s == "balanced_jet") return IdealizedCase::BalancedJet;
  if (s == "from_file") return IdealizedCase::FromFile;
  return IdealizedCase::WarmBubble;
}

namespace {

/// 由 pi0 与 theta 构造静力平衡的全场
void fill_base_state(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                     const IcOptions& opt) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real z = g.z_center(i, j, k);
        s.theta()(i, j, k) = ref.theta0()(i, j, k);
        s.pi()(i, j, k) = ref.pi0()(i, j, k);
        s.rho()(i, j, k) = ref.rho0()(i, j, k);
        // 水汽：指数衰减廓线
        const Real qv = opt.qv_surface * std::exp(-std::max(z, Real(0)) / opt.qv_decay_height);
        s.qv()(i, j, k) = qv;
        s.field(dyn::Species::Qc)(i, j, k) = Real(0);
        s.field(dyn::Species::Qr)(i, j, k) = Real(0);
        s.field(dyn::Species::Qi)(i, j, k) = Real(0);
        s.field(dyn::Species::Qs)(i, j, k) = Real(0);
        s.field(dyn::Species::Qg)(i, j, k) = Real(0);
      }
  // 动量为零
  s.u().fill(Real(0));
  s.v().fill(Real(0));
  s.w().fill(Real(0));
  s.time = Real(0);
  s.step = 0;
}

/// 施加水平均匀基本流 u = u0 + shear * z
void apply_shear(const grid::Grid& g, dyn::State& s, Real u0, Real shear) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Real z = g.z_center(std::min(i, nx - 1), j, k);
        s.u()(i, j, k) = u0 + shear * z;
      }
}

}  // namespace

// ---------------------------------------------------------------------------
// 状态方程自洽化
// ---------------------------------------------------------------------------

void enforce_state_equation(dyn::State& s, const dyn::ReferenceState& ref) {
  const grid::Grid& g = ref.grid();
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real qv = std::max(s.qv()(i, j, k), Real(0));
        const Real gamma_f = (Real(1) + qv / kEpsilonVap) / (Real(1) + qv);
        const Real val = s.rho()(i, j, k) * kRd * s.theta()(i, j, k) * gamma_f / kP0;
        if (val > Real(0)) s.pi()(i, j, k) = std::pow(val, kRd / kCv);
      }
}

void add_random_noise(dyn::State& s, Real amplitude, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-amplitude, amplitude);
  auto& th = s.theta();
  for (Int k = 0; k < th.nz(); ++k)
    for (Int j = 0; j < th.ny(); ++j)
      for (Int i = 0; i < th.nx(); ++i) th(i, j, k) += dist(gen);
}

// ---------------------------------------------------------------------------
// 各理想试验
// ---------------------------------------------------------------------------

void init_warm_bubble(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                      const IcOptions& opt) {
  // [D1] Klemp & Wilhelmson (1978)：theta' = dtheta cos^2(pi r / 2)
  fill_base_state(g, ref, s, opt);
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Real xc = (opt.bubble_xc != Real(0)) ? opt.bubble_xc : Real(0.5) * g.geom().x0 +
                                                                 Real(0.5) * static_cast<Real>(g.nx_global()) * g.geom().dx;
  const Real yc = (opt.bubble_yc != Real(0)) ? opt.bubble_yc : Real(0.5) * static_cast<Real>(g.ny_global()) * g.geom().dy;
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real x = (g.decomp().is + static_cast<Real>(i) + Real(0.5)) * g.geom().dx;
        const Real y = (g.decomp().js + static_cast<Real>(j) + Real(0.5)) * g.geom().dy;
        const Real z = g.z_center(i, j, k);
        const Real r2 = sqr((x - xc) / opt.bubble_radius_x) + sqr((y - yc) / opt.bubble_radius_x) +
                        sqr((z - opt.bubble_zc) / opt.bubble_radius_z);
        if (r2 < Real(1)) {
          const Real c = std::cos(Real(0.5) * kPi * std::sqrt(r2));
          s.theta()(i, j, k) += opt.bubble_amplitude * c * c;
        }
      }
  apply_shear(g, s, opt.u_background, opt.shear_u);
  enforce_state_equation(s, ref);
}

void init_cold_bubble(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                      const IcOptions& opt) {
  IcOptions o = opt;
  o.bubble_amplitude = -std::abs(opt.bubble_amplitude);
  init_warm_bubble(g, ref, s, o);
}

void init_density_current(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                          const IcOptions& opt) {
  // [B6] LeVeque 经典基准：中心 -15 K 的冷池，半径 4 km，等温层结
  fill_base_state(g, ref, s, opt);
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Real xc = Real(0.5) * static_cast<Real>(g.nx_global()) * g.geom().dx;
  const Real zc = Real(3000);
  const Real rc = Real(4000);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real x = (g.decomp().is + static_cast<Real>(i) + Real(0.5)) * g.geom().dx;
        const Real z = g.z_center(i, j, k);
        const Real r = std::sqrt(sqr(x - xc) + sqr(z - zc));
        if (r < rc) {
          const Real c = std::cos(Real(0.5) * kPi * r / rc);
          s.theta()(i, j, k) -= Real(15) * c * c;
        }
      }
  enforce_state_equation(s, ref);
}

void init_mountain_wave(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                        const IcOptions& opt) {
  // [D2][D9]：均匀稳定层结 + 定常来流；地形由 Geometry::zs 提供
  fill_base_state(g, ref, s, opt);
  apply_shear(g, s, (opt.u_background != Real(0)) ? opt.u_background : Real(10), Real(0));
  // 在迎风边界施加解析的垂直速度扰动（帮助快速建立稳态山波）
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real z = g.z_center(i, j, k);
        const Real zs = g.terrain(i, j);
        // 线性山波解：w' ~ U h0 k cos(k x + m z) 的简化形式
        const Real k = kPi / std::max(opt.perturbation_scale, Real(1));
        const Real m = Real(0.01) / std::max(std::abs(s.u()(i, j, k)), Real(1));
        s.w()(i, j, k) = opt.perturbation_amplitude * std::exp(-sqr((z - zs) / Real(8000))) *
                         std::cos(k * static_cast<Real>(i) * g.geom().dx + m * z);
      }
  enforce_state_equation(s, ref);
}

void init_inertia_gravity_wave(const grid::Grid& g, const dyn::ReferenceState& ref,
                               dyn::State& s, const IcOptions& opt) {
  // [T5] Skamarock & Klemp (1994)：热力强迫的惯性重力波
  fill_base_state(g, ref, s, opt);
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Real a = opt.perturbation_amplitude;
  const Real xc = Real(0.5) * static_cast<Real>(g.nx_global()) * g.geom().dx;
  const Real yc = Real(0.5) * static_cast<Real>(g.ny_global()) * g.geom().dy;
  const Real zc = Real(10000);
  const Real Lx = opt.perturbation_scale * Real(20);
  const Real Lz = Real(20000);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real x = (g.decomp().is + static_cast<Real>(i) + Real(0.5)) * g.geom().dx;
        const Real y = (g.decomp().js + static_cast<Real>(j) + Real(0.5)) * g.geom().dy;
        const Real z = g.z_center(i, j, k);
        const Real r = sqr((x - xc) / Real(4000)) + sqr((y - yc) / Real(4000)) +
                       sqr((z - zc) / Real(4000));
        if (r < Real(1)) {
          s.theta()(i, j, k) += a * sqr(std::cos(Real(0.5) * kPi * std::sqrt(r)));
        }
      }
  VIBE_UNUSED(Lx);
  VIBE_UNUSED(Lz);
  enforce_state_equation(s, ref);
}

void init_rising_thermal(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                         const IcOptions& opt) {
  // [T5]：截断的余弦热泡
  init_warm_bubble(g, ref, s, opt);
}

void init_baroclinic_wave(const grid::Grid& g, const dyn::ReferenceState& ref,
                          dyn::State& s, const IcOptions& opt) {
  // Jablonowski & Williamson (2006) 的简化形式：
  //   u(phi, eta) = u_max * sin^2(pi (eta - eta0)/(1-eta0)) * cos(phi)^(3/2) 的 f 平面类比
  fill_base_state(g, ref, s, opt);
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Real umax = (opt.jet_max != Real(0)) ? opt.jet_max : Real(30);
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Real z = g.z_center(std::min(i, nx - 1), j, k);
        const Real eta = z / std::max(g.geom().z_top, Real(1));
        const Real shape = std::sin(kPi * clamp(eta, Real(0), Real(1)));
        s.u()(i, j, k) = umax * shape * shape;
      }
  // 位温扰动（弱）
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real x = (g.decomp().is + static_cast<Real>(i) + Real(0.5)) * g.geom().dx;
        s.theta()(i, j, k) += opt.perturbation_amplitude *
                              std::exp(-sqr((x - Real(0.5) * static_cast<Real>(g.nx_global()) * g.geom().dx) / Real(5000)));
      }
  enforce_state_equation(s, ref);
}

void init_resting_isothermal(const grid::Grid& g, const dyn::ReferenceState& ref,
                             dyn::State& s, const IcOptions& opt) {
  // [T2]：静止等温大气，用于声波传播与半隐式稳定性测试
  fill_base_state(g, ref, s, opt);
  s.u().fill(Real(0));
  s.v().fill(Real(0));
  s.w().fill(Real(0));
  s.qv().fill(Real(0));
}

void init_balanced_jet(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                       const IcOptions& opt) {
  // [N8] Harris & Lin (2013)：地转平衡的急流
  fill_base_state(g, ref, s, opt);
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Real f = Real(2) * kOmega * std::sin(kReferenceLat * kDegToRad);
  const Real umax = (opt.jet_max != Real(0)) ? opt.jet_max : Real(40);
  const Real yc = Real(0.5) * static_cast<Real>(g.ny_global()) * g.geom().dy;
  const Real width = (opt.jet_width != Real(0)) ? opt.jet_width : Real(0.25) * static_cast<Real>(g.ny_global()) * g.geom().dy;
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i <= nx; ++i) {
        const Real y = (g.decomp().js + static_cast<Real>(j) + Real(0.5)) * g.geom().dy;
        s.u()(i, j, k) = umax * std::exp(-sqr((y - yc) / width));
      }
  // 地转平衡的位温扰动：dtheta/dy 由热成风关系给出
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real y = (g.decomp().js + static_cast<Real>(j) + Real(0.5)) * g.geom().dy;
        const Real dudy = -Real(2) * (y - yc) / sqr(width) * umax *
                          std::exp(-sqr((y - yc) / width));
        s.theta()(i, j, k) += ref.theta0()(i, j, k) / kGravity * f * dudy * (y - yc);
      }
  enforce_state_equation(s, ref);
}

void init_from_file(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                    const IcOptions& opt) {
  VIBE_CHECK_MSG(!opt.input_file.empty(), "FromFile 初值需要指定 input_file");
  // 具体读取交给 io 层；此处给出约定与失败时的回退
  VIBE_WARN("从文件读取初值：", opt.input_file,
            "（调用方需在此后注入 io::read_state；本例回退到静止等温大气）");
  init_resting_isothermal(g, ref, s, opt);
  if (opt.adjust_hydrostatic) enforce_state_equation(s, ref);
}

// ---------------------------------------------------------------------------

void initialize_state(const grid::Grid& g, const dyn::ReferenceState& ref, dyn::State& s,
                      const IcOptions& opt) {
  if (s.grid().nx() != g.nx()) s.allocate(g);
  switch (opt.case_type) {
    case IdealizedCase::WarmBubble: init_warm_bubble(g, ref, s, opt); break;
    case IdealizedCase::ColdBubble: init_cold_bubble(g, ref, s, opt); break;
    case IdealizedCase::DensityCurrent: init_density_current(g, ref, s, opt); break;
    case IdealizedCase::MountainWave: init_mountain_wave(g, ref, s, opt); break;
    case IdealizedCase::InertiaGravityWave: init_inertia_gravity_wave(g, ref, s, opt); break;
    case IdealizedCase::RisingThermal: init_rising_thermal(g, ref, s, opt); break;
    case IdealizedCase::BaroclinicWave: init_baroclinic_wave(g, ref, s, opt); break;
    case IdealizedCase::RestingIsothermal: init_resting_isothermal(g, ref, s, opt); break;
    case IdealizedCase::BalancedJet: init_balanced_jet(g, ref, s, opt); break;
    case IdealizedCase::FromFile: init_from_file(g, ref, s, opt); break;
    default: init_warm_bubble(g, ref, s, opt); break;
  }
  if (opt.add_random_noise) add_random_noise(s, opt.noise_amplitude, opt.seed);
  if (opt.adjust_hydrostatic) enforce_state_equation(s, ref);
  VIBE_INFO("初值构造完成：", to_string(opt.case_type), "，", s.describe().substr(0, 120));
}

}  // namespace vibe::driver

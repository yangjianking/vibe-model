#pragma once
/// @file initial_conditions.hpp
/// @brief 理想试验与真实个例的初值构造。
///
/// 提供的理想试验（每个都有明确的文献出处与解析解/参考解）
/// ------------------------------------------------------
///   * 暖泡对流（Warm bubble）        [D1] Klemp & Wilhelmson (1978)
///   * 冷泡 / 重力流（Density current）[B6] LeVeque (2002) 经典基准
///   * 山波（Mountain wave）           [D2][D9][D10]
///   * 惯性重力波（IGW）               [T5] Skamarock & Klemp (1994)
///   * 位温扰动上升热泡（Rising thermal）[T5] Wicker & Skamarock (2002)
///   * 斜压波（Baroclinic wave）        Jablonowski & Williamson (2006)
///   * 静止等温大气（声波测试）         [T2] Tapp & White (1976)
///   * 平衡急流（Balanced jet）         [N8] Harris & Lin (2013)
///   * 真实个例：从 NetCDF/GRIB 读取三维场并做静力平衡调整
///
/// 共同约定：所有构造函数把场写入 **全量** 变量（rho, theta, pi 为总场），
/// 参考态由 ReferenceState 单独提供；因此调用方需要先构造网格与参考态。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::driver {

/// 理想试验类型
enum class IdealizedCase {
  WarmBubble,
  ColdBubble,
  DensityCurrent,
  MountainWave,
  InertiaGravityWave,
  RisingThermal,
  BaroclinicWave,
  RestingIsothermal,
  BalancedJet,
  FromFile,
  Count
};

const char* to_string(IdealizedCase c) noexcept;
IdealizedCase case_from_string(const std::string& s);

/// 初值参数
struct IcOptions {
  IdealizedCase case_type = IdealizedCase::WarmBubble;
  Real bubble_amplitude = Real(2.0);      ///< 位温扰动幅度 (K)
  Real bubble_radius_x = Real(2000);      ///< 水平半径 (m)
  Real bubble_radius_z = Real(2000);      ///< 垂直半径 (m)
  Real bubble_zc = Real(2000);            ///< 中心高度 (m)
  Real bubble_xc = Real(0), bubble_yc = Real(0);
  Real surface_theta = Real(300);         ///< 地面位温 (K)
  Real surface_pressure = Real(100000);   ///< 地面气压 (Pa)
  Real qv_surface = Real(0.014);          ///< 地面水汽混合比
  Real qv_decay_height = Real(3000);      ///< 水汽指数衰减高度
  Real shear_u = Real(0);                 ///< 基本态风切变 du/dz
  Real u_background = Real(0);            ///< 均匀基本流
  Real jet_max = Real(0);                 ///< 急流最大风速
  Real jet_width = Real(0);
  Real jet_center = Real(0);
  Real perturbation_amplitude = Real(1.0);
  Real perturbation_scale = Real(1000);
  unsigned seed = 42;
  std::string input_file;                 ///< FromFile 时的路径
  bool adjust_hydrostatic = true;         ///< 是否做静力平衡调整
  bool add_random_noise = false;
  Real noise_amplitude = Real(0);
};

/// 主入口：按选项初始化状态
void initialize_state(const grid::Grid& g, const dyn::ReferenceState& ref,
                      dyn::State& s, const IcOptions& opt);

/// 各理想试验的独立实现（便于单元测试）
void init_warm_bubble(const grid::Grid& g, const dyn::ReferenceState& ref,
                      dyn::State& s, const IcOptions& opt);
void init_cold_bubble(const grid::Grid& g, const dyn::ReferenceState& ref,
                      dyn::State& s, const IcOptions& opt);
void init_density_current(const grid::Grid& g, const dyn::ReferenceState& ref,
                          dyn::State& s, const IcOptions& opt);
void init_mountain_wave(const grid::Grid& g, const dyn::ReferenceState& ref,
                        dyn::State& s, const IcOptions& opt);
void init_inertia_gravity_wave(const grid::Grid& g, const dyn::ReferenceState& ref,
                               dyn::State& s, const IcOptions& opt);
void init_rising_thermal(const grid::Grid& g, const dyn::ReferenceState& ref,
                         dyn::State& s, const IcOptions& opt);
void init_baroclinic_wave(const grid::Grid& g, const dyn::ReferenceState& ref,
                          dyn::State& s, const IcOptions& opt);
void init_resting_isothermal(const grid::Grid& g, const dyn::ReferenceState& ref,
                             dyn::State& s, const IcOptions& opt);
void init_balanced_jet(const grid::Grid& g, const dyn::ReferenceState& ref,
                       dyn::State& s, const IcOptions& opt);

/// 从文件读取（调用 io 层）
void init_from_file(const grid::Grid& g, const dyn::ReferenceState& ref,
                    dyn::State& s, const IcOptions& opt);

/// 质量场与 Exner 的自洽化：给定 rho, theta, qv 反解 pi
void enforce_state_equation(dyn::State& s, const dyn::ReferenceState& ref);

/// 叠加随机小扰动（用于触发对称不稳定/破对称）
void add_random_noise(dyn::State& s, Real amplitude, unsigned seed);

}  // namespace vibe::driver

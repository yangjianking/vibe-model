#pragma once
/// @file config.hpp
/// @brief 运行配置的数据结构与载入器。
///
/// 配置分层
/// --------
///   1. `config/model.yaml`   ：网格、时间步、数值格式、物理、并行
///   2. `config/nests.yaml`   ：嵌套层级与变分辨率设置（可选）
///   3. `config/da_4dvar.yaml`：同化窗口、B 矩阵、极小化、观测选择
///   4. `config/verify.yaml`  ：检验变量、阈值、方法、邻域半径
///
/// 载入策略：先读默认值，再按路径列表依次覆盖（`merge`），最后调用
/// `validate()` 做一致性检查（例如 CFL、层数、进程数与网格的整除性）。
/// 所有错误抛 `ConfigError`，附带字段路径。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/config/yaml_node.hpp"

namespace vibe::config {

// ============================== 模式配置 ====================================

struct DomainConfig {
  Int nx = 128, ny = 128, nz = 60;
  Real dx = Real(2000), dy = Real(2000);
  Real x0 = Real(0), y0 = Real(0);
  Real z_top = Real(22000);
  std::string terrain;                 ///< 地形文件路径；空 = 平坦
  std::vector<Real> zeta;              ///< 显式垂直层坐标；空 = 自动生成
  Real zeta_first_thickness = Real(60);
  Real zeta_stretch = Real(1.06);
  bool variable_resolution = false;
  Real h_min = Real(1000), h_max = Real(15000);
  std::string refinement;              ///< "circular" / "channel" / "none"
  Real refinement_cx = Real(0), refinement_cy = Real(0), refinement_radius = Real(0);
  Real refinement_transition = Real(0);
  bool periodic_x = false, periodic_y = false;
};

struct TimeConfig {
  Real dt = Real(10);
  Real run_length = Real(3600);        ///< 秒
  Int  acoustic_substeps = 6;
  std::string integrator = "rk3_acoustic";  ///< rk3_acoustic / semi_implicit
  Real cfl_target = Real(0.8);
  bool adaptive_dt = false;
  Real output_interval = Real(600);
  Real restart_interval = Real(0);     ///< 0 = 不写重启
  std::string start_time = "2000-01-01T00:00:00Z";
};

struct NumericsConfig {
  std::string advection = "central2";  ///< central2/central4/central6/weno5/upwind3
  bool flux_form = true;
  Real divergence_damping = Real(0.0);
  int  divergence_order = 2;
  Real sponge_alpha = Real(0.2);
  Real sponge_start_fraction = Real(0.8);
  std::string helmholtz_solver = "krylov_multigrid";
  int  helmholtz_max_iter = 200;
  Real helmholtz_tol = Real(1e-8);
  bool weno = false;
  std::string coriolis = "fplane";     ///< none/fplane/betaplane
  Real fplane_latitude = Real(45.0);
};

struct NestingConfig {
  bool enabled = false;
  Int levels = 0;
  Int ratio = 3;
  bool two_way = true;
  Int boundary_zone = 5;
  Real feedback_interval = Real(0);    ///< 0 = 每个父步都反馈
  std::vector<std::string> child_domains;
};

struct PhysicsConfig {
  std::string microphysics = "kessler";
  std::string radiation = "rrtmg_simple";
  std::string pbl = "ysu";
  std::string surface = "monin_obukhov";
  std::string cumulus = "none";
  Real radiation_cadence = Real(900);
  bool use_cloud_fraction = true;
  Real co2_ppm = Real(420);
  Real aerosol_optical_depth = Real(0.1);
  Int soil_layers = 5;
  bool enable_tendency_physics = true;
};

struct ParallelConfig {
  Int px = 1, py = 1;
  std::string io_backend = "netcdf";
  std::string output_dir = "output";
  bool write_double = true;
};

/// 完整模式配置
struct ModelConfig {
  std::string name = "vibe_run";
  std::string description;
  DomainConfig domain;
  TimeConfig time;
  NumericsConfig numerics;
  NestingConfig nesting;
  PhysicsConfig physics;
  ParallelConfig parallel;

  /// 从 YAML 节点构造
  static ModelConfig from_yaml(const YamlNode& root);
  /// 从单文件构造
  static ModelConfig load(const std::string& path);
  /// 从多文件叠加构造（后者覆盖前者）
  static ModelConfig load(const std::vector<std::string>& paths, bool allow_missing = true);

  /// 一致性校验；失败抛 ConfigError
  void validate() const;

  /// 估算最大稳定时间步（基于 dx、最大风速 100 m/s 与声速 350 m/s）
  Real suggested_dt() const;

  /// 摘要字符串（日志输出）
  std::string describe() const;
};

// ============================== 同化配置 ====================================

struct BackgroundErrorConfig {
  std::string method = "nmc";          ///< nmc / ensemble / hybrid
  std::string balance = "linear_balance";  ///< none / linear_balance / omega
  Real horizontal_length_scale = Real(200000);  ///< 米
  Real vertical_length_scale = Real(3000);
  Real diffusion_order = Real(2);
  Real variance_scale = Real(1.0);
  std::vector<Real> vertical_eof_fractions;  ///< EOF 截断能量比
  int  n_eofs = 10;
  Real hybrid_weight = Real(0.5);      ///< 混合 B 中集合占比
  std::string ensemble_dir;
};

struct CostFunctionConfig {
  bool use_penalty = true;
  Real penalty_weight = Real(1e-4);
  bool use_digital_filter = true;
  Real digital_filter_weight = Real(1e-5);
  bool use_model_error = false;
  Real model_error_scale = Real(0.0);
  bool use_weak_constraint = false;
};

struct MinimizerConfig {
  std::string method = "lbfgs";        ///< lbfgs / cg / lanczos
  int  max_iterations = 50;
  Real gradient_tolerance = Real(1e-6);
  Real cost_tolerance = Real(1e-8);
  int  memory = 10;                    ///< L-BFGS 存储对数
  Real initial_step = Real(0.1);
  bool preconditioned = true;
};

struct AssimilationWindow {
  std::string start = "2000-01-01T00:00:00Z";
  Real length = Real(21600);           ///< 6 小时
  Int  slot_interval = 12;             ///< 观测时隙数
};

struct DaConfig {
  bool enabled = false;
  AssimilationWindow window;
  BackgroundErrorConfig background;
  CostFunctionConfig cost;
  MinimizerConfig minimizer;
  int  outer_loops = 2;
  int  inner_loops = 50;
  Real inner_resolution_factor = Real(2.0);  ///< 内层循环的降分辨率因子
  bool assimilate_moisture = true;
  bool use_tl_ad = true;               ///< false = 用非线性模式做扰动预报
  std::string obs_list;                ///< 观测文件名
  std::string bias_correction = "variational_1d";
  bool check_adjoint = false;          ///< 启动时做点积检验

  static DaConfig from_yaml(const YamlNode& root);
  static DaConfig load(const std::string& path);
  void validate() const;
};

// ============================== 检验配置 ====================================

struct VerifyConfig {
  std::vector<std::string> variables{"t2", "u10", "v10", "ps"};
  std::vector<Real> thresholds{Real(0.1), Real(1.0), Real(5.0), Real(10.0)};
  std::vector<std::string> methods{"bias", "rmse", "mae", "correlation", "ets", "fss"};
  std::vector<int> neighborhood_radii{1, 3, 5, 9};
  std::string matchup = "nearest";     ///< nearest / bilinear / trilinear
  Real temporal_tolerance = Real(1800);///< 秒
  std::string climatology;             ///< 气候态文件（用于 ACC）
  bool compute_ensemble_scores = false;
  std::string output_format = "csv";

  static VerifyConfig from_yaml(const YamlNode& root);
  static VerifyConfig load(const std::string& path);
};

}  // namespace vibe::config

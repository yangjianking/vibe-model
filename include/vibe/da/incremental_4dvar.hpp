#pragma once
/// @file incremental_4dvar.hpp
/// @brief 增量 4D-Var 的主控流程。
///
/// 算法（[V3] Courtier et al. 1994；[V10] Rabier et al. 2000）
/// ----------------------------------------------------------
/// @verbatim
///   x_b  <- 背景（forecast）
///   for k = 1 .. n_outer:
///       x_k  <- 非线性模式从 x_{k-1} 积分（生成轨迹，分辨率 r_k）
///       d_i  <- y_i - H_i(x_k)                       （创新向量）
///       v    <- 0 或 上一外层的结果（热启动）
///       for j = 1 .. n_inner:
///           J(v), grad J(v) 需要一次 TL 前向 + AD 反向
///           v <- argmin J(v)                          （L-BFGS）
///       x_k  <- x_k + U v                             （更新分析/轨迹）
///   x_a = x_{n_outer}
/// @endverbatim
///
/// 分辨率递进
/// ----------
/// 内层循环通常在**降分辨率**网格上求解（代价更低），外层循环回到全分辨率
/// 重算轨迹与创新向量。这就是"增量"4D-Var 的含义（[V3]）。
///
/// 弱约束
/// ------
/// 可通过 `use_weak_constraint` 开启模式误差控制变量（[V15]），
/// 在代价函数中加入 Q^{-1} 项，但会显著增加控制向量维度。
///
/// 文献：[V1][V2][V3][V10][V11][V15][V22]。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/config/config.hpp"
#include "vibe/da/control_vector.hpp"
#include "vibe/da/cost_function.hpp"
#include "vibe/da/da_types.hpp"
#include "vibe/da/minimizer.hpp"
#include "vibe/da/tangent_linear.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/obs/obs_operator.hpp"
#include "vibe/obs/observations.hpp"
#include "vibe/time/integrator.hpp"

namespace vibe::da {

/// 非线性模式前向积分的回调（由 driver 注入，避免 da 依赖 driver）
using ForecastFunction =
    std::function<void(const dyn::State& x0, Real dt, int n_steps, dyn::State& xf)>;

/// 增量 4D-Var 配置
struct Incremental4DVarConfig {
  LoopStructure loops;
  MinimizerOptions minimizer;
  ControlVariableConfig control;
  ObservationTermConfig observation;
  PenaltyTerm penalty;
  PhysicsLinearization physics = PhysicsLinearization::Adiabatic;
  CheckpointStrategy checkpoint = CheckpointStrategy::StoreAll;
  bool use_weak_constraint = false;
  bool check_adjoint_on_start = false;
  bool verbose = true;
  int  output_interval = 1;

  static Incremental4DVarConfig from_da_config(const config::DaConfig& cfg);
};

/// 主控类
class Incremental4DVar {
 public:
  Incremental4DVar(const grid::Grid& g, const dyn::ReferenceState& ref,
                   const config::ModelConfig& model_cfg,
                   const config::DaConfig& da_cfg,
                   Incremental4DVarConfig cfg = {});

  /// 注入非线性模式（必需）
  void set_forecast( ForecastFunction f) { forecast_ = std::move(f); }
  /// 注入观测算子（必需）
  void set_observation_operator(std::unique_ptr<obs::ObservationOperator> h);
  /// 注入 B 矩阵构建器（可选；默认用配置）
  void set_background_error(std::unique_ptr<ControlVariableTransform> b);

  /// 主入口
  AnalysisResult run(const obs::ObsSpace& observations, dyn::State& x_background);

  /// 只做一次外层的诊断（用于调试与单元测试）
  AnalysisResult run_single_outer(const obs::ObsSpace& observations,
                                  dyn::State& x_background);

  const Incremental4DVarConfig& config() const noexcept { return cfg_; }

 private:
  /// 生成创新向量 y - H(x_b)
  void compute_innovations(const dyn::State& xb, const obs::ObsSpace& obs,
                           std::vector<Real>& d);

  /// 计算分析场相对背景的诊断
  void fill_diagnostics(AnalysisResult& res, const dyn::State& xb,
                        const obs::ObsSpace& obs) const;

  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  config::ModelConfig model_cfg_;
  config::DaConfig da_cfg_;
  Incremental4DVarConfig cfg_;

  ForecastFunction forecast_;
  std::unique_ptr<obs::ObservationOperator> h_;
  std::unique_ptr<ControlVariableTransform> b_;
  std::unique_ptr<TangentLinearModel> tl_;
  std::unique_ptr<AdjointModel> ad_;
};

}  // namespace vibe::da

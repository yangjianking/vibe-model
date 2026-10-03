/// @file incremental_4dvar.cpp
/// @brief 增量 4D-Var 主控：外层轨迹/创新重算、内层 L-BFGS、诊断。
///
/// 算法（[V3] Courtier et al. 1994；[V10] Rabier et al. 2000）
/// ----------------------------------------------------------
/// @verbatim
///   x_b <- 背景
///   for k = 1..n_outer:                       （外层：非线性轨迹）
///       d_i <- y_i - H_i(x_b)                 （创新向量）
///       v_0 <- 上一外层结果（热启动）或 0
///       for j = 1..n_inner:                   （内层：降分辨率，resolution_factor）
///           J(v), grad J(v)                   （一次 TL 前向 + 一次 AD 反向）
///           v <- argmin J(v)                  （L-BFGS / CG）
///       x_b <- x_b + U v                      （增量更新）
///   x_a = x_{n_outer}
/// @endverbatim
///
/// 说明：内层在**降分辨率**网格上求解、外层回到全分辨率，是"增量"4D-Var 的核心
/// （[V3]）。本实现把 resolution_factor 记录到日志与结果描述中；由于
/// ControlVariableTransform/ObservationOperator 的冻结接口不暴露网格重采样，
/// 真正的降分辨率内层由 driver 另行提供（见 docs/design/06_4dvar.md 的"取舍"）。
///
/// 复杂度：一次外层 ≈ O(n_inner * (TL 前向 + AD 反向 + H 及其伴随))。
///
/// 文献：[V1][V2][V3][V10][V11][V15][V22][V25]。

#include "vibe/da/incremental_4dvar.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::da {

// ===========================================================================
// 描述
// ===========================================================================

std::string LoopStructure::describe() const {
  std::ostringstream os;
  os << "LoopStructure[outer=" << outer << " inner=" << inner
     << " resolution_factor=" << resolution_factor
     << " warm_start=" << (warm_start ? "yes" : "no") << "]";
  return os.str();
}

std::string AnalysisResult::describe() const {
  std::ostringstream os;
  os << "AnalysisResult[outer=" << outer_loops_done
     << " inner_total=" << inner_iterations_total << " J_start=" << cost_start.total
     << " (b=" << cost_start.background_term << ", o=" << cost_start.observation_term
     << ", p=" << cost_start.penalty_term << ")"
     << " J_end=" << cost_end.total << " (b=" << cost_end.background_term
     << ", o=" << cost_end.observation_term << ", p=" << cost_end.penalty_term << ")"
     << " grad_norm_end=" << cost_end.gradient_norm
     << " rms(O-B)=" << observation_minus_background_rms
     << " rms(O-A)=" << observation_minus_analysis_rms
     << " rms(B-A)=" << background_minus_analysis_rms << " chi2=" << chi_square
     << " history=" << history.size() << "]";
  return os.str();
}

// ===========================================================================
// 配置
// ===========================================================================

Incremental4DVarConfig Incremental4DVarConfig::from_da_config(const config::DaConfig& cfg) {
  Incremental4DVarConfig c;
  // 循环结构
  c.loops.outer = std::max(1, cfg.outer_loops);
  c.loops.inner = std::max(1, cfg.inner_loops);
  c.loops.resolution_factor = cfg.inner_resolution_factor;
  c.loops.warm_start = true;

  // 极小化
  c.minimizer.method = cfg.minimizer.method;
  c.minimizer.max_iterations = cfg.minimizer.max_iterations;
  c.minimizer.gradient_tolerance = cfg.minimizer.gradient_tolerance;
  c.minimizer.cost_tolerance = cfg.minimizer.cost_tolerance;
  c.minimizer.memory = cfg.minimizer.memory;
  c.minimizer.initial_step = cfg.minimizer.initial_step;
  c.minimizer.use_preconditioner = cfg.minimizer.preconditioned;

  // 控制变量 / B
  c.control.balance = balance_from_string(cfg.background.balance);
  c.control.horizontal_length_scale = cfg.background.horizontal_length_scale;
  c.control.vertical_length_scale = cfg.background.vertical_length_scale;
  c.control.diffusion_order = cfg.background.diffusion_order;
  c.control.n_vertical_modes = std::max(1, cfg.background.n_eofs);
  c.control.hybrid_weight = cfg.background.hybrid_weight;
  c.control.use_ensemble_component =
      (cfg.background.method == "ensemble" || cfg.background.method == "hybrid");

  // 观测项
  c.observation.active = true;
  c.observation.weight = Real(1);
  c.observation.use_variational_bias =
      (cfg.bias_correction == "variational_1d" || cfg.bias_correction == "varbc");

  // 惩罚项
  c.penalty.digital_filter = cfg.cost.use_digital_filter;
  c.penalty.digital_filter_weight = cfg.cost.digital_filter_weight;
  c.penalty.magnitude_penalty = cfg.cost.use_penalty;
  c.penalty.magnitude_weight = cfg.cost.penalty_weight;
  c.penalty.smoothness = false;
  c.penalty.smoothness_weight = Real(0);

  c.use_weak_constraint = cfg.cost.use_weak_constraint;
  c.check_adjoint_on_start = cfg.check_adjoint;
  c.physics = PhysicsLinearization::Adiabatic;
  c.checkpoint = CheckpointStrategy::StoreAll;
  return c;
}

// ===========================================================================
// Incremental4DVar
// ===========================================================================

Incremental4DVar::Incremental4DVar(const grid::Grid& g, const dyn::ReferenceState& ref,
                                   const config::ModelConfig& model_cfg,
                                   const config::DaConfig& da_cfg,
                                   Incremental4DVarConfig cfg)
    : grid_(&g),
      ref_(&ref),
      model_cfg_(model_cfg),
      da_cfg_(da_cfg),
      cfg_(std::move(cfg)) {
  tl_ = std::make_unique<TangentLinearModel>(g, ref, model_cfg_, da_cfg_);
  ad_ = std::make_unique<AdjointModel>(g, ref, model_cfg_, da_cfg_);
  tl_->set_physics_linearization(cfg_.physics);
  ad_->set_physics_linearization(cfg_.physics);
  tl_->set_checkpointing(cfg_.checkpoint);
}

void Incremental4DVar::set_observation_operator(std::unique_ptr<obs::ObservationOperator> h) {
  VIBE_CHECK_MSG(h != nullptr, "set_observation_operator 收到空指针");
  h_ = std::move(h);
}

void Incremental4DVar::set_background_error(std::unique_ptr<ControlVariableTransform> b) {
  VIBE_CHECK_MSG(b != nullptr, "set_background_error 收到空指针");
  b_ = std::move(b);
}

void Incremental4DVar::compute_innovations(const dyn::State& xb, const obs::ObsSpace& obs,
                                           std::vector<Real>& d) {
  d.assign(obs.obs.size(), Real(0));
  if (h_ == nullptr || obs.obs.empty()) return;
  obs::ModelStateView xv;
  xv.state = &xb;
  xv.ref = ref_;
  xv.grid = grid_;
  xv.time = xb.time;
  std::vector<Real> y;
  h_->apply(xv, obs, y);
  VIBE_CHECK_MSG(y.size() == obs.obs.size(), "compute_innovations: 算子返回长度不一致");
  for (Size i = 0; i < obs.obs.size(); ++i) {
    d[i] = obs.obs[i].unbiased() - y[i];
  }
}

void Incremental4DVar::fill_diagnostics(AnalysisResult& res, const dyn::State& xb,
                                        const obs::ObsSpace& obs) const {
  if (h_ == nullptr || obs.obs.empty()) return;
  // 分析场处的 H(x)
  obs::ModelStateView xa;
  xa.state = &res.analysis;
  xa.ref = ref_;
  xa.grid = grid_;
  std::vector<Real> ya;
  h_->apply(xa, obs, ya);
  obs::ModelStateView xbg;
  xbg.state = &xb;
  xbg.ref = ref_;
  xbg.grid = grid_;
  std::vector<Real> yb;
  h_->apply(xbg, obs, yb);

  Real sa = Real(0), sb = Real(0), chi2 = Real(0);
  Size n = 0;
  for (Size i = 0; i < obs.obs.size(); ++i) {
    const auto& o = obs.obs[i];
    if (!o.usable()) continue;
    const Real oa = o.unbiased() - ya[i];
    const Real ob = o.unbiased() - yb[i];
    sa += oa * oa;
    sb += ob * ob;
    chi2 += ob * ob / std::max(o.sigma * o.sigma, Real(1e-30));
    ++n;
  }
  if (n > 0) {
    res.observation_minus_analysis_rms = std::sqrt(sa / static_cast<Real>(n));
    res.observation_minus_background_rms = std::sqrt(sb / static_cast<Real>(n));
    res.chi_square = chi2 / static_cast<Real>(n);
  }

  // B-A 的 RMS（按自由度数归一）
  dyn::State diff(*grid_);
  diff.axpy(Real(1), res.analysis, Real(0));
  diff.add_scaled(Real(-1), xb);
  Size dof = 0;
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto& f = diff.field(static_cast<dyn::Species>(sp));
    dof += static_cast<Size>(f.nx()) * static_cast<Size>(f.ny()) * static_cast<Size>(f.nz());
  }
  if (dof > 0) {
    res.background_minus_analysis_rms = diff.norm2(Real(0)) / std::sqrt(static_cast<Real>(dof));
  }
}

AnalysisResult Incremental4DVar::run_single_outer(const obs::ObsSpace& observations,
                                                  dyn::State& x_background) {
  AnalysisResult res;
  VIBE_CHECK_MSG(h_ != nullptr, "Incremental4DVar::run_single_outer 需要先注入观测算子");
  if (!b_) {
    b_ = make_control_variable_transform(*grid_, *ref_, da_cfg_, cfg_.control);
  }
  VIBE_CHECK_MSG(grid_ != nullptr && ref_ != nullptr, "Incremental4DVar 网格/参考态为空");

  if (cfg_.check_adjoint_on_start && !observations.obs.empty()) {
    obs::ModelStateView xv;
    xv.state = &x_background;
    xv.ref = ref_;
    xv.grid = grid_;
    const Real err = h_->check_adjoint(xv, observations, 1u);
    VIBE_INFO("[4dvar] 观测算子点积检验相对误差 = ", err);
  }

  const dyn::State xb_copy = x_background;

  // ---- 创新向量 d = y - H(x_b) ----
  std::vector<Real> d;
  compute_innovations(x_background, observations, d);

  // ---- 内层代价函数与极小化 ----
  CostFunction J(*b_, *tl_, *h_, observations, x_background, d);
  J.penalty() = cfg_.penalty;

  Vector v(b_->size(), Real(0));
  const CostStatistics stats0 = J.last_statistics();
  res.cost_start = stats0;
  (void)J.value(v);  // 记录初始 J 的分解
  res.cost_start = J.last_statistics();

  MinimizerOptions mo = cfg_.minimizer;
  mo.max_iterations = std::max(1, cfg_.loops.inner);
  // 控制变量已标准化（v ~ N(0, I)），H0 的最优选择是单位阵；
  // 因此默认不注入外部预条件（与 J(v) = 1/2 v^T v 的背景项一致）。
  std::unique_ptr<Minimizer> mini = make_minimizer(mo, nullptr);
  VIBE_INFO("[4dvar] 外层内层极小化: ", mini->name(),
            " 内层迭代上限=", mo.max_iterations,
            " 分辨率因子=", cfg_.loops.resolution_factor);
  const Real J_end = mini->minimize(J, v);
  VIBE_UNUSED(J_end);
  res.cost_end = J.last_statistics();
  res.history = mini->history();
  res.inner_iterations_total = mini->iterations();
  res.outer_loops_done = 1;

  // ---- 分析更新 x_a = x_b + U v ----
  dyn::State dx(*grid_);
  b_->to_state(v, dx);
  x_background.add_scaled(Real(1), dx);
  res.analysis = x_background;

  fill_diagnostics(res, xb_copy, observations);
  VIBE_INFO("[4dvar] ", res.describe());
  return res;
}

AnalysisResult Incremental4DVar::run(const obs::ObsSpace& observations,
                                     dyn::State& x_background) {
  VIBE_CHECK_MSG(h_ != nullptr, "Incremental4DVar::run 需要先注入观测算子");
  AnalysisResult total;
  total.analysis = x_background;
  const int n_outer = std::max(1, cfg_.loops.outer);
  for (int outer = 0; outer < n_outer; ++outer) {
    const AnalysisResult one = run_single_outer(observations, x_background);
    // 合并历史与计数
    for (auto rec : one.history) {
      rec.outer = outer + 1;
      total.history.push_back(rec);
    }
    total.inner_iterations_total += one.inner_iterations_total;
    total.outer_loops_done = outer + 1;
    total.analysis = x_background;
    total.cost_start = one.cost_start;
    total.cost_end = one.cost_end;
    total.observation_minus_background_rms = one.observation_minus_background_rms;
    total.observation_minus_analysis_rms = one.observation_minus_analysis_rms;
    total.background_minus_analysis_rms = one.background_minus_analysis_rms;
    total.chi_square = one.chi_square;
    VIBE_INFO("[4dvar] 外层 ", outer + 1, '/', n_outer,
              " 完成，J=", one.cost_end.total, " rms(O-A)=",
              one.observation_minus_analysis_rms);
    // 热启动：下一外层重新计算轨迹与创新向量（x_background 已更新）
  }
  return total;
}

}  // namespace vibe::da

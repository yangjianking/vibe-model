#pragma once
/// @file cost_function.hpp
/// @brief 增量 4D-Var 的代价函数与梯度。
///
/// 定义（[V3] Courtier et al. 1994）
/// --------------------------------
/// 以 v 为控制变量，dx = U v：
///
///     J(v) = 1/2 v^T v
///          + 1/2 sum_i ( H_i M_i U v - d_i )^T R^{-1} ( H_i M_i U v - d_i )
///          + J_c(v)
///
/// 其中 J_c 为弱约束惩罚项：
///   * 数字滤波惩罚（[V25]）：J_c = 1/2 * gamma * || DF(dx) ||^2
///   * 惩罚项（[V4]）：抑制过大的增量
///
/// 梯度
/// ----
///     grad J(v) = v + U^T sum_i M_i^T H_i^T R^{-1} ( H_i M_i U v - d_i ) + grad J_c
///
/// 实现要点：**一次梯度评估需要一次完整的切线性前向传播 + 一次伴随反向传播**，
/// 因此代价函数的评估次数直接决定了 4D-Var 的计算成本。本实现记录
/// 每次评估的分解（背景项、观测项、惩罚项），便于诊断。
///
/// 文献：[V1][V2][V3][V4][V10][V22][V25]。

#include <memory>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/da/control_vector.hpp"
#include "vibe/da/da_types.hpp"
#include "vibe/da/tangent_linear.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/obs/obs_operator.hpp"

namespace vibe::config { struct DaConfig; }

namespace vibe::da {

/// 观测项的配置
struct ObservationTermConfig {
  bool active = true;
  std::vector<obs::ObsType> types;   ///< 参与的观测类型
  Real weight = Real(1);
  bool use_variational_bias = true;
};

/// 弱约束惩罚项
struct PenaltyTerm {
  bool digital_filter = true;
  Real digital_filter_weight = Real(1e-5);
  bool magnitude_penalty = true;
  Real magnitude_weight = Real(1e-4);
  bool smoothness = false;
  Real smoothness_weight = Real(0);
};

/// 增量代价函数
class CostFunction {
 public:
  CostFunction(const ControlVariableTransform& b_transform,
               const TangentLinearModel& tl,
               const obs::ObservationOperator& h,
               const obs::ObsSpace& obs,
               const dyn::State& x_background,
               const std::vector<Real>& innovations);

  /// 计算 J(v)
  Real value(const Vector& v) const;

  /// 计算 grad J(v)
  void gradient(const Vector& v, Vector& g) const;

  /// 同时计算 J 与 grad（避免重复的前向/反向传播）
  Real value_and_gradient(const Vector& v, Vector& g) const;

  /// 上一次评估的分解统计
  const CostStatistics& last_statistics() const noexcept { return stats_; }

  /// 惩罚项配置
  PenaltyTerm& penalty() noexcept { return penalty_; }
  const PenaltyTerm& penalty() const noexcept { return penalty_; }

  /// 控制向量维度
  Size size() const noexcept { return b_->size(); }

  /// 评估次数（诊断用）
  int evaluations() const noexcept { return evaluations_; }
  void reset_counters() const { evaluations_ = 0; }

 private:
  /// 数字滤波惩罚
  Real digital_filter_penalty(const dyn::State& dx) const;
  /// 由 v 得到 dx
  void expand(const Vector& v, dyn::State& dx) const;

  const ControlVariableTransform* b_;
  const TangentLinearModel* tl_;
  const obs::ObservationOperator* h_;
  const obs::ObsSpace* obs_;
  const dyn::State* xb_;
  std::vector<Real> innovations_;
  PenaltyTerm penalty_;

  mutable CostStatistics stats_{};
  mutable int evaluations_ = 0;
};

}  // namespace vibe::da

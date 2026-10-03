#pragma once
/// @file da_types.hpp
/// @brief 变分同化的基础类型与统一记号。
///
/// 统一记号（[V14] Ide et al. 1997）
/// --------------------------------
///     x^b  背景（forecast）
///     x^a  分析
///     x^t  真值
///     y    观测
///     H    观测算子（非线性）
///     M    模式算子（非线性），切线性记 M'，伴随记 M'^T
///     B    背景误差协方差
///     R    观测误差协方差
///     Q    模式误差协方差
///
/// 强约束 4D-Var 的代价函数
/// ------------------------
///     J(x_0) = 1/2 || x_0 - x^b ||_B^2
///            + 1/2 sum_i || H_i M_{0->i}(x_0) - y_i ||_R^2
///
/// 增量形式（[V3] Courtier et al. 1994）
/// --------------------------------------
///     J(dx) = 1/2 dx^T B^{-1} dx
///           + 1/2 sum_i ( H_i M_{0->i} dx - d_i )^T R^{-1} ( ... )
///     d_i = y_i - H_i M_{0->i}(x^b)
/// 控制变量用 B = U U^T 的平方根 U 做预条件：dx = U v，
/// 于是 J(v) = 1/2 v^T v + 1/2 || H M U v - d ||_R^2，条件数大幅改善。
///
/// 文献：[V1][V2][V3][V4][V13][V14]。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/obs/observations.hpp"

namespace vibe::da {

/// 控制向量（一维连续存储）
using Vector = std::vector<Real>;

/// 稀疏/带状矩阵的通用最小接口（本模式中只用到矩阵-向量乘）
class LinearOperator {
 public:
  virtual ~LinearOperator() = default;
  virtual void apply(const Vector& x, Vector& y) const = 0;
  virtual Size size() const noexcept = 0;
  virtual const char* name() const noexcept = 0;
};

/// 对角矩阵（用于 R 与简单方差）
class DiagonalMatrix final : public LinearOperator {
 public:
  explicit DiagonalMatrix(Vector diag) : diag_(std::move(diag)) {}
  void apply(const Vector& x, Vector& y) const override;
  Size size() const noexcept override { return diag_.size(); }
  const char* name() const noexcept override { return "diagonal"; }
  const Vector& diag() const noexcept { return diag_; }

 private:
  Vector diag_;
};

/// 4D-Var 的循环结构
struct LoopStructure {
  int  outer = 2;              ///< 外层（非线性轨迹重算）
  int  inner = 50;             ///< 内层（切线性极小化迭代）
  Real resolution_factor = Real(2.0);  ///< 内层降分辨率因子
  bool warm_start = true;      ///< 内层是否沿用上一层的结果

  std::string describe() const;
};

/// 单次代价函数评估的统计
struct CostStatistics {
  Real background_term = Real(0);
  Real observation_term = Real(0);
  Real penalty_term = Real(0);
  Real total = Real(0);
  Real gradient_norm = Real(0);
  Size n_observations = 0;
  int  evaluations = 0;
};

/// 迭代记录（用于收敛曲线）
struct IterationRecord {
  int  outer = 0, inner = 0;
  Real cost = Real(0);
  Real gradient_norm = Real(0);
  Real step_length = Real(0);
  Real wall_time = Real(0);
};

/// 同化结果
struct AnalysisResult {
  dyn::State analysis;         ///< x^a
  CostStatistics cost_start;
  CostStatistics cost_end;
  std::vector<IterationRecord> history;
  Real observation_minus_background_rms = Real(0);
  Real observation_minus_analysis_rms = Real(0);
  Real background_minus_analysis_rms = Real(0);
  Real chi_square = Real(0);
  int  outer_loops_done = 0;
  int  inner_iterations_total = 0;

  std::string describe() const;
};

}  // namespace vibe::da

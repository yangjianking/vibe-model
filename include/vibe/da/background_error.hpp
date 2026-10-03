#pragma once
/// @file background_error.hpp
/// @brief 背景误差协方差 B 的估计与建模。
///
/// 三条技术路线（[V8][V9] 综述）
/// ---------------------------
///   1. **NMC 方法**（[V5] Parrish & Derber 1992）：
///         B ~ 0.5 E[(x^{T+24} - x^{T+12})(...)^T]
///      用同一时刻、不同预报时效的差作为误差代理；实现简单，是业务系统的经典做法。
///   2. **集合方法**（[V19][V20][V21]）：
///         B ~ 1/(N-1) sum (x_i - x_bar)(...)^T
///      可以给出流依赖（flow-dependent）的 B，但受集合规模限制。
///   3. **混合方法**（[V20][V23]）：
///         B = (1-w) B_static + w B_ensemble
///      兼顾静态 B 的满秩与集合 B 的流依赖。
///
/// 本模块提供：
///   * 从样本集合估计水平/垂直长度尺度与方差；`estimate_length_scales`
///   * NMC 样本的读取与统计；
///   * B 的诊断工具：`diagnose_ob_stats`（[V18] Desroziers 方法），
///     通过 O-B、O-A 的统计诊断 R、B、A 的合理性。
///
/// 文献：[V5][V8][V9][V12][V18][V19][V20][V23]。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/config/config.hpp"
#include "vibe/da/control_vector.hpp"
#include "vibe/da/da_types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::da {

/// 误差统计量
struct ErrorStatistics {
  std::vector<Real> variance;        ///< 各变量的方差
  Real horizontal_length_scale = Real(0);
  Real vertical_length_scale = Real(0);
  std::vector<Real> vertical_correlation;   ///< 垂直相关系数廓线
  int  samples = 0;
  std::string describe() const;
};

/// NMC 方法的误差估计
class NmcEstimator {
 public:
  /// 由 (x_{T+24} - x_{T+12}) 样本集估计
  static ErrorStatistics estimate(const std::vector<dyn::State>& samples,
                                  const grid::Grid& g);

  /// 从文件批量读取样本（调用 io 层；此处只声明）
  static std::vector<dyn::State> load_samples(const std::vector<std::string>& paths,
                                              const grid::Grid& g);
};

/// 集合估计
class EnsembleEstimator {
 public:
  static ErrorStatistics estimate(const std::vector<dyn::State>& members,
                                  const grid::Grid& g);
  /// 集合 B 的局地化（Schur 乘积，用 Gaussian 核）
  static void localize(std::vector<Real>& covariance, Real radius,
                       Real length_scale);
};

/// B 的诊断工具
struct DiagnosticRatios {
  Real ob_minus_b_rms = Real(0);
  Real ob_minus_a_rms = Real(0);
  Real b_minus_a_rms = Real(0);
  Real chi_square_per_obs = Real(0);
  Real gleit_ratio = Real(0);     ///< |O-A| / |O-B|，期望 ~ sqrt(0.5)
  std::string describe() const;
};

/// [V18] Desroziers 诊断：用 O-B、O-A 估计 R 与 B 的乘性偏差
DiagnosticRatios diagnose_ob_stats(const std::vector<Real>& omb,
                                   const std::vector<Real>& oma,
                                   const std::vector<Real>& bma,
                                   const std::vector<Real>& sigma);

/// B 矩阵构建器（把配置翻译为 ControlVariableTransform）
class BackgroundErrorBuilder {
 public:
  BackgroundErrorBuilder(const grid::Grid& g, const dyn::ReferenceState& ref,
                         const config::DaConfig& cfg);

  /// 设定统计量（若不设定则用配置中的默认值）
  void set_statistics(const ErrorStatistics& s) { stats_ = s; }

  /// 构建（可选地混入集合分量）
  std::unique_ptr<ControlVariableTransform> build() const;

  const ErrorStatistics& statistics() const noexcept { return stats_; }

 private:
  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  config::DaConfig cfg_;
  ErrorStatistics stats_{};
};

}  // namespace vibe::da

#pragma once
/// @file probabilistic.hpp
/// @brief 概率预报与集合预报检验：Brier、可靠性、CRPS、Talagrand 图、ROC/AUC、离散度。
///
/// 概率预报的评分必须是**严格正常评分**（strictly proper scoring rule，[E11]
/// Gneiting & Raftery 2007）：期望值在"预报概率 = 真实概率"处唯一取极小的评分，
/// 否则模式可以通过系统性偏报（hedging）获取虚假优势。本文件实现的
/// Brier 评分、CRPS、对数评分族均满足严格正常性（证明见 docs/design/09_verification.md）。
///
/// 公式
/// ----
///   事件指示 o in {0,1}，预报概率 p：
///     BS  = (1/n) sum_i (p_i - o_i)^2                                   [E3] Brier (1950)
///   可靠度-分辨度-不确定性分解（[E3][E17]，等概率箱 K 个，箱 k 内样本数 n_k，
///   预报概率均值 p_k，事件频率 o_k，气候频率 obar）：
///     BS  = REL - RES + UNC
///     REL = sum_k (n_k/n) (p_k - o_k)^2
///     RES = sum_k (n_k/n) (o_k - obar)^2
///     UNC = obar (1 - obar)
///   技巧评分（相对气候态参考预报 BS_ref = UNC）：
///     BSS = 1 - BS / UNC
///
///   集合 CRPS（[E12] Hersbach 2000）：
///     CRPS = int (F(x) - H(x - y))^2 dx
///          = (1/m) sum_i |x_i - y| - (1/(2 m^2)) sum_i sum_j |x_i - x_j|
///   对升序样本 x_(1) <= ... <= x_(m) 有恒等式
///     sum_i sum_j |x_i - x_j| = 2 sum_i (2i - m - 1) x_(i)      (i 从 1 计)
///   故 O(m log m) 可得；无偏（fair）版本把第二项改为 1/(m(m-1))（[E16]）。
///
///   Talagrand 秩直方图（[E9] Hamill 2001；[E10] Talagrand & Vautard 1997）：
///     秩 = 观测在 m 个成员中的位次（m+1 个箱）；理想平坦分布为 p_k = 1/(m+1)。
///     平坦度 flatness = 1 - (1/2) sum_k |p_k - 1/K| / (1 - 1/K) in [0,1]
///     秩偏差 bias     = sum_k p_k (k+0.5)/K - 1/2           in [-1/2, 1/2]
///
///   ROC / AUC（[E15] Mason & Graham 2002）：
///     AUC = P(p_pos > p_neg) + (1/2) P(p_pos = p_neg)
///     用分组秩统计量（Mann-Whitney U，组内并列按 0.5 计）精确计算。
///
/// 复杂度：Brier/可靠性 O(n)；CRPS O(m log m)；ROC/AUC O(n log n)；
///         秩直方图 O(n m log m)；离散度 O(n m)。
///
/// 文献：[E3][E9][E10][E11][E12][E15][E16][E17]；教材 [B9] 第 8-9 章。

#include <cstddef>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/verify/scores.hpp"

namespace vibe::verify {

// ---------------------------------------------------------------------------
// 可靠性分箱
// ---------------------------------------------------------------------------

/// 概率分箱策略。
enum class BinScheme {
  EqualWidth,      ///< 等宽：箱边界为 k/K（预报概率分布偏斜时样本数不均）
  EqualFrequency   ///< 等频：按分位数切分，每箱样本数尽量相等（[B9] 第 9 章）
};

inline const char* to_string(BinScheme s) noexcept {
  return s == BinScheme::EqualWidth ? "equal_width" : "equal_frequency";
}

/// 一个可靠性箱。
struct ReliabilityBin {
  Real p_lo = kNaN;    ///< 箱下边界（含）
  Real p_hi = kNaN;    ///< 箱上边界（含）
  Real p_mean = kNaN;  ///< 箱内预报概率均值
  Real o_mean = kNaN;  ///< 箱内事件频率（= 观测频率）
  std::size_t count = 0;  ///< 箱内样本数
};

/// 可靠性曲线；只返回非空箱（count > 0）。O(n log n)（等频需要排序）。
std::vector<ReliabilityBin> reliability_curve(const std::vector<Real>& p,
                                              const std::vector<Real>& o_event,
                                              int n_bins = 10,
                                              BinScheme scheme = BinScheme::EqualWidth);

/// Brier 评分分解结果（[E3][E17]）。
struct BrierDecomposition {
  Real brier = kNaN;            ///< BS
  Real reliability = kNaN;      ///< REL
  Real resolution = kNaN;       ///< RES
  Real uncertainty = kNaN;      ///< UNC
  Real brier_climatology = kNaN;///< 参考评分 BS_ref = UNC
  Real skill = kNaN;            ///< BSS = 1 - BS/UNC
  std::size_t n = 0;
};

/// Brier 评分及其可靠度-分辨度-不确定性分解。O(n)。
BrierDecomposition brier_decomposition(const std::vector<Real>& p,
                                       const std::vector<Real>& o_event,
                                       int n_bins = 10,
                                       BinScheme scheme = BinScheme::EqualWidth);

/// Brier 评分 BS = mean (p - o)^2；o 按 >= 0.5 二值化。O(n)。
Real brier_score(const std::vector<Real>& p, const std::vector<Real>& o);

/// Brier 技巧评分 BSS = 1 - BS/UNC；UNC = 0 时返回 kNaN。O(n)。
Real brier_skill_score(const std::vector<Real>& p, const std::vector<Real>& o);

// ---------------------------------------------------------------------------
// ROC / AUC
// ---------------------------------------------------------------------------

/// ROC 曲线（按阈值从高到低给出工作点）。
struct RocCurve {
  std::vector<Real> thresholds;  ///< 判别阈值（降序）
  std::vector<Real> pod;         ///< 命中率 H/(H+M)
  std::vector<Real> pofd;        ///< 空报率 FA/(FA+CN)
  Real auc = kNaN;               ///< 曲线下面积
  std::size_t n_pos = 0;
  std::size_t n_neg = 0;
};

/// 由概率预报构造 ROC 曲线；阈值取排序后相邻概率的中点，含并列合并。O(n log n)。
RocCurve roc_curve(const std::vector<Real>& p, const std::vector<Real>& o_event);

/// ROC 曲线下面积，精确含并列（AUC = P(p_pos > p_neg) + 0.5 P(p_pos = p_neg)）。O(n log n)。
Real roc_auc(const std::vector<Real>& p, const std::vector<Real>& o_event);

// ---------------------------------------------------------------------------
// CRPS
// ---------------------------------------------------------------------------

/// 集合 CRPS（标准有偏估计量，1/(2 m^2)）。
/// @param members 集合成员值（未排序亦可，内部排序）
/// @param obs     对应观测
/// 复杂度 O(m log m)。
Real crps_ensemble(const std::vector<Real>& members, Real obs);

/// 无偏（fair）CRPS：第二项改用 1/(m(m-1))，[E16] Ferro et al. (2008)。
/// m < 2 时退化为 |x_1 - obs|。O(m log m)。
Real crps_fair(const std::vector<Real>& members, Real obs);

/// 由预报 CDF 在给定网格上的取值计算 CRPS = int (F - H(x-y))^2 dx。
/// @param x    单调递增的评估网格
/// @param cdf  F(x_i)
/// @param obs  观测
/// 梯形法离散化，O(n)。
Real crps_from_cdf(const std::vector<Real>& x, const std::vector<Real>& cdf, Real obs);

/// 逐点 CRPS 的样本均值。O(sum m_i log m_i)。
Real mean_crps(const std::vector<std::vector<Real>>& members,
               const std::vector<Real>& obs);

// ---------------------------------------------------------------------------
// Talagrand 秩直方图
// ---------------------------------------------------------------------------

/// 秩直方图统计量（[E9][E10]）。
struct RankHistogram {
  std::size_t ensemble_size = 0;   ///< m
  std::size_t samples = 0;         ///< 参与统计的样本点数
  std::vector<Real> counts;        ///< 长度 m+1；并列时按 1/(#并列+1) 分数摊分
  Real flatness = kNaN;            ///< 平坦度 in [0,1]，1 = 完全平坦
  Real bias = kNaN;                ///< 秩偏差 in [-0.5,0.5]
  Real chi2 = kNaN;                ///< 卡方统计量（检验平坦性）
  Real dof = kNaN;                 ///< 自由度 = m
  Real p_value = kNaN;             ///< 卡方检验 p 值（不显著 => 集合可靠）

  /// 归一化频率 p_k = counts_k / sum counts。
  std::vector<Real> probabilities() const;
};

/// 计算秩直方图。成员可含 NaN（该点跳过）。
/// 并列（成员 == 观测）时把该观测以 1/(t+1) 的概率摊到 t+1 个相邻秩上，
/// 保持计数期望无偏（[E9] 第 3 节）。
RankHistogram rank_histogram(const std::vector<std::vector<Real>>& members,
                             const std::vector<Real>& obs);

/// 归一化秩直方图的平坦度 L1 度量，见文件头公式。O(K)。
Real rank_histogram_flatness(const std::vector<Real>& counts);

/// 归一化秩直方图的秩偏差，见文件头公式。O(K)。
Real rank_histogram_bias(const std::vector<Real>& counts);

/// 正则化上不完全 Gamma 函数 Q(a, x) = 1 - P(a, x)，用于卡方检验 p 值。
/// 级数 + 连分式（[B9] 附录；NR 风格），相对精度 ~1e-12。O(1) 次迭代。
Real gamma_q(Real a, Real x);

// ---------------------------------------------------------------------------
// 集合离散度
// ---------------------------------------------------------------------------

/// 单点集合的离散度：无偏样本标准差 sqrt( sum (x_i - xbar)^2 / (m-1) )。
Real ensemble_spread_members(const std::vector<Real>& members);

/// 多点平均集合离散度（跳过 NaN），即 Scores::spread。O(n m)。
Real ensemble_spread(const std::vector<std::vector<Real>>& members);

/// 集合平均的 RMSE（对观测），即 Scores::ensemble_mean_rmse。O(n m)。
Real ensemble_mean_rmse(const std::vector<std::vector<Real>>& members,
                        const std::vector<Real>& obs);

/// 离散度-技巧比 = spread / rmse；理想标定集合约为 1。rmse = 0 时返回 kNaN。
Real spread_skill_ratio(const std::vector<std::vector<Real>>& members,
                        const std::vector<Real>& obs);

/// 集合检验结果汇总。
struct EnsembleStats {
  Real spread = kNaN;
  Real rmse = kNaN;
  Real spread_skill_ratio = kNaN;
  Real bias = kNaN;
  Real crps = kNaN;
  std::size_t n = 0;
  /// 转换为通用 Scores（spread/spread_skill_ratio/crps/ensemble_mean_rmse/n）。
  Scores to_scores() const;
};

/// 一次性计算集合的离散度、集合平均 RMSE、离散度-技巧比、偏差与 CRPS。O(n m log m)。
EnsembleStats ensemble_scores(const std::vector<std::vector<Real>>& members,
                              const std::vector<Real>& obs);

// ---------------------------------------------------------------------------
// 批量接口（架构契约签名，冻结）
// ---------------------------------------------------------------------------

/// 概率型评分。o 按 >= 0.5 二值化为事件。
/// @return 填充 brier/brier_skill/roc_auc/brier_reliability/brier_resolution/
///         brier_uncertainty/n；其余字段为 kNaN
/// @throws DimensionError 长度不等
/// 复杂度：O(n log n)（AUC 排序），空间 O(n)。
Scores compute_probabilistic(const std::vector<Real>& p, const std::vector<Real>& o);

/// 概率型评分（可指定可靠性分箱数与分箱策略的扩展重载）。
Scores compute_probabilistic(const std::vector<Real>& p, const std::vector<Real>& o,
                             int n_bins, BinScheme scheme);

}  // namespace vibe::verify

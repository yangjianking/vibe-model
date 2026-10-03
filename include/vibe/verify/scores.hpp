#pragma once
/// @file scores.hpp
/// @brief 检验评分公共容器、确定性（连续型）评分与流式累加器。
///
/// 本文件是 \c vibe::verify 的**接口基线**（见 docs/design/00_architecture.md 第 9 节）。
/// 架构契约中列出的 \c Scores 字段与 \c compute_continuous 签名在此**原样冻结**；
/// 其余成员（\c ContingencyTable、增量累加器、辅助统计）为契约允许的**新增**扩展。
///
/// 方法论（[E1] Murphy 1993；[E2] Murphy & Winkler 1987；[B9] Wilks 2019 第 8 章）
/// --------------------------------------------------------------------------
/// 设第 i 对样本为 (f_i, o_i)，有效样本数 n：
///     ME   = (1/n) sum_i (f_i - o_i)                        "bias"（平均误差）
///     MAE  = (1/n) sum_i |f_i - o_i|
///     RMSE = sqrt( (1/n) sum_i (f_i - o_i)^2 )
///     r    = S_fo / sqrt(S_ff * S_oo),
///            S_xy = sum_i (x_i - xbar)(y_i - ybar)
///     ACC  = sum_i a_f,i a_o,i / sqrt( sum_i a_f,i^2 * sum_i a_o,i^2 ),
///            a_x,i = x_i - clim_i                （气候距平相关，[E1]）
///     sigma_e = sqrt( (1/(n-1)) sum_i (e_i - ebar)^2 ),  e_i = f_i - o_i
///
/// 缺失值约定与字段填充约定
/// ------------------------
///   1. 任何参与统计的量若为 NaN（\c std::isnan 为真），该**样本对整体跳过**；
///      有效对数记入 \c Scores::n。
///   2. 每个 \c compute_* 函数只填充本检验类型适用的字段，**其余字段一律置 0**
///      （driver 层会直接读取字段，必须保证没有未初始化值）。
///   3. 适用但当前样本下数学上无法定义的评分（如无事件时的 POD、缺气候态时的
///      ACC、两场全空时的 FSS）填 \c kNaN，以区分"未定义"与"0 分"（[E13]）。
///
/// 复杂度
/// ------
///   批量接口 O(n) 时间、O(1) 额外空间（除输入数组本身）；
///   增量累加器每样本 O(1) 时间、O(K) 空间（K 为概率分箱数，默认 10）。
///
/// 文献：[B9] Wilks (2019) 第 8 章；[B10] Jolliffe & Stephenson (2012)；
///       [E1] Murphy (1993)；[E2] Murphy & Winkler (1987)；[E13] Casati et al. (2008)。

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"

namespace vibe::verify {

// ---------------------------------------------------------------------------
// 检验类型枚举
// ---------------------------------------------------------------------------

/// 检验匹配类型：决定使用哪一组评分（[E2] 的预报-观测匹配框架）。
enum class MatchType {
  Continuous,     ///< 连续量（温度、风、位势高度）
  Categorical,    ///< 二值/分类事件（降水阈值、雾、雷暴）
  Probabilistic,  ///< 概率预报（集合概率、Brier/ROC）
  Spatial         ///< 空间/邻域（FSS、模糊检验）
};

/// 枚举 -> 稳定字符串（用于 CSV/JSON 输出，禁止依赖编译器内部名称）。
inline const char* to_string(MatchType m) noexcept {
  switch (m) {
    case MatchType::Continuous:    return "continuous";
    case MatchType::Categorical:   return "categorical";
    case MatchType::Probabilistic: return "probabilistic";
    case MatchType::Spatial:       return "spatial";
  }
  return "unknown";
}

/// 字符串 -> 枚举；非法取值抛出 \c ConfigError。
inline MatchType match_type_from_string(const std::string& s) {
  if (s == "continuous") return MatchType::Continuous;
  if (s == "categorical") return MatchType::Categorical;
  if (s == "probabilistic") return MatchType::Probabilistic;
  if (s == "spatial") return MatchType::Spatial;
  throw ConfigError("未知的检验类型: '" + s + "'（允许 continuous/categorical/probabilistic/spatial）");
}

// ---------------------------------------------------------------------------
// 数值约定
// ---------------------------------------------------------------------------

/// 缺省缺失值。
inline constexpr Real kNaN = std::numeric_limits<Real>::quiet_NaN();

/// 是否为缺失值（IEEE NaN 感知；不要用 -ffast-math 编译本模块）。
inline bool is_missing(Real x) noexcept { return std::isnan(x); }

/// 安全比值：分母非正（含 0）时返回 \c kNaN。用于 POD/FAR/ETS 等。
inline Real ratio_or_nan(Real num, Real den) noexcept {
  return den > Real(0) ? num / den : kNaN;
}

/// 把概率夹到 [eps, 1-eps]，供对数类评分（SEDI）使用。
inline Real clamp_probability(Real p, Real eps = Real(1e-6)) noexcept {
  return p < eps ? eps : (p > Real(1) - eps ? Real(1) - eps : p);
}

// ---------------------------------------------------------------------------
// Scores：检验评分容器（架构契约字段 + 扩展字段）
// ---------------------------------------------------------------------------

/// 一次检验输出的全部标量评分。
///
/// **架构冻结字段保持名称、类型与顺序不变**（前 19 个成员）。
/// 默认构造为零（driver 层可直接读取，不会遇到未初始化值）；
/// compute_* 只填充本检验类型适用的字段，其余保持 0；
/// 适用但当前样本下无法定义的评分填 NaN（见文件头"字段填充约定"）。
/// n 为该次检验的有效样本数。
struct Scores {
  // ---- 架构契约字段（docs/design/00_architecture.md 第 9 节，顺序冻结）----
  Real bias = Real(0);                ///< 平均误差 ME = mean(f) - mean(o)
  Real mae = Real(0);                 ///< 平均绝对误差
  Real rmse = Real(0);                ///< 均方根误差
  Real correlation = Real(0);         ///< Pearson 相关系数
  Real anomaly_correlation = Real(0); ///< 距平相关 ACC（需要气候态）
  Real pod = Real(0);                 ///< 命中率 H/(H+M)
  Real far = Real(0);                 ///< 空报率 FA/(H+FA)
  Real csi = Real(0);                 ///< 临界成功指数 H/(H+M+FA)
  Real ets = Real(0);                 ///< 公平技巧评分（含随机命中订正）
  Real frequency_bias = Real(0);      ///< 频率偏差 (H+FA)/(H+M)
  Real odds_ratio_skill = Real(0);    ///< 优势比技巧 ORSS
  Real brier = Real(0);               ///< Brier 评分
  Real brier_skill = Real(0);         ///< 相对气候态的 Brier 技巧评分
  Real crps = Real(0);                ///< 连续排序概率评分
  Real roc_auc = Real(0);             ///< ROC 曲线下面积
  Real fss = Real(0);                 ///< 邻域分数技巧评分
  Real spread = Real(0);              ///< 集合离散度（平均 ensemble std）
  Real spread_skill_ratio = Real(0);  ///< 离散度-技巧比
  std::size_t n = 0;                  ///< 有效样本数

  // ---- VIBE 扩展字段（基线之外的附加诊断量，可安全忽略）----
  Real mean_error = Real(0);            ///< 同 bias，显式命名
  Real std_error = Real(0);             ///< 误差标准差（n-1 无偏）
  Real mean_forecast = Real(0);         ///< mean(f)
  Real mean_observation = Real(0);      ///< mean(o)
  Real tss = Real(0);                   ///< Hanssen-Kuipers 技巧评分
  Real sedi = Real(0);                  ///< 对称极值依赖指数
  Real accuracy = Real(0);              ///< 正确率 (H+CN)/N
  Real brier_reliability = Real(0);     ///< Brier 分解：可靠度
  Real brier_resolution = Real(0);      ///< Brier 分解：分辨度
  Real brier_uncertainty = Real(0);     ///< Brier 分解：不确定性
  Real rank_histogram_flatness = Real(0);  ///< Talagrand 图平坦度 ∈ [0,1]
  Real rank_histogram_bias = Real(0);      ///< Talagrand 图秩偏差 ∈ [-0.5,0.5]
  Real ensemble_mean_rmse = Real(0);       ///< 集合平均的 RMSE
  Real ensemble_spread = Real(0);          ///< 平均集合离散度（同 spread）
};

/// 全 NaN 的评分容器（用于诊断/调试）。
Scores nan_scores(std::size_t n = 0) noexcept;

/// 全 0 的评分容器：\c compute_* 的起点，保证不适用字段为确定的 0。
Scores zero_scores(std::size_t n = 0) noexcept;

/// 把 \c Scores 展开为 (指标名, 数值) 列表，供 CSV/JSON 序列化与 Python 对照。
std::vector<std::pair<const char*, Real>> score_table(const Scores& s);

// ---------------------------------------------------------------------------
// 基础统计（对 NaN 感知）
// ---------------------------------------------------------------------------

/// 算术平均（跳过 NaN）；空序列或全 NaN 返回 kNaN。O(n)。
Real mean_value(const std::vector<Real>& x);

/// 无偏样本标准差 sqrt( sum (x-xbar)^2 / (n-1) )；n<2 返回 kNaN。O(n)。
Real stddev_value(const std::vector<Real>& x);

/// 均方根 sqrt( mean x^2 )（跳过 NaN）。O(n)。
Real root_mean_square(const std::vector<Real>& x);

/// Pearson 相关系数（见文件头公式）。
/// 约定：任一序列方差为 0（常数序列）时返回 0.0，而不是 NaN —— 与
/// [E13] 对"退化输入必须给出确定行为"的要求一致。O(n)。
Real pearson_correlation(const std::vector<Real>& f, const std::vector<Real>& o);

/// 距平相关 ACC，要求 clim 与 f/o 等长或长度为 1（常数气候态）。O(n)。
Real anomaly_correlation(const std::vector<Real>& f, const std::vector<Real>& o,
                         const std::vector<Real>& clim);

// ---------------------------------------------------------------------------
// 批量接口（架构契约签名，冻结；四个自由函数均在本头文件声明）
// ---------------------------------------------------------------------------

/// 连续型评分（实现见 src/verify/scores.cpp）。
///
/// @param f     预报序列（扁平，含 NaN 表示缺测）
/// @param o     观测序列（与 f 等长）
/// @param clim  可选气候态序列：等长（逐时气候态）或长度 1（常数），
///              为 nullptr 时 \c anomaly_correlation 为 NaN
/// @return      填充 bias/mae/rmse/correlation/anomaly_correlation/mean_error/
///              std_error/mean_forecast/mean_observation/n，其余字段为 0
/// @throws DimensionError 当 f 与 o 长度不等，或 clim 长度既非 0/1 也非 |f|
///
/// 复杂度：O(n) 时间，O(1) 额外空间。
Scores compute_continuous(const std::vector<Real>& f, const std::vector<Real>& o,
                          const std::vector<Real>* clim = nullptr);

/// 分类（事件）评分（实现见 src/verify/contingency.cpp）。
/// f >= threshold 且 o >= threshold 记为命中。
/// @return 填充 pod/far/csi/ets/frequency_bias/odds_ratio_skill/tss/sedi/accuracy/n，
///         其余字段为 0
/// @throws DimensionError 长度不等
/// 复杂度：O(n) 时间，O(1) 额外空间。
Scores compute_categorical(const std::vector<Real>& f, const std::vector<Real>& o,
                           Real threshold);

/// 概率型评分（实现见 src/verify/probabilistic.cpp）。o 按 >= 0.5 二值化为事件。
/// @return 填充 brier/brier_skill/roc_auc/brier_reliability/brier_resolution/
///         brier_uncertainty/n，其余字段为 0
/// @throws DimensionError 长度不等
/// 复杂度：O(n log n)，空间 O(n)。
Scores compute_probabilistic(const std::vector<Real>& p, const std::vector<Real>& o);

/// 邻域法（空间）评分（实现见 src/verify/spatial.cpp）。
/// @return 填充 fss/n，其余字段为 0
/// @throws DimensionError 尺寸不一致
/// 复杂度：O(nx*ny)（积分图）。
Scores compute_fractional_skill(const std::vector<Real>& f, const std::vector<Real>& o,
                                int nx, int ny, Real threshold,
                                int neighborhood_radius);

// ---------------------------------------------------------------------------
// 增量/流式累加器
// ---------------------------------------------------------------------------

/// 前向声明：完整定义在 contingency.hpp（避免与 Scores 相互包含）。
struct ContingencyTable;

/// 流式评分累加器：一次遍历即得到连续型 + 分类评分，内存 O(K)。
///
/// 数值方法
/// --------
///   1. 均值/方差/协方差使用 **Welford 在线更新**（[B11] Higham 第 1 章）：
///          n'   = n + 1
///          df   = f - mean_f,  do = o - mean_o
///          mean_f' = mean_f + df/n'
///          M2_f'   = M2_f + df*(f - mean_f')
///          C_fo'   = C_fo + df*(o - mean_o')
///      避免"大均值小方差"时的灾难性抵消；合并两批数据用 Chan 平行公式。
///   2. 概率评分使用**等宽直方图**（K 个箱）在线累加计数与事件数，
///      Brier 评分本身精确累加；AUC 为直方图（组内并列按 0.5 计）估计，
///      批量接口 \c compute_probabilistic 给出精确 AUC。
///
/// 复杂度：每样本 O(1) 时间；空间 O(K)。支持 \c merge 用于 MPI/OpenMP 归约。
class ScoreAccumulator {
 public:
  ScoreAccumulator() = default;

  /// @param threshold 分类检验的事件阈值（默认 0.5）
  /// @param n_bins    概率直方图箱数（默认 10）
  explicit ScoreAccumulator(Real threshold, std::size_t n_bins = 10);

  void set_threshold(Real t) noexcept { threshold_ = t; }
  void set_probability_bins(std::size_t k);
  Real threshold() const noexcept { return threshold_; }

  /// 加入一对连续样本；同时按 threshold_ 更新列联表计数。O(1)。
  void add(Real f, Real o);

  /// 加入一对连续样本并给出对应的气候态值（用于增量 ACC）。O(1)。
  void add(Real f, Real o, Real clim);

  /// 加入一对概率样本：p 为预报概率，o_event 为事件指示（0/1）。O(1)。
  void add_probability(Real p, Real o_event);

  /// 合并另一累加器（可重复调用，满足结合律交换律）。O(K)。
  void merge(const ScoreAccumulator& other);

  /// 清空全部累加量，保留配置。O(K)。
  void reset() noexcept;

  /// 已累加的连续/分类样本数。
  std::size_t count() const noexcept { return n_; }

  /// 已累加的概率样本数。
  std::size_t probability_count() const noexcept { return n_prob_; }

  /// 输出当前评分快照。
  Scores scores() const;

  /// 输出列联表（定义见 contingency.hpp）。
  ContingencyTable contingency() const;

  // ---- 在线统计工具（公开，供 free function / 并行归约 / Python 绑定复用）----

  /// 双变量在线更新（Welford / Chan）：见实现文件注释与 [B11]。O(1)。
  static void bivariate_update(std::size_t& n, Real& mean_f, Real& mean_o,
                               Real& m2_f, Real& m2_o, Real& c_fo,
                               Real f, Real o);

  /// 两个在线统计批次的合并（Chan 平行公式）。O(1)。
  static void bivariate_merge(std::size_t& n, Real& mean_f, Real& mean_o,
                              Real& m2_f, Real& m2_o, Real& c_fo,
                              std::size_t n2, Real mean_f2, Real mean_o2,
                              Real m2_f2, Real m2_o2, Real c_fo2);

 private:

  // 连续/分类
  std::size_t n_ = 0;
  Real mean_f_ = Real(0), mean_o_ = Real(0);
  Real m2_f_ = Real(0), m2_o_ = Real(0), co_fo_ = Real(0);
  Real sum_err_ = Real(0), sum_abs_err_ = Real(0), sum_sq_err_ = Real(0);

  // 距平
  std::size_t n_anom_ = 0;
  Real mean_af_ = Real(0), mean_ao_ = Real(0);
  Real m2_af_ = Real(0), m2_ao_ = Real(0), co_afao_ = Real(0);

  // 列联表计数
  Real threshold_ = Real(0.5);
  std::size_t hits_ = 0, misses_ = 0, false_alarms_ = 0, correct_negatives_ = 0;

  // 概率（Brier 精确 + 直方图）
  std::size_t n_prob_ = 0;
  Real sum_brier_ = Real(0);
  Real sum_event_ = Real(0);
  std::vector<std::size_t> prob_bin_count_;
  std::vector<Real> prob_bin_event_;
  std::vector<Real> prob_bin_sum_p_;
};

}  // namespace vibe::verify

#pragma once
/// @file contingency.hpp
/// @brief 2x2 列联表与分类（事件）检验评分。
///
/// 检验框架见 [E2] Murphy & Winkler (1987)、[B10] Jolliffe & Stephenson (2012) 第 3 章。
///
/// 列联表（阈值化后）
/// ------------------
///               观测=1        观测=0
///   预报=1       H (hits)      FA (false alarms)
///   预报=0       M (misses)    CN (correct negatives)
///
///     N = H + M + FA + CN
///
/// 评分定义（全部以列联表元素表达）
/// -------------------------------
///     frequency_bias = (H + FA) / (H + M)                          [B9] 式 (8.14)
///     POD  (命中率)  = H / (H + M)                                  [E19]
///     POFD (空报率)  = FA / (FA + CN)
///     FAR  (空报比)  = FA / (H + FA)                                [E6]
///     CSI  (TS)      = H / (H + M + FA)                             [E6] Schaefer (1990)
///     accuracy       = (H + CN) / N
///     TSS = POD - POFD = H/(H+M) - FA/(FA+CN)                       [E5] Hanssen-Kuipers
///     H_random = (H+M)(H+FA)/N
///     ETS = (H - H_random) / (H + M + FA - H_random)                [E4] Gandin & Murphy
///     OR = (H*CN)/(M*FA),  ORSS = (H*CN - M*FA)/(H*CN + M*FA)       [B10] 第 3 章
///     SEDI = [ln F - ln H + ln(1-H) - ln(1-F)]
///            / [ln F + ln H + ln(1-H) + ln(1-F)],  F = POFD         [E13][B10]
///
/// 边界与退化处理
/// --------------
///   * 分母为 0（例如无事件、无空报）时返回 \c kNaN，绝不返回伪 0；
///   * ETS 的分母 H+M+FA-H_random <= 0 时返回 kNaN；
///   * SEDI 中 H、F 先夹到 [1e-6, 1-1e-6] 再取对数，避免 ln(0)。
///
/// 复杂度：构造 O(n)，各评分 O(1)。
///
/// 文献：[E4] Gandin & Murphy (1992)；[E5] Hanssen & Kuipers (1965)；
///       [E6] Schaefer (1990)；[E13] Casati et al. (2008)；[B9][B10][E19]。

#include <cmath>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/verify/scores.hpp"

namespace vibe::verify {

/// 2x2 列联表；元素为实数以便支持分数计数（面积加权检验、并列概率分箱）。
struct ContingencyTable {
  Real hits = Real(0);
  Real misses = Real(0);
  Real false_alarms = Real(0);
  Real correct_negatives = Real(0);

  /// N = H + M + FA + CN。
  Real total() const noexcept { return hits + misses + false_alarms + correct_negatives; }

  /// 预报事件点数 H + FA。
  Real forecast_events() const noexcept { return hits + false_alarms; }

  /// 观测事件点数 H + M。
  Real observed_events() const noexcept { return hits + misses; }

  /// H + CN。
  Real correct() const noexcept { return hits + correct_negatives; }

  /// 频率偏差 (H+FA)/(H+M)；[B9] eq. (8.14)。
  Real frequency_bias() const noexcept {
    return ratio_or_nan(forecast_events(), observed_events());
  }

  /// 命中率 POD；[E19]。
  Real pod() const noexcept { return ratio_or_nan(hits, observed_events()); }

  /// 空报率 POFD = FA/(FA+CN)。
  Real pofd() const noexcept {
    return ratio_or_nan(false_alarms, false_alarms + correct_negatives);
  }

  /// 空报比 FAR = FA/(H+FA)；[E6]。
  Real far() const noexcept { return ratio_or_nan(false_alarms, forecast_events()); }

  /// 临界成功指数 CSI（= TS）；[E6]。
  Real csi() const noexcept {
    return ratio_or_nan(hits, hits + misses + false_alarms);
  }

  /// 随机命中数 H_random = (H+M)(H+FA)/N。
  Real random_hits() const noexcept {
    const Real N = total();
    return N > Real(0) ? observed_events() * forecast_events() / N : kNaN;
  }

  /// 公平技巧评分 ETS（含随机命中订正）；[E4]。
  Real ets() const noexcept {
    const Real hr = random_hits();
    if (is_missing(hr)) return kNaN;
    const Real den = hits + misses + false_alarms - hr;
    return den > Real(0) ? (hits - hr) / den : kNaN;
  }

  /// Hanssen-Kuipers 技巧评分 TSS = POD - POFD；[E5]。
  Real tss() const noexcept {
    const Real p = pod();
    const Real f = pofd();
    return (is_missing(p) || is_missing(f)) ? kNaN : p - f;
  }

  /// 优势比技巧评分 ORSS = (H*CN - M*FA)/(H*CN + M*FA)；[B10]。
  Real orss() const noexcept {
    const Real a = hits * correct_negatives;
    const Real b = misses * false_alarms;
    return ratio_or_nan(a - b, a + b);
  }

  /// 对称极值依赖指数 SEDI；[E13] 第 3 节、[B10] 第 3 章。
  ///
  /// 极值事件（大降水、强风）的 POD 与 POFD 会同时趋于 0，传统评分失去分辨力；
  /// SEDI 通过双重对数变换把"同时趋于 0"的行为映射为有限值：
  ///     SEDI in [-1, 1]，完美预报 = +1，随机预报 = 0（当 POD = POFD）。
  Real sedi(Real eps = Real(1e-6)) const noexcept {
    const Real H = clamp_probability(pod(), eps);
    const Real F = clamp_probability(pofd(), eps);
    if (is_missing(pod()) || is_missing(pofd())) return kNaN;
    const Real lhf = std::log(F), lhh = std::log(H);
    const Real l1mh = std::log(Real(1) - H), l1mf = std::log(Real(1) - F);
    const Real den = lhf + lhh + l1mh + l1mf;
    const Real num = lhf - lhh + l1mh - l1mf;
    return den != Real(0) ? num / den : kNaN;
  }

  /// 正确率 (H+CN)/N。
  Real accuracy() const noexcept { return ratio_or_nan(correct(), total()); }

  /// 由计数直接构造。
  static ContingencyTable from_counts(Real h, Real m, Real fa, Real cn) noexcept {
    ContingencyTable t;
    t.hits = h;
    t.misses = m;
    t.false_alarms = fa;
    t.correct_negatives = cn;
    return t;
  }

  /// 由 0/1 二值序列构造。两序列等长；NaN 样本对跳过；其余按 !=0 视为事件。
  static ContingencyTable from_binary(const std::vector<int>& f, const std::vector<int>& o);

  /// 由连续序列 + 事件阈值构造（f >= threshold 视为事件）。
  static ContingencyTable from_arrays(const std::vector<Real>& f,
                                      const std::vector<Real>& o, Real threshold);

  /// 人类可读摘要（调试与日志用）。
  std::string to_string() const;
};

/// 分类（事件）评分。
///
/// @param f         预报序列
/// @param o         观测序列
/// @param threshold 事件阈值：f >= threshold 且 o >= threshold 记为命中
/// @return          填充 pod/far/csi/ets/frequency_bias/odds_ratio_skill/tss/sedi/accuracy/n，
///                  连续型与概率型字段保持 kNaN
/// @throws DimensionError 长度不等
///
/// 复杂度：O(n) 时间，O(1) 空间。
Scores compute_categorical(const std::vector<Real>& f, const std::vector<Real>& o,
                           Real threshold);

}  // namespace vibe::verify

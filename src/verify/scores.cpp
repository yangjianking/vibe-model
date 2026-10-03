#include "vibe/verify/scores.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "vibe/verify/contingency.hpp"

namespace vibe::verify {
namespace {

/// 把 Scores 中与连续型检验无关的字段置 0（契约：不适用字段不得留未初始化值）。
void zero_non_continuous(Scores& s) noexcept {
  s.pod = Real(0);
  s.far = Real(0);
  s.csi = Real(0);
  s.ets = Real(0);
  s.frequency_bias = Real(0);
  s.odds_ratio_skill = Real(0);
  s.brier = Real(0);
  s.brier_skill = Real(0);
  s.crps = Real(0);
  s.roc_auc = Real(0);
  s.fss = Real(0);
  s.spread = Real(0);
  s.spread_skill_ratio = Real(0);
  s.tss = Real(0);
  s.sedi = Real(0);
  s.accuracy = Real(0);
  s.brier_reliability = Real(0);
  s.brier_resolution = Real(0);
  s.brier_uncertainty = Real(0);
  s.rank_histogram_flatness = Real(0);
  s.rank_histogram_bias = Real(0);
  s.ensemble_mean_rmse = Real(0);
  s.ensemble_spread = Real(0);
}

/// 由等宽概率直方图计算 Brier 的可靠度/分辨度/不确定性分解。
/// 公式见 probabilistic.hpp 文件头（[E3][E17]）。
void binned_brier_decomposition(const std::vector<std::size_t>& count,
                                const std::vector<Real>& events,
                                const std::vector<Real>& sum_p,
                                Real& reliability, Real& resolution,
                                Real& uncertainty, Real& obar) {
  Real total = Real(0), total_events = Real(0);
  for (std::size_t k = 0; k < count.size(); ++k) {
    total += static_cast<Real>(count[k]);
    total_events += events[k];
  }
  if (total <= Real(0)) {
    reliability = resolution = uncertainty = obar = kNaN;
    return;
  }
  obar = total_events / total;
  Real rel = Real(0), res = Real(0);
  for (std::size_t k = 0; k < count.size(); ++k) {
    if (count[k] == 0) continue;
    const Real w = static_cast<Real>(count[k]) / total;
    const Real p_k = sum_p[k] / static_cast<Real>(count[k]);
    const Real o_k = events[k] / static_cast<Real>(count[k]);
    rel += w * (p_k - o_k) * (p_k - o_k);
    res += w * (o_k - obar) * (o_k - obar);
  }
  reliability = rel;
  resolution = res;
  uncertainty = obar * (Real(1) - obar);
}

/// 由概率直方图（组内按并列处理）计算 AUC。
/// AUC = P(p_pos > p_neg) + 0.5 P(p_pos = p_neg)；组内贡献 = neg_k*cum_pos_high
/// + 0.5*pos_k*neg_k。见 probabilistic.hpp 文件头（[E15]）。
Real binned_auc(const std::vector<std::size_t>& count,
                const std::vector<Real>& events) {
  Real n_pos = Real(0), n_neg = Real(0);
  for (std::size_t k = 0; k < count.size(); ++k) {
    n_pos += events[k];
    n_neg += static_cast<Real>(count[k]) - events[k];
  }
  if (n_pos <= Real(0) || n_neg <= Real(0)) return kNaN;
  Real auc_num = Real(0), cum_pos = Real(0);
  for (std::size_t idx = count.size(); idx-- > 0;) {
    const Real pos = events[idx];
    const Real neg = static_cast<Real>(count[idx]) - pos;
    auc_num += neg * cum_pos + Real(0.5) * pos * neg;
    cum_pos += pos;
  }
  return auc_num / (n_pos * n_neg);
}

}  // namespace

// ---------------------------------------------------------------------------
// 容器工具
// ---------------------------------------------------------------------------

Scores nan_scores(std::size_t n) noexcept {
  Scores s;
  s.bias = s.mae = s.rmse = s.correlation = s.anomaly_correlation = kNaN;
  s.pod = s.far = s.csi = s.ets = s.frequency_bias = s.odds_ratio_skill = kNaN;
  s.brier = s.brier_skill = s.crps = s.roc_auc = kNaN;
  s.fss = s.spread = s.spread_skill_ratio = kNaN;
  s.mean_error = s.std_error = s.mean_forecast = s.mean_observation = kNaN;
  s.tss = s.sedi = s.accuracy = kNaN;
  s.brier_reliability = s.brier_resolution = s.brier_uncertainty = kNaN;
  s.rank_histogram_flatness = s.rank_histogram_bias = kNaN;
  s.ensemble_mean_rmse = s.ensemble_spread = kNaN;
  s.n = n;
  return s;
}

// 全 0 容器：compute_* 的统一起点，保证返回对象不含未初始化字段。O(1)。
Scores zero_scores(std::size_t n) noexcept {
  Scores s;
  s.bias = s.mae = s.rmse = s.correlation = s.anomaly_correlation = Real(0);
  s.pod = s.far = s.csi = s.ets = s.frequency_bias = s.odds_ratio_skill = Real(0);
  s.brier = s.brier_skill = s.crps = s.roc_auc = Real(0);
  s.fss = s.spread = s.spread_skill_ratio = Real(0);
  s.mean_error = s.std_error = s.mean_forecast = s.mean_observation = Real(0);
  s.tss = s.sedi = s.accuracy = Real(0);
  s.brier_reliability = s.brier_resolution = s.brier_uncertainty = Real(0);
  s.rank_histogram_flatness = s.rank_histogram_bias = Real(0);
  s.ensemble_mean_rmse = s.ensemble_spread = Real(0);
  s.n = n;
  return s;
}

std::vector<std::pair<const char*, Real>> score_table(const Scores& s) {
  return {
      {"bias", s.bias},
      {"mae", s.mae},
      {"rmse", s.rmse},
      {"correlation", s.correlation},
      {"anomaly_correlation", s.anomaly_correlation},
      {"pod", s.pod},
      {"far", s.far},
      {"csi", s.csi},
      {"ets", s.ets},
      {"frequency_bias", s.frequency_bias},
      {"odds_ratio_skill", s.odds_ratio_skill},
      {"brier", s.brier},
      {"brier_skill", s.brier_skill},
      {"crps", s.crps},
      {"roc_auc", s.roc_auc},
      {"fss", s.fss},
      {"spread", s.spread},
      {"spread_skill_ratio", s.spread_skill_ratio},
      {"mean_error", s.mean_error},
      {"std_error", s.std_error},
      {"mean_forecast", s.mean_forecast},
      {"mean_observation", s.mean_observation},
      {"tss", s.tss},
      {"sedi", s.sedi},
      {"accuracy", s.accuracy},
      {"brier_reliability", s.brier_reliability},
      {"brier_resolution", s.brier_resolution},
      {"brier_uncertainty", s.brier_uncertainty},
      {"rank_histogram_flatness", s.rank_histogram_flatness},
      {"rank_histogram_bias", s.rank_histogram_bias},
      {"ensemble_mean_rmse", s.ensemble_mean_rmse},
      {"ensemble_spread", s.ensemble_spread},
      {"n", static_cast<Real>(s.n)},
  };
}

// ---------------------------------------------------------------------------
// 基础统计
// ---------------------------------------------------------------------------

// 算术平均；跳过 NaN。复杂度 O(n)。
Real mean_value(const std::vector<Real>& x) {
  Real sum = Real(0);
  std::size_t n = 0;
  for (const Real v : x) {
    if (is_missing(v)) continue;
    sum += v;
    ++n;
  }
  return n > 0 ? sum / static_cast<Real>(n) : kNaN;
}

// 无偏样本标准差 sqrt( sum (x-xbar)^2 / (n-1) )，单遍 Welford，O(n)。
Real stddev_value(const std::vector<Real>& x) {
  std::size_t n = 0;
  Real mean = Real(0), m2 = Real(0);
  for (const Real v : x) {
    if (is_missing(v)) continue;
    const std::size_t n1 = n + 1;
    const Real d = v - mean;
    mean += d / static_cast<Real>(n1);
    m2 += d * (v - mean);
    n = n1;
  }
  if (n < 2) return kNaN;
  return std::sqrt(m2 / static_cast<Real>(n - 1));
}

// 均方根 sqrt( mean x^2 )；跳过 NaN。O(n)。
Real root_mean_square(const std::vector<Real>& x) {
  Real sum = Real(0);
  std::size_t n = 0;
  for (const Real v : x) {
    if (is_missing(v)) continue;
    sum += v * v;
    ++n;
  }
  return n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : kNaN;
}

// Pearson 相关系数；任一序列方差为 0 时返回 0（约定，见头文件）。O(n)。
Real pearson_correlation(const std::vector<Real>& f, const std::vector<Real>& o) {
  if (f.size() != o.size()) {
    throw DimensionError("pearson_correlation: 预报与观测长度不等 (" +
                         std::to_string(f.size()) + " vs " + std::to_string(o.size()) + ")");
  }
  std::size_t n = 0;
  Real mean_f = Real(0), mean_o = Real(0);
  Real m2_f = Real(0), m2_o = Real(0), c_fo = Real(0);
  for (std::size_t i = 0; i < f.size(); ++i) {
    if (is_missing(f[i]) || is_missing(o[i])) continue;
    ScoreAccumulator::bivariate_update(n, mean_f, mean_o, m2_f, m2_o, c_fo, f[i], o[i]);
  }
  if (n < 2) return kNaN;
  if (m2_f <= Real(0) || m2_o <= Real(0)) return Real(0);
  const Real r = c_fo / std::sqrt(m2_f * m2_o);
  return r < Real(-1) ? Real(-1) : (r > Real(1) ? Real(1) : r);
}

// 距平相关 ACC；clim 与 f/o 等长或长度为 1（常数气候态）。O(n)。
Real anomaly_correlation(const std::vector<Real>& f, const std::vector<Real>& o,
                         const std::vector<Real>& clim) {
  if (f.size() != o.size()) {
    throw DimensionError("anomaly_correlation: 预报与观测长度不等");
  }
  if (clim.empty()) {
    throw DimensionError("anomaly_correlation: 气候态序列为空（应提供等长或长度 1 的序列）");
  }
  if (clim.size() != 1 && clim.size() != f.size()) {
    throw DimensionError("anomaly_correlation: 气候态长度必须为 1 或与样本等长");
  }
  std::size_t n = 0;
  Real mean_af = Real(0), mean_ao = Real(0);
  Real m2_af = Real(0), m2_ao = Real(0), c_afao = Real(0);
  const Real clim_const = clim[0];
  for (std::size_t i = 0; i < f.size(); ++i) {
    const Real c = clim.size() == 1 ? clim_const : clim[i];
    if (is_missing(f[i]) || is_missing(o[i]) || is_missing(c)) continue;
    ScoreAccumulator::bivariate_update(n, mean_af, mean_ao, m2_af, m2_ao, c_afao,
                                       f[i] - c, o[i] - c);
  }
  if (n < 2) return kNaN;
  if (m2_af <= Real(0) || m2_ao <= Real(0)) return Real(0);
  const Real r = c_afao / std::sqrt(m2_af * m2_ao);
  return r < Real(-1) ? Real(-1) : (r > Real(1) ? Real(1) : r);
}

// ---------------------------------------------------------------------------
// 批量连续评分
// ---------------------------------------------------------------------------

// 见 scores.hpp：O(n) 时间、O(1) 额外空间；NaN 样本对整体跳过。
Scores compute_continuous(const std::vector<Real>& f, const std::vector<Real>& o,
                          const std::vector<Real>* clim) {
  if (f.size() != o.size()) {
    throw DimensionError("compute_continuous: 预报与观测长度不等 (" +
                         std::to_string(f.size()) + " vs " + std::to_string(o.size()) + ")");
  }
  const bool have_clim = (clim != nullptr) && !clim->empty();
  if (have_clim && clim->size() != 1 && clim->size() != f.size()) {
    throw DimensionError("compute_continuous: 气候态长度必须为 1 或与样本等长");
  }
  ScoreAccumulator acc;
  const Real clim_const = have_clim ? (*clim)[0] : Real(0);
  for (std::size_t i = 0; i < f.size(); ++i) {
    if (is_missing(f[i]) || is_missing(o[i])) continue;
    if (have_clim) {
      const Real c = clim->size() == 1 ? clim_const : (*clim)[i];
      if (is_missing(c)) continue;
      acc.add(f[i], o[i], c);
    } else {
      acc.add(f[i], o[i]);
    }
  }
  Scores s = acc.scores();
  zero_non_continuous(s);
  // 连续型中"需要外部输入却缺失"的评分保持 NaN（区别于不适用字段的 0）。
  if (!have_clim) s.anomaly_correlation = kNaN;
  return s;
}

// ---------------------------------------------------------------------------
// ScoreAccumulator
// ---------------------------------------------------------------------------

// 双变量在线更新（Welford）：见头文件与 [B11] Higham (2002) 第 1 章。O(1)。
void ScoreAccumulator::bivariate_update(std::size_t& n, Real& mean_f, Real& mean_o,
                                        Real& m2_f, Real& m2_o, Real& c_fo,
                                        Real f, Real o) {
  const std::size_t n1 = n + 1;
  const Real df = f - mean_f;
  const Real d_o = o - mean_o;
  const Real inv_n1 = Real(1) / static_cast<Real>(n1);
  mean_f += df * inv_n1;
  mean_o += d_o * inv_n1;
  m2_f += df * (f - mean_f);
  m2_o += d_o * (o - mean_o);
  c_fo += df * (o - mean_o);
  n = n1;
}

// Chan 平行合并公式（[B11] 第 1.9 节）：O(1)。
void ScoreAccumulator::bivariate_merge(std::size_t& n, Real& mean_f, Real& mean_o,
                                       Real& m2_f, Real& m2_o, Real& c_fo,
                                       std::size_t n2, Real mean_f2, Real mean_o2,
                                       Real m2_f2, Real m2_o2, Real c_fo2) {
  if (n2 == 0) return;
  if (n == 0) {
    n = n2;
    mean_f = mean_f2;
    mean_o = mean_o2;
    m2_f = m2_f2;
    m2_o = m2_o2;
    c_fo = c_fo2;
    return;
  }
  const std::size_t nt = n + n2;
  const Real inv_nt = Real(1) / static_cast<Real>(nt);
  const Real d_f = mean_f2 - mean_f;
  const Real d_o = mean_o2 - mean_o;
  const Real w = static_cast<Real>(n) * static_cast<Real>(n2) * inv_nt;
  mean_f += d_f * static_cast<Real>(n2) * inv_nt;
  mean_o += d_o * static_cast<Real>(n2) * inv_nt;
  m2_f += m2_f2 + d_f * d_f * w;
  m2_o += m2_o2 + d_o * d_o * w;
  c_fo += c_fo2 + d_f * d_o * w;
  n = nt;
}

ScoreAccumulator::ScoreAccumulator(Real threshold, std::size_t n_bins)
    : threshold_(threshold) {
  set_probability_bins(n_bins);
}

void ScoreAccumulator::set_probability_bins(std::size_t k) {
  if (k == 0) k = 1;
  prob_bin_count_.assign(k, 0);
  prob_bin_event_.assign(k, Real(0));
  prob_bin_sum_p_.assign(k, Real(0));
}

// 加入一对样本：连续 + 分类计数。O(1)。
void ScoreAccumulator::add(Real f, Real o) {
  if (is_missing(f) || is_missing(o)) return;
  bivariate_update(n_, mean_f_, mean_o_, m2_f_, m2_o_, co_fo_, f, o);
  const Real e = f - o;
  sum_err_ += e;
  sum_abs_err_ += std::abs(e);
  sum_sq_err_ += e * e;

  const bool fe = f >= threshold_;
  const bool oe = o >= threshold_;
  if (fe && oe) {
    ++hits_;
  } else if (!fe && !oe) {
    ++correct_negatives_;
  } else if (fe && !oe) {
    ++false_alarms_;
  } else {
    ++misses_;
  }
}

// 加入一对样本 + 气候态（增量 ACC）。O(1)。
void ScoreAccumulator::add(Real f, Real o, Real clim) {
  if (is_missing(f) || is_missing(o) || is_missing(clim)) return;
  add(f, o);
  bivariate_update(n_anom_, mean_af_, mean_ao_, m2_af_, m2_ao_, co_afao_,
                   f - clim, o - clim);
}

// 加入一对概率样本：Brier 精确累加 + 直方图（可靠性/ROC）。O(1)。
void ScoreAccumulator::add_probability(Real p, Real o_event) {
  if (is_missing(p) || is_missing(o_event)) return;
  const Real oe = o_event >= Real(0.5) ? Real(1) : Real(0);
  const Real e = p - oe;
  sum_brier_ += e * e;
  sum_event_ += oe;
  if (prob_bin_count_.empty()) set_probability_bins(10);
  const std::size_t K = prob_bin_count_.size();
  std::size_t k = p <= Real(0) ? 0
                  : (p >= Real(1) ? K - 1
                                  : static_cast<std::size_t>(p * static_cast<Real>(K)));
  if (k >= K) k = K - 1;
  prob_bin_count_[k] += 1;
  prob_bin_event_[k] += oe;
  prob_bin_sum_p_[k] += p;
  ++n_prob_;
}

// 合并另一累加器（结合律/交换律，用于并行归约）。O(K)。
void ScoreAccumulator::merge(const ScoreAccumulator& other) {
  bivariate_merge(n_, mean_f_, mean_o_, m2_f_, m2_o_, co_fo_,
                  other.n_, other.mean_f_, other.mean_o_,
                  other.m2_f_, other.m2_o_, other.co_fo_);
  bivariate_merge(n_anom_, mean_af_, mean_ao_, m2_af_, m2_ao_, co_afao_,
                  other.n_anom_, other.mean_af_, other.mean_ao_,
                  other.m2_af_, other.m2_ao_, other.co_afao_);
  sum_err_ += other.sum_err_;
  sum_abs_err_ += other.sum_abs_err_;
  sum_sq_err_ += other.sum_sq_err_;
  hits_ += other.hits_;
  misses_ += other.misses_;
  false_alarms_ += other.false_alarms_;
  correct_negatives_ += other.correct_negatives_;

  if (n_prob_ == 0) {
    n_prob_ = other.n_prob_;
    sum_brier_ = other.sum_brier_;
    sum_event_ = other.sum_event_;
    if (prob_bin_count_.size() == other.prob_bin_count_.size()) {
      for (std::size_t k = 0; k < prob_bin_count_.size(); ++k) {
        prob_bin_count_[k] = other.prob_bin_count_[k];
        prob_bin_event_[k] = other.prob_bin_event_[k];
        prob_bin_sum_p_[k] = other.prob_bin_sum_p_[k];
      }
    }
  } else if (other.n_prob_ > 0) {
    n_prob_ += other.n_prob_;
    sum_brier_ += other.sum_brier_;
    sum_event_ += other.sum_event_;
    if (prob_bin_count_.size() == other.prob_bin_count_.size()) {
      for (std::size_t k = 0; k < prob_bin_count_.size(); ++k) {
        prob_bin_count_[k] += other.prob_bin_count_[k];
        prob_bin_event_[k] += other.prob_bin_event_[k];
        prob_bin_sum_p_[k] += other.prob_bin_sum_p_[k];
      }
    }
  }
}

void ScoreAccumulator::reset() noexcept {
  n_ = 0;
  mean_f_ = mean_o_ = Real(0);
  m2_f_ = m2_o_ = co_fo_ = Real(0);
  sum_err_ = sum_abs_err_ = sum_sq_err_ = Real(0);
  n_anom_ = 0;
  mean_af_ = mean_ao_ = Real(0);
  m2_af_ = m2_ao_ = co_afao_ = Real(0);
  hits_ = misses_ = false_alarms_ = correct_negatives_ = 0;
  n_prob_ = 0;
  sum_brier_ = sum_event_ = Real(0);
  for (std::size_t k = 0; k < prob_bin_count_.size(); ++k) {
    prob_bin_count_[k] = 0;
    prob_bin_event_[k] = Real(0);
    prob_bin_sum_p_[k] = Real(0);
  }
}

// 输出评分快照。连续/分类/概率三组一起给出；未累加组保持 NaN。O(K)。
Scores ScoreAccumulator::scores() const {
  Scores s = zero_scores(n_);
  if (n_ > 0) {
    const Real inv_n = Real(1) / static_cast<Real>(n_);
    s.mean_forecast = mean_f_;
    s.mean_observation = mean_o_;
    s.mean_error = sum_err_ * inv_n;
    s.bias = s.mean_error;
    s.mae = sum_abs_err_ * inv_n;
    s.rmse = std::sqrt(sum_sq_err_ * inv_n);
    if (n_ >= 2) {
      const Real var_e = (sum_sq_err_ - sum_err_ * s.mean_error) / static_cast<Real>(n_ - 1);
      s.std_error = var_e > Real(0) ? std::sqrt(var_e) : Real(0);
    }
    if (n_ >= 2 && m2_f_ > Real(0) && m2_o_ > Real(0)) {
      const Real r = co_fo_ / std::sqrt(m2_f_ * m2_o_);
      s.correlation = r < Real(-1) ? Real(-1) : (r > Real(1) ? Real(1) : r);
    } else {
      s.correlation = Real(0);  // 常数序列约定（见头文件）
    }
  }
  if (n_anom_ >= 2 && m2_af_ > Real(0) && m2_ao_ > Real(0)) {
    const Real r = co_afao_ / std::sqrt(m2_af_ * m2_ao_);
    s.anomaly_correlation = r < Real(-1) ? Real(-1) : (r > Real(1) ? Real(1) : r);
  } else if (n_anom_ >= 2) {
    s.anomaly_correlation = Real(0);
  }

  if (n_ > 0) {
    const ContingencyTable t = contingency();
    if (t.observed_events() > Real(0)) s.pod = t.pod();
    if (t.forecast_events() > Real(0)) s.far = t.far();
    if (t.hits + t.misses + t.false_alarms > Real(0)) s.csi = t.csi();
    s.ets = t.ets();
    if (t.observed_events() > Real(0)) s.frequency_bias = t.frequency_bias();
    s.odds_ratio_skill = t.orss();
    s.tss = t.tss();
    s.sedi = t.sedi();
    s.accuracy = t.accuracy();
  }

  if (n_prob_ > 0) {
    const Real inv_n = Real(1) / static_cast<Real>(n_prob_);
    s.brier = sum_brier_ * inv_n;
    Real rel = kNaN, res = kNaN, unc = kNaN, obar = kNaN;
    binned_brier_decomposition(prob_bin_count_, prob_bin_event_, prob_bin_sum_p_,
                               rel, res, unc, obar);
    s.brier_reliability = rel;
    s.brier_resolution = res;
    s.brier_uncertainty = unc;
    s.brier_skill = (unc > Real(0)) ? Real(1) - s.brier / unc : kNaN;
    s.roc_auc = binned_auc(prob_bin_count_, prob_bin_event_);
  }
  return s;
}

}  // namespace vibe::verify

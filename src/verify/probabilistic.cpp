#include "vibe/verify/probabilistic.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace vibe::verify {
namespace {

/// 事件指示：o >= 0.5 记为 1。
inline Real event_of(Real o) noexcept { return o >= Real(0.5) ? Real(1) : Real(0); }

/// 单个可靠性箱的聚合量。
/// 注意：Brier 需要 sum (p-o)^2，而 sum_p 与 sum_o 无法还原该量
/// （仅当箱内 o 全同才成立），因此额外累加 sum_p2 = sum p^2 与
/// sum_po = sum p*o；对二值 o 有 o^2 = o。
struct BinAgg {
  std::size_t count = 0;
  Real sum_p = Real(0);
  Real sum_o = Real(0);
  Real sum_p2 = Real(0);
  Real sum_po = Real(0);
  Real min_p = kNaN;
  Real max_p = kNaN;
};

/// 等宽 / 等频分箱的公共实现，返回 K 个箱的聚合量（含空箱）。O(n log n)（等频需要排序）。
std::vector<BinAgg> bin_probabilities(const std::vector<Real>& p,
                                      const std::vector<Real>& o_event,
                                      int n_bins, BinScheme scheme) {
  if (p.size() != o_event.size()) {
    throw DimensionError("bin_probabilities: 概率与事件序列长度不等 (" +
                         std::to_string(p.size()) + " vs " +
                         std::to_string(o_event.size()) + ")");
  }
  const int K = n_bins > 0 ? n_bins : 1;
  std::vector<BinAgg> bins(static_cast<std::size_t>(K));

  if (scheme == BinScheme::EqualWidth) {
    for (std::size_t i = 0; i < p.size(); ++i) {
      if (is_missing(p[i]) || is_missing(o_event[i])) continue;
      const Real pc = p[i] <= Real(0) ? Real(0) : (p[i] >= Real(1) ? Real(1) : p[i]);
      std::size_t k = static_cast<std::size_t>(pc * static_cast<Real>(K));
      if (k >= static_cast<std::size_t>(K)) k = static_cast<std::size_t>(K) - 1;
      const Real oe = event_of(o_event[i]);
      BinAgg& b = bins[k];
      b.count += 1;
      b.sum_p += p[i];
      b.sum_o += oe;
      b.sum_p2 += p[i] * p[i];
      b.sum_po += p[i] * oe;
      if (b.count == 1 || p[i] < b.min_p) b.min_p = p[i];
      if (b.count == 1 || p[i] > b.max_p) b.max_p = p[i];
    }
    return bins;
  }

  // 等频：按预报概率排序后按序号均分，保证每箱样本数尽量相等。
  std::vector<std::pair<Real, Real>> v;
  v.reserve(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (is_missing(p[i]) || is_missing(o_event[i])) continue;
    v.emplace_back(p[i], event_of(o_event[i]));
  }
  if (v.empty()) return bins;
  std::sort(v.begin(), v.end(),
            [](const std::pair<Real, Real>& a, const std::pair<Real, Real>& b) {
              return a.first < b.first;
            });
  const std::size_t n = v.size();
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t k = (i * static_cast<std::size_t>(K)) / n;
    if (k >= static_cast<std::size_t>(K)) k = static_cast<std::size_t>(K) - 1;
    BinAgg& b = bins[k];
    b.count += 1;
    b.sum_p += v[i].first;
    b.sum_o += v[i].second;
    b.sum_p2 += v[i].first * v[i].first;
    b.sum_po += v[i].first * v[i].second;
    if (b.count == 1 || v[i].first < b.min_p) b.min_p = v[i].first;
    if (b.count == 1 || v[i].first > b.max_p) b.max_p = v[i].first;
  }
  return bins;
}

/// 并列感知的秩统计（Mann-Whitney U）：见 probabilistic.hpp 文件头。O(n log n)。
Real auc_from_pairs(std::vector<std::pair<Real, Real>>& v) {
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end(),
            [](const std::pair<Real, Real>& a, const std::pair<Real, Real>& b) {
              return a.first < b.first;
            });
  Real n_pos = Real(0), n_neg = Real(0);
  for (const auto& e : v) {
    if (e.second >= Real(0.5)) n_pos += Real(1); else n_neg += Real(1);
  }
  if (n_pos <= Real(0) || n_neg <= Real(0)) return kNaN;
  Real auc_num = Real(0), cum_pos = Real(0);
  std::size_t i = v.size();
  while (i > 0) {
    const Real pv = v[i - 1].first;
    std::size_t j = i;
    Real pos = Real(0), neg = Real(0);
    while (j > 0 && v[j - 1].first == pv) {
      if (v[j - 1].second >= Real(0.5)) pos += Real(1); else neg += Real(1);
      --j;
    }
    auc_num += neg * cum_pos + Real(0.5) * pos * neg;
    cum_pos += pos;
    i = j;
  }
  return auc_num / (n_pos * n_neg);
}

}  // namespace

// ---------------------------------------------------------------------------
// 可靠性曲线与 Brier 分解
// ---------------------------------------------------------------------------

std::vector<ReliabilityBin> reliability_curve(const std::vector<Real>& p,
                                              const std::vector<Real>& o_event,
                                              int n_bins, BinScheme scheme) {
  const int K = n_bins > 0 ? n_bins : 1;
  const std::vector<BinAgg> bins = bin_probabilities(p, o_event, K, scheme);
  std::vector<ReliabilityBin> out;
  for (std::size_t k = 0; k < bins.size(); ++k) {
    if (bins[k].count == 0) continue;
    ReliabilityBin rb;
    if (scheme == BinScheme::EqualWidth) {
      rb.p_lo = static_cast<Real>(k) / static_cast<Real>(K);
      rb.p_hi = static_cast<Real>(k + 1) / static_cast<Real>(K);
    } else {
      rb.p_lo = bins[k].min_p;
      rb.p_hi = bins[k].max_p;
    }
    rb.p_mean = bins[k].sum_p / static_cast<Real>(bins[k].count);
    rb.o_mean = bins[k].sum_o / static_cast<Real>(bins[k].count);
    rb.count = bins[k].count;
    out.push_back(rb);
  }
  return out;
}

BrierDecomposition brier_decomposition(const std::vector<Real>& p,
                                       const std::vector<Real>& o_event,
                                       int n_bins, BinScheme scheme) {
  const int K = n_bins > 0 ? n_bins : 1;
  const std::vector<BinAgg> bins = bin_probabilities(p, o_event, K, scheme);
  BrierDecomposition d;
  Real total = Real(0), total_o = Real(0), brier = Real(0);
  for (const BinAgg& b : bins) {
    total += static_cast<Real>(b.count);
    total_o += b.sum_o;
    // sum_{i in bin} (p_i - o_i)^2 = sum_p2 - 2 sum_po + sum_o  （二值 o）
    brier += b.sum_p2 - Real(2) * b.sum_po + b.sum_o;
  }
  if (total <= Real(0)) return d;
  d.n = static_cast<std::size_t>(total);
  d.brier = brier / total;
  const Real obar = total_o / total;
  Real rel = Real(0), res = Real(0);
  for (const BinAgg& b : bins) {
    if (b.count == 0) continue;
    const Real w = static_cast<Real>(b.count) / total;
    const Real p_k = b.sum_p / static_cast<Real>(b.count);
    const Real o_k = b.sum_o / static_cast<Real>(b.count);
    rel += w * (p_k - o_k) * (p_k - o_k);
    res += w * (o_k - obar) * (o_k - obar);
  }
  d.reliability = rel;
  d.resolution = res;
  d.uncertainty = obar * (Real(1) - obar);
  d.brier_climatology = d.uncertainty;
  d.skill = d.uncertainty > Real(0) ? Real(1) - d.brier / d.uncertainty : kNaN;
  return d;
}

// BS = mean (p - o)^2；O(n)。见 probabilistic.hpp 文件头 [E3]。
Real brier_score(const std::vector<Real>& p, const std::vector<Real>& o) {
  if (p.size() != o.size()) {
    throw DimensionError("brier_score: 概率与观测长度不等");
  }
  Real sum = Real(0);
  std::size_t n = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (is_missing(p[i]) || is_missing(o[i])) continue;
    const Real e = p[i] - event_of(o[i]);
    sum += e * e;
    ++n;
  }
  return n > 0 ? sum / static_cast<Real>(n) : kNaN;
}

// BSS = 1 - BS/UNC；UNC = 0 时返回 kNaN。O(n)。
Real brier_skill_score(const std::vector<Real>& p, const std::vector<Real>& o) {
  const BrierDecomposition d = brier_decomposition(p, o, 10, BinScheme::EqualWidth);
  return d.skill;
}

// ---------------------------------------------------------------------------
// ROC / AUC
// ---------------------------------------------------------------------------

// 由概率预报构造 ROC 工作点（阈值降序）。O(n log n)。
RocCurve roc_curve(const std::vector<Real>& p, const std::vector<Real>& o_event) {
  if (p.size() != o_event.size()) {
    throw DimensionError("roc_curve: 概率与事件序列长度不等");
  }
  std::vector<std::pair<Real, Real>> v;
  v.reserve(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (is_missing(p[i]) || is_missing(o_event[i])) continue;
    v.emplace_back(p[i], event_of(o_event[i]));
  }
  RocCurve c;
  if (v.empty()) return c;
  for (const auto& e : v) {
    if (e.second >= Real(0.5)) ++c.n_pos; else ++c.n_neg;
  }
  std::sort(v.begin(), v.end(),
            [](const std::pair<Real, Real>& a, const std::pair<Real, Real>& b) {
              return a.first > b.first;
            });
  // 逐阈值累积：阈值从高到低。
  Real pos = Real(0), neg = Real(0);
  const Real npos = static_cast<Real>(c.n_pos);
  const Real nneg = static_cast<Real>(c.n_neg);
  std::size_t i = 0;
  while (i < v.size()) {
    const Real t = v[i].first;
    while (i < v.size() && v[i].first == t) {
      if (v[i].second >= Real(0.5)) pos += Real(1); else neg += Real(1);
      ++i;
    }
    c.thresholds.push_back(t);
    c.pod.push_back(npos > Real(0) ? pos / npos : kNaN);
    c.pofd.push_back(nneg > Real(0) ? neg / nneg : kNaN);
  }
  c.auc = roc_auc(p, o_event);
  return c;
}

// AUC = P(p_pos > p_neg) + 0.5 P(p_pos = p_neg)。O(n log n)。[E15]
Real roc_auc(const std::vector<Real>& p, const std::vector<Real>& o_event) {
  if (p.size() != o_event.size()) {
    throw DimensionError("roc_auc: 概率与事件序列长度不等 (" +
                         std::to_string(p.size()) + " vs " +
                         std::to_string(o_event.size()) + ")");
  }
  std::vector<std::pair<Real, Real>> v;
  v.reserve(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (is_missing(p[i]) || is_missing(o_event[i])) continue;
    v.emplace_back(p[i], event_of(o_event[i]));
  }
  return auc_from_pairs(v);
}

// ---------------------------------------------------------------------------
// CRPS
// ---------------------------------------------------------------------------

// 排序样本估计式，见 probabilistic.hpp 文件头（[E12]）。O(m log m)。
Real crps_ensemble(const std::vector<Real>& members, Real obs) {
  if (is_missing(obs)) return kNaN;
  std::vector<Real> x;
  x.reserve(members.size());
  for (const Real v : members) {
    if (!is_missing(v)) x.push_back(v);
  }
  const std::size_t m = x.size();
  if (m == 0) return kNaN;
  if (m == 1) return std::abs(x[0] - obs);
  std::sort(x.begin(), x.end());
  Real s1 = Real(0), s2 = Real(0);
  const Real mf = static_cast<Real>(m);
  for (std::size_t i = 0; i < m; ++i) {
    s1 += std::abs(x[i] - obs);
    // i 从 0 计，对应 1 基的 (i+1)，故 2(i+1)-m-1 = 2i+1-m。
    s2 += (Real(2) * static_cast<Real>(i) + Real(1) - mf) * x[i];
  }
  return s1 / mf - s2 / (mf * mf);
}

// 无偏（fair）版本，[E16]。O(m log m)。
Real crps_fair(const std::vector<Real>& members, Real obs) {
  if (is_missing(obs)) return kNaN;
  std::vector<Real> x;
  x.reserve(members.size());
  for (const Real v : members) {
    if (!is_missing(v)) x.push_back(v);
  }
  const std::size_t m = x.size();
  if (m == 0) return kNaN;
  if (m == 1) return std::abs(x[0] - obs);
  std::sort(x.begin(), x.end());
  Real s1 = Real(0), s2 = Real(0);
  const Real mf = static_cast<Real>(m);
  for (std::size_t i = 0; i < m; ++i) {
    s1 += std::abs(x[i] - obs);
    s2 += (Real(2) * static_cast<Real>(i) + Real(1) - mf) * x[i];
  }
  return s1 / mf - s2 / (mf * (mf - Real(1)));
}

// CRPS = int (F - H(x-y))^2 dx；梯形法。O(n)。见 probabilistic.hpp 文件头。
Real crps_from_cdf(const std::vector<Real>& x, const std::vector<Real>& cdf, Real obs) {
  if (x.size() != cdf.size()) {
    throw DimensionError("crps_from_cdf: 网格与 CDF 长度不等");
  }
  if (x.size() < 2 || is_missing(obs)) return kNaN;
  Real integral = Real(0);
  for (std::size_t i = 0; i + 1 < x.size(); ++i) {
    const Real hi = x[i] >= obs ? Real(1) : Real(0);
    const Real hj = x[i + 1] >= obs ? Real(1) : Real(0);
    const Real gi = (cdf[i] - hi) * (cdf[i] - hi);
    const Real gj = (cdf[i + 1] - hj) * (cdf[i + 1] - hj);
    integral += Real(0.5) * (gi + gj) * (x[i + 1] - x[i]);
  }
  return integral;
}

// 逐点 CRPS 均值。O(sum m_i log m_i)。
Real mean_crps(const std::vector<std::vector<Real>>& members,
               const std::vector<Real>& obs) {
  if (members.size() != obs.size()) {
    throw DimensionError("mean_crps: 集合点数与观测数不等");
  }
  Real sum = Real(0);
  std::size_t n = 0;
  for (std::size_t i = 0; i < members.size(); ++i) {
    const Real c = crps_ensemble(members[i], obs[i]);
    if (is_missing(c)) continue;
    sum += c;
    ++n;
  }
  return n > 0 ? sum / static_cast<Real>(n) : kNaN;
}

// ---------------------------------------------------------------------------
// 秩直方图
// ---------------------------------------------------------------------------

std::vector<Real> RankHistogram::probabilities() const {
  Real total = Real(0);
  for (const Real c : counts) total += c;
  std::vector<Real> p(counts.size(), kNaN);
  if (total <= Real(0)) return p;
  for (std::size_t k = 0; k < counts.size(); ++k) {
    p[k] = counts[k] / total;
  }
  return p;
}

// 秩 = 观测在成员中的位次；并列按 1/(t+1) 摊分（[E9]）。O(n m log m)。
RankHistogram rank_histogram(const std::vector<std::vector<Real>>& members,
                             const std::vector<Real>& obs) {
  if (members.size() != obs.size()) {
    throw DimensionError("rank_histogram: 集合点数与观测数不等 (" +
                         std::to_string(members.size()) + " vs " +
                         std::to_string(obs.size()) + ")");
  }
  RankHistogram h;
  if (members.empty()) return h;
  const std::size_t m = members[0].size();
  for (const auto& e : members) {
    if (e.size() != m) {
      throw DimensionError("rank_histogram: 各点集合成员数不一致");
    }
  }
  h.ensemble_size = m;
  h.counts.assign(m + 1, Real(0));
  for (std::size_t i = 0; i < members.size(); ++i) {
    if (is_missing(obs[i])) continue;
    const std::vector<Real>& ens = members[i];
    bool ok = true;
    for (const Real v : ens) {
      if (is_missing(v)) { ok = false; break; }
    }
    if (!ok) continue;
    std::size_t lower = 0, equal = 0;
    for (const Real v : ens) {
      if (v < obs[i]) {
        ++lower;
      } else if (v == obs[i]) {
        ++equal;
      }
    }
    if (equal == 0) {
      h.counts[lower] += Real(1);
    } else {
      const Real share = Real(1) / static_cast<Real>(equal + 1);
      for (std::size_t k = lower; k <= lower + equal; ++k) {
        h.counts[k] += share;
      }
    }
    ++h.samples;
  }
  h.flatness = rank_histogram_flatness(h.counts);
  h.bias = rank_histogram_bias(h.counts);
  if (h.samples > 0 && h.counts.size() > 1) {
    const Real K = static_cast<Real>(h.counts.size());
    const std::vector<Real> pr = h.probabilities();
    Real acc = Real(0);
    for (std::size_t k = 0; k < pr.size(); ++k) {
      const Real d = pr[k] - Real(1) / K;
      acc += d * d;
    }
    h.chi2 = static_cast<Real>(h.samples) * K * acc;
    h.dof = K - Real(1);
    h.p_value = gamma_q(h.dof / Real(2), h.chi2 / Real(2));
  }
  return h;
}

// 平坦度 = 1 - (1/2) sum|p-1/K| / (1-1/K)。O(K)。
Real rank_histogram_flatness(const std::vector<Real>& counts) {
  const std::size_t K = counts.size();
  if (K == 0) return kNaN;
  if (K == 1) return Real(1);
  Real total = Real(0);
  for (const Real c : counts) total += c;
  if (total <= Real(0)) return kNaN;
  const Real uniform = Real(1) / static_cast<Real>(K);
  Real l1 = Real(0);
  for (const Real c : counts) {
    l1 += std::abs(c / total - uniform);
  }
  const Real scale = Real(1) - uniform;
  const Real f = Real(1) - Real(0.5) * l1 / scale;
  return f < Real(0) ? Real(0) : (f > Real(1) ? Real(1) : f);
}

// 秩偏差 = sum p_k (k+0.5)/K - 1/2。O(K)。
Real rank_histogram_bias(const std::vector<Real>& counts) {
  const std::size_t K = counts.size();
  if (K == 0) return kNaN;
  Real total = Real(0);
  for (const Real c : counts) total += c;
  if (total <= Real(0)) return kNaN;
  Real mean_rank = Real(0);
  for (std::size_t k = 0; k < K; ++k) {
    mean_rank += (counts[k] / total) * (static_cast<Real>(k) + Real(0.5)) /
                 static_cast<Real>(K);
  }
  return mean_rank - Real(0.5);
}

// 正则化上不完全 Gamma Q(a,x)：级数 + 连分式，相对精度 ~1e-14。O(1) 次迭代。
Real gamma_q(Real a, Real x) {
  if (is_missing(a) || is_missing(x)) return kNaN;
  if (a <= Real(0) || x < Real(0)) return kNaN;
  if (x == Real(0)) return Real(1);
  const Real gln = std::lgamma(a);
  const Real eps = Real(1e-14);
  const Real tiny = Real(1e-300);
  if (x < a + Real(1)) {
    // 级数展开求 P(a,x)，再取 Q = 1 - P。
    Real ap = a;
    Real sum = Real(1) / a;
    Real del = sum;
    for (int n = 1; n <= 500; ++n) {
      ap += Real(1);
      del *= x / ap;
      sum += del;
      if (std::abs(del) < std::abs(sum) * eps) break;
    }
    const Real p = sum * std::exp(-x + a * std::log(x) - gln);
    const Real q = Real(1) - p;
    return q < Real(0) ? Real(0) : (q > Real(1) ? Real(1) : q);
  }
  // 连分式求 Q(a,x)。
  Real b = x + Real(1) - a;
  Real c = Real(1) / tiny;
  Real d = Real(1) / b;
  Real h = d;
  for (int i = 1; i <= 500; ++i) {
    const Real an = -static_cast<Real>(i) * (static_cast<Real>(i) - a);
    b += Real(2);
    d = an * d + b;
    if (std::abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::abs(c) < tiny) c = tiny;
    d = Real(1) / d;
    const Real delta = d * c;
    h *= delta;
    if (std::abs(delta - Real(1)) < eps) break;
  }
  const Real q = std::exp(-x + a * std::log(x) - gln) * h;
  return q < Real(0) ? Real(0) : (q > Real(1) ? Real(1) : q);
}

// ---------------------------------------------------------------------------
// 集合离散度
// ---------------------------------------------------------------------------

// 无偏样本标准差（跳过 NaN）；m < 2 返回 kNaN。O(m)。
Real ensemble_spread_members(const std::vector<Real>& members) {
  std::size_t m = 0;
  Real mean = Real(0), m2 = Real(0);
  for (const Real v : members) {
    if (is_missing(v)) continue;
    const std::size_t m1 = m + 1;
    const Real d = v - mean;
    mean += d / static_cast<Real>(m1);
    m2 += d * (v - mean);
    m = m1;
  }
  if (m < 2) return kNaN;
  return std::sqrt(m2 / static_cast<Real>(m - 1));
}

// 多点平均离散度。O(n m)。
Real ensemble_spread(const std::vector<std::vector<Real>>& members) {
  Real sum = Real(0);
  std::size_t n = 0;
  for (const auto& e : members) {
    const Real s = ensemble_spread_members(e);
    if (is_missing(s)) continue;
    sum += s;
    ++n;
  }
  return n > 0 ? sum / static_cast<Real>(n) : kNaN;
}

// 集合平均 RMSE。O(n m)。
Real ensemble_mean_rmse(const std::vector<std::vector<Real>>& members,
                        const std::vector<Real>& obs) {
  if (members.size() != obs.size()) {
    throw DimensionError("ensemble_mean_rmse: 集合点数与观测数不等");
  }
  Real sum = Real(0);
  std::size_t n = 0;
  for (std::size_t i = 0; i < members.size(); ++i) {
    if (is_missing(obs[i])) continue;
    Real s = Real(0);
    std::size_t c = 0;
    for (const Real v : members[i]) {
      if (is_missing(v)) continue;
      s += v;
      ++c;
    }
    if (c == 0) continue;
    const Real e = s / static_cast<Real>(c) - obs[i];
    sum += e * e;
    ++n;
  }
  return n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : kNaN;
}

// 离散度-技巧比；rmse = 0 时返回 kNaN。O(n m)。
Real spread_skill_ratio(const std::vector<std::vector<Real>>& members,
                        const std::vector<Real>& obs) {
  const Real sp = ensemble_spread(members);
  const Real rm = ensemble_mean_rmse(members, obs);
  return (rm > Real(0)) ? sp / rm : kNaN;
}

Scores EnsembleStats::to_scores() const {
  Scores s = zero_scores(n);
  s.spread = spread;
  s.ensemble_spread = spread;
  s.spread_skill_ratio = spread_skill_ratio;
  s.ensemble_mean_rmse = rmse;
  s.crps = crps;
  s.bias = bias;
  s.mean_error = bias;
  return s;
}

// 一次性集合检验。O(n m log m)（CRPS 需要逐点排序）。
EnsembleStats ensemble_scores(const std::vector<std::vector<Real>>& members,
                              const std::vector<Real>& obs) {
  if (members.size() != obs.size()) {
    throw DimensionError("ensemble_scores: 集合点数与观测数不等");
  }
  EnsembleStats st;
  Real sum_spread = Real(0), sum_rmse2 = Real(0), sum_bias = Real(0), sum_crps = Real(0);
  std::size_t n = 0;
  for (std::size_t i = 0; i < members.size(); ++i) {
    if (is_missing(obs[i])) continue;
    Real s = Real(0);
    std::size_t c = 0;
    for (const Real v : members[i]) {
      if (is_missing(v)) continue;
      s += v;
      ++c;
    }
    if (c == 0) continue;
    const Real em = s / static_cast<Real>(c);
    sum_bias += em - obs[i];
    const Real e = em - obs[i];
    sum_rmse2 += e * e;
    const Real sp = ensemble_spread_members(members[i]);
    if (!is_missing(sp)) sum_spread += sp;
    const Real c_ = crps_ensemble(members[i], obs[i]);
    if (!is_missing(c_)) sum_crps += c_;
    ++n;
  }
  if (n == 0) return st;
  const Real fn = static_cast<Real>(n);
  st.n = n;
  st.spread = sum_spread / fn;
  st.rmse = std::sqrt(sum_rmse2 / fn);
  st.bias = sum_bias / fn;
  st.crps = sum_crps / fn;
  st.spread_skill_ratio = st.rmse > Real(0) ? st.spread / st.rmse : kNaN;
  return st;
}

// ---------------------------------------------------------------------------
// 批量概率评分
// ---------------------------------------------------------------------------

Scores compute_probabilistic(const std::vector<Real>& p, const std::vector<Real>& o) {
  return compute_probabilistic(p, o, 10, BinScheme::EqualWidth);
}

// 见 probabilistic.hpp：只填概率型字段，其余保持 kNaN。O(n log n)。
Scores compute_probabilistic(const std::vector<Real>& p, const std::vector<Real>& o,
                             int n_bins, BinScheme scheme) {
  if (p.size() != o.size()) {
    throw DimensionError("compute_probabilistic: 概率与观测长度不等 (" +
                         std::to_string(p.size()) + " vs " + std::to_string(o.size()) + ")");
  }
  const BrierDecomposition d = brier_decomposition(p, o, n_bins, scheme);
  Scores s = zero_scores(d.n);
  s.brier = d.brier;
  s.brier_reliability = d.reliability;
  s.brier_resolution = d.resolution;
  s.brier_uncertainty = d.uncertainty;
  s.brier_skill = d.skill;
  s.roc_auc = roc_auc(p, o);
  return s;
}

}  // namespace vibe::verify

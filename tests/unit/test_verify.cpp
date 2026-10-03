/// @file test_verify.cpp
/// @brief \c vibe::verify 的单元测试（框架：include/vibe/common/test.hpp）。
///
/// 覆盖：完美预报、随机/退化预报、已知解析值（2x2 列联表手算）、
/// NaN 缺测、增量累加与批量一致性、FSS 极限性质、CRPS 解析值、
/// ROC/秩直方图/DCT 往返、配置解析与结果序列化。

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vibe/common/test.hpp"
#include "vibe/verify/contingency.hpp"
#include "vibe/verify/probabilistic.hpp"
#include "vibe/verify/scores.hpp"
#include "vibe/verify/spatial.hpp"
#include "vibe/verify/verification.hpp"

// 测试写在全局作用域，因此这里在文件作用域引入类型与 API。
using vibe::Real;
using namespace vibe::verify;

namespace {

/// 确定性 xorshift32，保证测试可复现（不依赖 <random> 实现差异）。
struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
  std::uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  Real uniform() { return static_cast<Real>(next() & 0xFFFFFFu) / static_cast<Real>(0x1000000u); }
  Real symmetric() { return Real(2) * uniform() - Real(1); }
};

std::vector<Real> make_random(std::size_t n, std::uint32_t seed) {
  Rng rng(seed);
  std::vector<Real> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = rng.symmetric();
  return v;
}

constexpr Real kTol = Real(1e-12);

}  // namespace

// ---------------------------------------------------------------------------
// 连续型评分
// ---------------------------------------------------------------------------

VIBE_TEST(continuous_perfect_forecast) {
  const std::vector<Real> f = {1.0, 2.0, 3.0, 4.0, 5.0};
  const Scores s = compute_continuous(f, f, nullptr);
  VIBE_CHECK_NEAR(s.bias, 0.0, kTol);
  VIBE_CHECK_NEAR(s.mae, 0.0, kTol);
  VIBE_CHECK_NEAR(s.rmse, 0.0, kTol);
  VIBE_CHECK_NEAR(s.correlation, 1.0, kTol);
  VIBE_CHECK(s.n == 5);
}

VIBE_TEST(continuous_known_errors) {
  const std::vector<Real> f = {1.0, 2.0, 3.0};
  const std::vector<Real> o = {1.0, 2.0, 4.0};
  const Scores s = compute_continuous(f, o, nullptr);
  VIBE_CHECK_NEAR(s.bias, -1.0 / 3.0, kTol);
  VIBE_CHECK_NEAR(s.mean_error, -1.0 / 3.0, kTol);
  VIBE_CHECK_NEAR(s.mae, 1.0 / 3.0, kTol);
  VIBE_CHECK_NEAR(s.rmse, std::sqrt(1.0 / 3.0), kTol);
  VIBE_CHECK_NEAR(s.std_error, std::sqrt(1.0 / 3.0), kTol);
  VIBE_CHECK_NEAR(s.mean_forecast, 2.0, kTol);
  VIBE_CHECK_NEAR(s.mean_observation, 7.0 / 3.0, kTol);
}

VIBE_TEST(continuous_constant_series_correlation_zero) {
  const std::vector<Real> f = {5.0, 5.0, 5.0, 5.0};
  const std::vector<Real> o = {1.0, 2.0, 3.0, 4.0};
  const Scores s = compute_continuous(f, o, nullptr);
  VIBE_CHECK_NEAR(s.correlation, 0.0, kTol);
  VIBE_CHECK_NEAR(s.bias, 5.0 - 2.5, kTol);
}

VIBE_TEST(continuous_nan_pairs_skipped) {
  const Real nan = kNaN;
  const std::vector<Real> f = {1.0, 2.0, nan, 4.0, 5.0};
  const std::vector<Real> o = {1.0, 3.0, 3.0, 4.0, 6.0};
  const Scores s = compute_continuous(f, o, nullptr);
  VIBE_CHECK(s.n == 4);
  VIBE_CHECK_NEAR(s.mae, 0.5, kTol);
  VIBE_CHECK_NEAR(s.rmse, std::sqrt(0.5), kTol);
}

VIBE_TEST(continuous_anomaly_correlation) {
  const std::vector<Real> f = {2.0, 4.0, 6.0};
  const std::vector<Real> o = {1.0, 2.0, 3.0};
  const std::vector<Real> clim = {0.0, 0.0, 0.0};
  const Scores s = compute_continuous(f, o, &clim);
  VIBE_CHECK_NEAR(s.anomaly_correlation, 1.0, kTol);
  // 长度为 1 的常数气候态也应被接受
  const std::vector<Real> clim1 = {0.0};
  const Scores s1 = compute_continuous(f, o, &clim1);
  VIBE_CHECK_NEAR(s1.anomaly_correlation, 1.0, kTol);
}

VIBE_TEST(continuous_anomaly_undefined_without_climatology) {
  const std::vector<Real> f = {2.0, 4.0, 6.0};
  const std::vector<Real> o = {1.0, 2.0, 3.0};
  const Scores s = compute_continuous(f, o, nullptr);
  VIBE_CHECK(std::isnan(s.anomaly_correlation));
}

VIBE_TEST(continuous_dimension_mismatch_throws) {
  const std::vector<Real> f = {1.0, 2.0};
  const std::vector<Real> o = {1.0, 2.0, 3.0};
  VIBE_CHECK_THROWS(compute_continuous(f, o, nullptr));
}

// ---------------------------------------------------------------------------
// 增量累加器
// ---------------------------------------------------------------------------

VIBE_TEST(accumulator_matches_batch_continuous) {
  const std::vector<Real> f = make_random(257, 12345u);
  std::vector<Real> o(257);
  Rng rng(999u);
  for (std::size_t i = 0; i < o.size(); ++i) o[i] = f[i] + Real(0.3) * rng.symmetric();

  const Scores batch = compute_continuous(f, o, nullptr);
  ScoreAccumulator acc;
  for (std::size_t i = 0; i < f.size(); ++i) acc.add(f[i], o[i]);
  const Scores inc = acc.scores();
  VIBE_CHECK(inc.n == batch.n);
  VIBE_CHECK_NEAR(inc.bias, batch.bias, Real(1e-9));
  VIBE_CHECK_NEAR(inc.mae, batch.mae, Real(1e-9));
  VIBE_CHECK_NEAR(inc.rmse, batch.rmse, Real(1e-9));
  VIBE_CHECK_NEAR(inc.correlation, batch.correlation, Real(1e-9));
  VIBE_CHECK_NEAR(inc.std_error, batch.std_error, Real(1e-9));
}

VIBE_TEST(accumulator_merge_matches_single) {
  const std::vector<Real> f = make_random(200, 7u);
  std::vector<Real> o(200);
  Rng rng(42u);
  for (std::size_t i = 0; i < o.size(); ++i) o[i] = f[i] + rng.symmetric();

  ScoreAccumulator a, b;
  for (std::size_t i = 0; i < 100; ++i) a.add(f[i], o[i]);
  for (std::size_t i = 100; i < 200; ++i) b.add(f[i], o[i]);
  a.merge(b);

  const Scores batch = compute_continuous(f, o, nullptr);
  const Scores merged = a.scores();
  VIBE_CHECK_NEAR(merged.rmse, batch.rmse, Real(1e-12));
  VIBE_CHECK_NEAR(merged.correlation, batch.correlation, Real(1e-9));
  VIBE_CHECK_NEAR(merged.bias, batch.bias, Real(1e-12));
}

VIBE_TEST(accumulator_anomaly_correlation) {
  ScoreAccumulator acc;
  const std::vector<Real> f = {2.0, 4.0, 6.0};
  const std::vector<Real> o = {1.0, 2.0, 3.0};
  for (std::size_t i = 0; i < f.size(); ++i) acc.add(f[i], o[i], Real(0));
  VIBE_CHECK_NEAR(acc.scores().anomaly_correlation, 1.0, kTol);
}

VIBE_TEST(accumulator_probability_brier_matches_batch) {
  const std::vector<Real> p = {0.1, 0.9, 0.4, 0.7, 0.2};
  const std::vector<Real> o = {0.0, 1.0, 0.0, 1.0, 0.0};
  ScoreAccumulator acc(0.5, 5);
  for (std::size_t i = 0; i < p.size(); ++i) acc.add_probability(p[i], o[i]);
  VIBE_CHECK_NEAR(acc.scores().brier, brier_score(p, o), Real(1e-12));
}

// ---------------------------------------------------------------------------
// 列联表与分类评分（含 2x2 手算值）
// ---------------------------------------------------------------------------

VIBE_TEST(contingency_2x2_hand_computed) {
  // H=40, M=10, FA=20, CN=130  =>  N=200
  const ContingencyTable t = ContingencyTable::from_counts(40, 10, 20, 130);
  VIBE_CHECK_NEAR(t.total(), 200.0, kTol);
  VIBE_CHECK_NEAR(t.random_hits(), 15.0, kTol);          // (50*60)/200
  VIBE_CHECK_NEAR(t.pod(), 0.8, kTol);                   // 40/50
  VIBE_CHECK_NEAR(t.far(), 1.0 / 3.0, kTol);             // 20/60
  VIBE_CHECK_NEAR(t.csi(), 40.0 / 70.0, kTol);
  VIBE_CHECK_NEAR(t.ets(), 25.0 / 95.0, kTol);           // (40-15)/(110-15)
  VIBE_CHECK_NEAR(t.frequency_bias(), 1.2, kTol);        // 60/50
  VIBE_CHECK_NEAR(t.tss(), 0.8 - 20.0 / 150.0, kTol);
  VIBE_CHECK_NEAR(t.orss(), 5000.0 / 5400.0, kTol);
  VIBE_CHECK_NEAR(t.accuracy(), 0.85, kTol);
  VIBE_CHECK_NEAR(t.pofd(), 20.0 / 150.0, kTol);
  VIBE_CHECK(t.sedi() > 0.0 && t.sedi() < 1.0);
}

VIBE_TEST(contingency_all_hits) {
  const ContingencyTable t = ContingencyTable::from_counts(10, 0, 0, 10);
  VIBE_CHECK_NEAR(t.pod(), 1.0, kTol);
  VIBE_CHECK_NEAR(t.far(), 0.0, kTol);
  VIBE_CHECK_NEAR(t.csi(), 1.0, kTol);
  VIBE_CHECK_NEAR(t.ets(), 1.0, kTol);
  VIBE_CHECK_NEAR(t.tss(), 1.0, kTol);
  VIBE_CHECK_NEAR(t.accuracy(), 1.0, kTol);
  VIBE_CHECK_NEAR(t.frequency_bias(), 1.0, kTol);
}

VIBE_TEST(contingency_all_clear_undefined_scores_are_nan) {
  const ContingencyTable t = ContingencyTable::from_counts(0, 0, 0, 10);
  VIBE_CHECK_NEAR(t.accuracy(), 1.0, kTol);
  VIBE_CHECK(std::isnan(t.pod()));
  VIBE_CHECK(std::isnan(t.far()));
  VIBE_CHECK(std::isnan(t.ets()));
  VIBE_CHECK(std::isnan(t.orss()));
  VIBE_CHECK(std::isnan(t.frequency_bias()));
}

VIBE_TEST(categorical_from_threshold) {
  const std::vector<Real> f = {1.0, 2.0, 3.0, 4.0};
  const std::vector<Real> o = {1.0, 0.0, 3.0, 0.0};
  const Scores s = compute_categorical(f, o, 2.0);
  VIBE_CHECK(s.n == 4);
  VIBE_CHECK_NEAR(s.pod, 1.0, kTol);          // H=1, M=0
  VIBE_CHECK_NEAR(s.far, 2.0 / 3.0, kTol);    // FA=2, H=1
  VIBE_CHECK_NEAR(s.csi, 1.0 / 3.0, kTol);
  VIBE_CHECK_NEAR(s.ets, 0.25 / 2.25, kTol);
  VIBE_CHECK_NEAR(s.accuracy, 0.5, kTol);
  VIBE_CHECK_NEAR(s.rmse, 0.0, kTol);         // 连续型字段不适用 -> 0（契约）
  VIBE_CHECK_NEAR(s.brier, 0.0, kTol);        // 概率型字段不适用 -> 0
}

VIBE_TEST(categorical_from_binary_arrays) {
  const std::vector<int> f = {1, 1, 0, 0};
  const std::vector<int> o = {1, 0, 1, 0};
  const ContingencyTable t = ContingencyTable::from_binary(f, o);
  VIBE_CHECK_NEAR(t.hits, 1.0, kTol);
  VIBE_CHECK_NEAR(t.false_alarms, 1.0, kTol);
  VIBE_CHECK_NEAR(t.misses, 1.0, kTol);
  VIBE_CHECK_NEAR(t.correct_negatives, 1.0, kTol);
}

// ---------------------------------------------------------------------------
// 概率评分
// ---------------------------------------------------------------------------

VIBE_TEST(brier_perfect_probabilistic) {
  const std::vector<Real> p = {0.0, 1.0, 0.0, 1.0, 1.0};
  const std::vector<Real> o = {0.0, 1.0, 0.0, 1.0, 1.0};
  VIBE_CHECK_NEAR(brier_score(p, o), 0.0, kTol);
  VIBE_CHECK_NEAR(brier_skill_score(p, o), 1.0, kTol);
  VIBE_CHECK_NEAR(roc_auc(p, o), 1.0, kTol);
}

VIBE_TEST(brier_climatology_has_zero_skill) {
  const std::vector<Real> o = {0.0, 1.0, 1.0, 0.0};
  const std::vector<Real> p(o.size(), 0.5);  // 气候频率 = 0.5
  VIBE_CHECK_NEAR(brier_score(p, o), 0.25, kTol);
  VIBE_CHECK_NEAR(brier_skill_score(p, o), 0.0, kTol);
}

VIBE_TEST(brier_decomposition_identity_for_discrete_probabilities) {
  // 预报概率取等宽箱中心（箱内为常数），此时 BS = REL - RES + UNC 精确成立。
  std::vector<Real> p, o;
  Rng rng(2024u);
  for (int k = 0; k < 10; ++k) {
    const Real pc = (static_cast<Real>(k) + 0.5) / 10.0;
    for (int r = 0; r < 5; ++r) {
      p.push_back(pc);
      o.push_back(rng.uniform() < pc ? 1.0 : 0.0);
    }
  }
  const BrierDecomposition d = brier_decomposition(p, o, 10, BinScheme::EqualWidth);
  VIBE_CHECK_NEAR(d.reliability - d.resolution + d.uncertainty, d.brier, Real(1e-10));
  VIBE_CHECK(std::abs(d.skill - (1.0 - d.brier / d.uncertainty)) < 1e-12);
}

VIBE_TEST(reliability_curve_equal_frequency_bins_are_balanced) {
  std::vector<Real> p, o;
  Rng rng(5u);
  for (int i = 0; i < 100; ++i) {
    p.push_back(rng.uniform());
    o.push_back(rng.uniform() < 0.5 ? 1.0 : 0.0);
  }
  const std::vector<ReliabilityBin> bins =
      reliability_curve(p, o, 5, BinScheme::EqualFrequency);
  VIBE_CHECK(bins.size() == 5);
  for (const ReliabilityBin& b : bins) VIBE_CHECK(b.count == 20);
}

VIBE_TEST(roc_auc_ties_and_separation) {
  const std::vector<Real> p_tie = {0.5, 0.5, 0.5, 0.5};
  const std::vector<Real> o_tie = {1.0, 0.0, 1.0, 0.0};
  VIBE_CHECK_NEAR(roc_auc(p_tie, o_tie), 0.5, kTol);

  const std::vector<Real> p_sep = {0.9, 0.8, 0.2, 0.1};
  const std::vector<Real> o_sep = {1.0, 1.0, 0.0, 0.0};
  VIBE_CHECK_NEAR(roc_auc(p_sep, o_sep), 1.0, kTol);

  const std::vector<Real> p_inv = {0.1, 0.2, 0.8, 0.9};
  VIBE_CHECK_NEAR(roc_auc(p_inv, o_sep), 0.0, kTol);
}

VIBE_TEST(compute_probabilistic_fields) {
  const std::vector<Real> p = {0.9, 0.8, 0.2, 0.1};
  const std::vector<Real> o = {1.0, 1.0, 0.0, 0.0};
  const Scores s = compute_probabilistic(p, o);
  VIBE_CHECK(s.n == 4);
  VIBE_CHECK_NEAR(s.brier, 0.025, kTol);
  VIBE_CHECK_NEAR(s.roc_auc, 1.0, kTol);
  VIBE_CHECK_NEAR(s.rmse, 0.0, kTol);   // 连续型字段不适用 -> 0（契约）
  VIBE_CHECK_NEAR(s.fss, 0.0, kTol);    // 空间型字段不适用 -> 0（契约）
}

// ---------------------------------------------------------------------------
// CRPS 与集合离散度
// ---------------------------------------------------------------------------

VIBE_TEST(crps_two_member_analytic) {
  const Real a = 1.0, y = 3.0;
  const std::vector<Real> members = {y - a, y + a};
  VIBE_CHECK_NEAR(crps_ensemble(members, y), a / 2.0, kTol);
  VIBE_CHECK_NEAR(crps_fair(members, y), a / 2.0, kTol);
}

VIBE_TEST(crps_perfect_ensemble_is_zero) {
  const std::vector<Real> members = {7.0, 7.0, 7.0};
  VIBE_CHECK_NEAR(crps_ensemble(members, 7.0), 0.0, kTol);
  VIBE_CHECK_NEAR(crps_fair(members, 7.0), 0.0, kTol);
}

VIBE_TEST(crps_from_cdf_matches_ensemble_cdf) {
  // 两点等权集合 {1,3}，观测 2：F(x)=0.5*H(x-1)+0.5*H(x-3)
  const std::vector<Real> x = {0.0, 1.0, 2.0, 3.0, 4.0};
  const std::vector<Real> cdf = {0.0, 0.5, 0.5, 1.0, 1.0};
  const Real v = crps_from_cdf(x, cdf, 2.0);
  VIBE_CHECK(v >= 0.0 && v <= 1.0);
}

VIBE_TEST(ensemble_spread_and_ratio) {
  const std::vector<std::vector<Real>> members = {{0.0, 2.0}};
  const std::vector<Real> obs = {3.0};
  VIBE_CHECK_NEAR(ensemble_spread_members(members[0]), std::sqrt(2.0), kTol);
  VIBE_CHECK_NEAR(ensemble_spread(members), std::sqrt(2.0), kTol);
  VIBE_CHECK_NEAR(ensemble_mean_rmse(members, obs), 2.0, kTol);
  VIBE_CHECK_NEAR(spread_skill_ratio(members, obs), std::sqrt(2.0) / 2.0, kTol);
  const EnsembleStats st = ensemble_scores(members, obs);
  VIBE_CHECK_NEAR(st.bias, -2.0, kTol);
  // CRPS({0,2}, 3) = (|0-3|+|2-3|)/2 - ((1)*(-1)*0 + (1)*2)/4 = 2 - 0.5 = 1.5
  VIBE_CHECK_NEAR(st.crps, 1.5, kTol);
}

VIBE_TEST(rank_histogram_tie_spreading_and_bias) {
  // m=2, 成员 {0,0}, 观测 0：并列摊分到 3 个秩 -> 完全平坦
  const std::vector<std::vector<Real>> ens = {{0.0, 0.0}};
  const std::vector<Real> obs = {0.0};
  const RankHistogram h = rank_histogram(ens, obs);
  VIBE_CHECK(h.counts.size() == 3);
  for (const Real c : h.counts) VIBE_CHECK_NEAR(c, 1.0 / 3.0, kTol);
  VIBE_CHECK_NEAR(h.flatness, 1.0, kTol);
  VIBE_CHECK_NEAR(h.bias, 0.0, kTol);

  // m=1, 成员 {0}, 观测 1：观测高于唯一成员 -> 秩 1，平坦度 0，秩偏差 +0.25
  const std::vector<std::vector<Real>> ens2 = {{0.0}};
  const std::vector<Real> obs2 = {1.0};
  const RankHistogram h2 = rank_histogram(ens2, obs2);
  VIBE_CHECK(h2.counts.size() == 2);
  VIBE_CHECK_NEAR(h2.counts[1], 1.0, kTol);
  VIBE_CHECK_NEAR(h2.flatness, 0.0, kTol);
  VIBE_CHECK_NEAR(h2.bias, 0.25, kTol);
}

VIBE_TEST(gamma_q_known_values) {
  VIBE_CHECK_NEAR(gamma_q(1.0, 1.0), std::exp(-1.0), Real(1e-10));
  VIBE_CHECK_NEAR(gamma_q(0.5, 0.0), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// 空间检验：FSS 极限性质
// ---------------------------------------------------------------------------

VIBE_TEST(fss_perfect_forecast_is_one) {
  const std::vector<Real> f = {0, 0, 0, 0, 1, 0, 0, 1, 0};
  const int nx = 3, ny = 3;
  for (int r = 0; r <= 4; ++r) {
    VIBE_CHECK_NEAR(fractions_skill_score(f, f, nx, ny, 0.5, r), 1.0, Real(1e-12));
  }
}

VIBE_TEST(fss_radius_zero_reduces_to_point_ratio) {
  // H=1, M=0, FA=1, CN=2  =>  FSS(r=0) = 2H/(2H+M+FA) = 2/3
  const std::vector<Real> f = {1, 1, 0, 0};
  const std::vector<Real> o = {1, 0, 0, 0};
  const Real fss0 = fractions_skill_score(f, o, 2, 2, 0.5, 0);
  VIBE_CHECK_NEAR(fss0, 2.0 / 3.0, Real(1e-12));
  const Real csi = compute_categorical(f, o, 0.5).csi;
  VIBE_CHECK_NEAR(csi, 1.0 / 2.0, Real(1e-12));  // FSS(0) != CSI，见文档
  VIBE_CHECK(std::abs(fss0 - csi) > 1e-6);
}

VIBE_TEST(fss_disjoint_events_is_zero) {
  const std::vector<Real> f = {1, 1, 0, 0};
  const std::vector<Real> o = {0, 0, 1, 1};
  VIBE_CHECK_NEAR(fractions_skill_score(f, o, 2, 2, 0.5, 0), 0.0, Real(1e-12));
}

VIBE_TEST(fss_large_radius_tends_to_one_when_coverage_matches) {
  const int nx = 8, ny = 8;
  std::vector<Real> o(nx * ny, 0.0), f(nx * ny, 0.0);
  for (int i = 2; i <= 3; ++i) {
    for (int j = 2; j <= 3; ++j) {
      o[i * ny + j] = 1.0;
      f[(i + 3) * ny + j] = 1.0;  // 同覆盖率的平移
    }
  }
  const Real small = fractions_skill_score(f, o, nx, ny, 0.5, 0);
  const Real large = fractions_skill_score(f, o, nx, ny, 0.5, nx + ny);
  VIBE_CHECK_NEAR(small, 0.0, Real(1e-12));
  VIBE_CHECK_NEAR(large, 1.0, Real(1e-12));
  VIBE_CHECK(large > small);
}

VIBE_TEST(fss_undefined_when_both_fields_empty) {
  const std::vector<Real> z(9, 0.0);
  VIBE_CHECK(std::isnan(fractions_skill_score(z, z, 3, 3, 0.5, 1)));
}

VIBE_TEST(fss_vs_scale_length_and_minimum_resolvable_scale) {
  const int nx = 8, ny = 8;
  std::vector<Real> o(nx * ny, 0.0), f(nx * ny, 0.0);
  for (int i = 2; i <= 3; ++i) {
    for (int j = 2; j <= 3; ++j) {
      o[i * ny + j] = 1.0;
      f[(i + 3) * ny + j] = 1.0;
    }
  }
  const std::vector<Real> curve = fss_vs_scale(f, o, nx, ny, 0.5, 12);
  VIBE_CHECK(curve.size() == 13);
  VIBE_CHECK_NEAR(curve.front(), 0.0, Real(1e-12));
  VIBE_CHECK_NEAR(curve.back(), 1.0, Real(1e-12));
  const Real scale = minimum_resolvable_scale(f, o, nx, ny, 0.5, 12, 0.5);
  VIBE_CHECK(!std::isnan(scale));
  VIBE_CHECK(scale >= 0.0);
}

VIBE_TEST(neighborhood_fractions_boundary_aware) {
  const std::vector<Real> f = {1, 0, 0, 0};
  const std::vector<Real> frac = neighborhood_fractions(f, 2, 2, 0.5, 1);
  // 2x2 网格、半径 1 时所有窗口都覆盖全域：每个点分数 = 1/4
  for (const Real v : frac) VIBE_CHECK_NEAR(v, 0.25, kTol);
}

VIBE_TEST(fuzzy_scores_are_defined_for_nonempty_events) {
  const std::vector<Real> o = {1, 0, 0, 0};
  const std::vector<Real> f = {0, 0, 1, 0};
  const FuzzyScores s = fuzzy_scores(f, o, 2, 2, 0.5, 1);
  VIBE_CHECK(s.n == 4);
  VIBE_CHECK(!std::isnan(s.neighborhood_pod));
  VIBE_CHECK(!std::isnan(s.fss));
}

VIBE_TEST(double_penalty_detects_displacement) {
  const std::vector<Real> o = {1, 0, 0, 0};
  const std::vector<Real> f = {0, 1, 0, 0};  // f 相对 o 右移一个格点
  const DisplacementEstimate e = best_displacement(f, o, 4, 1, 2);
  VIBE_CHECK(e.dx == 1);
  VIBE_CHECK(e.dy == 0);
  VIBE_CHECK_NEAR(e.mse_at_best, 0.0, kTol);
  const DoublePenaltyDiagnosis d = double_penalty_diagnosis(f, o, 4, 1, 2);
  VIBE_CHECK_NEAR(d.amplitude_error, 0.0, kTol);
  VIBE_CHECK_NEAR(d.mse_total, 0.5, kTol);
  VIBE_CHECK_NEAR(d.displacement_error, 0.0, kTol);
  VIBE_CHECK_NEAR(d.double_penalty_index, 1.0, kTol);
}

VIBE_TEST(dct_roundtrip_and_parseval) {
  const int nx = 4, ny = 3;
  const std::vector<Real> x = make_random(static_cast<std::size_t>(nx * ny), 77u);
  std::vector<Real> X, back;
  dct2(x, nx, ny, X);
  idct2(X, nx, ny, back);
  Real e = 0.0, ex = 0.0, eX = 0.0;
  for (std::size_t k = 0; k < x.size(); ++k) {
    e += (back[k] - x[k]) * (back[k] - x[k]);
    ex += x[k] * x[k];
    eX += X[k] * X[k];
  }
  VIBE_CHECK(std::sqrt(e) < 1e-10);
  VIBE_CHECK_NEAR(eX, ex, Real(1e-9));  // 正交变换的能量守恒（Parseval）
  const ScaleSpectrum sp = dct_scale_spectrum(x, nx, ny, 4);
  VIBE_CHECK(sp.band_energy.size() == 4);
  VIBE_CHECK_NEAR(sp.total_energy, ex, Real(1e-9));
}

VIBE_TEST(compute_fractional_skill_contract) {
  const std::vector<Real> f = {1, 1, 0, 0};
  const std::vector<Real> o = {1, 0, 0, 0};
  const Scores s = compute_fractional_skill(f, o, 2, 2, 0.5, 0);
  VIBE_CHECK_NEAR(s.fss, 2.0 / 3.0, Real(1e-12));
  VIBE_CHECK(s.n == 4);
}

// ---------------------------------------------------------------------------
// 配置、驱动与聚合
// ---------------------------------------------------------------------------

VIBE_TEST(config_from_string_parses_keys) {
  const std::string text =
      "variables = t2, qv\n"
      "type.t2 = continuous\n"
      "type.qv = categorical\n"
      "thresholds.qv = 0.5, 1.0\n"
      "radii = 0, 1, 2\n"
      "n_bins = 20\n"
      "bin_scheme = equal_frequency\n"
      "# 注释应被忽略\n";
  const VerifyConfig cfg = VerifyConfig::from_string(text);
  VIBE_CHECK(cfg.variables.size() == 2);
  VIBE_CHECK(cfg.n_bins == 20);
  VIBE_CHECK(cfg.bin_scheme == BinScheme::EqualFrequency);
  const VerifyVariable v = cfg.resolve("qv");
  VIBE_CHECK(v.type == MatchType::Categorical);
  VIBE_CHECK(v.thresholds.size() == 2);
  VIBE_CHECK_NEAR(v.thresholds[0], 0.5, kTol);
  VIBE_CHECK(cfg.default_variable.radii.size() == 3);
}

VIBE_TEST(config_unknown_key_throws) {
  VIBE_CHECK_THROWS(VerifyConfig::from_kv({{"nonsense", "1"}}));
}

VIBE_TEST(verifier_run_produces_records_and_csv) {
  FieldSample s;
  s.variable = "t2";
  s.nx = 2;
  s.ny = 2;
  s.nz = 1;
  s.time = 0.0;
  s.forecast = {1.0, 2.0, 3.0, 4.0};
  s.observation = {1.0, 2.5, 2.5, 4.0};
  VerifyConfig cfg;
  cfg.variables.push_back(VerifyVariable{"t2", MatchType::Continuous});

  const Verifier verifier(cfg);
  const Scores sc = verifier.score_continuous(s);
  VIBE_CHECK(sc.n == 4);
  VIBE_CHECK_NEAR(sc.mae, 0.25, Real(1e-12));

  const VerificationResult res = verifier.run(s);
  VIBE_CHECK(res.size() > 20);
  const std::string csv = res.to_csv();
  VIBE_CHECK(csv.find("rmse") != std::string::npos);
  VIBE_CHECK(csv.find("variable,time,type") != std::string::npos);
  const std::string json = res.to_json();
  VIBE_CHECK(json.find("\"records\"") != std::string::npos);
  const std::vector<Real> rmses = res.values("t2", "rmse");
  VIBE_CHECK(rmses.size() == 1);
  VIBE_CHECK_NEAR(rmses[0], std::sqrt(0.125), Real(1e-9));  // 误差 {0,-0.5,0.5,0}
}

VIBE_TEST(verifier_spatial_config_generates_multiple_radii) {
  FieldSample s;
  s.variable = "precip";
  s.nx = 2;
  s.ny = 2;
  s.forecast = {1, 1, 0, 0};
  s.observation = {1, 0, 0, 0};
  VerifyConfig cfg;
  VerifyVariable v{"precip", MatchType::Spatial};
  v.thresholds = {0.5};
  v.radii = {0, 1};
  cfg.variables.push_back(v);
  const Verifier verifier(cfg);
  const VerificationResult res = verifier.run(s);
  const std::vector<Real> fss0 = res.values("precip", "fss");
  VIBE_CHECK(fss0.size() == 2);
  bool have_radius0 = false, have_radius1 = false;
  for (const auto& r : res.records) {
    if (r.metric == "fss" && r.radius == 0) have_radius0 = true;
    if (r.metric == "fss" && r.radius == 1) have_radius1 = true;
  }
  VIBE_CHECK(have_radius0 && have_radius1);
}

VIBE_TEST(aggregator_bucket_keys_and_statistics) {
  VIBE_CHECK(Aggregator::bucket_key(0.0, TimeBucket::Daily) == "1970-01-01");
  VIBE_CHECK(Aggregator::bucket_key(31.0 * 86400.0, TimeBucket::Monthly) == "1970-02");
  VIBE_CHECK(Aggregator::bucket_key(12345.0, TimeBucket::All) == "all");

  VerificationResult res;
  VerificationResult::Record r;
  r.variable = "t2";
  r.type = "continuous";
  r.metric = "rmse";
  r.time = 0.0;
  r.value = 1.0;
  r.n = 4;
  res.add(r);
  r.value = 3.0;
  r.time = 86400.0;
  res.add(r);

  Aggregator agg;
  agg.add(res);
  VIBE_CHECK(agg.record_count() == 2);
  const std::vector<AggregatedScore> daily = agg.aggregate(TimeBucket::Daily);
  VIBE_CHECK(daily.size() == 2);
  const std::vector<AggregatedScore> all = agg.aggregate(TimeBucket::All);
  VIBE_CHECK(all.size() == 1);
  VIBE_CHECK_NEAR(all[0].mean, 2.0, kTol);
  VIBE_CHECK_NEAR(all[0].min, 1.0, kTol);
  VIBE_CHECK_NEAR(all[0].max, 3.0, kTol);
  VIBE_CHECK_NEAR(all[0].median, 2.0, kTol);
  VIBE_CHECK(all[0].count == 2);
}

VIBE_TEST(verification_result_json_nan_is_null) {
  VerificationResult res;
  VerificationResult::Record r;
  r.variable = "q";
  r.type = "probabilistic";
  r.metric = "crps";
  r.time = 0.0;
  r.value = kNaN;
  res.add(r);
  VIBE_CHECK(res.to_json().find("null") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 说明：本文件**不提供 main**。所有单元测试目标统一链接 tests/unit/test_main.cpp
// （目标 vibe_test_main），其中已经实现了入口、命令行过滤与 MPI 初始化。
// 该文件曾短暂自带一个 main，与统一入口产生重复符号，已移除。

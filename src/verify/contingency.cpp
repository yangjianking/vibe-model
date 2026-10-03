#include "vibe/verify/contingency.hpp"

#include <sstream>

namespace vibe::verify {

// ---------------------------------------------------------------------------
// 列联表构造
// ---------------------------------------------------------------------------

// 由 0/1 二值序列构造；两序列必须等长，非零视为事件。O(n)。
ContingencyTable ContingencyTable::from_binary(const std::vector<int>& f,
                                               const std::vector<int>& o) {
  if (f.size() != o.size()) {
    throw DimensionError("ContingencyTable::from_binary: 长度不等 (" +
                         std::to_string(f.size()) + " vs " + std::to_string(o.size()) + ")");
  }
  ContingencyTable t;
  for (std::size_t i = 0; i < f.size(); ++i) {
    const bool fe = f[i] != 0;
    const bool oe = o[i] != 0;
    if (fe && oe) {
      t.hits += Real(1);
    } else if (!fe && !oe) {
      t.correct_negatives += Real(1);
    } else if (fe && !oe) {
      t.false_alarms += Real(1);
    } else {
      t.misses += Real(1);
    }
  }
  return t;
}

// 由连续序列 + 阈值构造（f >= threshold 记为事件）；NaN 样本对跳过。O(n)。
ContingencyTable ContingencyTable::from_arrays(const std::vector<Real>& f,
                                               const std::vector<Real>& o,
                                               Real threshold) {
  if (f.size() != o.size()) {
    throw DimensionError("ContingencyTable::from_arrays: 长度不等 (" +
                         std::to_string(f.size()) + " vs " + std::to_string(o.size()) + ")");
  }
  ContingencyTable t;
  for (std::size_t i = 0; i < f.size(); ++i) {
    if (is_missing(f[i]) || is_missing(o[i])) continue;
    const bool fe = f[i] >= threshold;
    const bool oe = o[i] >= threshold;
    if (fe && oe) {
      t.hits += Real(1);
    } else if (!fe && !oe) {
      t.correct_negatives += Real(1);
    } else if (fe && !oe) {
      t.false_alarms += Real(1);
    } else {
      t.misses += Real(1);
    }
  }
  return t;
}

std::string ContingencyTable::to_string() const {
  std::ostringstream os;
  os << "H=" << hits << " M=" << misses << " FA=" << false_alarms
     << " CN=" << correct_negatives;
  return os.str();
}

// ---------------------------------------------------------------------------
// 分类评分
// ---------------------------------------------------------------------------

// 见 contingency.hpp：O(n) 时间、O(1) 空间；未定义的评分保持 kNaN。
Scores compute_categorical(const std::vector<Real>& f, const std::vector<Real>& o,
                           Real threshold) {
  const ContingencyTable t = ContingencyTable::from_arrays(f, o, threshold);
  // 不适用字段置 0；适用但退化的评分保持 NaN（见 scores.hpp 约定）。
  Scores s = zero_scores(static_cast<std::size_t>(t.total()));
  s.pod = t.pod();
  s.far = t.far();
  s.csi = t.csi();
  s.ets = t.ets();
  s.frequency_bias = t.frequency_bias();
  s.odds_ratio_skill = t.orss();
  s.tss = t.tss();
  s.sedi = t.sedi();
  s.accuracy = t.accuracy();
  return s;
}

// ---------------------------------------------------------------------------
// ScoreAccumulator 的列联表视图
// ---------------------------------------------------------------------------

// 把在线计数转为列联表。O(1)。
ContingencyTable ScoreAccumulator::contingency() const {
  return ContingencyTable::from_counts(static_cast<Real>(hits_),
                                       static_cast<Real>(misses_),
                                       static_cast<Real>(false_alarms_),
                                       static_cast<Real>(correct_negatives_));
}

}  // namespace vibe::verify

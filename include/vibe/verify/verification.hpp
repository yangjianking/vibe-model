#pragma once
/// @file verification.hpp
/// @brief 检验框架：配置、批量评分驱动、结果序列化与时间序列聚合。
///
/// 设计（与 docs/design/00_architecture.md 第 3、9、10 节一致）
/// -----------------------------------------------------------
///   * \c Verifier **只接受内存中的数组**，不接触任何文件格式；
///     .vibebin / NetCDF 的读取由 \c verify_cli.cpp 或 \c vibe::io 负责。
///     这样 \c verify 不依赖 \c io，符合架构的"外部文件格式只在 io 层出现"约束。
///   * \c VerifyConfig 自带 \c from_kv：从"键 -> 值"字符串映射构造，
///     不依赖 \c vibe::config 的内部 YAML 解析器（避免模块环依赖）。
///     \c vibe::config 只需把已解析的键值对转发进来即可。
///   * \c VerificationResult 采用**长表（long format）**：
///     每行 = (变量, 时间, 阈值, 半径, 指标, 数值, n)，可直接落 CSV，
///     也便于 Python 侧 \c vibe_post.verify 用 pandas 透视。
///
/// 序列化约定
/// ----------
///   CSV  : 首行列名 \c variable,time,type,threshold,radius,metric,value,n
///   JSON : \c {"records":[{"variable":...,"value":...},...]}，UTF-8，数值用
///          \c %.17g 保证 double 往返无损；NaN 写成 \c null（JSON 无 NaN 字面量）。
///
/// 时间聚合
/// --------
///   时间统一为 **Unix epoch 秒（UTC）**。日/月键由纯算法从秒数换算
///   （Howard Hinnant 的 civil_from_days 算法），不依赖本地时区与 \c <ctime>。
///
/// 复杂度：单时次 O(nx*ny)（连续/分类）~ O(nx*ny*R)（空间多尺度）；
///         聚合 O(#records log #records)（中位数需要排序）。

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/verify/contingency.hpp"
#include "vibe/verify/probabilistic.hpp"
#include "vibe/verify/scores.hpp"
#include "vibe/verify/spatial.hpp"

namespace vibe::verify {

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

/// 单个检验变量的配置。
struct VerifyVariable {
  std::string name;                 ///< 预报变量名
  MatchType type = MatchType::Continuous;
  std::vector<Real> thresholds;     ///< 分类/空间检验的事件阈值列表
  std::vector<int> radii;           ///< 空间检验的邻域半径列表
  std::string obs_name;             ///< 对应观测变量名（空 = 与 name 同名）
  std::string climatology_name;     ///< 气候态变量名（空 = 无）
  Real scale = Real(1);             ///< 单位换算因子（预报 -> 观测单位）
  Real fss_target = Real(0.5);      ///< 最小可分辨尺度阈值
};

/// 检验配置。
///
/// 支持两种载入路径：
///   1. \c from_kv(vector<pair<string,string>>)：由 config 模块转发已解析的键值；
///   2. \c from_string("k=v\nk=v\n...")：CLI 与测试使用的极简文本形式。
struct VerifyConfig {
  std::vector<VerifyVariable> variables;   ///< 空 = 使用 \c default_variable
  VerifyVariable default_variable;         ///< 未在 variables 中匹配时使用的模板
  int n_bins = 10;                         ///< 概率/可靠性分箱数
  BinScheme bin_scheme = BinScheme::EqualWidth;
  int fss_max_radius = 10;                 ///< 多尺度 FSS 最大半径
  bool compute_continuous = true;
  bool compute_categorical = true;
  bool compute_probabilistic = true;
  bool compute_spatial = true;
  bool compute_double_penalty = false;     ///< 双惩罚诊断代价较高，默认关闭
  std::string csv_delimiter = ",";
  std::string missing_token = "NaN";       ///< CSV 中缺失值写法

  /// 从键值映射构造。识别的键：
  ///   variables              : 逗号分隔的变量名列表
  ///   type.<var>             : continuous|categorical|probabilistic|spatial
  ///   thresholds.<var>       : 逗号分隔阈值
  ///   radii.<var> / radii     : 逗号分隔半径
  ///   obs.<var>              : 观测变量名
  ///   climatology.<var>      : 气候态变量名
  ///   scale.<var>            : 单位换算因子
  ///   n_bins / bin_scheme / fss_max_radius / fss_target / csv_delimiter /
  ///   missing_token / compute_continuous / compute_categorical /
  ///   compute_probabilistic / compute_spatial / compute_double_penalty
  /// 未知键抛出 \c ConfigError（宁可显式失败也不要静默忽略拼写错误）。
  static VerifyConfig from_kv(const std::vector<std::pair<std::string, std::string>>& kv);

  /// \c from_kv 的文本包装：每行一个 \c key=value，忽略空行与 \c '#' 注释。
  static VerifyConfig from_string(const std::string& text);

  /// 查找变量配置；找不到返回 default_variable 的副本（name 置为查询名）。
  VerifyVariable resolve(const std::string& name) const;

  /// 逗号分隔的默认阈值列表。
  static std::vector<Real> parse_real_list(const std::string& text);
  /// 逗号分隔的整数列表。
  static std::vector<int> parse_int_list(const std::string& text);
  /// "true/1/yes/on" 解析为 true。
  static bool parse_bool(const std::string& text);
};

// ---------------------------------------------------------------------------
// 输入样本（内存数组）
// ---------------------------------------------------------------------------

/// 一个时次、一个变量的预报/观测对。
struct FieldSample {
  std::string variable;
  Int nx = 0, ny = 0, nz = 1;
  Real time = Real(0);       ///< Unix epoch 秒（UTC）
  Real dx = Real(1), dy = Real(1);
  std::vector<Real> forecast;      ///< 长度 nx*ny*nz（行主序，层优先）
  std::vector<Real> observation;   ///< 长度 nx*ny*nz
  std::vector<Real> climatology;   ///< 可选，长度 0 / 1 / nx*ny*nz
  Real scale = Real(1);            ///< 预报 -> 观测单位换算

  /// 水平层点数。
  std::size_t level_size() const noexcept {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  }
  /// 总点数。
  std::size_t total_size() const noexcept { return level_size() * static_cast<std::size_t>(nz); }
  /// 维度自检；不一致抛出 \c DimensionError。
  void validate() const;
};

// ---------------------------------------------------------------------------
// 结果
// ---------------------------------------------------------------------------

/// 长表检验结果。
struct VerificationResult {
  /// 单条结果记录。
  struct Record {
    std::string variable;
    std::string type;
    Real time = kNaN;        ///< Unix epoch 秒
    Real threshold = kNaN;   ///< 仅分类/空间检验有效
    int radius = -1;         ///< 仅空间检验有效
    std::string metric;
    Real value = kNaN;
    std::size_t n = 0;
  };

  std::vector<Record> records;

  /// 追加一条记录。
  void add(const Record& r) { records.push_back(r); }

  /// 把一次 \c Scores 展开为多条记录（指标名 -> 数值）。
  void append_scores(const Scores& s, const std::string& variable,
                     const std::string& type, Real time,
                     Real threshold = kNaN, int radius = -1);

  std::size_t size() const noexcept { return records.size(); }
  void clear() noexcept { records.clear(); }

  /// 按 (变量, 指标) 提取数值序列（时间升序保持原顺序）。
  std::vector<Real> values(const std::string& variable, const std::string& metric) const;

  /// 序列化为 CSV 文本（含表头）。O(#records)。
  std::string to_csv(const std::string& delimiter = ",",
                     const std::string& missing_token = "NaN") const;

  /// 序列化为 JSON 文本。NaN 输出为 \c null。O(#records)。
  std::string to_json() const;

  /// 写入 CSV 文件；失败抛 \c IoError。
  void write_csv(const std::string& path, const std::string& delimiter = ",",
                 const std::string& missing_token = "NaN") const;

  /// 写入 JSON 文件；失败抛 \c IoError。
  void write_json(const std::string& path) const;
};

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// 检验驱动：按配置对内存中的场批量计算全部评分。
class Verifier {
 public:
  explicit Verifier(VerifyConfig cfg = VerifyConfig{});

  const VerifyConfig& config() const noexcept { return cfg_; }

  /// 单时次检验（按该变量的 MatchType 分派）。O(N) ~ O(N*R)。
  VerificationResult run(const FieldSample& s) const;

  /// 批量多时次检验，结果按输入顺序累积在一个长表中。O(n_samples * N)。
  VerificationResult run(const std::vector<FieldSample>& samples) const;

  /// 只算连续评分（不论配置的 MatchType）。
  Scores score_continuous(const FieldSample& s) const;
  /// 只算分类评分（指定阈值）。
  Scores score_categorical(const FieldSample& s, Real threshold) const;
  /// 只算概率评分。
  Scores score_probabilistic(const FieldSample& s) const;
  /// 只算空间评分（指定阈值与半径）。
  Scores score_spatial(const FieldSample& s, Real threshold, int radius) const;

 private:
  /// 连续评分的公共入口：从 FieldSample 聚合扁平序列（含尺度换算与缺测）。
  Scores continuous_from_sample(const FieldSample& s) const;
  VerifyConfig cfg_;
};

// ---------------------------------------------------------------------------
// 时间聚合
// ---------------------------------------------------------------------------

/// 聚合时间桶。
enum class TimeBucket { All, Daily, Monthly };

inline const char* to_string(TimeBucket b) noexcept {
  switch (b) {
    case TimeBucket::All:     return "all";
    case TimeBucket::Daily:   return "daily";
    case TimeBucket::Monthly: return "monthly";
  }
  return "unknown";
}

/// 一条聚合结果。
struct AggregatedScore {
  std::string variable;
  std::string metric;
  std::string period;
  Real mean = kNaN;
  Real min = kNaN;
  Real max = kNaN;
  Real stddev = kNaN;
  Real median = kNaN;
  std::size_t count = 0;
};

/// 逐时次评分 -> 日/月统计的聚合器。
///
/// 评分本身是逐时次的标量序列（数量远小于样本点数），因此这里保存
/// 每个 (变量, 指标, 桶) 的数值列表以支持中位数与分位数；若只需均值/方差
/// 可改写为在线累加（见 docs/design/09_verification.md 的复杂度讨论）。
class Aggregator {
 public:
  Aggregator() = default;

  /// 累积一个结果集。
  void add(const VerificationResult& r);
  /// 累积一条记录。
  void add(const VerificationResult::Record& rec);
  void reset();

  std::size_t record_count() const noexcept { return count_; }

  /// 聚合并按 (period, variable, metric) 字典序输出。
  std::vector<AggregatedScore> aggregate(TimeBucket bucket = TimeBucket::All) const;

  /// 聚合结果序列化为 CSV。
  std::string to_csv(TimeBucket bucket = TimeBucket::All) const;

  /// 把 epoch 秒换算为桶键：All -> "all"，Daily -> "YYYY-MM-DD"，
  /// Monthly -> "YYYY-MM"。UTC，无时区与夏令时歧义。O(1)。
  static std::string bucket_key(Real seconds_since_epoch, TimeBucket bucket);

 private:
  /// 逐条评分记录；时间在 aggregate() 时按所选桶归组，因此这里保存原始 time。
  struct Entry {
    std::string variable;
    std::string metric;
    Real time = kNaN;
    Real value = kNaN;
  };
  std::vector<Entry> entries_;
  std::size_t count_ = 0;
};

/// 由 Unix epoch 秒计算 (year, month, day)（UTC），Hinnant 的 civil_from_days 算法。O(1)。
void civil_from_epoch(Real seconds_since_epoch, int& year, int& month, int& day);

}  // namespace vibe::verify

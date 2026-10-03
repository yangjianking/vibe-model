#include "vibe/verify/verification.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace vibe::verify {
namespace {

// ---------------------------- 字符串工具 -----------------------------------

std::string trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::string lower_copy(const std::string& s) {
  std::string r = s;
  for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return r;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (const char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

/// 数值格式化：缺测 -> token，否则 %.12g。O(1)。
std::string format_real(Real v, const std::string& missing_token = "NaN") {
  if (is_missing(v)) return missing_token;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.12g", static_cast<double>(v));
  return std::string(buf);
}

/// JSON 字符串转义（含控制字符）。O(len)。
std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (const char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

/// CSV 字段转义：含分隔符/引号/换行时用双引号包裹。
std::string csv_escape(const std::string& s, const std::string& delim) {
  const bool need = s.find(delim) != std::string::npos || s.find('"') != std::string::npos ||
                    s.find('\n') != std::string::npos || s.find('\r') != std::string::npos;
  if (!need) return s;
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"') out += "\"\"";
    else out.push_back(c);
  }
  out += "\"";
  return out;
}

/// 把 FieldSample 展平为 (预报*scale, 观测)，跳过缺测对。O(N)。
void flatten_sample(const FieldSample& s, std::vector<Real>& f, std::vector<Real>& o) {
  s.validate();
  const std::size_t n = s.total_size();
  f.clear();
  o.clear();
  f.reserve(n);
  o.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    if (is_missing(s.forecast[k]) || is_missing(s.observation[k])) continue;
    f.push_back(s.forecast[k] * s.scale);
    o.push_back(s.observation[k]);
  }
}

/// 变量属性（按变量名分组时的中间结构）。
struct VarBuilder {
  std::string name;
  MatchType type = MatchType::Continuous;
  bool type_set = false;
  std::vector<Real> thresholds;
  std::vector<int> radii;
  std::string obs_name;
  std::string clim_name;
  Real scale = Real(1);
  Real fss_target = Real(0.5);
};

}  // namespace

// ---------------------------------------------------------------------------
// VerifyConfig
// ---------------------------------------------------------------------------

std::vector<Real> VerifyConfig::parse_real_list(const std::string& text) {
  std::vector<Real> out;
  const std::string t = trim(text);
  if (t.empty()) return out;
  for (const std::string& part : split(t, ',')) {
    const std::string p = trim(part);
    if (p.empty()) continue;
    try {
      std::size_t used = 0;
      const double v = std::stod(p, &used);
      if (used != p.size()) throw std::invalid_argument("trailing");
      out.push_back(static_cast<Real>(v));
    } catch (const std::exception&) {
      throw ConfigError("阈值/数值列表中存在非法数值: '" + p + "'");
    }
  }
  return out;
}

std::vector<int> VerifyConfig::parse_int_list(const std::string& text) {
  std::vector<int> out;
  const std::string t = trim(text);
  if (t.empty()) return out;
  for (const std::string& part : split(t, ',')) {
    const std::string p = trim(part);
    if (p.empty()) continue;
    try {
      std::size_t used = 0;
      const long v = std::stol(p, &used);
      if (used != p.size()) throw std::invalid_argument("trailing");
      out.push_back(static_cast<int>(v));
    } catch (const std::exception&) {
      throw ConfigError("半径列表中存在非法整数: '" + p + "'");
    }
  }
  return out;
}

bool VerifyConfig::parse_bool(const std::string& text) {
  const std::string t = lower_copy(trim(text));
  if (t == "true" || t == "1" || t == "yes" || t == "on") return true;
  if (t == "false" || t == "0" || t == "no" || t == "off") return false;
  throw ConfigError("无法解析布尔值: '" + text + "'");
}

VerifyConfig VerifyConfig::from_kv(
    const std::vector<std::pair<std::string, std::string>>& kv) {
  VerifyConfig cfg;
  std::map<std::string, VarBuilder> builders;
  std::vector<std::string> order;

  // 第一遍：变量清单（保证 variables 先于逐变量属性生效）。
  for (const auto& item : kv) {
    const std::string key = trim(item.first);
    const std::string val = trim(item.second);
    if (key == "variables") {
      for (const std::string& name : split(val, ',')) {
        const std::string n = trim(name);
        if (n.empty()) continue;
        if (builders.find(n) == builders.end()) {
          builders[n] = VarBuilder{n};
          order.push_back(n);
        }
      }
    }
  }

  auto builder_for = [&](const std::string& name) -> VarBuilder& {
    auto it = builders.find(name);
    if (it == builders.end()) {
      builders[name] = VarBuilder{name};
      order.push_back(name);
      return builders[name];
    }
    return it->second;
  };

  // 第二遍：逐项解析。
  for (const auto& item : kv) {
    const std::string key = trim(item.first);
    const std::string val = trim(item.second);
    if (key.empty()) continue;
    if (key == "variables") continue;

    const std::size_t dot = key.find('.');
    const std::string prefix = dot == std::string::npos ? key : key.substr(0, dot);
    const std::string var = dot == std::string::npos ? std::string() : key.substr(dot + 1);

    if (prefix == "type") {
      if (var.empty()) throw ConfigError("键 'type' 缺少变量名后缀，例如 type.t2");
      VarBuilder& b = builder_for(var);
      b.type = match_type_from_string(val);
      b.type_set = true;
    } else if (prefix == "thresholds") {
      if (var.empty()) throw ConfigError("键 'thresholds' 缺少变量名后缀");
      builder_for(var).thresholds = parse_real_list(val);
    } else if (prefix == "radii") {
      if (var.empty()) {
        cfg.default_variable.radii = parse_int_list(val);
      } else {
        builder_for(var).radii = parse_int_list(val);
      }
    } else if (prefix == "obs") {
      if (var.empty()) throw ConfigError("键 'obs' 缺少变量名后缀");
      builder_for(var).obs_name = val;
    } else if (prefix == "climatology") {
      if (var.empty()) throw ConfigError("键 'climatology' 缺少变量名后缀");
      builder_for(var).clim_name = val;
    } else if (prefix == "scale") {
      if (var.empty()) {
        cfg.default_variable.scale = parse_real_list(val).empty() ? Real(1) : parse_real_list(val)[0];
      } else {
        const std::vector<Real> v = parse_real_list(val);
        if (v.size() != 1) throw ConfigError("键 'scale." + var + "' 需要单个数值");
        builder_for(var).scale = v[0];
      }
    } else if (prefix == "fss_target") {
      const std::vector<Real> v = parse_real_list(val);
      if (v.size() != 1) throw ConfigError("键 'fss_target' 需要单个数值");
      if (var.empty()) cfg.default_variable.fss_target = v[0];
      else builder_for(var).fss_target = v[0];
    } else if (key == "n_bins") {
      const std::vector<int> v = parse_int_list(val);
      if (v.size() != 1 || v[0] <= 0) throw ConfigError("键 'n_bins' 需要单个正整数");
      cfg.n_bins = v[0];
    } else if (key == "bin_scheme") {
      if (val == "equal_width" || val == "EqualWidth") cfg.bin_scheme = BinScheme::EqualWidth;
      else if (val == "equal_frequency" || val == "EqualFrequency") cfg.bin_scheme = BinScheme::EqualFrequency;
      else throw ConfigError("键 'bin_scheme' 只允许 equal_width / equal_frequency");
    } else if (key == "fss_max_radius") {
      const std::vector<int> v = parse_int_list(val);
      if (v.size() != 1 || v[0] < 0) throw ConfigError("键 'fss_max_radius' 需要非负整数");
      cfg.fss_max_radius = v[0];
    } else if (key == "csv_delimiter") {
      cfg.csv_delimiter = val;
    } else if (key == "missing_token") {
      cfg.missing_token = val;
    } else if (key == "compute_continuous") {
      cfg.compute_continuous = parse_bool(val);
    } else if (key == "compute_categorical") {
      cfg.compute_categorical = parse_bool(val);
    } else if (key == "compute_probabilistic") {
      cfg.compute_probabilistic = parse_bool(val);
    } else if (key == "compute_spatial") {
      cfg.compute_spatial = parse_bool(val);
    } else if (key == "compute_double_penalty") {
      cfg.compute_double_penalty = parse_bool(val);
    } else {
      throw ConfigError("VerifyConfig::from_kv: 未知键 '" + key + "'");
    }
  }

  for (const std::string& name : order) {
    const VarBuilder& b = builders[name];
    VerifyVariable v;
    v.name = b.name;
    v.type = b.type;
    v.thresholds = b.thresholds;
    v.radii = b.radii;
    v.obs_name = b.obs_name;
    v.climatology_name = b.clim_name;
    v.scale = b.scale;
    v.fss_target = b.fss_target;
    cfg.variables.push_back(v);
  }
  return cfg;
}

VerifyConfig VerifyConfig::from_string(const std::string& text) {
  std::vector<std::pair<std::string, std::string>> kv;
  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    const std::string t = trim(line);
    if (t.empty()) continue;
    const std::size_t eq = t.find('=');
    if (eq == std::string::npos) {
      throw ConfigError("VerifyConfig::from_string: 行缺少 '=' : '" + t + "'");
    }
    kv.emplace_back(trim(t.substr(0, eq)), trim(t.substr(eq + 1)));
  }
  return from_kv(kv);
}

VerifyVariable VerifyConfig::resolve(const std::string& name) const {
  for (const VerifyVariable& v : variables) {
    if (v.name == name) return v;
  }
  VerifyVariable v = default_variable;
  v.name = name;
  return v;
}

// ---------------------------------------------------------------------------
// FieldSample
// ---------------------------------------------------------------------------

void FieldSample::validate() const {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw DimensionError("FieldSample: 非正网格尺寸 (nx=" + std::to_string(nx) +
                         ", ny=" + std::to_string(ny) + ", nz=" + std::to_string(nz) + ")");
  }
  const std::size_t n = total_size();
  if (forecast.size() != n) {
    throw DimensionError("FieldSample '" + variable + "': 预报长度 " +
                         std::to_string(forecast.size()) + " 与 nx*ny*nz=" +
                         std::to_string(n) + " 不符");
  }
  if (observation.size() != n) {
    throw DimensionError("FieldSample '" + variable + "': 观测长度 " +
                         std::to_string(observation.size()) + " 与 nx*ny*nz=" +
                         std::to_string(n) + " 不符");
  }
  if (!climatology.empty() && climatology.size() != 1 && climatology.size() != n) {
    throw DimensionError("FieldSample '" + variable +
                         "': 气候态长度必须为 0 / 1 / nx*ny*nz");
  }
}

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

void VerificationResult::append_scores(const Scores& s, const std::string& variable,
                                       const std::string& type, Real time,
                                       Real threshold, int radius) {
  for (const auto& entry : score_table(s)) {
    Record r;
    r.variable = variable;
    r.type = type;
    r.time = time;
    r.threshold = threshold;
    r.radius = radius;
    r.metric = entry.first;
    r.value = entry.second;
    r.n = s.n;
    records.push_back(r);
  }
}

std::vector<Real> VerificationResult::values(const std::string& variable,
                                             const std::string& metric) const {
  std::vector<Real> out;
  for (const Record& r : records) {
    if (r.variable == variable && r.metric == metric && !is_missing(r.value)) {
      out.push_back(r.value);
    }
  }
  return out;
}

std::string VerificationResult::to_csv(const std::string& delimiter,
                                       const std::string& missing_token) const {
  std::ostringstream os;
  os << "variable" << delimiter << "time" << delimiter << "type" << delimiter
     << "threshold" << delimiter << "radius" << delimiter << "metric" << delimiter
     << "value" << delimiter << "n\n";
  for (const Record& r : records) {
    os << csv_escape(r.variable, delimiter) << delimiter
       << format_real(r.time, missing_token) << delimiter
       << csv_escape(r.type, delimiter) << delimiter
       << format_real(r.threshold, missing_token) << delimiter
       << r.radius << delimiter
       << csv_escape(r.metric, delimiter) << delimiter
       << format_real(r.value, missing_token) << delimiter
       << r.n << '\n';
  }
  return os.str();
}

std::string VerificationResult::to_json() const {
  std::ostringstream os;
  os << "{\n  \"records\": [\n";
  for (std::size_t i = 0; i < records.size(); ++i) {
    const Record& r = records[i];
    os << "    {\"variable\": \"" << json_escape(r.variable) << "\", "
       << "\"time\": " << format_real(r.time, "null") << ", "
       << "\"type\": \"" << json_escape(r.type) << "\", "
       << "\"threshold\": " << format_real(r.threshold, "null") << ", "
       << "\"radius\": " << r.radius << ", "
       << "\"metric\": \"" << json_escape(r.metric) << "\", "
       << "\"value\": " << format_real(r.value, "null") << ", "
       << "\"n\": " << r.n << "}";
    if (i + 1 < records.size()) os << ',';
    os << '\n';
  }
  os << "  ]\n}\n";
  return os.str();
}

void VerificationResult::write_csv(const std::string& path, const std::string& delimiter,
                                   const std::string& missing_token) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw IoError("无法写入 CSV 文件: " + path);
  const std::string text = to_csv(delimiter, missing_token);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!out) throw IoError("写入 CSV 失败: " + path);
}

void VerificationResult::write_json(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw IoError("无法写入 JSON 文件: " + path);
  const std::string text = to_json();
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!out) throw IoError("写入 JSON 失败: " + path);
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

Verifier::Verifier(VerifyConfig cfg) : cfg_(std::move(cfg)) {}

Scores Verifier::continuous_from_sample(const FieldSample& s) const {
  s.validate();
  const std::size_t n = s.total_size();
  const bool has_clim = !s.climatology.empty();
  const bool clim_const = has_clim && s.climatology.size() == 1;
  std::vector<Real> f, o, clim;
  f.reserve(n);
  o.reserve(n);
  if (has_clim) clim.reserve(clim_const ? 1 : n);
  for (std::size_t k = 0; k < n; ++k) {
    if (is_missing(s.forecast[k]) || is_missing(s.observation[k])) continue;
    Real c = Real(0);
    if (has_clim) {
      c = clim_const ? s.climatology[0] : s.climatology[k];
      if (is_missing(c)) continue;
    }
    f.push_back(s.forecast[k] * s.scale);
    o.push_back(s.observation[k]);
    if (has_clim) clim.push_back(c);
  }
  if (has_clim) return compute_continuous(f, o, &clim);
  return compute_continuous(f, o, nullptr);
}

Scores Verifier::score_continuous(const FieldSample& s) const {
  return continuous_from_sample(s);
}

Scores Verifier::score_categorical(const FieldSample& s, Real threshold) const {
  std::vector<Real> f, o;
  flatten_sample(s, f, o);
  return compute_categorical(f, o, threshold);
}

Scores Verifier::score_probabilistic(const FieldSample& s) const {
  std::vector<Real> f, o;
  flatten_sample(s, f, o);
  return compute_probabilistic(f, o, cfg_.n_bins, cfg_.bin_scheme);
}

Scores Verifier::score_spatial(const FieldSample& s, Real threshold, int radius) const {
  s.validate();
  const std::size_t L = s.level_size();
  // 空间评分是 2D 的：取最低层（k = 0）。多层检验请逐层调用。
  std::vector<Real> f(s.forecast.begin(), s.forecast.begin() + static_cast<std::ptrdiff_t>(L));
  std::vector<Real> o(s.observation.begin(),
                      s.observation.begin() + static_cast<std::ptrdiff_t>(L));
  return compute_fractional_skill(f, o, static_cast<int>(s.nx), static_cast<int>(s.ny),
                                  threshold, radius);
}

VerificationResult Verifier::run(const FieldSample& s) const {
  s.validate();
  const VerifyVariable v = cfg_.resolve(s.variable);
  VerificationResult res;
  switch (v.type) {
    case MatchType::Continuous: {
      if (!cfg_.compute_continuous) break;
      res.append_scores(continuous_from_sample(s), s.variable, "continuous", s.time);
      break;
    }
    case MatchType::Categorical: {
      if (!cfg_.compute_categorical) break;
      const std::vector<Real> th =
          v.thresholds.empty() ? std::vector<Real>{Real(0)} : v.thresholds;
      std::vector<Real> f, o;
      flatten_sample(s, f, o);
      for (const Real t : th) {
        res.append_scores(compute_categorical(f, o, t), s.variable, "categorical", s.time, t, -1);
      }
      break;
    }
    case MatchType::Probabilistic: {
      if (!cfg_.compute_probabilistic) break;
      res.append_scores(score_probabilistic(s), s.variable, "probabilistic", s.time);
      break;
    }
    case MatchType::Spatial: {
      if (!cfg_.compute_spatial) break;
      std::vector<Real> th =
          v.thresholds.empty() ? std::vector<Real>{Real(0)} : v.thresholds;
      std::vector<int> radii = v.radii;
      if (radii.empty()) {
        for (const int r : {0, 1, 2, 4, 8, 16}) {
          if (r <= cfg_.fss_max_radius) radii.push_back(r);
        }
        if (radii.empty()) radii.push_back(0);
      }
      for (const Real t : th) {
        for (const int r : radii) {
          res.append_scores(score_spatial(s, t, r), s.variable, "spatial", s.time, t, r);
        }
      }
      break;
    }
  }
  return res;
}

VerificationResult Verifier::run(const std::vector<FieldSample>& samples) const {
  VerificationResult all;
  for (const FieldSample& s : samples) {
    const VerificationResult r = run(s);
    all.records.insert(all.records.end(), r.records.begin(), r.records.end());
  }
  return all;
}

// ---------------------------------------------------------------------------
// 时间换算（UTC，Hinnant civil_from_days 算法）
// ---------------------------------------------------------------------------

void civil_from_epoch(Real seconds_since_epoch, int& year, int& month, int& day) {
  const long long secs = static_cast<long long>(std::floor(seconds_since_epoch));
  const long long days = secs >= 0 ? secs / 86400 : (secs - 86399) / 86400;
  const long long z = days + 719468;
  const long long era = (z >= 0 ? z : z - 146096) / 146097;
  const long long doe = z - era * 146097;                                   // [0, 146096]
  const long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const long long y = yoe + era * 400;
  const long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);            // [0, 365]
  const long long mp = (5 * doy + 2) / 153;                                 // [0, 11]
  const long long d = doy - (153 * mp + 2) / 5 + 1;                         // [1, 31]
  const long long m = mp + (mp < 10 ? 3 : -9);                              // [1, 12]
  year = static_cast<int>(y + (m <= 2 ? 1 : 0));
  month = static_cast<int>(m);
  day = static_cast<int>(d);
}

// ---------------------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------------------

void Aggregator::add(const VerificationResult& r) {
  for (const VerificationResult::Record& rec : r.records) add(rec);
}

void Aggregator::add(const VerificationResult::Record& rec) {
  if (is_missing(rec.value)) return;  // 未定义的评分不参与聚合
  Entry e;
  e.variable = rec.variable;
  e.metric = rec.metric;
  e.time = rec.time;
  e.value = rec.value;
  entries_.push_back(std::move(e));
  ++count_;
}

void Aggregator::reset() {
  entries_.clear();
  count_ = 0;
}

std::string Aggregator::bucket_key(Real seconds_since_epoch, TimeBucket bucket) {
  if (bucket == TimeBucket::All) return "all";
  int y = 0, m = 0, d = 0;
  civil_from_epoch(seconds_since_epoch, y, m, d);
  char buf[32];
  if (bucket == TimeBucket::Daily) {
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d);
  } else {
    std::snprintf(buf, sizeof(buf), "%04d-%02d", y, m);
  }
  return std::string(buf);
}

std::vector<AggregatedScore> Aggregator::aggregate(TimeBucket bucket) const {
  struct Group {
    std::string period;
    std::string variable;
    std::string metric;
    std::vector<Real> values;
  };
  std::map<std::string, Group> groups;
  for (const Entry& e : entries_) {
    const std::string period = bucket_key(e.time, bucket);
    const std::string key = period + '\x1f' + e.variable + '\x1f' + e.metric;
    auto it = groups.find(key);
    if (it == groups.end()) {
      Group g;
      g.period = period;
      g.variable = e.variable;
      g.metric = e.metric;
      g.values.push_back(e.value);
      groups.emplace(key, std::move(g));
    } else {
      it->second.values.push_back(e.value);
    }
  }

  std::vector<AggregatedScore> out;
  out.reserve(groups.size());
  for (const auto& kv : groups) {
    const Group& g = kv.second;
    if (g.values.empty()) continue;
    AggregatedScore a;
    a.period = g.period;
    a.variable = g.variable;
    a.metric = g.metric;
    a.count = g.values.size();
    const auto mm = std::minmax_element(g.values.begin(), g.values.end());
    a.min = *mm.first;
    a.max = *mm.second;
    Real sum = Real(0);
    for (const Real v : g.values) sum += v;
    a.mean = sum / static_cast<Real>(g.values.size());
    a.stddev = stddev_value(g.values);
    std::vector<Real> sorted = g.values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();
    a.median = (n % 2 == 1) ? sorted[n / 2]
                            : Real(0.5) * (sorted[n / 2 - 1] + sorted[n / 2]);
    out.push_back(a);
  }
  return out;
}

std::string Aggregator::to_csv(TimeBucket bucket) const {
  std::ostringstream os;
  os << "period,variable,metric,mean,min,max,stddev,median,count\n";
  for (const AggregatedScore& a : aggregate(bucket)) {
    os << a.period << ',' << csv_escape(a.variable, ",") << ','
       << csv_escape(a.metric, ",") << ',' << format_real(a.mean) << ','
       << format_real(a.min) << ',' << format_real(a.max) << ','
       << format_real(a.stddev) << ',' << format_real(a.median) << ','
       << a.count << '\n';
  }
  return os.str();
}

}  // namespace vibe::verify

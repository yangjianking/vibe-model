/// @file mini_yaml.cpp
/// @brief 极简 YAML 子集解析器实现（无外部依赖，见 include/vibe/config/yaml_node.hpp）。
///
/// 算法与复杂度
/// ------------
///   * 词法阶段 prepare()：逐字符扫描，剥离注释、检查制表符缩进、计算缩进量，
///     每个字符只访问常数次，因此 O(N)（N = 输入字符数），空间 O(N)。
///   * 语法阶段：LL(1) 递归下降，缩进即文法层级，parse_mapping /
///     parse_sequence / parse_block 互相递归，每一行只消费一次，O(N)。
///   * 行内 [a, b] / {a: 1} 用带括号深度与引号状态的顶层切分解析，O(n)。
///
/// 文献：[B3] Warner (2011) 第 2 章（模式配置与数值参数的组织）。

#include "vibe/config/yaml_node.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"

namespace vibe::config {
namespace {

// ===========================================================================
// 基础字符串工具
// ===========================================================================

std::string trim(const std::string& s) {
  Size b = 0;
  Size e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) --e;
  return s.substr(b, e - b);
}

std::string lower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

/// 统一的错误构造：带 origin、行号与出错行上下文。
[[noreturn]] void fail(const std::string& origin, int line_no, const std::string& raw_line,
                       const std::string& message) {
  std::ostringstream os;
  os << origin;
  if (line_no > 0) os << ':' << line_no;
  os << ": " << message;
  if (!raw_line.empty()) os << "\n  | " << raw_line;
  throw ConfigError(os.str());
}

/// 去掉注释：只有位于行首或前面是空白的 # 才生效（YAML 规则），
/// 引号内部与 abc#def 这类紧贴的 # 都不算注释。
std::string strip_comment(const std::string& line) {
  std::string out;
  out.reserve(line.size());
  char quote = '\0';
  bool escaped = false;
  for (Size i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quote != '\0') {
      out.push_back(c);
      if (escaped) {
        escaped = false;
      } else if (quote == '"' && c == '\\') {
        escaped = true;
      } else if (c == quote) {
        quote = '\0';
      }
      continue;
    }
    if (c == '\'' || c == '"') {
      quote = c;
      out.push_back(c);
      continue;
    }
    if (c == '#' && (i == 0 || std::isspace(static_cast<unsigned char>(line[i - 1])) != 0)) break;
    out.push_back(c);
  }
  return out;
}

/// 右侧去空白（保留左侧缩进）。
std::string rstrip(const std::string& s) {
  Size e = s.size();
  while (e > 0 && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) --e;
  return s.substr(0, e);
}

/// 找到顶层 key: 的冒号位置；跳过引号与 []{} 内部。
Size find_key_colon(const std::string& s) {
  int depth = 0;
  char quote = '\0';
  bool escaped = false;
  for (Size i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (quote != '\0') {
      if (escaped) {
        escaped = false;
      } else if (quote == '"' && c == '\\') {
        escaped = true;
      } else if (c == quote) {
        quote = '\0';
      }
      continue;
    }
    if (c == '\'' || c == '"') { quote = c; continue; }
    if (c == '[' || c == '{') { ++depth; continue; }
    if (c == ']' || c == '}') { --depth; continue; }
    if (c == ':' && depth == 0) {
      if (i + 1 == s.size() || std::isspace(static_cast<unsigned char>(s[i + 1])) != 0) return i;
    }
  }
  return std::string::npos;
}

/// 按顶层分隔符切分，忽略引号与括号内部。
std::vector<std::string> split_top_level(const std::string& s, char sep) {
  std::vector<std::string> parts;
  std::string cur;
  int depth = 0;
  char quote = '\0';
  bool escaped = false;
  for (char c : s) {
    if (quote != '\0') {
      cur.push_back(c);
      if (escaped) {
        escaped = false;
      } else if (quote == '"' && c == '\\') {
        escaped = true;
      } else if (c == quote) {
        quote = '\0';
      }
      continue;
    }
    if (c == '\'' || c == '"') { quote = c; cur.push_back(c); continue; }
    if (c == '[' || c == '{') ++depth;
    else if (c == ']' || c == '}') --depth;
    if (c == sep && depth == 0) { parts.push_back(cur); cur.clear(); continue; }
    cur.push_back(c);
  }
  parts.push_back(cur);
  return parts;
}

// ===========================================================================
// 标量解释（as_real / as_int / as_bool 共用）
// ===========================================================================

/// 去掉数字之间的下划线：1_000 -> 1000，1_0_0 -> 100。
std::string strip_digit_underscores(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (Size i = 0; i < s.size(); ++i) {
    if (s[i] == '_' && i > 0 && i + 1 < s.size() &&
        std::isdigit(static_cast<unsigned char>(s[i - 1])) != 0 &&
        std::isdigit(static_cast<unsigned char>(s[i + 1])) != 0) {
      continue;
    }
    out.push_back(s[i]);
  }
  return out;
}

/// 返回数字字面量（含符号、小数、科学计数法）在字符串中的结束位置；无数字返回 0。
Size scan_number_prefix(const std::string& s) {
  Size i = 0;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
  Size int_digits = 0;
  while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])) != 0) { ++i; ++int_digits; }
  Size frac_digits = 0;
  if (i < s.size() && s[i] == '.') {
    ++i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])) != 0) { ++i; ++frac_digits; }
  }
  if (int_digits == 0 && frac_digits == 0) return 0;
  if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
    Size j = i + 1;
    if (j < s.size() && (s[j] == '+' || s[j] == '-')) ++j;
    Size exp_digits = 0;
    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j])) != 0) { ++j; ++exp_digits; }
    if (exp_digits > 0) i = j;
  }
  return i;
}

/// 允许的长度/量纲后缀（与 python/vibe_post/config.py 的 _UNIT_FACTORS 对齐）。
bool unit_factor(const std::string& unit, Real& factor) {
  struct Entry { const char* name; Real factor; };
  static const Entry kUnits[] = {
      {"m", Real(1)},         {"km", Real(1e3)},      {"cm", Real(1e-2)},
      {"mm", Real(1e-3)},     {"s", Real(1)},         {"sec", Real(1)},
      {"min", Real(60)},      {"h", Real(3600)},      {"hr", Real(3600)},
      {"hour", Real(3600)},   {"d", Real(86400)},     {"day", Real(86400)},
      {"pa", Real(1)},        {"hpa", Real(100)},     {"mb", Real(100)},
      {"kpa", Real(1000)},    {"bar", Real(1e5)},     {"k", Real(1)},
      {"c", Real(1)},         {"deg", Real(1)},       {"degree", Real(1)},
      {"m/s", Real(1)},       {"mps", Real(1)},       {"ms", Real(1)},
      {"km/h", Real(1) / Real(3.6)}, {"kt", Real(0.514444)},
      {"knot", Real(0.514444)},       {"knots", Real(0.514444)},
      {"kg", Real(1)},        {"g", Real(1e-3)},      {"kg/kg", Real(1)},
      {"g/kg", Real(1e-3)},   {"w", Real(1)},         {"kw", Real(1e3)},
      {"j", Real(1)},         {"1", Real(1)},         {"-", Real(1)},
      {"s-1", Real(1)},       {"1/s", Real(1)},       {"%", Real(0.01)},
      {"percent", Real(0.01)},
  };
  const std::string u = lower_copy(trim(unit));
  for (const Entry& e : kUnits) {
    if (u == e.name) { factor = e.factor; return true; }
  }
  return false;
}

/// 解析标量为 Real。容忍：1000、1_000、1e3、3km、2.5 h、+.5。
bool parse_real_text(const std::string& text, Real& out) {
  const std::string s = strip_digit_underscores(trim(text));
  if (s.empty()) return false;
  // 1) 纯数值字面量（strtod 同时支持十进制与十六进制）
  {
    errno = 0;
    const char* begin = s.c_str();
    char* end = nullptr;
    const double v = std::strtod(begin, &end);
    if (end != begin && end != nullptr && *end == '\0') {
      out = static_cast<Real>(v);
      return std::isfinite(static_cast<double>(out));
    }
  }
  // 2) 数值 + 单位后缀
  const Size split = scan_number_prefix(s);
  if (split == 0 || split >= s.size()) return false;
  const std::string num = s.substr(0, split);
  const std::string unit = s.substr(split);
  const char* begin = num.c_str();
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin || end == nullptr || *end != '\0') return false;
  Real factor = Real(1);
  if (!unit_factor(unit, factor)) return false;
  out = static_cast<Real>(v * static_cast<double>(factor));
  return std::isfinite(static_cast<double>(out));
}

/// 去掉字符串两端的引号并处理转义；引号未闭合/后面有多余字符时抛错。
std::string unquote(const std::string& token, const std::string& origin, int line_no) {
  const char quote = token.front();
  std::string out;
  out.reserve(token.size());
  for (Size i = 1; i < token.size(); ++i) {
    const char c = token[i];
    if (quote == '"' && c == '\\' && i + 1 < token.size()) {
      const char n = token[++i];
      switch (n) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '0': out.push_back('\0'); break;
        default:
          out.push_back('\\');
          out.push_back(n);
          break;
      }
      continue;
    }
    if (c == quote) {
      if (quote == '\'' && i + 1 < token.size() && token[i + 1] == '\'') {
        out.push_back('\'');
        ++i;
        continue;
      }
      const std::string rest = trim(token.substr(i + 1));
      if (!rest.empty()) fail(origin, line_no, token, "引号字符串后存在多余字符: " + rest);
      return out;
    }
    out.push_back(c);
  }
  fail(origin, line_no, token, "引号未闭合");
}

// ===========================================================================
// 词法阶段
// ===========================================================================

struct Line {
  int indent = 0;
  std::string text;   ///< 去注释、去首尾空白后的内容
  int number = 0;     ///< 1 起的原始行号
  std::string raw;    ///< 原始行（错误消息用）
};

std::vector<Line> prepare(const std::string& text, const std::string& origin) {
  std::vector<Line> lines;
  std::istringstream in(text);
  std::string raw;
  int number = 0;
  while (std::getline(in, raw)) {
    ++number;
    if (!raw.empty() && raw.back() == '\r') raw.pop_back();

    const std::string no_comment = strip_comment(raw);
    if (trim(no_comment).empty()) continue;  // 空行或纯注释

    // 制表符缩进检查
    Size first_non_ws = 0;
    while (first_non_ws < no_comment.size() &&
           (no_comment[first_non_ws] == ' ' || no_comment[first_non_ws] == '\t')) {
      if (no_comment[first_non_ws] == '\t') {
        fail(origin, number, raw,
             "检测到制表符缩进；YAML 子集只允许空格缩进（建议每级 2 空格）");
      }
      ++first_non_ws;
    }
    const std::string body = rstrip(no_comment);
    const std::string trimmed = trim(body);
    if (trimmed == "---" || trimmed == "...") continue;  // 文档分隔符
    lines.push_back(Line{static_cast<int>(first_non_ws), trimmed, number, raw});
  }
  return lines;
}

// ===========================================================================
// 语法阶段：递归下降 + 缩进栈
// ===========================================================================

class Parser {
 public:
  std::vector<Line> lines;
  std::string origin = "<string>";

  YamlNode parse_document() {
    if (lines.empty()) return YamlNode{};
    Size pos = 0;
    YamlNode root = parse_block(pos, lines.front().indent);
    if (pos < lines.size()) {
      fail(origin, lines[pos].number, lines[pos].raw, "存在无法解析的剩余内容");
    }
    return root;
  }

 private:
  YamlNode parse_block(Size& pos, int indent) {
    if (pos >= lines.size()) return YamlNode{};
    const Line& ln = lines[pos];
    if (ln.text == "-" || starts_with(ln.text, "- ")) return parse_sequence(pos, indent);
    if (find_key_colon(ln.text) != std::string::npos) return parse_mapping(pos, indent);
    YamlNode v = parse_scalar_text(ln.text, ln.number);
    ++pos;
    return v;
  }

  YamlNode parse_mapping(Size& pos, int indent) {
    std::map<std::string, YamlNode> map;
    while (pos < lines.size()) {
      const Line& ln = lines[pos];
      if (ln.indent < indent) break;
      if (ln.indent > indent) {
        fail(origin, ln.number, ln.raw,
             "缩进过深：期望 " + std::to_string(indent) + " 个空格，实际 " +
                 std::to_string(ln.indent));
      }
      if (ln.text == "-" || starts_with(ln.text, "- ")) break;

      const Size colon = find_key_colon(ln.text);
      if (colon == std::string::npos) {
        fail(origin, ln.number, ln.raw, "不是合法的 key: value 行");
      }
      std::string key = trim(ln.text.substr(0, colon));
      if (key.empty()) fail(origin, ln.number, ln.raw, "键名不能为空");
      if (key.front() == '\'' || key.front() == '"') key = unquote(key, origin, ln.number);

      const std::string rest = trim(ln.text.substr(colon + 1));
      if (!rest.empty()) {
        map[key] = parse_scalar_text(rest, ln.number);
        ++pos;
        continue;
      }
      ++pos;  // 当前行已消费，值在后续缩进块中
      if (pos < lines.size() && lines[pos].indent > ln.indent) {
        map[key] = parse_block(pos, lines[pos].indent);
      } else if (pos < lines.size() && lines[pos].indent == ln.indent &&
                 (lines[pos].text == "-" || starts_with(lines[pos].text, "- "))) {
        map[key] = parse_sequence(pos, ln.indent);
      } else {
        map[key] = YamlNode{};  // 显式空值
      }
    }
    return YamlNode(std::move(map));
  }

  YamlNode parse_sequence(Size& pos, int indent) {
    std::vector<YamlNode> items;
    while (pos < lines.size()) {
      const Line& ln = lines[pos];
      if (ln.indent < indent) break;
      if (ln.indent > indent) {
        fail(origin, ln.number, ln.raw,
             "序列项缩进不一致：期望 " + std::to_string(indent) + " 个空格，实际 " +
                 std::to_string(ln.indent));
      }
      if (!(ln.text == "-" || starts_with(ln.text, "- "))) break;

      const std::string body = trim(ln.text.substr(1));
      if (body.empty()) {
        ++pos;
        if (pos < lines.size() && lines[pos].indent > ln.indent) {
          items.push_back(parse_block(pos, lines[pos].indent));
        } else {
          items.push_back(YamlNode{});
        }
        continue;
      }

      if (find_key_colon(body) != std::string::npos) {
        // 序列中的映射项：把 "- key: value" 还原为缩进映射后递归解析
        std::vector<Line> sub;
        sub.push_back(Line{ln.indent + 2, body, ln.number, ln.raw});
        Size q = pos + 1;
        while (q < lines.size() && lines[q].indent > ln.indent) {
          sub.push_back(lines[q]);
          ++q;
        }
        Parser sp;
        sp.lines = std::move(sub);
        sp.origin = origin;
        Size sub_pos = 0;
        items.push_back(sp.parse_block(sub_pos, sp.lines.front().indent));
        // sub[0] 对应 lines[pos]，sub[m] 对应 lines[pos+m]（m>=1）
        const Size consumed = (sub_pos == 0) ? Size(1) : sub_pos;
        pos += consumed;
        continue;
      }

      items.push_back(parse_scalar_text(body, ln.number));
      ++pos;
    }
    return YamlNode(std::move(items));
  }

  YamlNode parse_scalar_text(const std::string& text, int line_no) {
    const std::string t = trim(text);
    if (t.empty()) return YamlNode{};

    if (t.front() == '\'' || t.front() == '"') {
      return YamlNode(unquote(t, origin, line_no));
    }
    if (t.front() == '[') {
      if (t.back() != ']') fail(origin, line_no, t, "行内列表缺少 ]");
      const std::string inner = trim(t.substr(1, t.size() - 2));
      std::vector<YamlNode> items;
      if (!inner.empty()) {
        for (const std::string& part : split_top_level(inner, ',')) {
          const std::string p = trim(part);
          if (!p.empty()) items.push_back(parse_scalar_text(p, line_no));
        }
      }
      return YamlNode(std::move(items));
    }
    if (t.front() == '{') {
      if (t.back() != '}') fail(origin, line_no, t, "行内映射缺少 }");
      const std::string inner = trim(t.substr(1, t.size() - 2));
      std::map<std::string, YamlNode> map;
      if (!inner.empty()) {
        for (const std::string& part : split_top_level(inner, ',')) {
          const std::string p = trim(part);
          if (p.empty()) continue;
          const Size colon = find_key_colon(p);
          if (colon == std::string::npos) {
            fail(origin, line_no, t, "行内映射项缺少冒号: " + p);
          }
          std::string key = trim(p.substr(0, colon));
          if (key.empty()) fail(origin, line_no, t, "行内映射的键名不能为空");
          if (key.front() == '\'' || key.front() == '"') key = unquote(key, origin, line_no);
          map[key] = parse_scalar_text(trim(p.substr(colon + 1)), line_no);
        }
      }
      return YamlNode(std::move(map));
    }

    const std::string lowered = lower_copy(t);
    if (lowered == "~" || lowered == "null") return YamlNode{};
    return YamlNode(t);
  }
};

// ===========================================================================
// dump 辅助
// ===========================================================================

bool scalar_needs_quotes(const std::string& s) {
  if (s.empty()) return true;
  if (trim(s) != s) return true;
  if (s.front() == '-') return true;
  if (s.find_first_of(":#{}[],&*!|>'\"%@") != std::string::npos) return true;
  const std::string lw = lower_copy(s);
  if (lw == "null" || lw == "~" || lw == "true" || lw == "false" || lw == "yes" ||
      lw == "no" || lw == "on" || lw == "off") {
    return true;
  }
  Real probe = Real(0);
  if (parse_real_text(s, probe)) return true;  // 数字加引号以保持字符串语义
  return false;
}

std::string quote_dump(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default: out.push_back(c); break;
    }
  }
  out += "\"";
  return out;
}

std::string scalar_dump(const std::string& s) {
  return scalar_needs_quotes(s) ? quote_dump(s) : s;
}

}  // namespace

// ===========================================================================
// YamlNode 成员实现
// ===========================================================================

Real YamlNode::as_real(Real def) const {
  if (type_ != Type::Scalar) return def;
  Real v = Real(0);
  return parse_real_text(scalar_, v) ? v : def;
}

Int YamlNode::as_int(Int def) const {
  if (type_ != Type::Scalar) return def;
  Real v = Real(0);
  if (!parse_real_text(scalar_, v)) return def;
  const double r = std::llround(static_cast<double>(v));
  if (r > static_cast<double>(std::numeric_limits<Int>::max()) ||
      r < static_cast<double>(std::numeric_limits<Int>::min())) {
    return def;
  }
  return static_cast<Int>(r);
}

bool YamlNode::as_bool(bool def) const {
  if (type_ != Type::Scalar) return def;
  const std::string lw = lower_copy(trim(scalar_));
  if (lw == "true" || lw == "yes" || lw == "on" || lw == "y" || lw == "1") return true;
  if (lw == "false" || lw == "no" || lw == "off" || lw == "n" || lw == "0") return false;
  Real v = Real(0);
  if (parse_real_text(scalar_, v)) return v != Real(0);
  return def;
}

std::string YamlNode::as_string(const std::string& def) const {
  if (type_ == Type::Scalar) return scalar_;
  return def;
}

std::vector<Real> YamlNode::as_real_vector() const {
  std::vector<Real> out;
  if (type_ == Type::Sequence) {
    out.reserve(seq_.size());
    for (const YamlNode& n : seq_) {
      if (!n.is_scalar()) continue;
      Real v = Real(0);
      if (parse_real_text(n.scalar_, v)) out.push_back(v);
    }
  } else if (type_ == Type::Scalar) {
    Real v = Real(0);
    if (parse_real_text(scalar_, v)) out.push_back(v);
  }
  return out;
}

Size YamlNode::size() const noexcept {
  switch (type_) {
    case Type::Sequence: return seq_.size();
    case Type::Mapping: return map_.size();
    case Type::Scalar: return 1;
    case Type::Null: break;
  }
  return 0;
}

bool YamlNode::has(const std::string& key) const noexcept {
  return type_ == Type::Mapping && map_.find(key) != map_.end();
}

const YamlNode& YamlNode::operator[](const std::string& key) const noexcept {
  static const YamlNode kNullNode;
  if (type_ != Type::Mapping) return kNullNode;
  const auto it = map_.find(key);
  return it == map_.end() ? kNullNode : it->second;
}

const YamlNode& YamlNode::operator[](Size idx) const noexcept {
  static const YamlNode kNullNode;
  if (type_ != Type::Sequence || idx >= seq_.size()) return kNullNode;
  return seq_[idx];
}

std::vector<std::string> YamlNode::keys() const {
  std::vector<std::string> out;
  if (type_ == Type::Mapping) {
    out.reserve(map_.size());
    for (const auto& kv : map_) out.push_back(kv.first);
  }
  return out;
}

const YamlNode& YamlNode::get(const std::string& dotted_path) const noexcept {
  static const YamlNode kNullNode;
  if (dotted_path.empty()) return *this;
  const YamlNode* cur = this;
  Size begin = 0;
  while (begin <= dotted_path.size()) {
    Size end = dotted_path.find('.', begin);
    if (end == std::string::npos) end = dotted_path.size();
    const std::string seg = dotted_path.substr(begin, end - begin);
    if (cur->type_ == Type::Mapping) {
      const auto it = cur->map_.find(seg);
      if (it == cur->map_.end()) return kNullNode;
      cur = &it->second;
    } else if (cur->type_ == Type::Sequence) {
      bool all_digits = !seg.empty();
      for (char c : seg) {
        if (std::isdigit(static_cast<unsigned char>(c)) == 0) { all_digits = false; break; }
      }
      if (!all_digits) return kNullNode;
      Size idx = 0;
      for (char c : seg) idx = idx * 10 + static_cast<Size>(c - '0');
      if (idx >= cur->seq_.size()) return kNullNode;
      cur = &cur->seq_[idx];
    } else {
      return kNullNode;
    }
    if (end == dotted_path.size()) break;
    begin = end + 1;
  }
  return *cur;
}

Real YamlNode::require_real(const std::string& key) const {
  const YamlNode& n = get(key);
  if (n.is_null()) throw ConfigError("缺少必填字段: " + key);
  Real v = Real(0);
  if (!n.is_scalar() || !parse_real_text(n.scalar_, v)) {
    throw ConfigError("字段 " + key + " 需要数值，得到: " + trim(n.dump(0)));
  }
  return v;
}

Int YamlNode::require_int(const std::string& key) const {
  const YamlNode& n = get(key);
  if (n.is_null()) throw ConfigError("缺少必填字段: " + key);
  Real v = Real(0);
  if (!n.is_scalar() || !parse_real_text(n.scalar_, v)) {
    throw ConfigError("字段 " + key + " 需要整数，得到: " + trim(n.dump(0)));
  }
  const double r = std::llround(static_cast<double>(v));
  if (r > static_cast<double>(std::numeric_limits<Int>::max()) ||
      r < static_cast<double>(std::numeric_limits<Int>::min())) {
    throw ConfigError("字段 " + key + " 超出 Int 范围: " + n.scalar_);
  }
  return static_cast<Int>(r);
}

std::string YamlNode::require_string(const std::string& key) const {
  const YamlNode& n = get(key);
  if (n.is_null()) throw ConfigError("缺少必填字段: " + key);
  if (!n.is_scalar()) {
    throw ConfigError("字段 " + key + " 需要字符串，得到: " + trim(n.dump(0)));
  }
  return n.scalar_;
}

bool YamlNode::require_bool(const std::string& key) const {
  const YamlNode& n = get(key);
  if (n.is_null()) throw ConfigError("缺少必填字段: " + key);
  if (!n.is_scalar()) {
    throw ConfigError("字段 " + key + " 需要布尔值，得到: " + trim(n.dump(0)));
  }
  const std::string lw = lower_copy(trim(n.scalar_));
  if (lw == "true" || lw == "yes" || lw == "on" || lw == "1") return true;
  if (lw == "false" || lw == "no" || lw == "off" || lw == "0") return false;
  throw ConfigError("字段 " + key + " 不是合法布尔值: " + n.scalar_);
}

std::string YamlNode::dump(int indent) const {
  const int pad_n = indent > 0 ? indent : 0;
  const std::string pad(static_cast<Size>(pad_n), ' ');
  std::ostringstream os;
  switch (type_) {
    case Type::Null:
      os << pad << "null\n";
      break;
    case Type::Scalar:
      os << pad << scalar_dump(scalar_) << '\n';
      break;
    case Type::Sequence:
      if (seq_.empty()) {
        os << pad << "[]\n";
        break;
      }
      for (const YamlNode& item : seq_) {
        if (item.type_ == Type::Scalar) {
          os << pad << "- " << scalar_dump(item.scalar_) << '\n';
        } else if (item.type_ == Type::Null) {
          os << pad << "- null\n";
        } else {
          os << pad << "-\n" << item.dump(pad_n + 2);
        }
      }
      break;
    case Type::Mapping:
      if (map_.empty()) {
        os << pad << "{}\n";
        break;
      }
      for (const auto& kv : map_) {
        const YamlNode& v = kv.second;
        const std::string key = scalar_needs_quotes(kv.first) ? quote_dump(kv.first) : kv.first;
        if (v.type_ == Type::Scalar) {
          os << pad << key << ": " << scalar_dump(v.scalar_) << '\n';
        } else if (v.type_ == Type::Null) {
          os << pad << key << ": null\n";
        } else {
          os << pad << key << ":\n" << v.dump(pad_n + 2);
        }
      }
      break;
  }
  return os.str();
}

YamlNode YamlNode::parse(const std::string& text, const std::string& origin) {
  Parser parser;
  parser.origin = origin.empty() ? std::string("<string>") : origin;
  parser.lines = prepare(text, parser.origin);
  return parser.parse_document();
}

YamlNode YamlNode::parse_file(const std::string& path) {
  errno = 0;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    const int e = errno;
    std::ostringstream os;
    os << "无法打开 YAML 文件: " << path << " (errno=" << e << " " << std::strerror(e) << ")";
    throw ConfigError(os.str());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return parse(buffer.str(), path);
}

// ===========================================================================
// 深合并
// ===========================================================================

YamlNode merge(const YamlNode& base, const YamlNode& over) {
  // Null 表示"本层未指定"，不覆盖下层（YAML 的 ~ 视为未指定）
  if (over.is_null()) return base;
  if (base.is_mapping() && over.is_mapping()) {
    std::map<std::string, YamlNode> out;
    for (const std::string& k : base.keys()) out[k] = base[k];
    for (const std::string& k : over.keys()) {
      const auto it = out.find(k);
      out[k] = (it == out.end()) ? over[k] : merge(it->second, over[k]);
    }
    return YamlNode(std::move(out));
  }
  // 序列与标量整体替换（不做元素级合并）
  return over;
}

}  // namespace vibe::config

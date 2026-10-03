#pragma once
/// @file yaml_node.hpp
/// @brief 极简 YAML 子集解析器（无外部依赖）。
///
/// 支持的范围（刻意受限，避免引入第三方库）
/// ----------------------------------------
///   * 缩进表示的嵌套映射 `key: value`
///   * 序列 `- item`（可嵌套映射）
///   * 标量：整数、浮点、布尔、字符串（可加单/双引号）
///   * 行内列表 `[1, 2, 3]` 与行内映射 `{a: 1, b: 2}`
///   * 注释 `# ...`
///   * 空值与 `null`
/// **不支持**：锚点/别名、多文档、复杂键、流式块标量、制表符缩进。
///
/// 若解析到不支持的结构，抛出 `vibe::ConfigError` 并给出行号。

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"

namespace vibe::config {

/// 不可变 YAML 节点
class YamlNode {
 public:
  enum class Type { Null, Scalar, Sequence, Mapping };

  YamlNode() = default;
  explicit YamlNode(std::string scalar) : type_(Type::Scalar), scalar_(std::move(scalar)) {}
  explicit YamlNode(std::vector<YamlNode> seq) : type_(Type::Sequence), seq_(std::move(seq)) {}
  explicit YamlNode(std::map<std::string, YamlNode> map)
      : type_(Type::Mapping), map_(std::move(map)) {}

  Type type() const noexcept { return type_; }
  bool is_null() const noexcept { return type_ == Type::Null; }
  bool is_scalar() const noexcept { return type_ == Type::Scalar; }
  bool is_sequence() const noexcept { return type_ == Type::Sequence; }
  bool is_mapping() const noexcept { return type_ == Type::Mapping; }

  // ---- 标量访问 -----------------------------------------------------------
  const std::string& raw() const noexcept { return scalar_; }
  Real as_real(Real def = Real(0)) const;
  Int as_int(Int def = 0) const;
  bool as_bool(bool def = false) const;
  std::string as_string(const std::string& def = "") const;

  /// 序列形式的数值向量；标量时返回单元素
  std::vector<Real> as_real_vector() const;

  // ---- 结构访问 -----------------------------------------------------------
  const std::vector<YamlNode>& sequence() const { return seq_; }
  Size size() const noexcept;
  bool has(const std::string& key) const noexcept;
  const YamlNode& operator[](const std::string& key) const noexcept;
  const YamlNode& operator[](Size idx) const noexcept;
  std::vector<std::string> keys() const;

  /// 路径查询 "a.b.c"，缺失返回 Null 节点
  const YamlNode& get(const std::string& dotted_path) const noexcept;

  /// 带默认值与必填校验的取值助手
  Real require_real(const std::string& key) const;
  Int require_int(const std::string& key) const;
  std::string require_string(const std::string& key) const;
  bool require_bool(const std::string& key) const;

  // ---- 解析 ---------------------------------------------------------------
  static YamlNode parse(const std::string& text, const std::string& origin = "<string>");
  static YamlNode parse_file(const std::string& path);

  /// 调试输出
  std::string dump(int indent = 0) const;

 private:
  Type type_ = Type::Null;
  std::string scalar_;
  std::vector<YamlNode> seq_;
  std::map<std::string, YamlNode> map_;
};

/// 深度合并：over 中存在的键覆盖 base（递归处理映射）
YamlNode merge(const YamlNode& base, const YamlNode& over);

}  // namespace vibe::config

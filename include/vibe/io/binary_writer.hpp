#pragma once
/// @file binary_writer.hpp
/// @brief `.vibebin` 后备格式读写器（零依赖，用于调试、单元测试与 Python 后处理）。
///
/// 格式规范见 field_io.hpp 文件头注释；本文件同时给出读与写。
/// 设计目标：
///   * 跨平台确定（little-endian，定长名字，无对齐填充）；
///   * 可被 numpy 直接 memmap 读取；
///   * 支持多变量与单个 time 记录。

#include <cstdio>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/io/field_io.hpp"

namespace vibe::io {

inline constexpr const char* kBinaryMagic = "VIBEBIN1";
inline constexpr int kBinaryNameLength = 32;

/// 二进制写
class BinaryWriter final : public FieldWriter {
 public:
  ~BinaryWriter() override;
  void open(const std::string& path, const grid::Grid& g) override;
  void write(const FieldMeta& meta, const grid::Field<Real>& f) override;
  void write_time(Real t) override;
  void close() override;
  IoFormat format() const noexcept override { return IoFormat::Binary; }

 private:
  struct Record { FieldMeta meta; std::vector<Real> data; };
  std::FILE* fp_ = nullptr;
  grid::Geometry geom_{};
  Real time_ = Real(0);
  std::vector<Record> pending_;
  std::string path_;
};

/// 二进制读
class BinaryReader final : public FieldReader {
 public:
  ~BinaryReader() override;
  void open(const std::string& path) override;
  bool has(const std::string& name) const override;
  void read(const std::string& name, grid::Field<Real>& f) override;
  std::vector<std::string> variables() const override;
  FieldMeta meta(const std::string& name) const override;
  void close() override;

 private:
  struct Record { FieldMeta meta; Index offset; };
  std::FILE* fp_ = nullptr;
  std::vector<Record> records_;
  Int nx_ = 0, ny_ = 0, nz_ = 0;
  std::string path_;
};

/// 读取二进制文件头（供工具与测试使用）
struct BinaryHeader {
  Int nx = 0, ny = 0, nz = 0;
  int real_kind = 0;
  Real time = Real(0);
  Int nvars = 0;
  Int nlevels = 0;
};
BinaryHeader read_binary_header(const std::string& path);

}  // namespace vibe::io

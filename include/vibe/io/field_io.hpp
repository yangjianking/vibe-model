#pragma once
/// @file field_io.hpp
/// @brief 场输入输出的抽象接口。
///
/// **架构硬约束**：只有 `vibe::io` 层可以包含 NetCDF/GRIB 头文件。
/// 动力学、时间推进、物理、同化模块一律通过本文件声明的接口读写数据。
///
/// 变量命名约定（与 Python 包 vibe_post 一致）
/// ------------------------------------------
///   u, v, w, rho, theta, pi, qv, qc, qr, qi, qs, qg
///   维度顺序 (time, zeta, y, x)；`zeta` 为垂直层坐标（无单位，0-1）
///
/// 二进制后备格式 `.vibebin`（依赖 NetCDF 为零时使用）
/// -------------------------------------------------
///   文件头：
///     char    magic[8]   = "VIBEBIN1"
///     int32   nx, ny, nz
///     int32   real_kind  (0 = float64, 1 = float32)
///     float64 time
///     int32   nvars
///     int32   nlevels    (垂直层数，用于剖面型文件)
///   每个变量：
///     char    name[32]   （补 \0）
///     float64 data[nx*ny*nz]   （行主序，i 最快）
///
/// 文献：[D16] WRF ARW 第 5 章（I/O API 设计）。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::io {

/// 变量元数据
struct FieldMeta {
  std::string name;
  std::string units;
  std::string description;
  grid::Stagger stagger = grid::Stagger::Cell;
  Real missing_value = Real(-9999.0);
  int  level_type = 0;      ///< 0 = zeta, 1 = 气压, 2 = 高度
};

/// 写出的格式
enum class IoFormat { NetCDF, Binary, Both };

/// 写出器接口
class FieldWriter {
 public:
  virtual ~FieldWriter() = default;
  virtual void open(const std::string& path, const grid::Grid& g) = 0;
  virtual void write(const FieldMeta& meta, const grid::Field<Real>& f) = 0;
  virtual void write_time(Real t) = 0;
  virtual void close() = 0;
  virtual IoFormat format() const noexcept = 0;
};

std::unique_ptr<FieldWriter> make_netcdf_writer();
std::unique_ptr<FieldWriter> make_binary_writer();
std::unique_ptr<FieldWriter> make_writer(IoFormat f);

/// 读取器接口
class FieldReader {
 public:
  virtual ~FieldReader() = default;
  virtual void open(const std::string& path) = 0;
  virtual bool has(const std::string& name) const = 0;
  virtual void read(const std::string& name, grid::Field<Real>& f) = 0;
  virtual std::vector<std::string> variables() const = 0;
  virtual FieldMeta meta(const std::string& name) const = 0;
  virtual void close() = 0;
};

std::unique_ptr<FieldReader> make_reader(IoFormat f);

/// 重启文件信息
struct RestartInfo {
  Real time = Real(0);
  int  step = 0;
  std::string valid_time;
  std::string config_hash;      ///< 配置指纹，防止用错配置重启
};

/// 写出完整状态（所有预报变量）
void write_restart(const std::string& path, const dyn::State& s,
                   const RestartInfo& info);

/// 读取完整状态；要求目标 State 已经 allocate
RestartInfo read_restart(const std::string& path, dyn::State& s);

/// 把 State 的每个变量写出到 writer
void write_state(FieldWriter& w, const dyn::State& s);

/// 从 reader 填充 State（缺失变量保持原值）
void read_state(FieldReader& r, dyn::State& s);

/// 便捷函数：把单个场写成 `.vibebin`
void write_binary_field(const std::string& path, const FieldMeta& meta,
                        const grid::Field<Real>& f, Real time = Real(0));

/// 便捷函数：读取 `.vibebin` 中的第一个变量
void read_binary_field(const std::string& path, grid::Field<Real>& f);

}  // namespace vibe::io

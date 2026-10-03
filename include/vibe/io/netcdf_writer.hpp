#pragma once
/// @file netcdf_writer.hpp
/// @brief NetCDF-4 写出器（仅在 `VIBE_ENABLE_NETCDF` 时编译）。
///
/// 之所以单独成篇，是为了让 CMake 能条件编译：没有 NetCDF 的机器上
/// 整个翻译单元被排除，其余代码无需任何修改。
///
/// 文件结构
/// --------
///   dimensions : time (unlimited), zeta (nz+1), z (nz), y, x
///   variables  : 各预报/诊断量 + 坐标变量
///   global attr: title, source, history, Conventions="CF-1.10",
///                grid_type="arakawa_c", vertical_coordinate="terrain_following_height",
///                refinement_ratio, model_version
///
/// 文献：[D16] WRF ARW 第 5 章；CF 约定 1.10。

#include <memory>
#include <string>

#include "vibe/common/types.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/io/field_io.hpp"

namespace vibe::io {

/// 是否在编译期启用了 NetCDF
bool netcdf_available() noexcept;

/// 若未启用，调用 `make_netcdf_writer()` 会抛 `NotImplemented`
std::unique_ptr<FieldWriter> make_netcdf_writer_impl();

/// 一次写出坐标变量与全局属性
void write_coordinates(FieldWriter& w, const grid::Grid& g);

}  // namespace vibe::io

/// @file field_io.cpp
/// @brief 场 IO 工厂、State 读写与重启文件（见 include/vibe/io/field_io.hpp）。
///
/// 架构约束：只有本层（vibe::io）允许出现 NetCDF 依赖。本文件通过
/// netcdf_writer.hpp 声明的工厂间接使用 NetCDF，宏 VIBE_HAVE_NETCDF 只出现在
/// src/io/netcdf_writer.cpp 中，保证未启用 NetCDF 的机器上其余代码零修改。
///
/// 变量命名与单位表见下方 kSpeciesIo；与 Python 包 vibe_post 的命名一致，
/// 该一致性由 [D16] WRF ARW 第 5 章的 I/O API 设计约定保证。
/// 重启策略见 docs/design/11_config_and_io.md。

#include "vibe/io/field_io.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/io/binary_writer.hpp"
#include "vibe/io/netcdf_writer.hpp"

namespace vibe::io {
namespace {

// ===========================================================================
// State 变量元数据表
//
// 顺序严格等于 dyn::Species 的枚举顺序（dyn/state.hpp），
// 变量名用 dyn::species_name()，单位与错位由本表给出。
// ===========================================================================

struct SpeciesIo {
  const char* name;
  const char* units;
  const char* description;
  grid::Stagger stagger;
};

const SpeciesIo kSpeciesIo[] = {
    {"u", "m s-1", "zonal wind (Arakawa C-grid face x)", grid::Stagger::FaceX},
    {"v", "m s-1", "meridional wind (Arakawa C-grid face y)", grid::Stagger::FaceY},
    {"w", "m s-1", "vertical wind (Lorenz face z)", grid::Stagger::FaceZ},
    {"rho", "kg m-3", "dry air density", grid::Stagger::Cell},
    {"theta", "K", "potential temperature", grid::Stagger::Cell},
    {"pi", "1", "Exner pressure perturbation", grid::Stagger::Cell},
    {"qv", "kg kg-1", "water vapour mixing ratio", grid::Stagger::Cell},
    {"qc", "kg kg-1", "cloud water mixing ratio", grid::Stagger::Cell},
    {"qr", "kg kg-1", "rain water mixing ratio", grid::Stagger::Cell},
    {"qi", "kg kg-1", "cloud ice mixing ratio", grid::Stagger::Cell},
    {"qs", "kg kg-1", "snow mixing ratio", grid::Stagger::Cell},
    {"qg", "kg kg-1", "graupel mixing ratio", grid::Stagger::Cell},
};

static_assert(sizeof(kSpeciesIo) / sizeof(kSpeciesIo[0]) ==
                  static_cast<Size>(dyn::Species::Count),
              "kSpeciesIo 必须与 dyn::Species 一一对应");

FieldMeta species_meta(dyn::Species s) {
  const auto idx = static_cast<Size>(s);
  const SpeciesIo& io = kSpeciesIo[idx];
  FieldMeta meta;
  meta.name = io.name;
  meta.units = io.units;
  meta.description = io.description;
  meta.stagger = io.stagger;
  meta.missing_value = Real(-9999.0);
  meta.level_type = 0;  // zeta
  return meta;
}

/// "Both" 格式下二进制侧的文件名：把 .nc 换成 .vibebin，否则追加后缀。
std::string derive_binary_path(const std::string& path) {
  if (path.size() > 3 && path.compare(path.size() - 3, 3, ".nc") == 0) {
    return path.substr(0, path.size() - 3) + ".vibebin";
  }
  return path + ".vibebin";
}

// ===========================================================================
// 同时写 NetCDF 与二进制的复合写出器（IoFormat::Both）
// ===========================================================================

class TeeWriter final : public FieldWriter {
 public:
  TeeWriter(std::unique_ptr<FieldWriter> primary, std::unique_ptr<FieldWriter> secondary)
      : primary_(std::move(primary)), secondary_(std::move(secondary)) {}

  void open(const std::string& path, const grid::Grid& g) override {
    primary_path_ = path;
    secondary_path_ = derive_binary_path(path);
    primary_->open(primary_path_, g);
    secondary_->open(secondary_path_, g);
  }

  void write(const FieldMeta& meta, const grid::Field<Real>& f) override {
    primary_->write(meta, f);
    secondary_->write(meta, f);
  }

  void write_time(Real t) override {
    primary_->write_time(t);
    secondary_->write_time(t);
  }

  void close() override {
    primary_->close();
    secondary_->close();
  }

  IoFormat format() const noexcept override { return IoFormat::Both; }

 private:
  std::unique_ptr<FieldWriter> primary_;
  std::unique_ptr<FieldWriter> secondary_;
  std::string primary_path_;
  std::string secondary_path_;
};

}  // namespace

// ===========================================================================
// 工厂
// ===========================================================================

std::unique_ptr<FieldWriter> make_netcdf_writer() {
  if (!netcdf_available()) {
    throw NotImplemented(
        "NetCDF 写出器未编译启用（VIBE_HAVE_NETCDF 未定义）；请改用 make_binary_writer()");
  }
  return make_netcdf_writer_impl();
}

std::unique_ptr<FieldWriter> make_binary_writer() {
  return std::make_unique<BinaryWriter>();
}

std::unique_ptr<FieldWriter> make_writer(IoFormat f) {
  switch (f) {
    case IoFormat::NetCDF:
      return make_netcdf_writer();
    case IoFormat::Binary:
      return make_binary_writer();
    case IoFormat::Both: {
      auto nc = make_netcdf_writer();  // 未启用时抛 NotImplemented
      auto bin = make_binary_writer();
      return std::make_unique<TeeWriter>(std::move(nc), std::move(bin));
    }
  }
  throw IoError("未知的 IoFormat 取值");
}

std::unique_ptr<FieldReader> make_reader(IoFormat f) {
  switch (f) {
    case IoFormat::Binary:
    case IoFormat::Both:
      // 二进制侧是自描述程度最高、可被 numpy memmap 的格式，作为读取首选。
      return std::make_unique<BinaryReader>();
    case IoFormat::NetCDF:
      throw NotImplemented(
          "vibe::io 未提供 NetCDF 读取器（写出器层只负责输出）；"
          "请使用 .vibebin 或 Python 包 vibe_post 读取 NetCDF");
  }
  throw IoError("未知的 IoFormat 取值");
}

// ===========================================================================
// 状态读写
// ===========================================================================

void write_state(FieldWriter& w, const dyn::State& s) {
  w.write_time(s.time);
  for (int i = 0; i < dyn::kNumSpecies; ++i) {
    const auto sp = static_cast<dyn::Species>(i);
    const grid::Field<Real>& f = s.field(sp);
    if (f.empty()) {
      VIBE_WARN("[io] write_state: 变量 ", dyn::species_name(sp), " 未分配，已跳过");
      continue;
    }
    w.write(species_meta(sp), f);
  }
}

void read_state(FieldReader& r, dyn::State& s) {
  for (int i = 0; i < dyn::kNumSpecies; ++i) {
    const auto sp = static_cast<dyn::Species>(i);
    const char* name = dyn::species_name(sp);
    if (!r.has(name)) {
      VIBE_WARN("[io] read_state: 文件缺少变量 ", name, "（目标状态保持原值）");
      continue;
    }
    r.read(name, s.field(sp));
  }
}

// ===========================================================================
// 重启
//
// __restart_info__ 是本层的约定变量名：在 .vibebin 中以固定形状 (2,1,1)
// 存储 [time, step] 两个 float64。读重启时先找它，缺失则 time/step 置 0。
// ===========================================================================

void write_restart(const std::string& path, const dyn::State& s, const RestartInfo& info) {
  auto writer = make_binary_writer();
  writer->open(path, s.grid());
  // 文件头 time 由 write_state 从 s.time 写入；info.time 单独存进 __restart_info__。

  grid::Field<Real> tag(s.grid(), grid::Stagger::Cell, "__restart_info__");
  if (tag.size() < 2) throw IoError("write_restart: 网格太小，无法存放重启信息: " + path);
  tag.data()[0] = info.time;
  tag.data()[1] = static_cast<Real>(info.step);

  FieldMeta meta;
  meta.name = "__restart_info__";
  meta.units = "s";
  meta.description = "restart metadata: [time, step]";
  meta.stagger = grid::Stagger::Cell;
  meta.level_type = 0;
  writer->write(meta, tag);

  write_state(*writer, s);
  writer->close();
}

RestartInfo read_restart(const std::string& path, dyn::State& s) {
  BinaryReader reader;
  reader.open(path);

  RestartInfo info;
  if (reader.has("__restart_info__")) {
    grid::Field<Real> tag(s.grid(), grid::Stagger::Cell, "__restart_info__");
    reader.read("__restart_info__", tag);
    if (tag.size() >= 2) {
      info.time = tag.data()[0];
      info.step = static_cast<int>(std::llround(static_cast<double>(tag.data()[1])));
    }
  } else {
    VIBE_WARN("[io] read_restart: ", path, " 缺少 __restart_info__，time/step 置 0");
  }
  read_state(reader, s);
  reader.close();

  s.time = info.time;
  s.step = info.step;
  return info;
}

// ===========================================================================
// 单场便捷函数
// ===========================================================================

void write_binary_field(const std::string& path, const FieldMeta& meta,
                        const grid::Field<Real>& f, Real time) {
  if (f.empty()) throw IoError("write_binary_field: 场 '" + meta.name + "' 未分配");
  BinaryWriter writer;
  writer.open(path, f.grid());
  writer.write_time(time);
  writer.write(meta, f);
  writer.close();
}

void read_binary_field(const std::string& path, grid::Field<Real>& f) {
  if (f.empty()) throw IoError("read_binary_field: 目标场未分配（需先按网格构造）");
  BinaryReader reader;
  reader.open(path);
  const std::vector<std::string> names = reader.variables();
  if (names.empty()) {
    reader.close();
    throw IoError("read_binary_field: 文件中没有变量: " + path);
  }
  reader.read(names.front(), f);
  reader.close();
}

}  // namespace vibe::io

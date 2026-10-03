/// @file binary_writer.cpp
/// @brief .vibebin 后备格式的读与写（格式规范见 include/vibe/io/field_io.hpp 文件头）。
///
/// 字节布局（全部 little-endian，显式按字节拼装，不依赖主机字节序）
/// -----------------------------------------------------------------
///   offset  type      field
///   0       char[8]   magic = "VIBEBIN1"
///   8       int32     nx, ny, nz            （本地子域内部点数；串行时即全局）
///   20      int32     real_kind             （0=float64 写者, 1=float32 写者）
///   24      float64   time
///   32      int32     nvars
///   36      int32     nlevels               （= nz，供剖面型工具使用）
///   40      每个变量：char name[32] + float64 data[ni*nj*nk]
///
/// 关于变量形状：文件中只有 32 字节名字可以承载形状信息，因此本实现约定
/// **名字唯一决定错位**：u -> FaceX(nx+1)、v -> FaceY(ny+1)、w/zeta -> FaceZ(nz+1)，
/// 其余 -> Cell(nx,ny,nz)；特殊变量 __restart_info__ 固定为 (2,1,1)。
/// 写者会校验 FieldMeta.stagger 与名字隐含的错位一致，避免静默写坏数据。
/// 数据始终以 float64 存储，real_kind 只记录写者的原生 Real 类型，读时转回 Real。
///
/// 复杂度：写/读均为 O(N)，N = 变量数据点数；额外内存为一个变量的缓冲 O(N_var)。
/// 文献：[D16] WRF ARW 第 5 章（后备二进制格式与 I/O API 设计）。

#include "vibe/io/binary_writer.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"

namespace vibe::io {
namespace {

// ===========================================================================
// 小端字节序原语
// ===========================================================================

int real_kind_of_real() noexcept { return sizeof(Real) == 8 ? 0 : 1; }

std::string errno_message(int e) {
  return std::string(std::strerror(e)) + " (errno=" + std::to_string(e) + ")";
}

void write_bytes(std::FILE* fp, const void* data, Size n, const std::string& path) {
  if (n == 0) return;
  if (std::fwrite(data, 1, n, fp) != n) {
    const int e = errno;
    throw IoError("写入失败: " + path + " — " + errno_message(e));
  }
}

void write_i32_le(std::FILE* fp, Int value, const std::string& path) {
  const std::uint32_t u = static_cast<std::uint32_t>(value);
  unsigned char b[4];
  b[0] = static_cast<unsigned char>(u & 0xFFu);
  b[1] = static_cast<unsigned char>((u >> 8) & 0xFFu);
  b[2] = static_cast<unsigned char>((u >> 16) & 0xFFu);
  b[3] = static_cast<unsigned char>((u >> 24) & 0xFFu);
  write_bytes(fp, b, 4, path);
}

void write_f64_le(std::FILE* fp, double value, const std::string& path) {
  std::uint64_t u = 0;
  std::memcpy(&u, &value, sizeof(u));
  unsigned char b[8];
  for (int i = 0; i < 8; ++i) {
    b[i] = static_cast<unsigned char>((u >> (8 * i)) & 0xFFu);
  }
  write_bytes(fp, b, 8, path);
}

bool read_bytes(std::FILE* fp, void* data, Size n) {
  if (n == 0) return true;
  return std::fread(data, 1, n, fp) == n;
}

bool read_i32_le(std::FILE* fp, Int& value) {
  unsigned char b[4];
  if (!read_bytes(fp, b, 4)) return false;
  const std::uint32_t u = static_cast<std::uint32_t>(b[0]) |
                          (static_cast<std::uint32_t>(b[1]) << 8) |
                          (static_cast<std::uint32_t>(b[2]) << 16) |
                          (static_cast<std::uint32_t>(b[3]) << 24);
  value = static_cast<Int>(static_cast<std::int32_t>(u));
  return true;
}

bool read_f64_le(std::FILE* fp, double& value) {
  unsigned char b[8];
  if (!read_bytes(fp, b, 8)) return false;
  std::uint64_t u = 0;
  for (int i = 0; i < 8; ++i) {
    u |= static_cast<std::uint64_t>(b[i]) << (8 * i);
  }
  std::memcpy(&value, &u, sizeof(value));
  return true;
}

// ===========================================================================
// 形状规则
// ===========================================================================

struct Shape {
  Int ni = 0;
  Int nj = 0;
  Int nk = 0;
  Int count() const { return ni * nj * nk; }
};

grid::Stagger stagger_from_name(const std::string& name) {
  if (name == "u" || name == "u_facex") return grid::Stagger::FaceX;
  if (name == "v" || name == "v_facey") return grid::Stagger::FaceY;
  if (name == "w" || name == "zeta" || name == "w_facez") return grid::Stagger::FaceZ;
  return grid::Stagger::Cell;
}

Shape shape_for_stagger(grid::Stagger s, Int nx, Int ny, Int nz) {
  switch (s) {
    case grid::Stagger::FaceX: return Shape{nx + 1, ny, nz};
    case grid::Stagger::FaceY: return Shape{nx, ny + 1, nz};
    case grid::Stagger::FaceZ: return Shape{nx, ny, nz + 1};
    case grid::Stagger::Cell:
    case grid::Stagger::Corner: break;
  }
  return Shape{nx, ny, nz};
}

Shape shape_for_name(const std::string& name, Int nx, Int ny, Int nz) {
  if (name == "__restart_info__") return Shape{2, 1, 1};
  return shape_for_stagger(stagger_from_name(name), nx, ny, nz);
}

/// 供 BinaryReader::meta() 使用的名字 -> 单位/描述表（文件本身不存这些信息）。
struct NameMeta {
  const char* name;
  const char* units;
  const char* description;
};

const NameMeta kNameMeta[] = {
    {"u", "m s-1", "zonal wind"},
    {"v", "m s-1", "meridional wind"},
    {"w", "m s-1", "vertical wind"},
    {"rho", "kg m-3", "dry air density"},
    {"theta", "K", "potential temperature"},
    {"pi", "1", "Exner pressure perturbation"},
    {"qv", "kg kg-1", "water vapour mixing ratio"},
    {"qc", "kg kg-1", "cloud water mixing ratio"},
    {"qr", "kg kg-1", "rain water mixing ratio"},
    {"qi", "kg kg-1", "cloud ice mixing ratio"},
    {"qs", "kg kg-1", "snow mixing ratio"},
    {"qg", "kg kg-1", "graupel mixing ratio"},
    {"x", "m", "x coordinate of cell centres"},
    {"y", "m", "y coordinate of cell centres"},
    {"z", "m", "height of cell centres"},
    {"zeta", "1", "terrain-following vertical coordinate"},
    {"terrain", "m", "terrain height"},
    {"dx_cell", "m", "variable-resolution x cell width"},
    {"dy_cell", "m", "variable-resolution y cell width"},
    {"time", "s", "model time"},
    {"__restart_info__", "s", "restart metadata: [time, step]"},
};

FieldMeta meta_from_name(const std::string& name) {
  FieldMeta meta;
  meta.name = name;
  meta.stagger = stagger_from_name(name);
  for (const NameMeta& nm : kNameMeta) {
    if (name == nm.name) {
      meta.units = nm.units;
      meta.description = nm.description;
      return meta;
    }
  }
  meta.units = "";
  meta.description = "";
  return meta;
}

}  // namespace

// ===========================================================================
// BinaryWriter
// ===========================================================================

BinaryWriter::~BinaryWriter() {
  try {
    close();
  } catch (...) {
    // 析构中不允许抛出：失败只影响后备调试文件
  }
}

void BinaryWriter::open(const std::string& path, const grid::Grid& g) {
  if (path.empty()) throw IoError("BinaryWriter::open: 路径为空");
  if (fp_ != nullptr) {
    std::fclose(fp_);
    fp_ = nullptr;
  }
  path_ = path;
  geom_ = g.geom();
  // 记录"本地子域"维度：串行时等于全局维度，MPI 时每个 rank 写自己的文件。
  geom_.nx = g.nx();
  geom_.ny = g.ny();
  geom_.nz = g.nz();
  if (geom_.nx <= 0 || geom_.ny <= 0 || geom_.nz <= 0) {
    path_.clear();
    throw IoError("BinaryWriter::open: 网格维度非法 (" + std::to_string(geom_.nx) + "x" +
                  std::to_string(geom_.ny) + "x" + std::to_string(geom_.nz) + ")");
  }
  time_ = Real(0);
  pending_.clear();
}

void BinaryWriter::write_time(Real t) { time_ = t; }

void BinaryWriter::write(const FieldMeta& meta, const grid::Field<Real>& f) {
  if (path_.empty()) throw IoError("BinaryWriter::write: 必须先调用 open()");
  if (meta.name.empty()) throw IoError("BinaryWriter::write: 变量名不能为空");
  if (f.empty()) throw IoError("BinaryWriter::write: 场 '" + meta.name + "' 未分配");
  if (f.nx() != geom_.nx || f.ny() != geom_.ny || f.nz() != geom_.nz) {
    throw IoError("BinaryWriter::write: 场 '" + meta.name + "' 的网格 (" +
                  std::to_string(f.nx()) + "x" + std::to_string(f.ny()) + "x" +
                  std::to_string(f.nz()) + ") 与 open() 时给出的 (" +
                  std::to_string(geom_.nx) + "x" + std::to_string(geom_.ny) + "x" +
                  std::to_string(geom_.nz) + ") 不一致");
  }

  Record rec;
  rec.meta = meta;

  if (meta.name == "__restart_info__") {
    if (f.size() < 2) {
      throw IoError("BinaryWriter::write: __restart_info__ 需要至少 2 个元素的场");
    }
    rec.data.resize(2);
    rec.data[0] = f.data()[0];
    rec.data[1] = f.data()[1];
    pending_.push_back(std::move(rec));
    return;
  }

  const grid::Stagger implied = stagger_from_name(meta.name);
  if (implied != meta.stagger) {
    throw IoError("BinaryWriter::write: 变量名 '" + meta.name + "' 隐含错位 " +
                  grid::to_string(implied) + "，但 FieldMeta.stagger=" +
                  grid::to_string(meta.stagger) +
                  "；.vibebin 不存错位字段，名字必须唯一确定错位（u/v/w/zeta 为面量）");
  }

  const Shape sh = shape_for_stagger(meta.stagger, geom_.nx, geom_.ny, geom_.nz);
  rec.data.resize(static_cast<Size>(sh.count()));
  Size lin = 0;
  for (Int k = 0; k < sh.nk; ++k) {
    for (Int j = 0; j < sh.nj; ++j) {
      for (Int i = 0; i < sh.ni; ++i) {
        rec.data[lin++] = f.at(i, j, k);
      }
    }
  }
  pending_.push_back(std::move(rec));
}

void BinaryWriter::close() {
  if (path_.empty()) return;  // 已写出或从未 open

  errno = 0;
  std::FILE* fp = std::fopen(path_.c_str(), "wb");
  if (fp == nullptr) {
    const int e = errno;
    const std::string p = path_;
    path_.clear();
    pending_.clear();
    throw IoError("无法写入 " + p + ": " + errno_message(e));
  }
  fp_ = fp;

  try {
    write_bytes(fp, kBinaryMagic, 8, path_);
    write_i32_le(fp, geom_.nx, path_);
    write_i32_le(fp, geom_.ny, path_);
    write_i32_le(fp, geom_.nz, path_);
    write_i32_le(fp, static_cast<Int>(real_kind_of_real()), path_);
    write_f64_le(fp, static_cast<double>(time_), path_);
    write_i32_le(fp, static_cast<Int>(pending_.size()), path_);
    write_i32_le(fp, geom_.nz, path_);

    for (const Record& rec : pending_) {
      char name[kBinaryNameLength];
      std::memset(name, 0, sizeof(name));
      const Size n = std::min<Size>(rec.meta.name.size(),
                                    static_cast<Size>(kBinaryNameLength - 1));
      if (n > 0) std::memcpy(name, rec.meta.name.data(), n);
      write_bytes(fp, name, static_cast<Size>(kBinaryNameLength), path_);
      for (const Real value : rec.data) {
        write_f64_le(fp, static_cast<double>(value), path_);
      }
    }
  } catch (...) {
    std::fclose(fp);
    fp_ = nullptr;
    throw;
  }

  if (std::fclose(fp) != 0) {
    const int e = errno;
    fp_ = nullptr;
    throw IoError("关闭文件失败: " + path_ + " — " + errno_message(e));
  }
  fp_ = nullptr;
  pending_.clear();
  path_.clear();  // 防止析构重复写出
}

// ===========================================================================
// BinaryReader
// ===========================================================================

BinaryReader::~BinaryReader() {
  try {
    close();
  } catch (...) {
  }
}

void BinaryReader::open(const std::string& path) {
  if (path.empty()) throw IoError("BinaryReader::open: 路径为空");
  close();

  errno = 0;
  fp_ = std::fopen(path.c_str(), "rb");
  if (fp_ == nullptr) {
    const int e = errno;
    throw IoError("无法读取 " + path + ": " + errno_message(e));
  }
  path_ = path;
  records_.clear();

  char magic[8];
  if (!read_bytes(fp_, magic, 8) || std::memcmp(magic, kBinaryMagic, 8) != 0) {
    close();
    throw IoError("不是 .vibebin 文件（magic 不匹配）: " + path);
  }

  Int real_kind = 0;
  double time = 0.0;
  Int nvars = 0;
  Int nlevels = 0;
  if (!read_i32_le(fp_, nx_) || !read_i32_le(fp_, ny_) || !read_i32_le(fp_, nz_) ||
      !read_i32_le(fp_, real_kind) || !read_f64_le(fp_, time) || !read_i32_le(fp_, nvars) ||
      !read_i32_le(fp_, nlevels)) {
    close();
    throw IoError("文件头不完整: " + path);
  }
  if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) {
    close();
    throw IoError("非法网格维度: " + path);
  }
  if (nvars < 0 || nlevels < 0) {
    close();
    throw IoError("非法 nvars/nlevels: " + path);
  }
  if (real_kind != 0 && real_kind != 1) {
    close();
    throw IoError("未知 real_kind=" + std::to_string(real_kind) + ": " + path);
  }

  records_.reserve(static_cast<Size>(nvars));
  for (Int v = 0; v < nvars; ++v) {
    char name[kBinaryNameLength];
    if (!read_bytes(fp_, name, static_cast<Size>(kBinaryNameLength))) {
      close();
      throw IoError("读取变量名失败（第 " + std::to_string(v) + " 个变量）: " + path);
    }
    name[kBinaryNameLength - 1] = '\0';
    const std::string var_name(name);

    const Shape sh = shape_for_name(var_name, nx_, ny_, nz_);
    const long offset = std::ftell(fp_);
    if (offset < 0) {
      close();
      throw IoError("ftell 失败: " + path);
    }
    Record rec;
    rec.meta = meta_from_name(var_name);
    rec.offset = static_cast<Index>(offset);
    records_.push_back(std::move(rec));

    const long skip = static_cast<long>(sh.count()) * 8L;
    if (std::fseek(fp_, skip, SEEK_CUR) != 0) {
      close();
      throw IoError("跳过变量数据失败: " + var_name + ": " + path);
    }
  }
}

bool BinaryReader::has(const std::string& name) const {
  for (const Record& rec : records_) {
    if (rec.meta.name == name) return true;
  }
  return false;
}

std::vector<std::string> BinaryReader::variables() const {
  std::vector<std::string> names;
  names.reserve(records_.size());
  for (const Record& rec : records_) names.push_back(rec.meta.name);
  return names;
}

FieldMeta BinaryReader::meta(const std::string& name) const {
  for (const Record& rec : records_) {
    if (rec.meta.name == name) return rec.meta;
  }
  throw IoError("变量不存在: '" + name + "'（" + path_ + "）");
}

void BinaryReader::read(const std::string& name, grid::Field<Real>& f) {
  if (fp_ == nullptr) throw IoError("BinaryReader::read: 必须先调用 open()");
  const Record* rec = nullptr;
  for (const Record& r : records_) {
    if (r.meta.name == name) { rec = &r; break; }
  }
  if (rec == nullptr) throw IoError("变量不存在: '" + name + "'（" + path_ + "）");
  if (f.empty()) throw IoError("BinaryReader::read: 目标场未分配（变量 " + name + "）");

  if (name == "__restart_info__") {
    if (f.size() < 2) throw IoError("BinaryReader::read: __restart_info__ 目标场至少需 2 元素");
    if (std::fseek(fp_, static_cast<long>(rec->offset), SEEK_SET) != 0) {
      throw IoError("定位变量失败: " + name + ": " + path_);
    }
    double t0 = 0.0;
    double t1 = 0.0;
    if (!read_f64_le(fp_, t0) || !read_f64_le(fp_, t1)) {
      throw IoError("读取 __restart_info__ 数据失败: " + path_);
    }
    f.data()[0] = static_cast<Real>(t0);
    f.data()[1] = static_cast<Real>(t1);
    return;
  }

  if (f.nx() != nx_ || f.ny() != ny_ || f.nz() != nz_) {
    throw IoError("BinaryReader::read: 目标场维度 (" + std::to_string(f.nx()) + "x" +
                  std::to_string(f.ny()) + "x" + std::to_string(f.nz()) + ") 与文件 (" +
                  std::to_string(nx_) + "x" + std::to_string(ny_) + "x" +
                  std::to_string(nz_) + ") 不一致");
  }
  const grid::Stagger implied = stagger_from_name(name);
  if (f.stagger() != implied) {
    throw IoError("BinaryReader::read: 变量名 '" + name + "' 隐含错位 " +
                  grid::to_string(implied) + "，但目标场错位为 " +
                  grid::to_string(f.stagger()));
  }

  const Shape sh = shape_for_stagger(implied, nx_, ny_, nz_);
  if (std::fseek(fp_, static_cast<long>(rec->offset), SEEK_SET) != 0) {
    throw IoError("定位变量失败: " + name + ": " + path_);
  }
  std::vector<double> buffer(static_cast<Size>(sh.count()));
  for (Size n = 0; n < buffer.size(); ++n) {
    if (!read_f64_le(fp_, buffer[n])) {
      throw IoError("读取变量数据失败（截断）: " + name + ": " + path_);
    }
  }

  Size lin = 0;
  for (Int k = 0; k < sh.nk; ++k) {
    for (Int j = 0; j < sh.nj; ++j) {
      for (Int i = 0; i < sh.ni; ++i) {
        f.at(i, j, k) = static_cast<Real>(buffer[lin++]);
      }
    }
  }
}

void BinaryReader::close() {
  if (fp_ != nullptr) {
    std::fclose(fp_);
    fp_ = nullptr;
  }
  records_.clear();
}

// ===========================================================================
// 文件头读取（工具与测试用）
// ===========================================================================

BinaryHeader read_binary_header(const std::string& path) {
  errno = 0;
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) {
    const int e = errno;
    throw IoError("无法读取 " + path + ": " + errno_message(e));
  }

  BinaryHeader header;
  bool ok = true;
  char magic[8];
  Int real_kind = 0;
  double time = 0.0;
  ok = ok && read_bytes(fp, magic, 8) && std::memcmp(magic, kBinaryMagic, 8) == 0;
  if (!ok) {
    std::fclose(fp);
    throw IoError("不是 .vibebin 文件（magic 不匹配或文件过短）: " + path);
  }
  ok = ok && read_i32_le(fp, header.nx) && read_i32_le(fp, header.ny) &&
       read_i32_le(fp, header.nz) && read_i32_le(fp, real_kind) && read_f64_le(fp, time) &&
       read_i32_le(fp, header.nvars) && read_i32_le(fp, header.nlevels);
  std::fclose(fp);
  if (!ok) throw IoError("文件头不完整: " + path);
  if (header.nx <= 0 || header.ny <= 0 || header.nz <= 0) {
    throw IoError("非法网格维度: " + path);
  }
  header.real_kind = static_cast<int>(real_kind);
  header.time = static_cast<Real>(time);
  return header;
}

}  // namespace vibe::io

/// @file netcdf_writer.cpp
/// @brief NetCDF-4 写出器（仅当定义了 VIBE_HAVE_NETCDF 时编译出真实实现）。
///
/// 文件结构（CF-1.10 兼容）
/// -----------------------
///   dimensions : time (unlimited), zeta (nz+1), z (nz), y (ny), x (nx)
///   variables  : 各预报/诊断量 + 坐标变量 (time, x, y, z, zeta, terrain, ...)
///   global att: title, source, history, Conventions, grid_type,
///               vertical_coordinate, refinement_ratio, model_version
///
/// 说明：动量场在 Arakawa C 网格上位于面，其点数比体心多 1；为了保持所有
/// 变量共享同一组 CF 维度，这里把面量按体心点数写出（丢掉最东/最北/最顶层
/// 的那个面点），并在变量属性 stagger 中标明原始错位。精确的重启请使用
/// .vibebin（见 src/io/binary_writer.cpp）。
///
/// 文献：[D16] WRF ARW 第 5 章；CF 约定 1.10。

#include "vibe/io/netcdf_writer.hpp"

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/common/types.hpp"
#include "vibe/grid/field.hpp"

#ifdef VIBE_HAVE_NETCDF
#  include <netcdf.h>
#endif

namespace vibe::io {

bool netcdf_available() noexcept {
#ifdef VIBE_HAVE_NETCDF
  return true;
#else
  return false;
#endif
}

#ifdef VIBE_HAVE_NETCDF

namespace {

void check_nc(int rc, const char* what, const std::string& path) {
  if (rc != NC_NOERR) {
    throw IoError(std::string(what) + " 失败: " + nc_strerror(rc) + "（" + path + "）");
  }
}

bool is_coordinate_name(const std::string& name) {
  return name == "x" || name == "y" || name == "z" || name == "zeta" ||
         name == "terrain" || name == "dx_cell" || name == "dy_cell";
}

/// NetCDF-4 写出器。维度在 open() 时建立；变量按需定义（netCDF-4 允许
/// 多次 nc_redef / nc_enddef），因此无需预先知道要写哪些变量。
class NetCdfWriter final : public FieldWriter {
 public:
  ~NetCdfWriter() override {
    try {
      close();
    } catch (...) {
    }
  }

  void open(const std::string& path, const grid::Grid& g) override {
    if (path.empty()) throw IoError("NetCdfWriter::open: 路径为空");
    close();
    path_ = path;

    nx_ = g.nx();
    ny_ = g.ny();
    nz_ = g.nz();
    flat_terrain_ = g.geom().flat_terrain;
    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) {
      throw IoError("NetCdfWriter::open: 网格维度非法");
    }

    int rc = nc_create(path.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid_);
    check_nc(rc, "nc_create", path_);

    define_dim("time", NC_UNLIMITED, dim_time_);
    define_dim("zeta", nz_ + 1, dim_zeta_);
    define_dim("z", nz_, dim_z_);
    define_dim("y", ny_, dim_y_);
    define_dim("x", nx_, dim_x_);

    put_global_text("title", "VIBE-Model forecast output");
    put_global_text("source", "vibe-model");
    put_global_text("history", "created by vibe::io::NetCdfWriter");
    put_global_text("Conventions", "CF-1.10");
    put_global_text("grid_type", "arakawa_c");
    put_global_text("vertical_coordinate", "terrain_following_height");
    put_global_text("model_version", "vibe-0.1.0");
    put_global_int("refinement_ratio", 1);

    // 时间坐标变量（open 阶段定义，保证第一个 write_time 可写）
    int time_dims[1] = {dim_time_};
    rc = nc_def_var(ncid_, "time", NC_DOUBLE, 1, time_dims, &var_time_);
    check_nc(rc, "nc_def_var(time)", path_);
    put_att_text(var_time_, "units", "seconds since 1970-01-01 00:00:00 UTC");
    put_att_text(var_time_, "calendar", "standard");
    put_att_text(var_time_, "long_name", "model time");
    put_att_text(var_time_, "axis", "T");
    rc = nc_enddef(ncid_);
    check_nc(rc, "nc_enddef", path_);
    vars_["time"] = var_time_;
  }

  void write(const FieldMeta& meta, const grid::Field<Real>& f) override {
    if (ncid_ < 0) throw IoError("NetCdfWriter::write: 必须先调用 open()");
    if (meta.name.empty()) throw IoError("NetCdfWriter::write: 变量名不能为空");
    if (f.empty()) throw IoError("NetCdfWriter::write: 场 '" + meta.name + "' 未分配");
    if (is_coordinate_name(meta.name)) {
      write_coordinate(meta.name, f);
    } else {
      write_field(meta, f);
    }
  }

  void write_time(Real t) override {
    if (ncid_ < 0) throw IoError("NetCdfWriter::write_time: 必须先调用 open()");
    if (time_written_ && t != time_) ++time_index_;
    time_ = t;
    const size_t index = static_cast<size_t>(time_index_);
    double value = static_cast<double>(t);
    const int rc = nc_put_var1_double(ncid_, var_time_, &index, &value);
    check_nc(rc, "nc_put_var1_double(time)", path_);
    time_written_ = true;
  }

  void close() override {
    if (ncid_ >= 0) {
      const int rc = nc_close(ncid_);
      ncid_ = -1;
      if (rc != NC_NOERR) {
        throw IoError(std::string("nc_close 失败: ") + nc_strerror(rc) + "（" + path_ + "）");
      }
    }
    vars_.clear();
    time_written_ = false;
    time_index_ = 0;
  }

  IoFormat format() const noexcept override { return IoFormat::NetCDF; }

 private:
  void define_dim(const char* name, int length, int& id) {
    const int rc = nc_def_dim(ncid_, name, static_cast<size_t>(length), &id);
    check_nc(rc, "nc_def_dim", path_);
  }

  void put_global_text(const char* name, const std::string& value) {
    const int rc = nc_put_att_text(ncid_, NC_GLOBAL, name, value.size(), value.c_str());
    check_nc(rc, "nc_put_att_text(global)", path_);
  }

  void put_global_int(const char* name, int value) {
    const int rc = nc_put_att_int(ncid_, NC_GLOBAL, name, NC_INT, 1, &value);
    check_nc(rc, "nc_put_att_int(global)", path_);
  }

  void put_att_text(int varid, const char* name, const std::string& value) {
    const int rc = nc_put_att_text(ncid_, varid, name, value.size(), value.c_str());
    check_nc(rc, "nc_put_att_text", path_);
  }

  void put_att_double(int varid, const char* name, double value) {
    const int rc = nc_put_att_double(ncid_, varid, name, NC_DOUBLE, 1, &value);
    check_nc(rc, "nc_put_att_double", path_);
  }

  /// 按需定义变量：netCDF-4 允许在数据模式下重新进入定义模式。
  int ensure_variable(const std::string& name, const int* dims, int ndims,
                      const std::string& units, const std::string& long_name) {
    const auto it = vars_.find(name);
    if (it != vars_.end()) return it->second;
    int rc = nc_redef(ncid_);
    if (rc != NC_NOERR && rc != NC_EINDEFINE) check_nc(rc, "nc_redef", path_);
    int varid = -1;
    rc = nc_def_var(ncid_, name.c_str(), NC_DOUBLE, ndims, dims, &varid);
    check_nc(rc, "nc_def_var", path_);
    put_att_text(varid, "units", units);
    put_att_text(varid, "long_name", long_name);
    put_att_double(varid, "_FillValue", -9999.0);
    rc = nc_enddef(ncid_);
    check_nc(rc, "nc_enddef", path_);
    vars_[name] = varid;
    return varid;
  }

  /// 预报/诊断量：共享 (time, zeta, y, x) 维度。
  void write_field(const FieldMeta& meta, const grid::Field<Real>& f) {
    if (f.nx() != nx_ || f.ny() != ny_ || f.nz() != nz_) {
      throw IoError("NetCdfWriter::write: 场 '" + meta.name + "' 的网格与写入器不一致");
    }
    const int dims[4] = {dim_time_, dim_zeta_, dim_y_, dim_x_};
    const int varid = ensure_variable(meta.name, dims, 4, meta.units, meta.description);
    // stagger 属性：直接改属性需要重新进入定义模式
    {
      const int rc = nc_redef(ncid_);
      if (rc != NC_NOERR && rc != NC_EINDEFINE) check_nc(rc, "nc_redef", path_);
      put_att_text(varid, "stagger", grid::to_string(meta.stagger));
      put_att_text(varid, "coordinates", "x y zeta");
      const int rc2 = nc_enddef(ncid_);
      check_nc(rc2, "nc_enddef", path_);
    }

    std::vector<double> buffer(static_cast<Size>(nz_) * ny_ * nx_, 0.0);
    Size lin = 0;
    for (Int k = 0; k < nz_; ++k) {
      for (Int j = 0; j < ny_; ++j) {
        for (Int i = 0; i < nx_; ++i) {
          buffer[lin++] = static_cast<double>(f.at(i, j, k));
        }
      }
    }
    const size_t start[4] = {static_cast<size_t>(time_index_), 0, 0, 0};
    const size_t count[4] = {1, static_cast<size_t>(nz_), static_cast<size_t>(ny_),
                             static_cast<size_t>(nx_)};
    const int rc = nc_put_vara_double(ncid_, varid, start, count, buffer.data());
    check_nc(rc, "nc_put_vara_double", path_);
  }

  void write_coordinate(const std::string& name, const grid::Field<Real>& f) {
    if (f.nx() != nx_ || f.ny() != ny_ || f.nz() != nz_) {
      throw IoError("NetCdfWriter::write: 坐标场 '" + name + "' 的网格与写入器不一致");
    }
    if (name == "x" || name == "dx_cell") {
      const int dims[1] = {dim_x_};
      const char* long_name = name == "dx_cell" ? "variable-resolution x cell width"
                                                : "x coordinate of cell centres";
      const int varid = ensure_variable(name, dims, 1, "m", long_name);
      std::vector<double> buf(static_cast<Size>(nx_));
      for (Int i = 0; i < nx_; ++i) buf[static_cast<Size>(i)] = static_cast<double>(f.at(i, 0, 0));
      const int rc = nc_put_var_double(ncid_, varid, buf.data());
      check_nc(rc, "nc_put_var_double(x)", path_);
      return;
    }
    if (name == "y" || name == "dy_cell") {
      const int dims[1] = {dim_y_};
      const char* long_name = name == "dy_cell" ? "variable-resolution y cell width"
                                                : "y coordinate of cell centres";
      const int varid = ensure_variable(name, dims, 1, "m", long_name);
      std::vector<double> buf(static_cast<Size>(ny_));
      for (Int j = 0; j < ny_; ++j) buf[static_cast<Size>(j)] = static_cast<double>(f.at(0, j, 0));
      const int rc = nc_put_var_double(ncid_, varid, buf.data());
      check_nc(rc, "nc_put_var_double(y)", path_);
      return;
    }
    if (name == "zeta") {
      const int dims[1] = {dim_zeta_};
      const int varid = ensure_variable("zeta", dims, 1, "1",
                                        "terrain-following vertical coordinate");
      std::vector<double> buf(static_cast<Size>(nz_) + 1);
      for (Int k = 0; k <= nz_; ++k) {
        buf[static_cast<Size>(k)] = static_cast<double>(f.at(0, 0, k));
      }
      const int rc = nc_put_var_double(ncid_, varid, buf.data());
      check_nc(rc, "nc_put_var_double(zeta)", path_);
      return;
    }
    if (name == "terrain") {
      const int dims[2] = {dim_y_, dim_x_};
      const int varid = ensure_variable("terrain", dims, 2, "m", "terrain height");
      std::vector<double> buf(static_cast<Size>(ny_) * nx_, 0.0);
      Size lin = 0;
      for (Int j = 0; j < ny_; ++j) {
        for (Int i = 0; i < nx_; ++i) {
          buf[lin++] = static_cast<double>(f.at(i, j, 0));
        }
      }
      const int rc = nc_put_var_double(ncid_, varid, buf.data());
      check_nc(rc, "nc_put_var_double(terrain)", path_);
      return;
    }
    if (name == "z") {
      if (flat_terrain_) {
        const int dims[1] = {dim_z_};
        const int varid = ensure_variable("z", dims, 1, "m", "height of cell centres");
        std::vector<double> buf(static_cast<Size>(nz_));
        for (Int k = 0; k < nz_; ++k) {
          buf[static_cast<Size>(k)] = static_cast<double>(f.at(0, 0, k));
        }
        const int rc = nc_put_var_double(ncid_, varid, buf.data());
        check_nc(rc, "nc_put_var_double(z)", path_);
      } else {
        const int dims[3] = {dim_z_, dim_y_, dim_x_};
        const int varid = ensure_variable("z", dims, 3, "m", "height of cell centres");
        std::vector<double> buf(static_cast<Size>(nz_) * ny_ * nx_, 0.0);
        Size lin = 0;
        for (Int k = 0; k < nz_; ++k) {
          for (Int j = 0; j < ny_; ++j) {
            for (Int i = 0; i < nx_; ++i) {
              buf[lin++] = static_cast<double>(f.at(i, j, k));
            }
          }
        }
        const int rc = nc_put_var_double(ncid_, varid, buf.data());
        check_nc(rc, "nc_put_var_double(z3d)", path_);
      }
      return;
    }
    VIBE_WARN("[io] NetCdfWriter: 未知坐标变量 ", name, "，已忽略");
  }

  int ncid_ = -1;
  std::string path_;
  Int nx_ = 0, ny_ = 0, nz_ = 0;
  bool flat_terrain_ = true;
  int dim_time_ = -1, dim_zeta_ = -1, dim_z_ = -1, dim_y_ = -1, dim_x_ = -1;
  int var_time_ = -1;
  Real time_ = Real(0);
  bool time_written_ = false;
  int time_index_ = 0;
  std::map<std::string, int> vars_;
};

}  // namespace

std::unique_ptr<FieldWriter> make_netcdf_writer_impl() {
  return std::unique_ptr<FieldWriter>(new NetCdfWriter());
}

// ---------------------------------------------------------------------------
// 坐标变量与全局属性
//
// 通过 FieldWriter 的公开接口写出，因此对 BinaryWriter 同样可用（二进制侧
// 会按普通场存储；NetCDF 侧识别名字后写成 1-D/2-D 坐标变量）。
// 复杂度 O(nx*ny*nz)（把坐标广播成场），内存 O(nx*ny*nz)。
// ---------------------------------------------------------------------------
void write_coordinates(FieldWriter& w, const grid::Grid& g) {
  const Int nx = g.nx();
  const Int ny = g.ny();
  const Int nz = g.nz();
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw IoError("write_coordinates: 网格维度非法");
  }

  const auto make_meta = [](const char* name, const char* units, const char* desc,
                           grid::Stagger s) {
    FieldMeta meta;
    meta.name = name;
    meta.units = units;
    meta.description = desc;
    meta.stagger = s;
    return meta;
  };

  // 单元中心坐标（支持变分辨率：累加逐列/逐行宽度）
  std::vector<Real> xc(static_cast<Size>(nx));
  std::vector<Real> yc(static_cast<Size>(ny));
  {
    Real acc = g.geom().x0;
    for (Int i = 0; i < nx; ++i) {
      xc[static_cast<Size>(i)] = acc + Real(0.5) * g.dx_at(i);
      acc += g.dx_at(i);
    }
    acc = g.geom().y0;
    for (Int j = 0; j < ny; ++j) {
      yc[static_cast<Size>(j)] = acc + Real(0.5) * g.dy_at(j);
      acc += g.dy_at(j);
    }
  }

  {
    grid::Field<Real> x_field(g, grid::Stagger::Cell, "x");
    for (Int k = 0; k < nz; ++k) {
      for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) x_field(i, j, k) = xc[static_cast<Size>(i)];
      }
    }
    w.write(make_meta("x", "m", "x coordinate of cell centres", grid::Stagger::Cell), x_field);
  }
  {
    grid::Field<Real> y_field(g, grid::Stagger::Cell, "y");
    for (Int k = 0; k < nz; ++k) {
      for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) y_field(i, j, k) = yc[static_cast<Size>(j)];
      }
    }
    w.write(make_meta("y", "m", "y coordinate of cell centres", grid::Stagger::Cell), y_field);
  }
  {
    grid::Field<Real> zeta_field(g, grid::Stagger::FaceZ, "zeta");
    for (Int k = 0; k <= nz; ++k) {
      const Real z = static_cast<Size>(k) < g.geom().zeta.size() ? g.geom().zeta[static_cast<Size>(k)]
                                                                 : Real(0);
      for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) zeta_field(i, j, k) = z;
      }
    }
    w.write(make_meta("zeta", "1", "terrain-following vertical coordinate",
                      grid::Stagger::FaceZ),
            zeta_field);
  }
  {
    grid::Field<Real> z_field(g, grid::Stagger::Cell, "z");
    for (Int k = 0; k < nz; ++k) {
      for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) z_field(i, j, k) = g.z_center(i, j, k);
      }
    }
    w.write(make_meta("z", "m", "height of cell centres", grid::Stagger::Cell), z_field);
  }
  {
    grid::Field<Real> terrain(g, grid::Stagger::Cell, "terrain");
    for (Int k = 0; k < nz; ++k) {
      for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) terrain(i, j, k) = g.terrain(i, j);
      }
    }
    w.write(make_meta("terrain", "m", "terrain height", grid::Stagger::Cell), terrain);
  }
  if (g.geom().variable_resolution) {
    {
      grid::Field<Real> dxf(g, grid::Stagger::Cell, "dx_cell");
      for (Int k = 0; k < nz; ++k) {
        for (Int j = 0; j < ny; ++j) {
          for (Int i = 0; i < nx; ++i) dxf(i, j, k) = g.dx_at(i);
        }
      }
      w.write(make_meta("dx_cell", "m", "variable-resolution x cell width",
                        grid::Stagger::Cell),
              dxf);
    }
    {
      grid::Field<Real> dyf(g, grid::Stagger::Cell, "dy_cell");
      for (Int k = 0; k < nz; ++k) {
        for (Int j = 0; j < ny; ++j) {
          for (Int i = 0; i < nx; ++i) dyf(i, j, k) = g.dy_at(j);
        }
      }
      w.write(make_meta("dy_cell", "m", "variable-resolution y cell width",
                        grid::Stagger::Cell),
              dyf);
    }
  }
}

#else  // !VIBE_HAVE_NETCDF

std::unique_ptr<FieldWriter> make_netcdf_writer_impl() {
  throw NotImplemented(
      "NetCDF 未编译启用（VIBE_HAVE_NETCDF 未定义）；请配置 -DVIBE_ENABLE_NETCDF=ON "
      "或使用 make_binary_writer()");
}

void write_coordinates(FieldWriter& w, const grid::Grid& g) {
  VIBE_UNUSED(w);
  VIBE_UNUSED(g);
  throw NotImplemented("write_coordinates 需要 VIBE_HAVE_NETCDF 支持");
}

#endif  // VIBE_HAVE_NETCDF

}  // namespace vibe::io

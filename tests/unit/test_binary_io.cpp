/// @file test_binary_io.cpp
/// @brief .vibebin 读写器的单元测试（含用 std::ifstream 手工解析文件头，
///        不依赖 BinaryReader 自身）。
///
/// 注意：vibe/common/error.hpp 与 vibe/common/test.hpp 都定义了 VIBE_CHECK 宏，
/// 因此先在库头文件之后 undef，再引入测试框架（不改动任何头文件）。

#include "vibe/io/binary_writer.hpp"
#include "vibe/io/field_io.hpp"

#include "vibe/common/error.hpp"
#include "vibe/dyn/state.hpp"
#undef VIBE_CHECK
#include "vibe/common/test.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using vibe::Real;
using vibe::io::BinaryHeader;
using vibe::io::BinaryReader;
using vibe::io::BinaryWriter;
using vibe::io::FieldMeta;
using vibe::io::IoFormat;

grid::Grid make_grid(vibe::Int nx, vibe::Int ny, vibe::Int nz, vibe::Int halo = 2) {
  grid::Geometry geom;
  geom.nx = nx;
  geom.ny = ny;
  geom.nz = nz;
  geom.x0 = Real(0);
  geom.y0 = Real(0);
  geom.z_top = Real(10000);
  geom.dx = Real(1000);
  geom.dy = Real(1000);
  geom.zeta.assign(static_cast<vibe::Size>(nz) + 1, Real(0));
  for (vibe::Int k = 0; k <= nz; ++k) {
    geom.zeta[static_cast<vibe::Size>(k)] =
        static_cast<Real>(k) / static_cast<Real>(nz);
  }
  geom.flat_terrain = true;
  geom.name = "test";
  const grid::Decomposition dec = grid::Decomposition::make(nx, ny, nz, 1, 1, halo);
  return grid::Grid(geom, dec);
}

std::string temp_path(const std::string& tag) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path();
  return (dir / ("vibe_test_" + tag + ".vibebin")).string();
}

void remove_file(const std::string& p) {
  std::error_code ec;
  std::filesystem::remove(p, ec);
}

FieldMeta cell_meta(const std::string& name) {
  FieldMeta meta;
  meta.name = name;
  meta.units = "K";
  meta.description = "unit test field";
  meta.stagger = grid::Stagger::Cell;
  return meta;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. 单变量写读往返
// ---------------------------------------------------------------------------
VIBE_TEST(binary_roundtrip_single_variable) {
  const grid::Grid g = make_grid(4, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "theta");
  for (vibe::Int k = 0; k < 2; ++k) {
    for (vibe::Int j = 0; j < 3; ++j) {
      for (vibe::Int i = 0; i < 4; ++i) {
        f(i, j, k) = Real(100) + Real(10 * k) + Real(j) + Real(0.25) * Real(i);
      }
    }
  }
  const std::string p = temp_path("single");
  vibe::io::write_binary_field(p, cell_meta("theta"), f, Real(12.5));

  grid::Field<Real> r(g, grid::Stagger::Cell, "theta");
  vibe::io::read_binary_field(p, r);
  for (vibe::Int k = 0; k < 2; ++k) {
    for (vibe::Int j = 0; j < 3; ++j) {
      for (vibe::Int i = 0; i < 4; ++i) {
        VIBE_CHECK_NEAR(r(i, j, k), f(i, j, k), 0.0);
      }
    }
  }
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 2. 多变量（含面量）写读往返
// ---------------------------------------------------------------------------
VIBE_TEST(binary_roundtrip_multiple_variables) {
  const grid::Grid g = make_grid(4, 3, 2);
  grid::Field<Real> theta(g, grid::Stagger::Cell, "theta");
  grid::Field<Real> u(g, grid::Stagger::FaceX, "u");
  for (vibe::Int k = 0; k < 2; ++k) {
    for (vibe::Int j = 0; j < 3; ++j) {
      for (vibe::Int i = 0; i < 4; ++i) theta(i, j, k) = Real(300) + Real(i + j + k);
      for (vibe::Int i = 0; i <= 4; ++i) u(i, j, k) = Real(5) + Real(i) * Real(0.5) + Real(j);
    }
  }
  const std::string p = temp_path("multi");
  BinaryWriter w;
  w.open(p, g);
  FieldMeta mu = cell_meta("u");
  mu.stagger = grid::Stagger::FaceX;
  w.write(cell_meta("theta"), theta);
  w.write(mu, u);
  w.close();

  VIBE_CHECK(w.format() == IoFormat::Binary);
  const BinaryHeader h = vibe::io::read_binary_header(p);
  VIBE_CHECK(h.nvars == 2);

  BinaryReader r;
  r.open(p);
  VIBE_CHECK(r.variables().size() == 2);
  VIBE_CHECK(r.has("u"));
  VIBE_CHECK(r.has("theta"));
  VIBE_CHECK(!r.has("qv"));
  VIBE_CHECK(r.meta("u").stagger == grid::Stagger::FaceX);

  grid::Field<Real> u2(g, grid::Stagger::FaceX, "u");
  r.read("u", u2);
  for (vibe::Int k = 0; k < 2; ++k) {
    for (vibe::Int j = 0; j < 3; ++j) {
      for (vibe::Int i = 0; i <= 4; ++i) VIBE_CHECK_NEAR(u2(i, j, k), u(i, j, k), 0.0);
    }
  }
  grid::Field<Real> th2(g, grid::Stagger::Cell, "theta");
  r.read("theta", th2);
  for (vibe::Int k = 0; k < 2; ++k) {
    for (vibe::Int j = 0; j < 3; ++j) {
      for (vibe::Int i = 0; i < 4; ++i) VIBE_CHECK_NEAR(th2(i, j, k), theta(i, j, k), 0.0);
    }
  }
  r.close();
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 3. 文件头字段
// ---------------------------------------------------------------------------
VIBE_TEST(binary_header_fields_correct) {
  const grid::Grid g = make_grid(4, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "rho");
  f.fill(Real(1.25));
  const std::string p = temp_path("header");
  vibe::io::write_binary_field(p, cell_meta("rho"), f, Real(7.25));

  const BinaryHeader h = vibe::io::read_binary_header(p);
  VIBE_CHECK(h.nx == 4);
  VIBE_CHECK(h.ny == 3);
  VIBE_CHECK(h.nz == 2);
  VIBE_CHECK(h.nvars == 1);
  VIBE_CHECK(h.nlevels == 2);
  VIBE_CHECK_NEAR(h.time, 7.25, 0.0);
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 4. 名字定长截断（32 字节，含终止 NUL）
// ---------------------------------------------------------------------------
VIBE_TEST(binary_name_is_truncated_to_31) {
  const grid::Grid g = make_grid(3, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "long");
  f.fill(Real(2));
  const std::string long_name(40, 'a');
  const std::string p = temp_path("name");
  vibe::io::write_binary_field(p, cell_meta(long_name), f, Real(0));

  BinaryReader r;
  r.open(p);
  VIBE_CHECK(r.variables().size() == 1);
  VIBE_CHECK(r.variables()[0] == std::string(31, 'a'));
  VIBE_CHECK(r.has(std::string(31, 'a')));
  r.close();
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 5. real_kind 由 sizeof(Real) 决定，数据始终 float64
// ---------------------------------------------------------------------------
VIBE_TEST(binary_real_kind_matches_real) {
  const grid::Grid g = make_grid(3, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "pi");
  f.fill(Real(0.5));
  const std::string p = temp_path("realkind");
  vibe::io::write_binary_field(p, cell_meta("pi"), f, Real(0));

  const BinaryHeader h = vibe::io::read_binary_header(p);
  const int expected = (sizeof(Real) == 8) ? 0 : 1;
  VIBE_CHECK(h.real_kind == expected);

  // 数据区始终以 float64 存储：文件长度 = 40 字节头 + 32 字节名 + n*8
  std::error_code ec;
  const std::uintmax_t bytes = std::filesystem::file_size(p, ec);
  VIBE_CHECK(!ec);
  VIBE_CHECK(bytes == static_cast<std::uintmax_t>(40 + 32 + 3 * 3 * 2 * 8));
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 6. 读取不存在的变量抛异常
// ---------------------------------------------------------------------------
VIBE_TEST(binary_read_missing_variable_throws) {
  const grid::Grid g = make_grid(3, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "theta");
  f.fill(Real(1));
  const std::string p = temp_path("missing");
  vibe::io::write_binary_field(p, cell_meta("theta"), f, Real(0));

  BinaryReader r;
  r.open(p);
  grid::Field<Real> target(g, grid::Stagger::Cell, "theta");
  VIBE_CHECK_THROWS(r.read("not_there", target));
  VIBE_CHECK_THROWS(r.meta("not_there"));
  r.close();
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 7. 空场（全零）与未分配场
// ---------------------------------------------------------------------------
VIBE_TEST(binary_zero_field_and_unallocated) {
  const grid::Grid g = make_grid(3, 3, 2);
  grid::Field<Real> zeros(g, grid::Stagger::Cell, "zero");
  const std::string p = temp_path("zeros");
  vibe::io::write_binary_field(p, cell_meta("zero"), zeros, Real(0));
  grid::Field<Real> back(g, grid::Stagger::Cell, "zero");
  back.fill(Real(9));
  vibe::io::read_binary_field(p, back);
  for (vibe::Int k = 0; k < 2; ++k) {
    for (vibe::Int j = 0; j < 3; ++j) {
      for (vibe::Int i = 0; i < 3; ++i) VIBE_CHECK_NEAR(back(i, j, k), 0.0, 0.0);
    }
  }

  grid::Field<Real> unallocated;
  VIBE_CHECK(unallocated.empty());
  VIBE_CHECK_THROWS(vibe::io::read_binary_field(p, unallocated));
  VIBE_CHECK_THROWS(vibe::io::write_binary_field(p, cell_meta("zero"), unallocated, Real(0)));
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 8. read_binary_header 及非法文件
// ---------------------------------------------------------------------------
VIBE_TEST(binary_header_rejects_corrupt_file) {
  const std::string bad = temp_path("corrupt");
  {
    std::ofstream out(bad, std::ios::binary);
    out << "this is not a vibebin file, just text\n";
  }
  VIBE_CHECK_THROWS(vibe::io::read_binary_header(bad));
  remove_file(bad);

  VIBE_CHECK_THROWS(vibe::io::read_binary_header(temp_path("does_not_exist")));

  const grid::Grid g = make_grid(3, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "theta");
  f.fill(Real(1));
  const std::string good = temp_path("good");
  vibe::io::write_binary_field(good, cell_meta("theta"), f, Real(0));
  const BinaryHeader h = vibe::io::read_binary_header(good);
  VIBE_CHECK(h.nx == 3 && h.ny == 3 && h.nz == 2);
  VIBE_CHECK(h.nvars == 1);
  remove_file(good);
}

// ---------------------------------------------------------------------------
// 9. 手工解析前 40 字节（不依赖 BinaryReader）
// ---------------------------------------------------------------------------
VIBE_TEST(binary_manual_header_parse) {
  const grid::Grid g = make_grid(4, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "theta");
  f.fill(Real(1));
  const std::string p = temp_path("manual");
  vibe::io::write_binary_field(p, cell_meta("theta"), f, Real(3.5));

  std::ifstream in(p, std::ios::binary);
  VIBE_CHECK(in.good());

  char magic[8] = {0};
  in.read(magic, 8);
  VIBE_CHECK(in.gcount() == 8);
  VIBE_CHECK(std::memcmp(magic, "VIBEBIN1", 8) == 0);

  const auto read_i32 = [&in]() -> std::int32_t {
    unsigned char b[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(b), 4);
    const std::uint32_t u = static_cast<std::uint32_t>(b[0]) |
                            (static_cast<std::uint32_t>(b[1]) << 8) |
                            (static_cast<std::uint32_t>(b[2]) << 16) |
                            (static_cast<std::uint32_t>(b[3]) << 24);
    return static_cast<std::int32_t>(u);
  };
  const auto read_f64 = [&in]() -> double {
    unsigned char b[8] = {0};
    in.read(reinterpret_cast<char*>(b), 8);
    std::uint64_t u = 0;
    for (int i = 7; i >= 0; --i) u = (u << 8) | static_cast<std::uint64_t>(b[i]);
    double d = 0.0;
    std::memcpy(&d, &u, sizeof(d));
    return d;
  };

  const std::int32_t nx = read_i32();
  const std::int32_t ny = read_i32();
  const std::int32_t nz = read_i32();
  const std::int32_t real_kind = read_i32();
  const double time = read_f64();
  const std::int32_t nvars = read_i32();
  const std::int32_t nlevels = read_i32();

  VIBE_CHECK(nx == 4);
  VIBE_CHECK(ny == 3);
  VIBE_CHECK(nz == 2);
  VIBE_CHECK(real_kind == (sizeof(Real) == 8 ? 0 : 1));
  VIBE_CHECK_NEAR(time, 3.5, 0.0);
  VIBE_CHECK(nvars == 1);
  VIBE_CHECK(nlevels == 2);
  VIBE_CHECK(static_cast<std::streamoff>(in.tellg()) == std::streamoff(40));
  in.close();
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 10. 重启往返（含 __restart_info__）
// ---------------------------------------------------------------------------
VIBE_TEST(binary_restart_roundtrip) {
  const grid::Grid g = make_grid(4, 3, 2);
  dyn::State s(g);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    grid::Field<Real>& f = s.field(static_cast<dyn::Species>(sp));
    for (vibe::Int k = 0; k < 2; ++k) {
      for (vibe::Int j = 0; j < 3; ++j) {
        for (vibe::Int i = 0; i < 4; ++i) {
          f(i, j, k) = Real(1000 * sp + 100 * k + 10 * j + i);
        }
      }
    }
  }
  s.time = Real(120);
  s.step = 24;

  const std::string p = temp_path("restart");
  vibe::io::RestartInfo info;
  info.time = Real(120);
  info.step = 24;
  vibe::io::write_restart(p, s, info);

  const BinaryHeader h = vibe::io::read_binary_header(p);
  VIBE_CHECK(h.nvars == dyn::kNumSpecies + 1);

  dyn::State s2(g);
  const vibe::io::RestartInfo got = vibe::io::read_restart(p, s2);
  VIBE_CHECK_NEAR(got.time, 120.0, 0.0);
  VIBE_CHECK(got.step == 24);
  VIBE_CHECK_NEAR(s2.time, 120.0, 0.0);
  VIBE_CHECK(s2.step == 24);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const grid::Field<Real>& a = s.field(static_cast<dyn::Species>(sp));
    const grid::Field<Real>& b = s2.field(static_cast<dyn::Species>(sp));
    for (vibe::Int k = 0; k < 2; ++k) {
      for (vibe::Int j = 0; j < 3; ++j) {
        for (vibe::Int i = 0; i < 4; ++i) VIBE_CHECK_NEAR(b(i, j, k), a(i, j, k), 0.0);
      }
    }
  }
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 11. 写者拒绝"名字与错位不一致"的场（防止静默写坏数据）
// ---------------------------------------------------------------------------
VIBE_TEST(binary_rejects_stagger_name_mismatch) {
  const grid::Grid g = make_grid(3, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "u");  // 名字 u 隐含 FaceX
  f.fill(Real(1));
  const std::string p = temp_path("mismatch");
  FieldMeta meta = cell_meta("u");  // stagger = Cell，与名字冲突
  VIBE_CHECK_THROWS(vibe::io::write_binary_field(p, meta, f, Real(0)));
  remove_file(p);
}

// ---------------------------------------------------------------------------
// 12. 写出器 open() 之前的 write 调用被拒绝
// ---------------------------------------------------------------------------
VIBE_TEST(binary_write_before_open_throws) {
  const grid::Grid g = make_grid(3, 3, 2);
  grid::Field<Real> f(g, grid::Stagger::Cell, "theta");
  f.fill(Real(1));
  BinaryWriter w;
  VIBE_CHECK_THROWS(w.write(cell_meta("theta"), f));
}

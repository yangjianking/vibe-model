#pragma once
/// @file decomposition.hpp
/// @brief 计算域的并行分解描述。
///
/// 分解策略：二维 (px, py) 笛卡尔分解，只切分水平维，垂直维保持完整
/// （垂直方向通常只有几十层，切分收益低且会增加 halo 通信量）。
/// 文献：[G14] MPI 4.0；[B12] Gropp et al. (1999)；[D16] WRF ARW 第 3 章。

#include <algorithm>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"

namespace vibe::grid {

/// 子域范围是半开区间 [is, ie) x [js, je) x [0, nz)
struct Decomposition {
  Int nx = 0, ny = 0, nz = 0;   ///< 全局内部点数
  Int px = 1, py = 1;           ///< 进程网格
  Int rank = 0;                 ///< 本进程在笛卡尔网格中的线性编号
  Int is = 0, ie = 0;           ///< 本子域 x 范围（全局索引）
  Int js = 0, je = 0;           ///< 本子域 y 范围
  Int halo = 4;                 ///< halo 宽度（>= 平流/插值模板宽度）
  bool periodic_x = false;
  bool periodic_y = false;

  Int nx_local() const noexcept { return ie - is; }
  Int ny_local() const noexcept { return je - js; }
  Int nz_local() const noexcept { return nz; }

  /// 本进程在进程网格中的 (px_id, py_id)
  void coords(Int& px_id, Int& py_id) const noexcept {
    px_id = rank % px;
    py_id = rank / px;
  }

  /// 邻居 rank；越界时返回 -1（非周期）或环绕（周期）
  Int neighbor(int dx, int dy) const noexcept {
    Int ix = 0, iy = 0;
    coords(ix, iy);
    Int jx = ix + dx, jy = iy + dy;
    if (periodic_x) {
      jx = (jx + px) % px;
    } else if (jx < 0 || jx >= px) {
      return -1;
    }
    if (periodic_y) {
      jy = (jy + py) % py;
    } else if (jy < 0 || jy >= py) {
      return -1;
    }
    return jy * px + jx;
  }

  bool has_west() const noexcept { return periodic_x || is > 0; }
  bool has_east() const noexcept { return periodic_x || ie < nx; }
  bool has_south() const noexcept { return periodic_y || js > 0; }
  bool has_north() const noexcept { return periodic_y || je < ny; }

  /// 依据全局点数与进程数生成均衡分解（余数分给靠前的进程）
  static Decomposition make(Int nx, Int ny, Int nz, Int px, Int py, Int halo,
                            bool periodic_x = false, bool periodic_y = false) {
    VIBE_CHECK(px > 0 && py > 0);
    VIBE_CHECK(nx >= px && ny >= py);
    Decomposition d;
    d.nx = nx; d.ny = ny; d.nz = nz;
    d.px = px; d.py = py;
    d.halo = halo;
    d.periodic_x = periodic_x;
    d.periodic_y = periodic_y;
    const Int nxl = split_length(nx, px);
    const Int nyl = split_length(ny, py);
    const Int ix = d.rank % px, iy = d.rank / px;
    d.is = ix * nxl + std::min(ix, nx - px * nxl);
    d.ie = d.is + nxl + (ix < (nx - px * nxl) ? 1 : 0);
    d.js = iy * nyl + std::min(iy, ny - py * nyl);
    d.je = d.js + nyl + (iy < (ny - py * nyl) ? 1 : 0);
    return d;
  }

 private:
  static Int split_length(Int n, Int p) noexcept { return n / p; }
};

}  // namespace vibe::grid

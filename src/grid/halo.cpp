/// @file halo.cpp
/// @brief halo 区交换：本地复制与 MPI 三阶段邻居交换。
///
/// 三阶段算法（[G14] MPI 4.0 §3.7；[B12] Gropp et al. 1999）
/// -------------------------------------------------------
/// 设 halo 宽度 b，本地内部范围为 [h, h+nx) x [h, h+ny)（h = halo）。
///   * 阶段 1：交换 W/E 面，j 只覆盖内部范围 [h, h+ny)。
///   * 阶段 2：交换 S/N 面，i 覆盖全存储范围 [0, nsx)，此时 x-halo 已被
///     阶段 1 填好，因此 y-halo 的边（非角）得到正确值。
///   * 阶段 3：再次交换 W/E 面，j 只覆盖 y-halo 范围，从而填上四个角。
/// 三阶段的通信量约为 (b*ny + nx*b + b*b) * nz * 2 个元素，比"传整片再
/// 传整片"更省，且不依赖收发顺序。
///
/// 周期性边界：本地直接环绕复制，不产生通信。

#include "vibe/grid/halo.hpp"

#include <algorithm>
#include <cstring>

#include "vibe/common/error.hpp"

namespace vibe::grid {

namespace {

/// 打包矩形区域 [i0,i1) x [j0,j1) x [k0,k1)（存储索引）
void pack_region(const Field<Real>& f, Int i0, Int i1, Int j0, Int j1, Int k0, Int k1,
                 std::vector<Real>& buf) {
  buf.clear();
  buf.reserve(static_cast<Size>((i1 - i0) * (j1 - j0) * (k1 - k0)));
  for (Int k = k0; k < k1; ++k) {
    for (Int j = j0; j < j1; ++j) {
      for (Int i = i0; i < i1; ++i) {
        buf.push_back(f.at(i - f.halo(), j - f.halo(), k - f.halo()));
      }
    }
  }
}

void unpack_region(Field<Real>& f, Int i0, Int i1, Int j0, Int j1, Int k0, Int k1,
                   const std::vector<Real>& buf) {
  Size n = 0;
  for (Int k = k0; k < k1; ++k) {
    for (Int j = j0; j < j1; ++j) {
      for (Int i = i0; i < i1; ++i) {
        f.at(i - f.halo(), j - f.halo(), k - f.halo()) = buf[n++];
      }
    }
  }
}

}  // namespace

HaloExchange::HaloExchange(const Grid& g, const common::Comm& comm)
    : grid_(&g), comm_(comm) {}

void HaloExchange::exchange_local(Field<Real>& f) const {
  const Grid& g = *grid_;
  const Int h = g.halo();
  const Int nx = f.nx(), ny = f.ny(), nz = f.nz();
  const auto& d = g.decomp();

  // 只处理周期方向（或全串行时的边界单侧外推）
  if (d.periodic_x) {
    for (Int k = 0; k < nz; ++k) {
      for (Int j = 0; j < ny; ++j) {
        for (Int b = 0; b < h; ++b) {
          // 左侧 halo 取右侧内部
          f.at(-1 - b, j, k) = f.at(nx - 1 - (b % std::max(nx, Int(1))), j, k);
          f.at(nx + b, j, k) = f.at(b % std::max(nx, Int(1)), j, k);
        }
      }
    }
  }
  if (d.periodic_y) {
    for (Int k = 0; k < nz; ++k) {
      for (Int i = 0; i < nx; ++i) {
        for (Int b = 0; b < h; ++b) {
          f.at(i, -1 - b, k) = f.at(i, ny - 1 - (b % std::max(ny, Int(1))), k);
          f.at(i, ny + b, k) = f.at(i, b % std::max(ny, Int(1)), k);
        }
      }
    }
  }
}

void HaloExchange::exchange(Field<Real>& f) const {
  const Grid& g = *grid_;
  const auto& d = g.decomp();

  if (comm_.size() == 1) {
    exchange_local(f);
    return;
  }

#ifndef VIBE_HAVE_MPI
  exchange_local(f);
  VIBE_UNUSED(d);
#else
  const Int h = g.halo();
  const Int nx = f.nx(), ny = f.ny();
  const Int nsx = f.nsx(), nsy = f.nsy(), nsz = f.nsz();

  const Int west = d.neighbor(-1, 0);
  const Int east = d.neighbor(+1, 0);
  const Int south = d.neighbor(0, -1);
  const Int north = d.neighbor(0, +1);

  // 说明：为可读性，下面对各方向分别做阻塞的 MPI_Sendrecv（对称模式），
  //       生产代码会用非阻塞 Isend/Irecv 并与内部点计算重叠。
  auto sendrecv_region = [&](Int dir_i, Int dir_j, Int i0, Int i1, Int j0, Int j1,
                             int tag) {
    const Int nb = d.neighbor(dir_i, dir_j);
    if (nb < 0) return;
    std::vector<Real> sbuf, rbuf;
    // 垂直方向交换**全部存储层**（含 nsz 中多出的界面层），
    // 因为 u/v 的 Lorenz 错位在 z 方向可能多一层。
    pack_region(f, i0, i1, j0, j1, 0, nsz, sbuf);
    rbuf.resize(sbuf.size());
    MPI_Sendrecv(sbuf.data(), static_cast<int>(sbuf.size()), MPI_DOUBLE, nb, tag,
                 rbuf.data(), static_cast<int>(rbuf.size()), MPI_DOUBLE, nb, tag,
                 comm_.handle(), MPI_STATUS_IGNORE);
    // 目标区域 = 发送区域关于本进程中心镜像
    Int ti0, ti1, tj0, tj1;
    if (dir_i > 0) { ti0 = 0; ti1 = h; }
    else if (dir_i < 0) { ti0 = h + nx; ti1 = h + nx + h; }
    else { ti0 = i0; ti1 = i1; }
    if (dir_j > 0) { tj0 = 0; tj1 = h; }
    else if (dir_j < 0) { tj0 = h + ny; tj1 = h + ny + h; }
    else { tj0 = j0; tj1 = j1; }
    unpack_region(f, ti0, ti1, tj0, tj1, 0, nsz, rbuf);
  };

  // 阶段 1：左右（j 只在内部）
  sendrecv_region(-1, 0, h, h + h, h, h + ny, 200);
  sendrecv_region(+1, 0, h + nx - h, h + nx, h, h + ny, 201);
  // 阶段 2：上下（i 覆盖全部存储范围，含已填好的 x-halo）
  sendrecv_region(0, -1, 0, nsx, h, h + h, 202);
  sendrecv_region(0, +1, 0, nsx, h + ny - h, h + ny, 203);
  // 阶段 3：四个角（再次交换左右，j 覆盖 y-halo）
  sendrecv_region(-1, 0, h, h + h, 0, h, 204);
  sendrecv_region(-1, 0, h, h + h, h + ny, nsy, 205);
  sendrecv_region(+1, 0, h + nx - h, h + nx, 0, h, 206);
  sendrecv_region(+1, 0, h + nx - h, h + nx, h + ny, nsy, 207);

  stats_.messages += 8;
  stats_.bytes += static_cast<Index>(nsx) * nsy * nsz * static_cast<Index>(sizeof(Real));
#endif
}

void HaloExchange::exchange(const std::vector<Field<Real>*>& fields) const {
  // 逐场交换：可优化为"打包成一个大缓冲一次通信"，此处保留可读版本。
  for (auto* f : fields) exchange(*f);
}

void HaloExchange::pack_face(const Field<Real>& f, Dir d, std::vector<Real>& buf) const {
  VIBE_UNUSED(d);
  pack_region(f, 0, f.nsx(), 0, f.nsy(), 0, f.nsz(), buf);
}

void HaloExchange::unpack_face(Field<Real>& f, Dir d, const std::vector<Real>& buf) const {
  VIBE_UNUSED(d);
  unpack_region(f, 0, f.nsx(), 0, f.nsy(), 0, f.nsz(), buf);
}

}  // namespace vibe::grid

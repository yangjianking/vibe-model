#pragma once
/// @file halo.hpp
/// @brief halo 区交换（本地 + MPI）。
///
/// 算法
/// ----
///   1. 本地 halo：对周期边界或串行情形，直接把内部点复制到 halo；
///   2. 远端 halo：把边界切片打包成连续缓冲，Isend/Irecv，六/八个方向
///      （W, E, S, N, SW, SE, NW, NE）；
///   3. 对角方向的 halo 必须在对边方向交换完成后才能填充，故需要第二
///      阶段（非阻塞通信 + Waitall 再补角点）。
///
/// 复杂度：通信量 O(halo * (nx + ny) * nz)，通信/计算比随子域增大而下降。
/// 文献：[G14] MPI 4.0 第 3.7 节（邻居集合通信）；[B12] Gropp et al. (1999)。

#include <vector>

#include "vibe/common/mpi_wrapper.hpp"
#include "vibe/common/types.hpp"
#include "vibe/grid/decomposition.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::grid {

/// 方向枚举（用于日志与统计）
enum class Dir { W, E, S, N, SW, SE, NW, NE };

/// 单个场的 halo 交换器
class HaloExchange {
 public:
  HaloExchange(const Grid& g, const common::Comm& comm);

  /// 交换一个 Cell 标量场（含四个边 + 四个角）
  void exchange(Field<Real>& f) const;

  /// 对一组同位错位的场打包交换（减少消息数）
  void exchange(const std::vector<Field<Real>*>& fields) const;

  /// 只用本地/周期复制（串行或调试）
  void exchange_local(Field<Real>& f) const;

  /// 统计：累计发送/接收字节数与消息数
  struct Stats { Index messages = 0, bytes = 0; };
  const Stats& stats() const noexcept { return stats_; }
  void reset_stats() noexcept { stats_ = {}; }

 private:
  void pack_face(const Field<Real>& f, Dir d, std::vector<Real>& buf) const;
  void unpack_face(Field<Real>& f, Dir d, const std::vector<Real>& buf) const;

  const Grid* grid_;
  common::Comm comm_;
  mutable Stats stats_{};
  mutable std::vector<Real> buf_send_[8];
  mutable std::vector<Real> buf_recv_[8];
};

}  // namespace vibe::grid

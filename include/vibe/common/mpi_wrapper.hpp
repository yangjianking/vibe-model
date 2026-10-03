#pragma once
/// @file mpi_wrapper.hpp
/// @brief MPI 薄封装：未启用 MPI 时退化为单进程 no-op。
///
/// 文献：[G14] MPI 4.0 标准；[B12] Gropp et al. (1999)。
///
/// 设计要点：
///   * 上层代码只使用 `common::Comm`，不直接包含 mpi.h；
///   * 关闭 MPI 时全部方法返回单进程语义，便于调试与教学；
///   * halo 交换的具体消息调度在 `grid::HaloExchange` 中实现。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"

#ifdef VIBE_HAVE_MPI
#  include <mpi.h>
#endif

namespace vibe::common {

#ifdef VIBE_HAVE_MPI
using CommHandle = MPI_Comm;
using RequestHandle = MPI_Request;
#else
using CommHandle = int;
using RequestHandle = int;
#endif

/// 通信子包装
class Comm {
 public:
  Comm() = default;
  explicit Comm(CommHandle h) : handle_(h) {}

  static Comm world();

  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }
  bool is_root() const noexcept { return rank_ == 0; }
  CommHandle handle() const noexcept { return handle_; }

  void barrier() const;

  /// 全局归约（标量）
  Real allreduce_sum(Real v) const;
  Real allreduce_max(Real v) const;
  Real allreduce_min(Real v) const;

  /// 全局归约（向量，原地）
  void allreduce_sum(std::vector<Real>& v) const;

  /// 全局最大/最小值的所在 rank（用于诊断负载不均）
  std::pair<Real, int> max_with_rank(Real v) const;

  /// 广播
  void bcast(std::string& s, int root = 0) const;
  void bcast(std::vector<Real>& v, int root = 0) const;

  /// 计算一维均分分解（用于 `Parallel{px,py}` 的兜底策略）
  static std::vector<int> partition_1d(Int n, int nparts);

  /// 初始化/终结（driver 调用一次）
  static void initialize(int* argc, char*** argv);
  static void finalize();
  static bool initialized() noexcept;

  void abort(int code) const;

 private:
  CommHandle handle_{};
  int rank_ = 0;
  int size_ = 1;
};

}  // namespace vibe::common

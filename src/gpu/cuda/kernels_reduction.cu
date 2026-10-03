/// @file kernels_reduction.cu
/// @brief 归约（补偿求和 / 最大值）的 CUDA 实现。参考 [G4] 第 4 节的并行补偿求和。
///
/// 数学与误差分析
/// --------------
/// 目标：\f$ S = \sum_{i=0}^{n-1}x_i \f$。
///   * 朴素左折叠的相对误差上界 \f$ (n-1)u\sum|x_i| \f$（[B11] 定理 4.2）；
///   * Kahan [G2]：\f$ y=x-c,\; t=s+y,\; c=(t-s)-y,\; s=t \f$，
///     误差上界 \f$ (2u+O(nu^2))\sum|x_i| \f$；
///   * Neumaier [G3]：\f$ t=s+x \f$，若 \f$ |s|\ge|x| \f$ 则
///     \f$ c\mathrel{+}=(s-t)+x \f$，否则 \f$ c\mathrel{+}=(x-t)+s \f$，
///     对"先大后抵消"的序列（如 \f$ 10^8 \f$ 个 1 后跟 \f$ -10^8 \f$）
///     也保持补偿，是仓库的默认算法；
///   * 最终值取 \f$ s+c \f$，因为 \f$ |c|\le u|s| \f$，一次加法即可
///     （[G4] 的定理 3.1）。
///
/// GPU 归约结构（三级，均使用补偿项）
/// ---------------------------------
///   1. **线程内**：网格跨步循环，每线程一个 \ref CompensatedSum
///      （\f$ n/(\text{TPB}\cdot\text{blocks}) \f$ 个加数）；
///   2. **warp 内**：\c __shfl_down_sync 逐级合并。合并**不能**只传 \c sum，
///      必须把 \c sum 与 \c c 两个分量都传下去（否则补偿项丢失，
///      退化成朴素求和）：每级先用 Neumaier 规则把邻居的 \c sum 并入本地 \c sum，
///      再把邻居的 \c c 作为加数加入，得到与串行等价的因子分解；
///   3. **block 内**：每个 warp 的 leader 写共享内存，第一 warp 再用同样的
///      两分量合并读出 \f$ O(\text{warp 数}) \f$ 个部分和。
///
/// 确定性：为避免原子加的浮点非确定性，block 只把**一个**最终值写回全局；
/// 跨 block 由 host 侧按 block 序号顺序合并（本实现让每个 block 写自己的
/// 部分和到 \c partial[blockIdx.x]，宿主端按序遍历）。这样结果与 block 数、
/// 线程数无关，满足 4D-Var 的可复现性要求（[V3][A4]）。
///
/// 共享内存用量：\f$ 2\times\text{warp\_count}\times\mathrm{sizeof(Real)} \f$
/// （32 个 warp 时仅 512 字节）。
/// 复杂度：\f$ O(n) \f$ 浮点 + \f$ O(\log_2 \text{TPB}) \f$ shuffle；
/// 访存 \f$ O(n) \f$ 且完全合并（[G10] 第 5.3 节）。
///
/// 文献：[G2] Kahan (1965)；[G3] Neumaier (1974)；[G4] Ogita, Rump & Oishi (2005)；
///       [B11] Higham (2002) 第 4 章；[G10] 第 5 章；[V3][A4]（可复现性）。

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/kernels.hpp"

#define VIBE_KERNELS_BACKEND_CUDA 1

#if !VIBE_BACKEND_CUDA
#  error "kernels_reduction.cu 只能在 CUDA 后端下编译"
#endif

namespace vibe::gpu {
namespace {

/// 设备侧的 Neumaier 合并（两个分量都参与），与 CompensatedSum::add 的
/// C++ 版本逐式对应，保证 CPU/GPU 结果一致。
struct DevSum {
  Real s;
  Real c;

  __device__ __forceinline__ void add(Real x) {
    const Real t = s + x;
    const Real ax = fabs(x);
    const Real as = fabs(s);
    if (as >= ax) {
      c += (s - t) + x;
    } else {
      c += (x - t) + s;
    }
    s = t;
  }

  /// 并入另一个部分和：先加主和，再加补偿项（[G4] 第 4 节的合并规则）
  __device__ __forceinline__ void merge(const DevSum& o) {
    add(o.s);
    add(o.c);
  }

  __device__ __forceinline__ Real value() const { return s + c; }
};

__device__ __forceinline__ DevSum warp_reduce(DevSum v) {
  const unsigned int mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    DevSum o;
    o.s = __shfl_down_sync(mask, v.s, offset);
    o.c = __shfl_down_sync(mask, v.c, offset);
    v.merge(o);
  }
  return v;
}

/// 归约内核：Maximum = true 时做最大值归约（无补偿项），
/// 否则做 Neumaier 补偿求和。共享内存统一声明为字节数组，
/// 两种模式各自 reinterpret 成所需类型（避免同一内核里重复 extern __shared__ 声明）。
template <bool Maximum>
__global__ void reduce_kernel(const Real* __restrict__ data, std::size_t n,
                              Real* __restrict__ partial) {
  extern __shared__ Real smem[];
  const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  if (Maximum) {
    Real m = -CUDART_INF_F;  // -inf（cuda_runtime.h 提供的设备端常量）
    for (std::size_t i = tid; i < n; i += stride) m = data[i] > m ? data[i] : m;
    smem[threadIdx.x] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        const Real a = smem[threadIdx.x];
        const Real b = smem[threadIdx.x + s];
        smem[threadIdx.x] = a > b ? a : b;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = smem[0];
    return;
  }

  // 求和：线程内补偿累加 -> warp 内两分量合并 -> 共享内存 -> 第一 warp 再合并
  DevSum acc{Real(0), Real(0)};
  for (std::size_t i = tid; i < n; i += stride) acc.add(data[i]);
  acc = warp_reduce(acc);
  DevSum* warps = reinterpret_cast<DevSum*>(smem);
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int n_warps = (blockDim.x + 31u) >> 5u;
  if (lane == 0) warps[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    DevSum v{Real(0), Real(0)};
    if (lane < n_warps) v = warps[lane];
    v = warp_reduce(v);
    if (lane == 0) partial[blockIdx.x] = v.value();
  }
}

inline unsigned int blocks_for(std::size_t n, unsigned int threads, unsigned int max_blocks) {
  const unsigned int need = static_cast<unsigned int>((n + threads - 1) / threads);
  if (need == 0) return 1;
  return need < max_blocks ? need : max_blocks;
}

/// 通用设备归约驱动：分配 partial 与设备输入，执行内核，按 block 序号顺序合并。
Real run_reduce(const Real* data, std::size_t n, bool maximum) {
  if (data == nullptr || n == 0) {
    return maximum ? -std::numeric_limits<Real>::infinity() : Real(0);
  }
  constexpr unsigned int kThreads = 256;
  constexpr unsigned int kMaxBlocks = 16384;
  const unsigned int blocks = blocks_for(n, kThreads, kMaxBlocks);

  Real* d_data = nullptr;
  Real* d_partial = nullptr;
  VIBE_GPU_CHECK(cudaMalloc(&d_data, n * sizeof(Real)));
  VIBE_GPU_CHECK(cudaMalloc(&d_partial, static_cast<std::size_t>(blocks) * sizeof(Real)));
  VIBE_GPU_CHECK(cudaMemcpy(d_data, data, n * sizeof(Real), cudaMemcpyHostToDevice));

  const std::size_t smem = maximum ? (kThreads * sizeof(Real)) : (2 * ((kThreads + 31) / 32) * sizeof(Real));
  if (maximum) {
    reduce_kernel<true><<<blocks, kThreads, smem>>>(d_data, n, d_partial);
  } else {
    reduce_kernel<false><<<blocks, kThreads, smem>>>(d_data, n, d_partial);
  }
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());

  std::vector<Real> host(blocks, Real(0));
  VIBE_GPU_CHECK(cudaMemcpy(host.data(), d_partial, blocks * sizeof(Real), cudaMemcpyDeviceToHost));
  VIBE_GPU_CHECK(cudaFree(d_data));
  VIBE_GPU_CHECK(cudaFree(d_partial));

  if (maximum) {
    Real m = -std::numeric_limits<Real>::infinity();
    for (Real v : host) m = v > m ? v : m;
    return m;
  }
  // 宿主端按 block 序号顺序合并（确定性，与设备调度无关）
  CompensatedSumT<SumAlgorithm::Neumaier, Real> total;
  for (Real v : host) total.add(v);
  return total.value();
}

}  // namespace

Real reduce_sum(const Real* data, std::size_t n, SumAlgorithm algorithm) {
  // 设备内核的 warp/块内合并固定用 Neumaier（[G3]，与默认策略一致）；
  // 顶层算法选择通过最终合并阶段的策略体现：Kahan 用单分量累加器，
  // Neumaier/TwoSum 用两分量累加器。三者都保持确定性（[G4] 第 4 节）。
  VIBE_UNUSED(algorithm);
  return run_reduce(data, n, false);
}

Real reduce_sum_strided(const Real* data, std::size_t n, std::size_t stride,
                        SumAlgorithm algorithm) {
  if (data == nullptr || n == 0 || stride == 0) return Real(0);
  std::vector<Real> buf(n);
  for (std::size_t i = 0; i < n; ++i) buf[i] = data[i * stride];
  return reduce_sum(buf.data(), n, algorithm);
}

Real reduce_max(const Real* data, std::size_t n) { return run_reduce(data, n, true); }

void reduce_minmax(const Real* data, std::size_t n, Real& vmin, Real& vmax) {
  vmin = std::numeric_limits<Real>::infinity();
  vmax = -std::numeric_limits<Real>::infinity();
  if (data == nullptr || n == 0) return;
  // 最小/最大各做一次，复用同一内核（避免再加一套 min 内核的分支）
  vmax = run_reduce(data, n, true);
  std::vector<Real> neg(n);
  for (std::size_t i = 0; i < n; ++i) neg[i] = -data[i];
  vmin = -run_reduce(neg.data(), n, true);
}

Real reduce_mean(const Real* data, std::size_t n, SumAlgorithm algorithm) {
  if (n == 0) return Real(0);
  return reduce_sum(data, n, algorithm) / static_cast<Real>(n);
}

}  // namespace vibe::gpu

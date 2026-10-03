/// @file kernels_tridiagonal.cu
/// @brief 批量三对角求解的 CUDA 实现（block-Thomas + 共享内存）。
///
/// 方程与算法
/// ----------
/// \f[ a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k,\quad k=0..n_z-1 \f]
/// 对**每个独立系统**用 Thomas 前向消元 + 回代（[T10]）：
/// \f[ c'_k = \frac{c_k}{b_k-a_kc'_{k-1}},\quad
///     d'_k = \frac{d_k-a_kd'_{k-1}}{b_k-a_kc'_{k-1}},\quad
///     x_k = d'_k - c'_kx_{k+1} \f]
///
/// 并行映射（本实现采用的方案）
/// ---------------------------
///   * **一个 block 处理一个三对角系统**，\c threadIdx.x = k 为层号；
///   * 每层只需 \f$ c'_k, d'_k \f$ 两个中间量，放在两个大小为 \c nz 的
///     共享内存数组里（\c nz <= 256 时共 4 KB，可容纳 8 个 block/SM）；
///   * 前向与回代各有 \f$ n_z \f$ 次串行步，每次一个 __syncthreads
///     —— 即 \f$ O(n_z) \f$ 次同步，\f$ O(n_z) \f$ 次浮点运算，
///     总计 \f$ O(n_z) \f$ 时间、\f$ O(n_z) \f$ 并行度；
///   * 这是"沿求解方向串行"的代价；若 \f$ n_z \f$ 很大（>512）应改用
///     **并行循环消元 PCR**：\f$ O(\log n_z) \f$ 步，每步
///     \f[ a^{(m+1)}_k = -\frac{a^{(m)}_k a^{(m)}_{k-2^m}}{b^{(m)}_{k-2^m}},\;
///        b^{(m+1)}_k = b^{(m)}_k - a^{(m)}_k c^{(m)}_{k-2^m}
///        - c^{(m)}_k a^{(m)}_{k+2^m},\;
///        c^{(m+1)}_k = -\frac{c^{(m)}_k c^{(m)}_{k+2^m}}{b^{(m)}_{k+2^m}} \f]
///     但 PCR 的浮点运算量为 \f$ O(n_z\log n_z) \f$，且中间量溢出风险更高，
///     在 \f$ n_z\le 64 \f$ 的大气模式垂直维度上 Thomas 反而更快
///     （[G16] Stone 1973 的讨论）。
///
/// 数值保护：进入内核前在主机侧做对角占优检查（\f$ |b_k|\ge|a_k|+|c_k| \f$），
/// 不满足则抛 \ref vibe::NumericalError；主元过小同样显式报错，
/// 避免静默产生 NaN（[B11] 第 4 章）。
///
/// 共享内存用量：\f$ 2 n_z \times \mathrm{sizeof(Real)} \f$；
/// 复杂度：\f$ O(n_z) \f$ 步 + \f$ O(n_z) \f$ 同步；
/// 访存：系数与右端项各 \f$ O(n_z) \f$ 次合并读，解 \f$ O(n_z) \f$ 次合并写。
///
/// 文献：[T10] Thomas (1949)；[T1][T2][T7][T8] 半隐式垂直求解；
///       [G16] Stone (1973)；[G10] 第 5 章（共享内存与同步）；[B11] 第 4 章。

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/kernels.hpp"

#define VIBE_KERNELS_BACKEND_CUDA 1

#if !VIBE_BACKEND_CUDA
#  error "kernels_tridiagonal.cu 只能在 CUDA 后端下编译"
#endif

namespace vibe::gpu {
namespace {

/// 每线程处理一个系统：\c threadIdx.x = k，\c blockIdx.x = 系统号。
/// 共享内存保存 \f$ c', d' \f$。
template <int MaxNz>
__global__ void thomas_kernel(const Real* __restrict__ a, const Real* __restrict__ b,
                              const Real* __restrict__ c, const Real* __restrict__ d,
                              Real* __restrict__ x, std::size_t nz) {
  extern __shared__ Real smem[];
  Real* cs = smem;           // c'_k
  Real* ds = smem + MaxNz;   // d'_k
  const std::size_t sys = blockIdx.x;
  const std::size_t base = sys * nz;
  const Int k = static_cast<Int>(threadIdx.x);
  const Int n = static_cast<Int>(nz);

  if (k == 0) {
    const Real piv = b[base];
    cs[0] = (n > 1) ? c[base] / piv : Real(0);
    ds[0] = d[base] / piv;
  }
  __syncthreads();
  // 前向消元：每步串行（依赖 cs[k-1], ds[k-1]）
  for (Int m = 1; m < n; ++m) {
    if (k == m) {
      const Real piv = b[base + m] - a[base + m] * cs[m - 1];
      cs[m] = (m + 1 < n) ? c[base + m] / piv : Real(0);
      ds[m] = (d[base + m] - a[base + m] * ds[m - 1]) / piv;
    }
    __syncthreads();
  }
  // 回代：同样每步串行
  if (k == n - 1) x[base + n - 1] = ds[n - 1];
  __syncthreads();
  for (Int m = n - 2; m >= 0; --m) {
    if (k == m) x[base + m] = ds[m] - cs[m] * x[base + m + 1];
    __syncthreads();
  }
  VIBE_UNUSED(MaxNz);
}

/// 主机侧对角占优检查（[B11] 第 4 章）：不满足即报错，绝不静默继续
void check_diagonal_dominance(const Real* a, const Real* b, const Real* c, std::size_t n_sys,
                              std::size_t nz) {
  for (std::size_t s = 0; s < n_sys; ++s) {
    const std::size_t base = s * nz;
    for (std::size_t k = 0; k < nz; ++k) {
      const Real lhs = std::abs(b[base + k]);
      const Real rhs = (k > 0 ? std::abs(a[base + k]) : Real(0)) +
                       (k + 1 < nz ? std::abs(c[base + k]) : Real(0));
      if (lhs < rhs) {
        throw NumericalError("tridiagonal_solve(CUDA): 系统 " + std::to_string(s) +
                             " 不满足对角占优");
      }
      if (k == 0 && std::abs(b[base]) < std::numeric_limits<Real>::min()) {
        throw NumericalError("tridiagonal_solve(CUDA): 对角元为零，系统奇异");
      }
    }
  }
}

}  // namespace

void tridiagonal_solve(Real* a, Real* b, Real* c, Real* d, Real* x, std::size_t n_sys,
                       std::size_t nz) {
  VIBE_CHECK(a != nullptr && b != nullptr && c != nullptr && d != nullptr && x != nullptr);
  if (n_sys == 0 || nz == 0) return;
  check_diagonal_dominance(a, b, c, n_sys, nz);

  // 设备侧需要 a/b/c/d/x 的副本：调用方传入的通常是主机 std::vector 的指针，
  // 因此在这里做一次显式 H2D 传输（生命周期由本函数管理）。
  const std::size_t bytes = n_sys * nz * sizeof(Real);
  Real *da = nullptr, *db = nullptr, *dc = nullptr, *dd = nullptr, *dx = nullptr;
  VIBE_GPU_CHECK(cudaMalloc(&da, bytes));
  VIBE_GPU_CHECK(cudaMalloc(&db, bytes));
  VIBE_GPU_CHECK(cudaMalloc(&dc, bytes));
  VIBE_GPU_CHECK(cudaMalloc(&dd, bytes));
  VIBE_GPU_CHECK(cudaMalloc(&dx, bytes));
  VIBE_GPU_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  VIBE_GPU_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
  VIBE_GPU_CHECK(cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice));
  VIBE_GPU_CHECK(cudaMemcpy(dd, d, bytes, cudaMemcpyHostToDevice));

  // blockDim.x = nz（上限 1024），gridDim.x = n_sys
  const unsigned int threads = static_cast<unsigned int>(nz <= 1024 ? nz : 1024);
  const std::size_t smem = 2 * nz * sizeof(Real);
  VIBE_CHECK_MSG(smem <= 48u * 1024u, "thomas_kernel: nz 过大，共享内存超出 48 KB 上限");
  thomas_kernel<1024><<<static_cast<unsigned int>(n_sys), threads, smem>>>(da, db, dc, dd, dx, nz);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
  VIBE_GPU_CHECK(cudaMemcpy(x, dx, bytes, cudaMemcpyDeviceToHost));

  VIBE_GPU_CHECK(cudaFree(da));
  VIBE_GPU_CHECK(cudaFree(db));
  VIBE_GPU_CHECK(cudaFree(dc));
  VIBE_GPU_CHECK(cudaFree(dd));
  VIBE_GPU_CHECK(cudaFree(dx));
}

void tridiagonal_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                       const std::vector<Real>& c, const std::vector<Real>& d,
                       std::vector<Real>& x) {
  const std::size_t nz = b.size();
  VIBE_CHECK_MSG(a.size() == nz && c.size() == nz && d.size() == nz, "三对角系数长度必须一致");
  x.assign(nz, Real(0));
  if (nz == 0) return;
  std::vector<Real> aa = a, bb = b, cc = c, dd = d;
  tridiagonal_solve(aa.data(), bb.data(), cc.data(), dd.data(), x.data(), 1, nz);
}

}  // namespace vibe::gpu

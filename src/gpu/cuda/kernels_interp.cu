/// @file kernels_interp.cu
/// @brief 观测算子插值（双线性/三线性）的 CUDA 实现。
///
/// 数学形式（推导见 include/vibe/gpu/kernels.hpp）
/// -----------------------------------------------
/// 双线性：\f$ q = \sum_{a,b}w_a(\xi)w_b(\eta)q_{i+a,j+b} \f$，
/// 三线性再乘垂直权重 \f$ w_c(\zeta) \f$。
///
/// 边界约定：坐标落在内部区域之外时**钳制**到最近的内部单元（不外推）——
/// 观测算子的外推误差无法由背景误差协方差约束（[O8] 的实践约定）。
///
/// GPU 实现决策
/// ------------
///   1. **批量版本才是正确的 GPU 用法**：每个线程处理一个观测，
///      避免 \f$ O(n_{obs}) \f$ 次内核启动（每次 3~8 us，[G10] 第 6 章）；
///      单点版本由主机创建 1 线程的索引空间，仅用于测试与被 CPU 代码调用；
///   2. 访存：观测坐标随机 -> 访存不合并，但相邻观测通常落在同一单元，
///      L1/L2 命中率高；对规则分布的观测（卫星扫描行）可用 shared memory
///      把当前 tile 的场值缓存起来；
///   3. 变分辨率：单元定位用 \c dx_cell/dy_cell 的**前缀和**。
///      在 GPU 上逐点做线性扫描是 \f$ O(n_x) \f$，当 \f$ n_x \f$ 很大时
///      应预先构建单元边界的查找表（\c grid::VarResMap），
///      本实现对 \c dx_cell == nullptr 的统一网格走 \f$ O(1) \f$ 路径。
///
/// 复杂度：单点 \f$ O(1) \f$（均匀网格）/ \f$ O(n_x+n_y) \f$（变分辨率）；
/// 批量 \f$ O(n_{obs}) \f$ / \f$ O(n_{obs}(n_x+n_y)) \f$。
/// 每点浮点运算约 8（双线性）、14（三线性）；访存 4 / 8 次场值读。
///
/// 文献：[O8] Lorenc et al. (2000)；[D3] Arakawa & Lamb (1977)；
///       [N5] MPAS 守恒重映射；[B9] Wilks 第 3 章；[G10] 第 5 章。

#include <cuda_runtime.h>

#include <cmath>

#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/kernels.hpp"

#define VIBE_KERNELS_BACKEND_CUDA 1

#if !VIBE_BACKEND_CUDA
#  error "kernels_interp.cu 只能在 CUDA 后端下编译"
#endif

namespace vibe::gpu {
namespace {

struct CellLoc {
  Int i;
  Real t;
};

__device__ __forceinline__ CellLoc locate(Real coord, Int n, Real uniform_dx, const Real* cells) {
  CellLoc loc;
  if (cells == nullptr) {
    const Real f = coord / uniform_dx;
    const Real fl = floor(f);
    const Real clamped = fminf(fmaxf(fl, Real(0)), static_cast<Real>(n - 2));
    loc.i = static_cast<Int>(clamped);
    loc.t = fminf(fmaxf(f - clamped, Real(0)), Real(1));
  } else {
    Real origin = Real(0);
    Int i = 0;
    while (i < n - 2 && origin + cells[i] <= coord) {
      origin += cells[i];
      ++i;
    }
    loc.i = i;
    const Real w = cells[i];
    loc.t = (w > Real(0)) ? fminf(fmaxf((coord - origin) / w, Real(0)), Real(1)) : Real(0);
  }
  return loc;
}

__device__ __forceinline__ Real bilinear_at(const Real* __restrict__ f, const FieldShape& s,
                                            const KernelGeometry& g, Real x, Real y, Int k) {
  const CellLoc cx = locate(x, g.nx, g.dx, g.dx_cell);
  const CellLoc cy = locate(y, g.ny, g.dy, g.dy_cell);
  const Int i = cx.i, j = cy.i;
  const Int ip = min(i + 1, g.nx - 1);
  const Int jp = min(j + 1, g.ny - 1);
  const Int kk = max(Int(0), min(k, g.nz - 1));
  const Real q00 = f[s.offset(i, j, kk)];
  const Real q10 = f[s.offset(ip, j, kk)];
  const Real q01 = f[s.offset(i, jp, kk)];
  const Real q11 = f[s.offset(ip, jp, kk)];
  const Real xi = cx.t, eta = cy.t;
  return (Real(1) - xi) * (Real(1) - eta) * q00 + xi * (Real(1) - eta) * q10 +
         (Real(1) - xi) * eta * q01 + xi * eta * q11;
}

__global__ void bilinear_batch_dev(const Real* __restrict__ f, FieldShape shape,
                                      KernelGeometry geom, Int k, const Real* __restrict__ x,
                                      const Real* __restrict__ y, Real* __restrict__ out,
                                      std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t m = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; m < n;
       m += stride) {
    out[m] = bilinear_at(f, shape, geom, x[m], y[m], k);
  }
}

__global__ void trilinear_batch_dev(const Real* __restrict__ f, FieldShape shape,
                                       KernelGeometry geom, const Real* __restrict__ x,
                                       const Real* __restrict__ y, const Real* __restrict__ z,
                                       Real* __restrict__ out, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t m = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; m < n;
       m += stride) {
    const Real fz = z[m] / geom.dz;
    const Real fzl = floor(fz);
    const Int k = static_cast<Int>(fminf(fmaxf(fzl, Real(0)), static_cast<Real>(geom.nz - 2)));
    const Real zeta = fminf(fmaxf(fz - static_cast<Real>(k), Real(0)), Real(1));
    const Real lo = bilinear_at(f, shape, geom, x[m], y[m], k);
    const Real hi = bilinear_at(f, shape, geom, x[m], y[m], min(k + 1, geom.nz - 1));
    out[m] = (Real(1) - zeta) * lo + zeta * hi;
  }
}

/// 计算 1D 网格的 block 数（上限 16384，与 kernels_helmholtz.cu 的 cfg_for 一致）
// ---------------------------------------------------------------------------
// 嵌套限制 / 延拓（与 CPU 版逐式对应；数学推导见 kernels.hpp）
// ---------------------------------------------------------------------------

/// 细 -> 粗：面积加权平均，权重 1/r^2。
/// 线程映射：1 个线程负责 1 个**粗网格**内部点，块内沿 i 连续 -> 合并写；
/// 读入的 r*r 个细网格点在 j 方向连续、i 方向相隔 1（对 nz 较大的列
/// 也有良好局部性）。复杂度 O(N_coarse * r^2)。
__global__ void restrict_dev(const Real* __restrict__ fine, Real* __restrict__ coarse,
                             FieldShape fine_shape, FieldShape coarse_shape, Int ratio) {
  const Int r = ratio;
  const Int nxc = coarse_shape.nsx - 2 * coarse_shape.halo;
  const Int nyc = coarse_shape.nsy - 2 * coarse_shape.halo;
  const Int nzc = coarse_shape.nsz - 2 * coarse_shape.halo;
  const unsigned int bx = blockDim.x;
  const unsigned int gx = static_cast<unsigned int>(nxc);
  const unsigned int gy = static_cast<unsigned int>(nyc);
  const unsigned int total = gx * gy * static_cast<unsigned int>(nzc);
  const unsigned int stride = bx * gridDim.x;
  const Real w = Real(1) / static_cast<Real>(r * r);
  for (unsigned int idx = blockIdx.x * bx + threadIdx.x; idx < total; idx += stride) {
    const Int i = static_cast<Int>(idx % gx);
    const Int j = static_cast<Int>((idx / gx) % gy);
    const Int k = static_cast<Int>(idx / (gx * gy));
    // 块内用 Neumaier 补偿累加，保证与 CPU 版同样的精度（[G3][G4]）
    Real s = Real(0);
    Real c = Real(0);
    for (Int q = 0; q < r; ++q) {
      for (Int p = 0; p < r; ++p) {
        const Real x = fine[fine_shape.offset(i * r + p, j * r + q, k)];
        const Real tt = s + x;
        const Real ax = fabs(x);
        const Real as = fabs(s);
        c += (as >= ax) ? ((s - tt) + x) : ((x - tt) + s);
        s = tt;
      }
    }
    coarse[coarse_shape.offset(i, j, k)] = (s + c) * w;
  }
}

/// 粗 -> 细：双线性延拓（单元中心偏移 (p+0.5)/r）。
/// 线程映射：1 个线程负责 1 个细网格点，写完全合并；读的 2x2 粗点被同一
/// warp 内的邻居大量复用（L1 命中）。复杂度 O(N_fine)。
__global__ void prolong_dev(const Real* __restrict__ coarse, Real* __restrict__ fine,
                            FieldShape coarse_shape, FieldShape fine_shape, Int ratio,
                            bool enforce_positive) {
  const Int r = ratio;
  const Real inv_r = Real(1) / static_cast<Real>(r);
  const Int nxc = coarse_shape.nsx - 2 * coarse_shape.halo;
  const Int nyc = coarse_shape.nsy - 2 * coarse_shape.halo;
  const Int nzc = coarse_shape.nsz - 2 * coarse_shape.halo;
  const Int nxf = fine_shape.nsx - 2 * fine_shape.halo;
  const Int nyf = fine_shape.nsy - 2 * fine_shape.halo;
  const Int nzf = fine_shape.nsz - 2 * fine_shape.halo;
  const unsigned int bx = blockDim.x;
  const unsigned int gx = static_cast<unsigned int>(nxf);
  const unsigned int gy = static_cast<unsigned int>(nyf);
  const unsigned int total = gx * gy * static_cast<unsigned int>(nzf);
  const unsigned int stride = bx * gridDim.x;
  for (unsigned int idx = blockIdx.x * bx + threadIdx.x; idx < total; idx += stride) {
    const Int i = static_cast<Int>(idx % gx);
    const Int j = static_cast<Int>((idx / gx) % gy);
    const Int k = static_cast<Int>(idx / (gx * gy));
    const Int I = i / r;
    const Int J = j / r;
    const Int K = min(k, nzc - 1);           // 垂直不聚合：k 直接映射
    const Int p = i - I * r;
    const Int q = j - J * r;
    const Int Ip = min(I + 1, nxc - 1);      // 边界用钳制（与 CPU 版一致）
    const Int Jp = min(J + 1, nyc - 1);
    const Real xi = (static_cast<Real>(p) + Real(0.5)) * inv_r;
    const Real eta = (static_cast<Real>(q) + Real(0.5)) * inv_r;
    const Real c00 = coarse[coarse_shape.offset(I, J, K)];
    const Real c10 = coarse[coarse_shape.offset(Ip, J, K)];
    const Real c01 = coarse[coarse_shape.offset(I, Jp, K)];
    const Real c11 = coarse[coarse_shape.offset(Ip, Jp, K)];
    Real val = (Real(1) - xi) * (Real(1) - eta) * c00 + xi * (Real(1) - eta) * c10 +
               (Real(1) - xi) * eta * c01 + xi * eta * c11;
    if (enforce_positive && val < Real(0)) val = Real(0);
    fine[fine_shape.offset(i, j, k)] = val;
  }
}

inline unsigned int blocks_for(std::size_t n, unsigned int threads) {
  const std::size_t need = (n + threads - 1) / threads;
  const std::size_t cap = 16384;
  return static_cast<unsigned int>(need < cap ? need : cap);
}

}  // namespace

Real bilinear_interp(const Real* f, const FieldShape& shape, const KernelGeometry& geom, Int k,
                     Real x, Real y) {
  VIBE_CHECK(f != nullptr);
  // 单点：分配到设备执行（保证与批量版本逐位一致）
  Real* dx = nullptr;
  Real* dy = nullptr;
  Real* dout = nullptr;
  VIBE_GPU_CHECK(cudaMalloc(&dx, sizeof(Real)));
  VIBE_GPU_CHECK(cudaMalloc(&dy, sizeof(Real)));
  VIBE_GPU_CHECK(cudaMalloc(&dout, sizeof(Real)));
  VIBE_GPU_CHECK(cudaMemcpy(dx, &x, sizeof(Real), cudaMemcpyHostToDevice));
  VIBE_GPU_CHECK(cudaMemcpy(dy, &y, sizeof(Real), cudaMemcpyHostToDevice));
  bilinear_batch_dev<<<1, 1, 0, nullptr>>>(f, shape, geom, k, dx, dy, dout, 1);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
  Real result = Real(0);
  VIBE_GPU_CHECK(cudaMemcpy(&result, dout, sizeof(Real), cudaMemcpyDeviceToHost));
  VIBE_GPU_CHECK(cudaFree(dx));
  VIBE_GPU_CHECK(cudaFree(dy));
  VIBE_GPU_CHECK(cudaFree(dout));
  return result;
}

void bilinear_interp_batch(const Real* f, const FieldShape& shape, const KernelGeometry& geom,
                           Int k, const Real* x, const Real* y, Real* out, std::size_t n) {
  VIBE_CHECK(f != nullptr && x != nullptr && y != nullptr && out != nullptr);
  if (n == 0) return;
  constexpr unsigned int kThreads = 256;
  bilinear_batch_dev<<<blocks_for(n, kThreads), kThreads, 0, nullptr>>>(f, shape, geom, k, x, y,
                                                                        out, n);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

Real trilinear_interp(const Real* f, const FieldShape& shape, const KernelGeometry& geom, Real x,
                      Real y, Real z) {
  VIBE_CHECK(f != nullptr);
  Real* dxyz = nullptr;
  Real* dout = nullptr;
  VIBE_GPU_CHECK(cudaMalloc(&dxyz, 3 * sizeof(Real)));
  VIBE_GPU_CHECK(cudaMalloc(&dout, sizeof(Real)));
  const Real host[3] = {x, y, z};
  VIBE_GPU_CHECK(cudaMemcpy(dxyz, host, 3 * sizeof(Real), cudaMemcpyHostToDevice));
  trilinear_batch_dev<<<1, 1, 0, nullptr>>>(f, shape, geom, dxyz, dxyz + 1, dxyz + 2, dout, 1);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
  Real result = Real(0);
  VIBE_GPU_CHECK(cudaMemcpy(&result, dout, sizeof(Real), cudaMemcpyDeviceToHost));
  VIBE_GPU_CHECK(cudaFree(dxyz));
  VIBE_GPU_CHECK(cudaFree(dout));
  return result;
}

void trilinear_interp_batch(const Real* f, const FieldShape& shape, const KernelGeometry& geom,
                            const Real* x, const Real* y, const Real* z, Real* out, std::size_t n) {
  VIBE_CHECK(f != nullptr && x != nullptr && y != nullptr && z != nullptr && out != nullptr);
  if (n == 0) return;
  constexpr unsigned int kThreads = 256;
  trilinear_batch_dev<<<blocks_for(n, kThreads), kThreads, 0, nullptr>>>(f, shape, geom, x, y, z,
                                                                           out, n);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

// ===========================================================================
// 嵌套：限制与延拓的主机侧包装
// ===========================================================================
void restrict_2to1(const Real* fine, Real* coarse, const FieldShape& fine_shape,
                   const FieldShape& coarse_shape, Int ratio) {
  VIBE_CHECK(fine != nullptr && coarse != nullptr);
  VIBE_CHECK_MSG(ratio >= 1, "restrict_2to1: ratio 必须 >= 1");
  const std::size_t n = static_cast<std::size_t>(coarse_shape.nsx - 2 * coarse_shape.halo) *
                        static_cast<std::size_t>(coarse_shape.nsy - 2 * coarse_shape.halo) *
                        static_cast<std::size_t>(coarse_shape.nsz - 2 * coarse_shape.halo);
  if (n == 0) return;
  constexpr unsigned int kThreads = 256;
  restrict_dev<<<blocks_for(n, kThreads), kThreads, 0, nullptr>>>(fine, coarse, fine_shape,
                                                                 coarse_shape, ratio);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

void prolong_1to2(const Real* coarse, Real* fine, const FieldShape& coarse_shape,
                  const FieldShape& fine_shape, Int ratio, bool enforce_positive) {
  VIBE_CHECK(coarse != nullptr && fine != nullptr);
  VIBE_CHECK_MSG(ratio >= 1, "prolong_1to2: ratio 必须 >= 1");
  const std::size_t n = static_cast<std::size_t>(fine_shape.nsx - 2 * fine_shape.halo) *
                        static_cast<std::size_t>(fine_shape.nsy - 2 * fine_shape.halo) *
                        static_cast<std::size_t>(fine_shape.nsz - 2 * fine_shape.halo);
  if (n == 0) return;
  constexpr unsigned int kThreads = 256;
  prolong_dev<<<blocks_for(n, kThreads), kThreads, 0, nullptr>>>(coarse, fine, coarse_shape,
                                                                fine_shape, ratio,
                                                                enforce_positive);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

}  // namespace vibe::gpu

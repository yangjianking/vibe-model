/// @file kernels_diffusion.cu
/// @brief 扩散（Laplacian / biharmonic）内核的 CUDA 实现。
///
/// 数学与离散化见 \c include/vibe/gpu/kernels.hpp 的 \c diffusion 声明：
/// \f[
///   \nabla^2 q = \frac{q_{i+1}-2q_i+q_{i-1}}{\Delta x_i^2} + (y,z)
///   \qquad
///   \nabla^4 q = \nabla^2(\nabla^2 q)
/// \f]
///
/// GPU 实现决策
/// ------------
///   1. biharmonic 分成**两次内核启动**（中间量写 \c work），而不是一个内核里
///      做两次模板：后者需要把 13 点模板完全展开（约 36 次全局读/点），
///      寄存器压力大；两次 7 点模板总共 12 读 4 写更省（[G10] 第 5 章）；
///   2. 共享内存：不显式使用。扩散是"读多写少"且模板对称，L1 命中率很高；
///      若 nz 大、nx*ny 小，可用 2D tile 减少重复读；
///   3. 访存：7 点模板中 3 个方向各 2 次邻居读，其中 x 方向被同一 warp 的
///      相邻线程复用 -> 实际每点约 4 次全局事务（[G10] 第 5.3.2 节）；
///   4. 精度：biharmonic 误差按 \f$ 1/\Delta x^4 \f$ 放大，因此上层必须
///      保证 compute >= FP32（kernels.hpp 与 CPU 实现同款检查）。
///
/// 复杂度：Laplacian \f$ O(N) \f$，约 8 flops/点；biharmonic 两次调用
/// \f$ O(2N) \f$，约 16 flops/点 + 1 次中间往返。
///
/// 文献：[B2] 第 2、3 章；[D12] 尺度选择阻尼；[G6] 混合精度误差预算；
///       [G10] 第 5 章。

#include <cuda_runtime.h>

#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/kernels.hpp"

#define VIBE_KERNELS_BACKEND_CUDA 1

#if !VIBE_BACKEND_CUDA
#  error "kernels_diffusion.cu 只能在 CUDA 后端下编译"
#endif

namespace vibe::gpu {
namespace {

__global__ void laplacian_kernel(const Real* __restrict__ src, Real* __restrict__ dst,
                                 FieldShape shape, KernelGeometry geom) {
  const unsigned int bx = blockDim.x;
  const unsigned int gx = static_cast<unsigned int>(geom.nx);
  const unsigned int gy = static_cast<unsigned int>(geom.ny);
  const unsigned int total = gx * gy * static_cast<unsigned int>(geom.nz);
  const unsigned int stride = bx * gridDim.x;
  for (unsigned int idx = blockIdx.x * bx + threadIdx.x; idx < total; idx += stride) {
    const Int i = static_cast<Int>(idx % gx);
    const Int j = static_cast<Int>((idx / gx) % gy);
    const Int k = static_cast<Int>(idx / (gx * gy));
    const std::size_t c = shape.offset(i, j, k);
    const Real inv2x = geom.inv_dx_at(i) * geom.inv_dx_at(i);
    const Real inv2y = geom.inv_dy_at(j) * geom.inv_dy_at(j);
    const Real inv2z = Real(1) / (geom.dz * geom.dz);
    dst[c] = (src[shape.offset(i + 1, j, k)] - Real(2) * src[c] + src[shape.offset(i - 1, j, k)]) * inv2x +
             (src[shape.offset(i, j + 1, k)] - Real(2) * src[c] + src[shape.offset(i, j - 1, k)]) * inv2y +
             (src[shape.offset(i, j, k + 1)] - Real(2) * src[c] + src[shape.offset(i, j, k - 1)]) * inv2z;
  }
}

/// 原地缩放内核：dst = factor * dst（把 \f$ \kappa \f$ 的乘法从模板内核里提出来）
__global__ void scale_kernel(Real* __restrict__ dst, std::size_t n, Real factor) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    dst[i] *= factor;
  }
}

inline LaunchConfig cfg_for(const KernelGeometry& g) {
  const std::size_t total = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
  const std::size_t cap = static_cast<std::size_t>(16384) * 256;
  return make_1d_strided(total < cap ? total : cap, 256, 1);
}

inline void run_laplacian(const Real* src, Real* dst, const FieldShape& shape,
                          const KernelGeometry& geom) {
  const LaunchConfig cfg = cfg_for(geom);
  laplacian_kernel<<<static_cast<unsigned int>(cfg.block_count()),
                     static_cast<unsigned int>(cfg.block.x), 0, nullptr>>>(src, dst, shape, geom);
  VIBE_GPU_CHECK(cudaGetLastError());
}

}  // namespace

void diffusion(const Real* q, Real* out, Real* work, const FieldShape& shape,
               const KernelGeometry& geom, Real kappa, DiffusionKind kind) {
  VIBE_CHECK(q != nullptr && out != nullptr);
  if (kind == DiffusionKind::Biharmonic) {
    VIBE_CHECK_MSG(work != nullptr, "biharmonic 扩散需要 work 中间缓冲");
    VIBE_CHECK_MSG(precision_rank(current_policy().compute) >= precision_rank(Precision::FP32),
                   "biharmonic 扩散要求 compute 精度 >= FP32");
    run_laplacian(q, work, shape, geom);    // 第一遍：nabla^2 q
    run_laplacian(work, out, shape, geom);  // 第二遍：nabla^4 q
    const std::size_t n = shape.size();
    const unsigned int blocks = static_cast<unsigned int>((n + 255) / 256);
    scale_kernel<<<blocks, 256, 0, nullptr>>>(out, n, -kappa);
    VIBE_GPU_CHECK(cudaGetLastError());
  } else {
    run_laplacian(q, out, shape, geom);
    const std::size_t n = shape.size();
    const unsigned int blocks = static_cast<unsigned int>((n + 255) / 256);
    scale_kernel<<<blocks, 256, 0, nullptr>>>(out, n, kappa);
    VIBE_GPU_CHECK(cudaGetLastError());
  }
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

}  // namespace vibe::gpu

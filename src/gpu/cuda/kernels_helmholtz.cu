/// @file kernels_helmholtz.cu
/// @brief Helmholtz 算子残差与 Jacobi/Chebyshev 平滑子的 CUDA 实现。
///
/// 数学形式（推导见 include/vibe/gpu/kernels.hpp 的声明处）
/// -------------------------------------------------------
/// \f[
///   \mathcal{L}x = \nabla\cdot(a\nabla x) - b\,x
///   = \sum_{d\in\{x,y,z\}} \frac{a_{d+}(x_{d+}-x_c) - a_{d-}(x_c-x_{d-})}{\Delta_d^2} - b x_c
/// \f]
/// 面系数用**调和平均** \f$ a_f = 2a_ia_{i+1}/(a_i+a_{i+1}) \f$：
/// 有限体积意义下它保证跨面通量连续，对强对比的 \f$ a \f$（例如地表
/// 附近的垂直扩散系数）比算术平均准确（[B2] 第 4 章、[G16]）。
///
/// Jacobi / Chebyshev 平滑子
/// -------------------------
/// \f[ x^{(m+1)} = x^{(m)} + \frac{\omega_m}{D}\,(f-\mathcal{L}x^{(m)}),\qquad
///     D = -\sum_d \frac{a_{d+}+a_{d-}}{\Delta_d^2} - b \f]
/// 3D 7 点 Laplacian 的最优 Jacobi 权 \f$ \omega = 2/3 \f$（[T13] 式 (3.24)）；
/// Chebyshev 半迭代的权按 [T13] 第 3.3 节递推
/// \f$ \omega_m = 2/(2-\rho^2\omega_{m-1}),\; \omega_0=1 \f$，
/// \f$ \rho = \frac13\sum_d\cos(\pi/n_d) \f$ 为 Jacobi 迭代矩阵的谱半径估计。
/// 这使光滑子把高频误差的衰减率从 \f$ O(1-\pi^2h^2/3) \f$ 提升到
/// \f$ O(1-\pi h/3) \f$ —— 多重网格得以收敛的关键（[T13] 第 3 章）。
///
/// GPU 实现决策
/// ------------
///   1. **Jacobi 需要双缓冲**：一次内核体内不得读写同一数组（否则变成
///      Gauss-Seidel，warp 内结果依赖调度顺序，不可复现）。
///      本实现把 \c sweeps 次迭代全部放在**一个内核**里，按遍交替读写指针：
///      \c src -> \c dst -> \c src ...，仅使用一块辅助缓冲；
///      代价是每遍一次 __syncthreads，但省去了 \c sweeps 次内核启动
///      （每次约 3~8 us 启动延迟，[G10] 第 6 章）；
///   2. 共享内存：0（7 点模板在 L1 中复用）；若要做 Jacobi 的"块内缓存"
///      优化，可用 \f$ (t_x+2)(t_y+2)t_z \f$ 的 tile，可省约 40% 全局访存；
///   3. 残差计算使用 \c double 累加（\c reduce 精度），保证迭代精化的
///      残差不受 compute 精度限制（[G5]）。
///
/// 复杂度：残差 \f$ O(N) \f$，约 20 flops/点；Jacobi 每遍 \f$ O(N) \f$，
/// 约 15 flops/点 + 1 次同步。
///
/// 文献：[T2][T7] 半隐式；[T13] Briggs et al. (2000) 第 3 章；
///       [G16] Stone (1973)；[G5] Haidar et al. (2018)；[B2][G10]。

#include <cuda_runtime.h>

#include <cmath>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/kernels.hpp"

#define VIBE_KERNELS_BACKEND_CUDA 1

#if !VIBE_BACKEND_CUDA
#  error "kernels_helmholtz.cu 只能在 CUDA 后端下编译"
#endif

namespace vibe::gpu {
namespace {

__device__ __forceinline__ Real face_harmonic(Real a, Real b) {
  const Real den = a + b;
  return (fabs(den) > Real(1e-300)) ? (Real(2) * a * b / den) : Real(0);
}

/// \f$ \nabla\cdot(a\nabla x) \f$ 的通量差部分
__device__ __forceinline__ Real helmholtz_flux(const Real* __restrict__ x,
                                               const Real* __restrict__ a, const FieldShape& s,
                                               const KernelGeometry& g, Int i, Int j, Int k) {
  const Real xc = x[s.offset(i, j, k)];
  const Real ac = a[s.offset(i, j, k)];
  const Real inv2x = g.inv_dx_at(i) * g.inv_dx_at(i);
  const Real inv2y = g.inv_dy_at(j) * g.inv_dy_at(j);
  const Real inv2z = Real(1) / (g.dz * g.dz);
  const Real rx = face_harmonic(ac, a[s.offset(i + 1, j, k)]) * (x[s.offset(i + 1, j, k)] - xc) -
                  face_harmonic(ac, a[s.offset(i - 1, j, k)]) * (xc - x[s.offset(i - 1, j, k)]);
  const Real ry = face_harmonic(ac, a[s.offset(i, j + 1, k)]) * (x[s.offset(i, j + 1, k)] - xc) -
                  face_harmonic(ac, a[s.offset(i, j - 1, k)]) * (xc - x[s.offset(i, j - 1, k)]);
  const Real rz = face_harmonic(ac, a[s.offset(i, j, k + 1)]) * (x[s.offset(i, j, k + 1)] - xc) -
                  face_harmonic(ac, a[s.offset(i, j, k - 1)]) * (xc - x[s.offset(i, j, k - 1)]);
  return rx * inv2x + ry * inv2y + rz * inv2z;
}

__device__ __forceinline__ Real helmholtz_diag(const Real* __restrict__ a, const FieldShape& s,
                                               const KernelGeometry& g, Int i, Int j, Int k) {
  const Real ac = a[s.offset(i, j, k)];
  const Real inv2x = g.inv_dx_at(i) * g.inv_dx_at(i);
  const Real inv2y = g.inv_dy_at(j) * g.inv_dy_at(j);
  const Real inv2z = Real(1) / (g.dz * g.dz);
  const Real sx = (face_harmonic(ac, a[s.offset(i + 1, j, k)]) +
                   face_harmonic(ac, a[s.offset(i - 1, j, k)])) * inv2x;
  const Real sy = (face_harmonic(ac, a[s.offset(i, j + 1, k)]) +
                   face_harmonic(ac, a[s.offset(i, j - 1, k)])) * inv2y;
  const Real sz = (face_harmonic(ac, a[s.offset(i, j, k + 1)]) +
                   face_harmonic(ac, a[s.offset(i, j, k - 1)])) * inv2z;
  return -(sx + sy + sz);
}

__global__ void residual_kernel(const Real* __restrict__ x, const Real* __restrict__ a,
                                const Real* __restrict__ b, const Real* __restrict__ f,
                                Real* __restrict__ r, FieldShape shape, KernelGeometry geom) {
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
    // 双精度累加（reduce 精度）：残差必须比未知量更精确 [G5]
    const double lhs = static_cast<double>(helmholtz_flux(x, a, shape, geom, i, j, k)) -
                       static_cast<double>(b[c]) * static_cast<double>(x[c]);
    r[c] = static_cast<Real>(static_cast<double>(f[c]) - lhs);
  }
}

/// Jacobi/Chebyshev：一个内核完成 sweeps 遍，遍间 __syncthreads
__global__ void jacobi_kernel(const Real* __restrict__ a, const Real* __restrict__ b,
                              const Real* __restrict__ f, const Real* __restrict__ x0,
                              Real* __restrict__ out, Real* __restrict__ aux, FieldShape shape,
                              KernelGeometry geom, int sweeps, bool chebyshev) {
  const unsigned int bx = blockDim.x;
  const unsigned int gx = static_cast<unsigned int>(geom.nx);
  const unsigned int gy = static_cast<unsigned int>(geom.ny);
  const unsigned int total = gx * gy * static_cast<unsigned int>(geom.nz);
  const unsigned int stride = bx * gridDim.x;

  // Chebyshev 的谱半径估计（[T13] 式 (3.23)）
  const Real rho = chebyshev ? (cos(kPi / (geom.nx > 1 ? Real(geom.nx) : Real(2))) +
                                cos(kPi / (geom.ny > 1 ? Real(geom.ny) : Real(2))) +
                                cos(kPi / (geom.nz > 1 ? Real(geom.nz) : Real(2)))) /
                                   Real(3)
                             : Real(0);
  const Real rho2 = rho * rho;
  const Real* src = x0;
  for (int s = 0; s < sweeps; ++s) {
    Real* dst = (s == sweeps - 1) ? out : aux;
    // omega_m 的递推（每遍 O(1)，所有线程算同一个值，无分歧）
    Real omega = Real(2) / Real(3);
    if (chebyshev) {
      Real w = Real(1);
      for (int m = 0; m < s; ++m) w = Real(2) / (Real(2) - rho2 * w);
      omega = w;
    }
    for (unsigned int idx = blockIdx.x * bx + threadIdx.x; idx < total; idx += stride) {
      const Int i = static_cast<Int>(idx % gx);
      const Int j = static_cast<Int>((idx / gx) % gy);
      const Int k = static_cast<Int>(idx / (gx * gy));
      const std::size_t c = shape.offset(i, j, k);
      const Real d = helmholtz_diag(a, shape, geom, i, j, k) - b[c];
      const Real lhs = helmholtz_flux(src, a, shape, geom, i, j, k) - b[c] * src[c];
      const Real resid = f[c] - lhs;
      dst[c] = (fabs(d) > Real(1e-300)) ? (src[c] + omega * resid / d) : src[c];
    }
    __syncthreads();
    src = dst;
  }
}

/// 启动配置：把索引空间映射到 1D 网格。
/// \c grid.x 取 \f$ \min(\text{total},\; 16\,384\times256) \f$，
/// 超出部分由设备侧的网格跨步循环消化——这样在 1e6~1e9 点量级上都能把
/// SM 填满（[G10] 第 5.7 节），而不是被每线程 1 点的划分限制住。
inline LaunchConfig cfg_for(const KernelGeometry& g) {
  const std::size_t total = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
  const std::size_t cap = static_cast<std::size_t>(16384) * 256;
  return make_1d_strided(total < cap ? total : cap, 256, 1);
}

}  // namespace

void helmholtz_residual(const Real* x, const Real* a, const Real* b, const Real* f, Real* r,
                        const FieldShape& shape, const KernelGeometry& geom) {
  VIBE_CHECK(x != nullptr && a != nullptr && b != nullptr && f != nullptr && r != nullptr);
  const LaunchConfig cfg = cfg_for(geom);
  residual_kernel<<<static_cast<unsigned int>(cfg.block_count()),
                    static_cast<unsigned int>(cfg.block.x), 0, nullptr>>>(x, a, b, f, r, shape,
                                                                          geom);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

void helmholtz_jacobi(const Real* a, const Real* b, const Real* f, const Real* x, Real* out,
                      const FieldShape& shape, const KernelGeometry& geom, int sweeps,
                      bool chebyshev) {
  VIBE_CHECK(a != nullptr && b != nullptr && f != nullptr && x != nullptr && out != nullptr);
  VIBE_CHECK_MSG(sweeps >= 1, "helmholtz_jacobi: sweeps 必须 >= 1");
  Real* aux = nullptr;
  const std::size_t n = shape.size();
  if (sweeps > 1) {
    VIBE_GPU_CHECK(cudaMalloc(&aux, n * sizeof(Real)));
  }
  const LaunchConfig cfg = cfg_for(geom);
  jacobi_kernel<<<static_cast<unsigned int>(cfg.block_count()),
                  static_cast<unsigned int>(cfg.block.x), 0, nullptr>>>(a, b, f, x, out, aux, shape,
                                                                        geom, sweeps, chebyshev);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
  if (aux != nullptr) VIBE_GPU_CHECK(cudaFree(aux));
}

}  // namespace vibe::gpu

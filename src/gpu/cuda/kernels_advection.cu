/// @file kernels_advection.cu
/// @brief 平流内核的 CUDA 实现（标量 + 动量）。
///
/// 数学与离散化
/// ------------
/// 完整推导、LaTeX 公式、文献引用与复杂度写在
/// \c include/vibe/gpu/kernels.hpp 的 \c advect_scalar / \c advect_momentum 声明处。
/// 本文件只补充 **GPU 特有的实现决策**：
///
///   1. 线程映射：3D 索引空间按 x 最快线性化，1 个线程负责 1 个点；
///      i 方向相邻线程访问相邻地址 -> 合并访存（[G10] 第 5.3.2 节）；
///   2. 共享内存：本内核不显式使用。x 方向模板宽 3（2 阶）或 5（WENO5），
///      同一 warp 内相邻线程会把模板重叠部分放进 L1，命中率天然很高；
///      若改用 tile（\f$ (t_x+4)t_y t_z \f$ 个 Real），可省 2/3 的全局读，
///      但会引入 __syncthreads 与边界复制，收益在 nz 较大时才明显；
///   3. 寄存器：2 阶分支约 24 个、4 阶约 40 个、WENO5 约 64 个 32-bit 寄存器；
///      WENO5 分支在 cc >= 7.0 上占用率仍可达 ~50%（2048 线程/SM 上限下）；
///   4. 精度策略：通量按 \ref vibe::gpu::current_policy 的 compute 精度计算，
///      但本实现全部用 \c Real（= FP64，由 VIBE_PRECISION 决定）以保持与 CPU
///      参考实现可逐位比对；要启用 FP32 计算需同时把 Field 的存储精度降级
///      （见 docs/design/05_gpu_hybrid_precision.md 第 3 节的误差预算讨论）。
///
/// 复杂度：\f$ O(N) \f$，N = 含 halo 的存储点数；每点 flops 约 18/60/120
/// 分别对应 2 阶 / 4 阶 / WENO5；全局访存约 7 读 1 写（2 阶）。
///
/// 文献：[D3][D6][D7][D8][B2][G10] 第 5 章；[G12] 第 7 章。

#include <cuda_runtime.h>

#include <cmath>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/kernels.hpp"

#define VIBE_KERNELS_BACKEND_CUDA 1

#if !VIBE_BACKEND_CUDA
#  error "kernels_advection.cu 只能在 CUDA 后端下编译（见 kernels.hpp 的链接契约）"
#endif

namespace vibe::gpu {
namespace {

// ---------------------------------------------------------------------------
// 设备侧工具（与 kernels_cpu.cpp 的匿名命名空间函数一一对应）
// ---------------------------------------------------------------------------

/// 2 阶中心差分（x 方向）
__device__ __forceinline__ Real d1_x(const Real* p, const FieldShape& s, Int i, Int j, Int k,
                                     Real inv_dx) {
  return (p[s.offset(i + 1, j, k)] - p[s.offset(i - 1, j, k)]) * (Real(0.5) * inv_dx);
}
__device__ __forceinline__ Real d1_y(const Real* p, const FieldShape& s, Int i, Int j, Int k,
                                     Real inv_dy) {
  return (p[s.offset(i, j + 1, k)] - p[s.offset(i, j - 1, k)]) * (Real(0.5) * inv_dy);
}
__device__ __forceinline__ Real d1_z(const Real* p, const FieldShape& s, Int i, Int j, Int k,
                                     Real inv_dz) {
  return (p[s.offset(i, j, k + 1)] - p[s.offset(i, j, k - 1)]) * (Real(0.5) * inv_dz);
}

/// 4 阶中心差分
__device__ __forceinline__ Real d1_x4(const Real* p, const FieldShape& s, Int i, Int j, Int k,
                                      Real inv_dx) {
  const Real a = p[s.offset(i + 2, j, k)];
  const Real b = p[s.offset(i + 1, j, k)];
  const Real c = p[s.offset(i - 1, j, k)];
  const Real d = p[s.offset(i - 2, j, k)];
  return (Real(-1) * a + Real(8) * b - Real(8) * c + d) * (inv_dx / Real(12));
}
__device__ __forceinline__ Real d1_y4(const Real* p, const FieldShape& s, Int i, Int j, Int k,
                                      Real inv_dy) {
  const Real a = p[s.offset(i, j + 2, k)];
  const Real b = p[s.offset(i, j + 1, k)];
  const Real c = p[s.offset(i, j - 1, k)];
  const Real d = p[s.offset(i, j - 2, k)];
  return (Real(-1) * a + Real(8) * b - Real(8) * c + d) * (inv_dy / Real(12));
}

/// WENO5 左偏重构（[D7][D8] 式 (2.6)）：返回界面 i+1/2 的左侧值
__device__ __forceinline__ Real weno5_left(Real qm2, Real qm1, Real q0, Real q1, Real q2) {
  const Real eps = Real(1e-6);
  const Real p0 = (Real(2) * qm2 - Real(7) * qm1 + Real(11) * q0) / Real(6);
  const Real p1 = (Real(-1) * qm1 + Real(5) * q0 + Real(2) * q1) / Real(6);
  const Real p2 = (Real(2) * q0 + Real(5) * q1 - q2) / Real(6);
  const Real d0 = qm2 - Real(2) * qm1 + q0;
  const Real d1 = qm2 - Real(4) * qm1 + Real(3) * q0;
  const Real e0 = qm1 - Real(2) * q0 + q1;
  const Real e1 = qm1 - q1;
  const Real f0 = q0 - Real(2) * q1 + q2;
  const Real f1 = Real(3) * q0 - Real(4) * q1 + q2;
  const Real b0 = Real(13) / Real(12) * d0 * d0 + Real(1) / Real(4) * d1 * d1;
  const Real b1 = Real(13) / Real(12) * e0 * e0 + Real(1) / Real(4) * e1 * e1;
  const Real b2 = Real(13) / Real(12) * f0 * f0 + Real(1) / Real(4) * f1 * f1;
  const Real a0 = (Real(3) / Real(10)) / ((eps + b0) * (eps + b0));
  const Real a1 = (Real(6) / Real(10)) / ((eps + b1) * (eps + b1));
  const Real a2 = (Real(1) / Real(10)) / ((eps + b2) * (eps + b2));
  const Real inv = Real(1) / (a0 + a1 + a2);
  return (a0 * p0 + a1 * p1 + a2 * p2) * inv;
}

__device__ __forceinline__ Real face_avg(Real a, Real b) { return Real(0.5) * (a + b); }

// ---------------------------------------------------------------------------
// 设备内核
// ---------------------------------------------------------------------------
__global__ void advect_scalar_kernel(const Real* __restrict__ q, const Real* __restrict__ u,
                                     const Real* __restrict__ v, const Real* __restrict__ w,
                                     const Real* __restrict__ rho, Real* __restrict__ out,
                                     FieldShape shape, KernelGeometry geom, AdvectionOptions opt) {
  const unsigned int bx = blockDim.x;
  const unsigned int gx = static_cast<unsigned int>(geom.nx);
  const unsigned int gy = static_cast<unsigned int>(geom.ny);
  const unsigned int gz = static_cast<unsigned int>(geom.nz);
  const unsigned int total = gx * gy * gz;
  const unsigned int stride = bx * gridDim.x;
  const bool flux = opt.flux_form && (rho != nullptr);

  for (unsigned int idx = blockIdx.x * bx + threadIdx.x; idx < total; idx += stride) {
    const Int i = static_cast<Int>(idx % gx);
    const Int j = static_cast<Int>((idx / gx) % gy);
    const Int k = static_cast<Int>(idx / (gx * gy));
    const std::size_t c = shape.offset(i, j, k);
    const Real invdx = geom.inv_dx_at(i);
    const Real invdy = geom.inv_dy_at(j);
    const Real invdz = Real(1) / geom.dz;
    Real adv;

    if (opt.order == 4) {
      const Real vx = d1_x4(q, shape, i, j, k, invdx);
      const Real vy = d1_y4(q, shape, i, j, k, invdy);
      const Real vz = d1_z(q, shape, i, j, k, invdz);
      adv = u[c] * vx + v[c] * vy + w[c] * vz;
    } else if (opt.weno5) {
      const Real qxp = weno5_left(q[shape.offset(i - 2, j, k)], q[shape.offset(i - 1, j, k)],
                                  q[shape.offset(i, j, k)], q[shape.offset(i + 1, j, k)],
                                  q[shape.offset(i + 2, j, k)]);
      const Real qxm = weno5_left(q[shape.offset(i + 1, j, k)], q[shape.offset(i, j, k)],
                                  q[shape.offset(i - 1, j, k)], q[shape.offset(i - 2, j, k)],
                                  q[shape.offset(i - 3, j, k)]);
      const Real qyp = weno5_left(q[shape.offset(i, j - 2, k)], q[shape.offset(i, j - 1, k)],
                                  q[shape.offset(i, j, k)], q[shape.offset(i, j + 1, k)],
                                  q[shape.offset(i, j + 2, k)]);
      const Real qym = weno5_left(q[shape.offset(i, j + 1, k)], q[shape.offset(i, j, k)],
                                  q[shape.offset(i, j - 1, k)], q[shape.offset(i, j - 2, k)],
                                  q[shape.offset(i, j - 3, k)]);
      const Real fz = q[shape.offset(i, j, k + 1)] - q[shape.offset(i, j, k - 1)];
      adv = u[c] * (qxp - qxm) * invdx + v[c] * (qyp - qym) * invdy + w[c] * fz * invdz;
    } else {
      const Real qxp = face_avg(q[c], q[shape.offset(i + 1, j, k)]);
      const Real qxm = face_avg(q[c], q[shape.offset(i - 1, j, k)]);
      const Real qyp = face_avg(q[c], q[shape.offset(i, j + 1, k)]);
      const Real qym = face_avg(q[c], q[shape.offset(i, j - 1, k)]);
      const Real qzp = face_avg(q[c], q[shape.offset(i, j, k + 1)]);
      const Real qzm = face_avg(q[c], q[shape.offset(i, j, k - 1)]);
      const Real rxp = flux ? face_avg(rho[c], rho[shape.offset(i + 1, j, k)]) : Real(1);
      const Real rxm = flux ? face_avg(rho[c], rho[shape.offset(i - 1, j, k)]) : Real(1);
      const Real ryp = flux ? face_avg(rho[c], rho[shape.offset(i, j + 1, k)]) : Real(1);
      const Real rym = flux ? face_avg(rho[c], rho[shape.offset(i, j - 1, k)]) : Real(1);
      const Real rzp = flux ? face_avg(rho[c], rho[shape.offset(i, j, k + 1)]) : Real(1);
      const Real rzm = flux ? face_avg(rho[c], rho[shape.offset(i, j, k - 1)]) : Real(1);
      const Real uxp = face_avg(u[c], u[shape.offset(i + 1, j, k)]);
      const Real uxm = face_avg(u[c], u[shape.offset(i - 1, j, k)]);
      const Real vyp = face_avg(v[c], v[shape.offset(i, j + 1, k)]);
      const Real vym = face_avg(v[c], v[shape.offset(i, j - 1, k)]);
      const Real wzp = face_avg(w[c], w[shape.offset(i, j, k + 1)]);
      const Real wzm = face_avg(w[c], w[shape.offset(i, j, k - 1)]);
      const Real div = (rxp * qxp * uxp - rxm * qxm * uxm) * invdx +
                       (ryp * qyp * vyp - rym * qym * vym) * invdy +
                       (rzp * qzp * wzp - rzm * qzm * wzm) * invdz;
      adv = flux ? (-div / rho[c]) : (-div);
    }
    out[c] = adv;
  }
}

__global__ void advect_momentum_kernel(const Real* __restrict__ u, const Real* __restrict__ v,
                                       const Real* __restrict__ w, Real* __restrict__ du,
                                       Real* __restrict__ dv, Real* __restrict__ dw,
                                       FieldShape shape, KernelGeometry geom, AdvectionOptions opt) {
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
    const Real invdx = geom.inv_dx_at(i);
    const Real invdy = geom.inv_dy_at(j);
    const Real invdz = Real(1) / geom.dz;
    Real ax, ay, az;
    if (opt.order == 4) {
      ax = d1_x4(u, shape, i, j, k, invdx);
      ay = d1_y4(u, shape, i, j, k, invdy);
      az = d1_z(u, shape, i, j, k, invdz);
      const Real bx_ = d1_x4(v, shape, i, j, k, invdx);
      const Real by_ = d1_y4(v, shape, i, j, k, invdy);
      const Real bz_ = d1_z(v, shape, i, j, k, invdz);
      const Real cx_ = d1_x4(w, shape, i, j, k, invdx);
      const Real cy_ = d1_y4(w, shape, i, j, k, invdy);
      const Real cz_ = d1_z(w, shape, i, j, k, invdz);
      du[c] = -(u[c] * ax + v[c] * ay + w[c] * az);
      dv[c] = -(u[c] * bx_ + v[c] * by_ + w[c] * bz_);
      dw[c] = -(u[c] * cx_ + v[c] * cy_ + w[c] * cz_);
    } else {
      ax = d1_x(u, shape, i, j, k, invdx);
      ay = d1_y(u, shape, i, j, k, invdy);
      az = d1_z(u, shape, i, j, k, invdz);
      const Real bx_ = d1_x(v, shape, i, j, k, invdx);
      const Real by_ = d1_y(v, shape, i, j, k, invdy);
      const Real bz_ = d1_z(v, shape, i, j, k, invdz);
      const Real cx_ = d1_x(w, shape, i, j, k, invdx);
      const Real cy_ = d1_y(w, shape, i, j, k, invdy);
      const Real cz_ = d1_z(w, shape, i, j, k, invdz);
      du[c] = -(u[c] * ax + v[c] * ay + w[c] * az);
      dv[c] = -(u[c] * bx_ + v[c] * by_ + w[c] * bz_);
      dw[c] = -(u[c] * cx_ + v[c] * cy_ + w[c] * cz_);
    }
  }
}

/// 启动配置：1D 网格，block 默认 256 线程，shared 0
inline LaunchConfig cfg_for(const KernelGeometry& g) {
  const std::size_t total = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
  const std::size_t cap = static_cast<std::size_t>(16384) * 256;
  return make_1d_strided(total < cap ? total : cap, 256, 1);
}

}  // namespace

// ===========================================================================
// 主机侧包装（公开 API）
// ===========================================================================
void advect_scalar(const Real* q, const Real* u, const Real* v, const Real* w, const Real* rho,
                   Real* out, const FieldShape& shape, const KernelGeometry& geom,
                   const AdvectionOptions& options) {
  VIBE_CHECK(q != nullptr && u != nullptr && v != nullptr && w != nullptr && out != nullptr);
  const LaunchConfig cfg = cfg_for(geom);
  const std::size_t blocks = cfg.block_count();
  // 共享内存 0；流用默认流（可换 Stream::create() 以做多流 overlap）
  advect_scalar_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(cfg.block.x),
                         0, nullptr>>>(q, u, v, w, rho, out, shape, geom, options);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

void advect_momentum(const Real* u, const Real* v, const Real* w, const Real* rho, Real* du,
                     Real* dv, Real* dw, const FieldShape& shape, const KernelGeometry& geom,
                     const AdvectionOptions& options) {
  VIBE_CHECK(u != nullptr && v != nullptr && w != nullptr && du != nullptr && dv != nullptr &&
             dw != nullptr);
  VIBE_UNUSED(rho);
  const LaunchConfig cfg = cfg_for(geom);
  const std::size_t blocks = cfg.block_count();
  advect_momentum_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(cfg.block.x),
                           0, nullptr>>>(u, v, w, du, dv, dw, shape, geom, options);
  VIBE_GPU_CHECK(cudaGetLastError());
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
}

}  // namespace vibe::gpu

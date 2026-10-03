/// @file kernels_cpu.cpp
/// @brief 数值内核的 CPU（OpenMP）实现——参考实现，也是无 GPU 环境的唯一实现。
///
/// 定位
/// ----
///   * 与 \c src/gpu/cuda/kernels_*.cu **逐符号对应**：函数名、参数、
///     语义完全一致，因此同一份测试用例可以同时验证两条路径；
///   * 用 \f$ O(\Delta x^2) \f$ / \f$ O(\Delta x^4) \f$ 模板直接展开，
///     不做任何向量化内建函数，保证与解析解的可逐位比对；
///   * 每个内核的数学推导、离散化、文献引用与复杂度写在
///     \c include/vibe/gpu/kernels.hpp 的声明处（单一出处），此处只补充
///     CPU 特有的实现说明（循环顺序、并行策略、数值保护）。
///
/// 编译契约
/// --------
/// 本文件定义的是 \c vibe::gpu 命名空间下与 CUDA 版**同名**的函数。
/// 一次链接只能包含一组，因此文件内用 \c #if !VIBE_HAVE_GPU 守卫；
/// 详见 include/vibe/gpu/kernels.hpp 的"链接契约"说明。
///
/// 并行策略
/// --------
///   * 逐点内核：外层 OpenMP 并行、内层按 i 连续遍历（i 最快 ⇒ 缓存友好，
///     与 GPU 的合并访存一致）；
///   * 逐列内核（三对角）：并行维度是系统号，列内串行（依赖链无法并行）；
///   * 归约：分块 + 每块 CompensatedSum + 块间确定性合并（与线程数无关）。
///
/// 文献：与 kernels.hpp 声明处相同，主要为 [D3][D6][T10][T13][N4][O8][G2][G3][G4][B11]。

#include "vibe/gpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/precision.hpp"

#define VIBE_KERNELS_BACKEND_CPU 1

#if VIBE_HAVE_GPU
#  error "kernels_cpu.cpp 是 CPU 后端实现；CUDA/HIP/SYCL 构建请编译 src/gpu/cuda/*.cu（见 kernels.hpp 的链接契约）"
#endif

#if defined(_OPENMP)
#  include <omp.h>
#endif

namespace vibe::gpu {

namespace {

// ---------------------------------------------------------------------------
// 内部工具
// ---------------------------------------------------------------------------

/// 把 CPU 上"逐点内核"的公共循环骨架收敛到一处：
/// 外层 k/j（可选并行），内层 i 连续。\c f(i,j,k) 由各内核提供。
template <class F>
inline void for_each_point(const KernelGeometry& g, F&& f) {
  const Int nx = g.nx, ny = g.ny, nz = g.nz;
#if defined(_OPENMP)
#  pragma omp parallel for collapse(2) schedule(static)
#endif
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 0; i < nx; ++i) f(i, j, k);
    }
  }
}

/// 三线性权重（用于限制/延拓的常数一致性检查）
inline Real w0(Real t) noexcept { return Real(1) - t; }
inline Real w1(Real t) noexcept { return t; }

/// 2 阶中心差分 \f$ (f_{i+1}-f_{i-1})/(2\Delta x_i) \f$
inline Real d1_x(const Real* p, const FieldShape& s, Int i, Int j, Int k, Real inv_dx) noexcept {
  return (p[s.offset(i + 1, j, k)] - p[s.offset(i - 1, j, k)]) * (Real(0.5) * inv_dx);
}
inline Real d1_y(const Real* p, const FieldShape& s, Int i, Int j, Int k, Real inv_dy) noexcept {
  return (p[s.offset(i, j + 1, k)] - p[s.offset(i, j - 1, k)]) * (Real(0.5) * inv_dy);
}
inline Real d1_z(const Real* p, const FieldShape& s, Int i, Int j, Int k, Real inv_dz) noexcept {
  return (p[s.offset(i, j, k + 1)] - p[s.offset(i, j, k - 1)]) * (Real(0.5) * inv_dz);
}

/// 4 阶中心差分：\f$ (-f_{i+2}+8f_{i+1}-8f_{i-1}+f_{i-2})/(12\Delta x) \f$
inline Real d1_x4(const Real* p, const FieldShape& s, Int i, Int j, Int k, Real inv_dx) noexcept {
  const Real a = p[s.offset(i + 2, j, k)];
  const Real b = p[s.offset(i + 1, j, k)];
  const Real c = p[s.offset(i - 1, j, k)];
  const Real d = p[s.offset(i - 2, j, k)];
  return (Real(-1) * a + Real(8) * b - Real(8) * c + Real(1) * d) * (inv_dx / Real(12));
}
inline Real d1_y4(const Real* p, const FieldShape& s, Int i, Int j, Int k, Real inv_dy) noexcept {
  const Real a = p[s.offset(i, j + 2, k)];
  const Real b = p[s.offset(i, j + 1, k)];
  const Real c = p[s.offset(i, j - 1, k)];
  const Real d = p[s.offset(i, j - 2, k)];
  return (Real(-1) * a + Real(8) * b - Real(8) * c + Real(1) * d) * (inv_dy / Real(12));
}

/// WENO5 的 JS 权重重建（[D7][D8] 式 (2.6)）
/// \f$ \alpha_k = d_k/(\epsilon+\beta_k)^2,\; \omega_k = \alpha_k/\sum\alpha \f$
/// @return 单元界面 i+1/2 处的左偏重构值
inline Real weno5_left(Real qm2, Real qm1, Real q0, Real q1, Real q2) noexcept {
  constexpr Real eps = Real(1e-6);
  // 三个候选 3 阶多项式（界面 i+1/2 处的左偏通量）
  const Real p0 = (Real(2) * qm2 - Real(7) * qm1 + Real(11) * q0) / Real(6);
  const Real p1 = (Real(-1) * qm1 + Real(5) * q0 + Real(2) * q1) / Real(6);
  const Real p2 = (Real(2) * q0 + Real(5) * q1 - Real(1) * q2) / Real(6);
  // 光滑因子（[D8] 式 (2.6)）
  const Real b0 = Real(13) / Real(12) * (qm2 - Real(2) * qm1 + q0) * (qm2 - Real(2) * qm1 + q0) +
                   Real(1) / Real(4) * (qm2 - Real(4) * qm1 + Real(3) * q0) * (qm2 - Real(4) * qm1 + Real(3) * q0);
  const Real b1 = Real(13) / Real(12) * (qm1 - Real(2) * q0 + q1) * (qm1 - Real(2) * q0 + q1) +
                   Real(1) / Real(4) * (qm1 - q1) * (qm1 - q1);
  const Real b2 = Real(13) / Real(12) * (q0 - Real(2) * q1 + q2) * (q0 - Real(2) * q1 + q2) +
                   Real(1) / Real(4) * (Real(3) * q0 - Real(4) * q1 + q2) * (Real(3) * q0 - Real(4) * q1 + q2);
  // 理想权 d = (3/10, 3/5, 1/10)
  const Real a0 = (Real(3) / Real(10)) / ((eps + b0) * (eps + b0));
  const Real a1 = (Real(6) / Real(10)) / ((eps + b1) * (eps + b1));
  const Real a2 = (Real(1) / Real(10)) / ((eps + b2) * (eps + b2));
  const Real inv = Real(1) / (a0 + a1 + a2);
  return (a0 * p0 + a1 * p1 + a2 * p2) * inv;
}

/// 由相邻两个单元中心值构造面通量的"标量值"（2 阶中心插值）
inline Real face_avg(Real a, Real b) noexcept { return Real(0.5) * (a + b); }

/// 调和平均面系数：对变系数扩散保持通量连续（[B2] 第 4 章）
/// \f$ a_{f} = \dfrac{2 a_i a_{i+1}}{a_i + a_{i+1}} \f$
inline Real face_harmonic(Real a, Real b) noexcept {
  const Real den = a + b;
  if (std::abs(den) < std::numeric_limits<Real>::min()) return Real(0);
  return Real(2) * a * b / den;
}

/// 7 点变系数算子的"通量差"部分（不含 -b x 项），用于残差与 Jacobi。
/// 使用调和平均面系数（与 CUDA 实现严格一致）。
inline Real helmholtz_lhs(const Real* x, const Real* a, const FieldShape& s, const KernelGeometry& g,
                          Int i, Int j, Int k) {
  const Real xc = x[s.offset(i, j, k)];
  const Real ac = a[s.offset(i, j, k)];
  const Real inv2x = g.inv_dx_at(i) * g.inv_dx_at(i);
  const Real inv2y = g.inv_dy_at(j) * g.inv_dy_at(j);
  const Real inv2z = Real(1) / (g.dz * g.dz);
  const Real axp = face_harmonic(ac, a[s.offset(i + 1, j, k)]);
  const Real axm = face_harmonic(ac, a[s.offset(i - 1, j, k)]);
  const Real ayp = face_harmonic(ac, a[s.offset(i, j + 1, k)]);
  const Real aym = face_harmonic(ac, a[s.offset(i, j - 1, k)]);
  const Real azp = face_harmonic(ac, a[s.offset(i, j, k + 1)]);
  const Real azm = face_harmonic(ac, a[s.offset(i, j, k - 1)]);
  const Real rx = axp * (x[s.offset(i + 1, j, k)] - xc) - axm * (xc - x[s.offset(i - 1, j, k)]);
  const Real ry = ayp * (x[s.offset(i, j + 1, k)] - xc) - aym * (xc - x[s.offset(i, j - 1, k)]);
  const Real rz = azp * (x[s.offset(i, j, k + 1)] - xc) - azm * (xc - x[s.offset(i, j, k - 1)]);
  return rx * inv2x + ry * inv2y + rz * inv2z;
}

/// 算子的对角元（用于 Jacobi 权 \f$ \omega/D \f$）
inline Real helmholtz_diag(const Real* a, const FieldShape& s, const KernelGeometry& g, Int i, Int j,
                           Int k) {
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
  return (sx + sy + sz) * Real(-1);  // 对角为负（L = sum a d2 - b 的符号约定）
}

/// 双线性权重与单元定位（供 bilinear/prolong 复用）
struct CellLoc {
  Int i = 0;
  Real xi = Real(0);
};

/// 物理坐标 x -> 单元索引 + 局部坐标（变分辨率用逐列累加）
inline CellLoc locate_x(const KernelGeometry& g, Real x) {
  CellLoc loc;
  Real origin = Real(0);
  if (g.dx_cell == nullptr) {
    const Real fi = (x - Real(0)) / g.dx;
    const Real fl = std::floor(fi);
    loc.i = static_cast<Int>(std::max(Real(0), std::min(fl, static_cast<Real>(g.nx - 2))));
    loc.xi = std::max(Real(0), std::min(fi - static_cast<Real>(loc.i), Real(1)));
    return loc;
  }
  Int i = 0;
  while (i < g.nx - 2 && origin + g.dx_cell[i] <= x) {
    origin += g.dx_cell[i];
    ++i;
  }
  loc.i = i;
  const Real w = g.dx_cell[i];
  loc.xi = w > Real(0) ? std::max(Real(0), std::min((x - origin) / w, Real(1))) : Real(0);
  return loc;
}

inline CellLoc locate_y(const KernelGeometry& g, Real y) {
  CellLoc loc;
  Real origin = Real(0);
  if (g.dy_cell == nullptr) {
    const Real fj = y / g.dy;
    const Real fl = std::floor(fj);
    loc.i = static_cast<Int>(std::max(Real(0), std::min(fl, static_cast<Real>(g.ny - 2))));
    loc.xi = std::max(Real(0), std::min(fj - static_cast<Real>(loc.i), Real(1)));
    return loc;
  }
  Int j = 0;
  while (j < g.ny - 2 && origin + g.dy_cell[j] <= y) {
    origin += g.dy_cell[j];
    ++j;
  }
  loc.i = j;
  const Real w = g.dy_cell[j];
  loc.xi = w > Real(0) ? std::max(Real(0), std::min((y - origin) / w, Real(1))) : Real(0);
  return loc;
}

/// 归约分块大小：与线程数无关（确定性），65536 个元素/块
constexpr std::size_t kReduceChunk = 65536;

}  // namespace

// ===========================================================================
// KernelGeometry::from_grid
// ===========================================================================
KernelGeometry KernelGeometry::from_grid(const grid::Grid& g) {
  KernelGeometry k;
  k.nx = g.nx();
  k.ny = g.ny();
  k.nz = g.nz();
  k.halo = g.halo();
  k.dx = g.geom().dx;
  k.dy = g.geom().dy;
  // 垂直层厚取第 0 层的物理厚度（含地形追随雅可比前的名义值）
  k.dz = g.dzeta(0) * g.geom().z_top;
  const auto& dxc = g.geom().dx_cell;
  const auto& dyc = g.geom().dy_cell;
  k.dx_cell = dxc.empty() ? nullptr : dxc.data();
  k.dy_cell = dyc.empty() ? nullptr : dyc.data();
  return k;
}

// ===========================================================================
// 1. 平流
// ===========================================================================
void advect_scalar(const Real* q, const Real* u, const Real* v, const Real* w, const Real* rho,
                   Real* out, const FieldShape& shape, const KernelGeometry& geom,
                   const AdvectionOptions& options) {
  VIBE_CHECK(q != nullptr && u != nullptr && v != nullptr && w != nullptr && out != nullptr);
  const bool flux = options.flux_form && (rho != nullptr);

  for_each_point(geom, [&](Int i, Int j, Int k) {
    const std::size_t c = shape.offset(i, j, k);
    Real adv = Real(0);
    if (options.order == 4) {
      // 4 阶中心：advection form 与 flux form 在均匀网格上一致，此处取平流形式
      const Real vx = d1_x4(q, shape, i, j, k, geom.inv_dx_at(i));
      const Real vy = d1_y4(q, shape, i, j, k, geom.inv_dy_at(j));
      const Real vz = (q[shape.offset(i, j, k + 1)] - q[shape.offset(i, j, k - 1)]) *
                      (Real(0.5) / geom.dz);
      adv = u[c] * vx + v[c] * vy + w[c] * vz;
      if (flux) adv = (Real(1) / rho[c]) * (rho[c] * adv);  // 形式统一，便于对照
    } else if (options.weno5) {
      // WENO5：在 i+1/2 与 i-1/2 分别重构，得到通量差（[D7][D8]）
      const Real qm2 = q[shape.offset(i - 2, j, k)];
      const Real qm1 = q[shape.offset(i - 1, j, k)];
      const Real q0 = q[shape.offset(i, j, k)];
      const Real q1 = q[shape.offset(i + 1, j, k)];
      const Real q2 = q[shape.offset(i + 2, j, k)];
      // x 方向：左偏重构给出 q_{i+1/2}^-，反向模板给出 q_{i-1/2}^+。
      // 反向模板的第 5 个点应为 i-3；halo 只保证 2 层，故将其**钳制**到
      // i-2（对 WENO 的权函数而言该点权重最小，钳制引入的误差是 O(dx^3)，
      // 与 5 阶重构相容；需要严格 5 点模板时把 halo 加到 3）。
      const Real qm3 = q[shape.offset(std::max(i - 3, -shape.halo), j, k)];
      const Real qxp = weno5_left(qm2, qm1, q0, q1, q2);
      const Real qxm = weno5_left(q1, q0, qm1, qm2, qm3);
      const Real fx = qxp - qxm;  // 与 2 阶版本同为"界面值之差"

      // y 方向：同一函数逐维使用（[D7] 的 dimension-by-dimension 策略）
      const Real qyp = weno5_left(q[shape.offset(i, j - 2, k)], q[shape.offset(i, j - 1, k)], q0,
                                  q[shape.offset(i, j + 1, k)], q[shape.offset(i, j + 2, k)]);
      const Real qym = weno5_left(q[shape.offset(i, j + 1, k)], q0, q[shape.offset(i, j - 1, k)],
                                  q[shape.offset(i, j - 2, k)], q[shape.offset(i, j - 3, k)]);
      const Real fy = qyp - qym;

      // z 方向：用相邻两层界面通量的 2 阶中心（垂直方向 halo 更贵，且
      // 垂直 CFL 由声波子步控制，对 WENO 的需求较低）
      const Real fz = q[shape.offset(i, j, k + 1)] - q[shape.offset(i, j, k - 1)];

      adv = u[c] * fx * geom.inv_dx_at(i) + v[c] * fy * geom.inv_dy_at(j) + w[c] * fz * (Real(1) / geom.dz);
    } else {
      // 2 阶中心（守恒通量形式，[D6]）
      // x 方向：F_{i+1/2} = (rho q)|_{i+1/2} u|_{i+1/2}
      const Real qxp = face_avg(q[c], q[shape.offset(i + 1, j, k)]);
      const Real rxp = flux ? face_avg(rho[c], rho[shape.offset(i + 1, j, k)]) : Real(1);
      const Real qxm = face_avg(q[c], q[shape.offset(i - 1, j, k)]);
      const Real rxm = flux ? face_avg(rho[c], rho[shape.offset(i - 1, j, k)]) : Real(1);
      const Real uxp = face_avg(u[c], u[shape.offset(i + 1, j, k)]);
      const Real uxm = face_avg(u[c], u[shape.offset(i - 1, j, k)]);
      const Real dzx = (rxp * qxp * uxp - rxm * qxm * uxm) * geom.inv_dx_at(i);

      const Real qyp = face_avg(q[c], q[shape.offset(i, j + 1, k)]);
      const Real ryp = flux ? face_avg(rho[c], rho[shape.offset(i, j + 1, k)]) : Real(1);
      const Real qym = face_avg(q[c], q[shape.offset(i, j - 1, k)]);
      const Real rym = flux ? face_avg(rho[c], rho[shape.offset(i, j - 1, k)]) : Real(1);
      const Real vyp = face_avg(v[c], v[shape.offset(i, j + 1, k)]);
      const Real vym = face_avg(v[c], v[shape.offset(i, j - 1, k)]);
      const Real dzy = (ryp * qyp * vyp - rym * qym * vym) * geom.inv_dy_at(j);

      const Real qzp = face_avg(q[c], q[shape.offset(i, j, k + 1)]);
      const Real rzp = flux ? face_avg(rho[c], rho[shape.offset(i, j, k + 1)]) : Real(1);
      const Real qzm = face_avg(q[c], q[shape.offset(i, j, k - 1)]);
      const Real rzm = flux ? face_avg(rho[c], rho[shape.offset(i, j, k - 1)]) : Real(1);
      const Real wzp = face_avg(w[c], w[shape.offset(i, j, k + 1)]);
      const Real wzm = face_avg(w[c], w[shape.offset(i, j, k - 1)]);
      const Real dzz = (rzp * qzp * wzp - rzm * qzm * wzm) / geom.dz;

      if (flux) {
        const Real rinv = Real(1) / rho[c];
        adv = -(dzx + dzy + dzz) * rinv;
      } else {
        adv = -(dzx + dzy + dzz);
      }
    }
    // 散度阻尼（[D12]）：对趋势施加一次 Laplacian 型衰减
    if (options.divergence_damping > Real(0)) {
      const Real lap = (q[shape.offset(i + 1, j, k)] - Real(2) * q[c] + q[shape.offset(i - 1, j, k)]) *
                           (geom.inv_dx_at(i) * geom.inv_dx_at(i)) +
                       (q[shape.offset(i, j + 1, k)] - Real(2) * q[c] + q[shape.offset(i, j - 1, k)]) *
                           (geom.inv_dy_at(j) * geom.inv_dy_at(j)) +
                       (q[shape.offset(i, j, k + 1)] - Real(2) * q[c] + q[shape.offset(i, j, k - 1)]) /
                           (geom.dz * geom.dz);
      adv += options.divergence_damping * lap;
    }
    out[c] = adv;
  });
}

void advect_momentum(const Real* u, const Real* v, const Real* w, const Real* rho, Real* du,
                     Real* dv, Real* dw, const FieldShape& shape, const KernelGeometry& geom,
                     const AdvectionOptions& options) {
  VIBE_CHECK(u != nullptr && v != nullptr && w != nullptr && du != nullptr && dv != nullptr &&
             dw != nullptr);
  VIBE_UNUSED(rho);
  for_each_point(geom, [&](Int i, Int j, Int k) {
    const std::size_t c = shape.offset(i, j, k);
    const Real invdx = geom.inv_dx_at(i);
    const Real invdy = geom.inv_dy_at(j);
    const Real invdz = Real(1) / geom.dz;
    if (options.order == 4) {
      const Real ux = d1_x4(u, shape, i, j, k, invdx);
      const Real uy = d1_y4(u, shape, i, j, k, invdy);
      const Real uz = (u[shape.offset(i, j, k + 1)] - u[shape.offset(i, j, k - 1)]) * (Real(0.5) * invdz);
      const Real vx = d1_x4(v, shape, i, j, k, invdx);
      const Real vy = d1_y4(v, shape, i, j, k, invdy);
      const Real vz = (v[shape.offset(i, j, k + 1)] - v[shape.offset(i, j, k - 1)]) * (Real(0.5) * invdz);
      const Real wx = d1_x4(w, shape, i, j, k, invdx);
      const Real wy = d1_y4(w, shape, i, j, k, invdy);
      const Real wz = (w[shape.offset(i, j, k + 1)] - w[shape.offset(i, j, k - 1)]) * (Real(0.5) * invdz);
      du[c] = -(u[c] * ux + v[c] * uy + w[c] * uz);
      dv[c] = -(u[c] * vx + v[c] * vy + w[c] * vz);
      dw[c] = -(u[c] * wx + v[c] * wy + w[c] * wz);
    } else {
      // 2 阶：对流形式（非守恒），与标量的守恒形式配对使用（[D6] 的 C-grid 方案）
      du[c] = -(u[c] * d1_x(u, shape, i, j, k, invdx) + v[c] * d1_y(u, shape, i, j, k, invdy) +
                w[c] * d1_z(u, shape, i, j, k, invdz));
      dv[c] = -(u[c] * d1_x(v, shape, i, j, k, invdx) + v[c] * d1_y(v, shape, i, j, k, invdy) +
                w[c] * d1_z(v, shape, i, j, k, invdz));
      dw[c] = -(u[c] * d1_x(w, shape, i, j, k, invdx) + v[c] * d1_y(w, shape, i, j, k, invdy) +
                w[c] * d1_z(w, shape, i, j, k, invdz));
    }
  });
}

// ===========================================================================
// 2. 扩散
// ===========================================================================
void diffusion(const Real* q, Real* out, Real* work, const FieldShape& shape,
               const KernelGeometry& geom, Real kappa, DiffusionKind kind) {
  VIBE_CHECK(q != nullptr && out != nullptr);
  const auto laplacian = [&](const Real* src, Real* dst) {
    for_each_point(geom, [&](Int i, Int j, Int k) {
      const std::size_t c = shape.offset(i, j, k);
      const Real inv2x = geom.inv_dx_at(i) * geom.inv_dx_at(i);
      const Real inv2y = geom.inv_dy_at(j) * geom.inv_dy_at(j);
      const Real inv2z = Real(1) / (geom.dz * geom.dz);
      dst[c] = (src[shape.offset(i + 1, j, k)] - Real(2) * src[c] + src[shape.offset(i - 1, j, k)]) * inv2x +
               (src[shape.offset(i, j + 1, k)] - Real(2) * src[c] + src[shape.offset(i, j - 1, k)]) * inv2y +
               (src[shape.offset(i, j, k + 1)] - Real(2) * src[c] + src[shape.offset(i, j, k - 1)]) * inv2z;
    });
  };
  if (kind == DiffusionKind::Laplacian) {
    laplacian(q, out);
    for_each_point(geom, [&](Int i, Int j, Int k) { out[shape.offset(i, j, k)] *= kappa; });
    return;
  }
  VIBE_CHECK_MSG(work != nullptr, "biharmonic 扩散需要 work 中间缓冲");
  // 双调和算子把误差按 1/Delta x^4 放大，半精度中间量会直接毁掉解，
  // 因此要求 compute >= FP32（[G6] 的误差预算约束）。
  VIBE_CHECK_MSG(precision_rank(current_policy().compute) >= precision_rank(Precision::FP32),
                 "biharmonic 扩散要求 compute 精度 >= FP32");
  laplacian(q, work);      // 第一遍：nabla^2 q
  laplacian(work, out);    // 第二遍：nabla^4 q
  for_each_point(geom, [&](Int i, Int j, Int k) { out[shape.offset(i, j, k)] *= -kappa; });
}

// ===========================================================================
// 3. 三对角求解（批量 Thomas）
// ===========================================================================
void tridiagonal_solve(Real* a, Real* b, Real* c, Real* d, Real* x, std::size_t n_sys,
                       std::size_t nz) {
  VIBE_CHECK(a != nullptr && b != nullptr && c != nullptr && d != nullptr && x != nullptr);
  if (n_sys == 0 || nz == 0) return;

#if defined(_OPENMP)
#  pragma omp parallel for schedule(static) if (n_sys > 1)
#endif
  for (std::int64_t s = 0; s < static_cast<std::int64_t>(n_sys); ++s) {
    const std::size_t base = static_cast<std::size_t>(s) * nz;
    Real* as = a + base;
    Real* bs = b + base;
    Real* cs = c + base;
    Real* ds = d + base;
    Real* xs = x + base;
    // 对角占优检查（[B11] 第 4 章）：|b| >= |a| + |c| 必须成立，否则 Thomas
    // 法的无主元增长性不成立，明确报错而不是给出不可信的解。
    for (std::size_t k = 0; k < nz; ++k) {
      const Real lhs = std::abs(bs[k]);
      const Real rhs = (k > 0 ? std::abs(as[k]) : Real(0)) + (k + 1 < nz ? std::abs(cs[k]) : Real(0));
      if (lhs < rhs) {
        throw NumericalError("tridiagonal_solve: 第 " + std::to_string(s) + " 个系统不满足对角占优");
      }
    }
    // 前向消元
    Real piv = bs[0];
    if (std::abs(piv) < std::numeric_limits<Real>::min()) {
      throw NumericalError("tridiagonal_solve: 对角元为零，系统奇异");
    }
    cs[0] = cs[0] / piv;
    ds[0] = ds[0] / piv;
    for (std::size_t k = 1; k < nz; ++k) {
      piv = bs[k] - as[k] * cs[k - 1];
      if (std::abs(piv) < std::numeric_limits<Real>::min()) {
        throw NumericalError("tridiagonal_solve: 前向消元中出现零主元（系统奇异）");
      }
      cs[k] = (k + 1 < nz) ? cs[k] / piv : Real(0);
      ds[k] = (ds[k] - as[k] * ds[k - 1]) / piv;
    }
    // 回代
    xs[nz - 1] = ds[nz - 1];
    for (std::size_t kk = nz - 1; kk-- > 0;) {
      xs[kk] = ds[kk] - cs[kk] * xs[kk + 1];
    }
  }
}

void tridiagonal_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                       const std::vector<Real>& c, const std::vector<Real>& d,
                       std::vector<Real>& x) {
  const std::size_t nz = b.size();
  VIBE_CHECK_MSG(a.size() == nz && c.size() == nz && d.size() == nz, "三对角系数长度必须一致");
  std::vector<Real> aa = a, bb = b, cc = c, dd = d;
  x.assign(nz, Real(0));
  if (nz == 0) return;
  tridiagonal_solve(aa.data(), bb.data(), cc.data(), dd.data(), x.data(), 1, nz);
}

// ===========================================================================
// 4. Helmholtz
// ===========================================================================
void helmholtz_residual(const Real* x, const Real* a, const Real* b, const Real* f, Real* r,
                        const FieldShape& shape, const KernelGeometry& geom) {
  VIBE_CHECK(x != nullptr && a != nullptr && b != nullptr && f != nullptr && r != nullptr);
  for_each_point(geom, [&](Int i, Int j, Int k) {
    const std::size_t c = shape.offset(i, j, k);
    // r = f - (div(a grad x) - b x)。残差用 FP64 累加（compute 提升到 reduce），
    // 保证迭代精化所依赖的残差精度（[G5]）。
    const Real lhs = helmholtz_lhs(x, a, shape, geom, i, j, k) - b[c] * x[c];
    r[c] = static_cast<Real>(widen(static_cast<double>(f[c]) - static_cast<double>(lhs)));
  });
}

void helmholtz_jacobi(const Real* a, const Real* b, const Real* f, const Real* x, Real* out,
                      const FieldShape& shape, const KernelGeometry& geom, int sweeps,
                      bool chebyshev) {
  VIBE_CHECK(a != nullptr && b != nullptr && f != nullptr && x != nullptr && out != nullptr);
  VIBE_CHECK_MSG(sweeps >= 1, "helmholtz_jacobi: sweeps 必须 >= 1");
  // 单缓冲交替是安全的：每遍只读上一遍的 src、写另一块 dst，
  // 遍内不存在"读新值"的依赖（这正是 Jacobi 可与 Gauss-Seidel 区分的性质）。
  std::vector<Real> aux;
  if (sweeps > 1) aux.resize(shape.size(), Real(0));
  const Real* src = x;
  for (int s = 0; s < sweeps; ++s) {
    Real* dst = (s == sweeps - 1) ? out : aux.data();
    // omega：Jacobi 最优 2/3（3D 7 点，[T13] 式 (3.24)）；
    // Chebyshev 半迭代按 [T13] 第 3.3 节递推
    Real omega = Real(2) / Real(3);
    if (chebyshev) {
      const Real rho = (std::cos(kPi / std::max<Real>(Real(2), Real(geom.nx))) +
                        std::cos(kPi / std::max<Real>(Real(2), Real(geom.ny))) +
                        std::cos(kPi / std::max<Real>(Real(2), Real(geom.nz)))) /
                       Real(3);
      const Real rho2 = rho * rho;
      Real w = Real(1);
      for (int m = 0; m < s; ++m) w = Real(2) / (Real(2) - rho2 * w);
      omega = w;
    }
    for_each_point(geom, [&](Int i, Int j, Int k) {
      const std::size_t c = shape.offset(i, j, k);
      const Real d = helmholtz_diag(a, shape, geom, i, j, k) - b[c];
      const Real lhs = helmholtz_lhs(src, a, shape, geom, i, j, k) - b[c] * src[c];
      const Real resid = f[c] - lhs;
      // 保护零对角（b 极大时的退化情形）
      dst[c] = (std::abs(d) > std::numeric_limits<Real>::min())
                   ? src[c] + omega * resid / d
                   : src[c];
    });
    src = dst;
  }
}

// ===========================================================================
// 5. 嵌套限制 / 延拓
// ===========================================================================
void restrict_2to1(const Real* fine, Real* coarse, const FieldShape& fine_shape,
                   const FieldShape& coarse_shape, Int ratio) {
  VIBE_CHECK(fine != nullptr && coarse != nullptr);
  VIBE_CHECK_MSG(ratio >= 1, "restrict_2to1: ratio 必须 >= 1");
  const Int r = ratio;
  const Real w = Real(1) / static_cast<Real>(r * r);  // 面积权重 1/r^2（守恒）
  const Int nxc = coarse_shape.nsx - 2 * coarse_shape.halo;
  const Int nyc = coarse_shape.nsy - 2 * coarse_shape.halo;
  const Int nzc = coarse_shape.nsz - 2 * coarse_shape.halo;
#if defined(_OPENMP)
#  pragma omp parallel for collapse(2) schedule(static)
#endif
  for (Int k = 0; k < nzc; ++k) {
    for (Int j = 0; j < nyc; ++j) {
      for (Int i = 0; i < nxc; ++i) {
        CompensatedSum<Real, SumAlgorithm::Neumaier> acc;
        for (Int q = 0; q < r; ++q) {
          for (Int p = 0; p < r; ++p) {
            acc.add(fine[fine_shape.offset(i * r + p, j * r + q, k)]);
          }
        }
        coarse[coarse_shape.offset(i, j, k)] = acc.value() * w;
      }
    }
  }
}

void prolong_1to2(const Real* coarse, Real* fine, const FieldShape& coarse_shape,
                  const FieldShape& fine_shape, Int ratio, bool enforce_positive) {
  VIBE_CHECK(coarse != nullptr && fine != nullptr);
  VIBE_CHECK_MSG(ratio >= 1, "prolong_1to2: ratio 必须 >= 1");
  const Int r = ratio;
  const Real inv_r = Real(1) / static_cast<Real>(r);
  const Int nxc = coarse_shape.nsx - 2 * coarse_shape.halo;
  const Int nyc = coarse_shape.nsy - 2 * coarse_shape.halo;
  const Int nzc = coarse_shape.nsz - 2 * coarse_shape.halo;
  // 每个细网格单元的中心在粗网格坐标中的位置：I + (p + 0.5)/r
  const Int nxf = fine_shape.nsx - 2 * fine_shape.halo;
  const Int nyf = fine_shape.nsy - 2 * fine_shape.halo;
  const Int nzf = fine_shape.nsz - 2 * fine_shape.halo;
  VIBE_CHECK_MSG(nxf == nxc * r && nyf == nyc * r && nzf == nzc, "延拓要求细网格尺寸是粗网格的 r 倍（垂直一致）");
  for (Int k = 0; k < nzf; ++k) {
    const Int K = std::min(k, nzc - 1);
    for (Int J = 0; J < nyc; ++J) {
      for (Int I = 0; I < nxc; ++I) {
        // 预取 2x2 粗单元（边界上用线性外推：用相邻单元差分延长）
        const Real c00 = coarse[coarse_shape.offset(I, J, K)];
        const Real c10 = coarse[coarse_shape.offset(std::min(I + 1, nxc - 1), J, K)];
        const Real c01 = coarse[coarse_shape.offset(I, std::min(J + 1, nyc - 1), K)];
        const Real c11 = coarse[coarse_shape.offset(std::min(I + 1, nxc - 1), std::min(J + 1, nyc - 1), K)];
        for (Int q = 0; q < r; ++q) {
          const Real eta = (static_cast<Real>(q) + Real(0.5)) * inv_r;
          for (Int p = 0; p < r; ++p) {
            const Real xi = (static_cast<Real>(p) + Real(0.5)) * inv_r;
            Real val = w0(xi) * w0(eta) * c00 + w1(xi) * w0(eta) * c10 + w0(xi) * w1(eta) * c01 +
                       w1(xi) * w1(eta) * c11;
            if (enforce_positive) val = std::max(val, Real(0));
            fine[fine_shape.offset(I * r + p, J * r + q, k)] = val;
          }
        }
      }
    }
  }
}

// ===========================================================================
// 6. 插值
// ===========================================================================
Real bilinear_interp(const Real* f, const FieldShape& shape, const KernelGeometry& geom, Int k,
                     Real x, Real y) {
  VIBE_CHECK(f != nullptr);
  const CellLoc cx = locate_x(geom, x);
  const CellLoc cy = locate_y(geom, y);
  const Int i = cx.i, j = cy.i;
  const Real xi = cx.xi, eta = cy.xi;
  const Int kk = std::max(Int(0), std::min(k, geom.nz - 1));
  const Real q00 = f[shape.offset(i, j, kk)];
  const Real q10 = f[shape.offset(std::min(i + 1, geom.nx - 1), j, kk)];
  const Real q01 = f[shape.offset(i, std::min(j + 1, geom.ny - 1), kk)];
  const Real q11 = f[shape.offset(std::min(i + 1, geom.nx - 1), std::min(j + 1, geom.ny - 1), kk)];
  return w0(xi) * w0(eta) * q00 + w1(xi) * w0(eta) * q10 + w0(xi) * w1(eta) * q01 + w1(xi) * w1(eta) * q11;
}

void bilinear_interp_batch(const Real* f, const FieldShape& shape, const KernelGeometry& geom,
                           Int k, const Real* x, const Real* y, Real* out, std::size_t n) {
  VIBE_CHECK(f != nullptr && x != nullptr && y != nullptr && out != nullptr);
#if defined(_OPENMP)
#  pragma omp parallel for schedule(static) if (n > 512)
#endif
  for (std::int64_t m = 0; m < static_cast<std::int64_t>(n); ++m) {
    out[m] = bilinear_interp(f, shape, geom, k, x[m], y[m]);
  }
}

Real trilinear_interp(const Real* f, const FieldShape& shape, const KernelGeometry& geom, Real x,
                      Real y, Real z) {
  VIBE_CHECK(f != nullptr);
  const CellLoc cx = locate_x(geom, x);
  const CellLoc cy = locate_y(geom, y);
  Real fz = z / geom.dz;
  Real fzl = std::floor(fz);
  Int k = static_cast<Int>(std::max(Real(0), std::min(fzl, static_cast<Real>(geom.nz - 2))));
  Real zeta = std::max(Real(0), std::min(fz - static_cast<Real>(k), Real(1)));
  const Int kp = std::min(k + 1, geom.nz - 1);
  const Real q000 = f[shape.offset(cx.i, cy.i, k)];
  const Real q100 = f[shape.offset(std::min(cx.i + 1, geom.nx - 1), cy.i, k)];
  const Real q010 = f[shape.offset(cx.i, std::min(cy.i + 1, geom.ny - 1), k)];
  const Real q110 = f[shape.offset(std::min(cx.i + 1, geom.nx - 1), std::min(cy.i + 1, geom.ny - 1), k)];
  const Real q001 = f[shape.offset(cx.i, cy.i, kp)];
  const Real q101 = f[shape.offset(std::min(cx.i + 1, geom.nx - 1), cy.i, kp)];
  const Real q011 = f[shape.offset(cx.i, std::min(cy.i + 1, geom.ny - 1), kp)];
  const Real q111 = f[shape.offset(std::min(cx.i + 1, geom.nx - 1), std::min(cy.i + 1, geom.ny - 1), kp)];
  const Real za = w0(zeta), zb = w1(zeta);
  const Real xy0 = w0(cx.xi) * w0(cy.xi) * q000 + w1(cx.xi) * w0(cy.xi) * q100 +
                   w0(cx.xi) * w1(cy.xi) * q010 + w1(cx.xi) * w1(cy.xi) * q110;
  const Real xy1 = w0(cx.xi) * w0(cy.xi) * q001 + w1(cx.xi) * w0(cy.xi) * q101 +
                   w0(cx.xi) * w1(cy.xi) * q011 + w1(cx.xi) * w1(cy.xi) * q111;
  VIBE_UNUSED(k);
  return za * xy0 + zb * xy1;
}

void trilinear_interp_batch(const Real* f, const FieldShape& shape, const KernelGeometry& geom,
                            const Real* x, const Real* y, const Real* z, Real* out, std::size_t n) {
  VIBE_CHECK(f != nullptr && x != nullptr && y != nullptr && z != nullptr && out != nullptr);
#if defined(_OPENMP)
#  pragma omp parallel for schedule(static) if (n > 512)
#endif
  for (std::int64_t m = 0; m < static_cast<std::int64_t>(n); ++m) {
    out[m] = trilinear_interp(f, shape, geom, x[m], y[m], z[m]);
  }
}

// ===========================================================================
// 7. 归约
// ===========================================================================
namespace {

/// 分块补偿求和的核心：把 [0,n) 切成与线程数无关的固定块，
/// 每块用一个 CompensatedSum，最后按块序号用 Neumaier 规则合并。
/// 该结构保证**任意线程数下逐位相同**的结果（4D-Var 可复现性要求）。
template <SumAlgorithm S>
Real reduce_sum_blocked(const Real* data, std::size_t n) {
  if (n == 0) return Real(0);
  const std::size_t n_chunks = (n + kReduceChunk - 1) / kReduceChunk;
  std::vector<Real> partial(n_chunks, Real(0));
#if defined(_OPENMP)
#  pragma omp parallel for schedule(static) if (n_chunks > 1)
#endif
  for (std::int64_t c = 0; c < static_cast<std::int64_t>(n_chunks); ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * kReduceChunk;
    const std::size_t end = std::min(begin + kReduceChunk, n);
    CompensatedSumT<S, Real> acc;
    for (std::size_t i = begin; i < end; ++i) acc.add(data[i]);
    partial[static_cast<std::size_t>(c)] = acc.value();
  }
  CompensatedSumT<SumAlgorithm::Neumaier, Real> total;
  for (std::size_t c = 0; c < n_chunks; ++c) total.add(partial[c]);
  return total.value();
}

}  // namespace

Real reduce_sum(const Real* data, std::size_t n, SumAlgorithm algorithm) {
  if (data == nullptr || n == 0) return Real(0);
  switch (algorithm) {
    case SumAlgorithm::Kahan: return reduce_sum_blocked<SumAlgorithm::Kahan>(data, n);
    case SumAlgorithm::TwoSum: return reduce_sum_blocked<SumAlgorithm::TwoSum>(data, n);
    case SumAlgorithm::Neumaier:
    default: return reduce_sum_blocked<SumAlgorithm::Neumaier>(data, n);
  }
}

Real reduce_sum_strided(const Real* data, std::size_t n, std::size_t stride,
                        SumAlgorithm algorithm) {
  if (data == nullptr || n == 0 || stride == 0) return Real(0);
  // 步伐归约：先抽取到连续缓冲，再用同一套分块补偿求和（保持确定性）
  std::vector<Real> buf(n);
  for (std::size_t i = 0; i < n; ++i) buf[i] = data[i * stride];
  return reduce_sum(buf.data(), n, algorithm);
}

Real reduce_max(const Real* data, std::size_t n) {
  if (data == nullptr || n == 0) return -std::numeric_limits<Real>::infinity();
  const std::size_t n_chunks = (n + kReduceChunk - 1) / kReduceChunk;
  std::vector<Real> partial(n_chunks, -std::numeric_limits<Real>::infinity());
#if defined(_OPENMP)
#  pragma omp parallel for schedule(static) if (n_chunks > 1)
#endif
  for (std::int64_t c = 0; c < static_cast<std::int64_t>(n_chunks); ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * kReduceChunk;
    const std::size_t end = std::min(begin + kReduceChunk, n);
    Real m = -std::numeric_limits<Real>::infinity();
    for (std::size_t i = begin; i < end; ++i) m = data[i] > m ? data[i] : m;
    partial[static_cast<std::size_t>(c)] = m;
  }
  Real m = -std::numeric_limits<Real>::infinity();
  for (std::size_t c = 0; c < n_chunks; ++c) m = partial[c] > m ? partial[c] : m;
  return m;
}

void reduce_minmax(const Real* data, std::size_t n, Real& vmin, Real& vmax) {
  vmin = std::numeric_limits<Real>::infinity();
  vmax = -std::numeric_limits<Real>::infinity();
  if (data == nullptr || n == 0) return;
  const std::size_t n_chunks = (n + kReduceChunk - 1) / kReduceChunk;
  std::vector<Real> pmin(n_chunks, std::numeric_limits<Real>::infinity());
  std::vector<Real> pmax(n_chunks, -std::numeric_limits<Real>::infinity());
#if defined(_OPENMP)
#  pragma omp parallel for schedule(static) if (n_chunks > 1)
#endif
  for (std::int64_t c = 0; c < static_cast<std::int64_t>(n_chunks); ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * kReduceChunk;
    const std::size_t end = std::min(begin + kReduceChunk, n);
    Real lo = std::numeric_limits<Real>::infinity();
    Real hi = -std::numeric_limits<Real>::infinity();
    for (std::size_t i = begin; i < end; ++i) {
      const Real v = data[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    pmin[static_cast<std::size_t>(c)] = lo;
    pmax[static_cast<std::size_t>(c)] = hi;
  }
  for (std::size_t c = 0; c < n_chunks; ++c) {
    if (pmin[c] < vmin) vmin = pmin[c];
    if (pmax[c] > vmax) vmax = pmax[c];
  }
}

Real reduce_mean(const Real* data, std::size_t n, SumAlgorithm algorithm) {
  if (n == 0) return Real(0);
  return reduce_sum(data, n, algorithm) / static_cast<Real>(n);
}

}  // namespace vibe::gpu

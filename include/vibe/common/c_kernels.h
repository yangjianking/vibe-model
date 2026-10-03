/* ============================================================================
 *  include/vibe/common/c_kernels.h
 *
 *  纯 C11 热点内核层。
 *
 *  为什么在这里用 C 而不是 C++：
 *    1. 这些例程是"数组进、数组出"的紧循环，不需要任何抽象；
 *    2. C 版本可以被 FORTRAN / Python(ctypes) / 其它语言直接调用，
 *       便于与外部模式耦合和快速原型验证；
 *    3. 部分老牌 HPC 工具链对 C 的向量化诊断更成熟。
 *  框架与数据结构仍然全部用 C++（RAII、模板、异常），只有这里的内循环是 C。
 *
 *  约定
 *  ----
 *    * 所有标量类型为 double（对应 VIBE_PRECISION=0 的构建）；
 *      单精度构建使用 *_f32 后缀的同名函数。
 *    * 所有函数不分配内存；调用者提供 workspace。
 *    * 所有函数无全局状态，可重入，可被 OpenMP 线程安全调用。
 *    * 失败返回非 0 错误码，不设置 errno。
 *
 *  文献：[T10] Thomas (1949)；[D7] Shu (1998)；[D8] Jiang & Shu (1996)；
 *        [G3] Neumaier (1974)。
 * ==========================================================================*/
#ifndef VIBE_COMMON_C_KERNELS_H
#define VIBE_COMMON_C_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 错误码 */
#define VIBE_C_OK 0
#define VIBE_C_ERR_SINGULAR 1
#define VIBE_C_ERR_BAD_ARG 2

/* ---------------------------------------------------------------------------
 * 三对角求解（Thomas 算法，[T10]）
 *
 *   求解   a[k] x[k-1] + b[k] x[k] + c[k] x[k+1] = d[k],  k = 0..n-1
 *   其中 a[0] 与 c[n-1] 不参与运算。
 *
 * 参数：
 *   a, c   长度 n-1 的次对角/上对角（可为 NULL，此时视为 0）
 *   b      长度 n   的主对角
 *   d      长度 n   的右端项
 *   x      长度 n   的输出解
 *   ws     长度 n   的工作区（必须提供）
 *
 * 复杂度 O(n)，无除法以外的浮点运算发散。
 * 返回 VIBE_C_OK 或 VIBE_C_ERR_SINGULAR / VIBE_C_ERR_BAD_ARG。
 * -------------------------------------------------------------------------*/
int vibe_tridiagonal_solve_f64(const double *a, const double *b, const double *c,
                               const double *d, int n, double *x, double *ws);

/* ---------------------------------------------------------------------------
 * WENO5 面值重构（[D8] 式 (2.6)）
 *
 *   qm2..q2 为面 i+1/2 两侧各 2 个单元的平均值；
 *   输出该面左右偏重构值 q_left（右行波用）与 q_right（左行波用）。
 *   非线性权重 d = (0.1, 0.6, 0.3)，epsilon = 1e-6。
 * -------------------------------------------------------------------------*/
void vibe_weno5_reconstruct_f64(double qm2, double qm1, double q0, double q1,
                                double q2, double *q_left, double *q_right);

/* ---------------------------------------------------------------------------
 * 双线性插值（观测算子与嵌套共用）
 *
 *   f 为带 halo 的行主序数组，存储维度 nsx x nsy x nsz，halo 为 halo 宽度。
 *   (xi, yj) 为"体心分数索引"（0.0 表示第 0 个体心），k 为层号。
 *   越界时钳制到边界（与 grid::Field::clamp_at 语义一致）。
 * -------------------------------------------------------------------------*/
double vibe_bilinear_f64(const double *f, int halo, int nsx, int nsy, int nz,
                         double xi, double yj, int k);

/* ---------------------------------------------------------------------------
 * 声波/平流 CFL 扫描
 *
 *   返回 max over 格点 of
 *       dt * ( |u|/dx + |v|/dy + |w| / (dzeta[k] * (z_top - zs)) )
 *   额外通过输出参数返回 (max_wind, min_dz)。
 *   u/v/w 为不含 halo 的体心插值风速（长度 nx*ny*nz，行主序 i 最快）。
 *   zs 可为 NULL（平坦地形）。
 * -------------------------------------------------------------------------*/
double vibe_cfl_scan_f64(const double *u, const double *v, const double *w,
                         const double *dzeta, const double *zs, int nx, int ny,
                         int nz, double dx, double dy, double z_top, double dt,
                         double *max_wind_out, double *min_dz_out);

/* ---------------------------------------------------------------------------
 * Neumaier 补偿求和（[G3]）
 *
 *   对长度 n 的数组求和，误差与 O(u^2) 成正比而非 O(n u)。
 *   返回精确到双精度极限的累加值。
 * -------------------------------------------------------------------------*/
double vibe_neumaier_sum_f64(const double *x, size_t n);

/* ---------------------------------------------------------------------------
 * 通量形式的标量平流内循环（二阶中心面通量）
 *
 *   dqdt[i] = -[ (Fx[i+1]-Fx[i])/dx + (Fy[j+1]-Fy[j])/dy + (Fz[k+1]-Fz[k])/dz ] / rho
 *   Fx[i] = u[i] * 0.5*(rho[i-1]+rho[i]) * 0.5*(q[i-1]+q[i])
 *
 *   所有输入均为体心布局、行主序、无 halo、i 最快；u/v/w 已插值到体心，
 *   其 i+1 位置由调用者保证（数组在边界外多分配一圈或使用周期索引）。
 *   本函数只处理内部点 1..nx-2（i 方向）。
 * -------------------------------------------------------------------------*/
void vibe_advect_scalar_flux_f64(const double *q, const double *rho,
                                 const double *u, const double *v, const double *w,
                                 const double *dzeta, const double *zs, int nx,
                                 int ny, int nz, double dx, double dy,
                                 double z_top, double *dqdt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VIBE_COMMON_C_KERNELS_H */

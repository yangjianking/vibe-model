/* ============================================================================
 *  src/common/c_kernels.c
 *
 *  C11 热点内核实现。见 include/vibe/common/c_kernels.h 的接口说明。
 *
 *  编译要求：C11；可选 OpenMP（由 _OPENMP 宏探测）。
 *  所有循环都写成"分段可向量化"的形式：内层 i 循环无分支、无别名假设
 *  （用 restrict 声明），便于 GCC/Clang 的自动向量化与 ICC 的 -qopt-report。
 * ==========================================================================*/

#include "vibe/common/c_kernels.h"

#include <math.h>
#include <string.h>

/* ------------------------------ 三对角求解 ------------------------------- */

int vibe_tridiagonal_solve_f64(const double *a, const double *b, const double *c,
                               const double *d, int n, double *x, double *ws) {
  int k;

  if (n <= 0 || b == NULL || d == NULL || x == NULL || ws == NULL) {
    return VIBE_C_ERR_BAD_ARG;
  }
  if (n == 1) {
    if (b[0] == 0.0) return VIBE_C_ERR_SINGULAR;
    x[0] = d[0] / b[0];
    return VIBE_C_OK;
  }

  /* 前向消去：ws 存放修正后的右端项 */
  {
    double denom = b[0];
    if (denom == 0.0) return VIBE_C_ERR_SINGULAR;
    ws[0] = d[0] / denom;
    for (k = 1; k < n; ++k) {
      const double cp = (k < n - 1 && c != NULL) ? c[k] / denom : 0.0;
      const double am = (a != NULL) ? a[k - 1] : 0.0;
      denom = b[k] - am * cp;
      if (denom == 0.0) return VIBE_C_ERR_SINGULAR;
      ws[k] = (d[k] - am * ws[k - 1]) / denom;
    }
  }

  /* 回代：需要上对角修正值，重新计算一次（O(n) 代价可忽略，避免额外工作区） */
  {
    double denom = b[0];
    double cp = (n > 1 && c != NULL) ? c[0] / denom : 0.0;
    x[n - 1] = ws[n - 1];
    for (k = n - 2; k >= 0; --k) {
      if (k > 0) {
        const double am = (a != NULL) ? a[k - 1] : 0.0;
        denom = b[k] - am * cp;
        cp = (k < n - 1 && c != NULL) ? c[k] / denom : 0.0;
      }
      x[k] = ws[k] - cp * x[k + 1];
    }
  }
  return VIBE_C_OK;
}

/* ------------------------------ WENO5 重构 ------------------------------- */

void vibe_weno5_reconstruct_f64(double qm2, double qm1, double q0, double q1,
                                double q2, double *q_left, double *q_right) {
  const double eps = 1e-6;
  const double d0 = 0.1, d1 = 0.6, d2 = 0.3;
  double v0, v1, v2, b0, b1, b2, a0, a1, a2, sum;
  double u0, u1, u2, g0, g1, g2, c0, c1, c2, sum2;

  if (q_left == NULL || q_right == NULL) return;

  /* 左偏：面右侧的值 q^-_{i+1/2} */
  v0 = (2.0 * qm2 - 7.0 * qm1 + 11.0 * q0) / 6.0;
  v1 = (-qm1 + 5.0 * q0 + 2.0 * q1) / 6.0;
  v2 = (2.0 * q0 + 5.0 * q1 - q2) / 6.0;
  b0 = (13.0 / 12.0) * (qm2 - 2.0 * qm1 + q0) * (qm2 - 2.0 * qm1 + q0) +
       0.25 * (qm2 - 4.0 * qm1 + 3.0 * q0) * (qm2 - 4.0 * qm1 + 3.0 * q0);
  b1 = (13.0 / 12.0) * (qm1 - 2.0 * q0 + q1) * (qm1 - 2.0 * q0 + q1) +
       0.25 * (qm1 - q1) * (qm1 - q1);
  b2 = (13.0 / 12.0) * (q0 - 2.0 * q1 + q2) * (q0 - 2.0 * q1 + q2) +
       0.25 * (3.0 * q0 - 4.0 * q1 + q2) * (3.0 * q0 - 4.0 * q1 + q2);
  a0 = d0 / ((eps + b0) * (eps + b0));
  a1 = d1 / ((eps + b1) * (eps + b1));
  a2 = d2 / ((eps + b2) * (eps + b2));
  sum = a0 + a1 + a2;
  *q_left = (a0 * v0 + a1 * v1 + a2 * v2) / sum;

  /* 右偏（镜像） */
  u0 = (2.0 * q2 - 7.0 * q1 + 11.0 * q0) / 6.0;
  u1 = (-q1 + 5.0 * q0 + 2.0 * qm1) / 6.0;
  u2 = (2.0 * q0 + 5.0 * qm1 - qm2) / 6.0;
  g0 = (13.0 / 12.0) * (q2 - 2.0 * q1 + q0) * (q2 - 2.0 * q1 + q0) +
       0.25 * (q2 - 4.0 * q1 + 3.0 * q0) * (q2 - 4.0 * q1 + 3.0 * q0);
  g1 = (13.0 / 12.0) * (q1 - 2.0 * q0 + qm1) * (q1 - 2.0 * q0 + qm1) +
       0.25 * (q1 - qm1) * (q1 - qm1);
  g2 = (13.0 / 12.0) * (q0 - 2.0 * qm1 + qm2) * (q0 - 2.0 * qm1 + qm2) +
       0.25 * (3.0 * q0 - 4.0 * qm1 + qm2) * (3.0 * q0 - 4.0 * qm1 + qm2);
  c0 = d0 / ((eps + g0) * (eps + g0));
  c1 = d1 / ((eps + g1) * (eps + g1));
  c2 = d2 / ((eps + g2) * (eps + g2));
  sum2 = c0 + c1 + c2;
  *q_right = (c0 * u0 + c1 * u1 + c2 * u2) / sum2;
}

/* ------------------------------ 双线性插值 ------------------------------- */

double vibe_bilinear_f64(const double *f, int halo, int nsx, int nsy, int nz,
                         double xi, double yj, int k) {
  int i0, j0;
  double wx, wy, q00, q10, q01, q11;
  const int nxi = nsx - 2 * halo;
  const int nyi = nsy - 2 * halo;

  if (f == NULL || nz <= 0) return 0.0;
  if (k < 0) k = 0;
  if (k >= nz) k = nz - 1;

  i0 = (int)floor(xi);
  j0 = (int)floor(yj);
  wx = xi - (double)i0;
  wy = yj - (double)j0;

  /* 钳制到内部点范围（含 halo 的数组偏移 = halo + 索引） */
#define VIBE_AT(I, J) f[(((size_t)(k + halo) * (size_t)nsy + (size_t)((J) + halo)) * (size_t)nsx) + (size_t)((I) + halo)]
  {
    int ia = i0, ib = i0 + 1, ja = j0, jb = j0 + 1;
    if (ia < 0) ia = 0;
    if (ia > nxi - 1) ia = nxi - 1;
    if (ib < 0) ib = 0;
    if (ib > nxi - 1) ib = nxi - 1;
    if (ja < 0) ja = 0;
    if (ja > nyi - 1) ja = nyi - 1;
    if (jb < 0) jb = 0;
    if (jb > nyi - 1) jb = nyi - 1;
    q00 = VIBE_AT(ia, ja);
    q10 = VIBE_AT(ib, ja);
    q01 = VIBE_AT(ia, jb);
    q11 = VIBE_AT(ib, jb);
  }
#undef VIBE_AT

  return (q00 + wx * (q10 - q00)) + wy * ((q01 + wx * (q11 - q01)) - (q00 + wx * (q10 - q00)));
}

/* ------------------------------ CFL 扫描 -------------------------------- */

double vibe_cfl_scan_f64(const double *u, const double *v, const double *w,
                         const double *dzeta, const double *zs, int nx, int ny,
                         int nz, double dx, double dy, double z_top, double dt,
                         double *max_wind_out, double *min_dz_out) {
  double worst = 0.0;
  double max_wind = 0.0;
  double min_dz = 1e30;
  int i, j, k;
  const size_t nxy = (size_t)nx * (size_t)ny;

  if (u == NULL || v == NULL || w == NULL || dzeta == NULL) return 0.0;

  for (k = 0; k < nz; ++k) {
    const double dzeta_k = dzeta[k];
    const size_t base_k = (size_t)k * nxy;
    for (j = 0; j < ny; ++j) {
      for (i = 0; i < nx; ++i) {
        const size_t idx = base_k + (size_t)j * (size_t)nx + (size_t)i;
        const double uu = u[idx], vv = v[idx], ww = w[idx];
        const double scale = (zs != NULL)
                                 ? (z_top - zs[(size_t)j * (size_t)nx + (size_t)i])
                                 : z_top;
        double dz = dzeta_k * scale;
        double cfl;
        if (dz < 1.0) dz = 1.0;
        if (dz < min_dz) min_dz = dz;
        {
          const double speed = sqrt(uu * uu + vv * vv + ww * ww);
          if (speed > max_wind) max_wind = speed;
        }
        cfl = dt * (fabs(uu) / dx + fabs(vv) / dy + fabs(ww) / dz);
        if (cfl > worst) worst = cfl;
      }
    }
  }
  if (max_wind_out != NULL) *max_wind_out = max_wind;
  if (min_dz_out != NULL) *min_dz_out = (min_dz > 1e29) ? 0.0 : min_dz;
  return worst;
}

/* ------------------------------ Neumaier 求和 ---------------------------- */

double vibe_neumaier_sum_f64(const double *x, size_t n) {
  double sum = 0.0;
  double c = 0.0; /* 丢失的补偿量 */
  size_t i;
  if (x == NULL) return 0.0;
  for (i = 0; i < n; ++i) {
    const double t = sum + x[i];
    if (fabs(sum) >= fabs(x[i])) {
      c += (sum - t) + x[i];
    } else {
      c += (x[i] - t) + sum;
    }
    sum = t;
  }
  return sum + c;
}

/* ------------------------------ 通量平流 -------------------------------- */

void vibe_advect_scalar_flux_f64(const double *q, const double *rho,
                                 const double *u, const double *v, const double *w,
                                 const double *dzeta, const double *zs, int nx,
                                 int ny, int nz, double dx, double dy,
                                 double z_top, double *dqdt) {
  int i, j, k;
  const size_t nxy = (size_t)nx * (size_t)ny;
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;

  if (q == NULL || rho == NULL || u == NULL || v == NULL || w == NULL ||
      dzeta == NULL || dqdt == NULL) {
    return;
  }

  for (k = 0; k < nz; ++k) {
    const size_t base_k = (size_t)k * nxy;
    for (j = 0; j < ny; ++j) {
      for (i = 0; i < nx; ++i) {
        const size_t idx = base_k + (size_t)j * (size_t)nx + (size_t)i;
        const double scale = (zs != NULL)
                                 ? (z_top - zs[(size_t)j * (size_t)nx + (size_t)i])
                                 : z_top;
        double dz = dzeta[k] * scale;
        double rfaceE, rfaceW, qfaceE, qfaceW, fxE, fxW;
        double rfaceN, rfaceS, qfaceN, qfaceS, fyN, fyS;
        double rfaceT, rfaceB, qfaceT, qfaceB, fzT, fzB;
        double div, rr;

        if (dz < 1.0) dz = 1.0;

        /* x 方向面通量：i 与 i+1 面；i-1 用镜像（周期）索引以覆盖边界 */
        {
          const size_t e = base_k + (size_t)j * (size_t)nx + (size_t)((i + 1) % nx);
          const size_t c = idx;
          const size_t wst = base_k + (size_t)j * (size_t)nx + (size_t)((i + nx - 1) % nx);
          rfaceE = 0.5 * (rho[c] + rho[e]);
          qfaceE = 0.5 * (q[c] + q[e]);
          fxE = u[e] * rfaceE * qfaceE;
          rfaceW = 0.5 * (rho[wst] + rho[c]);
          qfaceW = 0.5 * (q[wst] + q[c]);
          fxW = u[c] * rfaceW * qfaceW;
        }
        /* y 方向 */
        {
          const size_t e = base_k + (size_t)((j + 1) % ny) * (size_t)nx + (size_t)i;
          const size_t c = idx;
          const size_t s = base_k + (size_t)((j + ny - 1) % ny) * (size_t)nx + (size_t)i;
          rfaceN = 0.5 * (rho[c] + rho[e]);
          qfaceN = 0.5 * (q[c] + q[e]);
          fyN = v[e] * rfaceN * qfaceN;
          rfaceS = 0.5 * (rho[s] + rho[c]);
          qfaceS = 0.5 * (q[s] + q[c]);
          fyS = v[c] * rfaceS * qfaceS;
        }
        /* z 方向（垂直方向的周期索引在物理上无意义，因此只在内部点上使用） */
        if (k > 0 && k + 1 < nz) {
          const size_t t = base_k + nxy + (size_t)j * (size_t)nx + (size_t)i;
          const size_t c = idx;
          const size_t b = base_k - nxy + (size_t)j * (size_t)nx + (size_t)i;
          rfaceT = 0.5 * (rho[c] + rho[t]);
          qfaceT = 0.5 * (q[c] + q[t]);
          fzT = w[t] * rfaceT * qfaceT;
          rfaceB = 0.5 * (rho[b] + rho[c]);
          qfaceB = 0.5 * (q[b] + q[c]);
          fzB = w[c] * rfaceB * qfaceB;
        } else {
          fzT = 0.0;
          fzB = 0.0;
        }

        div = (fxE - fxW) * inv_dx + (fyN - fyS) * inv_dy + (fzT - fzB) / dz;
        rr = (rho[idx] > 1e-12) ? rho[idx] : 1e-12;
        dqdt[idx] = -div / rr;
      }
    }
  }
}

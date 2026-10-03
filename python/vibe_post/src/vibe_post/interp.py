"""插值：双/三线性、单调 PCHIP 垂直插值、守恒粗化与延拓。

对应 C++ `vibe::grid::interpolation`（include/vibe/grid/interpolation.hpp）。
全部为 numpy 向量化的纯函数，不依赖 scipy。

坐标约定
--------
* 体心场 `f[k, j, i]` 的物理坐标：`x_i = x0 + (i + 1/2) dx`，
  `y_j = y0 + (j + 1/2) dy`，`z` 由 :class:`vibe_post.grid.GridSpec` 的
  地形追随坐标给出（[D2]）。
* 垂直方向第一维（axis=0）为层序；:func:`interp_to_pressure` 内部会把
  递减气压翻转为递增坐标后再插值。
* 越界点默认**零梯度外推**（钳位索引），与 C++ `grid::bilinear` 语义一致；
  可用 `extrapolate=False` 改为返回 NaN。

文献：[D3] Arakawa & Lamb (1977) 错位插值；[B9] Wilks 第 3 章（单调插值）；
[N1] Davies (1976)；[N5] MPAS 守恒重映射；[D16] WRF ARW 第 3 章。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "bilinear",
    "trilinear",
    "pchip",
    "pchip_slopes",
    "interp_to_height",
    "interp_to_pressure",
    "column_interp_to_height",
    "column_interp_to_pressure",
    "conservative_coarsen",
    "bilinear_prolong",
    "extrapolate_profile",
    "REGISTRY",
]


# ---------------------------------------------------------------------------
# 水平插值
# ---------------------------------------------------------------------------


def bilinear(
    field: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    dx: float | ArrayLike = 1.0,
    dy: float | ArrayLike = 1.0,
    extrapolate: bool = True,
) -> NDArray[np.float64]:
    """双线性插值，支持批量点（与 C++ `grid::bilinear` 语义一致）。

    公式
    ----
    令 :math:`(\\xi, \\eta)` 为体心索坐标，`i = \\lfloor \\xi \\rfloor`、
    `a = \\xi - i`（j、b 同理）：

    .. math::

        f(\\xi,\\eta) =
        (1-a)(1-b) f_{i,j} + a(1-b) f_{i+1,j}
        + (1-a) b  f_{i,j+1} + a b f_{i+1,j+1}

    其中体心坐标与物理坐标的关系为 `\\xi = x/dx - 1/2`（C-grid 体心偏移半格）。
    边界处索引被钳位到 `[0, n-1]`，等价于**零梯度（单侧常数）外推**；
    `extrapolate=False` 时越界点返回 NaN。

    离散化与误差：该格式在 `dx\\to 0` 时为二阶精确
    :math:`O(\\Delta x^2)`（[D3] 第 3 节）。

    复杂度：O(M)，M = 插值点数；内存 O(M)。
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("bilinear 需要至少二维场 (..., y, x)")
    ny, nx = arr.shape[-2], arr.shape[-1]
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr, y_arr = np.broadcast_arrays(x_arr, y_arr)

    dx_arr = np.asarray(dx, dtype=np.float64)
    dy_arr = np.asarray(dy, dtype=np.float64)

    if dx_arr.ndim == 0:
        xi = (x_arr - x0) / float(dx_arr) - 0.5
    else:
        xi = _staggered_to_index(x_arr, dx_arr, x0)
    if dy_arr.ndim == 0:
        eta = (y_arr - y0) / float(dy_arr) - 0.5
    else:
        eta = _staggered_to_index(y_arr, dy_arr, y0)

    out_of_range = (xi < 0.0) | (xi > nx - 1.0) | (eta < 0.0) | (eta > ny - 1.0)
    i0 = np.clip(np.floor(xi).astype(np.intp), 0, nx - 1)
    j0 = np.clip(np.floor(eta).astype(np.intp), 0, ny - 1)
    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    a = np.clip(xi - i0, 0.0, 1.0)
    b = np.clip(eta - j0, 0.0, 1.0)

    f00 = arr[..., j0, i0]
    f10 = arr[..., j1, i0]
    f01 = arr[..., j0, i1]
    f11 = arr[..., j1, i1]
    result = ((1.0 - a) * (1.0 - b) * f00 + a * (1.0 - b) * f01
              + (1.0 - a) * b * f10 + a * b * f11)
    if not extrapolate:
        result = np.where(out_of_range, np.nan, result)
    return np.asarray(result, dtype=np.float64)


def _staggered_to_index(coord: NDArray[np.float64], spacing: NDArray[np.float64],
                        origin: float) -> NDArray[np.float64]:
    """非均匀网格（`dx_cell`/`dy_cell`）上：物理坐标 -> 体心索引（连续值）。

    体心位置：`xc_0 = origin + dx_0/2`，`xc_n = xc_{n-1} + (dx_{n-1}+dx_n)/2`。
    用 `np.searchsorted` 定位区间后线性插值索引，复杂度 O(M log n)。
    """
    centers = origin + np.cumsum(spacing) - 0.5 * spacing
    idx = np.searchsorted(centers, coord, side="right").astype(np.float64) - 1.0
    idx = np.clip(idx, 0.0, max(centers.size - 2, 0))
    i0 = idx.astype(np.intp)
    i1 = np.clip(i0 + 1, 0, centers.size - 1)
    c0 = centers[i0]
    c1 = centers[i1]
    span = np.where(c1 > c0, c1 - c0, 1.0)
    return i0 + (coord - c0) / span


def trilinear(
    field: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    height: ArrayLike,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    dx: float | ArrayLike = 1.0,
    dy: float | ArrayLike = 1.0,
    extrapolate: bool = True,
) -> NDArray[np.float64]:
    """三线性插值 `f(x, y, z)`，批量点。

    公式
    ----
    先在垂直方向按 :math:`c` 权重取两个相邻层：

    .. math::

        F_{i,j} = (1-c) f_{k,j,i} + c f_{k+1,j,i}, \\quad
        c = \\frac{z - z_k}{z_{k+1} - z_k}

    再做双线性 :func:`bilinear`。垂直索引 `k` 由 `height` 场（形状
    `(nz, ny, nx)`，:meth:`GridSpec.height_array` 生成）按列单调查找得到
    （逐列二分，复杂度 O(M log nz)）。

    复杂度：O(M log nz + M)，M 为点数；内存 O(M)。
    """
    f3 = np.asarray(field, dtype=np.float64)
    if f3.ndim != 3:
        raise ValueError("trilinear 需要三维场 (nz, ny, nx)")
    h3 = np.asarray(height, dtype=np.float64)
    if h3.shape != f3.shape:
        raise ValueError("height 形状 {0} 与场 {1} 不一致".format(h3.shape, f3.shape))
    x_arr, y_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64))

    nz = f3.shape[0]
    # 逐列垂直查找：先按 (y, x) 取列，再在列内二分
    dx_arr = np.asarray(dx, dtype=np.float64)
    dy_arr = np.asarray(dy, dtype=np.float64)
    if dx_arr.ndim == 0:
        xi = (x_arr - x0) / float(dx_arr) - 0.5
    else:
        xi = _staggered_to_index(x_arr, dx_arr, x0)
    if dy_arr.ndim == 0:
        eta = (y_arr - y0) / float(dy_arr) - 0.5
    else:
        eta = _staggered_to_index(y_arr, dy_arr, y0)
    ny, nx = f3.shape[1], f3.shape[2]
    i0 = np.clip(np.floor(xi).astype(np.intp), 0, nx - 1)
    j0 = np.clip(np.floor(eta).astype(np.intp), 0, ny - 1)
    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    a = np.clip(xi - i0, 0.0, 1.0)
    b = np.clip(eta - j0, 0.0, 1.0)

    column = h3[:, j0, i0]                       # (nz, M)
    k0 = np.empty(z_arr.shape, dtype=np.intp)
    c = np.empty(z_arr.shape, dtype=np.float64)
    # 逐列处理：列数 = 点数，列内层数 nz（向量化 slice 搜索）
    flat_z = z_arr.ravel()
    flat_k0 = k0.ravel()
    flat_c = c.ravel()
    flat_j0 = j0.ravel()
    flat_i0 = i0.ravel()
    col_flat = column.reshape(nz, -1)
    # 预先判断每列的单调方向（地形追随高度恒为递增，但插值到气压层时会传入递减场）
    increasing = col_flat[-1] >= col_flat[0]
    for m in range(col_flat.shape[1]):
        zcol = col_flat[:, m]
        if not increasing[m]:
            zcol = zcol[::-1]
        target = flat_z[m]
        idx = np.searchsorted(zcol, target, side="right") - 1
        idx = int(np.clip(idx, 0, nz - 2))
        z_lo, z_hi = float(zcol[idx]), float(zcol[idx + 1])
        weight = 0.0 if z_hi == z_lo else (target - z_lo) / (z_hi - z_lo)
        flat_k0[m] = idx
        flat_c[m] = weight
    k1 = np.clip(k0 + 1, 0, nz - 1)
    c = np.clip(c, 0.0, 1.0)

    n_batch = int(np.asarray(x_arr).size)
    x_batch = np.broadcast_to(np.asarray(x_arr).reshape(n_batch, 1), (n_batch, 1))
    y_batch = np.broadcast_to(np.asarray(y_arr).reshape(n_batch, 1), (n_batch, 1))
    f_at_lo = np.empty((n_batch, 1), dtype=np.float64)
    f_at_hi = np.empty((n_batch, 1), dtype=np.float64)
    k0_flat = np.asarray(k0).ravel()
    k1_flat = np.asarray(k1).ravel()
    i0_flat = np.asarray(i0).ravel()
    j0_flat = np.asarray(j0).ravel()
    for m in range(n_batch):
        f_at_lo[m, 0] = bilinear(f3[k0_flat[m], :, :], x_batch[m:m + 1, 0], y_batch[m:m + 1, 0],
                                 x0=x0, y0=y0, dx=dx, dy=dy, extrapolate=extrapolate)[0]
        f_at_hi[m, 0] = bilinear(f3[k1_flat[m], :, :], x_batch[m:m + 1, 0], y_batch[m:m + 1, 0],
                                 x0=x0, y0=y0, dx=dx, dy=dy, extrapolate=extrapolate)[0]
    f_at_lo = f_at_lo.reshape(np.asarray(x_arr).shape)
    f_at_hi = f_at_hi.reshape(np.asarray(x_arr).shape)

    result = (1.0 - c) * f_at_lo + c * f_at_hi
    if not extrapolate:
        z_min = h3.min(axis=0)
        z_max = h3.max(axis=0)
        outside = (z_arr < z_min[j0, i0]) | (z_arr > z_max[j0, i0])
        result = np.where(outside, np.nan, result)
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# 单调三次插值（PCHIP，自实现，不依赖 scipy）
# ---------------------------------------------------------------------------


def pchip_slopes(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """PCHIP 节点导数（Fritsch-Carlson 单调性限制）。

    公式（[B9] Wilks 第 3 章；Fritsch & Carlson 1980）
    ------------------------------------------------
    差商 :math:`\\delta_k = (y_{k+1}-y_k)/(x_{k+1}-x_k)`；
    内部节点取加权调和平均

    .. math::

        d_k = \\frac{w_1 + w_2}{\\dfrac{w_1}{\\delta_{k-1}} + \\dfrac{w_2}{\\delta_k}},
        \\quad w_1 = 2 h_k + h_{k-1},\\; w_2 = h_k + 2 h_{k-1}

    若 :math:`\\delta_{k-1}\\delta_k \\le 0` 则 :math:`d_k = 0`（局部极值处
    置零，保证不越界）。端点用单侧三点公式后再做同样的单调限制。

    复杂度：O(n)（n 为节点数），全程 numpy 向量化，无 Python 循环。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x 与 y 长度不一致")
    n = x.size
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    h = np.diff(x)
    if np.any(h <= 0.0):
        raise ValueError("pchip 要求 x 严格单调递增（请先排序或翻转坐标）")
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=np.float64)

    if n == 2:
        d[:] = delta[0]
        return d

    # 内部节点
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    same_sign = delta[:-1] * delta[1:] > 0.0
    idx = np.arange(1, n - 1)
    d[idx] = np.where(same_sign, harmonic, 0.0)

    # 端点：三点单侧公式 + 限制
    def endpoint(h0: float, h1: float, d0: float, d1: float, d2: float) -> float:
        value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(value) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(value) > abs(3.0 * d0):
            return 3.0 * d0
        return value

    d[0] = endpoint(h[0], h[1], delta[0], delta[1],
                   delta[2] if n > 3 else delta[1])
    d[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2],
                    delta[-3] if n > 3 else delta[-2])
    return d


def pchip(x: ArrayLike, y: ArrayLike, xq: ArrayLike) -> NDArray[np.float64]:
    """单调分段三次 Hermite 插值（PCHIP）。

    公式（分段 Hermite，[B9] Wilks 第 3 章）
    --------------------------------------
    在 :math:`[x_k, x_{k+1}]` 上，令 :math:`t = (x_q - x_k)/h_k`：

    .. math::

        p(t) = h_{00}(t)\\, y_k + h_{10}(t)\\, h_k d_k
             + h_{01}(t)\\, y_{k+1} + h_{11}(t)\\, h_k d_{k+1}

    其中 :math:`h_{00}=2t^3-3t^2+1`、:math:`h_{10}=t^3-2t^2+t`、
    :math:`h_{01}=-2t^3+3t^2`、:math:`h_{11}=t^3-t^2`；
    导数 :math:`d_k` 由 :func:`pchip_slopes` 给出。该格式保证
    **单调性保持**（不产生过冲），对水汽、位温等正定量尤其重要。

    越界处理：`xq` 在节点范围之外时**钳到端点值**（零梯度外推），这是垂直
    插值到模式层以上高度时的标准做法（[D16] 第 3 章）。

    复杂度：O(n)（斜率）+ O(M log n)（搜索）；内存 O(M+n)。
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    xq_arr = np.asarray(xq, dtype=np.float64)
    if x_arr.size < 2:
        if x_arr.size == 1:
            return np.full(xq_arr.shape, float(y_arr[0]))
        return np.full(xq_arr.shape, np.nan)
    order = np.argsort(x_arr, kind="stable")
    if not np.all(order == np.arange(x_arr.size)):
        x_arr, y_arr = x_arr[order], y_arr[order]
    d = pchip_slopes(x_arr, y_arr)
    h = np.diff(x_arr)
    delta = np.diff(y_arr) / h

    flat = np.clip(xq_arr, x_arr[0], x_arr[-1]).ravel()
    k = np.clip(np.searchsorted(x_arr, flat, side="right") - 1, 0, x_arr.size - 2)
    t = (flat - x_arr[k]) / h[k]
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    values = (h00 * y_arr[k] + h10 * h[k] * d[k]
              + h01 * y_arr[k + 1] + h11 * h[k] * d[k + 1])
    return values.reshape(xq_arr.shape)


# ---------------------------------------------------------------------------
# 逐列垂直插值
# ---------------------------------------------------------------------------


def _broadcast_target(target: ArrayLike, tail: tuple[int, ...]):
    """把插值目标广播到与垂直廓线点集同形。

    返回 (目标数组, 输出形状)：标量目标 -> tail；与 tail 同形或可广播的数组 ->
    广播后的形状。若形状既不等于 tail 也无法广播，则退化为 (n_points, n_targets)
    的“点 x 目标”二维形式（批处理模式）。复杂度 O(n)。
    """
    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim == 0:
        n_points = int(np.prod(tail, dtype=int)) if tail else 1
        return np.full(n_points, float(arr)), tail
    try:
        broadcast = np.broadcast_to(arr, tail)
        return np.array(broadcast, copy=True), tail
    except ValueError:
        n_points = int(np.prod(tail, dtype=int)) if tail else 1
        if arr.size != n_points:
            raise ValueError("target 形状 {0} 无法广播到 {1}".format(arr.shape, tail))
        return arr, (n_points, 1)

def _column_interp(
    coord: ArrayLike,
    values: ArrayLike,
    target: ArrayLike,
    *,
    monotonic: bool = True,
    auto_flip: bool = True,
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    """逐廓线 PCHIP 垂直插值。

    记 P 为廓线数、T 为目标数，约定如下：

    * coord / values 的第一维都是层；廓线网格 tail = values.shape[1:]；
    * target 为标量时 T=1，所有廓线插值到同一坐标，输出形状为 tail；
    * target 形状为 (P,) 时 T=P，逐廓线配对，输出形状为 (P,)；
    * target 形状为 (T,) 且 T != P 时，对每个目标独立插值，输出形状为 tail + (T,)。

    坐标沿层递减（如气压）时自动翻转为递增；NaN 层被跳过；目标越界用端点值。
    复杂度：O(nlev x P x T)。
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim < 1:
        raise ValueError("values 至少一维")
    nlev = v.shape[0]
    tail = v.shape[1:]
    n_profiles = int(np.prod(tail, dtype=int)) if tail else 1
    v_flat = v.reshape(nlev, n_profiles)

    coord_arr = np.asarray(coord, dtype=np.float64)
    if coord_arr.ndim == 1:
        if coord_arr.size != nlev:
            raise ValueError("coord 长度 {0} 与层数 {1} 不一致".format(coord_arr.size, nlev))
        c_flat = np.broadcast_to(coord_arr.reshape(nlev, 1), (nlev, n_profiles))
    elif coord_arr.ndim == v.ndim and coord_arr.shape == v.shape:
        c_flat = coord_arr.reshape(nlev, n_profiles)
    else:
        raise ValueError("coord 形状 {0} 与 values 形状 {1} 不一致".format(coord_arr.shape, v.shape))

    target_arr = np.asarray(target, dtype=np.float64)
    if target_arr.ndim == 0:
        tf = target_arr.reshape(1).astype(np.float64)
        paired = True
        base_shape: tuple[int, ...] = ()
    elif target_arr.shape == tail:
        tf = target_arr.reshape(n_profiles)
        paired = True
        base_shape = tail
    elif target_arr.ndim == 1 and tail == () and target_arr.size != n_profiles:
        # 一维廓线 + 多个目标：对每个目标独立插值
        tf = target_arr
        paired = False
        base_shape = ()
    elif target_arr.ndim == 1 and target_arr.size == n_profiles:
        tf = target_arr
        paired = True
        base_shape = (n_profiles,)
    elif target_arr.ndim == 1:
        tf = target_arr
        paired = False
        base_shape = tail
    else:
        raise ValueError("target 形状 {0}（size {1}）无法匹配廓线网格 {2}".format(
            target_arr.shape, target_arr.size, tail))

    n_targets = int(tf.size)
    # 内部统一按 (n_profiles, n_targets)；配对模式只有对角元有效
    effective_targets = n_profiles if paired else n_targets
    out = np.full((n_profiles, effective_targets), fill_value, dtype=np.float64)
    if auto_flip:
        decreasing = c_flat[-1] < c_flat[0]
    else:
        decreasing = np.zeros(n_profiles, dtype=bool)
    for m in range(n_profiles):
        ycol = v_flat[:, m]
        zcol = c_flat[:, m]
        if decreasing[m]:
            # 递减坐标（气压）：先翻转为递增
            zcol = zcol[::-1]
            ycol = ycol[::-1]
        valid = np.isfinite(zcol) & np.isfinite(ycol)
        if valid.sum() < 2:
            continue
        zv = zcol[valid]
        yv = ycol[valid]
        order = np.argsort(zv, kind="stable")
        if not np.all(order == np.arange(zv.size)):
            zv, yv = zv[order], yv[order]
        if monotonic and np.any(np.diff(zv) <= 0.0):
            keep = np.concatenate(([True], np.diff(zv) > 0.0))
            zv, yv = zv[keep], yv[keep]
            if zv.size < 2:
                continue
        if paired:
            t = float(tf[m]) if n_targets > 1 else float(tf[0])
            if t < zv[0]:
                out[m, m] = yv[0]
            elif t > zv[-1]:
                out[m, m] = yv[-1]
            else:
                out[m, m] = float(pchip(zv, yv, t))
        else:
            inside = (tf >= zv[0]) & (tf <= zv[-1])
            piece = np.empty(n_targets, dtype=np.float64)
            piece[~inside] = np.where(tf[~inside] < zv[0], yv[0], yv[-1])
            if inside.any():
                piece[inside] = pchip(zv, yv, tf[inside])
            out[m, :] = piece
    if paired:
        diagonal = np.diagonal(out) if effective_targets == n_profiles else out[:, 0]
        return diagonal.reshape(base_shape if target_arr.ndim == 0 else target_arr.shape)
    return out.T.reshape(base_shape + (n_targets,))

def interp_to_height(
    z: ArrayLike,
    values: ArrayLike,
    z_target: ArrayLike,
    *,
    axis: int = 0,
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    """垂直插值到给定高度层（单调 PCHIP，避免过冲）。

    公式：在 :math:`z_k` 与 :math:`z_{k+1}` 之间用 :func:`pchip` 的分段三次
    Hermite 多项式求值（[B9] 第 3 章）。相比线性插值，PCHIP 在层结急剧变化处
    既不引入虚假极值，又保持 :math:`C^1` 连续。

    参数
    ----
    z : array_like, shape (nlev,) 或 (nlev, ...)
        层中心高度 [m]，沿 `axis` 单调（递增或递减均可）。
    values : array_like, shape (nlev, ...)
        被插值变量；`NaN` 层会被跳过（不做外推填充）。
    z_target : array_like
        目标高度；标量或形状与 `values.shape[1:]` 一致。
    axis : int
        层所在轴，默认 0。非 0 时内部转置到第一维再转回。

    复杂度：O(nlev × M)；内存 O(M)。
    """
    v = np.asarray(values, dtype=np.float64)
    zc = np.asarray(z, dtype=np.float64)
    if axis != 0:
        v = np.moveaxis(v, axis, 0)
        zc = np.moveaxis(zc, axis, 0) if zc.ndim > 1 else zc
    result = _column_interp(zc, v, z_target, fill_value=fill_value)
    return result


def interp_to_pressure(
    p: ArrayLike,
    values: ArrayLike,
    p_target: ArrayLike,
    *,
    axis: int = 0,
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    """垂直插值到给定气压层。

    与 :func:`interp_to_height` 相同，但坐标是气压（通常随高度**递减**）。
    实现上先按层翻转为递增坐标，再做 PCHIP；这等价于以 :math:`-\\ln p` 为
    自变量的单调插值（[D16] 第 3 章）。

    复杂度：O(nlev × M)。
    """
    v = np.asarray(values, dtype=np.float64)
    pc = np.asarray(p, dtype=np.float64)
    if axis != 0:
        v = np.moveaxis(v, axis, 0)
        pc = np.moveaxis(pc, axis, 0) if pc.ndim > 1 else pc
    return _column_interp(pc, v, p_target, monotonic=True, auto_flip=True,
                          fill_value=fill_value)


def column_interp_to_height(
    height: ArrayLike,
    values: ArrayLike,
    z_target: float | ArrayLike,
    *,
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    """把三维场 `(nz, ny, nx)` 逐列插值到高度 `z_target`。复杂度 O(nz nx ny)。"""
    h3 = np.asarray(height, dtype=np.float64)
    v3 = np.asarray(values, dtype=np.float64)
    if h3.shape != v3.shape:
        raise ValueError("height 与 values 形状不一致：{0} vs {1}".format(h3.shape, v3.shape))
    if np.isscalar(z_target) or np.asarray(z_target).ndim == 0:
        target = np.full(h3.shape[1:], float(z_target))
    else:
        target = np.asarray(z_target, dtype=np.float64)
        if target.shape != h3.shape[1:]:
            target = np.broadcast_to(target, h3.shape[1:])
    return _column_interp(h3, v3, target, fill_value=fill_value)


def column_interp_to_pressure(
    pressure: ArrayLike,
    values: ArrayLike,
    p_target: float | ArrayLike,
    *,
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    """把三维场 `(nz, ny, nx)` 逐列插值到气压 `p_target`。复杂度 O(nz nx ny)。"""
    p3 = np.asarray(pressure, dtype=np.float64)
    v3 = np.asarray(values, dtype=np.float64)
    if p3.shape != v3.shape:
        raise ValueError("pressure 与 values 形状不一致：{0} vs {1}".format(p3.shape, v3.shape))
    if np.isscalar(p_target) or np.asarray(p_target).ndim == 0:
        target = np.full(p3.shape[1:], float(p_target))
    else:
        target = np.asarray(p_target, dtype=np.float64)
        if target.shape != p3.shape[1:]:
            target = np.broadcast_to(target, p3.shape[1:])
    return _column_interp(p3, v3, target, fill_value=fill_value)


# ---------------------------------------------------------------------------
# 守恒粗化 / 延拓
# ---------------------------------------------------------------------------


def conservative_coarsen(
    field: ArrayLike,
    ratio: int,
    *,
    axes: Sequence[int] = (-2, -1),
    nan_policy: str = "skip",
) -> NDArray[np.float64]:
    """保守面积加权粗化（fine -> coarse），对应 C++ `grid::conservative_restrict`。

    公式（[N5] 守恒重映射；[D13] 有限体积）
    -------------------------------------
    均匀网格上守恒粗化即 r x r 分块均值：

    .. math::

        \\bar{f}_{I,J} = \\frac{1}{r^2} \\sum_{p=0}^{r-1} \\sum_{q=0}^{r-1}
        f_{rI+p,\\, rJ+q}

    与 C++ 的权重 1/r^2 完全一致（architecture 第 5.4 节守恒约束）。
    逐偏移累加实现，天然支持任意轴组合（含 3-D 场的垂直粗化）。

    nan_policy：'skip' 忽略 NaN 并按有效点数重归一化；'propagate' 有 NaN 即 NaN。
    复杂度：O(N)；内存 O(N)。
    """
    arr = np.asarray(field, dtype=np.float64)
    r = int(ratio)
    if r < 1:
        raise ValueError("ratio 必须 >= 1")
    if nan_policy not in ("skip", "propagate"):
        raise ValueError("nan_policy 只能是 'skip' 或 'propagate'")
    normalized = tuple(ax % arr.ndim for ax in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError("axes 不能重复")
    # 把待粗化的轴移到末尾（本函数内部约定），输出时再还原
    others = [ax for ax in range(arr.ndim) if ax not in normalized]
    perm = others + list(normalized)
    moved = np.transpose(arr, perm)
    block_shape = moved.shape[:arr.ndim - len(normalized)]
    reducible = moved.shape[arr.ndim - len(normalized):]
    for size in reducible:
        if size % r != 0:
            raise ValueError("被粗化轴长度 {0} 不能被 ratio={1} 整除".format(size, r))
    out_dims = tuple(size // r for size in reducible)
    total = np.zeros(block_shape + out_dims, dtype=np.float64)
    valid = np.zeros(block_shape + out_dims, dtype=np.float64)
    index = tuple(slice(None) for _ in block_shape)
    for offset in np.ndindex(*([r] * len(reducible))):
        selector = index + tuple(
            slice(off, size - (r - 1) + off, r) for off, size in zip(offset, reducible))
        block = moved[selector]
        finite = np.isfinite(block)
        total += np.where(finite, block, 0.0)
        valid += finite.astype(np.float64)
    if nan_policy == "propagate":
        count = float(r ** len(reducible))
        result = total / count
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(valid > 0.0, total / np.maximum(valid, 1.0), np.nan)
    # 还原到原轴顺序
    result_perm = others + [ax for ax in range(arr.ndim) if ax not in others]
    return np.transpose(result, np.argsort(result_perm))

def conservative_coarsen_area(
    field: ArrayLike,
    area: ArrayLike,
    ratio: int,
    *,
    axes: Sequence[int] = (-2, -1),
) -> NDArray[np.float64]:
    """面积加权的守恒粗化（变分辨率网格）。

    公式（[N5]）
    -----------
    .. math::

        \\bar{f}_c = \\frac{\\sum_{f \\in c} f_f A_f}{\\sum_{f \\in c} A_f}

    即"面元加权平均"，当 :math:`A_f` 为常数时退化为 :func:`conservative_coarsen`。
    复杂度：O(N)。
    """
    arr = np.asarray(field, dtype=np.float64)
    weight = np.asarray(area, dtype=np.float64)
    if weight.shape != arr.shape:
        try:
            weight = np.broadcast_to(weight, arr.shape)
        except ValueError as exc:
            raise ValueError("area 形状 {0} 无法广播到 field 形状 {1}".format(
                weight.shape, arr.shape)) from exc
    r = int(ratio)
    normalized = tuple(ax % arr.ndim for ax in axes)
    others = [ax for ax in range(arr.ndim) if ax not in normalized]
    perm = others + list(normalized)
    moved = np.transpose(arr, perm)
    wmoved = np.transpose(weight, perm)
    split_shape = moved.shape[:len(others)] + tuple(
        part for dim in moved.shape[len(others):] for part in (dim // r, r))
    reduce_axes = tuple(range(len(others), len(split_shape), 2))
    numerator = moved.reshape(split_shape).sum(axis=reduce_axes)
    denominator = wmoved.reshape(split_shape).sum(axis=reduce_axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denominator != 0.0, numerator / denominator, np.nan)
    order = others + [ax for ax in range(arr.ndim) if ax not in others]
    return np.transpose(result, np.argsort(order))


def bilinear_prolong(
    field: ArrayLike,
    ratio: int,
    *,
    enforce_positive: bool = False,
) -> NDArray[np.float64]:
    """水平双线性延拓（coarse -> fine），对应 C++ `grid::bilinear_prolong`。

    公式（[N1] Davies 1976 的插值部分；[D16] 第 3 章）
    ------------------------------------------------
    细化比 :math:`r` 时，细网格体心 :math:`x^f_{i} = x^c_{I} + (\\alpha + 1/2) \\Delta x^c/r - \\Delta x^c/(2r)`，
    等价于在粗网格索引坐标上做双线性：

    .. math::

        i_c = \\frac{i + 1/2}{r} - \\frac12,

    然后复用 :func:`bilinear` 的公式。边界外以零梯度外推（保持粗网格边界值），
    符合 Davies 松弛区外侧"由父域提供边界"的语义。

    `enforce_positive=True` 时把结果裁剪到 :math:`\\ge 0`（水物质保正）。

    复杂度：O(r^2 N_c)；内存 O(r^2 N_c)。
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("bilinear_prolong 需要至少二维场")
    r = int(ratio)
    if r < 1:
        raise ValueError("ratio 必须 >= 1")
    ny_c, nx_c = arr.shape[-2], arr.shape[-1]
    ny_f, nx_f = ny_c * r, nx_c * r
    # 细网格体心 -> 粗网格连续索引
    i_f = (np.arange(nx_f) + 0.5) / r - 0.5
    j_f = (np.arange(ny_f) + 0.5) / r - 0.5
    i0 = np.clip(np.floor(i_f).astype(np.intp), 0, nx_c - 1)
    j0 = np.clip(np.floor(j_f).astype(np.intp), 0, ny_c - 1)
    i1 = np.clip(i0 + 1, 0, nx_c - 1)
    j1 = np.clip(j0 + 1, 0, ny_c - 1)
    a = np.clip(i_f - i0, 0.0, 1.0)
    b = np.clip(j_f - j0, 0.0, 1.0)
    # 逐粗单元求和（分离变量形式），避免构造 4 个完整临时场
    f00 = arr[..., j0[:, None], i0[None, :]]
    f01 = arr[..., j0[:, None], i1[None, :]]
    f10 = arr[..., j1[:, None], i0[None, :]]
    f11 = arr[..., j1[:, None], i1[None, :]]
    wa = a[None, :]
    wb = b[:, None]
    result = ((1.0 - wa) * (1.0 - wb) * f00 + wa * (1.0 - wb) * f01
              + (1.0 - wa) * wb * f10 + wa * wb * f11)
    if enforce_positive:
        result = np.maximum(result, 0.0)
    return result


def extrapolate_profile(
    coord: ArrayLike,
    values: ArrayLike,
    *,
    fill: str = "nearest",
    axis: int = 0,
) -> NDArray[np.float64]:
    """用单调外推填充剖面中的 NaN（垂直外推 / 观测廓线填补）。

    策略（[B9] 第 3 章）
    ------------------
    * `"nearest"`：用最近的有效层值（零梯度）；
    * `"linear"`：用端点两个有效层做线性外推；
    * `"log"`：对正定量在对数空间线性外推（水汽、气压常用）。

    仅填补**首尾**的 NaN；序列中间的空洞保持 NaN（避免伪造垂直结构）。
    复杂度：O(N)。
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 0 or v.shape[0] == 0:
        return v.copy()
    if axis != 0:
        v = np.moveaxis(v, axis, 0)
    out = v.copy()
    nlev = out.shape[0]
    flat = out.reshape(nlev, -1)
    coord_arr = np.asarray(coord, dtype=np.float64)
    coord_flat = (np.broadcast_to(coord_arr.reshape(nlev, 1), flat.shape)
                  if coord_arr.ndim == 1 else coord_arr.reshape(nlev, -1))
    for m in range(flat.shape[1]):
        column = flat[:, m]
        valid = np.isfinite(column)
        if valid.sum() == 0:
            continue
        first = int(np.argmax(valid))
        last = int(nlev - np.argmax(valid[::-1]) - 1)
        if first > 0:
            column[:first] = _extrapolate_plain(
                coord_flat[first, m], coord_flat[first + 1, m] if first + 1 <= last else None,
                column[first], column[first + 1] if first + 1 <= last else None,
                coord_flat[:first, m], fill, forward=True)
        if last < nlev - 1:
            column[last + 1:] = _extrapolate_plain(
                coord_flat[last - 1, m] if last - 1 >= first else None, coord_flat[last, m],
                column[last - 1] if last - 1 >= first else None, column[last],
                coord_flat[last + 1:, m], fill, forward=False)
    if axis != 0:
        out = np.moveaxis(out, axis, 0)
    return out


def _extrapolate_plain(z0: float | None, z1: float | None, v0: float | None, v1: float | None,
                       targets: NDArray[np.float64], fill: str,
                       *, forward: bool) -> NDArray[np.float64]:
    """执行一次端点外推。复杂度 O(目标点数)。"""
    if fill == "nearest" or z0 is None or z1 is None or v0 is None or v1 is None:
        return np.full(targets.shape, float(v0 if forward else v1))
    if fill == "log":
        if v0 <= 0.0 or v1 <= 0.0:
            return np.full(targets.shape, float(v0 if forward else v1))
        slope = (np.log(v1) - np.log(v0)) / (z1 - z0)
        return np.exp(np.log(v0) + slope * (targets - z0))
    if fill == "linear":
        slope = (v1 - v0) / (z1 - z0) if z1 != z0 else 0.0
        return v0 + slope * (targets - z0)
    raise ValueError("fill 只能是 'nearest' / 'linear' / 'log'")


#: 方法名 -> 实现，供 CLI / 配置选择
REGISTRY: Mapping[str, Callable[..., Any]] = {
    "bilinear": bilinear,
    "trilinear": trilinear,
    "pchip": pchip,
    "pchip_slopes": pchip_slopes,
    "interp_to_height": interp_to_height,
    "interp_to_pressure": interp_to_pressure,
    "column_interp_to_height": column_interp_to_height,
    "column_interp_to_pressure": column_interp_to_pressure,
    "conservative_coarsen": conservative_coarsen,
    "bilinear_prolong": bilinear_prolong,
    "extrapolate_profile": extrapolate_profile,
}

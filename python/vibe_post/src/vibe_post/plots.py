"""绘图：matplotlib 填色图、剖面、简化 skew-T、检验图与动能谱。

对应 docs/design/00_architecture.md 第 12 节"绘图依赖 matplotlib 为可选 extras"。
本模块**所有** matplotlib 导入都放在函数内部，缺失时抛出带安装提示的
:class:`vibe_post.io.MissingDependencyError`（而不是裸 ImportError），
因此纯 numpy 后端在无绘图依赖的环境下仍可工作。

绘图约定
--------
* 返回 `(fig, ax)`（单面板）或 `(fig, axes)`（多面板），由调用方决定 `savefig`；
* 所有函数都接受"原始数组"或 `:class:`vibe_post.io.Dataset` + 变量名；
* 单位在轴标签中显式标注（与 :mod:`vibe_post.diagnostics` 的 SI 约定一致）；
* 色标一律用感知均匀的 `viridis`/`magma`/`RdBu_r`（[B9] 第 3 章的绘图建议）。

文献：[B9] Wilks (2019) 第 3 章（资料展示与谱分析）；[E7] Roberts & Lean (2008)；
[E9] Hamill (2001)；[B4] Holton & Hakim (2013) 第 9 章（skew-T 热力学图）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "get_mpl",
    "plot_field",
    "plot_vertical_section",
    "plot_skewt",
    "plot_profile",
    "plot_timeseries",
    "plot_reliability",
    "plot_rank_histogram",
    "plot_fss_vs_scale",
    "plot_spectra",
    "kinetic_energy_spectrum",
]


def get_mpl() -> tuple[Any, Any]:
    """惰性导入 matplotlib，返回 `(pyplot, matplotlib)`。

    缺失时抛 :class:`vibe_post.io.MissingDependencyError`。复杂度 O(1)。
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 依赖缺失路径
        from .io import MissingDependencyError
        raise MissingDependencyError("matplotlib", "绘图", "plots") from exc
    return plt, matplotlib


def _resolve(data: Any, var: str | None = None) -> NDArray[np.float64]:
    """从数组或 Dataset 中取出场数据。复杂度 O(1)。"""
    from .io import Dataset

    if isinstance(data, Dataset):
        if var is None:
            raise ValueError("传入 Dataset 时必须给出 var 名称")
        if var not in data.fields:
            raise KeyError("Dataset 中没有变量 {0!r}；可用：{1}".format(
                var, sorted(data.fields)))
        return np.asarray(data.fields[var], dtype=np.float64)
    arr = np.asarray(data, dtype=np.float64)
    return arr


def _coords(data: Any, nx: int, ny: int, kind: str = "xy",
            dx: float = 1.0, dy: float = 1.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """从 Dataset 的 coords 或均匀间距生成坐标（单位 km）。复杂度 O(n)。"""
    from .io import Dataset

    coords = getattr(data, "coords", {}) if isinstance(data, Dataset) else {}
    x = np.asarray(coords.get("x", (np.arange(nx) + 0.5) * dx), dtype=np.float64) / 1000.0
    if kind == "xy":
        y = np.asarray(coords.get("y", (np.arange(ny) + 0.5) * dy), dtype=np.float64) / 1000.0
    else:
        y = np.asarray(coords.get("z", coords.get("zeta", np.arange(ny) * dy)), dtype=np.float64)
    return x, y


def plot_field(field: Any, var: str | None = None, *, level: int = 0,
               u: Any | None = None, v: Any | None = None,
               vector_stride: int = 4, cmap: str = "viridis",
               title: str | None = None, units: str = "", ax: Any | None = None,
               colorbar: bool = True, **kwargs: Any) -> tuple[Any, Any]:
    """等经纬/平面填色图（可选叠加风矢量）。

    参数
    ----
    field : array_like 或 Dataset
        三维 `(nz, ny, nx)` 或二维 `(ny, nx)`；`level` 只在三维时使用。
    u, v : 可选，同形状的风分量；给定时按 `vector_stride` 抽稀画矢量。
    level : int
        垂直层号（用于 `pcolormesh` 的着色体）。

    返回值：`(fig, ax)`。复杂度：O(N)。
    """
    plt, _ = get_mpl()
    arr = _resolve(field, var)
    if arr.ndim == 3:
        if not (0 <= level < arr.shape[0]):
            raise IndexError("level={0} 超出层数 {1}".format(level, arr.shape[0]))
        plane = arr[level]
    elif arr.ndim == 2:
        plane = arr
    else:
        raise ValueError("plot_field 需要二维或三维场，得到 {0} 维".format(arr.ndim))
    ny, nx = plane.shape
    x, y = _coords(field, nx, ny, "xy")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(x, y, plane, cmap=cmap, shading="auto", **kwargs)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(units or "值")
    if u is not None and v is not None:
        u_arr = _resolve(u) if not isinstance(u, str) else _resolve(field, u)
        v_arr = _resolve(v) if not isinstance(v, str) else _resolve(field, v)
        if u_arr.ndim == 3:
            u_arr = u_arr[min(level, u_arr.shape[0] - 1)]
        if v_arr.ndim == 3:
            v_arr = v_arr[min(level, v_arr.shape[0] - 1)]
        step = max(int(vector_stride), 1)
        yy, xx = np.meshgrid(y[::step], x[::step], indexing="ij")
        ax.quiver(xx, yy, u_arr[::step, ::step], v_arr[::step, ::step],
                  color="k", pivot="middle", width=0.0025)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or ("填色图" + (" — {0}".format(var) if var else "")))
    return fig, ax


def plot_vertical_section(field: Any, var: str | None = None, *, j: int | None = None,
                          i: int | None = None, z: ArrayLike | None = None,
                          cmap: str = "magma", title: str | None = None,
                          units: str = "", ax: Any | None = None,
                          colorbar: bool = True, **kwargs: Any) -> tuple[Any, Any]:
    """垂直剖面图（x-z 或 y-z）。

    参数
    ----
    field : 三维 `(nz, ny, nx)`
    j : 固定 y 索引，得到 x-z 剖面（沿 x 方向）
    i : 固定 x 索引，得到 y-z 剖面
    z : 高度 `(nz, ny, nx)` 或 `(nz,)`；None 时用层号。

    返回值 `(fig, ax)`。复杂度 O(N)。
    """
    plt, _ = get_mpl()
    arr = _resolve(field, var)
    if arr.ndim != 3:
        raise ValueError("plot_vertical_section 需要三维场 (nz, ny, nx)")
    nz, ny, nx = arr.shape
    if (j is None) == (i is None):
        raise ValueError("请且仅请给出 j 或 i 之一")
    if j is not None:
        if not (0 <= j < ny):
            raise IndexError("j={0} 超出 ny={1}".format(j, ny))
        section = arr[:, j, :]
        horizontal = np.arange(nx) + 0.5
        xlabel = "x [km]"
    else:
        if not (0 <= i < nx):
            raise IndexError("i={0} 超出 nx={1}".format(i, nx))
        section = arr[:, :, i]
        horizontal = np.arange(ny) + 0.5
        xlabel = "y [km]"
    if z is not None:
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 3:
            levels = z_arr[:, j, :] if j is not None else z_arr[:, :, i]
        elif z_arr.ndim == 1 and z_arr.size == nz:
            levels = np.repeat(z_arr[:, None], horizontal.size, axis=1)
        else:
            raise ValueError("z 形状 {0} 与剖面不匹配".format(z_arr.shape))
        ylabel = "高度 [m]"
    else:
        levels = np.repeat(np.arange(nz)[:, None], horizontal.size, axis=1)
        ylabel = "层号"
    from .io import Dataset
    coords = getattr(field, "coords", {}) if isinstance(field, Dataset) else {}
    dx = float(getattr(field, "attrs", {}).get("dx", 1000.0)) if isinstance(field, Dataset) else 1000.0
    if j is not None and "x" in coords:
        horizontal = np.asarray(coords["x"], dtype=np.float64) / 1000.0
    elif i is not None and "y" in coords:
        horizontal = np.asarray(coords["y"], dtype=np.float64) / 1000.0
    else:
        horizontal = horizontal * dx / 1000.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(horizontal, levels, section, cmap=cmap, shading="auto", **kwargs)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(units or "值")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or ("垂直剖面" + (" — {0}".format(var) if var else "")))
    return fig, ax


def plot_skewt(p: ArrayLike, t: ArrayLike, td: ArrayLike | None = None, *,
               z: ArrayLike | None = None, ax: Any | None = None,
               p_bottom: float = 105000.0, p_top: float = 20000.0,
               skew: float = 0.9, **kwargs: Any) -> tuple[Any, Any]:
    """简化 skew-T 对数气压图（自绘，不依赖 MetPy）。

    坐标变换（[B4] 第 9 章）
    ----------------------
    * 纵轴：:math:`\\ln p`（气压从下到上递减，实现取 :math:`-\\ln p`）；
    * 横轴：skew 变换 :math:`x = T + s\\,(\\ln p_0 - \\ln p)`，
      其中 :math:`s` 为 `skew` 因子（单位 K/ln(hPa)）。

    图上叠加：
    1. 干绝热线（等位温，:math:`T=\\theta(p/p_0)^\\kappa`）；
    2. 湿绝热线（用 :func:`vibe_post.diagnostics._moist_temperature` 反解）；
    3. 饱和混合比线（:math:`q_{v,sat}(T,p)` 等值线）；
    4. 温度廓线与露点廓线。

    返回值 `(fig, ax)`。复杂度：O(nz × n_line)。
    """
    plt, _ = get_mpl()
    from . import diagnostics as dg

    p_arr = np.asarray(p, dtype=np.float64).ravel()
    order = np.argsort(-p_arr)
    p_arr = p_arr[order]
    t_arr = np.asarray(t, dtype=np.float64).ravel()[order]
    ax_created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 8.0), constrained_layout=True)
    else:
        fig = ax.figure

    def y_of(pressure: NDArray[np.float64]) -> NDArray[np.float64]:
        return -np.log(np.maximum(pressure, 1.0) / 100.0)

    def x_of(temp: NDArray[np.float64], pressure: NDArray[np.float64]) -> NDArray[np.float64]:
        return temp + skew * y_of(pressure)

    p_grid = np.logspace(np.log10(p_bottom), np.log10(p_top), 120)
    t_grid = np.linspace(180.0, 340.0, 160)
    # 干绝热线
    for theta in range(240, 480, 20):
        t_line = theta * (p_grid / 100000.0) ** dg.KAPPA
        ax.plot(x_of(t_line, p_grid), y_of(p_grid), color="tab:orange", lw=0.6, alpha=0.7)
    # 湿绝热线：theta_e = theta，从底部起算
    for theta_e_value in range(240, 380, 20):
        target = np.full_like(p_grid, float(theta_e_value))
        guess = theta_e_value * (p_grid / 100000.0) ** dg.KAPPA
        t_moist = dg._moist_temperature(p_grid, target, guess, n_iter=30)
        ax.plot(x_of(t_moist, p_grid), y_of(p_grid), color="tab:green", lw=0.6, alpha=0.6)
    # 饱和混合比线
    for q_value in (0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.020):
        e_target = q_value * p_grid / (dg.EPSILON + q_value)
        t_sat = dg.T0 + (243.5 * np.log(np.maximum(e_target, 1.0e-6) / 611.2)) / (
            17.67 - np.log(np.maximum(e_target, 1.0e-6) / 611.2))
        ax.plot(x_of(t_sat, p_grid), y_of(p_grid), color="tab:purple", lw=0.5, alpha=0.5)
    # 温度与露点廓线
    ax.plot(x_of(t_arr, p_arr), y_of(p_arr), color="tab:red", lw=2.0, label="T")
    if td is not None:
        td_arr = np.asarray(td, dtype=np.float64).ravel()[order]
        ax.plot(x_of(td_arr, p_arr), y_of(p_arr), color="tab:blue", lw=2.0, label="Td")
    if z is not None:
        z_arr = np.asarray(z, dtype=np.float64).ravel()[order]
        for height in (1000.0, 3000.0, 6000.0, 9000.0):
            index = int(np.argmin(np.abs(z_arr - height)))
            ax.axhline(y_of(p_arr[index]), color="gray", lw=0.5, ls=":")
            ax.text(x_of(np.array([t_grid[0]]), np.array([p_arr[index]]))[0],
                    y_of(p_arr[index]), " {0:.0f} m".format(height),
                    va="bottom", ha="left", fontsize=7, color="gray")
    ax.set_yscale("linear")
    p_ticks = np.array([100000.0, 85000.0, 70000.0, 50000.0, 30000.0, 20000.0])
    p_ticks = p_ticks[(p_ticks <= p_bottom) & (p_ticks >= p_top)]
    ax.set_yticks(y_of(p_ticks))
    ax.set_yticklabels(["{0:.0f}".format(v / 100.0) for v in p_ticks])
    ax.set_ylabel("气压 [hPa]")
    ax.set_xlabel("温度 [K]（skew 因子 {0}）".format(skew))
    ax.set_title("skew-T / log-p 简图")
    ax.grid(True, ls=":", lw=0.4, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    return fig, ax


def plot_profile(values: ArrayLike, coord: ArrayLike, *, invert: bool = True,
                 xlabel: str = "值", ylabel: str = "气压 [hPa]",
                 ax: Any | None = None, label: str | None = None,
                 marker: str = "-", **kwargs: Any) -> tuple[Any, Any]:
    """垂直廓线图（单条或多条）。

    `values` 形状 `(nz,)` 或 `(n, nz)`；后者按 n 条曲线绘制。
    `invert=True` 时纵轴翻转（气压自下而上递减）。返回值 `(fig, ax)`。
    复杂度：O(N)。
    """
    plt, _ = get_mpl()
    vals = np.asarray(values, dtype=np.float64)
    coord_arr = np.asarray(coord, dtype=np.float64).ravel()
    if vals.ndim == 1:
        vals = vals[None, :]
    if vals.shape[-1] != coord_arr.size:
        raise ValueError("values 末维 {0} 与 coord 长度 {1} 不一致".format(
            vals.shape[-1], coord_arr.size))
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 6.4), constrained_layout=True)
    else:
        fig = ax.figure
    for row in vals:
        ax.plot(row, coord_arr, marker, label=label, **kwargs)
    if invert:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label is not None:
        ax.legend(fontsize=8)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    return fig, ax


def plot_timeseries(time: ArrayLike, series: ArrayLike | Mapping[str, ArrayLike], *,
                    obs: Mapping[str, ArrayLike] | None = None,
                    xlabel: str = "时间 [h]", ylabel: str = "值",
                    ax: Any | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """时间序列图（可叠加观测）。

    `series` 为 `(nt,)` 或 `{"名称": (nt,)}`；`obs` 为同形映射（虚线）。
    返回值 `(fig, ax)`。复杂度：O(N)。
    """
    plt, _ = get_mpl()
    t = np.asarray(time, dtype=np.float64).ravel()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    else:
        fig = ax.figure
    if isinstance(series, Mapping):
        for name, values in series.items():
            ax.plot(t, np.asarray(values, dtype=np.float64).ravel(), label=str(name), **kwargs)
    else:
        arr = np.asarray(series, dtype=np.float64)
        if arr.ndim == 1:
            ax.plot(t, arr, label="forecast", **kwargs)
        else:
            for k, row in enumerate(arr):
                ax.plot(t, row, label="member {0}".format(k), **kwargs)
    if obs:
        for name, values in obs.items():
            ax.plot(t, np.asarray(values, dtype=np.float64).ravel(), ls="--",
                    color="k", label="{0} (obs)".format(name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    return fig, ax


def plot_reliability(curve: Mapping[str, Any] | None = None, *, p: ArrayLike | None = None,
                     o: ArrayLike | None = None, n_bins: int = 10,
                     ax: Any | None = None, label: str | None = None,
                     **kwargs: Any) -> tuple[Any, Any]:
    """可靠性图（预报概率对观测频率，含 1:1 参考线与样本直方数）。

    `curve` 可直接用 :func:`vibe_post.verify.reliability_curve` 的返回值；
    也可以给 `p`/`o` 由本函数计算。返回值 `(fig, ax)`。复杂度 O(N + n_bins)。
    """
    plt, _ = get_mpl()
    if curve is None:
        from .verify import reliability_curve
        if p is None or o is None:
            raise ValueError("需要 curve，或同时给出 p 与 o")
        curve = reliability_curve(p, o, n_bins=n_bins)
    if ax is None:
        fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(5.4, 6.4),
                                          gridspec_kw={"height_ratios": [3, 1]},
                                          constrained_layout=True, sharex=True)
    else:
        fig = ax.figure
        ax_hist = ax
    p_mean = np.asarray(curve["p_mean"], dtype=np.float64)
    reliability = np.asarray(curve["reliability"], dtype=np.float64)
    counts = np.asarray(curve["counts"], dtype=np.float64)
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8, label="完美校准")
    ax.plot(p_mean, reliability, marker="o", label=label or "预报", **kwargs)
    ax.set_ylabel("观测频率")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    centres = 0.5 * (np.asarray(curve["bins"])[1:] + np.asarray(curve["bins"])[:-1])
    width = float(np.diff(np.asarray(curve["bins"])).mean() * 0.9)
    ax_hist.bar(centres, counts, width=width, color="tab:blue", alpha=0.7)
    ax_hist.set_xlabel("预报概率")
    ax_hist.set_ylabel("样本数")
    ax_hist.grid(True, ls=":", lw=0.4, alpha=0.5)
    return fig, ax


def plot_rank_histogram(ensemble: ArrayLike | None = None,
                        obs: ArrayLike | None = None, *,
                        histogram: Mapping[str, Any] | None = None,
                        ax: Any | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """Talagrand 排序直方图（含平坦参考线）。

    `histogram` 可直接用 :func:`vibe_post.verify.rank_histogram` 的返回值；
    也可给 `ensemble`/`obs` 由本函数计算。返回值 `(fig, ax)`。
    复杂度：O(M log M × N)。
    """
    plt, _ = get_mpl()
    if histogram is None:
        from .verify import rank_histogram
        if ensemble is None or obs is None:
            raise ValueError("需要 histogram，或同时给出 ensemble 与 obs")
        histogram = rank_histogram(ensemble, obs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    else:
        fig = ax.figure
    rank = np.asarray(histogram["rank"], dtype=np.float64)
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    expected = np.asarray(histogram["expected"], dtype=np.float64)
    ax.bar(rank, counts, color="tab:blue", alpha=0.75, **kwargs)
    ax.plot(rank, expected, color="k", ls="--", lw=0.9, label="均匀期望")
    ax.set_xlabel("集合排序秩")
    ax.set_ylabel("频数")
    ax.set_title("Talagrand 排序直方图 (E9)")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5, axis="y")
    return fig, ax


def plot_fss_vs_scale(curve: Mapping[str, Any] | None = None, *,
                      f: ArrayLike | None = None, o: ArrayLike | None = None,
                      threshold: float = 0.5,
                      radii: Sequence[int] = (0, 1, 2, 4, 8, 16),
                      ax: Any | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """FSS 随邻域尺度变化的曲线（含"有用技巧"阈值线）。

    `curve` 直接用 :func:`vibe_post.verify.fss_vs_scale` 的返回值。
    返回值 `(fig, ax)`。复杂度：O(len(radii) × N)。
    """
    plt, _ = get_mpl()
    if curve is None:
        from .verify import fss_vs_scale
        if f is None or o is None:
            raise ValueError("需要 curve，或同时给出 f 与 o")
        curve = fss_vs_scale(f, o, threshold, radii)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(curve["window"], curve["fss"], marker="o", color="tab:blue",
            label="FSS", **kwargs)
    useful = float(np.asarray(curve["fss_useful"]))
    ax.axhline(useful, color="tab:red", ls="--", lw=0.9,
               label="有用技巧阈值 = 0.5 + f_o/2 = {0:.3f}".format(useful))
    first = float(np.asarray(curve["first_useful_radius"]))
    if np.isfinite(first):
        ax.axvline(2 * first + 1, color="gray", ls=":", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("邻域窗口 w（格点数，w = 2r+1）")
    ax.set_ylabel("FSS")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("邻域分数技巧评分 (E7, E14)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    return fig, ax


def kinetic_energy_spectrum(field: ArrayLike, dx: float = 1.0, *,
                            axis: int = -1, remove_mean: bool = True,
                            window: str = "hann") -> dict[str, NDArray[np.float64]]:
    """一维动能/方差谱（Welch 式单段周期图）。

    公式（[B9] 第 3 章）
    ------------------
    对沿 `axis` 的每条剖面 :math:`f(n)`（长度 N）做窗函数加权的 FFT：

    .. math::

        P(k) = \\frac{|\\hat f_k|^2}{N\\,W}\\sum_n w_n^2, \\qquad
        k = 0,\\dots,N/2

    其中 :math:`\\hat f_k = \\sum_n w_n (f_n-\\bar f)e^{-i2\\pi kn/N}`。
    对水平两个方向平均后再按波数取平均，得到"谱密度 -> 波长"关系；
    谱斜率（对数坐标下的最小二乘拟合）用于检验 -5/3 惯性子区（[B9]）。

    返回 `{"wavenumber", "wavelength", "spectrum", "slope", "fit_wavelength"}`。
    复杂度：O(N log N)。
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if remove_mean:
        arr = arr - np.nanmean(arr, axis=axis, keepdims=True)
    n = arr.shape[axis]
    if n < 4:
        raise ValueError("谱分析至少需要 4 个点，得到 {0}".format(n))
    if window == "hann":
        w = np.hanning(n)
    elif window == "none":
        w = np.ones(n)
    else:
        raise ValueError("window 只能是 'hann' 或 'none'")
    shape = [1] * arr.ndim
    shape[axis] = n
    weighted = np.nan_to_num(arr, nan=0.0) * w.reshape(shape)
    spectrum_complex = np.fft.rfft(weighted, axis=axis)
    power = (np.abs(spectrum_complex) ** 2) / (n * float(np.sum(w ** 2)))
    k_index = np.arange(power.shape[axis])
    power = np.take(power, k_index, axis=axis)
    flat_axes = tuple(a for a in range(power.ndim) if a != (axis % power.ndim))
    averaged = power.mean(axis=flat_axes)
    wavenumber = k_index / (n * dx)
    valid = k_index > 0
    wavelength = np.full(k_index.shape, np.inf)
    wavelength[valid] = 1.0 / wavenumber[valid]
    spectrum = averaged
    # 谱斜率：在中间 1/4 ~ 3/4 波段做 log-log 最小二乘
    lo = max(int(n * 0.15), 1)
    hi = max(int(n * 0.5), lo + 2)
    lo = min(lo, k_index.size - 1)
    hi = min(hi, k_index.size)
    seg = slice(lo, hi)
    if hi - lo >= 3:
        xfit = np.log(wavenumber[seg])
        yfit = np.log(np.maximum(spectrum[seg], 1.0e-300))
        slope = float(np.polyfit(xfit, yfit, 1)[0])
    else:
        slope = float("nan")
    return {"wavenumber": wavenumber, "wavelength": wavelength, "spectrum": spectrum,
            "slope": np.asarray(slope),
            "fit_wavelength": wavelength[seg] if hi - lo >= 1 else np.zeros(0)}


def plot_spectra(fields: Any, var: str | Sequence[str] | None = None, *,
                 dx: float = 1.0, ax: Any | None = None,
                 labels: Sequence[str] | None = None,
                 reference_slope: float = -5.0 / 3.0,
                 **kwargs: Any) -> tuple[Any, Any]:
    """动能（方差）谱图：横轴波长、纵轴谱密度，附 :math:`-5/3` 参考线。

    `fields` 可以是单个数组、`(n, ...)` 数组序列，或 `{名称: 数组}` 映射。
    返回值 `(fig, ax)`。复杂度：O(n × N log N)。
    """
    plt, _ = get_mpl()
    if isinstance(fields, Mapping):
        items = list(fields.items())
    elif isinstance(fields, (list, tuple)):
        items = [("field {0}".format(k), item) for k, item in enumerate(fields)]
    else:
        items = [(str(var) if var is not None else "field", fields)]
    if labels is not None:
        items = [(str(label), value) for label, (_, value) in zip(labels, items)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    else:
        fig = ax.figure
    reference = None
    for name, value in items:
        result = kinetic_energy_spectrum(value, dx)
        k = result["wavenumber"]
        mask = k > 0
        ax.loglog(1.0 / k[mask], result["spectrum"][mask], lw=1.2, label=str(name), **kwargs)
        if reference is None:
            reference = (1.0 / k[mask], result["spectrum"][mask])
    if reference is not None and np.isfinite(reference_slope):
        wavelength_ref = reference[0]
        anchor_k = 0.5 * (wavelength_ref.min() + wavelength_ref.max())
        anchor_index = int(np.argmin(np.abs(wavelength_ref - anchor_k)))
        scale = reference[1][anchor_index] * (anchor_k ** -reference_slope)
        ax.loglog(wavelength_ref, scale * wavelength_ref ** reference_slope,
                  color="gray", ls="--", lw=0.9,
                  label="参考斜率 {0:.2f}".format(reference_slope))
    ax.set_xlabel("波长 [m]")
    ax.set_ylabel("谱密度")
    ax.set_title("方差/动能谱 (B9)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    return fig, ax

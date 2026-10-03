"""网格几何、Arakawa C-grid 错位插值与地形追随高度。

对应 C++ `vibe::grid`（include/vibe/grid/geometry.hpp、interpolation.hpp）。
本模块是 Python 侧对 C++ 几何语义的**只读镜像**，不做并行分解、不含 halo 通信。

坐标系
------
水平：Arakawa C-grid [D3]，`u` 位于 x 面心、`v` 位于 y 面心、标量位于体心。
垂直：Lorenz 错位 [D4]，`w` 位于层界面，标量位于层中心。
垂向坐标：地形追随高度坐标 [D2]（Gal-Chen & Somerville 1975）

.. math::

    \\zeta = H \\frac{z - z_s(x,y)}{H - z_s(x,y)},
    \\qquad z = z_s + \\zeta \\left(1 - \\frac{z_s}{H}\\right)

其中 `H = z_top`，`\\zeta` 的量纲为米（与 C++ `Geometry::zeta` 一致）。

变分辨率网格 [N5][N6] 通过逐列 `dx_cell` 与逐行 `dy_cell` 表示；动力学与后处理
看到的是同一套度量项接口，因此不需要区分均匀/变分辨率。

文献：[D2][D3][D4][D9][D10][N1][N5][B3]。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "GridSpec",
    "zeta_levels",
    "stagger_to_cell",
    "cell_to_face",
    "STAGGER_CELL",
    "STAGGER_FACE_X",
    "STAGGER_FACE_Y",
    "STAGGER_FACE_Z",
    "STAGGER_ALIASES",
]

#: 错位类型（字符串常量，对应 C++ `grid::Stagger`）
STAGGER_CELL = "cell"
STAGGER_FACE_X = "face_x"
STAGGER_FACE_Y = "face_y"
STAGGER_FACE_Z = "face_z"
STAGGER_CORNER = "corner"

#: 变量名/别名 -> 错位类型；与 C++ `vibe::io` 的输出命名一致
STAGGER_ALIASES: dict[str, str] = {
    "u": STAGGER_FACE_X, "u10": STAGGER_FACE_X, "u_10m": STAGGER_FACE_X,
    "v": STAGGER_FACE_Y, "v10": STAGGER_FACE_Y, "v_10m": STAGGER_FACE_Y,
    "w": STAGGER_FACE_Z, "wa": STAGGER_FACE_Z,
    "rho": STAGGER_CELL, "theta": STAGGER_CELL, "pi": STAGGER_CELL,
    "pi0": STAGGER_CELL, "p": STAGGER_CELL, "pres": STAGGER_CELL,
    "qv": STAGGER_CELL, "qc": STAGGER_CELL, "qr": STAGGER_CELL,
    "qi": STAGGER_CELL, "qs": STAGGER_CELL, "qg": STAGGER_CELL,
    "t": STAGGER_CELL, "tk": STAGGER_CELL, "t2": STAGGER_CELL,
    "zeta": STAGGER_CELL, "zs": STAGGER_CELL, "terrain": STAGGER_CELL,
    "rain": STAGGER_CELL, "rainc": STAGGER_CELL, "rainnc": STAGGER_CELL,
    "dbz": STAGGER_CELL, "reflectivity": STAGGER_CELL,
    "cape": STAGGER_CELL, "cin": STAGGER_CELL, "srh": STAGGER_CELL,
}


def stagger_of(varname: str) -> str:
    """按变量名推断错位类型，未知变量按体心（Cell）处理。

    复杂度：O(1)。
    """
    key = varname.strip().lower()
    if key in STAGGER_ALIASES:
        return STAGGER_ALIASES[key]
    base = key.split("_")[0]
    return STAGGER_ALIASES.get(base, STAGGER_CELL)


# ---------------------------------------------------------------------------
# 垂直层生成
# ---------------------------------------------------------------------------


def zeta_levels(
    nz: int,
    z_top: float = 20000.0,
    *,
    first_thickness: float = 20.0,
    stretch: float = 1.08,
    mode: str = "geometric",
) -> NDArray[np.float64]:
    """生成单调递增的垂直层界面坐标 `zeta[0..nz]`（单位：米）。

    几何拉伸（`mode="geometric"`，C++ `make_stretched_zeta`）
    ----------------------------------------------------------
    面间距按几何级数增长

    .. math::

        \\Delta \\zeta_k = \\Delta \\zeta_0 s^k,
        \\qquad \\Delta \\zeta_0 = \\frac{H (s-1)}{s^{n_z} - 1}

    其中 :math:`s` 为 `stretch`。该式保证 `\\sum_k \\Delta\\zeta_k = H`，
    即模式顶严格落在 `H` 上（离散化：一阶精确求和，迭代 O(nz)）。

    平滑拉伸（`mode="smooth"`，[D9][D10] 建议的连续形式）
    ----------------------------------------------------
    .. math::

        w(\\sigma) = 1 + (\\beta - 1)\\sigma^2, \\quad
        \\Delta z_i \\propto \\frac{w(i)}{\\sum_j w(j)}

    其中 `\\beta` 由 :math:`\\Delta z_0 / \\Delta z_{n_z-1}` 确定；
    该分布在高/低层都保持连续导数，抑制陡峭地形上的虚假气压梯度。

    复杂度：O(nz)；空间 O(nz)。
    """
    if nz < 1:
        raise ValueError(f"nz 必须 >= 1，得到 {nz}")
    if z_top <= 0.0:
        raise ValueError(f"z_top 必须为正值，得到 {z_top}")
    if mode == "smooth":
        sigma = np.linspace(0.0, 1.0, nz)
        ratio = max(first_thickness / z_top, 1.0e-6)
        beta = max(1.0, ratio * nz)
        weight = 1.0 + (beta - 1.0) * sigma ** 2
        dz = z_top * weight / weight.sum()
    else:
        step = max(first_thickness, 1.0e-6)
        if abs(stretch - 1.0) < 1.0e-12:
            dz = np.full(nz, z_top / nz)
        else:
            total = (stretch ** nz - 1.0) / (stretch - 1.0)
            dz = z_top * stretch ** np.arange(nz) / total
            if dz[0] > z_top:
                dz = np.full(nz, z_top / nz)
    zeta = np.concatenate(([0.0], np.cumsum(dz)))
    zeta[-1] = z_top
    return zeta


# ---------------------------------------------------------------------------
# GridSpec
# ---------------------------------------------------------------------------


@dataclass
class GridSpec:
    """网格几何描述（C++ `grid::Geometry` 的 Python 镜像）。

    Attributes
    ----------
    nx, ny, nz : int
        内部点数（不含 halo）。
    dx, dy : float
        名义水平分辨率 [m]；变分辨率时为最小分辨率。
    x0, y0 : float
        左下角物理坐标 [m]。
    z_top : float
        模式顶高度 [m]。
    halo : int
        halo 宽度；Python 侧只记录，不参与索引。
    zeta : ndarray, shape (nz+1,)
        垂直层界面坐标 [m]，单调递增，`zeta[-1] == z_top`。
    terrain : ndarray, shape (ny, nx)
        地形高度 `z_s` [m]（行主序：j 慢变、i 快变，与 C++ 一致）。
    dx_cell : ndarray, shape (nx,)
        逐列 x 单元宽度 [m]；变分辨率时非均匀。
    dy_cell : ndarray, shape (ny,)
        逐行 y 单元宽度 [m]。
    name : str
        网格名（如 `d01`、`d02`）。
    """

    nx: int = 64
    ny: int = 64
    nz: int = 40
    dx: float = 3000.0
    dy: float = 3000.0
    x0: float = 0.0
    y0: float = 0.0
    z_top: float = 20000.0
    halo: int = 4
    zeta: NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))
    terrain: NDArray[np.float64] | None = None
    dx_cell: NDArray[np.float64] | None = None
    dy_cell: NDArray[np.float64] | None = None
    name: str = "d01"

    # -- 构造后校验 -------------------------------------------------------
    def __post_init__(self) -> None:
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.nz = int(self.nz)
        self.halo = int(self.halo)
        self.zeta = np.asarray(self.zeta, dtype=np.float64).ravel()
        if self.zeta.size != self.nz + 1:
            self.zeta = zeta_levels(self.nz, self.z_top)
        if self.terrain is not None:
            self.terrain = np.asarray(self.terrain, dtype=np.float64)
            if self.terrain.shape == (self.nx, self.ny):
                # 容忍 (nx, ny) 传入：转置为 C++ 的行主序 (ny, nx)
                self.terrain = self.terrain.T.copy()
            elif self.terrain.shape != (self.ny, self.nx):
                raise ValueError(
                    f"terrain 形状应为 (ny={self.ny}, nx={self.nx})，得到 {self.terrain.shape}"
                )
        if self.dx_cell is not None:
            self.dx_cell = np.asarray(self.dx_cell, dtype=np.float64).ravel()
            if self.dx_cell.size != self.nx:
                raise ValueError(f"dx_cell 长度应为 nx={self.nx}，得到 {self.dx_cell.size}")
        if self.dy_cell is not None:
            self.dy_cell = np.asarray(self.dy_cell, dtype=np.float64).ravel()
            if self.dy_cell.size != self.ny:
                raise ValueError(f"dy_cell 长度应为 ny={self.ny}，得到 {self.dy_cell.size}")

    # -- 基本属性 ---------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        """体心场的数组形状 `(nz, ny, nx)`（Python 数组下标顺序）。"""
        return (self.nz, self.ny, self.nx)

    @property
    def variable_resolution(self) -> bool:
        return self.dx_cell is not None or self.dy_cell is not None

    @property
    def flat_terrain(self) -> bool:
        return self.terrain is None or bool(np.all(self.terrain == 0.0))

    @property
    def lx(self) -> float:
        return float(self.dx_cell.sum()) if self.dx_cell is not None else self.nx * self.dx

    @property
    def ly(self) -> float:
        return float(self.dy_cell.sum()) if self.dy_cell is not None else self.ny * self.dy

    # -- 坐标 -------------------------------------------------------------
    def x_centers(self) -> NDArray[np.float64]:
        """体心 x 坐标（长度 nx）。均匀网格时为 `x0 + (i+0.5) dx`，复杂度 O(nx)。"""
        if self.dx_cell is None:
            return self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        return self.x0 + np.cumsum(self.dx_cell) - 0.5 * self.dx_cell

    def y_centers(self) -> NDArray[np.float64]:
        """体心 y 坐标（长度 ny）。复杂度 O(ny)。"""
        if self.dy_cell is None:
            return self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        return self.y0 + np.cumsum(self.dy_cell) - 0.5 * self.dy_cell

    def x_faces(self) -> NDArray[np.float64]:
        """x 面坐标（长度 nx+1，u 所在位置）。复杂度 O(nx)。"""
        if self.dx_cell is None:
            return self.x0 + np.arange(self.nx + 1) * self.dx
        return self.x0 + np.concatenate(([0.0], np.cumsum(self.dx_cell)))

    def y_faces(self) -> NDArray[np.float64]:
        """y 面坐标（长度 ny+1，v 所在位置）。复杂度 O(ny)。"""
        if self.dy_cell is None:
            return self.y0 + np.arange(self.ny + 1) * self.dy
        return self.y0 + np.concatenate(([0.0], np.cumsum(self.dy_cell)))

    def dx_at(self, i: ArrayLike) -> NDArray[np.float64]:
        """第 i 列的单元宽度。复杂度 O(1)。"""
        if self.dx_cell is None:
            return np.full_like(np.asarray(i, dtype=np.float64), self.dx)
        return self.dx_cell[np.asarray(i, dtype=np.intp)]

    def dy_at(self, j: ArrayLike) -> NDArray[np.float64]:
        """第 j 行的单元宽度。复杂度 O(1)。"""
        if self.dy_cell is None:
            return np.full_like(np.asarray(j, dtype=np.float64), self.dy)
        return self.dy_cell[np.asarray(j, dtype=np.intp)]

    # -- 垂直几何 ---------------------------------------------------------
    def zeta_centers(self) -> NDArray[np.float64]:
        """层中心的 zeta 值（长度 nz）。复杂度 O(nz)。"""
        return 0.5 * (self.zeta[:-1] + self.zeta[1:])

    def dzeta(self) -> NDArray[np.float64]:
        """层厚 `\\Delta\\zeta_k`（长度 nz）。复杂度 O(nz)。"""
        return np.diff(self.zeta)

    def jacobian(self) -> NDArray[np.float64] | float:
        """雅可比 :math:`G^{1/2} = \\partial z/\\partial \\zeta = 1 - z_s/H`。

        平坦地形时返回标量 1.0；否则返回 `(ny, nx)` 数组。复杂度 O(nx ny)。
        """
        if self.terrain is None:
            return 1.0
        return 1.0 - self.terrain / self.z_top

    def height(self, i: ArrayLike, j: ArrayLike, zeta: ArrayLike) -> NDArray[np.float64]:
        """地形追随坐标下 (i, j, zeta) 处的**海拔高度** [m]。

        公式（[D2] Gal-Chen & Somerville 1975）：

        .. math::

            z(i,j,\\zeta) = z_s(i,j) + \\zeta \\left(1 - \\frac{z_s(i,j)}{H}\\right)

        该映射对每个 `(i,j)` 是 :math:`\\zeta \\to z` 的**线性**变换，因此
        层中心的 `z` 等于层中心 `zeta` 的像（离散化无额外误差）。

        复杂度：O(1)（逐点），向量化调用 O(N)。
        """
        i_arr = np.asarray(i, dtype=np.intp)
        j_arr = np.asarray(j, dtype=np.intp)
        zeta_arr = np.asarray(zeta, dtype=np.float64)
        zs = 0.0 if self.terrain is None else self.terrain[j_arr, i_arr]
        factor = 1.0 - zs / self.z_top
        return zs + zeta_arr * factor

    def height_array(self, *, staggered_w: bool = False) -> NDArray[np.float64]:
        """三维高度场 `z[k, j, i]`，形状 `(nz, ny, nx)`。

        `staggered_w=True` 时返回层界面（w 所在）高度，形状 `(nz+1, ny, nx)`。

        复杂度：O(nx ny nz)，内存 O(nx ny nz)。
        """
        zs = np.zeros((self.ny, self.nx)) if self.terrain is None else self.terrain
        levels = self.zeta if staggered_w else self.zeta_centers()
        factor = 1.0 - zs / self.z_top                      # (ny, nx)
        return zs[None, :, :] + levels[:, None, None] * factor[None, :, :]

    def geopotential_height(self, *, staggered_w: bool = False) -> NDArray[np.float64]:
        """位势高度 [m]（海平面以上）。等于 :meth:`height_array`。复杂度同上。"""
        return self.height_array(staggered_w=staggered_w)

    def dz(self) -> NDArray[np.float64]:
        """层中心的层厚 :math:`\\Delta z_k = \\Delta\\zeta_k (1 - z_s/H)`。

        平坦地形时返回长度 nz 的一维数组；否则返回 `(nz, ny, nx)`。
        复杂度：O(nx ny nz)。
        """
        thickness = np.diff(self.zeta)                          # (nz,)
        jac = self.jacobian()
        if np.isscalar(jac):
            return thickness
        return thickness[:, None, None] * np.asarray(jac)[None, :, :]

    def cell_volume(self) -> NDArray[np.float64] | float:
        """有限体积单元体积 :math:`\\Delta V = \\Delta x \\Delta y \\Delta z`。

        复杂度：O(nx ny nz)。
        """
        dx = self.dx_cell if self.dx_cell is not None else np.full(self.nx, self.dx)
        dy = self.dy_cell if self.dy_cell is not None else np.full(self.ny, self.dy)
        return self.dz() * dy[None, :, None] * dx[None, None, :]

    def describe(self) -> str:
        """一行网格摘要。复杂度 O(1)。"""
        vr = "变分辨率" if self.variable_resolution else "均匀"
        terrain = "平坦" if self.flat_terrain else f"地形 max={float(self.terrain.max()):.1f} m"
        return (
            f"{self.name}: {self.nx}x{self.ny}x{self.nz} ({vr}, {terrain}), "
            f"Lx={self.lx / 1000.0:.1f} km, Ly={self.ly / 1000.0:.1f} km, "
            f"H={self.z_top / 1000.0:.1f} km"
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典（数组转列表）。复杂度 O(N)。"""
        out = asdict(self)
        out["zeta"] = self.zeta.tolist()
        out["terrain"] = None if self.terrain is None else self.terrain.tolist()
        out["dx_cell"] = None if self.dx_cell is None else self.dx_cell.tolist()
        out["dy_cell"] = None if self.dy_cell is None else self.dy_cell.tolist()
        return out

    # -- 构造入口 ---------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GridSpec":
        """从 JSON/N YAML 元数据映射构造。

        识别键：`nx, ny, nz, dx, dy, x0, y0, z_top, halo, zeta, terrain/zs,
        dx_cell, dy_cell, name`；长度字符串（如 `"3km"`）由
        :func:`vibe_post.config.parse_quantity` 解析。
        复杂度：O(N + nx ny)。
        """
        from .config import parse_quantity

        def num(key: str, default: float) -> float:
            return parse_quantity(data.get(key, default)) if key in data else float(default)

        spec = cls(
            nx=int(data.get("nx", 64)),
            ny=int(data.get("ny", 64)),
            nz=int(data.get("nz", 40)),
            dx=num("dx", 3000.0),
            dy=num("dy", 3000.0),
            x0=num("x0", 0.0),
            y0=num("y0", 0.0),
            z_top=num("z_top", 20000.0),
            halo=int(data.get("halo", 4)),
            name=str(data.get("name", "d01")),
        )
        nz = spec.nz
        if "zeta" in data and data["zeta"] is not None:
            spec.zeta = np.asarray(data["zeta"], dtype=np.float64).ravel()
        else:
            spec.zeta = zeta_levels(
                nz, spec.z_top,
                first_thickness=num("first_thickness", 20.0),
                stretch=float(data.get("stretch", data.get("vertical_stretch", 1.08))),
            )
        terrain = data.get("terrain", data.get("zs"))
        if terrain is not None:
            spec.terrain = np.asarray(terrain, dtype=np.float64)
        if data.get("dx_cell") is not None:
            spec.dx_cell = np.asarray(data["dx_cell"], dtype=np.float64)
        if data.get("dy_cell") is not None:
            spec.dy_cell = np.asarray(data["dy_cell"], dtype=np.float64)
        spec.__post_init__()
        return spec

    @classmethod
    def from_json(cls, path: str | Path) -> "GridSpec":
        """读取 JSON 网格元数据。复杂度 O(文件大小)。"""
        target = Path(path)
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "grid" in payload and isinstance(payload["grid"], Mapping):
            payload = payload["grid"]
        return cls.from_mapping(payload)

    @classmethod
    def from_dataset(cls, dataset: Any) -> "GridSpec":
        """由 :class:`vibe_post.io.Dataset` 或 xarray Dataset 推断网格。

        推断规则：取任一三维场确定 `(nz, ny, nx)`；`x`/`y`/`zeta` 坐标
        若存在则使用，否则由 `dx`/`dy`/`z_top` 属性或默认值生成。
        复杂度：O(N)。
        """
        coords = dict(getattr(dataset, "coords", {}) or {})
        attrs = dict(getattr(dataset, "attrs", {}) or {})
        dims_map = dict(getattr(dataset, "dims", {}) or {})
        fields_map = dict(getattr(dataset, "fields", {}) or {})

        shape: tuple[int, int, int] | None = None
        for name, array in fields_map.items():
            arr = np.asarray(array)
            names = dims_map.get(name)
            if names is not None and "zeta" in names and ("x" in names or "ny" in names):
                order = [names.index(d) for d in ("zeta", "y", "x")]
                shape = tuple(int(arr.shape[o]) for o in order)  # type: ignore[assignment]
                break
            if arr.ndim == 3 and arr.shape[0] <= 500:
                shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
                break
        if shape is None:
            shape = (int(attrs.get("nz", 40)), int(attrs.get("ny", 64)), int(attrs.get("nx", 64)))
        nz, ny, nx = shape

        def coordinate(name: str, count: int, fallback_spacing: float, fallback_origin: float) -> NDArray[np.float64]:
            for key in (name, name.upper(), name + "_coord"):
                if key in coords:
                    values = np.asarray(coords[key], dtype=np.float64).ravel()
                    if values.size == count:
                        return values
            return fallback_origin + (np.arange(count) + 0.5) * fallback_spacing

        dx = float(attrs.get("dx", attrs.get("DX", 3000.0)))
        dy = float(attrs.get("dy", attrs.get("DY", dx)))
        x = coordinate("x", nx, dx, float(attrs.get("x0", 0.0)))
        y = coordinate("y", ny, dy, float(attrs.get("y0", 0.0)))
        z_top = float(attrs.get("z_top", attrs.get("H", 20000.0)))

        if "zeta" in coords and np.asarray(coords["zeta"]).size in (nz, nz + 1):
            zeta = np.asarray(coords["zeta"], dtype=np.float64).ravel()
            if zeta.size == nz:
                zeta = np.concatenate(([0.0], 0.5 * (zeta[1:] + zeta[:-1]), [z_top]))
            zeta = zeta.copy()
            zeta[-1] = max(z_top, float(zeta[-1]))
        else:
            zeta = zeta_levels(nz, z_top)

        terrain = None
        for key in ("terrain", "zs", "HGT", "hgt", "topo"):
            if key in coords:
                candidate = np.asarray(coords[key], dtype=np.float64)
                if candidate.shape == (ny, nx):
                    terrain = candidate
                    break
            if key in fields_map:
                candidate = np.asarray(fields_map[key], dtype=np.float64)
                if candidate.shape == (ny, nx):
                    terrain = candidate
                    break

        dx_cell = None
        dy_cell = None
        if x.size >= 2:
            edges = np.concatenate(([x[0] - 0.5 * (x[1] - x[0])], 0.5 * (x[1:] + x[:-1]),
                                    [x[-1] + 0.5 * (x[-1] - x[-2])]))
            dx_cell = np.diff(edges)
            if float(np.ptp(dx_cell)) < 1.0e-6 * max(dx, 1.0):
                dx_cell = None
        if y.size >= 2:
            edges = np.concatenate(([y[0] - 0.5 * (y[1] - y[0])], 0.5 * (y[1:] + y[:-1]),
                                    [y[-1] + 0.5 * (y[-1] - y[-2])]))
            dy_cell = np.diff(edges)
            if float(np.ptp(dy_cell)) < 1.0e-6 * max(dy, 1.0):
                dy_cell = None

        return cls(
            nx=nx, ny=ny, nz=nz,
            dx=float(dx_cell.mean()) if dx_cell is not None else dx,
            dy=float(dy_cell.mean()) if dy_cell is not None else dy,
            x0=float(x[0] - 0.5 * (dx_cell[0] if dx_cell is not None else dx)),
            y0=float(y[0] - 0.5 * (dy_cell[0] if dy_cell is not None else dy)),
            z_top=z_top, zeta=zeta, terrain=terrain,
            dx_cell=dx_cell, dy_cell=dy_cell,
            halo=int(attrs.get("halo", 4)),
            name=str(attrs.get("name", attrs.get("grid", "d01"))),
        )


# ---------------------------------------------------------------------------
# Arakawa C-grid 错位插值（与 C++ grid::stagger_to_cell 语义一致）
# ---------------------------------------------------------------------------


def _average_with_clamped_index(
    array: NDArray[np.float64], offset: int, axis: int
) -> NDArray[np.float64]:
    """沿 `axis` 做 `0.5 (f[i] + f[i+offset])`，越界索引钳位。

    离散化
    ------
    .. math::

        f_{i}^{cell} = \\tfrac12 \\left( f_{i-\\frac12} + f_{i+\\frac12} \\right),
        \\qquad i = 0 \\ldots n-1

    对 `offset = +1`（FaceX/FaceY，长度 n+1）取 `i-1/2 -> i-1`、`i+1/2 -> i`；
    对 `offset = 0`（FaceZ，长度 n+1）取 `i-1/2 -> i`、`i+1/2 -> i+1`。
    边界处（`i=0` 或 `i=n-1`）把越界索引钳到边界，等价于零梯度/单侧外推，
    与 C++ `stagger_to_cell` 的边界处理**逐点一致**。

    复杂度：O(N)，内存 O(N)。
    """
    if offset == 1:
        low = np.clip(np.arange(array.shape[axis]) - 1, 0, array.shape[axis] - 1)
        high = np.clip(np.arange(array.shape[axis]), 0, array.shape[axis] - 1)
    else:
        low = np.clip(np.arange(array.shape[axis]), 0, array.shape[axis] - 1)
        high = np.clip(np.arange(array.shape[axis]) + 1, 0, array.shape[axis] - 1)
    first = np.take(array, low, axis=axis)
    second = np.take(array, high, axis=axis)
    return 0.5 * (first + second)


def stagger_to_cell(
    field: ArrayLike,
    stagger: str,
    grid: GridSpec | None = None,
    *,
    axis: int | None = None,
) -> NDArray[np.float64]:
    """把错位场插值到体心（标量位置）。

    公式（二阶中心平均，[D3] Arakawa & Lamb 1977 第 3 节）
    ----------------------------------------------------
    .. math::

        u^{c}_{i,j,k} &= \\tfrac12 \\left(u_{i-\\frac12,j,k} + u_{i+\\frac12,j,k}\\right) \\\\
        v^{c}_{i,j,k} &= \\tfrac12 \\left(v_{i,j-\\frac12,k} + v_{i,j+\\frac12,k}\\right) \\\\
        w^{c}_{i,j,k} &= \\tfrac12 \\left(w_{i,j,k-\\frac12} + w_{i,j,k+\\frac12}\\right)

    离散化：数组下标顺序统一为 `(..., z, y, x)`（即 numpy 的
    `(k, j, i)`），与 C++ `Grid::flatten` 的行主序（i 最快）一致。
    边界点取单侧值（零梯度外推），保证输出与输入形状相同。

    参数
    ----
    field : array_like
        错位场，最后一维为 x，倒数第二维为 y，倒数第三维（若有）为 z。
    stagger : str
        `"u"`/`"face_x"`、`"v"`/`"face_y"`、`"w"`/`"face_z"`、`"cell"`；
        也接受变量名（内部调用 :func:`stagger_of`）。
    grid : GridSpec, 可选
        用于校验形状；为 None 时按数组形状推断。
    axis : int, 可选
        显式指定要平均的轴，覆盖 `stagger` 的推断。

    复杂度：O(N)，N = 场点数；内存 O(N)。
    """
    arr = np.asarray(field, dtype=np.float64)
    kind = stagger if stagger in (STAGGER_CELL, STAGGER_FACE_X, STAGGER_FACE_Y, STAGGER_FACE_Z,
                                  STAGGER_CORNER) else stagger_of(stagger)
    if kind in (STAGGER_CELL, STAGGER_CORNER) or arr.ndim == 1:
        return arr.copy()
    if axis is None:
        if kind == STAGGER_FACE_X:
            axis = arr.ndim - 1
            offset = 1
        elif kind == STAGGER_FACE_Y:
            axis = arr.ndim - 2
            offset = 1
        elif kind == STAGGER_FACE_Z:
            axis = arr.ndim - 3 if arr.ndim >= 3 else 0
            offset = 0
        else:  # pragma: no cover - 已被上面的 kind 判断排除
            return arr.copy()
    else:
        offset = 1 if kind in (STAGGER_FACE_X, STAGGER_FACE_Y) else 0
    axis = axis % arr.ndim
    return _average_with_clamped_index(arr, offset, axis)


def cell_to_face(
    field: ArrayLike,
    axis: int,
    grid: GridSpec | None = None,
    *,
    closed: bool = False,
) -> NDArray[np.float64]:
    """体心 -> 面（长度加 1），二阶精度（C++ `cell_to_face_x/y/z`）。

    公式
    ----
    .. math::

        f_{i+\\frac12} = \\tfrac12 (f_i + f_{i+1}), \\quad i = 0 \\ldots n-2

    端点用线性外推 `f_{-1/2} = 1.5 f_0 - 0.5 f_1`、
    `f_{n-1/2} = 1.5 f_{n-1} - 0.5 f_{n-2}`（`closed=True` 时改为周期端点，
    用于嵌套边界与周期试验）。

    复杂度：O(N)，内存 O(N)。
    """
    arr = np.asarray(field, dtype=np.float64)
    axis = axis % arr.ndim
    n = arr.shape[axis]
    if n < 2:
        raise ValueError("cell_to_face 需要该轴至少 2 个点")
    low = np.take(arr, np.arange(n - 1), axis=axis)
    high = np.take(arr, np.arange(1, n), axis=axis)
    interior = 0.5 * (low + high)
    if closed:
        first = 0.5 * (np.take(arr, n - 1, axis=axis) + np.take(arr, 0, axis=axis))
        last = first
    else:
        first = 1.5 * np.take(arr, 0, axis=axis) - 0.5 * np.take(arr, 1, axis=axis)
        last = 1.5 * np.take(arr, n - 1, axis=axis) - 0.5 * np.take(arr, n - 2, axis=axis)
    return np.concatenate((np.expand_dims(first, axis),
                           interior,
                           np.expand_dims(last, axis)), axis=axis)


def unstagger(field: ArrayLike, stagger: str, grid: GridSpec | None = None) -> NDArray[np.float64]:
    """`stagger_to_cell` 的别名，便于 CL I 与脚本使用。复杂度 O(N)。"""
    return stagger_to_cell(field, stagger, grid)

"""文件 IO：NetCDF / GRIB2 / `.vibebin` 三后端，自动探测 + 优雅降级。

对应 C++ `vibe::io`（docs/design/00_architecture.md 第 10.2 节、
include/vibe/io/field_io.hpp）。本模块是**唯一**接触文件格式的地方，
其余模块只使用 `Dataset`。

后端优先级（`open_dataset(..., backend="auto")`）
------------------------------------------------
1. **xarray/netCDF4** —— VIBE 输出 NetCDF，变量 `u,v,w,rho,theta,pi,qv,qc,qr,qi,qs,qg`，
   维度 `(time, zeta, y, x)` 或 `(zeta, y, x)`；
2. **cfgrib** —— GRIB2 侧边界/初始场；
3. **纯 numpy 二进制** —— C++ `io::BinaryWriter` 的 `.vibebin`
   （字节布局见 `_newio_part` 模块 docstring / docs/design/10_postprocessing.md 第 4 节）。

维度与内存顺序约定
------------------
C++ 侧逻辑维度顺序为 `(time, zeta, y, x)`，且 `i`（x 方向）变化最快。
numpy 是 C 序，故数组下标顺序是逻辑顺序的**反转**：`(t, k, j, i)`。
`Dataset.dims` 记录逻辑维度名（顺序与 C++ 一致），`fields[name].shape` 与其反转一致。

文献：[B3] Warner (2011) 第 1 章；[D16] WRF ARW 第 5 章。
"""

from __future__ import annotations

import json
import os
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Dataset", "VariableInfo", "VibeBinFile", "VibeBinHeader",
    "MissingDependencyError", "UnsupportedFormatError", "VibeBinError",
    "VIBE_BIN_MAGIC", "VIBE_BIN_NAME_BYTES", "VIBE_BIN_HEADER_BYTES",
    "CANONICAL_VARIABLES", "VIBE_BIN_MAGIC",
    "open_dataset", "read_vibebin", "write_vibebin", "read_netcdf", "read_grib",
    "to_netcdf", "to_numpy", "summarize", "available_backends", "sniff_format",
    "normalize_dims", "read_binary_header", "shape_for_name", "stagger_from_name",
]

class MissingDependencyError(ImportError):
    """可选依赖缺失，错误信息中带安装提示（"优雅降级"的唯一出口）。"""

    def __init__(self, package: str, purpose: str, extra: str | None = None) -> None:
        extra = extra or package
        hint = "pip install 'vibe-post[{0}]'  或  pip install {1}".format(extra, package)
        super().__init__(
            "读取/写入 {0} 需要可选依赖 {1!r}，当前环境未安装。\n  安装方式：{2}".format(
                purpose, package, hint))
        self.package = package
        self.purpose = purpose


class UnsupportedFormatError(ValueError):
    """文件格式无法识别或不受支持。"""


class VibeBinError(ValueError):
    """`.vibebin` 读写错误（magic / 维度 / 布局不匹配等）。"""

#: 扩展名 -> 后端
_EXTENSION_BACKENDS: dict[str, str] = {
    ".nc": "netcdf", ".nc4": "netcdf", ".cdf": "netcdf", ".netcdf": "netcdf",
    ".grib": "grib", ".grib2": "grib", ".grb": "grib", ".grb2": "grib",
    ".vibebin": "vibebin", ".bin": "vibebin", ".vbin": "vibebin",
}

#: 维度别名 -> 规范名（吸收 WRF/COSMO/xarray 的写法差异）
_DIM_ALIASES: dict[str, str] = {
    "z": "zeta", "lev": "zeta", "level": "zeta", "nz": "zeta", "zeta": "zeta",
    "znu": "zeta", "bottom_top": "zeta", "sig": "zeta", "eta": "zeta",
    "j": "y", "ny": "y", "lat": "y", "latitude": "y", "south_north": "y",
    "i": "x", "nx": "x", "lon": "x", "longitude": "x", "west_east": "x",
    "t": "time", "time": "time", "time_counter": "time", "nt": "time",
}


def normalize_dims(dims: Sequence[str]) -> tuple[str, ...]:
    """维度名 -> 规范名 `(time, zeta, y, x)`。复杂度 O(ndim)。"""
    return tuple(_DIM_ALIASES.get(str(d).lower(), str(d).lower()) for d in dims)


def to_numpy(value: Any) -> NDArray[Any]:
    """统一取 numpy 数组：ndarray / xarray DataArray / 带 `.values` 的对象。

    复杂度：O(1)（不复制数据）。"""
    if isinstance(value, Dataset):
        raise TypeError("to_numpy 不接受 Dataset；请传 dataset['变量名'] 或 dataset.fields")
    if isinstance(value, np.ndarray):
        return value
    values = getattr(value, "values", None)
    if values is not None:
        return np.asarray(values)
    return np.asarray(value)


def available_backends() -> dict[str, bool]:
    """三种后端可用性（不抛异常）。复杂度 O(1)。"""
    status = {"vibebin": True, "netcdf": False, "grib": False}
    try:
        import xarray  # noqa: F401
        status["netcdf"] = True
    except Exception:
        pass
    try:
        import cfgrib  # noqa: F401
        status["grib"] = True
    except Exception:
        pass
    return status


def _require(package: str, purpose: str, extra: str | None = None) -> Any:
    """导入可选依赖；缺失时抛 `MissingDependencyError`。复杂度 O(1)。"""
    try:
        return __import__(package)
    except Exception as exc:
        raise MissingDependencyError(package, purpose, extra) from exc


def sniff_format(path: str | os.PathLike[str]) -> str:
    """按扩展名 + magic 判定后端（`vibebin`/`netcdf`/`grib`）。复杂度 O(1)。"""
    target = Path(path)
    backend = _EXTENSION_BACKENDS.get(target.suffix.lower(), "")
    try:
        with target.open("rb") as handle:
            magic = handle.read(8)
    except OSError:
        return backend or "unknown"
    if magic[:8] == VIBE_BIN_MAGIC:
        return "vibebin"
    if magic[:3] == b"CDF" or magic[:4] == b"\x89HDF":
        return "netcdf"
    if magic[:4] == b"GRIB":
        return "grib"
    if magic[:4] == b"\x1f\x8b":
        return backend or "grib"
    return backend or "unknown"



#: `.vibebin` 文件 magic（8 字节 ASCII），对应 C++ `io::kBinaryMagic`
VIBE_BIN_MAGIC = b"VIBEBIN1"

#: 变量名定长 ASCII 字节数（C++ `io::kBinaryNameLength`）
VIBE_BIN_NAME_BYTES = 32

#: 文件头长度：magic(8) + nx,ny,nz,real_kind(4 x int32) + time(float64)
#:             + nvars(int32) + nlevels(int32) = 40
VIBE_BIN_HEADER_BYTES = 8 + 4 * 4 + 8 + 4 + 4

_TRAILER_MAGIC = b"VIBETRL1"
_RESTART_INFO = "__restart_info__"

#: 扩展名 -> 后端
_EXTENSION_BACKENDS: dict[str, str] = {
    ".nc": "netcdf", ".nc4": "netcdf", ".cdf": "netcdf", ".netcdf": "netcdf",
    ".grib": "grib", ".grib2": "grib", ".grb": "grib", ".grb2": "grib",
    ".vibebin": "vibebin", ".bin": "vibebin", ".vbin": "vibebin",
}

#: C++ 侧状态/诊断的规范逻辑维度顺序（architecture 第 6.1、10.2 节）
CANONICAL_VARIABLES: dict[str, tuple[str, ...]] = {
    name: ("time", "zeta", "y", "x")
    for name in ("u", "v", "w", "rho", "theta", "pi", "qv", "qc", "qr", "qi", "qs", "qg")
}
CANONICAL_VARIABLES.update({
    "zs": ("y", "x"), "terrain": ("y", "x"), "psfc": ("y", "x"), "t2": ("y", "x"),
    "q2": ("y", "x"), "u10": ("y", "x"), "v10": ("y", "x"), "rain": ("y", "x"),
})

#: 维度别名 -> 规范名（吸收 WRF/COSMO/xarray 的写法差异）
_DIM_ALIASES: dict[str, str] = {
    "z": "zeta", "lev": "zeta", "level": "zeta", "nz": "zeta", "zeta": "zeta",
    "znu": "zeta", "bottom_top": "zeta", "sig": "zeta", "eta": "zeta",
    "j": "y", "ny": "y", "lat": "y", "latitude": "y", "south_north": "y",
    "i": "x", "nx": "x", "lon": "x", "longitude": "x", "west_east": "x",
    "t": "time", "time": "time", "time_counter": "time", "nt": "time",
}

@dataclass
class Dataset:
    """VIBE 后处理统一数据对象。

    `fields` 中数组的下标顺序是**逻辑维度顺序的反转**（numpy C 序）：
    `dims["theta"] == ("time","zeta","y","x")` 对应
    `fields["theta"].shape == (nt, nz, ny, nx)`，且 `fields["theta"][t,k,j,i]`
    与 C++ `Field::at(i,j,k)`（第 t 个时刻）逐点对应。
    """

    fields: dict[str, NDArray[Any]] = field(default_factory=dict)
    dims: dict[str, tuple[str, ...]] = field(default_factory=dict)
    coords: dict[str, NDArray[Any]] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    time: float = 0.0

    def __post_init__(self) -> None:
        self.fields = {str(k): np.asarray(v) for k, v in dict(self.fields).items()}
        self.coords = {str(k): np.asarray(v) for k, v in dict(self.coords).items()}
        self.dims = {str(k): normalize_dims(v) for k, v in dict(self.dims).items()}
        for name, array in list(self.fields.items()):
            if name in self.dims:
                continue
            canonical = CANONICAL_VARIABLES.get(name)
            if canonical is not None and array.ndim == len(canonical):
                self.dims[name] = canonical
            else:
                self.dims[name] = _infer_dims(array, name)
        self.attrs = dict(self.attrs)
        self.attrs.setdefault("source", "memory")
        self.attrs.setdefault("real_kind", "float64")

    # -- 容器协议 ---------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self.fields or key in self.coords

    def __getitem__(self, key: str) -> NDArray[Any]:
        if key in self.fields:
            return self.fields[key]
        if key in self.coords:
            return self.coords[key]
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def keys(self) -> Iterable[str]:
        return self.fields.keys()

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.fields:
            return self.fields[key]
        return self.coords.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover - 展示用
        names = ", ".join(sorted(self.fields)[:8])
        more = ", ..." if len(self.fields) > 8 else ""
        return "Dataset(n={0}, time={1:g}, vars=[{2}{3}])".format(len(self.fields), self.time, names, more)

    # -- 变换 -------------------------------------------------------------
    def with_field(self, name: str, array: ArrayLike,
                   dims: Sequence[str] | None = None) -> "Dataset":
        """返回带新变量的新 Dataset（浅拷贝元数据，不复制数组）。复杂度 O(1)。"""
        clone = self.copy(deep=False)
        arr = np.asarray(array)
        clone.fields[str(name)] = arr
        clone.dims[str(name)] = (normalize_dims(dims) if dims is not None
                                 else self.dims.get(str(name), _infer_dims(arr, str(name))))
        return clone

    def copy(self, *, deep: bool = False) -> "Dataset":
        """复制；`deep=True` 时也复制数组。复杂度 O(N)（deep）/ O(1)（浅）。"""
        return Dataset(
            fields={k: (v.copy() if deep else v) for k, v in self.fields.items()},
            dims=dict(self.dims),
            coords={k: (v.copy() if deep else v) for k, v in self.coords.items()},
            attrs=dict(self.attrs),
            time=float(self.time),
        )

    def merge(self, other: "Dataset", *, overwrite: bool = True) -> "Dataset":
        """合并；键冲突时按 `overwrite` 决定。复杂度 O(变量数)。"""
        out = self.copy(deep=False)
        for name, array in other.fields.items():
            if name in out.fields and not overwrite:
                continue
            out.fields[name] = array
            out.dims[name] = other.dims.get(name, out.dims.get(name, _infer_dims(array, name)))
        for name, array in other.coords.items():
            out.coords.setdefault(name, array)
        out.attrs.update(other.attrs)
        return out

    def select_time(self, index: int) -> "Dataset":
        """取 `time` 维第 `index` 层并去掉该维。复杂度 O(N)。"""
        out = self.copy(deep=False)
        for name, array in list(self.fields.items()):
            dims = self.dims.get(name, ())
            if dims and dims[0] == "time":
                out.fields[name] = np.take(array, index, axis=0)
                out.dims[name] = dims[1:]
        coord = out.coords.get("time")
        if coord is not None and np.asarray(coord).ndim == 1 and np.asarray(coord).size > index:
            out.time = float(np.asarray(coord)[index])
            out.coords["time"] = np.asarray([out.time])
        return out

    def shape_of(self, name: str) -> tuple[int, ...]:
        """变量逻辑形状（C++ 顺序）。复杂度 O(ndim)。"""
        return tuple(self.fields[name].shape[::-1])

    def grid_shape(self) -> tuple[int, int, int]:
        """推断 `(nx, ny, nz)`；优先属性，其次三维场。复杂度 O(变量数)。"""
        if {"nx", "ny", "nz"} <= set(self.attrs):
            return int(self.attrs["nx"]), int(self.attrs["ny"]), int(self.attrs["nz"])
        for name, array in self.fields.items():
            dims = self.dims.get(name, ())
            if array.ndim >= 3 and "y" in dims and "x" in dims:
                nx = int(array.shape[dims.index("x")])
                ny = int(array.shape[dims.index("y")])
                nz = int(array.shape[dims.index("zeta")]) if "zeta" in dims else int(
                    array.shape[dims.index("y") - 1]
                )
                return nx, ny, nz
        nx = int(np.asarray(self.coords["x"]).size) if "x" in self.coords else int(self.attrs.get("nx", 0))
        ny = int(np.asarray(self.coords["y"]).size) if "y" in self.coords else int(self.attrs.get("ny", 0))
        nz = int(np.asarray(self.coords["zeta"]).size) if "zeta" in self.coords else int(self.attrs.get("nz", 0))
        return nx, ny, nz

    def to_xarray(self) -> Any:
        """转 xarray.Dataset（需要 xarray）。复杂度 O(N)。"""
        xarray = _require("xarray", "转换为 xarray.Dataset", "netcdf")
        data: dict[str, Any] = {}
        for name, array in self.fields.items():
            dims = self.dims.get(name, _infer_dims(array, name))
            coords: dict[str, Any] = {}
            for dim, size in zip(dims, array.shape):
                if dim in self.coords and np.asarray(self.coords[dim]).shape == (size,):
                    coords[dim] = np.asarray(self.coords[dim])
            data[name] = xarray.DataArray(array, dims=dims, coords=coords, name=name)
        return xarray.Dataset(data, attrs=_jsonable(self.attrs))

    def summary(self) -> str:
        """多行文本摘要。复杂度 O(N)。"""
        lines = ["Dataset: {0} 个变量, time = {1:g} s".format(len(self.fields), self.time)]
        shown = {k: v for k, v in list(self.attrs.items())[:6]}
        lines.append("  attrs: {0}".format(shown))
        for name in sorted(self.fields):
            info = VariableInfo.of(name, self.fields[name], self.dims.get(name, ()))
            lines.append("  " + info.describe())
        return "\n".join(lines)

    # -- 序列化 -----------------------------------------------------------
    def write_vibebin(self, path: str | os.PathLike[str], *,
                      real_kind: int | None = None,
                      attrs: Mapping[str, Any] | None = None) -> Path:
        """写出 .vibebin。复杂度 O(N)。"""
        return write_vibebin(path, self, real_kind=real_kind, attrs=attrs)

def _infer_dims(array: NDArray[Any], name: str = "") -> tuple[str, ...]:
    """按数组形状推断维度名（末尾为 x）。复杂度 O(1)。"""
    if array.ndim == 0:
        return ()
    if array.ndim == 1:
        return ("x",)
    if array.ndim == 2:
        return ("y", "x")
    if array.ndim == 3:
        return ("zeta", "y", "x")
    if array.ndim == 4:
        return ("time", "zeta", "y", "x")
    return tuple("dim{0}".format(k) for k in range(array.ndim))


def _jsonable(value: Any) -> Any:
    """把 numpy 标量/数组递归转成 JSON 可序列化对象。复杂度 O(N)。"""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class VariableInfo:
    """单变量形状与统计摘要。"""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dims: tuple[str, ...]
    vmin: float
    vmax: float
    mean: float
    std: float
    nan_count: int
    nbytes: int

    @classmethod
    def of(cls, name: str, array: ArrayLike, dims: Sequence[str] = ()) -> "VariableInfo":
        """计算摘要（全 NaN / 空数组安全）。复杂度 O(N)。"""
        arr = np.asarray(array)
        if arr.size == 0:
            nan = float("nan")
            return cls(name, arr.shape, str(arr.dtype), tuple(dims), nan, nan, nan, nan, 0, 0)
        flat = np.asarray(arr, dtype=np.float64).ravel()
        mask = np.isfinite(flat)
        nan_count = int(flat.size - int(mask.sum()))
        if nan_count == flat.size:
            return cls(name, arr.shape, str(arr.dtype), tuple(dims),
                       float("nan"), float("nan"), float("nan"), float("nan"),
                       nan_count, int(arr.nbytes))
        valid = flat[mask]
        return cls(name, arr.shape, str(arr.dtype), tuple(dims),
                   float(valid.min()), float(valid.max()),
                   float(valid.mean()), float(valid.std()),
                   nan_count, int(arr.nbytes))

    def describe(self) -> str:
        """一行摘要文本。复杂度 O(1)。"""
        return "{0:<12s} shape={1:<18s} {2:<8s} [{3:.4g}, {4:.4g}] mean={5:.4g} nan={6}".format(
            self.name, str(self.shape), self.dtype, self.vmin, self.vmax, self.mean, self.nan_count)


def summarize(dataset: Dataset | str | os.PathLike[str]) -> str:
    """返回 Dataset 或文件的摘要文本（CLI `info` 用）。复杂度 O(N)。"""
    if not isinstance(dataset, Dataset):
        dataset = open_dataset(dataset)
    lines = [dataset.summary()]
    try:
        from .grid import GridSpec
        lines.append("  grid: " + GridSpec.from_dataset(dataset).describe())
    except Exception as exc:  # pragma: no cover - 网格推断失败不影响 info
        lines.append("  grid: 推断失败（{0}）".format(exc))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 纯 numpy 二进制后端：VibeBinFile
# ---------------------------------------------------------------------------



"""`.vibebin` 读写：与 C++ `vibe::io::BinaryWriter` 字节级兼容。

文件布局（全部 little-endian，无对齐填充）
------------------------------------------
.. code-block:: text

    offset  type      field
    0       char[8]   magic = b"VIBEBIN1"
    8       int32     nx, ny, nz            （本地子域内部点数；串行时即全局）
    20      int32     real_kind             （0=float64 写者, 1=float32 写者）
    24      float64   time
    32      int32     nvars
    36      int32     nlevels               （= nz，供剖面型工具使用）
    40      每个变量：char name[32] + float64 data[ni*nj*nk]（行主序，i 最快）

变量形状由**名字**唯一决定（与 C++ `shape_for_name` 一致）：

* ``u`` / ``u_facex`` -> (nx+1, ny, nz)
* ``v`` / ``v_facey`` -> (nx, ny+1, nz)
* ``w`` / ``zeta`` / ``w_facez`` -> (nx, ny, nz+1)
* ``__restart_info__`` -> (2, 1, 1)
* 其余 -> (nx, ny, nz)

数据**始终以 float64 存储**，``real_kind`` 只记录写者的原生 ``Real`` 类型。

向后兼容扩展（可选，默认关闭）
-------------------------------
``extended=True`` 时每条记录在名字后追加 ``int32[nx,ny,nz]`` 形状与 ``uint32`` CRC32，
并在文件尾附加 ``b"VIBETRL1"`` + ``int32 attr_bytes`` + UTF-8 JSON 属性。
读取器先按 classic 布局解析，若记录的字节数不吻合再尝试 extended，因此**两种布局都能读**；
C++ 读取器只认 classic 布局，所以默认写出的是 classic。

复杂度：读/写均为 O(N)；额外内存 O(1)（逐变量缓冲）。
文献：[D16] WRF ARW 第 5 章（后备二进制格式与 I/O API 设计）。
"""


import json
import os
import struct
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "VIBE_BIN_MAGIC", "VIBE_BIN_NAME_BYTES", "VIBE_BIN_HEADER_BYTES",
    "VibeBinError", "VibeBinHeader", "VibeBinFile",
    "read_vibebin", "write_vibebin", "shape_for_name", "stagger_from_name",
]

#: 文件 magic（8 字节 ASCII）

#: 变量名定长 ASCII 字节数

#: 文件头长度：magic(8) + nx,ny,nz,real_kind(4x int32) + time(float64)
#:             + nvars(int32) + nlevels(int32) = 40

#: 扩展尾部 magic

#: 固定 2 元素的特殊变量




def stagger_from_name(name: str) -> str:
    """变量名 -> 错位类型（与 C++ `stagger_from_name` 一致）。复杂度 O(1)。"""
    if name in ("u", "u_facex"):
        return "face_x"
    if name in ("v", "v_facey"):
        return "face_y"
    if name in ("w", "zeta", "w_facez"):
        return "face_z"
    return "cell"


def shape_for_name(name: str, nx: int, ny: int, nz: int) -> tuple[int, int, int]:
    """变量名 -> `(nx_v, ny_v, nz_v)`（与 C++ `shape_for_name` 一致）。复杂度 O(1)。"""
    if name == _RESTART_INFO:
        return (2, 1, 1)
    kind = stagger_from_name(name)
    if kind == "face_x":
        return (nx + 1, ny, nz)
    if kind == "face_y":
        return (nx, ny + 1, nz)
    if kind == "face_z":
        return (nx, ny, nz + 1)
    return (nx, ny, nz)


def _crc32(array: NDArray[Any]) -> int:
    """变量数据的 CRC32（仅扩展布局使用）。复杂度 O(N)。"""
    return int(zlib.crc32(np.ascontiguousarray(array).tobytes()) & 0xFFFFFFFF)


def _pack_name(name: str) -> bytes:
    """变量名 -> 32 字节定长 ASCII（补 NUL）。复杂度 O(len(name))。"""
    raw = name.encode("ascii", errors="replace")[:VIBE_BIN_NAME_BYTES - 1]
    return raw + b"\x00" * (VIBE_BIN_NAME_BYTES - len(raw))


def _unpack_name(raw: bytes) -> str:
    """32 字节定长 ASCII -> 变量名。复杂度 O(32)。"""
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


@dataclass
class VibeBinHeader:
    """`.vibebin` 文件头。"""

    nx: int = 0
    ny: int = 0
    nz: int = 0
    real_kind: int = 0
    time: float = 0.0
    nvars: int = 0
    nlevels: int = 0

    @classmethod
    def unpack(cls, buffer: bytes) -> "VibeBinHeader":
        """从文件头字节解析。复杂度 O(1)。"""
        if len(buffer) < VIBE_BIN_HEADER_BYTES:
            raise VibeBinError("文件短于 {0} 字节，不是 .vibebin".format(VIBE_BIN_HEADER_BYTES))
        if buffer[:8] != VIBE_BIN_MAGIC:
            raise VibeBinError("magic 不匹配：期望 {0!r}，得到 {1!r}".format(
                VIBE_BIN_MAGIC, buffer[:8]))
        nx, ny, nz, real_kind = struct.unpack_from("<iiii", buffer, 8)
        (time,) = struct.unpack_from("<d", buffer, 24)
        nvars, nlevels = struct.unpack_from("<ii", buffer, 32)
        if min(nx, ny, nz) <= 0 or nvars < 0:
            raise VibeBinError("文件头维度非法：({0}, {1}, {2}), nvars={3}".format(
                nx, ny, nz, nvars))
        return cls(nx=nx, ny=ny, nz=nz, real_kind=real_kind, time=time,
                   nvars=nvars, nlevels=nlevels)

    def pack(self) -> bytes:
        """序列化为 40 字节文件头。复杂度 O(1)。"""
        head = bytearray(VIBE_BIN_MAGIC)
        head += struct.pack("<iiii", self.nx, self.ny, self.nz, self.real_kind)
        head += struct.pack("<d", self.time)
        head += struct.pack("<ii", self.nvars, self.nlevels)
        if len(head) != VIBE_BIN_HEADER_BYTES:
            raise VibeBinError("文件头长度错误：{0} != {1}".format(
                len(head), VIBE_BIN_HEADER_BYTES))
        return bytes(head)


@dataclass
class _Record:
    name: str
    data: NDArray[np.float64]
    checksum: int = 0


class VibeBinFile:
    """`.vibebin` 读写器（C++ `vibe::io::BinaryWriter` 的 Python 对应实现）。

    使用方式::

        with VibeBinFile("state.vibebin", mode="r") as handle:
            dataset = handle.read_dataset()

        handle = VibeBinFile(mode="w")
        handle.set_grid(nx, ny, nz, real_kind=0, time=0.0)
        handle.add_variable("theta", theta_3d)
        handle.write("out.vibebin")

    ``extended=False``（默认）写出与 C++ 字节级一致的 classic 布局；
    ``extended=True`` 追加形状/CRC32 与 JSON 属性尾部。
    复杂度：读写均 O(N)。
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *,
                 mode: str = "r", extended: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.mode = mode
        self.extended = bool(extended)
        self.header = VibeBinHeader()
        self.attrs: dict[str, Any] = {}
        self._records: dict[str, _Record] = {}
        self._buffer: bytes | None = None
        self._layout = "classic"
        self._end_offset = VIBE_BIN_HEADER_BYTES
        self._loaded = False
        if self.path is not None and mode == "r":
            self.read_header()

    # -- 基本信息 ---------------------------------------------------------
    @property
    def nx(self) -> int:
        return self.header.nx

    @property
    def ny(self) -> int:
        return self.header.ny

    @property
    def nz(self) -> int:
        return self.header.nz

    @property
    def real_kind(self) -> int:
        return self.header.real_kind

    @property
    def time(self) -> float:
        return self.header.time

    @property
    def shape(self) -> tuple[int, int, int]:
        """网格形状 `(nz, ny, nx)`。复杂度 O(1)。"""
        return (self.header.nz, self.header.ny, self.header.nx)

    @property
    def variables(self) -> tuple[str, ...]:
        """已载入/待写出的变量名（按插入顺序）。复杂度 O(nvars)。"""
        return tuple(self._records)

    def __contains__(self, name: str) -> bool:
        return name in self._records

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        return self._records[name].data

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __enter__(self) -> "VibeBinFile":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """释放数据（不持有文件句柄）。复杂度 O(1)。"""
        self._buffer = None

    def describe(self) -> str:
        """一行文件摘要。复杂度 O(1)。"""
        kind = "float64" if self.header.real_kind == 0 else "float32"
        name = self.path.name if self.path else "<memory>"
        return "{0}: {1}x{2}x{3} {4}, nvars={5}, time={6:g} s, layout={7}".format(
            name, self.header.nx, self.header.ny, self.header.nz, kind,
            len(self._records), self.header.time, self._layout)

    # -- 解析 -------------------------------------------------------------
    def _try_layout(self, buffer: bytes, layout: str) -> tuple[dict[str, _Record], int]:
        """按指定布局解析全部变量；不吻合则抛 `VibeBinError`。复杂度 O(N)。"""
        header = self.header
        records: dict[str, _Record] = {}
        offset = VIBE_BIN_HEADER_BYTES
        for _ in range(header.nvars):
            if offset + VIBE_BIN_NAME_BYTES > len(buffer):
                raise VibeBinError("记录名被截断")
            name = _unpack_name(buffer[offset:offset + VIBE_BIN_NAME_BYTES])
            offset += VIBE_BIN_NAME_BYTES
            checksum = 0
            if layout == "extended":
                if offset + 16 > len(buffer):
                    raise VibeBinError("扩展记录头被截断")
                nx_v, ny_v, nz_v = struct.unpack_from("<iii", buffer, offset)
                offset += 12
                (checksum,) = struct.unpack_from("<I", buffer, offset)
                offset += 4
                if min(nx_v, ny_v, nz_v) <= 0:
                    raise VibeBinError("扩展记录形状非法：" + str((nx_v, ny_v, nz_v)))
                shape = (nx_v, ny_v, nz_v)
            else:
                if name == _RESTART_INFO:
                    shape = (2, 1, 1)
                else:
                    shape = shape_for_name(name, header.nx, header.ny, header.nz)
            count = int(shape[0]) * int(shape[1]) * int(shape[2])
            nbytes = count * 8          # 数据始终为 float64
            if offset + nbytes > len(buffer):
                raise VibeBinError("变量 {0!r} 数据被截断".format(name))
            data = np.frombuffer(buffer, dtype="<f8", count=count, offset=offset).copy()
            offset += nbytes
            if checksum and _crc32(data) != checksum:
                raise VibeBinError("变量 {0!r} CRC32 校验失败".format(name))
            records[name] = _Record(name, data.reshape((shape[2], shape[1], shape[0])), checksum)
        return records, offset

    def _detect_layout(self, buffer: bytes) -> str:
        """判定 classic / extended 布局并缓存解析结果。复杂度 O(nvars + N)。"""
        if self._loaded:
            return self._layout
        if buffer.rfind(_TRAILER_MAGIC) >= 0:
            order = ["extended", "classic"]
        else:
            order = ["classic", "extended"]
        errors: list[str] = []
        for layout in order:
            try:
                records, end = self._try_layout(buffer, layout)
            except (VibeBinError, ValueError, struct.error) as exc:
                errors.append(str(exc))
                continue
            attrs = self._read_trailer(buffer, end)
            if end != len(buffer) and not attrs:
                # 既不是文件结尾也没有尾部属性 -> 布局判定错误，换另一种
                errors.append("剩余 {0} 字节无法解释".format(len(buffer) - end))
                continue
            self._records = records
            self._layout = layout
            self._end_offset = end
            self.attrs = attrs
            self._loaded = True
            return layout
        raise VibeBinError("无法解析 .vibebin（classic/extended 均失败）：" + "; ".join(errors))

    @staticmethod
    def _read_trailer(buffer: bytes, offset: int) -> dict[str, Any]:
        """读取可选 JSON 尾部属性；不存在时返回空字典。复杂度 O(尾部长度)。"""
        if offset + 12 > len(buffer) or buffer[offset:offset + 8] != _TRAILER_MAGIC:
            return {}
        (length,) = struct.unpack_from("<i", buffer, offset + 8)
        payload = buffer[offset + 12: offset + 12 + max(0, length)]
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}

    # -- 读 ---------------------------------------------------------------
    def read_header(self) -> "VibeBinFile":
        """读文件头 + 解析全部变量。复杂度 O(文件大小)。"""
        if self.path is None:
            raise VibeBinError("未指定文件路径")
        buffer = Path(self.path).read_bytes()
        self.header = VibeBinHeader.unpack(buffer)
        self._buffer = buffer
        self._loaded = False
        self._records = {}
        self.attrs = {}
        self._detect_layout(buffer)
        return self

    def read(self, *, variables: Sequence[str] | None = None) -> "VibeBinFile":
        """已解析全部变量；`variables` 给出时只保留这些。复杂度 O(nvars)。"""
        if not self._loaded:
            self.read_header()
        if variables is not None:
            wanted = set(variables)
            self._records = {k: v for k, v in self._records.items() if k in wanted}
        return self

    def self_check(self) -> list[str]:
        """自检：返回问题列表（空表示通过）。复杂度 O(N)。"""
        problems: list[str] = []
        if self.header.nvars != len(self._records):
            problems.append("nvars={0} 与实际记录数 {1} 不一致".format(
                self.header.nvars, len(self._records)))
        for name, record in self._records.items():
            expected = shape_for_name(name, self.header.nx, self.header.ny, self.header.nz)
            if name != _RESTART_INFO and record.data.shape != (expected[2], expected[1], expected[0]):
                problems.append("{0}: 形状 {1} 与名字隐含的 {2} 不一致".format(
                    name, record.data.shape, (expected[2], expected[1], expected[0])))
            if np.isinf(record.data).any():
                problems.append("{0}: 含 Inf".format(name))
        return problems

    def read_dataset(self, *, variables: Sequence[str] | None = None) -> Any:
        """解析为 `Dataset`（含网格坐标与属性）。复杂度 O(N)。"""
        from .io import Dataset  # 延迟导入，避免模块循环

        if not self._loaded:
            self.read(variables=variables)
        header = self.header
        fields: dict[str, NDArray[np.float64]] = {}
        dims: dict[str, tuple[str, ...]] = {}
        for name, record in self._records.items():
            if name == _RESTART_INFO:
                continue
            fields[name] = record.data
            dims[name] = ("zeta", "y", "x")
        coords: dict[str, NDArray[np.float64]] = {
            "x": (np.arange(header.nx) + 0.5),
            "y": (np.arange(header.ny) + 0.5),
            "zeta": np.arange(header.nz, dtype=np.float64),
        }
        attrs = dict(self.attrs)
        attrs.update({
            "source": str(self.path) if self.path else "memory",
            "nx": header.nx, "ny": header.ny, "nz": header.nz,
            "real_kind": "float64" if header.real_kind == 0 else "float32",
            "layout": self._layout,
        })
        return Dataset(fields=fields, dims=dims, coords=coords, attrs=attrs,
                       time=float(header.time))

    # -- 写 ---------------------------------------------------------------
    def set_grid(self, nx: int, ny: int, nz: int, *, real_kind: int = 0,
                 time: float = 0.0, nlevels: int | None = None) -> None:
        """设置文件头。复杂度 O(1)。"""
        self.header = VibeBinHeader(nx=int(nx), ny=int(ny), nz=int(nz),
                                    real_kind=int(real_kind), time=float(time),
                                    nvars=0, nlevels=int(nz if nlevels is None else nlevels))

    def add_variable(self, name: str, data: ArrayLike) -> None:
        """加入一个变量；形状必须与名字隐含的错位一致。复杂度 O(N)。"""
        arr = np.asarray(data, dtype="<f8")
        if arr.ndim != 3:
            raise VibeBinError("变量 {0!r} 必须是三维 (nz, ny, nx)".format(name))
        nx_v, ny_v, nz_v = shape_for_name(name, self.header.nx, self.header.ny, self.header.nz)
        expected = (nz_v, ny_v, nx_v)
        if arr.shape != expected:
            raise VibeBinError("变量 {0!r} 形状 {1} 与名字隐含的 {2} 不一致".format(
                name, arr.shape, expected))
        stored = np.ascontiguousarray(arr, dtype="<f8")
        self._records[str(name)] = _Record(str(name), stored, _crc32(stored))
        self.header.nvars = len(self._records)

    def write(self, path: str | os.PathLike[str] | None = None, *,
              attrs: Mapping[str, Any] | None = None) -> Path:
        """写出文件。`extended` 由构造参数决定。复杂度 O(N)。"""
        target = Path(path) if path is not None else self.path
        if target is None:
            raise VibeBinError("未指定输出路径")
        if attrs:
            self.attrs.update(dict(attrs))
        self.header.nvars = len(self._records)
        buffer = bytearray(self.header.pack())
        for name, record in self._records.items():
            buffer += _pack_name(name)
            if self.extended:
                nx_v, ny_v, nz_v = shape_for_name(name, self.header.nx, self.header.ny,
                                                 self.header.nz)
                buffer += struct.pack("<iii", nx_v, ny_v, nz_v)
                buffer += struct.pack("<I", _crc32(record.data))
            buffer += np.ascontiguousarray(record.data, dtype="<f8").tobytes()
        if self.extended:
            payload = json.dumps(_jsonable(self.attrs), ensure_ascii=False).encode("utf-8")
            buffer += _TRAILER_MAGIC
            buffer += struct.pack("<i", len(payload))
            buffer += payload
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.write(bytes(buffer))
        self.path = target
        return target

    @classmethod
    def from_dataset(cls, dataset: Any, *, extended: bool = False,
                     real_kind: int | None = None) -> "VibeBinFile":
        """由 `Dataset` 构造（只写出形状合法的三维变量）。复杂度 O(N)。"""
        nx, ny, nz = dataset.grid_shape()
        if real_kind is None:
            real_kind = 1 if str(dataset.attrs.get("real_kind", "")).startswith("float32") else 0
        handle = cls(mode="w", extended=extended)
        handle.set_grid(nx, ny, nz, real_kind=real_kind, time=float(dataset.time))
        handle.attrs = dict(_jsonable(dataset.attrs))
        for name, array in dataset.fields.items():
            arr = np.asarray(array, dtype=np.float64)
            expected = shape_for_name(name, nx, ny, nz)
            if arr.ndim == 3 and arr.shape == (expected[2], expected[1], expected[0]):
                handle.add_variable(name, arr)
        return handle

    def __repr__(self) -> str:  # pragma: no cover - 展示用
        return "VibeBinFile({0}, variables={1})".format(self.describe(), list(self._records))


def read_vibebin(path: str | os.PathLike[str], *,
                 variables: Sequence[str] | None = None) -> Any:
    """读取 `.vibebin` 为 `Dataset`。复杂度 O(N)。"""
    handle = VibeBinFile(path, mode="r")
    handle.read(variables=variables)
    problems = handle.self_check()
    if problems:
        warnings.warn("vibebin 自检警告：{0}".format("; ".join(problems)), stacklevel=2)
    return handle.read_dataset()


def write_vibebin(path: str | os.PathLike[str], dataset: Any, *,
                  real_kind: int | None = None, extended: bool = False,
                  attrs: Mapping[str, Any] | None = None) -> Path:
    """把 `Dataset` 写成 `.vibebin`（默认与 C++ 字节级兼容）。复杂度 O(N)。"""
    handle = VibeBinFile.from_dataset(dataset, extended=extended, real_kind=real_kind)
    return handle.write(path, attrs=attrs)


def read_binary_header(path: str | os.PathLike[str]) -> VibeBinHeader:
    """只读 `.vibebin` 文件头（与 C++ `io::read_binary_header` 对应）。复杂度 O(1)。"""
    with open(path, "rb") as handle:
        head = handle.read(VIBE_BIN_HEADER_BYTES)
    return VibeBinHeader.unpack(head)

def _normalize_dataset(data: Any, source: str) -> Dataset:
    """把 xarray.Dataset 归一化为 :class:`Dataset`（维度名/顺序统一）。

    复杂度：O(N)（需要转置时复制一次；否则零拷贝视图）。
    """
    fields: dict[str, NDArray[Any]] = {}
    dims: dict[str, tuple[str, ...]] = {}
    coords: dict[str, NDArray[Any]] = {}
    for name, variable in data.variables.items():
        array = np.asarray(variable.values)
        logical = normalize_dims(tuple(variable.dims))
        if name in data.coords:
            coords[name] = array
            continue
        if array.ndim == 0:
            fields[name] = array
            dims[name] = ()
            continue
        # xarray 维度顺序 = numpy 数组顺序（最慢在前）；要求 zeta 在最慢侧
        order = list(range(array.ndim))
        if "zeta" in logical:
            slow_index = 0 if logical[0] != "time" else 1
            zeta_index = logical.index("zeta")
            if zeta_index != slow_index:
                order.insert(slow_index, order.pop(zeta_index))
                array = np.transpose(array, order)
                logical = [logical[o] for o in order]
        fields[name] = array
        dims[name] = tuple(logical)
    attrs = {str(k): _jsonable(v) for k, v in dict(data.attrs).items()}
    attrs["source"] = source
    time = 0.0
    if "time" in coords:
        coord = np.asarray(coords["time"]).ravel()
        if coord.size:
            raw = coord[0]
            try:
                time = float(raw)
            except (TypeError, ValueError):
                time = 0.0
            if np.issubdtype(np.asarray(raw).dtype, np.datetime64):
                attrs["time_iso"] = str(np.datetime_as_string(np.asarray(raw), unit="s"))
    return Dataset(fields=fields, dims=dims, coords=coords, attrs=attrs, time=time)


def read_netcdf(path: str | os.PathLike[str]) -> Dataset:
    """读取 NetCDF（xarray + netCDF4/h5netcdf）。复杂度 O(N)。"""
    xarray = _require("xarray", "读取 NetCDF", "netcdf")
    try:
        data = xarray.open_dataset(path, engine="netcdf4")
    except (ImportError, ModuleNotFoundError, ValueError):
        try:
            data = xarray.open_dataset(path)
        except Exception as exc:  # pragma: no cover - 环境相关
            raise MissingDependencyError("netCDF4", "读取 NetCDF", "netcdf") from exc
    try:
        dataset = _normalize_dataset(data, str(path))
    finally:
        data.close()
    return dataset


def to_netcdf(path: str | os.PathLike[str], dataset: Dataset, *,
              unlimited_dims: Sequence[str] = (), compress: bool = False) -> Path:
    """把 :class:`Dataset` 写成 NetCDF。复杂度 O(N)。"""
    target = Path(path)
    frame = dataset.to_xarray()
    encoding = {}
    if compress:
        for name in frame.data_vars:
            encoding[name] = {"zlib": True, "complevel": 4}
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"encoding": encoding} if encoding else {}
    if unlimited_dims:
        kwargs["unlimited_dims"] = list(unlimited_dims)
    frame.to_netcdf(target, **kwargs)
    return target


# ---------------------------------------------------------------------------
# GRIB2 后端
# ---------------------------------------------------------------------------


def read_grib(path: str | os.PathLike[str], *, filter_by_keys: Mapping[str, Any] | None = None,
              rename: Mapping[str, str] | None = None) -> Dataset:
    """读取 GRIB2（cfgrib/eccodes），用于侧边界与初始场。

    变量名按 cfgrib 的 `shortName` 或 `cfVarName` 命名；可用 `rename` 映射到
    VIBE 的规范名（例如 `{"t": "theta"}`）。复杂度 O(N)。
    """
    cfgrib = _require("cfgrib", "读取 GRIB2", "grib")
    xarray = _require("xarray", "读取 GRIB2", "grib")
    backend_kwargs: dict[str, Any] = {"indexpath": ""}
    if filter_by_keys:
        backend_kwargs["filter_by_keys"] = dict(filter_by_keys)
    try:
        data = xarray.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)
    except Exception as exc:
        raise UnsupportedFormatError("cfgrib 无法解析 {0}：{1}".format(path, exc)) from exc
    try:
        dataset = _normalize_dataset(data, str(path))
    finally:
        data.close()
    dataset.attrs["format"] = "grib2"
    if rename:
        for old, new in rename.items():
            if old in dataset.fields:
                dataset.fields[new] = dataset.fields.pop(old)
                dataset.dims[new] = dataset.dims.pop(old)
    return dataset


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


def open_dataset(path: str | os.PathLike[str], *, backend: str = "auto",
                 variables: Sequence[str] | None = None,
                 engine: str | None = None) -> Dataset:
    """打开 VIBE 输出文件，自动选择后端。

    参数
    ----
    backend : {"auto", "netcdf", "grib", "vibebin"}
        `"auto"` 时按 :func:`sniff_format` 判定，失败则依次尝试其余后端。
    variables : 序列, 可选
        只读取这些变量（`.vibebin` 后端支持真正的按需读取；NetCDF/GRIB 后端
        在归一化后做子集选择）。
    engine : str, 可选
        `"netcdf"` / `"grib"` 时忽略；保留给未来使用（如 `"h5netcdf"`）。

    复杂度：O(N)；内存 O(N)。
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError("文件不存在：{0}".format(target))
    resolved = sniff_format(target) if backend == "auto" else backend
    order = [resolved] + [b for b in ("netcdf", "grib", "vibebin") if b != resolved]

    errors: list[str] = []
    for candidate in order:
        try:
            if candidate == "vibebin":
                dataset = read_vibebin(target, variables=variables)
            elif candidate == "netcdf":
                dataset = read_netcdf(target)
            elif candidate == "grib":
                dataset = read_grib(target)
            else:
                raise UnsupportedFormatError("未知后端：{0}".format(candidate))
            if variables is not None and candidate != "vibebin":
                subset = {k: v for k, v in dataset.fields.items() if k in set(variables)}
                dataset = Dataset(fields=subset,
                                  dims={k: dataset.dims[k] for k in subset},
                                  coords=dict(dataset.coords), attrs=dict(dataset.attrs),
                                  time=dataset.time)
            dataset.attrs.setdefault("backend", candidate)
            return dataset
        except MissingDependencyError as exc:
            errors.append("{0}: {1}".format(candidate, str(exc).splitlines()[0]))
            if backend != "auto":
                raise
        except (UnsupportedFormatError, VibeBinError, ValueError, OSError) as exc:
            errors.append("{0}: {1}".format(candidate, exc))
            if backend != "auto":
                raise UnsupportedFormatError(
                    "以 {0} 后端读取 {1} 失败：{2}".format(backend, target, exc)) from exc
    raise UnsupportedFormatError(
        "无法以任何后端读取 {0}：\n  - {1}".format(target, "\n  - ".join(errors)))

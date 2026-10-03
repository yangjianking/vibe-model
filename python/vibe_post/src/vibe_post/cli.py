"""`vibe-post` 命令行接口。

子命令
------
.. code-block:: text

    vibe-post info   <file> [--backend auto]
    vibe-post interp <file> --var u --to-height 5000 --out out.nc
    vibe-post derive <file> --list | --diag cape --out out.nc
    vibe-post verify --forecast f.nc --obs o.nc --var t2 --method rmse
    vibe-post plot   <file> --var theta --level 10 --out fig.png

设计约定
--------
* 只依赖标准库 argparse 与 numpy；NetCDF/绘图后端按需惰性导入；
* 子命令的 `--help` 给出完整用法与示例；
* 输出文件按扩展名选择格式（`.nc` -> NetCDF，`.vibebin` -> 二进制）；
* 沿用 docs/design/00_architecture.md 的变量命名（u, v, w, rho, theta, pi, qv…）。

文献引用见 docs/design/10_postprocessing.md。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import __version__
from .io import Dataset, open_dataset, to_netcdf, write_vibebin

#: 诊断名 -> CLI 关键字（与 diagnostics.list_diagnostics 的键一致）
_DIAG_CHOICES = (
    "wind_speed", "wind_direction", "dewpoint", "relative_humidity",
    "temperature_from_exner", "pressure_from_exner", "theta_e", "geopotential",
    "vorticity", "divergence", "absolute_vorticity", "potential_vorticity",
    "cape", "srh", "slpm", "reflectivity", "dbz", "brightness_temperature",
    "k_index", "showalter_index", "accum",
)


def _build_parser() -> argparse.ArgumentParser:
    """构造 argparse 解析器（含全部子命令）。复杂度 O(1)。"""
    parser = argparse.ArgumentParser(
        prog="vibe-post",
        description="VIBE-Model 后处理：文件摘要、插值、诊断、检验与绘图。",
        epilog="示例：vibe-post derive run/wrfout.nc --diag cape --out cape.nc",
    )
    parser.add_argument("--version", action="version", version="vibe-post {0}".format(__version__))
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<命令>")

    info = subparsers.add_parser("info", help="打印文件/变量/网格摘要")
    info.add_argument("file", help="输入文件（.nc / .grib2 / .vibebin）")
    info.add_argument("--backend", default="auto",
                      choices=("auto", "netcdf", "grib", "vibebin"),
                      help="强制后端（默认自动探测）")
    info.add_argument("--json", action="store_true", help="以 JSON 输出便于脚本解析")

    interp = subparsers.add_parser("interp", help="水平/垂直插值")
    interp.add_argument("file", help="输入文件")
    interp.add_argument("--var", default="theta", help="要插值的变量名（默认 theta）")
    interp.add_argument("--to-height", type=float, default=None, metavar="Z",
                        help="插值到给定高度 [m]")
    interp.add_argument("--to-pressure", type=float, default=None, metavar="P",
                        help="插值到给定气压 [Pa]")
    interp.add_argument("--x", type=float, default=None, help="水平插值目标 x [m]")
    interp.add_argument("--y", type=float, default=None, help="水平插值目标 y [m]")
    interp.add_argument("--radius", type=float, default=None, metavar="R",
                        help="点插值模式的半径 [m]（给出则输出径向平均剖面）")
    interp.add_argument("--out", default=None, help="输出文件（.nc 或 .vibebin）")
    interp.add_argument("--backend", default="auto",
                        choices=("auto", "netcdf", "grib", "vibebin"))

    derive = subparsers.add_parser("derive", help="诊断量计算")
    derive.add_argument("file", help="输入文件")
    derive.add_argument("--list", action="store_true", help="列出可用诊断量后退出")
    derive.add_argument("--diag", default=None, choices=_DIAG_CHOICES,
                        help="要计算的诊断量")
    derive.add_argument("--var", default=None, help="输出变量名（默认同诊断名）")
    derive.add_argument("--out", default=None, help="输出文件（.nc 或 .vibebin）")
    derive.add_argument("--backend", default="auto",
                        choices=("auto", "netcdf", "grib", "vibebin"))

    verify = subparsers.add_parser("verify", help="预报检验")
    verify.add_argument("--forecast", required=True, help="预报文件")
    verify.add_argument("--obs", required=True, help="观测文件（网格或站点）")
    verify.add_argument("--var", default="t2", help="预报变量名（默认 t2）")
    verify.add_argument("--obs-var", default=None, help="观测变量名（默认同 --var）")
    verify.add_argument("--method", default="rmse",
                        choices=("bias", "mae", "rmse", "correlation", "acc",
                                 "ets", "ts", "csi", "pod", "far", "fss",
                                 "brier", "bss", "crps", "spread_skill", "all"),
                        help="检验方法")
    verify.add_argument("--threshold", type=float, default=None,
                        help="分类/邻域检验阈值")
    verify.add_argument("--window", type=int, default=1, help="FSS 邻域窗口（奇数）")
    verify.add_argument("--clim", default=None, help="气候态文件（用于 ACC）")
    verify.add_argument("--out", default=None, help="检验结果 JSON 输出路径")
    verify.add_argument("--backend", default="auto",
                        choices=("auto", "netcdf", "grib", "vibebin"))

    plot = subparsers.add_parser("plot", help="绘图")
    plot.add_argument("file", help="输入文件")
    plot.add_argument("--var", default="theta", help="变量名（默认 theta）")
    plot.add_argument("--kind", default="field",
                      choices=("field", "section", "profile", "skewt", "spectra", "timeseries"),
                      help="图类型")
    plot.add_argument("--level", type=int, default=0, help="垂直层号（field 模式）")
    plot.add_argument("--j", type=int, default=None, help="剖面固定 y 索引")
    plot.add_argument("--i", type=int, default=None, help="剖面固定 x 索引")
    plot.add_argument("--out", default="vibe_post.png", help="输出图片路径")
    plot.add_argument("--cmap", default=None, help="matplotlib 色标名")
    plot.add_argument("--backend", default="auto",
                      choices=("auto", "netcdf", "grib", "vibebin"))
    return parser


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _cell_fields(dataset: Dataset) -> dict[str, np.ndarray]:
    """把数据集里的场统一到体心视图（错位场取回体心并裁剪到体心网格）。

    `.vibebin` 按 C++ 约定把 u/v/w 存成面场（各多一层/一行/一列），而诊断量
    要求所有输入同形。约定：u[i] 与体心 i **共享同一物理位置**（C++
    `Grid::index` 语义），因此直接取前 nx 列；w 在层界面，取相邻界面的平均。
    复杂度：O(N)。
    """
    fields = {name: np.asarray(value, dtype=np.float64)
              for name, value in dataset.fields.items()}
    reference = fields.get("theta")
    if reference is None or reference.ndim != 3:
        return fields
    nz, ny, nx = reference.shape
    out: dict[str, np.ndarray] = {}
    for name, arr in fields.items():
        if arr.ndim != 3:
            out[name] = arr
            continue
        if arr.shape == (nz + 1, ny, nx):          # w / zeta（层界面）
            out[name] = 0.5 * (arr[:-1] + arr[1:])
        elif arr.shape == (nz, ny, nx + 1):        # u（x 面）
            out[name] = arr[..., :nx]
        elif arr.shape == (nz, ny + 1, nx):        # v（y 面）
            out[name] = arr[:, :ny, :]
        elif arr.shape == (nz, ny, nx):
            out[name] = arr
        else:
            out[name] = arr
    return out

def _write_dataset(dataset: Dataset, path: str | None) -> str | None:
    """把 Dataset 写出到 `path`（.nc / .vibebin / .json），返回实际路径。复杂度 O(N)。"""
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    if suffix in (".nc", ".nc4", ".cdf"):
        to_netcdf(path, dataset)
    elif suffix in (".vibebin", ".bin", ".vbin"):
        write_vibebin(path, dataset)
    elif suffix == ".json":
        payload = {name: np.asarray(value).tolist() for name, value in dataset.fields.items()}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    else:
        raise ValueError("不支持的输出扩展名 {0!r}（用 .nc / .vibebin / .json）".format(suffix))
    return path


def _grid_of(dataset: Dataset):
    """由 Dataset 构造 :class:`GridSpec`。复杂度 O(N)。"""
    from .grid import GridSpec
    return GridSpec.from_dataset(dataset)


def _level_coordinate(dataset: Dataset, grid: Any) -> np.ndarray:
    """返回三维层中心高度 [m]。复杂度 O(nx ny nz)。"""
    return grid.height_array()


def _pressure_field(dataset: Dataset, grid: Any) -> np.ndarray:
    """确定三维气压场 [Pa]：优先 `p`/`pres`，其次 Exner 函数，最后静力近似。

    复杂度：O(nx ny nz)。
    """
    from . import diagnostics as dg

    for name in ("p", "pres", "pressure", "P"):
        if name in dataset.fields and np.asarray(dataset.fields[name]).ndim == 3:
            return np.asarray(dataset.fields[name], dtype=np.float64)
    if "pi" in dataset.fields:
        return dg.pressure_from_exner(dataset.fields["pi"])
    if "theta" in dataset.fields:
        # 静力 + 理想气体：p = p_sfc * exp(-g z /(Rd T_mean))
        theta = np.asarray(dataset.fields["theta"], dtype=np.float64)
        height = _level_coordinate(dataset, grid)
        p_sfc = float(dataset.attrs.get("p_sfc", 101325.0))
        mean_t = np.mean(theta, axis=0, keepdims=True)
        return p_sfc * np.exp(-dg.GRAVITY * height / (dg.RD * np.maximum(mean_t, 200.0)))
    raise KeyError("数据集中缺少 pi 或 p，无法建立气压坐标")


def _temperature_field(dataset: Dataset) -> np.ndarray | None:
    """取出三维温度 [K]（优先 `t`/`tk`，其次由位温与 Exner 反算）。复杂度 O(N)。"""
    from . import diagnostics as dg

    for name in ("t", "tk", "temp", "temperature", "T"):
        if name in dataset.fields and np.asarray(dataset.fields[name]).ndim == 3:
            return np.asarray(dataset.fields[name], dtype=np.float64)
    if "theta" in dataset.fields and "pi" in dataset.fields:
        return dg.temperature_from_exner(dataset.fields["pi"], dataset.fields["theta"])
    return None


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    """`vibe-post info`：打印文件/变量/网格摘要。复杂度 O(N)。"""
    from .io import available_backends, sniff_format, summarize, VariableInfo

    dataset = open_dataset(args.file, backend=args.backend)
    if args.json:
        payload = {
            "file": str(args.file),
            "format": sniff_format(args.file),
            "time": dataset.time,
            "attrs": {k: str(v) for k, v in dataset.attrs.items()},
            "grid": list(dataset.grid_shape()),
            "variables": {
                name: {
                    "shape": list(dataset.fields[name].shape),
                    "dims": list(dataset.dims.get(name, ())),
                    "dtype": str(dataset.fields[name].dtype),
                    "min": VariableInfo.of(name, dataset.fields[name]).vmin,
                    "max": VariableInfo.of(name, dataset.fields[name]).vmax,
                    "nan": VariableInfo.of(name, dataset.fields[name]).nan_count,
                }
                for name in sorted(dataset.fields)
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print("后端可用性: {0}".format(available_backends()))
    print(summarize(dataset))
    return 0


def cmd_interp(args: argparse.Namespace) -> int:
    """`vibe-post interp`：水平 / 垂直插值。复杂度 O(N)。"""
    from . import interp as ip

    dataset = open_dataset(args.file, backend=args.backend)
    if args.var not in dataset.fields:
        raise KeyError("文件里没有变量 {0!r}；可用：{1}".format(args.var, sorted(dataset.fields)))
    field = np.asarray(dataset.fields[args.var], dtype=np.float64)
    grid = _grid_of(dataset)
    out = dataset.copy(deep=False)

    if args.x is not None and args.y is not None:
        if field.ndim == 3:
            height = _level_coordinate(dataset, grid)
            values = ip.trilinear(field, args.x, args.y, height, height,
                                  x0=grid.x0, y0=grid.y0, dx=grid.dx, dy=grid.dy)
        else:
            values = ip.bilinear(field, args.x, args.y, x0=grid.x0, y0=grid.y0,
                                 dx=grid.dx, dy=grid.dy)
        print("{0}(x={1:.1f} m, y={2:.1f} m) = {3}".format(args.var, args.x, args.y,
                                                           np.asarray(values)))
        return 0

    if field.ndim != 3:
        raise ValueError("垂直插值需要三维场 (nz, ny, nx)，得到 {0} 维".format(field.ndim))
    height = _level_coordinate(dataset, grid)
    name = "{0}_interp".format(args.var)
    if args.to_height is not None:
        result = ip.column_interp_to_height(height, field, float(args.to_height))
        out = out.with_field(name, result, dims=("y", "x"))
        out.attrs["interp_target"] = "height={0} m".format(args.to_height)
        print("插值到高度 {0} m，结果形状 {1}".format(args.to_height, result.shape))
    elif args.to_pressure is not None:
        pressure = _pressure_field(dataset, grid)
        result = ip.column_interp_to_pressure(pressure, field, float(args.to_pressure))
        out = out.with_field(name, result, dims=("y", "x"))
        out.attrs["interp_target"] = "pressure={0} Pa".format(args.to_pressure)
        print("插值到气压 {0} Pa，结果形状 {1}".format(args.to_pressure, result.shape))
    else:
        raise SystemExit("请给出 --to-height、--to-pressure 或 --x/--y 之一")

    written = _write_dataset(out, args.out)
    if written:
        print("已写出 {0}".format(written))
    return 0


def cmd_derive(args: argparse.Namespace) -> int:
    """`vibe-post derive`：诊断量计算。复杂度 O(N)（CAPE 为 O(nz n_iter N)）。"""
    from . import diagnostics as dg
    from . import interp as ip

    if args.list:
        for name, description in sorted(dg.list_diagnostics().items()):
            print("{0:<28s} {1}".format(name, description))
        return 0
    if args.diag is None:
        raise SystemExit("请给出 --diag <诊断名> 或 --list")

    dataset = open_dataset(args.file, backend=args.backend)
    grid = _grid_of(dataset)
    fields = _cell_fields(dataset)
    diag = args.diag
    name = args.var or diag
    out = dataset.copy(deep=False)
    result: Any

    if diag in ("wind_speed", "wind_direction"):
        if "u" not in fields or "v" not in fields:
            raise KeyError("需要 u 与 v")
        function = dg.wind_speed if diag == "wind_speed" else dg.wind_direction
        result = function(fields["u"], fields["v"])
        out = out.with_field(name, result, dims=dataset.dims.get("u"))
    elif diag in ("dewpoint", "relative_humidity", "theta_e"):
        if "pi" not in fields or "theta" not in fields or "qv" not in fields:
            raise KeyError("需要 pi、theta 与 qv")
        pressure = _pressure_field(dataset, grid)
        temperature = dg.temperature_from_exner(fields["pi"], fields["theta"])
        if diag == "dewpoint":
            result = dg.dewpoint(pressure, fields["qv"])
        elif diag == "relative_humidity":
            result = dg.relative_humidity(pressure, temperature, fields["qv"])
        else:
            result = dg.theta_e(pressure, temperature, fields["qv"])
        out = out.with_field(name, result, dims=dataset.dims.get("theta"))
    elif diag == "temperature_from_exner":
        result = dg.temperature_from_exner(fields["pi"], fields["theta"])
        out = out.with_field(name, result, dims=dataset.dims.get("theta"))
    elif diag == "pressure_from_exner":
        result = dg.pressure_from_exner(fields["pi"])
        out = out.with_field(name, result, dims=dataset.dims.get("pi"))
    elif diag == "geopotential":
        result = dg.geopotential(_level_coordinate(dataset, grid))
        out = out.with_field(name, result, dims=("zeta", "y", "x"))
    elif diag in ("vorticity", "divergence"):
        if "u" not in fields or "v" not in fields:
            raise KeyError("需要 u 与 v")
        function = dg.vorticity if diag == "vorticity" else dg.divergence
        result = function(fields["u"], fields["v"], grid.dx, grid.dy)
        out = out.with_field(name, result, dims=dataset.dims.get("u"))
    elif diag in ("absolute_vorticity", "potential_vorticity"):
        if "u" not in fields or "v" not in fields or "theta" not in fields:
            raise KeyError("需要 u、v 与 theta")
        if diag == "absolute_vorticity":
            result = dg.absolute_vorticity(fields["u"], fields["v"],
                                           float(dataset.attrs.get("lat", 45.0)),
                                           grid.dx, grid.dy)
        else:
            result = dg.potential_vorticity(
                fields["theta"], fields["u"], fields["v"],
                p=_pressure_field(dataset, grid),
                qv=fields.get("qv"),
                z=_level_coordinate(dataset, grid),
                lat=float(dataset.attrs.get("lat", 45.0)),
                dx=grid.dx, dy=grid.dy)
        out = out.with_field(name, result, dims=dataset.dims.get("theta"))
    elif diag in ("cape",):
        needed = ("theta", "pi", "qv")
        if any(key not in fields for key in needed):
            raise KeyError("CAPE 需要 theta、pi 与 qv")
        temperature = dg.temperature_from_exner(fields["pi"], fields["theta"])
        pressure = _pressure_field(dataset, grid)
        height = _level_coordinate(dataset, grid)
        result = dg.cape_cin(height, pressure, temperature, fields["qv"])
        out = out.with_field(name, np.asarray(result["cape"]), dims=("y", "x"))
        out = out.with_field("{0}_cin".format(name), np.asarray(result["cin"]), dims=("y", "x"))
        out = out.with_field("{0}_lfc".format(name), np.asarray(result["lfc"]), dims=("y", "x"))
        out = out.with_field("{0}_el".format(name), np.asarray(result["el"]), dims=("y", "x"))
    elif diag in ("srh",):
        if "u" not in fields or "v" not in fields:
            raise KeyError("SRH 需要 u 与 v")
        result = dg.storm_relative_helicity(_level_coordinate(dataset, grid),
                                            fields["u"], fields["v"])
        out = out.with_field(name, np.asarray(result["srh"]), dims=("y", "x"))
    elif diag in ("slpm",):
        if "pi" not in fields or "theta" not in fields:
            raise KeyError("SLP 需要 pi 与 theta")
        height = _level_coordinate(dataset, grid)
        pressure = _pressure_field(dataset, grid)
        temperature = dg.temperature_from_exner(fields["pi"], fields["theta"])
        result = dg.sea_level_pressure(pressure[0], temperature[0], height[0])
        out = out.with_field(name, result, dims=("y", "x"))
    elif diag in ("reflectivity", "dbz"):
        if "qr" not in fields:
            raise KeyError("反射率需要 qr（可选 qs、qg）")
        density = fields.get("rho", np.ones_like(fields["qr"]))
        result = dg.reflectivity_dbz(fields["qr"], fields.get("qs"), fields.get("qg"),
                                     density, fields.get("t"))
        out = out.with_field(name, result, dims=dataset.dims.get("qr"))
    elif diag == "brightness_temperature":
        key = "olr" if "olr" in fields else ("flux" if "flux" in fields else None)
        if key is None:
            raise KeyError("亮度温度需要 olr 或 flux 变量")
        result = dg.brightness_temperature(fields[key])
        out = out.with_field(name, result, dims=dataset.dims.get(key))
    elif diag in ("k_index", "showalter_index"):
        pressure = _pressure_field(dataset, grid)
        temperature = _temperature_field(dataset)
        if temperature is None:
            raise KeyError("需要温度场（t 或 theta+pi）")
        qv = fields.get("qv")
        if qv is None:
            raise KeyError("需要 qv")
        height = _level_coordinate(dataset, grid)
        if diag == "k_index":
            t850 = ip_level(ip,
                            pressure=pressure, values=temperature, target=85000.0)
            td850 = ip_level(ip,
                             pressure=pressure, values=dg.dewpoint(pressure, qv), target=85000.0)
            t500 = ip_level(ip,
                            pressure=pressure, values=temperature, target=50000.0)
            t700 = ip_level(ip,
                            pressure=pressure, values=temperature, target=70000.0)
            td700 = ip_level(ip,
                             pressure=pressure, values=dg.dewpoint(pressure, qv), target=70000.0)
            result = dg.k_index(t850, t500, td850, t700, td700)
        else:
            t850 = ip_level(ip,
                            pressure=pressure, values=temperature, target=85000.0)
            td850 = ip_level(ip,
                             pressure=pressure, values=dg.dewpoint(pressure, qv), target=85000.0)
            t500 = ip_level(ip,
                            pressure=pressure, values=temperature, target=50000.0)
            result = dg.showalter_index(t850, td850, t500)
        out = out.with_field(name, result, dims=("y", "x"))
    elif diag == "accum":
        key = None
        for candidate in ("rain", "rainc", "rainnc", "pr", "precip"):
            if candidate in fields:
                key = candidate
                break
        if key is None:
            raise KeyError("降水累积需要 rain / rainc / rainnc 之一")
        dt = float(dataset.attrs.get("output_interval", 3600.0))
        result = dg.precipitation_accumulation(fields[key], dt)
        out = out.with_field(name, result, dims=dataset.dims.get(key))
    else:
        raise SystemExit("未知诊断量 {0!r}".format(diag))

    written = _write_dataset(out, args.out)
    print("诊断 {0} 完成，形状 {1}".format(diag, np.asarray(result).shape))
    if written:
        print("已写出 {0}".format(written))
    return 0


def ip_level(interp_module: Any, pressure: np.ndarray, values: np.ndarray,
             target: float) -> np.ndarray:
    """把三维场逐列插值到目标气压层（cmd_derive 内的便捷包装）。复杂度 O(N)。"""
    return interp_module.column_interp_to_pressure(pressure, values, float(target))


def cmd_verify(args: argparse.Namespace) -> int:
    """`vibe-post verify`：计算指定的检验评分。复杂度 O(N)。"""
    from . import verify as vf

    forecast = open_dataset(args.forecast, backend=args.backend)
    obs = open_dataset(args.obs, backend=args.backend)
    obs_var = args.obs_var or args.var
    if args.var not in forecast.fields:
        raise KeyError("预报文件缺少变量 {0!r}".format(args.var))
    if obs_var not in obs.fields:
        raise KeyError("观测文件缺少变量 {0!r}".format(obs_var))
    f = _cell_fields(forecast)[args.var]
    o = _cell_fields(obs)[obs_var]
    threshold = args.threshold
    if threshold is None:
        threshold = float(np.nanpercentile(o, 90.0)) if o.size else 0.0

    results: dict[str, Any] = {"var": args.var, "n_forecast": int(f.size),
                               "n_obs": int(o.size), "threshold": threshold}
    method = args.method
    if method in ("bias",):
        results["bias"] = vf.bias(f, o)
    elif method == "mae":
        results["mae"] = vf.mae(f, o)
    elif method == "rmse":
        results["rmse"] = vf.rmse(f, o)
    elif method == "correlation":
        results["correlation"] = vf.correlation(f, o)
    elif method == "acc":
        if args.clim is None:
            raise SystemExit("--method acc 需要 --clim 气候态文件")
        clim = open_dataset(args.clim, backend=args.backend)
        field = clim.fields.get(args.var)
        if field is None:
            raise KeyError("气候态文件缺少变量 {0!r}".format(args.var))
        results["acc"] = vf.anomaly_correlation(f, o, field)
    elif method in ("ets", "ts", "csi", "pod", "far"):
        table = vf.contingency_table(f, o, threshold)
        results.update({k: float(v) for k, v in table.items()})
        function = {"ets": vf.ets, "csi": vf.csi,
                    "pod": vf.pod, "far": vf.far}[method]
        results[method] = function(f, o, threshold)
    elif method == "fss":
        if f.ndim < 2:
            raise SystemExit("FSS 需要二维或更高维的网格场")
        plane_f = f[-1, :, :] if f.ndim > 2 else f
        plane_o = o[-1, :, :] if o.ndim > 2 else o
        results["fss"] = vf.fractions_skill_score(plane_f, plane_o, threshold, args.window)
    elif method in ("brier", "bss"):
        probability = np.clip((f - np.nanmin(f)) / max(np.nanmax(f) - np.nanmin(f), 1e-12), 0, 1)
        binary = (np.nan_to_num(o, nan=0.0) >= threshold).astype(np.float64)
        results["brier"] = vf.brier_score(probability, binary)
        if method == "bss":
            results["bss"] = vf.brier_skill_score(probability, binary)
    elif method in ("crps", "spread_skill"):
        if f.ndim < 3:
            raise SystemExit("{0} 需要集合维（至少三维 (M, ...)）".format(method))
        if method == "crps":
            results["crps"] = float(np.nanmean(vf.crps_ensemble(f, o)))
        else:
            results["spread_skill_ratio"] = vf.spread_skill_ratio(f, o)
    elif method == "all":
        results["bias"] = vf.bias(f, o)
        results["mae"] = vf.mae(f, o)
        results["rmse"] = vf.rmse(f, o)
        results["correlation"] = vf.correlation(f, o)
        results.update({k: float(v) for k, v in vf.contingency_table(f, o, threshold).items()})
        for name, function in (("pod", vf.pod), ("far", vf.far), ("csi", vf.csi),
                               ("ets", vf.ets), ("tss", vf.tss),
                               ("frequency_bias", vf.frequency_bias), ("orss", vf.orss)):
            results[name] = function(f, o, threshold)
        if f.ndim >= 2 and min(f.shape[-2:]) >= args.window:
            results["fss"] = vf.fractions_skill_score(f[-1, :, :], o[-1, :, :], threshold, args.window)
    else:  # pragma: no cover - argparse 已限制取值
        raise SystemExit("未知方法 {0!r}".format(method))

    text = json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                       for k, v in results.items()}, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("已写出 {0}".format(args.out))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """`vibe-post plot`：绘图并保存图片。复杂度 O(N)（谱为 O(N log N)）。"""
    from . import diagnostics as dg
    from . import plots as pl

    dataset = open_dataset(args.file, backend=args.backend)
    if args.var not in dataset.fields:
        raise KeyError("文件里没有变量 {0!r}".format(args.var))
    grid = _grid_of(dataset)
    kind = args.kind
    cmap = args.cmap

    if kind in ("field",):
        kwargs = {"cmap": cmap} if cmap else {}
        fig, _ = pl.plot_field(dataset, args.var, level=args.level,
                               title="{0} @ level {1}".format(args.var, args.level), **kwargs)
    elif kind == "section":
        fig, _ = pl.plot_vertical_section(
            dataset, args.var, j=args.j, i=args.i,
            z=grid.height_array(),
            title="{0} 垂直剖面".format(args.var))
    elif kind == "profile":
        pressure = _pressure_field(dataset, grid)
        column = np.asarray(dataset.fields[args.var], dtype=np.float64)[:, 0, 0]
        fig, _ = pl.plot_profile(column, pressure[:, 0, 0],
                                 xlabel=args.var, label=args.var)
    elif kind == "skewt":
        pressure = _pressure_field(dataset, grid)
        temperature = _temperature_field(dataset)
        if temperature is None:
            raise KeyError("skew-T 需要温度场")
        qv = dataset.fields.get("qv")
        dewpoint = dg.dewpoint(pressure, qv) if qv is not None else None
        fig, _ = pl.plot_skewt(pressure[:, 0, 0], temperature[:, 0, 0],
                               None if dewpoint is None else dewpoint[:, 0, 0],
                               z=grid.height_array()[:, 0, 0])
    elif kind == "spectra":
        fig, _ = pl.plot_spectra(np.asarray(dataset.fields[args.var])[-1], args.var,
                                 dx=grid.dx)
    elif kind == "timeseries":
        field = np.asarray(dataset.fields[args.var], dtype=np.float64)
        time = np.asarray(dataset.coords.get("time", np.arange(field.shape[0])), dtype=np.float64)
        fig, _ = pl.plot_timeseries(time / 3600.0, field.reshape(field.shape[0], -1)[:, 0],
                                    xlabel="时间 [h]", ylabel=args.var)
    else:  # pragma: no cover
        raise SystemExit("未知图类型 {0!r}".format(kind))

    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("已写出 {0}".format(args.out))
    return 0


_COMMANDS = {
    "info": cmd_info,
    "interp": cmd_interp,
    "derive": cmd_derive,
    "verify": cmd_verify,
    "plot": cmd_plot,
}


def main(argv: Sequence[str] | None = None) -> int:
    """`vibe-post` 入口。复杂度 O(N)。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse required=True 已兜底
        parser.error("未知命令 {0!r}".format(args.command))
        return 2
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

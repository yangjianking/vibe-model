"""VIBE-Model 后处理包根模块。

本包与 C++ 侧 `vibe::io` / `vibe::verify` 一一对应，但**不依赖 C++ 运行时**：
所有数据通过文件交换（NetCDF / GRIB2 / `.vibebin`）。

设计要点
--------
* 数据模型：唯一的交换对象是 :class:`vibe_post.io.Dataset`（fields + dims + coords + attrs + time）。
* 惰性依赖：xarray / cfgrib / matplotlib / PyYAML 全部按需导入，缺失时抛出
  :class:`vibe_post.io.MissingDependencyError`，便于在最小环境中只使用 numpy 后端。
* 无状态诊断：:mod:`vibe_post.diagnostics` 全部是纯函数，输入/输出 numpy 数组，
  便于单元测试，也便于与 GPU 内核逐点对齐。

文献编号（如 `[D5]`、`[E7]`）见 `docs/design/references.md`。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

#: 与 C++ `io::BinaryWriter` 文件格式版本号一致
GRID_VERSION = 1

from .config import (
    ConfigBundle,
    DaConfig,
    ModelConfig,
    VerifyConfig,
    find_config_dir,
    load_config_bundle,
    load_da_config,
    load_model_config,
    load_verify_config,
    parse_yaml,
)
from .grid import GridSpec, cell_to_face, stagger_to_cell, zeta_levels
from .interp import (
    REGISTRY,
    bilinear,
    bilinear_prolong,
    column_interp_to_height,
    column_interp_to_pressure,
    conservative_coarsen,
    extrapolate_profile,
    interp_to_height,
    interp_to_pressure,
    pchip,
    trilinear,
)
from .io import (
    Dataset,
    MissingDependencyError,
    UnsupportedFormatError,
    VibeBinError,
    VibeBinFile,
    available_backends,
    open_dataset,
    read_grib,
    read_netcdf,
    read_vibebin,
    summarize,
    to_netcdf,
    write_vibebin,
)

__all__ = [
    "__version__",
    "GRID_VERSION",
    # config
    "ModelConfig", "DaConfig", "VerifyConfig", "ConfigBundle",
    "load_model_config", "load_da_config", "load_verify_config",
    "load_config_bundle", "parse_yaml", "find_config_dir",
    # grid
    "GridSpec", "stagger_to_cell", "cell_to_face", "zeta_levels",
    # io
    "Dataset", "VibeBinFile", "open_dataset", "read_vibebin", "write_vibebin",
    "read_netcdf", "read_grib", "to_netcdf", "summarize", "available_backends",
    "MissingDependencyError", "UnsupportedFormatError", "VibeBinError",
    # interp
    "bilinear", "trilinear", "pchip", "interp_to_height", "interp_to_pressure",
    "conservative_coarsen", "bilinear_prolong", "extrapolate_profile",
    "column_interp_to_pressure", "column_interp_to_height", "REGISTRY",
]

#: 诊断量名（与 C++ `dyn::diagnose_all` 的字段命名保持同名）
_DIAG_NAMES = (
    "wind_speed", "wind_direction", "dewpoint", "relative_humidity",
    "mixing_ratio", "saturation_mixing_ratio", "theta_e",
    "pressure_from_exner", "temperature_from_exner", "geopotential",
    "vorticity", "divergence", "absolute_vorticity", "potential_vorticity",
    "cape_cin", "storm_relative_helicity", "precipitation_accumulation",
    "sea_level_pressure", "reflectivity_dbz", "brightness_temperature",
    "k_index", "showalter_index", "celsius", "hpa",
)

_PLOT_NAMES = (
    "plot_field", "plot_vertical_section", "plot_skewt", "plot_profile",
    "plot_timeseries", "plot_reliability", "plot_rank_histogram",
    "plot_fss_vs_scale", "plot_spectra",
)


def __getattr__(name: str) -> Any:
    """惰性暴露 diagnostics / verify / plots 中的公开名字。

    `import vibe_post` 只加载轻量模块；`vibe_post.rmse` 这类访问才真正导入
    numpy 密集的模块。这样在没有任何可选依赖的环境中 `
    python -c "import vibe_post"` 也能成功。
    """
    if name in _DIAG_NAMES:
        return getattr(import_module(".diagnostics", __name__), name)
    if name in _PLOT_NAMES:
        return getattr(import_module(".plots", __name__), name)
    module = import_module(".verify", __name__)
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + list(_DIAG_NAMES) + list(_PLOT_NAMES)))

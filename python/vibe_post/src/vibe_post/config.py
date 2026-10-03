"""VIBE-Model 运行配置读取。

对应 C++ `vibe::config`（见 docs/design/00_architecture.md 第 10.1 节）。
YAML 解析优先使用 PyYAML；缺失时退回到**内置极简解析器**（本文件
:func:`parse_yaml`），它刻意只支持 C++ `src/config/mini_yaml.cpp` 的同一子集：

* 缩进映射 `key: value`；
* `- ` 序列（标量 / 行内映射 / 嵌套映射）；
* 标量：整数、浮点、布尔、null、引号字符串、非引号字符串；
* 行内列表 `[a, b, c]` 与行内映射 `{a: 1, b: 2}`；
* `#` 注释、空行、多文档分隔符 `---`。

**不支持**锚点/别名/多行标量/标签，与 C++ 侧保持一致，避免两边行为分叉。

单位规约（与 `config/model.yaml` 的写法一致）：
`dx = 3km`、`run_length = 6h`、`dt = 10s` 这类"数值+单位后缀"字符串由
:func:`parse_quantity` 转成 SI 数值。

文献：[B3] Warner (2011) 第 2 章（模式配置与数值参数）。
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "parse_yaml",
    "parse_quantity",
    "MiniYamlError",
    "DomainConfig",
    "TimeConfig",
    "NumericsConfig",
    "NestingConfig",
    "PhysicsConfig",
    "ParallelConfig",
    "ModelConfig",
    "DaConfig",
    "VerifyConfig",
    "ConfigBundle",
    "load_model_config",
    "load_da_config",
    "load_verify_config",
    "load_config_bundle",
    "find_config_dir",
    "DEFAULT_CONFIG_DIR",
]

DEFAULT_CONFIG_DIR = Path("config")

# ---------------------------------------------------------------------------
# 极简 YAML 子集解析器
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}
_NULLS = {"null", "~", ""}
_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_HEX_RE = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
_FLOAT_RE = re.compile(r"^[+-]?((\d+\.?\d*)|(\.\d+))([eE][+-]?\d+)?$")
_KEY_RE = re.compile(r"^([^:#\[\]{}]+):(?:\s+(.*))?$")
_QUANTITY_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)$")

#: 单位 -> SI 换算因子
_UNIT_FACTORS: dict[str, float] = {
    "m": 1.0, "km": 1.0e3, "cm": 1.0e-2, "mm": 1.0e-3,
    "s": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0, "d": 86400.0, "day": 86400.0,
    "pa": 1.0, "hpa": 100.0, "mb": 100.0, "kpa": 1000.0, "bar": 1.0e5,
    "k": 1.0, "c": 1.0, "deg": 1.0,
    "ms": 1.0, "m/s": 1.0, "mps": 1.0, "km/h": 1.0 / 3.6, "kt": 0.514444,
    "kg": 1.0, "g": 1.0e-3, "kg/kg": 1.0, "g/kg": 1.0e-3,
    "w": 1.0, "kw": 1.0e3, "j": 1.0, "k": 1.0,
    "1": 1.0, "s-1": 1.0, "%": 0.01, "percent": 0.01,
}


class MiniYamlError(ValueError):
    """极简 YAML 解析失败（含行号）。"""


def parse_quantity(value: Any, *, default_unit: float = 1.0) -> float:
    """把 `"3km"` / `"6 h"` / `3.5` 解析成以 SI 单位表示的浮点数。

    复杂度：O(1)。
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = _QUANTITY_RE.match(text)
    if match is None:
        return float(text) * default_unit
    number, unit = float(match.group(1)), match.group(2).lower()
    if unit not in _UNIT_FACTORS:
        raise MiniYamlError(f"未知单位后缀 {unit!r}（输入 {value!r}）")
    return number * _UNIT_FACTORS[unit]


def _strip_comment(line: str) -> str:
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if quote is not None:
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == quote:
                quote = None
            out.append(ch)
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def _unescape(text: str) -> str:
    return (text.replace("\\\\", "\\")
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\t", "\t"))


def _split_top_level(text: str) -> list[str]:
    """按顶层逗号切分，忽略引号与括号内部。复杂度 O(n)。"""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for ch in text:
        if quote is not None:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            continue
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(text: str) -> Any:
    token = text.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        inner = token[1:-1]
        return inner if token[0] == "'" else _unescape(inner)
    lowered = token.lower()
    if lowered in _NULLS:
        return None
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    if _HEX_RE.match(token):
        return int(token, 16)
    if _INT_RE.match(token):
        return int(token)
    if _FLOAT_RE.match(token):
        return float(token)
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        return [] if not inner else [_parse_scalar(p) for p in _split_top_level(inner)]
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        result: dict[str, Any] = {}
        if inner:
            for part in _split_top_level(inner):
                key, _, value = part.partition(":")
                result[str(_parse_scalar(key.strip()))] = _parse_scalar(value)
        return result
    return token


@dataclass
class _Line:
    indent: int
    text: str
    number: int


def _prepare(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = _strip_comment(raw.replace("\t", "    "))
        if not stripped.strip():
            continue
        if stripped.strip() in ("---", "..."):
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append(_Line(indent=indent, text=stripped.strip(), number=number))
    return lines


def _parse_block(lines: Sequence[_Line], pos: int, indent: int) -> tuple[Any, int]:
    """解析一个缩进层级上的块；返回 (值, 下一行位置)。"""
    if pos >= len(lines):
        return None, pos
    if lines[pos].text.startswith("- "):
        return _parse_sequence(lines, pos, indent)
    if lines[pos].text == "-":
        return _parse_sequence(lines, pos, indent)
    if ":" in lines[pos].text:
        return _parse_mapping(lines, pos, indent)
    return _parse_scalar(lines[pos].text), pos + 1


def _parse_sequence(lines: Sequence[_Line], pos: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while pos < len(lines):
        line = lines[pos]
        if line.indent < indent or not (line.text == "-" or line.text.startswith("- ")):
            break
        body = line.text[1:].strip()
        if not body:
            value, pos = _parse_block(lines, pos + 1, line.indent + 2)
            items.append(value)
            continue
        if ":" in body and not body.startswith(("[", "{", "'", '"')):
            # 序列中的行内映射项：把 "- key: value" 当成缩进映射
            synthetic = _Line(indent=line.indent + 2, text=body, number=line.number)
            sub = [synthetic]
            pos += 1
            while pos < len(lines) and lines[pos].indent > line.indent:
                sub.append(lines[pos])
                pos += 1
            value, _ = _parse_mapping(sub, 0, line.indent + 2)
            items.append(value)
            continue
        items.append(_parse_scalar(body))
        pos += 1
    return items, pos


def _parse_mapping(lines: Sequence[_Line], pos: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while pos < len(lines):
        line = lines[pos]
        if line.indent < indent:
            break
        if line.text == "-" or line.text.startswith("- "):
            break
        match = _KEY_RE.match(line.text)
        if match is None:
            raise MiniYamlError(f"第 {line.number} 行不是合法的 'key: value'：{line.text!r}")
        key = str(_parse_scalar(match.group(1).strip()))
        inline = match.group(2)
        if inline is not None and inline.strip() != "":
            result[key] = _parse_scalar(inline)
            pos += 1
            continue
        # 值为嵌套块：下一行必须缩进更深
        if pos + 1 < len(lines) and lines[pos + 1].indent > line.indent:
            value, pos = _parse_block(lines, pos + 1, lines[pos + 1].indent)
            result[key] = value
        elif pos + 1 < len(lines) and lines[pos + 1].text.startswith("- ") and lines[pos + 1].indent == line.indent:
            value, pos = _parse_sequence(lines, pos + 1, line.indent)
            result[key] = value
        else:
            result[key] = None
            pos += 1
    return result, pos


def parse_yaml(text: str) -> dict[str, Any]:
    """解析 YAML 子集（无 PyYAML 时的后备实现，也是唯一实现）。

    公式/算法：LL(1) 递归下降，缩进即文法层级；每个字符只扫描常数次，
    因此**复杂度 O(N)**（N 为字符数），空间 O(深度)。
    与 C++ `vibe::config::mini_yaml` 的行为保持一致（见 [B3] 第 2 章）。

    >>> parse_yaml("domain:\n  nx: 64\n  dx: 3km\n")
    {'domain': {'nx': 64, 'dx': '3km'}}
    """
    if not isinstance(text, str):
        raise MiniYamlError("parse_yaml 需要 str 输入")
    lines = _prepare(text)
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0].indent)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MiniYamlError("配置文件顶层必须是映射（mapping）")
    return value


def _load_yaml_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return parse_yaml(text)
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise MiniYamlError(f"{path} 顶层不是映射")
    return {str(k): v for k, v in data.items()}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class DomainConfig:
    """模式域几何。长度单位为米，与 C++ `config::ModelConfig::Domain` 一致。"""

    nx: int = 64
    ny: int = 64
    nz: int = 40
    dx: float = 3000.0
    dy: float = 3000.0
    x0: float = 0.0
    y0: float = 0.0
    z_top: float = 20000.0
    halo: int = 4
    terrain_file: str = ""
    vertical_stretch: float = 1.08
    flat_terrain: bool = True


@dataclass
class TimeConfig:
    dt: float = 10.0
    run_length: float = 21600.0
    acoustic_substeps: int = 6
    integrator: str = "rk3"
    output_interval: float = 3600.0
    start_time: str = "2024-01-01T00:00:00Z"


@dataclass
class NumericsConfig:
    advection_order: int = 5
    weno: bool = True
    divergence_damping: float = 0.05
    sponge_depth: float = 0.25
    rayleigh_damping: float = 0.001
    helmholtz_solver: str = "vertical_tridiagonal"
    krylov_tolerance: float = 1.0e-8


@dataclass
class NestingConfig:
    enabled: bool = False
    ratio: int = 3
    levels: int = 1
    two_way: bool = True
    relaxation_zone: int = 8
    feedback_interval: int = 1


@dataclass
class PhysicsConfig:
    microphysics: str = "thompson"
    radiation: str = "rrtmg"
    pbl: str = "myj"
    surface: str = "noah"
    cumulus: str = "none"
    radiation_interval: float = 900.0
    pbl_interval: float = 0.0


@dataclass
class ParallelConfig:
    px: int = 1
    py: int = 1
    openmp_threads: int = 1
    gpu_backend: str = "cpu"
    precision: str = "double"

    def nproc(self) -> int:
        return max(1, self.px) * max(1, self.py)


@dataclass
class ModelConfig:
    """对应 C++ `config::ModelConfig`。"""

    domain: DomainConfig = field(default_factory=DomainConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    nesting: NestingConfig = field(default_factory=NestingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    name: str = "vibe-run"
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("raw", None)
        return out


@dataclass
class DaConfig:
    """对应 C++ `config::DaConfig`（增量 4D-Var，[V3][V13]）。"""

    window_length: float = 3600.0
    n_outer: int = 2
    n_inner: int = 50
    minimizer: str = "lbfgs"
    background_error: str = "diffusion"
    correlation_scale: float = 200.0       # 水平相关尺度 (m)，[V7]
    vertical_scale: float = 1000.0         # 垂直相关尺度 (m)
    control_variables: tuple[str, ...] = ("u", "v", "theta", "qv")
    observation_types: tuple[str, ...] = ("radiosonde", "surface")
    adjoint_check: bool = True
    eps_check: float = 1.0e-6
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("raw", None)
        return out


@dataclass
class VerifyConfig:
    """对应 C++ `config::VerifyConfig` / `vibe::verify`（[B9][E13]）。"""

    variables: tuple[str, ...] = ("t2", "q2", "u10", "v10", "psfc", "rain")
    thresholds: dict[str, float] = field(
        default_factory=lambda: {"rain": 1.0e-3, "reflectivity": 35.0, "wind10": 10.0}
    )
    neighborhood_radii: tuple[int, ...] = (1, 3, 5, 9, 17)
    metrics: tuple[str, ...] = ("bias", "mae", "rmse", "correlation", "ets", "fss")
    climatology_file: str = ""
    ensemble_size: int = 0
    bootstrap: int = 0
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("raw", None)
        return out


@dataclass
class ConfigBundle:
    """一次读取的三个配置对象；缺失的文件对应字段为 None。"""

    model: ModelConfig | None = None
    da: DaConfig | None = None
    verify: VerifyConfig | None = None
    config_dir: str = ""

    def require_model(self) -> ModelConfig:
        if self.model is None:
            raise FileNotFoundError("未找到 config/model.yaml；请显式提供路径或设置 VIBE_POST_CONFIG")
        return self.model

    def __getitem__(self, key: str) -> Any:
        mapping = {"model": self.model, "da": self.da, "verify": self.verify}
        if key not in mapping:
            raise KeyError(key)
        return mapping[key]


# ---------------------------------------------------------------------------
# 映射 -> 数据类
# ---------------------------------------------------------------------------


def _as_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise MiniYamlError(f"{what} 需要映射，得到 {type(value).__name__}")


def _pick(data: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def _to_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return (value,)


def _set_fields(instance: Any, data: Mapping[str, Any], converters: Mapping[str, Any]) -> None:
    valid = {f.name for f in fields(instance)}
    for key, value in data.items():
        if key not in valid or key == "raw":
            continue
        converter = converters.get(key)
        setattr(instance, key, converter(value) if converter else value)


def da_config_from_mapping(data: Mapping[str, Any]) -> DaConfig:
    cfg = DaConfig()
    da = dict(_as_mapping(_pick(data, "da_4dvar", "4dvar", "da", default=data), "da"))
    nested = _pick(da, "background", "background_error", "b")
    bg = _as_mapping(nested, "da.background") if nested is not None else {}
    scale = _as_mapping(_pick(bg, "correlation", "scale", default={}), "da.background.scale")
    cfg.window_length = parse_quantity(
        _pick(da, "window_length", "window", "assimilation_window", default=cfg.window_length),
        default_unit=1.0,
    )
    cfg.n_outer = int(_pick(da, "n_outer", "outer_loops", default=cfg.n_outer))
    cfg.n_inner = int(_pick(da, "n_inner", "inner_loops", default=cfg.n_inner))
    cfg.minimizer = str(_pick(da, "minimizer", "solver", default=cfg.minimizer))
    if isinstance(bg, Mapping) and "type" in bg:
        cfg.background_error = str(bg["type"])
    cfg.background_error = str(_pick(da, "background_error", "b_type", default=cfg.background_error))
    cfg.correlation_scale = parse_quantity(
        _pick(scale, "horizontal", "horizontal_length", default=_pick(da, "correlation_scale", default=cfg.correlation_scale))
    )
    cfg.vertical_scale = parse_quantity(
        _pick(scale, "vertical", "vertical_length", default=_pick(da, "vertical_scale", default=cfg.vertical_scale))
    )
    cfg.control_variables = _to_tuple(_pick(da, "control_variables", "cv", default=cfg.control_variables))
    cfg.observation_types = _to_tuple(_pick(da, "observation_types", "obs_types", default=cfg.observation_types))
    cfg.adjoint_check = bool(_pick(da, "adjoint_check", "check_adjoint", default=cfg.adjoint_check))
    cfg.eps_check = float(_pick(da, "eps_check", "epsilon", default=cfg.eps_check))
    cfg.raw = dict(data)
    return cfg


def verify_config_from_mapping(data: Mapping[str, Any]) -> VerifyConfig:
    cfg = VerifyConfig()
    ver = dict(_as_mapping(_pick(data, "verify", "verification", default=data), "verify"))
    cfg.variables = _to_tuple(_pick(ver, "variables", "fields", default=cfg.variables))
    thresholds = _pick(ver, "thresholds", "threshold", default=None)
    if isinstance(thresholds, Mapping):
        cfg.thresholds = {
            str(k): parse_quantity(v, default_unit=1.0) if not isinstance(v, Mapping) else 0.0
            for k, v in thresholds.items()
        }
    elif thresholds is not None:
        cfg.thresholds = {str(name): parse_quantity(thresholds) for name in cfg.variables}
    radii = _pick(ver, "neighborhood_radii", "neighborhood", "radii", default=cfg.neighborhood_radii)
    cfg.neighborhood_radii = tuple(int(r) for r in _to_tuple(radii))
    cfg.metrics = _to_tuple(_pick(ver, "metrics", "methods", default=cfg.metrics))
    cfg.climatology_file = str(_pick(ver, "climatology_file", "clim", default=cfg.climatology_file))
    cfg.ensemble_size = int(_pick(ver, "ensemble_size", "n_members", "members", default=cfg.ensemble_size))
    cfg.bootstrap = int(_pick(ver, "bootstrap", default=cfg.bootstrap))
    cfg.raw = dict(data)
    return cfg


def model_config_from_mapping(data: Mapping[str, Any]) -> ModelConfig:
    """把 YAML 映射装配成 :class:`ModelConfig`（未知键忽略，缺失键取默认）。"""
    cfg = ModelConfig()
    cfg.name = str(_pick(data, "name", "run_name", default=cfg.name))

    dom = _as_mapping(_pick(data, "domain", "grid", default={}), "domain")
    cfg.domain.nx = int(_pick(dom, "nx", default=cfg.domain.nx))
    cfg.domain.ny = int(_pick(dom, "ny", default=cfg.domain.ny))
    cfg.domain.nz = int(_pick(dom, "nz", default=cfg.domain.nz))
    cfg.domain.dx = parse_quantity(_pick(dom, "dx", default=cfg.domain.dx))
    cfg.domain.dy = parse_quantity(_pick(dom, "dy", default=_pick(dom, "dx", default=cfg.domain.dy)))
    cfg.domain.x0 = parse_quantity(_pick(dom, "x0", "x_min", default=cfg.domain.x0))
    cfg.domain.y0 = parse_quantity(_pick(dom, "y0", "y_min", default=cfg.domain.y0))
    cfg.domain.z_top = parse_quantity(_pick(dom, "z_top", "ztop", "top", default=cfg.domain.z_top))
    cfg.domain.halo = int(_pick(dom, "halo", "nhalo", default=cfg.domain.halo))
    cfg.domain.terrain_file = str(_pick(dom, "terrain_file", "terrain", default=cfg.domain.terrain_file))
    cfg.domain.vertical_stretch = float(_pick(dom, "vertical_stretch", "stretch", default=cfg.domain.vertical_stretch))
    cfg.domain.flat_terrain = bool(_pick(dom, "flat_terrain", default=cfg.domain.flat_terrain))

    tim = _as_mapping(_pick(data, "time", "timestep", default={}), "time")
    cfg.time.dt = parse_quantity(_pick(tim, "dt", "time_step", default=cfg.time.dt))
    cfg.time.run_length = parse_quantity(_pick(tim, "run_length", "length", default=cfg.time.run_length))
    cfg.time.acoustic_substeps = int(_pick(tim, "acoustic_substeps", "n_sub", default=cfg.time.acoustic_substeps))
    cfg.time.integrator = str(_pick(tim, "integrator", "scheme", default=cfg.time.integrator))
    cfg.time.output_interval = parse_quantity(_pick(tim, "output_interval", "output_dt", default=cfg.time.output_interval))
    cfg.time.start_time = str(_pick(tim, "start_time", "start", default=cfg.time.start_time))

    num = _as_mapping(_pick(data, "numerics", "dynamics", default={}), "numerics")
    cfg.numerics.advection_order = int(_pick(num, "advection_order", "order", default=cfg.numerics.advection_order))
    cfg.numerics.weno = bool(_pick(num, "weno", default=cfg.numerics.weno))
    cfg.numerics.divergence_damping = float(_pick(num, "divergence_damping", "damping", default=cfg.numerics.divergence_damping))
    cfg.numerics.sponge_depth = float(_pick(num, "sponge_depth", default=cfg.numerics.sponge_depth))
    cfg.numerics.rayleigh_damping = float(_pick(num, "rayleigh_damping", default=cfg.numerics.rayleigh_damping))
    cfg.numerics.helmholtz_solver = str(_pick(num, "helmholtz_solver", "solver", default=cfg.numerics.helmholtz_solver))
    cfg.numerics.krylov_tolerance = float(_pick(num, "krylov_tolerance", "tol", default=cfg.numerics.krylov_tolerance))

    nest = _as_mapping(_pick(data, "nesting", "nest", default={}), "nesting")
    cfg.nesting.enabled = bool(_pick(nest, "enabled", default=cfg.nesting.enabled))
    cfg.nesting.ratio = int(_pick(nest, "ratio", default=cfg.nesting.ratio))
    cfg.nesting.levels = int(_pick(nest, "levels", "n_levels", default=cfg.nesting.levels))
    cfg.nesting.two_way = bool(_pick(nest, "two_way", default=cfg.nesting.two_way))
    cfg.nesting.relaxation_zone = int(_pick(nest, "relaxation_zone", "n_zone", default=cfg.nesting.relaxation_zone))
    cfg.nesting.feedback_interval = int(_pick(nest, "feedback_interval", default=cfg.nesting.feedback_interval))

    phys = _as_mapping(_pick(data, "physics", default={}), "physics")
    cfg.physics.microphysics = str(_pick(phys, "microphysics", "mp", default=cfg.physics.microphysics))
    cfg.physics.radiation = str(_pick(phys, "radiation", "ra", default=cfg.physics.radiation))
    cfg.physics.pbl = str(_pick(phys, "pbl", default=cfg.physics.pbl))
    cfg.physics.surface = str(_pick(phys, "surface", "lsm", default=cfg.physics.surface))
    cfg.physics.cumulus = str(_pick(phys, "cumulus", "cu", default=cfg.physics.cumulus))
    cfg.physics.radiation_interval = parse_quantity(_pick(phys, "radiation_interval", default=cfg.physics.radiation_interval))
    cfg.physics.pbl_interval = parse_quantity(_pick(phys, "pbl_interval", default=cfg.physics.pbl_interval))

    par = _as_mapping(_pick(data, "parallel", "mpi", default={}), "parallel")
    cfg.parallel.px = int(_pick(par, "px", "nproc_x", default=cfg.parallel.px))
    cfg.parallel.py = int(_pick(par, "py", "nproc_y", default=cfg.parallel.py))
    cfg.parallel.openmp_threads = int(_pick(par, "openmp_threads", "threads", default=cfg.parallel.openmp_threads))
    cfg.parallel.gpu_backend = str(_pick(par, "gpu_backend", "backend", default=cfg.parallel.gpu_backend))
    cfg.parallel.precision = str(_pick(par, "precision", default=cfg.parallel.precision))

    cfg.raw = dict(data)
    return cfg


# ---------------------------------------------------------------------------
# 文件入口
# ---------------------------------------------------------------------------


def find_config_dir(start: str | os.PathLike[str] | None = None) -> Path | None:
    """自 `start`（默认 cwd）向上查找含 `model.yaml` 的目录，或 `$VIBE_POST_CONFIG`。

    复杂度：O(目录深度)。
    """
    env = os.environ.get("VIBE_POST_CONFIG")
    if env:
        return Path(env).expanduser()
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        cfg_dir = candidate / DEFAULT_CONFIG_DIR
        if (cfg_dir / "model.yaml").is_file() or (cfg_dir / "da_4dvar.yaml").is_file():
            return cfg_dir
    return None


def _resolve(path: str | os.PathLike[str] | None, config_dir: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if config_dir is not None and (config_dir / candidate.name).is_file():
        return config_dir / candidate.name
    return candidate


def load_model_config(
    path: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    config_dir: str | os.PathLike[str] | None = None,
) -> ModelConfig:
    """读取 `config/model.yaml`。

    参数可以是文件路径、已解析的映射、或 None（自动查找）。
    """
    if isinstance(path, Mapping):
        return model_config_from_mapping(path)
    target = _resolve(path, find_config_dir(config_dir)) or (Path(config_dir or DEFAULT_CONFIG_DIR) / "model.yaml")
    if not target.is_file():
        raise FileNotFoundError(f"找不到模式配置：{target}")
    cfg = model_config_from_mapping(_load_yaml_file(target))
    cfg.source_path = str(target)
    return cfg


def load_da_config(
    path: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    config_dir: str | os.PathLike[str] | None = None,
) -> DaConfig:
    """读取 `config/da_4dvar.yaml`。"""
    if isinstance(path, Mapping):
        return da_config_from_mapping(path)
    cfg_dir = find_config_dir(config_dir)
    target = _resolve(path, cfg_dir) or ((cfg_dir or Path(config_dir or DEFAULT_CONFIG_DIR)) / "da_4dvar.yaml")
    if not target.is_file():
        raise FileNotFoundError(f"找不到同化配置：{target}")
    cfg = da_config_from_mapping(_load_yaml_file(target))
    cfg.source_path = str(target)
    return cfg


def load_verify_config(
    path: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    config_dir: str | os.PathLike[str] | None = None,
) -> VerifyConfig:
    """读取 `config/verify.yaml`。"""
    if isinstance(path, Mapping):
        return verify_config_from_mapping(path)
    cfg_dir = find_config_dir(config_dir)
    target = _resolve(path, cfg_dir) or ((cfg_dir or Path(config_dir or DEFAULT_CONFIG_DIR)) / "verify.yaml")
    if not target.is_file():
        raise FileNotFoundError(f"找不到检验配置：{target}")
    cfg = verify_config_from_mapping(_load_yaml_file(target))
    cfg.source_path = str(target)
    return cfg


def load_config_bundle(
    config_dir: str | os.PathLike[str] | None = None,
    *,
    model: str | os.PathLike[str] | None = None,
    da: str | os.PathLike[str] | None = None,
    verify: str | os.PathLike[str] | None = None,
    strict: bool = False,
) -> ConfigBundle:
    """一次性读取三个配置文件；缺失的文件在结果中为 None（`strict=True` 则抛错）。"""
    cfg_dir = Path(config_dir) if config_dir is not None else find_config_dir()
    bundle = ConfigBundle(config_dir=str(cfg_dir) if cfg_dir else "")
    for key, loader, explicit in (("model", load_model_config, model),
                                  ("da", load_da_config, da),
                                  ("verify", load_verify_config, verify)):
        target = explicit
        if target is None and cfg_dir is not None:
            candidate = {"model": "model.yaml", "da": "da_4dvar.yaml", "verify": "verify.yaml"}[key]
            if (cfg_dir / candidate).is_file():
                target = cfg_dir / candidate
        if target is None:
            if strict:
                raise FileNotFoundError(f"缺少配置文件：{key}")
            continue
        setattr(bundle, key, loader(target))
    return bundle

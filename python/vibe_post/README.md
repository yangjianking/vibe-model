# vibe_post — VIBE-Model Python 后处理包

`vibe_post` 是 VIBE-Model 的**独立 Python 后处理库**，与 C++ 侧完全解耦：
它只读取模式输出文件（NetCDF / GRIB2 / `.vibebin`），在 numpy 上完成诊断、检验与绘图。
接口基线见 `docs/design/00_architecture.md` 与 `docs/design/10_postprocessing.md`。

## 安装

~~~bash
# 仅核心（numpy，含纯 numpy 二进制后端）
pip install ./python/vibe_post

# 带 NetCDF / GRIB / 绘图 / YAML 支持
pip install "./python/vibe_post[all]"

# 开发模式
pip install -e "./python/vibe_post[all,test]"
~~~

## 快速上手

~~~python
import numpy as np
import vibe_post as vp

ds = vp.open_dataset("wrfout.vibebin")           # 自动探测后端
T  = vp.temperature_from_exner(ds.fields["pi"], ds.fields["theta"])
ds = ds.with_field("T", T)
print(vp.summarize(ds))

grid = vp.GridSpec.from_dataset(ds)
u_c  = vp.stagger_to_cell(ds.fields["u"], "u", grid)
~~~

命令行：

~~~bash
vibe-post info   run/wrfout.nc
vibe-post interp run/wrfout.nc --var theta --to-height 5000 --out th5km.nc
vibe-post derive run/wrfout.nc --list
vibe-post derive run/wrfout.nc --diag cape --out diag.nc
vibe-post verify --forecast f.nc --obs o.nc --var t2 --method rmse
vibe-post plot   run/wrfout.nc --var theta --level 10 --out theta.png
~~~

## 模块一览

| 模块 | 职责 | 关键 API |
|---|---|---|
| `config` | 读取 `config/*.yaml`（PyYAML 或内置极简解析器） | `load_model_config`, `load_config_bundle` |
| `grid` | GridSpec、地形追随高度、Arakawa C-grid 错位 | `GridSpec`, `stagger_to_cell` |
| `io` | Dataset、NetCDF / cfgrib / `.vibebin` 三种后端 | `open_dataset`, `VibeBinFile` |
| `interp` | 双/三线性、单调 PCHIP 垂直插值、保守粗化 | `bilinear`, `interp_to_height`, `conservative_coarsen` |
| `diagnostics` | 20+ 无状态诊断量（numpy 向量化） | `cape_cin`, `potential_vorticity`, `theta_e` |
| `verify` | 连续 / 分类 / 概率 / 邻域四类检验评分 | `rmse`, `ets`, `crps_ensemble`, `fss` |
| `plots` | matplotlib 填色图、剖面、skew-T、动能谱 | `plot_field`, `plot_skewt`, `plot_spectra` |

所有可选后端均**惰性导入**：缺少 xarray/cfgrib/matplotlib/PyYAML 时给出带安装提示的
`MissingDependencyError`，而不是裸 `ImportError`。

## 数据模型

统一交换对象是 :class:`vibe_post.io.Dataset`：

~~~python
Dataset(
    fields = {"theta": ndarray, ...},   # 变量 -> 数组，规范维度顺序
    dims   = {"theta": ("time", "zeta", "y", "x")},
    coords = {"x": ..., "y": ..., "zeta": ..., "time": ...},
    attrs  = {"source": "...", "real_kind": "float64"},
    time   = 0.0,                        # 秒，相对起报时刻
)
~~~

## `.vibebin` 与 C++ 的兼容性

`vibe_post.io.VibeBinFile` **字节级兼容** C++ `vibe::io::BinaryWriter`（定义见
`include/vibe/io/field_io.hpp`）：40 字节文件头 + 每条记录 `name[32] + float64 数据`，
变量形状由名字唯一决定（`u` 加一列、`v` 加一行、`w`/`zeta` 加一层、其余为体心）。

因为面场与体心场形状不同，诊断量所需的“体心视图”要显式取：

~~~python
u_cell = ds.fields["u"][..., :grid.nx]      # x 面 -> 体心
v_cell = ds.fields["v"][:, :grid.ny, :]      # y 面 -> 体心
w_cell = 0.5 * (ds.fields["w"][:-1] + ds.fields["w"][1:])  # 层界面 -> 层中心
~~~

`write_vibebin(..., extended=True)` 额外写出形状、CRC32 与 JSON 属性尾部，读取器
自动识别两种布局；C++ 读取器只认 classic 布局，因此默认写出 classic。

垂直坐标名为 **`zeta`**（地形追随坐标值，量纲为米），与 C++
`grid::Geometry::zeta` 语义一致；物理高度由 :meth:`GridSpec.height_array` 计算。

## 单位约定

内部一律使用 **SI 基本单位**：米、秒、开尔文、帕斯卡、kg/kg。
仅在绘图/输出时按需转换（`diagnostics.celsius`、`diagnostics.hpa`）。
风分量 `u,v` 沿物理坐标 `+x/+y` 方向。

## 许可

MIT。文献引用编号（`[D5]`、`[E7]` …）对应 `docs/design/references.md`。

# 教程 · 第一次运行

本教程带你从零跑通一次温泡对流理想试验，并画出结果。

---

## 1. 准备

~~~bash
git clone <repo> vibe-model && cd vibe-model

cmake --preset gcc-release
cmake --build --preset gcc-release -j
~~~

若你的机器没有 MPI/NetCDF，CMake 会自动关闭对应开关并给出警告，这不影响运行。

---

## 2. 快速检验（不写文件）

~~~bash
./build/src/vibe_model --config config/model.yaml --ic warm_bubble --dry-run --log-level info
~~~

输出包含：

~~~text
[INFO ] VIBE-Model 运行配置
名称      : vibe_idealized
格点 vibe_idealized: 全局 128x64x60, 本地 128x64x60, 分解 1x1 (rank 0), 平坦地形, dx=2000 ...
参考态：theta0 平均 300 K, 地面气压 100000 Pa, 顶层气压 ... Pa, 静力残差(相对) ...
初值构造完成：warm_bubble ...
驱动初始化完成：vibe_idealized，积分器 rk3_acoustic
~~~

---

## 3. 正式运行

编辑 config/model.yaml，把 run_length 设为 7200（秒），然后：

~~~bash
mkdir -p output
./build/src/vibe_model --config config/model.yaml --ic warm_bubble --log-level info
~~~

每 50 步会输出一行状态：

~~~text
[INFO ] t=300 s step=50 dt=6 | CFL 平流=0.42 声波=2.10 垂直=0.61 最大风速=31.2 最小层厚=58.0 m
~~~

结束时输出运行摘要（步数、模拟时长、墙钟耗时、能量收支首末对比）。

---

## 4. 用 Python 后处理

~~~bash
python -m pip install -e "python/vibe_post[all]"      # 含 xarray / matplotlib
~~~

~~~python
import vibe_post as vp

ds = vp.open_dataset("output/vibe_idealized_final.nc")
print(vp.summarize(ds))

# 第 30 层的位温扰动
theta = ds.fields["theta"]
diag = vp.diagnostics.derived_fields(ds, ["theta_e", "rh", "vorticity", "reflectivity"])

import matplotlib
matplotlib.use("Agg")
vp.plots.plot_field(diag["reflectivity"], level=20, out="figs/refl.png", cmap="turbo")
~~~

---

## 5. 短时回归（CI 风格）

~~~bash
ctest --test-dir build -L smoke --output-on-failure
~~~

---

## 6. 下一步

* [idealized_cases.md](idealized_cases.md)：全部理想试验的物理含义与推荐配置；
* [../design/02_dynamical_core.md](../design/02_dynamical_core.md)：读懂方程组；
* [../design/04_time_integration.md](../design/04_time_integration.md)：切换积分器并对比；
* [../design/06_4dvar.md](../design/06_4dvar.md)：加入观测并做同化。

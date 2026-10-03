# 02 · 动力学核心

> 本章给出 VIBE-Model 控制方程的完整推导、离散化、守恒性分析与边界条件。
> 文献编号见 [references.md](references.md)。

---

## 1. 控制方程

### 1.1 全可压非静力方程组

干空气状态方程与热力学关系：

$$p = \rho R_d T\,\frac{1 + q_v/\epsilon}{1 + q_v}, \qquad
\pi \equiv \left(\frac{p}{p_{00}}\right)^{R_d/c_p}, \qquad T = \theta\,\pi .$$

动量、质量、位温与水物质方程：

$$\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} &= -\mathbf{u}\cdot\nabla\mathbf{u} - c_p \theta \nabla\pi - f\,\mathbf{k}\times\mathbf{u} + \mathbf{g} + \mathcal{D}_u,\\
\frac{\partial \rho}{\partial t} &= -\nabla\cdot(\rho\mathbf{u}),\\
\frac{\partial \theta}{\partial t} &= -\mathbf{u}\cdot\nabla\theta + \frac{\dot{Q}}{\Pi},\\
\frac{\partial q_x}{\partial t} &= -\mathbf{u}\cdot\nabla q_x + S_x .
\end{aligned}$$

浮力项显式计入虚温效应与水凝物负重 [D1] 式 (2.8)：

$$B = g\left[\frac{\theta'}{\theta_0} + \left(\frac{R_v}{R_d}-1\right)q_v - (q_c + q_r + q_i + q_s + q_g)\right],
\qquad \frac{R_v}{R_d}-1 = 0.608 .$$

### 1.2 地形追随高度坐标

令 $H$ 为模式顶、$z_s(x,y)$ 为地形 [D2]：

$$\zeta = H\,\frac{z - z_s(x,y)}{H - z_s(x,y)} \in [0, H].$$

映射的雅可比与度量项：

$$\frac{\partial z}{\partial \zeta} = \frac{H - z_s}{H} \equiv G^{1/2},
\qquad z = z_s + \zeta\,\frac{H - z_s}{H},
\qquad \mathrm{d}V = \mathrm{d}x\,\mathrm{d}y\,G^{1/2}\,\mathrm{d}\zeta .$$

**陡坡问题。** 上式在陡坡上会产生虚假气压梯度力的截断误差，量级与 $\partial z_s/\partial x$ 成正比。
本模式实现四种抑制手段（均为可配置选项）：

1. 地形平滑（1-2-1 滤波 + 斜率限幅），实现于 `grid::smooth_terrain`；
2. 层间平滑（Klemp 2011 的 smoothed coordinate surfaces）[D10]；
3. $\sigma$-$z$ 混合坐标 [D9]；
4. 解的限幅 [D11]。

### 1.3 参考态分离 [D1]

$$\rho = \rho_0(z) + \rho', \qquad \theta = \theta_0(z) + \theta', \qquad \pi = \pi_0(z) + \pi' .$$

参考态水平均匀、时间不变、满足静力平衡与状态方程：

$$\frac{\mathrm{d}\pi_0}{\mathrm{d}z} = -\frac{g}{c_p\theta_0},
\qquad \rho_0 = \frac{p_{00}\,\pi_0^{\,c_v/R_d}}{R_d\,\theta_0},
\qquad \pi_0(0) = \left(\frac{p_{\mathrm{sfc}}}{p_{00}}\right)^{R_d/c_p}.$$

**离散格式**（与动力学的垂直差分相容，二阶）：

$$\pi_0(z_{k+1}) = \pi_0(z_k) - \frac{g\,(z_{k+1}-z_k)}{c_p\,\theta_0^{\mathrm{mid}}},
\qquad \theta_0^{\mathrm{mid}} = \tfrac12\left(\theta_0(z_k)+\theta_0(z_{k+1})\right).$$

代码入口：`ReferenceState::isothermal` / `from_profile` / `standard_atmosphere`。
一致性检验：`ReferenceState::hydrostatic_residual()` 返回相对残差，单元测试要求小于 $5\times10^{-3}$。

---

## 2. 空间离散

### 2.1 Arakawa C-grid 与 Lorenz 垂直错位

~~~text
   w(k+1) -----------------------------------  层界面 (zeta 界面)
          |             |             |
   u(i) --+--- theta ---+--- u(i+1) --+       层中心 (标量位置)
   (x 面) |  rho, pi,   |   (x 面)    |
          |  qv, qc...  |             |
   w(k)   -----------------------------------  层界面
        v(j)                        v(j+1)     (y 面)
~~~

| 变量 | 位置 |
|---|---|
| $u$ | $x$ 面心，$y,z$ 体心 |
| $v$ | $y$ 面心，$x,z$ 体心 |
| $w$ | $z$ 界面，$x,y$ 体心 |
| $\rho,\theta,\pi,q_x$ | 体心 |

该错位的两个关键性质 [D3][D4]：

1. **梯度与散度互为负转置**：在周期边界下
   $\sum \rho u \,\partial_x \pi'$ 与 $\sum \pi' \partial_x(\rho u)$ 精确配对，离散动能守恒；
2. **Lorenz 错位使离散垂直动能与位能可以精确互换**，是离散总能量守恒的必要条件。

### 2.2 差分算子

| 算子 | 表达式 | 代码 |
|---|---|---|
| 一阶导数（2 阶） | $(q_{i+1}-q_{i-1})/(2\Delta x_i)$ | `grad_x` |
| 一阶导数（4 阶） | $(q_{i-2}-8q_{i-1}+8q_{i+1}-q_{i+2})/(12\Delta x_i)$ | `grad_x` |
| 一阶导数（6 阶） | $(-q_{i-3}+9q_{i-2}-45q_{i-1}+45q_{i+1}-9q_{i+2}+q_{i+3})/(60\Delta x_i)$ | `grad_x` |
| 水平散度 | $(u_{i+1}-u_i)/\Delta x_i + (v_{j+1}-v_j)/\Delta y_j$ | `dyn::divergence` |
| 相对涡度 | $(v_{i+1}-v_{i-1})/(2\Delta x) - (u_{j+1}-u_{j-1})/(2\Delta y)$ | `dyn::relative_vorticity` |
| 垂直差分（面） | $(f_k - f_{k-1})/\Delta z_k$ | `ddz_face` |

在变分辨率网格上 $\Delta x_i$ 逐列不同（`Grid::dx_at(i)`），其余公式不变。
**代价是 CFL 由最小单元决定**，因此变分辨率网格适合"局部加密 + 全局粗化"的场景 [N5][N6]。

### 2.3 平流格式

四种格式在配置项 `numerics.advection` 中切换。

**（a）二阶中心（默认）**，能量守恒错位 [D3]：

$$(u\,\partial_x q)_i \approx \tfrac12(u_i + u_{i+1})\,\frac{q_{i+1}-q_{i-1}}{2\Delta x}.$$

**（b）四阶 / 六阶中心**：仅替换梯度模板（表见 2.2），$u$ 仍取面平均。

**（c）保守有限体积形式**（`flux_form: true`）：

$$\frac{\partial(\rho q)}{\partial t} = -\nabla\cdot(\rho q\,\mathbf{u}),$$

面上通量取二阶中心插值：

$$F_x(i) = u_i \cdot \tfrac12(\rho_{i-1}+\rho_i)\cdot \tfrac12(q_{i-1}+q_i).$$

在周期边界下 $\sum_{ijk}\rho_{ijk}q_{ijk}\Delta V_{ijk}$ 守恒到浮点舍入量级 [D6][D13]。

**（d）五阶 WENO** [D7][D8]。在 $i+1/2$ 面上做左右偏重构，左偏的三个候选值为

$$v_0 = \frac{2q_{i-2}-7q_{i-1}+11q_i}{6},\quad
v_1 = \frac{-q_{i-1}+5q_i+2q_{i+1}}{6},\quad
v_2 = \frac{2q_i+5q_{i+1}-q_{i+2}}{6},$$

光滑性指标（以 $\beta_0$ 为例）

$$\beta_0 = \tfrac{13}{12}(q_{i-2}-2q_{i-1}+q_i)^2 + \tfrac14(q_{i-2}-4q_{i-1}+3q_i)^2,$$

$\beta_1$ 的第二项为 $\tfrac14(q_{i-1}-q_{i+1})^2$，$\beta_2$ 与 $\beta_0$ 镜像。非线性权重

$$\alpha_k = \frac{d_k}{(\epsilon+\beta_k)^2},\quad (d_0,d_1,d_2)=(0.1,\,0.6,\,0.3),\quad
w_k = \frac{\alpha_k}{\sum_j \alpha_j},\qquad
q^-_{i+1/2} = \sum_k w_k\,v_k .$$

右偏重构为镜像操作。通量采用局部 Lax-Friedrichs 分裂：

$$\hat{F} = \tfrac12\left[\,u\,(q^- + q^+) - |u|\,(q^+ - q^-)\,\right].$$

WENO5 模板宽度为 5，因此要求 halo 宽度不小于 4；该约束在 `Advection` 构造函数中以
`VIBE_CHECK` 静态检查（运行时断言）。

---

## 3. 离散守恒性

### 3.1 质量守恒

质量方程严格采用通量形式。离散后

$$\frac{\mathrm{d}}{\mathrm{d}t}\sum_{ijk}\rho_{ijk}\,\Delta V_{ijk}
= -\sum_{\text{faces}} F\cdot\mathbf{n}\,A .$$

在周期边界（或所有面通量为零）下右端逐项相消，**总质量在机器精度内守恒**。
单元体积由 `Grid::cell_volume` 给出：

$$\Delta V_{ijk} = \Delta x_i\,\Delta y_j\,G^{1/2}_{ij}\,\Delta\zeta_k\,H .$$

集成测试对周期边界检验相对质量漂移小于 $5\times10^{-3}$。

### 3.2 能量守恒

在无物理过程、无摩擦、周期边界条件下，离散总能量

$$E = \sum_{ijk}\left[\tfrac12\rho(u^2+v^2+w^2) + \rho c_v T + \rho g z + \rho L_v q_v\right]\Delta V_{ijk}$$

的漂移来自三个来源，本模式逐个处理：

| 来源 | 机制 | 处理 |
|---|---|---|
| 时间分裂 | 大/小步之间通量不一致 | 采用守恒分裂显式结构 [D6] |
| 垂直平流 | 面平均的相位误差 | 保证 $w\,\partial_\zeta\theta$ 的配对 |
| 上层海绵层 | 人为耗散 | 仅在 $z > 0.8H$ 开启，默认 $\alpha_{\max}=0.2$ |

守恒性检验接口 `dyn::energy_budget` 返回动能、内能、位能、潜热能与总质量，
driver 在日志与运行摘要中给出首末对比。

### 3.3 Ertel 位涡

[1] 位涡是检验离散格式质量的重要工具：在绝热无摩擦的等熵面上 PV 沿轨迹守恒。

$$\mathrm{PV} = \frac{1}{\rho}\,\boldsymbol{\omega}_a\cdot\nabla\theta,\qquad
\boldsymbol{\omega}_a = \left(-\frac{\partial v}{\partial z},\;
\frac{\partial u}{\partial z},\;
f + \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right).$$

实现见 `dyn::ertel_pv`；`dyn::diagnose_all` 一次性计算全部诊断量。

---

## 4. 边界条件

### 4.1 水平

* **周期**（理想试验）：`Decomposition::periodic_x/y`，halo 直接环绕复制；
* **固壁**：法向速度置零；
* **开放 / 嵌套**：由 Davies 松弛区提供，见 [03_grid_and_nesting.md](03_grid_and_nesting.md)。

### 4.2 垂直

* 下边界 $\zeta = 0$：$w = 0$；$u,v$ 的地面通量由物理模块提供；
* 上边界 $\zeta = H$：$w = 0$（刚盖），并在其上 20% 高度内施加 Rayleigh 海绵层：

$$\alpha(z) = \alpha_{\max}\sin^2\!\left(\frac{\pi}{2}\,
\frac{z - z_{\mathrm{sponge}}}{H - z_{\mathrm{sponge}}}\right),\qquad
\phi \leftarrow \frac{\phi + \Delta t\,\alpha\,\phi_{\mathrm{ref}}}{1 + \Delta t\,\alpha}.$$

后向欧拉保证无条件稳定。实现见 `Damping::apply_sponge`。

---

## 5. 代码对照表

| 数学对象 | 类 / 函数 | 文件 |
|---|---|---|
| 参考态 $\rho_0,\theta_0,\pi_0$ | `dyn::ReferenceState` | [reference_state.hpp](../../include/vibe/dyn/reference_state.hpp) |
| 预报状态 | `dyn::State` | [state.hpp](../../include/vibe/dyn/state.hpp) |
| 趋势容器 | `dyn::Tendency` | [tendency.hpp](../../include/vibe/dyn/tendency.hpp) |
| 平流（4 种格式） | `dyn::Advection` | [advection.cpp](../../src/dyn/advection.cpp) |
| 科氏力 | `dyn::Coriolis` | [coriolis.cpp](../../src/dyn/coriolis.cpp) |
| 海绵层 / 散度阻尼 | `dyn::Damping` | [damping.cpp](../../src/dyn/damping.cpp) |
| 方程装配 | `dyn::Equations` | [equations.cpp](../../src/dyn/equations.cpp) |
| 诊断量 | `dyn::diagnose_all` | [diagnostics.cpp](../../src/dyn/diagnostics.cpp) |
| 网格与度量 | `grid::Grid` | [geometry.cpp](../../src/grid/geometry.cpp) |

---

## 6. 参考文献

[D1] Klemp & Wilhelmson (1978) · [D2] Gal-Chen & Somerville (1975) ·
[D3] Arakawa & Lamb (1977) · [D4] Lorenz (1960) · [D5] Skamarock & Klemp (2008) ·
[D6] Klemp, Skamarock & Dudhia (2008) · [D7] Shu (1998) · [D8] Jiang & Shu (1996) ·
[D9] Schär et al. (2002) · [D10] Klemp (2011) · [D11] Zängl (2012) ·
[D13] Satoh (2002) · [D14] Harris & Durran (2010) ·
[B4] Holton & Hakim (2013) · [B6] LeVeque (2002)

# 01 · 总体设计

> 文献编号见 [references.md](references.md)。本文回答三个问题：
> **VIBE-Model 是什么、它由哪些数值过程组成、这些过程如何组织成软件。**

---

## 1. 定位

VIBE-Model 是一个面向**教学与科研**的数值天气预报模式。它刻意选择了"足够完整但不求业务化"的设计点：

| 维度 | VIBE-Model 的选择 | 理由 |
|---|---|---|
| 控制方程 | 全可压非静力 Euler 方程 | 覆盖对流尺度到天气尺度，无静力近似误差 |
| 垂直坐标 | 地形追随高度坐标（Gal-Chen & Somerville） | 物理直观，便于教学 |
| 水平网格 | Arakawa C-grid | 离散动能守恒、梯度/散度配对 [D3] |
| 垂直错位 | Lorenz 错位（w 在半层） | 离散能量守恒 [D4] |
| 时间推进 | RK3 + 声波子步 **或** 半隐式 Helmholtz | 两条主流路线的并列实现 |
| 嵌套 | 固定比 r = 3 双向嵌套 + Davies 松弛 | 区域高分辨率的标准做法 [N1][N3][N4] |
| 变分辨率 | 平滑映射构造的变分辨率网格 | 与嵌套共用度量项接口 [N5][N6] |
| 同化 | 增量 4D-Var（强约束，可选弱约束） | 伴随方法的教学与科研平台 [V3][V10] |
| 并行 | MPI 域分解 + OpenMP + GPU（CUDA/HIP/SYCL） | 覆盖传统 HPC 集群到 GPU 集群 |
| 精度 | double / single / mixed 三档 | 混合精度是当代模式的重要议题 [G6][G7] |

**明确的非目标**：业务化资料接收、集合预报全流程、化学/气溶胶、海洋耦合。

---

## 2. 数值过程的完整链条

~~~text
   初值/边界场                       观测（探空/地面/卫星/雷达）
        |                                        |
        v                                        v
  +------------------+                  +----------------------+
  |  初始化与静力调整  |                  |  质量控制与偏差订正     |
  +------------------+                  +----------------------+
        |                                        |
        v                                        v
  +---------------------------------------------------------+
  |                    增量 4D-Var 同化                       |
  |      外循环：非线性轨迹 + 创新向量                          |
  |      内循环：TL 前向 + AD 反向 + L-BFGS 极小化               |
  +---------------------------------------------------------+
        |  分析场 x^a
        v
  +---------------------------------------------------------+
  |                       模式时间积分                         |
  |   慢过程：平流 / 科氏 / 物理 / 阻尼 -> RK3 或 AB2            |
  |   快过程：线性声波-重力波                                  |
  |        分裂显式：n 个前向-后向子步                          |
  |        半隐式  ：3D Helmholtz（Crank-Nicolson）            |
  |   每个大时间步：halo 交换（MPI） -> 嵌套父子交换（双向）        |
  +---------------------------------------------------------+
        |  预报场
        v
  +------------------+     +---------------------------+
  |  IO（NetCDF /     |     |  误差检验与评估             |
  |  .vibebin 重启）   |     |  Bias/RMSE/ETS/FSS/Brier  |
  +------------------+     +---------------------------+
        |                           |
        v                           v
  +---------------------------------------------------------+
  |              Python 后处理包 vibe_post                     |
  +---------------------------------------------------------+
~~~

---

## 3. 控制方程组（概览）

预报量：$u, v, w, \rho, \theta, \pi', q_v, q_c, q_r, q_i, q_s, q_g$。

采用静力平衡参考态分离 [D1]：

$$\rho = \rho_0(z) + \rho', \qquad \theta = \theta_0(z) + \theta', \qquad \pi = \pi_0(z) + \pi'$$

其中 $\pi = (p/p_{00})^{R_d/c_p}$ 为 Exner 函数。分离的收益是：**线性声波-重力波项的系数只依赖 $z$**，
因此半隐式的 Helmholtz 算子系数可以离线构造并复用。

动量方程（地形追随坐标，$G^{1/2} = (H - z_s)/H$）：

$$\begin{aligned}
\frac{\partial u}{\partial t} &= -\mathbf{u}\cdot\nabla u - c_p \theta \frac{\partial \pi'}{\partial x} + f v + D_u,\\
\frac{\partial v}{\partial t} &= -\mathbf{u}\cdot\nabla v - c_p \theta \frac{\partial \pi'}{\partial y} - f u + D_v,\\
\frac{\partial w}{\partial t} &= -\mathbf{u}\cdot\nabla w - c_p \theta\,G^{-1/2}\frac{\partial \pi'}{\partial \zeta}
                                 + g\left[\frac{\theta'}{\theta_0} + 0.608\,q_v - \sum_x q_x\right] + D_w .
\end{aligned}$$

质量守恒（严格守恒形式）与热力学、水物质方程：

$$\frac{\partial \rho}{\partial t} = -\nabla\cdot(\rho \mathbf{u}), \qquad
\frac{\partial \theta}{\partial t} = -\mathbf{u}\cdot\nabla\theta + \frac{\theta_0}{\rho_0 c_p T_0}\dot{Q}, \qquad
\frac{\partial q_x}{\partial t} = -\mathbf{u}\cdot\nabla q_x + S_x .$$

状态方程与线性化 Exner 方程：

$$\pi = \left[\frac{\rho R_d \theta}{p_{00}}\cdot\frac{1 + q_v/\epsilon}{1 + q_v}\right]^{R_d/c_v},
\qquad \frac{\partial \pi'}{\partial t} = \frac{R_d}{c_v}\frac{\pi_0}{\rho_0}\frac{\partial \rho'}{\partial t}.$$

完整推导、离散化与守恒性分析见 [02_dynamical_core.md](02_dynamical_core.md)。

---

## 4. 空间与时间离散

### 4.1 空间

* **Arakawa C-grid** [D3]：$u$ 在 $x$ 面，$v$ 在 $y$ 面，标量在体心，$w$ 在 $z$ 面（Lorenz 错位）。
  该错位使梯度与散度算子互为负转置，因而离散动能守恒。
* **地形追随高度坐标** [D2]：$\zeta = H (z - z_s)/(H - z_s)$。陡坡上的截断误差由
  地形平滑与平滑层抑制 [D9][D10][D11]。
* **变分辨率**：$\Delta x$、$\Delta y$ 变成逐列/逐行的数组（Grid 的 dx_cell 字段），
  动力学代码完全不变，CFL 由最小单元决定。

### 4.2 时间

两条并列路线：

| | 分裂显式（rk3_acoustic） | 半隐式（semi_implicit） |
|---|---|---|
| 慢过程 | Wicker-Skamarock 低存储 RK3 [T5] | Adams-Bashforth 2 阶外推 |
| 快过程 | n 个前向-后向声波子步 [D1][D6] | Crank-Nicolson + 3D Helmholtz [T2][T6][T7] |
| 时间步上限 | min(平流 CFL, n x 声波子步极限) | 平流 CFL |
| 每步成本 | 无全局线性求解 | 一次（或数次）Helmholtz 求解 |
| 适用 | 中小规模、GPU 友好、实现简单 | 大时间步、大规模、陡地形 |

细节见 [04_time_integration.md](04_time_integration.md)。

---

## 5. 软件模块与依赖

~~~text
vibe::common    基础类型 / 常量 / 错误 / 日志 / MPI 封装 / 计时
vibe::config    YAML 子集解析 + 配置结构 + 一致性校验
vibe::grid      几何、度量、Field、halo、插值、嵌套、变分辨率
vibe::dyn       参考态、State、平流、科氏、阻尼、方程装配、诊断
vibe::gpu       设备抽象、内存、内核启动、混合精度策略
vibe::physics   微物理、辐射、PBL、陆面、积云
vibe::timeint   RK3、声波子步、Helmholtz、半隐式、时间步控制
vibe::io        NetCDF / .vibebin 读写、重启、检查点
vibe::obs       观测数据模型、观测算子（含 TL/AD）、快速辐射传输
vibe::da        控制变量、B 矩阵、代价函数、极小化、TL/AD、增量 4D-Var
vibe::verify    确定性/分类/概率/邻域四类评分
vibe::driver    顶层装配与 main
~~~

**两条硬约束**：只有 vibe::io 允许包含 NetCDF 头文件；只有 vibe::driver 允许调用所有模块。

---

## 6. 精度与并行策略

三级并行：

1. **MPI**：二维 $(p_x, p_y)$ 域分解，只切水平维。halo 交换采用三阶段算法
   （左右 -> 上下 -> 角），见 [03_grid_and_nesting.md](03_grid_and_nesting.md)。
2. **OpenMP**：在内核最外层循环上并行，对 Field 的行主序布局天然友好。
3. **GPU**：vibe::gpu 提供 DeviceBuffer、Stream 与类型擦除的 launch()，
   业务代码不出现内核启动语法。

混合精度策略 [G6][G7][B11]：

* **状态存储一律 FP64**（守恒量），保证长时间积分的质量守恒不被舍入污染；
* **趋势与内核算术默认 FP32**，可配置为 FP64；
* **归约使用补偿求和**（Kahan [G2] / Neumaier [G3]），把 FP32 归约误差压到 FP64 量级；
* **Helmholtz 采用迭代精化** [G5]：FP32 求解 + FP64 残差修正，直到 FP64 残差达标。

---

## 7. 学习路径建议

1. [02_dynamical_core.md](02_dynamical_core.md)：把方程组与离散化读通；
2. [03_grid_and_nesting.md](03_grid_and_nesting.md)：网格、度量项、嵌套与变分辨率；
3. [04_time_integration.md](04_time_integration.md)：时间分裂与半隐式；
4. [05_gpu_hybrid_precision.md](05_gpu_hybrid_precision.md)：后端抽象与混合精度；
5. [06_4dvar.md](06_4dvar.md) 与 [07_tangent_linear_adjoint.md](07_tangent_linear_adjoint.md)：同化与伴随；
6. [08_physics.md](08_physics.md)：物理参数化；
7. [09_verification.md](09_verification.md) 与 [10_postprocessing.md](10_postprocessing.md)：检验与后处理；
8. [11_config_and_io.md](11_config_and_io.md) 与 [12_build_and_test.md](12_build_and_test.md)：工程化；
9. [00_architecture.md](00_architecture.md)：接口基线（写代码前必读）。

---

## 8. 本章直接引用的文献

[D1] [D2] [D3] [D4] [D6] [D9] [D10] [D11] [G2] [G3] [G5] [G6] [G7]
[N1] [N3] [N4] [N5] [N6] [T2] [T5] [T6] [T7] [V3] [V10] [B11]
—— 完整条目见 [references.md](references.md)。

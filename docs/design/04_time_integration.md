# 04 · 时间积分

> 本章给出 VIBE-Model 的两条时间推进路线：**分裂显式**（RK3 + 声波子步）与
> **半隐式**（Crank-Nicolson + 3D Helmholtz），以及它们的稳定性、精度与实现细节。
> 文献编号见 [references.md](references.md)。

---

## 1. 问题的来源：声波与重力波的刚性

大气模式中特征时间尺度的分离非常剧烈。取 $\Delta x = 2$ km、$c_s \approx 340$ m/s：

| 过程 | 特征速度 | 显式稳定的时间步 |
|---|---|---|
| 声波 | $c_s \approx 340$ m/s | $2\Delta x/c_s \approx 12$ s（含安全系数约 6 s） |
| 平流 | 风速约 20 m/s | $0.8\Delta x/U \approx 80$ s |
| 深重力波 | $NH/\pi \approx 60$ m/s | 约 27 s |

若全部显式，时间步被声波限制，浪费 5 到 20 倍的计算量。两种标准解法：

1. **分裂显式**：大时间步用显式格式，声波项在更小的时间间隔上用子步推进 [D1][D5][D6][T5]；
2. **半隐式**：对线性声波-重力波项用隐式格式（Crank-Nicolson），代价是求解椭圆型方程
   [T1][T2][T6][T7][T8]。

---

## 2. 路线 A：RK3 + 声波子步

### 2.1 慢过程的 RK3

采用 Wicker & Skamarock (2002) 的低存储 SSP-RK3 [T5]：

$$\begin{aligned}
\phi^{(1)} &= \phi^{t} + \frac{\Delta t}{3}\,R(\phi^{t}),\\
\phi^{(2)} &= \phi^{t} + \frac{\Delta t}{2}\,R(\phi^{(1)}),\\
\phi^{t+\Delta t} &= \phi^{t} + \Delta t\,R(\phi^{(2)}),
\end{aligned}$$

其中 $R$ 是**慢过程**趋势：平流、科氏力、物理参数化与数值阻尼。

**为什么用低存储形式？** 只需保存 $\phi^{t}$ 与当前状态，内存开销为两倍而不是四倍，
在 GPU 上尤其重要。

### 2.2 声波子步（前向-后向）

在每个 RK 阶段内，把该阶段的慢趋势冻结为常数，用 $n$ 个前向-后向子步
（$\Delta\tau = \Delta t_{\text{stage}}/n$）推进线性声波-重力波项 [D1][D6]：

$$\begin{aligned}
\rho^{\tau+\Delta\tau} &= \rho^{\tau} - \Delta\tau\,\nabla\cdot(\rho_0\mathbf{u}^{\tau}),\\
\pi'^{\tau+\Delta\tau} &= \pi'^{\tau} + \Delta\tau\,\frac{R_d}{c_v}\frac{\pi_0}{\rho_0}
   \Bigl[-\nabla\cdot(\rho_0\mathbf{u}^{\tau})\Bigr],\\
\mathbf{u}^{\tau+\Delta\tau} &= \mathbf{u}^{\tau}
   - \Delta\tau\,c_p\theta_0\,\nabla\pi'^{\tau+\Delta\tau}.
\end{aligned}$$

语义就是「先更新质量与 Exner，再用新的 Exner 更新动量」。第 2 式与第 1 式合起来
等价于在定 $\theta$ 的假设下对状态方程做时间微分。

### 2.3 稳定性

对线性声波，前向-后向格式等价于蛙跳格式，放大因子模恒为 1（中性）。稳定性条件为

$$\Delta\tau \le \frac{2}{c_s\sqrt{\dfrac{1}{\Delta x^2}+\dfrac{1}{\Delta y^2}+\dfrac{1}{\Delta z^2}}}.$$

实践中取安全系数 0.8，并按域内最大声速与最小层厚选择子步数：

$$n \ge \frac{2\,c_s^{\max}\,\Delta t}{0.8\,\min(\Delta x,\Delta y,\Delta z_{\min})}.$$

接口：`AcousticSubstepper::required_substeps` 与 `AcousticSubstepper::acoustic_cfl`。

### 2.4 HEVI：垂直隐式声波

若把垂直声波项改用 Crank-Nicolson，$n$ 只受**水平**声波约束；在细垂直网格
（例如近地层 20 m 层厚）或陡地形下收益显著 [T2][T6]。离散后得到逐列三对角系统

$$-A_k\,\pi'_{k-1} + B_k\,\pi'_k - C_k\,\pi'_{k+1} = f_k,$$

$$A_k = \left(\frac{\Delta\tau}{2}\right)^2 \frac{c_p \theta_{0,k}\,\rho_{0,k}}{\Delta z_k^2},
\qquad
B_k = \frac{1}{(\rho_0 c_s^2)_k} + A_k + A_{k+1}.$$

$A_k$ 只依赖参考态与 $\Delta\tau$，因此可以在整个时间步内预计算并复用 LU 分解
（`thomas_factorize` 与 `thomas_solve_factored`）。

---

## 3. 路线 B：半隐式 + 3D Helmholtz

### 3.1 推导

线性声波-重力波子系统（$\mathbf{N}$ 为非线性与物理项）：

$$\frac{\partial \mathbf{u}}{\partial t} = -c_p\theta_0\nabla\pi' + \mathbf{N},\qquad
\frac{\partial \pi'}{\partial t} = -\kappa\,\nabla\cdot(\rho_0\mathbf{u}),\qquad
\kappa \equiv \frac{R_d}{c_v}\frac{\pi_0}{\rho_0}.$$

对线性项用 Crank-Nicolson、对 $\mathbf{N}$ 用二阶外推：

$$\mathbf{u}^{n+1} = \mathbf{u}^n + \Delta t\,\mathbf{N}
  - \frac{\Delta t}{2}c_p\theta_0\nabla(\pi'^{n+1}+\pi'^n),$$

$$\pi'^{n+1} = \pi'^n - \Delta t\,\kappa\,G
  + \left(\frac{\Delta t}{2}\right)^2\kappa\,\nabla\cdot\!\bigl(\rho_0 c_p\theta_0\nabla(\pi'^{n+1}+\pi'^n)\bigr),$$

其中

$$G = \nabla\cdot(\rho_0\mathbf{u}^n) + \frac{\Delta t}{2}\,\nabla\cdot(\rho_0\mathbf{N}).$$

整理后得到关于 $\pi'^{n+1}$ 的 Helmholtz 方程：

$$\pi'^{n+1} - \nabla\cdot\bigl(a\,\nabla\pi'^{n+1}\bigr)
= \pi'^n - \Delta t\,\kappa\,G + \nabla\cdot\bigl(a\,\nabla\pi'^n\bigr),
\qquad a = \left(\frac{\Delta t}{2}\right)^2\!\frac{R_d}{c_v}\,\pi_0\,c_p\,\theta_0 .$$

注意 $a$ **不含 $\rho_0$**（$(R_d/c_v)\pi_0$ 已吸收密度的角色），这使系数的构造与存储更省。
实现中把方程写成 $-\nabla\cdot(a\nabla x) + b\,x = f$、$b \equiv 1$ 的形式，
交给 `HelmholtzSolver` 接口。

### 3.2 求解器族

| 求解器 | 算法 | 复杂度 | 适用 |
|---|---|---|---|
| `VerticalTridiagonal` | 逐列 Thomas [T10] | 精确，$O(n_x n_y n_z)$ | HEVI、教学 |
| `KrylovJacobi` | Jacobi 预条件 BiCGSTAB [T12] | 迭代，每步 $O(n)$ | 一般三维问题 |
| `KrylovMultigrid` | 几何多重网格 V-cycle 预条件 [T13] | 收敛率与网格无关 | 大 $n_z$、强各向异性 |
| 迭代精化包装 [G5] | FP32 求解 + FP64 残差修正 | 额外常数次残差计算 | 混合精度 |

收敛判据：

$$\frac{\|r\|_2}{\|f\|_2} < \varepsilon, \qquad r = f - L(x).$$

### 3.3 稳定性与精度

Crank-Nicolson 对线性声波与重力波的放大因子模恒为 1（中性、无条件稳定）[T7]。
整体稳定性由平流项的外推格式稳定域约束：

$$\Delta t \le \frac{\text{CFL}_{\text{adv}}\,\min(\Delta x,\Delta y)}{|\mathbf{U}|_{\max}},
\qquad \text{CFL}_{\text{adv}} \approx 0.8\ (\text{四阶}),\quad 0.4\ (\text{二阶}).$$

**中性格式的隐患**：不耗散意味着短波噪声不会被自动抑制，必须辅以

* Robert-Asselin-Williams 时间滤波（`set_time_filter`）[T3]；
* 水平散度阻尼（`numerics.divergence_damping`）[D6]；
* 上层海绵层（`Damping::apply_sponge`）。

### 3.4 迭代中心隐式

Bénard [T7][T8] 指出，对含重力波的方程组，「解耦」与「耦合」线性系统的选择显著影响
稳定性与精度。本模式通过 `SemiImplicitIntegrator::set_outer_iterations` 支持
**迭代中心隐式**：把 $\pi'^{n+1}$ 的当前估计反馈回非线性项的评估，重复若干次。
每个外层迭代需要一次额外的 Helmholtz 求解，但对大时间步的精度提升明显。

---

## 4. 时间步控制与稳定性监控

`timeint::TimeStepController` 的自适应规则：

$$\Delta t_{\text{new}} = \mathrm{clamp}\!\left(
\gamma\,\frac{\text{CFL}_{\text{target}}}{\text{CFL}_{\text{now}}}\Delta t_{\text{old}},\;
\Delta t_{\min},\; \Delta t_{\max}\right), \qquad \gamma = 0.85 .$$

监控量（`StabilityReport`）：平流 CFL、声波 CFL、垂直 CFL、域内最大风速与最小层厚、
非有限值检测、物理过程的隐含稳定性指标。

**中止判据**：出现非有限值，或平流 CFL 超过 4（此时继续积分已经没有意义）。

---

## 5. 建议的对比实验

| 实验 | 目的 | 期望结果 |
|---|---|---|
| 静止等温大气，两种积分器各跑 1 小时 | 静止解的保持能力 | 速度扰动保持在机器精度量级 |
| 暖泡对流，$\Delta x = 2$ km，两种积分器 | 对比有效分辨率 | 单体位置与强度一致（差异小于 5%） |
| 山波 + 陡地形 | 检验 HEVI 的必要性 | 全显式声波子步需要 $n \ge 10$ 才稳定 |
| 声波脉冲传播 | 相速与振幅误差 | 半隐式中性格式无振幅衰减但存在相速误差 |
| 时间步减半两次 | 二阶收敛性 | 误差以 $O(\Delta t^2)$ 下降 |

---

## 6. 代码对照表

| 功能 | 接口 | 文件 |
|---|---|---|
| 积分器接口 | `timeint::Integrator` | [integrator.hpp](../../include/vibe/time/integrator.hpp) |
| RK3 + 声波子步 | `RungeKutta3Integrator` | [runge_kutta.cpp](../../src/time/runge_kutta.cpp) |
| 声波子步与 HEVI | `AcousticSubstepper` | [acoustic_substep.cpp](../../src/time/acoustic_substep.cpp) |
| Helmholtz 求解器 | `HelmholtzSolver` 与工厂 | [helmholtz.cpp](../../src/time/helmholtz.cpp) |
| 半隐式 | `SemiImplicitIntegrator` | [semi_implicit.cpp](../../src/time/semi_implicit.cpp) |
| 时间步控制 | `TimeStepController` | [timestep_control.cpp](../../src/time/timestep_control.cpp) |

---

## 7. 参考文献

[D1] Klemp & Wilhelmson (1978) · [D5] Skamarock & Klemp (2008) ·
[D6] Klemp, Skamarock & Dudhia (2008) · [T1] Kwizak & Robert (1971) ·
[T2] Tapp & White (1976) · [T3] Robert (1982) · [T5] Wicker & Skamarock (2002) ·
[T6] Cullen (1990) · [T7] Bénard (2003) · [T8] Bénard (2004) ·
[T10] Thomas (1949) · [T11] Saad & Schultz (1986) · [T12] van der Vorst (1992) ·
[T13] Briggs et al. (2000) · [T14] Wood et al. (2014) · [T15] Baldauf (2010) ·
[G5] Haidar et al. (2018)

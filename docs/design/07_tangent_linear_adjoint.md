# 07 · 切线性模式与伴随模式

> 本文给出切线性（TL）与伴随（AD）的数学定义、手写伴随的构造规则与检查清单、
> 点积检验的理论与数值细节、不连续过程的处理、检查点策略、内存/计算权衡表、
> 常见错误清单，以及与本项目代码的对照与验证流程。
> 文献编号见 [references.md](references.md)。

---

## 1. 切线性的定义与有效性条件

### 1.1 Fréchet 导数

设模式算子 $M: \mathbb R^n \to \mathbb R^n$ 把窗口起点的状态映到某时刻的状态。对基础态 $x^b$ 与扰动 $\delta x$，若存在线性算子 $M'(x^b)$ 使得

$$
M(x^b + \delta x) = M(x^b) + M'(x^b)\,\delta x + o(\|\delta x\|),
\qquad \delta x \to 0 ,
\tag{7.1}
$$

则称 $M'(x^b)$ 是 $M$ 在 $x^b$ 处的 **Fréchet 导数**（切线性算子）。切线性模式就是它的计算机实现 [A4] Errico (1997)。

**离散视角**：如果 $M$ 由有限个算术/初等函数组合而成，$M'(x^b)$ 就是 $M$ 的 Jacobi 矩阵，可用**逐语句前向微分**构造（[A1] Giering & Kaminski 1998）：

$$
\frac{\partial}{\partial x}(a \cdot b) = (\partial a)\cdot b + a\cdot(\partial b),\qquad
\frac{\partial}{\partial x}(a / b) = \frac{\partial a}{b} - \frac{a\,\partial b}{b^2},
$$
$$
\frac{\partial}{\partial x}\phi(a) = \phi'(a)\,\partial a,\qquad
\frac{\partial}{\partial x}\max(a, b) = \text{对应分支的导数（冻结开关）}.
$$

### 1.2 有效性条件

TL 近似只有在下列条件下才"有效"（[A4][A8]）：

1. **可微性**：$M$ 在 $x^b$ 的邻域内可微。不连续过程（饱和调整、对流开关、正定截断、WENO 权重切换）必须特殊处理（§6）。
2. **扰动足够小但不可忽略**：$\|\delta x\|$ 要小到高阶项可忽略，又要大到高于浮点噪声（见 §5 的最优 $\epsilon$）。
3. **有效时间窗**：$M'$ 的误差随 $\|\delta x\|$ 与积分时间增长。定义为

$$
\tau_{\text{valid}} \approx \frac{\ln(\text{容差})}{\lambda_{\max}},
\qquad \lambda_{\max} = \max_k \Re\,\mu_k\!\left(M'\right),
\tag{7.2}
$$

即误差放大到超过容差所需的时间。混沌系统中 $\lambda_{\max} > 0$，有效窗为几天量级 [A4]。因此 4D-Var 的窗口长度、外层重线性化间隔必须与 $\tau_{\text{valid}}$ 匹配。

4. **线性化点更新**：增量 4D-Var 每个外层都要重新线性化（[V3]），否则 TL 的偏差会累积成系统性误差 [V22]。

---

## 2. 伴随的数学定义与内积选择

### 2.1 定义

给定内积 $\langle \cdot, \cdot \rangle$，线性算子 $L: X \to Y$ 的**伴随** $L^{\mathrm T}: Y \to X$ 由下式唯一确定：

$$
\big\langle L\,u,\ v \big\rangle_Y = \big\langle u,\ L^{\mathrm T} v \big\rangle_X ,
\qquad \forall u \in X,\ v \in Y .
\tag{7.3}
$$

4D-Var 的梯度正是伴随模式对观测加权残差的解：

$$
\nabla_{x_0} J = \sum_i M_i^{\mathrm T} H_i'^{\mathrm T} R^{-1}\big(H_i M_i x_0 - y_i\big) + B^{-1}(x_0 - x^b).
\tag{7.4}
$$

### 2.2 内积的选择

**这是最容易出错的环节。** 内积不同，伴随矩阵不同：

* **欧氏内积** $\langle u, v\rangle = \sum_i u_i v_i$：伴随 = 矩阵转置。最简单、最常用于点积检验。
* **格点体积加权** $\langle u, v\rangle_W = \sum_g w_g u_g v_g$（$w_g$ = 单元体积）：伴随 = $W^{-1} L^{\mathrm T} W$。若 $L$ 是插值（稀疏权重 $w_{ig}$），加权伴随需要除以格点体积。
* **观测误差加权** $\langle u, v\rangle_{R^{-1}}$：等价于先乘 $R^{-1}$，实现中通常把 $R^{-1}$ 作为独立的一步。

**约定**：本项目的算子层点积检验（`check_adjoint`）使用**欧氏内积**，此时 `scatter_adjoint` 就是 `sample` 的逐项矩阵转置，误差只来自浮点舍入。$B$ 的格点体积加权在控制变量变换 $U$ 内部体现。若要对 $\Delta x$ 侧使用体积加权度量，则 `applyAD` 必须同时除以单元体积，否则 (7.3) 不成立。这个取舍在 `src/obs/obs_operator.cpp` 的注释与 `docs/design/06_4dvar.md` §8.4 中均明确说明。

### 2.3 转置与逆的区别

* 转置（伴随）：$L^{\mathrm T}$，满足 (7.3)；
* 逆：$L^{-1}$，恢复输入。

二者只有在 $L$ 正交时才相同。四维变分中的常见混淆是把 `from_state`（$U^{-1}$）当成 $U^{\mathrm T}$ 使用；本实现用恒等式

$$
U^{\mathrm T} = U^{-1}\,(U U^{\mathrm T}) = \texttt{applyInvSqrtB}\big(\texttt{applyB}(w)\big)
\tag{7.5}
$$

严格求出 $U^{\mathrm T}$，不依赖 $U$ 对称的假设（见 `src/da/cost_function.cpp`）。

---

## 3. 手写伴随的构造规则与检查清单

[ A1 ] 给出了一套机械化的规则。设前向代码是一串语句 $S_1, S_2, \dots, S_K$，每条语句把某些输入变量 $u_1, \dots, u_m$ 更新为输出 $v_1, \dots, v_l$。

### 3.1 构造规则

1. **保存中间量**：TL 需要基础态在每个语句处的中间量（如插值权重、$\theta_b$、$\pi_b$）。存储或重算（§7）。
2. **逐语句前向微分**：把 TL 写成与基础态相同的语句序列，只对扰动线性化，系数冻结。
3. **逆序**：AD 从末语句到首语句。
4. **每条语句替换为其转置**：若前向语句为 $\delta v_j = \sum_i A_{ji}\,\delta u_i$，则 AD 累加

$$
\bar u_i \mathrel{+}= \sum_j A_{ji}\,\bar v_j .
\tag{7.6}
$$

5. **累加而非覆盖**：一个输入可能被多条语句使用，AD 必须把贡献**累加**到同一个伴随变量上。
6. **清零**：伴随变量在每次反向传播前必须清零（本项目的算子把 `applyAD` 定义为累加，调用方负责清零）。
7. **不连续语句**：用基础态的开关状态（冻结开关），或光滑化，或从 TL/AD 中删除该物理过程（§6）。

### 3.2 逐语句规则表

| 前向语句 | TL | AD（逆序、累加） |
|---|---|---|
| $c = a \cdot b$ | $\delta c = \delta a \cdot b + a\cdot\delta b$ | $\bar a \mathrel{+}= b\,\bar c$；$\bar b \mathrel{+}= a\,\bar c$ |
| $c = a / b$ | $\delta c = \delta a/b - a\,\delta b/b^2$ | $\bar a \mathrel{+}= \bar c/b$；$\bar b \mathrel{-}= a\,\bar c/b^2$ |
| $c = f(a)$ | $\delta c = f'(a)\,\delta a$ | $\bar a \mathrel{+}= f'(a)\,\bar c$ |
| $c += a$（累加） | $\delta c += \delta a$ | $\bar a \mathrel{+}= \bar c$ |
| $c = 0$ | $\delta c = 0$ | 无需处理（丢弃 $\bar c$） |
| $\text{if }(p)\, S$ | 用基础态的 $p$ 决定是否包含 $S$ 的微分 | 同样用基础态的 $p$ |
| $\text{for } k=1..N$ | 同一循环 | **逆序**循环 $k=N..1$ |
| $y_i = \sum_g w_{ig} x_g$（插值） | $\delta y_i = \sum_g w_{ig}\delta x_g$ | $\bar x_g \mathrel{+}= \sum_i w_{ig}\bar y_i$（**转置使用权重**） |
| $x = x + \alpha y$ | $\delta x = \delta x + \alpha\delta y$ | $\bar y \mathrel{+}= \alpha\bar x$；$\bar x$ 保持 |

### 3.3 检查清单（每次写完 AD 后逐条核对）

- [ ] 所有循环都逆序了吗？（RL 阶段、时隙、物种、层）
- [ ] 所有前向语句都有对应的 AD 语句吗？（逐行对照）
- [ ] 每条 AD 语句都是对应前向语句的**转置**吗？（乘除号、符号、除法中的平方）
- [ ] 插值权重的转置用对了吗？（$w_{ig}$ 与 $w_{gi}$ 的区别；边界钳制的重复累加）
- [ ] 基础态依赖都包含了吗？（$H'$ 的系数是否都冻结在 $x^b$ 上而非 $\delta x$）
- [ ] 所有伴随累加都是 `+=` 而不是 `=`？
- [ ] 每次反向传播前伴随变量清零了吗？
- [ ] 不连续过程用冻结开关了吗？
- [ ] 点积检验用**多个**随机种子通过了吗？

---

## 4. 离散伴随 vs 连续伴随

两条路线 [A3] Sirkes & Tziperman (1997)：

$$
\underbrace{\text{连续方程} \xrightarrow{\text{离散}} \text{模式}}_{\text{先离散}} \xrightarrow{\text{线性化}} \text{TL} \xrightarrow{\text{转置}} \text{AD}_{\text{离散}},
\qquad
\underbrace{\text{连续方程} \xrightarrow{\text{线性化}} \text{连续 TL} \xrightarrow{\text{转置}} \text{连续 AD} \xrightarrow{\text{离散}} \text{AD}_{\text{离散}}}_{\text{先线性化}} .
$$

两者**不交换**：

* 先离散后伴随得到的是**精确的离散伴随**，点积检验（在离散内积下）达到机器精度；这是 4D-Var 的正确选择，因为代价函数的极小化发生在离散空间。
* 先线性化后离散，若 TL 与 AD 使用不同的离散格式（例如 TL 用中心差分、AD 用单侧差分），点积检验会失败，误差为 $O(\Delta x)$。
* 另一个陷阱：**非线性模式的离散格式本身**（如通量形式的平流）与其 TL 的乘积法则必须一致。若前向用 $$-\nabla\cdot(\rho u)$$ 的离散通量形式，TL 必须逐项微分**同一个离散算子**，而不能用连续形式再离散。本项目的 TL 核在 `src/da/tangent_linear.cpp` 中把密度方程写成 `-rho div(u) - u.grad(rho)` 的离散分裂形式，并按乘积法则补齐 `-u'.grad(rho_b)` 项，保证它的 Fréchet 导数与所写的 TL 语句逐项一致（该点由 `tests/unit/test_tl_ad.cpp` 中同构的非线性模式验证）。

---

## 5. 点积检验：理论与数值细节

### 5.1 算法

$$
\text{LHS} = \big\langle M'\delta x,\ \delta y \big\rangle,\qquad
\text{RHS} = \big\langle \delta x,\ M'^{\mathrm T}\delta y \big\rangle,\qquad
\varepsilon_{\text{rel}} = \frac{|\text{LHS} - \text{RHS}|}{|\text{LHS}| + |\text{RHS}|} .
\tag{7.7}
$$

`vibe::da::check_adjoint` 与 `obs::ObservationOperator::check_adjoint` 实现上式，返回对称相对误差（`vibe::rel_error`），并用固定种子生成随机向量以保证可复现。

### 5.2 为什么期望 $\sim 10^{-15}$

若 AD 是 TL 的严格转置，则 LHS 与 RHS 在精确算术下相等。实际差异只来自：

1. 两种求和顺序造成的浮点舍入（每次 FMA 约 $10^{-16}$ 相对误差，累积后 $\sim n \cdot 10^{-16}$，对 $n \sim 10^3$ 维向量仍约 $10^{-13}$）；
2. 编译器的 FMA/向量化重排；
3. 混合精度（若启用）。

因此 $\varepsilon_{\text{rel}} \lesssim 10^{-13}$ 是正常值；容差 $10^{-8}$ 留了 5 个数量级的余量，可过滤真正的实现错误。若误差在 $10^{-3} \sim 10^{-1}$，几乎肯定是伴随代码有 bug，而不是数值问题。

### 5.3 随机向量的选择

* 用**零均值单位方差正态**随机向量（不要用全 1 或稀疏向量，后者会掩盖未转置的稀疏结构）；
* 至少换 3–5 个种子；某些实现错误只在特定索引组合下暴露；
* 状态向量的尺度差异很大（$u \sim 10$、$\pi \sim 10^{-4}$），可先对 $\delta x$ 按变量做归一化再检验，避免某一变量掩盖其它变量的误差。

### 5.4 与有限差分结合的顺序

标准流程是先做有限差分（TL 与前向一致），再做点积检验（TL 与 AD 互为转置）：

$$
\text{FD 检验：} \quad M(x+\epsilon\delta x) \xrightarrow{\text{差分}} \text{对比 } M'\delta x ,
$$
$$
\text{AD 检验：} \quad M'\delta x \ \text{与}\ M'^{\mathrm T}\delta y \ \text{做点积} .
$$

两者都通过，TL/AD 才能用于 4D-Var。

---

## 6. 不连续过程的处理

气象模式中不连续（不可微）的过程包括：饱和调整/凝结阈值、对流触发与关闭、云顶夹卷、正定截断、WENO/JFNK 权重切换、边界与地形掩膜。

三种处理方式（[A6] Zou et al. 1993；[A8] Mahfouf 1999）：

### 6.1 冻结开关（frozen switch）

用**基础态**的开关状态决定 TL/AD 的路径，并把开关本身视为常数（导数为 0 或 1）：

$$
H(x) = \begin{cases} H_1(x), & s(x^b) = 1 \\ H_2(x), & s(x^b) = 0 \end{cases}
\quad\Longrightarrow\quad
H'\,\delta x = \begin{cases} H_1'\,\delta x, & s(x^b)=1 \\ H_2'\,\delta x, & s(x^b)=0 \end{cases}.
$$

优点：保持伴随的严格性；缺点：忽略开关的敏感性（在开关边界附近误差大）。这是业务系统的默认选择。

### 6.2 光滑化

用连续函数逼近阶跃，例如用 `smoothstep` 或 $\tanh$ 替代 `max`/`min`：

$$
\max(0, a) \approx \frac{a + \sqrt{a^2 + \epsilon^2}}{2},\qquad
\tanh(a/\epsilon) \text{ 替代符号函数}.
$$

优点：处处可微；缺点：引入偏差，且 $\epsilon$ 需要调参（太小则 TL 病态，太大则物理失真）。

### 6.3 简化物理（simplified physics）

在 TL/AD 中只保留**可微且主导**的物理过程（如大尺度凝结、垂直扩散），完全删除不连续过程（深对流、微物理相变）。[A5] Navon et al. (1992) 用绝热版本模式做变分同化是经典做法；[A8] 系统比较了各物理过程对 TL 近似的影响。

本项目的 `PhysicsLinearization` 枚举提供四种选择：`Adiabatic`（默认，绝热）、`Simplified`、`FrozenSwitch`、`FullPhysics`。TL 核当前实现的是绝热动力学。

---

## 7. 检查点与 revolve 算法

AD 需要基础态轨迹上的中间量。四种策略（`CheckpointStrategy`）：

| 策略 | 内存 | 时间 | 说明 |
|---|---|---|---|
| `StoreAll` | $O(N_{\text{steps}})$ 个状态 | 1 次前向 | 最简单，内存最大 |
| `Multilevel` | $O(\sqrt{N})$ | $\sim 2$ 次前向 | 分块检查点：存块边界，块内重算 |
| `Revolve` | $O(\log N)$ 或 $O(\sqrt{N})$ | $\sim 2\!-\!3$ 次前向 | Griewank-Walther 最优调度 [A2] §12 |
| `Recompute` | $O(1)$ | $O(N^2)$ | 时间换内存，仅教学用 |

**Revolve 的基本思想**：把 $N$ 步分成若干"检查点段"，只保存段首状态；反向传播时从最近的检查点重算段内轨迹。设 $c$ 个检查点、每段 $N/c$ 步，则

$$
\text{内存} = c \cdot n,\qquad
\text{时间} = N + N/c \cdot \frac{N}{2} \quad(\text{最坏情形}),
$$

最优 $c = \sqrt{N}$ 给出内存 $O(\sqrt N)$、时间约 2 倍。本项目 `Trajectory` 与 `memory_bytes()` 对应 `StoreAll`；策略枚举已暴露给调用方。生产实现在 $N$ 大、$n$ 大时必须启用 revolve，否则内存会超过观测与状态本身的量级。

---

## 8. 内存与计算权衡表

设 $n$ = 状态自由度、$N$ = 时间步数、$s$ = 每步保存的中间量大小。

| 方案 | 内存 | 前向次数 | 适用 |
|---|---|---|---|
| StoreAll | $N \cdot n \cdot 8$ B | 1 | $N n \lesssim 10^9$（例如 $n=10^6, N=100$ → 0.8 GB） |
| StoreAll（只存基础态） | $N \cdot n \cdot 8$ B | 1 | 推荐：趋势可由基础态重算 |
| Multilevel | $\sqrt{N}\,n\,8$ B | 2 | 大 $N$、内存受限 |
| Revolve | $O(\log N)\,n\,8$ B | 2–3 | 约束记忆体的最优选择 |
| Recompute | $n\,8$ B | $N(N+1)/2$ | 仅教学 |

**计算成本**：一次 TL 前向约 $2\times$ 非线性一步；一次 AD 反向约 $2\!-\!4\times$（取决于算术强度与内存访问）。因此一次 4D-Var 梯度评估的总成本约

$$
C_{\text{grad}} \approx N\big(C_M + C_{\text{TL}} + C_{\text{AD}}\big) + 2 p\,C_H
\approx N\,(1 + 2 + 3)\,C_M ,
$$

即约 6 倍"把窗口积一遍"的成本。这解释了为什么内层迭代数与降分辨率因子是 4D-Var 成本的主要旋钮。

---

## 9. 常见错误清单

| # | 错误 | 症状 | 修正 |
|---|---|---|---|
| 1 | 插值矩阵未转置（$$\bar x_g \mathrel{+}= w_{ig}\bar y_i$$ 写成了 $$w_{gi}$$） | 点积检验 $\sim 10^{-2}$ | 用同一权重表重写散射；对照 `sample/scatter` |
| 2 | 遗漏基础态依赖（如 $$-u'\cdot\nabla\rho_b$$） | 点积检验通过但 FD 检验呈 $O(1)$ 偏差 | 用非线性模式做 FD 检验；补齐乘积法则项 |
| 3 | 语句顺序错乱（AD 未逆序） | 点积检验失败，误差随 $N$ 增大 | 逐语句逆序对照 |
| 4 | 边界权重重复累加（钳制索引未按同一映射累加） | 点积检验在边界附近失败 | 用 `off_clamped` 统一索引映射 |
| 5 | 原地覆盖而非累加（`=` vs `+=`） | 后写的结果覆盖前面算子的贡献 | 所有 AD 语句用累加 |
| 6 | halo 未交换 | 并行与串行结果不一致 | 在 TL/AD 每步后交换 halo |
| 7 | RK 阶段遗漏（只写了第一个阶段） | FD 检验误差 $O(\Delta t)$ | 逐阶段写出并逆序 |
| 8 | 伴随变量未清零 | 第二次调用结果异常 | 每次反向传播前清零 |
| 9 | 把 $$U^{-1}$$ 当 $$U^{\mathrm T}$$ | 代价函数梯度下降缓慢或振荡 | 用 (7.5) 的恒等式 |
| 10 | 时间积分器中的非线性限制器进入 TL/AD | 点积检验失败 | 冻结开关或删除该限制器 |
| 11 | 用了未来时刻的基础态（"先时间"错误） | 点积检验在长时间窗失败 | 检查轨迹索引与时间递增方向 |
| 12 | 单精度累积 | 点积检验 $\sim 10^{-4}$ | 伴随累加用 double |

---

## 10. 代码对照表与本项目验证流程

### 10.1 代码对照

| 概念 / 算法 | 类 / 函数 | 文件 | 文献 |
|---|---|---|---|
| 离散伴随构造规则 | 匿名命名空间内的 `tendency_tl` / `tendency_ad` | `src/da/tangent_linear.cpp` | [A1] |
| 切线性模式 | `da::TangentLinearModel::step/propagate/linear_growth_rate` | `src/da/tangent_linear.cpp` | [A4][A5] |
| 伴随模式 | `da::AdjointModel::step_adjoint/propagate/propagate_with_trajectory` | `src/da/tangent_linear.cpp` | [A1][A2] |
| 轨迹存储与内存统计 | `da::Trajectory` | `src/da/tangent_linear.cpp` | [A2] |
| 检查点策略 | `da::CheckpointStrategy` | `include/vibe/da/tangent_linear.hpp` | [A2] |
| 不连续过程处理 | `da::PhysicsLinearization` | `include/vibe/da/tangent_linear.hpp` | [A6][A8] |
| 点积检验 | `da::check_adjoint`、`AdjointCheckResult` | `src/da/tangent_linear.cpp` | [A3] |
| 切线一致性检验 | `da::check_tangent`、`TangentCheckResult` | `src/da/tangent_linear.cpp` | [A3][A4] |
| 观测算子点积检验 | `obs::ObservationOperator::check_adjoint/check_tangent` | `src/obs/obs_operator.cpp` | [A3][O8] |
| 观测算子插值转置 | `sample_scalar/sample_staggered/scatter_adjoint` | `src/obs/obs_operator.cpp` | [O8] |
| 辐射率解析雅可比 | `detail::jacobian_column` | `src/obs/radiative_transfer.cpp` | [O1][O3] |
| $$U^{\mathrm T}$$ 的组合实现 | `CostFunction::value_and_gradient` | `src/da/cost_function.cpp` | [V3][V14] |
| 测试 | `tests/unit/test_tl_ad.cpp`、`test_obs_operator.cpp` | `tests/unit/` | — |

### 10.2 本项目的验证流程（验收清单）

1. **单步伴随**：`tl.step` 与 `ad.step_adjoint` 的点积检验 $<10^{-8}$（`test_tl_ad.cpp` 用例 5）。
2. **多步伴随**：`da::check_adjoint(tl, ad, xb, dt, n, seed)` 在多个种子上通过（用例 1、2）。
3. **线性性**：`TL(a\,dx_1 + b\,dx_2) = a\,TL(dx_1) + b\,TL(dx_2)` 与 AD 的对应性质（用例 3、4）。
4. **FD 一致性**：用与 TL 核同构的非线性模式做前向差分，误差曲线呈 **U 形**（用例 7），
   $$ \varepsilon(\epsilon) \approx C_1 \epsilon + C_2\,\epsilon_{\text{mach}}/\epsilon ,
   \qquad \epsilon^{*} = \sqrt{C_2\epsilon_{\text{mach}}/C_1}. $$
   本项目的测试在 $\epsilon \in [10^{-10}, 10^{-1}]$ 上取点，要求最小值落在区间**内部**，且两端误差都大于最小值。
5. **最优 $\epsilon$**：`check_tangent` 返回的 `optimal_eps` 必须是输入列表中使误差最小的那个（用例 8）。
6. **轨迹与内存**：`Trajectory` 的状态/趋势可访问、时间戳按 $dt$ 递增、`memory_bytes()` 随步数线性增长（用例 9、10）。
7. **增长率**：零扰动返回 0，非零扰动返回有限值（用例 11）。
8. **观测算子**：探空、地面（含 10 m 对数律）、掩星、雷达、复合算子的点积检验均 $<10^{-8}$（`test_obs_operator.cpp`）。
9. **辐射传输**：解析雅可比对中心差分一致（相对误差 $<10^{-3}$），亮温落在物理范围（`test_obs_operator.cpp` 用例 13、12）。
10. **回归**：所有测试在改动 TL/AD 后必须重跑；点积检验是**不可妥协**的门槛（[A3]）。

### 10.3 本项目的实现取舍

* **自包含 TL 核**：冻结的 `dyn::Equations` 没有 TL/AD 版本，因此 `src/da/tangent_linear.cpp` 在匿名命名空间实现了自包含的 RK3 切线性核（平流用基础态速度、梯度项用基础态 `theta/pi/rho` 的**全量**（参考态 + 扰动）冻结系数，浮力用 `g\,\theta'/\theta_b`）。生产实现应复用 dyn 层的模板化内核，一次编译出非线性/TL/AD 三个版本（[A1] 的"同一份代码三种模式"原则）。
* **`eq_` 为空**：`TangentLinearModel` / `AdjointModel` 的冻结头文件没有声明析构函数，而 `dyn::Equations*` 是裸指针；本实现不持有它的所有权以避免泄漏。若要接入 dyn 层，应在头文件中改为 `std::unique_ptr<dyn::Equations>`。
* **`const_cast`**：`check_adjoint` / `check_tangent` 接收 `const` 引用，而 `propagate` 在冻结接口中是非 const 方法；实现中用 `const_cast` 推进并刷新轨迹缓存，副作用已在源码注释中说明。
* **基础态轨迹**：`record_trajectory` 在 `eq_ == nullptr` 时按"冻结基础态"记录（时间戳递增、状态不变），这保证 AD 与 TL 使用同一线性化点、点积检验严格通过；生产实现应由非线性模式逐步推进基础态。
* **共位化处理**：教学版把 `u,v,w` 视为体心共位，错位模板的差异留给 dyn 层。这不影响 TL/AD 的转置结构与点积检验，但会影响与非线性模式的一致性，因此 FD 检验的"同构非线性模式"也采用同一共位约定。

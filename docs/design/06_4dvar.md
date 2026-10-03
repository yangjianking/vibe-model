# 06 · 变分同化与增量 4D-Var

> 本文给出 VIBE-Model 4D-Var 同化系统的数学推导、离散实现规范与代码对照。
> 记号遵循 [V14] Ide et al. (1997)；所有文献编号见 [references.md](references.md)。
> 代码位置：`include/vibe/da/`、`src/da/`、`include/vibe/obs/`、`src/obs/`。

---

## 1. 统一记号与问题设置

### 1.1 符号

| 符号 | 含义 | 代码 |
|---|---|---|
| $x$ | 模式状态向量（离散自由度 $n$） | `dyn::State` / `da::Vector` |
| $x^t$ | 真值 | — |
| $x^b$ | 背景（forecast），$x^b = x^t + \varepsilon^b$ | `x_background` |
| $x^a$ | 分析，$x^a = x^t + \varepsilon^a$ | `AnalysisResult::analysis` |
| $y$ | 观测向量（$p$ 个标量） | `obs::ObsSpace` |
| $H$ | 非线性观测算子 $x \mapsto y$ | `obs::ObservationOperator` |
| $M_{0\to i}$ | 非线性模式算子（从窗口起点积到 $t_i$） | `ForecastFunction` |
| $M_i$ | $M_{0\to i}$ 的切线性算子 | `da::TangentLinearModel` |
| $M_i^{\mathrm T}$ | 其伴随 | `da::AdjointModel` |
| $B$ | 背景误差协方差 $E[\varepsilon^b \varepsilon^{b\mathrm T}]$ | `da::ControlVariableTransform` |
| $R$ | 观测误差协方差（对角近似） | `ObsSpace::inverse_variance` |
| $Q$ | 模式误差协方差（弱约束） | 接口保留 |
| $U$ | $B$ 的平方根，$B = U U^{\mathrm T}$，$\Delta x = U v$ | `applySqrtB` |
| $d_i$ | 创新向量 $y_i - H_i(x^b)$ | `CostFunction::innovations_` |

### 1.2 三个统计假设

1. 背景与观测误差无偏：$E[\varepsilon^b] = 0$，$E[\varepsilon^o] = 0$；
2. 两者不相关：$E[\varepsilon^b \varepsilon^{o\mathrm T}] = 0$；
3. 误差高斯，且 $B$、$R$ 已知（4D-Var 中 $B$ 是静态的，流依赖由集合分量或外层重线性化提供）。

在这三个假设下，Bayes 公式给出的分析是后验概率密度的众数（最大后验），等价于极小化代价函数。

---

## 2. 强约束 4D-Var

### 2.1 代价函数

强约束假设模式在窗口内完美（$Q = 0$），状态完全由初始条件决定：

$$
J(x_0) = \frac{1}{2}\left\| x_0 - x^b \right\|_{B^{-1}}^2
        + \frac{1}{2}\sum_{i=0}^{N} \left\| H_i\big(M_{0\to i}(x_0)\big) - y_i \right\|_{R^{-1}}^2 ,
\qquad \|z\|_C^2 \equiv z^{\mathrm T} C z .
\tag{6.1}
$$

$x_0$ 是窗口起点的控制量，$N$ 为时间时隙数。

### 2.2 梯度与伴随方法

将 (6.1) 对 $x_0$ 求导：

$$
\nabla_{x_0} J
= B^{-1}(x_0 - x^b)
+ \sum_i M_i^{\mathrm T} H_i'^{\mathrm T}\, R^{-1}\!\left( H_i M_i x_0 - y_i \right).
\tag{6.2}
$$

$H_i'$ 是 $H_i$ 在轨迹点上的切线性算子，$M_i$ 是切线性模式（[V1][V2][V24]）。求和**从最晚时隙向最早时隙累加**：实现顺序必须先 TL 前向记录轨迹，再从末时隙反向经 AD 传播，见 [07_tangent_linear_adjoint.md](07_tangent_linear_adjoint.md)。

### 2.3 拉格朗日乘子视角

把模式约束写成 $x_{i+1} = M_i(x_i)$，构造

$$
\mathcal L = J + \sum_i \lambda_{i+1}^{\mathrm T}\big(x_{i+1} - M_i(x_i)\big),
$$

对 $x_i$ 求导给出 $\lambda_i = M_i^{\mathrm T}\lambda_{i+1} + (\text{观测项})$，即伴随方程；$\lambda_0 = \nabla_{x_0} J$。因此 4D-Var 的梯度就是伴随模式的解，计算量为一次非线性轨迹 + 一次 TL + 一次 AD（[V2]）。

---

## 3. 弱约束 4D-Var

放宽完美模式假设，引入模式误差 $\eta_i \sim \mathcal N(0, Q_i)$：

$$
J(x_0, \{\eta_i\}) = \frac{1}{2}\|x_0 - x^b\|_{B^{-1}}^2
+ \frac{1}{2}\sum_i \|\eta_i\|_{Q_i^{-1}}^2
+ \frac{1}{2}\sum_i \|H_i(x_i) - y_i\|_{R^{-1}}^2 .
\tag{6.3}
$$

弱约束把控制向量扩展到 $\{x_0, \eta_1, \dots, \eta_N\}$，维数增加约 $(N+1)$ 倍，但能校正模式偏差（[V15]）。本实现保留 `Incremental4DVarConfig::use_weak_constraint` 开关；求 $Q^{-1}$ 项需要模式误差控制变量与相应变换，属于驱动层职责。

---

## 4. 增量方法与分辨率递进

### 4.1 增量形式

直接极小化 (6.1) 的困难：每次迭代都要积分非线性模式；观测算子虽为弱非线性，但外循环代价高；条件数由 $B$ 与 $R$ 的尺度比决定。

Courtier 等 (1994) [V3] 的**增量方法**：以当前背景 $x^b_k$ 的轨迹为线性化点，令 $\Delta x = x_0 - x^b_k$，只保留二次项：

$$
J_k(\Delta x) = \frac{1}{2}\Delta x^{\mathrm T} B^{-1} \Delta x
+ \frac{1}{2}\sum_i \Big( H_i' M_i \Delta x - d_i \Big)^{\mathrm T} R^{-1} \Big( H_i' M_i \Delta x - d_i \Big),
\tag{6.4}
$$
$$
d_i = y_i - H_i\big(M_{0\to i}(x^b_k)\big).
\tag{6.5}
$$

得到 $\Delta x_k = \arg\min J_k$ 后更新 $x^b_{k+1} = x^b_k + \Delta x_k$，进入下一外层。

### 4.2 预条件（控制变量变换）

设 $B = U U^{\mathrm T}$，令 $\Delta x = U v$，则

$$
J_k(v) = \frac{1}{2} v^{\mathrm T} v
+ \frac{1}{2}\sum_i \Big( H_i' M_i U v - d_i \Big)^{\mathrm T} R^{-1}\Big( H_i' M_i U v - d_i \Big).
\tag{6.6}
$$

背景项的条件数由 $B$ 的谱决定；写成 $v$ 之后其 Hessian 的背景部分为**单位阵**，条件数大幅下降。相应梯度为

$$
\nabla_v J = v + U^{\mathrm T} \sum_i M_i^{\mathrm T} H_i'^{\mathrm T} R^{-1}\big(H_i' M_i U v - d_i\big) + \nabla_v J_c .
\tag{6.7}
$$

本实现中 $U^{\mathrm T}$ 通过冻结接口的组合恒等式实现：

$$
U^{\mathrm T} w = U^{-1}\,(U U^{\mathrm T})\,w = \texttt{applyInvSqrtB}(\texttt{applyB}(w)),
\tag{6.8}
$$

不需要假设 $U$ 对称。

### 4.3 分辨率递进

外层必须在**全分辨率**上重算轨迹与创新（否则 $d_i$ 与线性化点不一致，[V22] 指出非切线性模式的增量 4D-Var 会引入系统性偏差）；内层可用**降分辨率**网格求解 (6.6)，因为极小化只需要近似解，且内层解的表达能力受限于控制变量的有效自由度（[V3][V10]）。本实现把 `LoopStructure::resolution_factor` 记录在日志与结果描述中；由于冻结接口不暴露网格重采样，真正的降分辨率算子由 driver 层注入（见 §12）。

$$
\underbrace{x^b_k}_{\text{全分辨率}} \xrightarrow[\text{非线性}]{\text{外层}} \text{轨迹},\ d_i
\;\longrightarrow\;
\underbrace{v_k}_{\text{降分辨率}} \xrightarrow[\text{L-BFGS}]{\text{内层}} \Delta x_k
\;\longrightarrow\; x^b_{k+1}.
$$

---

## 5. 背景误差协方差 B 的建模

### 5.1 顺序变换分解

$$
U = A_{\text{bal}}\,\Sigma\,E_{\text{eof}}\,K_h ,
\qquad B = U U^{\mathrm T},
\tag{6.9}
$$

对应 `StandardControlVariableTransform`（`src/da/control_vector.cpp`）。$K_h$ 为水平相关，$E_{\text{eof}}$ 为垂直相关，$\Sigma$ 为方差，$A_{\text{bal}}$ 为多变量平衡。

### 5.2 水平相关：隐式扩散

存储 $n\times n$ 的相关矩阵在业务尺度不可行（$n \sim 10^8$）。Weaver & Courtier (2001) [V7] 用广义扩散方程构造相关模型：

$$
C = \left( I - \frac{L^2}{2p}\nabla^2 \right)^{-p},
\qquad
\widehat{C}(k) = \left( 1 + \frac{L^2}{2p}k^2 \right)^{-p}
\xrightarrow[\ p\to\infty\ ]{} \exp\left(-\frac{L^2 k^2}{2}\right).
\tag{6.10}
$$

其傅里叶逆变换是方差为 $L^2$ 的高斯核，e-folding 距离约 $L$。$K_h = C^{1/2} = (I - \tfrac{L^2}{4p}\nabla^2)^{-p/2}$，在本实现中由 `DiffusionCorrelation` 用 $p/2$ 次隐式平滑近似，每次平滑用 CG 求解 $(I - c\nabla^2)y = x$（反射边界，$c = L_h^2/(4p)$）。反射边界保证常数是 $\nabla^2$ 的零空间、算子对称正定。

优点：内存 $O(n)$、计算 $O(n_{\text{iter}} n)$、天然多分辨率；缺点是相关函数形式受限，$p$ 小时与目标高斯核有偏差。$p = 4 \sim 8$ 是常用折中。

### 5.3 垂直相关：EOF 展开

对每个水平点，用 NMC 样本的垂直协方差 $C_v$ 做特征分解：

$$
C_v \phi_m = \lambda_m \phi_m,\qquad
E = \Phi\,\Lambda^{1/2}\Phi^{\mathrm T} .
\tag{6.11}
$$

`VerticalEofTransform::estimate_from_samples` 用**自实现的 Jacobi 旋转**（不依赖 LAPACK，`src/da/control_vector.cpp`）求对称矩阵的全部特征对，按特征值降序取前 $n_{\text{modes}}$ 个模态。`explained_variance(k)` 给出前 $k$ 个模态的累计能量占比 $\sum_{m<k}\lambda_m / \sum_m \lambda_m$。

**约定与取舍**：冻结接口要求 $B$ 的算子（`applyB/applyBinv/applySqrtB/applyInvSqrtB`）作用在同一维度的 `Vector` 上，即 $U$ 必须是**方阵**。因此本实现在垂直方向保留完整正交基（前 `n_vertical_modes` 个模态承载全部方差，其余给正则化小方差 $10^{-4}$），而不是截断降维。这样 `applyInvSqrtB` 严格可逆；代价是控制向量维数没有降低。生产实现若要用截断 EOF 降维，需要扩展接口以区分"控制空间"与"状态空间"维数。

### 5.4 多变量平衡

线性平衡关系（[V8]）由地转平衡导出：

$$
u_g = -\frac{1}{f}\frac{\partial \phi'}{\partial y}, \qquad
v_g = +\frac{1}{f}\frac{\partial \phi'}{\partial x}, \qquad
\phi' = c_p\,\theta_0\,\pi' ,
\tag{6.12}
$$

其中 $\pi'$ 是 Exner 扰动，$f = 2\Omega\sin\varphi_0$（f 平面近似，`f0_`）。`BalanceOperator::derive_wind` 实现 (6.12)，`derive_mass_adjoint` 是它的**严格转置**（注意权重含 $c_p\theta_0$ 在邻点的取值）。控制变量中的 $u,v$ 是**不平衡**风，组装时加上 (6.12) 得到的平衡风；这样 $B$ 的质量-风耦合被显式表达，避免风场与质量场各自独立同化造成的虚假不平衡。

`BalanceForm::OmegaEquation` 在冻结接口下退化为带稳定度修正因子的线性平衡（接口只暴露 $u,v$，没有 $\omega$ 输出），完整 $\omega$ 方程需要扩展接口。

### 5.5 NMC 方法

Parrish & Derber (1992) [V5]：

$$
B \approx \frac{1}{2} E\left[ \big(x^{T+24} - x^{T+12}\big)\big(x^{T+24} - x^{T+12}\big)^{\mathrm T} \right],
\tag{6.13}
$$

用同一时刻、不同预报时效的差作为误差代理。`NmcEstimator::estimate`（`src/da/background_error.cpp`）据此给出逐变量方差、水平相关廓线（沿 $x$ 的 lag 1..8）与垂直相关廓线，并用**对数线性回归**拟合长度尺度：

$$
\ln \rho(r) = a - r/L \;\Longrightarrow\; L = -1/\text{slope},
\qquad r = m\,\Delta x .
\tag{6.14}
$$

若回归失败（相关廓线非正或过短），回退到 e-folding 距离（$\rho$ 首次降到 $e^{-1}$ 的距离，线性插值）。

### 5.6 集合与混合方法

集合估计（[V19]）：

$$
B \approx \frac{1}{N-1}\sum_{i=1}^{N} \big(x_i - \bar x\big)\big(x_i - \bar x\big)^{\mathrm T},
\tag{6.15}
$$

`EnsembleEstimator::estimate` 逐变量计算集合方差（用 $1/(N-1)$ 归一），相关长度用集合偏差的相关廓线拟合。有限集合会低估方差并产生虚假远距离相关，因此需要**局地化**（Schur 乘积）：

$$
B \leftarrow B \circ \rho,\qquad
\rho(z) = \text{Gaspari-Cohn}(z),\quad |z|\ge 2 \Rightarrow \rho = 0 .
\tag{6.16}
$$

Gaspari-Cohn 是紧支撑 $C^2$ 核（[V20]），`EnsembleEstimator::localize` 在方阵（展平）情形下按索引距离施加 Schur 乘积，非方阵情形视作逐 lag 的协方差向量。混合 $B = (1-w)B_{\text{static}} + w B_{\text{ens}}$（[V23]）：`BackgroundErrorBuilder` 读取 `BackgroundErrorConfig::method`（`nmc`/`ensemble`/`hybrid`）与 `hybrid_weight`，把统计量翻译为 `ControlVariableConfig` 后调用 `make_control_variable_transform`。

---

## 6. 观测误差 R 与偏差订正

### 6.1 对角 R

`ObsSpace::inverse_variance()` 返回 $\mathrm{diag}(R^{-1}) = 1/\sigma_i^2$，不可用观测（`qc_flag > 0` 或 $\sigma \le 0$）取 0，等价于把该观测从代价函数中剔除（对角 $R$）。这是业务系统的标准近似（[O8]）。

### 6.2 Desroziers 诊断

Desroziers 等 (2005) [V18] 给出只用观测量即可诊断 $B$、$R$、$A$ 的恒等式。`diagnose_ob_stats` 计算：

$$
\text{rms}(O-B) = \sqrt{\frac{1}{p}\sum_i d_i^2},\qquad
\text{rms}(O-A) = \sqrt{\frac{1}{p}\sum_i (d_i - K d_i)^2},
$$
$$
\chi^2 = \frac{1}{p}\sum_i \frac{d_i^2}{\sigma_i^2}\ \ (\text{期望}\approx 1),
\qquad
\text{Gleit} = \frac{\text{rms}(O-A)}{\text{rms}(O-B)}\ \ (\text{期望}\approx \sqrt{1/2}).
\tag{6.17}
$$

若 $\chi^2 \gg 1$：$R$ 被低估或 $B$ 被高估；若 $\chi^2 \ll 1$：相反。Gleit 比偏大说明分析对观测的拟合不足（$B/R$ 权重偏大）。

### 6.3 变分偏差订正（VarBC）

卫星偏差通常与观测几何、气团状态有关（[V16][V17]）。设偏差 $b = \sum_j X_j \beta_j = X\beta$，$X_j$ 为**预报因子**（`BiasPredictor`：常数、扫描角、扫描角平方、层厚、下垫面类型、地面气压、云量），系数 $\beta$ 与状态一起在同化中求解：

$$
J(\beta) = \frac{1}{2}\sum_i \frac{\big(d_i - X_i\beta\big)^2}{\sigma_i^2}
+ \frac{1}{2}\big(\beta - \beta^b\big)^{\mathrm T} B_\beta^{-1} \big(\beta - \beta^b\big).
\tag{6.18}
$$

令 $A = \sum_i X_i^{\mathrm T}X_i/\sigma_i^2$、$b = \sum_i X_i^{\mathrm T}d_i/\sigma_i^2$，则 $\nabla_\beta J = -b + A\beta + B_\beta^{-1}(\beta-\beta^b)$，一次 Gauss-Newton（正规方程）更新为

$$
\Delta\beta = \big(A + B_\beta^{-1}\big)^{-1}\Big( b - A\beta - B_\beta^{-1}\beta \Big),
\qquad
\beta \leftarrow \beta + \alpha\,\Delta\beta .
\tag{6.19}
$$

`VariationalBiasCorrection::update` 按 $(\text{ObsType}, \text{channel})$ 分组各自解一个 $7\times 7$ 的对称正定系统（Cholesky，`src/obs/bias_correction.cpp`），因此不同通道的系数互不污染。`predict/correct/correct_inplace` 是 $b$ 的正演与订正；`correct_inplace` 把 $b$ 累加进 `Observation::bias`，具有可逆性。

---

## 7. 极小化

### 7.1 为什么用 L-BFGS

(6.6) 的 Hessian 良态（背景部分为单位阵），但维数巨大（$10^6 \sim 10^8$）。L-BFGS（[V13]；[B8] 第 7 章）用最近 $m$ 对 $(s_k, y_k)$ 隐式表示逆 Hessian：

$$
H_0 = \gamma I,\qquad \gamma = \frac{s_{k-1}^{\mathrm T} y_{k-1}}{y_{k-1}^{\mathrm T} y_{k-1}},
$$
$$
q = g_k;\quad
\text{for } i = k-1..k-m:\ a_i = \rho_i s_i^{\mathrm T} q,\ q \leftarrow q - a_i y_i;
$$
$$
r = H_0 q;\quad
\text{for } i = k-m..k-1:\ b = \rho_i y_i^{\mathrm T} r,\ r \leftarrow r + s_i(a_i - b);
\qquad
d_k = -r .
\tag{6.20}
$$

内存 $O(mn)$，每次迭代 $O(mn)$。本实现支持对角预条件：把 $H_0 = \gamma I$ 换成 $H_0 = \gamma P$（$P$ 对角正定），在 $r = H_0 q$ 一步逐元素乘 $P_i$。

### 7.2 非线性 CG

Polak-Ribière+（带重启）：

$$
\beta_k = \max\left(0,\ \frac{g_{k+1}^{\mathrm T}(g_{k+1} - g_k)}{g_k^{\mathrm T} g_k}\right),
\qquad
d_{k+1} = -g_{k+1} + \beta_k d_k ,
\tag{6.21}
$$

当 $g_{k+1}^{\mathrm T} d_{k+1} \ge 0$（丢失下降性）或每 $n$ 步强制 $\beta = 0$ 重启。CG 内存 $O(n)$，适合超大问题的低内存模式，但收敛对条件数更敏感。

### 7.3 强 Wolfe 线搜索

[B8] 算法 3.5 + 3.6：先用扩张找到满足 Armijo 条件或曲率条件的区间，再在区间内 zoom。强 Wolfe 条件：

$$
f(x + \alpha d) \le f(x) + c_1 \alpha\, g^{\mathrm T} d
\qquad\text{(Armijo, 充分下降)},
\tag{6.22}
$$
$$
\left|\, g(x+\alpha d)^{\mathrm T} d \,\right| \le c_2\,|\,g^{\mathrm T} d\,|
\qquad\text{(曲率, 排除过短步长)} .
\tag{6.23}
$$

本实现取 $c_1 = 10^{-4}$、$c_2 = 0.9$，zoom 用二分（安全、收敛可靠）。线搜索是唯一需要额外函数/梯度评估的地方：4D-Var 中一次评估 = 一次 TL 前向 + 一次 AD 反向，因此 `MinimizerOptions::use_strong_wolfe` 是在收敛质量与每次迭代成本之间权衡。

### 7.4 Lanczos 谱估计

对对称正定 $B$（或代价函数的 Hessian）做 $m$ 步 Lanczos 三对角化 $T_m$，其 Ritz 值逼近极值特征值：

$$
T_m = \begin{pmatrix}\alpha_1&\beta_1&&\\ \beta_1&\alpha_2&\ddots&\\ &\ddots&\ddots&\beta_{m-1}\\ &&\beta_{m-1}&\alpha_m\end{pmatrix},
\qquad Q_m^{\mathrm T} A Q_m = T_m .
\tag{6.24}
$$

`vibe::da::detail::lanczos_tridiagonalize` 用完全重正交抵抗舍入误差累积，Ritz 值由 **Sturm 序列 + 二分**求出（`src/da/minimizer.cpp`）。用途：估计 $B$ 的谱确定预条件、诊断 Hessian 病态程度、构造低秩近似。`make_minimizer("lanczos")` 返回的极小化器只做谱估计（不移动 $x$），Ritz 值记录在 `history()` 的 `gradient_norm` 字段中。

### 7.5 收敛判据

$$
\frac{\|\nabla J\|}{\max(1, |J|)} < \text{gtol}
\qquad\text{或}\qquad
|J_k - J_{k-1}| < \text{ftol},
\tag{6.25}
$$

同时用内层迭代上限（`LoopStructure::inner`）兜底。`IterationRecord` 记录每次迭代的 $J$、梯度范数、步长，用于画收敛曲线与发现"代价下降但梯度不降"的异常。

---

## 8. 观测算子的稀疏性与实现规范

### 8.1 三件套

每个 `ObservationOperator` 必须提供（[O8]）：

* `apply`：$y = H(x)$；
* `applyTL`：$dy = H'(x)\,\Delta x$，**所有系数冻结在基础态**；
* `applyAD`：$\Delta x^{*} \mathrel{+}= H'^{\mathrm T}(x)\,dy$，严格按 `applyTL` 的逆序转置、**累加**（便于复合算子串联）。

### 8.2 稀疏性

每个观测只依赖少数格点：水平双线性 4 点 × 垂直线性 2 点 = 8 个格点；辐射率依赖整条廓线（$n_z$ 层）但水平仍只依赖 4 点。`applyAD` 必须显式利用该稀疏结构（`scatter_adjoint` / `scatter_level`），禁止构造稠密矩阵。

### 8.3 边界钳制与严格转置

`src/obs/obs_operator.cpp` 的 `sample_scalar/sample_staggered` 与 `scatter_adjoint` 使用**同一套索引钳制**（`off_clamped`），因此散射是插值的逐项转置，点积检验的相对误差只来自浮点（$\sim 10^{-15}$）。`sample_staggered` 先做错位→体心的算术平均（各自 $1/2$ 权重），再做水平双线性与垂直线性；`scatter_adjoint` 按相反顺序把权重分摊回错位场。

### 8.4 点积检验

$$
\big\langle H'(x)\,\Delta x,\ dy \big\rangle_R
=
\big\langle \Delta x,\ H'^{\mathrm T}(x)\,dy \big\rangle_B .
\tag{6.26}
$$

`ObservationOperator::check_adjoint` 用固定种子生成随机 $\Delta x$、$dy$，返回对称相对误差；`check_tangent` 用有限差分比较 $H'(x)\Delta x$ 与 $(H(x+\epsilon\Delta x) - H(x))/\epsilon$。

**度量选择的说明**：本实现两侧都用欧氏内积。$B$ 的格点体积加权在控制变量变换 $U$ 内部体现（`variance()` 给出的方差、`DiffusionCorrelation` 的度量）；若在点积检验中对 $\Delta x$ 侧施加体积权重，则 `applyAD` 必须同时除以单元体积才是同一度量下的伴随，这会改变算子的语义。该取舍在 `obs_operator.cpp` 的注释中有明确说明。

---

## 9. 诊断与调优

| 诊断量 | 期望 | 偏大/偏小的含义 | 代码 |
|---|---|---|---|
| $J_b$ | $O(n/2)$ | 背景项主导说明观测信息不足 | `CostStatistics::background_term` |
| $J_o$ | $O(p/2)$ | 远大于 $p/2$ 说明 $R$ 被低估或 $B$ 被高估 | `CostStatistics::observation_term` |
| $J_c$ | 远小于 $J_b + J_o$ | 惩罚项过大会拖慢收敛、平滑掉真实信号 | `CostStatistics::penalty_term` |
| $\chi^2/p$ | $\approx 1$ | 见 (6.17) | `AnalysisResult::chi_square` |
| rms(O-A) / rms(O-B) | $\approx \sqrt{1/2}$ | Gleit 比 | `AnalysisResult::observation_minus_analysis_rms` |
| $\|\nabla J\|$ | 单调下降 | 不降说明线搜索失败/伴随有错 | `final_gradient_norm()` |
| 每次迭代 $J$ | 单调不增 | 上升说明线搜索未满足 Wolfe | `history()` |

调优经验：

1. 先固定 $B$（$L_h$、$L_v$、方差），只用观测项调 $R$ 的整体尺度，使 $\chi^2 \approx 1$；
2. 再调 $B$ 的长度尺度，使 rms(O-A) 达到目标（分析既不过拟合也不欠拟合）；
3. 打开 `check_adjoint_on_start`，点积检验必须先通过（$<10^{-8}$），否则所有关于收敛的结论都不可信；
4. 内层迭代数不必太大：增量 4D-Var 的外层会补偿内层未收敛，通常外层 2–3 次、内层 30–60 次即可。

---

## 10. 计算成本分析

记状态自由度 $n$、观测数 $p$、时隙数 $N$、内层迭代 $K$、L-BFGS 内存 $m$。

| 项 | 成本 | 说明 |
|---|---|---|
| 一次非线性轨迹（每外层 1 次） | $N \cdot C_M$ | $C_M$ = 一步模式成本 |
| TL 前向（每内层迭代 1 次） | $N \cdot C_{\text{TL}}$，通常 $C_{\text{TL}} \approx 2 C_M$ | 需要同时计算基础态导数 |
| AD 反向（每内层迭代 1 次） | $N \cdot C_{\text{AD}}$，通常 $C_{\text{AD}} \approx 2 \sim 4 C_M$ | 逆序 + 转置 |
| H 及伴随 | $p \cdot C_H$ | 稀疏，$C_H$ 很小；辐射率为 $p n_z$ |
| 控制变量变换 | $O(n \log n)$ | 扩散 CG + EOF 矩阵乘 |
| L-BFGS 每迭代 | $O(mn)$ | 向量运算，可忽略 |
| 外层总成本 | $O\!\big(n_{\text{outer}}(N C_M + K(N C_{\text{TL}} + N C_{\text{AD}} + p C_H))\big)$ | |

量级示例：$n = 10^6$、$p = 10^5$、$N = 12$、$K = 50$、$n_{\text{outer}} = 2$，一次分析约需 $2\times(12 + 50\times 12\times 5) \approx 6\times10^3$ 倍单步模式成本。这就是内层要用降分辨率（factor $=2$ 使成本降约 8 倍）以及伴随代码效率至关重要的原因。

**内存**：`Trajectory::memory_bytes()` 给出 $N_{\text{steps}} \times n \times 8$ 字节（含慢/声波趋势）。$n=10^6$、$N=12$ 时约 0.2 GB（只存基础态）或 0.6 GB（含趋势）；checkpointing（见文档 07）用时间换内存。

---

## 11. 与代码的对应表

| 数学 / 算法 | 类 / 函数 | 文件 | 文献 |
|---|---|---|---|
| 统一记号、代价函数 (6.1) | `da::AnalysisResult`、`da::CostStatistics` | `include/vibe/da/da_types.hpp` | [V14] |
| 增量代价函数 (6.6) | `da::CostFunction::value/value_and_gradient` | `src/da/cost_function.cpp` | [V3] |
| 梯度 (6.7)(6.8) | `CostFunction::gradient` | `src/da/cost_function.cpp` | [V1][V2] |
| 数字滤波弱约束 | `CostFunction::digital_filter_penalty`、Dolph-Chebyshev 高通 FIR | `src/da/cost_function.cpp` | [V25] |
| $B=UU^{\mathrm T}$ 顺序变换 (6.9) | `StandardControlVariableTransform` | `src/da/control_vector.cpp` | [V7][V8][V9] |
| 扩散相关 (6.10) | `da::DiffusionCorrelation` | `src/da/control_vector.cpp` | [V6][V7] |
| 垂直 EOF (6.11) | `da::VerticalEofTransform`、`jacobi_eigen` | `src/da/control_vector.cpp` | [V9][V12] |
| 线性平衡 (6.12) | `da::BalanceOperator::derive_wind/derive_mass_adjoint` | `src/da/control_vector.cpp` | [V8] |
| NMC 估计 (6.13)(6.14) | `da::NmcEstimator::estimate` | `src/da/background_error.cpp` | [V5] |
| 集合估计与局地化 (6.15)(6.16) | `da::EnsembleEstimator::estimate/localize` | `src/da/background_error.cpp` | [V19][V20][V23] |
| Desroziers 诊断 (6.17) | `da::diagnose_ob_stats` | `src/da/background_error.cpp` | [V18] |
| B 构建 | `da::BackgroundErrorBuilder::build` | `src/da/background_error.cpp` | [V8][V9] |
| L-BFGS (6.20) | `LbfgsMinimizer` | `src/da/minimizer.cpp` | [V13][B8] |
| 非线性 CG (6.21) | `CgMinimizer` | `src/da/minimizer.cpp` | [B8] |
| 强 Wolfe 线搜索 (6.22)(6.23) | `da::strong_wolfe_line_search` | `src/da/minimizer.cpp` | [B8] |
| Lanczos 谱估计 (6.24) | `detail::lanczos_tridiagonalize` | `src/da/minimizer.cpp` | [B8] |
| 增量 4D-Var 主控 | `da::Incremental4DVar::run/run_single_outer` | `src/da/incremental_4dvar.cpp` | [V3][V10] |
| 循环结构与分辨率递进 | `da::LoopStructure::describe`、`Incremental4DVarConfig` | `src/da/incremental_4dvar.cpp` | [V3][V22] |
| 弱约束开关 | `Incremental4DVarConfig::use_weak_constraint` | `src/da/incremental_4dvar.cpp` | [V15] |
| 观测三件套与点积检验 (6.26) | `obs::ObservationOperator` | `src/obs/obs_operator.cpp` | [O8][A3] |
| 探空/地面/导风/掩星/雷达/辐射率 | `obs::SoundingOperator` 等 | `src/obs/obs_operators.cpp` | [O1][O4][O6][O8] |
| 快速辐射传输 | `obs::RadiativeTransfer` | `src/obs/radiative_transfer.cpp` | [O1][O2][O3] |
| 折射率 $N = 77.6 p/T + 3.73\cdot 10^5 q_v p/T^2$ | `GnssRoOperator::refractivity` | `src/obs/obs_operators.cpp` | [O4][O5] |
| 反射率 $Z = a(\rho q)^b$ | `RadarOperator`、`radar_terms` | `src/obs/obs_operators.cpp` | [P3][O6] |
| VarBC (6.18)(6.19) | `obs::VariationalBiasCorrection` | `src/obs/bias_correction.cpp` | [V16][V17] |
| 观测数据模型与 CSV | `obs::ObsSpace`、`obs::ObsReader` | `src/obs/observations.cpp` | [O9] |

---

## 12. 数值实验建议、取舍与常见失败模式

### 12.1 建议的最小验证序列

1. **点积检验**：`ObservationOperator::check_adjoint` 与 `da::check_adjoint`，容差 $10^{-8}$；
2. **单观测实验**：给一个位于格点的探空温度观测，$B$ 用单位方差、$R$ 用小值，检查分析增量符号与量级符合最优插值公式 $x^a = x^b + BH^{\mathrm T}(HBH^{\mathrm T}+R)^{-1}d$；
3. **L-BFGS vs CG**：同一代价函数，比较迭代数与最终梯度范数；
4. **切线检验**：`check_tangent` 的误差曲线应呈 U 形（截断 $O(\epsilon)$ + 舍入 $O(1/\epsilon)$）；
5. **$\chi^2$ 与 Gleit 比**：用合成观测检验 `diagnose_ob_stats` 是否落在期望附近；
6. **分辨率敏感性**：改变 `inner_resolution_factor`，检查分析质量的退化是否可接受。

### 12.2 本实现的取舍与偏差

* **观测项的时刻**：`CostFunction` 的冻结构造函数没有时间步长，也没有可写（非 const）的 TL 模型，因此观测项在**分析时刻**用 $H'$ 及其伴随求值（(6.7) 中 $M_i = I$ 的形式）。逐时隙的 $M_i$ 由 `Incremental4DVar` 与 driver 层负责。这是增量 3D 形式；多时隙 4D 的完整形式需要扩展 `CostFunction` 接口。
* **数字滤波的方向**：冻结接口只给单一增量（无时间维），因此 `digital_filter_penalty` 实现为**垂直方向的对称高通 FIR**（Dolph-Chebyshev 窗，零直流增益，反射边界保证自伴）。生产实现应在增量轨迹的时间方向施加同样处理（[V25]）。
* **垂直 EOF 的完整性**：为保证 $U$ 可逆（见 §5.3），垂直基是完整的 $n_z$ 个模态，只对核函数谱做方差加权。
* **`TangentLinearModel::eq_` 为空**：冻结头文件没有为 `TangentLinearModel` 声明析构函数，而 `dyn::Equations*` 是裸指针，因此本实现不持有 `dyn::Equations` 的所有权（TL 核自包含），避免资源泄漏。生产实现应在头文件中加入 `std::unique_ptr<dyn::Equations>` 与析构函数。
* **降分辨率内层**：`resolution_factor` 只被记录与打印；真正在粗网格上求解需要网格重采样算子，属于 driver 层。
* **`check_adjoint` / `check_tangent` 的 const_cast**：冻结接口把 `propagate` 声明为非 const 方法，而检验函数接收 `const TangentLinearModel&`。实现中通过 `const_cast` 调用并刷新轨迹缓存，副作用已在 `src/da/tangent_linear.cpp` 注释中说明。
* **领域分解**：本模块的向量与场都按本地子域操作（`Grid::nx()` 返回本地点数），跨进程 halo 交换由 grid/mpi 层在算子之外完成。

### 12.3 常见失败模式

| 现象 | 可能原因 | 检查 |
|---|---|---|
| 点积检验失败（$10^{-2}$ 量级） | 插值矩阵未转置、系数未冻结在基础态、累加顺序错误 | 逐算子 `check_adjoint`；对照 `sample`/`scatter` 的权重表 |
| $J$ 不下降 | 线搜索失败、梯度符号错、$R$ 尺度错 | 打印 `history()`；先在小问题上用二次型验证极小化器 |
| 分析增量爆炸 | $B$ 方差过大、$R$ 过小、$\chi^2 \gg 1$ | 检查 `variance()` 与观测 `sigma` |
| 分析出现格点噪声 | 水平/垂直相关长度太小、内层未收敛 | 增大 $L_h$/$L_v$；增加内层迭代 |
| 平流层出现虚假振荡 | 数字滤波惩罚太弱、垂直模态截断 | 调大 `digital_filter_weight`；检查垂直相关 |
| 卫星通道偏差被吸收进状态 | VarBC 未启用或系数未收敛 | 检查 `use_variational_bias` 与 `describe()` |

---

## 13. 小结

* 增量 4D-Var 的三要素：**外层重线性化**（保证 $d_i$ 与 $M_i$ 一致）、**内层预条件极小化**（$J(v)=\tfrac12 v^{\mathrm T}v + \ldots$）、**分辨率递进**（成本控制）；
* $B$ 的建模决定分析质量的上限：扩散相关给可扩展的水平相关、EOF 给垂直相关、平衡关系给多变量耦合、NMC/集合给统计量；
* $R$ 与偏差订正决定观测信息的正确使用：对角 $R$ + Desroziers 诊断 + VarBC；
* 一切结论的前提是**切线/伴随的正确性**——点积检验必须先通过，见 [07_tangent_linear_adjoint.md](07_tangent_linear_adjoint.md)。

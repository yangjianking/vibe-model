# 09 检验与评估模块（vibe::verify）

> 本文描述 VIBE-Model 的误差检验与预报评估模块：接口契约、四类评分的完整公式与出处、
> 列联表手算示例、邻域法 FSS 算法、概率评分的严格正常性、实现与复杂度，以及与
> Python 侧 `vibe_post.verify` 的公式对照。
>
> 接口基线见 [00_architecture.md](00_architecture.md) 第 9 节；文献编号见
> [references.md](references.md)。本文只**新增**成员，不改写已冻结的签名。

---

## 1. 范围与接口基线

### 1.1 模块边界

`vibe::verify` 依赖 `vibe::common`，**不依赖** `vibe::io`、`vibe::dyn`、
`vibe::config` 的内部实现：

* `Verifier` 只接受内存中的 `FieldSample`（扁平 `std::vector<Real>` + 网格尺寸）；
* 文件读取（`.vibebin`、NetCDF）交给 `verify_cli.cpp`（本模块内实现 `.vibebin`）
  或 `vibe::io`；
* `VerifyConfig::from_kv` 接受"键 -> 值"字符串映射，由 `vibe::config` 转发已解析结果，
  从而避免 `config` 的内部解析器泄漏到检验模块（也避免模块环依赖）。

这条边界对应架构第 3 节的硬约束："外部文件格式只在 io 层出现"。`verify_cli.cpp`
不包含任何 netCDF 头文件。

### 1.2 冻结的接口（架构第 9 节）

```cpp
namespace vibe::verify {

struct ContingencyTable { Real hits, misses, false_alarms, correct_negatives; };
struct Scores {
  Real bias, mae, rmse, correlation, anomaly_correlation;
  Real pod, far, csi, ets, frequency_bias, odds_ratio_skill;
  Real brier, brier_skill, crps, roc_auc;
  Real fss, spread, spread_skill_ratio;
  std::size_t n;
};
Scores compute_continuous(const std::vector<Real>& f, const std::vector<Real>& o,
                          const std::vector<Real>* clim = nullptr);
Scores compute_categorical(const std::vector<Real>& f, const std::vector<Real>& o,
                           Real threshold);
Scores compute_probabilistic(const std::vector<Real>& p,
                             const std::vector<Real>& o);
Scores compute_fractional_skill(const std::vector<Real>& f,
                                const std::vector<Real>& o,
                                int nx, int ny, Real threshold,
                                int neighborhood_radius);

}  // namespace vibe::verify
```

实现中上述 19 个 `Scores` 字段名称、类型与顺序**逐字保持**；附加诊断量
（`mean_error`、`std_error`、`tss`、`sedi`、`accuracy`、
Brier 三项分解、秩直方图统计、`ensemble_mean_rmse` 等）追加在 `n` 之后，
属于契约允许的"只能新增"扩展，旧代码按字段名访问不受影响。

四个 `compute_*` 自由函数**全部在 `scores.hpp` 中声明**（契约要求），实现分散在
`scores.cpp / contingency.cpp / probabilistic.cpp / spatial.cpp`，以避免 `scores.hpp`
反向包含其它头文件。返回值中**不适用字段一律为 0**（`zero_scores` 起点），
适用但当前样本下无法定义的评分为 `NaN`（见 2.3 节）。

---

## 2. 检验方法论

### 2.1 三维质量框架

Murphy (1993) [E1] 与 Murphy & Winkler (1987) [E2] 指出：预报质量包含三个正交维度，
只报告单一评分会掩盖真实信息。

| 维度 | 含义 | 本模块对应评分 |
|---|---|---|
| **一致性 consistency** | 预报与观测在统计上的对应关系 | `correlation`、`anomaly_correlation`、ROC/AUC |
| **质量 quality** | 预报相对于观测的准确程度 | `mae`、`rmse`、`bias`、`csi`、`ets`、`brier`、`crps` |
| **价值 value** | 预报相对参考（气候态/持续性）带来的收益 | `brier_skill`、`ets`、`tss`、`fss` |

因此模块同时输出**绝对评分**与**技巧评分**，且每个评分都必须能追溯到 [references.md]
中的文献编号。

### 2.2 匹配框架与四类检验

按预报/观测的匹配类型（`MatchType`）分成四类：

1. **连续型 Continuous**：温度、风、位势高度等实数值，[B9] 第 8 章；
2. **分类型 Categorical**：降水阈值、雾、雷暴等事件，列联表方法 [B10] 第 3 章；
3. **概率型 Probabilistic**：集合/概率预报，Brier、CRPS、Talagrand、ROC，
   [E3][E9][E11][E12][E15]；
4. **空间型 Spatial**：高分辨率网格的邻域/尺度检验，FSS 与模糊检验 [E7][E8][E14]。

Casati et al. (2008) [E13] 强调：不同尺度的可预报性不同，单一评分无法覆盖；
因此本模块要求空间检验必须**多尺度**报告（`fss_vs_scale`）。

### 2.3 缺失值、不适用字段与"未定义"的三分处理

* **缺失值**：任何参与统计的量若为 IEEE NaN，该**样本对整体跳过**；有效对数记入 `Scores::n`。
* **不适用字段置 0**：每个 `compute_*` 函数只计算本检验类型适用的评分，
  其余字段一律显式置 0（`zero_scores` 起点）。driver 层会直接读取字段，
  因此返回值中不允许出现未初始化内容。
* **适用但未定义 -> NaN**：分母为 0 的评分返回 `kNaN`，绝不返回伪 0。例如"无事件"时
  `pod`、`far`、`ets`、`orss` 为 NaN 而 `accuracy` = 1；缺少气候态时
  `anomaly_correlation` 为 NaN；两场全无事件时 FSS 为 NaN。
* 后两条的组合是 [E13] 对"可比较性"的要求：不适用 ≠ 未定义 ≠ 0 分，
  把"未定义"当成 0 会系统性扭曲多年平均评分。

---

## 3. 确定性（连续型）评分

设有效样本对为 `(f_i, o_i)`，`n` 为有效对数，`e_i = f_i - o_i`。

\[
\begin{aligned}
\text{ME (bias)}   &= \frac{1}{n}\sum_{i=1}^{n} e_i, &
\text{MAE}         &= \frac{1}{n}\sum_{i=1}^{n} |e_i|,\\
\text{RMSE}        &= \sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^2}, &
\sigma_e           &= \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(e_i-\bar e)^2}.
\end{aligned}
\]

Pearson 相关系数（[B9] 式 (3.21)）：

\[
r=\frac{S_{fo}}{\sqrt{S_{ff}S_{oo}}},\qquad
S_{xy}=\sum_i (x_i-\bar x)(y_i-\bar y).
\]

距平相关（anomaly correlation, [E1]；[B9] 第 8 章）：

\[
\text{ACC}=\frac{\sum_i a_{f,i}a_{o,i}}
{\sqrt{\left(\sum_i a_{f,i}^2\right)\left(\sum_i a_{o,i}^2\right)}},\qquad
a_{x,i}=x_i-\text{clim}_i .
\]

实现要点：

* 均值/方差/协方差用 **Welford 在线更新**，避免"大均值小方差"时的灾难性抵消
  （[B11] Higham 2002 第 1 章）。对升序样本的等价合并使用 **Chan 平行公式**：
  \[
  \begin{aligned}
  n &= n_a+n_b, & \delta_x &= \bar x_b-\bar x_a,\\
  \bar x &= \bar x_a + \delta_x \frac{n_b}{n}, &
  M_{2,x} &= M_{2,x}^{(a)}+M_{2,x}^{(b)}+\delta_x^2\frac{n_a n_b}{n},\\
  C_{xy} &= C_{xy}^{(a)}+C_{xy}^{(b)}+\delta_x\delta_y\frac{n_a n_b}{n}.
  \end{aligned}
  \]
* **退化约定**：若任一序列方差为 0（常数序列），`r` 与 `anomaly_correlation` 返回
  **0.0** 而非 NaN——这是明确的工程约定（测试 `continuous_constant_series_correlation_zero`），
  因为"常数预报与任何观测都没有线性相关"在业务上是有意义的结论。
* 气候态 `clim` 允许长度为 1（常数）或与样本等长（逐时气候态）。

复杂度：批量 `O(n)` 时间、`O(1)` 额外空间；增量累加器每样本 `O(1)`。

---

## 4. 分类（事件）检验

### 4.1 列联表与评分公式

```text
              观测=1          观测=0
预报=1     H (hits)        FA (false alarms)
预报=0     M (misses)      CN (correct negatives)
```

令 `N=H+M+FA+CN`。

\[
\begin{aligned}
\text{frequency bias} &= \frac{H+FA}{H+M}, &
\text{POD} &= \frac{H}{H+M},\\
\text{POFD} &= \frac{FA}{FA+CN}, &
\text{FAR} &= \frac{FA}{H+FA},\\
\text{CSI} &= \frac{H}{H+M+FA}, &
\text{accuracy} &= \frac{H+CN}{N},\\
\text{TSS} &= \text{POD}-\text{POFD}, &
\text{OR} &= \frac{H\cdot CN}{M\cdot FA},\quad
\text{ORSS}=\frac{H\cdot CN-M\cdot FA}{H\cdot CN+M\cdot FA}.
\end{aligned}
\]

出处：[E5] Hanssen & Kuipers (1965)（TSS）；[E6] Schaefer (1990)（CSI/TS）；
[B9] 式 (8.14)（频率偏差）；[B10] 第 3 章（ORSS/POFD）。

**公平技巧评分 ETS**（Gandin & Murphy 1992 [E4]）先扣除随机命中：

\[
H_{\text{rand}}=\frac{(H+M)(H+FA)}{N},\qquad
\text{ETS}=\frac{H-H_{\text{rand}}}{H+M+FA-H_{\text{rand}}}.
\]

当分母 `H+M+FA-H_rand <= 0`（例如全域皆为事件）时返回 NaN。

**对称极值依赖指数 SEDI**（[E13] 第 3 节；[B10] 第 3 章）针对极值事件
`POD -> 0`、`POFD -> 0` 导致传统评分失去分辨力的问题：

\[
\text{SEDI}=\frac{\ln F-\ln H+\ln(1-H)-\ln(1-F)}
{\ln F+\ln H+\ln(1-H)+\ln(1-F)},\qquad H=\text{POD},\; F=\text{POFD}.
\]

实现中先把 `H`、`F` 夹到 `[10^{-6}, 1-10^{-6}]` 再取对数。
性质：完美预报 `+1`，随机预报（`H=F`）为 `0`，范围 `[-1,1]`。

### 4.2 2x2 手算示例（可复算）

取 `H=40`、`M=10`、`FA=20`、`CN=130`，则 `N=200`。

| 评分 | 计算 | 数值 |
|---|---|---|
| frequency bias | `(40+20)/(40+10) = 60/50` | 1.200000 |
| POD | `40/50` | 0.800000 |
| POFD | `20/(20+130)=20/150` | 0.133333 |
| FAR | `20/60` | 0.333333 |
| CSI | `40/(40+10+20)=40/70` | 0.571429 |
| accuracy | `(40+130)/200` | 0.850000 |
| TSS | `0.8-0.133333` | 0.666667 |
| `H_rand` | `(50)(60)/200` | 15.0 |
| ETS | `(40-15)/(110-15)=25/95` | 0.263158 |
| ORSS | `(40\cdot130-10\cdot20)/(40\cdot130+10\cdot20)=5000/5400` | 0.925926 |

这些数值就是单元测试 `contingency_2x2_hand_computed` 的期望值。

### 4.3 复杂度

构造列联表 `O(n)` 时间、`O(1)` 空间；各评分 `O(1)`。
`ContingencyTable` 元素取 `Real` 而非整数，以支持面积加权检验
（如变分辨率网格的雅可比加权）与并列概率分箱的分数计数。

---

## 5. 概率评分与严格正常性

### 5.1 严格正常评分（strictly proper scoring rule）

Gneiting & Raftery (2007) [E11] 定义：设观测 `y` 服从 `Q`，评分 `S(P,y)`，
若
\[
\mathbb{E}_{Y\sim Q}\big[S(Q,Y)\big]\le \mathbb{E}_{Y\sim Q}\big[S(P,Y)\big]
\quad\text{对一切预报分布 }P,
\]
且等号仅在 `P=Q` 时成立，则 `S` 为**严格正常评分**。否则模式可以通过
"偏报"（hedging，例如永远报气候频率或把概率推向 0/1）取得虚假的技巧提升，
使得检验结论无法反映真实预报能力。

本模块采用的评分全部满足严格正常性：

| 评分 | 类型 | 正常性 |
|---|---|---|
| Brier 评分 `(p-o)^2` | 二值概率 | 严格正常（Brier 1950 [E3]） |
| 对数评分 `-\ln p_o` | 二值概率 | 严格正常（[E11]） |
| CRPS | 连续概率（含集合） | 严格正常（[E11][E12]） |
| 集合离散度 | 标定诊断 | 非评分，仅诊断 |

**注意**：`bias`、`csi`、`ets`、`roc_auc`、`fss` 等
**不是**正常评分；它们不满足"诚实是最优策略"，必须与严格正常评分配合使用，
这正体现 [E1] 的多维质量框架。

### 5.2 Brier 评分与可靠度-分辨度-不确定性分解

事件指示 `o\in\{0,1\}`，预报概率 `p`：

\[
\text{BS}=\frac{1}{n}\sum_{i=1}^{n}(p_i-o_i)^2 .
\]

按预报概率分箱（`K` 个箱，箱 `k` 内样本数 `n_k`，箱内预报概率均值 `p_k`，
观测频率 `o_k`，气候频率 `\bar o`）：

\[
\begin{aligned}
\text{REL} &= \sum_k \frac{n_k}{n}(p_k-o_k)^2, &
\text{RES} &= \sum_k \frac{n_k}{n}(o_k-\bar o)^2, &
\text{UNC} &= \bar o(1-\bar o),\\
\text{BS} &= \text{REL}-\text{RES}+\text{UNC} \quad\text{（离散预报精确成立）}, &
\text{BSS} &= 1-\frac{\text{BS}}{\text{UNC}}.
\end{aligned}
\]

出处：[E3] Brier (1950)、[E17] Brier & Allen (1951)、[B9] 第 9 章。

> **恰当代数注意**：`BS=REL-RES+UNC` 对"预报概率在箱内为常数"的离散预报严格成立。
> 对连续预报概率，箱内预报方差会引入附加项
> \[
> \text{BS} = \text{REL}-\text{RES}+\text{UNC}
> + \underbrace{\frac1n\sum_i \delta_i^2}_{\text{箱内预报方差}}
> - \underbrace{\frac2n\sum_i \delta_i\varepsilon_i}_{\text{箱内协方差}},
> \quad \delta_i=p_i-p_{k(i)},\ \varepsilon_i=o_i-o_{k(i)} .
> \]
> 本模块的 `brier_decomposition` 返回**精确** `BS`（按 `\sum(p-o)^2` 累加）
> 与分箱 `REL/RES/UNC`；单元测试
> `brier_decomposition_identity_for_discrete_probabilities` 只在箱内为常数时断言恒等式。

分箱策略（`BinScheme`）：

* **等宽 EqualWidth**：箱边界 `k/K`，实现简单，但预报概率分布偏斜时箱内样本数悬殊；
* **等频 EqualFrequency**：按分位数切分，每箱样本数尽量相等（可靠性图更稳健，[B9] 第 9 章）。

### 5.3 集合 CRPS

Hersbach (2000) [E12] 给出集合预报的连续排序概率评分：

\[
\text{CRPS}=\int_{-\infty}^{\infty}\big(F(x)-H(x-y)\big)^2\,dx,
\]

其中 `F` 为集合经验分布函数，`H` 为 Heaviside 阶跃函数，`y` 为观测。
对 `m` 个成员的样本估计式：

\[
\text{CRPS}=\frac{1}{m}\sum_{i=1}^{m}|x_i-y|
-\frac{1}{2m^2}\sum_{i=1}^{m}\sum_{j=1}^{m}|x_i-x_j| .
\]

对升序成员 `x_{(1)}\le\cdots\le x_{(m)}` 有恒等式

\[
\sum_{i,j}|x_i-x_j| = 2\sum_{i=1}^{m}(2i-m-1)\,x_{(i)},
\]

因此单点 CRPS 只需 `O(m\log m)`（排序）。无偏（fair）版本把第二项系数
改为 `1/(m(m-1))`，用于校正小集合的负偏差（[E16] Ferro et al. 2008）。

### 5.4 Talagrand 秩直方图

对每个样本点，把观测插入 `m` 个成员排序序列，得到 `m+1` 个"秩箱"
（[E9] Hamill 2001；[E10] Talagrand & Vautard 1997）。理想标定集合的秩分布为均匀分布
`p_k=1/(m+1)`。

* 平坦度（本模块定义，`\in[0,1]`，1 为完全平坦）：
  \[
  \text{flatness}=1-\frac{\tfrac12\sum_k|p_k-1/K|}{1-1/K},\qquad K=m+1 ;
  \]
* 秩偏差（`\in[-1/2,1/2]`）：
  \[
  \text{bias}=\sum_{k=0}^{K-1}p_k\frac{k+1/2}{K}-\frac12 ;
  \]
* 卡方统计量（检验平坦性，自由度 `K-1`，[B9] 第 5 章）：
  \[
  \chi^2=nK\sum_k\left(p_k-\frac1K\right)^2,\qquad
  p\text{-value}=Q\!\left(\frac{K-1}{2},\frac{\chi^2}{2}\right),
  \]
  `Q` 为正则化上不完全 Gamma 函数（级数 + 连分式实现）。

**并列处理**：当若干成员恰等于观测（`t` 个），把该观测以 `1/(t+1)` 的概率
摊到 `t+1` 个相邻秩上，保持计数期望无偏（[E9] 第 3 节）。因此
`RankHistogram::counts` 为 `Real` 而非整数。

### 5.5 ROC 曲线与 AUC

Mason & Graham (2002) [E15]：ROC 曲线以命中率 `\text{POD}(c)` 对空报率
`\text{POFD}(c)` 作图，`c` 为判别阈值。曲线下面积

\[
\text{AUC}=P(p_{\text{pos}}>p_{\text{neg}})+\tfrac12 P(p_{\text{pos}}=p_{\text{neg}}).
\]

**并列处理**：把相同预报概率归为一组，按概率从高到低累积正样本数 `\text{cum}_+`：

\[
\text{AUC}=\frac{1}{n_+n_-}\sum_{\text{组}\,g}
\Big[\,\text{neg}_g\cdot\text{cum}_+^{\text{higher}}
+\tfrac12\,\text{pos}_g\cdot\text{neg}_g\Big].
\]

其中第一项计"高于本组的正样本"（严格大于），第二项把组内并列按 0.5 计。
这与 Mann-Whitney U 统计量的并列订正完全一致，做到**精确**而非近似。

增量累加器 `ScoreAccumulator` 为节省内存只保存概率直方图（`K` 个箱），
AUC 按组内并列估计；批量接口 `compute_probabilistic` 给出精确 AUC。

### 5.6 集合离散度

\[
\sigma_{\text{ens}}=\sqrt{\frac{1}{m-1}\sum_{i=1}^{m}(x_i-\bar x)^2},\qquad
\text{spread}=\overline{\sigma_{\text{ens}}},\qquad
\text{RMSE}=\sqrt{\overline{(\bar x-y)^2}},\qquad
\text{SSR}=\frac{\text{spread}}{\text{RMSE}} .
\]

理想标定集合满足 `\text{SSR}\approx 1`：离散度过小表示集合过于自信（underdispersive），
过大表示欠自信。`spread` 与 `spread_skill_ratio` 是架构 `Scores` 的冻结字段。

---

## 6. 空间检验

### 6.1 双惩罚问题

高分辨率模式对对流单体的**位移误差**在逐点检验中同时产生 misses 与 false alarms，
使 ETS/CSI 在"形态与量级都合理、只是偏了一个格点"时低于粗网格模式——这就是
double penalty（[E7][E8][E13]）。邻域法先在窗口内把二值场平滑成事件**分数场**，
再比较分数场，从而对位置误差宽容、对量级/覆盖率误差敏感。

### 6.2 邻域分数与 FSS

二值化与邻域分数：

\[
I(\mathbf x)=\mathbf 1[\,\phi(\mathbf x)\ge \tau\,],\qquad
F(\mathbf x)=\frac{1}{|N_r(\mathbf x)|}\sum_{\mathbf y\in N_r(\mathbf x)}I(\mathbf y),
\]

\[
N_r(\mathbf x)=\{\,\mathbf y:\,|i_y-i_x|\le r,\ |j_y-j_x|\le r\,\}\cap\text{网格}.
\]

边界处窗口自动收缩（分母为窗口内实际格点数），避免把域外当作"无事件"。
FSS（[E7] 式 (4)；[E14]）：

\[
\text{FSS}=1-\frac{\sum_{\mathbf x}\big(F_f(\mathbf x)-F_o(\mathbf x)\big)^2}
{\sum_{\mathbf x}F_f(\mathbf x)^2+\sum_{\mathbf x}F_o(\mathbf x)^2}.
\]

**极限性质**（单元测试覆盖）：

* `r\to 0`：`F` 退化为 `I`，
  \[
  \text{FSS}(0)=1-\frac{M+FA}{2H+M+FA}=\frac{2H}{2H+M+FA}.
  \]
  这是逐点点数比 `2H/(2H+M+FA)`；**它不等于** `\text{CSI}=H/(H+M+FA)`
  （[E7] 第 3 节特别提醒）。两者只在 `H=0` 或某些退化情形相等，
  文档与测试均显式区分。
* `r\to\infty`：两场分数都变成域平均覆盖率 `a`、`b`，
  \[
  \text{FSS}\to 1-\frac{(a-b)^2}{a^2+b^2},
  \]
  当覆盖率相等（`a=b`）时趋于 1。因此"大半径极限为 1"只在事件覆盖率一致的
  前提下成立（例如同一事件的平移）。
* 两场全无事件时分母为 0，`\text{FSS}=\text{NaN}`（未定义，而非 0 或 1）。

### 6.3 FSS 算法伪代码（积分图实现）

朴素邻域求和代价为 `O(n_x n_y r^2)`；本模块用 **2D 积分图
（summed-area table）**把每个半径的代价降到 `O(n_x n_y)`，与 `r` 无关。

```text
输入: 场 f, o (nx*ny), 阈值 tau, 半径 r
输出: FSS

1. 对 f 和 o 分别构造二值场 I(x) = 1 if field(x) >= tau else 0
2. 构造积分图 P (尺寸 (nx+1)*(ny+1)):
      P[i+1][j+1] = I[i][j] + P[i][j+1] + P[i+1][j] - P[i][j]
3. 对每个格点 (i,j):
      i0 = max(0, i-r); i1 = min(nx-1, i+r)
      j0 = max(0, j-r); j1 = min(ny-1, j+r)
      S  = P[i1+1][j1+1] - P[i0][j1+1] - P[i1+1][j0] + P[i0][j0]   // O(1)
      F[i][j] = S / ((i1-i0+1)*(j1-j0+1))
4. num = sum_x (Ff - Fo)^2
   den = sum_x Ff^2 + sum_x Fo^2
5. return (den > 0) ? 1 - num/den : NaN
```

多尺度：对 `r=0,1,\dots,r_{\max}` 重复步骤 3-5，得到 FSS 曲线；
最小可分辨尺度 = 首个满足 `\text{FSS}(r)\ge\text{fss\\_target}`（默认 0.5）的
半径换算出的物理尺度 `2r\,\Delta x`。

### 6.4 模糊检验

Ebert (2008) [E8] 的"邻域法"：只要半径内有观测事件，就认为预报在该点"模糊命中"。
本模块构造模糊列联表：

\[
\begin{aligned}
\text{fuzzy hit} &: F_f>0 \;\wedge\; F_o>0, &
\text{fuzzy miss} &: F_f=0 \;\wedge\; F_o>0,\\
\text{fuzzy FA} &: F_f>0 \;\wedge\; F_o=0, &
\text{fuzzy CN} &: F_f=0 \;\wedge\; F_o=0,
\end{aligned}
\]

并在其上计算 POD/FAR/CSI/ETS 与频率偏差（`FuzzyScores`）。
多半径的模糊评分序列与 FSS 曲线一起给出预报的"有效分辨率"。

### 6.5 双惩罚与误差分解

在全场（或 `\pm s` 邻域）搜索使 MSE 最小的整数位移 `(dx^*,dy^*)`：

\[
(dx^*,dy^*)=\arg\min_{|dx|,|dy|\le s}\frac{1}{|N_{\text{ov}}|}
\sum_{\mathbf x\in N_{\text{ov}}}\big(f(x+dx,y+dy)-o(x,y)\big)^2 .
\]

分解：

\[
\underbrace{\text{MSE}_{\text{total}}}_{\text{零位移 MSE}},
\qquad
A=\overline{f}-\overline{o}\ \ (\text{幅度误差}),
\qquad
D=\sqrt{\max(\text{MSE}^*-A^2,\,0)}\ \ (\text{位移/形态误差}),
\]

\[
\text{DPI}=\frac{\text{MSE}_{\text{total}}}{\max(\text{MSE}^*,\epsilon)}\ \ge 1 .
\]

`DPI` 接近 1 表示误差主要来自位置（"双惩罚"显著，逐点评分低估了真实技巧）；
`|A|` 大而 `D` 小表示主要是量级/覆盖率偏差。该分解为工程近似，
用于诊断而非严格能量分解，出处 [E8][E13]。

### 6.6 尺度分解（简化 2D DCT）

正交 DCT-II 把二维场分解到余弦基：

\[
X[k,l]=\sum_{i=0}^{n_x-1}\sum_{j=0}^{n_y-1}x[i,j]\,c_k c_l
\cos\!\frac{\pi k(2i+1)}{2n_x}\cos\!\frac{\pi l(2j+1)}{2n_y},
\quad
c_0=\sqrt{\tfrac1{n}},\ c_{k>0}=\sqrt{\tfrac2{n}} .
\]

正交性给出 Parseval 关系 `\sum x^2=\sum X^2`（单元测试
`dct_roundtrip_and_parseval` 验证往返精度与能量守恒）。能量按归一化径向波数

\[
\kappa=\frac{1}{\sqrt2}\sqrt{\left(\frac{k}{n_x}\right)^2+\left(\frac{l}{n_y}\right)^2}\in[0,1]
\]

分为 `n_{\text{bands}}` 个等宽带，得到误差能量谱（`ScaleSpectrum`），
用于对照 FSS 的"有效分辨率"。可分离实现复杂度 `O(n_x n_y(n_x+n_y))`。

---

## 7. 检验框架、结果与聚合

### 7.1 Verifier 工作流

```text
FieldSample  ──>  Verifier::run  ──>  VerificationResult (长表)  ──>  CSV / JSON
 (内存数组)        (按 MatchType 分派)      (variable,time,type,threshold,radius,metric,value,n)
```

* `FieldSample`：`nx,ny,nz` + 扁平 `forecast/observation` + 可选 `climatology`
  + `scale`（单位换算）+ `time`（Unix epoch 秒）；
* 空间评分为 2D，取最低层 `k=0`（多层检验请逐层构造 `FieldSample`）；
* `VerificationResult` 采用**长表**，一行一个指标，便于 pandas 透视，
  且新增字段不需要改变列结构；
* CSV 表头固定为 `variable,time,type,threshold,radius,metric,value,n`；
  JSON 中 NaN 写为 `null`（JSON 无 NaN 字面量），数值用 `%.12g` 输出。

### 7.2 时间聚合

`Aggregator` 把逐时次评分按 **All / Daily / Monthly** 桶聚合成
mean / min / max / stddev / median / count。时间桶由纯算法从 Unix epoch 秒换算
（Howard Hinnant 的 `civil_from_days` 算法，UTC），不依赖 `<ctime>`
的本地时区与夏令时，保证跨平台、跨节点一致。

### 7.3 .vibebin 读取（verify_cli.cpp）

```text
offset  type        field
0       char[8]     magic = "VIBEBIN1"
8       int32       nx
12      int32       ny
16      int32       nz
20      int32       real_kind   (0 = float64, 1 = float32)
24      float64     time
32      int32       nvars
36      ...         每个变量: char[32] name + nx*ny*nz 个对应类型的数据
```

该文件只包含标准库头文件，**不包含 netCDF**，与 io 层的 `.vibebin` 约定一致。

---

## 8. 实现与复杂度

| 例程 | 时间 | 空间 | 备注 |
|---|---|---|---|
| `compute_continuous` | `O(n)` | `O(1)` | Welford/Chan，数值稳定 |
| `ScoreAccumulator::add` | `O(1)` | `O(K)` | `K` = 概率箱数（默认 10） |
| `ScoreAccumulator::merge` | `O(K)` | `O(K)` | MPI/OpenMP 归约 |
| `compute_categorical` | `O(n)` | `O(1)` | 单遍列联表 |
| `brier_decomposition` | `O(n)`（等宽）/ `O(n\log n)`（等频） | `O(K)` | Brier 值精确 |
| `roc_auc` | `O(n\log n)` | `O(n)` | 排序 + 并列分组 |
| `crps_ensemble` | `O(m\log m)` | `O(m)` | 升序恒等式 |
| `rank_histogram` | `O(nm\log m)` | `O(nm)` | 输入即成员矩阵 |
| `fractions_skill_score` | `O(n_x n_y)` | `O(n_x n_y)` | 积分图，与 `r` 无关 |
| `fss_vs_scale` | `O(n_x n_y R)` | `O(n_x n_y)` | `R` = 最大半径 |
| `double_penalty_diagnosis` | `O(n_x n_y s^2)` | `O(n_x n_y)` | 位移搜索 |
| `dct2 / idct2` | `O(n_x n_y(n_x+n_y))` | `O(n_x n_y)` | 可分离 |
| `Aggregator::aggregate` | `O(N_{\text{rec}}\log N_{\text{rec}})` | `O(N_{\text{rec}})` | 中位数需要排序 |

**内存策略**：`ScoreAccumulator` 面向"样本量极大、内存受限"的场景
（例如逐点检验全场的逐时次累加），只保留常数个累加器与 `O(K)` 概率直方图，
不存储样本；`merge` 支持并行归约。批量接口因需要精确统计量而 `O(n)` 存储输入。

---

## 9. 与 Python 侧 vibe_post.verify 的公式对照

Python 后处理包 `vibe_post.verify` 只做包装与可视化，**公式必须与本模块逐字一致**；
下表是两侧的契约对照（Python 侧接口为规划约定，实现时以此表为准）。

| 评分 | C++（本模块） | Python `vibe_post.verify` | 公式/出处 |
|---|---|---|---|
| 平均误差 | `Scores::bias / mean_error` | `bias(f,o)` | `\bar f-\bar o`，[B9] |
| MAE | `Scores::mae` | `mae(f,o)` | `\frac1n\sum|e|` |
| RMSE | `Scores::rmse` | `rmse(f,o)` | `\sqrt{\frac1n\sum e^2}` |
| 相关 | `Scores::correlation` | `correlation(f,o)` | Pearson [B9] |
| 距平相关 | `Scores::anomaly_correlation` | `anomaly_correlation(f,o,clim)` | [E1] |
| POD/FAR | `compute_categorical` | `pod/cat_scores` | [E19] |
| CSI | `Scores::csi` | `csi` | [E6] |
| ETS | `Scores::ets` | `ets` | [E4] |
| TSS | `Scores::tss` | `tss` | [E5] |
| ORSS | `Scores::odds_ratio_skill` | `orss` | [B10] |
| SEDI | `Scores::sedi / ContingencyTable::sedi` | `sedi` | [E13][B10] |
| Brier | `Scores::brier` | `brier_score` | [E3] |
| Brier 分解 | `brier_decomposition` | `brier_decomposition` | [E3][E17] |
| BSS | `Scores::brier_skill` | `brier_skill_score` | `1-BS/UNC` |
| CRPS | `crps_ensemble / crps_fair` | `crps` | [E12][E16] |
| 秩直方图 | `rank_histogram` | `rank_histogram` | [E9][E10] |
| ROC/AUC | `roc_curve / roc_auc` | `roc_auc` | [E15] |
| FSS | `fractions_skill_score` | `fss` | [E7][E14] |
| 模糊评分 | `fuzzy_scores` | `fuzzy_scores` | [E8] |
| 离散度/SSR | `ensemble_scores` | `spread_skill` | [B9] |

两侧共同约定：

1. 事件阈值语义一致：`field >= threshold` 记为事件；
2. NaN 跳过样本对，`n` 只计有效对；
3. 适用但数学上未定义的评分为 NaN，**不落 0**；不适用字段为 0；
4. 概率事件指示统一为 `o >= 0.5`；
5. 分箱策略统一为 `equal_width / equal_frequency`。

---

## 10. 测试覆盖

`tests/unit/test_verify.cpp` 使用内置框架 `vibe::test`，共 40+ 个用例，覆盖：

* 完美预报（连续、分类、概率、FSS 全 1）；
* 已知解析值：`H=40,M=10,FA=20,CN=130` 的完整 2x2 手算、
  CRPS 两成员解析解 `a/2`、`\gamma` 函数已知值；
* 退化输入：常数序列相关系数为 0、全空事件的 NaN、两场全空的 FSS=NaN；
* 缺失值：NaN 样本对跳过且 `n` 正确；
* 增量一致性：`ScoreAccumulator` 与批量计算逐项一致，`merge` 结合律；
* FSS 极限性质：`r=0` 退化为 `2H/(2H+M+FA)` 且与 CSI 显式区分，
  `r\to\infty`（覆盖率一致）趋于 1；
* 位移诊断：平移事件的最优位移被正确恢复；
* DCT 往返精度与 Parseval 能量守恒；
* 配置解析（`from_string` / 未知键抛异常）、结果 CSV/JSON 序列化、时间桶聚合。

---

## 11. 与架构基线的偏差说明

1. **`VerifyConfig` 的命名空间**：架构第 10 节列出 `vibe::config::VerifyConfig`，
   本模块定义 `vibe::verify::VerifyConfig`。原因是架构第 3 节要求检验模块不与
   `config` 内部解析器耦合；`from_kv` 接受已解析的键值对，由 `vibe::config`
   在组装时填入即可。这是"新增"而非"改写"：`vibe::config` 仍可保留同名类型并做转换。
2. **`Scores` 扩展字段**：在冻结的 19 个字段之后追加了 `mean_error`、`std_error`、
   `tss`、`sedi`、`accuracy`、Brier 三项分解、秩直方图统计与
   `ensemble_mean_rmse/ensemble_spread`。字段名与顺序对既有字段无影响。
3. **`ContingencyTable` 位置**：定义在 `contingency.hpp`（契约只要求存在于
   `vibe::verify`）。四个 `compute_*` 自由函数**全部在 `scores.hpp` 中声明**
   （契约要求），其实现分别落在 `scores.cpp / contingency.cpp / probabilistic.cpp /
   spatial.cpp`，避免 `scores.hpp` 反向包含其它头文件。
   `Scores` 中不适用字段置 0，适用但退化的评分置 NaN（见 2.3 节）。
4. **`compute_probabilistic` 不含 CRPS**：CRPS 需要集合成员，而契约签名的输入是
   单值概率序列，无法计算。集合 CRPS 由 `crps_ensemble` / `ensemble_scores` 提供，
   其 `spread`、`spread_skill_ratio` 才能被填入 `Scores`。
5. **FSS(0) 与 CSI 不等**：实现遵循 Roberts & Lean (2008) 原始定义；文档与测试
   明确记录 `\text{FSS}(0)=2H/(2H+M+FA)`，避免与 `\text{CSI}` 混淆。

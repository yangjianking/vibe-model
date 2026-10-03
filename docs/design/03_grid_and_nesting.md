# 03 · 网格、嵌套与变分辨率

> 本章说明 VIBE-Model 的网格表示、度量项、并行分解、halo 交换、双向嵌套与
> 平滑变分辨率网格。文献编号见 [references.md](references.md)。

---

## 1. 网格表示

### 1.1 三层抽象

~~~text
Geometry        只描述几何：nx, ny, nz, dx, dy, z_top, zeta[], zs[], dx_cell[], dy_cell[]
   +
Decomposition   只描述并行：px, py, rank, [is,je) x [js,je), halo, 周期性
   =
Grid            几何 + 拓扑 + 预计算度量项（面积、体积、1/dz、雅可比）
   +
Field<T>        单数组 + halo 布局的网格量
~~~

这种切分让「网格」与「数据」彻底分离：同一份 `Geometry` 可以配不同的 `Decomposition`，
而 `Field` 只依赖 `Grid`。

### 1.2 垂直坐标与层生成

地形追随高度坐标下的层定义 [D2]：

$$z(\zeta) = z_s(x,y) + \zeta\,\bigl(H - z_s(x,y)\bigr), \qquad \zeta \in [0, H].$$

层界面的 $\zeta$ 序列由几何拉伸生成（`make_stretched_zeta`）：给定首层厚度
$\Delta_1$ 与拉伸比 $r$，

$$\Delta_k = \Delta_1\, r^{k-1}, \qquad
\zeta_k = \frac{\sum_{m=0}^{k-1}\Delta_m}{\sum_{m=0}^{n_z-1}\Delta_m}.$$

默认参数（首层 60 m、拉伸比 1.055、$n_z = 60$）给出的层厚从约 60 m 增长到约 1.4 km，
兼顾边界层解析度与计算成本。也可以在配置中直接给出完整的 $\zeta$ 数组。

**不变量**：$\zeta_0 = 0$、$\zeta_{n_z} = 1$、严格单调递增。单元测试逐项检查。

### 1.3 度量项

| 量 | 公式 | 接口 |
|---|---|---|
| 雅可比 | $G^{1/2} = (H - z_s)/H$ | `Grid::jacobian` |
| 物理高度（任意分数层） | $z_s + \zeta(k)\,(H - z_s)$ | `Grid::height` |
| 层中心高度 | $z_s + \tfrac12(\zeta_k+\zeta_{k+1})(H-z_s)$ | `Grid::z_center` |
| 层界面高度 | 同 height，整数参数 | `Grid::z_interface` |
| 垂直层厚 | $\Delta\zeta_k\,(H-z_s)$ | `Grid::dzeta` 乘尺度因子 |
| 单元体积 | $\Delta x_i\,\Delta y_j\,G^{1/2}\,\Delta\zeta_k\,H$ | `Grid::cell_volume` |

注意：`Grid::height` 采用「分数层索引」约定——整数参数表示层界面，
$k+0.5$ 表示层中心，因此层中心的 $\zeta$ 严格取上下界面的平均（即使层厚不均匀）。

### 1.4 地形

三种地形由网格模块直接生成，便于理想试验：

* **Witch of Agnesi**（[D2] 经典山波试验）：$h = h_0 a^2/(a^2 + d^2)$，
  其中 $d^2 = (x-x_c)^2+(y-y_c)^2$；
* **Schär 余弦山**（[D9]）：$h = h_0\cos^2\!\bigl(\pi d/(2\lambda)\bigr)$，$|d|\le\lambda$；
* **外部文件**：纯文本矩阵（$n_x$ 行 $n_y$ 列），读入后自动做 3 次 1-2-1 平滑并限幅。

平滑算子的写法（`smooth_terrain`）：

$$z_s^{\text{new}}_{ij} = \tfrac14\left(z_{i-1,j} + z_{i+1,j} + z_{i,j-1} + z_{i,j+1}\right)
+ \tfrac12 z_{ij},$$

随后做**只降不升**的限幅（取邻点最大值）与斜率限幅 $|\Delta z_s| \le c$，
以避免平滑后产生新的极值点。

---

## 2. 并行分解与 halo 交换

### 2.1 分解策略

只切水平维：$(p_x, p_y)$ 笛卡尔分解，垂直维在每个进程上完整保留。理由：

1. 垂直层数通常只有几十层，切分的通信/计算比很差；
2. 半隐式的垂直三对角求解（HEVI）需要整列数据；
3. 垂直方向的数据局部性对 cache 友好。

子域范围的余数分配（`Decomposition::make`）把余数分给靠前的进程：

$$n_x^{\text{local}} = \left\lfloor n_x/p_x\right\rfloor
+ \mathbf{1}\{\,i_x < n_x \bmod p_x\,\}.$$

### 2.2 三阶段 halo 交换

halo 宽度 $b$（默认 4，满足 WENO5 的 5 点模板与 4 阶中心差分）。设本地内部范围为
$[b, b+n_x)\times[b, b+n_y)$：

| 阶段 | 方向 | 覆盖范围 | 作用 |
|---|---|---|---|
| 1 | W/E | $j \in [b, b+n_y)$ | 填左右边（不含角） |
| 2 | S/N | $i \in [0, n_{sx})$ | 利用已填好的左右 halo 填上下边与部分角 |
| 3 | W/E | $j \in [0,b)\cup[b+n_y, n_{sy})$ | 填四个角 |

图示（$b = 2$，内部 4x4 点）：

~~~text
   阶段 1：填左右两侧的 2 列
   阶段 2：填上下两侧的行（此时左右 halo 已有效，因此边上的角也正确）
   阶段 3：填四个 2x2 角块
~~~

通信量约为 $(b\,n_y + n_x b + b^2)\,n_z$ 个元素，比整片交换省一个量级。
实现使用 `MPI_Sendrecv`（对称模式，天然避免死锁）；生产代码可改为
`MPI_Isend/MPI_Irecv` 并与内部点计算重叠。

**周期性边界**不产生通信：halo 在本地环绕复制（`HaloExchange::exchange_local`），
这也是理想试验与单元测试的主要模式。

---

## 3. 双向嵌套

### 3.1 Davies 侧边界松弛 [N1][N2][N9]

子域边界处用父域插值值松弛。设松弛区厚度为 $N$ 个细网格点，第 $n$ 个点
（$n = 0$ 为最外边界）的权重为

$$\alpha(n) = \alpha_{\max}\cdot\tfrac12\left[1 - \cos\!\left(\pi\frac{N-n}{N}\right)\right],$$

于是 $\alpha(0) = \alpha_{\max}$、$\alpha(N) = 0$，且一阶导数连续（不引入反射源）。
更新公式（对每个预报量）：

$$\phi^{\text{child}} \leftarrow (1-\alpha)\,\phi^{\text{child}} + \alpha\,\phi^{\text{parent}} .$$

实现：`davies_profile` 与 `Nest::apply_lateral_boundary`；松弛区厚度由配置项
`nesting.boundary_zone` 控制（默认 5 个细网格点）。

### 3.2 父到子插值（prolongation）

子域点 $(i,j)$ 的中心在父域坐标中的位置为

$$x_p = \frac{i_0 + i + 0.5}{r} - 0.5, \qquad
y_p = \frac{j_0 + j + 0.5}{r} - 0.5,$$

随后做双线性（可升级为双三次）插值。本模式内外层垂直坐标相同，父子层一一对应，
因此不需要垂直插值。

### 3.3 子到父反馈（restriction）与守恒 [N3][N4]

父单元覆盖 $r\times r$ 个子单元，体积加权平均：

$$\phi^{\text{parent}}_{ij} = \frac{1}{r^2}
\sum_{i'=0}^{r-1}\sum_{j'=0}^{r-1}\phi^{\text{child}}_{\,r i + i',\; r j + j'} .$$

对密度类变量，先乘体积再平均，保证总质量守恒：

$$\sum_{\text{children}}\rho_c\,\Delta V_c = \rho_p\,\Delta V_p .$$

实现：`Nest::restrict_to_parent`。单元测试用常数块验证该恒等式。

### 3.4 时间子循环

父域一个大步 $\Delta t$ 对应子域 $r$ 个小步 $\Delta t/r$，父子在共同同步时刻交换一次
[N4][N8]：

~~~text
父:  |----------------- dt ------------------|
子:  |--- dt/3 ---|---- dt/3 ----|--- dt/3 ---|
              ^ 下传边界            ^ 上行反馈
~~~

这维持了二阶时间精度；若不做子循环，子域会因 CFL 而被迫使用与父域相同的时间步，
失去嵌套的意义。

---

## 4. 平滑变分辨率网格

### 4.1 基本思想 [N5][N6][N7][T19]

给定目标分辨率函数 $h(x,y) \in [h_{\min}, h_{\max}]$，构造单调映射

$$\xi = F(x) = \frac{1}{L}\int_0^x \frac{\mathrm{d}s}{h(s)},
\qquad x = F^{-1}(\xi).$$

在计算坐标上等距取 $\xi_k = k/n$，映射回物理坐标即得单元界面 $x_k$。
$h$ 小的地方积分增长快，因此物理间距小，实现自动加密。

**缓变条件**：为保证映射光滑且离散误差可控，要求

$$\left|\frac{\mathrm{d}h}{\mathrm{d}x}\right| \le C\,\frac{h}{L}.$$

实现中由 `smooth_refinement` 做迭代平滑（1-2-1 滤波）并逐点限幅，
$C$ 对应配置中的最大梯度参数。

### 4.2 数值实现

~~~text
1. 在细采样（n*16 点）上用 Simpson 积分累积 F(x)，归一化到 [0,1]
2. 对每个 k，在 F 的逆上做线性插值得到 x_k
3. 单元宽度 dx_k = x_{k+1} - x_k，断言严格为正（映射单调 -> 无翻转单元）
4. 二维情形：x、y 方向分别构造，网格为张量积
~~~

复杂度 $O(16n + n\log n)$；实现于 [variable_resolution.cpp](../../src/grid/variable_resolution.cpp)。

### 4.3 与嵌套的关系

两者在动力学代码中**共用同一套度量项接口**（`Grid::dx_at` 与 `Grid::dy_at`），
因此动力学层不需要知道当前是嵌套还是变分辨率。差别在于：

| | 嵌套 | 变分辨率 |
|---|---|---|
| 分辨率跳变 | 界面处不连续（$r$ 倍） | 处处连续 |
| 守恒性 | 需要显式的 restriction 算子 | 由连续映射自然满足 |
| 反射 | 界面可能产生虚假反射 | 无界面，反射小 |
| 计算效率 | 只在子域加密，效率高 | 过渡区也有细网格，效率略低 |
| CFL | 父子各自满足 | 全局由最小单元决定 |

---

## 5. 精度与性能注意事项

1. **CFL 由最小单元决定**。`max_stable_dt` 给出显式意义下的最大稳定步长，
   变分辨率配置应以此为准。
2. **halo 宽度是硬约束**。平流模板宽度、插值阶数、嵌套松弛区厚度的最大值决定了
   所需 halo；配置不一致时 `VIBE_CHECK` 会立即报错。
3. **面积与体积必须与离散算子自洽**。所有守恒性检验（质量、能量、位涡）都依赖
   `Grid::cell_volume` 与通量公式使用同一个度量项。
4. **对角 halo 不可省略**。双线性插值、嵌套反馈以及高阶平流的交叉模板都需要角点数据。

---

## 6. 代码对照表

| 功能 | 接口 | 文件 |
|---|---|---|
| 几何与度量 | `grid::Geometry`、`grid::Grid` | [geometry.cpp](../../src/grid/geometry.cpp) |
| 并行分解 | `grid::Decomposition` | [decomposition.hpp](../../include/vibe/grid/decomposition.hpp) |
| 网格量容器 | `grid::FieldT` | [field.hpp](../../include/vibe/grid/field.hpp) |
| halo 交换 | `grid::HaloExchange` | [halo.cpp](../../src/grid/halo.cpp) |
| 插值与重映射 | `conservative_restrict`、`bilinear_prolong` | [interpolation.cpp](../../src/grid/interpolation.cpp) |
| 嵌套 | `grid::Nest`、`grid::NestHierarchy` | [nest.cpp](../../src/grid/nest.cpp) |
| 变分辨率 | `grid::VarResMap1D`、`VarResGridBuilder` | [variable_resolution.cpp](../../src/grid/variable_resolution.cpp) |

---

## 7. 参考文献

[N1] Davies (1976) · [N2] Davies (1983) · [N3] Clark & Farley (1984) ·
[N4] Skamarock & Klemp (1993) · [N5] Skamarock et al. (2012) · [N6] Ringler et al. (2013) ·
[N7] Tomita & Satoh (2004) · [N8] Harris & Lin (2013) · [N9] Warner et al. (1997) ·
[T19] Zängl et al. (2015) · [D2] Gal-Chen & Somerville (1975) · [D9] Schär et al. (2002)

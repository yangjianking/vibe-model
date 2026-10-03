# 后处理包 `vibe_post` 设计说明

> 本文是 Python 后处理包的**接口与格式规范**，与 [00_architecture.md](00_architecture.md) 第 10 节
> （配置与 IO 契约）、第 9 节（检验契约）以及 [references.md](references.md) 配套。
> 文中公式编号沿用 references.md 的方括号编号（如 `[D2]`、`[E7]`）。

---

## 1. 定位与边界

`vibe_post` 是 VIBE-Model 的独立 Python 后处理库，**不依赖 C++ 运行时**，
只通过文件与模式交换数据。对应关系：

| 方向 | 规则 |
|---|---|
| C++ -> Python | 模式输出 NetCDF / `.vibebin`，Python 只读 |
| Python -> C++ | Python 写 `.vibebin` / NetCDF，C++ `io` 层只读 |
| 共享 | 变量命名、逻辑维度顺序 `(time, zeta, y, x)`、`.vibebin` 二进制布局 |
| 不共享 | C++ 类型定义、halo/并行分解、MPI 通信（Python 侧完全无感知） |

**硬边界**：Python 侧不实现任何时间推进、物理参数化与同化；它只做
"读 -> 诊断 -> 插值 -> 检验 -> 绘图"这一条链。稠密线性代数（Helmholtz 求解、
B 矩阵作用）也留在 C++，Python 只消费其结果。

### 1.1 惰性依赖策略

| 能力 | 依赖 | 缺失时的行为 |
|---|---|---|
| 核心（数组、诊断、检验、`.vibebin`） | numpy | 必需 |
| NetCDF 读写 | xarray + netCDF4 | `MissingDependencyError`，附 `pip install 'vibe-post[netcdf]'` |
| GRIB2 读取 | cfgrib + eccodes | `MissingDependencyError`，附安装提示 |
| 绘图 | matplotlib | `MissingDependencyError`，附安装提示 |
| 配置解析 | PyYAML（可选） | 自动回退到内置极简 YAML 解析器 |

所有可选依赖**只在函数内部导入**，因此 `import vibe_post` 在最小 numpy 环境中总是成功。
`vibe_post.open_dataset` 在 `backend="auto"` 下遇到依赖缺失会自动尝试下一后端，
只有全部失败才抛出汇总错误。

---

## 2. 包结构

~~~
python/vibe_post/
  pyproject.toml              打包与 extras 声明（netcdf/grib/plots/yaml/all/test）
  README.md                   面向使用者的上手说明
  src/vibe_post/
    __init__.py               公开 API 汇总 + __getattr__ 惰性暴露
    config.py                 配置读取 + 内置极简 YAML 子集解析器
    grid.py                   GridSpec、地形追随高度、Arakawa C-grid 错位
    io.py                     Dataset、NetCDF/cfgrib/VibeBin 三后端
    interp.py                 双/三线性、PCHIP、保守粗化、延拓
    diagnostics.py            热力学 / 动力 / 遥感诊断量（无状态）
    verify.py                 连续 / 分类 / 概率 / 邻域四类检验评分
    plots.py                  matplotlib 绘图（含自绘 skew-T、动能谱）
    cli.py                    `vibe-post` argparse 命令行
    __main__.py               `python -m vibe_post` 入口
  examples/quickstart.py      端到端示例（合成数据，无需外部文件）
  tests/test_io.py            `.vibebin` 与后端探测（11 个测试）
  tests/test_interp.py        插值与粗化（12 个测试）
  tests/test_diagnostics.py   诊断量（11 个测试）
  tests/test_verify.py        检验评分（10 个测试）
~~~

模块依赖是**单向**的：

~~~
config -> (无)
grid   -> config(仅 parse_quantity)
io     -> config(仅 parse_quantity)
interp -> (仅 numpy)
diagnostics -> interp(仅 SRH/垂直插值)
verify -> (仅 numpy)
plots  -> io(取 Dataset 类型) + diagnostics(湿绝热反解)
cli    -> 全部
~~~

不存在循环依赖；`io` 与 `verify` 都不导入 `plots`。

---

## 3. 数据模型

### 3.1 Dataset

唯一的数据交换对象是 :class:`vibe_post.io.Dataset`：

~~~python
@dataclass
class Dataset:
    fields: dict[str, np.ndarray]     # 变量名 -> 数组
    dims:   dict[str, tuple[str, ...]]  # 变量名 -> 逻辑维度名
    coords: dict[str, np.ndarray]     # 坐标本身也是 field
    attrs:  dict[str, Any]            # 全局属性（JSON 可序列化）
    time:   float                     # 秒，相对起报时刻
~~~

**维度与内存顺序**（与 C++ 严格一致）：

* 逻辑维度顺序为 `(time, zeta, y, x)`，其中 `i`（x 方向）变化最快；
* numpy 是 C 序，故数组下标顺序是逻辑顺序的**反转**：`(t, k, j, i)`；
* `fields["theta"].shape == (nt, nz, ny, nx)`，且
  `fields["theta"][t, k, j, i]` 对应 C++ `Field::at(i, j, k)`（第 t 个时刻）。

垂直坐标名为 **`zeta`**，量纲为米（地形追随坐标值），与 C++
`grid::Geometry::zeta` 同义；物理高度由 :meth:`GridSpec.height_array` 计算。
这样命名是为了避免与 NetCDF 里的几何高度 `z` 混淆。

### 3.2 规范变量名

与 architecture 第 6.1 节一致：`u, v, w, rho, theta, pi, qv, qc, qr, qi, qs, qg`，
诊断量沿用 C++ `dyn` 的命名（`t, p, zs, dbz, cape, cin, srh, slp`…）。
维度别名表把 WRF/COSMO/xarray 的写法折叠到同一套名字：
`z/lev/level/bottom_top -> zeta`、`lat -> y`、`lon -> x`、`time_counter -> time`。

---

## 4. `.vibebin` 二进制格式规范（与 C++ 逐字节一致）

权威定义见 `include/vibe/io/field_io.hpp` 文件头与 `src/io/binary_writer.cpp`；
Python 侧 `vibe_post.io.VibeBinFile` 是它的**逐字节对应实现**。

### 4.1 布局

全部小端（little-endian），显式按字节拼装，不依赖主机字节序。

~~~text
偏移   长度   内容
----   ----   ------------------------------------------------------
0      8      magic = b"VIBEBIN1"（ASCII）
8      4      int32  nx
12     4      int32  ny
16     4      int32  nz
20     4      int32  real_kind   (0 = float64 写者, 1 = float32 写者)
24     8      float64 time      [s]
32     4      int32  nvars
36     4      int32  nlevels    (= nz，供剖面型工具使用)
----   以下重复 nvars 次 ----------------------------------------
+0     32     变量名（定长 ASCII，不足补 NUL）
+32    N*8    数据（float64，行主序：i 最快、k 最慢）
~~~

**关键约定**：文件里只有 32 字节名字可以承载形状信息，因此**名字唯一决定错位**
（与 C++ `shape_for_name` 完全一致）：

| 变量名 | 形状 `(nx_v, ny_v, nz_v)` |
|---|---|
| `u` / `u_facex` | `(nx+1, ny, nz)` |
| `v` / `v_facey` | `(nx, ny+1, nz)` |
| `w` / `zeta` / `w_facez` | `(nx, ny, nz+1)` |
| `__restart_info__` | `(2, 1, 1)` |
| 其余 | `(nx, ny, nz)` |

数据**始终以 float64 存储**，`real_kind` 只记录写者的原生 `Real` 类型。

### 4.2 可选扩展（向后兼容，默认关闭）

`write_vibebin(..., extended=True)` 时每条记录在名字后追加
`int32[nx_v, ny_v, nz_v]` 形状与 `uint32` CRC32，并在文件尾追加
`b"VIBETRL1" + int32 attr_bytes + UTF-8 JSON 属性`。读取器先按 classic 解析，
字节数不吻合再尝试 extended，因此两种布局都能读；C++ 读取器只认 classic，
所以 Python 默认写出 classic。

### 4.3 与 C++ 的边界

| 项 | C++ `io` | Python `vibe_post.io` |
|---|---|---|
| magic / 头字段 / 顺序 | 见上表 | 相同（40 字节） |
| 记录布局 | `name[32] + float64[N]` | 相同 |
| 数据顺序 | 行主序，i 最快 | 相同 |
| 形状来源 | 变量名 | `shape_for_name`（同表） |
| 头解析入口 | `read_binary_header` | `read_binary_header` / `VibeBinHeader.unpack` |
| 自检 | 无（读取即校验维度） | `self_check()` 检查形状与 Inf |
| 属性 | 无 | 仅 extended 尾部（C++ 可忽略） |

### 4.4 体心视图（使用约定）

诊断量要求所有输入同形，而 u/v/w 是面场，因此读取后需要显式取回体心：

~~~python
u_cell = ds.fields["u"][..., :nx]        # x 面：u[i] 与体心 i 共享物理位置（C++ Grid::index 语义）
v_cell = ds.fields["v"][:, :ny, :]        # y 面
w_cell = 0.5 * (ds.fields["w"][:-1] + ds.fields["w"][1:])   # 层界面 -> 层中心
~~~

CLI 的 `derive` / `verify` 子命令内部用 `_cell_fields()` 完成该转换。

## 5. 网格与 Arakawa C-grid

### 5.1 坐标变换

地形追随高度坐标 [D2]（Gal-Chen & Somerville 1975）：

.. math::

    \\zeta = H \\frac{z - z_s(x,y)}{H - z_s(x,y)},
    \qquad
    z = z_s + \\zeta \\left(1 - \\frac{z_s}{H}\\right)

其中 :math:`H = z_{top}`。对每个 :math:`(i,j)` 该映射是线性的，因此层中心的 :math:`z`
等于层中心 :math:`\\zeta` 的像（离散化无额外误差）。
雅可比 :math:`G^{1/2} = \\partial z/\\partial\\zeta = 1 - z_s/H`，单元体积
:math:`\\Delta V = \\Delta x\\,\\Delta y\\,(1-z_s/H)\\,\\Delta\\zeta`。

`:class:`GridSpec`` 提供 `height_array()`（层中心与 w 层界面）、
`dz()`、`jacobian()`、`cell_volume()`、`x_centers()/x_faces()`。
变分辨率网格用逐列 `dx_cell` 与逐行 `dy_cell` 表示 [N5][N6]。

### 5.2 错位插值

Arakawa C-grid [D3] 的错位 -> 体心用二阶中心平均：

.. math::

    u^{c}_{i,j,k} = \\tfrac12\\left(u_{i-1/2,j,k} + u_{i+1/2,j,k}\\right), \\quad
    v^{c}_{i,j,k} = \\tfrac12\\left(v_{i,j-1/2,k} + v_{i,j+1/2,k}\\right), \\quad
    w^{c}_{i,j,k} = \\tfrac12\\left(w_{i,j,k-1/2} + w_{i,j,k+1/2}\\right)

边界点索引钳位，等价于零梯度（单侧）外推，与 C++
`grid::stagger_to_cell` 的边界处理逐点一致。反向（体心 -> 面）用
`cell_to_face`，端点线性外推。

### 5.3 垂直层生成

几何拉伸（`zeta_levels(mode="geometric")`）：

.. math::

    \\Delta\\zeta_k = \\Delta\\zeta_0 s^k,
    \\qquad \\Delta\\zeta_0 = \\frac{H(s-1)}{s^{n_z}-1}

保证 :math:`\\sum_k \\Delta\\zeta_k = H`（模式顶严格落在 :math:`H`）。
`mode="smooth"` 用 [D9][D10] 的连续形式
:math:`w(\\sigma)=1+(\\beta-1)\\sigma^2` 分配层厚，抑制陡坡上的虚假气压梯度。

---

## 6. 插值

### 6.1 水平

双线性（[D3] 第 3 节）在体心索引坐标 :math:`\\xi = x/\\Delta x - 1/2` 上做；

.. math::

    f(\\xi,\\eta) = (1-a)(1-b)f_{i,j} + a(1-b)f_{i+1,j}
                     + (1-a)b\\,f_{i,j+1} + ab\\,f_{i+1,j+1}

`bilinear` 支持批量点（数组广播），越界可选零梯度钳位或返回 NaN。
三线性 `trilinear` 先在逐列高度场上二分定位层，再做双线性。

### 6.2 垂直：单调 PCHIP

自实现（不依赖 scipy）的单调分段三次 Hermite [B9] 第 3 章：
节点导数用 Fritsch-Carlson 限制的加权调和平均

.. math::

    d_k = \\frac{w_1+w_2}{w_1/\\delta_{k-1} + w_2/\\delta_k},
    \\quad w_1 = 2h_k+h_{k-1},\\; w_2 = h_k+2h_{k-1}

若 :math:`\\delta_{k-1}\\delta_k\\le0` 则 :math:`d_k=0`。该格式**保持单调、不过冲**，
对水汽、位温等正定量尤其重要；相比线性插值在急剧层结处既不引入虚假极值，
又保持 :math:`C^1` 连续。

`interp_to_height` / `interp_to_pressure` 支持逐列不同坐标；气压递减时自动翻转为
递增再插值。NaN 层被跳过，目标越界用端点值（零梯度外推）。

### 6.3 守恒粗化与延拓

细 -> 粗的面积加权守恒重映射 [N5]：

.. math::

    \\bar f_c = \\frac{\\sum_{f\\in c} f_f A_f}{\\sum_{f\\in c} A_f}

均匀网格上退化为 :math:`r\\times r` 分块均值（权重 :math:`1/r^2`），
与 architecture 第 5.4 节的守恒约束一致。`conservative_coarsen` 用逐偏移累加实现，
支持任意轴组合；`nan_policy="skip"` 忽略 NaN 并按有效点数重归一化，
`"propagate"` 则把 NaN 计入分母。
粗 -> 细用双线性延拓（Davies 插值部分 [N1]），可选保正。

---

## 7. 诊断量：公式与文献

全部为纯函数（无状态），单位 SI。`[D16]` 指 WRF ARW 技术说明，
`[B4]` 指 Holton & Hakim。

| 函数 | 公式 | 文献 |
|---|---|---|
| `pressure_from_exner` | :math:`p = p_0\\,\\Pi^{c_p/R_d}` | [D16] 第 3 章 |
| `temperature_from_exner` | :math:`T = \\theta\\,\\Pi` | [D16] |
| `saturation_vapor_pressure` | :math:`e_s = 611.2\\exp[17.67(T-273.15)/(T-29.65)]`（冰面 22.46/0.55） | [B4] 第 3 章 |
| `saturation_mixing_ratio` | :math:`q_{v,sat} = \\varepsilon e_s/(p-e_s)` | [B4] |
| `mixing_ratio` | :math:`q_v = \\varepsilon e/(p-e)`（精确式） | [B4] |
| `dewpoint` | :math:`T_d = \\frac{a_1(T_0-a_2)\\ln(e/e_0)}{a_1-\\ln(e/e_0)}+T_0` | [B4] |
| `relative_humidity` | :math:`RH = q_v/q_{v,sat}` | [D16] |
| `theta_e` | :math:`\\theta_e = T(p_0/p)^{\\kappa}\\exp[L_vq_v/(c_pT)]` | [B4] |
| `wind_speed` / `wind_direction` | :math:`\\sqrt{u^2+v^2}` / :math:`270^\\circ - \\arctan_2(v,u)` | [O9] |
| `geopotential` | :math:`\\Phi = gz` | [B4] 第 1 章 |
| `vorticity` | :math:`\\zeta = \\partial v/\\partial x - \\partial u/\\partial y` | [B4] 第 4 章、[D3] |
| `divergence` | :math:`D = \\partial u/\\partial x + \\partial v/\\partial y` | [B4] |
| `absolute_vorticity` | :math:`\\eta = f+\\zeta`，:math:`f=2\\Omega\\sin\\varphi` | [B4] |
| `potential_vorticity` | :math:`PV = \\rho^{-1}\\boldsymbol\\omega_a\\cdot\\nabla\\theta` | [B4] 第 4 章 |
| `cape_cin` | :math:`\\int g\\,(T_{v,p}-T_{v,e})/T_{v,e}\\,dz` | [P11]、[B4] 第 3 章 |
| `storm_relative_helicity` | :math:`\\int[(u-c_u)\\partial_zv-(v-c_v)\\partial_zu]dz` | [B4] 第 8 章 |
| `precipitation_accumulation` | :math:`P=\\sum_n R_n\\Delta t_n` | [P1] |
| `sea_level_pressure` | :math:`p_{SL}=p_s[1+\\Gamma z_s/T_s]^{g/(R_d\\Gamma)}` | [D16] 第 3 章 |
| `reflectivity_dbz` | :math:`Z\\approx3.63\\times10^9(\\rho q_r)^{1.75}`，:math:`10\\log_{10}Z` | [O6]、[P3] |
| `brightness_temperature` | :math:`T_b=(F_{up}/\\varepsilon\\sigma)^{1/4}` | [P5] |
| `k_index` | :math:`(T_{850}-T_{500})+T_{d,850}-(T_{700}-T_{d,700})` | [B4] 第 3 章、[E19] |
| `showalter_index` | :math:`SI = T_{500}-T_{parcel,500}` | [B4] 第 3 章 |

### 7.1 水平差分的 C-grid 约定

`vorticity` / `divergence` 用"索引对齐的中心差分"：C-grid 上 `u` 与 `v` 的存储
索引与体心 (i,j,k) 一一对应（C++ `Grid::index(Stagger,i,j,k)` 语义），因此

.. math::

    \\left(\\frac{\\partial f}{\\partial x}\\right)_i
    = \\frac{f_{i+1/2}-f_{i-1/2}}{\\Delta x}
    \\;\\to\\;
    \\frac{f_i - f_{i-1}}{\\Delta x}

内部点为二阶 :math:`O(\\Delta x^2)`，边界退化为单侧一阶，与 C++
`dyn::relative_vorticity` 逐点一致。

### 7.2 CAPE/CIN 的离散步骤

1. 起点（最低层或指定高度）的 :math:`(p_0,T_0,q_{v,0})`；
2. 抬升凝结高度：:math:`T_{LCL}` 用 Bolton 近似，:math:`p_{LCL}=p_0(T_{LCL}/T_0)^{1/\\kappa}`；
3. LCL 以下干绝热 :math:`T=\\theta(p/p_0)^\\kappa`，以上用 Newton 迭代反解
   :math:`\\theta_e(T,p,q_{v,sat})=\\theta_{e,LCL}`（单调，全局收敛）；
4. 虚温 :math:`T_v=T(1+0.608q_v)`，浮力 :math:`B=g(T_{v,p}-T_{v,e})/T_{v,e}`；
5. 梯形法积分得累计浮力，取正/负面积分别为 CAPE/CIN，LFC/EL 取首个/最后一个正浮力层。

---

## 8. 检验评分：公式与文献

### 8.1 连续量

| 评分 | 公式 | 文献 |
|---|---|---|
| bias | :math:`N^{-1}\\sum(f_i-o_i)` | [B9] 第 8 章 |
| MAE | :math:`N^{-1}\\sum|f_i-o_i|` | [B9] |
| RMSE | :math:`\\sqrt{N^{-1}\\sum(f_i-o_i)^2}` | [B9] |
| 相关系数 | :math:`r=\\frac{\\sum(f-\\bar f)(o-\\bar o)}{\\sqrt{\\sum(f-\\bar f)^2\\sum(o-\\bar o)^2}}` | [B9] 第 3 章 |
| ACC | 同上但以气候态 :math:`c` 为基准 | [E19] |

零方差、N<2、空数组、全 NaN 一律返回 NaN（数学上未定义），不抛异常。

### 8.2 分类量

列联表 (H, M, F, C)，:math:`N=H+M+F+C`：

| 评分 | 公式 | 文献 |
|---|---|---|
| POD | :math:`H/(H+M)` | [E18] |
| FAR | :math:`F/(H+F)` | [E18] |
| CSI | :math:`H/(H+M+F)` | [E6] |
| ETS | :math:`(H-H_r)/(H+M+F-H_r)`，:math:`H_r=(H+M)(H+F)/N` | [E4]、[E19] |
| TSS | :math:`H/(H+M)-F/(F+C)` | [E5] |
| 频率偏差 | :math:`(H+F)/(H+M)` | [E18] |
| ORSS | :math:`(HC-FM)/(HC+FM)` | [E4]、[B10] 第 3 章 |

### 8.3 概率量

| 评分 | 公式 | 文献 |
|---|---|---|
| Brier | :math:`N^{-1}\\sum(p_i-o_i)^2` | [E3]、[E11] |
| BSS | :math:`1-\\mathrm{BS}/[p_c(1-p_c)]` | [E11] |
| 可靠性曲线 | 分箱后的 :math:`\\bar p_k` 对 :math:`\\bar o_k` | [E2]、[B9] |
| ROC / AUC | 秩和公式（并列用平均秩） | [E15] |
| CRPS | :math:`\\mathrm{MAE}-\\frac{1}{2M(M-1)}\\sum_{m,n}|x_m-x_n|` | [E12]、[E16] |
| 排序直方图 | :math:`r=\\#\\{m:x_{(m)}<y\\}` | [E9]、[E10] |
| SSR | :math:`\\overline{\\sigma_{ens}}/\\mathrm{RMSE}(\\bar x,y)` | [B9] 第 8 章 |

CRPS 的集合项是本包**唯一**需要 O(M^2) 的地方（逐点成对绝对差），
与文档公式逐项对应，便于与 C++ 实现交叉核对。

### 8.4 邻域：FSS

.. math::

    F^{(w)}(i,j) = \\frac{1}{n_{ij}}\\sum_{p,q\\in\\mathcal N_w(i,j)} I(x_{p,q})

.. math::

    \\mathrm{FSS} = \\frac{2\\,\\mathrm{FBS}_w}{\\mathrm{FBS}_w+\\mathrm{FBS}_{ref}},
    \\quad
    \\mathrm{FBS}_{ref} = \\sum(F_f)^2+\\sum(F_o)^2

`neighborhood_fractions` 用积分图实现，复杂度 O(N)（与窗口无关）；
边界用有效点数归一，避免零填充稀释。有用技巧判据
:math:`\\mathrm{FSS}>0.5+f_o/2`（[E7] 式 (5)），由 `fss_vs_scale` 给出。

**约定**：FSS 需要二维水平面；CLI/示例用 `f[-1, :, :]` 取单层，
若差异来自垂直方向需要先自行插值到同一高度。

---

## 9. 绘图约定

* 所有 matplotlib 导入在函数内部；缺依赖抛 `MissingDependencyError`；
* 单面板返回 `(fig, ax)`、多面板返回 `(fig, axes)`，由调用方 `savefig`；
* 轴标签必带单位；水平坐标默认 km、垂直坐标默认 m 或 hPa；
* 色标用感知均匀的 `viridis`/`magma`/`RdBu_r`；
* 函数接受数组或 `Dataset` + 变量名两种调用形式；
* `plot_skewt` 自绘（干绝热线、湿绝热线、饱和混合比线），不依赖 MetPy；
* `kinetic_energy_spectrum` 做 Hann 加窗周期图，返回谱斜率与拟合波段。

---

## 10. 与 C++ 侧的接口边界（核对清单）

以下逐项对照 `include/vibe/io/`、`include/vibe/verify/`、`include/vibe/grid/`
与 `include/vibe/dyn/state.hpp` 的实际定义。

| 检查项 | C++ | Python | 状态 |
|---|---|---|---|
| 状态变量名 | `dyn::State` 字段 | `CANONICAL_VARIABLES` | 一致 |
| 逻辑维度顺序 | `(time, zeta, y, x)`，i 最快 | 同（numpy 下标为反转） | 一致 |
| 垂直坐标 | `grid::Geometry::zeta` | `GridSpec.zeta` | 一致 |
| 高度公式 | `Grid::height` = `z_s + ζ(1-z_s/H)` | `GridSpec.height` | 一致 |
| 错位插值边界 | 零梯度钳位 | `stagger_to_cell` 相同 | 一致 |
| 守恒粗化权重 | `1/r^2` | `conservative_coarsen` 分块均值 | 一致 |
| 涡度/散度差分 | `dyn::relative_vorticity` 索引对齐中心差分 | `vorticity` 相同 | 一致 |
| `.vibebin` magic / 头 / 记录 | `io::BinaryWriter` | `VibeBinFile` | **逐字节一致** |
| 变量形状规则 | `shape_for_name`（名字决定错位） | `shape_for_name` | 一致 |
| 数据精度 | 始终 float64 存储 | 同 | 一致 |
| CRC32 / 形状冗余 / JSON 属性 | 无 | 仅 `extended=True` | 扩展，默认关闭 |
| 连续量评分 | `verify::compute_continuous` | `verify.py` | 一致（含零方差 -> 0.0） |
| 分类量评分 | `ContingencyTable` 成员函数 | `contingency_table` + 各评分 | 一致（分母 0 -> NaN） |
| FSS | `1 - FBS/FBS_ref`，窗口 `2r+1` | `fractions_skill_score` | 一致 |
| CRPS | `1/M Σ|x-y| - 1/(2M(M-1)) ΣΣ|x_i-x_j|`（fair） | `crps_ensemble(fair=True)` | 一致 |

### 10.1 明确的接口约定

1. **面场形状**：`u/v/w` 按名字隐含形状存储（各加一列/一行/一层），因此
   跨语言交换时**不要**按“所有变量同形”假设写出；Python 的 `add_variable`
   会校验形状并在不匹配时报 `VibeBinError`，避免静默写坏数据。
2. **体心视图**：诊断量要求同形输入，Python 侧显式取回体心（见 4.4 节），
   不做隐式插值——隐式插值会掩盖错位错误。
3. **相关系数退化**：常数序列（方差为 0）返回 `0.0`，与 C++
   `verify::pearson_correlation` 一致；`n < 2` 仍返回 NaN。
4. **评分未定义**：分母为 0 的评分（无事件时的 POD、缺气候态时的 ACC 等）
   返回 NaN，以区分“未定义”与“0 分”（[E13]）。
5. **CAPE 起点**：Python 默认取最低层（surface-based），可用 `z_start=` 指定；
   C++ `dyn::cape_cin_column` 要求显式 `z_start`，语义一致、默认值不同。
6. **位涡密度**：缺 `rho` 时由 `p/(R_d T_v)` 反算；两者都不给则按单位密度计算
   并发出 `UserWarning`，不静默出错。

## 11. 复杂度总表

| 模块/函数 | 时间 | 空间 |
|---|---|---|
| `parse_yaml` | O(N) 字符 | O(深度) |
| `GridSpec.height_array` | O(nx ny nz) | O(nx ny nz) |
| `stagger_to_cell` | O(N) | O(N) |
| `bilinear` / `trilinear` | O(M) | O(M) |
| `pchip` | O(n + M log n) | O(n+M) |
| `interp_to_height` | O(nlev x N) | O(N) |
| `conservative_coarsen` | O(N) | O(N) |
| `cape_cin` | O(nz x N_h x n_iter) | O(nz x N_h) |
| `storm_relative_helicity` | O(n_sub x N_h) | O(n_sub x N_h) |
| `potential_vorticity` | O(N) | O(N) |
| `crps_ensemble` | O(M^2 x N_h) | O(M^2 x N_h) |
| `fractions_skill_score` | O(N) | O(N) |
| `kinetic_energy_spectrum` | O(N log N) | O(N) |

其中 N 为场点数、M 为插值/集合点数。

---

## 12. 测试策略

* 全部使用 numpy 合成数据，不访问网络、不依赖外部数据文件；
* 解析可验证的构造优先：线性场验证双/三线性精确性、等熵大气验证位涡、
  旋转风廓线验证 SRH、指数风廓线验证 CAPE；
* 边界情形必须覆盖：全 NaN、零方差、空数组、单点、重复层、坐标乱序、
  比率不整除、窗口非奇数、维度不匹配；
* 每个 test_*.py 自带 `_run_all()`，无 pytest 时可直接 `python tests/test_xxx.py`；
* 共 44 个测试函数，覆盖 io(9) / interp(12) / diagnostics(11) / verify(12)。

---

## 13. 文献

本文引用的编号全部来自 [references.md](references.md)：

* 控制方程与垂直坐标：[D2] Gal-Chen & Somerville (1975)、[D3] Arakawa & Lamb (1977)、
  [D9][D10][D11] 平滑坐标面、[D13] 守恒离散、[D16] WRF ARW；
* 嵌套与变分辨率：[N1] Davies (1976)、[N5] 守恒重映射、[N6] 变分辨率；
* 物理参数化：[P1] Kessler (1969)、[P3] Thompson et al. (2008)、
  [P5] Mlawer et al. (1997)、[P11] Kain & Fritsch (1990)；
* 观测与算子：[O6] Sun & Crook (1997)、[O9] WMO-No.8；
* 误差检验：[E2]–[E19]；
* 教材：[B4] Holton & Hakim、[B9] Wilks、[B10] Jolliffe & Stephenson。

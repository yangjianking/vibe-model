# VIBE-Model 物理参数化设计（08_physics）

> 对应代码：<code>include/vibe/physics/</code>、<code>src/physics/</code>、
> <code>tests/unit/test_physics.cpp</code>。
> 文献编号 [P*]/[B*]/[D*] 取自 [references.md](references.md)；
> 接口基线见 [00_architecture.md](00_architecture.md) 第 3、6、12 节。
>
> **记号约定**：$F^{+}$ 为向上通量、$F^{-}$ 为向下通量；
> 降水率 1 mm/s 等价于 1 kg m$^{-2}$ s$^{-1}$（$\rho_w=1000$ kg/m$^3$）；
> 水物质为混合比（kg/kg）；H、LE 向上为正（W/m$^2$）。

---

## 1. 目标、范围与接口约定

物理参数化模块把**次网格尺度**的动量、热量与水物质输送表达为可分辨尺度变量的倾向，
按固定顺序交给时间积分器。

**冻结接口**

| 类型 / 函数 | 位置 | 说明 |
|---|---|---|
| <code>physics::PhysicsColumn</code> | <code>physics_types.hpp</code> | 单列视图（z,p,rho,theta,qv..qg,u,v,tke,dz）+ <code>resize(nz)</code> |
| <code>physics::SurfaceState</code> | <code>physics_types.hpp</code> | 地形/海陆掩膜/粗糙度/反照率/五层土壤/2m,10m/通量/辐射 |
| <code>physics::PhysicsOptions</code> | <code>physics_types.hpp</code> | 全部方案开关与可调参数（第 9 节参数表） |
| <code>physics::PhysicsDiagnostics</code> | <code>physics_types.hpp</code> | 对流/网格降水、云量、大气顶收支、PBL 高度 |
| <code>MicrophysicsBase::step_column</code> | <code>microphysics.hpp</code> | 单列微物理，返回地面降水率（mm/s） |
| <code>PhysicsDriver::step(s,g,ref,out,dt)</code> | <code>physics_driver.hpp</code> | **唯一主入口**：把 $\Delta t$ 内物理倾向累加到 <code>dyn::PhysicsTendency& out</code> |
| <code>make_physics_driver(g,cfg)</code> | <code>physics_driver.hpp</code> | driver 层唯一构造入口（读 <code>config::ModelConfig</code>） |

**实现要点**

1. 物理过程不改写 <code>dyn::State</code>，只写 <code>dyn::PhysicsTendency</code>；
   例外是跨时间步保持的陆面状态（土壤温度/湿度、表皮温度）与 TKE，
   由 <code>PhysicsDriver</code> 内部持有（<code>surface_states()</code>、<code>tke_storage()</code>）。
2. 所有水物质过程必须正定且质量守恒（列内增量 + 地面降水 = 0，机器精度），
   残差写入 <code>PhysicsDiagnostics::mass_conservation_residual</code>。
3. 所有相变都必须把潜热写回 theta：$\Delta\theta=L\,\Delta q_{\mathrm{cond}}/(c_p\pi)$，
   $\pi=(p/p_0)^{R_d/c_p}$。
4. 每个数值例程上方必须有注释块：公式（LaTeX 形式）+ 离散化 + 文献 + 复杂度。

---

## 2. 物理过程在模式中的位置与时序

### 2.1 依赖方向

~~~text
common <- grid <- dyn <- timeint <- driver
                    <- physics <-
~~~

physics 只依赖 common / grid / dyn（只读状态 + 参考态），不依赖 io 与 da。
物理量在**列**上组织，水平循环由 driver 承担。

### 2.2 单步调用顺序

<code>PhysicsDriver::step</code> 对每个 (i, j) 列依次执行：

| 次序 | 过程 | 输入 | 输出 |
|---|---|---|---|
| 0 | 状态 → 单列 | State, ReferenceState, Grid | PhysicsColumn |
| 1 | **辐射**（按 radiation_cadence） | 廓线、云水/云冰路径、CO2/O3、太阳高度角 | dθ/dt (K/s)、地面/大气顶通量 |
| 2 | **地面层 + 陆面/海面** | 最低层风、温度、湿度、地表状态 | u\*, θ\*, q\*, H, LE, τ, T_g, 土壤廓线 |
| 3 | **边界层**（YSU/MYJ） | K 廓线或 TKE、地面通量 | dθ/dt, dq/dt, du/dt, dv/dt, de/dt |
| 4 | **微物理** | 饱和调整 + 源汇 + 沉降 | dθ/dt, dq_x/dt, 网格降水 |
| 5 | **积云** | plume、CAPE 闭合 | dθ/dt, dq/dt, du/dt, dv/dt, 对流降水 |
| 6 | **次网格动量通量** | Smagorinsky 水平扩散 | du/dt, dv/dt |
| 7 | 单列 → 倾向 | ColumnTendency | dyn::PhysicsTendency（累加） |

顺序的物理理由：辐射先于陆面（地表能量平衡需要向下短波/长波）；
地面层先于边界层（后者需要 $u_*,\theta_*,q_*$ 作下边界通量）；
微物理先于积云（积云的环境场应含网格尺度凝结结果）；
次网格动量通量最后（避免被其他过程的动量倾向干扰）。

### 2.3 时间分裂与"加热率平流"

物理过程用大时间步 $\Delta t$（<code>dt_physics</code>）调用，与声波子步解耦
（[D5] Skamarock & Klemp 2008；[D16] WRF ARW 技术说明第 8 章）。
配置给出的 <code>radiation_cadence</code> 以**秒**为单位，构造时换算为步数
<code>n_rad = max(1, round(radiation_cadence/dt))</code>。
未调用辐射的步上沿用上一次加热率：

$$
\left.\frac{\partial\theta}{\partial t}\right|_{\mathrm{rad}}^{(n)}
=\alpha\,h_{\mathrm{new}}+(1-\alpha)\,h_{\mathrm{old}},
\qquad \alpha=\texttt{radiation\_heating\_relax}
$$

这既实现了辐射加热率的时间平流（避免阶梯式加热），也把辐射调用频率与稳定性解耦。

---

## 3. 微物理

### 3.1 饱和调整（Kessler / Thompson 共用）

在定压、绝热假定下求解凝结-潜热耦合系统

$$
q_v-\delta=q_s\!\left(p,\;T_0+\frac{L}{c_p}\delta\right),
\qquad \Delta\theta=\frac{L\,\delta}{c_p\pi}
$$

用 Newton 迭代求 $\delta$（正为凝结）：

$$
f(\delta)=q_v-\delta-q_s(T(\delta)),\qquad
f'(\delta)=-1-\frac{L}{c_p}\frac{\partial q_s}{\partial T},\qquad
\frac{\partial q_s}{\partial T}=\frac{L\,q_s}{R_vT^{2}}
$$

约束 $\delta\in[-q_c,q_v]$ 保证正定；初值取线性化解
$\delta_0=(q_v-q_s)/(1+L^{2}q_s/(c_pR_vT^{2}))$。文献 [P1]。
复杂度 $O(n_{\mathrm{iter}})\approx O(3)$。

### 3.2 Kessler 暖雨方案 [P1]

水物质：$q_v,q_c,q_r$（忽略冰相，状态中的冰相原样保留）。

| 过程 | 方程 |
|---|---|
| 自动转换 | $P_{\mathrm{auto}}=k_1\max(q_c-q_{c0},0)$，$k_1=10^{-3}\,\mathrm{s^{-1}}$，$q_{c0}=10^{-3}$ |
| 收集（吸积） | $P_{\mathrm{acc}}=k_2q_cq_r$，$k_2=2.2$ |
| 雨落速 | $V_t=a(\rho q_r)^{b}$，$a=36.34$，$b=0.1364$ |
| 雨蒸发 | Maxwell-Mason 群体积分（下） |

**蒸发/凝华的统一形式（[P2] 式 (A1)-(A6)）**

单个粒子相变质量速率

$$
\frac{dm}{dt}=\frac{4\pi r f_v(\rho_v-\rho_{v,s})}
{1+\dfrac{L^{2}D_v\rho_{v,s}}{K_aR_vT^{2}}},
\qquad f_v=0.78+0.308\,Sc^{1/3}Re^{1/2}
$$

对指数谱 $N(D)=N_0e^{-\lambda D}$ 积分（$N_r=N_0/\lambda$，$\bar r=1/(2\lambda)$）：

$$
\left.\frac{dq_x}{dt}\right|_{\mathrm{evap}}
=\frac{N_r}{\rho}\left.\frac{dm}{dt}\right|_{r=\bar r},
\qquad
\lambda=\left(\frac{aN_0\Gamma(1+b)}{\rho q_x}\right)^{1/b}
$$

雨的谱参数：$N_{0r}=8\times10^{6}$ m$^{-4}$，$m(D)=\frac{\pi}{6}\rho_wD^{3}$。

**雨沉降：上游通量法 + 可用质量限制（无条件正定）**

自顶向下传递入流，第 k 层：

$$
\text{available}_k=\rho_kq_k\Delta z_k+F_{\mathrm{in},k}\Delta t,\qquad
F_{\mathrm{out},k}=\min\!\big(\rho_kq_kV_{t,k}\Delta t,\ \text{available}_k\big)
$$

离开列底的通量即地面降水率。该格式 (a) 严格正定；(b) 列内质量守恒；
(c) 落速 CFL $<1$ 时退化为标准一阶上游格式 [B6]。

### 3.3 Thompson 6 类方案 [P3]

水物质：$q_v,q_c,q_r,q_i,q_s,q_g$。粒子谱与落速（[P3] 表 1）

$$
m(D)=aD^{b},\quad V(D)=cD^{d},\quad
\lambda=\left(\frac{aN_0\Gamma(1+b)}{\rho q}\right)^{1/b},\quad
\langle D^{d}\rangle_m=\frac{\Gamma(4+d)}{\Gamma(4)}\lambda^{-d}
$$

| 类别 | $m=aD^{b}$ | $V=cD^{d}$ | $N_0$ (m$^{-4}$) |
|---|---|---|---|
| 雨 | $a=\pi\rho_w/6,\ b=3$ | $4854\,D\,e^{-195D}$ | $8\times10^{6}$ |
| 雪 | $a=0.069,\ b=2$ | $40\,D^{0.55}$ | $2\times10^{7}$ |
| 霰 | $a=\pi\cdot400/6,\ b=3$ | $442\,D^{0.89}$ | $4\times10^{6}$ |
| 云冰 | $a=\pi\cdot890/6,\ b=3$ | — | $10^{7}$ |

落速含空气密度修正 $(\rho_0/\rho)^{0.54}$（[P3] 式 (A14)）。

**过程清单**

1. 云水→雨水自动转换（Berry-Reinhardt 的临界半径判据简化）：
   $$
   r_{vol}=\left(\frac{3\rho q_c}{4\pi\rho_wN_c}\right)^{1/3},\qquad
   P_{ra}=\frac{q_c}{\tau}\max\!\left(0,\,1-\frac{r_{crit}}{r_{vol}}\right)
   $$
   $r_{crit}=10\,\mu\mathrm{m}$（含弱温度依赖），$\tau=10^{3}$ s。
2. 雨水收集云水：$P_{ac}=k_2q_cq_r$。
3. Bergeron 过程（[P3] 式 (A9)）：
   $$
   \beta=c\max\!\left(\frac{q_v}{q_{s,i}}-1,0\right),\qquad
   \Delta q_c=q_c\big(1-e^{-\beta\Delta t}\big),\qquad
   c=5.9\times10^{-3}\ \mathrm{s^{-1}}
   $$
4. 冻结：云水均匀冻结（$T\le233.15$ K，速率 1 s$^{-1}$）；
   云水浸泡冻结 $P=q_c/\tau_f[e^{A'(T_0-T)}-1]$（$\tau_f=10^{4}$ s，$A'=0.66$ K$^{-1}$）；
   雨水 Bigg 冻结
   $$
   P^{r}_{frz}=\frac{\pi^{2}\rho_wB'}{12\rho}
   \big[e^{A'(T_0-T)}-1\big]\int_0^{\infty}\!D^{6}N(D)\,dD,
   \qquad \int D^{6}N\,dD=N_{0r}\Gamma(7)\lambda_r^{-7}
   $$
   $B'=100$ m$^{-3}$ s$^{-1}$。
5. 云冰→雪：$P_{is}=k_{is}\max(q_i-q_{i0},0)$。
6. 雪淞附→霰：$P_{sg}=c_{rim}q_cq_s(\rho_0/\rho)^{1/2}$。
7. 沉积/凝华：Maxwell-Mason 冰面相变（同 3.2，$L\to L_s$、$q_{v,s}\to q_{s,i}$），
   增长受"不使层内过饱和"的潜热-饱和度约束限制。
8. 融化（[P2] 式 (A20)）：
   $$
   \left.\frac{dq_s}{dt}\right|_{\mathrm{melt}}=\frac{N_s}{\rho}\cdot
   \frac{4\pi rf_v\big[K_a(T-T_0)+L_vD_v\rho(q_v-q_{s,0})\big]}{L_f}
   $$
9. 沉降：雨、雪、霰分别用上游通量法（同 3.2）。

**正定与守恒保护**：每个子步后对 $q_x$ 做非负截断，并把截断造成的质量亏缺
按层质量比例归还给 $q_v$，使整列质量严格守恒（残差约 $10^{-15}$）。

### 3.4 与状态量的耦合

微物理就地更新 <code>PhysicsColumn::theta</code> 与水物质，driver 用有限差分给出倾向

$$
\left(\frac{\partial\theta}{\partial t}\right)_{\mathrm{mp}}
=\frac{\theta^{n+1}-\theta^{n}}{\Delta t},\qquad
\left(\frac{\partial q_x}{\partial t}\right)_{\mathrm{mp}}
=\frac{q_x^{n+1}-q_x^{n}}{\Delta t}
$$

---

## 4. 辐射

### 4.1 谱带与相关 k 分布

长波用 RRTM 的 16 个谱带（波数边界 10-3250 cm$^{-1}$），
短波用 RRTMG 的 14 个谱带（820-50000 cm$^{-1}$，对应 12.2-0.2 μm），带数可配置
（[P5][P6]）。带内透过率用相关 k 分布：

$$
T_b(u)=\sum_{g=1}^{N_g}w_g\exp\!\left(-\sum_s k_{g,s}(p,T)\,u_s\right),
\qquad \sum_g w_g=1,\quad \sum_g w_gk_g=\bar k
$$

g 点用 Gauss-Laguerre 求积生成（$\int_0^\infty e^{-k}e^{-ku}dk=1/(1+u)$）：

$$
L_0=1,\ L_1=1-x,\quad (j+1)L_{j+1}=(2j+1-x)L_j-jL_{j-1},\qquad
w_i=\frac{x_i}{(N+1)^{2}[L_{N+1}(x_i)]^{2}}
$$

节点用 Newton 迭代求根（$L_N'(x)=(NL_N-NL_{N-1})/x$），复杂度 $O(n^{2})$。

### 4.2 黑体份额

带发射/吸收需要累积 Planck 函数（无量纲）：

$$
f(\lambda T)=\frac{15}{\pi^{4}}\sum_{n=1}^{\infty}
\frac{e^{-na/(\lambda T)}}{n^{4}}
\left[\frac{a^{3}}{(n\lambda T)^{3}}+\frac{3a^{2}}{(n\lambda T)^{2}}
+\frac{6a}{n\lambda T}+6\right],
\qquad a=\frac{hc}{k}=14388\ \mu\mathrm{m\,K}
$$

数值校验：$f(2898\,\mu\mathrm{m\,K})=0.2495$（Wien 峰值）、
$f(4100)=0.4942$（中值）、$f(\infty)=1$。
长波带份额按 3.08-1000 μm 归一化；短波带份额用太阳黑体 $T_\odot=5776$ K 计算并归一化。

### 4.3 长波：发射率/吸收率累加

逐带逐层计算 $T_{b,k}$ 与带发射 $E_{b,k}=\sigma T_k^{4}f_b(T_k)$，然后

$$
F^{+}_{b,k+1}=T_{b,k}F^{+}_{b,k}+(1-T_{b,k})E_{b,k}\quad(\text{自下而上}),
$$

$$
F^{-}_{b,k}=T_{b,k}F^{-}_{b,k+1}+(1-T_{b,k})E_{b,k}\quad(\text{自上而下}),
$$

地表边界（发射 + 反射）：

$$
F^{+}_{b,0}=\varepsilon\sigma T_g^{4}f_b(T_g)+(1-\varepsilon)F^{-}_{b,0},
\qquad F^{-}_{b,\mathrm{top}}=0
$$

吸收剂质量路径：$u_{\mathrm{H_2O}}=\rho q_v\Delta z$；
$u_{\mathrm{CO_2}}=\rho\cdot6.38\times10^{-4}\Delta z$（420 ppmv）；
$u_{\mathrm{O_3}}$ 由总柱量（DU）按高斯权重（中心 22 km、宽 5 km）分配。

### 4.4 短波：直射 + 单次散射 + 漫射

1. 太阳位置
   $$
   \delta=23.45^\circ\cos\!\frac{2\pi(172-N)}{365},\qquad
   \cos z=\sin\varphi\sin\delta+\cos\varphi\cos\delta\cos h,\qquad
   h=(\mathrm{UTC}-12)\cdot15^\circ+\lambda
   $$
   日地距离修正 $(r_0/r)^{2}=1+0.033\cos(2\pi N/365)$。
2. 直射光束（Beer-Lambert，含倾角）

$$
T_{\mathrm{dir}}=\sum_gw_ge^{-(\tau_{\mathrm{abs},g}+\tau_{\mathrm{sca}})/\mu_0},
\qquad
T_{\mathrm{abs}}=\sum_gw_ge^{-\tau_{\mathrm{abs},g}/\mu_0}
$$

   被散射出光束的量 $F_{\mathrm{dir}}(T_{\mathrm{abs}}-T_{\mathrm{dir}})$，
   按 $(1+g_{\mathrm{eff}})/2$ 分到下半球、$(1-g_{\mathrm{eff}})/2$ 到上半球。
3. 漫射：以扩散因子 $D=1.66$ 衰减，
   $T_{\mathrm{diff}}=\sum_gw_ge^{-D(\tau_{\mathrm{abs},g}+\tau_{\mathrm{sca}})}$，
   源项置于层中心并衰减半层；自顶向下累加得到向下漫射，经地表反射
   $F^{+}_0=\alpha(F_{\mathrm{dir},0}+F^{-}_0)$ 后自下而上累加。
4. 云光学厚度（几何光学，[P6]）

$$
\tau_c=\frac{3\,\mathrm{LWP}}{2\rho_wr_e}+\frac{3\,\mathrm{IWP}}{2\rho_ir_{e,i}},
\qquad\text{按云量线性缩放}
$$

5. 气溶胶：按 550 nm 光学厚度以 $\lambda^{-1.3}$ 分配；
   Rayleigh 用 $k_R=9.7\times10^{-6}(0.55/\lambda)^{4.05}$ m$^{2}$/kg。
6. 夜间分支：$\cos z\le$ 阈值时完全跳过短波（地面与大气顶短波置零），长波照常。

**加热率**（净通量散度）：

$$
\frac{\partial T}{\partial t}=-\frac{1}{\rho c_p}\frac{\partial F_{\mathrm{net}}}{\partial z},
\qquad
\frac{\partial\theta}{\partial t}=\frac{1}{\pi}\frac{\partial T}{\partial t}
$$

短波用净向下约定、长波用净向上约定；实现中统一为一个"净向上"散度函数，
短波调用时交换参数。

---

## 5. 边界层（PBL）

### 5.1 共用的隐式扩散求解器

两个方案最终都化为

$$
\frac{\partial X}{\partial t}=\frac{\partial}{\partial z}\left(K\frac{\partial X}{\partial z}\right)
-\lambda X+S
$$

的全隐式离散（界面通量 + Thomas 三对角）：

$$
\big[1+\Delta t(a_k+b_k+\lambda_k)\big]X_k^{n+1}
-\Delta t\,a_kX_{k-1}^{n+1}-\Delta t\,b_kX_{k+1}^{n+1}
=X_k^{n}+\Delta tS_k,
$$

$$
a_k=\frac{K_{k-1/2}}{\Delta z_k\Delta z_{k-1/2}},\qquad
b_k=\frac{K_{k+1/2}}{\Delta z_k\Delta z_{k+1/2}}
$$

下边界为地面层运动学通量（向上为正）：$H=\rho c_p\overline{w'\theta'}_0$，
故 $\overline{w'\theta'}_0=-u_*\theta_*$；$\overline{w'q'}_0=-u_*q_*$；
$\overline{w'u'}_0=-u_*^{2}u_1/|V_1|$。
全隐式对任意 $\Delta t$、$K\ge0$ 无条件稳定（[B2] 第 3 章）。

### 5.2 YSU（非局地 K 廓线）[P7]

1. PBL 高度（总体 Richardson 数）
   $$
   Ri_b(z)=\frac{gz[\theta_v(z)-\theta_{v,s}]}{\theta_{v,1}|V(z)|^{2}},
   \qquad Ri_b(h)=Ri_c=0.25
   $$
2. 速度尺度与 K 廓线
   $$
   w_*^{3}=\frac{g}{\theta_0}h\,\overline{(w'\theta')}_0,\qquad
   w_s=\left(u_*^{3}+\phi_m\kappa w_*^{3}\frac{z}{h}\right)^{1/3},
   $$
   $$
   K_m=\kappa w_sz\left(1-\frac{z}{h}\right)^{2}\ (z\le h),\qquad
   K_h=\frac{K_m}{Pr},\quad Pr=1+2.1\frac{z}{h}
   $$
   $z>h$ 时 $K=K_\infty+K(h)e^{-(z-h)/(0.1h)}$，$K_\infty=0.1$ m$^{2}$/s。
3. 逆梯度（非局地）项
   $$
   \gamma=C\frac{\overline{(w'\theta')}_0}{w_sh},\qquad C=15
   $$
   以源项形式进入扩散方程：
   $S_\gamma=[K_{k+1/2}\gamma_{k+1/2}-K_{k-1/2}\gamma_{k-1/2}]/\Delta z_k$。
4. 夹卷：PBL 顶向上通量取地面通量的 0.15 倍
   $F_e=0.15\,\overline{(w'\theta')}_0$，以 $\pm F_e/\Delta z$ 施加在 PBL 顶上下相邻层
   （PBL 顶降温、自由大气增温，整列能量守恒）。

### 5.3 MYJ（Mellor-Yamada 2.5 阶 TKE 闭合）[P8][P18]

**TKE 预报方程**（$e=\frac12q^{2}$）

$$
\frac{\partial e}{\partial t}
=\frac{\partial}{\partial z}\left(K_e\frac{\partial e}{\partial z}\right)
+K_mS^{2}-K_h\frac{g}{\theta_0}\frac{\partial\theta_v}{\partial z}
-\frac{2e^{3/2}}{B_1l},
\qquad
S^{2}=\left(\frac{\partial u}{\partial z}\right)^{2}
+\left(\frac{\partial v}{\partial z}\right)^{2}
$$

$$
K_m=l\sqrt{2e}\,S_M,\qquad K_h=l\sqrt{2e}\,S_H,\qquad
K_e=l\sqrt{2e}\,S_q\ (S_q=0.2)
$$

耗散项线性化为 $\lambda=2\sqrt{e^{n}}/(B_1l)$ 后隐式处理（稳定）。

**稳定性函数**（[P8] 式 (35)；采用 [P18] 的非奇异化系数）

$$
G_M=S^{2},\qquad G_H=\frac{g}{\theta_0}\frac{\partial\theta_v}{\partial z}
\ (\text{稳定}>0),\qquad \epsilon=\left(\frac{l}{q}\right)^{2}
$$

$$
D=(A_{DNM}G_M+A_{DNH}G_H)G_H\epsilon^{2}
+(B_{DNM}G_M+B_{DNH}G_H)\epsilon+1
$$

$$
S_M=\frac{B_{SMH}G_H\epsilon+C_{ESM}}{D},\qquad
S_H=\frac{(B_{SHM}G_M+B_{SHH}G_H)\epsilon+C_{ESH}}{D}
$$

其中（$A_1,A_2,B_1,B_2,C_1=0.65989,0.65742,11.87799,7.22697,8.3096\times10^{-4}$，
$B_{TG}=g/273$）：

$$
\begin{aligned}
A_{DNH}&=9A_1A_2^{2}(12A_1+3B_2)B_{TG}^{2},&
A_{DNM}&=18A_1^{2}A_2(B_2-3A_2)B_{TG},\\
B_{DNH}&=3A_2(7A_1+B_2)B_{TG},&
B_{DNM}&=6A_1^{2},\\
B_{SHH}&=9A_1A_2^{2}B_{TG},&
B_{SHM}&=18A_1^{2}A_2C_1,\\
B_{SMH}&=-3A_1A_2(3A_2+3B_2C_1+12A_1C_1-B_2)B_{TG},&
C_{ESH}&=A_2,\ C_{ESM}=A_1(1-3C_1).
\end{aligned}
$$

**混合长**

* 平衡长度 ELM：解 $(l/q)^{2}$ 的二次方程（稳定分支 $z^{2}+B_{UBR}z+A_{UBR}C_{UBR}=0$；
  不稳定分支 $z^{2}+B_{DEN}z+A_{DEN}=0$），取正根后 $ELM=\sqrt{q^{2}/x}$；
  $G_M/G_H\le REQUIRE$ 的"湍流禁区"取 $EL=0.32$ m；求根用数值稳定形式
  （避免 $b>0$ 时的相消）。
* Blackadar 渐近长度
  $$
  EL_0=\alpha\frac{\int q\,z\,dz}{\int q\,dz}\in[EL_{0min},EL_{0max}],\qquad
  \alpha=0.3
  $$
* 廓线：PBL 内 $EL=\min(\kappa z/(1+\kappa z/EL_0),ELM)$，
  PBL 上 $EL=\min(0.23\Delta z,ELM)$，再做一次 1-2-1 平滑。
* 下边界条件（MY 平衡 TKE）：$e(0)\ge\frac12B_1^{2/3}u_*^{2}$。

**PBL 高度**：自顶向下第一个 $e\le e_{\min}\cdot1.01$ 的层。

---

## 6. 地面层与陆面/海面

### 6.1 Monin-Obukhov 相似理论 [P9][P10]

$$
\Phi_m(\zeta)=\begin{cases}(1-16\zeta)^{-1/4}&\zeta<0\\ 1+5\zeta&\zeta\ge0\end{cases},
\qquad
\Phi_h(\zeta)=\begin{cases}(1-16\zeta)^{-1/2}&\zeta<0\\ 1+5\zeta&\zeta\ge0\end{cases}
$$

积分形式（Paulson 1970，$x=(1-16\zeta)^{1/4}$）：

$$
\begin{aligned}
\zeta<0:&\quad \psi_m=2\ln\frac{1+x}{2}+\ln\frac{1+x^{2}}{2}-2\arctan x+\frac{\pi}{2},
&\psi_h=2\ln\frac{1+x^{2}}{2},\\
\zeta\ge0:&\quad \psi_m=\psi_h=-5\zeta.
\end{aligned}
$$

廓线与通量：

$$
\frac{\theta(z)-\theta_s}{\theta_*}=\frac{1}{\kappa}
\left[\ln\frac{z}{z_{0h}}-\psi_h(\zeta)+\psi_h(\zeta_{0h})\right],
\qquad
H=-\rho c_pu_*\theta_*,\quad LE=-\rho L_vu_*q_*,\quad \tau=\rho u_*^{2}
$$

### 6.2 迭代算法伪代码

~~~text
solve(col, sfc):
    z1 = z[0] - terrain ; |V1| = max(sqrt(u1^2+v1^2), 0.1)
    z0 = 陆地: sfc.roughness ; 海面: Charnock 初值 1e-4
    z0h = z0q = max(0.1*z0, 2e-5)
    theta_s = T_g/pi(p_sfc) ; q_s = beta*qsat(p_sfc, T_g)
    u* = kappa*|V1|/ln(z1/z0)                       # 中性初值
    L = +inf ; zeta = 0
    repeat n = 1..max_iter:
        zeta1 = clamp(z1/L, -50, 20)
        u* = kappa*|V1|      / [ln(z1/z0)  - psim(zeta1) + psim(zeta0m)]
        t* = kappa*(th1-ths) / [ln(z1/z0h) - psih(zeta1) + psih(zeta0h)]
        q* = kappa*(q1-qs)   / [ln(z1/z0q) - psih(zeta1) + psih(zeta0q)]
        if 海面: z0 = alpha_c*u*^2/g + 0.11*nu/u* ; z0h = z0q = max(0.1*z0, 2e-5)
        thv* = t*(1+0.608*q1) + 0.608*th1*q*
        if |thv*| <= 1e-12: L = +inf ; converged = true ; break
        L_new = u*^2 / [kappa*(g/th1)*thv*]
        L <- L + relax*(L_new - L)                  # relax = 0.5 (n<=4), 1.0 (n>4)
        if |dL|/|L| <= tol: converged = true ; break
    H = -rho1*cp*u**t* ; LE = -rho1*Lv*u**q* ; tau = rho1*u*^2
    Cd = u*^2/|V1|^2 ; Ch = kappa*u*/den_h ; Cq = kappa*u*/den_q
    2m/10m 诊断：按同样的相似廓线外推
~~~

### 6.3 陆面（五层，Noilhan-Planton 简化）[P14][P15]

**表皮能量平衡**（Newton 求 $T_g$）

$$
R_n=H+LE+G,\qquad
R_n=(1-a)S^{-}+\varepsilon(L^{-}-\sigma T_g^{4}),
$$

$$
H=\rho c_pC_h|V|\left(\frac{T_g}{\pi_s}-\theta_1\right),\qquad
LE=\rho L_vC_q|V|\big(\beta q_{sat}(T_g)-q_1\big),\qquad
G=\lambda_s\frac{T_g-T_{s,1}}{0.5\Delta z_1}
$$

$$
\frac{dR_n}{dT_g}=-4\varepsilon\sigma T_g^{3},\qquad
\frac{dH}{dT_g}=\frac{\rho c_pC_h|V|}{\pi_s},\qquad
\frac{dLE}{dT_g}=\rho L_vC_q|V|\beta\frac{dq_{sat}}{dT},\qquad
\frac{dG}{dT_g}=\frac{\lambda_s}{0.5\Delta z_1}
$$

**土壤温度/湿度**（上边界通量、下边界 Dirichlet，全隐式三对角）

$$
C_s\frac{\partial T}{\partial t}
=\frac{\partial}{\partial z}\left(\lambda\frac{\partial T}{\partial z}\right),
\qquad
\frac{\partial w}{\partial t}
=\frac{\partial}{\partial z}\left(D_w\frac{\partial w}{\partial z}\right)
$$

土壤湿度对蒸发的限制：$\beta=\min(1,w_1/w_{fc})$（低于凋萎点时指数衰减）。

### 6.4 海面

固定 SST、反照率 $a_{sea}=0.06$、发射率 0.98；Charnock 粗糙度

$$
z_0=\frac{\alpha_cu_*^{2}}{g}+0.11\frac{\nu}{u_*},\qquad \alpha_c=0.018
$$

---

## 7. 积云对流（Kain-Fritsch [P11]）

### 7.1 触发

在最低 3 km 内取相当位温最大的层为源层；抬升气块（干绝热 + 湿绝热）后要求

$$
\mathrm{CAPE}\ge\mathrm{CAPE}_{\min}=1000\ \mathrm{J/kg},\qquad
\text{存在 LFC},\qquad z_{top}-z_{base}\ge4\ \mathrm{km}
$$

### 7.2 卷入/卷出 plume

$$
\frac{1}{m}\frac{\partial m}{\partial z}=\epsilon(z)-\delta(z),
\qquad m_k=m_{k-1}e^{(\epsilon-\delta)\Delta z}
$$

云底以上沿饱和湿绝热线（$\theta_e$ 守恒，Newton 反解 T）：

$$
F(T)=\frac{T}{\pi}\exp\!\left(\frac{L_sq_{s,i}(T)}{c_pT}\right)-\theta_e=0
$$

卷入按质量加权混合 $\theta_e$；浮力 $b=g(T_{v,p}-T_{v,e})/T_{v,e}<0$ 时按 $\delta$ 卷出，
连续两层负浮力即确定云顶。卷入率 $\epsilon=3\times10^{-4}$ m$^{-1}$、
卷出率 $\delta=1.5\times10^{-4}$ m$^{-1}$（见参数表与第 10 节）。

### 7.3 CAPE 消耗闭合

$$
\Sigma=\frac{1}{z_t-z_b}\int_{z_b}^{z_t}\frac{g}{c_p\theta_{v,e}}
\left|\frac{\partial s_v}{\partial z}\right|dz,\qquad
s_v=c_pT+gz+L_vq_v,
$$

$$
m_b=\frac{\rho_b\,\mathrm{CAPE}}{\tau_c\,\Sigma}
\qquad[\mathrm{kg\,m^{-2}s^{-1}}]
$$

量纲校验：
$\rho[\mathrm{kg\,m^{-3}}]\times\mathrm{CAPE}[\mathrm{m^{2}s^{-2}}]
/(\tau[\mathrm{s}]\times\Sigma[\mathrm{s^{-2}}])=\mathrm{kg\,m^{-2}s^{-1}}$。

### 7.4 环境响应与降水

$$
\left.\frac{\partial\theta}{\partial t}\right|_{cu}
=-\frac{1}{\rho}m_b\frac{\partial\hat\theta}{\partial z}
+\frac{\Delta m_k(\hat\theta_k-\bar\theta_k)}{\rho_k\Delta z_k}
+\text{(云下层补偿下沉)}
$$

降水效率：浅对流 0.2；深对流按陆/海取值（0.5 / 0.7）。
水物质收支的精确闭合：设整列水汽净减少率为 $L$（kg m$^{-2}$ s$^{-1}$），则

$$
P=\varepsilon_pL,\qquad
E_{re}=(1-\varepsilon_p)L\ \text{（下沉气流再蒸发，返回云下层并降温）}
$$

于是 $\int\rho\,\partial_tq\,dz+P=0$ 严格成立，残差写入诊断。

### 7.5 Grell-Devenyi 集合框架 [P12]

对 N 个成员施加确定性参数扰动（卷入率、卷出率、闭合时间、降水效率、CAPE 阈值），
每个成员用 KF 积分，最后等权重平均倾向与降水：

$$
\bar T=\frac{1}{N}\sum_{i=1}^{N}T_i,\qquad
\bar P=\frac{1}{N}\sum_{i=1}^{N}P_i
$$

成员数由 <code>gd_ensemble_size</code> 控制（上限 32）。

---

## 8. 与动力学的时间分裂耦合、稳定性与时间步约束

### 8.1 时间分裂位置

~~~text
for n = 0 .. N-1:                               # 大时间步 dt
    tend = 0
    equations.tendencies(state, tend)           # 平流/科氏/阻尼
    physics.step(state, grid, ref, tend, dt)    # 本模块：累加物理倾向
    timeint.integrator.step(state)              # RK3 + 声波子步 / 半隐式 Helmholtz
~~~

物理倾向只进入大时间步右端项，声波子步内不重复调用（[D5][D16]）。
半隐式格式下物理倾向与线性声波-重力波项线性叠加，不改变 Helmholtz 算子系数
（物理倾向在声波子步内保持常数）。

### 8.2 数值稳定性

| 过程 | 格式 | 稳定性条件 |
|---|---|---|
| 微物理源汇（转换/收集/蒸发） | 显式向前欧拉 | $\Delta t\le1/\max(P/q)$，正定截断兜底 |
| 饱和调整 | Newton 隐式 | 无条件（受 $\delta$ 边界约束） |
| 粒子沉降 | 上游通量 + 可用质量限制 | 无条件正定（CFL>1 时降为一阶） |
| 垂直扩散（PBL/土壤） | 全隐式三对角 | 无条件稳定 |
| 辐射加热 | 显式（K/s） | 由 <code>max_heating_rate</code> 限幅 |
| 陆面表皮温度 | Newton 迭代 | 步长限幅兜底 |

诊断量：

$$
C_h=\frac{\max(|u|,|v|)\Delta t}{\Delta x},\qquad
C_f=\frac{V_{t,\max}\Delta t}{\min\Delta z},\qquad
C_d=\frac{2K_{\max}\Delta t}{(\min\Delta z)^{2}}
$$

$C_h$ 受大时间步 CFL 约束（<code>cfl_limit</code>）；
$C_f$、$C_d$ 受 <code>vertical_cfl_limit</code> 约束，
超限时 <code>PhysicsDriver::stability_report()</code> 给出警告。

### 8.3 正定与守恒的强制手段

1. 相变速率乘 $\Delta t$ 后取 $\min(\text{rate}\cdot\Delta t,\text{available})$；
2. 沉降用可用质量限制；
3. Thompson 每步末尾把负值截断并按质量比例归还水汽；
4. 微物理的地面降水计入整列守恒检查；
5. 积云用 $\varepsilon_p$ 与再蒸发的互补关系保证精确闭合。

---

## 9. 参数表

### 9.1 时间与网格

| 参数 | 默认 | 单位 | 说明 |
|---|---|---|---|
| <code>dt_physics</code> | 10 | s | 物理调用步长 |
| <code>radiation_cadence</code> | 1 | 步 | 辐射调用间隔（配置以秒给出，构造时换算） |
| <code>radiation_heating_relax</code> | 1.0 | — | 新加热率权重（<1 时做时间平滑） |
| <code>pbl_cadence</code>, <code>microphysics_cadence</code> | 1 | 步 | 调用间隔（预留） |
| <code>max_heating_rate</code> | 0.02 | K/s | 加热率限幅 |
| <code>enable_tendency_physics</code> | true | — | 关闭时只做诊断、不写倾向 |

### 9.2 微物理

| 参数 | 默认 | 单位 | 说明 |
|---|---|---|---|
| <code>kessler_qc0</code> | 1.0e-3 | kg/kg | 自动转换阈值 |
| <code>kessler_k1</code> | 1.0e-3 | 1/s | 自动转换率 [P1] |
| <code>kessler_k2</code> | 2.2 | — | 雨收集云水系数 [P1] |
| <code>kessler_rain_a</code> / <code>_b</code> | 36.34 / 0.1364 | — | 雨落速 $V_t=a(\rho q_r)^{b}$ |
| <code>thompson_nc/nr/ni/ns/ng</code> | 1e8/1e6/1e5/3e4/4e4 | m$^{-3}$ | 各类粒子数浓度 |
| <code>thompson_bergeron_rate</code> | 5.9e-3 | 1/s | Bergeron 系数 c [P3] |
| <code>thompson_bigg_a</code> / <code>_b</code> | 0.66 / 100 | 1/K, m$^{-3}$s$^{-1}$ | Bigg 冻结系数 |
| <code>thompson_autoconv_time</code> | 1000 | s | 云水→雨水松弛时间 |
| <code>thompson_autoconv_radius_um</code> | 10 | μm | 临界体积平均半径 |
| <code>thompson_riming_coef</code> | 0.5 | — | 雪淞附系数 |

### 9.3 辐射

| 参数 | 默认 | 单位 | 说明 |
|---|---|---|---|
| <code>n_longwave_bands</code> | 16 | — | RRTM 长波谱带 [P5] |
| <code>n_shortwave_bands</code> | 14 | — | RRTMG 短波谱带 [P6] |
| <code>n_g_points</code> | 4 | — | 相关 k 分布求积点数 |
| <code>co2_ppm</code> | 420 | ppmv | CO2 浓度 |
| <code>o3_column_du</code> | 300 | DU | 臭氧柱量 |
| <code>aerosol_optical_depth</code> | 0.10 | — | 550 nm 气溶胶光学厚度 |
| <code>cloud_droplet_radius_um</code> / <code>cloud_ice_radius_um</code> | 10 / 30 | μm | 有效半径 |
| <code>solar_constant</code> | 1361 | W/m$^2$ | 太阳常数 |
| <code>radiation_min_cos_zenith</code> | 0.01 | — | 夜间判据 |

> 注：各谱带吸收系数由谱带中心波长的解析式生成，是**教学用简化参数化**
> （量级与 RRTM 各带一致，但不是业务系数表）。替换为正式 RRTM 系数表时只需替换
> <code>radiation_rrtmg.cpp</code> 中的 <code>kbar_*</code> 函数。

### 9.4 边界层与地面层

| 参数 | 默认 | 单位 | 说明 |
|---|---|---|---|
| <code>ysu_ri_critical</code> | 0.25 | — | PBL 顶临界 Richardson 数 [P7] |
| <code>ysu_entrainment_coef</code> | 0.15 | — | 夹卷通量比 [P7] |
| <code>ysu_prandtl_coef</code> | 2.1 | — | $Pr=1+c\,z/h$ |
| <code>ysu_countergradient_coef</code> | 15 | — | 逆梯度系数 C |
| <code>my.*</code> | 0.65989, 0.65742, 11.87799, 7.22697, 8.31e-4 | — | MY 闭合常数 [P18]；<code>original_my82()</code> 给出 0.92, 0.74, 16.6, 10.1, 0.08 |
| <code>myj_el0max</code> / <code>myj_el0min</code> | 1000 / 1 | m | Blackadar 渐近混合长限幅 |
| <code>myj_alph</code> | 0.30 | — | 渐近混合长系数 |
| <code>myj_e_min</code> | 1e-4 | m$^2$/s$^2$ | TKE 下限 |
| <code>mo_max_iterations</code> / <code>mo_tolerance</code> | 40 / 1e-8 | — | M-O 迭代上限与收敛判据 |
| <code>charnock_alpha</code> | 0.018 | — | Charnock 系数 |
| <code>soil_thermal_conductivity</code> | 1.0 | W/(m K) | 土壤导热率 |
| <code>soil_heat_capacity</code> | 2.0e6 | J/(m$^3$K) | 土壤热容量 |
| <code>soil_hydraulic_diffusivity</code> | 2.0e-7 | m$^2$/s | 土壤水分扩散率 |
| <code>soil_field_capacity</code> / <code>soil_wilting_point</code> | 0.3 / 0.1 | m$^3$/m$^3$ | 田间持水量 / 凋萎点 |

### 9.5 积云

| 参数 | 默认 | 单位 | 说明 |
|---|---|---|---|
| <code>kf_entrainment_rate</code> | 3.0e-4 | 1/m | 卷入率（见第 10 节单位说明） |
| <code>kf_detrainment_rate</code> | 1.5e-4 | 1/m | 卷出率 |
| <code>kf_min_cape</code> | 1000 | J/kg | 触发所需最小 CAPE |
| <code>kf_min_cloud_depth</code> / <code>kf_max_cloud_depth</code> | 4000 / 16000 | m | 云厚判据 |
| <code>kf_closure_time</code> | 1800 | s | CAPE 消耗时间尺度 $\tau_c$ |
| <code>kf_precip_efficiency_land</code> / <code>_sea</code> | 0.5 / 0.7 | — | 降水效率 |
| <code>gd_ensemble_size</code> / <code>gd_spread</code> | 12 / 0.7 | — | GD 成员数与扰动幅度 |

### 9.6 次网格动量与稳定

| 参数 | 默认 | 单位 | 说明 |
|---|---|---|---|
| <code>use_smagorinsky</code> | true | — | 水平次网格动量通量开关 [P16] |
| <code>smagorinsky_coef</code> | 0.20 | — | Smagorinsky 常数 |
| <code>cfl_limit</code> / <code>vertical_cfl_limit</code> | 0.5 / 0.9 | — | CFL 限值（诊断） |

---

## 10. 已知简化与扩展点

1. **微物理**：Kessler 为暖雨（忽略冰相）；Thompson 的云水自动转换使用临界半径松弛式
   而非 [P3] 的谱宽公式；Morrison/WSM6 仅保留枚举与工厂接口（抛 NotImplemented）。
2. **辐射**：长波用发射率/吸收率累加；短波用"直射 + 单次散射 + 漫射衰减"，
   未做完整多次散射两流求解；带吸收系数为教学用简化解析式。
   扩展点：把 <code>RadiationBand</code> 的 <code>k_h2o/k_co2/k_o3</code> 换成 RRTM 系数表，
   并在 <code>shortwave_column</code> 内换成 delta-Eddington 两流（可逐 g 点求解）。
3. **PBL**：YSU 的夹卷与逆梯度项为简化实现；MYJ 用层中心差分（原文为界面差分），
   混合长只做一次平滑。
4. **积云**：GD 框架以 KF 为唯一成员方案；下沉气流以"再蒸发 + 云下层冷却"的积分形式
   表达，未显式积分下沉气块方程；[P11] 的卷入率以 0.03 的无量纲形式给出，
   本实现统一为 1/m（$3\times10^{-4}$ m$^{-1}$，对应每公里约 35% 的质量增加）。
5. **陆面**：五层土壤为均匀热力参数；未包含植被/雪盖/冻土过程。
6. **动量倾向的错位**：物理在单元中心计算，u/v 倾向按列写入对应的 FaceX/FaceY 点
   （等价于 WRF 物理包直接作用于错位点的做法）；如需严格面心插值，可在
   <code>PhysicsDriver::scatter</code> 中做两列平均。
7. **TKE 持久化**：<code>dyn::State</code> 没有 TKE 分量，MYJ 的 $e$ 由
   <code>PhysicsDriver</code> 内部保存（<code>tke_storage()</code>）；
   4D-Var 的切线性/伴随若需要 TKE 控制变量，必须把它一并纳入控制向量。

---

## 11. 单元测试对照

<code>tests/unit/test_physics.cpp</code> 用 <code>VIBE_TEST</code> 覆盖 30 个用例：

| 用例 | 检验内容 |
|---|---|
| <code>physics_saturation_vapor_pressure_bolton_known_values</code> | $e_s(0°C)=611.2$ Pa、$e_s(20°C)\approx2337$ Pa、冰面 < 液面 |
| <code>physics_saturation_mixing_ratio_and_rh</code> | 饱和混合比量级、RH=1、露点自洽 |
| <code>physics_exner_and_virtual_temperature</code> | Exner 往返、虚温符号 |
| <code>physics_ideal_sounding_hydrostatic_and_finite</code> | 静力平衡 $\partial p/\partial z=-\rho g$ |
| <code>physics_column_water_paths_and_cloud_fraction</code> | 水路径与云量范围 |
| <code>physics_saturation_adjust_condenses_to_rh_one</code> | 凝结后 RH=1、守恒、增温、迭代次数 |
| <code>physics_saturation_adjust_evaporates_cloud</code> | 云水蒸发、守恒、不超饱和 |
| <code>physics_kessler_no_cloud_no_change</code> | 无云时无变化、无降水 |
| <code>physics_kessler_autoconversion_threshold</code> | $q_{c0}$ 阈值行为 |
| <code>physics_kessler_mass_conservation_and_positivity</code> | 整列守恒（含降水）与正定 |
| <code>physics_kessler_rain_sedimentation_reaches_ground</code> | 雨沉降达地面、落速单调 |
| <code>physics_thompson_mass_conservation</code> | 6 类方案守恒与正定 |
| <code>physics_thompson_bergeron_ice_production</code> | 冷云 Bergeron/凝华生成云冰 |
| <code>physics_thompson_extreme_values_stay_positive</code> | 极端水物质下正定与有限性 |
| <code>physics_monin_obukhov_neutral_log_profile</code> | 中性对数风廓线、$L\to\infty$、$C_d=\kappa^{2}/\ln^{2}$ |
| <code>physics_monin_obukhov_iteration_converges</code> | 迭代收敛、稳定/不稳定符号、$\psi$ 符号 |
| <code>physics_surface_energy_balance_closes</code> | 能量平衡残差 < 1e-6 W/m$^2$、与土壤热通量一致 |
| <code>physics_sea_surface_charnock</code> | Charnock 粗糙度随风速增大 |
| <code>physics_ysu_pbl_height_and_diffusivity</code> | PBL 高度、K 廓线形状、逆梯度项 |
| <code>physics_myj_tke_positive_and_pbl_height</code> | TKE 正定有界、稳定性函数随稳定度减小、混合长 |
| <code>physics_cumulus_kf_no_trigger_in_stable_profile</code> | 稳定层结不触发 |
| <code>physics_cumulus_kf_trigger_and_water_budget</code> | 触发、降水、水物质收支闭合 |
| <code>physics_grell_devenyi_ensemble_framework</code> | 成员扰动与集合平均 |
| <code>physics_radiation_planck_and_bands</code> | Planck 已知值、谱带份额归一、Gauss-Laguerre 精度 |
| <code>physics_radiation_night_shortwave_zero</code> | 夜间短波为 0、长波冷却、OLR 量级 |
| <code>physics_radiation_daytime_and_solar_geometry</code> | 白天短波收支、太阳几何、云光学厚度 |
| <code>physics_factories_and_config_mapping</code> | 工厂/字符串解析/配置映射（cadence 换算） |
| <code>physics_cfl_diagnostics</code> | CFL 计算与边界情形 |
| <code>physics_diagnostics_aggregate</code> | 域平均、最大值与 reset |
| （工厂缺省路径） | <code>make_physics_driver</code> 在无方案时返回空实现（driver 级集成） |

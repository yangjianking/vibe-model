# 教程 · 理想试验清单

每个理想试验都有明确的文献出处、解析或参考解，以及推荐的网格与时间步配置。
初值实现在 [initial_conditions.cpp](../../src/driver/initial_conditions.cpp)。

---

## 1. 暖泡对流（warm_bubble）

**出处**：[D1] Klemp & Wilhelmson (1978)。

**初值**：静止等温大气 + 位温扰动

$$\theta'(x,y,z) = \Delta\theta\cos^2\!\left(\frac{\pi}{2}\,r\right),\quad
r^2 = \left(\frac{x-x_c}{r_x}\right)^2+\left(\frac{y-y_c}{r_y}\right)^2+\left(\frac{z-z_c}{r_z}\right)^2 .$$

**推荐配置**：$\Delta x = \Delta y = 2$ km，$n_z = 60$，$H = 22$ km，$\Delta t = 6$ s，
声波子步 6，$\Delta\theta = 2$ K，$r_x = r_y = 2$ km，$r_z = 2$ km，$z_c = 2$ km，周期边界。

**检验要点**：上升速度峰值约 15–25 m/s；对流单体在 $t \approx 20$ min 达到中性浮力层；
总质量守恒（无物理过程时）到 $10^{-8}$ 相对量级。

---

## 2. 冷泡 / 重力流（density_current）

**出处**：LeVeque (2002) 第 4 章经典基准 [B6]；也见 [D1] 的冷池试验。

**初值**：等温层结 + 中心 $-15$ K 的冷池，半径 4 km。

**推荐配置**：$\Delta x = 50$–100 m（二维剖面），$\Delta t = 0.5$ s。

**检验要点**：冷池前缘传播速度接近理论值 $\sqrt{g\,H\,\Delta\theta/\theta_0}$；
**双惩罚问题**的典型试验：位置误差会同时惩罚空报与漏报（见
[../design/09_verification.md](../design/09_verification.md)）。

---

## 3. 山波（mountain_wave）

**出处**：[D2] Gal-Chen & Somerville (1975)；[D9] Schär et al. (2002)；[D10] Klemp (2011)。

**初值**：均匀稳定层结 + 定常来流（$U = 10$ m/s，$N = 0.01$ s$^{-1}$）+ Agnesi 或余弦山。

**推荐配置**：$\Delta x = 1$ km，$\Delta z \approx 200$ m，$\Delta t = 6$ s，开放边界，
松弛区宽度约为域宽的 10%。

**检验要点**：稳态山波解可与线性理论解对比（$w'$ 的振幅与相位）；
陡地形（$h_0/a > 0.5$）时检验非线性破碎与波阻；
**该试验最能体现 HEVI/半隐式的价值**——细的垂直网格使显式声波子步数剧增。

---

## 4. 惯性重力波（inertia_gravity_wave）

**出处**：[T5] Skamarock & Klemp (2002/2004) 系列的惯性重力波试验。

**初值**：三维高斯位温扰动，在 $f$ 平面上激发惯性重力波。

**推荐配置**：$\Delta x = 2$ km，$\Delta t = 6$ s，周期边界。

**检验要点**：波的传播方向与频率满足离散色散关系
$\omega^2 = f^2 + N^2 k_h^2/(k_h^2 + m^2)$；能量在波列中近似守恒。

---

## 5. 上升热泡（rising_thermal）

**出处**：[T5] Wicker & Skamarock (2002)。

**初值**：截断余弦热泡（同暖泡，但常用更强的截断以产生陡梯度）。

**用途**：检验高阶平流格式（四阶/六阶/WENO5）在陡梯度下的表现。

**推荐配置**：$\Delta x = 125$ m，$\Delta t = 1$ s，对比 central2 / central4 / weno5
在同一时刻的温度场（WENO5 应显著抑制过冲）。

---

## 6. 斜压波（baroclinic_wave）

**出处**：Jablonowski & Williamson (2006) 的斜压不稳定试验（本实现为其 $f$ 平面简化版本）。

**初值**：解析的急流廓线 + 局地位温扰动，随后由斜压不稳定自然发展。

**推荐配置**：$\Delta x = 100$ km 级，积分 10–20 天，$\beta$ 平面。

**检验要点**：扰动能量在约第 8 天达到峰值；位涡场的精细结构与参考解的比对。

---

## 7. 静止等温大气（resting_isothermal）

**出处**：[T2] Tapp & White (1976)；所有声波相关的稳定性测试。

**初值**：所有动量为零，气压与密度严格静力平衡。

**用途**：
1. 检验模式能否保持静止解（差分格式的"静水平衡保持性"）；
2. 检验半隐式与分裂显式在**远超显式稳定极限**的时间步下是否稳定；
3. 声波脉冲传播试验的基底态。

**推荐配置**：$\Delta t = 10$–100 s，对比两种积分器。

**预期结果**：速度扰动的最大幅值应保持在 $10^{-10}$ m/s 量级（机器精度）。

---

## 8. 平衡急流（balanced_jet）

**出处**：[N8] Harris & Lin (2013)。

**初值**：高斯型急流 $u(y) = U_{\max}\exp[-(y-y_0)^2/w^2]$，由热成风关系
$f\,\partial u/\partial y = -(g/\theta_0)\,\partial\theta/\partial y$ 给出位温场。

**用途**：检验地转平衡的保持能力与嵌套边界的质量（把急流放在子域边界附近）。

---

## 9. 从文件初始化（from_file）

**用途**：真实个例。需要提供 NetCDF/GRIB 三维场（$u,v,w,\rho,\theta,\pi,q_v,q_c,q_r,q_i,q_s,q_g$）。
读入后由 `enforce_state_equation` 做静力与状态方程的自洽化调整。

**边界**：可由粗分辨率模式输出（或再分析）经垂直插值后提供侧边界，配 Davies 松弛。

---

## 10. 试验矩阵建议

| 试验 | 网格 | $\Delta t$ | 声波子步 | 平流格式 | 主要检验对象 |
|---|---|---|---|---|---|
| warm_bubble | 128x64x60, 2 km | 6 | 6 | central2 | 动力学核心 + 时间推进 |
| warm_bubble（对比） | 同上 | 12 | 1 次 Helmholtz | weno5 | 半隐式 |
| density_current | 400x1x100, 0.05 km | 0.5 | 4 | central4 | 陡梯度、双惩罚 |
| mountain_wave | 200x100x80, 1 km | 6 | 8 | central4 | 陡地形、HEVI |
| inertia_gravity_wave | 128x128x40, 2 km | 6 | 6 | central6 | 色散关系 |
| baroclinic_wave | 256x128x40, 100 km | 900 | 1（半隐式） | central4 | 长时间积分、嵌套 |

每个试验都应在 [../design/09_verification.md](../design/09_verification.md) 定义的评分框架下
与参考解或观测做定量对比。

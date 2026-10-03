# examples/ —— 嵌入示例

| 示例 | 说明 |
|---|---|
| [embed_model.cpp](embed_model.cpp) | 把 VIBE-Model 当作库使用的最小可运行程序：手工装配网格、参考态、状态、方程与积分器，然后积分并输出能量守恒诊断 |

## 为什么需要这个示例

命令行驱动（`vibe_model`）把"读配置 -> 建网格 -> 造初值 -> 跑积分 -> 写输出"全部封装了，
掩盖了模块之间的真实依赖顺序。嵌入示例暴露这条链路，适合：

* 在自己的程序里调用模式（参数扫描、集合成员、耦合、教学演示）；
* 调试时替换某一个组件（例如换成自定义的积分器或物理方案）；
* 理解"哪些对象必须活到积分结束"（ReferenceState、Equations、State 的生命周期）。

## 运行

~~~bash
cmake --build build --target vibe_embed_example
./build/examples/vibe_embed_example 100      # 100 个时间步
~~~

典型输出：

~~~text
格点 embed: 全局 64x64x48, 本地 64x64x48, 分解 1x1 (rank 0), 平坦地形, dx=2000 ...
参考态：theta0 平均 300 K, 地面气压 100000 Pa, 顶层气压 ... Pa, 静力残差(相对) ...
初始：能量收支: KE=... IE=... PE=... LE=... 总计=... J, 总质量=... kg
  step   10  t =     60.0 s  w_max =    1.234 m/s  theta_min = 299.100 K
  ...
结束：能量收支: ...
总能量相对变化 = ...
总质量相对变化 = ...
~~~

## 嵌入时的注意点

1. **生命周期**：`ReferenceState` 必须比 `State` 与 `Equations` 活得更久
   （`State` 只持有它的裸指针）。
2. **并行**：若启用了 MPI，需要在 `main` 开头调用
   `common::Comm::initialize(&argc, &argv)`，并用真实 `rank` 构造
   `Decomposition`；halo 交换用 `grid::HaloExchange`。
3. **初值**：`driver::initialize_state` 提供全部理想试验；真实个例可自行调用 `io::read_state`。
4. **物理**：把 `physics::make_physics_driver` 的返回值通过
   `integrator.set_physics(ptr.get())` 注入；积分器不拥有它。
5. **时间步**：用 `timeint::TimeStepController` 做自适应与稳定性监控。

# VIBE-Model

本人的专业是计算机软件，没有气象背景，但在气象系统工作多年。
此项目的目的不是为了开发一个真正的数值天气预报模式，而只是为了验证一个问题：
基于 AI-Agent 和合适的大语言模式，可以生成一个数值天气预报的模式框架。
此问题我在三四月份曾经提出过，并在七月二十九日单位群里面明确说出它的可能性。
可惜一直没有回应，所以最终还是由本人亲自来实现。

——————对传统的气象科研方法来说，这是一个斯普特尼克时刻。

**提示词
在 vibe-model 下交付一个结构清晰、可学习理解的气象数值预报模式项目：
核心模式（含嵌套/可变分辨率网格、半隐式时间推进、GPU 内核与混合精度抽象）、
4D-Var 同化（含明确观测接口）、
切线性伴随技术、
提供误差检验与评估工具、
Python 后处理包、
配置文件、测试与构建系统，
以及带理论文献出处的详细设计文档；
与密集计算有关的地方，优先考虑C++和C语言，
运行环境基于传统的高性能并行计算机上。
只需要代码和文档输出，不需要编译和运行测试。


**可嵌套 · 变分辨率 · 半隐式 · GPU 混合精度** 的数值天气预报模式（教学与科研用）。

VIBE-Model 是一个"从方程到产品"完整链路的开源 NWP 模式：非静力可压动力核心、
双向嵌套与平滑变分辨率网格、RK3/声波子步与半隐式两条时间推进路线、
增量 4D-Var 同化与手写切线性/伴随、物理参数化、误差检验与评估、
以及独立的 Python 后处理包。所有数值过程都在设计文档里给出了公式与文献出处。

---

## 目录

| 内容 | 位置 |
|---|---|
| 接口基线（写代码前必读） | [docs/design/00_architecture.md](docs/design/00_architecture.md) |
| 设计文档 | [docs/design/](docs/design/) |
| 上手教程 | [docs/tutorial/first_run.md](docs/tutorial/first_run.md) |
| 理想试验清单 | [docs/tutorial/idealized_cases.md](docs/tutorial/idealized_cases.md) |
| 文献出处总表 | [docs/design/references.md](docs/design/references.md) |
| 配置示例 | [config/](config/) |
| C++ 源码 | [src/](src/) 与 [include/vibe/](include/vibe/) |
| Python 后处理包 | [python/vibe_post/](python/vibe_post/) |
| 测试 | [tests/](tests/) |

---

## 特性一览

**动力核心**
- 全可压非静力 Euler 方程，地形追随高度坐标 [D1][D2]
- Arakawa C-grid + Lorenz 垂直错位，离散动能/能量守恒 [D3][D4]
- 2/4/6 阶中心差分与 5 阶 WENO 平流 [D7][D8]
- 静力平衡参考态分离，线性声波-重力波系数离线构造 [D1]

**网格系统**
- MPI 二维域分解，三阶段 halo 交换（含对角）
- 固定比 r = 3 双向嵌套 + Davies 侧边界松弛 [N1][N3][N4]
- 平滑变分辨率网格，与嵌套共用同一套度量项接口 [N5][N6]
- 地形平滑与陡坡处理选项 [D9][D10][D11]

**时间推进**
- RK3（SSP 低存储）+ 前向-后向声波子步 [T5][D6]
- HEVI 垂直隐式（三对角）与半隐式全隐式（3D Helmholtz）[T2][T6][T7]
- 垂直三对角直接法、BiCGSTAB、多重网格预条件 [T10][T12][T13]
- 自适应时间步与稳定性监控

**GPU 与混合精度**
- CPU / CUDA / HIP / SYCL 四后端统一抽象，业务代码零内核启动语法
- 状态 FP64、计算 FP32、归约补偿求和（Kahan/Neumaier）[G2][G3][G4]
- Helmholtz 迭代精化（FP32 求解 + FP64 残差）[G5]
- 半精度类型与误差预算工具，不依赖 cuda_fp16.h [G1][G6][G7]

**资料同化**
- 增量 4D-Var，内/外循环与分辨率递进 [V3][V10]
- B 矩阵：扩散型水平相关 [V7] + 垂直 EOF [V9] + 多变量平衡 [V8]
- 观测算子带完整 TL/AD：探空、地面、飞机、AMV、散射计、GNSS 掩星、
  卫星辐射率（含快速辐射传输与解析雅可比）、雷达反射率 [O1]–[O8]
- 手写伴随的构造规范与点积检验 [A1][A3][A4]
- 变分偏差订正 [V16][V17] 与 Desroziers 诊断 [V18]
- L-BFGS / 非线性 CG / Lanczos 极小化 [V13][B8]

**物理参数化**
- 微物理：Kessler 暖雨、Thompson 六类、Morrison 双参数 [P1][P3][P4]
- 辐射：RRTM 相关 k 分布（长波 + 短波）[P5][P6]
- 边界层：YSU 非局地、Mellor-Yamada 2.5 阶 TKE [P7][P8]
- 地面层：Monin-Obukhov 相似理论 [P9][P10]
- 陆面：Noilhan-Planton 五层土壤 [P14][P15]
- 积云：Kain-Fritsch 质量通量 [P11][P12]

**检验与评估**
- 确定性：Bias / MAE / RMSE / 相关 / 距平相关
- 分类：POD / FAR / CSI / ETS / TSS / ORSS / SEDI [E4][E5][E6]
- 概率：Brier（含分解）/ BSS / CRPS / ROC-AUC / 可靠性图 / 秩直方图 [E3][E9][E12][E15]
- 空间：FSS 与多尺度 FSS、模糊检验、双惩罚诊断 [E7][E8][E14]
- 流式累加器，支持 MPI 归约与日月聚合

**C 与 C++ 分工**
- 框架与算法组织用 C++20；紧循环用纯 C11，集中在 include/vibe/common/c_kernels.h
  （Thomas 三对角、WENO5 重构、双线性插值、CFL 扫描、Neumaier 补偿求和、通量平流）
- C 部分 ABI 稳定，可被 FORTRAN / Python(ctypes) / Julia 直接调用；同一公式在 C 与 C++ 中
  各有实现并逐点对比测试
- 约束：C 文件无全局可变状态，工作区由调用者提供，错误以返回码表达

**Python 后处理**
- 三种输入后端：xarray/NetCDF、cfgrib/GRIB2、纯 numpy 的 .vibebin
- 派生诊断：RH、露点、相当位温、涡度、散度、Ertel PV、CAPE/CIN、反射率、海平面气压
- 插值：双线性/三线性、单调 PCHIP 垂直插值、守恒粗化
- 绘图：填色图、垂直剖面、简化 skew-T、可靠性图、秩直方图、动能谱
- 命令行工具 vibe-post

---

## 快速开始

~~~bash
# 1. 构建
cmake --preset gcc-release
cmake --build --preset gcc-release -j

# 2. 只装配与校验
./build/src/vibe_model --config config/model.yaml --dry-run

# 3. 跑一个暖泡对流试验
./build/src/vibe_model --config config/model.yaml --ic warm_bubble

# 4. 测试
ctest --test-dir build --output-on-failure

# 5. Python 后处理
python -m pip install -e "python/vibe_post[all]"
python -m vibe_post info output/vibe_idealized_final.nc

# 6. 仓库自检（不需要编译器：校验构建清单、头文件解析与文档链接）
python scripts/check_repo.py
~~~

---

## 目录结构

~~~text
vibe-model/
  CMakeLists.txt          顶层构建（选项、依赖探测、安装导出）
  CMakePresets.json       预设：gcc-release / mixed-precision / cuda-release / hip-release / debug-asan
  cmake/                  CompilerOptions / GPUOptions / FindNetCDF / 包配置模板
  config/                 model.yaml / nests.yaml / da_4dvar.yaml / physics.yaml / verify.yaml
                          variable_resolution.yaml
  include/vibe/           公开头文件（唯一对外接口，按模块分子目录）
  src/                    实现：common config grid dynamics time physics gpu io obs da verify driver
  python/vibe_post/       Python 后处理包（src 布局 + tests + examples）
  examples/               把模式当作库使用的嵌入示例（手工装配各模块）
  tests/                  unit / integration / python / data
  docs/design/            00 架构 01 总览 02 动力核心 03 网格嵌套 04 时间积分
                          05 GPU 混合精度 06 4D-Var 07 切线性伴随 08 物理
                          09 检验 10 后处理 11 配置与 IO 12 构建与测试 + references
  docs/tutorial/          first_run / idealized_cases
  scripts/                构建与运行脚本
  tools/                  独立小工具
~~~

---

## 设计原则

1. **公式可追溯**：每个数值例程的注释块都给出公式、离散化、文献编号与复杂度；
   设计文档里的每个公式都能在 [references.md](docs/design/references.md) 找到出处。
2. **接口冻结**：模块间只通过 include/vibe 下的公开头文件交互；
   依赖方向严格单向，禁止循环依赖。
3. **IO 隔离**：只有 vibe::io 可以包含 NetCDF 头文件；动力、物理、同化模块完全不感知文件格式。
4. **零强制依赖**：没有 MPI/NetCDF/GPU 时仍可构建为纯串行版本；
   YAML 解析器与测试框架都是内置的。
5. **可微分**：每条切线性路径都有对应的伴随路径，并以点积检验作为回归测试。
6. **精度可审计**：存储/计算/归约三种精度角色分离，混合精度策略是可配置、可验证的对象。

---

## 引用

如果你在研究中使用了本模式，请引用对应的原始文献（见 [references.md](docs/design/references.md)），
并注明 VIBE-Model 版本号（`VIBE_VERSION` 宏，构建时由 CMake 注入）。

---

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。

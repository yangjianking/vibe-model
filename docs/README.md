# 文档索引

VIBE-Model 的文档分成三层：**接口基线**（怎么写代码）、**设计文档**（为什么这样写）、
**教程**（怎么用）。

---

## 设计文档（docs/design/）

| 编号 | 标题 | 内容 | 主要文献 |
|---|---|---|---|
| 00 | [体系结构与接口契约](design/00_architecture.md) | 目录结构、命名空间、冻结的接口签名、构建选项、编码规范 | — |
| 01 | [总体设计](design/01_overview.md) | 定位、数值过程链条、模块图、精度与并行策略、学习路径 | 综述 |
| 02 | [动力学核心](design/02_dynamical_core.md) | 控制方程、地形追随坐标、C-grid 离散、平流格式、守恒性、边界条件 | [D1]–[D14] |
| 03 | [网格、嵌套与变分辨率](design/03_grid_and_nesting.md) | 几何与度量、halo 三阶段交换、Davies 松弛、双向反馈、变分辨率映射 | [N1]–[N9] |
| 04 | [时间积分](design/04_time_integration.md) | RK3+声波子步、HEVI、半隐式 Helmholtz 推导、稳定性、时间步控制 | [T1]–[T19] |
| 05 | [GPU 与混合精度](design/05_gpu_hybrid_precision.md) | 后端抽象、内存模型、补偿求和、半精度、迭代精化、能力矩阵 | [G1]–[G18] |
| 06 | [4D-Var 同化](design/06_4dvar.md) | 统一记号、增量方法、B 建模、R 与偏差订正、极小化、观测算子规范 | [V1]–[V25] |
| 07 | [切线性与伴随](design/07_tangent_linear_adjoint.md) | 伴随构造规则、点积检验、不连续过程、检查点、错误清单 | [A1]–[A10] |
| 08 | [物理参数化](design/08_physics.md) | 微物理、辐射、PBL、地面层、陆面、积云与耦合时序 | [P1]–[P18] |
| 09 | [误差检验与评估](design/09_verification.md) | 确定性/分类/概率/空间四类评分的公式与实现 | [E1]–[E19] |
| 10 | [Python 后处理](design/10_postprocessing.md) | 包结构、数据模型、.vibebin 格式、诊断量与绘图 | [B9][E7][E9] |
| 11 | [配置与 IO](design/11_config_and_io.md) | 配置分层与叠加语义、YAML 子集语法、.vibebin 字节规范、CF 属性 | [D16] |
| 12 | [构建、测试与工程实践](design/12_build_and_test.md) | CMake 预设、测试分层与方法论、调试清单、剖析 | [B8][G17] |
| — | [文献出处总表](design/references.md) | 全部引用条目（按 B/D/T/N/G/V/A/O/P/E 分组） | — |

---

## 教程（docs/tutorial/）

| 文档 | 内容 |
|---|---|
| [first_run.md](tutorial/first_run.md) | 从零构建、运行暖泡试验、用 Python 画图 |
| [idealized_cases.md](tutorial/idealized_cases.md) | 全部理想试验的初值公式、推荐配置与检验要点 |

---

## 配置（config/）

| 文件 | 用途 |
|---|---|
| [model.yaml](../config/model.yaml) | 主配置：网格、时间步、数值格式、物理、并行 |
| [nests.yaml](../config/nests.yaml) | 嵌套层级与子域范围 |
| [da_4dvar.yaml](../config/da_4dvar.yaml) | 同化窗口、B 矩阵、极小化、观测选择 |
| [physics.yaml](../config/physics.yaml) | 物理方案组合（叠加在 model.yaml 之上） |
| [verify.yaml](../config/verify.yaml) | 检验变量、阈值、方法与邻域半径 |
| [variable_resolution.yaml](../config/variable_resolution.yaml) | 变分辨率网格示例 |

配置的叠加语义：后给的文件的键覆盖先给的键（映射递归合并，序列整体替换）。
加载入口为 `config::ModelConfig::load(paths)`。

---

## 可运行示例

| 示例 | 位置 | 内容 |
|---|---|---|
| 命令行运行 | [tutorial/first_run.md](tutorial/first_run.md) | 用 vibe_model 跑理想试验 |
| 库内嵌入 | [../examples/embed_model.cpp](../examples/embed_model.cpp) | 手工装配网格/参考态/方程/积分器，输出守恒诊断 |
| 批量试验矩阵 | [../scripts/run_idealized.sh](../scripts/run_idealized.sh) | 多组积分器与格式的对比运行 |
| 仓库自检 | [../scripts/check_repo.py](../scripts/check_repo.py) | 校验构建清单、头文件解析与文档链接 |

---

## 阅读顺序建议

**第一次接触** → [01_overview.md](design/01_overview.md) → [first_run.md](tutorial/first_run.md)
→ [02_dynamical_core.md](design/02_dynamical_core.md)

**要改动力核心或时间推进** → 先读 [00_architecture.md](design/00_architecture.md) 第 4–6 节
→ [02](design/02_dynamical_core.md) → [04](design/04_time_integration.md)

**要做同化** → [06_4dvar.md](design/06_4dvar.md) → [07_tangent_linear_adjoint.md](design/07_tangent_linear_adjoint.md)
→ 跑 tools/adjoint_check

**要做 GPU 移植或混合精度** → [05_gpu_hybrid_precision.md](design/05_gpu_hybrid_precision.md)
→ [00_architecture.md](design/00_architecture.md) 第 7 节

**要做检验与后处理** → [09_verification.md](design/09_verification.md)
→ [10_postprocessing.md](design/10_postprocessing.md)

# 贡献指南

VIBE-Model 的目标是"可学习、可验证、可扩展"。因此对贡献的要求不只是"能跑"，
而是**能被读懂、能被检验、能追溯到文献**。

---

## 1. 提交前检查清单

- [ ] 新增的数值过程有设计文档章节，公式标注了 references.md 中的文献编号；
- [ ] 新代码的每个数值函数上方有注释块：公式 + 离散化 + 文献 + 复杂度；
- [ ] 有对应的单元测试（VIBE_TEST），且覆盖退化情形与性质检验；
- [ ] 新增配置项有默认值，并在 config/*.yaml 中给出示例；
- [ ] 没有引入新的强制第三方依赖（可选依赖必须能被优雅关闭）；
- [ ] 没有破坏 00_architecture.md 中已冻结的接口签名（只能新增，不能改写）；
- [ ] 若修改了 TL/AD 代码，跑过 tools/adjoint_check 且相对误差 < 1e-8；
- [ ] 更新了 docs/README.md 的索引（若新增文档）。

---

## 2. 代码规范摘要

完整规范见 [docs/design/00_architecture.md 第 12 节](docs/design/00_architecture.md)。

* C++20；`#pragma once`；公开头文件全部在 include/vibe/ 下。
* 命名：类型 CamelCase、函数与变量 snake_case、常量 kCamelCase、私有成员尾下划线。
* 禁止裸 new/delete；禁止在热点循环中调用虚函数；禁止在 dyn/timeint/physics/da 中出现
  NetCDF 头文件。
* 索引统一用 `vibe::Int` / `vibe::Index`，禁止 unsigned 参与下标运算。
* 浮点常量统一用 `Real(x)` 包裹，避免精度切换时的隐式提升意外。

---

## 3. 添加一个新物理方案的步骤

1. 在 include/vibe/physics/ 中声明类（可只放最小接口）；
2. 在 src/physics/ 中实现，文件名为 `<过程>_<方案>.cpp`；
3. 在 `physics::make_physics_driver` 的工厂里注册新的方案名；
4. 在 config/physics.yaml 与白名单常量中加入方案名；
5. 写单元测试（至少包含：守恒性、退化情形、与解析解的比对）；
6. 在 docs/design/08_physics.md 中新增一节，列出方程与文献。

---

## 4. 添加一个新的观测算子的步骤

观测算子必须提供**三件套**：`apply`（非线性）、`applyTL`（切线性）、`applyAD`（伴随）。

1. 在 include/vibe/obs/obs_operators.hpp 声明；
2. 实现时必须遵守：TL 中所有系数冻结在基础态；AD 严格按 TL 的逆序转置；
3. 不连续过程采用冻结开关（见 [docs/design/07](docs/design/07_tangent_linear_adjoint.md)）；
4. 在 `make_operator` 工厂注册；
5. 单元测试必须包含：插值权重和为 1、点落在格点上时精确、
   `check_adjoint` 通过（相对误差 < 1e-8）、`check_tangent` 呈 U 形误差曲线。

---

## 5. 报告数值问题

请在 issue 中提供：

1. 完整的配置文件（含所有叠加的 YAML）；
2. 使用的构建预设与精度设置；
3. 复现所需的最小步数；
4. 每 50 步输出的 CFL 报告（模式会自动打印）；
5. 若涉及同化，附上 tools/adjoint_check 的输出。

---

## 6. 许可

贡献即表示你同意以 Apache License 2.0 授权你的贡献。
请勿提交任何许可不兼容的代码或数据（尤其是真实观测资料）。

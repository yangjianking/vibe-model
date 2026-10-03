# 12 · 构建、测试与工程实践

> 本章说明如何配置、构建、测试与调试 VIBE-Model，以及编码规范与性能剖析方法。

---

## 1. 依赖

| 依赖 | 版本 | 必需 | 说明 |
|---|---|---|---|
| CMake | 3.20+ | 是 | 构建系统 |
| C++ 编译器 | 支持 C++20 | 是 | GCC 11+ / Clang 13+ / MSVC 19.30+ |
| C 编译器 | C11 | 是 | GPU 与少量 C 内核 |
| MPI | 3.1+ | 否 | 域分解并行，未找到时自动关闭 |
| OpenMP | 4.5+ | 否 | 共享内存并行 |
| NetCDF | 4.x（C 接口） | 否 | 输出；未找到时回退到 .vibebin |
| CUDA | 11.0+ | 否 | GPU 后端 |
| HIP | 5.0+ | 否 | AMD GPU 后端 |
| SYCL | 2020 | 否 | 跨厂商 GPU 后端 |
| Python | 3.10+ | 否 | 后处理包与 Python 测试 |
| pytest | 7.0+ | 否 | Python 测试 |

**零强制第三方依赖**：即使没有 MPI、OpenMP、NetCDF、GPU，模式仍可构建为纯串行版本
（YAML 解析器与测试框架都是仓库内置的）。

---

## 2. 构建

### 2.1 CMake 预设（推荐）

~~~bash
# 查看可用预设
cmake --list-presets

# GCC + MPI + OpenMP + NetCDF，Release
cmake --preset gcc-release
cmake --build --preset gcc-release

# 混合精度（状态 FP64，内核 FP32）
cmake --preset mixed-precision
cmake --build --preset mixed-precision

# CUDA
cmake --preset cuda-release
cmake --build --preset cuda-release

# Debug + AddressSanitizer/UBSan（无 MPI）
cmake --preset debug-asan
cmake --build --preset debug-asan
~~~

### 2.2 手工配置

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DVIBE_ENABLE_MPI=ON \
      -DVIBE_ENABLE_OPENMP=ON \
      -DVIBE_ENABLE_NETCDF=ON \
      -DVIBE_ENABLE_4DVAR=ON \
      -DVIBE_PRECISION=mixed \
      -DVIBE_GPU_ARCH=80
cmake --build build -j 8
~~~

配置结束后 CMake 会打印一份摘要（构建类型、精度、各后端开关），便于记录在实验日志中。

### 2.3 构建产物

| 目标 | 类型 | 说明 |
|---|---|---|
| vibe_common ... vibe_driver | 静态库 | 各模块库 |
| vibe::model | INTERFACE | 聚合接口，下游只需链接它 |
| vibe_model | 可执行文件 | 主程序 |
| vibe_verify_cli | 可执行文件 | 独立的评分工具（读 .vibebin） |
| vibe_adjoint_check | 可执行文件 | 切线性/伴随点积检验 |
| vibe_embed_example | 可执行文件 | 把模式当库使用的嵌入示例 |
| test_* | 可执行文件 | 单元测试 |
| test_integration | 可执行文件 | 集成测试 |

---

## 3. 测试

### 3.1 运行

~~~bash
ctest --test-dir build --output-on-failure          # 全部
ctest --test-dir build -L unit                       # 只跑单元测试
ctest --test-dir build -L integration                # 集成测试
ctest --test-dir build -L smoke                      # 短时理想试验
./build/tests/test_grid                              # 直接运行
./build/tests/test_grid thomas                       # 只跑名字含 thomas 的用例
~~~

### 3.2 测试分层

| 层 | 标签 | 内容 | 运行时间 |
|---|---|---|---|
| 单元 | unit | 类型/常量/日志/**C 内核**、YAML、二进制 IO、网格几何、场运算、halo、插值、嵌套、参考态、平流、科氏、阻尼、方程、诊断、三对角、Helmholtz、声波子步、RK3、半隐式、时间步控制、GPU 精度、物理、检验、观测算子、极小化、TL/AD | 秒级 |
| 集成 | integration | 完整驱动跑全部理想试验、两种积分器对比、变分辨率网格、检查点往返 | 十秒级 |
| 冒烟 | smoke | 用配置文件跑 4 个理想试验（配置解析 + 装配 + 短积分的端到端检查） | 十秒级 |
| Python | python | vibe_post 的 IO/诊断/检验/插值测试 | 秒级 |

### 3.3 测试框架

内置头文件框架（无外部依赖），见 [test.hpp](../../include/vibe/common/test.hpp)：

~~~cpp
#include "vibe/common/test.hpp"

VIBE_TEST(helmholtz_vertical_tridiagonal_recovers_manufactured_solution) {
  // 制造解方法（Method of Manufactured Solutions）
  VIBE_CHECK(err < Real(1e-8));
  VIBE_CHECK_NEAR(solver->iterations(), 1, 0);
}
~~~

宏：`VIBE_TEST`、`VIBE_CHECK`、`VIBE_CHECK_NEAR`、`VIBE_CHECK_REL`、`VIBE_CHECK_THROWS`。
测试入口统一在 [test_main.cpp](../../tests/unit/test_main.cpp)，支持命令行过滤。

### 3.4 数值测试的方法论

1. **制造解（MMS）**：给定解析解，反算右端项，验证离散算子与求解器的收敛率。
   示例：Helmholtz 三对角测试。
2. **解析解**：在线性问题中使用。示例：三对角矩阵的常数系数解析解。
3. **守恒性检验**：验证质量、能量、位涡在理想条件下的守恒。示例：周期边界质量守恒。
4. **性质检验**：验证离散算子的代数性质。示例：平流的线性性、插值的权重和为 1、
   限制/延拓的守恒性、科氏力的反对称性。
5. **极限检验**：验证退化为已知极限。示例：FSS 在半径趋于无穷时趋于 1、
   WENO5 对光滑函数的高阶精度。
6. **伴随点积检验**：见 [07_tangent_linear_adjoint.md](07_tangent_linear_adjoint.md)。

---

## 4. 运行

~~~bash
# 基本运行
./build/src/vibe_model --config config/model.yaml

# 选择理想试验与日志级别
./build/src/vibe_model --config config/model.yaml --ic warm_bubble --log-level info

# 只装配与校验（不积分）
./build/src/vibe_model --config config/model.yaml --dry-run

# MPI 并行（4 进程，2x2 分解）
mpirun -np 4 ./build/src/vibe_model --config config/model.yaml
#   config 中设置 parallel: { px: 2, py: 2 }

# 从检查点重启
./build/src/vibe_model --config config/model.yaml --restart output/restart_120.vibebin
~~~

退出码：0 正常，1 配置或运行错误，2 数值发散，3 命令行错误。

---

## 5. 调试与剖析

### 5.1 数值问题排查清单

| 症状 | 首要检查项 |
|---|---|
| 立即出现 NaN | 配置的 dt 是否过大；zeta 是否单调；地形是否超出 z_top |
| 积分若干步后发散 | 打印 CFL 报告（每 50 步自动输出）；检查声波子步数 |
| 质量不守恒 | 是否启用了物理过程的水物质源汇；边界是否周期 |
| 上边界反射 | 海绵层起始高度与 alpha_max |
| 陡坡上出现噪声 | 地形平滑次数、层间平滑、是否启用 WENO |
| 4D-Var 不收敛 | 先跑伴随点积检验（`check_adjoint`），再检查 B 的长度尺度 |
| 并行结果与串行不一致 | halo 宽度是否足够；分解是否整除 |

### 5.2 计时

`common::TimerRegistry` 按名称聚合，`ScopedTimer` 打点。运行结束时
driver 调用 `timers_.report()` 输出分解。

### 5.3 性能剖析

~~~bash
# CPU
perf record -g ./build/src/vibe_model --config config/model.yaml
perf report

# GPU
nsys profile --stats=true ./build/src/vibe_model --config config/model.yaml
ncu --set full --kernel-name regex:advect ./build/src/vibe_model ...
~~~

**并行效率诊断**（[G17] Amdahl 定律）：固定问题规模，扫描 MPI 进程数，
记录每模拟日墙钟时间；`Comm::max_with_rank` 可用于定位负载最重的进程。

---

## 6. 编码规范

1. C++20；头文件用 `#pragma once`；公开头文件全部位于 include/vibe/。
2. 命名：类型 CamelCase，函数与变量 snake_case，编译期常量 kCamelCase，
   私有成员 trailing_underscore_。
3. 每个数值例程上方必须写注释块：**公式 + 离散化 + 文献编号 + 复杂度**。
4. 热点循环内禁止虚函数调用（物理参数化除外，其成本占比低）。
5. 禁止裸 new/delete；容器用 std::vector；GPU 缓冲用 `gpu::DeviceBuffer`。
6. 单元测试文件名与被测源文件同名：test_x.cpp。
7. 错误处理：配置与 IO 失败抛异常；数值内部使用 `VIBE_ASSERT`（Release 下消失）。
8. 新增物理或数值过程时，必须同时提交：实现、单元测试、设计文档更新、
   配置项与默认值、以及至少一个理想试验的验证结果。

---

## 6b. C 与 C++ 的分工

密集计算优先用 C / C++，但两者有明确分工：

| 层 | 语言 | 位置 | 内容 |
|---|---|---|---|
| 框架与算法组织 | C++20 | src/ 其余部分 | 网格、状态、方程装配、积分器、物理、同化、IO |
| 热点内循环 | C11 | src/common/c_kernels.c | Thomas 三对角、WENO5 重构、双线性插值、CFL 扫描、Neumaier 求和、通量平流 |
| GPU 内核 | CUDA C++ | src/gpu/cuda/*.cu | 与 C 版内核逐式对应的设备实现 |

选择 C 的三条理由：

1. **可被外部直接调用**：C ABI 稳定，FORTRAN / Python(ctypes, cffi) / Julia 都能直接链接，
   便于与已有模式或脚本耦合；
2. **便于对照验证**：同一公式在 C 与 C++ 中各有一份实现，单元测试逐点对比
   （见 tests/unit/test_common.cpp 里的 c_kernel_weno5_matches_cpp_implementation）；
3. **工具链诊断**：部分 HPC 工具链的向量化报告与性能计数器对 C 更成熟。

约束：C 文件中不得有任何全局可变状态；所有工作区由调用者提供；函数必须可重入且
线程安全；错误以返回码表达（VIBE_C_OK / VIBE_C_ERR_SINGULAR / VIBE_C_ERR_BAD_ARG），
由 C++ 侧翻译为异常。

---

## 7. 持续集成建议

~~~yaml
# .github/workflows/ci.yml 的推荐结构（本仓库不内置以保持零依赖）
jobs:
  build-and-test:
    strategy:
      matrix:
        preset: [gcc-release, debug-asan, mixed-precision]
    steps:
      - cmake --preset ${{ matrix.preset }}
      - cmake --build --preset ${{ matrix.preset }}
      - ctest --preset ${{ matrix.preset }}
  python:
    steps:
      - python -m pip install -e python/vibe_post[all]
      - python -m pytest python/vibe_post/tests
~~~

---

## 8. 参考文献

[B8] Nocedal & Wright (2006)（数值优化的收敛判据） · [G17] Gustafson (1988)（性能评估）

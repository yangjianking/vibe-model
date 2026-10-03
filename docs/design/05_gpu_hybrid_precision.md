# GPU 内核与混合精度抽象层（vibe::gpu）

> 本文是 [00_architecture.md](00_architecture.md) 第 7 节（GPU 与混合精度契约）的展开，
> 说明四条后端（CPU / CUDA / HIP / SYCL）如何共用同一套源码、内存模型如何与 halo 布局
> 对齐、混合精度的误差预算是多少、以及正确性如何被验证。
> 全文公式的文献编号见 [references.md](references.md)。

---

## 1. 后端抽象与移植策略

### 1.1 编译期探测，运行时选择

后端在**编译期**唯一确定（@VIBE_BACKEND_CPU@ 等宏），运行时只能查询、
不能改变编译产物支持的后端集合：

~~~text
探测顺序：用户显式定义 -> __CUDACC__ -> __HIPCC__ -> SYCL_LANGUAGE_VERSION -> CPU
不变量：  VIBE_BACKEND_CPU + CUDA + HIP + SYCL == 1   （backend.hpp 的 static_assert）
~~~

这条不变量换来三个好处：

1. 所有 @#if@ 都是编译期常量，分支被完全消除，定点循环里没有虚函数（架构第 12 节规范 4）；
2. 设备内存/内核/流三类对象在 CPU 后端有"退化实现"，业务代码不需要写条件编译；
3. 不会出现"编译了两套内核又同时链接"的符号冲突（见 1.3 的链接契约）。

### 1.2 四类抽象对象

| 抽象 | 头文件 | CPU 后端语义 | 加速器后端语义 |
|---|---|---|---|
| @Backend@ / @BackendCaps@ | backend.hpp | 值语义 + 能力矩阵 | 同左（张量核、统一内存、事件计时为真） |
| @Device@ / @DeviceInfo@ | device.hpp | 一个伪设备（线程数、物理内存） | @cudaGetDeviceProperties@ 等 |
| @DeviceBuffer<T>@ / @MemoryPool@ | memory.hpp | 堆内存（设备指针 == 主机指针） | 设备内存 + 页锁定镜像 / 统一内存 |
| @Stream@ / @LaunchConfig@ | launch.hpp | no-op 流、OpenMP 分块 | 真流、1D 网格、共享内存 |

CPU 后端的"退化"不是敷衍，而是**参考实现**：它的结果用于反证 GPU 路径的正确性
（第 9 节验证清单）。

### 1.3 链接契约与内核组织

架构第 7 节要求"CPU 与 CUDA 各给一份实现"，同时又要求"业务代码禁止写三角括号启动语法"。
在不修改构建系统的前提下，本层采用如下组织：

~~~text
include/vibe/gpu/kernels.hpp            声明（唯一公开接口，签名冻结）
src/gpu/cpu/kernels_cpu.cpp             定义：串行 + OpenMP（#if !VIBE_HAVE_GPU）
src/gpu/cuda/kernels_advection.cu       定义：__global__ 设备内核 + 主机启动包装
src/gpu/cuda/kernels_diffusion.cu       ...
src/gpu/cuda/kernels_{tridiagonal,helmholtz,interp,reduction}.cu
~~~

**一次链接只能包含其中一组**。CPU-only 构建编译 @kernels_cpu.cpp@；
CUDA 构建编译 @src/gpu/cuda/*.cu@（由 @VIBE_ENABLE_CUDA@ 决定）。
两个实现文件各自用守护宏拒绝被同时编译：

~~~cpp
#define VIBE_KERNELS_BACKEND_CPU 1     // kernels_cpu.cpp
#if VIBE_HAVE_GPU
#  error "kernels_cpu.cpp 是 CPU 后端实现；CUDA 构建请编译 src/gpu/cuda/*.cu"
#endif
~~~

这与架构第 7 节的字面表述（一个函数两份定义）有差异，但保留了它的**意图**：
接口唯一、实现按后端替换、业务代码零条件编译。差异已记入交付总结。

### 1.4 索引空间约定（CPU 与 GPU 必须逐位一致）

内核索引空间是三维的（对应含 halo 的存储维度），线性化顺序固定为 **x 最快**：

```
flat = (k * g_y + j) * g_x + i,   g_x = nsx, g_y = nsy, g_z = nsz
```

* CPU：@for_each_index_3d@ 把索引空间按 @block.x*block.y*block.z@ 分块，
  块间 OpenMP 并行、块内按 flat 顺序串行；
* GPU：设备内核里同样用 flat 分解，块内线程沿 i 连续 -> 合并访存。

两条路径的**算术表达式逐字相同**（CPU 版与 CUDA 版的模板函数一一对应），
因此同一输入得到相同结果（除浮点结合顺序外逐位一致）。

### 1.5 移植检查单（新增一个后端时）

1. 在 backend.hpp 增加探测宏与 @BackendCaps@ 取值；
2. 在 device.cpp / memory.cpp 的四处 @#if@ 链里补分支（分配、释放、拷贝、同步）；
3. 在 launch.hpp 的索引迭代里确认 OpenMP/工作组的两级并行映射；
4. 在内核侧新增一份实现目录（@src/gpu/<backend>@），设备内核签名与 CPU 版一致；
5. 用第 9 节的验证清单逐项对拍。

---

## 2. 内存模型与 halo 交换的 GPU 化

### 2.1 三种分配策略的取舍

| 策略 | 底层 API | 适用 | 代价 |
|---|---|---|---|
| @Device@ | @cudaMalloc@ + @cudaMallocHost@ 镜像 | 全部预报场与趋势场 | 每次上传/下载各一次 O(n) 传输 |
| @Pinned@ | @cudaMallocHost@ | halo 缓冲、观测向量、小规模标量 | 页锁定内存不可换出，过量会拖慢主机 |
| @Unified@ | @cudaMallocManaged@ | 大场、访存稀疏、跨模块共享 | 缺页迁移抖动，性能不可预测（[G10] 第 6.2.4 节） |

默认策略的选择原则：**时间步内反复访问的场用 @Device@**（显式传输、时序可预测），
**每个时间步只碰一次的大场用 @Unified@**（省一次显式拷贝），
**同步点之间的临时缓冲用 @Pinned@**。池化（@MemoryPool@）把 @Device@ 的
分配代价摊销到 O(log B)，这是避免"每步 cudaMalloc"这一常见性能陷阱的关键（[G10] 第 6.2 节）。

### 2.2 halo 布局的 GPU 对齐

@grid::Field@ 是"单数组 + halo"（grid/field.hpp）：

```
off(i,j,k) = ((k + h) * nsy + (j + h)) * nsx + (i + h)
```

设备副本就是同一段线性内存的镜像（@FieldMirror@），因此：

* **不需要**在 GPU 上重排数据，也不需要设备端的索引表；
* 内核把 halo 当真实数据访问（一阶/二阶模板会越出内部区域），
  这与 CPU 侧 @Field::at(i,j,k)@ 允许负索引的约定完全一致；
* halo 区的宽度要求：2 阶模板 h >= 2、4 阶与 WENO5 h >= 3。

### 2.3 halo 交换的三阶段方案

GPU 上 halo 交换的本质是"六/八个方向的边界切片搬运 + 邻居 MPI 通信"，标准做法是：

~~~text
阶段 1  pack    : 设备内核把边界切片写入连续的 pinned 缓冲（无需回主机内存）
阶段 2  comm    : cudaMemcpyAsync(pinned -> 邻居) / MPI_Isend/Irecv
阶段 3  unpack  : 设备内核从 pinned 缓冲写回 halo 区
                 对角方向必须等对边方向完成后再做（grid/halo.hpp 的两阶段约定）
~~~

关键设计决策：

1. **pinned 缓冲 + 异步流**：把 pack/comm/unpack 三条流水线重叠起来，
   在 halo/内部点比例较大时能把通信代价藏进计算（[G10] 第 6.2.4 节）；
2. **同一错位共享一次交换**：@rho/theta/pi/qv@ 都是 Cell 场，
   打包成一次连续传输（halo.hpp 的 @exchange(vector<Field*> )@），减少消息数；
3. **不要用统一内存做 halo**：缺页迁移会在通信路径上引入不可预测的停顿，
   与"halo 交换必须与计算重叠"的目标冲突（[G6] 的实践结论）。

### 2.4 内存预算

@check_memory_budget@ 在时间步开始时检查"当前占用 + 本次新增"是否超过
设备容量的 90%。理由：统一内存与碎片化会让 @cudaMalloc@ 在接近容量时失败，
提前失败（抛 @vibe::Error@）比运行到一半崩溃更容易诊断。

---

## 3. 混合精度理论与误差传播

### 3.1 三种精度角色

@PrecisionPolicy@ 把精度分成三个**角色**（[G6] 的核心思想）：

| 角色 | 默认 | 约束来源 |
|---|---|---|
| @storage@（状态量存储） | FP64 | 守恒量必须精确存储（架构第 4 节铁律 1） |
| @compute@（趋势/通量） | FP32 | 每步从 FP64 状态重新出发，误差不跨步累积 |
| @reduce@（归约/累加） | FP64 | 长链累加的误差按 n 增长，必须补偿或加宽 |

### 3.2 单位舍入误差与单步误差

记浮点格式的单位舍入误差 u = eps/2（FP64 为 2^-53，FP32 为 2^-24，
FP16 为 2^-11，BF16 为 2^-8）。对一次基本运算 z = x o y：

```
z = (x o y) (1 + delta),   |delta| <= u
```

一个时间步的局部舍入误差量级为

```
eps_step  ~  u_compute * ||q||
```

### 3.3 累积：随机游走还是线性漂移

若每步误差相互独立、零均值，则 N 步后

```
||eps_N||  ~  sqrt(N) * u_compute * ||q||
```

若误差相关（保守格式的系统性收支偏差、有偏的舍入），则

```
||eps_N||  ~  N * u_compute * ||q||
```

这条区别决定了工程决策：

* **趋势/通量（compute 角色）**：可以安全降到 FP32。
  以 u32 = 6e-8、N = 1e4 步为例，随机游走上界约 6e-6 倍的场量级 —— 远小于模式自身的
  离散误差（O(dx^2) 约 1e-3）（[G7] 的基准结论）；
* **质量/能量收支（reduce 角色）**：绝不能降到 FP32。
  同样 N = 1e4 步线性累积上界为 6e-4，已经与可分辨的信号同量级 ——
  这正是 [G6] 报告的"质量收支必须在双精度下累加"的原因。

### 3.4 离散化误差与舍入误差的交叉点

```
C * dx^p   vs   sqrt(N) * u_compute * ||q||
```

令两者相等可得"低于此分辨率再用高精度就没有收益"的分辨率。以 3 阶精度
dx = 100 m 的平流为例，C dx^3 已达 1e-3 量级，因此 FP32 计算在 dx >= 100 m
是充分的；只有 dx < 10 m（LES 极限）或 4D-Var 的多重外循环
（误差被放大 sqrt(cond(B)) 约 100 倍）才需要 FP64 计算（[G6][G9]）。

### 3.5 迭代精化：让"低精度求解 + 高精度修正"成立

Helmholtz 方程 A x = b 用 FP32 求解、FP64 计算残差：

~~~text
step 1  x_0 = A_32^{-1} b                 （FP32 因子分解与回代）
step 2  r = b - A_64 x_0                  （FP64 残差，误差 ~ u_64 * ||b||）
step 3  d = A_32^{-1} r                   （FP32 修正量求解）
step 4  x_1 = x_0 + d                     （FP64 累加）
重复 step 2-4 直到 ||r|| < tol 或达到迭代上限
~~~

收敛性（[G5] 的定理 3.1）：若 ||I - A32^{-1} A64|| = rho < 1，
则 ||x_{k+1} - x|| <= rho ||x_k - x||，k 次迭代后达到 O(rho^k u64) 的精度 ——
**与 FP32 的 u 无关**。
本层要求 Helmholtz 的残差始终以 @reduce@（默认 FP64）累加
（@helmholtz_residual@ 内部用 double），否则精化会退化为原地踏步。

---

## 4. 补偿求和

### 4.1 误差上界

| 方法 | 误差上界 | 每项代价 |
|---|---|---|
| 朴素左折叠 | (n-1) u sum|x_i| | 1 flop |
| Kahan [G2] | (2u + O(n u^2)) sum|x_i| | 4 flops |
| Neumaier [G3] | (2u + O(n u^2)) sum|x_i|，且对 |x_i| > |s| 仍有效 | 4~5 flops |
| @TwoSum@ 骨架 [G4] | 保留全部误差项，可用于精确点积 | 6 flops |

Kahan 的补偿项在 |x| > |s| 时会**丢失**（[G3] 的反例）：
s = 1e16 时加上 1.0 与 -1e16 的序列，Kahan 得到 0 而 Neumaier 得到 1。
因此本层默认 @SumAlgorithm::Neumaier@。

### 4.2 并行归约中的补偿

并行归约不能"各家算完朴素相加"—— 那会把各块的补偿项丢掉。
正确做法是**两分量合并**（[G4] 第 4 节）：

```
merge(s1,c1; s2,c2):  先 add(s2)、再 add(c2)，在 (s1,c1) 上执行 Neumaier 步
```

本层在三级结构里都使用它：

~~~text
线程内（网格跨步）：每线程一个 CompensatedSum
warp 内            ：__shfl_down_sync 同时传 sum 与 c，逐级 merge
block 内           ：共享内存存 2*(warp 数) 个 Real，第一 warp 再 merge
跨 block           ：每 block 只写一个值到 partial[blockIdx.x]，宿主端按序合并
~~~

**确定性**：跨 block 用"按块序号的串行合并"而不是原子加，
因为原子加的浮点求和顺序不确定，会破坏 4D-Var 的可复现性（[V3][A4]）。
代价是每个 block 多写 8 字节并多一次 O(blocks) 的宿主循环，可忽略。

### 4.3 归约分块与线程数无关性

CPU 实现把 [0,n) 切成固定大小 65536 的块（与线程数无关），
每块一个 @CompensatedSum@，再按块序号合并。于是**任意 OpenMP 线程数下结果逐位相同**。
GPU 实现的块划分由 @blocks_for(n, 256, 1024)@ 决定，同样是 n 的函数而非调度时间的函数。

---

## 5. 半精度与张量核的适用边界

### 5.1 两种 16 位格式

| 格式 | 位布局 | u | 动态范围 | 适用 |
|---|---|---|---|---|
| @half_t@（binary16） | 1+5+10 | 2^-11 | 6e-8 ~ 6.5e4 | 观测/辐射传输中间量、张量核输入 |
| @bfloat16_t@ | 1+8+7 | 2^-8 | 与 FP32 相同 | 动力学变量的降精度实验 [G7] |

BF16 的指数位与 FP32 相同，因此**不会在气象量程两端溢出**，
这是它在数值模式里比 half 更受青睐的根本原因；代价是尾数只有 7 位（u = 2^-8）。

### 5.2 为什么动力学不能用 16 位存状态

由第 3.3 节，N = 1e4 步、u16 = 2^-11 时随机游走误差上界约 1e-2 倍的场量级，
已经与信号同量级。因此架构第 4 节铁律 1（守恒量必须 FP64）不可违反，
@PrecisionPolicy::valid()@ 会在默认配置下拒绝 @storage != FP64@。

### 5.3 张量核的边界

张量核（16x16x16 的 FP16 乘累加，[G1][G5]）要求：

1. 操作可以写成矩阵乘或卷积形式（Helmholtz 的 stencil 不能直接映射）；
2. 累加在 FP32 中进行（MMA 的累加器精度），否则 u16 * n 会失控；
3. 有足够的规则批处理（如观测算子的批量插值、辐射传输的批量通道）。

本层对张量核的态度是：**暴露能力（@BackendCaps::tensor_core@），但不在动力学内核里使用**。
可用于的候选位置是第 4 章的批量插值与辐射传输模块（未来扩展）。

---

## 6. Helmholtz 求解的迭代精化实现

### 6.1 算子离散

```
L x = div(a grad x) - b x
```

面系数用**调和平均** a_f = 2 a_i a_{i+1} / (a_i + a_{i+1}) 使跨面通量连续。
展开后等价于对称形式

```
L x = div(a grad x) - b x
```_SYM

### 6.2 精化循环

~~~text
x = 0
for outer = 1..n_refine:
    r   = f - L(x)                    # FP64 残差（helmholtz_residual）
    if ||r|| < tol: break             # 用 reduce_sum 的补偿求和，避免判据本身不可靠
    d   = solve_FP32(L, r, b)         # 多网格/PCG，低精度因子
    x  += d                           # FP64 累加
~~~

* 残差判据必须用 @reduce_sum@（补偿）而不是朴素求和：否则残差范数自身的误差
  就可能大于判据阈值（当 ||r||/||f|| < 1e-8 时）；
* Chebyshev + 多重网格平滑子用第 4 节的 Jacobi/Chebyshev 内核（[T13]）；
* 迭代次数上限与外循环次数由 timeint 层配置。

### 6.3 无条件稳定性

架构第 6.4 节要求线性声波-重力波在隐式格式下无条件稳定。
Helmholtz 算子是对称负定（a > 0, b >= 0）的，Chebyshev/Jacobi 的迭代矩阵特征值
落在 (-1,1) 内，因此迭代不放大误差；精化循环只改善解的精度，不改变稳定性（[T7][T13]）。

---

## 7. 四后端能力矩阵

| 能力 | CPU | CUDA | HIP | SYCL |
|---|---|---|---|---|
| 独立设备内存 | 否（统一地址） | 是 | 是 | 是 |
| 统一内存 | 是（天然） | cudaMallocManaged | hipMallocManaged | USM shared [G11] |
| 页锁定主机内存 | 否 | cudaMallocHost | hipHostMalloc | USM host |
| 多流并发 | 否（OpenMP 线程组） | 是 | 是 | 是（多 queue） |
| warp/wavefront 宽度 | 1（无锁步） | 32 [G10] | 64（GCN/RDNA）[G12] | 32（sub-group） |
| 原生 FP16 运算 | 软件模拟 | cc >= 5.3 | 是 | 是 |
| 原生 BF16 | 软件模拟 | cc >= 8.0 | CDNA2/RDNA3 | SYCL 2020 |
| 张量核 / MFMA | 否 | cc >= 7.0 [G1] | 是 | joint_matrix 扩展 |
| 事件计时 | steady_clock | cudaEvent [G10] | hipEvent | sycl::event |
| 全局同步 | 隐式 | cooperative groups | cooperative groups | work-group barrier |
| 共享/本地内存 | 否 | __shared__ 48~96 KB | LDS 64 KB | local accessor |
| 浮点原子加 | 是 | 是 | 是 | 是 |

共享内存与块大小的默认选择：block.x = 256、三对角沿求解方向一个 block
（共享内存 2*nz 个 Real）、归约共享内存 2*(warp 数) 个 Real。

---

## 8. 性能剖析方法

### 8.1 内核级计时

* GPU：用**事件计时**（cudaEventRecord/ElapsedTime）测**设备侧**时间，
  排除主机抖动与启动延迟；CPU：用 @steady_clock@ 测挂钟时间（[G10] 第 6 章）；
* @KernelTimings@ 按内核名聚合，输出 calls / total_ms / mean_ms
  与降序排序（@sorted()@），直接对应 Amdahl 定律中的"热点占比"（[G17]）：

```
S_max = 1 / ( (1 - p) + p / s )
```

其中 p 为可加速部分占比、s 为加速比。若 p = 0.95，即使 s 趋于无穷
也只能得到 20 倍 —— 这就是"必须同时优化 halo 交换与 IO"的量化依据。

### 8.2 四个必查指标

| 指标 | 目标 | 手段 |
|---|---|---|
| 占用率（occupancy） | >= 50%（受寄存器/共享内存约束） | @Device::occupancy()@；调 @__launch_bounds__@ |
| 访存效率 | 合并事务数 / 请求数 >= 90% | i 最快序 + 对齐（见 1.4） |
| 算术强度 | 每字节访存 >= 2 flops | 提高每点工作量（WENO5、biharmonic 双遍） |
| 传输/计算比 | copy_to_device 时间 < 内核时间 | 减少同步点；用流重叠（2.3 节） |

### 8.3 常见反模式

1. 在时间步内调用 @DeviceBuffer::resize@（隐式分配 + 同步）—— 用 @MemoryPool@；
2. 每步 @copy_to_host@ 取范数做诊断 —— 把归约留在设备侧（@reduce_sum@）；
3. 用统一内存承载每步多次访问的场 —— 缺页迁移抖动；
4. 把 @shared_bytes@ 设成 0 却在设备内核里申明大数组 —— 编译期就会超限。

---

## 9. 正确性验证清单

每一项都必须在 CPU 参考实现与目标后端上**同时**通过
（tests/unit/test_gpu_precision.cpp 已覆盖 1–5、7，以及第 6 项的一部分）。

| # | 检验 | 判据 | 对应文献/需求 |
|---|---|---|---|
| 1 | 补偿求和精确性 | 1e8 个 1.0 之和与项数之差 <= 1；Neumaier 在抵消序列上不劣于 Kahan | [G2][G3][B11] 第 4 章 |
| 2 | 半精度往返与舍入 | half 相对误差 <= 2^-11、BF16 <= 2^-8；0/Inf/NaN/次正规正确 | [G1] 第 3 节 |
| 3 | 保常数性 | 常速度场平流常标量场，趋势恒为 0（对 2/4 阶） | [D6] |
| 4 | 守恒检验 | 周期边界下 sum(rho q dV) 相对漂移 < 1e-12 / 步 | 架构 6.4 节约束 2 |
| 5 | 线性精确性 | 双线性插值对线性场精确；z 落在层中心时三线性退化为双线性 | [O8] |
| 6 | 三对角解析解 | 常系数 (-1,2,-1) 系统与解析解误差 <= 1e-10 | [T10] |
| 7 | 限制/延拓 | restrict 常数不变、总量守恒（权重 1/r^2）；prolong 常数不变；restrict(prolong) 恒等 | [N4][N5] |
| 8 | 点积检验 | 两组实现在随机场上 <y,Ax> = <A^T y,x> 相对误差 < 1e-12（TL/AD 自检） | [A4] 第 3 节 |
| 9 | 收敛率对比 | 与 FP64 基准解相比，误差随分辨率按设计阶数下降（2/4/5 阶的双对数斜率） | [B2] 第 2 章 |
| 10 | 精度敏感性 | FP32 计算 + FP64 存储 与全 FP64 的 24 h 预报差异 < 1e-3；能量收支差异 < 1e-8 | [G6][G7] |
| 11 | 迭代精化收敛 | 残差范数按 rho^k 递减，最终达到 O(u64)（与 FP32 因子无关） | [G5] |
| 12 | 确定性 | 同一输入在不同线程数/块数下 reduce_sum 逐位相同 | [V3][A4] |

第 8、9、10、11 项的完整实现依赖 dyn/timeint 模块，本层已提供所需内核与配置接口。

---

## 10. 与架构契约的对应关系

| 架构第 7 节条目 | 本层实现 |
|---|---|
| enum class Backend { CPU, CUDA, HIP, SYCL } | backend.hpp（附加 to_string/from_string/BackendCaps） |
| DeviceInfo（含 compute_major/minor、multiprocessors） | device.hpp（架构给出的是精简版，本层为**扩展**） |
| enumerate_devices() / active_backend() | device.cpp（另加 set_active_backend/device_count） |
| DeviceBuffer<T>（device_ptr/host_ptr/copy_*/size） | memory.hpp（另加 AllocKind/resize/fill/release） |
| Stream、LaunchConfig{grid,block,shared_bytes,stream} | launch.hpp（dim3 置于 vibe::gpu 内） |
| launch(kernel, cfg, args...) | launch.hpp 提供 for_each_index_1d/3d[_strided] 与 launch；内核实现内使用 |
| Precision / PrecisionPolicy（五字段） | precision.hpp（字段完全一致） |
| CompensatedSum<T>（sum, c, add, value） | precision.hpp（模板参数增加算法变体，默认 Neumaier） |
| kernels 函数列表 | kernels.hpp（参数顺序统一为"裸指针 + 形状 + 几何"，见交付总结的偏差说明） |

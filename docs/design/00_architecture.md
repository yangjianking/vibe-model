# VIBE-Model 体系结构与接口契约

> 本文是整个仓库的**接口基线（interface baseline）**。所有模块的头文件、命名空间、类型名、
> 文件布局都必须与本文一致；若某模块需要扩展接口，只能**新增**，不得改写已冻结的签名。
> 文中所有数学符号与公式的文献出处见 [references.md](references.md)。

---

## 1. 设计目标与范围

VIBE-Model 是一个面向**教学与科研**的区域/全球一体化数值天气预报模式，强调"可读、可验证、可扩展"。

| 目标 | 具体要求 |
|---|---|
| 可学习 | 每个物理/数值过程独立成文件，公式与文献一一对应 |
| 可移植 | 同一套源码支持 CPU（MPI+OpenMP）与 GPU（CUDA/HIP/SYCL） |
| 可嵌套 | 支持固定比率双向嵌套与平滑变分辨率（variable-resolution）网格 |
| 可同化 | 提供增量 4D-Var，观测算子接口显式、可插拔 |
| 可微分 | 切线性（TL）与伴随（AD）代码由模板/宏约束成对出现，可做点积检验 |
| 可检验 | 内置确定性/分类/概率/邻域四类检验评分 |
| 可后处理 | 独立 Python 包 `vibe_post`，不依赖 C++ 侧运行 |

**不在范围内**：业务化资料接收、集合预报全流程（只提供接口）、化学/气溶胶模块。

---

## 2. 目录结构

~~~text
vibe-model/
  CMakeLists.txt              顶层构建
  CMakePresets.json           预设（gcc-release / cuda-debug / hip-release ...）
  cmake/                      构建辅助模块
  config/                     YAML 运行配置
  docs/design/                设计文档（本文所在）
  docs/tutorial/              上手教程
  include/vibe/               公开头文件（唯一对外接口，按模块分子目录）
  src/                        实现（.cpp / .c / .cu）
  examples/                   嵌入示例（把模式当库用，手工装配各模块）
  obs/                        示例合成观测（CSV）与格式说明
  python/vibe_post/           Python 后处理包（src 布局 + tests + examples）
  tests/                      单元 / 集成 / Python 测试与测试数据说明
  scripts/                    构建、批量运行、地形生成、仓库自检
  tools/                      独立小工具（adjoint 点积检验器）
~~~

---

## 3. 命名空间与模块依赖

命名空间统一为 `vibe::<module>`，禁止跨模块直接访问对方内部 `detail` 命名空间。

~~~text
vibe::common      基础类型、常量、错误、日志、MPI 封装、计时器
vibe::config      YAML 配置读取与校验
vibe::grid        网格几何、错位网格、度量项、插值、嵌套、halo
vibe::dyn         预报方程、参考态、平流、科氏、阻尼、诊断量
vibe::timeint     时间推进：RK3 / 声波子步 / 半隐式 Helmholtz
vibe::physics     物理参数化：微物理、辐射、PBL、陆面、积云
vibe::gpu         设备抽象、内存、内核启动、混合精度策略
vibe::io          NetCDF / 二进制场 IO、重启、检查点
vibe::da          控制变量、B 矩阵、代价函数、极小化、增量 4D-Var
vibe::obs         观测类型、观测空间、观测算子（含 TL/AD）
vibe::verify      检验评分与检验框架
vibe::driver      顶层积分驱动 main 程序
~~~

依赖方向（箭头表示"可依赖"）：

~~~text
common  <-  config  <-  grid  <-  dyn  <-  timeint  <-  driver
                                     <-  physics  <-
common  <-  gpu     <-  (grid/dyn/timeint/physics/da 的加速后端)
common  <-  io      <-  driver
common  <-  da      ->  obs, dyn, timeint, grid
common  <-  verify  ->  (读取 NetCDF 预报/观测，不依赖 dyn 内部状态)
~~~

**硬约束**：`dyn`、`timeint`、`physics`、`da` 不得包含 NetCDF/GRIB 头文件；
所有外部文件格式只在 `io` 层出现。

---

## 4. 基础类型契约（include/vibe/common/types.hpp）

~~~cpp
namespace vibe {

// 编译期精度策略：VIBE_PRECISION = 0(double) / 1(single) / 2(mixed)
#if VIBE_PRECISION == 1
using Real = float;
#else
using Real = double;
#endif

using Int   = std::int32_t;   // 网格索引
using Index = std::int64_t;   // 全局序号、数组偏移
using Size  = std::size_t;

// 混合精度：存储/通量/累加三种角色可分别取型
namespace gpu {
  template <class T> struct TypeTag { using type = T; };
  using StorageReal = ...;   // 见 gpu/precision.hpp
}

} // namespace vibe
~~~

三条铁律：

1. **状态量守恒**：任何预报量（质量、动量、位温、水物质）必须用 `Real`（=double）存储，
   即使 GPU 后端以 FP32 计算趋势。
2. **索引无符号化禁止**：一律用 `Int`，禁止 `unsigned` 参与下标运算。
3. **裸指针只在 `gpu` 与 `common` 层出现**，其余模块只使用 `grid::Field`。

---

## 5. 网格与场契约（include/vibe/grid/）

### 5.1 坐标系与错位

* 水平：Arakawa **C-grid**（u 在 x 面，v 在 y 面，标量在中心）。
* 垂直：Lorenz 错位（w 在半层，标量在整层）。
* 垂直坐标：地形追随高度坐标（Gal-Chen & Somerville 1975）
  \( \zeta = H \dfrac{z - z_s(x,y)}{H - z_s(x,y)} \)。

### 5.2 关键类型

~~~cpp
namespace vibe::grid {

struct Index3 { Int i, j, k; };

enum class Stagger { Cell, FaceX, FaceY, FaceZ, Corner };

// 网格几何：只描述几何，不含物理
struct Geometry {
  Int  nx, ny, nz;          // 内部点数（不含 halo）
  Int  halo;                // halo 宽度，默认 4
  Real x0, y0;              // 左下角（米）
  Real z_top;               // 模式顶（米）
  Real dx, dy;              // 水平分辨率（可变分辨率网格为最小分辨率）
  std::vector<Real> zeta;   // 垂直层坐标 (nz+1)
  std::vector<Real> zs;     // 地形高度 (nx*ny)，可为 nullptr
  // 变分辨率：每个单元的实际宽度 / 雅可比
  std::vector<Real> dx_cell, dy_cell, jacobian;
};

// 网格描述 + 拓扑（halo、邻居、并行分解）
class Grid {
public:
  const Geometry& geom() const noexcept;
  Int  size(Stagger s) const noexcept;      // 该错位网格的总点数
  Index3 index(Stagger s, Int i, Int j, Int k) const noexcept;
  const std::vector<Real>& metric(Stagger s) const;  // 度量系数
};

} // namespace vibe::grid
~~~

### 5.3 场（Field）

```cpp
namespace vibe::grid {

template <class T>
class FieldT {
public:
  FieldT() = default;
  FieldT(const Grid& g, Stagger s, std::string name);
  T&       at(Int i, Int j, Int k);          // 含 halo 的本地索引
  const T& at(Int i, Int j, Int k) const;
  T*       data();                            // 连续内存，CPU 或设备
  const T* data() const;
  Int      size() const;
  Stagger  stagger() const;
  const std::string& name() const;
  void     fill(T v);
  void     axpy(T a, const FieldT& x, T b);   // *this = a*x + b*(*this)
};

template <class T> using Field = FieldT<T>;

} // namespace vibe::grid
~~~

`Field` 的存储是 **single-array + halo**（不做多块 patch），并行分解由 MPI 子域负责；
GPU 后端下相同的内存布局同时可被主机与设备访问（unified 或显式 mirror，见 gpu 模块）。

### 5.4 嵌套

~~~cpp
namespace vibe::grid {

// Davies (1976) 松弛系数：外层 n 点内从 0 平滑升到 1
std::vector<Real> davies_profile(Int n_zone, Real alpha_max = 1.0);

class Nest {
public:
  Int  ratio() const noexcept;                 // 细化比，通常 3
  const Grid& grid() const noexcept;           // 子域网格
  void  interpolate_from_parent(const Field<Real>& src, Field<Real>& dst) const;
  void  restrict_to_parent  (const Field<Real>& src, Field<Real>& dst,
                             Real weight = 1.0) const;   // 守恒面积加权
  void  apply_lateral_boundary(std::vector<Field<Real>>& state) const;
};

class NestHierarchy {                          // 双向嵌套管理
public:
  void add_level(std::unique_ptr<Nest> n);
  void exchange(Real dt);                      // 一次父子交换（时间子循环）
};

} // namespace vibe::grid
~~~

**守恒约束**：`restrict_to_parent` 对质量类变量必须满足
\( \sum_{\text{children}} \rho_c \Delta V_c = \rho_p \Delta V_p \)，
即细化比 \( r \) 时权重为 \( 1/r^2 \)（水平）或 \( 1/r^2 \)（垂直已单独处理）。

### 5.5 变分辨率网格

~~~cpp
namespace vibe::grid {

struct RefinementFunction {
  Real h_min;                     // 最小分辨率 (m)
  Real h_max;                     // 最大分辨率 (m)
  Real (*density)(Real x, Real y);// 目标密度函数，可为 nullptr（用解析拉伸）
};

// 由目标密度构造单调映射 x -> xi，保证单元不翻转且光滑
class VarResMap {
public:
  explicit VarResMap(RefinementFunction f, Real Lx, Real Ly, Int nhint);
  Real coordinate(Real xi) const;         // 计算坐标 -> 物理坐标
  Real jacobian(Real xi) const;           // dx/dxi（度量项）
};

} // namespace vibe::grid
~~~

变分辨率网格与嵌套网格在动力学代码里**共用同一套度量项接口**，因此
`dyn` 层不需要知道当前是嵌套还是变分辨率。

---

## 6. 动力学与时间推进契约

### 6.1 预报状态

~~~cpp
namespace vibe::dyn {

struct State {
  grid::Field<Real> u, v, w;        // 风 (FaceX/FaceY/FaceZ)
  grid::Field<Real> rho;            // 干空气密度 (Cell)
  grid::Field<Real> theta;          // 位温 (Cell)
  grid::Field<Real> pi;             // Exner 全量（pi0 + pi'），见 state.hpp 的说明
  grid::Field<Real> qv;             // 水汽 (Cell)
  grid::Field<Real> qc, qr, qi, qs, qg;
  Real time = 0.0;

  State() = default;
  State(const grid::Grid& g);       // 按格点分配全部场
  State clone() const;
  void  axpy(Real a, const State& x, Real b);   // *this = a*x + b*(*this)
  Real  norm2() const;
  std::vector<Real> pack() const;               // 打包为 1D（供 4D-Var）
  void  unpack(const std::vector<Real>& v);
};

} // namespace vibe::dyn
~~~

### 6.2 参考态

~~~cpp
namespace vibe::dyn {

// 静力平衡、水平均匀、时间不变（Klemp & Wilhelmson 1978）
class ReferenceState {
public:
  static ReferenceState isothermal(Real theta0, Real p_surf, Real z_top, Int nz);
  static ReferenceState from_profile(const std::vector<Real>& z,
                                     const std::vector<Real>& theta);
  const grid::Field<Real>& rho0() const;
  const grid::Field<Real>& theta0() const;
  const grid::Field<Real>& pi0() const;
};

} // namespace vibe::dyn
~~~

### 6.3 趋势与方程

~~~cpp
namespace vibe::dyn {

struct Tendency {          // 与 State 同布局的 d/dt
  grid::Field<Real> u, v, w, rho, theta, pi, qv, qc, qr, qi, qs, qg;
  Tendency() = default;
  explicit Tendency(const grid::Grid& g);
  void zero();
};

// 方程装配器：只负责非线性项，线性声波/重力波项交给 timeint
class Equations {
public:
  Equations(const grid::Grid& g, const ReferenceState& ref,
            const config::ModelConfig& cfg);
  void tendencies(const State& s, Tendency& d) const;   // 大时间步趋势
  void acoustic_tendencies(const State& s, Tendency& d) const;
  void diagnose(State& s) const;                        // pi, 诊断量
private:
  Advection  adv_;
  Coriolis   cor_;
  Damping    damp_;
};

// 平流（可选 2/4/6 阶中心或 5 阶 WENO）
class Advection {
public:
  Advection(const grid::Grid& g, int order, bool weno);
  void scalar(const grid::Field<Real>& q, const grid::Field<Real>& u,
              const grid::Field<Real>& v, const grid::Field<Real>& w,
              const grid::Field<Real>& rho, grid::Field<Real>& dqdt) const;
  void momentum(const State& s, Tendency& d) const;
};

} // namespace vibe::dyn
~~~

### 6.4 时间推进接口

~~~cpp
namespace vibe::timeint {

struct StepContext {
  Real dt;           // 大时间步
  Real dt_acoustic;  // 声波子步（<= dt）
  int  acoustic_substeps;
  Real time;
};

// 统一的单步接口：所有积分器实现它
class Integrator {
public:
  virtual ~Integrator() = default;
  virtual void step(dyn::State& s, const StepContext& ctx) = 0;
  virtual const char* name() const noexcept = 0;
};

// 1) Wicker & Skamarock (2002) RK3 + 声波子步（HEVI 的显式版本）
class RungeKutta3Integrator final : public Integrator { ... };

// 2) 半隐式（Crank-Nicolson 处理线性声波-重力波）-> 3D Helmholtz
class SemiImplicitIntegrator final : public Integrator {
public:
  SemiImplicitIntegrator(..., std::unique_ptr<HelmholtzSolver> solver);
  void step(dyn::State& s, const StepContext& ctx) override;
};

// 3D Helmholtz:  (div(a grad) - b) pi' = f
class HelmholtzSolver {
public:
  virtual void solve(const grid::Field<Real>& a,
                     const grid::Field<Real>& b,
                     const grid::Field<Real>& f,
                     grid::Field<Real>& x) = 0;
  virtual int  iterations() const noexcept = 0;
  virtual Real residual() const noexcept = 0;
};
std::unique_ptr<HelmholtzSolver> make_helmholtz_solver(HelmholtzKind kind,
                                                       const grid::Grid& g,
                                                       const config::ModelConfig& cfg);
enum class HelmholtzKind { VerticalTridiagonal, KrylovMultigrid, KrylovJacobi };

} // namespace vibe::timeint
~~~

**数值约束**：积分器必须满足

1. 线性声波-重力波在隐式格式下**无条件稳定**（放大因子模 ≤ 1）；
2. 守恒形式平流在周期边界下保持 \( \int \rho q\, dV \) 到机器精度量级；
3. 大时间步 CFL 由水平风决定，垂直声波 CFL 只由 `dt_acoustic` 决定。

---

## 7. GPU 与混合精度契约（include/vibe/gpu/）

~~~cpp
namespace vibe::gpu {

enum class Backend { CPU, CUDA, HIP, SYCL };

struct DeviceInfo { Backend backend; int id; std::string name; std::size_t memory_bytes; };
std::vector<DeviceInfo> enumerate_devices();
Backend active_backend() noexcept;

// 内存：主机/设备统一接口
template <class T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(Size n);
  T*       device_ptr();
  T*       host_ptr();          // 需要时自动同步
  Size     size() const;
  void     copy_to_device();
  void     copy_to_host();
  ~DeviceBuffer();
};

class Stream { ... };
struct LaunchConfig { dim3 grid, block; Size shared_bytes; Stream* stream; };

// 所有内核通过类型擦除的启动器调用，禁止在业务代码里写 <<< >>>
template <class Kernel, class... Args>
void launch(Kernel k, const LaunchConfig& cfg, Args&&... args);

} // namespace vibe::gpu
~~~

混合精度策略：

~~~cpp
namespace vibe::gpu {

enum class Precision { FP16, BF16, FP32, FP64 };

// 角色化精度：存储（状态）、计算（趋势）、累加（归约）
struct PrecisionPolicy {
  Precision storage  = Precision::FP64;   // 守恒量必须 FP64
  Precision compute  = Precision::FP32;   // 默认单精度算趋势
  Precision reduce   = Precision::FP64;   // Kahan/Neumaier 累加
  bool      compensated_summation = true;
  bool      iterative_refinement  = true; // Helmholtz 解的双精度 refinement
};

template <class T> struct CompensatedSum { T sum, c; void add(T x); T value() const; };

} // namespace vibe::gpu
~~~

对应内核（在 `include/vibe/gpu/kernels.hpp` 声明，`src/gpu/cuda/*.cu` 与
`src/gpu/cpu/*.cpp` 各给一份实现）：

~~~cpp
namespace vibe::gpu {
void advect_scalar(const grid::Field<Real>& q, ..., grid::Field<Real>& dqdt);
void advect_momentum(...);
void diffusion(...);
void tridiagonal_solve(std::vector<Real>& a, std::vector<Real>& b,
                       std::vector<Real>& c, std::vector<Real>& d,
                       std::vector<Real>& x);
void helmholtz_residual(...);
void restrict_2to1(...);
void prolong_1to2(...);
void bilinear_interp(...);      // 观测算子用
void trilinear_interp(...);
}
~~~

> **实现注记（2025 修订）**：上表中的 `grid::Field<Real>&` 是**语义占位**。
> 实际实现采用「裸指针 + `FieldShape` + `KernelGeometry`」的参数形式：
> `Field` 是模板类型，无法跨主机/设备边界传递；而纯裸指针又缺少形状信息。
> 最终签名见 [include/vibe/gpu/kernels.hpp](../../include/vibe/gpu/kernels.hpp)：
>
> ~~~cpp
> void advect_scalar(const Real* q, const Real* u, const Real* v, const Real* w,
>                    const Real* rho, Real* dqdt, const FieldShape& shape,
>                    const KernelGeometry& geom, AdvectionOptions opt);
> ~~~
>
> 上游业务代码仍然**不写任何内核启动语法**：CPU 后端通过
> `for_each_index_1d/3d[_strided]` 并行，CUDA/HIP 后端在 `.cu` 内使用
> 网格跨步循环。`git` 层的这一约定由 `vibe::gpu::kernels.hpp` 唯一承载。

---

## 8. 观测与同化契约（include/vibe/obs/, include/vibe/da/）

### 8.1 观测

~~~cpp
namespace vibe::obs {

enum class ObsType {
  Radiosonde, Surface, Aircraft, AMV, Scatterometer,
  GnssRo, Radiance, RadarReflectivity, Profiler
};

struct Observation {
  ObsType  type;
  Real     time;                 // 相对同化窗口起点（秒）
  Real     x, y, z;              // 物理坐标（米）
  int      variable;             // 见 VarKind
  Real     value;                // 观测值
  Real     sigma;                // 观测误差标准差
  Real     bias_correction = 0;  // 已估偏差
  int      qc_flag = 0;          // 0 = 可用
  Int      station_id = -1;
};

enum class VarKind { U, V, W, T, Q, PS, P, Radiance, Refractivity, Reflectivity };

struct ObsSpace {
  std::vector<Observation> obs;
  Real window_start = 0, window_length = 0;
  std::vector<Real> r_inv;       // 1/sigma^2
  Size size() const;
};

} // namespace vibe::obs
~~~

### 8.2 观测算子（含 TL/AD）

~~~cpp
namespace vibe::obs {

// 所有观测算子必须提供三者：非线性、切线性、伴随
class ObservationOperator {
public:
  virtual ~ObservationOperator() = default;
  virtual const char* name() const noexcept = 0;
  virtual void apply  (const dyn::State& x, const ObsSpace& obs,
                       std::vector<Real>& y) const = 0;
  virtual void applyTL(const dyn::State& x, const dyn::State& dx,
                       const ObsSpace& obs, std::vector<Real>& dy) const = 0;
  virtual void applyAD(const dyn::State& x, const std::vector<Real>& dy,
                       const ObsSpace& obs, dyn::State& dx) const = 0;
};

std::unique_ptr<ObservationOperator> make_operator(obs::ObsType t,
                                                   const grid::Grid& g,
                                                   const config::DaConfig& cfg);

} // namespace vibe::obs
~~~

### 8.3 4D-Var

~~~cpp
namespace vibe::da {

// 控制变量变换 B = U U^T
class ControlVariableTransform {
public:
  virtual void to_state  (const Vector& v, dyn::State& dx) const = 0;
  virtual void from_state(const dyn::State& dx, Vector& v) const = 0;
  virtual void applyB    (const Vector& v, Vector& Bv) const = 0;
  virtual void applyBinv (const Vector& v, Vector& Binvv) const = 0;
};

class CostFunction {                  // 增量代价函数
public:
  Real value(const Vector& v) const;
  void gradient(const Vector& v, Vector& g) const;   // 需要 AD 模式
};

class Minimizer {                     // L-BFGS / CG / Lanczos
public:
  virtual void minimize(CostFunction& J, Vector& x) = 0;
  virtual int  iterations() const noexcept = 0;
};

class Incremental4DVar {
public:
  Incremental4DVar(..., int n_outer = 2, int n_inner = 50);
  void run(const obs::ObsSpace& obs, dyn::State& x_b, dyn::State& x_a);
};

// 切线性/伴随自检（Liu & Nocedal 的点积检验）
Real check_adjoint (...);
Real check_tangent (...);

} // namespace vibe::da
~~~

---

## 9. 检验契约（include/vibe/verify/）

~~~cpp
namespace vibe::verify {

struct ContingencyTable { Real hits, misses, false_alarms, correct_negatives; };
struct Scores {
  Real bias, mae, rmse, correlation, anomaly_correlation;
  Real pod, far, csi, ets, frequency_bias, odds_ratio_skill;
  Real brier, brier_skill, crps, roc_auc;
  Real fss, spread, spread_skill_ratio;
  std::size_t n;
};
Scores compute_continuous(const std::vector<Real>& f, const std::vector<Real>& o,
                          const std::vector<Real>* clim = nullptr);
Scores compute_categorical(const std::vector<Real>& f, const std::vector<Real>& o,
                           Real threshold);
Scores compute_probabilistic(const std::vector<Real>& p,
                             const std::vector<Real>& o);
Scores compute_fractional_skill(const std::vector<Real>& f,
                                const std::vector<Real>& o,
                                int nx, int ny, Real threshold,
                                int neighborhood_radius);

} // namespace vibe::verify
~~~

---

## 10. 配置与 IO 契约

### 10.1 配置

~~~cpp
namespace vibe::config {

struct ModelConfig {
  struct Domain { Int nx, ny, nz; Real dx, dy, z_top; std::string terrain_file; };
  struct Time   { Real dt, run_length; Int acoustic_substeps; std::string integrator; };
  struct Numerics { int advection_order; bool weno; Real divergence_damping; };
  struct Nesting { bool enabled; Int ratio; Int levels; bool two_way; };
  struct Physics { std::string microphysics, radiation, pbl, surface, cumulus; };
  struct Parallel { Int px, py; };
  Domain domain; Time time; Numerics numerics; Nesting nesting;
  Physics physics; Parallel parallel;
};
struct DaConfig { ... };
// 注：vibe::verify 另外定义了一个同名的轻量评分配置（见
// include/vibe/verify/verification.hpp），使 verify 模块可以独立于 config 使用。
// 用户面向的配置对象始终是 vibe::config::VerifyConfig。
struct VerifyConfig { ... };

ModelConfig  load_model_config (const std::string& path);
DaConfig     load_da_config    (const std::string& path);
VerifyConfig load_verify_config(const std::string& path);
ModelConfig  load_model_config(/* 层叠覆盖 */ const std::vector<std::string>& paths);

} // namespace vibe::config
~~~

YAML 采用内置的**极简解析器**（`src/config/mini_yaml.cpp`），支持：缩进映射、`-` 序列、
标量、引号、行内 `[a,b]`/`{a: b}`、注释。刻意不支持锚点/别名/多文档，以避免外部依赖。

### 10.2 IO

~~~cpp
namespace vibe::io {

class FieldWriter {                 // 仅此层允许出现 netCDF
public:
  virtual void open(const std::string& path, const grid::Grid& g) = 0;
  virtual void write(const std::string& name, const grid::Field<Real>& f) = 0;
  virtual void close() = 0;
};
std::unique_ptr<FieldWriter> make_netcdf_writer();
std::unique_ptr<FieldWriter> make_binary_writer();

void write_restart(const std::string& path, const dyn::State& s, Real t);
void read_restart (const std::string& path, dyn::State& s);

} // namespace vibe::io
~~~

---

## 11. 构建选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `VIBE_ENABLE_MPI` | ON | 域分解并行 |
| `VIBE_ENABLE_OPENMP` | ON | 共享内存并行 |
| `VIBE_ENABLE_CUDA` | OFF | CUDA 后端 |
| `VIBE_ENABLE_HIP` | OFF | HIP 后端 |
| `VIBE_ENABLE_SYCL` | OFF | SYCL 后端 |
| `VIBE_ENABLE_NETCDF` | ON | NetCDF 输出 |
| `VIBE_ENABLE_4DVAR` | ON | 同化模块 |
| `VIBE_PRECISION` | `double` | `double` / `single` / `mixed` |
| `VIBE_BUILD_TESTS` | ON | 测试 |
| `VIBE_BUILD_PYTHON` | ON | 安装 Python 包 |

目标命名：`vibe::common`、`vibe::grid`、`vibe::dyn`、`vibe::timeint`、
`vibe::physics`、`vibe::gpu`、`vibe::io`、`vibe::da`、`vibe::obs`、`vibe::verify`、
`vibe::driver`；聚合接口目标 `vibe::model`。

---

## 12. 编码规范

1. C++20；头文件用 `#pragma once`；公开头文件全部在 `include/vibe/`。
2. 命名：类型 `CamelCase`，函数/变量 `snake_case`，常量 `kCamelCase`，
   私有成员 `trailing_underscore_`。
3. 每个数值例程上方必须有注释块：**公式（LaTeX）+ 离散化 + 文献引用 + 复杂度**。
4. 定点循环禁止虚函数调用；物理参数化可以虚函数。
5. 禁止 `new/delete`，容器用 `std::vector`；GPU 缓冲用 `gpu::DeviceBuffer`。
6. 单元测试文件与源文件同名：`test_<file>.cpp`，测试框架为内置的 `vibe::test`（header-only）。
7. Python：类型注解 + `numpy` 向量化；绘图依赖 `matplotlib` 为可选 extras。

8. **C 与 C++ 的分工**：框架、数据结构、异常、配置、IO 与算法组织一律用 C++20；
   只有「数组进、数组出」的紧循环允许写成纯 C11，统一放在
   [include/vibe/common/c_kernels.h](../../include/vibe/common/c_kernels.h) 与
   [src/common/c_kernels.c](../../src/common/c_kernels.c)，用 extern "C" 暴露给 C++。
   采用 C 的三条理由：可被 FORTRAN / Python(ctypes) 直接调用、便于与外部模式耦合、
   部分工具链的向量化诊断更成熟。**C 文件中禁止出现全局可变状态**，
   必须可重入且线程安全（约定见该头文件）。

---

## 13. 参考文献

见 [references.md](references.md)，包含动力学、半隐式、嵌套、变分辨率、GPU 混合精度、
4D-Var、切线性伴随、误差检验七组文献，正文公式均标注对应编号。

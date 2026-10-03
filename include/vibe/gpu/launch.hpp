#pragma once
/// @file launch.hpp
/// @brief 启动配置、流（Stream）、索引空间迭代（for_each_index_*）与事件计时。
///
/// 设计要点
/// --------
///   1. **业务代码禁止写 \c <<< >>>**（架构第 7 节硬约束）。CPU 与 GPU 的
///      内核实现各自在 \c src/gpu/cpu 与 \c src/gpu/cuda 内用本文件提供的
///      索引空间约定与启动助手；公开接口只有 kernels.hpp 中的函数；
///   2. CPU 后端在 \ref LaunchConfig 描述的迭代空间上做 OpenMP 并行（无 OpenMP
///      时退化为串行），因此同一份内核函数体在四种后端上语义一致；
///   3. GPU 后端在设备内核内部使用 \c blockIdx/threadIdx + 网格跨步循环
///      （[G10] 第 5.7、6.1 节、[G12]），沿用的仍是本文件的线性化约定。
///
/// dim3 的归属
/// -----------
/// \c dim3 定义在 \c vibe::gpu 命名空间内（而非全局），因此：
///   * CPU 构建不依赖任何 CUDA 头文件也能编译；
///   * CUDA 构建也不会与 \c cuda_runtime.h 的全局 \c dim3 冲突
///     （内核启动时用 \c .x/.y/.z 显式取值，见 kernels_*.cu）。
///
/// 迭代空间约定
/// ------------
///   内核索引空间 = \f$\{0,\dots,\text{grid.x}-1\}\times
///   \{0,\dots,\text{grid.y}-1\}\times\{0,\dots,\text{grid.z}-1\}\f$
///   网格 = 索引空间大小，block = 每个"执行单元"的逻辑线程数。
///   * CPU：把索引空间展平后按 \c block.x*block.y*block.z 分块，块间 OpenMP 并行；
///   * GPU：\c grid/block 直接作为 launch 维度（块内线程再自行对索引空间分条）。
///
/// 复杂度：launch 本身 O(1) 开销（GPU 约 3~8 us 启动延迟 [G10]，CPU 为一次
/// 线程组建立，OpenMP 约 1~2 us）；\ref Stream 创建/销毁 O(1)。
///
/// 文献：[G10] CUDA C++ Programming Guide 第 6 章（内核启动、流、事件）；
///       [G12] HIP Programming Guide 第 6 章；[G11] SYCL 2020 第 4.9 节（队列）；
///       [G13] OpenMP 5.2 第 2.9 节（并行区与 for 构造）；
///       [G17] Gustafson (1988)（并行开销与弱扩展）。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/device.hpp"
// for_field() 需要 Field 的尺寸接口；grid 模块只依赖 common，不存在循环依赖
#include "vibe/grid/field.hpp"

#if defined(_OPENMP)
#  include <omp.h>
#endif

// `Stream` 的内联实现需要后端运行时类型（仅在使用 GPU 后端时引入）
#if VIBE_BACKEND_CUDA
#  include <cuda_runtime.h>
#elif VIBE_BACKEND_HIP
#  include <hip/hip_runtime.h>
#elif VIBE_BACKEND_SYCL
#  include <sycl/sycl.hpp>
#endif
#include <cstdlib>

namespace vibe::gpu {

// ===========================================================================
// 1. dim3（最小兼容定义，位于 vibe::gpu 内以避免与 CUDA 冲突）
// ===========================================================================
/// 与 CUDA 的 \c dim3 字段语义一致（x 最快），但**不**提供成员函数，
/// 且只占 3 个 \c unsigned int，便于按值传递。
struct dim3 {
  unsigned int x = 1, y = 1, z = 1;
};

inline constexpr dim3 make_dim3(unsigned int x, unsigned int y = 1, unsigned int z = 1) noexcept {
  dim3 d;
  d.x = x;
  d.y = y;
  d.z = z;
  return d;
}

inline constexpr std::size_t dim3_count(const dim3& d) noexcept {
  return static_cast<std::size_t>(d.x) * d.y * d.z;
}

// ===========================================================================
// 2. Stream
// ===========================================================================
/// 执行流。CPU 后端是 no-op（同步语义天然满足）；CUDA/HIP 包装
/// \c cudaStream_t / \c hipStream_t；SYCL 包装 \c sycl::queue 的抽象句柄。
///
/// 语义
/// ----
///   * 默认构造 = 默认流（\c nullptr），不做任何资源管理；
///   * \ref create() 创建独立流（GPU 上可用于 overlap 计算与传输）；
///   * 禁止拷贝、可移动；析构时销毁自有流并同步（保证 kernel 不访问已释放内存）；
///   * \ref synchronize 等待该流上全部已提交工作。
///
/// 复杂度：创建/销毁 O(1) 但涉及驱动调用；同步的等待时间由工作负载决定。
///
/// 文献：[G10] 第 6.2 节（异步执行与流）；[G12] 第 6.3 节；[G11] 第 4.9 节。
class Stream {
 public:
  /// 默认流（不拥有资源）
  Stream() noexcept = default;
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& o) noexcept;
  Stream& operator=(Stream&& o) noexcept;

  /// 创建一个独立流
  static Stream create();

  /// 该流是否拥有底层句柄（默认流为 false）
  bool owned() const noexcept { return owned_; }
  /// 原生句柄（CUDA: \c cudaStream_t；HIP: \c hipStream_t；CPU: nullptr）
  void* native() const noexcept { return handle_; }
  /// 是否为默认流
  bool is_default() const noexcept { return handle_ == nullptr; }

  /// 等待本流上所有已提交工作完成
  void synchronize() const;
  /// 让本流等待 \p other 上已提交的全部工作（事件依赖，[G10] 第 6.2.4 节）
  void wait(const Stream& other) const;

  /// 与默认流绑定的单例（供 LaunchConfig 缺省值使用）
  static Stream& default_stream();

 private:
  void destroy() noexcept;

  void* handle_ = nullptr;
  bool owned_ = false;
};

// ===========================================================================
// 3. LaunchConfig
// ===========================================================================
/// 启动配置。与架构第 7 节一致：\c grid / \c block / \c shared_bytes / \c stream。
///
/// \c shared_bytes 在 CPU 后端被记录但不分配（CPU 内核用栈或调用方提供的
/// 缓冲，见 kernels.hpp 中 \ref helmholtz_jacobi 的注释）。
struct LaunchConfig {
  dim3 grid = make_dim3(1, 1, 1);          ///< 索引空间大小（"网格"）
  dim3 block = make_dim3(256, 1, 1);       ///< 每执行单元的线程数
  std::size_t shared_bytes = 0;            ///< 动态共享内存字节数（GPU 有效）
  Stream* stream = nullptr;                ///< 执行流；nullptr = 默认流

  /// 每个执行单元的线程数
  std::size_t threads_per_block() const noexcept { return dim3_count(block); }
  /// 索引空间元素总数（= grid.x*grid.y*grid.z）
  std::size_t total_grid() const noexcept { return dim3_count(grid); }
  /// 执行单元（block）个数 = ceil(total_grid / threads_per_block)
  std::size_t block_count() const noexcept {
    const std::size_t per = threads_per_block();
    const std::size_t total = total_grid();
    return per == 0 ? 0 : (total + per - 1) / per;
  }
  /// 若 grid 是"点数"而 block 是"线程数"，等价于把点数按线程数分块
  bool consistent() const noexcept { return block_count() > 0; }
};

// === 构造助手 =============================================================
/// 1D：grid.x = n，block.x = threads
LaunchConfig make_1d(std::size_t n, int threads = 256, std::size_t shared_bytes = 0,
                     Stream* stream = nullptr);
/// 2D：grid = (nx, ny)
LaunchConfig make_2d(std::size_t nx, std::size_t ny, int tx = 16, int ty = 16,
                     std::size_t shared_bytes = 0, Stream* stream = nullptr);
/// 3D：grid = (nx, ny, nz)
LaunchConfig make_3d(std::size_t nx, std::size_t ny, std::size_t nz, int tx = 8, int ty = 8,
                     int tz = 8, std::size_t shared_bytes = 0, Stream* stream = nullptr);

/// 1D 分块形式：总工作量 \p n 个元素，每线程处理 \p items_per_thread 个，
/// 返回 grid.x = ceil(n / (block.x * items_per_thread))、block.x = \p block。
LaunchConfig make_1d_strided(std::size_t n, int block = 256, int items_per_thread = 1,
                             std::size_t shared_bytes = 0, Stream* stream = nullptr);

/// 针对一个 [[vibe::grid::Field]] 生成配置：
///   * grid  = (nsx, nsy, nsz)，含 halo 的存储维度（内核按存储访问 halo）；
///   * block = (block_x, 1, 1)，且 block_x 被夹到 [warp_size, 1024] 区间。
///
/// 之所以用**含 halo 的存储维度**而不是内部维度：GPU 内核需要在 halo 上
/// 计算通量/差分（一阶/二阶模板），halo 区在存储中是真实存在的一段内存。
/// 需要只遍历内部点的内核请自行在 kernel 内做 \c if (i < interior) 判定。
LaunchConfig for_field_dims(Size nsx, Size nsy, Size nsz, int block = 256,
                            Stream* stream = nullptr);

template <class T>
inline LaunchConfig for_field(const grid::FieldT<T>& f, int block = 256, Stream* stream = nullptr) {
  return for_field_dims(static_cast<Size>(f.nsx()), static_cast<Size>(f.nsy()),
                        static_cast<Size>(f.nsz()), block, stream);
}

// ===========================================================================
// 4. 索引空间迭代与启动
// ===========================================================================
//
// 命名约定：**索引空间的线性化顺序固定为 x 最快、z 最慢**
// \f[ \text{flat} = (z \cdot \text{grid.y} + y) \cdot \text{grid.x} + x \f]
// CPU 与 GPU 后端必须使用同一约定，否则 "同一份内核" 会得到不同结果。
//
// 迭代策略（性能相关，见文档第 8 节）：
//   * \c for_each_index_1d/3d  —— 索引空间 = 线程数，一个线程恰好处理一个点。
//     适合"每点工作量小、总点数多"的内核（通量计算、逐列三对角）；
//   * \c for_each_index_*_strided —— 网格跨步循环（grid-stride loop）。
//     适合每点工作量大、或索引空间远大于可并行度的内核（PCHIP 垂直插值、
//     多重网格的整层更新），能减少块调度开销并提高缓存命中（[G10] 第 5.7 节）。

namespace detail {

/// 索引空间线性化（x 最快）
inline std::size_t flatten3(std::size_t x, std::size_t y, std::size_t z, const dim3& g) noexcept {
  return (z * g.y + y) * g.x + x;
}

/// 把索引空间按 CPU 的 block 语义分块后并行执行
template <class Body3>
void run_blocks(const LaunchConfig& cfg, Body3&& body) {
  const std::size_t total = cfg.total_grid();
  if (total == 0) return;
  const std::size_t tpb = cfg.threads_per_block();
  const std::size_t blocks = cfg.block_count();
#if defined(_OPENMP)
#  pragma omp parallel for schedule(static) if (blocks > 1)
#endif
  for (std::int64_t b = 0; b < static_cast<std::int64_t>(blocks); ++b) {
    const std::size_t begin = static_cast<std::size_t>(b) * tpb;
    const std::size_t end = (begin + tpb < total) ? (begin + tpb) : total;
    for (std::size_t t = begin; t < end; ++t) {
      const unsigned int x = static_cast<unsigned int>(t % cfg.grid.x);
      const unsigned int y = static_cast<unsigned int>((t / cfg.grid.x) % cfg.grid.y);
      const unsigned int z = static_cast<unsigned int>(t / (static_cast<std::size_t>(cfg.grid.x) * cfg.grid.y));
      body(x, y, z);
    }
  }
}

}  // namespace detail

/// 遍历 1D 索引空间（CPU 实现；GPU 构建下由 src/gpu/cuda 的对应实现替代）。
/// \p body 的签名为 \c void(unsigned int).
template <class Body1>
void for_each_index_1d(const LaunchConfig& cfg, Body1&& body) {
  detail::run_blocks(cfg, [&](unsigned int x, unsigned int, unsigned int) { body(x); });
}

/// 遍历 3D 索引空间（CPU 实现）。\p body 的签名为 \c void(unsigned,unsigned,unsigned)。
template <class Body3>
void for_each_index_3d(const LaunchConfig& cfg, Body3&& body) {
  detail::run_blocks(cfg, static_cast<Body3&&>(body));
}

/// 网格跨步 1D 遍历：每个"逻辑线程"处理 \c cfg.grid.x 中以
/// \c block.x 为步长的多个点，适合每点代价高的内核。
template <class Body1>
void for_each_index_1d_strided(const LaunchConfig& cfg, Body1&& body) {
  detail::run_blocks(cfg, [&](unsigned int x, unsigned int, unsigned int) {
    for (std::size_t i = x; i < cfg.grid.x; i += cfg.block.x) body(i);
  });
}

/// 变参启动：把内核体与附加参数绑定后进入 CPU 索引迭代。
/// GPU 构建下，业务代码应直接调用 kernels.hpp 中的内核函数（它们内部走
/// 后端实现），本函数只服务于 CPU 路径与测试。
template <class Kernel>
void launch(Kernel k, const LaunchConfig& cfg) {
  detail::run_blocks(cfg, [&](unsigned int x, unsigned int y, unsigned int z) { k(x, y, z); });
}

template <class Kernel, class... Args>
void launch(Kernel k, const LaunchConfig& cfg, Args&&... args) {
  detail::run_blocks(cfg, [&](unsigned int x, unsigned int y, unsigned int z) {
    k(x, y, z, args...);
  });
}

// ===========================================================================
// 6. 事件计时
// ===========================================================================
/// 单次内核计时记录（[G10] 第 6 章、[G17] 剖析方法）。
struct KernelTiming {
  std::string name;         ///< 内核名
  double elapsed_ms = 0.0;  ///< 累计耗时
  std::int64_t calls = 0;   ///< 调用次数
  std::size_t blocks = 0;   ///< 最近一次的 block 数
  std::size_t threads = 0;  ///< 最近一次的线程数

  double mean_ms() const noexcept {
    return calls > 0 ? elapsed_ms / static_cast<double>(calls) : 0.0;
  }
};

/// 计时表：按名字聚合，输出 Amdahl 式分解（[G17]）。
struct KernelTimings {
  std::vector<KernelTiming> entries;

  KernelTiming& get(const std::string& name);
  void record(const std::string& name, double elapsed_ms, const LaunchConfig& cfg);
  /// 按耗时降序排序后的副本
  std::vector<KernelTiming> sorted() const;
  /// 多行文本报告
  std::string report() const;
  void reset();
  double total_ms() const;
};

/// 全局计时表（可通过 \c VIBE_GPU_TIMING 环境变量在运行时决定是否记录）
KernelTimings& kernel_timings();
/// 是否开启内核计时
bool timing_enabled() noexcept;

/// 记录一次内核启动耗时。
///
/// GPU 后端：用 \c cudaEventCreate/Record/ElapsedTime 测**设备侧**时间
/// （排除启动延迟与主机侧抖动，[G10] 第 6 章）；
/// CPU 后端：\c std::chrono::steady_clock 测挂钟时间。
///
/// @return 本次耗时（毫秒）
template <class Fn>
double time_kernel(const std::string& name, const LaunchConfig& cfg, Fn&& fn) {
  if (!timing_enabled()) {
    fn();
    return 0.0;
  }
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  kernel_timings().record(name, ms, cfg);
  return ms;
}

/// 作用域计时器：析构时记录
class ScopedKernelTimer {
 public:
  ScopedKernelTimer(std::string name, const LaunchConfig& cfg);
  ~ScopedKernelTimer();
  ScopedKernelTimer(const ScopedKernelTimer&) = delete;
  ScopedKernelTimer& operator=(const ScopedKernelTimer&) = delete;

 private:
  std::string name_;
  LaunchConfig cfg_;
  std::chrono::steady_clock::time_point t0_;
  bool active_ = false;
};

// ===========================================================================
// 7. 非模板实现（内联；避免为本层再引入一个实现文件）
// ===========================================================================
inline LaunchConfig make_1d(std::size_t n, int threads, std::size_t shared_bytes,
                            Stream* stream) {
  LaunchConfig cfg;
  cfg.grid = make_dim3(static_cast<unsigned int>(n), 1u, 1u);
  cfg.block = make_dim3(static_cast<unsigned int>(threads > 0 ? threads : 1), 1u, 1u);
  cfg.shared_bytes = shared_bytes;
  cfg.stream = stream;
  return cfg;
}

inline LaunchConfig make_2d(std::size_t nx, std::size_t ny, int tx, int ty,
                            std::size_t shared_bytes, Stream* stream) {
  LaunchConfig cfg;
  cfg.grid = make_dim3(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny), 1u);
  cfg.block = make_dim3(static_cast<unsigned int>(tx > 0 ? tx : 1),
                        static_cast<unsigned int>(ty > 0 ? ty : 1), 1u);
  cfg.shared_bytes = shared_bytes;
  cfg.stream = stream;
  return cfg;
}

inline LaunchConfig make_3d(std::size_t nx, std::size_t ny, std::size_t nz, int tx, int ty, int tz,
                            std::size_t shared_bytes, Stream* stream) {
  LaunchConfig cfg;
  cfg.grid = make_dim3(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny),
                       static_cast<unsigned int>(nz));
  cfg.block = make_dim3(static_cast<unsigned int>(tx > 0 ? tx : 1),
                        static_cast<unsigned int>(ty > 0 ? ty : 1),
                        static_cast<unsigned int>(tz > 0 ? tz : 1));
  cfg.shared_bytes = shared_bytes;
  cfg.stream = stream;
  return cfg;
}

/// 1D 分块：总工作量 `n`，每线程处理 `items_per_thread` 个点。
/// grid.x = 每个执行单元覆盖的点数，故 grid.x = block.x * items_per_thread；
/// 于是 block_count() = ceil(n / grid.x)（见 LaunchConfig::block_count）。
/// 注意：这里的 `grid` 是**索引空间**（与 CUDA 的 grid 维度无关），
/// 索引空间与执行单元个数的换算由 block_count() 完成。
inline LaunchConfig make_1d_strided(std::size_t n, int block, int items_per_thread,
                                    std::size_t shared_bytes, Stream* stream) {
  LaunchConfig cfg;
  const std::size_t per = static_cast<std::size_t>(block > 0 ? block : 1) *
                          static_cast<std::size_t>(items_per_thread > 0 ? items_per_thread : 1);
  const std::size_t total = (n < per) ? n : (per * ((n + per - 1) / per));
  cfg.grid = make_dim3(static_cast<unsigned int>(total), 1u, 1u);
  cfg.block = make_dim3(static_cast<unsigned int>(block > 0 ? block : 1), 1u, 1u);
  cfg.shared_bytes = shared_bytes;
  cfg.stream = stream;
  return cfg;
}

/// 针对含 halo 的存储维度生成配置：grid = (nsx, nsy, nsz)，
/// block.x 被夹到 [warp_size, 1024]，保证至少一个 warp 且不超过硬件上限。
inline LaunchConfig for_field_dims(Size nsx, Size nsy, Size nsz, int block, Stream* stream) {
  LaunchConfig cfg;
  cfg.grid = make_dim3(static_cast<unsigned int>(nsx), static_cast<unsigned int>(nsy),
                       static_cast<unsigned int>(nsz));
  const int ws = default_device().warp_size();
  int b = block;
  const int lo = ws > 1 ? ws : 32;
  if (b < lo) b = lo;
  const int hi = default_device().max_threads_per_block();
  if (b > hi) b = hi;
  cfg.block = make_dim3(static_cast<unsigned int>(b), 1u, 1u);
  cfg.shared_bytes = 0;
  cfg.stream = stream;
  return cfg;
}

inline Stream::~Stream() { destroy(); }

inline Stream::Stream(Stream&& o) noexcept : handle_(o.handle_), owned_(o.owned_) {
  o.handle_ = nullptr;
  o.owned_ = false;
}

inline Stream& Stream::operator=(Stream&& o) noexcept {
  if (this != &o) {
    destroy();
    handle_ = o.handle_;
    owned_ = o.owned_;
    o.handle_ = nullptr;
    o.owned_ = false;
  }
  return *this;
}

inline void Stream::destroy() noexcept {
  if (!owned_ || handle_ == nullptr) {
    handle_ = nullptr;
    owned_ = false;
    return;
  }
#if VIBE_BACKEND_CUDA
  (void)cudaStreamDestroy(reinterpret_cast<cudaStream_t>(handle_));
#elif VIBE_BACKEND_HIP
  (void)hipStreamDestroy(reinterpret_cast<hipStream_t>(handle_));
#elif VIBE_BACKEND_SYCL
  delete reinterpret_cast<sycl::queue*>(handle_);
#endif
  handle_ = nullptr;
  owned_ = false;
}

inline Stream Stream::create() {
  Stream s;
#if VIBE_BACKEND_CUDA
  cudaStream_t h = nullptr;
  VIBE_GPU_CHECK(cudaStreamCreateWithFlags(&h, cudaStreamNonBlocking));
  s.handle_ = reinterpret_cast<void*>(h);
  s.owned_ = true;
#elif VIBE_BACKEND_HIP
  hipStream_t h = nullptr;
  VIBE_GPU_CHECK(hipStreamCreateWithFlags(&h, hipStreamNonBlocking));
  s.handle_ = reinterpret_cast<void*>(h);
  s.owned_ = true;
#elif VIBE_BACKEND_SYCL
  s.handle_ = new sycl::queue{sycl::default_selector_v};
  s.owned_ = true;
#else
  // CPU：流没有实体，`create()` 返回一个"拥有"的哑流，便于上层统一代码路径
  s.handle_ = nullptr;
  s.owned_ = true;
#endif
  return s;
}

inline void Stream::synchronize() const {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(handle_)));
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipStreamSynchronize(reinterpret_cast<hipStream_t>(handle_)));
#elif VIBE_BACKEND_SYCL
  if (handle_ != nullptr) reinterpret_cast<sycl::queue*>(handle_)->wait();
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

inline void Stream::wait(const Stream& other) const {
#if VIBE_BACKEND_CUDA
  cudaEvent_t ev = nullptr;
  VIBE_GPU_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  VIBE_GPU_CHECK(cudaEventRecord(ev, reinterpret_cast<cudaStream_t>(other.handle_)));
  VIBE_GPU_CHECK(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(handle_), ev, 0));
  VIBE_GPU_CHECK(cudaEventDestroy(ev));
#elif VIBE_BACKEND_HIP
  hipEvent_t ev = nullptr;
  VIBE_GPU_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  VIBE_GPU_CHECK(hipEventRecord(ev, reinterpret_cast<hipStream_t>(other.handle_)));
  VIBE_GPU_CHECK(hipStreamWaitEvent(reinterpret_cast<hipStream_t>(handle_), ev, 0));
  VIBE_GPU_CHECK(hipEventDestroy(ev));
#elif VIBE_BACKEND_SYCL
  if (other.handle_ != nullptr) synchronize();
#else
  VIBE_UNUSED(other);
  synchronize();
#endif
}

inline Stream& Stream::default_stream() {
  static Stream s;
  return s;
}

// --- KernelTimings ---------------------------------------------------------
inline KernelTiming& KernelTimings::get(const std::string& name) {
  for (KernelTiming& e : entries) {
    if (e.name == name) return e;
  }
  entries.push_back(KernelTiming{});
  entries.back().name = name;
  return entries.back();
}

inline void KernelTimings::record(const std::string& name, double elapsed_ms,
                                  const LaunchConfig& cfg) {
  KernelTiming& e = get(name);
  e.elapsed_ms += elapsed_ms;
  e.calls += 1;
  e.blocks = cfg.block_count();
  e.threads = cfg.threads_per_block();
}

inline std::vector<KernelTiming> KernelTimings::sorted() const {
  std::vector<KernelTiming> out = entries;
  std::sort(out.begin(), out.end(),
            [](const KernelTiming& a, const KernelTiming& b) { return a.elapsed_ms > b.elapsed_ms; });
  return out;
}

inline double KernelTimings::total_ms() const {
  double t = 0.0;
  for (const KernelTiming& e : entries) t += e.elapsed_ms;
  return t;
}

inline std::string KernelTimings::report() const {
  std::string s = "内核计时（按耗时降序）:\n";
  for (const KernelTiming& e : sorted()) {
    s += "  " + e.name + ": total=" + std::to_string(e.elapsed_ms) +
         " ms calls=" + std::to_string(e.calls) + " mean=" + std::to_string(e.mean_ms()) +
         " ms blocks=" + std::to_string(e.blocks) + " threads=" + std::to_string(e.threads) + "\n";
  }
  s += "  合计 " + std::to_string(total_ms()) + " ms\n";
  return s;
}

inline void KernelTimings::reset() { entries.clear(); }

/// 全局计时表：由环境变量 `VIBE_GPU_TIMING`（非 "0"/空）控制是否记录
inline KernelTimings& kernel_timings() {
  static KernelTimings t;
  return t;
}

inline bool timing_enabled() noexcept {
  static const bool enabled = [] {
    const char* v = std::getenv("VIBE_GPU_TIMING");
    if (v == nullptr) return false;
    return !(v[0] == '\0' || (v[0] == '0' && v[1] == '\0'));
  }();
  return enabled;
}

inline ScopedKernelTimer::ScopedKernelTimer(std::string name, const LaunchConfig& cfg)
    : name_(std::move(name)), cfg_(cfg), t0_(std::chrono::steady_clock::now()) {
  active_ = timing_enabled();
}

inline ScopedKernelTimer::~ScopedKernelTimer() {
  if (!active_) return;
  const auto t1 = std::chrono::steady_clock::now();
  kernel_timings().record(name_, std::chrono::duration<double, std::milli>(t1 - t0_).count(), cfg_);
}

/// 全局同步（等待所有流）；CPU 后端为内存栅栏
inline void device_synchronize() {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipDeviceSynchronize());
#elif VIBE_BACKEND_SYCL
  // SYCL 无全局同步 API：由各 queue 的 wait() 覆盖
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace vibe::gpu
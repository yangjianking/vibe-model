#pragma once
/// @file device.hpp
/// @brief 设备枚举、设备选择与设备属性查询。
///
/// 分层
/// ----
///   * \ref DeviceInfo —— 纯数据描述（POD），可跨 MPI rank 序列化写入日志；
///   * \ref enumerate_devices —— 后端相关的发现过程；
///   * \ref Device —— RAII 句柄，负责"选定当前设备"，析构时不做破坏性操作；
///   * \ref active_backend / \ref set_active_backend —— 全局当前后端（进程级）。
///
/// 为什么把"当前后端"做成进程级全局
/// ---------------------------------
/// 一个模式在一次运行里只用一个后端，但配置读取发生在设备选择之前，因此需要
/// 一个可查询、可设置的全局状态。多线程下该状态在启动阶段设置、运行阶段只读，
/// 故不加锁（见 \ref docs/design/05_gpu_hybrid_precision.md 第 1 节）。
///
/// 与 CUDA/HIP/SYCL 的关系
/// -----------------------
/// \ref enumerate_devices 在 CUDA 构建里调用 \c cudaGetDeviceProperties，
/// HIP 构建里调用 \c hipGetDeviceProperties，SYCL 构建里调用
/// \c sycl::device::get_devices；CPU 后端返回一个伪设备（id=0）。
/// 三种加速器后端的属性字段（\c compute_major/minor、\c multiprocessors）
/// 在 CPU 与 SYCL 下取保守值。
///
/// 文献：[G10] CUDA C++ Programming Guide 第 4 章（设备管理）与第 6 章（事件计时）；
///       [G12] HIP Programming Guide 第 5 章；[G11] SYCL 2020 第 4 章；
///       [G17] Gustafson (1988)（并行度与负载均衡）；[G18] Dennis & Edwards (2004)。

#include <cstddef>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/gpu/backend.hpp"

namespace vibe::gpu {

// ---------------------------------------------------------------------------
// 设备描述
// ---------------------------------------------------------------------------
/// 单设备的静态属性。所有字段在 \ref enumerate_devices 时一次填满，
/// 之后只读；因此可安全地在多线程间共享。
struct DeviceInfo {
  Backend backend = Backend::CPU;   ///< 后端类型
  int id = 0;                       ///< 后端内的设备序号（CPU 恒 0）
  std::string name;                 ///< 设备名（cudaDeviceProp::name 等）
  std::size_t memory_bytes = 0;     ///< 全局显存/内存字节数
  int compute_major = 0;            ///< 计算能力主版本（CPU 取 0）
  int compute_minor = 0;            ///< 计算能力次版本
  int multiprocessors = 0;          ///< 流多处理器 (SM/CU) 个数

  /// 派生的计算能力浮点数，便于与架构要求比较（如 >= 7.0 才有张量核，[G1]）
  int compute_capability() const noexcept { return compute_major * 10 + compute_minor; }
  /// 是否为支持 FP16 张量核的架构（Volta 及以上，[G1][G5]）
  bool has_tensor_cores() const noexcept {
    return (backend == Backend::CUDA || backend == Backend::HIP) && compute_capability() >= 70;
  }
  /// 一行摘要（日志/测试用）
  std::string describe() const;
};

/// 枚举当前编译后端下的全部设备。CPU 后端返回恰好 1 个伪设备，
/// \c memory_bytes 取进程可用物理内存的保守估计。
/// GPU 后端下若运行时初始化失败（驱动缺失/无可见设备），返回空向量，
/// **不抛异常**——是否致命由调用方决定（见 \ref Device 构造函数）。
std::vector<DeviceInfo> enumerate_devices();

/// 枚举指定后端的设备；若后端非当前编译后端，返回空向量。
std::vector<DeviceInfo> enumerate_devices(Backend backend);

// ---------------------------------------------------------------------------
// 全局当前后端
// ---------------------------------------------------------------------------
/// 当前激活后端（默认 = 编译期后端 \ref compiled_backend）
Backend active_backend() noexcept;

/// 设置当前后端。若目标后端非本编译产物支持（\ref is_compiled 为 false），
/// 抛出 \ref vibe::NotImplemented 并保持原值不变。
void set_active_backend(Backend b);

/// 当前后端的可见设备数（= \ref enumerate_devices 的大小，结果被缓存）
int device_count();

/// 重置内部缓存（切换后端后自动失效；此接口供测试使用）
void reset_device_cache();

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
/// 设备句柄：选定设备、查询占用率与 warp 尺寸、同步。
///
/// 语义
/// ----
///   * 构造时把设备设为"当前设备"（CPU 后端为空操作）；
///   * 析构**不**释放设备、不重置设备状态（与 CUDA 习惯一致）；
///   * 移动语义可转移"当前设备"的所有权标记，拷贝被禁止。
///
/// 线程安全：同一 \ref Device 对象不可跨线程共享；不同对象指向不同设备时可并行。
class Device {
 public:
  /// 使用当前后端的第 0 号设备；无可用设备时抛 \ref vibe::Error
  Device();
  /// 使用当前后端的第 \p id 号设备
  explicit Device(int id);
  /// 指定后端与设备号（会临时切换 \ref active_backend）
  Device(Backend backend, int id);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) noexcept;
  Device& operator=(Device&&) noexcept;

  // ---- 属性 ---------------------------------------------------------------
  const DeviceInfo& info() const noexcept { return info_; }
  Backend backend() const noexcept { return info_.backend; }
  int id() const noexcept { return info_.id; }
  const std::string& name() const noexcept { return info_.name; }
  std::size_t memory_bytes() const noexcept { return info_.memory_bytes; }
  int multiprocessors() const noexcept { return info_.multiprocessors; }
  bool valid() const noexcept { return valid_; }

  // ---- 查询 ---------------------------------------------------------------
  /// warp（CUDA）/ wavefront（AMD GCN、RDNA）/ sub-group（SYCL）宽度。
  /// CUDA 恒 32（[G10] 第 5.4 节）；HIP NVIDIA 为 32、AMD 为 64（[G12]）；
  /// CPU 后端返回 1（无锁步执行，仅用于统一归约代码的步长逻辑）。
  int warp_size() const noexcept;

  /// 归约树的最大共享内存字节数（CPU 后端返回 0，表示用栈/寄存器）
  std::size_t max_shared_memory() const noexcept;

  /// 每 block 的最大线程数（CPU 后端返回 1024，与 OpenMP 线程上限无关）
  int max_threads_per_block() const noexcept;

  /// 占用率估计：给定每线程寄存器数 \p regs_per_thread、每线程共享内存
  /// \p shared_per_block 与 block 线程数 \p threads_per_block，
  /// 返回 (活跃 warp 数) / (理论最大 warp 数)。CUDA 用
  /// \c cudaOccupancyMaxActiveBlocksPerMultiprocessor；
  /// CPU 后端用 Little 定律式的保守估计（受 available_parallelism 限制）。
  double occupancy(int regs_per_thread, std::size_t shared_per_block,
                   int threads_per_block) const;

  /// 每线程可用寄存器数上限（CPU 返回 0）
  int registers_per_thread() const noexcept;

  // ---- 同步 ---------------------------------------------------------------
  /// 等待设备上所有已提交工作完成。CPU 后端为内存栅栏 + no-op。
  void synchronize() const;

  /// 把当前设备设回本对象（多线程交替使用多设备时调用）
  void make_current() const;

 private:
  DeviceInfo info_{};
  bool valid_ = false;
};

/// 当前线程/进程的默认设备（按后端缓存；首次调用时构造）
Device& default_device();

// ---------------------------------------------------------------------------
// 便捷查询（等价于 default_device().xxx()）
// ---------------------------------------------------------------------------
int  warp_size();
std::size_t max_shared_memory();
int  max_threads_per_block();

}  // namespace vibe::gpu

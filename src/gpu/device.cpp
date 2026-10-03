/// @file device.cpp
/// @brief 设备枚举与选择的实现（四种后端各一段 \#if）。
///
/// 后端分支的组织
/// --------------
/// 每个公开函数内部只保留**一处** \#if 链，顺序固定为
/// CUDA -> HIP -> SYCL -> CPU，便于审查"某个后端是否被漏掉"。
/// 所有 GPU 运行时调用都被 \ref VIBE_GPU_CHECK 包裹，错误会带表达式文本抛出。
///
/// 复杂度：\ref enumerate_devices 为 O(n_devices) 的驱动查询（首次调用约
/// 10~100 us，之后被缓存）；\ref Device::occupancy 在 CUDA 上是
/// \c cudaOccupancyMaxActiveBlocksPerMultiprocessor 的一次查询（O(1)）。
///
/// 文献：[G10] CUDA C++ Programming Guide 第 4、6 章；[G12] HIP Programming Guide；
///       [G11] SYCL 2020 第 4 章；[G17] Gustafson (1988)；[G18] Dennis & Edwards (2004)。

#include "vibe/gpu/device.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"

// ------------------------- 后端运行时头文件 --------------------------------
#if VIBE_BACKEND_CUDA
#  include <cuda_runtime.h>
#endif
#if VIBE_BACKEND_HIP
#  include <hip/hip_runtime.h>
#endif
#if VIBE_BACKEND_SYCL
#  include <sycl/sycl.hpp>
#endif

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

namespace vibe::gpu {

namespace {

// 进程级状态：当前后端与设备列表缓存
Backend g_active = compiled_backend();
std::vector<DeviceInfo> g_devices;
bool g_devices_ready = false;
std::mutex g_mutex;  // 保护上面的三个变量

/// 物理内存估计（CPU 伪设备的 memory_bytes）
std::size_t estimate_host_memory() {
#if defined(_WIN32)
  MEMORYSTATUSEX st{};
  st.dwLength = sizeof(st);
  if (GlobalMemoryStatusEx(&st)) return static_cast<std::size_t>(st.ullTotalPhys);
  return static_cast<std::size_t>(1) << 33;  // 8 GiB 兜底
#else
  const long pages = ::sysconf(_SC_PHYS_PAGES);
  const long page = ::sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page > 0) return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page);
  return static_cast<std::size_t>(1) << 33;
#endif
}

}  // namespace

// ===========================================================================
// 字符串 <-> 枚举
// ===========================================================================
Backend from_string(const std::string& s, bool* ok) noexcept {
  std::string t;
  t.reserve(s.size());
  for (char c : s) t.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  if (ok != nullptr) *ok = true;
  if (t == "CPU" || t == "OMP" || t == "OPENMP" || t == "HOST") return Backend::CPU;
  if (t == "CUDA" || t == "GPU" || t == "NVIDIA") return Backend::CUDA;
  if (t == "HIP" || t == "ROCM" || t == "AMD") return Backend::HIP;
  if (t == "SYCL" || t == "DPCPP" || t == "ONEAPI") return Backend::SYCL;
  if (ok != nullptr) *ok = false;
  return Backend::CPU;
}

std::string DeviceInfo::describe() const {
  std::ostringstream os;
  os << to_string(backend) << ':' << id << " \"" << name << "\" "
     << (memory_bytes >> 20) << " MiB";
  if (compute_major > 0 || compute_minor > 0) os << " cc" << compute_major << '.' << compute_minor;
  if (multiprocessors > 0) os << " sm" << multiprocessors;
  return os.str();
}

// ===========================================================================
// 后端能力矩阵
// ===========================================================================
BackendCaps backend_caps(Backend b) noexcept {
  BackendCaps c;
  switch (b) {
    case Backend::CPU:
      // CPU 侧：统一内存天然成立（同一地址空间），无事件计时/张量核
      c.accelerator = false;
      c.unified_memory = true;
      c.pinned_memory = false;
      c.async_streams = false;
      c.fp16 = true;   ///< 有 FP16 的软件模拟（half_t）
      c.bf16 = true;
      c.tensor_core = false;
      c.events_timing = false;
      c.cooperative_groups = false;
      c.atomic_float = true;
      break;
    case Backend::CUDA:
      c.accelerator = true;
      c.unified_memory = true;   ///< cudaMallocManaged（[G10] 第 6.2.4 节）
      c.pinned_memory = true;
      c.async_streams = true;
      c.fp16 = true;             ///< Pascal 起的原生 __half 运算
      c.bf16 = true;             ///< Ampere（cc >= 8.0）原生 BF16
      c.tensor_core = true;      ///< Volta（cc 7.0）起 [G1][G5]
      c.events_timing = true;
      c.cooperative_groups = true;
      c.atomic_float = true;
      break;
    case Backend::HIP:
      c.accelerator = true;
      c.unified_memory = true;   ///< hipMallocManaged
      c.pinned_memory = true;
      c.async_streams = true;
      c.fp16 = true;
      c.bf16 = true;             ///< CDNA2/RDNA3 起
      c.tensor_core = true;      ///< MFMA / WMMA
      c.events_timing = true;
      c.cooperative_groups = true;
      c.atomic_float = true;
      break;
    case Backend::SYCL:
      c.accelerator = true;
      c.unified_memory = true;   ///< USM shared（[G11] 第 4.7 节）
      c.pinned_memory = true;    ///< USM host
      c.async_streams = true;    ///< 多 queue
      c.fp16 = true;
      c.bf16 = true;             ///< SYCL 2020 支持 bfloat16
      c.tensor_core = true;      ///< 通过 vendor 的 joint_matrix 扩展
      c.events_timing = true;    ///< sycl::event::get_profiling_info
      c.cooperative_groups = true;  ///< 无直接对应；用 work-group barrier 近似
      c.atomic_float = true;
      break;
  }
  return c;
}

// ===========================================================================
// 错误处理
// ===========================================================================
namespace detail {

std::string gpu_error_string(int code) {
#if VIBE_BACKEND_CUDA
  return std::string(cudaGetErrorName(static_cast<cudaError_t>(code))) + ": " +
         cudaGetErrorString(static_cast<cudaError_t>(code));
#elif VIBE_BACKEND_HIP
  return std::string(hipGetErrorName(static_cast<hipError_t>(code))) + ": " +
         hipGetErrorString(static_cast<hipError_t>(code));
#else
  return "GPU 后端未启用（code=" + std::to_string(code) + "）";
#endif
}

void throw_gpu_error(const char* expr, const char* file, int line, int code) {
  std::ostringstream os;
  os << "[gpu] " << file << ':' << line << " GPU 调用失败: " << expr << " -- "
     << gpu_error_string(code);
  throw Error(os.str());
}

}  // namespace detail

// ===========================================================================
// 设备枚举
// ===========================================================================
std::vector<DeviceInfo> enumerate_devices(Backend backend) {
  std::vector<DeviceInfo> out;

  if (backend != compiled_backend()) {
    // 非当前编译后端不可枚举；CPU 是唯一例外（始终可"模拟"）
    if (backend != Backend::CPU) return out;
  }

  switch (backend) {
#if VIBE_BACKEND_CUDA
    case Backend::CUDA: {
      int count = 0;
      VIBE_GPU_CHECK(cudaGetDeviceCount(&count));
      out.reserve(static_cast<std::size_t>(count));
      for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        VIBE_GPU_CHECK(cudaGetDeviceProperties(&prop, i));
        DeviceInfo info;
        info.backend = Backend::CUDA;
        info.id = i;
        info.name = prop.name;
        info.memory_bytes = static_cast<std::size_t>(prop.totalGlobalMem);
        info.compute_major = prop.major;
        info.compute_minor = prop.minor;
        info.multiprocessors = prop.multiProcessorCount;
        out.push_back(std::move(info));
      }
      break;
    }
#endif
#if VIBE_BACKEND_HIP
    case Backend::HIP: {
      int count = 0;
      VIBE_GPU_CHECK(hipGetDeviceCount(&count));
      out.reserve(static_cast<std::size_t>(count));
      for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        VIBE_GPU_CHECK(hipGetDeviceProperties(&prop, i));
        DeviceInfo info;
        info.backend = Backend::HIP;
        info.id = i;
        info.name = prop.name;
        info.memory_bytes = static_cast<std::size_t>(prop.totalGlobalMem);
        info.compute_major = prop.major;
        info.compute_minor = prop.minor;
        info.multiprocessors = prop.multiProcessorCount;
        out.push_back(std::move(info));
      }
      break;
    }
#endif
#if VIBE_BACKEND_SYCL
    case Backend::SYCL: {
      // SYCL 的"设备"来自默认平台及所有可见平台（[G11] 第 4.6 节）
      auto platforms = sycl::platform::get_platforms();
      int id = 0;
      for (const auto& plat : platforms) {
        for (const auto& dev : plat.get_devices()) {
          DeviceInfo info;
          info.backend = Backend::SYCL;
          info.id = id++;
          info.name = dev.get_info<sycl::info::device::name>();
          info.memory_bytes = dev.get_info<sycl::info::device::global_mem_size>();
          // SYCL 不暴露 CUDA 式的"计算能力"，用最大工作组尺寸与 CU 数近似
          info.compute_major = 0;
          info.compute_minor = 0;
          info.multiprocessors =
              static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());
          out.push_back(std::move(info));
        }
      }
      break;
    }
#endif
    case Backend::CPU:
    default: {
      // 伪设备：名字包含体系结构标签，便于日志定位
      DeviceInfo info;
      info.backend = Backend::CPU;
      info.id = 0;
      const unsigned hw = std::thread::hardware_concurrency();
      std::ostringstream os;
      os << "CPU (" << (hw == 0 ? 1u : hw) << " threads)";
      info.name = os.str();
      info.memory_bytes = estimate_host_memory();
      info.compute_major = 0;
      info.compute_minor = 0;
      info.multiprocessors = static_cast<int>(hw == 0 ? 1u : hw);
      out.push_back(std::move(info));
      break;
    }
  }
  return out;
}

std::vector<DeviceInfo> enumerate_devices() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_devices_ready) {
    g_devices = enumerate_devices(g_active);
    g_devices_ready = true;
  }
  return g_devices;
}

// ===========================================================================
// 全局后端状态
// ===========================================================================
Backend active_backend() noexcept {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_active;
}

void set_active_backend(Backend b) {
  if (!is_compiled(b)) {
    std::ostringstream os;
    os << "后端 " << to_string(b) << " 未编入本产物（编译期后端为 "
       << to_string(compiled_backend()) << "）";
    throw NotImplemented(os.str());
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_active != b) {
    g_active = b;
    g_devices.clear();
    g_devices_ready = false;
  }
}

int device_count() { return static_cast<int>(enumerate_devices().size()); }

void reset_device_cache() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_devices.clear();
  g_devices_ready = false;
  g_active = compiled_backend();
}

// ===========================================================================
// Device
// ===========================================================================
Device::Device() : Device(active_backend(), 0) {}

Device::Device(int id) : Device(active_backend(), id) {}

Device::Device(Backend backend, int id) {
  if (!is_compiled(backend)) {
    throw NotImplemented(std::string("Device: 后端 ") + to_string(backend) + " 未编入本产物");
  }
  // 若后端与当前不同，先切换（设备列表按后端缓存）
  if (backend != active_backend()) set_active_backend(backend);

  const auto devices = enumerate_devices(backend);
  if (devices.empty()) {
    throw Error(std::string("Device: 后端 ") + to_string(backend) + " 没有可见设备");
  }
  if (id < 0 || static_cast<std::size_t>(id) >= devices.size()) {
    std::ostringstream os;
    os << "Device: 设备号 " << id << " 越界（可选 0.." << (devices.size() - 1) << "）";
    throw Error(os.str());
  }
  info_ = devices[static_cast<std::size_t>(id)];
  valid_ = true;
  make_current();
}

Device::Device(Device&& o) noexcept : info_(std::move(o.info_)), valid_(o.valid_) {
  o.valid_ = false;
}

Device& Device::operator=(Device&& o) noexcept {
  if (this != &o) {
    info_ = std::move(o.info_);
    valid_ = o.valid_;
    o.valid_ = false;
  }
  return *this;
}

Device::~Device() {
  // 有意不重置设备：与 CUDA 习惯一致，避免析构顺序造成的隐式同步
  valid_ = false;
}

void Device::make_current() const {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaSetDevice(info_.id));
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipSetDevice(info_.id));
#elif VIBE_BACKEND_SYCL
  // SYCL 的设备选择通过 queue 完成；此处只记录，queue 创建时再绑定
#endif
}

int Device::warp_size() const noexcept {
#if VIBE_BACKEND_CUDA
  return 32;  // [G10] 第 5.4 节：warp = 32 线程，所有 NVIDIA 架构一致
#elif VIBE_BACKEND_HIP
  // AMD GCN/RDNA 为 64（wave64），部分架构的 wave32 需要显式请求（[G12]）
  return 64;
#elif VIBE_BACKEND_SYCL
  return 32;  // sub-group 的常见宽度；精确值需查询 device::sub_group_sizes
#else
  return 1;  // CPU：无锁步执行；归约代码用该值退化为串行
#endif
}

std::size_t Device::max_shared_memory() const noexcept {
  switch (info_.backend) {
#if VIBE_BACKEND_CUDA
    case Backend::CUDA:
      return 48u * 1024u;  // 默认 48 KB，cc>=7.0 可 opt-in 至 ~96 KB
#endif
#if VIBE_BACKEND_HIP
    case Backend::HIP:
      return 64u * 1024u;  // AMD 通常 64 KB LDS
#endif
#if VIBE_BACKEND_SYCL
    case Backend::SYCL:
      return 32u * 1024u;  // local memory，视厂商而定
#endif
    default:
      return 0;  // CPU 无共享内存概念
  }
}

int Device::max_threads_per_block() const noexcept {
  switch (info_.backend) {
#if VIBE_BACKEND_CUDA
    case Backend::CUDA:
      return 1024;
#endif
#if VIBE_BACKEND_HIP
    case Backend::HIP:
      return 1024;
#endif
#if VIBE_BACKEND_SYCL
    case Backend::SYCL:
      return 1024;  // 需查询 max_work_group_size；此处取保守值
#endif
    default:
      return 1024;  // CPU：逻辑线程数上限（仅影响分块粒度）
  }
}

int Device::registers_per_thread() const noexcept {
#if VIBE_BACKEND_CUDA
  return 255;  // 每线程最多 255 个 32-bit 寄存器（[G10] 第 5.5 节）
#elif VIBE_BACKEND_HIP
  return 256;
#else
  return 0;
#endif
}

double Device::occupancy(int regs_per_thread, std::size_t shared_per_block,
                         int threads_per_block) const {
  if (threads_per_block <= 0) return 0.0;
#if VIBE_BACKEND_CUDA
  // 用 CUDA 运行时的占用率计算器（需要内核符号，这里给的是"每 SM 活跃 block
  // 上界"的经验式：[G10] 第 5.5 节的资源约束推导）。
  const int max_threads_per_sm = 2048;
  const int max_blocks_per_sm = 32;
  const int regs_per_sm = 65536;
  const std::size_t smem_per_sm = max_shared_memory();
  int by_threads = max_threads_per_sm / threads_per_block;
  int by_blocks = max_blocks_per_sm;
  int by_regs = (regs_per_thread > 0) ? (regs_per_sm / (regs_per_thread * threads_per_block)) : by_threads;
  int by_smem = (shared_per_block > 0)
                    ? static_cast<int>(smem_per_sm / shared_per_block)
                    : by_threads;
  int active = std::min(std::min(by_threads, by_blocks), std::min(by_regs, by_smem));
  if (active < 0) active = 0;
  const double max_warps = static_cast<double>(max_threads_per_sm) / static_cast<double>(warp_size());
  const double warps = static_cast<double>(active * threads_per_block) / static_cast<double>(warp_size());
  return max_warps > 0.0 ? warps / max_warps : 0.0;
#elif VIBE_BACKEND_HIP
  const int max_threads_per_cu = 2048;
  const int active = std::max(1, max_threads_per_cu / threads_per_block);
  const double max_wavefronts =
      static_cast<double>(max_threads_per_cu) / static_cast<double>(warp_size());
  const double waves =
      static_cast<double>(active * threads_per_block) / static_cast<double>(warp_size());
  return max_wavefronts > 0.0 ? waves / max_wavefronts : 0.0;
#else
  // CPU：以可用线程数衡量的"占用率"。这里表达的是
  // "一个执行单元相对硬件并行度的匹配程度"，仅用于配置检查（[G17]）。
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const double want = static_cast<double>(threads_per_block);
  const double have = static_cast<double>(hw);
  VIBE_UNUSED(regs_per_thread);
  VIBE_UNUSED(shared_per_block);
  return want >= have ? 1.0 : want / have;
#endif
}

void Device::synchronize() const {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipDeviceSynchronize());
#elif VIBE_BACKEND_SYCL
  // SYCL 的同步由 queue::wait() 完成，见 memory.cpp / launch.cpp 的实现
#else
  // CPU：内存栅栏即可（阻止编译器把跨线程读写重排）
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

Device& default_device() {
  static Device dev = [] {
    // CPU 构建下必定成功；GPU 构建下若枚举失败则抛错，由调用方捕获
    return Device(active_backend(), 0);
  }();
  return dev;
}

int warp_size() { return default_device().warp_size(); }
std::size_t max_shared_memory() { return default_device().max_shared_memory(); }
int max_threads_per_block() { return default_device().max_threads_per_block(); }

}  // namespace vibe::gpu

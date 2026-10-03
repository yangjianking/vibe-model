/// @file memory.cpp
/// @brief 内存原语、全局统计、内存池与 FieldMirror 的实现。
///
/// 后端分支组织
/// ------------
/// \ref detail::backend_alloc / \ref detail::backend_copy_* / \ref detail::backend_memset
/// 是仅有四处涉及后端运行时的函数，其余代码（池、镜像、统计）全部后端无关。
/// CPU 后端下："设备内存"就是 \c new/\c delete 得到的堆内存，
/// "拷贝"就是 \c memcpy，"同步"是内存栅栏。于是上游内核代码
/// （kernels_cpu.cpp 与 kernels_*.cu）无需任何 \#if 就能共用同一套缓冲语义。
///
/// 复杂度：\ref detail::backend_alloc 为 O(1) 系统调用 + O(n) 页表建立；
/// 拷贝 O(n)；池的 acquire/release 摊销 O(log B)；
/// \ref FieldMirror 的 upload/download 为一次 O(n) 传输。
///
/// 文献：[G10] CUDA C++ Programming Guide 第 6 章（内存管理、统一内存、
///       异步拷贝与 \c cudaMemsetAsync）；[G12] HIP Programming Guide 第 6 章；
///       [G11] SYCL 2020 第 4.7 节（USM：device/host/shared 分配）；
///       [G17] Gustafson (1988)（摊销分配成本）；[B11] Higham (2002)（位级清零）。

#include "vibe/gpu/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#include "vibe/common/error.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/device.hpp"

#if VIBE_BACKEND_CUDA
#  include <cuda_runtime.h>
#endif
#if VIBE_BACKEND_HIP
#  include <hip/hip_runtime.h>
#endif
#if VIBE_BACKEND_SYCL
#  include <sycl/sycl.hpp>
#endif

namespace vibe::gpu {

namespace {

std::mutex g_stats_mutex;
MemoryStats g_stats{};

}  // namespace

// ===========================================================================
// 全局统计
// ===========================================================================
MemoryStats memory_stats() {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  return g_stats;
}

void reset_memory_stats() {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  g_stats.reset();
}

std::string memory_stats_string() {
  const MemoryStats s = memory_stats();
  std::ostringstream os;
  os << "当前 " << (s.current >> 20) << " MiB / 峰值 " << (s.peak >> 20) << " MiB"
     << " | 设备 " << (s.device_bytes >> 20) << " MiB"
     << " | 主机 " << (s.host_bytes >> 20) << " MiB"
     << " | 分配 " << s.allocations << " 次"
     << " | 池命中 " << s.pool_hits << " / 未命中 " << s.pool_misses;
  return os.str();
}

namespace detail {

void note_alloc(std::size_t bytes, bool device_mem) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  g_stats.current += bytes;
  g_stats.allocations += 1;
  if (device_mem) {
    g_stats.device_bytes += bytes;
  } else {
    g_stats.host_bytes += bytes;
  }
  if (g_stats.current > g_stats.peak) g_stats.peak = g_stats.current;
}

void note_free(std::size_t bytes, bool device_mem) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  g_stats.current = (g_stats.current > bytes) ? (g_stats.current - bytes) : 0;
  g_stats.deallocations += 1;
  if (device_mem) {
    g_stats.device_bytes = (g_stats.device_bytes > bytes) ? (g_stats.device_bytes - bytes) : 0;
  } else {
    g_stats.host_bytes = (g_stats.host_bytes > bytes) ? (g_stats.host_bytes - bytes) : 0;
  }
}

// ===========================================================================
// 后端内存原语
// ===========================================================================
void* backend_alloc(std::size_t bytes, bool device, bool pinned) {
#if VIBE_BACKEND_CUDA
  void* p = nullptr;
  if (device) {
    VIBE_GPU_CHECK(cudaMalloc(&p, bytes));
  } else if (pinned) {
    VIBE_GPU_CHECK(cudaMallocHost(&p, bytes));
  } else {
    VIBE_GPU_CHECK(cudaMallocManaged(&p, bytes));  // 统一内存
  }
  return p;
#elif VIBE_BACKEND_HIP
  void* p = nullptr;
  if (device) {
    VIBE_GPU_CHECK(hipMalloc(&p, bytes));
  } else if (pinned) {
    VIBE_GPU_CHECK(hipHostMalloc(&p, bytes));
  } else {
    VIBE_GPU_CHECK(hipMallocManaged(&p, bytes));
  }
  return p;
#elif VIBE_BACKEND_SYCL
  // USM：device / host / shared 三种分配对应本层的三种策略（[G11] 第 4.7 节）
  if (device) return sycl::malloc_device(bytes, sycl::queue{sycl::default_selector_v});
  if (pinned) return sycl::aligned_alloc_host(64, bytes, sycl::queue{sycl::default_selector_v});
  return sycl::malloc_shared(bytes, sycl::queue{sycl::default_selector_v});
#else
  // CPU：三种策略都退化为对齐的堆分配（设备指针 == 主机指针）
  VIBE_UNUSED(device);
  VIBE_UNUSED(pinned);
  void* p = ::operator new(bytes, std::nothrow);
  if (p == nullptr) throw Error("[gpu] 主机内存分配失败（请求 " + std::to_string(bytes) + " 字节）");
  return p;
#endif
}

void backend_free(void* ptr, bool pinned) noexcept {
  if (ptr == nullptr) return;
#if VIBE_BACKEND_CUDA
  if (pinned) {
    (void)cudaFreeHost(ptr);
  } else {
    (void)cudaFree(ptr);
  }
#elif VIBE_BACKEND_HIP
  if (pinned) {
    (void)hipHostFree(ptr);
  } else {
    (void)hipFree(ptr);
  }
#elif VIBE_BACKEND_SYCL
  VIBE_UNUSED(pinned);
  (void)sycl::free(ptr, sycl::queue{sycl::default_selector_v});
#else
  VIBE_UNUSED(pinned);
  ::operator delete(ptr);
#endif
}

void backend_copy_h2d(void* dst, const void* src, std::size_t bytes, void* stream) {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                 reinterpret_cast<cudaStream_t>(stream)));
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice,
                                reinterpret_cast<hipStream_t>(stream)));
#elif VIBE_BACKEND_SYCL
  VIBE_UNUSED(stream);
  sycl::queue{sycl::default_selector_v}.memcpy(dst, src, bytes).wait();
#else
  VIBE_UNUSED(stream);
  std::memcpy(dst, src, bytes);
#endif
}

void backend_copy_d2h(void* dst, const void* src, std::size_t bytes, void* stream) {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                 reinterpret_cast<cudaStream_t>(stream)));
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
                                reinterpret_cast<hipStream_t>(stream)));
#elif VIBE_BACKEND_SYCL
  VIBE_UNUSED(stream);
  sycl::queue{sycl::default_selector_v}.memcpy(dst, src, bytes).wait();
#else
  VIBE_UNUSED(stream);
  std::memcpy(dst, src, bytes);
#endif
}

void backend_copy_d2d(void* dst, const void* src, std::size_t bytes, void* stream) {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                                 reinterpret_cast<cudaStream_t>(stream)));
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice,
                                reinterpret_cast<hipStream_t>(stream)));
#elif VIBE_BACKEND_SYCL
  VIBE_UNUSED(stream);
  sycl::queue{sycl::default_selector_v}.memcpy(dst, src, bytes).wait();
#else
  VIBE_UNUSED(stream);
  if (dst != src) std::memmove(dst, src, bytes);
#endif
}

void backend_memset(void* dst, int value, std::size_t bytes, void* stream) {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaMemsetAsync(dst, value, bytes, reinterpret_cast<cudaStream_t>(stream)));
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipMemsetAsync(dst, value, bytes, reinterpret_cast<hipStream_t>(stream)));
#elif VIBE_BACKEND_SYCL
  VIBE_UNUSED(stream);
  sycl::queue{sycl::default_selector_v}.memset(dst, value, bytes).wait();
#else
  VIBE_UNUSED(stream);
  std::memset(dst, value, bytes);
#endif
}

void backend_sync() {
#if VIBE_BACKEND_CUDA
  VIBE_GPU_CHECK(cudaDeviceSynchronize());
#elif VIBE_BACKEND_HIP
  VIBE_GPU_CHECK(hipDeviceSynchronize());
#elif VIBE_BACKEND_SYCL
  // SYCL 的同步粒度是 queue；无全局同步 API，这里由 queue::wait 覆盖
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace detail

// ===========================================================================
// MemoryPool
// ===========================================================================
MemoryPool::MemoryPool(std::size_t max_cached_bytes) : max_cached_bytes_(max_cached_bytes) {}

MemoryPool::~MemoryPool() { clear(); }

MemoryPool& MemoryPool::instance() {
  static MemoryPool pool;
  return pool;
}

bool MemoryPool::pop_block(std::size_t element_size, AllocKind kind, Size n, CachedBlock& out) {
  // 1) 精确命中：同类型、同策略、同长度
  for (Bucket& b : buckets_) {
    if (b.element_size != element_size || b.kind != kind) continue;
    for (std::size_t i = 0; i < b.blocks.size(); ++i) {
      if (b.blocks[i].elements == n) {
        out = b.blocks[i];
        b.blocks.erase(b.blocks.begin() + static_cast<std::ptrdiff_t>(i));
        cached_bytes_ -= out.elements * out.element_size;
        return true;
      }
    }
  }
  // 2) best-fit：找到不小于请求长度的最小块
  Bucket* best_bucket = nullptr;
  std::size_t best_index = 0;
  std::size_t best_elements = 0;
  for (Bucket& b : buckets_) {
    if (b.element_size != element_size || b.kind != kind) continue;
    for (std::size_t i = 0; i < b.blocks.size(); ++i) {
      const std::size_t elems = b.blocks[i].elements;
      if (elems < n) continue;
      if (best_bucket == nullptr || elems < best_elements) {
        best_bucket = &b;
        best_index = i;
        best_elements = elems;
      }
    }
  }
  if (best_bucket == nullptr) return false;
  out = best_bucket->blocks[best_index];
  best_bucket->blocks.erase(best_bucket->blocks.begin() + static_cast<std::ptrdiff_t>(best_index));
  cached_bytes_ -= out.elements * out.element_size;
  return true;
}

void MemoryPool::push_block(std::size_t element_size, AllocKind kind, Size n, void* device,
                            void* host) {
  if (device == nullptr && host == nullptr) return;
  for (Bucket& b : buckets_) {
    if (b.element_size == element_size && b.kind == kind) {
      CachedBlock blk;
      blk.device = device;
      blk.host = host;
      blk.elements = n;
      blk.element_size = element_size;
      blk.kind = kind;
      b.blocks.push_back(blk);
      cached_bytes_ += n * element_size;
      return;
    }
  }
  Bucket b;
  b.element_size = element_size;
  b.kind = kind;
  CachedBlock blk;
  blk.device = device;
  blk.host = host;
  blk.elements = n;
  blk.element_size = element_size;
  blk.kind = kind;
  b.blocks.push_back(blk);
  buckets_.push_back(std::move(b));
  cached_bytes_ += n * element_size;
}

std::size_t MemoryPool::cached_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t total = 0;
  for (const Bucket& b : buckets_) total += b.blocks.size();
  return total;
}

std::size_t MemoryPool::cached_bytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cached_bytes_;
}

MemoryStats MemoryPool::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void MemoryPool::reset_stats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.reset();
}

void MemoryPool::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (Bucket& b : buckets_) {
    for (CachedBlock& blk : b.blocks) {
      if (blk.device != nullptr) {
        detail::backend_free(blk.device, blk.kind == AllocKind::Pinned);
        detail::note_free(blk.elements * blk.element_size, blk.kind == AllocKind::Device);
      }
      if (blk.host != nullptr && blk.host != blk.device) {
        detail::backend_free(blk.host, true);
        detail::note_free(blk.elements * blk.element_size, false);
      }
    }
    b.blocks.clear();
  }
  buckets_.clear();
  cached_bytes_ = 0;
}

// ===========================================================================
// FieldMirror
// ===========================================================================
FieldMirror::FieldMirror(std::size_t bytes, AllocKind kind) { resize(bytes, kind); }

FieldMirror::~FieldMirror() { free_all(); }

FieldMirror::FieldMirror(FieldMirror&& o) noexcept
    : device_(o.device_), host_(o.host_), bytes_(o.bytes_), kind_(o.kind_) {
  o.device_ = nullptr;
  o.host_ = nullptr;
  o.bytes_ = 0;
}

FieldMirror& FieldMirror::operator=(FieldMirror&& o) noexcept {
  if (this != &o) {
    free_all();
    device_ = o.device_;
    host_ = o.host_;
    bytes_ = o.bytes_;
    kind_ = o.kind_;
    o.device_ = nullptr;
    o.host_ = nullptr;
    o.bytes_ = 0;
  }
  return *this;
}

void FieldMirror::free_all() noexcept {
  if (device_ != nullptr) {
    detail::backend_free(device_, kind_ == AllocKind::Pinned);
    detail::note_free(bytes_, kind_ == AllocKind::Device);
  }
  if (host_ != nullptr && host_ != device_) {
    detail::backend_free(host_, true);
    detail::note_free(bytes_, false);
  }
  device_ = nullptr;
  host_ = nullptr;
  bytes_ = 0;
}

void FieldMirror::resize(std::size_t bytes, AllocKind kind) {
  if (bytes == bytes_ && kind == kind_) return;
  free_all();
  kind_ = kind;
  bytes_ = bytes;
  if (bytes == 0) return;
  switch (kind) {
    case AllocKind::Unified:
      device_ = static_cast<unsigned char*>(detail::backend_alloc(bytes, false, false));
      host_ = device_;
      detail::note_alloc(bytes, false);
      break;
    case AllocKind::Pinned:
      device_ = static_cast<unsigned char*>(detail::backend_alloc(bytes, false, true));
      host_ = device_;
      detail::note_alloc(bytes, false);
      break;
    case AllocKind::Device:
      device_ = static_cast<unsigned char*>(detail::backend_alloc(bytes, true, false));
      host_ = static_cast<unsigned char*>(detail::backend_alloc(bytes, false, true));
      detail::note_alloc(bytes, true);
      detail::note_alloc(bytes, false);
      break;
  }
}

void FieldMirror::upload(const void* host_src, std::size_t bytes) {
  VIBE_CHECK_MSG(bytes == bytes_, "FieldMirror::upload 长度不匹配");
  if (bytes == 0) return;
  detail::backend_copy_h2d(device_, host_src, bytes, nullptr);
  if (host_ != nullptr && host_ != device_) std::memcpy(host_, host_src, bytes);
}

void FieldMirror::download(void* host_dst, std::size_t bytes) const {
  VIBE_CHECK_MSG(bytes == bytes_, "FieldMirror::download 长度不匹配");
  if (bytes == 0) return;
  detail::backend_copy_d2h(host_dst, device_, bytes, nullptr);
}

// ===========================================================================
// 预算检查
// ===========================================================================
void check_memory_budget(std::size_t extra_bytes, Real fraction) {
  const auto devices = enumerate_devices();
  if (devices.empty()) return;
  const std::size_t capacity = devices.front().memory_bytes;
  if (capacity == 0) return;
  const std::size_t used = memory_stats().device_bytes;
  const double limit = static_cast<double>(capacity) * static_cast<double>(fraction);
  if (static_cast<double>(used + extra_bytes) > limit) {
    std::ostringstream os;
    os << "[gpu] 内存预算超限：已用 " << (used >> 20) << " MiB + 本次 " << (extra_bytes >> 20)
       << " MiB > 上限 " << (static_cast<std::size_t>(limit) >> 20) << " MiB（容量 "
       << (capacity >> 20) << " MiB 的 " << static_cast<double>(fraction) << " 倍）";
    throw Error(os.str());
  }
}

}  // namespace vibe::gpu

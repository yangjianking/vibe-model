#pragma once
/// @file memory.hpp
/// @brief 设备缓冲、分配策略与内存池（模板实现全部内联在头文件）。
///
/// 三种分配策略（\ref AllocKind）
/// ------------------------------
///   * \c Device  —— 后端设备内存（\c cudaMalloc / \c hipMalloc /
///     \c sycl::malloc_device）＋一块**页锁定**主机镜像。有了主机镜像，
///     \ref DeviceBuffer::host_ptr 不需要每次临时分配；传输用
///     \c cudaMemcpyAsync 加同步流完成；
///   * \c Pinned  —— 页锁定主机内存（\c cudaMallocHost）。设备可直接访问，
///     适合同步点少、传输频繁的小数组（halo 交换缓冲、观测向量）；
///   * \c Unified —— 统一内存（\c cudaMallocManaged，[G10] 第 6 章）。
///     同一指针在主机与设备均有效，按需缺页迁移；实现最简，但可能产生
///     page-fault 抖动，故默认只用于"每步少量访存"的大场。
///
/// CPU 后端下三种策略全部退化为 \c std::vector 式的堆内存（设备指针 == 主机指针），
/// 于是全部上游代码无需任何 \#if。
///
/// 与 halo 布局的关系
/// ------------------
/// \c grid::Field 是"单数组 + halo"布局（见 grid/field.hpp），
/// \ref DeviceBuffer 只负责一块**线性**存储，不感知错位与 halo；
/// Field 与设备副本的耦合由 \ref FieldMirror 完成，映射细节与 halo 交换的
/// GPU 化见 docs/design/05_gpu_hybrid_precision.md 第 2 节。
///
/// 复杂度：分配 O(1)（池命中）/ O(1)+O(n) 建页表（未命中）；拷贝 O(n)；
/// \ref MemoryPool::acquire 摊销 O(log B)（B = 桶数）。
///
/// 文献：[G10] CUDA C++ Programming Guide 第 6 章（设备内存、统一内存、
///       页锁定内存与异步分配）；[G12] HIP Programming Guide 第 6 章；
///       [G11] SYCL 2020 第 4.7 节（USM）；[G17] Gustafson (1988)（摊销成本与扩展性）；
///       [B11] Higham (2002)（按位清零与浮点表示的关系）。

#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/device.hpp"

namespace vibe::gpu {

// ---------------------------------------------------------------------------
// 分配策略
// ---------------------------------------------------------------------------
enum class AllocKind {
  Device = 0,   ///< 设备私有内存 + 页锁定主机镜像
  Pinned = 1,   ///< 页锁定主机内存（设备可直接访问）
  Unified = 2   ///< 统一内存（主机/设备同一指针）
};

inline const char* to_string(AllocKind k) noexcept {
  switch (k) {
    case AllocKind::Device: return "device";
    case AllocKind::Pinned: return "pinned";
    case AllocKind::Unified: return "unified";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// 内存统计
// ---------------------------------------------------------------------------
/// 进程级内存计数器。所有 \ref DeviceBuffer 的分配/释放都会更新它。
/// 统计以字节为单位，跨后端统一。
struct MemoryStats {
  std::size_t current = 0;        ///< 当前占用字节
  std::size_t peak = 0;           ///< 峰值占用字节
  std::size_t allocations = 0;    ///< 累计分配次数（含池命中）
  std::size_t deallocations = 0;  ///< 累计释放次数
  std::size_t pool_hits = 0;      ///< 池命中次数
  std::size_t pool_misses = 0;    ///< 池未命中（真正向系统申请）次数
  std::size_t device_bytes = 0;   ///< 当前设备内存字节
  std::size_t host_bytes = 0;     ///< 当前主机（含页锁定）内存字节

  void reset() noexcept { *this = MemoryStats{}; }
};

MemoryStats memory_stats();
void reset_memory_stats();
std::string memory_stats_string();

// ---------------------------------------------------------------------------
// 后端无关的内存原语（在 src/gpu/memory.cpp 中实现）
// ---------------------------------------------------------------------------
namespace detail {
void* backend_alloc(std::size_t bytes, bool device, bool pinned);
void backend_free(void* ptr, bool pinned) noexcept;
void backend_copy_h2d(void* dst, const void* src, std::size_t bytes, void* stream);
void backend_copy_d2h(void* dst, const void* src, std::size_t bytes, void* stream);
void backend_copy_d2d(void* dst, const void* src, std::size_t bytes, void* stream);
void backend_memset(void* dst, int value, std::size_t bytes, void* stream);
void backend_sync();
void note_alloc(std::size_t bytes, bool device_mem);
void note_free(std::size_t bytes, bool device_mem);
}  // namespace detail

// ---------------------------------------------------------------------------
// DeviceBuffer<T>
// ---------------------------------------------------------------------------
/// 主机/设备统一缓冲。
///
/// 语义要点
/// --------
///   * 仅可移动，禁止拷贝（避免隐式 O(n) 传输）；
///   * \ref host_ptr 在 \c Device 策略下返回**主机镜像**指针。修改镜像后
///     必须显式 \ref copy_to_device——不做隐式脏标记，因为隐式同步会掩盖性能问题；
///   * \ref device_ptr 在 CPU 后端返回与 \ref host_ptr 相同的地址；
///   * 析构释放内存；\ref release 后缓冲变空。
///
/// 线程安全：单个 Buffer 不可跨线程并发读写；全局统计计数器内部加锁。
template <class T>
class DeviceBuffer {
 public:
  using value_type = T;
  static_assert(std::is_trivially_copyable<T>::value,
                "DeviceBuffer 只支持平凡可拷贝类型（比特级拷贝语义）");

  DeviceBuffer() noexcept = default;
  explicit DeviceBuffer(Size n, AllocKind kind = AllocKind::Device) { allocate(n, kind); }
  DeviceBuffer(Size n, AllocKind kind, const T& init) {
    allocate(n, kind);
    fill(init);
  }
  ~DeviceBuffer() { free_all(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept { move_from(o); }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      free_all();
      move_from(o);
    }
    return *this;
  }

  // ---- 元信息 -------------------------------------------------------------
  Size size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  AllocKind kind() const noexcept { return kind_; }
  std::size_t bytes() const noexcept { return size_ * sizeof(T); }

  // ---- 指针 ---------------------------------------------------------------
  T* device_ptr() noexcept { return device_; }
  const T* device_ptr() const noexcept { return device_; }
  T* host_ptr() noexcept { return host_; }
  const T* host_ptr() const noexcept { return host_; }

  /// 设备可直访的裸指针（内核启动用）。CPU/Pinned/Unified 下与 host_ptr 相同。
  T* data() noexcept { return device_; }
  const T* data() const noexcept { return device_; }

  // ---- 形状与填充 ---------------------------------------------------------
  /// 重新分配（内容不保留）。长度不变时为空操作。
  void resize(Size n, AllocKind kind = AllocKind::Device) {
    if (n == size_ && kind == kind_) return;
    free_all();
    allocate(n, kind);
  }

  /// 释放全部资源，回到空缓冲
  void release() noexcept { free_all(); }

  /// 填充。\c Device 策略下写入设备内存（经镜像与一次 H2D 拷贝）。
  void fill(const T& value) {
    if (size_ == 0) return;
    if (host_ != nullptr) {
      for (Size n = 0; n < size_; ++n) host_[n] = value;
      if (kind_ == AllocKind::Device) copy_to_device();
    } else if (is_all_byte_pattern(value)) {
      detail::backend_memset(device_, static_cast<int>(static_cast<unsigned char>(value)), bytes(), nullptr);
    } else {
      // 无主机镜像且不能用 memset：退化为按元素写设备内存
      detail::backend_memset(device_, 0, bytes(), nullptr);
      fill_impl_direct(value);
    }
  }

  /// 按字节清零（浮点 0 的位模式全零，符合 IEEE-754 规范值）
  void zero() {
    if (size_ == 0) return;
    detail::backend_memset(device_, 0, bytes(), nullptr);
    if (host_ != nullptr && host_ != device_) std::memset(host_, 0, bytes());
  }

  // ---- 数据传输 -----------------------------------------------------------
  /// 主机镜像 -> 设备（Pinned/Unified/CPU 下为空操作）
  void copy_to_device() {
    if (kind_ != AllocKind::Device || size_ == 0) return;
    detail::backend_copy_h2d(device_, host_, bytes(), nullptr);
  }
  /// 设备 -> 主机镜像
  void copy_to_host() {
    if (kind_ != AllocKind::Device || size_ == 0) return;
    detail::backend_copy_d2h(host_, device_, bytes(), nullptr);
  }

  /// 主机内存 -> 本缓冲（长度必须相等）
  void upload(const T* src, Size n) {
    VIBE_CHECK_MSG(n == size_, "DeviceBuffer::upload 长度不匹配");
    if (size_ == 0) return;
    detail::backend_copy_h2d(device_, src, n * sizeof(T), nullptr);
    if (host_ != nullptr && host_ != device_) std::memcpy(host_, src, n * sizeof(T));
  }
  /// 本缓冲 -> 主机内存
  void download(T* dst, Size n) const {
    VIBE_CHECK_MSG(n == size_, "DeviceBuffer::download 长度不匹配");
    if (size_ == 0) return;
    detail::backend_copy_d2h(dst, device_, n * sizeof(T), nullptr);
  }

  /// \ref resize + \ref upload 的合并形式
  void assign(const T* src, Size n, AllocKind kind = AllocKind::Device) {
    resize(n, kind);
    if (n > 0 && src != nullptr) upload(src, n);
  }

  // ---- 交换 ---------------------------------------------------------------
  void swap(DeviceBuffer& o) noexcept {
    std::swap(device_, o.device_);
    std::swap(host_, o.host_);
    std::swap(size_, o.size_);
    std::swap(kind_, o.kind_);
  }

  // ---- 供 MemoryPool 使用（不参与常规调用）--------------------------------
  /// 接管一块已由池分配的存储（不更新全局统计）
  void adopt(T* device, T* host, Size n, AllocKind kind) noexcept {
    free_all();
    device_ = device;
    host_ = host;
    size_ = n;
    kind_ = kind;
  }
  /// 交出设备指针所有权（之后本对象为空，不再释放它）
  T* steal_device() noexcept { T* p = device_; device_ = nullptr; return p; }
  /// 交出主机镜像指针所有权
  T* steal_host() noexcept { T* p = host_; host_ = nullptr; return p; }

 private:
  /// 值是否为"全字节重复"模式（可用 memset 填充）：仅对整数成立
  static constexpr bool is_all_byte_pattern(const T&) noexcept {
    return std::is_integral<T>::value && sizeof(T) == 1;
  }

  void fill_impl_direct(const T& value);

  void move_from(DeviceBuffer& o) noexcept {
    device_ = o.device_;
    host_ = o.host_;
    size_ = o.size_;
    kind_ = o.kind_;
    o.device_ = nullptr;
    o.host_ = nullptr;
    o.size_ = 0;
  }

  void free_all() noexcept {
    if (device_ == nullptr && host_ == nullptr) {
      size_ = 0;
      return;
    }
    const std::size_t nbytes = bytes();
    if (device_ != nullptr) {
      detail::backend_free(device_, kind_ == AllocKind::Pinned);
      detail::note_free(nbytes, kind_ == AllocKind::Device);
    }
    if (host_ != nullptr && host_ != device_) {
      detail::backend_free(host_, true);
      detail::note_free(nbytes, false);
    }
    device_ = nullptr;
    host_ = nullptr;
    size_ = 0;
  }

  void allocate(Size n, AllocKind kind);

  T* device_ = nullptr;   ///< 设备（或统一/CPU）内存
  T* host_ = nullptr;     ///< 主机镜像；Pinned 与 Unified 下 == device_
  Size size_ = 0;
  AllocKind kind_ = AllocKind::Device;
};

template <class T>
inline void swap(DeviceBuffer<T>& a, DeviceBuffer<T>& b) noexcept { a.swap(b); }

template <class T>
void DeviceBuffer<T>::allocate(Size n, AllocKind kind) {
  kind_ = kind;
  size_ = n;
  if (n == 0) return;
  const std::size_t nbytes = n * sizeof(T);
  switch (kind) {
    case AllocKind::Unified:
      // 一块内存既是设备也是主机；CPU 后端等价于普通堆分配
      device_ = static_cast<T*>(detail::backend_alloc(nbytes, false, false));
      host_ = device_;
      break;
    case AllocKind::Pinned:
      device_ = static_cast<T*>(detail::backend_alloc(nbytes, false, true));
      host_ = device_;
      break;
    case AllocKind::Device:
      device_ = static_cast<T*>(detail::backend_alloc(nbytes, true, false));
      host_ = static_cast<T*>(detail::backend_alloc(nbytes, false, true));
      break;
  }
  if (kind == AllocKind::Device) {
    detail::note_alloc(nbytes, true);   // 设备内存
    detail::note_alloc(nbytes, false);  // 页锁定主机镜像
  } else {
    detail::note_alloc(nbytes, false);  // Pinned / Unified 都记在主机侧
  }
}

template <class T>
void DeviceBuffer<T>::fill_impl_direct(const T& value) {
  // 极端回退路径：把元素逐个写到设备内存（仅当无主机镜像时到达）
  for (Size n = 0; n < size_; ++n) detail::backend_copy_h2d(device_ + n, &value, sizeof(T), nullptr);
}

// ---------------------------------------------------------------------------
// MemoryPool
// ---------------------------------------------------------------------------
/// 尺寸分级池：把"同一元素类型 + 同一策略 + 相近元素数"的释放缓冲留在池里复用，
/// 避免时间步循环内反复 cudaMalloc/Free（分配是全局同步点，代价高）。
///
/// 策略
/// ----
///   * 先按精确长度查找（精确命中）；
///   * 未命中则取**不小于**请求长度的最小可用块（best-fit），命中后整块移交；
///   * 缓存总量超过 \ref max_cached_bytes 时拒绝缓存，直接释放（防止显存无界增长）。
///
/// 复杂度：acquire/release 平均 O(log B)，B = 桶数；桶内线性扫描。
/// 线程安全：全部接口加互斥。
///
/// 文献：[G10] 第 6.2 节（异步分配与池化）；[G17] 摊销成本论证。
class MemoryPool {
 public:
  explicit MemoryPool(std::size_t max_cached_bytes = 256u * 1024u * 1024u);
  ~MemoryPool();

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  /// 取出至少 \p n 个元素的缓冲；池内没有则新建
  template <class T>
  DeviceBuffer<T> acquire(Size n, AllocKind kind = AllocKind::Device) {
    DeviceBuffer<T> buf;
    if (n == 0) return buf;
    std::lock_guard<std::mutex> lock(mutex_);
    CachedBlock blk;
    if (pop_block(sizeof(T), kind, n, blk)) {
      ++stats_.pool_hits;
      buf.adopt(static_cast<T*>(blk.device), static_cast<T*>(blk.host), static_cast<Size>(blk.elements), kind);
    } else {
      ++stats_.pool_misses;
      buf.resize(n, kind);
    }
    return buf;
  }

  /// 归还缓冲（缓冲交给池或被真正释放）
  template <class T>
  void release(DeviceBuffer<T>& buf) {
    if (buf.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t want = static_cast<std::size_t>(buf.size()) * sizeof(T);
    if (cached_bytes_ + want <= max_cached_bytes_) {
      push_block(sizeof(T), buf.kind(), buf.size(), buf.steal_device(), buf.steal_host());
      buf.release();  // 指针已被交出，这里只把长度清零
    } else {
      buf.release();
    }
  }

  std::size_t cached_blocks() const;
  std::size_t cached_bytes() const;
  MemoryStats stats() const;
  void reset_stats();
  /// 清空池（真正归还系统）
  void clear();

  static MemoryPool& instance();

 private:
  struct CachedBlock {
    void* device = nullptr;
    void* host = nullptr;
    std::size_t elements = 0;
    std::size_t element_size = 0;
    AllocKind kind = AllocKind::Device;
  };
  struct Bucket {
    std::size_t element_size = 0;
    AllocKind kind = AllocKind::Device;
    std::vector<CachedBlock> blocks;
  };

  bool pop_block(std::size_t element_size, AllocKind kind, Size n, CachedBlock& out);
  void push_block(std::size_t element_size, AllocKind kind, Size n, void* device, void* host);

  std::vector<Bucket> buckets_;
  std::size_t max_cached_bytes_ = 0;
  std::size_t cached_bytes_ = 0;
  mutable std::mutex mutex_;
  MemoryStats stats_{};
};

// ---------------------------------------------------------------------------
// FieldMirror
// ---------------------------------------------------------------------------
/// \c grid::Field 的 GPU 镜像（按字节，不依赖 grid 头文件）。
///
/// 设备侧的 Field 只要求"布局与主机 Field 完全相同"：
/// 长度 \c nx*ny*nz*sizeof(Real) 的线性缓冲，halo 已在 Field 的
/// \ref grid::FieldT::offset 中编码，因此镜像无需理解 halo。
///
/// @code
///   grid::Field<Real> q(g, Stagger::Cell, "q");
///   gpu::FieldMirror m(q.size() * sizeof(Real));
///   m.upload(q.data(), q.size() * sizeof(Real));
///   // ... 内核在 m.device_ptr<Real>() 上工作 ...
///   m.download(q.data(), q.size() * sizeof(Real));
/// @endcode
class FieldMirror {
 public:
  FieldMirror() = default;
  explicit FieldMirror(std::size_t bytes, AllocKind kind = AllocKind::Device);
  ~FieldMirror();

  FieldMirror(const FieldMirror&) = delete;
  FieldMirror& operator=(const FieldMirror&) = delete;
  FieldMirror(FieldMirror&&) noexcept;
  FieldMirror& operator=(FieldMirror&&) noexcept;

  void resize(std::size_t bytes, AllocKind kind = AllocKind::Device);
  std::size_t bytes() const noexcept { return bytes_; }
  bool empty() const noexcept { return bytes_ == 0; }
  AllocKind kind() const noexcept { return kind_; }

  /// 主机 -> 设备（长度必须相同）
  void upload(const void* host_src, std::size_t bytes);
  /// 设备 -> 主机
  void download(void* host_dst, std::size_t bytes) const;

  void* device_raw() noexcept { return device_; }
  const void* device_raw() const noexcept { return device_; }

  template <class T>
  T* device_ptr() noexcept { return static_cast<T*>(device_); }
  template <class T>
  const T* device_ptr() const noexcept { return static_cast<const T*>(device_); }
  template <class T>
  std::size_t count() const noexcept { return bytes_ / sizeof(T); }

 private:
  void free_all() noexcept;

  unsigned char* device_ = nullptr;
  unsigned char* host_ = nullptr;
  std::size_t bytes_ = 0;
  AllocKind kind_ = AllocKind::Device;
};

/// 显存预算检查：当前占用 + \p extra_bytes 超过设备容量的 \p fraction 倍时
/// 抛 \ref vibe::Error。建议在每个时间步开始处调用一次（[G10] 第 6 章）。
void check_memory_budget(std::size_t extra_bytes, Real fraction = Real(0.9));

}  // namespace vibe::gpu

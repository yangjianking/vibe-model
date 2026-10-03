#pragma once
/// @file backend.hpp
/// @brief 计算后端枚举、编译期探测与统一错误检查宏。
///
/// 设计目标
/// --------
/// 同一套源码在四种后端上编译：
///
///   * \c CPU  —— 串行或 OpenMP 多线程（参考实现，始终可用）；
///   * \c CUDA —— NVIDIA GPU（\c nvcc，\c cuda_runtime.h）；
///   * \c HIP  —— AMD GPU（\c hipcc，\c hip_runtime.h）；
///   * \c SYCL —— 厂商无关异构（\c icpx/dpcpp，\c <sycl/sycl.hpp>，[G11]）。
///
/// 探测规则（编译期，见 \ref VIBE_BACKEND_CPU 等宏）
/// --------------------------------------------------
///   1. 任何后端编译单元都会先把 \c VIBE_BACKEND_* 解析成 0/1；
///   2. 若用户显式定义了 \c VIBE_BACKEND_<X>，以用户值为准（可强制 CPU 后端做调试）；
///   3. 否则由编译器内建宏探测：\c __CUDACC__ -> CUDA，\c __HIPCC__ -> HIP，
///      \c SYCL_LANGUAGE_VERSION -> SYCL；
///   4. 全部未命中时使用 \c CPU。
/// 恰好一个后端宏为 1；其余为 0。\ref active_backend 返回该编译期后端。
///
/// 编译期常量（供 \#if 使用，不要在业务代码里写 GPU 运行时 API）
/// --------------------------------------------------------------
///   \c VIBE_BACKEND_CPU  \c VIBE_BACKEND_CUDA  \c VIBE_BACKEND_HIP  \c VIBE_BACKEND_SYCL
///   \c VIBE_HAVE_CUDA    \c VIBE_HAVE_HIP     \c VIBE_HAVE_SYCL    \c VIBE_HAVE_GPU
///
/// 运行时错误检查
/// --------------
///   \ref VIBE_GPU_CHECK(expr)  —— 在 GPU 后端检查 API 返回码并抛 \ref vibe::Error；
///   在 CPU 后端退化为普通调用（不引入任何 GPU 依赖）。
///
/// 文献：[G10] CUDA C++ Programming Guide；[G11] SYCL 2020；[G12] HIP Programming Guide；
///       [G13] OpenMP 5.2；[B11] Higham (2002) 第 1 章（浮点异常与非规格化数判定）。

#include <string>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"

// ---------------------------------------------------------------------------
// 1. 后端探测
// ---------------------------------------------------------------------------
#if !defined(VIBE_BACKEND_CPU)
#  define VIBE_BACKEND_CPU 0
#endif
#if !defined(VIBE_BACKEND_CUDA)
#  if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#    define VIBE_BACKEND_CUDA 1
#  else
#    define VIBE_BACKEND_CUDA 0
#  endif
#endif
#if !defined(VIBE_BACKEND_HIP)
#  if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
#    define VIBE_BACKEND_HIP 1
#  else
#    define VIBE_BACKEND_HIP 0
#  endif
#endif
#if !defined(VIBE_BACKEND_SYCL)
#  if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_DEVICE_ONLY__)
#    define VIBE_BACKEND_SYCL 1
#  else
#    define VIBE_BACKEND_SYCL 0
#  endif
#endif

// 一个都没开启 -> CPU 后端
#if !VIBE_BACKEND_CUDA && !VIBE_BACKEND_HIP && !VIBE_BACKEND_SYCL && !VIBE_BACKEND_CPU
#  define VIBE_BACKEND_CPU 1
#endif

// 若用户显式开启了 GPU 后端，则关闭 CPU 标志（CPU 只是"无 GPU"时的回退）
#if VIBE_BACKEND_CUDA || VIBE_BACKEND_HIP || VIBE_BACKEND_SYCL
#  undef VIBE_BACKEND_CPU
#  define VIBE_BACKEND_CPU 0
#endif

#if !VIBE_BACKEND_CPU && !VIBE_BACKEND_CUDA && !VIBE_BACKEND_HIP && !VIBE_BACKEND_SYCL
#  error "VIBE: 未探测到任何可用后端，请显式定义 VIBE_BACKEND_CPU=1"
#endif

// 别名宏：语义更贴近使用点
#define VIBE_HAVE_CUDA VIBE_BACKEND_CUDA
#define VIBE_HAVE_HIP  VIBE_BACKEND_HIP
#define VIBE_HAVE_SYCL VIBE_BACKEND_SYCL
#define VIBE_HAVE_GPU  (VIBE_BACKEND_CUDA || VIBE_BACKEND_HIP || VIBE_BACKEND_SYCL)

// ---------------------------------------------------------------------------
// 2. 设备函数限定符（CPU 与 SYCL-host 路径下为空）
// ---------------------------------------------------------------------------
#if VIBE_HAVE_CUDA
#  define VIBE_HD   __host__ __device__
#  define VIBE_INLINE_DEVICE __device__ __forceinline__
#elif VIBE_HAVE_HIP
#  define VIBE_HD   __host__ __device__
#  define VIBE_INLINE_DEVICE __device__ inline
#else
#  define VIBE_HD
#  define VIBE_INLINE_DEVICE inline
#endif

namespace vibe::gpu {

// ---------------------------------------------------------------------------
// 3. 后端枚举
// ---------------------------------------------------------------------------
/// 计算后端。数值与 \ref device.hpp 中的探测宏无耦合：CPU 构建下
/// \ref active_backend 恒为 \c Backend::CPU，即使枚举里存在其它取值。
enum class Backend {
  CPU = 0,   ///< 主机多核（串行或 OpenMP），参考实现
  CUDA = 1,  ///< NVIDIA GPU
  HIP = 2,   ///< AMD GPU（或 HIP-on-NVIDIA）
  SYCL = 3   ///< 厂商无关异构运行时
};

/// 后端名（大写，稳定标识，可用于配置与日志）
inline const char* to_string(Backend b) noexcept {
  switch (b) {
    case Backend::CPU: return "CPU";
    case Backend::CUDA: return "CUDA";
    case Backend::HIP: return "HIP";
    case Backend::SYCL: return "SYCL";
  }
  return "UNKNOWN";
}

/// 由字符串解析后端；接受大小写不敏感的 "cpu"/"cuda"/"hip"/"sycl"，
/// 以及常见别名 "gpu"（= CUDA，取自配置文件的习惯写法）。
/// 无法识别时返回 \c Backend::CPU 并把 \p ok 置 false。
Backend from_string(const std::string& s, bool* ok = nullptr) noexcept;

/// 编译期后端。仅用于 \c constexpr 分支；运行时分支请用 \ref active_backend。
inline constexpr Backend compiled_backend() noexcept {
#if VIBE_BACKEND_CUDA
  return Backend::CUDA;
#elif VIBE_BACKEND_HIP
  return Backend::HIP;
#elif VIBE_BACKEND_SYCL
  return Backend::SYCL;
#else
  return Backend::CPU;
#endif
}

/// 该后端是否为异构加速器（需要显式内存拷贝与流同步）
inline constexpr bool is_accelerator(Backend b) noexcept {
  return b == Backend::CUDA || b == Backend::HIP || b == Backend::SYCL;
}

/// 该后端是否为当前编译产物所支持（编译期常量，运行时常量折叠）
inline constexpr bool is_compiled(Backend b) noexcept {
  return b == Backend::CPU ||
#if VIBE_BACKEND_CUDA
         b == Backend::CUDA ||
#endif
#if VIBE_BACKEND_HIP
         b == Backend::HIP ||
#endif
#if VIBE_BACKEND_SYCL
         b == Backend::SYCL ||
#endif
         false;
}

/// 后端能力：把"能否做某件事"集中在一处，避免业务代码散落 \#if
struct BackendCaps {
  bool accelerator = false;      ///< 是否独立设备内存
  bool unified_memory = false;   ///< 是否支持统一寻址（CPU 恒 true）
  bool pinned_memory = false;    ///< 是否支持页锁定主机内存
  bool async_streams = false;    ///< 是否支持多流并发
  bool fp16 = false;             ///< 是否原生支持 FP16 运算
  bool bf16 = false;             ///< 是否原生支持 BF16 运算
  bool tensor_core = false;      ///< 是否可访问矩阵乘累加单元（[G1][G5]）
  bool events_timing = false;    ///< 是否有事件计时（[G10] 第 6 章）
  bool cooperative_groups = false;  ///< 是否有 cooperative groups / 全局同步
  bool atomic_float = false;     ///< 是否支持浮点原子加
};

/// 查询某后端的能力矩阵（CPU 返回保守值）
BackendCaps backend_caps(Backend b) noexcept;

// ---------------------------------------------------------------------------
// 4. 运行时错误检查
// ---------------------------------------------------------------------------
namespace detail {
/// GPU 运行时错误转字符串（CUDA: cudaGetErrorString；HIP: hipGetErrorString）
std::string gpu_error_string(int code);
/// 统一抛出点：p expr 为出错的表达式文本，p code 为 API 返回码
[[noreturn]] void throw_gpu_error(const char* expr, const char* file, int line, int code);
}  // namespace detail

namespace vibe::gpu::detail {
/// CPU 后端的空检查（模板化以接受任意表达式）
template <class T>
inline void check_cpu_fallback(T&&) noexcept {}
}  // namespace vibe::gpu::detail

// 编译期自检：恰好一个后端被激活（[B11] 编译期不变量）
static_assert((VIBE_BACKEND_CPU + VIBE_BACKEND_CUDA + VIBE_BACKEND_HIP + VIBE_BACKEND_SYCL) == 1,
              "VIBE: 必须恰好激活一个计算后端（CPU/CUDA/HIP/SYCL）");

// ---------------------------------------------------------------------------
// 5. 检查宏
// ---------------------------------------------------------------------------
#if VIBE_HAVE_GPU
/// 执行 GPU 运行时 API 并检查返回码；失败时抛 \ref vibe::Error（含错误码文本）。
/// CPU 后端下等价于 \c ((void)(expr))。
#  define VIBE_GPU_CHECK(expr)                                                    \
    do {                                                                          \
      const int vibe_gpu_code_ = static_cast<int>(expr);                          \
      if (VIBE_UNLIKELY(vibe_gpu_code_ != 0)) {                                   \
        ::vibe::gpu::detail::throw_gpu_error(#expr, __FILE__, __LINE__, vibe_gpu_code_); \
      }                                                                           \
    } while (false)
#else
#  define VIBE_GPU_CHECK(expr) ::vibe::gpu::detail::check_cpu_fallback((void)(expr))
#endif

}  // namespace vibe::gpu

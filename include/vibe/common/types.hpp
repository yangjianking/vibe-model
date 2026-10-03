#pragma once
/// @file types.hpp
/// @brief 全局基础类型与数值工具。
///
/// 设计约束（见 docs/design/00_architecture.md 第 4 节）：
///   1. 守恒量的存储一律为 `Real`（默认 double），与 GPU 计算精度无关；
///   2. 索引一律为有符号 `Int` / `Index`，禁止 unsigned 参与下标运算；
///   3. 裸指针只允许出现在 `vibe::common` 与 `vibe::gpu` 层。
///
/// 文献：[G2] Kahan 补偿求和、[G6] 混合精度、[B11] 浮点误差分析。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// 精度选择宏
//   VIBE_PRECISION = 0 : double 全精度（默认，数值研究 / 同化）
//   VIBE_PRECISION = 1 : single 全单精度（教学 / 低内存）
//   VIBE_PRECISION = 2 : mixed  状态 double，趋势 float（见 gpu/precision.hpp）
// ---------------------------------------------------------------------------
#ifndef VIBE_PRECISION
#define VIBE_PRECISION 0
#endif

// ---------------------------------------------------------------------------
// 编译期属性宏
// ---------------------------------------------------------------------------
#define VIBE_UNUSED(x) ((void)(x))
#if defined(_MSC_VER)
#  define VIBE_FORCE_INLINE __forceinline
#  define VIBE_RESTRICT __restrict
#  define VIBE_NODISCARD [[nodiscard]]
#else
#  define VIBE_FORCE_INLINE inline __attribute__((always_inline))
#  define VIBE_RESTRICT __restrict__
#  define VIBE_NODISCARD [[nodiscard]]
#endif
#define VIBE_LIKELY(x) (x)
#define VIBE_UNLIKELY(x) (x)

namespace vibe {

// ---------------------------------------------------------------------------
// 基础标量类型
// ---------------------------------------------------------------------------
#if VIBE_PRECISION == 1
using Real = float;
#else
using Real = double;
#endif

using Int   = std::int32_t;   ///< 网格索引（局部、含 halo）
using Index = std::int64_t;   ///< 全局序号、大数组偏移
using Size  = std::size_t;

inline constexpr Int  kInvalidIndex = static_cast<Int>(-1);
inline constexpr Real kEpsilon = std::numeric_limits<Real>::epsilon();
inline constexpr Real kHuge    = std::numeric_limits<Real>::max();

/// 编译期类型名，用于日志与 NetCDF 属性
inline constexpr const char* real_type_name() noexcept {
#if VIBE_PRECISION == 1
  return "float32";
#elif VIBE_PRECISION == 2
  return "mixed(double-state/float-tendency)";
#else
  return "float64";
#endif
}

// ---------------------------------------------------------------------------
// 标量工具
// ---------------------------------------------------------------------------
template <class T> constexpr T sqr(T x) noexcept { return x * x; }
template <class T> constexpr T cube(T x) noexcept { return x * x * x; }
template <class T> constexpr T pow4(T x) noexcept { return sqr(sqr(x)); }

template <class T> constexpr int sign(T x) noexcept { return (T(0) < x) - (x < T(0)); }
template <class T> constexpr T clamp(T v, T lo, T hi) noexcept { return v < lo ? lo : (v > hi ? hi : v); }

/// 线性插值：a + w (b - a)
template <class T> constexpr T lerp(T a, T b, Real w) noexcept { return a + w * (b - a); }

/// 三次 Hermite 平滑阶跃，w in [0,1]；用于 Davies 松弛区 [N1]
template <class T> constexpr T smoothstep(T w) noexcept { return w * w * (T(3) - T(2) * w); }

/// 对称相对误差，用于测试与点积检验
template <class T> constexpr T rel_error(T a, T b) noexcept {
  const T d = std::abs(a - b);
  const T s = std::abs(a) + std::abs(b);
  return s > T(0) ? d / s : d;
}

/// 带容差的近似比较
template <class T> constexpr bool approx(T a, T b, T tol = T(1e-10)) noexcept {
  return rel_error(a, b) <= tol;
}

/// 把角度折回 (-pi, pi]
inline Real wrap_angle(Real a) noexcept {
  constexpr Real kTwoPi = Real(6.283185307179586);
  a -= kTwoPi * std::floor((a + kTwoPi / 2) / kTwoPi);
  return a;
}

/// 保证数值安全的除法（分母加符号相关的微扰）
template <class T> constexpr T safe_div(T num, T den, T eps = T(1e-30)) noexcept {
  const T d = den >= T(0) ? (den + eps) : (den - eps);
  return num / d;
}

}  // namespace vibe

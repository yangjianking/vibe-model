#pragma once
/// @file precision.hpp
/// @brief 精度枚举、精度特征、混合精度策略、补偿求和与半精度可移植实现。
///
/// 本文件要实现的核心思想是**角色化精度（role-based precision）**：
/// 同一份动力学代码里，"守恒量怎么存"、"趋势怎么算"、"归约怎么累加"三件事
/// 允许取不同的浮点类型，从而在可接受的误差预算内换取吞吐
/// （[G6] ENDGame 混合精度、[G7] 精度基准、[G1] 混合精度训练）。
///
/// 四类实体
/// --------
///   1. \ref PrecisionTraits<P> —— 编译期特征（类型、epsilon、最大值、名称）；
///   2. \ref PrecisionPolicy  —— 运行时策略（storage/compute/reduce + 两个开关）；
///   3. \ref CompensatedSum<S,T> —— Kahan [G2] / Neumaier [G3] 补偿求和；
///   4. \ref half_t / \ref bfloat16_t / \ref demote / \ref promote
///      —— 不依赖 \c cuda_fp16.h 的最小可移植半精度，保证无 GPU 环境可编译。
///
/// 半精度的位级约定（[G1] 第 3 节、IEEE-754-2008 binary16/bfloat16）
/// ---------------------------------------------------------------
///   \c half_t      : 1 符号 + 5 指数（偏置 15）+ 10 尾数
///   \c bfloat16_t  : 1 符号 + 8 指数（偏置 127）+ 7 尾数（= FP32 截断高 16 位）
///
/// 舍入策略：\ref half_t::to_float / \ref bfloat16_t::to_float 为精确提升；
/// 降精度使用 **round-to-nearest-even**（\ref demote 的默认模式），
/// 与 CUDA 的 \c __float2half_rn 一致；也提供截断模式用于误差对照实验。
///
/// 复杂度：所有转换 O(1)；\ref CompensatedSum::add O(1)；
/// \ref PrecisionGuard 构造/析构 O(1)（只改一个线程局部指针）。
///
/// 文献：[G1] Micikevicius et al. (2018) 混合精度训练；[G2] Kahan (1965)；
///       [G3] Neumaier (1974)；[G4] Ogita, Rump & Oishi (2005) 精确求和与点积；
///       [G5] Haidar et al. (2018) 张量核迭代精化；[G6] Klöwer et al. (2019)；
///       [G7] Düben & Palmer (2014)；[B11] Higham (2002) 第 1、4 章。

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/gpu/backend.hpp"

namespace vibe::gpu {

// ===========================================================================
// 1. Precision 枚举与特征
// ===========================================================================
/// 浮点精度档位。顺序即"位宽"，可用于排序与取较大者。
enum class Precision {
  FP16 = 0,  ///< IEEE-754 binary16：1+5+10，epsilon = 2^-11 = 4.883e-4
  BF16 = 1,  ///< bfloat16：1+8+7，epsilon = 2^-8 = 3.906e-3
  FP32 = 2,  ///< IEEE-754 binary32：1+8+23，epsilon = 2^-24 = 5.960e-8
  FP64 = 3   ///< IEEE-754 binary64：1+11+52，epsilon = 2^-53 = 1.110e-16
};

inline const char* to_string(Precision p) noexcept {
  switch (p) {
    case Precision::FP16: return "FP16";
    case Precision::BF16: return "BF16";
    case Precision::FP32: return "FP32";
    case Precision::FP64: return "FP64";
  }
  return "UNKNOWN";
}

/// 大小写折叠为小写 ASCII（不依赖 <cctype> 与区域设置，便于内联与 constexpr 使用）
inline std::string lower_ascii(const std::string& s) {
  std::string t(s.size(), '\0');
  for (std::size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    t[i] = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  }
  return t;
}

/// 解析精度字符串（大小写不敏感；接受 "half"、"single"、"double"、"bfloat" 等别名）
inline Precision precision_from_string(const std::string& s, bool* ok = nullptr) noexcept {
  const std::string t = lower_ascii(s);
  if (ok != nullptr) *ok = true;
  if (t == "fp16" || t == "half" || t == "binary16" || t == "h") return Precision::FP16;
  if (t == "bf16" || t == "bfloat16" || t == "bfloat") return Precision::BF16;
  if (t == "fp32" || t == "single" || t == "float" || t == "binary32") return Precision::FP32;
  if (t == "fp64" || t == "double" || t == "real" || t == "binary64") return Precision::FP64;
  if (ok != nullptr) *ok = false;
  return Precision::FP64;
}

/// 精度位宽（存储用位数）
inline constexpr int precision_bits(Precision p) noexcept {
  return p == Precision::FP16 ? 16 : (p == Precision::BF16 ? 16 : (p == Precision::FP32 ? 32 : 64));
}

/// 精度档位的序数（FP16=0 … FP64=3），便于取 max/min
inline constexpr int precision_rank(Precision p) noexcept { return static_cast<int>(p); }

/// 取两者中更"宽"的精度（max(rank)）
inline constexpr Precision precision_max(Precision a, Precision b) noexcept {
  return precision_rank(a) >= precision_rank(b) ? a : b;
}

// --- 前向声明：本文件后面定义的可移植半精度类型 ---------------------------
struct half_t;
struct bfloat16_t;

// ===========================================================================
// 2. PrecisionTraits<P>
// ===========================================================================
/// 编译期精度特征。主模板只声明不定义，四个特化分别给出
/// \c type、\c storage_bits、\c epsilon()、\c max()、\c min_positive()、
/// \c name()、\c is_half()。
///
/// epsilon 的定义与 IEEE-754 保持一致：**1 与下一个可表示数之差**，
/// 即 FP16 = 2^-10 的 2 倍关系说明见下：
///   * FP16 尾数 10 位，最大相对舍入误差为 2^-11，故
///     \f$ \epsilon = 2^{-10} \f$（1 到下一个可表示数的间隔的一半的两倍）……
///     为与 \c std::numeric_limits 习惯一致，这里给出
///     \c epsilon = 2^{-11}（\c numeric_limits<float>::epsilon 的定义在
///     binary32 下为 2^-23；同理 binary16 为 2^-10 时对应 "1 的下一个间隔"，
///     本实现统一采用 \c numeric_limits 语义：|1-下一次可表示数|，
///     FP16 为 2^-10，BF16 为 2^-7）。
///
/// 为避免歧义，本文件同时提供：
///   * \ref PrecisionTraits<P>::epsilon —— \c numeric_limits 语义（1 的间隔）
///   * \ref PrecisionTraits<P>::unit_roundoff —— 单位舍入误差 u = epsilon/2
/// 误差传播分析（\ref docs/design/05_gpu_hybrid_precision.md 第 3 节）使用 u。
template <Precision P>
struct PrecisionTraits;

namespace detail {
/// 整数幂的 constexpr 版本（C++20 的 std::pow 非 constexpr）
template <int E>
constexpr double pow2() noexcept {
  double v = 1.0;
  for (int i = 0; i < E; ++i) v *= 2.0;
  for (int i = 0; i > E; --i) v *= 0.5;
  return v;
}
}  // namespace detail

template <>
struct PrecisionTraits<Precision::FP16> {
  using type = half_t;
  static constexpr Precision value = Precision::FP16;
  static constexpr int mantissa_bits = 10;
  static constexpr int exponent_bits = 5;
  static constexpr int total_bits = 16;
  static constexpr int max_exponent = 15;          ///< 2^15 = 32768，正规数上界
  static constexpr int min_exponent = -14;         ///< 正规数下界 2^-14
  /// numeric_limits 语义：|1 - nextafter(1)| = 2^-10
  static constexpr double epsilon() noexcept { return detail::pow2<-10>(); }
  /// 单位舍入误差 u = epsilon/2 = 2^-11
  static constexpr double unit_roundoff() noexcept { return detail::pow2<-11>(); }
  static constexpr double max() noexcept { return (2.0 - detail::pow2<-10>()) * detail::pow2<15>(); }
  static constexpr double min_positive() noexcept { return detail::pow2<-14>(); }
  static constexpr double min_subnormal() noexcept { return detail::pow2<-24>(); }
  static constexpr double decimal_digits() noexcept { return 3.31; }
  static constexpr bool is_half() noexcept { return true; }
  static constexpr bool native_arithmetic() noexcept { return false; }
  static constexpr const char* name() noexcept { return "FP16"; }
};

template <>
struct PrecisionTraits<Precision::BF16> {
  using type = bfloat16_t;
  static constexpr Precision value = Precision::BF16;
  static constexpr int mantissa_bits = 7;
  static constexpr int exponent_bits = 8;
  static constexpr int total_bits = 16;
  static constexpr int max_exponent = 127;
  static constexpr int min_exponent = -126;
  /// |1 - nextafter(1)| = 2^-7
  static constexpr double epsilon() noexcept { return detail::pow2<-7>(); }
  static constexpr double unit_roundoff() noexcept { return detail::pow2<-8>(); }
  static constexpr double max() noexcept { return static_cast<double>(3.38953139e38f); }
  static constexpr double min_positive() noexcept { return detail::pow2<-126>(); }
  static constexpr double min_subnormal() noexcept { return detail::pow2<-133>(); }
  static constexpr double decimal_digits() noexcept { return 2.41; }
  static constexpr bool is_half() noexcept { return true; }
  static constexpr bool native_arithmetic() noexcept { return false; }
  static constexpr const char* name() noexcept { return "BF16"; }
};

template <>
struct PrecisionTraits<Precision::FP32> {
  using type = float;
  static constexpr Precision value = Precision::FP32;
  static constexpr int mantissa_bits = 23;
  static constexpr int exponent_bits = 8;
  static constexpr int total_bits = 32;
  static constexpr int max_exponent = 127;
  static constexpr int min_exponent = -126;
  static constexpr double epsilon() noexcept { return detail::pow2<-23>(); }
  static constexpr double unit_roundoff() noexcept { return detail::pow2<-24>(); }
  static constexpr double max() noexcept { return static_cast<double>(std::numeric_limits<float>::max()); }
  static constexpr double min_positive() noexcept { return static_cast<double>(std::numeric_limits<float>::min()); }
  static constexpr double min_subnormal() noexcept { return static_cast<double>(std::numeric_limits<float>::denorm_min()); }
  static constexpr double decimal_digits() noexcept { return 6.92; }
  static constexpr bool is_half() noexcept { return false; }
  static constexpr bool native_arithmetic() noexcept { return true; }
  static constexpr const char* name() noexcept { return "FP32"; }
};

template <>
struct PrecisionTraits<Precision::FP64> {
  using type = double;
  static constexpr Precision value = Precision::FP64;
  static constexpr int mantissa_bits = 52;
  static constexpr int exponent_bits = 11;
  static constexpr int total_bits = 64;
  static constexpr int max_exponent = 1023;
  static constexpr int min_exponent = -1022;
  static constexpr double epsilon() noexcept { return detail::pow2<-52>(); }
  static constexpr double unit_roundoff() noexcept { return detail::pow2<-53>(); }
  static constexpr double max() noexcept { return std::numeric_limits<double>::max(); }
  static constexpr double min_positive() noexcept { return std::numeric_limits<double>::min(); }
  static constexpr double min_subnormal() noexcept { return std::numeric_limits<double>::denorm_min(); }
  static constexpr double decimal_digits() noexcept { return 15.95; }
  static constexpr bool is_half() noexcept { return false; }
  static constexpr bool native_arithmetic() noexcept { return true; }
  static constexpr const char* name() noexcept { return "FP64"; }
};

/// 由 C++ 标量类型反查 \ref Precision（用于模板元编程）
template <class T>
struct precision_of;
template <> struct precision_of<float> { static constexpr Precision value = Precision::FP32; };
template <> struct precision_of<double> { static constexpr Precision value = Precision::FP64; };
template <> struct precision_of<half_t> { static constexpr Precision value = Precision::FP16; };
template <> struct precision_of<bfloat16_t> { static constexpr Precision value = Precision::BF16; };

// ===========================================================================
// 3. 可移植半精度
// ===========================================================================
/// IEEE-754 binary16 的最小可移植实现（不包含 cuda_fp16.h）。
///
/// 位布局（16 位）：\c s(1) e(5) m(10)，指数偏置 15。
///   * \c e = 0,          m = 0     -> ±0
///   * \c e = 0,          0 < m     -> 次正规数  \f$ \pm m \cdot 2^{-24} \f$
///   * \c 1 <= e <= 30              -> 正规数    \f$ \pm (1 + m/1024) \cdot 2^{e-15} \f$
///   * \c e = 31,         m = 0     -> ±Inf
///   * \c e = 31,         m != 0    -> NaN
///
/// 存储用 \c std::uint16_t，保证 2 字节、可平凡拷贝，才能放进 \c DeviceBuffer。
/// 算术运算在 host 侧先提升到 \c float 再降回（\ref native_arithmetic 为 false），
/// 与 [G1] 强调的"半精度只用于存储与乘累加、累加必须更宽"完全一致。
struct half_t {
  std::uint16_t bits = 0;

  constexpr half_t() noexcept = default;
  constexpr explicit half_t(std::uint16_t raw) noexcept : bits(raw) {}

  /// float -> half，round-to-nearest-even（等价于 \c __float2half_rn，[G10]）
  ///
  /// 分支结构对应 IEEE-754 的四种情形（正规 / 次正规 / Inf / NaN）：
  ///   * \c absf >= 0x7f800000          -> Inf 或 NaN（保持 quiet 位）
  ///   * \c absf >= 0x47800000          -> 幅值超过 half 最大值，上溢为 Inf
  ///   * \c absf <  0x38800000          -> 次正规或 0
  ///   * 其余                             -> 正规数（指数 -112，尾数右移 13 位）
  static half_t from_float(float x) noexcept {
    const std::uint32_t f = bits_of(x);
    const std::uint16_t sign = static_cast<std::uint16_t>((f >> 16) & 0x8000u);
    const std::uint32_t absf = f & 0x7fffffffu;
    if (absf >= 0x7f800000u) {  // Inf / NaN
      const bool is_nan = absf > 0x7f800000u;
      return half_t(static_cast<std::uint16_t>(sign | 0x7c00u | (is_nan ? 0x0200u : 0u)));
    }
    if (absf >= 0x47800000u) {  // 上溢 -> Inf
      return half_t(static_cast<std::uint16_t>(sign | 0x7c00u));
    }
    if (absf < 0x38800000u) {   // 次正规或 0
      if (absf < 0x33000000u) return half_t(sign);  // 舍入到 ±0
      const int shift = 126 - static_cast<int>((absf >> 23) & 0xffu);
      const std::uint32_t mant = (absf & 0x007fffffu) | 0x00800000u;
      std::uint32_t out = mant >> static_cast<unsigned>(shift + 1);
      const std::uint32_t rem = mant & ((1u << static_cast<unsigned>(shift + 1)) - 1u);
      const std::uint32_t half_bit = 1u << static_cast<unsigned>(shift);
      if (rem > half_bit || (rem == half_bit && (out & 1u))) ++out;
      return half_t(static_cast<std::uint16_t>(sign | out));  // 进位溢出即进入正规数区间
    }
    // 正规数：指数调整 + round-to-nearest-even
    std::uint32_t exp = ((absf >> 23) & 0xffu) - 112u;
    std::uint32_t mant = absf & 0x007fffffu;
    const std::uint32_t round_part = mant & 0x1fffu;
    mant >>= 13;
    if (round_part > 0x1000u || (round_part == 0x1000u && (mant & 1u))) {
      ++mant;
      if (mant == 0x400u) {
        mant = 0;
        ++exp;
        if (exp >= 0x1fu) return half_t(static_cast<std::uint16_t>(sign | 0x7c00u));
      }
    }
    return half_t(static_cast<std::uint16_t>(sign | (exp << 10) | mant));
  }

  /// half -> float（精确，无舍入）
  float to_float() const noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x03ffu;
    std::uint32_t f = 0;
    if (exp == 0) {
      if (mant == 0) {
        f = sign;  // ±0
      } else {
        // 次正规 -> 归一化
        std::uint32_t e = 127u - 15u + 1u;
        std::uint32_t m = mant;
        while ((m & 0x0400u) == 0) {
          m <<= 1;
          --e;
        }
        m &= 0x03ffu;
        f = sign | (e << 23) | (m << 13);
      }
    } else if (exp == 0x1fu) {
      f = sign | 0x7f800000u | (mant ? 0x00400000u : 0u);  // Inf / NaN
    } else {
      f = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return from_bits(f);
  }

  /// 截断模式（round-toward-zero）；用于误差上界实验与 rpe 风格模拟 [G8]
  static half_t from_float_trunc(float x) noexcept {
    const std::uint32_t f = bits_of(x);
    const std::uint16_t sign = static_cast<std::uint16_t>((f >> 16) & 0x8000u);
    const std::int32_t e = static_cast<std::int32_t>((f >> 23) & 0xffu) - 127 + 15;
    const std::uint32_t mant = f & 0x007fffffu;
    if (((f >> 23) & 0xffu) == 0xffu) {
      return half_t(static_cast<std::uint16_t>(sign | 0x7c00u | (mant ? 0x0200u : 0u)));
    }
    if (e >= 31) return half_t(static_cast<std::uint16_t>(sign | 0x7c00u));
    if (e <= 0) {
      if (e < -10) return half_t(sign);
      const std::uint32_t m = (mant | 0x00800000u) >> static_cast<unsigned>(1 - e);
      return half_t(static_cast<std::uint16_t>(sign | (m >> 13)));
    }
    return half_t(static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10) | (mant >> 13)));
  }

  constexpr bool is_nan() const noexcept { return ((bits >> 10) & 0x1fu) == 0x1fu && (bits & 0x03ffu) != 0; }
  constexpr bool is_inf() const noexcept { return ((bits >> 10) & 0x1fu) == 0x1fu && (bits & 0x03ffu) == 0; }
  constexpr bool is_zero() const noexcept { return (bits & 0x7fffu) == 0; }
  constexpr bool sign_bit() const noexcept { return (bits & 0x8000u) != 0; }

  static constexpr half_t zero() noexcept { return half_t(static_cast<std::uint16_t>(0)); }
  static constexpr half_t inf() noexcept { return half_t(static_cast<std::uint16_t>(0x7c00u)); }
  static constexpr half_t quiet_nan() noexcept { return half_t(static_cast<std::uint16_t>(0x7e00u)); }

  friend bool operator==(half_t a, half_t b) noexcept { return a.bits == b.bits; }
  friend bool operator!=(half_t a, half_t b) noexcept { return a.bits != b.bits; }

 private:
  static std::uint32_t bits_of(float x) noexcept {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
  }
  static float from_bits(std::uint32_t u) noexcept {
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
  }
};

/// bfloat16：与 FP32 同指数范围，取高 16 位（尾数 7 位）。
/// 因为指数位与 FP32 完全相同，\c from_float 只是"舍入到最近的 16 位截断"，
/// 溢出行为与 FP32 一致（不会像 half 一样轻易上溢）——这正是它在数值模式里
/// 比 half 更受欢迎的原因（[G1][G6]）。
struct bfloat16_t {
  std::uint16_t bits = 0;

  constexpr bfloat16_t() noexcept = default;
  constexpr explicit bfloat16_t(std::uint16_t raw) noexcept : bits(raw) {}

  /// round-to-nearest-even（等价于 \c __float2bfloat16_rn）
  static bfloat16_t from_float(float x) noexcept {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN：保持 quiet 位并存高 16 位
      return bfloat16_t(static_cast<std::uint16_t>((u >> 16) | 0x0040u));
    }
    const std::uint32_t lsb = (u >> 16) & 1u;
    const std::uint32_t rounding_bias = 0x7fffu + lsb;
    u += rounding_bias;
    return bfloat16_t(static_cast<std::uint16_t>(u >> 16));
  }

  /// 截断（round-toward-zero）
  static bfloat16_t from_float_trunc(float x) noexcept {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return bfloat16_t(static_cast<std::uint16_t>(u >> 16));
  }

  /// bfloat16 -> float 是精确的（左移 16 位补零）
  float to_float() const noexcept {
    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
  }

  constexpr bool is_nan() const noexcept {
    return ((bits >> 7) & 0xffu) == 0xffu && (bits & 0x007fu) != 0;
  }
  constexpr bool is_inf() const noexcept {
    return ((bits >> 7) & 0xffu) == 0xffu && (bits & 0x007fu) == 0;
  }
  constexpr bool is_zero() const noexcept { return (bits & 0x7fffu) == 0; }
  constexpr bool sign_bit() const noexcept { return (bits & 0x8000u) != 0; }

  static constexpr bfloat16_t zero() noexcept { return bfloat16_t(static_cast<std::uint16_t>(0)); }
  static constexpr bfloat16_t inf() noexcept { return bfloat16_t(static_cast<std::uint16_t>(0x7f80u)); }
  static constexpr bfloat16_t quiet_nan() noexcept { return bfloat16_t(static_cast<std::uint16_t>(0x7fc0u)); }

  friend bool operator==(bfloat16_t a, bfloat16_t b) noexcept { return a.bits == b.bits; }
  friend bool operator!=(bfloat16_t a, bfloat16_t b) noexcept { return a.bits != b.bits; }
};

static_assert(sizeof(half_t) == 2, "half_t 必须是 2 字节");
static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t 必须是 2 字节");
static_assert(std::is_trivially_copyable<half_t>::value, "half_t 必须平凡可拷贝");
static_assert(std::is_trivially_copyable<bfloat16_t>::value, "bfloat16_t 必须平凡可拷贝");

// ===========================================================================
// 4. demote / promote
// ===========================================================================
/// 降精度：\c Dst(demote<Src,Dst>(src))。
/// 支持 float<->half/bfloat、double->float、half<->bfloat 及恒等映射。
/// CPU 后端不做任何隐式行为变化，语义与 GPU 的 \c __float2half_rn 一致。
template <class Src, class Dst>
inline Dst demote(Src src) noexcept {
  if constexpr (std::is_same<Src, Dst>::value) {
    return src;
  } else if constexpr (std::is_same<Src, double>::value && std::is_same<Dst, float>::value) {
    return static_cast<float>(src);
  } else if constexpr (std::is_same<Src, float>::value && std::is_same<Dst, double>::value) {
    return static_cast<double>(src);
  } else if constexpr (std::is_same<Dst, half_t>::value && std::is_same<Src, float>::value) {
    return half_t::from_float(src);
  } else if constexpr (std::is_same<Dst, half_t>::value && std::is_same<Src, double>::value) {
    return half_t::from_float(static_cast<float>(src));
  } else if constexpr (std::is_same<Dst, bfloat16_t>::value && std::is_same<Src, float>::value) {
    return bfloat16_t::from_float(src);
  } else if constexpr (std::is_same<Dst, bfloat16_t>::value && std::is_same<Src, double>::value) {
    return bfloat16_t::from_float(static_cast<float>(src));
  } else if constexpr (std::is_same<Dst, bfloat16_t>::value && std::is_same<Src, half_t>::value) {
    return bfloat16_t::from_float(src.to_float());
  } else if constexpr (std::is_same<Dst, half_t>::value && std::is_same<Src, bfloat16_t>::value) {
    return half_t::from_float(src.to_float());
  } else if constexpr (std::is_same<Dst, float>::value && (std::is_same<Src, half_t>::value || std::is_same<Src, bfloat16_t>::value)) {
    return src.to_float();
  } else if constexpr (std::is_same<Dst, double>::value && (std::is_same<Src, half_t>::value || std::is_same<Src, bfloat16_t>::value)) {
    return static_cast<double>(src.to_float());
  } else {
    static_assert(sizeof(Src) == 0, "demote: 不支持的精度组合");
  }
}

/// 截断版本（round-toward-zero），用于构造误差上界场景 [G8]
template <class Src, class Dst>
inline Dst demote_trunc(Src src) noexcept {
  if constexpr (std::is_same<Dst, half_t>::value) {
    return half_t::from_float_trunc(static_cast<float>(src));
  } else if constexpr (std::is_same<Dst, bfloat16_t>::value) {
    return bfloat16_t::from_float_trunc(static_cast<float>(src));
  } else {
    return demote<Src, Dst>(src);
  }
}

/// 升精度：始终精确（半精度 -> float/double 无损）
template <class Src, class Dst>
inline Dst promote(Src src) noexcept {
  if constexpr (std::is_same<Src, half_t>::value) {
    return static_cast<Dst>(src.to_float());
  } else if constexpr (std::is_same<Src, bfloat16_t>::value) {
    return static_cast<Dst>(src.to_float());
  } else {
    return static_cast<Dst>(src);
  }
}

/// 把任意标量统一提升到 \c double 参与归约（最后一步"终极精度"）
template <class T>
inline double widen(T x) noexcept {
  if constexpr (std::is_same<T, half_t>::value || std::is_same<T, bfloat16_t>::value) {
    return static_cast<double>(x.to_float());
  } else {
    return static_cast<double>(x);
  }
}

// ===========================================================================
// 5. PrecisionPolicy
// ===========================================================================
/// 角色化精度策略。
///
/// 三个角色 + 两个开关
/// -------------------
///   * \c storage —— 预报量的**存储**精度。按 [G6] 与架构第 4 节铁律 1，
///     守恒量必须 FP64，即使计算用 FP32；
///   * \c compute —— 趋势/通量的**计算**精度。默认 FP32，误差在
///     时间步内不累积（RK 每步重新从 FP64 状态出发）；
///   * \c reduce —— 归约（范数、点积、质量收支）的累加精度，默认 FP64；
///   * \c compensated_summation —— 是否启用 Kahan/Neumaier 补偿 [G2][G3]；
///   * \c iterative_refinement —— Helmholtz 解是否做双精度残差修正 [G5]。
///
/// 误差预算（详见 docs/design/05_gpu_hybrid_precision.md 第 3 节）：
/// 单步舍入误差 \f$ \epsilon_{step} \approx u_{compute}\|q\| \f$，
/// 若每步误差**独立随机**，N 步累积为 \f$ \sqrt{N}\,u_{compute}\|q\| \f$；
/// 若系统误差相关（例如保守格式的收支漂移），则按 N 线性累积，
/// 因此质量/能量收支必须走补偿求和 + FP64。
struct PrecisionPolicy {
  Precision storage = Precision::FP64;  ///< 守恒量必须 FP64（架构铁律）
  Precision compute = Precision::FP32;  ///< 默认单精度算趋势
  Precision reduce = Precision::FP64;   ///< Kahan/Neumaier 累加
  bool compensated_summation = true;    ///< 启用补偿求和 [G2][G3]
  bool iterative_refinement = true;     ///< Helmholtz 双精度精化 [G5]
  /// 存储精度低于 FP64 时是否允许（教学实验用；默认禁止）
  bool allow_lossy_storage = false;

  /// 研究模式：全 FP64 + 补偿求和 + 迭代精化（数值研究/同化的基准）
  static PrecisionPolicy research() noexcept {
    PrecisionPolicy p;
    p.storage = Precision::FP64;
    p.compute = Precision::FP64;
    p.reduce = Precision::FP64;
    p.compensated_summation = true;
    p.iterative_refinement = true;
    return p;
  }

  /// 默认模式：FP64 存储、FP32 计算、FP64 补偿归约、迭代精化
  static PrecisionPolicy default_policy() noexcept { return PrecisionPolicy{}; }

  /// 快速模式：存储仍 FP64（铁律），计算 FP32，归约 FP32 且关闭补偿求和。
  /// 用于教学演示"关掉补偿求和会漂多远"，不建议做业务生产。
  static PrecisionPolicy fast() noexcept {
    PrecisionPolicy p;
    p.storage = Precision::FP64;
    p.compute = Precision::FP32;
    p.reduce = Precision::FP32;
    p.compensated_summation = false;
    p.iterative_refinement = false;
    return p;
  }

  /// 半精度实验模式：存储 FP32（损失可控），计算 BF16，归约 FP32 补偿
  static PrecisionPolicy half_experiment() noexcept {
    PrecisionPolicy p;
    p.storage = Precision::FP32;
    p.compute = Precision::BF16;
    p.reduce = Precision::FP32;
    p.compensated_summation = true;
    p.iterative_refinement = true;
    p.allow_lossy_storage = true;
    return p;
  }

  /// 计算精度对应的单位舍入误差 u（误差传播用）
  double compute_u() const noexcept { return unit_roundoff(compute); }
  /// 存储精度对应的单位舍入误差
  double storage_u() const noexcept { return unit_roundoff(storage); }

  static double unit_roundoff(Precision p) noexcept {
    switch (p) {
      case Precision::FP16: return PrecisionTraits<Precision::FP16>::unit_roundoff();
      case Precision::BF16: return PrecisionTraits<Precision::BF16>::unit_roundoff();
      case Precision::FP32: return PrecisionTraits<Precision::FP32>::unit_roundoff();
      case Precision::FP64: return PrecisionTraits<Precision::FP64>::unit_roundoff();
    }
    return 0.0;
  }

  /// 校验：默认策略下守恒量必须 FP64（除非显式 allow_lossy_storage）
  bool valid() const noexcept {
    if (!allow_lossy_storage && storage != Precision::FP64) return false;
    if (precision_rank(compute) > precision_rank(storage)) return false;  // 计算精度不得超过存储
    if (precision_rank(reduce) < precision_rank(compute)) return false;   // 归约不得低于计算
    return true;
  }
  /// 不满足约束时抛 \ref vibe::Error
  void validate() const {
    if (valid()) return;
    throw Error(std::string("PrecisionPolicy 非法: ") + describe());
  }
  std::string describe() const {
    std::string s = "storage=";
    s += to_string(storage);
    s += " compute=";
    s += to_string(compute);
    s += " reduce=";
    s += to_string(reduce);
    s += compensated_summation ? " +kahan" : " -kahan";
    s += iterative_refinement ? " +refine" : " -refine";
    if (allow_lossy_storage) s += " (lossy-storage-allowed)";
    return s;
  }
};

// ===========================================================================
// 6. CompensatedSum
// ===========================================================================
/// 补偿求和算法选择
enum class SumAlgorithm {
  Kahan = 0,     ///< Kahan (1965) 经典补偿 [G2]
  Neumaier = 1,  ///< Neumaier (1974) 变体，对 |x| > |sum| 的情形也正确 [G3]
  TwoSum = 2     ///< 只用 Twosum 误差项的"精确累加骨架"，供 [G4] 的 TwoProd 复用
};

/// 补偿求和状态。模板参数 \c T 是**累加类型**（通常 float 或 double），
/// 内部误差项 \c c 与 \c T 同型——这是与 [G4] 完全一致的做法：
/// 补偿项不能跨精度累积，否则补偿失效。
///
/// Kahan [G2]：
/// \f[
///   y = x - c,\quad t = s + y,\quad c = (t - s) - y,\quad s = t
/// \f]
/// 有效误差 \f$ s + c \f$，即返回 \c value() = s + c。
/// 经典 Kahan 在 \f$ |x| > |s| \f$ 时会丢失补偿，因此有了 Neumaier。
///
/// Neumaier [G3]：
/// \f[
///   t = s + x
/// \f]
/// \f[
///   \text{if } |s| \ge |x|:\; c \mathrel{+}= (s - t) + x
///   \quad\text{else}:\; c \mathrel{+}= (x - t) + s
/// \f]
/// \f[ s = t \f]
/// Neumaier 对大小相消序列（如 1e8 个 1.0 后紧跟 -1e8）显著优于 Kahan。
///
/// 复杂度：add O(1)（Kahan 3 flops，Neumaier 分支 4~5 flops）；
/// 无分支版本可用 \c fabs 选择，见 \ref add_branchless。
///
/// 注意：本结构**不是**线程安全归约的终点，但可以安全地做 warp 内
/// shuffle 归约（每线程一个实例，最后按 Neumaier 合并，见 [G4] 第 4 节）。
template <SumAlgorithm S, class T>
struct CompensatedSumT {
  static_assert(std::is_floating_point<T>::value, "CompensatedSum 只支持浮点累加类型");
  T sum{};      ///< 主累加和
  T c{};        ///< 补偿项（误差项）
  T max_abs{};  ///< |x| 的历史最大值，供 Neumaier 的规模自适应与被舍弃项估计
  std::size_t count = 0;  ///< 加数个数（用于误差估计与测试）

  CompensatedSumT() noexcept = default;
  explicit CompensatedSumT(T init) noexcept : sum(init) {}
  explicit CompensatedSumT(const std::vector<T>& v) noexcept {
    sum = T(0);
    for (T x : v) add(x);
  }

  /// 加入一个加数（S = Kahan 时为 Kahan 步；S = Neumaier 时为 Neumaier 步）
  VIBE_FORCE_INLINE void add(T x) noexcept {
    ++count;
    const T ax = x < T(0) ? -x : x;
    if (ax > max_abs) max_abs = ax;
    if constexpr (S == SumAlgorithm::Kahan) {
      const T y = x - c;
      const T t = sum + y;
      c = (t - sum) - y;
      sum = t;
    } else if constexpr (S == SumAlgorithm::Neumaier) {
      const T t = sum + x;
      const T asum = sum < T(0) ? -sum : sum;
      if (asum >= ax) {
        c += (sum - t) + x;
      } else {
        c += (x - t) + sum;
      }
      sum = t;
    } else {  // TwoSum：与 Neumaier 同构但始终用 Twosum 公式，保留全部误差
      const T t = sum + x;
      c += (sum - t) + x;
      sum = t;
    }
  }

  /// 无分支 Neumaier（浮点比较不产生分支，便于 GPU 上的 warp 一致性）
  VIBE_FORCE_INLINE void add_branchless(T x) noexcept {
    ++count;
    const T ax = x < T(0) ? -x : x;
    if (ax > max_abs) max_abs = ax;
    if constexpr (S == SumAlgorithm::Kahan) {
      add(x);
    } else {
      const T t = sum + x;
      const T asum = sum < T(0) ? -sum : sum;
      const bool big = asum >= ax;
      const T e1 = big ? ((sum - t) + x) : ((x - t) + sum);
      c += e1;
      sum = t;
    }
  }

  /// 累加值（主和 + 补偿；一次加法即可，因 |c| <= u|sum|）
  VIBE_NODISCARD T value() const noexcept { return sum + c; }

  /// 只取主和（用于对照实验）
  VIBE_NODISCARD T raw() const noexcept { return sum; }
  /// 补偿项
  VIBE_NODISCARD T correction() const noexcept { return c; }

  /// 清空
  void reset() noexcept {
    sum = T(0);
    c = T(0);
    max_abs = T(0);
    count = 0;
  }

  /// 合并两个部分和：把 \p other 的两个分量按 Neumaier 规则并入本实例。
  /// 这是块级归约的正确合并方式（[G4] 第 4 节）：直接 s1+s2 会丢掉
  /// 各自的补偿项。
  void merge(const CompensatedSumT& other) noexcept {
    add(other.sum);
    add(other.c);
    if (other.max_abs > max_abs) max_abs = other.max_abs;
    // add() 会把 other 的两个分量各记一次，这里扣掉其一以保持 count = 加数个数
    count += other.count - 1;
  }

  /// 误差估计：|value() - 精确值| 的上界近似 \f$ n\,u\,|\sum|x_i|| \f$
  VIBE_NODISCARD T error_bound(T unit_roundoff) const noexcept {
    return static_cast<T>(2 * count) * unit_roundoff * (max_abs * static_cast<T>(count));
  }
};

/// 架构契约中的默认形式：Kahan 与 Neumaier 由 \ref sum_kind 选择。
/// 默认使用 Neumaier（对消减序列更稳，见 tests/unit/test_gpu_precision.cpp）。
template <class T, SumAlgorithm S = SumAlgorithm::Neumaier>
using CompensatedSum = CompensatedSumT<S, T>;

/// 按运行期算法选择做一次归约的便捷函数
template <class T>
inline T compensated_sum(const T* data, std::size_t n, SumAlgorithm algorithm,
                         T unit_roundoff = PrecisionTraits<Precision::FP64>::unit_roundoff()) noexcept {
  VIBE_UNUSED(unit_roundoff);
  switch (algorithm) {
    case SumAlgorithm::Kahan: {
      CompensatedSum<T, SumAlgorithm::Kahan> s;
      for (std::size_t i = 0; i < n; ++i) s.add(data[i]);
      return s.value();
    }
    case SumAlgorithm::TwoSum: {
      CompensatedSum<T, SumAlgorithm::TwoSum> s;
      for (std::size_t i = 0; i < n; ++i) s.add(data[i]);
      return s.value();
    }
    case SumAlgorithm::Neumaier:
    default: {
      CompensatedSum<T, SumAlgorithm::Neumaier> s;
      for (std::size_t i = 0; i < n; ++i) s.add(data[i]);
      return s.value();
    }
  }
}

/// 朴素求和（对照基线：tests 中用于展示补偿求和的收益）
template <class T>
inline T naive_sum(const T* data, std::size_t n) noexcept {
  T s = T(0);
  for (std::size_t i = 0; i < n; ++i) s += data[i];
  return s;
}

// ===========================================================================
// 7. PrecisionGuard
// ===========================================================================
/// RAII 精度策略切换。当前策略存放在**线程局部**变量中
/// （\ref current_policy），因为 GPU 内核按流执行、CPU 后端按线程执行，
/// 策略不应跨线程共享。
///
/// 用法
/// ----
/// @code
///   {
///     gpu::PrecisionGuard g(gpu::PrecisionPolicy::half_experiment());
///     advect_scalar(...);   // 内部通过 current_policy() 决定降精度行为
///   }                        // 自动恢复
/// @endcode
///
/// 复杂度：构造/析构 O(1)；未做动态分配。可以嵌套（保存/恢复）。
class PrecisionGuard {
 public:
  explicit PrecisionGuard(const PrecisionPolicy& policy) noexcept;
  /// 只改某个角色精度，其余继承当前策略
  PrecisionGuard(Precision storage, Precision compute, Precision reduce) noexcept;
  ~PrecisionGuard() noexcept;

  PrecisionGuard(const PrecisionGuard&) = delete;
  PrecisionGuard& operator=(const PrecisionGuard&) = delete;

  const PrecisionPolicy& policy() const noexcept { return policy_; }
  /// 是否处于激活状态（未被移动/未提前 release）
  bool active() const noexcept { return active_; }
  /// 提前恢复（析构时不再重复恢复）
  void release() noexcept;

 private:
  PrecisionPolicy saved_{};
  PrecisionPolicy policy_{};
  bool active_ = false;
};

/// 当前线程的精度策略（默认 = \ref PrecisionPolicy::default_policy）。
/// 线程局部：GPU 内核按流执行、CPU 内核按线程执行，策略不应跨线程共享。
inline PrecisionPolicy& policy_slot() noexcept {
  static thread_local PrecisionPolicy policy = PrecisionPolicy::default_policy();
  return policy;
}

inline const PrecisionPolicy& current_policy() noexcept { return policy_slot(); }

/// 直接设置当前线程策略（\ref PrecisionGuard 之外的低层入口）
inline void set_current_policy(const PrecisionPolicy& p) noexcept { policy_slot() = p; }

/// RAII 切换：构造时保存并写入新策略，析构或 \ref release 时恢复
inline PrecisionGuard::PrecisionGuard(const PrecisionPolicy& policy) noexcept
    : saved_(policy_slot()), policy_(policy) {
  policy_slot() = policy_;
  active_ = true;
}

inline PrecisionGuard::PrecisionGuard(Precision storage, Precision compute,
                                      Precision reduce) noexcept
    : saved_(policy_slot()), policy_(policy_slot()) {
  policy_.storage = storage;
  policy_.compute = compute;
  policy_.reduce = reduce;
  policy_slot() = policy_;
  active_ = true;
}

inline PrecisionGuard::~PrecisionGuard() noexcept { release(); }

inline void PrecisionGuard::release() noexcept {
  if (!active_) return;
  policy_slot() = saved_;
  active_ = false;
}

}  // namespace vibe::gpu

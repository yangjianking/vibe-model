#pragma once
/// @file error.hpp
/// @brief 异常层次与断言宏。
///
/// 约定：模式内部**不抛出**用于控制流的异常；异常只用于不可恢复的
/// 配置错误、维度不一致与 IO 失败。

#include <sstream>
#include <stdexcept>
#include <string>

#include "vibe/common/types.hpp"

namespace vibe {

/// 基类：所有 VIBE 异常
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

/// 配置解析/校验失败
class ConfigError : public Error {
 public:
  explicit ConfigError(const std::string& what) : Error("[config] " + what) {}
};

/// 维度/索引不一致
class DimensionError : public Error {
 public:
  explicit DimensionError(const std::string& what) : Error("[dimension] " + what) {}
};

/// 数值求解失败（不收敛、出现 NaN）
class NumericalError : public Error {
 public:
  explicit NumericalError(const std::string& what) : Error("[numerical] " + what) {}
};

/// IO 失败
class IoError : public Error {
 public:
  explicit IoError(const std::string& what) : Error("[io] " + what) {}
};

/// 未实现的可选后端
class NotImplemented : public Error {
 public:
  explicit NotImplemented(const std::string& what) : Error("[not-implemented] " + what) {}
};

namespace detail {
inline std::string make_message(const char* file, int line, const char* expr) {
  std::ostringstream os;
  os << file << ':' << line << " 断言失败: " << expr;
  return os.str();
}
}  // namespace detail

}  // namespace vibe

/// 永远检查的断言（用于配置与用户输入）
#define VIBE_CHECK(expr)                                                     \
  do {                                                                       \
    if (VIBE_UNLIKELY(!(expr))) {                                            \
      throw ::vibe::Error(::vibe::detail::make_message(__FILE__, __LINE__, #expr)); \
    }                                                                        \
  } while (false)

/// 仅在 Debug 构建中检查（用于热点循环内的前置条件）
#ifdef NDEBUG
#  define VIBE_ASSERT(expr) VIBE_UNUSED(expr)
#else
#  define VIBE_ASSERT(expr) VIBE_CHECK(expr)
#endif

#define VIBE_CHECK_MSG(expr, msg)                                            \
  do {                                                                       \
    if (VIBE_UNLIKELY(!(expr))) {                                            \
      throw ::vibe::Error(::vibe::detail::make_message(__FILE__, __LINE__, #expr) + " — " + (msg)); \
    }                                                                        \
  } while (false)

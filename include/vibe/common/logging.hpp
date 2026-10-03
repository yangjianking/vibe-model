#pragma once
/// @file logging.hpp
/// @brief 极简日志设施（MPI 感知、线程安全、可重定向）。
///
/// 设计：热点循环内**不得**调用日志；日志只出现在初始化、每 N 步的状态输出，
/// 以及错误路径。这样可保证 CPU/GPU 后端行为一致。

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>

#include "vibe/common/types.hpp"

namespace vibe::common {

enum class LogLevel : int { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Off = 5 };

/// 全局日志器。单例，进程内唯一；MPI 下只有 rank 0 输出（可强制全输出）。
class Logger {
 public:
  static Logger& instance();

  void set_level(LogLevel l) noexcept { level_ = l; }
  LogLevel level() const noexcept { return level_; }
  void set_rank(int r) noexcept { rank_ = r; }
  void set_all_ranks(bool v) noexcept { all_ranks_ = v; }
  void set_file(std::FILE* f) noexcept { out_ = f; }

  void write(LogLevel l, const std::string& msg);

 private:
  Logger() = default;
  LogLevel level_ = LogLevel::Info;
  int rank_ = 0;
  bool all_ranks_ = false;
  std::FILE* out_ = nullptr;
  std::mutex mutex_;
};

namespace detail {
inline std::string concat(std::ostringstream& os) { return os.str(); }

template <class T, class... Rest>
std::string concat(std::ostringstream& os, T&& first, Rest&&... rest) {
  os << std::forward<T>(first);
  return concat(os, std::forward<Rest>(rest)...);
}

template <class... Args>
std::string format(Args&&... args) {
  std::ostringstream os;
  return concat(os, std::forward<Args>(args)...);
}
}  // namespace detail

inline void log(LogLevel l, const std::string& msg) { Logger::instance().write(l, msg); }

}  // namespace vibe::common

#define VIBE_LOG(level, ...)                                                  \
  do {                                                                        \
    if (static_cast<int>(level) >=                                           \
        static_cast<int>(::vibe::common::Logger::instance().level())) {       \
      ::vibe::common::Logger::instance().write(                               \
          level, ::vibe::common::detail::format(__VA_ARGS__));                \
    }                                                                         \
  } while (false)

#define VIBE_TRACE(...) VIBE_LOG(::vibe::common::LogLevel::Trace, __VA_ARGS__)
#define VIBE_DEBUG(...) VIBE_LOG(::vibe::common::LogLevel::Debug, __VA_ARGS__)
#define VIBE_INFO(...)  VIBE_LOG(::vibe::common::LogLevel::Info,  __VA_ARGS__)
#define VIBE_WARN(...)  VIBE_LOG(::vibe::common::LogLevel::Warn,  __VA_ARGS__)
#define VIBE_ERROR(...) VIBE_LOG(::vibe::common::LogLevel::Error, __VA_ARGS__)

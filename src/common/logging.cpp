/// @file logging.cpp
/// @brief 日志器实现。

#include "vibe/common/logging.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>

namespace vibe::common {

Logger& Logger::instance() {
  static Logger logger;
  return logger;
}

namespace {
const char* level_tag(LogLevel l) noexcept {
  switch (l) {
    case LogLevel::Trace: return "TRACE";
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info:  return "INFO ";
    case LogLevel::Warn:  return "WARN ";
    case LogLevel::Error: return "ERROR";
    case LogLevel::Off:   return "OFF  ";
  }
  return "?????";
}

/// 自进程启动以来的毫秒数（用于日志时间戳）
long long uptime_ms() {
  using namespace std::chrono;
  static const auto t0 = steady_clock::now();
  return duration_cast<milliseconds>(steady_clock::now() - t0).count();
}
}  // namespace

void Logger::write(LogLevel l, const std::string& msg) {
  if (static_cast<int>(l) < static_cast<int>(level_)) return;
  if (level_ == LogLevel::Off) return;
  if (rank_ != 0 && !all_ranks_) return;

  std::lock_guard<std::mutex> lock(mutex_);
  std::FILE* out = (out_ != nullptr) ? out_ : ((l >= LogLevel::Warn) ? stderr : stdout);

  std::ostringstream head;
  head << "[" << level_tag(l) << "]["
       << std::setw(5) << std::setfill('0') << uptime_ms() << "ms]";
  if (rank_ != 0 || all_ranks_) head << "[r" << rank_ << "]";
  head << " ";

  std::fprintf(out, "%s%s\n", head.str().c_str(), msg.c_str());
  std::fflush(out);
}

}  // namespace vibe::common

/// @file timer.cpp
/// @brief 计时器的非常规实现（寄存器化 report），以及 %CPU 与内存占用的诊断。

#include "vibe/common/timer.hpp"

#include <cstdio>
#include <sstream>

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#elif defined(__linux__) || defined(__APPLE__)
#  include <sys/resource.h>
#  include <unistd.h>
#endif

namespace vibe::common {

namespace {

/// 当前进程的常驻内存（字节）；不可用时返回 0
std::size_t resident_memory_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return static_cast<std::size_t>(pmc.WorkingSetSize);
  }
  return 0;
#elif defined(__linux__)
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (f == nullptr) return 0;
  long total = 0, resident = 0;
  if (std::fscanf(f, "%ld %ld", &total, &resident) != 2) {
    std::fclose(f);
    return 0;
  }
  std::fclose(f);
  return static_cast<std::size_t>(resident) * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}

}  // namespace

/// 供 driver 与工具调用的运行时诊断字符串
std::string runtime_memory_report() {
  std::ostringstream os;
  const std::size_t rss = resident_memory_bytes();
  os << "常驻内存 = " << (static_cast<double>(rss) / (1024.0 * 1024.0)) << " MiB";
  return os.str();
}

}  // namespace vibe::common

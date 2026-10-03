#pragma once
/// @file timer.hpp
/// @brief 墙钟计时器与作用域计时。
///
/// 用途：内核调优、负载均衡诊断（[G17] Amdahl 定律）、以及每个时间步的性能日志。
/// 计时使用 steady_clock，避免系统时间调整造成的跳变。

#include <chrono>
#include <string>
#include <vector>

#include "vibe/common/logging.hpp"
#include "vibe/common/types.hpp"

namespace vibe::common {

using SteadyClock = std::chrono::steady_clock;

/// 简单累加计时器（可多次 start/stop）
class Timer {
 public:
  Timer() = default;
  explicit Timer(std::string name) : name_(std::move(name)) {}

  void start() noexcept { t0_ = SteadyClock::now(); running_ = true; }

  /// 停止并累加，返回本次时长（秒）
  Real stop() noexcept {
    if (!running_) return Real(0);
    const auto t1 = SteadyClock::now();
    const Real dt = std::chrono::duration<Real>(t1 - t0_).count();
    total_ += dt;
    calls_ += 1;
    running_ = false;
    return dt;
  }

  Real total() const noexcept { return total_; }
  Index calls() const noexcept { return calls_; }
  Real mean() const noexcept { return calls_ > 0 ? total_ / static_cast<Real>(calls_) : Real(0); }
  void reset() noexcept { total_ = Real(0); calls_ = 0; }
  const std::string& name() const noexcept { return name_; }

 private:
  std::string name_;
  SteadyClock::time_point t0_{};
  Real total_ = Real(0);
  Index calls_ = 0;
  bool running_ = false;
};

/// 作用域计时器：析构时打印
class ScopedTimer {
 public:
  explicit ScopedTimer(std::string name)
      : name_(std::move(name)) {
    timer_.start();
  }
  ~ScopedTimer() {
    const Real dt = timer_.stop();
    VIBE_DEBUG(name_, " 耗时 ", dt, " s");
  }
  Real elapsed() noexcept { return timer_.stop(); }

 private:
  std::string name_;
  Timer timer_;
};

/// 一组具名计时器的集合，用于按时间步输出性能分解
class TimerRegistry {
 public:
  Timer& get(const std::string& name) {
    for (auto& t : timers_) {
      if (t.name() == name) return t;
    }
    timers_.emplace_back(name);
    return timers_.back();
  }
  void report() const {
    for (const auto& t : timers_) {
      VIBE_INFO("[timer] ", t.name(), " total=", t.total(), "s calls=", t.calls(),
                " mean=", t.mean(), "s");
    }
  }
  void reset() {
    for (auto& t : timers_) t.reset();
  }

 private:
  std::vector<Timer> timers_;
};

/// 运行时内存占用报告（不可用时返回说明文本），供 driver 与诊断使用
std::string runtime_memory_report();

}  // namespace vibe::common

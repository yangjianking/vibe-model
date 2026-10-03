#pragma once
/// @file test.hpp
/// @brief 头文件形式的极简测试框架，避免外部依赖。
///
/// 用法：
/// @code
///   #include "vibe/common/test.hpp"
///   VIBE_TEST(helmholtz_1d_constant_coefficients) {
///     VIBE_CHECK_NEAR(solve(...), exact, 1e-12);
///   }
/// @endcode
///
/// 与主流框架的取舍：不支持 fixture/参数化，但支持浮点容差比较与
/// "预期抛出" 检查，足以覆盖模式中的数值单元测试。

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"

namespace vibe::test {

struct TestCase {
  std::string name;
  std::function<void()> fn;
};

/// 全局注册表（C++17 起 inline 变量保证跨 TU 唯一）
inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

inline int& failure_count() {
  static int n = 0;
  return n;
}

inline int& check_count() {
  static int n = 0;
  return n;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) {
    registry().push_back({name, std::move(fn)});
  }
};

inline void report_failure(const char* file, int line, const std::string& msg) {
  ++failure_count();
  std::fprintf(stderr, "  [FAIL] %s:%d  %s\n", file, line, msg.c_str());
}

/// 运行全部注册用例；name_filter 非空时只运行名字包含该子串的用例
inline int run_all(const std::string& name_filter = "") {
  int run = 0, failed_before = 0;
  std::printf("运行 %zu 个测试用例\n", registry().size());
  for (auto& tc : registry()) {
    if (!name_filter.empty() && tc.name.find(name_filter) == std::string::npos) continue;
    const int before = failure_count();
    ++run;
    try {
      tc.fn();
    } catch (const std::exception& e) {
      report_failure("<exception>", 0, std::string("未捕获异常: ") + e.what());
    }
    const int now = failure_count();
    std::printf("  [%s] %s\n", now == before ? "PASS" : "FAIL", tc.name.c_str());
    VIBE_UNUSED(failed_before);
  }
  std::printf("共 %d 个用例，%d 个检查，%d 个失败\n", run, check_count(), failure_count());
  return failure_count() == 0 ? 0 : 1;
}

}  // namespace vibe::test

#define VIBE_TEST(name)                                                       \
  static void vibe_test_##name();                                             \
  static ::vibe::test::Registrar vibe_test_reg_##name(#name, &vibe_test_##name); \
  static void vibe_test_##name()

#define VIBE_CHECK(cond)                                                      \
  do {                                                                        \
    ++::vibe::test::check_count();                                            \
    if (!(cond)) ::vibe::test::report_failure(__FILE__, __LINE__, #cond);     \
  } while (false)

#define VIBE_CHECK_NEAR(a, b, tol)                                            \
  do {                                                                        \
    ++::vibe::test::check_count();                                            \
    const double va_ = static_cast<double>(a);                                \
    const double vb_ = static_cast<double>(b);                                \
    const double vt_ = static_cast<double>(tol);                              \
    if (!(std::abs(va_ - vb_) <= vt_)) {                                      \
      char buf_[256];                                                         \
      std::snprintf(buf_, sizeof(buf_), "%s (=%.17g) vs %s (=%.17g) 容差 %.3g", \
                    #a, va_, #b, vb_, vt_);                                    \
      ::vibe::test::report_failure(__FILE__, __LINE__, buf_);                 \
    }                                                                         \
  } while (false)

#define VIBE_CHECK_REL(a, b, tol)                                             \
  do {                                                                        \
    ++::vibe::test::check_count();                                            \
    if (::vibe::rel_error(static_cast<double>(a), static_cast<double>(b)) >   \
        static_cast<double>(tol)) {                                           \
      ::vibe::test::report_failure(__FILE__, __LINE__,                        \
                                   std::string(#a) + " 相对误差超过 " + #tol); \
    }                                                                         \
  } while (false)

#define VIBE_CHECK_THROWS(expr)                                               \
  do {                                                                        \
    ++::vibe::test::check_count();                                            \
    bool thrown_ = false;                                                     \
    try {                                                                     \
      (void)(expr);                                                           \
    } catch (...) {                                                           \
      thrown_ = true;                                                         \
    }                                                                         \
    if (!thrown_)                                                             \
      ::vibe::test::report_failure(__FILE__, __LINE__, #expr " 未抛出异常");   \
  } while (false)

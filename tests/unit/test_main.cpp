/// @file test_main.cpp
/// @brief 所有单元测试可执行文件的统一入口。
///
/// 用法：  ./test_grid              运行全部用例
///         ./test_grid helmholtz   只运行名字包含 "helmholtz" 的用例

#include <iostream>
#include <string>

#include "vibe/common/mpi_wrapper.hpp"
#include "vibe/common/test.hpp"

int main(int argc, char** argv) {
  vibe::common::Comm::initialize(&argc, &argv);
  const std::string filter = (argc > 1) ? std::string(argv[1]) : std::string();
  const int rc = vibe::test::run_all(filter);
  vibe::common::Comm::finalize();
  return rc;
}

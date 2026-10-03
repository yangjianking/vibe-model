# =============================================================================
#  tests/CMakeLists.txt —— 单元测试、集成测试与 CTest 注册
# =============================================================================

add_library(vibe_test_main STATIC unit/test_main.cpp)
target_link_libraries(vibe_test_main PUBLIC vibe::common vibe::compile_options)

# 每个测试可执行文件对应一个单元测试源文件；用 VIBE_TEST 宏注册用例。
function(vibe_add_unit_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vibe_test_main vibe::model vibe::compile_options)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "unit" TIMEOUT 300)
endfunction()

vibe_add_unit_test(test_common      unit/test_common.cpp)
vibe_add_unit_test(test_config      unit/test_config.cpp)
vibe_add_unit_test(test_binary_io   unit/test_binary_io.cpp)
vibe_add_unit_test(test_grid        unit/test_grid.cpp)
vibe_add_unit_test(test_dynamics    unit/test_dynamics.cpp)
vibe_add_unit_test(test_time        unit/test_time.cpp)
vibe_add_unit_test(test_gpu_precision unit/test_gpu_precision.cpp)
vibe_add_unit_test(test_physics     unit/test_physics.cpp)
vibe_add_unit_test(test_verify      unit/test_verify.cpp)
vibe_add_unit_test(test_obs_operator unit/test_obs_operator.cpp)
vibe_add_unit_test(test_minimizer   unit/test_minimizer.cpp)
vibe_add_unit_test(test_tl_ad       unit/test_tl_ad.cpp)

# 集成测试：完整的理想试验短积分（串行、小网格）
add_executable(test_integration integration/test_idealized_cases.cpp)
target_link_libraries(test_integration PRIVATE vibe::model vibe::driver vibe::compile_options)
add_test(NAME test_integration COMMAND test_integration)
set_tests_properties(test_integration PROPERTIES LABELS "integration" TIMEOUT 600)

# 理想试验的短时运行（作为 CTest 的 smoke test）
foreach(case warm_bubble density_current mountain_wave resting_isothermal)
  add_test(NAME smoke_${case}
           COMMAND vibe_model_exe
                   --config ${CMAKE_SOURCE_DIR}/config/model.yaml
                   --ic ${case}
                   --log-level warn)
  set_tests_properties(smoke_${case} PROPERTIES
    LABELS "smoke"
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endforeach()

# Python 测试（若存在 pytest）
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME python_vibe_post
           COMMAND ${Python3_EXECUTABLE} -m pytest -q
                   ${CMAKE_SOURCE_DIR}/python/vibe_post/tests)
  set_tests_properties(python_vibe_post PROPERTIES
    LABELS "python" TIMEOUT 300
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python/vibe_post/src")
endif()

# =============================================================================
#  VIBE-Model —— 可嵌套变分辨率、半隐式、GPU 混合精度数值天气预报模式
#
#  构建选项见 docs/design/00_architecture.md 第 11 节。
#  典型用法：
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVIBE_ENABLE_MPI=ON
#     cmake --build build -j
#     ctest --test-dir build --output-on-failure
# =============================================================================
cmake_minimum_required(VERSION 3.20)

project(VIBE-Model
        VERSION 0.9.0
        DESCRIPTION "Versatile Implicit Balanced-Equation model: nested, variable-resolution NWP core with 4D-Var"
        LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# 选项
# ---------------------------------------------------------------------------
option(VIBE_ENABLE_MPI        "启用 MPI 域分解"            ON)
option(VIBE_ENABLE_OPENMP     "启用 OpenMP 共享内存并行"    ON)
option(VIBE_ENABLE_CUDA       "启用 CUDA 后端"             OFF)
option(VIBE_ENABLE_HIP        "启用 HIP 后端"              OFF)
option(VIBE_ENABLE_SYCL       "启用 SYCL 后端"             OFF)
option(VIBE_ENABLE_NETCDF     "启用 NetCDF 输出"           ON)
option(VIBE_ENABLE_4DVAR      "编译变分同化模块"            ON)
option(VIBE_ENABLE_PHYSICS    "编译物理参数化模块"          ON)
option(VIBE_BUILD_TESTS       "构建测试"                   ON)
option(VIBE_BUILD_PYTHON      "安装 Python 后处理包"        ON)
option(VIBE_WARNINGS_AS_ERRORS "把编译警告视为错误"         OFF)
option(VIBE_ENABLE_SANITIZERS "启用 Address/UB sanitizer"  OFF)

set(VIBE_PRECISION "double" CACHE STRING "数值精度：double | single | mixed")
set_property(CACHE VIBE_PRECISION PROPERTY STRINGS double single mixed)
set(VIBE_GPU_ARCH "" CACHE STRING "GPU 目标架构（如 80 表示 sm_80）；空 = 自动探测")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

include(CompilerOptions)
include(GPUOptions)
include(CTest)

# ---------------------------------------------------------------------------
# 第三方依赖
# ---------------------------------------------------------------------------
if(VIBE_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX QUIET)
  if(MPI_CXX_FOUND)
    message(STATUS "VIBE: MPI 已启用 (${MPI_CXX_LIBRARY_VERSION_STRING})")
  else()
    message(WARNING "VIBE: 未找到 MPI，VIBE_ENABLE_MPI 自动关闭")
    set(VIBE_ENABLE_MPI OFF)
  endif()
endif()

if(VIBE_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(NOT OpenMP_CXX_FOUND)
    message(WARNING "VIBE: 未找到 OpenMP，VIBE_ENABLE_OPENMP 自动关闭")
    set(VIBE_ENABLE_OPENMP OFF)
  endif()
endif()

if(VIBE_ENABLE_NETCDF)
  find_package(NetCDF COMPONENTS C QUIET)
  if(NOT NetCDF_FOUND)
    message(WARNING "VIBE: 未找到 NetCDF，VIBE_ENABLE_NETCDF 自动关闭；将使用 .vibebin 后备格式")
    set(VIBE_ENABLE_NETCDF OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# 编译期宏
# ---------------------------------------------------------------------------
add_library(vibe_build_config INTERFACE)
target_compile_definitions(vibe_build_config INTERFACE
  VIBE_VERSION="${PROJECT_VERSION}")

if(VIBE_PRECISION STREQUAL "single")
  target_compile_definitions(vibe_build_config INTERFACE VIBE_PRECISION=1)
elseif(VIBE_PRECISION STREQUAL "mixed")
  target_compile_definitions(vibe_build_config INTERFACE VIBE_PRECISION=2)
else()
  target_compile_definitions(vibe_build_config INTERFACE VIBE_PRECISION=0)
endif()

if(VIBE_ENABLE_MPI)
  target_compile_definitions(vibe_build_config INTERFACE VIBE_HAVE_MPI=1)
  target_link_libraries(vibe_build_config INTERFACE MPI::MPI_CXX)
endif()
if(VIBE_ENABLE_OPENMP)
  target_compile_definitions(vibe_build_config INTERFACE VIBE_HAVE_OPENMP=1)
  target_link_libraries(vibe_build_config INTERFACE OpenMP::OpenMP_CXX)
endif()
if(VIBE_ENABLE_NETCDF)
  target_compile_definitions(vibe_build_config INTERFACE VIBE_HAVE_NETCDF=1)
  target_link_libraries(vibe_build_config INTERFACE NetCDF::NetCDF_C)
endif()
if(NOT VIBE_ENABLE_PHYSICS)
  target_compile_definitions(vibe_build_config INTERFACE VIBE_NO_PHYSICS=1)
endif()
if(NOT VIBE_ENABLE_4DVAR)
  target_compile_definitions(vibe_build_config INTERFACE VIBE_NO_4DVAR=1)
endif()

# ---------------------------------------------------------------------------
# 子目录
# ---------------------------------------------------------------------------
add_subdirectory(src)

add_subdirectory(tools)
add_subdirectory(examples)

if(VIBE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(VIBE_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# ---------------------------------------------------------------------------
# 安装与导出
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY config/  DESTINATION ${CMAKE_INSTALL_DATADIR}/vibe-model/config)
install(DIRECTORY docs/    DESTINATION ${CMAKE_INSTALL_DATADIR}/vibe-model/docs)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/VibeModelConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeModelConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/VibeModelConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VibeModel)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/VibeModelConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/VibeModelConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VibeModel)

message(STATUS "")
message(STATUS "VIBE-Model ${PROJECT_VERSION} 配置摘要")
message(STATUS "  构建类型   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  精度       : ${VIBE_PRECISION}")
message(STATUS "  MPI        : ${VIBE_ENABLE_MPI}")
message(STATUS "  OpenMP     : ${VIBE_ENABLE_OPENMP}")
message(STATUS "  CUDA       : ${VIBE_ENABLE_CUDA}")
message(STATUS "  HIP        : ${VIBE_ENABLE_HIP}")
message(STATUS "  SYCL       : ${VIBE_ENABLE_SYCL}")
message(STATUS "  NetCDF     : ${VIBE_ENABLE_NETCDF}")
message(STATUS "  4D-Var     : ${VIBE_ENABLE_4DVAR}")
message(STATUS "  物理       : ${VIBE_ENABLE_PHYSICS}")
message(STATUS "  测试       : ${VIBE_BUILD_TESTS}")
message(STATUS "")

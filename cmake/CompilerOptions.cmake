# =============================================================================
#  cmake/CompilerOptions.cmake —— 统一的编译选项与告警集
# =============================================================================

# 一个 INTERFACE 目标承载所有公共编译选项
add_library(vibe_compile_options INTERFACE)
add_library(vibe::compile_options ALIAS vibe_compile_options)

set(_vibe_cxx_warnings
  -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
  -Wcast-align -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion
  -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough -Wnull-dereference)

set(_vibe_c_warnings
  -Wall -Wextra -Wpedantic -Wshadow -Wcast-align -Wunused -Wformat=2)

if(MSVC)
  set(_vibe_msvc_warnings /W4 /permissive- /Zc:__cplusplus /utf-8)
  target_compile_options(vibe_compile_options INTERFACE ${_vibe_msvc_warnings})
  target_compile_options(vibe_compile_options INTERFACE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Debug>:/Od /Zi>)
  if(VIBE_ENABLE_OPENMP)
    target_compile_options(vibe_compile_options INTERFACE /openmp)
  endif()
else()
  target_compile_options(vibe_compile_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${_vibe_cxx_warnings}>
    $<$<COMPILE_LANGUAGE:C>:${_vibe_c_warnings}>)

  # 架构相关的优化：优先使用本机架构
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" VIBE_HAVE_MARCH_NATIVE)
  if(VIBE_HAVE_MARCH_NATIVE AND NOT VIBE_ENABLE_CUDA AND NOT VIBE_ENABLE_HIP)
    target_compile_options(vibe_compile_options INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
      $<$<COMPILE_LANGUAGE:C>:-march=native>)
  endif()

  target_compile_options(vibe_compile_options INTERFACE
    $<$<CONFIG:Release>:-O3 -funroll-loops -fno-math-errno>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>)

  if(VIBE_ENABLE_SANITIZERS)
    target_compile_options(vibe_compile_options INTERFACE
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_options(vibe_compile_options INTERFACE -fsanitize=address,undefined)
  endif()

  # 对含大量公式的实现文件关闭 -Wconversion 的噪声（集中管理，避免用 pragma 污染源码）
  target_compile_options(vibe_compile_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-float-conversion>)
endif()

if(VIBE_WARNINGS_AS_ERRORS)
  if(MSVC)
    target_compile_options(vibe_compile_options INTERFACE /WX)
  else()
    target_compile_options(vibe_compile_options INTERFACE -Werror)
  endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT MSVC)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _vibe_ipo_ok OUTPUT _vibe_ipo_msg)
  if(_vibe_ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(STATUS "VIBE: 本编译器不支持 IPO/LTO：${_vibe_ipo_msg}")
  endif()
endif()

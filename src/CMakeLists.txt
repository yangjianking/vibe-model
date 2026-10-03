# =============================================================================
#  src/CMakeLists.txt —— 各模块库与可执行文件
#
#  依赖方向严格遵守 docs/design/00_architecture.md 第 3 节：
#      common <- config <- grid <- dyn <- timeint <- driver
#      common <- gpu    <- (grid/dyn/timeint/physics/da 的加速后端)
#      common <- io     <- driver
#      common <- da -> obs, dyn, timeint, grid
#      common <- verify -> (只接受内存数组)
# =============================================================================

# ---------------------------------------------------------------------------
# 公共工具函数：统一设置目标属性
# ---------------------------------------------------------------------------
function(vibe_add_library target)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
  add_library(${target} STATIC ${ARG_SOURCES})
  add_library(vibe::${target} ALIAS ${target})
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(${target} PUBLIC vibe::compile_options vibe_build_config
                                          vibe_gpu_backend)
  target_link_libraries(${target} PUBLIC ${ARG_DEPS})
  set_target_properties(${target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
endfunction()

# ---------------------------------------------------------------------------
# 1. common —— 基础类型、常量、日志、MPI 封装、计时器
# ---------------------------------------------------------------------------
vibe_add_library(common
  SOURCES
    common/logging.cpp
    common/mpi_wrapper.cpp
    common/timer.cpp
    # 纯 C11 热点内核（Thomas 三对角、WENO5 重构、双线性插值、CFL 扫描、
    # Neumaier 补偿求和、通量平流内循环）。接口见 include/vibe/common/c_kernels.h。
    common/c_kernels.c)

# ---------------------------------------------------------------------------
# 2. config —— YAML 子集解析与配置校验
# ---------------------------------------------------------------------------
vibe_add_library(config
  SOURCES
    config/mini_yaml.cpp
    config/config.cpp
  DEPS vibe::common)

# ---------------------------------------------------------------------------
# 3. grid —— 几何、度量、场、halo、插值、嵌套、变分辨率
# ---------------------------------------------------------------------------
vibe_add_library(grid
  SOURCES
    grid/geometry.cpp
    grid/halo.cpp
    grid/interpolation.cpp
    grid/nest.cpp
    grid/variable_resolution.cpp
  DEPS vibe::common vibe::config)

# ---------------------------------------------------------------------------
# 4. dyn —— 控制方程、状态、平流、科氏、阻尼、诊断
# ---------------------------------------------------------------------------
vibe_add_library(dyn
  SOURCES
    dyn/reference_state.cpp
    dyn/state.cpp
    # Tendency / PhysicsTendency 的分配与向量运算（TL/AD 与时间推进都依赖）
    dyn/tendency.cpp
    dyn/advection.cpp
    dyn/coriolis.cpp
    dyn/damping.cpp
    dyn/equations.cpp
    dyn/diagnostics.cpp
  DEPS vibe::common vibe::config vibe::grid)

# ---------------------------------------------------------------------------
# 5. gpu —— 设备抽象、内存、混合精度、内核
# ---------------------------------------------------------------------------
set(_vibe_gpu_sources
    gpu/device.cpp
    gpu/memory.cpp
    gpu/cpu/kernels_cpu.cpp)

if(VIBE_ENABLE_CUDA)
  list(APPEND _vibe_gpu_sources
    gpu/cuda/kernels_advection.cu
    gpu/cuda/kernels_diffusion.cu
    gpu/cuda/kernels_tridiagonal.cu
    gpu/cuda/kernels_helmholtz.cu
    gpu/cuda/kernels_interp.cu
    gpu/cuda/kernels_reduction.cu)
endif()

vibe_add_library(gpu
  SOURCES ${_vibe_gpu_sources}
  DEPS vibe::common)

# ---------------------------------------------------------------------------
# 6. physics —— 微物理、辐射、PBL、陆面、积云
# ---------------------------------------------------------------------------
if(VIBE_ENABLE_PHYSICS)
  vibe_add_library(physics
    SOURCES
      physics/physics_types.cpp
      physics/physics_driver.cpp
      physics/microphysics_kessler.cpp
      physics/microphysics_thompson.cpp
      physics/radiation_rrtmg.cpp
      physics/pbl_myj.cpp
      physics/pbl_ysu.cpp
      physics/surface_layer.cpp
      physics/cumulus_kf.cpp
    DEPS vibe::common vibe::config vibe::grid vibe::dyn)
else()
  add_library(physics INTERFACE)
  add_library(vibe::physics ALIAS physics)
  target_link_libraries(physics INTERFACE vibe::dyn)
endif()

# ---------------------------------------------------------------------------
# 7. timeint —— 积分器、声波子步、Helmholtz、时间步控制
# ---------------------------------------------------------------------------
vibe_add_library(timeint
  SOURCES
    time/helmholtz.cpp
    time/acoustic_substep.cpp
    time/runge_kutta.cpp
    time/semi_implicit.cpp
    time/timestep_control.cpp
  DEPS vibe::common vibe::config vibe::grid vibe::dyn vibe::gpu vibe::physics)

# ---------------------------------------------------------------------------
# 8. io —— NetCDF 与 .vibebin 读写、重启
# ---------------------------------------------------------------------------
vibe_add_library(io
  SOURCES
    io/field_io.cpp
    io/binary_writer.cpp
    io/netcdf_writer.cpp
  DEPS vibe::common vibe::config vibe::grid vibe::dyn)

# ---------------------------------------------------------------------------
# 9. obs —— 观测数据模型、观测算子、快速辐射传输、偏差订正
# ---------------------------------------------------------------------------
set(_vibe_obs_sources
    obs/observations.cpp
    obs/obs_operator.cpp
    obs/obs_operators.cpp
    obs/radiative_transfer.cpp
    obs/bias_correction.cpp)

vibe_add_library(obs
  SOURCES ${_vibe_obs_sources}
  DEPS vibe::common vibe::config vibe::grid vibe::dyn)

# ---------------------------------------------------------------------------
# 10. da —— 控制变量、B 矩阵、代价函数、极小化、TL/AD、增量 4D-Var
# ---------------------------------------------------------------------------
if(VIBE_ENABLE_4DVAR)
  vibe_add_library(da
    SOURCES
      da/control_vector.cpp
      da/background_error.cpp
      da/cost_function.cpp
      da/minimizer.cpp
      da/tangent_linear.cpp
      da/incremental_4dvar.cpp
    DEPS vibe::common vibe::config vibe::grid vibe::dyn vibe::timeint vibe::obs)
else()
  add_library(da INTERFACE)
  add_library(vibe::da ALIAS da)
  target_link_libraries(da INTERFACE vibe::obs)
endif()

# ---------------------------------------------------------------------------
# 11. verify —— 检验评分
# ---------------------------------------------------------------------------
vibe_add_library(verify
  SOURCES
    verify/scores.cpp
    verify/contingency.cpp
    verify/probabilistic.cpp
    verify/spatial.cpp
    verify/verification.cpp
  DEPS vibe::common vibe::config)

# ---------------------------------------------------------------------------
# 12. driver —— 顶层装配与 main
# ---------------------------------------------------------------------------
vibe_add_library(driver
  SOURCES
    driver/driver.cpp
    driver/initial_conditions.cpp
  DEPS vibe::common vibe::config vibe::grid vibe::dyn vibe::timeint
       vibe::physics vibe::io vibe::da vibe::obs vibe::verify vibe::gpu)

# 聚合接口目标：下游只需 target_link_libraries(... vibe::model)
add_library(vibe_model INTERFACE)
add_library(vibe::model ALIAS vibe_model)
target_link_libraries(vibe_model INTERFACE
  vibe::common vibe::config vibe::grid vibe::dyn vibe::timeint
  vibe::physics vibe::io vibe::da vibe::obs vibe::verify vibe::gpu)

# 主可执行文件
add_executable(vibe_model_exe driver/main.cpp)
set_target_properties(vibe_model_exe PROPERTIES OUTPUT_NAME vibe_model)
target_link_libraries(vibe_model_exe PRIVATE vibe::driver vibe::compile_options)
install(TARGETS vibe_model_exe RUNTIME DESTINATION bin)

# 检验命令行工具
add_executable(vibe_verify_cli verify/verify_cli.cpp)
target_link_libraries(vibe_verify_cli PRIVATE vibe::verify vibe::compile_options)
install(TARGETS vibe_verify_cli RUNTIME DESTINATION bin)

# 库与头文件安装
install(TARGETS common config grid dyn gpu physics timeint io obs da verify driver
        EXPORT VibeModelTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(EXPORT VibeModelTargets
        FILE VibeModelTargets.cmake
        NAMESPACE vibe::
        DESTINATION lib/cmake/VibeModel)

# =============================================================================
#  python/CMakeLists.txt —— Python 后处理包的安装
#
#  仅做安装与测试注册，不做编译。使用 pip 的本地安装：
#      python -m pip install -e python/vibe_post
# =============================================================================

find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_Interpreter_FOUND)
  # 把运行时依赖的 C++ 侧配置一并拷贝到 Python 包的数据目录，便于后处理复用
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/config/
          DESTINATION ${CMAKE_INSTALL_DATADIR}/vibe-model/config)

  add_custom_target(vibe_post_install
    COMMAND ${Python3_EXECUTABLE} -m pip install --quiet
            ${CMAKE_SOURCE_DIR}/python/vibe_post
    COMMENT "安装 Python 后处理包 vibe_post"
    VERBATIM)
else()
  message(STATUS "VIBE: 未找到 Python3，跳过 Python 包安装目标")
endif()

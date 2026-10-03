# =============================================================================
#  tools/CMakeLists.txt —— 独立命令行小工具
# =============================================================================

add_executable(vibe_adjoint_check adjoint_check.cpp)
target_link_libraries(vibe_adjoint_check PRIVATE
  vibe::model vibe::driver vibe::da vibe::compile_options)
install(TARGETS vibe_adjoint_check RUNTIME DESTINATION bin)

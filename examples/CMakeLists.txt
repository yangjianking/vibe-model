# =============================================================================
#  examples/CMakeLists.txt —— 嵌入示例（不参与安装，仅作为可运行文档）
# =============================================================================

add_executable(vibe_embed_example embed_model.cpp)
target_link_libraries(vibe_embed_example PRIVATE
  vibe::model vibe::driver vibe::compile_options)
set_target_properties(vibe_embed_example PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

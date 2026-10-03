#!/usr/bin/env bash
# =============================================================================
#  scripts/build.sh —— VIBE-Model 构建脚本
#
#  用法：
#     scripts/build.sh                     # 默认 gcc-release
#     scripts/build.sh mixed-precision     # 指定 CMake 预设
#     scripts/build.sh gcc-release --clean # 先清理构建目录
#     scripts/build.sh debug-asan --test   # 构建并运行测试
# =============================================================================
set -euo pipefail

PRESET="${1:-gcc-release}"
CLEAN=0
RUN_TESTS=0

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --test)  RUN_TESTS=1 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> 预设：$PRESET"

if [ "$CLEAN" -eq 1 ]; then
  echo "==> 清理 build 目录"
  rm -rf "build/$PRESET"
fi

echo "==> 配置"
cmake --preset "$PRESET"

echo "==> 构建"
cmake --build --preset "$PRESET"

if [ "$RUN_TESTS" -eq 1 ]; then
  echo "==> 测试"
  ctest --preset "$PRESET" --output-on-failure
fi

echo "==> 完成。可执行文件位于 build/$PRESET/src/"

#!/usr/bin/env bash
# =============================================================================
#  scripts/run_idealized.sh —— 批量运行理想试验矩阵
#
#  用法：
#     scripts/run_idealized.sh [build 目录] [输出目录]
#     scripts/run_idealized.sh build/gcc-release output/experiments
#
#  脚本会依次运行若干理想试验，并把运行摘要收集到一个汇总文件中，
#  便于对比不同积分器、平流格式与时间步。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/build/gcc-release}"
OUT_DIR="${2:-$ROOT/output/experiments}"
EXE="$BUILD_DIR/src/vibe_model"

if [ ! -x "$EXE" ]; then
  echo "找不到可执行文件：$EXE" >&2
  echo "请先运行 scripts/build.sh" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.txt"
: > "$SUMMARY"

# 每个条目：试验名|积分器|平流格式|时间步|声波子步|模拟时长
CASES=(
  "warm_bubble|rk3_acoustic|central2|6|6|3600"
  "warm_bubble|semi_implicit|central2|12|1|3600"
  "density_current|rk3_acoustic|central4|2|4|1200"
  "mountain_wave|rk3_acoustic|central4|6|8|3600"
  "inertia_gravity_wave|rk3_acoustic|central6|6|6|3600"
  "resting_isothermal|rk3_acoustic|central2|10|20|3600"
  "resting_isothermal|semi_implicit|central2|100|1|3600"
)

for entry in "${CASES[@]}"; do
  IFS='|' read -r ic integ adv dt nsub runlen <<< "$entry"
  tag="${ic}_${integ}_${adv}_dt${dt}"
  echo "==> $tag"

  # 用环境变量覆盖配置（通过临时 YAML 叠加实现，保持零侵入）
  tmp_cfg="$OUT_DIR/$tag.yaml"
  cat > "$tmp_cfg" <<YAML
name: $tag
description: "自动生成的试验配置：$ic / $integ / $adv / dt=$dt"
time:
  dt: $dt
  run_length: $runlen
  acoustic_substeps: $nsub
  integrator: $integ
numerics:
  advection: $adv
parallel:
  output_dir: "$OUT_DIR/$tag"
YAML

  echo "=== $tag ===" >> "$SUMMARY"
  "$EXE" \
    --config "$ROOT/config/model.yaml" \
    --config "$tmp_cfg" \
    --ic "$ic" \
    --log-level info 2>&1 | tee -a "$SUMMARY"
  echo "" >> "$SUMMARY"
done

echo "==> 全部完成。摘要：$SUMMARY"

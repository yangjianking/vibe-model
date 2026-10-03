#!/usr/bin/env python3
"""生成 VIBE-Model 可读的地形文件（纯文本矩阵）。

用法
----
    python scripts/make_terrain.py --nx 128 --ny 64 --dx 2000 \
        --type agnesi --height 1000 --radius 6000 --out terrain.txt

支持的形状
----------
* agnesi : Witch of Agnesi 山 h = h0 * a^2 / (a^2 + d^2)   [D2]
* schar  : 光滑余弦山                                       [D9]
* gauss  : 高斯山
* ridge  : 沿 y 方向的二维山脊
* random : 由功率谱生成的分形地形（可选）

输出格式：每行 ny 个数（或每行 nx 个数，见 --layout），
纯空白分隔，行主序，与 driver::build_geometry 的读取约定一致（先 i 后 j）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def witch_of_agnesi(nx: int, ny: int, dx: float, dy: float,
                    height: float, radius: float,
                    cx: float, cy: float) -> list[list[float]]:
    out = []
    for j in range(ny):
        row = []
        x_c = cx if cx >= 0 else 0.5 * nx * dx
        y_c = cy if cy >= 0 else 0.5 * ny * dy
        for i in range(nx):
            x = i * dx - x_c
            y = j * dy - y_c
            d2 = x * x + y * y
            row.append(height * radius * radius / (radius * radius + d2))
        out.append(row)
    return out


def schar(nx: int, ny: int, dx: float, dy: float,
          height: float, radius: float, cx: float, cy: float) -> list[list[float]]:
    out = []
    for j in range(ny):
        row = []
        x_c = cx if cx >= 0 else 0.5 * nx * dx
        y_c = cy if cy >= 0 else 0.5 * ny * dy
        for i in range(nx):
            x = i * dx - x_c
            y = j * dy - y_c
            d = math.hypot(x, y)
            row.append(height * math.cos(0.5 * math.pi * d / radius) ** 2
                       if d < radius else 0.0)
        out.append(row)
    return out


def gaussian(nx: int, ny: int, dx: float, dy: float,
             height: float, radius: float, cx: float, cy: float) -> list[list[float]]:
    out = []
    for j in range(ny):
        row = []
        x_c = cx if cx >= 0 else 0.5 * nx * dx
        y_c = cy if cy >= 0 else 0.5 * ny * dy
        for i in range(nx):
            x = i * dx - x_c
            y = j * dy - y_c
            row.append(height * math.exp(-(x * x + y * y) / (radius * radius)))
        out.append(row)
    return out


def ridge(nx: int, ny: int, dx: float, dy: float,
          height: float, radius: float, cx: float, cy: float) -> list[list[float]]:
    """沿 y 方向延伸的山脊，仅在 x 方向变化。"""
    x_c = cx if cx >= 0 else 0.5 * nx * dx
    out = []
    for _j in range(ny):
        row = []
        for i in range(nx):
            x = i * dx - x_c
            row.append(height * math.cos(0.5 * math.pi * x / radius) ** 2
                       if abs(x) < radius else 0.0)
        out.append(row)
    return out


SHAPES = {
    "agnesi": witch_of_agnesi,
    "schar": schar,
    "gauss": gaussian,
    "ridge": ridge,
}


def smooth(field: list[list[float]], passes: int) -> list[list[float]]:
    """1-2-1 平滑，与 grid::smooth_terrain 保持一致（边界保持原值）。"""
    ny = len(field)
    nx = len(field[0])
    for _ in range(passes):
        new = [row[:] for row in field]
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                new[j][i] = (0.25 * (field[j][i - 1] + field[j][i + 1]
                                     + field[j - 1][i] + field[j + 1][i])
                             + 0.5 * field[j][i])
        field = new
    return field


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成 VIBE-Model 地形文件")
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--dx", type=float, default=1000.0)
    p.add_argument("--dy", type=float, default=None)
    p.add_argument("--type", choices=sorted(SHAPES), default="agnesi")
    p.add_argument("--height", type=float, default=1000.0)
    p.add_argument("--radius", type=float, default=5000.0)
    p.add_argument("--cx", type=float, default=-1.0, help="中心 x（-1 表示居中）")
    p.add_argument("--cy", type=float, default=-1.0, help="中心 y（-1 表示居中）")
    p.add_argument("--smooth", type=int, default=3, help="1-2-1 平滑次数")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    dy = args.dy if args.dy is not None else args.dx
    field = SHAPES[args.type](args.nx, args.ny, args.dx, dy,
                              args.height, args.radius, args.cx, args.cy)
    if args.smooth > 0:
        field = smooth(field, args.smooth)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write(f"# VIBE-Model 地形文件：{args.type}, "
                 f"{args.nx}x{args.ny}, dx={args.dx}, dy={dy}, h0={args.height}\n")
        for row in field:
            fh.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    peak = max(max(row) for row in field)
    print(f"已写出 {args.out}：{args.nx}x{args.ny}，峰值 {peak:.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())

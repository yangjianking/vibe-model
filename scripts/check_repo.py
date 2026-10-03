#!/usr/bin/env python3
"""VIBE-Model 仓库自检（不需要编译器）。

用法
----
    python scripts/check_repo.py [--root .] [--quiet]

检查项
------
1. **构建完整性**：CMakeLists.txt 中列出的每个源文件都真实存在；
2. **头文件解析**：源码中所有 #include "vibe/..." 都能在 include/ 下找到；
3. **文档链接**：所有 Markdown 的相对链接都能解析到实际文件；
4. **接口一致性**：include/vibe 下的公开头文件都被至少一个源文件或测试引用
   （孤立头文件会被报告为提示，不是错误）；
5. **统计**：文件数、代码行数、各模块规模。

退出码：0 全部通过；1 存在错误。
设计原则：只依赖 Python 标准库；不编译、不执行任何项目代码。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_RE = re.compile(r"^\s{4,}([A-Za-z0-9_/.\-]+\.(?:cpp|cu|c|hpp|h))\s*$", re.MULTILINE)
INCLUDE_RE = re.compile(r'#include\s+"(vibe/[^"]+)"')
MDLINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")

SKIP_DIRS = {"build", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
CODE_EXT = {".hpp", ".h", ".cpp", ".cu", ".c", ".py", ".sh", ".yaml", ".toml"}


def walk(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            yield p


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def check_cmake(root: Path, files: list[Path], rep: Report) -> int:
    n = 0
    for cf in [f for f in files if f.name == "CMakeLists.txt"]:
        text = cf.read_text(encoding="utf-8", errors="replace")
        for m in SOURCE_RE.finditer(text):
            rel = m.group(1)
            if "${" in rel:
                continue
            if not (cf.parent / rel).exists():
                rep.error(f"CMake 引用了不存在的文件：{cf.relative_to(root)} -> {rel}")
            else:
                n += 1
    return n


def check_includes(root: Path, files: list[Path], rep: Report) -> tuple[int, set[str]]:
    inc_root = root / "include"
    n = 0
    used: set[str] = set()
    for f in files:
        if f.suffix not in {".hpp", ".h", ".cpp", ".cu", ".c"}:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in INCLUDE_RE.finditer(text):
            hdr = m.group(1)
            used.add(hdr)
            n += 1
            if not (inc_root / hdr).exists():
                rep.error(f"未解析的头文件：{f.relative_to(root)} -> {hdr}")
    return n, used


def check_md_links(root: Path, files: list[Path], rep: Report) -> int:
    n = 0
    for f in [x for x in files if x.suffix == ".md"]:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in MDLINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            n += 1
            resolved = (f.parent / target).resolve()
            if not resolved.exists():
                rep.error(f"失效的文档链接：{f.relative_to(root)} -> {target}")
    return n


def check_orphan_headers(root: Path, used: set[str], rep: Report) -> None:
    inc_root = root / "include"
    if not inc_root.exists():
        return
    for h in sorted(inc_root.rglob("*.hpp")):
        rel = h.relative_to(inc_root).as_posix()
        if rel not in used:
            rep.note(f"公开头文件未被任何源文件/测试引用：include/{rel}")


def statistics(root: Path, files: list[Path]) -> str:
    by_dir: Counter[str] = Counter()
    lines_by_dir: defaultdict[str, int] = defaultdict(int)
    total_lines = 0
    for f in files:
        rel = f.relative_to(root)
        key = "/".join(rel.parts[:2]) if len(rel.parts) > 1 else rel.parts[0]
        by_dir[key] += 1
        if f.suffix in CODE_EXT:
            try:
                lines = sum(1 for _ in f.open("r", encoding="utf-8", errors="replace"))
            except OSError:
                lines = 0
            lines_by_dir[key] += lines
            total_lines += lines

    out = ["", "=== 规模统计 ===", f"文件总数: {len(files)}    代码/配置行数: {total_lines}", ""]
    out.append(f"{'目录':<28}{'文件数':>8}{'行数':>10}")
    for key, cnt in sorted(by_dir.items(), key=lambda kv: -lines_by_dir[kv[0]]):
        out.append(f"{key:<28}{cnt:>8}{lines_by_dir[key]:>10}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VIBE-Model 仓库自检")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root: Path = args.root.resolve()
    files = list(walk(root))
    rep = Report()

    n_cmake = check_cmake(root, files, rep)
    n_inc, used = check_includes(root, files, rep)
    n_md = check_md_links(root, files, rep)
    check_orphan_headers(root, used, rep)

    print(f"检查根目录：{root}")
    print(f"  CMake 源文件引用 : {n_cmake}")
    print(f"  头文件引用       : {n_inc}")
    print(f"  Markdown 链接    : {n_md}")

    if rep.notes and not args.quiet:
        print(f"\n提示（{len(rep.notes)} 条）：")
        for m in rep.notes[:40]:
            print("  -", m)

    if rep.errors:
        print(f"\n错误（{len(rep.errors)} 条）：")
        for m in rep.errors:
            print("  x", m)
        print(statistics(root, files))
        return 1

    print("\n全部检查通过。")
    print(statistics(root, files))
    return 0


if __name__ == "__main__":
    sys.exit(main())

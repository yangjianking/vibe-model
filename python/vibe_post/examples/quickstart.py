"""vibe_post 快速上手示例。

运行：
    python examples/quickstart.py [--out-dir .]

本脚本**不依赖任何外部数据**：先在内存里合成一份 VIBE 风格的探空/网格场，
写成 `.vibebin`，再走完整的"读 -> 诊断 -> 插值 -> 检验 -> 绘图"链路。
绘图部分在缺少 matplotlib 时自动跳过（体现惰性依赖与优雅降级）。

单位一律 SI：米、秒、开尔文、帕斯卡、kg/kg。
文献编号见 docs/design/references.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vibe_post as vp  # noqa: E402


def make_synthetic_run(nx: int = 48, ny: int = 36, nz: int = 32,
                       z_top: float = 18000.0) -> vp.Dataset:
    """合成一份 VIBE 输出（位温、Exner 函数、水汽、三维风）。

    垂直方向用等温参考态 + 逆温层的理想探空：
    :math:`\\theta(z) = \\theta_0 + \\Gamma z + \\Delta\\theta
    \\exp(-((z-z_i)/h)^2)`，保证 CAPE/位涡等诊断有非平凡结构。
    复杂度：O(nx ny nz)。
    """
    dx = dy = 5000.0
    zeta = np.linspace(0.0, z_top, nz)                       # 层中心高度 [m]
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    # 理想探空
    theta0 = 300.0
    gamma = 3.5e-3
    theta_1d = theta0 + gamma * zeta
    qv_1d = 0.014 * np.exp(-zeta / 2500.0)
    p_1d = 100000.0 * np.exp(-zeta / 8000.0)
    pi_1d = (p_1d / 100000.0) ** 0.28571

    # 三维：加一个暖泡扰动与风切变
    bump = 2.5 * np.exp(-(((xx - xx.mean()) / 40000.0) ** 2
                          + ((yy - yy.mean()) / 30000.0) ** 2))
    theta = theta_1d[:, None, None] * (1.0 + 0.002 * bump)[None, :, :]
    pi = np.broadcast_to(pi_1d[:, None, None], (nz, ny, nx)).copy()
    qv = np.broadcast_to(qv_1d[:, None, None], (nz, ny, nx)).copy()
    shear = zeta / max(z_top, 1.0)
    # 按 C++ `.vibebin` 约定：u/v 为面场（各加一列/一行），w 为层界面场
    u_cell = (5.0 + 12.0 * shear)[:, None, None] + 1.0 + 0.0 * theta
    v_cell = (2.0 + 3.0 * np.sin(2.0 * np.pi * xx / (nx * dx)))[None, :, :] + 0.0 * theta
    w_cell = 0.05 * np.sin(2.0 * np.pi * zeta / z_top)[:, None, None] + 0.0 * theta
    u = np.concatenate((u_cell, u_cell[..., -1:]), axis=-1)
    v = np.concatenate((v_cell, v_cell[:, -1:, :]), axis=-2)
    w = np.concatenate((w_cell, w_cell[-1:, :, :]), axis=0)

    fields = {"theta": theta, "pi": pi, "qv": qv, "u": u, "v": v, "w": w}
    return vp.Dataset(
        fields=fields,
        dims={name: ("zeta", "y", "x") for name in fields},
        coords={"zeta": zeta, "x": x, "y": y},
        attrs={"dx": dx, "dy": dy, "z_top": z_top, "nx": nx, "ny": ny, "nz": nz,
               "lat": 45.0, "source": "synthetic"},
        time=0.0,
    )


def main(argv: list[str] | None = None) -> int:
    """运行完整示例。复杂度 O(N)（绘图为 O(N log N)）。"""
    parser = argparse.ArgumentParser(description="vibe_post 快速上手示例")
    parser.add_argument("--out-dir", default=".", help="输出目录（默认当前目录）")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 合成并写出 .vibebin（纯 numpy 后端，无需任何可选依赖）
    dataset = make_synthetic_run()
    path = out_dir / "quickstart.vibebin"
    vp.write_vibebin(path, dataset)
    print("[1] 已写出 {0}（{1} 字节）".format(path, path.stat().st_size))

    # 2) 读回并打印摘要
    loaded = vp.open_dataset(path)
    print("[2] 读回：{0} 个变量，网格 {1}".format(len(loaded.fields), loaded.grid_shape()))
    print(vp.summarize(loaded))

    # 3) 网格与高度场
    grid = vp.GridSpec.from_dataset(loaded)
    height = grid.height_array()
    pressure = vp.pressure_from_exner(loaded.fields["pi"])
    temperature = vp.temperature_from_exner(loaded.fields["pi"], loaded.fields["theta"])
    print("[3] 网格：{0}".format(grid.describe()))
    print("    地面气压 {0:.1f} hPa，最低层温度 {1:.2f} K".format(
        pressure[0].mean() / 100.0, temperature[0].mean()))

    # 3.5) 错位场 -> 体心视图（u/v/w 是面场，诊断量要求同形）
    cell = {
        "theta": np.asarray(loaded.fields["theta"]),
        "pi": np.asarray(loaded.fields["pi"]),
        "qv": np.asarray(loaded.fields["qv"]),
        "u": np.asarray(loaded.fields["u"])[..., :grid.nx],
        "v": np.asarray(loaded.fields["v"])[:, :grid.ny, :],
        "w": 0.5 * (np.asarray(loaded.fields["w"])[:-1] + np.asarray(loaded.fields["w"])[1:]),
    }
    print("[3.5] 体心视图形状：u={0}, theta={1}".format(cell["u"].shape, cell["theta"].shape))

    # 4) 诊断量
    dewpoint = vp.dewpoint(pressure, loaded.fields["qv"])
    rh = vp.relative_humidity(pressure, temperature, loaded.fields["qv"])
    zeta_rel = vp.vorticity(cell["u"], cell["v"], grid.dx, grid.dy)
    pv = vp.potential_vorticity(cell["theta"], cell["u"], cell["v"],
                                p=pressure, qv=cell["qv"], z=height,
                                lat=45.0, dx=grid.dx, dy=grid.dy)
    cape = vp.cape_cin(height, pressure, temperature, cell["qv"])
    print("[4] 诊断：RH(近地面)={0:.3f}，Td(近地面)={1:.2f} K".format(
        float(rh[0].mean()), float(dewpoint[0].mean())))
    print("    相对涡度范围 [{0:.3e}, {1:.3e}] 1/s".format(
        float(zeta_rel.min()), float(zeta_rel.max())))
    print("    Ertel 位涡范围 [{0:.3e}, {1:.3e}] K m^2 kg^-1 s^-1".format(
        float(np.nanmin(pv)), float(np.nanmax(pv))))
    print("    CAPE 最大 {0:.1f} J/kg，CIN 最小 {1:.1f} J/kg".format(
        float(np.nanmax(cape["cape"])), float(np.nanmin(cape["cin"]))))

    # 5) 垂直插值到 5000 m 与 500 hPa
    theta_5km = vp.column_interp_to_height(height, loaded.fields["theta"], 5000.0)
    theta_500 = vp.column_interp_to_pressure(pressure, loaded.fields["theta"], 50000.0)
    print("[5] theta@5km 均值 {0:.2f} K；theta@500hPa 均值 {1:.2f} K".format(
        float(np.nanmean(theta_5km)), float(np.nanmean(theta_500))))

    # 6) 检验：把 5 km 高度作为"真值"，把模式 theta 插值场作为"预报"
    truth = np.broadcast_to(theta_5km.mean(), theta_5km.shape)
    metrics = {
        "bias": vp.bias(theta_5km, truth),
        "rmse": vp.rmse(theta_5km, truth),
        "ets": vp.ets(theta_5km, truth, float(np.nanpercentile(truth, 90.0))),
        "fss": vp.fractions_skill_score(theta_5km, truth, float(np.nanpercentile(truth, 90.0)), 3),
    }
    print("[6] 检验：" + "，".join("{0}={1:.3f}".format(k, v) for k, v in metrics.items()))

    # 7) 绘图（可选依赖）
    try:
        from vibe_post import plots as pl
    except Exception as exc:            # pragma: no cover - 依赖缺失路径
        print("[7] 跳过绘图：{0}".format(exc))
        return 0
    try:
        fig, _ = pl.plot_field(loaded, "theta", level=0, title="theta @ 第 1 层")
        fig.savefig(out_dir / "quickstart_theta.png", dpi=120, bbox_inches="tight")
        pl.get_mpl()[0].close(fig)
        fig, _ = pl.plot_spectra(cell["w"][-1], dx=grid.dx)
        fig.savefig(out_dir / "quickstart_spectra.png", dpi=120, bbox_inches="tight")
        pl.get_mpl()[0].close(fig)
        print("[7] 已写出 quickstart_theta.png 与 quickstart_spectra.png")
    except Exception as exc:            # pragma: no cover - 缺依赖
        print("[7] 绘图失败（可能缺少 matplotlib）：{0}".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

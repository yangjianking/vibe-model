"""`vibe_post.interp` 单元测试：双/三线性、PCHIP、保守粗化与延拓。

运行：`pytest tests/test_interp.py -q`。
全部使用解析函数做基准（线性场、二次场），覆盖边界与退化情形。
文献：[D3] Arakawa & Lamb (1977)；[B9] Wilks (2019) 第 3 章；[N5] 保守重映射。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vibe_post.interp import (
    REGISTRY,
    bilinear,
    bilinear_prolong,
    column_interp_to_height,
    column_interp_to_pressure,
    conservative_coarsen,
    extrapolate_profile,
    interp_to_height,
    interp_to_pressure,
    pchip,
    pchip_slopes,
    trilinear,
)


def test_bilinear_reproduces_linear_field() -> None:
    """双线性对双线性场是**精确**的（二阶格式的再生性）。"""
    nx, ny, dx = 8, 6, 100.0
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    field = 2.0 + 3.0 * xx + 5.0 * yy

    px = np.array([150.0, 350.0, 650.0])
    py = np.array([250.0, 150.0, 450.0])
    expected = 2.0 + 3.0 * px + 5.0 * py
    np.testing.assert_allclose(bilinear(field, px, py, dx=dx, dy=dx), expected, rtol=1e-12)


def test_bilinear_out_of_range_modes() -> None:
    """越界点：extrapolate=True 用零梯度钳位，False 返回 NaN。"""
    field = np.arange(9, dtype=np.float64).reshape(3, 3)
    clamped = bilinear(field, np.array([-50.0]), np.array([-50.0]), dx=1.0, dy=1.0)
    assert clamped[0] == pytest.approx(field[0, 0])

    outside = bilinear(field, np.array([5.0]), np.array([5.0]), dx=1.0, dy=1.0,
                       extrapolate=False)
    assert np.isnan(outside[0])

    # 点 (1.0, 1.0) 落在体心 (0.5,0.5) 与 (1.5,1.5) 之间，权重各半
    edge = bilinear(field, np.array([1.0]), np.array([1.0]), dx=1.0, dy=1.0)
    assert edge[0] == pytest.approx(0.5 * (field[0, 0] + field[1, 1]))  # edge check


def test_trilinear_vertical_linear_exact() -> None:
    """三线性对线性垂直廓线精确，并正确使用逐列高度场。"""
    nz, ny, nx = 5, 3, 4
    heights = np.linspace(0.0, 4000.0, nz)
    height = np.broadcast_to(heights[:, None, None], (nz, ny, nx)).copy()
    field = 10.0 + 0.01 * height                     # 与 z 线性
    z = np.array([500.0, 2500.0, 3900.0])
    out = trilinear(field, np.array([0.5, 1.5, 2.5]), np.array([0.5, 1.5, 0.5]), z, height)
    np.testing.assert_allclose(out, 10.0 + 0.01 * z, rtol=1e-10)

    with pytest.raises(ValueError):
        trilinear(field, 0.5, 0.5, 100.0, height[..., :2])


def test_pchip_monotone_and_no_overshoot() -> None:
    """PCHIP 必须保持单调、不过冲，并在节点上精确插值。"""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])       # 含平坦段
    xq = np.linspace(-0.5, 4.5, 101)
    out = pchip(x, y, xq)

    assert np.all(np.diff(out) >= -1e-12)          # 单调不减
    assert out.min() >= -1e-12 and out.max() <= 1.0 + 1e-12   # 无过冲
    np.testing.assert_allclose(pchip(x, y, x), y, atol=1e-12)
    # 越界钳位到端点值
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)

    slopes = pchip_slopes(x, y)
    assert slopes[1] == pytest.approx(0.0)
    assert slopes[3] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pchip_slopes(np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]))


def test_pchip_handles_degenerate_inputs() -> None:
    """单点、空输入与含 NaN 的剖面不能崩溃。"""
    assert pchip([1.0], [5.0], [0.0, 1.0, 2.0])[0] == pytest.approx(5.0)
    empty = pchip(np.array([]), np.array([]), np.array([1.0]))
    assert np.isnan(empty[0])

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, np.nan, 2.0, 3.0])
    out = interp_to_height(x, y, np.array([0.5, 1.5, 2.5]))
    assert np.all(np.isfinite(out))               # NaN 层被跳过
    # 层序乱序时 pchip 自动排序，结果与升序一致
    x_unsorted = np.array([2.0, 0.0, 1.0, 3.0])
    out2 = interp_to_height(x_unsorted, np.array([2.0, 0.0, 1.0, 3.0]), np.array([1.5]))
    assert out2[0] == pytest.approx(1.5, abs=1e-9)


def test_column_interp_shapes_and_pressure_flip() -> None:
    """逐列插值：形状跟随目标；气压递减坐标自动翻转。"""
    nz, ny, nx = 6, 3, 4
    z = np.linspace(0.0, 5000.0, nz)
    height = np.broadcast_to(z[:, None, None], (nz, ny, nx)).copy()
    field = np.broadcast_to(z[:, None, None] * 2.0, (nz, ny, nx)).copy()

    plane = column_interp_to_height(height, field, 2500.0)
    assert plane.shape == (ny, nx)
    np.testing.assert_allclose(plane, 5000.0, rtol=1e-9)

    per_point = column_interp_to_height(height, field, np.full((ny, nx), 1000.0))
    assert per_point.shape == (ny, nx)
    np.testing.assert_allclose(per_point, 2000.0, rtol=1e-9)

    pressure = np.broadcast_to((100000.0 - z * 10.0)[:, None, None], (nz, ny, nx)).copy()
    at_p = column_interp_to_pressure(pressure, field, 95000.0)
    assert at_p.shape == (ny, nx)
    np.testing.assert_allclose(at_p, 1000.0, rtol=1e-9)

    with pytest.raises(ValueError):
        column_interp_to_height(height, field[:, :2, :], 1000.0)


def test_conservative_coarsen_preserves_mean() -> None:
    """面积均匀时，保守粗化必须保持区域平均（守恒性）。"""
    field = np.arange(64, dtype=np.float64).reshape(8, 8).T   # 行主序：i 最快
    # 转置后 field[:2, :2] = [[0, 8], [1, 9]]，均值 4.5
    coarse = conservative_coarsen(field, 2)
    assert coarse.shape == (4, 4)
    assert coarse.mean() == pytest.approx(field.mean(), rel=1e-12)
    assert coarse.sum() * 4 == pytest.approx(field.sum(), rel=1e-12)
    np.testing.assert_allclose(coarse[0, 0], field[:2, :2].mean())  # 4.5

    with pytest.raises(ValueError):
        conservative_coarsen(np.zeros((7, 8)), 2)
    with pytest.raises(ValueError):
        conservative_coarsen(np.zeros((8, 8)), 0)


def test_conservative_coarsen_nan_policy() -> None:
    """nan_policy='skip' 忽略 NaN 并重归一化；全 NaN 块返回 NaN。"""
    field = np.ones((4, 4))
    field[0, 0] = np.nan
    skipped = conservative_coarsen(field, 2, nan_policy="skip")
    assert skipped[0, 0] == pytest.approx(1.0)   # 忽略 NaN 并重归一化
    propagated = conservative_coarsen(field, 2, nan_policy="propagate")
    assert propagated[0, 0] == pytest.approx(0.75)  # NaN 视为 0，分母仍是 4

    all_nan = np.full((2, 2), np.nan)
    assert np.isnan(conservative_coarsen(all_nan, 2)[0, 0])
    with pytest.raises(ValueError):
        conservative_coarsen(field, 2, nan_policy="bogus")


def test_bilinear_prolong_constant_and_positive() -> None:
    """延拓：常数场保持常数；线性场的细网格值落在粗网格线性插值上。"""
    coarse = np.full((3, 4), 7.5)
    fine = bilinear_prolong(coarse, 2)
    assert fine.shape == (6, 8)
    np.testing.assert_allclose(fine, 7.5, rtol=1e-12)

    ramp = np.arange(4, dtype=np.float64)[None, :] * np.ones((3, 1))
    fine_ramp = bilinear_prolong(ramp, 2)
    assert fine_ramp.shape == (6, 8)
    assert np.all(np.diff(fine_ramp[0]) >= -1e-12)      # 单调性保持

    negative = np.array([[1.0, -0.5], [0.2, 0.3]])
    clamped = bilinear_prolong(negative, 2, enforce_positive=True)
    assert clamped.min() >= 0.0
    with pytest.raises(ValueError):
        bilinear_prolong(coarse, 0)


def test_extrapolate_profile_modes() -> None:
    """端点外推：nearest/linear/log 三种策略，中间空洞保持 NaN。"""
    z = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([np.nan, 2.0, np.nan, 8.0])
    nearest = extrapolate_profile(z, values, fill="nearest")
    assert nearest[0] == pytest.approx(2.0)
    assert np.isnan(nearest[2])                     # 中间空洞不填补

    full = np.array([1.0, 2.0, 3.0, 4.0])
    tail = np.array([1.0, 2.0, np.nan, np.nan])
    linear = extrapolate_profile(z, tail, fill="linear")
    np.testing.assert_allclose(linear[2:], [3.0, 4.0])
    log_fill = extrapolate_profile(z, np.array([1.0, 10.0, np.nan, np.nan]), fill="log")
    assert log_fill[-1] > log_fill[-2] > 0.0
    with pytest.raises(ValueError):
        extrapolate_profile(z, values, fill="bogus")


def test_registry_exposes_all_functions() -> None:
    """REGISTRY 必须覆盖对外承诺的算法名。"""
    for name in ("bilinear", "trilinear", "pchip", "interp_to_height",
                 "interp_to_pressure", "conservative_coarsen", "bilinear_prolong",
                 "extrapolate_profile"):
        assert name in REGISTRY


def _run_all() -> int:
    import traceback

    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS", name)
            except Exception:
                failures += 1
                print("FAIL", name)
                traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())

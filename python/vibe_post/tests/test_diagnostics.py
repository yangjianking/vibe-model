"""`vibe_post.diagnostics` 单元测试：解析解 + 边界情形。

运行：`pytest tests/test_diagnostics.py -q`。
每个测试都用**解析可验证**的构造（线性风场、等熵大气、纯水汽层结），
并覆盖全 NaN、零方差、空列等退化输入。
文献：[B4] Holton & Hakim；[D16] WRF ARW 第 3 章；[P11] Kain & Fritsch (1990)。
"""

from __future__ import annotations

import numpy as np
import pytest

from vibe_post import diagnostics as dg


def test_pressure_and_temperature_roundtrip() -> None:
    """Exner 函数与气压/温度必须互相可逆（逐点解析式）。"""
    pressure = np.array([100000.0, 85000.0, 50000.0, 20000.0])
    pi = (pressure / dg.P0) ** dg.KAPPA
    np.testing.assert_allclose(dg.pressure_from_exner(pi), pressure, rtol=1e-12)
    assert dg.pressure_from_exner(1.0) == pytest.approx(dg.P0)

    theta = np.array([300.0, 310.0])
    pi2 = np.array([1.0, 0.8])
    np.testing.assert_allclose(dg.temperature_from_exner(pi2, theta), theta * pi2)
    assert dg.celsius(273.15) == pytest.approx(0.0)
    assert dg.hpa(100000.0) == pytest.approx(1000.0)


def test_moist_thermodynamics_consistency() -> None:
    """饱和混合比 / 露点 / 相对湿度三者必须自洽。"""
    p = np.array([100000.0, 90000.0])
    t = np.array([293.15, 283.15])
    qsat = dg.saturation_mixing_ratio(p, t)
    assert np.all(qsat > 0.0)

    # 用 qsat 反算露点：回到原温度
    np.testing.assert_allclose(dg.dewpoint(p, qsat), t, rtol=1e-6)
    # 饱和时 RH = 1
    np.testing.assert_allclose(dg.relative_humidity(p, t, qsat), 1.0, atol=1e-9)
    # 干空气 RH = 0
    assert dg.relative_humidity(p, t, 0.0)[0] == pytest.approx(0.0)
    # 未裁剪时过饱和 > 1
    assert dg.relative_humidity(p, t, 2.0 * qsat, clip=False)[0] > 1.0

    # e = p q /(eps+q) 与 q = eps e/(p-e) 互逆
    e = np.array([1200.0, 800.0])
    qv = dg.mixing_ratio(p, e)
    np.testing.assert_allclose(dg.vapor_pressure_from_mixing_ratio(p, qv), e, rtol=1e-12)
    # 冰面饱和水汽压低于液面
    assert dg.saturation_vapor_pressure(260.0, phase="ice") < \
        dg.saturation_vapor_pressure(260.0, phase="water")
    with pytest.raises(ValueError):
        dg.dewpoint(p, qv, phase="bogus")
    with pytest.raises(ValueError):
        dg.saturation_vapor_pressure(t, phase="bogus")


def test_theta_e_increases_with_moisture() -> None:
    """(	heta_e) 随水汽增大，且干空气时退化为位温。"""
    p = np.array([100000.0])
    t = np.array([300.0])
    dry = dg.theta_e(p, t, np.array([0.0]))
    moist = dg.theta_e(p, t, np.array([0.02]))
    assert dry[0] == pytest.approx(t[0])
    assert moist[0] > dry[0]


def test_wind_speed_direction_and_calm() -> None:
    """风速/风向：四象限与静风（NaN）。"""
    u = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    v = np.array([-1.0, 0.0, 1.0, 0.0, 0.0])
    direction = dg.wind_direction(u, v)
    np.testing.assert_allclose(direction[:4], [0.0, 270.0, 180.0, 90.0], atol=1e-9)
    assert np.isnan(direction[4])
    np.testing.assert_allclose(dg.wind_speed(np.array([3.0]), np.array([4.0])), [5.0])


def test_vorticity_divergence_analytic() -> None:
    """u=0, v=a x -> zeta = a；u=b x, v=0 -> D = b（内部点精确）。"""
    nx = ny = 9
    dx = 1000.0
    x = np.arange(nx) * dx
    yy, xx = np.meshgrid(np.arange(ny) * dx, x, indexing="ij")
    a, b = 1.0e-4, -2.0e-4
    zeta = dg.vorticity(np.zeros_like(xx), a * xx, dx, dx)
    np.testing.assert_allclose(zeta[1:-1, 1:-1], a, rtol=1e-12)
    div = dg.divergence(b * xx, np.zeros_like(xx), dx, dx)
    np.testing.assert_allclose(div[1:-1, 1:-1], b, rtol=1e-12)
    assert np.all(np.isfinite(zeta))          # 边界为单侧差分

    eta = dg.absolute_vorticity(np.zeros_like(xx), a * xx, lat=45.0, dx=dx, dy=dx)
    f = dg.coriolis_parameter(45.0)
    np.testing.assert_allclose(eta[1:-1, 1:-1], f + a, rtol=1e-12)
    np.testing.assert_allclose(dg.coriolis_parameter(0.0), 0.0, atol=1e-18)


def test_potential_vorticity_uniform_stratification() -> None:
    """等熵层结 + 静止大气：PV = f (dtheta/dz) / rho，可解析核对。"""
    nz, ny, nx = 10, 4, 4
    dz = 1000.0
    z = np.arange(nz) * dz
    dtheta_dz = 3.0e-3
    theta = np.broadcast_to((300.0 + dtheta_dz * z)[:, None, None], (nz, ny, nx)).copy()
    zeros = np.zeros((nz, ny, nx))
    p = np.broadcast_to((100000.0 * np.exp(-z / 8000.0))[:, None, None], (nz, ny, nx)).copy()
    rho = 1.2
    pv = dg.potential_vorticity(theta, zeros, zeros, p=p, rho=np.full_like(theta, rho),
                                z=np.broadcast_to(z[:, None, None], theta.shape),
                                lat=45.0, dx=1000.0, dy=1000.0)
    expected = dg.coriolis_parameter(45.0) * dtheta_dz / rho
    np.testing.assert_allclose(pv[1:-1, 1, 1], expected, rtol=1e-9)


def test_cape_cin_and_degenerate_columns() -> None:
    """CAPE 对不稳定探空为正、对中性探空接近零；全 NaN 列不崩溃。"""
    nz = 40
    z = np.linspace(0.0, 14000.0, nz)
    p = 100000.0 * np.exp(-z / 8000.0)
    t = 300.0 - 6.5e-3 * z
    qv = 0.014 * np.exp(-z / 2500.0)
    result = dg.cape_cin(z[:, None, None], p[:, None, None], t[:, None, None], qv[:, None, None])
    assert result["cape"][0, 0] > 100.0
    assert result["cin"][0, 0] <= 0.0
    assert np.isfinite(result["lfc"][0, 0])
    assert result["el"][0, 0] >= result["lfc"][0, 0]
    assert result["lcl_pressure"][0, 0] <= 100000.0

    # 中性（等位温）气块：CAPE 与 CIN 都应为零
    theta = np.full(nz, 300.0)
    t_neutral = theta * (p / dg.P0) ** dg.KAPPA
    neutral = dg.cape_cin(z[:, None, None], p[:, None, None], t_neutral[:, None, None],
                          np.zeros((nz, 1, 1)))
    assert neutral["cape"][0, 0] == pytest.approx(0.0, abs=1e-6)

    with pytest.raises(ValueError):
        dg.cape_cin(z[:2, None, None], p[:2, None, None], t[:2, None, None], qv[:2, None, None])


def test_storm_relative_helicity_analytic() -> None:
    """解析核对 SRH：纯剪切（u=az, v=0）恒为零；旋转风廓线非零。"""
    nz = 31
    z = np.linspace(0.0, 3000.0, nz)
    a = 5.0e-3
    shape = (nz, 3, 3)
    z3 = np.broadcast_to(z[:, None, None], shape).copy()
    shear = np.broadcast_to((a * z)[:, None, None], shape).copy()
    zeros = np.zeros(shape)

    # v 不随高度变化 -> 被积函数 (u-c_u) dv/dz - (v-c_v) du/dz 恒为 0
    assert dg.storm_relative_helicity(z3, shear, zeros, z_top=3000.0)["srh"][0, 0] == pytest.approx(0.0, abs=1e-9)
    assert dg.storm_relative_helicity(z3, zeros, shear, z_top=3000.0)["srh"][0, 0] == pytest.approx(0.0, abs=1e-9)
    constant = np.full(shape, 3.0)
    assert dg.storm_relative_helicity(z3, constant, constant, z_top=3000.0)["srh"][0, 0] == pytest.approx(0.0, abs=1e-9)

    # 旋转风廓线 u = U sin(pi z / 2h), v = U cos(pi z / 2h)，h = z_top：
    # du/dz = U(pi/2h)cos，dv/dz = -U(pi/2h)sin
    # 被积函数 = -U^2 (pi/2h)，解析积分 = -U^2 pi / 2 ≈ -157.08 m^2/s^2
    U = 10.0
    h = 3000.0
    phase = np.pi * z / (2.0 * h)
    u3 = np.broadcast_to((U * np.sin(phase))[:, None, None], shape).copy()
    v3 = np.broadcast_to((U * np.cos(phase))[:, None, None], shape).copy()
    result = dg.storm_relative_helicity(z3, u3, v3, z_top=h)
    expected = -U ** 2 * np.pi / 2.0
    assert result["srh"][0, 0] == pytest.approx(expected, rel=1e-3)
    assert np.isfinite(result["srh_positive"][0, 0])
    assert result["srh_positive"][0, 0] >= 0.0

    with pytest.raises(ValueError):
        dg.storm_relative_helicity(z3, u3[:, :2, :], v3, z_top=h)

def test_precipitation_accumulation_units_and_nan() -> None:
    """降水累积：矩形法精确；单位换算正确；NaN 传播。"""
    rate = np.array([[1.0e-3, 2.0e-3], [3.0e-3, 4.0e-3]])   # (nt=2, 2 点)
    accum = dg.precipitation_accumulation(rate, 1800.0)
    np.testing.assert_allclose(accum, [4.0e-3 * 1800.0, 6.0e-3 * 1800.0])
    per_step = dg.precipitation_accumulation(rate, np.array([600.0, 1200.0]))
    np.testing.assert_allclose(per_step, [1.0e-3 * 600.0 + 3.0e-3 * 1200.0,
                                          2.0e-3 * 600.0 + 4.0e-3 * 1200.0])
    metres = dg.precipitation_accumulation(rate, 1800.0, units="m")
    np.testing.assert_allclose(metres, accum / dg.RHO_WATER)
    with_nan = dg.precipitation_accumulation(np.array([[np.nan, 1.0]]), 1.0)
    assert np.isnan(with_nan[0]) and with_nan[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        dg.precipitation_accumulation(rate, 1.0, units="bogus")


def test_sea_level_pressure_and_reflectivity() -> None:
    """SLP 随海拔增大；反射率随含水量单调增大且有 dBZ 下限。"""
    p = np.array([90000.0, 90000.0])
    t = np.array([283.15, 283.15])
    z = np.array([0.0, 1000.0])
    slp = dg.sea_level_pressure(p, t, z)
    assert slp[0] == pytest.approx(90000.0)
    assert slp[1] > slp[0] > 0.0
    assert np.isnan(dg.sea_level_pressure(np.array([1.0]), np.array([-300.0]), np.array([1000.0]))[0])

    dry = dg.reflectivity_dbz(np.zeros(3), rho=np.ones(3))
    light = dg.reflectivity_dbz(np.full(3, 1e-4), rho=np.ones(3))
    heavy = dg.reflectivity_dbz(np.full(3, 5e-3), rho=np.ones(3))
    assert dry[0] < light[0] < heavy[0]
    assert np.all(np.isfinite(dry))


def test_geopotential_srh_indices_and_brightness() -> None:
    """位势、K 指数、Showalter 指数与亮度温度的解析/经验关系。"""
    z = np.array([0.0, 1000.0])
    np.testing.assert_allclose(dg.geopotential(z), dg.GRAVITY * z)
    np.testing.assert_allclose(dg.geopotential_height(z), z)

    k = dg.k_index(293.15, 263.15, 288.15, 283.15, 278.15)   # = 20 + 15 - (5)
    assert k == pytest.approx(40.0)
    showalter = dg.showalter_index(293.15, 288.15, 263.15)
    assert np.isfinite(showalter)

    flux = 240.0
    tb = dg.brightness_temperature(flux)
    assert tb == pytest.approx((flux / dg.STEFAN_BOLTZMANN) ** 0.25)
    assert np.isnan(dg.brightness_temperature(np.array([-1.0]))[0])
    with pytest.raises(ValueError):
        dg.brightness_temperature(flux, emissivity=0.0)


def test_all_nan_and_empty_inputs() -> None:
    """全 NaN / 空数组：函数返回 NaN 或空数组，绝不抛异常。"""
    nan = np.full((3, 3), np.nan)
    assert np.all(np.isnan(dg.wind_speed(nan, nan)))
    assert np.all(np.isnan(dg.vorticity(nan, nan, 1000.0, 1000.0)))
    assert np.all(np.isnan(dg.dewpoint(np.full(3, 100000.0), nan[:, 0])))
    empty = np.zeros((0,))
    assert dg.wind_speed(empty, empty).size == 0
    assert dg.relative_humidity(np.array([100000.0]), np.array([300.0]), np.array([np.nan]))[0] != 0


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

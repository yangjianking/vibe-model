"""诊断量：由预报状态导出热力学、动力与遥感量（numpy 向量化，无状态）。

对应 C++ `vibe::dyn`（include/vibe/dyn/diagnostics.hpp）。**函数式 API**：
每个函数都是纯函数，输入/输出 numpy 数组，不持有状态，便于单元测试与 GPU
内核逐点对齐。

单位约定（全部 SI）
------------------
位温 :math:`\\theta` [K]，Exner 函数 :math:`\\Pi` [无量纲]，
气压 :math:`p` [Pa]，温度 :math:`T` [K]，水汽混合比 :math:`q_v` [kg/kg]，
密度 :math:`\\rho` [kg/m^3]，风速 [m/s]，高度 [m]。

索引约定
--------
所有场按 numpy C 序存为 `(k, j, i)`（即 `(z, y, x)`），与 C++
`Grid::flatten` 的行主序（i 最快）一致。水平差分用**索引对齐的中心差分**
（内部二阶、边界退化为单侧一阶），与 C++ `dyn::relative_vorticity` 的
C-grid 约定逐点一致。

文献：[B4] Holton & Hakim 第 3、4、8 章；[D3] Arakawa & Lamb (1977)；
[D16] WRF ARW 第 3 章；[P3] Thompson et al. (2008)；[P11] Kain & Fritsch (1990)；
[O6] Sun & Crook (1997)。
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "GRAVITY", "RD", "RV", "CP", "CV", "KAPPA", "EPSILON", "P0", "T0",
    "LV", "LS", "LF", "RHO_WATER", "STEFAN_BOLTZMANN", "OMEGA", "EARTH_RADIUS",
    "celsius", "hpa",
    "pressure_from_exner", "temperature_from_exner",
    "saturation_vapor_pressure", "saturation_mixing_ratio", "mixing_ratio",
    "relative_humidity", "dewpoint", "theta_e",
    "wind_speed", "wind_direction", "geopotential", "geopotential_height",
    "coriolis_parameter", "relative_vorticity", "vorticity", "divergence",
    "absolute_vorticity", "potential_vorticity",
    "cape_cin", "storm_relative_helicity", "precipitation_accumulation",
    "sea_level_pressure", "reflectivity_dbz", "brightness_temperature",
    "k_index", "showalter_index", "list_diagnostics",
]

# ---------------------------------------------------------------------------
# 物理常数（与 include/vibe/common/constants.hpp 一致）
# ---------------------------------------------------------------------------
GRAVITY: float = 9.80665
RD: float = 287.05
RV: float = 461.51
CP: float = 1004.64
CV: float = 717.63
KAPPA: float = 0.28571
EPSILON: float = 0.62197
P0: float = 100000.0
T0: float = 273.15
LV: float = 2.501e6
LS: float = 2.834e6
LF: float = 3.337e5
RHO_WATER: float = 1000.0
STEFAN_BOLTZMANN: float = 5.670374419e-8
OMEGA: float = 7.2921159e-5
EARTH_RADIUS: float = 6.371229e6

_DEG = 180.0 / np.pi

#: 参考态的干空气虚温系数（Tv = T(1 + 0.608 q)）
_EPS_TV: float = 1.0 / EPSILON - 1.0     # = 0.6078

#: 雷达反射率的经验系数（[O6]）
_Z_COEFF: float = 3.63e9


# ---------------------------------------------------------------------------
# 单位换算
# ---------------------------------------------------------------------------


def celsius(t: ArrayLike) -> NDArray[np.float64]:
    """开尔文 -> 摄氏度 :math:`T_C = T_K - 273.15`。复杂度 O(N)。"""
    return np.asarray(t, dtype=np.float64) - T0


def hpa(p: ArrayLike) -> NDArray[np.float64]:
    """帕斯卡 -> 百帕 :math:`p_{hPa} = p_{Pa}/100`。复杂度 O(N)。"""
    return np.asarray(p, dtype=np.float64) / 100.0


def _as(*arrays: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """统一转 float64 ndarray。复杂度 O(1)（不复制时）。"""
    return tuple(np.asarray(a, dtype=np.float64) for a in arrays)


# ---------------------------------------------------------------------------
# 热力学
# ---------------------------------------------------------------------------


def pressure_from_exner(pi: ArrayLike, p0: float = P0) -> NDArray[np.float64]:
    """Exner 函数 -> 气压。

    公式（[D16] WRF ARW 第 3 章）
    ---------------------------
    .. math::

        \\Pi = \\left(\\frac{p}{p_0}\\right)^{R_d/c_p}
        \\quad\\Longleftrightarrow\\quad
        p = p_0\\,\\Pi^{c_p/R_d} = p_0\\,\\Pi^{1/\\kappa}

    离散化：逐点代数关系，无离散误差。
    复杂度：O(N)。
    """
    pi_arr = np.asarray(pi, dtype=np.float64)
    return p0 * np.power(pi_arr, 1.0 / KAPPA)


def temperature_from_exner(pi: ArrayLike, theta: ArrayLike) -> NDArray[np.float64]:
    """位温 + Exner 函数 -> 温度 :math:`T = \\theta\\,\\Pi`。

    参数顺序与 C++ `dyn::exner_to_temperature(pi, theta)` 一致。
    复杂度：O(N)。
    """
    pi_arr, theta_arr = _as(pi, theta)
    return theta_arr * pi_arr


def saturation_vapor_pressure(t: ArrayLike, *, phase: str = "auto") -> NDArray[np.float64]:
    """饱和水汽压（Tetens / Bolton 1980 拟合）。

    公式（[B4] Holton & Hakim 第 3 章）
    ---------------------------------
    液面（:math:`T\\ge 273.15` K）：

    .. math::

        e_s = 611.2\\,\\exp\\!\\left(
        \\frac{17.67\\,(T-273.15)}{T-29.65}\\right)

    冰面（:math:`T<273.15` K）：

    .. math::

        e_s = 611.2\\,\\exp\\!\\left(
        \\frac{22.46\\,(T-273.15)}{T-0.55}\\right)

    `phase="auto"` 按温度切换，`"water"`/`"ice"` 强制单一相态。

    离散化：逐点代数式；在 :math:`-40\\ldots+50\\,^\\circ`C 内相对误差 < 0.3%。
    复杂度：O(N)。
    """
    t_arr = np.asarray(t, dtype=np.float64)
    tc = t_arr - T0
    es_water = 611.2 * np.exp(17.67 * tc / (t_arr - 29.65))
    es_ice = 611.2 * np.exp(22.46 * tc / (t_arr - 0.55))
    if phase == "water":
        return es_water
    if phase == "ice":
        return es_ice
    if phase != "auto":
        raise ValueError("phase 只能是 'auto' / 'water' / 'ice'")
    return np.where(t_arr >= T0, es_water, es_ice)


def saturation_mixing_ratio(p: ArrayLike, t: ArrayLike, *,
                            phase: str = "auto") -> NDArray[np.float64]:
    """饱和混合比 :math:`q_{v,sat} = \\varepsilon e_s/(p-e_s)`。

    分母取 :math:`\\max(p-e_s,\\;0.01|p|)`，对应 C++ `common::safe_div` 的
    防除零语义。复杂度：O(N)。
    """
    p_arr, t_arr = _as(p, t)
    es = saturation_vapor_pressure(t_arr, phase=phase)
    denominator = np.maximum(p_arr - es, 1.0e-2 * np.maximum(np.abs(p_arr), 1.0))
    return EPSILON * es / denominator


def mixing_ratio(p: ArrayLike, e: ArrayLike) -> NDArray[np.float64]:
    """水汽混合比（精确式）:math:`q_v = \\varepsilon e/(p-e)`。

    与近似的 :math:`q_v \\approx \\varepsilon e/p` 不同，此式在热带湿空气中
    仍保持 < 0.1% 误差（[B4] 第 3 章）。逆变换
    :math:`e = p q_v/(\\varepsilon+q_v)` 在 :func:`dewpoint` 中使用。
    复杂度：O(N)。
    """
    p_arr, e_arr = _as(p, e)
    denominator = np.maximum(p_arr - e_arr, 1.0e-6 * np.maximum(np.abs(p_arr), 1.0))
    return EPSILON * e_arr / denominator


def vapor_pressure_from_mixing_ratio(p: ArrayLike, qv: ArrayLike) -> NDArray[np.float64]:
    """由混合比反解水汽压 :math:`e = p q_v/(\\varepsilon+q_v)`。复杂度 O(N)。"""
    p_arr, qv_arr = _as(p, qv)
    return p_arr * qv_arr / (EPSILON + qv_arr)


def relative_humidity(p: ArrayLike, t: ArrayLike, qv: ArrayLike, *,
                      phase: str = "auto", clip: bool = True) -> NDArray[np.float64]:
    """相对湿度 :math:`RH = q_v/q_{v,sat}(p,T) `（0~1）。

    `clip=True` 时限制到 :math:`[0,1]`（对应模式对过饱和的截断，[D16] 第 3 章）；
    `clip=False` 保留过饱和信息，供检验使用。
    复杂度：O(N)。
    """
    p_arr, t_arr, qv_arr = _as(p, t, qv)
    qsat = saturation_mixing_ratio(p_arr, t_arr, phase=phase)
    with np.errstate(invalid="ignore", divide="ignore"):
        rh = np.where(qsat > 0.0, qv_arr / qsat, np.nan)
    if clip:
        rh = np.clip(rh, 0.0, 1.0)
    return rh


def dewpoint(p: ArrayLike, qv: ArrayLike, *, phase: str = "water") -> NDArray[np.float64]:
    """露点温度（由水汽压解析反解 Tetens 公式）。

    公式（[B4] 第 3 章）
    ------------------
    .. math::

        e = \\frac{p\\,q_v}{\\varepsilon + q_v}, \\qquad
        T_d = \\frac{a_1(T_0-a_2)\\ln(e/e_0)}{a_1 - \\ln(e/e_0)} + T_0

    液面 :math:`(a_1,a_2)=(17.67,29.65)`、:math:`e_0=611.2` Pa；
    `phase="ice"` 用 :math:`(22.46,0.55)`（霜点 :math:`T_f`）。

    离散化：解析逆变换，与前向 :math:`e_s(T)` 同一拟合，故二者互逆。
    复杂度：O(N)。
    """
    p_arr, qv_arr = _as(p, qv)
    e = np.maximum(vapor_pressure_from_mixing_ratio(p_arr, qv_arr), 1.0e-10)
    ln_ratio = np.log(e / 611.2)
    if phase == "ice":
        a1, a2 = 22.46, 0.55
    elif phase == "water":
        a1, a2 = 17.67, 29.65
    else:
        raise ValueError("phase 只能是 'water' 或 'ice'")
    return (T0 - a2) * ln_ratio / (a1 - ln_ratio) + T0


def theta_e(p: ArrayLike, t: ArrayLike, qv: ArrayLike, *,
            lv: float = LV, cp: float = CP) -> NDArray[np.float64]:
    """相当位温（忽略液态水热容的常用形式）。

    公式
    ----
    .. math::

        \\theta_e = T\\left(\\frac{p_0}{p}\\right)^{R_d/c_p}
        \\exp\\!\\left(\\frac{L_v q_v}{c_p T}\\right)

    与 C++ `dyn::theta_e_from_rho` 使用同一形式（[B4] 第 3 章），用于判别
    气团性质与 CAPE 的气块抬升守恒量。`qv` 超过饱和值时调用方应先饱和化。
    复杂度：O(N)。
    """
    p_arr, t_arr, qv_arr = _as(p, t, qv)
    t_safe = np.maximum(t_arr, 1.0)
    exponent = KAPPA * np.log(P0 / np.maximum(p_arr, 1.0))
    return t_arr * np.exp(exponent) * np.exp(lv * np.maximum(qv_arr, 0.0) / (cp * t_safe))


# ---------------------------------------------------------------------------
# 风与位势
# ---------------------------------------------------------------------------


def wind_speed(u: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """风速 :math:`|\\mathbf V|=\\sqrt{u^2+v^2}` [m/s]。复杂度 O(N)。"""
    u_arr, v_arr = _as(u, v)
    return np.hypot(u_arr, v_arr)


def wind_direction(u: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """气象风向（度，风的**来向**，0=北、90=东）。

    公式
    ----
    .. math::

        \\phi = \\left[270^\\circ - \\frac{180}{\\pi}
        \\operatorname{atan2}(v,u)\\right] \\bmod 360^\\circ

    静风返回 NaN。复杂度：O(N)。
    """
    u_arr, v_arr = _as(u, v)
    with np.errstate(invalid="ignore"):
        direction = (270.0 - _DEG * np.arctan2(v_arr, u_arr)) % 360.0
    return np.where((u_arr == 0.0) & (v_arr == 0.0), np.nan, direction)


def geopotential(z: ArrayLike, g: float = GRAVITY) -> NDArray[np.float64]:
    """位势 :math:`\\Phi = g z` [m^2/s^2]，输入为海拔高度 [m]。复杂度 O(N)。"""
    return g * np.asarray(z, dtype=np.float64)


def geopotential_height(z: ArrayLike) -> NDArray[np.float64]:
    """位势高度 [m]（WMO 定义）：海平面以上几何高度。复杂度 O(N)。"""
    return np.asarray(z, dtype=np.float64)


def coriolis_parameter(lat_deg: ArrayLike) -> NDArray[np.float64]:
    """科氏参数 :math:`f = 2\\Omega\\sin\\varphi` [1/s]。复杂度 O(N)。"""
    return 2.0 * OMEGA * np.sin(np.asarray(lat_deg, dtype=np.float64) / _DEG)


# ---------------------------------------------------------------------------
# 动力诊断（C-grid 差分）
# ---------------------------------------------------------------------------


def _pad_x(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """沿最后一维（x）补一个零梯度副本，长度 +1。

    C++ `Grid::index(Stagger,i,j,k)` 中同一 (i,j,k) 在各错位上指向同一物理位置，
    Python 侧输入的错位场按 N 点解释（与体心场逐点对齐），故末尾补边界副本。
    复杂度 O(N)。
    """
    return np.concatenate((arr, arr[..., -1:]), axis=-1)


def _centered_diff(field: NDArray[np.float64], axis: int, spacing: float) -> NDArray[np.float64]:
    """索引对齐的一阶差分（内部中心、边界单侧）。

    公式（内部点）
    -------------
    .. math::

        \\left(\\frac{\\partial f}{\\partial x}\\right)_i
        = \\frac{f_{i+1/2} - f_{i-1/2}}{\\Delta x}
        = \\frac{f_{i} - f_{i-1}}{\\Delta x}
        \\quad (0 < i < N-1)

    注意最后一步等式利用了 C-grid 的**索引对齐**性质：面场第 i 个存储点与
    体心第 i 点位于同一物理位置，故中心差分退化为相邻差分，避免了一次额外的
    错位平均（[D3] 第 3 节）。末端点用前一差分复制（单侧一阶）。

    复杂度：O(N)；内存 O(N)。
    """
    moved = np.moveaxis(field, axis, -1)
    diff = np.diff(_pad_x(moved), axis=-1)
    result = diff if axis in (-1, field.ndim - 1) else np.moveaxis(diff, -1, axis)
    last = [slice(None)] * result.ndim
    prev = [slice(None)] * result.ndim
    last[axis] = slice(-1, None)
    prev[axis] = slice(-2, -1)
    result[tuple(last)] = result[tuple(prev)]
    return result / spacing


def _spacing(value: Any, axis: str, default: float = 1.0) -> float:
    """从标量或 GridSpec 取水平间距。复杂度 O(1)。"""
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) or default
    if axis == "x":
        return float(getattr(value, "dx", default)) or default
    return float(getattr(value, "dy", getattr(value, "dx", default))) or default


def vorticity(u: ArrayLike, v: ArrayLike, dx: Any = 1.0, dy: Any = None) -> NDArray[np.float64]:
    """相对涡度 :math:`\\zeta = \\partial v/\\partial x - \\partial u/\\partial y` [1/s]。

    公式（[B4] Holton & Hakim 第 4 章）
    ----------------------------------
    .. math::

        \\zeta = \\frac{\\partial v}{\\partial x}
                - \\frac{\\partial u}{\\partial y}

    离散化（C-grid，与 C++ `dyn::relative_vorticity` 一致）
    ------------------------------------------------------
    `u`（x 面）与 `v`（y 面）的存储索引与体心 (i,j,k) 一一对齐，故直接对
    两个场做索引对齐差分（:func:`_centered_diff`），内部点 :math:`O(\\Delta^2)`、
    边界点 :math:`O(\\Delta)`（[D3] 第 3 节）。

    参数
    ----
    u, v : array_like, shape (ny, nx) 或 (nz, ny, nx)
    dx, dy : float 或 GridSpec
        `dy=None` 时取 `dx`。

    复杂度：O(N)。
    """
    u_arr, v_arr = _as(u, v)
    dx_val = _spacing(dx, "x")
    dy_val = _spacing(dx if dy is None else dy, "y")
    return _centered_diff(v_arr, -1, dx_val) - _centered_diff(u_arr, -2, dy_val)


#: C++ 命名 `dyn::relative_vorticity` 的别名
relative_vorticity = vorticity


def divergence(u: ArrayLike, v: ArrayLike, dx: Any = 1.0, dy: Any = None) -> NDArray[np.float64]:
    """水平散度 :math:`D=\\partial u/\\partial x + \\partial v/\\partial y` [1/s]。

    公式与离散化同 :func:`vorticity`（[B4] 第 4 章）。
    复杂度：O(N)。
    """
    u_arr, v_arr = _as(u, v)
    dx_val = _spacing(dx, "x")
    dy_val = _spacing(dx if dy is None else dy, "y")
    return _centered_diff(u_arr, -1, dx_val) + _centered_diff(v_arr, -2, dy_val)


def absolute_vorticity(u: ArrayLike, v: ArrayLike, lat: ArrayLike | float = 45.0,
                       dx: Any = 1.0, dy: Any = None) -> NDArray[np.float64]:
    """绝对涡度 :math:`\\eta = f + \\zeta` [1/s]。

    `lat` 可为标量（f 平面）或与水平网格同形的数组（:math:`\\beta` 平面）。
    复杂度：O(N)。
    """
    zeta = vorticity(u, v, dx, dy)
    f = coriolis_parameter(np.asarray(lat, dtype=np.float64))
    try:
        f = np.broadcast_to(f, zeta.shape)
    except ValueError:
        f = np.broadcast_to(f.reshape(f.shape + (1,) * (zeta.ndim - f.ndim)), zeta.shape)
    return f + zeta


def _vertical_derivative(field: NDArray[np.float64], z: Any, dz: Any) -> NDArray[np.float64]:
    """沿层方向（axis=0）的垂直导数，结果位于层中心。

    公式
    ----
    .. math::

        \\left(\\frac{\\partial f}{\\partial z}\\right)_k
        = \\frac{f_{k+1}-f_k}{z_{k+1}-z_k}
        \\quad\\text{（层间）},\\qquad
        \\left.\\frac{\\partial f}{\\partial z}\\right|_{k+1/2}
        \\to \\text{层中心取相邻层间导数的算术平均}

    端点用单侧层间导数。`z` 为高度场时按非均匀层厚；否则用常数 `dz`。
    `z` 缺省且 `dz` 缺省时取单位间距（仅用于无量纲测试）。
    复杂度：O(N)。
    """
    if z is not None:
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.shape != field.shape:
            try:
                z_arr = np.broadcast_to(z_arr, field.shape)
            except ValueError as exc:
                raise ValueError("z 形状 {0} 无法广播到场形状 {1}".format(
                    z_arr.shape, field.shape)) from exc
        thickness = np.diff(z_arr, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            layer = np.where(thickness != 0.0, np.diff(field, axis=0) / thickness, 0.0)
    else:
        spacing = 1.0 if dz is None else float(np.asarray(dz, dtype=np.float64).ravel()[0])
        if spacing == 0.0:
            raise ValueError("dz 不能为 0")
        layer = np.diff(field, axis=0) / spacing

    out = np.empty_like(field)
    out[0] = layer[0]
    out[-1] = layer[-1]
    if field.shape[0] > 2:
        out[1:-1] = 0.5 * (layer[:-1] + layer[1:])
    return out


def potential_vorticity(theta: ArrayLike, u: ArrayLike, v: ArrayLike, *,
                        p: ArrayLike | None = None, rho: ArrayLike | None = None,
                        t: ArrayLike | None = None, exner: ArrayLike | None = None,
                        qv: ArrayLike | None = None, lat: ArrayLike | float = 45.0,
                        z: ArrayLike | None = None, dz: ArrayLike | None = None,
                        dx: Any = 1.0, dy: Any = None) -> NDArray[np.float64]:
    """Ertel 位涡（干形式）。

    公式（[B4] Holton & Hakim 第 4 章）
    ----------------------------------
    .. math::

        PV = \\frac{1}{\\rho}\\,\\boldsymbol\\omega_a\\cdot\\nabla\\theta,
        \\qquad
        \\boldsymbol\\omega_a=\\left(-\\frac{\\partial v}{\\partial z},\\;
        \\frac{\\partial u}{\\partial z},\\;
        f+\\frac{\\partial v}{\\partial x}-\\frac{\\partial u}{\\partial y}\\right)

    展开（[B4] 式 4.26）：

    .. math::

        PV=\\frac{1}{\\rho}\\left[
          -\\frac{\\partial v}{\\partial z}\\frac{\\partial\\theta}{\\partial x}
          +\\frac{\\partial u}{\\partial z}\\frac{\\partial\\theta}{\\partial y}
          +\\eta\\,\\frac{\\partial\\theta}{\\partial z}\\right]

    离散化
    ------
    * 水平导数：索引对齐中心差分（同 :func:`vorticity`），内部 :math:`O(\\Delta^2)`；
    * 垂直导数：以 `z` 为自变量的层间差分映射到层中心（:func:`_vertical_derivative`）；
    * `rho` 缺省时由 :math:`\\rho=p/(R_d T_v)` 反算，其中
      :math:`T_v=T(1+\\varepsilon' q_v)`（`qv` 给了才启用；`t` 缺省时用
      :math:`T=\\theta\\Pi`）。三者皆缺时按单位密度计算并给出警告。

    单位 :math:`[\\mathrm{K\\,m^2\\,kg^{-1}\\,s^{-1}}]`；除以
    :math:`10^{-6}` 即得 PVU。复杂度：O(N)。
    """
    theta_arr, u_arr, v_arr = _as(theta, u, v)
    if u_arr.shape != theta_arr.shape:
        raise ValueError("u 形状 {0} 与 theta 形状 {1} 不一致".format(u_arr.shape, theta_arr.shape))
    dx_val = _spacing(dx, "x")
    dy_val = _spacing(dx if dy is None else dy, "y")

    dthdx = _centered_diff(theta_arr, -1, dx_val)
    dthdy = _centered_diff(theta_arr, -2, dy_val)
    dudz = _vertical_derivative(u_arr, z, dz)
    dvdz = _vertical_derivative(v_arr, z, dz)
    dthdz = _vertical_derivative(theta_arr, z, dz)
    eta = absolute_vorticity(u_arr, v_arr, lat, dx_val, dy_val)
    density = _density_from(p, rho, t, exner, qv, theta_arr)

    numerator = -dvdz * dthdx + dudz * dthdy + eta * dthdz
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(density > 0.0, numerator / density, np.nan)


def _density_from(p: Any, rho: Any, t: Any, exner: Any, qv: Any,
                  theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """确定密度场：显式 `rho` > 理想气体 :math:`p/(R_dT_v)` > 单位值。复杂度 O(N)。"""
    if rho is not None:
        return np.asarray(rho, dtype=np.float64)
    if p is not None:
        p_arr = np.asarray(p, dtype=np.float64)
        if t is not None:
            t_arr = np.asarray(t, dtype=np.float64)
        elif exner is not None:
            t_arr = theta * np.asarray(exner, dtype=np.float64)
        else:
            t_arr = np.full_like(p_arr, 288.15)
        if qv is not None:
            t_arr = t_arr * (1.0 + _EPS_TV * np.maximum(np.asarray(qv, dtype=np.float64), 0.0))
        return p_arr / (RD * np.maximum(t_arr, 1.0))
    warnings.warn("potential_vorticity: 未提供 rho/p，按单位密度计算", stacklevel=3)
    return np.ones_like(theta)


# ---------------------------------------------------------------------------
# CAPE / CIN（气块法，逐列）
# ---------------------------------------------------------------------------


def _lcl_temperature(t: NDArray[np.float64], qv: NDArray[np.float64],
                     p: NDArray[np.float64]) -> NDArray[np.float64]:
    """抬升凝结温度 :math:`T_{LCL}`（Bolton 1980 近似）。

    公式
    ----
    .. math::

        T_d = \\frac{17.67(T_0-29.65)\\ln(e/611.2)}{17.67-\\ln(e/611.2)}+T_0,
        \\qquad e=\\frac{p q_v}{\\varepsilon+q_v}

    .. math::

        T_{LCL}=\\left[\\frac{1}{T_d-56}
        +\\frac{\\ln(T/T_d)}{800}\\right]^{-1}+56

    复杂度：O(N)。
    """
    e = np.maximum(p * qv / (EPSILON + qv), 1.0e-10)
    ln_ratio = np.log(e / 611.2)
    td = 17.67 * (T0 - 29.65) * ln_ratio / (17.67 - ln_ratio) + T0
    td = np.clip(td, 100.0, t)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_lcl = 1.0 / (1.0 / np.maximum(td - 56.0, 1.0)
                       + np.log(np.maximum(t, 1.0) / np.maximum(td, 1.0)) / 800.0) + 56.0
    return np.where(np.isfinite(t_lcl), t_lcl, t)


def _dry_lapse(theta: NDArray[np.float64], p_target: ArrayLike) -> NDArray[np.float64]:
    """干绝热关系：由位温与目标气压求温度。

    公式
    ----
    .. math::

        T = \\theta (p_{target}/p_0)^{R_d/c_p} = \\theta (p_{target}/p_0)^{\\kappa}

    这是位温守恒的直接推论（[B4] 第 3 章）；离散化无误差。复杂度：O(N)。
    """
    p_arr = np.asarray(p_target, dtype=np.float64)
    return theta * np.power(np.maximum(p_arr, 1.0) / P0, KAPPA)

def _moist_temperature(p: NDArray[np.float64], theta_e_target: NDArray[np.float64],
                       t_guess: NDArray[np.float64], *, n_iter: int = 40,
                       tol: float = 1.0e-4) -> NDArray[np.float64]:
    """由守恒的 :math:`\\theta_e` 反解饱和气块温度（Newton + 钳位）。

    公式
    ----
    求 :math:`T` 使

    .. math::

        F(T)=T\\left(\\frac{p_0}{p}\\right)^{\\kappa}
        \\exp\\!\\left(\\frac{L_v\\,q_{v,sat}(T)}{c_p T}\\right)
        -\\theta_e^{target}=0

    :math:`F` 关于 :math:`T` 单调递增（:math:`q_{v,sat}` 随 :math:`T` 指数增），
    故 Newton 迭代自左向右全局收敛；步长限幅 ±20 K、解钳位 :math:`[120,400]` K
    保证在奇异点附近仍稳定。

    复杂度：O(n_iter × N)；内存 O(N)。
    """
    t = np.clip(np.asarray(t_guess, dtype=np.float64), 120.0, 400.0)
    ratio = np.power(P0 / np.maximum(p, 1.0), KAPPA)
    for _ in range(n_iter):
        qsat = saturation_mixing_ratio(p, t)
        with np.errstate(over="ignore"):
            f = t * ratio * np.exp(np.minimum(LV * qsat / (CP * t), 50.0)) - theta_e_target
        dt = 0.05
        t2 = t + dt
        qsat2 = saturation_mixing_ratio(p, t2)
        with np.errstate(over="ignore"):
            f2 = t2 * ratio * np.exp(np.minimum(LV * qsat2 / (CP * t2), 50.0)) - theta_e_target
        dfdt = (f2 - f) / dt
        step = np.where(np.abs(dfdt) > 1.0e-30, f / dfdt, 0.0)
        t_next = np.clip(t - np.clip(step, -20.0, 20.0), 120.0, 400.0)
        converged = np.all(np.abs(t_next - t) < tol)
        t = t_next
        if converged:
            break
    return t


def cape_cin(z: ArrayLike, p: ArrayLike, t: ArrayLike, qv: ArrayLike, *,
             qc: ArrayLike | None = None, z_start: float | ArrayLike | None = None,
             n_iter: int = 40) -> dict[str, NDArray[np.float64]]:
    """逐列气块法 CAPE / CIN / LFC / EL / LCL 气压。

    公式（[B4] 第 3 章；[P11] Kain & Fritsch (1990)）
    ------------------------------------------------
    .. math::

        CAPE = \\int_{z_{LFC}}^{z_{EL}}
               g\\,\\frac{T_{v,parcel}-T_{v,env}}{T_{v,env}}\\,dz
        \\\\
        CIN  = -\\int_{z_{start}}^{z_{LFC}}
               \\min\\!\\left(0,\\;
               g\\,\\frac{T_{v,parcel}-T_{v,env}}{T_{v,env}}\\right)dz

    虚温 :math:`T_v = T(1+0.608\\,q_v)`；云水计入虚温修正
    :math:`q_v \\to q_v + q_c`。

    离散化
    ------
    抬升路径：起点以下按 :math:`\\theta` 守恒的**干绝热**，LCL 以上按
    :math:`\\theta_e` 守恒的**伪湿绝热**（[P11] 标准步骤）。层界面浮力用
    相邻层平均，再用**梯形法**积分（对线性分布精确）；LFC 取首个正浮力层、
    EL 取其上方最后一个正浮力层（离散定义与 WRF `wrf_cape_2d` 一致，[D16]）。

    参数
    ----
    z, p, t, qv : (nz, ny, nx)
        层中心高度 [m]、气压 [Pa]、温度 [K]、水汽混合比 [kg/kg]。
    qc : 可选，云水混合比。
    z_start : 抬升起点高度；`None` 取最低层。
    n_iter : 湿绝热 Newton 迭代次数。

    返回
    ----
    `{"cape": J/kg, "cin": J/kg, "lfc": m, "el": m, "lcl_pressure": Pa}`，
    形状均为水平网格 `(ny, nx)`；无水平维时形状为 `()`。

    复杂度：O(nz × N_h × n_iter)；内存 O(nz × N_h)。
    """
    z_arr, p_arr, t_arr, qv_arr = _as(z, p, t, qv)
    if not (z_arr.shape == p_arr.shape == t_arr.shape == qv_arr.shape):
        raise ValueError("z / p / t / qv 形状必须一致，得到 {0}".format(
            (z_arr.shape, p_arr.shape, t_arr.shape, qv_arr.shape)))
    nz = z_arr.shape[0]
    if nz < 3:
        raise ValueError("CAPE 至少需要 3 层，得到 nz={0}".format(nz))
    horizontal = z_arr.shape[1:]

    qv_eff = qv_arr if qc is None else qv_arr + np.asarray(qc, dtype=np.float64)
    tv_env = t_arr * (1.0 + _EPS_TV * qv_eff)

    index_grid = np.arange(nz).reshape((nz,) + (1,) * len(horizontal))
    if z_start is None:
        start_index = np.zeros(horizontal, dtype=np.intp)
    else:
        target = np.broadcast_to(np.asarray(z_start, dtype=np.float64), horizontal)
        start_index = np.argmin(np.abs(z_arr - target[None, ...]), axis=0).astype(np.intp)
    start_index = np.clip(start_index, 0, nz - 2)

    take = start_index[None, ...]
    p_start = np.take_along_axis(p_arr, take, axis=0)[0]
    t_start = np.take_along_axis(t_arr, take, axis=0)[0]
    qv_start = np.maximum(np.take_along_axis(qv_arr, take, axis=0)[0], 0.0)
    z_start_val = np.take_along_axis(z_arr, take, axis=0)[0]
    theta_start = t_start * np.power(P0 / np.maximum(p_start, 1.0), KAPPA)

    t_lcl = _lcl_temperature(t_start, qv_start, p_start)
    p_lcl = np.maximum(p_start * np.power(t_lcl / np.maximum(t_start, 1.0), 1.0 / KAPPA), 1.0)
    qsat_lcl = saturation_mixing_ratio(p_lcl, t_lcl)
    theta_e_lcl = theta_e(p_lcl, t_lcl, qsat_lcl)

    t_dry = _dry_lapse(theta_start[None, ...], p_arr)
    t_moist = _moist_temperature(p_arr, np.broadcast_to(theta_e_lcl, p_arr.shape), t_dry,
                                 n_iter=n_iter)
    above_lcl = p_arr < p_lcl[None, ...]
    t_parcel = np.where(above_lcl, t_moist, t_dry)
    qv_parcel = np.where(above_lcl, saturation_mixing_ratio(p_arr, t_parcel),
                         np.broadcast_to(qv_start[None, ...], p_arr.shape))
    tv_parcel = t_parcel * (1.0 + _EPS_TV * np.maximum(qv_parcel, 0.0))

    at_or_above = index_grid >= start_index[None, ...]
    with np.errstate(invalid="ignore", divide="ignore"):
        buoyancy = GRAVITY * (tv_parcel - tv_env) / np.maximum(tv_env, 1.0)
    buoyancy = np.where(at_or_above, buoyancy, 0.0)
    buoyancy = np.nan_to_num(buoyancy, nan=0.0, posinf=0.0, neginf=0.0)

    dz = np.maximum(np.diff(z_arr, axis=0), 0.0)
    layer_buoyancy = 0.5 * (buoyancy[:-1] + buoyancy[1:])
    increments = np.concatenate(
        (np.zeros((1,) + horizontal), np.cumsum(layer_buoyancy * dz, axis=0)), axis=0)

    positive = buoyancy > 0.0
    has_positive = positive.any(axis=0)
    first_positive = np.argmax(positive, axis=0)
    last_positive = nz - 1 - np.argmax(positive[::-1], axis=0)

    lfc = np.where(has_positive,
                   np.take_along_axis(z_arr, first_positive[None, ...], axis=0)[0], np.nan)
    el = np.where(has_positive,
                  np.take_along_axis(z_arr, last_positive[None, ...], axis=0)[0], np.nan)

    cape = np.where(has_positive, np.maximum(increments[-1], 0.0), 0.0)
    below_lfc = index_grid <= first_positive[None, ...]
    negative_route = np.where(below_lfc, increments, np.inf)
    cin = np.where(has_positive, -np.minimum(negative_route.min(axis=0), 0.0), 0.0)
    cin = np.where(np.isfinite(cin), cin, 0.0)

    return {
        "cape": np.asarray(cape, dtype=np.float64),
        "cin": np.asarray(cin, dtype=np.float64),
        "lfc": np.asarray(lfc, dtype=np.float64),
        "el": np.asarray(el, dtype=np.float64),
        "lcl_pressure": np.asarray(p_lcl, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 其他诊断
# ---------------------------------------------------------------------------


def storm_relative_helicity(z: ArrayLike, u: ArrayLike, v: ArrayLike, *,
                            storm_u: float = 0.0, storm_v: float = 0.0,
                            z_top: float = 3000.0, z_bottom: float = 0.0,
                            n_sub: int | None = None) -> dict[str, NDArray[np.float64]]:
    """风暴相对螺旋度 SRH [m^2/s^2]。

    公式（[B4] 第 8 章；Davies-Jones 1990）
    --------------------------------------
    .. math::

        SRH=\\int_{z_b}^{z_t}\\left[
        (u-c_u)\\frac{\\partial v}{\\partial z}
        -(v-c_v)\\frac{\\partial u}{\\partial z}\\right]dz

    离散化
    ------
    先把 u、v 逐列线性插值到 :math:`\\Delta z = 100` m 的均匀细网格，再用**梯形法**
    积分（[B4] 第 8 章的常用离散）。对均匀输入层该插值是恒等映射，误差只来自横向
    离散；``srh_positive`` 只累计非负被积函数（业务常用的 0–3 km 正螺旋度）。

    返回 ``{"srh": ..., "srh_positive": ...}``，形状为水平网格。
    复杂度：O(n_sub x N_h)。
    """
    z_arr, u_arr, v_arr = _as(z, u, v)
    if not (z_arr.shape == u_arr.shape == v_arr.shape):
        raise ValueError("z / u / v 形状必须一致")
    nz = z_arr.shape[0]
    horizontal = z_arr.shape[1:]
    columns = int(np.prod(horizontal, dtype=int)) if horizontal else 1
    if n_sub is None:
        n_sub = max(int(np.ceil((z_top - z_bottom) / 100.0)) + 1, 2)
    n_sub = int(max(n_sub, 2))
    z_fine = np.linspace(z_bottom, z_top, n_sub)

    z_flat = z_arr.reshape(nz, columns)
    u_fine = np.empty((n_sub, columns), dtype=np.float64)
    v_fine = np.empty((n_sub, columns), dtype=np.float64)
    for m in range(columns):
        order = np.argsort(z_flat[:, m], kind="stable")
        zc = z_flat[order, m]
        u_fine[:, m] = np.interp(z_fine, zc, u_arr.reshape(nz, columns)[order, m])
        v_fine[:, m] = np.interp(z_fine, zc, v_arr.reshape(nz, columns)[order, m])

    dz = float(z_fine[1] - z_fine[0])
    du = np.diff(u_fine, axis=0) / dz          # (n_sub-1, columns)
    dv = np.diff(v_fine, axis=0) / dz
    u_mid = 0.5 * (u_fine[:-1] + u_fine[1:])
    v_mid = 0.5 * (v_fine[:-1] + v_fine[1:])
    integrand = (u_mid - storm_u) * dv - (v_mid - storm_v) * du
    srh_flat = dz * np.sum(integrand, axis=0)
    positive_flat = dz * np.sum(np.maximum(integrand, 0.0), axis=0)
    srh = srh_flat.reshape(horizontal) if horizontal else srh_flat.reshape(())
    positive = positive_flat.reshape(horizontal) if horizontal else positive_flat.reshape(())
    return {"srh": np.asarray(srh, dtype=np.float64),
            "srh_positive": np.asarray(positive, dtype=np.float64)}

def precipitation_accumulation(rate: ArrayLike, dt: float | ArrayLike, *,
                               units: str = "mm") -> NDArray[np.float64]:
    """降水累积 :math:`P = \\sum_n R_n\\Delta t_n`。

    参数
    ----
    rate : array_like, shape (nt, ...) 或 (...)
        降水率 [kg m^-2 s^-1]，与水深率 mm/s 数值等价。
    dt : float 或 array_like, shape (nt,)
        时间间隔 [s]；标量时对所有时间段使用同一值。
    units : {"mm", "m", "kg/m2"}
        输出单位。mm 与 kg/m^2 数值相同（:math:`\\rho_w = 1000\\,\\mathrm{kg/m^3}`）。

    离散化：矩形法。对模式输出的"区间平均率"是**精确**的；若 `rate` 是瞬时值，
    则为一阶近似。复杂度：O(N)。
    """
    rate_arr = np.asarray(rate, dtype=np.float64)
    dt_arr = np.asarray(dt, dtype=np.float64)
    if units not in ("mm", "m", "kg/m2"):
        raise ValueError("units 只能是 'mm' / 'm' / 'kg/m2'")
    if rate_arr.ndim == 0:
        accum = rate_arr * (float(dt_arr.ravel()[0]) if dt_arr.size else 0.0)
    elif dt_arr.ndim >= 1 and dt_arr.size == rate_arr.shape[0]:
        accum = np.sum(rate_arr * dt_arr.reshape((-1,) + (1,) * (rate_arr.ndim - 1)), axis=0)
    elif dt_arr.size == 1 or dt_arr.ndim == 0:
        accum = np.sum(rate_arr, axis=0) * float(dt_arr.ravel()[0])
    else:
        accum = rate_arr * dt_arr
    scale = 1.0 if units in ("mm", "kg/m2") else 1.0 / RHO_WATER
    return accum * scale


def sea_level_pressure(p_sfc: ArrayLike, t_sfc: ArrayLike,
                       z_sfc: ArrayLike, *, gamma: float = 0.0065,
                       g: float = GRAVITY, rd: float = RD) -> NDArray[np.float64]:
    """海平面气压（静力外推）。

    公式（[D16] WRF ARW 第 3 章）
    ----------------------------
    .. math::

        p_{SL}=p_{sfc}\\left[1+\\frac{\\Gamma\\,z_{sfc}}{T_{sfc}}\\right]^{g/(R_d\\Gamma)}

    即静力方程在**等温递减率** :math:`\\Gamma=6.5\\ \\mathrm{K/km}` 下的解析解。
    离散化：逐点代数式；:math:`T_{sfc}+\\Gamma z_{sfc}\\le 0`（非物理）返回 NaN。
    复杂度：O(N)。
    """
    p_arr, t_arr, z_arr = _as(p_sfc, t_sfc, z_sfc)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = 1.0 + gamma * z_arr / np.maximum(t_arr, 1.0)
        result = p_arr * np.power(np.maximum(factor, 1.0e-6), g / (rd * gamma))
    return np.where((t_arr + gamma * z_arr) > 0.0, result, np.nan)


def reflectivity_dbz(qr: ArrayLike, qs: ArrayLike | None = None,
                     qg: ArrayLike | None = None, rho: ArrayLike | None = None,
                     t: ArrayLike | None = None, *,
                     qr_scale: float = 1.0) -> NDArray[np.float64]:
    """雷达反射率因子 [dBZ]（简化 Marshall-Palmer 形式）。

    公式（[O6] Sun & Crook (1997)；[P3] Thompson et al. (2008) 附录）
    --------------------------------------------------------------
    .. math::

        Z = \\frac{720\\times10^{18}}{\\pi\\rho_w^{1.75}N_0^{0.75}}
            (\\rho q_r)^{1.75}
          \\;\\approx\\;
            3.63\\times10^{9}\\,(\\rho q_r)^{1.75}\\ \\mathrm{mm^6/m^3}

    .. math::

        Z_{dBZ} = 10\\log_{10}\\max(Z, Z_{\\min}), \\qquad
        Z_{\\min}=10^{-4}\\ \\mathrm{mm^6/m^3}

    雪、霰按同样幂律累加，冰相乘以 :math:`(\\rho_w/\\rho_i)^{1.75}\\approx1.24`
    的密度修正（[P3]）。

    离散化：逐点代数式，下限避免 :math:`-\\infty` dBZ。
    复杂度：O(N)。
    """
    qr_arr = np.asarray(qr, dtype=np.float64)
    rho_arr = (np.ones_like(qr_arr) if rho is None
               else np.broadcast_to(np.asarray(rho, dtype=np.float64), qr_arr.shape))
    mass = np.maximum(rho_arr * np.maximum(qr_arr, 0.0) * qr_scale, 0.0)
    z_linear = _Z_COEFF * np.power(mass, 1.75)
    for species, factor in ((qs, 1.24), (qg, 1.0)):
        if species is not None:
            mass_ice = np.maximum(rho_arr * np.maximum(np.asarray(species, dtype=np.float64), 0.0), 0.0)
            z_linear = z_linear + factor * _Z_COEFF * np.power(mass_ice, 1.75)
    dbz = 10.0 * np.log10(np.maximum(z_linear, 1.0e-4))
    if t is not None:
        dbz = np.where(np.asarray(t, dtype=np.float64) < T0, np.minimum(dbz, 60.0), dbz)
    return np.asarray(dbz, dtype=np.float64)


def brightness_temperature(flux: ArrayLike, *, emissivity: float = 1.0,
                           band: str = "thermal") -> NDArray[np.float64]:
    """亮度温度（简化灰体近似）。

    公式（Stefan-Boltzmann）
    -----------------------
    .. math::

        T_b=\\left(\\frac{F_{up}}{\\varepsilon\\,\\sigma}\\right)^{1/4}

    `band="solar"` 时把 `flux` 解释为反射短波通量、`emissivity` 解释为反照率
    :math:`\\alpha`：:math:`T_b=((1-\\alpha)F/\\sigma)^{1/4}`。

    离散化：逐点代数式；负通量返回 NaN。复杂度：O(N)。
    """
    if emissivity <= 0.0:
        raise ValueError("emissivity 必须为正")
    flux_arr = np.asarray(flux, dtype=np.float64)
    if band == "solar":
        effective = np.maximum((1.0 - emissivity) * flux_arr, 0.0)
        valid = flux_arr >= 0.0
    elif band == "thermal":
        effective = np.maximum(flux_arr, 0.0) / emissivity
        valid = flux_arr >= 0.0
    else:
        raise ValueError("band 只能是 'thermal' 或 'solar'")
    t_b = np.power(effective / STEFAN_BOLTZMANN, 0.25)
    return np.where(valid, t_b, np.nan)


def k_index(t850: ArrayLike, t500: ArrayLike, td850: ArrayLike,
            t700: ArrayLike, td700: ArrayLike) -> NDArray[np.float64]:
    """K 指数 [K]。

    公式（[B4] 第 3 章；[E19] WMO 检验手册）
    --------------------------------------
    .. math::

        K=(T_{850}-T_{500})+T_{d,850}-(T_{700}-T_{d,700})

    所有温度先转换为摄氏度；经验阈值 30 K 以上指示中等以上对流潜势。
    离散化：逐点代数式，无积分误差。复杂度：O(N)。
    """
    t850_arr, t500_arr, td850_arr, t700_arr, td700_arr = _as(t850, t500, td850, t700, td700)
    return ((celsius(t850_arr) - celsius(t500_arr))
            + celsius(td850_arr)
            - (celsius(t700_arr) - celsius(td700_arr)))

def showalter_index(t850: ArrayLike, td850: ArrayLike, t500: ArrayLike, *,
                    p_low: float = 85000.0, p_high: float = 50000.0,
                    n_iter: int = 40) -> NDArray[np.float64]:
    """Showalter 指数 [K]。

    公式（[B4] 第 3 章）
    ------------------
    把 850 hPa 的气块按干绝热抬升到抬升凝结高度（LCL），再沿伪湿绝热（守恒
    :math:`\\theta_e`）抬升到 500 hPa，得到气块温度
    :math:`T_{parcel,500}`：

    .. math::

        SI = T_{500} - T_{parcel,500}

    :math:`SI \\le -3\\ \\mathrm{K}` 指示强对流潜势；:math:`SI>3` 稳定。
    实现复用 :func:`_moist_temperature` 的湿绝热反解，与 CAPE 完全一致。

    离散化：干绝热段用解析式 :math:`T\\propto p^{\\kappa}`；湿绝热段用
    Newton 迭代反解 :math:`\\theta_e`（:func:`_moist_temperature`），
    迭代误差 < :math:`10^{-4}` K。

    复杂度：O(n_iter × N)。
    """
    t850_arr, td850_arr, t500_arr = _as(t850, td850, t500)
    es850 = saturation_vapor_pressure(td850_arr)
    p_arr = np.full_like(t850_arr, float(p_low))
    qv850 = mixing_ratio(p_arr, es850)
    theta850 = t850_arr * (P0 / p_low) ** KAPPA
    t_lcl = _lcl_temperature(t850_arr, qv850, p_arr)
    p_lcl = np.maximum(float(p_low) * (t_lcl / np.maximum(t850_arr, 1.0)) ** (1.0 / KAPPA), 1.0)
    theta_e_lcl = theta_e(p_lcl, t_lcl, saturation_mixing_ratio(p_lcl, t_lcl))
    t_dry_high = _dry_lapse(theta850, p_high)
    parcel_500 = _moist_temperature(
        np.full_like(theta850, float(p_high)), theta_e_lcl, t_dry_high, n_iter=n_iter)
    return t500_arr - parcel_500


def list_diagnostics() -> Mapping[str, str]:
    """可用诊断量名 -> 一句话说明（供 CLI `derive --list`）。复杂度 O(1)。"""
    return {
        "wind_speed": "风速 (m/s)",
        "wind_direction": "风向 (deg, 来向)",
        "dewpoint": "露点 (K)",
        "relative_humidity": "相对湿度 (0-1)",
        "mixing_ratio": "水汽混合比 (kg/kg)",
        "saturation_mixing_ratio": "饱和混合比 (kg/kg)",
        "theta_e": "相当位温 (K)",
        "pressure_from_exner": "气压 (Pa)",
        "temperature_from_exner": "温度 (K)",
        "geopotential": "位势 (m^2/s^2)",
        "vorticity": "相对涡度 (1/s)",
        "divergence": "水平散度 (1/s)",
        "absolute_vorticity": "绝对涡度 (1/s)",
        "potential_vorticity": "Ertel 位涡 (K m^2 kg^-1 s^-1)",
        "cape_cin": "CAPE/CIN/LFC/EL/LCL 气压",
        "storm_relative_helicity": "0-3 km 风暴相对螺旋度 (m^2/s^2)",
        "precipitation_accumulation": "降水累积 (mm)",
        "sea_level_pressure": "海平面气压 (Pa)",
        "reflectivity_dbz": "雷达反射率 (dBZ)",
        "brightness_temperature": "亮度温度 (K)",
        "k_index": "K 指数 (K)",
        "showalter_index": "Showalter 指数 (K)",
    }

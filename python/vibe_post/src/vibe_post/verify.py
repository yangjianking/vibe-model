"""预报检验评分（纯 numpy，与 C++ `include/vibe/verify/scores.hpp` 公式一致）。

四类检验（docs/design/00_architecture.md 第 9 节）
------------------------------------------------
1. **连续量**：bias、mae、rmse、correlation、anomaly_correlation；
2. **分类量**：contingency_table、pod、far、csi、ets、tss、frequency_bias、orss；
3. **概率量**：brier_score、brier_skill_score、reliability_curve、roc_curve/roc_auc、
   crps_ensemble、rank_histogram、spread_skill_ratio；
4. **邻域/空间**：fractions_skill_score / fss。

通用约定
--------
* 所有函数对 `NaN` **成对剔除**（forecast 或 obs 任一为 NaN 即丢弃该样本），
  有效样本数为 0 时返回 `NaN`（而不是抛异常），便于批量检验时稳健聚合；
* 零方差（常数预报/观测）时相关系数返回 `NaN`（数学上未定义），
  而不是 0；
* 数组可任意维；需要网格形状的函数显式要求二维或三维；
* 所有函数无状态、无副作用，便于与 C++ 同名函数逐值比对。

文献：[B9] Wilks (2019)；[B10] Jolliffe & Stephenson (2012)；[E3] Brier (1950)；
[E4] Gandin & Murphy (1992)；[E6] Schaefer (1990)；[E7] Roberts & Lean (2008)；
[E9] Hamill (2001)；[E11] Gneiting & Raftery (2007)；[E12] Hersbach (2000)；
[E14] Mittermaier & Roberts (2010)；[E15] Mason & Graham (2002)；
[E16] Ferro et al. (2008)；[E18] Stanski et al. (1989)；[E19] WMO (2012)。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "bias", "mean_error", "mae", "rmse", "correlation", "anomaly_correlation",
    "contingency_table", "pod", "far", "csi", "ets", "tss", "frequency_bias",
    "orss", "brier_score", "brier_skill_score", "reliability_curve",
    "roc_curve", "roc_auc", "crps_ensemble", "rank_histogram",
    "spread_skill_ratio", "fractions_skill_score", "fss", "fss_vs_scale",
    "neighborhood_fractions", "scores_table",
]


def _nan() -> float:
    """标量 NaN。复杂度 O(1)。"""
    return float("nan")


def valid_pairs(f: ArrayLike, o: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """成对剔除 NaN/Inf，返回等长的一维有效样本。复杂度 O(N)。"""
    f_arr = np.asarray(f, dtype=np.float64).ravel()
    o_arr = np.asarray(o, dtype=np.float64).ravel()
    if f_arr.size != o_arr.size:
        try:
            f_arr, o_arr = np.broadcast_arrays(f_arr, o_arr)
        except ValueError as exc:
            raise ValueError("forecast 与 observation 形状不匹配：{0} vs {1}".format(
                f_arr.shape, o_arr.shape)) from exc
    mask = np.isfinite(f_arr) & np.isfinite(o_arr)
    return f_arr[mask], o_arr[mask]


def _safe_div(numerator: float, denominator: float) -> float:
    """安全除法：分母为 0 时返回 NaN。复杂度 O(1)。"""
    if denominator == 0.0:
        return _nan()
    return float(numerator) / float(denominator)


# ---------------------------------------------------------------------------
# 连续量评分
# ---------------------------------------------------------------------------


def bias(f: ArrayLike, o: ArrayLike) -> float:
    """平均偏差（mean error）。

    公式（[B9] 第 8 章）
    ------------------
    .. math::

        \\mathrm{bias} = \\frac{1}{N}\\sum_{i=1}^{N}(f_i-o_i)

    正值表示预报系统偏大。离散化：算术平均（一阶），复杂度 O(N)。
    """
    fv, ov = valid_pairs(f, o)
    if fv.size == 0:
        return _nan()
    return float(np.mean(fv - ov))


def mean_error(f: ArrayLike, o: ArrayLike) -> float:
    """`bias` 的别名（平均误差 ME）。复杂度 O(N)。"""
    return bias(f, o)


def mae(f: ArrayLike, o: ArrayLike) -> float:
    """平均绝对误差 MAE。

    公式
    ----
    .. math::

        \\mathrm{MAE}=\\frac{1}{N}\\sum_i |f_i-o_i|

    对离群值不敏感（相比 RMSE），是 L1 损失的经验估计。复杂度 O(N)。
    """
    fv, ov = valid_pairs(f, o)
    if fv.size == 0:
        return _nan()
    return float(np.mean(np.abs(fv - ov)))


def rmse(f: ArrayLike, o: ArrayLike) -> float:
    """均方根误差 RMSE。

    公式
    ----
    .. math::

        \\mathrm{RMSE}=\\sqrt{\\frac{1}{N}\\sum_i (f_i-o_i)^2}

    对应集合平均意义上误差标准差的无偏估计（高斯假设下等于误差标准差）。
    离散化：先平方和、再开方，避免中间溢出。复杂度 O(N)。
    """
    fv, ov = valid_pairs(f, o)
    if fv.size == 0:
        return _nan()
    return float(np.sqrt(np.mean((fv - ov) ** 2)))


def correlation(f: ArrayLike, o: ArrayLike) -> float:
    """Pearson 相关系数。

    公式（[B9] 第 3 章）
    ------------------
    .. math::

        r = \\frac{\\sum_i (f_i-\\bar f)(o_i-\\bar o)}
        {\\sqrt{\\sum_i (f_i-\\bar f)^2}\\sqrt{\\sum_i (o_i-\\bar o)^2}}

    任一序列方差为 0（包括 N<2 或常量场）时返回 NaN。
    复杂度：O(N)（两趟均值 + 一趟求和）。
    """
    fv, ov = valid_pairs(f, o)
    if fv.size < 2:
        return _nan()
    fm = fv - fv.mean()
    om = ov - ov.mean()
    denominator = float(np.sqrt(np.sum(fm ** 2) * np.sum(om ** 2)))
    # 与 C++ verify::pearson_correlation 一致：常数序列（方差为 0）返回 0.0
    return 0.0 if denominator == 0.0 else float(np.sum(fm * om)) / denominator


def anomaly_correlation(f: ArrayLike, o: ArrayLike, clim: ArrayLike) -> float:
    """距平相关系数 ACC。

    公式（[E19] WMO 检验手册；[B9] 第 8 章）
    --------------------------------------
    .. math::

        ACC=\\frac{\\sum_i (f_i-c_i)(o_i-c_i)}
        {\\sqrt{\\sum_i (f_i-c_i)^2}\\sqrt{\\sum_i (o_i-c_i)^2}}

    其中 :math:`c_i` 为气候态（逐点，可以是标量或与样本同形）。
    ACC 对系统性偏差不敏感，是业务中期预报的核心指标。复杂度 O(N)。
    """
    f_arr = np.asarray(f, dtype=np.float64).ravel()
    o_arr = np.asarray(o, dtype=np.float64).ravel()
    c_arr = np.broadcast_to(np.asarray(clim, dtype=np.float64), f_arr.shape).ravel()
    mask = np.isfinite(f_arr) & np.isfinite(o_arr) & np.isfinite(c_arr)
    fa = f_arr[mask] - c_arr[mask]
    oa = o_arr[mask] - c_arr[mask]
    if fa.size < 2:
        return _nan()
    denominator = float(np.sqrt(np.sum(fa ** 2) * np.sum(oa ** 2)))
    return _safe_div(float(np.sum(fa * oa)), denominator)


# ---------------------------------------------------------------------------
# 分类量评分（列联表）
# ---------------------------------------------------------------------------


def contingency_table(f: ArrayLike, o: ArrayLike, threshold: float) -> dict[str, float]:
    """二分类列联表（相对阈值）。

    定义（[E18] Stanski et al. 1989；[E19] WMO 2012）
    -----------------------------------------------
    * `hits`（H）：:math:`f\\ge T` 且 :math:`o\\ge T`；
    * `misses`（M）：:math:`f<T` 且 :math:`o\\ge T`；
    * `false_alarms`（F）：:math:`f\\ge T` 且 :math:`o<T`；
    * `correct_negatives`（C）：:math:`f<T` 且 :math:`o<T`。

    离散化：逐样本计数（0/1 掩膜求和），复杂度 O(N)。
    """
    fv, ov = valid_pairs(f, o)
    if fv.size == 0:
        return {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0,
                "correct_negatives": 0.0, "n": 0.0}
    fmask = fv >= threshold
    omask = ov >= threshold
    hits = float(np.sum(fmask & omask))
    misses = float(np.sum((~fmask) & omask))
    false_alarms = float(np.sum(fmask & (~omask)))
    correct = float(np.sum((~fmask) & (~omask)))
    return {"hits": hits, "misses": misses, "false_alarms": false_alarms,
            "correct_negatives": correct, "n": float(fv.size)}


def _table(f: ArrayLike, o: ArrayLike, threshold: float) -> dict[str, float]:
    """内部用：允许直接传入已算好的列联表。复杂度 O(N)。"""
    if isinstance(f, Mapping):
        return dict(f)
    return contingency_table(f, o, threshold)


def pod(f: ArrayLike, o: ArrayLike, threshold: float | None = None) -> float:
    """命中率 POD（probability of detection / recall）。

    公式（[E18]）
    ------------
    .. math::

        \\mathrm{POD}=\\frac{H}{H+M}

    完全漏报时为 0；无观测事件时为 NaN。复杂度 O(N)。
    """
    table = _table(f, o, 0.0 if threshold is None else threshold)
    return _safe_div(table["hits"], table["hits"] + table["misses"])


def far(f: ArrayLike, o: ArrayLike, threshold: float | None = None) -> float:
    """空报率 FAR。

    公式
    ----
    .. math::

        \\mathrm{FAR}=\\frac{F}{H+F}

    无预报事件时为 NaN。复杂度 O(N)。
    """
    table = _table(f, o, 0.0 if threshold is None else threshold)
    return _safe_div(table["false_alarms"], table["hits"] + table["false_alarms"])


def csi(f: ArrayLike, o: ArrayLike, threshold: float | None = None) -> float:
    """临界成功指数 CSI（Threat Score）。

    公式（[E6] Schaefer (1990)）
    --------------------------
    .. math::

        \\mathrm{CSI}=\\frac{H}{H+M+F}

    相比 POD 兼顾了空报，是强对流检验最常用的评分之一。复杂度 O(N)。
    """
    table = _table(f, o, 0.0 if threshold is None else threshold)
    return _safe_div(table["hits"], table["hits"] + table["misses"] + table["false_alarms"])


def ets(f: ArrayLike, o: ArrayLike, threshold: float | None = None) -> float:
    """公平技巧评分 ETS（Gilbert skill score）。

    公式（[E4] Gandin & Murphy (1992)；[E19]）
    -----------------------------------------
    随机命中期望

    .. math::

        H_{random}=\\frac{(H+M)(H+F)}{N}

    .. math::

        \\mathrm{ETS}=\\frac{H-H_{random}}
        {H+M+F-H_{random}}

    ETS 对随机预报的期望为 0，取值 :math:`(-1/3, 1]`。复杂度 O(N)。
    """
    table = _table(f, o, 0.0 if threshold is None else threshold)
    total = table["n"]
    if total == 0:
        return _nan()
    hits, misses, false_alarms = table["hits"], table["misses"], table["false_alarms"]
    expected = (hits + misses) * (hits + false_alarms) / total
    return _safe_div(hits - expected, hits + misses + false_alarms - expected)


def tss(f: ArrayLike, o: ArrayLike, threshold: float | None = None) -> float:
    """True Skill Statistic（Peirce skill score / Hanssen-Kuipers）。

    公式（[E5] Hanssen & Kuipers (1965)）
    -----------------------------------
    .. math::

        \\mathrm{TSS}=\\mathrm{POD}-\\mathrm{POFD}
        =\\frac{H}{H+M}-\\frac{F}{F+C}

    复杂度：O(N)。
    """
    table = _table(f, o, 0.0 if threshold is None else threshold)
    sensitivity = _safe_div(table["hits"], table["hits"] + table["misses"])
    false_rate = _safe_div(table["false_alarms"],
                           table["false_alarms"] + table["correct_negatives"])
    if np.isnan(sensitivity) or np.isnan(false_rate):
        return _nan()
    return sensitivity - false_rate


def frequency_bias(f: ArrayLike, o: ArrayLike, threshold: float | None = None) -> float:
    """频率偏差（bias score）。

    公式
    ----
    .. math::

        B=\\frac{H+F}{H+M}

    :math:`B=1` 表示事件频率匹配；:math:`B>1` 过度预报。复杂度 O(N)。
    """
    table = _table(f, o, 0.0 if threshold is None else threshold)
    return _safe_div(table["hits"] + table["false_alarms"], table["hits"] + table["misses"])


def orss(f: ArrayLike, o: ArrayLike, threshold: float | None = None) -> float:
    """Odds Ratio Skill Score（Yule's Q）。

    公式（[E4]；[B10] 第 3 章）
    --------------------------
    .. math::

        \\mathrm{ORSS}=\\frac{H\\,C-F\\,M}{H\\,C+F\\,M}

    相比 TSS 更"公平"（对基础率不敏感），取值 :math:`[-1,1]`。复杂度 O(N)。
    """
    table = _table(f, o, 0.0 if threshold is None else threshold)
    hits = table["hits"]
    correct = table["correct_negatives"]
    false_alarms = table["false_alarms"]
    misses = table["misses"]
    numerator = hits * correct - false_alarms * misses
    denominator = hits * correct + false_alarms * misses
    return _safe_div(numerator, denominator)


def scores_table(f: ArrayLike, o: ArrayLike, threshold: float) -> dict[str, float]:
    """一次给出全部分类评分。复杂度 O(N)。"""
    table = contingency_table(f, o, threshold)
    return {
        "n": table["n"],
        "hits": table["hits"],
        "misses": table["misses"],
        "false_alarms": table["false_alarms"],
        "correct_negatives": table["correct_negatives"],
        "pod": pod(table, None, None),
        "far": far(table, None, None),
        "csi": csi(table, None, None),
        "ets": ets(table, None, None),
        "tss": tss(table, None, None),
        "frequency_bias": frequency_bias(table, None, None),
        "orss": orss(table, None, None),
    }


# ---------------------------------------------------------------------------
# 概率量评分
# ---------------------------------------------------------------------------


def brier_score(p: ArrayLike, o: ArrayLike, *, threshold: float | None = None) -> float:
    """Brier 评分 BS（概率预报的均方误差）。

    公式（[E3] Brier (1950)；[E11] Gneiting & Raftery (2007)）
    ---------------------------------------------------------
    .. math::

        \\mathrm{BS}=\\frac{1}{N}\\sum_i (p_i-o_i)^2

    其中 :math:`p_i` 为事件发生的预报概率，:math:`o_i\\in\\{0,1\\}` 为观测
    是否发生。`threshold` 给出时先把非二值的 `o` 二值化，便于直接用连续
    观测（如降水）检验。离散化：算术平均（一阶）。复杂度：O(N)。
    """
    o_arr = np.asarray(o, dtype=np.float64)
    if threshold is not None:
        o_arr = (o_arr >= threshold).astype(np.float64)
    pv, ov = valid_pairs(p, o_arr)
    if pv.size == 0:
        return _nan()
    return float(np.mean((pv - ov) ** 2))


def brier_skill_score(p: ArrayLike, o: ArrayLike, p_clim: float | ArrayLike = 0.5) -> float:
    """Brier 技巧评分 BSS（相对气候态概率）。

    公式（[E11]）
    ------------
    .. math::

        \\mathrm{BSS}=1-\\frac{\\mathrm{BS}}{\\mathrm{BS}_{clim}},
        \\qquad \\mathrm{BS}_{clim}=p_c(1-p_c)

    :math:`p_c` 由 `p_clim` 给出（标量或逐点气候概率场）。`BSS>0` 表示优于
    气候态。复杂度：O(N)。
    """
    o_arr = np.asarray(o, dtype=np.float64)
    pv, ov = valid_pairs(p, o_arr)
    if pv.size == 0:
        return _nan()
    bs = float(np.mean((pv - ov) ** 2))
    clim = np.asarray(p_clim, dtype=np.float64)
    pc = float(clim) if clim.ndim == 0 else float(np.mean(clim))
    reference = pc * (1.0 - pc)
    if reference <= 0.0:
        return _nan()
    return 1.0 - bs / reference


def reliability_curve(p: ArrayLike, o: ArrayLike, *, n_bins: int = 10,
                      bins: ArrayLike | None = None) -> dict[str, NDArray[np.float64]]:
    """可靠性曲线（预报概率 -> 观测频率）与频率直方图。

    公式（[B9] 第 8 章；[E2] Murphy & Winkler (1987)）
    -------------------------------------------------
    把 :math:`[0,1]` 均分为 `n_bins` 个区间，对每个区间取

    .. math::

        \\bar p_k=\\frac{1}{n_k}\\sum_{i\\in k}p_i, \\qquad
        \\bar o_k=\\frac{1}{n_k}\\sum_{i\\in k}o_i

    返回 `{"bins","p_mean","reliability","counts"}`；空 bin 填 NaN 以便绘图断开。
    复杂度：O(N + n_bins)。
    """
    pv, ov = valid_pairs(p, o)
    if bins is None:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    else:
        edges = np.asarray(bins, dtype=np.float64).ravel()
        if edges.size < 2:
            raise ValueError("bins 至少需要 2 个边界")
    if pv.size == 0:
        empty = np.full(edges.size - 1, np.nan)
        return {"bins": edges, "p_mean": empty, "reliability": empty,
                "counts": np.zeros(edges.size - 1)}
    index = np.clip(np.digitize(pv, edges[1:-1], right=False), 0, edges.size - 2)
    counts = np.zeros(edges.size - 1, dtype=np.float64)
    p_mean = np.full(edges.size - 1, np.nan)
    o_mean = np.full(edges.size - 1, np.nan)
    for k in range(edges.size - 1):
        mask = index == k
        count = int(mask.sum())
        counts[k] = count
        if count > 0:
            p_mean[k] = float(pv[mask].mean())
            o_mean[k] = float(ov[mask].mean())
    return {"bins": edges, "p_mean": p_mean, "reliability": o_mean, "counts": counts}


def roc_curve(p: ArrayLike, o: ArrayLike, *,
              thresholds: ArrayLike | None = None) -> dict[str, NDArray[np.float64]]:
    """ROC 曲线（命中率对空报率）。

    公式（[E15] Mason & Graham (2002)；[B9] 第 8 章）
    ------------------------------------------------
    对阈值 :math:`\\tau`：:math:`\\mathrm{POD}(\\tau)` 对
    :math:`\\mathrm{POFD}(\\tau)` 作图；端点
    :math:`(0,0)`（:math:`\\tau=\\infty`）与 :math:`(1,1)`（:math:`\\tau=-\\infty`）
    保证曲线闭合。

    实现用排序法，复杂度 O(N log N)。
    """
    pv, ov = valid_pairs(p, o)
    if pv.size == 0:
        return {"thresholds": np.zeros(0), "pod": np.zeros(0), "pofd": np.zeros(0)}
    order = np.argsort(-pv, kind="stable")
    p_sorted = pv[order]
    o_sorted = ov[order]
    positives = float((o_sorted > 0.5).sum())
    negatives = float(o_sorted.size - positives)
    distinct = np.unique(p_sorted)[::-1]
    pod_vals = np.empty(distinct.size, dtype=np.float64)
    pofd_vals = np.empty(distinct.size, dtype=np.float64)
    cumulative_pos = 0.0
    cumulative_neg = 0.0
    cursor = 0
    for k, tau in enumerate(distinct):
        while cursor < p_sorted.size and p_sorted[cursor] >= tau:
            if o_sorted[cursor] > 0.5:
                cumulative_pos += 1.0
            else:
                cumulative_neg += 1.0
            cursor += 1
        pod_vals[k] = cumulative_pos / positives if positives > 0 else np.nan
        pofd_vals[k] = cumulative_neg / negatives if negatives > 0 else np.nan
    full_thresholds = np.concatenate(([np.inf], distinct, [-np.inf]))
    pod_full = np.concatenate(([0.0], pod_vals, [1.0 if positives > 0 else np.nan]))
    pofd_full = np.concatenate(([0.0], pofd_vals, [1.0 if negatives > 0 else np.nan]))
    if thresholds is not None:
        wanted = np.asarray(thresholds, dtype=np.float64)
        return {"thresholds": wanted,
                "pod": np.interp(wanted, full_thresholds[::-1], pod_full[::-1]),
                "pofd": np.interp(wanted, full_thresholds[::-1], pofd_full[::-1])}
    return {"thresholds": full_thresholds, "pod": pod_full, "pofd": pofd_full}


def roc_auc(p: ArrayLike, o: ArrayLike) -> float:
    """ROC 曲线下面积 AUC（等价于 Mann-Whitney U 统计量）。

    公式（[E15]）
    ------------
    .. math::

        \\mathrm{AUC}=\\frac{1}{N_+N_-}\\sum_{i\\in+}\\sum_{j\\in-}
        \\left[\\mathbb 1(p_i>p_j)+\\tfrac12\\mathbb 1(p_i=p_j)\\right]

    与阈值无关，衡量概率预报的**分辨能力**；0.5 表示无技巧。并列用平均秩处理。
    实现用秩和公式，复杂度 O(N log N)。
    """
    pv, ov = valid_pairs(p, o)
    positives = pv[ov > 0.5]
    negatives = pv[ov <= 0.5]
    if positives.size == 0 or negatives.size == 0:
        return _nan()
    combined = np.concatenate((positives, negatives))
    order = np.argsort(combined, kind="stable")
    sorted_values = combined[order]
    ranks = np.empty(combined.size, dtype=np.float64)
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    rank_sum = float(ranks[:positives.size].sum())
    n_pos = float(positives.size)
    n_neg = float(negatives.size)
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def crps_ensemble(ensemble: ArrayLike, obs: ArrayLike, *, fair: bool = True
                  ) -> NDArray[np.float64]:
    """集合连续排序概率评分 CRPS。

    公式（[E12] Hersbach (2000)；[E16] Ferro et al. (2008)）
    -------------------------------------------------------
    令升序成员 x_(1) <= ... <= x_(M)、观测 y：

    .. math::

        CRPS = \\frac{1}{M}\\sum_{m=1}^{M}|x_m-y|
               - \\frac{1}{2M(M-1)}\\sum_{m=1}^{M}\\sum_{n=1}^{M}|x_m-x_n|

    第一项是集合的平均绝对误差，第二项是集合的**离散度惩罚**：确定性集合
    （所有成员相同）退化为 MAE。`fair=True` 用 :math:`M(M-1)` 的公平分母（[E16]），
    `fair=False` 用有偏的 :math:`M^2` 分母。

    `ensemble` 形状 `(M, ...)`、`obs` 形状 `(...)`，返回 `(...)`。
    复杂度：O(M^2 x N_h)（成对距离）；内存 O(M x N_h)。
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    obs_arr = np.asarray(obs, dtype=np.float64)
    if ens.ndim == 0:
        raise ValueError("ensemble 至少需要两维 (M, ...)")
    members = ens.shape[0]
    if members < 2:
        raise ValueError("CRPS 至少需要 2 个集合成员，得到 {0}".format(members))
    obs_b = np.broadcast_to(obs_arr, ens.shape[1:])
    valid = np.isfinite(ens) & np.isfinite(obs_b)[None, ...]
    counts = valid.sum(axis=0).astype(np.float64)
    term1 = np.nansum(np.abs(ens - obs_b[None, ...]), axis=0) / np.maximum(counts, 1.0)
    # 成对绝对差（含两次计数与对角零元）
    pairwise = np.abs(ens[:, None, ...] - ens[None, :, ...])
    pairwise = np.where(np.isfinite(pairwise), pairwise, 0.0)
    pair_total = pairwise.sum(axis=(0, 1))
    if fair:
        divisor = counts * (counts - 1.0)
    else:
        divisor = counts ** 2
    spread = pair_total / np.maximum(divisor, 1.0)
    return np.where(counts >= 2, term1 - 0.5 * spread, np.nan)

def rank_histogram(ensemble: ArrayLike, obs: ArrayLike, *,
                   n_bins: int | None = None) -> dict[str, NDArray[np.float64]]:
    """Talagrand 排序直方图。

    公式（[E9] Hamill (2001)；[E10] Talagrand & Vautard (1997)）
    ------------------------------------------------------------
    对每个样本把观测插入升序集合得到秩

    .. math::

        r_i=\\#\\{m: x_{(m)}<y_i\\}\\in\\{0,\\dots,M\\}

    统计各秩频数；平坦直方图表示集合离散度可靠，U 形表示欠离散、L 形表示过离散。
    离散化：逐样本按"严格小于"计数，与 C++ `rank_histogram` 一致。
    复杂度：O(M log M × N)；返回 `{"rank","counts","expected"}`。
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    obs_arr = np.asarray(obs, dtype=np.float64)
    if ens.ndim < 2:
        raise ValueError("ensemble 至少需要两维 (M, ...)")
    members = ens.shape[0]
    bins = int(n_bins if n_bins is not None else members + 1)
    if bins < 2:
        raise ValueError("n_bins 至少为 2")
    obs_b = np.broadcast_to(obs_arr, ens.shape[1:])
    counts = np.zeros(bins, dtype=np.float64)
    total = 0
    flat_ens = ens.reshape(members, -1)
    flat_obs = obs_b.reshape(-1)
    for m in range(flat_ens.shape[1]):
        column = flat_ens[:, m]
        target = flat_obs[m]
        column = column[np.isfinite(column)]
        if not np.isfinite(target) or column.size == 0:
            continue
        rank = int(np.clip(np.sum(column < target), 0, bins - 1))
        counts[rank] += 1.0
        total += 1
    expected = np.full(bins, float(total) / float(bins)) if bins else np.zeros(0)
    return {"rank": np.arange(bins, dtype=np.float64), "counts": counts,
            "expected": expected}


def spread_skill_ratio(ensemble: ArrayLike, obs: ArrayLike, *,
                       spread: str = "std", ddof: int = 0) -> float:
    """离散度-技巧比 SSR。

    公式（[B9] 第 8 章；[B10] 第 6 章）
    ----------------------------------
    .. math::

        \\mathrm{SSR}=\\frac{\\overline{\\sigma_{ens}}}
        {\\mathrm{RMSE}(\\bar x, y)}

    理想集合应满足 :math:`\\mathrm{SSR}\\approx1`。`spread="std"` 用成员标准差，
    `"range"` 用半极差（对粗集合更稳健）。复杂度：O(M × N)。
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    obs_arr = np.asarray(obs, dtype=np.float64)
    if ens.ndim < 2:
        raise ValueError("ensemble 至少需要两维 (M, ...)")
    valid = np.isfinite(ens)
    counts = valid.sum(axis=0).astype(np.float64)
    ensemble_mean = np.nansum(np.where(valid, ens, 0.0), axis=0) / np.maximum(counts, 1.0)
    if spread == "std":
        variance = np.nansum(np.where(valid, (ens - ensemble_mean[None, ...]) ** 2, 0.0),
                             axis=0) / np.maximum(counts - ddof, 1.0)
        spread_values = np.sqrt(np.maximum(variance, 0.0))
    elif spread == "range":
        max_values = np.nanmax(np.where(valid, ens, -np.inf), axis=0)
        min_values = np.nanmin(np.where(valid, ens, np.inf), axis=0)
        spread_values = 0.5 * (max_values - min_values)
    else:
        raise ValueError("spread 只能是 'std' 或 'range'")
    obs_b = np.broadcast_to(obs_arr, ens.shape[1:])
    mask = np.isfinite(ensemble_mean) & np.isfinite(obs_b) & np.isfinite(spread_values)
    if not np.any(mask):
        return _nan()
    skill = float(np.sqrt(np.mean((ensemble_mean[mask] - obs_b[mask]) ** 2)))
    return _safe_div(float(np.mean(spread_values[mask])), skill)


# ---------------------------------------------------------------------------
# 邻域（空间）检验：FSS
# ---------------------------------------------------------------------------


def neighborhood_fractions(field: ArrayLike, window: int) -> NDArray[np.float64]:
    """邻域内事件发生的比例。

    公式（[E7] Roberts & Lean (2008)）
    ---------------------------------
    .. math::

        F^{(w)}(i,j)=\\frac{1}{n_{ij}}\\sum_{p,q\\in\\mathcal N_w(i,j)}I(x_{p,q})

    其中 :math:`\\mathcal N_w` 为边长 `w`（奇数）的方形邻域、
    :math:`n_{ij}` 为域内实际参与统计的格点数（边界处小于 :math:`w^2`，
    避免零填充稀释边缘比例）。

    实现用**积分图**（summed-area table）：先累积和再取矩形差，
    复杂度 O(N)（与窗口大小无关）；内存 O(N)。
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("neighborhood_fractions 需要二维场，得到 {0} 维".format(arr.ndim))
    win = int(window)
    if win < 1 or win % 2 == 0:
        raise ValueError("window 必须是正奇数，得到 {0}".format(window))
    if win > min(arr.shape):
        raise ValueError("window={0} 超过网格尺寸 {1}".format(win, arr.shape))
    half = win // 2
    ny, nx = arr.shape
    padded = np.pad(arr, half, mode="constant", constant_values=0.0)
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    j = np.arange(ny)
    i = np.arange(nx)
    j1 = j + win
    i1 = i + win
    box = (integral[np.ix_(j1, i1)] - integral[np.ix_(j, i1)]
           - integral[np.ix_(j1, i)] + integral[np.ix_(j, i)])
    if half == 0:
        return box / float(win * win)
    jj = j[:, None]
    ii = i[None, :]
    count_j = (np.minimum(jj + half, ny - 1) - np.maximum(jj - half, 0) + 1).astype(np.float64)
    count_i = (np.minimum(ii + half, nx - 1) - np.maximum(ii - half, 0) + 1).astype(np.float64)
    return box / (count_j * count_i)


def fractions_skill_score(f: ArrayLike, o: ArrayLike, threshold: float,
                          window: int = 1) -> float:
    '''分数技巧评分 FSS（单一尺度）。

        公式（[E7] Roberts & Lean (2008) 式 (2)(3)；[E14] Mittermaier & Roberts (2010)）
        -------------------------------------------------------------------------
        .. math::

            \\mathrm{FBS}_w=\\sum_{i,j}\\left(F^{(w)}_f-F^{(w)}_o\\right)^2

        .. math::

            \\mathrm{FBS}_{ref}=\\sum_{i,j}\\left(F^{(w)}_f\\right)^2
            +\\sum_{i,j}\\left(F^{(w)}_o\\right)^2

        .. math::

            \\mathrm{FSS}=\\frac{2\\,\\mathrm{FBS}_w}
            {\\mathrm{FBS}_w+\\mathrm{FBS}_{ref}}
            =1-\\frac{\\mathrm{FBS}_w}
            {\\frac12\\mathrm{FBS}_{ref}}

        两种写法完全等价（第二种即 Roberts & Lean 的原始定义）；本实现与 C++
        verify::compute_fractional_skill 使用第一种。FSS 趋近 1 表示完美，
        FSS > 0.5 + f_o/2（见 fss_vs_scale）判为有用技巧尺度。

        复杂度：O(N)。
    '''

    f_arr = np.asarray(f, dtype=np.float64)
    o_arr = np.asarray(o, dtype=np.float64)
    if f_arr.shape != o_arr.shape:
        raise ValueError("forecast 与 observation 形状必须一致：{0} vs {1}".format(
            f_arr.shape, o_arr.shape))
    f_frac = neighborhood_fractions((f_arr >= threshold).astype(np.float64), window)
    o_frac = neighborhood_fractions((o_arr >= threshold).astype(np.float64), window)
    fbs = float(np.sum((f_frac - o_frac) ** 2))
    denominator = float(np.sum(f_frac ** 2)) + float(np.sum(o_frac ** 2))
    return _safe_div(denominator - fbs, denominator)  # FSS = 1 - FBS/FBS_ref（C++ 同式）

def fss(f: ArrayLike, o: ArrayLike, threshold: float, *,
        window: int = 1, radii: Sequence[int] | None = None,
        return_curve: bool = False) -> Any:
    """FSS 统一入口（单尺度或尺度序列）。

    `radii` 给出或 `return_curve=True` 时返回
    `{"radii", "window", "fss"}`，窗口取 :math:`w=2r+1`；
    否则返回单一尺度（`window`）的标量。复杂度：O(len(radii) × N)。
    """
    if radii is not None or return_curve:
        radius_list = [int(r) for r in (radii if radii is not None else [0, 1, 2, 4, 8])]
        windows = np.array([2 * r + 1 for r in radius_list], dtype=np.int64)
        values = np.array([fractions_skill_score(f, o, threshold, int(w)) for w in windows],
                          dtype=np.float64)
        return {"radii": np.asarray(radius_list, dtype=np.float64),
                "window": windows, "fss": values}
    return fractions_skill_score(f, o, threshold, window)


def fss_vs_scale(f: ArrayLike, o: ArrayLike, threshold: float,
                 radii: Sequence[int] = (0, 1, 2, 4, 8, 16)
                 ) -> dict[str, NDArray[np.float64]]:
    """FSS 随邻域尺度变化的曲线，并给出"有用技巧尺度"。

    判定准则（[E7] 式 (5)）
    ----------------------
    .. math::

        \\mathrm{FSS}_{useful}=\\frac12+\\frac{f_o}{2},
        \\qquad f_o=\\overline{I(o\\ge T)}

    返回 `{"radii","window","fss","f_o","fss_useful","first_useful_radius"}`；
    没有满足准则的尺度时 `first_useful_radius` 为 NaN。
    复杂度：O(len(radii) × N)。
    """
    radii_arr = np.asarray(list(radii), dtype=np.int64)
    curve = fss(f, o, threshold, radii=radii_arr, return_curve=True)
    o_arr = np.asarray(o, dtype=np.float64)
    f_o = float(np.mean((o_arr >= threshold).astype(np.float64))) if o_arr.size else _nan()
    useful = 0.5 + 0.5 * (f_o if np.isfinite(f_o) else 0.0)
    passing = np.where(curve["fss"] >= useful)[0]
    first = float(radii_arr[passing[0]]) if passing.size else _nan()
    return {"radii": curve["radii"], "window": curve["window"], "fss": curve["fss"],
            "f_o": np.asarray(f_o), "fss_useful": np.asarray(useful),
            "first_useful_radius": np.asarray(first)}


# ---------------------------------------------------------------------------
# 概率量评分
# ---------------------------------------------------------------------------

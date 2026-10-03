"""`vibe_post.verify` 单元测试：连续 / 分类 / 概率 / 邻域四类评分。

运行：`pytest tests/test_verify.py -q`。
每个公式都用可解析验证的构造核对，并覆盖空数组、全 NaN、零方差等边界。
文献：[B9] Wilks (2019)；[E3] Brier (1950)；[E7] Roberts & Lean (2008)；
[E9] Hamill (2001)；[E11] Gneiting & Raftery (2007)；[E12] Hersbach (2000)。
"""

from __future__ import annotations

import numpy as np
import pytest

from vibe_post import verify as vf


def test_continuous_scores_perfect_and_known_bias() -> None:
    """完美预报：误差类评分为 0、相关为 1；常数偏移给出解析 bias。"""
    o = np.linspace(0.0, 10.0, 21)
    assert vf.bias(o, o) == pytest.approx(0.0)
    assert vf.mae(o, o) == pytest.approx(0.0)
    assert vf.rmse(o, o) == pytest.approx(0.0)
    assert vf.correlation(o, o) == pytest.approx(1.0)

    shifted = o + 2.0
    assert vf.bias(shifted, o) == pytest.approx(2.0)
    assert vf.mae(shifted, o) == pytest.approx(2.0)
    assert vf.rmse(shifted, o) == pytest.approx(2.0)
    assert vf.correlation(shifted, o) == pytest.approx(1.0)
    assert vf.mean_error(shifted, o) == pytest.approx(vf.bias(shifted, o))


def test_continuous_degenerate_cases() -> None:
    """零方差、单点、空数组与全 NaN：返回 NaN 而不是抛异常。"""
    assert vf.correlation(np.zeros(5), np.arange(5)) == 0.0   # 与 C++ 一致：零方差 -> 0
    assert np.isnan(vf.correlation(np.array([1.0]), np.array([1.0])))  # n<2 -> NaN
    assert np.isnan(vf.rmse(np.array([]), np.array([])))
    assert np.isnan(vf.bias(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))
    # 成对剔除 NaN：只有一对有效
    assert vf.bias(np.array([1.0, np.nan]), np.array([1.0, 5.0])) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        vf.rmse(np.zeros(3), np.zeros(4))


def test_anomaly_correlation() -> None:
    """ACC：距平完全相关为 1；气候态等于观测时为 NaN。"""
    clim = np.linspace(270.0, 290.0, 10)
    obs = clim + np.linspace(-1.0, 1.0, 10)
    fcst = clim + 2.0 * np.linspace(-1.0, 1.0, 10)
    assert vf.anomaly_correlation(fcst, obs, clim) == pytest.approx(1.0)
    assert np.isnan(vf.anomaly_correlation(clim, clim, clim))


def test_contingency_table_and_categorical_scores() -> None:
    """用人工列联表核对 POD/FAR/CSI/ETS/TSS/频率偏差/ORSS 的解析值。"""
    # H=8, M=2, F=4, C=6  -> POD=0.8, FAR=1/3, CSI=8/14, FB=12/10
    f = np.concatenate([np.ones(8), np.zeros(2), np.ones(4), np.zeros(6)])
    o = np.concatenate([np.ones(8), np.ones(2), np.zeros(4), np.zeros(6)])
    table = vf.contingency_table(f, o, 0.5)
    assert table["hits"] == 8 and table["misses"] == 2
    assert table["false_alarms"] == 4 and table["correct_negatives"] == 6

    assert vf.pod(f, o, 0.5) == pytest.approx(0.8)
    assert vf.far(f, o, 0.5) == pytest.approx(4.0 / 12.0)
    assert vf.csi(f, o, 0.5) == pytest.approx(8.0 / 14.0)
    assert vf.frequency_bias(f, o, 0.5) == pytest.approx(12.0 / 10.0)
    expected_random = 10.0 * 12.0 / 20.0
    assert vf.ets(f, o, 0.5) == pytest.approx((8 - expected_random) / (14 - expected_random))
    assert vf.tss(f, o, 0.5) == pytest.approx(0.8 - 4.0 / 10.0)
    assert vf.orss(f, o, 0.5) == pytest.approx((8 * 6 - 4 * 2) / (8 * 6 + 4 * 2))

    scores = vf.scores_table(f, o, 0.5)
    assert scores["n"] == 20 and scores["csi"] == pytest.approx(8.0 / 14.0)


def test_categorical_scores_no_events() -> None:
    """无事件 / 全事件：分母为 0 的评分必须返回 NaN（而不是崩溃或 0）。"""
    f = np.zeros(10)
    o = np.zeros(10)
    assert np.isnan(vf.pod(f, o, 0.5))
    assert np.isnan(vf.far(f, o, 0.5))
    assert np.isnan(vf.csi(f, o, 0.5))
    assert np.isnan(vf.frequency_bias(f, o, 0.5))
    assert np.isnan(vf.ets(f, o, 0.5))
    assert np.isnan(vf.tss(f, o, 0.5))
    assert np.isnan(vf.orss(f, o, 0.5))
    everything = vf.contingency_table(np.ones(5), np.ones(5), 0.5)
    assert everything["hits"] == 5.0
    assert vf.pod(np.ones(5), np.ones(5), 0.5) == pytest.approx(1.0)


def test_brier_and_reliability() -> None:
    """Brier 评分与可靠性曲线：完美概率预报 BS=0、BSS=1。"""
    o = np.array([0.0, 0.0, 1.0, 1.0])
    perfect = o.copy()
    assert vf.brier_score(perfect, o) == pytest.approx(0.0)
    assert vf.brier_skill_score(perfect, o, p_clim=0.5) == pytest.approx(1.0)

    worst = 1.0 - o
    assert vf.brier_score(worst, o) == pytest.approx(1.0)
    assert np.isnan(vf.brier_skill_score(perfect, o, p_clim=0.0))
    assert vf.brier_score(np.array([0.9, 0.1]), np.array([5.0, 1.0]), threshold=2.0) > 0.0

    curve = vf.reliability_curve(perfect, o, n_bins=2)
    assert curve["counts"].sum() == 4.0
    np.testing.assert_allclose(curve["reliability"][0], 0.0)
    np.testing.assert_allclose(curve["reliability"][1], 1.0)
    empty = vf.reliability_curve(np.array([]), np.array([]), n_bins=3)
    assert np.all(np.isnan(empty["reliability"]))


def test_roc_auc_limits() -> None:
    """AUC：完美 = 1，反向 = 0，随机 ≈ 0.5；含并列值时用平均秩。"""
    o = np.array([0, 0, 1, 1], dtype=float)
    assert vf.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), o) == pytest.approx(1.0)
    assert vf.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), o) == pytest.approx(0.0)
    assert vf.roc_auc(np.full(4, 0.5), o) == pytest.approx(0.5)
    assert np.isnan(vf.roc_auc(np.array([0.1, 0.2]), np.array([1.0, 1.0])))

    curve = vf.roc_curve(np.array([0.1, 0.2, 0.8, 0.9]), o)
    assert curve["pod"][0] == pytest.approx(0.0)
    assert curve["pod"][-1] == pytest.approx(1.0)
    assert curve["pofd"][-1] == pytest.approx(1.0)
    sampled = vf.roc_curve(np.array([0.1, 0.2, 0.8, 0.9]), o, thresholds=[0.5])
    assert sampled["pod"][0] == pytest.approx(1.0)


def test_crps_ensemble_analytic() -> None:
    """CRPS：确定性集合退化为 MAE；离散度惩罚可解析核对。"""
    obs = np.array([1.0])
    # 确定性集合（子代与观测一致）-> CRPS = 0
    exact = np.array([[0.0], [0.0]])
    assert vf.crps_ensemble(np.array([[1.0], [1.0]]), obs)[0] == pytest.approx(0.0)
    # 成员 {0, 2}、观测 1：MAE = 1，成对平均绝对差 = 2（fair 分母 2）
    members = np.array([[0.0], [2.0]])
    assert vf.crps_ensemble(members, obs)[0] == pytest.approx(0.0)
    # 成员 {0, 0, 2}、观测 1：MAE = 2/3，fair 集合项 = 8/6
    three = np.array([[0.0], [0.0], [2.0]])
    expected = 1.0 / 3.0
    assert vf.crps_ensemble(three, obs)[0] == pytest.approx(expected)
    with pytest.raises(ValueError):
        vf.crps_ensemble(np.zeros((1, 3)), np.zeros(3))
    with_nan = vf.crps_ensemble(np.array([[1.0, np.nan], [2.0, np.nan]]), np.array([1.0, 0.0]))
    assert np.isnan(with_nan[1])


def test_rank_histogram_and_spread_skill() -> None:
    """排序直方图：无偏集合趋近平坦；超出集合范围落在端 bin。"""
    rng = np.random.default_rng(3)
    members = 11
    ensemble = rng.normal(size=(members, 400, 4))
    truth = rng.normal(size=(400, 4))
    histogram = vf.rank_histogram(ensemble, truth)
    assert histogram["counts"].sum() == ensemble.shape[1] * ensemble.shape[2]
    assert histogram["counts"].size == members + 1
    assert histogram["expected"][0] == pytest.approx(histogram["counts"].sum() / (members + 1))

    outside = vf.rank_histogram(np.array([[0.0], [1.0]]), np.array([5.0]))
    assert outside["counts"][-1] == 1.0

    ssr = vf.spread_skill_ratio(ensemble, truth)
    assert 0.5 < ssr < 2.0                       # 合理集合的 SSR 应接近 1
    assert vf.spread_skill_ratio(ensemble, truth, spread="range") > 0.0
    with pytest.raises(ValueError):
        vf.spread_skill_ratio(ensemble, truth, spread="bogus")
    with pytest.raises(ValueError):
        vf.spread_skill_ratio(np.zeros(3), np.zeros(3))


def test_fss_limits_and_scales() -> None:
    """FSS：完美 = 1；互不重叠随尺度单调增大；尺度曲线给出有用尺度。"""
    ones = np.ones((12, 12))
    assert vf.fractions_skill_score(ones, ones, 0.5, 1) == pytest.approx(1.0)

    forecast = np.zeros((32, 32))
    forecast[4:12, 4:12] = 1.0
    obs = np.zeros((32, 32))
    obs[7:15, 7:15] = 1.0
    small = vf.fractions_skill_score(forecast, obs, 0.5, 1)
    large = vf.fractions_skill_score(forecast, obs, 0.5, 15)
    assert 0.0 <= small < large <= 1.0

    curve = vf.fss_vs_scale(forecast, obs, 0.5, radii=(0, 1, 2, 4, 8))
    assert curve["fss"].size == 5
    assert np.all(np.diff(curve["fss"]) > -1e-9)
    assert np.isfinite(float(curve["first_useful_radius"]))

    with pytest.raises(ValueError):
        vf.fractions_skill_score(forecast, obs, 0.5, 4)      # 偶数窗口
    with pytest.raises(ValueError):
        vf.neighborhood_fractions(np.zeros((4, 4)), 9)
    with pytest.raises(ValueError):
        vf.fractions_skill_score(forecast, obs[:4], 0.5, 3)


def test_neighborhood_fractions_edge_counting() -> None:
    """邻域比例在边界用有效格点数归一（不被零填充稀释）。"""
    field = np.ones((10, 10))
    fractions = vf.neighborhood_fractions(field, 3)
    np.testing.assert_allclose(fractions, 1.0)
    single = np.zeros((5, 5))
    single[2, 2] = 1.0
    local = vf.neighborhood_fractions(single, 3)
    assert local[2, 2] == pytest.approx(1.0 / 9.0)   # 3x3 窗口只有中心是 1
    assert local[0, 0] == pytest.approx(0.0)


def test_valid_pairs_and_nan_handling() -> None:
    """`valid_pairs` 成对剔除 NaN，长度不一致时报错。"""
    f, o = vf.valid_pairs(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, np.nan]))
    np.testing.assert_array_equal(f, [1.0])
    with pytest.raises(ValueError):
        vf.valid_pairs(np.zeros(3), np.zeros(4))
    assert vf.fss(np.zeros((4, 4)), np.zeros((4, 4)), 1.0, window=1) is not None


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

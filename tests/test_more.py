"""Decomposition, intermittent demand, cleaning, ensembles, TBATS and the
thorough backtest against results recorded by the Rust crate."""

import math

import pytest

import foresight as fs
from conftest import COMPUTED, SEARCHED, logs, near, recorded

MORE = recorded("rust_more.json")


def test_stl(air):
    y = logs(air)
    cases = {
        "periodic": dict(),
        "span13": dict(seasonal_window=13),
        "robust7": dict(seasonal_window=7, robust=True),
        "robust5rounds": dict(seasonal_window=7, robust=True, inner=1, outer=5),
        "degree1": dict(seasonal_window=11, degrees=(1, 1, 1), trend_window=21),
        "flat": dict(seasonal_window=9, degrees=(0, 0, 0)),
    }
    for name, options in cases.items():
        d = fs.stl(y, 12, **options)
        w = MORE["stl"][name]
        near(d.seasonal[0], w["seasonal"], COMPUTED, name)
        near(d.trend, w["trend"], COMPUTED, name)
        near(d.trend_strength, w["trend_strength"], COMPUTED, name)
        near(d.seasonal_strength(0), w["seasonal_strength"], COMPUTED, name)
        for i, v in enumerate(y):
            assert abs(d.trend[i] + d.seasonal[0][i] + d.remainder[i] - v) < 1e-12


def two_patterns():
    weekly = [5, 0, -2, -3, 0, 1, -1]
    return [
        100 + 0.05 * t + weekly[t % 7] + 8 * math.sin(2 * math.pi * t / 30)
        + 2 * math.sin(t * 1.7) * math.cos(t * 0.3)
        for t in range(420)
    ]


def test_mstl():
    w = MORE["mstl"]
    d = fs.mstl(two_patterns(), [30, 7])
    assert d.periods == [7, 30]
    near(d.seasonal[0], w["seasonal7"], 1e-8, "weekly")
    near(d.seasonal[1], w["seasonal30"], 1e-8, "monthly")
    near(d.trend, w["trend"], 1e-8, "trend")


def test_decomposed(icms):
    w = MORE["decomposed"]
    y = fs.Series(icms, period=12)
    near(fs.Decomposed(fs.Naive()).forecast(y, 12), w["icms_naive"], COMPUTED, "naive")
    near(fs.Decomposed(fs.Drift()).forecast(y, 12), w["icms_drift"], COMPUTED, "drift")
    two = fs.Decomposed(fs.Drift(), periods=[7, 30]).forecast(fs.Series(two_patterns(), 7), 40)
    near(two, w["two"], 1e-8, "two periods")
    assert fs.Decomposed(fs.Drift()).name == "stl_drift"


def demand():
    return [float(1 + (t * 3) % 5) if (t * 7) % 10 < 3 else 0.0 for t in range(60)]


def test_croston():
    w = MORE["croston"]
    y = demand()
    for name, model in {
        "croston": fs.Croston(),
        "croston3": fs.Croston(alpha=0.3),
        "sba": fs.Croston("sba", alpha=0.2),
        "tsb": fs.Croston("tsb", alpha=0.2, beta=0.1),
    }.items():
        f = model.forecast(y, 3)
        near(f[0], w[name], COMPUTED, name)
        assert f[2] == f[0]
    near(fs.Croston(optimised=True).forecast(y, 1)[0], w["optimised"], SEARCHED, "optimised")
    with pytest.raises(ValueError):
        fs.Croston().fit([1, 0, -2, 0, 3, 0, 0, 1])


def test_clean(air, icms, fpe):
    dirty = logs(air)
    dirty[29] += 0.8
    dirty[99] -= 0.7
    dirty[60] = float("nan")
    dirty[61] = None  # None is a gap too
    doubled = list(icms)
    doubled[39] *= 2
    doubled[89] *= 0.4
    for name, values, period in [
        ("air", dirty, 12), ("icms", doubled, 12), ("fpe", fpe, 12), ("air_flat", dirty, 1),
    ]:
        w = MORE["clean"][name]
        found = fs.outliers(values, period)
        assert [o.index for o in found] == [int(i) for i in w["index"]]
        near([o.replacement for o in found], w["replacement"], COMPUTED, name)
        near(fs.clean(values, period), w["clean"], COMPUTED, name)
        near(fs.interpolate(values, period), w["filled"], COMPUTED, name)


@pytest.mark.parametrize("name", ["icms", "fpe"])
def test_ensemble(series, name):
    v, first = series[name]
    y = fs.monthly(v, first)
    for key, options in {
        "inverse_error": {},
        "equal": {"weighting": "equal"},
        "median": {"weighting": "median"},
        "stacked": {"weighting": "stacked"},
        "top3": {"top": 3},
    }.items():
        w = MORE["ensemble"][name][key]
        fit = fs.Ensemble(fs.defaults(), **options).fit(y)
        assert list(fit.params) == w["names"]
        near(list(fit.params.values()), w["weights"], 1e-4, key)
        near(fit.forecast(12), w["forecast"], SEARCHED, key)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["icms", "fpe"])
def test_tbats(series, name):
    v, _ = series[name]
    y = fs.Series(v, period=12)
    for key, model in {
        "plain": fs.Tbats(harmonics=[3], box_cox=False, trend=True, damped=False, arma_errors=False),
        "boxcox": fs.Tbats(harmonics=[3], box_cox=True, trend=True, damped=True, arma_errors=False),
        "arma": fs.Tbats(harmonics=[2], box_cox=False, trend=False, arma_orders=(1, 1)),
        "auto": fs.Tbats(),
    }.items():
        w = MORE["tbats"][name][key]
        fit = model.fit(y)
        d = fit.details
        assert d["seasonal"][0][1] == w["harmonics"]
        assert d["trend"] == w["trend"]
        assert (d["damping"] is not None and d["damping"] < 1) == w["damped"]
        assert (d["lambda"] is not None) == w["box_cox"]
        assert list(d["arma"]) == w["arma"]
        near(d["minus_two_log_likelihood"], w["likelihood"], 1e-6, key)
        near(fit.aic, w["aic"], 1e-6, key)
        near(fit.forecast(12), w["forecast"], 5e-3, key)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["icms", "fpe"])
def test_thorough_backtest(series, name):
    w = MORE["thorough"][name]
    v, first = series[name]
    r = fs.backtest(fs.monthly(v, first), fs.thorough())
    assert r.best.name == w["chosen"]
    assert [c.name for c in r.candidates] == [s[0] for s in w["scores"]]
    for c, (_, score) in zip(r.candidates, w["scores"]):
        near(c.score, score, 1e-3, c.name)
    near([p.mean for p in r.best.forecast], w["forecast"], 1e-4, "forecast")

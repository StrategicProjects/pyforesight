"""The Python side: inputs, errors, names and conversions."""

import math

import pytest

import foresight as fs


def sales():
    pattern = [1.1, 0.8, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.1, 1.3]
    out, level = [], 100.0
    for t in range(96):
        out.append(level * pattern[t % 12] + ((t * 37) % 11 - 5) * 1.5)
        level += 0.5
    return out


def test_inputs_of_any_kind():
    y = sales()
    want = fs.Theta().forecast(y, 3, period=12)
    assert fs.Theta().forecast(tuple(y), 3, period=12) == want
    assert fs.Theta().forecast(fs.Series(y, period=12), 3) == want
    np = pytest.importorskip("numpy")
    assert fs.Theta().forecast(np.array(y), 3, period=12) == want
    pd = pytest.importorskip("pandas")
    assert fs.Theta().forecast(pd.Series(y), 3, period=12) == want


def test_series():
    y = fs.monthly(sales(), first_month=3)
    assert (len(y), y.period, y.first_season) == (96, 12, 3)
    assert fs.quarterly([1, 2, 3, 4, 5], first_quarter=2).first_season == 2
    with pytest.raises(ValueError):
        fs.Series([1, 2, 3], period=12, first_season=13)
    with pytest.raises(ValueError):
        fs.Theta().fit(y, period=4)  # contradicts the series
    with pytest.raises(ValueError):
        fs.Series("123")


def test_errors_are_exceptions():
    with pytest.raises(ValueError):
        fs.HoltWinters().fit([1.0, 2.0, 3.0], period=12)
    with pytest.raises(ValueError):
        fs.Ets("XYZ")
    with pytest.raises(ValueError):
        fs.backtest(sales(), period=12, metric="r2")
    with pytest.raises(ValueError):
        fs.backtest([1.0] * 10, period=12)


def test_models_and_names():
    assert isinstance(fs.Arima.airline(), fs.Arima)
    assert isinstance(fs.Arima.airline(), fs.Model)
    assert fs.log(fs.Arima.airline()).name == "log_arima_011_011"
    assert fs.Transformed(fs.Theta()).name == "log_theta"
    assert fs.Transformed(fs.Theta(), "guerrero").fit(sales(), period=12).forecast(2)
    renamed = fs.Theta().named("my_theta", "Theta, renamed")
    assert (renamed.name, renamed.description) == ("my_theta", "Theta, renamed")
    assert [m.name for m in fs.defaults()][:3] == ["naive", "drift", "seasonal_naive"]
    assert len(fs.defaults()) == 11 and len(fs.thorough()) == 18
    assert repr(fs.Theta()) == "<Theta theta>"


def test_backtest_report():
    y = fs.monthly(sales())
    r = fs.backtest(y, [fs.Theta(), fs.HoltWinters().named("hw")], origins=12, horizon=6)
    assert [c.name for c in r.candidates] == ["theta", "hw", "mean(hw+theta)"]
    assert r["hw"].name == "hw" and r[-1].name == r.candidates[-1].name
    with pytest.raises(KeyError):
        r["nothing"]
    best = r.best
    assert best is not None and r.chosen == [c.name for c in r.candidates].index(best.name)
    p = best.forecast[0]
    lo, hi = p.interval(0.8)
    assert lo <= p.mean <= hi and set(p.intervals) == {0.8, 0.95}
    total = best.cumulative(6)
    assert math.isclose(total.mean, sum(q.mean for q in best.forecast))
    assert best.horizons[0]["horizon"] == 1 and "mape" in best.horizons[0]
    with pytest.raises(IndexError):
        best.cumulative(7)
    pd = pytest.importorskip("pandas")
    assert list(r.to_pandas().columns) == ["name", "score", "chosen", "components", "description"]
    assert list(best.to_pandas().columns)[:4] == ["horizon", "mean", "lower_80", "upper_80"]
    assert "seasonal_12" in fs.stl(sales(), 12).to_pandas().columns


def test_accuracy():
    assert fs.mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert fs.bias([100, 200], [110, 220]) == pytest.approx(10.0)
    assert fs.mae([1, 2], [2, 4]) == pytest.approx(1.5)
    assert fs.rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert fs.mase([3, 4], [3, 4], [1, 2, 3, 4, 5]) == pytest.approx(0.0)
    assert fs.inv_box_cox(fs.box_cox([1.0, 2.0], 0.3), 0.3) == pytest.approx([1.0, 2.0])
    with pytest.raises(ValueError):
        fs.mape([1, 2], [1])


# ---- what a review found (0.1.1)

def test_fit_can_cross_threads():
    from concurrent.futures import ThreadPoolExecutor

    y = fs.monthly(sales())
    with ThreadPoolExecutor(2) as pool:
        fits = list(pool.map(lambda _: fs.Theta().fit(y), range(4)))
    want = fs.Theta().fit(y).forecast(3)
    assert all(f.forecast(3) == want for f in fits)
    assert repr(fits[0]) == "<Fit theta>"


def test_regressors_must_reach_the_horizon():
    y = fs.monthly(sales())
    x = fs.Regressors({"x": [math.sin(t * 0.7) for t in range(96)]})  # no future rows
    model = fs.Arima((1, 0, 0), regressors=x)
    with pytest.raises(ValueError):
        model.forecast(y, 3)
    with pytest.raises(ValueError):
        fs.log(model).forecast(y, 3)
    with pytest.raises(ValueError, match="regressors"):
        model.fit(y).forecast(3)
    with pytest.warns(UserWarning, match="arima_100_x"):
        r = fs.backtest(y, [model, fs.Naive()], origins=12, horizon=6)
    assert r.dropped == ["arima_100_x"] and r.best.name == "naive"
    assert all(math.isfinite(p.mean) for p in r.best.forecast)
    with pytest.raises(ValueError, match="differ in length"):
        fs.Regressors({"a": [1.0, 2.0], "b": [1.0]})


def test_backtest_says_what_is_wrong():
    y = fs.monthly(sales())
    for levels in ([80], [-0.5], [float("nan")], [0.8, 1.0]):
        with pytest.raises(ValueError, match="levels"):
            fs.backtest(y, levels=levels)
    with pytest.raises(ValueError, match="horizon"):
        fs.backtest(y, horizon=0)
    with pytest.raises(ValueError, match="too short"):
        fs.backtest(y, origins=6)
    with pytest.raises(ValueError, match="not finite"):
        fs.backtest([1.0, None] * 40, period=12)
    negative = fs.monthly([-v for v in sales()])
    with pytest.warns(UserWarning, match="holt_winters"):
        r = fs.backtest(negative, [fs.Naive(), fs.HoltWinters()])
    assert r.dropped == ["holt_winters"]
    for p in r.best.forecast:
        lo, hi = p.interval(0.8)
        assert lo <= hi
    with pytest.raises(ValueError, match="no candidate"):
        fs.backtest(negative, [fs.HoltWinters()])


def test_a_series_brings_its_period():
    v = sales()
    v[41] = None
    monthly = fs.monthly(v)
    assert fs.interpolate(monthly) == fs.interpolate(v, 12)
    assert fs.interpolate(monthly) != fs.interpolate(v)
    assert fs.clean(monthly) == fs.clean(v, 12)
    whole = fs.monthly(sales(), first_month=3)
    again = fs.Series(whole)
    assert (again.period, again.first_season) == (12, 3)
    assert fs.stl(whole).periods == [12]
    assert fs.seasonal_strength(whole) == fs.seasonal_strength(sales(), 12)
    with pytest.raises(ValueError, match="contradicts"):
        fs.stl(whole, 4)


def test_sizes_within_reason():
    y = fs.monthly(sales())
    for call in (
        lambda: fs.Naive().forecast(y, 2**62),
        lambda: fs.Naive().fit(y).forecast(2**62),
        lambda: fs.acf(sales(), 2**62),
        lambda: fs.Regressors.fourier(12, 1, 2**62),
        lambda: fs.Arima((60, 0, 0)),
        lambda: fs.AutoArima(d=99),
    ):
        with pytest.raises(ValueError, match="at most"):
            call()


def test_threads_can_be_limited():
    y = fs.monthly(sales())
    free = fs.backtest(y)
    assert fs.max_threads() >= 1
    fs.set_max_threads(1)
    try:
        assert fs.max_threads() == 1
        one = fs.backtest(y)
    finally:
        fs.set_max_threads(0)
    assert [c.score for c in one] == [c.score for c in free]
    assert "theta" in free and "nothing" not in free


def test_exact_series():
    assert fs.AutoArima().forecast([5.0] * 60, 2, period=12) == pytest.approx([5.0, 5.0])
    assert fs.outliers([3.0] * 40) == []
    fit = fs.Tbats().fit([7.0] * 60, period=12)
    assert fit.log_likelihood is None and "minus_two_log_likelihood" in fit.details
    short = fs.Prophet().forecast([52.0, 57.0, 51.0, 59.0, 55.0, 53.0, 58.0, 54.0], 6, period=12)
    assert all(30 < v < 90 for v in short)
    pd = pytest.importorskip("pandas")
    r = fs.backtest(fs.monthly(sales()), [fs.Naive()], levels=[0.8, 0.995], combine=0)
    assert list(r.best.to_pandas().columns) == [
        "horizon", "mean", "lower_80", "upper_80", "lower_99.5", "upper_99.5",
    ]


def test_a_horizon_without_errors_has_no_interval():
    pd = pytest.importorskip("pandas")
    y = [5.0] * 60 + [0.0] * 60  # the naive forecast is zero at every origin
    r = fs.backtest(y, [fs.Naive()], period=12, origins=12, horizon=6, combine=0)
    frame = r.best.to_pandas()
    assert list(frame["mean"]) == [0.0] * 6
    assert r.best.forecast[0].intervals == {}
    y = [float(v) for v in range(1, 121)]
    y[107] = 0.0  # one origin forecasts zero for the same season a year later
    r = fs.backtest(fs.monthly(y), [fs.SeasonalNaive()], origins=12, horizon=12, combine=0)
    frame = r.best.to_pandas()
    assert len(frame) == 12 and "lower_80" in frame.columns

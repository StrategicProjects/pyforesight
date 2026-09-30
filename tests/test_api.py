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

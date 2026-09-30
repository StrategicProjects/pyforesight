"""The models against results recorded by the Rust crate on x86-64 Linux.

The package runs the crate itself, so the only differences are those of
floating point between platforms; the tolerances are those of the Go
edition, which reproduces the same numbers independently.
"""

import math

import pytest

import foresight as fs
from conftest import COMPUTED, SEARCHED, logs, near, recorded

MODELS = recorded("rust_models.json")
NAMES = ["air", "icms", "fpe"]


@pytest.mark.parametrize("name", NAMES)
def test_arima(series, name):
    want = MODELS[name]
    v, first = series[name]
    y, ly = fs.monthly(v, first), fs.monthly(logs(v), first)

    a = fs.Arima.airline().fit(ly)
    # a coefficient on the edge of invertibility is flat in the likelihood
    near(a.details["ma"], want["airline"]["ma"], 1e-3, "airline ma")
    near(a.details["seasonal_ma"], want["airline"]["sma"], 1e-3, "airline sma")
    near(a.log_likelihood, want["airline"]["loglik"], 1e-7, "airline loglik")
    near(a.aicc, want["airline"]["aicc"], 1e-7, "airline aicc")
    near(fs.log(fs.Arima.airline()).forecast(y, 12), want["airline"]["forecast"], SEARCHED, "airline")

    m = fs.Arima((1, 1, 1), seasonal=(1, 0, 0), constant=True).fit(ly)
    near(m.details["ar"], want["mixed"]["ar"], 1e-3, "mixed ar")
    near(m.details["seasonal_ar"], want["mixed"]["sar"], 1e-3, "mixed sar")
    near(m.details["constant"], want["mixed"]["constant"], 1e-4, "mixed constant")
    near(m.log_likelihood, want["mixed"]["loglik"], 1e-7, "mixed loglik")
    near(m.forecast(12), want["mixed"]["forecast"], SEARCHED, "mixed")

    auto = fs.AutoArima().fit(ly)
    order = list(auto.details["order"]) + list(auto.details["seasonal_order"])
    assert order == want["auto_arima"]["order"]
    assert (auto.details["constant"] is not None) == want["auto_arima"]["constant"]
    near(auto.aicc, want["auto_arima"]["aicc"], 1e-6, "auto aicc")
    near(auto.forecast(12), want["auto_arima"]["forecast"], 1e-4, "auto")


def test_regression(icms, ipca, air):
    want = MODELS["regression_index"]
    model = fs.Arima((0, 1, 1), seasonal=(0, 1, 1), regressors=fs.Regressors({"ipca": logs(ipca)}))
    fit = model.fit(fs.Series(logs(icms)[:100], period=12))
    near(fit.details["regression"]["ipca"], want["slope"], 1e-4, "slope")
    near(fit.log_likelihood, want["loglik"], 1e-7, "loglik")
    near(fit.forecast(12), want["forecast"], SEARCHED, "forecast")

    want = MODELS["regression_fourier"]
    harmonic = fs.Arima((1, 1, 1), constant=True, regressors=fs.Regressors.fourier(12, 3, 156))
    fit = harmonic.fit(fs.Series(logs(air), period=12))
    near(fit.details["constant"], want["constant"], 1e-4, "constant")
    near(fit.log_likelihood, want["loglik"], 1e-7, "loglik")
    near(fit.forecast(12), want["forecast"], SEARCHED, "forecast")
    assert harmonic.name == "arima_111_x"
    assert list(fit.details["regression"])[0] == "sin1_12"


@pytest.mark.parametrize("name", NAMES)
def test_ets(series, name):
    want = MODELS[name]
    v, first = series[name]
    y = fs.monthly(v, first)
    for code, w in want["ets"].items():
        fit = fs.Ets(code).fit(y)
        assert fit.details["code"] == code
        # many numbers are searched at once; what counts is the likelihood
        near(fit.log_likelihood, w["loglik"], 1e-6, code)
        near(fit.aicc, w["aicc"], 1e-6, code)
        near(fit.forecast(12), w["forecast"], 2e-3, code)
    auto = fs.AutoEts().fit(y)
    assert auto.details["code"] == want["auto_ets"]["code"]
    near(auto.aicc, want["auto_ets"]["aicc"], 1e-6, "auto ets")


@pytest.mark.parametrize("name", NAMES)
def test_prophet(series, name):
    want = MODELS[name]
    v, first = series[name]
    for model, y, w in [
        (fs.Prophet(), fs.monthly(v, first), want["prophet"]),
        (fs.Prophet(fourier_order=5), fs.monthly(v, first), want["prophet5"]),
        (fs.Prophet(), fs.monthly(logs(v), first), want["prophet_log"]),
    ]:
        fit = model.fit(y)
        near(fit.forecast(12), w["forecast"], 1e-8, "prophet")
        near(fit.details["sigma"], w["sigma"], 1e-8, "sigma")
        assert [p for p, _ in fit.details["changepoints"]] == [int(b) for b in w["bends"]]


def test_prophet_events():
    want = MODELS["events"]
    pattern = [5, -3, 0, 2, -4, 1, 3, -2, 0, 4, -5, 4]
    y = []
    for i in range(120):
        v = 200 + 0.8 * i + pattern[i % 12] + ((i * 37) % 11) * 0.3
        if i in (10, 34, 58, 82, 106):
            v += 20
        if i >= 80:
            v -= 15
        y.append(v)
    model = fs.Prophet(
        changepoints=0,
        events=[("campaign", [10, 34, 58, 82, 106, 126])],
        steps=[("new_law", 80)],
    )
    fit = model.fit(fs.monthly(y))
    effects = fit.details["effects"]
    near([effects["campaign"], effects["new_law"]], want["effects"], 1e-8, "effects")
    near(fit.forecast(12), want["forecast"], 1e-8, "forecast")


@pytest.mark.parametrize("name", NAMES)
def test_diagnostics(series, name):
    want = MODELS[name]
    v, first = series[name]
    lv = logs(v)
    near(fs.guerrero(fs.monthly(v, first)), want["guerrero"], 1e-6, "guerrero")
    near(fs.kpss(lv), want["kpss"], 1e-8, "kpss")
    assert fs.ndiffs(lv) == want["ndiffs"]
    assert fs.nsdiffs(lv, 12) == want["nsdiffs"]
    near(fs.seasonal_strength(lv, 12), want["strength"], 1e-8, "strength")


@pytest.mark.parametrize("name", NAMES)
def test_default_backtest(series, name):
    want = MODELS[name]["backtest"]
    v, first = series[name]
    r = fs.backtest(fs.monthly(v, first))
    assert r.best.name == want["chosen"]
    assert [c.name for c in r.candidates] == [s[0] for s in want["scores"]]
    for c, (_, score) in zip(r.candidates, want["scores"]):
        near(c.score, score, 1e-4, c.name)
    near([p.mean for p in r.best.forecast], want["forecast"], SEARCHED, "forecast")

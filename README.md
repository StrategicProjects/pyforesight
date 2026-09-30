# pyforesight

[![PyPI](https://img.shields.io/pypi/v/pyforesight.svg)](https://pypi.org/project/pyforesight/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23050402.svg)](https://doi.org/10.5281/zenodo.23050402)
[![CI](https://github.com/StrategicProjects/pyforesight/actions/workflows/ci.yml/badge.svg)](https://github.com/StrategicProjects/pyforesight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](pyproject.toml)

Time series forecasting in Python that picks its model by what would have
worked.

`foresight` fits several models to a series, replays the past to see how each
would have done, chooses by out-of-sample error and reports intervals taken
from the errors actually observed, including intervals for the total of the
next k periods.

The models, the backtest and the utilities are the Rust crate
[foresight](https://github.com/milkway/foresight), compiled into the package:
the numbers are the crate's, the backtest runs on all cores, and nothing else
is needed at run time. NumPy and pandas are accepted and, for pandas, produced
on request, but neither is required.

<p align="center"><img src="https://raw.githubusercontent.com/StrategicProjects/pyforesight/main/docs/figures/architecture.svg" alt="Architecture: the Rust crate foresight holds every computation; pyforesight (Python, PyO3) and foresightr (R, extendr) call it; foresight-go is an independent Go port checked against it." width="100%"></p>

**Website:** <https://strategicprojects.github.io/pyforesight/> ·
[Português](README.pt-BR.md)

## Install

```bash
pip install pyforesight
```

Python 3.9 or later. Wheels are ready for Linux (x86-64 and aarch64), macOS
(Intel and Apple silicon) and Windows; elsewhere pip compiles the Rust code,
which needs a Rust toolchain (<https://rustup.rs>). The package is imported
as `foresight`.

## Use

```python
import foresight as fs

# monthly data whose first observation is in March
y = fs.monthly(values, first_month=3)

# replay the last 36 months, 12 months ahead, with the 11 default models
report = fs.backtest(y)

best = report.best
print(f"{best.name}: MAPE {best.score:.1f}%")
for p in best.forecast:
    lo, hi = p.interval(0.80)
    print(p.horizon, round(p.mean), round(lo), round(hi))

half_year = best.cumulative(6)      # the total of the next six months, with its own interval
report.to_pandas()                  # one row per candidate (needs pandas)
best.to_pandas()                    # the forecast with its intervals
```

One model on its own:

```python
# seasonal ARIMA on the log scale
fit = fs.log(fs.Arima.airline()).fit(y)
next_year = fit.forecast(12)

# orders chosen from the data, inspected
auto = fs.AutoArima().fit(fs.monthly(log_values, first_month=3))
auto.details["order"], auto.details["seasonal_order"], auto.aicc
```

A trend that bends, with dated events:

```python
model = fs.Prophet(
    events={"campaign": [10, 34, 58, 82, 106, 130]},  # future ones included
    steps={"new_law": 80},                              # a lasting change of level
)
fit = model.fit(y)
fit.details["changepoints"], fit.details["effects"]
```

Several models combined, and the wider set of candidates:

```python
ensemble = fs.Ensemble(fs.defaults(), weighting="stacked")
report = fs.backtest(y, fs.thorough() + [ensemble.named("my_ensemble")])
```

Any sequence of numbers works where a series is expected: a list, a NumPy
array, a pandas Series. Only the values are read (an index of dates is not),
so say what the series is with `fs.monthly`, `fs.quarterly` or `fs.Series`, or
pass the seasonal period: `fs.Theta().fit(values, period=12)`. Missing values
(`None`, NaN) are only accepted by the cleaning functions.

Good to know:

- Positions count from 0 at the first observation: Prophet's `events` and
  `steps`, `Outlier.index`, `Report.first_origin` and the changepoints of a
  fit. Months and quarters (`first_month`, `first_season`) count from 1.
- A candidate that cannot forecast at every origin and from the whole series
  is left out of the report, with a warning; `report.dropped` names them.
- `fs.set_max_threads(n)` limits the threads of backtests and ensembles; the
  results do not depend on it. The GIL is released while models are fitted.
- Models, fits and reports hold Rust objects and cannot be pickled or
  copied; a model is cheap to build again in another process.

## What is in it

| Piece | What it does |
|---|---|
| `Series`, `monthly`, `quarterly` | values + seasonal period + season of the first observation |
| `Model` / `Fit` | fit once, forecast any horizon, inspect `params`, `details`, likelihood and residuals |
| Models | `Mean`, `Naive`, `Drift`, `SeasonalNaive`, `Theta`, `HoltWinters`, `LogLinear` (optionally deflated by a price index), `Arima` (seasonal, exact maximum likelihood, optionally with regressors), `AutoArima`, `Ets`, `AutoEts`, `Prophet` (changepoints, Fourier seasonality, dated events and steps), `Tbats` (several seasonal periods, not necessarily whole numbers), `Croston` (with SBA and TSB) |
| `Transformed`, `log` | any model on the log or another Box-Cox scale, λ fixed or by Guerrero's method |
| `Decomposed`, `stl`, `mstl` | trend, seasonal patterns and remainder by LOESS; any model on the seasonally adjusted series |
| `Ensemble` | average, median, weights by inverse error or stacked weights |
| `Regressors` | external variables, Fourier terms, seasonal dummies |
| `defaults`, `thorough` | ready sets of 11 and 18 candidates |
| `backtest`, `set_max_threads` | rolling origin (expanding or fixed window) on all cores, or as many as allowed; MAPE, MAE, RMSE, MASE and bias by horizon; average of the best models; choice by out-of-sample error; empirical intervals by horizon and for totals |
| `interpolate`, `outliers`, `clean` | gaps filled and outliers found and replaced, with the season taken into account |
| Measures and tests | `mape`, `bias`, `mae`, `rmse`, `mase`, `acf`, `difference`, `kpss`, `ndiffs`, `nsdiffs`, `seasonal_strength`, `box_cox`, `inv_box_cox`, `guerrero` |

## How it differs from the usual toolkits

<p align="center"><img src="https://raw.githubusercontent.com/StrategicProjects/pyforesight/main/docs/figures/backtest.svg" alt="How a model is chosen: every candidate is refitted at each origin and forecasts ahead; the errors by horizon rank the candidates and give the empirical intervals." width="100%"></p>

Most forecasting libraries choose a model by an in-sample information
criterion and derive intervals from distributional assumptions. Here the
choice and the intervals both come from forecasts made without seeing the
future they are judged against. The interval for a total (say, the rest of a
fiscal year) is measured on totals, because adding up monthly limits
overstates its uncertainty.

## Checked

The package runs the Rust crate, so its numbers are the crate's; the tests
check that nothing is lost on the way, against results recorded by the crate:
ARIMA, regression with ARIMA errors, ETS, Prophet, TBATS, STL and MSTL,
Croston, cleaning, ensembles, tests of stationarity and seasonality, and the
backtests of 11 and 18 candidates on three public series. The crate itself is
compared with the R packages `forecast` 9.0.2 and `prophet` 1.1.7, and
reproduced independently by the Go edition
[foresight-go](https://github.com/milkway/foresight-go). The same methods are
available in R: [foresightr](https://strategicprojects.github.io/foresightr/).

```bash
pip install maturin pytest
maturin develop --release
pytest                 # about 30 s; pytest -m "not slow" skips TBATS and the thorough backtest
```

## Data

`tests/data` has two public series: the monthly ICMS and FPE revenue of the
state of Piauí, Brazil (Siconfi/STN, with the IPCA price index from the
Central Bank of Brazil), and the airline passengers of Box & Jenkins.

## Citation

Zenodo: <https://doi.org/10.5281/zenodo.23050402> (all versions); see also
[CITATION.cff](CITATION.cff).

## Authors

André Leite, Marcos Wasiliew, Hugo Vasconcelos, Carlos Amorim and Diogo
Bezerra.

## License

MIT.

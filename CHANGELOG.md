# Changelog

## 0.1.1

On the crate foresight 0.7.2 (see its changelog): a candidate that cannot
give its final forecast is left out instead of voiding the backtest; intervals
of negative forecasts come in order; Prophet needs two cycles for seasonal
terms; automatic ARIMA keeps exact fits; forecasts by decomposition keep the
position of the series; no spurious outliers on exact data; TBATS on a
constant series; no threads inside threads.

In the package:

- A `Fit` can be used from any thread, and the GIL is released while any
  model is fitted.
- Regressors that stop short of the horizon raise `ValueError` instead of
  giving NaN, also inside `log`, `Decomposed` and ensembles; in a backtest the
  candidate is left out.
- `backtest` checks `levels`, `horizon` and the length of the series and says
  which is wrong; candidates left out raise a warning and are listed in
  `Report.dropped`.
- `set_max_threads` and `max_threads`.
- A `Series` brings its period to `clean`, `interpolate`, `outliers`, `stl`,
  `nsdiffs`, `seasonal_strength` and `Series(...)`.
- Sizes that would exhaust memory (`h=2**62`) raise `ValueError`.
- `Fit.log_likelihood` is `None` for TBATS, whose objective is in
  `details["minus_two_log_likelihood"]`.
- `to_pandas` names levels such as 0.995 `lower_99.5`; `Report` can be
  iterated and tested with `in`; columns of `Regressors` must have the same
  length.

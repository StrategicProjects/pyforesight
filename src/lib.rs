//! Python bindings of the Rust crate `foresight`.
//!
//! Every model, the backtest and the utilities run in Rust; this module only
//! converts between Python objects and the crate's types. The backtest
//! releases the GIL while it runs on all cores.

use std::sync::Arc;

use foresight as fs;
use fs::decompose::{Mstl, SeasonalWindow, Stl};
use fs::models as m;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::PyClassInitializer;

// ---------------------------------------------------------------- series

/// A regularly spaced series with a seasonal period.
///
/// ``period`` is the length of the seasonal cycle (12 for monthly data with a
/// yearly cycle, 4 for quarterly, 1 for none) and ``first_season`` the
/// position of the first observation in that cycle, counting from 1 (for
/// monthly data, the calendar month: 1 = January).
#[pyclass(frozen, module = "foresight")]
struct Series {
    values: Vec<f64>,
    period: usize,
    phase: usize,
}

#[pymethods]
impl Series {
    #[new]
    #[pyo3(signature = (values, period = None, first_season = None))]
    fn new(
        values: &Bound<'_, PyAny>,
        period: Option<usize>,
        first_season: Option<usize>,
    ) -> PyResult<Self> {
        // another Series keeps its period and season unless told otherwise
        let given = values.cast::<Series>().ok().map(|s| {
            let s = s.get();
            (s.period, s.phase + 1)
        });
        let period = within("period", period.or(given.map(|g| g.0)).unwrap_or(1), MOST)?.max(1);
        let first_season = first_season
            .or(given.filter(|g| g.0 == period).map(|g| g.1))
            .unwrap_or(1);
        if first_season == 0 || first_season > period {
            return Err(PyValueError::new_err(format!(
                "first_season must be between 1 and the period ({period})"
            )));
        }
        Ok(Series {
            values: floats(values)?,
            period,
            phase: first_season - 1,
        })
    }

    #[getter]
    fn values(&self) -> Vec<f64> {
        self.values.clone()
    }

    #[getter]
    fn period(&self) -> usize {
        self.period
    }

    #[getter]
    fn first_season(&self) -> usize {
        self.phase + 1
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Series({} values, period={}, first_season={})",
            self.values.len(),
            self.period,
            self.phase + 1
        )
    }
}

/// Monthly data with a yearly cycle; ``first_month`` is the calendar month of
/// the first observation (1 = January).
#[pyfunction]
#[pyo3(signature = (values, first_month = 1))]
fn monthly(values: &Bound<'_, PyAny>, first_month: usize) -> PyResult<Series> {
    Series::new(values, Some(12), Some(first_month))
}

/// Quarterly data with a yearly cycle; ``first_quarter`` is 1 for Q1.
#[pyfunction]
#[pyo3(signature = (values, first_quarter = 1))]
fn quarterly(values: &Bound<'_, PyAny>, first_quarter: usize) -> PyResult<Series> {
    Series::new(values, Some(4), Some(first_quarter))
}

/// Numbers from any iterable (list, tuple, NumPy array, pandas Series);
/// ``None`` becomes NaN.
fn floats(values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if values.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyValueError::new_err("expected numbers, got a string"));
    }
    if let Ok(s) = values.cast::<Series>() {
        return Ok(s.get().values.clone());
    }
    let mut out = Vec::new();
    for item in values.try_iter()? {
        let item = item?;
        out.push(if item.is_none() {
            f64::NAN
        } else {
            item.extract::<f64>()?
        });
    }
    Ok(out)
}

/// The values, period and phase of a series given either as a [`Series`] or
/// as numbers plus an optional period.
struct Owned {
    values: Vec<f64>,
    period: usize,
    phase: usize,
}

impl Owned {
    fn view(&self) -> fs::Series<'_> {
        fs::Series::new(&self.values, self.period).with_phase(self.phase)
    }
}

fn owned(y: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<Owned> {
    if let Ok(s) = y.cast::<Series>() {
        let s = s.get();
        if let Some(p) = period {
            if p.max(1) != s.period {
                return Err(PyValueError::new_err(format!(
                    "period={p} contradicts the period of the series ({})",
                    s.period
                )));
            }
        }
        return Ok(Owned {
            values: s.values.clone(),
            period: s.period,
            phase: s.phase,
        });
    }
    Ok(Owned {
        values: floats(y)?,
        period: within("period", period.unwrap_or(1), MOST)?.max(1),
        phase: 0,
    })
}

/// Sizes that drive allocations or loops are kept within reason: a larger
/// number is a mistake, and would exhaust memory or never return.
const MOST: usize = 1_000_000;

fn within(what: &str, value: usize, most: usize) -> PyResult<usize> {
    if value > most {
        return Err(PyValueError::new_err(format!(
            "{what} must be at most {most}, got {value}"
        )));
    }
    Ok(value)
}

fn unfit(what: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{what}: the series is too short, not finite, or otherwise unsuitable for the model"
    ))
}

// ---------------------------------------------------------------- models

/// A model shared between Python objects and the threads of a backtest.
#[derive(Clone)]
struct Shared(Arc<dyn fs::Model>);

impl fs::Model for Shared {
    fn name(&self) -> String {
        self.0.name()
    }

    fn description(&self) -> String {
        self.0.description()
    }

    fn fit(&self, y: fs::Series<'_>) -> Option<Box<dyn fs::Fitted>> {
        self.0.fit(y)
    }

    // the model's own checks (regressors that reach the horizon) still apply
    fn forecast(&self, y: fs::Series<'_>, h: usize) -> Option<Vec<f64>> {
        self.0.forecast(y, h)
    }
}

/// Models whose estimate says more than the forecasts: kept to fill
/// [`Fit::details`].
#[derive(Clone)]
enum Rich {
    Arima(m::Arima),
    ArimaX(m::ArimaX),
    AutoArima(m::AutoArima),
    Ets(m::Ets),
    AutoEts(m::AutoEts),
    Prophet(m::Prophet),
    Tbats(m::Tbats),
}

impl Rich {
    fn estimate(&self, y: &Owned) -> Option<RichFit> {
        let v = y.view();
        Some(match self {
            Rich::Arima(a) => RichFit::Arima(a.estimate(v)?),
            Rich::ArimaX(a) => RichFit::Arima(a.estimate(v)?),
            Rich::AutoArima(a) => RichFit::Arima(a.select(v)?),
            Rich::Ets(e) => RichFit::Ets(e.estimate(v)?),
            Rich::AutoEts(e) => RichFit::Ets(e.select(v)?),
            Rich::Prophet(p) => RichFit::Prophet(p.estimate(v)?),
            Rich::Tbats(t) => RichFit::Tbats(t.select(v)?),
        })
    }
}

/// A forecasting method, before seeing any data.
///
/// Fit it to a series with :meth:`fit`, or forecast in one go with
/// :meth:`forecast`. Models are plain configuration: the same object can be
/// fitted to many series and take part in many backtests.
#[pyclass(subclass, frozen, module = "foresight")]
struct Model {
    inner: Arc<dyn fs::Model>,
    rich: Option<Rich>,
    name: String,
    description: String,
}

impl Model {
    fn of(model: impl fs::Model + 'static) -> Self {
        Model {
            name: model.name(),
            description: model.description(),
            inner: Arc::new(model),
            rich: None,
        }
    }

    fn rich(model: impl fs::Model + 'static, rich: Rich) -> Self {
        Model {
            rich: Some(rich),
            ..Model::of(model)
        }
    }

    fn shared(&self) -> Shared {
        Shared(self.inner.clone())
    }

    fn candidate(&self) -> fs::Candidate {
        fs::Candidate::new(self.shared()).named(self.name.clone(), self.description.clone())
    }
}

#[pymethods]
impl Model {
    /// Short identifier, e.g. ``holt_winters``.
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// One-line description of the method.
    #[getter]
    fn description(&self) -> String {
        self.description.clone()
    }

    /// The same model under another name (and description), as it will
    /// appear in a backtest.
    #[pyo3(signature = (name, description = None))]
    fn named(&self, name: String, description: Option<String>) -> Model {
        Model {
            inner: self.inner.clone(),
            rich: self.rich.clone(),
            name,
            description: description.unwrap_or_else(|| self.description.clone()),
        }
    }

    /// Estimates the model on ``y`` (a :class:`Series` or numbers, with
    /// ``period`` for the latter).
    #[pyo3(signature = (y, period = None))]
    fn fit(&self, py: Python<'_>, y: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<Fit> {
        let y = owned(y, period)?;
        // the GIL is released while the model is estimated, which may take
        // seconds
        let estimate = py
            .detach(|| match &self.rich {
                Some(rich) => rich.estimate(&y).map(Estimate::Rich),
                None => self.inner.fit(y.view()).map(Estimate::Plain),
            })
            .ok_or_else(|| unfit(&self.name))?;
        Ok(Fit {
            model: self.name.clone(),
            estimate,
        })
    }

    /// Fits and forecasts the ``h`` periods after the last observation.
    #[pyo3(signature = (y, h, period = None))]
    fn forecast(
        &self,
        py: Python<'_>,
        y: &Bound<'_, PyAny>,
        h: usize,
        period: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        let y = owned(y, period)?;
        let h = within("h", h, MOST)?;
        py.detach(|| self.inner.forecast(y.view(), h))
            .ok_or_else(|| unfit(&self.name))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let class = slf.get_type().name()?;
        Ok(format!("<{class} {}>", slf.get().name))
    }
}

fn init<T: pyo3::PyClass<BaseType = Model>>(model: Model, sub: T) -> PyClassInitializer<T> {
    PyClassInitializer::from(model).add_subclass(sub)
}

/// The mean of the history.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Mean;

#[pymethods]
impl Mean {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        init(Model::of(m::Mean), Mean)
    }
}

/// The last value (random walk).
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Naive;

#[pymethods]
impl Naive {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        init(Model::of(m::Naive), Naive)
    }
}

/// The last value plus the average change (random walk with drift).
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Drift;

#[pymethods]
impl Drift {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        init(Model::of(m::Drift), Drift)
    }
}

/// The same season of the last cycle; with ``growth``, scaled by the growth
/// of the last cycle over the one before.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct SeasonalNaive;

#[pymethods]
impl SeasonalNaive {
    #[new]
    #[pyo3(signature = (growth = false))]
    fn new(growth: bool) -> PyClassInitializer<Self> {
        let model = if growth {
            m::SeasonalNaive::with_growth()
        } else {
            m::SeasonalNaive::new()
        };
        init(Model::of(model), SeasonalNaive)
    }
}

/// The Theta method (Assimakopoulos & Nikolopoulos, 2000), seasonally
/// adjusted by classical decomposition when the series is seasonal.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Theta;

#[pymethods]
impl Theta {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        init(Model::of(m::Theta), Theta)
    }
}

/// Holt-Winters with multiplicative seasonality, smoothing chosen by
/// one-step squared error.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct HoltWinters;

#[pymethods]
impl HoltWinters {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        init(Model::of(m::HoltWinters), HoltWinters)
    }
}

/// Regression of the log on a trend and seasonal dummies; optionally on the
/// last ``window`` observations only and deflated by a price index
/// (``deflator``, one value per period from the first observation on; the
/// future is not read: forecasts are inflated back at the index's growth over
/// the last cycle).
#[pyclass(extends = Model, frozen, module = "foresight")]
struct LogLinear;

#[pymethods]
impl LogLinear {
    #[new]
    #[pyo3(signature = (window = None, deflator = None))]
    fn new(
        window: Option<usize>,
        deflator: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut model = m::LogLinear::new();
        if let Some(n) = window {
            model = model.window(n);
        }
        if let Some(index) = deflator {
            model = model.deflated_by(floats(index)?);
        }
        Ok(init(Model::of(model), LogLinear))
    }
}

fn order3(o: Option<(usize, usize, usize)>) -> (usize, usize, usize) {
    o.unwrap_or((0, 0, 0))
}

/// Seasonal ARIMA estimated by exact maximum likelihood.
///
/// ``order`` is (p, d, q), ``seasonal`` is (P, D, Q). ``constant`` asks for a
/// mean (undifferenced series) or a drift (one difference); by default only
/// the mean is estimated. With ``regressors`` it becomes a regression with
/// ARIMA errors.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Arima;

fn arima(
    order: (usize, usize, usize),
    seasonal: Option<(usize, usize, usize)>,
    constant: Option<bool>,
    regressors: Option<&Regressors>,
) -> PyResult<Model> {
    let (sp, sd, sq) = order3(seasonal);
    for (what, value, most) in [
        ("p", order.0, 50),
        ("d", order.1, 5),
        ("q", order.2, 50),
        ("P", sp, 50),
        ("D", sd, 5),
        ("Q", sq, 50),
    ] {
        within(what, value, most)?;
    }
    let mut a = m::Arima::new(order.0, order.1, order.2).seasonal(sp, sd, sq);
    if let Some(c) = constant {
        a = a.constant(c);
    }
    Ok(match regressors {
        Some(r) => {
            let x = a.with_regressors(r.inner.clone());
            Model::rich(x.clone(), Rich::ArimaX(x))
        }
        None => Model::rich(a, Rich::Arima(a)),
    })
}

#[pymethods]
impl Arima {
    #[new]
    #[pyo3(signature = (order = (0, 1, 1), seasonal = None, constant = None, regressors = None))]
    fn new(
        order: (usize, usize, usize),
        seasonal: Option<(usize, usize, usize)>,
        constant: Option<bool>,
        regressors: Option<PyRef<'_, Regressors>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(init(
            arima(order, seasonal, constant, regressors.as_deref())?,
            Arima,
        ))
    }

    /// ARIMA(0,1,1)(0,1,1), the "airline model": a good default for seasonal
    /// series, usually on the log scale.
    #[staticmethod]
    fn airline(py: Python<'_>) -> PyResult<Py<Arima>> {
        Py::new(
            py,
            init(arima((0, 1, 1), Some((0, 1, 1)), None, None)?, Arima),
        )
    }
}

fn criterion(name: &str) -> PyResult<m::Criterion> {
    match name.to_ascii_lowercase().as_str() {
        "aicc" => Ok(m::Criterion::Aicc),
        "aic" => Ok(m::Criterion::Aic),
        "bic" => Ok(m::Criterion::Bic),
        _ => Err(PyValueError::new_err(
            "criterion must be 'aicc', 'aic' or 'bic'",
        )),
    }
}

/// ARIMA with orders chosen automatically: differences by the KPSS test and
/// the strength of seasonality, orders by stepwise search on ``criterion``.
/// Fitting chooses again, so a backtest judges the whole procedure.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct AutoArima;

#[pymethods]
impl AutoArima {
    #[new]
    #[pyo3(signature = (criterion = "aicc", d = None, seasonal_d = None, max_order = None, regressors = None))]
    fn new(
        criterion: &str,
        d: Option<usize>,
        seasonal_d: Option<usize>,
        max_order: Option<(usize, usize, usize, usize)>,
        regressors: Option<PyRef<'_, Regressors>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut a = m::AutoArima::new().criterion(self::criterion(criterion)?);
        a.d = d.map(|d| within("d", d, 5)).transpose()?;
        a.seasonal_d = seasonal_d.map(|d| within("seasonal_d", d, 5)).transpose()?;
        if let Some((p, q, sp, sq)) = max_order {
            for v in [p, q, sp, sq] {
                within("max_order", v, 50)?;
            }
            a = a.max_orders(p, q, sp, sq);
        }
        if let Some(r) = regressors {
            a = a.regressors(r.inner.clone());
        }
        Ok(init(Model::rich(a.clone(), Rich::AutoArima(a)), AutoArima))
    }
}

/// A member of the exponential smoothing family in state space form, by its
/// code: error (A, M), trend (N, A, Ad) and season (N, A, M), e.g. ``"MAM"``
/// or ``"AAdN"``.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Ets;

#[pymethods]
impl Ets {
    #[new]
    fn new(code: &str) -> PyResult<PyClassInitializer<Self>> {
        let e = m::Ets::from_code(code)
            .ok_or_else(|| PyValueError::new_err(format!("{code:?} is not an ETS code")))?;
        Ok(init(Model::rich(e, Rich::Ets(e)), Ets))
    }
}

/// The member of the exponential smoothing family with the best
/// ``criterion``; multiplicative parts only for positive series.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct AutoEts;

#[pymethods]
impl AutoEts {
    #[new]
    #[pyo3(signature = (criterion = "aicc"))]
    fn new(criterion: &str) -> PyResult<PyClassInitializer<Self>> {
        let e = m::AutoEts::new().criterion(self::criterion(criterion)?);
        Ok(init(Model::rich(e, Rich::AutoEts(e)), AutoEts))
    }
}

/// Prophet (Taylor & Letham, 2018) without Stan: piecewise linear trend with
/// changepoints, Fourier seasonality, and optional ``events`` (name → the
/// positions where they happen, future ones included) and ``steps``
/// (name → the position from which a lasting change applies). Positions
/// count from 0 at the first observation.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Prophet;

#[pymethods]
impl Prophet {
    #[new]
    #[pyo3(signature = (
        changepoints = None, changepoint_range = None, changepoint_prior_scale = None,
        seasonality_prior_scale = None, fourier_order = None, event_prior_scale = None,
        events = None, steps = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        changepoints: Option<usize>,
        changepoint_range: Option<f64>,
        changepoint_prior_scale: Option<f64>,
        seasonality_prior_scale: Option<f64>,
        fourier_order: Option<usize>,
        event_prior_scale: Option<f64>,
        events: Option<&Bound<'_, PyAny>>,
        steps: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut p = m::Prophet::new();
        if let Some(n) = changepoints {
            p = p.changepoints(n);
        }
        if let Some(r) = changepoint_range {
            p = p.changepoint_range(r);
        }
        if let Some(s) = changepoint_prior_scale {
            p = p.changepoint_prior_scale(s);
        }
        if let Some(s) = seasonality_prior_scale {
            p = p.seasonality_prior_scale(s);
        }
        if let Some(k) = fourier_order {
            p = p.fourier_order(k);
        }
        if let Some(s) = event_prior_scale {
            p = p.event_prior_scale(s);
        }
        for (name, positions) in pairs(events, |v| v.extract::<Vec<usize>>())? {
            p = p.event(name, &positions);
        }
        for (name, from) in pairs(steps, |v| v.extract::<usize>())? {
            p = p.step(name, from);
        }
        Ok(init(Model::rich(p.clone(), Rich::Prophet(p)), Prophet))
    }
}

/// Name-value pairs from a dict or from a sequence of pairs.
fn pairs<T>(
    obj: Option<&Bound<'_, PyAny>>,
    value: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<(String, T)>> {
    let Some(obj) = obj else {
        return Ok(Vec::new());
    };
    if let Ok(d) = obj.cast::<PyDict>() {
        return d
            .iter()
            .map(|(k, v)| Ok((k.extract()?, value(&v)?)))
            .collect();
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.push((k, value(&v)?));
    }
    Ok(out)
}

/// TBATS (De Livera, Hyndman & Snyder, 2011): trigonometric seasonality for
/// one or several ``periods``, not necessarily whole numbers, Box-Cox, damped
/// trend and ARMA errors. Whatever is left as ``None`` is chosen by AIC.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Tbats;

#[pymethods]
impl Tbats {
    #[new]
    #[pyo3(signature = (
        periods = None, harmonics = None, box_cox = None, trend = None, damped = None,
        arma_errors = None, arma_orders = None,
    ))]
    fn new(
        periods: Option<Vec<f64>>,
        harmonics: Option<Vec<usize>>,
        box_cox: Option<bool>,
        trend: Option<bool>,
        damped: Option<bool>,
        arma_errors: Option<bool>,
        arma_orders: Option<(usize, usize)>,
    ) -> PyClassInitializer<Self> {
        let mut t = m::Tbats::new(&periods.unwrap_or_default());
        if let Some(h) = harmonics {
            t = t.harmonics(&h);
        }
        if let Some(b) = box_cox {
            t = t.box_cox(b);
        }
        if let Some(b) = trend {
            t = t.trend(b);
        }
        if let Some(b) = damped {
            t = t.damped(b);
        }
        if let Some(b) = arma_errors {
            t = t.arma_errors(b);
        }
        if let Some((p, q)) = arma_orders {
            t = t.arma_orders(p, q);
        }
        init(Model::rich(t.clone(), Rich::Tbats(t)), Tbats)
    }
}

/// Intermittent demand: ``variant`` ``"croston"`` (Croston, 1972), ``"sba"``
/// (Syntetos-Boylan correction) or ``"tsb"`` (Teunter-Syntetos-Babai).
/// Smoothing ``alpha`` (default 0.1) and, for TSB, ``beta``; ``optimised``
/// chooses ``alpha`` by squared error.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Croston;

#[pymethods]
impl Croston {
    #[new]
    #[pyo3(signature = (variant = "croston", alpha = None, beta = None, optimised = false))]
    fn new(
        variant: &str,
        alpha: Option<f64>,
        beta: Option<f64>,
        optimised: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let v = match variant.to_ascii_lowercase().as_str() {
            "croston" => m::Intermittent::Croston,
            "sba" => m::Intermittent::Sba,
            "tsb" => m::Intermittent::Tsb,
            _ => {
                return Err(PyValueError::new_err(
                    "variant must be 'croston', 'sba' or 'tsb'",
                ))
            }
        };
        let mut c = m::Croston::new().variant(v);
        if let Some(a) = alpha {
            c = c.alpha(a);
        }
        if let Some(b) = beta {
            c = c.beta(b);
        }
        if optimised {
            c = c.optimised();
        }
        Ok(init(Model::of(c), Croston))
    }
}

/// ``model`` on the seasonally adjusted series (STL, or MSTL for several
/// ``periods``); the seasonal pattern of the last cycle is added back.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Decomposed;

#[pymethods]
impl Decomposed {
    #[new]
    #[pyo3(signature = (model, periods = None, robust = false))]
    fn new(
        model: PyRef<'_, Model>,
        periods: Option<Vec<usize>>,
        robust: bool,
    ) -> PyClassInitializer<Self> {
        let mut d = m::Decomposed::new(model.shared()).robust(robust);
        if let Some(p) = periods {
            d = d.periods(&p);
        }
        init(Model::of(d), Decomposed)
    }
}

/// Several models combined. ``weighting``: ``"inverse_error"`` (default),
/// ``"equal"``, ``"median"`` or ``"stacked"``; the weights come from forecasts
/// of the last ``origins`` periods, ``horizon`` ahead; ``top`` keeps only the
/// best members.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Ensemble;

#[pymethods]
impl Ensemble {
    #[new]
    #[pyo3(signature = (members, weighting = "inverse_error", origins = None, horizon = None, top = None))]
    fn new(
        members: Vec<PyRef<'_, Model>>,
        weighting: &str,
        origins: Option<usize>,
        horizon: Option<usize>,
        top: Option<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let w = match weighting.to_ascii_lowercase().as_str() {
            "inverse_error" => m::Weighting::InverseError,
            "equal" => m::Weighting::Equal,
            "median" => m::Weighting::Median,
            "stacked" => m::Weighting::Stacked,
            _ => {
                return Err(PyValueError::new_err(
                    "weighting must be 'inverse_error', 'equal', 'median' or 'stacked'",
                ))
            }
        };
        let mut e = m::Ensemble::new(members.iter().map(|m| m.candidate()).collect()).weighting(w);
        if let Some(n) = origins {
            e = e.origins(within("origins", n, MOST)?);
        }
        if let Some(h) = horizon {
            e = e.horizon(within("horizon", h, MOST)?);
        }
        if let Some(k) = top {
            e = e.top(k);
        }
        Ok(init(Model::of(e), Ensemble))
    }
}

/// ``model`` on the Box-Cox scale: ``lam`` = 0 is the log (the default), a
/// number is that λ, and ``"guerrero"`` chooses λ at each fit. The series
/// must be positive.
#[pyclass(extends = Model, frozen, module = "foresight")]
struct Transformed;

fn transformed(model: &Model, lam: &Bound<'_, PyAny>) -> PyResult<Model> {
    let shared = model.shared();
    if let Ok(name) = lam.extract::<String>() {
        if name.eq_ignore_ascii_case("guerrero") {
            return Ok(Model::of(fs::Transformed::auto(shared)));
        }
        return Err(PyValueError::new_err("lam must be a number or 'guerrero'"));
    }
    Ok(Model::of(fs::Transformed::box_cox(shared, lam.extract()?)))
}

#[pymethods]
impl Transformed {
    #[new]
    #[pyo3(signature = (model, lam = None))]
    fn new(
        model: PyRef<'_, Model>,
        lam: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let inner = match lam {
            Some(l) => transformed(&model, l)?,
            None => Model::of(fs::Transformed::log(model.shared())),
        };
        Ok(init(inner, Transformed))
    }
}

/// ``model`` on the log scale.
#[pyfunction]
fn log(py: Python<'_>, model: PyRef<'_, Model>) -> PyResult<Py<Transformed>> {
    Py::new(
        py,
        init(Model::of(fs::Transformed::log(model.shared())), Transformed),
    )
}

fn listed(py: Python<'_>, candidates: Vec<fs::Candidate>) -> PyResult<Vec<Py<Model>>> {
    candidates
        .into_iter()
        .map(|c| {
            Py::new(
                py,
                Model {
                    inner: Arc::from(c.model),
                    rich: None,
                    name: c.name,
                    description: c.description,
                },
            )
        })
        .collect()
}

/// The benchmarks and every model that needs no external data and fits in a
/// moment: 11 candidates.
#[pyfunction]
fn defaults(py: Python<'_>) -> PyResult<Vec<Py<Model>>> {
    listed(py, m::defaults())
}

/// :func:`defaults` plus two ensembles of them, ETS chosen automatically
/// (alone and after STL, on the original and the log scale) and ARIMA with
/// automatic orders (original and log): 18 candidates, seconds rather than
/// milliseconds in a backtest.
#[pyfunction]
fn thorough(py: Python<'_>) -> PyResult<Vec<Py<Model>>> {
    listed(py, m::thorough())
}

// ---------------------------------------------------------------- regressors

/// External variables for a regression with ARIMA errors, one column per
/// variable and one row per period from the first observation on. To
/// forecast, the rows must also cover the horizon.
#[pyclass(frozen, module = "foresight")]
struct Regressors {
    inner: fs::Regressors,
}

#[pymethods]
impl Regressors {
    /// ``columns`` maps each name to its values.
    #[new]
    #[pyo3(signature = (columns = None))]
    fn new(columns: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut r = fs::Regressors::new();
        if let Some(c) = columns {
            let mut rows = None;
            for (name, values) in c.iter() {
                let name = name.extract::<String>()?;
                let values = floats(&values)?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(PyValueError::new_err(format!(
                        "regressor {name} has values that are not finite"
                    )));
                }
                if *rows.get_or_insert(values.len()) != values.len() {
                    return Err(PyValueError::new_err("the regressors differ in length"));
                }
                r = r.with(name, values);
            }
        }
        Ok(Regressors { inner: r })
    }

    /// Sine and cosine pairs of the given ``period`` up to ``order``
    /// harmonics, for ``rows`` periods.
    #[staticmethod]
    fn fourier(period: f64, order: usize, rows: usize) -> PyResult<Self> {
        if !(period.is_finite() && period > 1.0) {
            return Err(PyValueError::new_err("period must be above 1"));
        }
        Ok(Regressors {
            inner: fs::Regressors::fourier(
                period,
                within("order", order, 1000)?,
                within("rows", rows, 10 * MOST)?,
            ),
        })
    }

    /// One dummy per season but the first, for ``rows`` periods.
    #[staticmethod]
    fn seasonal_dummies(period: usize, rows: usize) -> PyResult<Self> {
        Ok(Regressors {
            inner: fs::Regressors::seasonal_dummies(
                within("period", period, 1000)?,
                within("rows", rows, 10 * MOST)?,
            ),
        })
    }

    /// The columns of both.
    fn __add__(&self, other: PyRef<'_, Regressors>) -> Self {
        Regressors {
            inner: self.inner.clone().and(other.inner.clone()),
        }
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    fn __repr__(&self) -> String {
        format!(
            "Regressors({}, rows={})",
            self.inner.names().join(", "),
            self.inner.rows()
        )
    }
}

// ---------------------------------------------------------------- fits

/// Estimates that can be sent between threads.
enum RichFit {
    Arima(m::ArimaFit),
    Ets(m::EtsFit),
    Prophet(m::ProphetFit),
    Tbats(m::TbatsFit),
}

// one per fit: the size of the variants does not matter
#[allow(clippy::large_enum_variant)]
enum Estimate {
    Plain(Box<dyn fs::Fitted>),
    Rich(RichFit),
}

impl Estimate {
    fn fitted(&self) -> &dyn fs::Fitted {
        match self {
            Estimate::Plain(f) => f.as_ref(),
            Estimate::Rich(RichFit::Arima(f)) => f,
            Estimate::Rich(RichFit::Ets(f)) => f,
            Estimate::Rich(RichFit::Prophet(f)) => f,
            Estimate::Rich(RichFit::Tbats(f)) => f,
        }
    }
}

/// A model estimated on a series. It is plain data: it can be kept and used
/// from any thread.
#[pyclass(frozen, module = "foresight")]
struct Fit {
    model: String,
    estimate: Estimate,
}

fn params_dict<'py>(py: Python<'py>, params: &[(String, f64)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in params {
        d.set_item(k, v)?;
    }
    Ok(d)
}

#[pymethods]
impl Fit {
    /// Name of the model that was fitted.
    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    /// Point forecasts for the ``h`` periods after the last observation.
    /// ``ValueError`` when they are not finite: regressors that do not reach
    /// the horizon, for instance.
    fn forecast(&self, h: usize) -> PyResult<Vec<f64>> {
        let forecast = self.estimate.fitted().forecast(within("h", h, MOST)?);
        if forecast.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "{}: the forecast is not finite {h} periods ahead; regressors must cover the \
                 series and the horizon",
                self.model
            )));
        }
        Ok(forecast)
    }

    /// Estimated parameters by name.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        params_dict(py, &self.estimate.fitted().params())
    }

    /// Log-likelihood, for ARIMA and ETS.
    #[getter]
    fn log_likelihood(&self) -> Option<f64> {
        match &self.estimate {
            Estimate::Rich(RichFit::Arima(f)) => Some(f.log_likelihood),
            Estimate::Rich(RichFit::Ets(f)) => Some(f.log_likelihood),
            // TBATS reports −2 log-likelihood up to a constant: see details
            _ => None,
        }
    }

    #[getter]
    fn aic(&self) -> Option<f64> {
        match &self.estimate {
            Estimate::Rich(RichFit::Arima(f)) => Some(f.aic),
            Estimate::Rich(RichFit::Ets(f)) => Some(f.aic),
            Estimate::Rich(RichFit::Tbats(f)) => Some(f.aic()),
            _ => None,
        }
    }

    #[getter]
    fn aicc(&self) -> Option<f64> {
        match &self.estimate {
            Estimate::Rich(RichFit::Arima(f)) => Some(f.aicc),
            Estimate::Rich(RichFit::Ets(f)) => Some(f.aicc),
            _ => None,
        }
    }

    #[getter]
    fn bic(&self) -> Option<f64> {
        match &self.estimate {
            Estimate::Rich(RichFit::Arima(f)) => Some(f.bic),
            Estimate::Rich(RichFit::Ets(f)) => Some(f.bic),
            _ => None,
        }
    }

    /// One-step errors of the history, when the model has them.
    #[getter]
    fn residuals(&self) -> Option<Vec<f64>> {
        match &self.estimate {
            Estimate::Rich(RichFit::Arima(f)) => Some(f.residuals.clone()),
            Estimate::Rich(RichFit::Ets(f)) => Some(f.residuals.clone()),
            Estimate::Rich(RichFit::Tbats(f)) => Some(f.residuals().to_vec()),
            _ => None,
        }
    }

    /// What was chosen and estimated, beyond :attr:`params`: orders and
    /// coefficients for ARIMA, the code and smoothing for ETS, changepoints
    /// (positions from 0 at the first observation) and event effects for
    /// Prophet, the structure for TBATS and ``minus_two_log_likelihood`` (up
    /// to a constant; the lower the better, as its AIC). Empty for the other
    /// models.
    #[getter]
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        match &self.estimate {
            Estimate::Plain(_) => {}
            Estimate::Rich(RichFit::Arima(f)) => {
                d.set_item("order", f.order())?;
                d.set_item("seasonal_order", f.seasonal_order())?;
                d.set_item("period", f.period())?;
                d.set_item("ar", f.ar.clone())?;
                d.set_item("ma", f.ma.clone())?;
                d.set_item("seasonal_ar", f.seasonal_ar.clone())?;
                d.set_item("seasonal_ma", f.seasonal_ma.clone())?;
                d.set_item("constant", f.constant)?;
                d.set_item("regression", params_dict(py, &f.regression)?)?;
                d.set_item("sigma2", f.sigma2)?;
            }
            Estimate::Rich(RichFit::Ets(f)) => {
                d.set_item("code", f.model().code())?;
                d.set_item("alpha", f.alpha)?;
                d.set_item("beta", f.beta)?;
                d.set_item("gamma", f.gamma)?;
                d.set_item("phi", f.phi)?;
                d.set_item("sigma2", f.sigma2)?;
                d.set_item("fitted", f.fitted.clone())?;
                d.set_item("initial_level_and_trend", f.initial_level_and_trend())?;
                d.set_item("initial_seasonal", f.initial_seasonal().to_vec())?;
            }
            Estimate::Rich(RichFit::Prophet(f)) => {
                d.set_item("changepoints", f.changepoints())?;
                d.set_item("effects", params_dict(py, &f.effects())?)?;
                d.set_item("sigma", f.sigma())?;
                d.set_item("fitted", f.fitted())?;
            }
            Estimate::Rich(RichFit::Tbats(f)) => {
                d.set_item("seasonal", f.seasonal())?;
                d.set_item("lambda", f.lambda())?;
                d.set_item("trend", f.trend().is_some())?;
                d.set_item("damping", f.trend())?;
                d.set_item("arma", f.arma())?;
                d.set_item("smoothing", f.smoothing())?;
                d.set_item("minus_two_log_likelihood", f.likelihood())?;
            }
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!("<Fit {}>", self.model)
    }
}

/// A level as a percentage for column names: 0.8 is "80", 0.995 is "99.5".
fn percent(level: f64) -> String {
    let text = format!("{:.4}", level * 100.0);
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// A pandas DataFrame from columns in order; pandas is imported only here.
fn frame<'py>(py: Python<'py>, columns: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
    let pandas = py.import("pandas").map_err(|_| {
        pyo3::exceptions::PyImportError::new_err("to_pandas needs pandas: pip install pandas")
    })?;
    pandas.getattr("DataFrame")?.call1((columns,))
}

// ---------------------------------------------------------------- backtest

/// A forecast with its intervals.
#[pyclass(frozen, module = "foresight")]
struct Point {
    inner: fs::Point,
}

#[pymethods]
impl Point {
    /// Periods ahead (for a cumulative forecast, how many were added up).
    #[getter]
    fn horizon(&self) -> usize {
        self.inner.horizon
    }

    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    /// Coverage → (lower, upper).
    #[getter]
    fn intervals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for i in &self.inner.intervals {
            d.set_item(i.level, (i.lower, i.upper))?;
        }
        Ok(d)
    }

    /// The interval with the given coverage, e.g. 0.8.
    fn interval(&self, level: f64) -> PyResult<(f64, f64)> {
        self.inner
            .interval(level)
            .map(|i| (i.lower, i.upper))
            .ok_or_else(|| PyKeyError::new_err(format!("no interval of level {level}")))
    }

    fn __repr__(&self) -> String {
        let bands: Vec<String> = self
            .inner
            .intervals
            .iter()
            .map(|i| format!("{}%: {:.6}–{:.6}", percent(i.level), i.lower, i.upper))
            .collect();
        format!(
            "Point(h={}, mean={:.6}, {})",
            self.inner.horizon,
            self.inner.mean,
            bands.join(", ")
        )
    }
}

fn bands<'py>(py: Python<'py>, bands: &[fs::Band]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for b in bands {
        d.set_item(b.level, (b.lower, b.upper))?;
    }
    Ok(d)
}

/// Backtest and forecast of one candidate (a model or an average of
/// models).
#[pyclass(frozen, module = "foresight")]
struct Candidate {
    inner: fs::CandidateReport,
}

#[pymethods]
impl Candidate {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    /// Names of the models involved: one, or several for an average.
    #[getter]
    fn components(&self) -> Vec<String> {
        self.inner.components.clone()
    }

    /// The ranking metric averaged over the horizons (infinite when it could
    /// not be computed).
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    /// Parameters fitted on the whole series (empty for an average).
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        params_dict(py, &self.inner.params)
    }

    /// What the backtest measured at each horizon (index 0 is one period
    /// ahead): pairs evaluated, MAPE, bias, MAE, RMSE, MASE, and the
    /// quantiles of the relative error for each period and for the total.
    #[getter]
    fn horizons<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (i, h) in self.inner.horizons.iter().enumerate() {
            let d = PyDict::new(py);
            d.set_item("horizon", i + 1)?;
            d.set_item("n", h.n)?;
            d.set_item("mape", h.mape)?;
            d.set_item("bias", h.bias)?;
            d.set_item("mae", h.mae)?;
            d.set_item("rmse", h.rmse)?;
            d.set_item("mase", h.mase)?;
            d.set_item("bands", bands(py, &h.bands)?)?;
            d.set_item("cumulative", bands(py, &h.cumulative)?)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Backtest forecasts, ``[origin][h - 1]``.
    #[getter]
    fn trajectories(&self) -> Vec<Vec<f64>> {
        self.inner.trajectories.clone()
    }

    /// Forecast from the whole series, with the intervals of the backtest.
    #[getter]
    fn forecast(&self) -> Vec<Point> {
        self.inner
            .forecast
            .iter()
            .map(|p| Point { inner: p.clone() })
            .collect()
    }

    /// Forecast of the total of the next ``k`` periods, with intervals
    /// measured on totals.
    fn cumulative(&self, k: usize) -> PyResult<Point> {
        self.inner
            .cumulative(k)
            .map(|p| Point { inner: p })
            .ok_or_else(|| PyIndexError::new_err("k must be between 1 and the horizon"))
    }

    /// The forecast as a pandas DataFrame: horizon, mean and the bounds of
    /// each interval (``lower_80``, ``upper_80``…).
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = PyDict::new(py);
        let f = &self.inner.forecast;
        d.set_item("horizon", f.iter().map(|p| p.horizon).collect::<Vec<_>>())?;
        d.set_item("mean", f.iter().map(|p| p.mean).collect::<Vec<_>>())?;
        // a horizon without usable errors has no interval: NaN there
        let mut levels: Vec<f64> = Vec::new();
        for i in f.iter().flat_map(|p| &p.intervals) {
            if !levels.contains(&i.level) {
                levels.push(i.level);
            }
        }
        for level in levels {
            let pct = percent(level);
            let bound = |lower: bool| -> Vec<f64> {
                f.iter()
                    .map(|p| {
                        p.interval(level)
                            .map_or(f64::NAN, |i| if lower { i.lower } else { i.upper })
                    })
                    .collect()
            };
            d.set_item(format!("lower_{pct}"), bound(true))?;
            d.set_item(format!("upper_{pct}"), bound(false))?;
        }
        frame(py, &d)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Candidate {} score={:.4}>",
            self.inner.name, self.inner.score
        )
    }
}

/// Result of a backtest.
#[pyclass(frozen, module = "foresight")]
struct Report {
    candidates: Vec<Py<Candidate>>,
    names: Vec<String>,
    chosen: usize,
    origins: usize,
    first_origin: usize,
    horizon: usize,
    metric: String,
    dropped: Vec<String>,
}

#[pymethods]
impl Report {
    /// The models that went through the whole backtest, in the order given,
    /// then the average of the best ones.
    #[getter]
    fn candidates(&self, py: Python<'_>) -> Vec<Py<Candidate>> {
        self.candidates.iter().map(|c| c.clone_ref(py)).collect()
    }

    /// The candidate with the smallest error.
    #[getter]
    fn best(&self, py: Python<'_>) -> Py<Candidate> {
        self.candidates[self.chosen].clone_ref(py)
    }

    /// Position of the best candidate in :attr:`candidates`.
    #[getter]
    fn chosen(&self) -> usize {
        self.chosen
    }

    /// Origins actually used.
    #[getter]
    fn origins(&self) -> usize {
        self.origins
    }

    /// Names of the candidates left out: they could not forecast at some
    /// origin or from the whole series.
    #[getter]
    fn dropped(&self) -> Vec<String> {
        self.dropped.clone()
    }

    /// Position in the series, counting from 0, of the first period forecast
    /// in the backtest.
    #[getter]
    fn first_origin(&self) -> usize {
        self.first_origin
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.horizon
    }

    #[getter]
    fn metric(&self) -> String {
        self.metric.clone()
    }

    fn __len__(&self) -> usize {
        self.candidates.len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.candidates.iter().map(|c| c.clone_ref(py)))?
            .into_any()
            .try_iter()
            .map(|i| i.into_any())
    }

    /// Whether a candidate of that name is in the report.
    fn __contains__(&self, name: &Bound<'_, PyAny>) -> bool {
        name.extract::<String>()
            .is_ok_and(|n| self.names.contains(&n))
    }

    /// A candidate by name or position.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<Candidate>> {
        if let Ok(i) = key.extract::<isize>() {
            let n = self.candidates.len() as isize;
            let j = if i < 0 { i + n } else { i };
            if !(0..n).contains(&j) {
                return Err(PyIndexError::new_err("candidate index out of range"));
            }
            return Ok(self.candidates[j as usize].clone_ref(py));
        }
        let name: String = key.extract()?;
        self.names
            .iter()
            .position(|n| *n == name)
            .map(|i| self.candidates[i].clone_ref(py))
            .ok_or_else(|| PyKeyError::new_err(name))
    }

    /// One row per candidate: name, score, whether it was chosen, the
    /// models involved and the description.
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = PyDict::new(py);
        let all: Vec<_> = self.candidates.iter().map(|c| c.get()).collect();
        d.set_item("name", self.names.clone())?;
        d.set_item(
            "score",
            all.iter().map(|c| c.inner.score).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "chosen",
            (0..all.len()).map(|i| i == self.chosen).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "components",
            all.iter()
                .map(|c| c.inner.components.join(" + "))
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "description",
            all.iter()
                .map(|c| c.inner.description.clone())
                .collect::<Vec<_>>(),
        )?;
        frame(py, &d)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Report {} candidates, best {}, {} origins>",
            self.candidates.len(),
            self.names[self.chosen],
            self.origins
        )
    }
}

/// Replays the past with each candidate, forecasting from each of the last
/// ``origins`` periods up to ``horizon`` ahead, ranks the candidates by the
/// average ``metric`` (``"mape"``, ``"mae"``, ``"rmse"`` or ``"mase"``),
/// adds the simple average of the best ``combine``, and forecasts from the
/// whole series with intervals of the given ``levels`` taken from the
/// errors observed. ``candidates`` defaults to :func:`defaults`. Candidates
/// that cannot forecast at every origin and from the whole series are left
/// out, with a warning; see :attr:`Report.dropped`. With ``window``, every
/// fit uses the last ``window`` observations only.
#[pyfunction]
#[pyo3(signature = (
    y, candidates = None, *, period = None, origins = 36, horizon = 12, min_train = 48,
    window = None, combine = 2, levels = vec![0.8, 0.95], metric = "mape", parallel = true,
))]
#[allow(clippy::too_many_arguments)]
fn backtest(
    py: Python<'_>,
    y: &Bound<'_, PyAny>,
    candidates: Option<Vec<PyRef<'_, Model>>>,
    period: Option<usize>,
    origins: usize,
    horizon: usize,
    min_train: usize,
    window: Option<usize>,
    combine: usize,
    levels: Vec<f64>,
    metric: &str,
    parallel: bool,
) -> PyResult<Report> {
    let metric = match metric.to_ascii_lowercase().as_str() {
        "mape" => fs::Metric::Mape,
        "mae" => fs::Metric::Mae,
        "rmse" => fs::Metric::Rmse,
        "mase" => fs::Metric::Mase,
        _ => {
            return Err(PyValueError::new_err(
                "metric must be 'mape', 'mae', 'rmse' or 'mase'",
            ))
        }
    };
    let y = owned(y, period)?;
    let candidates: Vec<fs::Candidate> = match candidates {
        Some(c) => c.iter().map(|m| m.candidate()).collect(),
        None => m::defaults(),
    };
    if candidates.is_empty() {
        return Err(PyValueError::new_err("no candidates"));
    }
    if levels.iter().any(|l| !(*l > 0.0 && *l < 1.0)) {
        return Err(PyValueError::new_err(
            "levels must be above 0 and below 1, e.g. 0.8 for an 80% interval",
        ));
    }
    if horizon == 0 {
        return Err(PyValueError::new_err("horizon must be at least 1"));
    }
    if window == Some(0) {
        return Err(PyValueError::new_err("window must be at least 1"));
    }
    within("horizon", horizon, MOST)?;
    if y.values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "the series has values that are not finite: fill the gaps first, e.g. with clean()",
        ));
    }
    let usable = origins.min(y.values.len().saturating_sub(min_train));
    if usable < horizon {
        return Err(PyValueError::new_err(format!(
            "the series is too short: {} observations with min_train={min_train} leave {usable} \
             origins, fewer than the horizon ({horizon}); lower min_train or horizon",
            y.values.len()
        )));
    }
    let asked: Vec<String> = candidates.iter().map(|c| c.name.clone()).collect();
    let config = fs::Backtest {
        origins,
        horizon,
        min_train,
        window,
        combine,
        levels,
        metric,
        parallel,
    };
    let report = py
        .detach(|| config.run(y.view(), &candidates))
        .ok_or_else(|| {
            PyValueError::new_err(
                "no candidate could forecast at every origin and from the whole series",
            )
        })?;
    let names: Vec<String> = report.candidates.iter().map(|c| c.name.clone()).collect();
    let dropped: Vec<String> = asked.into_iter().filter(|n| !names.contains(n)).collect();
    if !dropped.is_empty() {
        let message = std::ffi::CString::new(format!(
            "left out of the backtest (no forecast at some origin or from the whole series): {}",
            dropped.join(", ")
        ))
        .unwrap_or_default();
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            &message,
            1,
        )?;
    }
    let mut out = Vec::with_capacity(report.candidates.len());
    for c in report.candidates {
        out.push(Py::new(py, Candidate { inner: c })?);
    }
    Ok(Report {
        candidates: out,
        names,
        chosen: report.chosen,
        origins: report.origins,
        first_origin: report.first_origin,
        horizon: report.horizon,
        metric: format!("{:?}", report.metric).to_lowercase(),
        dropped,
    })
}

/// Limits the threads used by backtests and ensembles, in the whole process;
/// 0, the default, stands for every core. The results do not depend on it.
#[pyfunction]
fn set_max_threads(threads: usize) -> PyResult<()> {
    fs::set_max_threads(within("threads", threads, 4096)?);
    Ok(())
}

/// The most threads a backtest or an ensemble will use.
#[pyfunction]
fn max_threads() -> usize {
    fs::max_threads()
}

// ---------------------------------------------------------------- decomposition

/// Trend, seasonal patterns and remainder.
#[pyclass(frozen, module = "foresight")]
struct Decomposition {
    inner: fs::decompose::Decomposition,
}

#[pymethods]
impl Decomposition {
    #[getter]
    fn periods(&self) -> Vec<usize> {
        self.inner.periods.clone()
    }

    #[getter]
    fn trend(&self) -> Vec<f64> {
        self.inner.trend.clone()
    }

    /// One pattern per period, in increasing order of period.
    #[getter]
    fn seasonal(&self) -> Vec<Vec<f64>> {
        self.inner.seasonal.clone()
    }

    #[getter]
    fn remainder(&self) -> Vec<f64> {
        self.inner.remainder.clone()
    }

    /// The series without its seasonal patterns.
    #[getter]
    fn seasonally_adjusted(&self) -> Vec<f64> {
        self.inner.seasonally_adjusted()
    }

    /// Strength of the trend, from 0 to 1 (Wang, Smith & Hyndman, 2006).
    #[getter]
    fn trend_strength(&self) -> f64 {
        self.inner.trend_strength()
    }

    /// Strength of the ``i``-th seasonal pattern, from 0 to 1.
    #[pyo3(signature = (i = 0))]
    fn seasonal_strength(&self, i: usize) -> PyResult<f64> {
        self.inner
            .seasonal_strength(i)
            .ok_or_else(|| PyIndexError::new_err("no such seasonal pattern"))
    }

    /// Trend, one ``seasonal_<period>`` column per pattern and remainder.
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = PyDict::new(py);
        d.set_item("trend", self.inner.trend.clone())?;
        for (p, s) in self.inner.periods.iter().zip(&self.inner.seasonal) {
            d.set_item(format!("seasonal_{p}"), s.clone())?;
        }
        d.set_item("remainder", self.inner.remainder.clone())?;
        frame(py, &d)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Decomposition periods={:?}, {} values>",
            self.inner.periods,
            self.inner.trend.len()
        )
    }
}

/// STL (Cleveland et al., 1990) for one ``period`` (that of ``y`` when it is
/// a :class:`Series`). ``seasonal_window`` is the LOESS window over the cycles
/// (odd, usually 7 or more; an even number is taken as the next odd one), or
/// ``None`` for the same pattern in every cycle. ``degrees`` are the LOESS degrees (0 or 1)
/// of the seasonal, trend and low-pass smoothers.
#[pyfunction]
#[pyo3(signature = (
    y, period = None, seasonal_window = None, *, trend_window = None, low_pass_window = None,
    degrees = None, robust = false, inner = None, outer = None,
))]
#[allow(clippy::too_many_arguments)]
fn stl(
    y: &Bound<'_, PyAny>,
    period: Option<usize>,
    seasonal_window: Option<usize>,
    trend_window: Option<usize>,
    low_pass_window: Option<usize>,
    degrees: Option<(usize, usize, usize)>,
    robust: bool,
    inner: Option<usize>,
    outer: Option<usize>,
) -> PyResult<Decomposition> {
    let y = owned(y, period)?;
    let (values, period) = (y.values, y.period);
    if let Some((a, b, c)) = degrees {
        if a > 1 || b > 1 || c > 1 {
            return Err(PyValueError::new_err("degrees must be 0 or 1"));
        }
    }
    let inner = inner.map(|n| within("inner", n, 1000)).transpose()?;
    let outer = outer.map(|n| within("outer", n, 1000)).transpose()?;
    let window = seasonal_window.map_or(SeasonalWindow::Periodic, SeasonalWindow::Span);
    let mut s = Stl::new(period, window).robust(robust);
    if let Some(n) = trend_window {
        s = s.trend_window(n);
    }
    if let Some(n) = low_pass_window {
        s = s.low_pass_window(n);
    }
    if let Some((a, b, c)) = degrees {
        s = s.degrees(a, b, c);
    }
    if inner.is_some() || outer.is_some() {
        let (i, o) = if robust { (1, 15) } else { (2, 0) };
        s = s.iterations(inner.unwrap_or(i), outer.unwrap_or(o));
    }
    s.decompose(&values)
        .map(|inner| Decomposition { inner })
        .ok_or_else(|| {
            PyValueError::new_err(
                "STL needs finite values, a period of 2 or more and a series longer than two \
                 full cycles",
            )
        })
}

/// MSTL (Bandara, Hyndman & Bergmeir, 2021): STL applied in turn to each of
/// several ``periods``.
#[pyfunction]
#[pyo3(signature = (y, periods, *, windows = None, iterations = None, robust = false))]
fn mstl(
    y: &Bound<'_, PyAny>,
    periods: Vec<usize>,
    windows: Option<Vec<usize>>,
    iterations: Option<usize>,
    robust: bool,
) -> PyResult<Decomposition> {
    let values = floats(y)?;
    let mut s = Mstl::new(&periods).robust(robust);
    if let Some(w) = windows {
        s = s.seasonal_windows(&w);
    }
    if let Some(n) = iterations {
        s = s.iterations(within("iterations", n, 100)?);
    }
    s.decompose(&values)
        .map(|inner| Decomposition { inner })
        .ok_or_else(|| {
            PyValueError::new_err(
                "MSTL needs finite values and a series longer than two cycles of a period",
            )
        })
}

// ---------------------------------------------------------------- cleaning

/// An observation that does not fit with the others: its ``index`` (from 0
/// at the first observation), its ``value`` and the ``replacement`` that the
/// neighbours and the season suggest.
#[pyclass(frozen, module = "foresight")]
struct Outlier {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    value: f64,
    #[pyo3(get)]
    replacement: f64,
}

#[pymethods]
impl Outlier {
    fn __repr__(&self) -> String {
        format!(
            "Outlier(index={}, value={}, replacement={})",
            self.index, self.value, self.replacement
        )
    }
}

/// Gaps (NaN or ``None``) filled, following the season when there is one.
#[pyfunction]
#[pyo3(signature = (values, period = None))]
fn interpolate(values: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<Vec<f64>> {
    let y = owned(values, period)?;
    fs::clean::interpolate(&y.values, y.period)
        .ok_or_else(|| PyValueError::new_err("nothing to interpolate from"))
}

/// Observations far from what the trend and the season suggest. Their
/// ``index`` counts from 0 at the first observation.
#[pyfunction]
#[pyo3(signature = (values, period = None))]
fn outliers(values: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<Vec<Outlier>> {
    let y = owned(values, period)?;
    let found = fs::clean::outliers(&y.values, y.period)
        .ok_or_else(|| PyValueError::new_err("the series is too short"))?;
    Ok(found
        .into_iter()
        .map(|o| Outlier {
            index: o.index,
            value: o.value,
            replacement: o.replacement,
        })
        .collect())
}

/// Gaps filled and outliers replaced.
#[pyfunction]
#[pyo3(signature = (values, period = None))]
fn clean(values: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<Vec<f64>> {
    let y = owned(values, period)?;
    fs::clean::clean(&y.values, y.period)
        .ok_or_else(|| PyValueError::new_err("the series is too short"))
}

// ---------------------------------------------------------------- diagnostics

/// Autocorrelations at lags 1 to ``max_lag``.
#[pyfunction]
fn acf(y: &Bound<'_, PyAny>, max_lag: usize) -> PyResult<Vec<f64>> {
    Ok(fs::diagnostics::acf(
        &floats(y)?,
        within("max_lag", max_lag, MOST)?,
    ))
}

/// Differences at the given ``lag``.
#[pyfunction]
#[pyo3(signature = (y, lag = 1))]
fn difference(y: &Bound<'_, PyAny>, lag: usize) -> PyResult<Vec<f64>> {
    Ok(fs::diagnostics::difference(&floats(y)?, lag))
}

/// KPSS statistic for level stationarity (Kwiatkowski et al., 1992); large
/// values (above 0.463 at 5%) call for a difference.
#[pyfunction]
fn kpss(y: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    Ok(fs::diagnostics::kpss(&floats(y)?))
}

/// Differences needed for stationarity by the KPSS test, at most ``max``.
#[pyfunction]
#[pyo3(signature = (y, max = 2))]
fn ndiffs(y: &Bound<'_, PyAny>, max: usize) -> PyResult<usize> {
    Ok(fs::diagnostics::ndiffs(&floats(y)?, within("max", max, 5)?))
}

/// Seasonal differences needed, by the strength of seasonality.
#[pyfunction]
#[pyo3(signature = (y, period = None))]
fn nsdiffs(y: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<usize> {
    let y = owned(y, period)?;
    Ok(fs::diagnostics::nsdiffs(&y.values, y.period))
}

/// Strength of seasonality from 0 to 1 (Wang, Smith & Hyndman, 2006).
#[pyfunction]
#[pyo3(signature = (y, period = None))]
fn seasonal_strength(y: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<Option<f64>> {
    let y = owned(y, period)?;
    Ok(fs::diagnostics::seasonal_strength(&y.values, y.period))
}

/// Box-Cox transformation with parameter ``lam`` (0 is the log).
#[pyfunction]
fn box_cox(y: &Bound<'_, PyAny>, lam: f64) -> PyResult<Vec<f64>> {
    let t = fs::BoxCox::new(lam);
    Ok(floats(y)?.into_iter().map(|v| t.apply(v)).collect())
}

/// Inverse of :func:`box_cox`.
#[pyfunction]
fn inv_box_cox(z: &Bound<'_, PyAny>, lam: f64) -> PyResult<Vec<f64>> {
    let t = fs::BoxCox::new(lam);
    Ok(floats(z)?.into_iter().map(|v| t.invert(v)).collect())
}

/// λ of the Box-Cox transformation chosen by Guerrero's method (1993).
#[pyfunction]
#[pyo3(signature = (y, period = None))]
fn guerrero(y: &Bound<'_, PyAny>, period: Option<usize>) -> PyResult<f64> {
    let y = owned(y, period)?;
    fs::BoxCox::guerrero(y.view())
        .map(|b| b.lambda())
        .ok_or_else(|| {
            PyValueError::new_err("Guerrero's method needs positive values and two cycles")
        })
}

// ---------------------------------------------------------------- accuracy

fn pair(actual: &Bound<'_, PyAny>, forecast: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (a, f) = (floats(actual)?, floats(forecast)?);
    if a.len() != f.len() {
        return Err(PyValueError::new_err(
            "actual and forecast differ in length",
        ));
    }
    Ok((a, f))
}

/// Mean absolute percentage error, in percent.
#[pyfunction]
fn mape(actual: &Bound<'_, PyAny>, forecast: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let (a, f) = pair(actual, forecast)?;
    Ok(fs::accuracy::mape(&a, &f))
}

/// Mean of (forecast − actual)/actual, in percent; positive when forecasts
/// ran high.
#[pyfunction]
fn bias(actual: &Bound<'_, PyAny>, forecast: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let (a, f) = pair(actual, forecast)?;
    Ok(fs::accuracy::bias(&a, &f))
}

/// Mean absolute error.
#[pyfunction]
fn mae(actual: &Bound<'_, PyAny>, forecast: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let (a, f) = pair(actual, forecast)?;
    Ok(fs::accuracy::mae(&a, &f))
}

/// Root mean squared error.
#[pyfunction]
fn rmse(actual: &Bound<'_, PyAny>, forecast: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let (a, f) = pair(actual, forecast)?;
    Ok(fs::accuracy::rmse(&a, &f))
}

/// Mean absolute error scaled by the in-sample seasonal naive error of
/// ``train`` (Hyndman & Koehler, 2006).
#[pyfunction]
#[pyo3(signature = (actual, forecast, train, period = None))]
fn mase(
    actual: &Bound<'_, PyAny>,
    forecast: &Bound<'_, PyAny>,
    train: &Bound<'_, PyAny>,
    period: Option<usize>,
) -> PyResult<Option<f64>> {
    let (a, f) = pair(actual, forecast)?;
    let period = match train.cast::<Series>() {
        Ok(s) if period.is_none() => s.get().period,
        _ => period.unwrap_or(1),
    };
    Ok(fs::accuracy::mase_scale(&floats(train)?, period)
        .and_then(|scale| fs::accuracy::mase(&a, &f, scale)))
}

// ---------------------------------------------------------------- module

#[pymodule]
fn _foresight(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CRATE_VERSION", "0.7.2")?;
    m.add_class::<Series>()?;
    m.add_class::<Model>()?;
    m.add_class::<Mean>()?;
    m.add_class::<Naive>()?;
    m.add_class::<Drift>()?;
    m.add_class::<SeasonalNaive>()?;
    m.add_class::<Theta>()?;
    m.add_class::<HoltWinters>()?;
    m.add_class::<LogLinear>()?;
    m.add_class::<Arima>()?;
    m.add_class::<AutoArima>()?;
    m.add_class::<Ets>()?;
    m.add_class::<AutoEts>()?;
    m.add_class::<Prophet>()?;
    m.add_class::<Tbats>()?;
    m.add_class::<Croston>()?;
    m.add_class::<Decomposed>()?;
    m.add_class::<Ensemble>()?;
    m.add_class::<Transformed>()?;
    m.add_class::<Regressors>()?;
    m.add_class::<Fit>()?;
    m.add_class::<Point>()?;
    m.add_class::<Candidate>()?;
    m.add_class::<Report>()?;
    m.add_class::<Decomposition>()?;
    m.add_class::<Outlier>()?;
    for f in [
        wrap_pyfunction!(monthly, m)?,
        wrap_pyfunction!(quarterly, m)?,
        wrap_pyfunction!(log, m)?,
        wrap_pyfunction!(defaults, m)?,
        wrap_pyfunction!(thorough, m)?,
        wrap_pyfunction!(backtest, m)?,
        wrap_pyfunction!(set_max_threads, m)?,
        wrap_pyfunction!(max_threads, m)?,
        wrap_pyfunction!(stl, m)?,
        wrap_pyfunction!(mstl, m)?,
        wrap_pyfunction!(interpolate, m)?,
        wrap_pyfunction!(outliers, m)?,
        wrap_pyfunction!(clean, m)?,
        wrap_pyfunction!(acf, m)?,
        wrap_pyfunction!(difference, m)?,
        wrap_pyfunction!(kpss, m)?,
        wrap_pyfunction!(ndiffs, m)?,
        wrap_pyfunction!(nsdiffs, m)?,
        wrap_pyfunction!(seasonal_strength, m)?,
        wrap_pyfunction!(box_cox, m)?,
        wrap_pyfunction!(inv_box_cox, m)?,
        wrap_pyfunction!(guerrero, m)?,
        wrap_pyfunction!(mape, m)?,
        wrap_pyfunction!(bias, m)?,
        wrap_pyfunction!(mae, m)?,
        wrap_pyfunction!(rmse, m)?,
        wrap_pyfunction!(mase, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}

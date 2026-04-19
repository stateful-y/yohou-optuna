---
name: create-yohou-forecaster
description: "Step-by-step guide for implementing new point or interval forecasters in Yohou. Covers BasePointForecaster, BaseIntervalForecaster, and BaseReductionForecaster patterns. Includes class templates with fit/predict, _parameter_constraints, panel data support, fitted attributes, and test file templates with _yield_yohou_forecaster_checks (27 systematic checks). Use when creating, extending, or testing any forecaster class."
---

# Creating New Forecasters

## Quick Decision Tree

- **Pattern-based/Statistical** → Extend `BasePointForecaster` in `src/yohou/point/`
- **ML-based reduction** → Extend `PointReductionForecaster` or `BaseReductionForecaster`
- **Interval forecasting** → Extend `BaseIntervalForecaster` in `src/yohou/interval/`

---

## Templates

Use these templates as starting points:

- [Forecaster class template](./forecaster_template.py) — `MyForecaster(BasePointForecaster)` with full docstrings, `_parameter_constraints`, `fit()`, `predict()`
- [Test file template](./test_forecaster_template.py) — Full test with `_yield_yohou_forecaster_checks` parametrization + specific tests

---

## Parameter Constraints

All forecasters MUST define `_parameter_constraints` for sklearn validation:

```python
_parameter_constraints: dict = {
    **ParentForecaster._parameter_constraints,
    "int_param": [Interval(numbers.Integral, 1, None, closed="left")],     # Integer ≥ 1
    "float_param": [Interval(numbers.Real, 0.0, 1.0, closed="both")],      # Float in [0, 1]
    "positive_float": [Interval(numbers.Real, 0.0, None, closed="neither")], # Float > 0
    "optional_param": [Interval(numbers.Real, 0.0, 1.0, closed="both"), None],
    "transformer_param": [BaseTransformer, None],
}
```

Validation timing: (1) automatic at fit via `@_fit_context`, (2) domain-specific in `fit()` body, (3) **never** in `__init__`.

---

## Panel Data Support

```python
def fit(self, y, X, forecasting_horizon, **params):
    y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
    if self.panel_group_names_ is not None:
        for col_name in self.local_y_schema_:
            pass  # Process each series (local_y_schema_ is dict[str, pl.DataType])
    else:
        pass  # Global data
```

---

## Fitted Attributes

All forecasters MUST set at least one fitted attribute (trailing underscore `_`) in `fit()`:

```python
def fit(self, y, X, forecasting_horizon, **params):
    y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

    # Base class sets these automatically:
    # - self._forecasting_horizon
    # - self._observation_horizon
    # - self._y_observed (last observation_horizon rows)
    # - self._X_observed (if X provided)
    # - self.panel_group_names_ (if panel data)
    # - self.local_y_schema_ (dict[str, pl.DataType], if panel data)

    # Your forecaster MUST set custom fitted attributes:
    self.model_ = ...           # Fitted model/estimator
    self.coefficients_ = ...    # Model parameters
    self.last_values_ = ...     # State for recursive prediction

    return self
```

**sklearn's `check_is_fitted()` will verify these exist before `predict()`.**

---

## Testing Forecasters

### Systematic Checks

Use `_yield_yohou_forecaster_checks` (27 checks):
- Fit attributes and fitted state validation
- Prediction time column structure
- Observe/rewind behavior
- Parameter validation
- Clone compatibility
- Tag consistency

### Tags for Generator

```python
tags = {
    "forecaster_type": "point",          # or "interval" or "both"
    "uses_reduction": False,             # True for ReductionForecasters
    "supports_panel_data": True,
    "uses_target_transformer": False,    # True if target_transformer parameter exists
    "uses_feature_transformer": False,   # True if feature_transformer parameter exists
    "tracks_observations": True,         # False for meta-forecasters delegating to children
}
```

### Available Test Fixtures

```python
def test_my_forecaster(y_X_factory):
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

def test_panel(y_X_factory):
    y, X = y_X_factory(length=100, n_targets=2, panel=True)
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/<module>/<file>.py`
2. `uvx ruff format src/yohou/<module>/<file>.py`
3. `uvx ty check src/yohou/<module>/<file>.py`
4. `uvx interrogate src/yohou/<module>/<file>.py` (docstring coverage)
5. `uv run pytest tests/<module>/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/<module>/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Missing time columns**: Always call `self._add_time_columns(y_pred)` before returning from `predict()`
- **Panel data not handled**: Check `self.panel_group_names_` and iterate `self.local_y_schema_`
- **Transformers not applied**: Use `_pre_fit()` to get transformed data (`y_t`, `X_t`), NOT raw `y`, `X`
- **No fitted attributes**: Must set at least one attribute with trailing `_` in `fit()`
- **Doctest repr mismatches**: Use exact repr format: `MyForecaster(param1=5)`
- **Mutable default args**: Use `None` and set default in method body, not `[]` or `{}`
- **Parameter validation in `__init__`**: sklearn validates at `fit()` time, NOT construction time
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

---

## Real-World Examples to Study

**Pattern-based forecasters** (no ML, statistical patterns):
- `src/yohou/point/naive.py` — SeasonalNaive (simplest example)
- `src/yohou/stationarity/seasonality.py` — Seasonality forecasters
- `src/yohou/stationarity/trend.py` — Trend forecasters

**Model-based forecasters** (ML/statistical models):
- `src/yohou/point/reduction.py` — PointReductionForecaster (sklearn integration)
- `src/yohou/interval/reduction.py` — IntervalReductionForecaster

**Meta-forecasters** (combine other forecasters):
- `src/yohou/compose/decomposition_pipeline.py` — DecompositionPipeline
- `src/yohou/compose/column_forecaster.py` — ColumnForecaster
- `src/yohou/compose/forecasted_feature_forecaster.py` — ForecastedFeatureForecaster

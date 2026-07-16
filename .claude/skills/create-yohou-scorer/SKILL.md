---
name: create-yohou-scorer
description: "Step-by-step guide for implementing new point and interval metrics (scorers) in Yohou. Covers BasePointScorer and BaseIntervalScorer with score() method pattern, aggregation methods (timewise/componentwise/groupwise/coveragewise/all), time weighting support, panel data scoring, and test templates with _yield_yohou_scorer_checks (11 check functions, variable yield count). Use when creating or testing any metric/scorer class."
---

# Creating New Scorers

## Quick Decision Tree

- **Point forecast metric** → Extend `BasePointScorer` in `src/yohou/metrics/point.py`
- **Interval forecast metric** → Extend `BaseIntervalScorer` in `src/yohou/metrics/interval.py`
- **Coverage/calibration metric** → See `src/yohou/metrics/conformity.py`

---

## Templates

- [Point scorer template](./point_scorer_template.py) — `MyMetric(BasePointScorer)` with `score()` method
- [Interval scorer template](./interval_scorer_template.py) — `MyIntervalMetric(BaseIntervalScorer)` skeleton
- [Test file template](./test_scorer_template.py) — Full test with `_yield_yohou_scorer_checks` + specific tests

**Note**: Concrete scorers implement `score()` directly (the abstract method from `BaseScorer`). The template's `_compute_scores()` is an internal organization suggestion, not a framework method.

---

## Aggregation Patterns

All scorers inherit `_aggregate_scores()` from `BaseScorer`, which handles:

1. **Time weighting**: Applies `time_weight` to per-timestep scores
2. **Hierarchical aggregation**: Based on `aggregation_method`

```python
# Scalar (fully aggregated)
metric = MyMetric(aggregation_method="all")
score = metric.score(y_truth, y_pred)  # float

# Per-component (aggregate across time)
metric = MyMetric(aggregation_method="timewise")
scores = metric.score(y_truth, y_pred)  # DataFrame with one row per component

# Per-timestep (aggregate across components)
metric = MyMetric(aggregation_method="componentwise")
scores = metric.score(y_truth, y_pred)  # DataFrame with one row per timestep

# Per-coverage (interval scorers only, aggregate across coverage rates)
metric = MyIntervalMetric(aggregation_method="coveragewise")
scores = metric.score(y_truth, y_pred)  # DataFrame aggregated per coverage rate

# Panel-only aggregation
metric = MyMetric(aggregation_method="groupwise")
scores = metric.score(y_truth, y_pred)  # DataFrame with component x timestep
```

---

## Time Weighting Support

Scorers automatically support `time_weight` via `_aggregate_scores()`:

```python
# DataFrame weight (explicit per-group weights)
time_weight = pl.DataFrame({
    "time": [...],
    "sales__store_1_weight": [1.0, 1.5, 2.0, ...],
})
score = metric.score(y_truth, y_pred, time_weight=time_weight)

# Callable weight (global, 1 parameter)
def recency_weight(y_truth):
    return pl.DataFrame({
        "time": y_truth["time"],
        "weight": np.linspace(0.5, 1.0, len(y_truth)),
    })
score = metric.score(y_truth, y_pred, time_weight=recency_weight)

# Callable weight (panel-aware, 2 parameters)
def group_specific_weight(y_truth, group_name):
    weights = {...}
    return pl.DataFrame({"time": y_truth["time"], f"{group_name}_weight": weights})
score = metric.score(y_truth, y_pred, time_weight=group_specific_weight)
```

**Key**: `_aggregate_scores()` calls `_process_time_weights()` automatically.

---

## Testing Scorers

### Systematic Checks

`_yield_yohou_scorer_checks` (11 check functions, variable yield count based on scorer type):
- Tags: `check_scorer_tags_accessible_before_fit`, `check_scorer_tags_static_after_fit`, `check_scorer_tags_match_capabilities`
- Core: `check_scorer_lower_is_better`, `check_scorer_methods_call_check_is_fitted`
- Aggregation: `check_scorer_aggregation_methods` (if aggregation_method param)
- Coverage: `check_scorer_coverage_rate_subselection` (interval only)
- Validation: `check_scorer_parameter_validation` (parametrized for panel_group_names, component_names, aggregation_method, coverage_rates)
- Standalone (not in generator): `check_scorer_prediction_type_compatibility`, `check_scorer_panel_subselection`, `check_scorer_component_subselection`

### Tags for Generator

```python
tags = {
    "prediction_type": "point",  # or "interval"
    "lower_is_better": True,     # False for accuracy-like metrics
    "requires_calibration": False,
}
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/metrics/<file>.py`
2. `uvx ruff format src/yohou/metrics/<file>.py`
3. `uvx ty check src/yohou/metrics/<file>.py`
4. `uvx interrogate src/yohou/metrics/<file>.py` (docstring coverage)
5. `uv run pytest tests/metrics/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/metrics/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Aggregation not implemented**: `_compute_scores()` must return per-timestep, per-component scores
- **Time column missing**: Scores DataFrame must include `"time"` column
- **Wrong sign for `lower_is_better`**: Error metrics (MAE, RMSE) = `True`, accuracy metrics (R²) = `False`
- **Search sign convention**: `GridSearchCV`/`RandomizedSearchCV` **negate** scores for `lower_is_better=True` scorers (sklearn convention). Scores in `cv_results_` and `best_score_` will be negative for error metrics. To recover the raw value: `raw_mae = -best_score_`. If your scorer has `lower_is_better=False` (e.g. `EmpiricalCoverage`), override `__sklearn_tags__` to set `scorer_tags.lower_is_better = False`.
- **Time weights not supported**: Don't override `_aggregate_scores()` unless necessary
- **Panel data not handled**: Base class handles panel groups automatically
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

---

## Real-World Examples to Study

**Point metrics**:
- `src/yohou/metrics/point.py` — MeanAbsoluteError, MeanSquaredError, RootMeanSquaredScaledError

**Interval metrics**:
- `src/yohou/metrics/interval.py` — EmpiricalCoverage, MeanIntervalWidth, IntervalScore, PinballLoss, CalibrationError

**Conformity metrics**:
- `src/yohou/metrics/conformity.py` — Residual, AbsoluteResidual, GammaResidual, AbsoluteGammaResidual

**Testing**:
- `tests/metrics/test_point.py`, `tests/metrics/test_interval.py`
- `src/yohou/testing/scorer.py` — Check functions (8 checks)

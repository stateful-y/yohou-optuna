---
name: create-yohou-transformer
description: "Step-by-step guide for implementing new time series transformers in Yohou. Covers stateful (observation_horizon, memory) and stateless patterns, validate_transformer_data usage, inverse_transform, panel data support, and feature name output. Includes class templates for BaseTransformer subclasses and test templates with _yield_yohou_transformer_checks (26 systematic checks). Use when creating, extending, or testing any transformer class."
---

# Creating New Transformers

## 5-Minute Quickstart

**TL;DR**: Copy the [transformer template](./transformer_template.py), modify, run tests.

```python
import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.utils.validation import check_is_fitted
from yohou.base import BaseTransformer
from yohou.utils import validate_transformer_data


class MyTransformer(BaseTransformer):
    """One-line description."""

    def __init__(self, window: StrictInt = 5):
        self.window = window

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **params):
        X = validate_transformer_data(self, X=X, reset=True)
        self._observation_horizon = self.window  # OPTIONAL: For stateful transformers
        BaseTransformer.fit(self, X, y, **params)
        self.fitted_ = True
        return self

    def transform(self, X, **params):
        check_is_fitted(self, ["fitted_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)
        return X
```

---

## Quick Decision Tree

- **Stateful** (needs observation_horizon) → Set `self._observation_horizon` in `fit()`
- **Stateless** (no memory) → Don't set `_observation_horizon`
- **Windowing/Lag features** → Extend `LagTransformer` or use `tabularize()`
- **Stationarization** → See `src/yohou/stationarity/transformers.py`

---

## Templates

- [Transformer class template](./transformer_template.py) — `MyTransformer(BaseTransformer)` with fit/transform/inverse_transform + docstrings
- [Test file template](./test_transformer_template.py) — Full test with `_yield_yohou_transformer_checks` + specific tests

---

## Critical: Always Use `validate_transformer_data`

**MANDATORY**: Call `validate_transformer_data` in ALL methods:

| Method | Call Pattern |
|--------|--------------|
| `fit()` | `X = validate_transformer_data(self, X=X, reset=True)` |
| `transform()` | `X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)` |
| `inverse_transform()` | Stateless: `X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)` |
| | Stateful: `X_t, X_p = validate_transformer_data(self, X=X, reset=False, inverse=True, X_p=X_p, observation_horizon=self.observation_horizon, stateful=True)` |

---

## Stateful vs. Stateless Transformers

### Stateless (No Memory)

**Example**: Scaling, polynomial features, Fourier features
```python
def fit(self, X, y=None, **params):
    X = validate_transformer_data(self, X=X, reset=True)
    # DO NOT set self._observation_horizon
    BaseTransformer.fit(self, X, y, **params)
    self.mean_ = X.select(~cs.by_name("time")).mean()
    return self
```

### Stateful (Needs Memory)

**Example**: Lag features, differencing, rolling windows
```python
def fit(self, X, y=None, **params):
    X = validate_transformer_data(self, X=X, reset=True)
    # CRITICAL: Set observation_horizon BEFORE calling BaseTransformer.fit()
    self._observation_horizon = 5
    BaseTransformer.fit(self, X, y, **params)
    return self

def transform(self, X, **params):
    combined = pl.concat([self._X_observed, X])  # Prepend memory
    X_t = ...
    return X_t
```

---

## Memory Management: `observe()` and `rewind()`

```python
transformer.observe(X_new)   # Appends then calls rewind()
transformer.rewind(X_all)    # Keeps last N rows based on observation_horizon
# Memory is in self._X_observed (always == observation_horizon after rewind())
```

---

## Panel Data Support

Transformers automatically support panel data (prefixed columns). No special handling needed unless custom logic required.

---

## Feature Names for Output

Implement `get_feature_names_out()` if transformation changes column count/names:

```python
from sklearn.utils.validation import _check_feature_names_in

def get_feature_names_out(self, input_features=None):
    input_features = _check_feature_names_in(self, input_features)
    return [f"{col}_lag_{self.lag}" for col in input_features]
```

---

## Testing Transformers

### Systematic Checks

Use `_yield_yohou_transformer_checks` (26 checks):
- `check_fit_sets_attributes` - fit() creates required attributes
- `check_observation_horizon_not_fitted` - horizon not accessible before fit
- `check_observation_horizon_after_fit` - horizon correct after fit
- `check_transform_output_structure` - output structure preserved
- `check_feature_names_out_match` - feature names match output
- `check_transformers_unfitted_stateless` - unfitted raises appropriately
- `check_transformer_methods_call_check_is_fitted` - methods validate fitted state
- `check_tags_accessible_before_fit` / `check_tags_static_after_fit` / `check_tags_match_capabilities` - tag system
- `check_rewind_transform_behavior` - rewind_transform works correctly
- Stateful: `check_observe_concatenates_memory`, `check_observe_transform_equivalence`, `check_rewind_updates_memory`, `check_memory_bounded`, `check_insufficient_data_raises`, `check_observe_transform_sequential_consistency`
- Invertible: `check_inverse_transform_identity`, `check_inverse_transform_round_trip`, `check_inverse_observe_transform_identity`
- `check_transformer_preserve_dtypes`, `check_fit_idempotent`, `check_fit_transform_equivalence`
- Panel: `check_panel_data_support`
- Metadata: `check_metadata_routing_default_request`, `check_metadata_routing_get_metadata_routing`

### Available Test Fixtures

```python
def test_my_func(time_series_factory):
    X = time_series_factory(length=100, n_components=2, seed=42)

def test_panel(panel_time_series_factory):
    X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

def test_split(time_series_train_test_factory):
    X_train, X_test = time_series_train_test_factory(train_length=80, test_length=20)
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/preprocessing/<file>.py`
2. `uvx ruff format src/yohou/preprocessing/<file>.py`
3. `uvx ty check src/yohou/preprocessing/<file>.py`
4. `uvx interrogate src/yohou/preprocessing/<file>.py` (docstring coverage)
5. `uv run pytest tests/preprocessing/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/preprocessing/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Time column missing**: Always preserve `"time"` column in output
- **Stateful without observation_horizon**: Must set `self._observation_horizon` in `fit()` BEFORE calling `BaseTransformer.fit()`
- **Memory not used**: Stateful transformers should use `self._X_observed` in `transform()`
- **No fitted attributes**: Must set at least one attribute with trailing `_`
- **BaseTransformer.**init**() not needed**: BaseTransformer inherits from `BaseEstimator` but has no custom `__init__`. Your subclass only needs to define `__init__` with its own params (no need to call `super().__init__()`)
- **Validation order wrong**: Call `validate_transformer_data()` BEFORE setting `_observation_horizon`
- **Panel data broken**: Ensure transformation works with prefixed columns
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

---

## Real-World Examples to Study

**Stateless transformers**:
- `src/yohou/stationarity/transformers.py` - Detrending

**Stateful transformers** (with observation_horizon):
- `src/yohou/preprocessing/window.py` - LagTransformer (windowing/tabularization)

**Testing**:
- `tests/preprocessing/test_window.py` - Comprehensive transformer tests
- `src/yohou/testing/transformer.py` - Check functions for systematic testing

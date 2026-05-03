---
name: create-yohou-search
description: "Guide for extending hyperparameter search in Yohou (advanced). Covers custom search strategy implementation extending BaseSearchCV, with the _run_search abstract method pattern. Includes built-in GridSearchCV/RandomizedSearchCV usage examples and test templates with _yield_yohou_search_checks (19 systematic checks). Most users should use built-in search classes — extend only for novel strategies like Bayesian optimization."
---

# Creating New Hyperparameter Search Classes

**Note**: This is an advanced use case. Most users should use the built-in `GridSearchCV` and `RandomizedSearchCV` classes.

---

## When to Extend Search Classes

Consider extending only if:
- You need custom sampling strategies (e.g., Bayesian optimization, genetic algorithms)
- You want to add time series-specific early stopping or pruning
- You're implementing ensemble-based hyperparameter tuning

**For most use cases**, use built-in classes with custom scorers or splitters instead.

---

## Templates

- [Search class template](./search_template.py) — `MyCustomSearch(BaseSearchCV)` with full fit/predict/sampling logic
- [Test file template](./test_search_template.py) — Full test with `_yield_yohou_search_checks` + specific tests

---

## Built-In Search Classes

### GridSearchCV

```python
from yohou.model_selection import GridSearchCV
from yohou.point import PointReductionForecaster
from yohou.metrics import MeanAbsoluteError

search = GridSearchCV(
    forecaster=PointReductionForecaster(),
    param_grid={
        "estimator__alpha": [0.1, 1.0, 10.0],
        "estimator__fit_intercept": [True, False],
    },
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5),
)
search.fit(y, X_actual=X, forecasting_horizon=3)
print(search.best_params_)
```

### RandomizedSearchCV

```python
from yohou.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

search = RandomizedSearchCV(
    forecaster=PointReductionForecaster(),
    param_distributions={
        "estimator__alpha": uniform(0.01, 10.0),
        "estimator__max_iter": randint(100, 1000),
    },
    n_iter=20,
    scoring=MeanAbsoluteError(),
)
search.fit(y, X_actual=X, forecasting_horizon=3)
```

---

## Testing Search Classes

### Systematic Checks

`_yield_yohou_search_checks` (19 checks):
- Fit/predict workflow verification
- best_params_, best_score_, best_forecaster_ attributes
- CV results structure
- Parameter validation
- Clone compatibility
- Observe/predict on best forecaster
- Tag consistency

---

## Common Pitfalls

- **Not using time series CV**: Always use `ExpandingWindowSplitter` or `SlidingWindowSplitter`, NOT `KFold`
- **Refitting on wrong data**: Refit best model on full training data, not last CV fold
- **Ignoring scorer direction**: Check `scoring.lower_is_better` to determine best score
- **Metadata routing issues**: Pass `**params` through to forecaster methods
- **Parallel execution complexity**: Time series CV is sequential by nature
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

---

## Recommendation

**For 99% of use cases**: Use `GridSearchCV` or `RandomizedSearchCV` with custom `scoring`, `cv`, and `param_grid`/`param_distributions`.

**Only extend if**: You're implementing novel search strategies (e.g., Bayesian optimization, SMAC, Optuna integration).

---

## Real-World Examples to Study

**Built-in search classes**:
- `src/yohou/model_selection/search.py` — GridSearchCV, RandomizedSearchCV

**Testing**:
- `tests/model_selection/test_search.py` — Search tests
- `src/yohou/testing/search.py` — Check functions (19 checks)

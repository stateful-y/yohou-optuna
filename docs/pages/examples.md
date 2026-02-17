# Examples

Explore real-world applications of Yohou-Optuna through interactive examples.

## What can Yohou-Optuna do?

### Quickstart Search

A minimal end-to-end example: load data, define distributions, run `OptunaSearchCV`, and inspect results. This is the best starting point for new users.

[:material-notebook: optuna_search.py](../examples/optuna_search/)

### Composed Forecaster Tuning

Tune nested parameters in a `DecompositionPipeline` — optimize the decomposer, trend forecaster, and residual forecaster simultaneously using `__` parameter routing.

[:material-notebook: composed_tuning.py](../examples/composed_tuning/)

### Search Visualization

Visualize optimization history, parameter importances, and parallel coordinate plots using Optuna's built-in visualization tools with `search.study_`.

[:material-notebook: search_visualization.py](../examples/search_visualization/)

### Multi-Metric Search

Evaluate multiple scoring metrics (MAE, RMSE, etc.) in a single search pass. Learn how to use `refit` to select the best forecaster based on your preferred metric.

[:material-notebook: multi_metric_search.py](../examples/multi_metric_search/)

### Samplers and Persistence

Compare different Optuna samplers (TPE, CMA-ES, Random) and persist studies to a SQLite database for resumption and analysis.

[:material-notebook: samplers_and_persistence.py](../examples/samplers_and_persistence/)

## Running Examples Locally

All examples are [Marimo](https://marimo.io) reactive notebooks stored as `.py` files:

```bash
# Interactive editing (recommended)
uv run marimo edit examples/optuna_search.py

# Run as script (non-interactive)
uv run marimo run examples/optuna_search.py

# Or use the justfile shortcut
just example optuna_search.py
```

## Next Steps

- Browse the [API Reference](api-reference.md) for detailed documentation
- Check the [User Guide](user-guide.md) to understand core concepts

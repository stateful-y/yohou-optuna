# Examples

Explore real-world applications of Yohou-Optuna through interactive examples. Each notebook uses real datasets from `yohou.datasets` and visualizations from `yohou.plotting`.

## What can Yohou-Optuna do?

### Quickstart Search

Load the Air Passengers dataset, define search distributions, run `OptunaSearchCV`, and visualize results with `plot_cv_results_scatter` and `plot_forecast`. The best starting point for new users.

[:material-notebook: optuna_search.py](../examples/optuna_search/)

### Composed Forecaster Tuning

Tune nested parameters in a `PointReductionForecaster` with `LagTransformer` -- use autocorrelation analysis to motivate lag selection, optimize with `ExpandingWindowSplitter`, and diagnose residuals. Uses the Sunspots dataset.

[:material-notebook: composed_tuning.py](../examples/composed_tuning/)

### Multi-Metric Search

Evaluate multiple scoring metrics (MAE, RMSE, MSE) in a single search pass. Compare rankings across metrics with `plot_model_comparison_bar` and visualize multivariate data from the ETT-M1 dataset.

[:material-notebook: multi_metric_search.py](../examples/multi_metric_search/)

### Search Visualization

Combine Optuna's built-in optimization plots (history, importances, contour) with yohou's forecast diagnostics (`plot_cv_results_scatter`, `plot_forecast`, `plot_residual_time_series`). Uses Victoria Electricity data.

[:material-notebook: search_visualization.py](../examples/search_visualization/)

### Panel Data Tuning

Tune forecasters on grouped time series from the Australian Tourism dataset. Visualize CV splits with `plot_splits`, compare Random vs TPE samplers, and use `MaxTrialsCallback` for early stopping.

[:material-notebook: panel_tuning.py](../examples/panel_tuning/)

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

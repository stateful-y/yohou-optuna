# Examples

Learn Yohou-Optuna through focused, interactive examples. Each notebook demonstrates one core workflow using real datasets from `yohou.datasets` and visualizations from `yohou.plotting`. Examples are organized from basic usage to advanced patterns and are runnable and editable locally or online.

## Getting Started

### Quickstart Search ([View](/examples/optuna_search/) | [Editable](/examples/optuna_search/edit/))

**Your First Hyperparameter Search**

Start here to understand the fundamental `OptunaSearchCV` workflow. This example loads the Air Passengers dataset, defines search distributions for Ridge regression parameters, runs a complete hyperparameter search, and visualizes results. You'll learn the three essential steps every search needs: defining parameter distributions, fitting the search, and inspecting results with `plot_cv_results_scatter` and `plot_forecast`.

## Core Workflows

### Composed Forecaster Tuning ([View](/examples/composed_tuning/) | [Editable](/examples/composed_tuning/edit/))

**Tuning Nested Parameters in Reduction Forecasters**

Dive into tuning composed estimators by optimizing both a Ridge regressor and its `LagTransformer` feature pipeline. This example uses the Sunspots dataset and autocorrelation analysis to motivate lag selection, then searches over nested parameters with `ExpandingWindowSplitter` cross-validation. You'll see how the double-underscore syntax (`feature_transformer__lag`) reaches into nested components, and how to diagnose results with residual plots.

### Multi-Metric Search ([View](/examples/multi_metric_search/) | [Editable](/examples/multi_metric_search/edit/))

**Evaluating Multiple Scoring Metrics Simultaneously**

Evaluate MAE, RMSE, and MSE in a single search pass instead of running separate searches for each metric. This example uses multivariate data from the ETT-M1 dataset and demonstrates how different metrics can rank the same trials differently. You'll compare the best trial selected by each metric using `plot_model_comparison_bar`, and understand when `refit` matters for choosing the final forecaster.

## Advanced Topics

### Search Visualization ([View](/examples/search_visualization/) | [Editable](/examples/search_visualization/edit/))

**Combining Optuna and Yohou Visualization**

Explore the full diagnostic toolkit by combining Optuna's built-in optimization plots (history, parameter importances, contour) with yohou's forecast diagnostics (`plot_cv_results_scatter`, `plot_forecast`, `plot_residual_time_series`). This example uses Victoria Electricity demand data and shows how the two visualization ecosystems complement each other -- Optuna for understanding the search process, yohou for evaluating forecast quality.

### Panel Data Tuning ([View](/examples/panel_tuning/) | [Editable](/examples/panel_tuning/edit/))

**Hyperparameter Search on Grouped Time Series**

Apply `OptunaSearchCV` to panel data from the Australian Tourism dataset, where multiple related time series are tuned jointly. This example visualizes cross-validation splits with `plot_splits`, compares Random and TPE sampling strategies side by side, and demonstrates `MaxTrialsCallback` for early stopping. Essential for understanding how yohou-optuna handles real-world datasets with hierarchical structure.

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

- **[User Guide](user-guide.md)**: Deep dive into core concepts and architecture
- **[API Reference](api-reference.md)**: Complete OptunaSearchCV documentation
- **[Contributing](contributing.md)**: Add your own examples or improve existing ones

# How-to Guides

Task-oriented guides for common workflows. Each guide assumes you have already
completed the [Getting Started](../tutorials/getting-started.md) tutorial.

## Optimization

- [Configure OptunaSearchCV](configure.md): choose samplers, add callbacks, persist studies, and customize cross-validation
- [Prune Unpromising Trials](prune-trials.md): stop a trial after a few backtest folds when it is reliably worse than the best one, and choose which folds run first
- [Score Multiple Metrics](multi-metric-search.md): evaluate configurations against multiple scoring metrics simultaneously
- [Visualize Search Results](visualize-study.md): plot optimization history and parameter importance with Optuna and yohou diagnostics

## Composition

- [Tune Composed Forecasters](composed-forecasters.md): tune nested parameters across pipelines, decomposition steps, and column-wise forecasters
- [Tune on Panel Data](panel-data.md): optimize forecasters on grouped time series with panel strategies

## Debugging

- [Troubleshooting](troubleshooting.md): solutions to common problems with installation, search, parameter routing, and storage

## Project

- [Contributing](contribute.md): set up a development environment, run tests, and submit changes

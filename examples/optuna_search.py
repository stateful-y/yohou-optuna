# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optuna",
#     "polars",
#     "scikit-learn",
#     "yohou",
#     "yohou-optuna",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

__gallery__ = {
    "title": "OptunaSearchCV Quickstart",
    "description": "Your first hyperparameter search with OptunaSearchCV on the Air Passengers dataset.",
    "category": "tutorial",
    "companion": "pages/tutorials/getting-started.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OptunaSearchCV Quickstart

        In this notebook, we will run a Bayesian hyperparameter search on the
        Air Passengers dataset using
        [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/).
        We will load the data, define a search space, fit the search, inspect
        cross-validation results, and generate a forecast with the best model.

        **Prerequisites**: basic familiarity with scikit-learn's fit/predict API and time series forecasting concepts.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import optuna
    import polars as pl
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import load_air_passengers
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_cv_results_scatter, plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster

    from yohou_optuna import OptunaSearchCV, Sampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        CategoricalDistribution,
        FloatDistribution,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        Sampler,
        load_air_passengers,
        optuna,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load and Explore the Data

        Let's start with the Air Passengers dataset - 144 monthly observations of
        international airline passenger counts from 1949 to 1960.
        """
    )
    return


@app.cell
def _(load_air_passengers, plot_time_series):
    y = load_air_passengers()
    plot_time_series(y, title="Air Passengers (1949-1960)")
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We split into training (first 120 months) and test (last 24 months) sets.
        """
    )
    return


@app.cell
def _(y):
    y_train = y.head(120)
    y_test = y.tail(24)
    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Define Forecaster and Search Space

        We wrap a Ridge regressor in a `PointReductionForecaster` and define
        distributions for `alpha` and `fit_intercept`. The TPE sampler will
        guide the search through this space.
        """
    )
    return


@app.cell
def _(
    CategoricalDistribution,
    FloatDistribution,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    optuna,
):
    param_distributions = {
        "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        "estimator__fit_intercept": CategoricalDistribution([True, False]),
    }

    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=20,
        cv=3,
        refit=True,
        verbose=0,
    )
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Run the Search

        Let's fit the search on the training data.
        """
    )
    return


@app.cell
def _(search, y_train):
    search.fit(y_train, forecasting_horizon=12)
    return


@app.cell
def _(mo, search):
    mo.md(
        f"""
        **Best parameters:** `{search.best_params_}`

        **Best score (MAE):** `{search.best_score_:.4f}`

        **Number of trials:** `{len(search.trials_)}`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Inspect Cross-Validation Results

        Let's look at how regularization strength relates to the MAE score.
        """
    )
    return


@app.cell
def _(pl, search):
    results_df = pl.DataFrame(
        {
            "trial": list(range(len(search.cv_results_["params"]))),
            "alpha": [p.get("estimator__alpha", None) for p in search.cv_results_["params"]],
            "fit_intercept": [p.get("estimator__fit_intercept", None) for p in search.cv_results_["params"]],
            "mean_test_score": search.cv_results_["mean_test_score"],
            "std_test_score": search.cv_results_["std_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    ).sort("rank")
    results_df
    return


@app.cell
def _(plot_cv_results_scatter, search):
    plot_cv_results_scatter(
        search.cv_results_,
        param_name="estimator__alpha",
        higher_is_better=False,
        title="CV Score vs Regularization Strength",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Forecast with the Best Model

        Since we set `refit=True`, the best forecaster is already fitted on the
        full training data. Let's generate predictions and compare against the
        held-out test set.
        """
    )
    return


@app.cell
def _(search):
    y_pred = search.predict(forecasting_horizon=12)
    y_pred
    return (y_pred,)


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=36,
        title="Best Forecaster: Predicted vs Actual",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What We Built

        We ran a complete hyperparameter search: loaded the Air Passengers
        dataset, defined a search space over Ridge parameters, ran 20
        Bayesian-optimized trials with `OptunaSearchCV`, inspected
        cross-validation rankings, and generated a forecast with the best model.

        ## Next Steps

        - [How to Tune Composed Forecasters](/examples/composed_tuning/): tune forecasters with feature transformers like `LagTransformer`
        - [How to Run a Multi-Metric Search](/examples/multi_metric_search/): track multiple metrics (MAE, RMSE, MSE) simultaneously
        - [How to Visualize Search Results](/examples/search_visualization/): use Optuna's built-in optimization plots
        """
    )
    return


if __name__ == "__main__":
    app.run()

"""Test configuration and fixtures for Yohou-Optuna."""

from datetime import datetime, timedelta

import numpy as np
import optuna
import polars as pl
import pytest
from optuna.distributions import FloatDistribution
from sklearn.linear_model import Ridge
from yohou.base import BaseForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.point import PointReductionForecaster

from yohou_optuna import OptunaSearchCV, Sampler

# Suppress noisy optuna logs during tests
optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def y_X_factory():
    """Factory for generating (y, X) tuples for forecaster testing.

    Returns a callable that generates polars DataFrames with a ``"time"``
    column and numeric target/feature columns.

    Returns
    -------
    callable
        Factory function accepting length, n_targets, n_features, seed,
        panel, and n_groups parameters.

    """

    def _factory(length=100, n_targets=2, n_features=3, seed=42, panel=False, n_groups=2):
        """Generate forecaster test data.

        Parameters
        ----------
        length : int
            Number of time steps.
        n_targets : int
            Number of target columns.
        n_features : int
            Number of feature columns (0 for None).
        seed : int
            Random seed.
        panel : bool
            Whether to create panel data with __ separator.
        n_groups : int
            Number of panel groups when panel=True.

        Returns
        -------
        y : pl.DataFrame
            Target data with "time" column.
        X : pl.DataFrame or None
            Features with "time" column, or None if n_features=0.

        """
        rng = np.random.default_rng(seed)

        time_col = pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        if panel:
            y = pl.DataFrame({"time": time_col})
            for i in range(n_targets):
                base_values = rng.random(length)
                for group_idx in range(n_groups):
                    variation = group_idx * 0.1
                    col_name = f"y_{i}__group_{group_idx}"
                    y = y.with_columns(pl.Series(col_name, base_values + variation))

            X = None
            if n_features > 0:
                X = pl.DataFrame({"time": time_col})
                for i in range(n_features):
                    base_values = rng.random(length)
                    for group_idx in range(n_groups):
                        variation = group_idx * 0.05
                        col_name = f"X_{i}__group_{group_idx}"
                        X = X.with_columns(pl.Series(col_name, base_values + variation))
        else:
            y = pl.DataFrame({"time": time_col})
            for i in range(n_targets):
                y = y.with_columns(pl.Series(f"y_{i}", rng.random(length)))

            X = None
            if n_features > 0:
                X = pl.DataFrame({"time": time_col})
                for i in range(n_features):
                    X = X.with_columns(pl.Series(f"X_{i}", rng.random(length)))

        return y, X

    return _factory


@pytest.fixture
def default_forecaster():
    """Create a default PointReductionForecaster for testing.

    Uses Ridge estimator which has an alpha parameter suitable for
    Optuna distribution-based search.

    Returns
    -------
    PointReductionForecaster
        A forecaster instance with Ridge estimator.

    """
    return PointReductionForecaster(estimator=Ridge())


@pytest.fixture
def default_param_distributions():
    """Create default Optuna parameter distributions for testing.

    Returns
    -------
    dict
        Dictionary mapping parameter names to Optuna distributions.

    """
    return {
        "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
    }


@pytest.fixture
def default_scorer():
    """Create a default MeanAbsoluteError scorer for testing.

    Returns
    -------
    MeanAbsoluteError
        A scorer instance.

    """
    return MeanAbsoluteError()


@pytest.fixture
def default_sampler():
    """Create a deterministic Optuna sampler for reproducible tests.

    Returns
    -------
    Sampler
        A wrapped TPESampler with fixed seed.

    """
    return Sampler(sampler=optuna.samplers.TPESampler, seed=42)


@pytest.fixture
def optuna_search_cv(default_forecaster, default_param_distributions, default_scorer, default_sampler):
    """Create a default OptunaSearchCV instance for testing.

    Parameters
    ----------
    default_forecaster : PointReductionForecaster
        Forecaster to optimize.
    default_param_distributions : dict
        Parameter distributions.
    default_scorer : MeanAbsoluteError
        Scoring function.
    default_sampler : Sampler
        Deterministic sampler.

    Returns
    -------
    OptunaSearchCV
        Configured search instance.

    """
    return OptunaSearchCV(
        forecaster=default_forecaster,
        param_distributions=default_param_distributions,
        scoring=default_scorer,
        sampler=default_sampler,
        n_trials=3,
        cv=2,
        refit=True,
    )


class FailingForecaster(BaseForecaster):
    """A mock forecaster that always raises during fit or predict.

    Parameters
    ----------
    fail_on : str
        Method that should raise. One of ``"fit"``, ``"predict"``, or
        ``"both"``.
    exception_cls : type
        Exception class to raise.

    """

    _parameter_constraints: dict = {
        **BaseForecaster._parameter_constraints,
        "fail_on": [str],
        "exception_cls": "no_validation",
    }

    def __init__(self, fail_on="fit", exception_cls=ValueError):
        super().__init__()
        self.fail_on = fail_on
        self.exception_cls = exception_cls

    def fit(self, y, X=None, forecasting_horizon=1, **fit_params):
        """Raise if fail_on includes fit.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int, default=1
            Forecast horizon.
        **fit_params : dict
            Additional parameters.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``fail_on`` is ``"fit"`` or ``"both"``.

        """
        if self.fail_on in ("fit", "both"):
            raise self.exception_cls("FailingForecaster intentional error in fit")
        self._y_train = y
        self.is_fitted_ = True
        return self

    def predict(self, forecasting_horizon=None, X=None, **predict_params):
        """Raise if fail_on includes predict.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps to forecast.
        X : pl.DataFrame or None, default=None
            Exogenous features.
        **predict_params : dict
            Additional parameters.

        Returns
        -------
        pl.DataFrame

        Raises
        ------
        ValueError
            If ``fail_on`` is ``"predict"`` or ``"both"``.

        """
        if self.fail_on in ("predict", "both"):
            raise self.exception_cls("FailingForecaster intentional error in predict")
        return self._y_train.tail(forecasting_horizon)


@pytest.fixture
def failing_forecaster():
    """Create a FailingForecaster that raises during fit.

    Returns
    -------
    FailingForecaster
        A forecaster that always raises ValueError in fit.

    """
    return FailingForecaster(fail_on="fit")


@pytest.fixture
def interval_forecaster():
    """Create an IntervalReductionForecaster for interval prediction testing.

    Returns
    -------
    IntervalReductionForecaster
        A forecaster that supports interval prediction.

    """
    from sklearn.linear_model import QuantileRegressor
    from yohou.interval import IntervalReductionForecaster

    return IntervalReductionForecaster(estimator=QuantileRegressor())


@pytest.fixture
def large_param_distributions():
    """Create parameter distributions with many parameters for stress testing.

    Returns
    -------
    dict
        Dictionary mapping parameter names to Optuna distributions.

    """
    from optuna.distributions import CategoricalDistribution

    return {
        "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
        "estimator__fit_intercept": CategoricalDistribution([True, False]),
        "estimator__copy_X": CategoricalDistribution([True, False]),
        "estimator__positive": CategoricalDistribution([True, False]),
    }

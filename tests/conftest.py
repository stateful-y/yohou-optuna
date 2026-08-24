"""Test configuration and fixtures for Yohou-Optuna."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import optuna
import polars as pl
import pytest
from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase
from optuna.distributions import FloatDistribution
from sklearn.linear_model import Ridge
from yohou.base import BaseForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import PointReductionForecaster

from yohou_optuna import OptunaSearchCV, Sampler

# Suppress noisy optuna logs during tests
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Hypothesis remembers failing examples so a rerun replays them first. That example
# database defaults to `.hypothesis/` at the repo root; this puts it under
# `.artifacts/` with every other piece of throwaway output. It has no config-file
# key, so registering and loading a profile is the only way to set it -- which is why
# this lives here rather than in pyproject.toml.
#
# This moves the example database ONLY. Hypothesis also writes a `.hypothesis/`
# storage directory for its own constants and unicode caches, which no setting
# relocates. Newer versions drop a self-ignoring `.gitignore` inside it and older
# ones do not, so `.gitignore` lists it explicitly rather than depending on which
# version resolved.
settings.register_profile("default", database=DirectoryBasedExampleDatabase(".artifacts/hypothesis"))
settings.load_profile("default")


def run_checks(
    estimator: Any,
    checks: Generator[tuple[str, Any, dict], None, None],
    *,
    expected_failures: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Run all checks from a generator, collecting and reporting all failures.

    Unlike a simple for-loop, this function does **not** stop at the first
    failure. All checks are executed and a single ``pytest.fail`` is raised
    at the end summarising every unexpected failure (and every expected
    failure that unexpectedly passed).

    Parameters
    ----------
    estimator : object
        Fitted estimator instance passed as the first positional argument
        to each check function.
    checks : generator of (str, callable, dict)
        Output of a ``_yield_yohou_*_checks`` generator.
    expected_failures : set of str, optional
        Check names that are expected to fail.  Unexpected passes are
        reported alongside unexpected failures.

    """
    failures: list[str] = []
    xfail_passed: list[str] = []

    for check_name, check_func, check_kwargs in checks:
        passes_estimator = (
            "splitter" in check_kwargs
            or "splitter_class" in check_kwargs
            or "scorer" in check_kwargs
            or "scorer_class" in check_kwargs
        )

        try:
            if passes_estimator:
                check_func(**check_kwargs)
            else:
                check_func(estimator, **check_kwargs)
        except Exception as exc:
            if check_name in expected_failures:
                continue
            failures.append(f"  {check_name}: {type(exc).__name__}: {exc}")
        else:
            if check_name in expected_failures:
                xfail_passed.append(check_name)

    messages: list[str] = []
    if failures:
        messages.append(f"{len(failures)} check(s) failed:\n" + "\n".join(failures))
    if xfail_passed:
        xfail_lines = "\n".join(f"  {name}" for name in xfail_passed)
        messages.append(f"{len(xfail_passed)} expected failure(s) unexpectedly passed:\n" + xfail_lines)

    if messages:
        pytest.fail("\n\n".join(messages))


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
            # Convention: <entity>__<variable> (e.g., group_0__y_0)
            y = pl.DataFrame({"time": time_col})
            for group_idx in range(n_groups):
                for i in range(n_targets):
                    base_values = rng.random(length)
                    variation = group_idx * 0.1
                    col_name = f"group_{group_idx}__y_{i}"
                    y = y.with_columns(pl.Series(col_name, base_values + variation))

            X = None
            if n_features > 0:
                X = pl.DataFrame({"time": time_col})
                for group_idx in range(n_groups):
                    for i in range(n_features):
                        base_values = rng.random(length)
                        variation = group_idx * 0.05
                        col_name = f"group_{group_idx}__X_{i}"
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

    def fit(self, y, X_actual=None, forecasting_horizon=1, **fit_params):
        """Raise if fail_on includes fit.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None, default=None
            Actual observation features.
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

    def predict(self, X_future=None, X_forecast=None, forecasting_horizon=None, **predict_params):
        """Raise if fail_on includes predict.

        Parameters
        ----------
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        forecasting_horizon : int or None, default=None
            Number of steps to forecast.
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


class ThresholdFailingForecaster(PointReductionForecaster):
    """A real reduction forecaster whose fit fails when the training series is short.

    With an expanding-window cross-validation, earlier folds train on less
    data, so a threshold strictly between two folds' training lengths fails
    exactly the earlier folds and passes the later ones. That makes partial
    fold failure deterministic without any call counting. Folds above the
    threshold fit and score as the parent class, so this mock exercises the
    real scoring path rather than a stub.

    Parameters
    ----------
    estimator : object or None
        Passed to ``PointReductionForecaster``.
    min_length : int
        Minimum training length below which ``fit`` raises ValueError.

    """

    _parameter_constraints: dict = {
        **PointReductionForecaster._parameter_constraints,
        "min_length": [int],
    }

    def __init__(self, estimator=None, min_length=50):
        super().__init__(estimator=estimator)
        self.min_length = min_length

    def fit(self, y, X_actual=None, forecasting_horizon=1, **fit_params):
        """Raise below ``min_length``, otherwise fit as the parent class.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None, default=None
            Actual observation features.
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
            If the training series is shorter than ``min_length``.

        """
        if len(y) < self.min_length:
            msg = f"training series of length {len(y)} is shorter than {self.min_length}"
            raise ValueError(msg)
        return super().fit(y, X_actual=X_actual, forecasting_horizon=forecasting_horizon, **fit_params)


@pytest.fixture
def threshold_failing_forecaster():
    """Create a ThresholdFailingForecaster with its default threshold.

    Tests set the effective threshold per trial through the search's
    parameter distributions, so the default here is irrelevant.

    Returns
    -------
    ThresholdFailingForecaster
        A forecaster that raises in fit below a training-length threshold.

    """
    return ThresholdFailingForecaster(estimator=Ridge())


class MarkerRecordingSplitter(ExpandingWindowSplitter):
    """An expanding-window splitter whose ``split`` accepts a metadata key.

    The extra ``split_marker`` parameter on ``split`` makes the key
    requestable through sklearn's signature scraping, so after
    ``set_split_request(split_marker=True)`` a search routes it via the
    splitter bucket. Every ``split`` call records the received marker in
    ``split_calls_``, which lets a test assert the routed value reached each
    per-trial split. ``get_n_splits`` accepts the key as well because the
    search forwards the same bucket there.

    Parameters
    ----------
    n_splits : int, default=3
        Number of expanding-window folds.
    max_train_size : int or None, default=None
        As in ``ExpandingWindowSplitter``.
    test_size : int or None, default=None
        As in ``ExpandingWindowSplitter``.

    """

    def __init__(self, n_splits=3, *, max_train_size=None, test_size=None):
        super().__init__(n_splits, max_train_size=max_train_size, test_size=test_size)
        self.split_calls_: list = []

    def split(self, y, X_actual=None, split_marker=None):
        """Record the marker, then split as the parent class.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None, default=None
            Actual features.
        split_marker : object, default=None
            Requestable metadata key recorded in ``split_calls_``.

        Yields
        ------
        train : ndarray
            Training set row indices for that split.
        test : ndarray
            Test set row indices for that split.

        """
        self.split_calls_.append(split_marker)
        yield from super().split(y, X_actual)

    def get_n_splits(self, y=None, X_actual=None, split_marker=None):
        """Return the fold count, accepting the routed key.

        Parameters
        ----------
        y : pl.DataFrame or None, default=None
            Not used.
        X_actual : pl.DataFrame or None, default=None
            Not used.
        split_marker : object, default=None
            Accepted and ignored; the search forwards the splitter bucket
            here as well as to ``split``.

        Returns
        -------
        int
            The number of cross-validation folds.

        """
        return super().get_n_splits(y, X_actual)


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
def class_proba_forecaster():
    """Create a ClassProbaReductionForecaster for class probability testing.

    Returns
    -------
    ClassProbaReductionForecaster
        A forecaster that supports class probability prediction.

    """
    from sklearn.linear_model import LogisticRegression
    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.compose import FeaturePipeline
    from yohou.preprocessing import LagTransformer

    return ClassProbaReductionForecaster(
        estimator=LogisticRegression(),
        reduction_strategy="direct",
        actual_transformer=FeaturePipeline([("lag", LagTransformer(lag=[1, 2, 3]))]),
    )


@pytest.fixture
def y_class_proba_factory():
    """Factory for generating binary classification time series data.

    Returns a callable that generates a polars DataFrame with a ``"time"``
    column and binary class columns suitable for class probability
    forecasters.

    Returns
    -------
    callable
        Factory function accepting length and seed parameters.

    """

    def _factory(length=100, seed=42):
        rng = np.random.default_rng(seed)
        time_col = pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )
        classes = rng.choice([0.0, 1.0], size=length)
        y = pl.DataFrame({"time": time_col, "label": classes})
        return y

    return _factory


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

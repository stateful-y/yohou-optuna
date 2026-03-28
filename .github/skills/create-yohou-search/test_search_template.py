"""Tests for <Name> search class.

Tests <Name> using both the check generator pattern and search-specific tests.
"""

import pytest
from sklearn.base import clone
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import MyCustomSearch
from yohou.point import PointReductionForecaster
from yohou.testing import _yield_yohou_search_checks


@pytest.mark.parametrize(
    "search,expected_failures",
    [
        (
            MyCustomSearch(
                forecaster=PointReductionForecaster(),
                param_space={"estimator__alpha": [0.1, 1.0]},
                scoring=MeanAbsoluteError(),
            ),
            [],
        ),
    ],
    ids=["default"],
)
def test_search_systematic_checks(search, expected_failures, y_X_factory):
    """Run all applicable checks for search."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = X[:80], X[80:]

    # Fit search
    search_fitted = clone(search)
    search_fitted.fit(y_train, X_train, forecasting_horizon=5)

    # Run all checks from generator
    expected_failures_set = set(expected_failures)

    for check_name, check_func, check_kwargs in _yield_yohou_search_checks(
        search_fitted, y_train, X_train, y_test, X_test
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(search_fitted, **check_kwargs)


def test_search_basic_fit(y_X_factory):
    """Test basic fit behavior."""
    y, X = y_X_factory(length=100)

    search = MyCustomSearch(
        forecaster=PointReductionForecaster(),
        param_space={"estimator__alpha": [0.1, 1.0, 10.0]},
        scoring=MeanAbsoluteError(),
    )
    search.fit(y, X, forecasting_horizon=5)

    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert hasattr(search, "best_forecaster_")


def test_search_predict(y_X_factory):
    """Test predict with best forecaster."""
    y, X = y_X_factory(length=100)

    search = MyCustomSearch(
        forecaster=PointReductionForecaster(),
        param_space={"estimator__alpha": [0.1, 1.0]},
        scoring=MeanAbsoluteError(),
    )
    search.fit(y[:80], X[:80], forecasting_horizon=5)
    y_pred = search.predict(forecasting_horizon=5)

    assert len(y_pred) == 5
    assert "time" in y_pred.columns


def test_search_best_params_valid(y_X_factory):
    """Test that best_params_ contains valid parameter values."""
    y, X = y_X_factory(length=100)
    param_space = {"estimator__alpha": [0.1, 1.0, 10.0]}

    search = MyCustomSearch(
        forecaster=PointReductionForecaster(),
        param_space=param_space,
        scoring=MeanAbsoluteError(),
    )
    search.fit(y, X, forecasting_horizon=5)

    assert search.best_params_["estimator__alpha"] in param_space["estimator__alpha"]


def test_search_cv_results(y_X_factory):
    """Test cv_results_ structure."""
    y, X = y_X_factory(length=100)

    search = MyCustomSearch(
        forecaster=PointReductionForecaster(),
        param_space={"estimator__alpha": [0.1, 1.0]},
        scoring=MeanAbsoluteError(),
    )
    search.fit(y, X, forecasting_horizon=5)

    assert hasattr(search, "cv_results_")
    assert len(search.cv_results_) >= 2


def test_search_clone(y_X_factory):
    """Test that search can be cloned."""
    y, X = y_X_factory(length=100)

    search = MyCustomSearch(
        forecaster=PointReductionForecaster(),
        param_space={"estimator__alpha": [0.1, 1.0]},
        scoring=MeanAbsoluteError(),
    )
    search.fit(y, X, forecasting_horizon=5)

    cloned = clone(search)
    assert not hasattr(cloned, "best_params_")


def test_search_get_set_params():
    """Test get_params and set_params work correctly."""
    search = MyCustomSearch(
        forecaster=PointReductionForecaster(),
        param_space={"estimator__alpha": [0.1]},
        scoring=MeanAbsoluteError(),
    )

    params = search.get_params()
    assert "forecaster" in params
    assert "scoring" in params

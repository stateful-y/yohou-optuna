"""Tests for <Name> forecaster.

Tests <Name> using both the check generator pattern and forecaster-specific tests.
"""

import pytest
from sklearn.base import clone
from yohou.point import MyForecaster
from yohou.testing import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,expected_failures",
    [
        (MyForecaster(param1=1), []),
        (MyForecaster(param1=5), []),
    ],
    ids=["param1_1", "param1_5"],
)
def test_forecaster_systematic_checks(
    forecaster,
    expected_failures,
    y_X_factory,
):
    """Run all applicable checks for forecaster."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    # Split for train/test
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = X[:80], X[80:] if X is not None else (None, None)

    # Fit forecaster
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X_train, forecasting_horizon=5)

    # Run all checks from generator
    expected_failures_set = set(expected_failures)
    tags = {"forecaster_type": "point", "uses_reduction": False}

    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted, y_train, X_train, y_test, X_test, tags=tags
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)


def test_forecaster_basic_fit_predict(y_X_factory):
    """Test basic fit and predict workflow."""
    y, X = y_X_factory(length=100)
    forecaster = MyForecaster(param1=1)
    forecaster.fit(y[:80], X_actual=X[:80] if X else None, forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)

    # Predictions have required columns
    assert "time" in y_pred.columns
    assert "observed_time" in y_pred.columns
    assert len(y_pred) == 5


def test_forecaster_observe_predict(y_X_factory):
    """Test observe_predict workflow."""
    y, X = y_X_factory(length=100)
    forecaster = MyForecaster(param1=1)
    forecaster.fit(y[:80], X_actual=X[:80] if X else None, forecasting_horizon=5)

    # Observe new data and predict
    y_pred = forecaster.observe_predict(y[80:85], X_actual=X[80:85] if X else None)
    assert len(y_pred) == 5


def test_forecaster_panel_data(y_X_factory):
    """Test forecaster handles panel data (prefixed columns)."""
    y, X = y_X_factory(length=100, n_targets=2, seed=42, panel=True)
    forecaster = MyForecaster(param1=1)
    forecaster.fit(y[:80], X_actual=X[:80] if X else None, forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)

    # Predictions have all panel groups
    assert forecaster.panel_group_names_ is not None
    assert len(y_pred) == 5


def test_forecaster_clone(y_X_factory):
    """Test that forecaster can be cloned."""
    y, X = y_X_factory(length=100)
    forecaster = MyForecaster(param1=1)
    forecaster.fit(y[:80], forecasting_horizon=5)

    cloned = clone(forecaster)
    # Cloned should not be fitted
    from sklearn.utils.validation import check_is_fitted

    with pytest.raises(Exception):  # noqa: B017
        check_is_fitted(cloned)


def test_forecaster_get_set_params():
    """Test get_params and set_params work correctly."""
    forecaster = MyForecaster(param1=5)

    params = forecaster.get_params()
    assert params["param1"] == 5

    forecaster.set_params(param1=10)
    assert forecaster.get_params()["param1"] == 10

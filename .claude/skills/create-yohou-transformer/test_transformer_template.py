"""Tests for <Name> transformers.

Tests <Name> using both the check generator pattern and transformer-specific tests.
"""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from yohou.preprocessing import MyTransformer
from yohou.testing import _yield_yohou_transformer_checks


@pytest.mark.parametrize(
    "transformer,expected_failures",
    [
        (MyTransformer(param1=1), []),
        (MyTransformer(param1=5), []),
    ],
    ids=["param1_1", "param1_5"],
)
def test_transformer_systematic_checks(
    transformer,
    expected_failures,
    time_series_train_test_factory,
):
    """Run all applicable checks for transformer."""
    # Generate continuous train/test data
    min_horizon = transformer._observation_horizon if hasattr(transformer, "_observation_horizon") else 10
    X_train, X_test = time_series_train_test_factory(train_length=min_horizon + 50, test_length=min_horizon + 20)

    # Get tags and adjust data if needed
    tags = transformer.__sklearn_tags__()
    if tags.input_tags and tags.input_tags.min_value is not None:
        offset = max(0.0, tags.input_tags.min_value + 1.0)
        X_train = X_train.select([pl.col("time"), (pl.all().exclude("time") + offset)])
        X_test = X_test.select([pl.col("time"), (pl.all().exclude("time") + offset)])

    # Fit transformer
    transformer_fitted = clone(transformer)
    transformer_fitted.fit(X_train)

    # Run all checks from generator
    expected_failures_set = set(expected_failures)

    for check_name, check_func, check_kwargs in _yield_yohou_transformer_checks(
        transformer_fitted, X_train, None, X_test
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(transformer_fitted, **check_kwargs)


def test_my_transformer_basic(time_series_factory):
    """Test basic fit/transform behavior."""
    X = time_series_factory(length=50, n_components=2)
    transformer = MyTransformer(param1=1)
    transformer.fit(X)

    X_trans = transformer.transform(X)

    # Time column should be preserved
    assert "time" in X_trans.columns
    assert_frame_equal(X_trans.select(pl.col("time")), X.select(pl.col("time")))


def test_my_transformer_inverse_transform(time_series_factory):
    """Test inverse_transform recovers original data (if invertible)."""
    X = time_series_factory(length=50, n_components=2)
    transformer = MyTransformer(param1=1)
    transformer.fit(X)

    X_trans = transformer.transform(X)
    X_recovered = transformer.inverse_transform(X_trans)

    for col in X.columns:
        if col == "time":
            continue
        np.testing.assert_allclose(
            X_recovered[col].to_numpy(),
            X[col].to_numpy(),
            rtol=1e-5,
        )


def test_my_transformer_observation_horizon(time_series_factory):
    """Test observation_horizon is correctly set."""
    X = time_series_factory(length=50)
    transformer = MyTransformer(param1=5)
    transformer.fit(X)

    # Verify observation_horizon matches expected value
    expected_horizon = 5  # Based on your transformer logic
    assert transformer.observation_horizon == expected_horizon


def test_my_transformer_with_panel_data(panel_time_series_factory):
    """Test transformer handles panel data (prefixed columns)."""
    X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

    transformer = MyTransformer(param1=1)
    transformer.fit(X)
    X_trans = transformer.transform(X)

    # Time column preserved, all other columns present
    assert "time" in X_trans.columns
    assert len(X_trans.columns) == len(X.columns)


def test_my_transformer_clone(time_series_factory):
    """Test that transformer can be cloned."""
    X = time_series_factory(length=50)
    transformer = MyTransformer(param1=1)
    transformer.fit(X)

    cloned = clone(transformer)
    # Cloned should not be fitted
    assert not hasattr(cloned, "fitted_attr_") or cloned.fitted_attr_ is None


def test_my_transformer_get_set_params():
    """Test get_params and set_params work correctly."""
    transformer = MyTransformer(param1=5)

    params = transformer.get_params()
    assert params["param1"] == 5

    transformer.set_params(param1=10)
    assert transformer.get_params()["param1"] == 10

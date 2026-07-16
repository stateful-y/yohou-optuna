"""Tests for <Name> scorer.

Tests <Name> using both the check generator pattern and scorer-specific tests.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from yohou.metrics import MyMetric
from yohou.testing import _yield_yohou_scorer_checks


@pytest.fixture
def y_truth_pred():
    """Generate sample truth and prediction DataFrames."""
    from datetime import datetime, timedelta

    base_time = datetime(2020, 1, 1)
    n = 10

    y_truth = pl.DataFrame({
        "time": [base_time + timedelta(days=i) for i in range(n)],
        "value": [float(i) for i in range(n)],
    })

    y_pred = pl.DataFrame({
        "observed_time": [base_time - timedelta(days=1)] * n,
        "time": [base_time + timedelta(days=i) for i in range(n)],
        "value": [float(i) + 0.5 for i in range(n)],  # Slightly off
    })

    return y_truth, y_pred


@pytest.mark.parametrize(
    "scorer,expected_failures",
    [
        (MyMetric(), []),
        (MyMetric(aggregation_method="timewise"), []),
    ],
    ids=["default", "timewise"],
)
def test_scorer_systematic_checks(scorer, expected_failures, y_truth_pred):
    """Run all applicable checks for scorer."""
    y_truth, y_pred = y_truth_pred

    # Run all checks from generator
    expected_failures_set = set(expected_failures)
    tags = {"prediction_type": "point", "lower_is_better": True}

    for check_name, check_func, check_kwargs in _yield_yohou_scorer_checks(scorer, y_truth, y_pred, tags=tags):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(scorer, **check_kwargs)


def test_scorer_basic_score(y_truth_pred):
    """Test basic scoring behavior."""
    y_truth, y_pred = y_truth_pred
    metric = MyMetric()

    score = metric.score(y_truth, y_pred)
    assert isinstance(score, float)
    assert score >= 0  # For error metrics


def test_scorer_lower_is_better(y_truth_pred):
    """Test lower_is_better property."""
    metric = MyMetric()
    assert metric.lower_is_better is True  # For error metrics


def test_scorer_aggregation_timewise(y_truth_pred):
    """Test timewise aggregation returns per-component DataFrame."""
    y_truth, y_pred = y_truth_pred
    metric = MyMetric(aggregation_method="timewise")

    result = metric.score(y_truth, y_pred)
    assert isinstance(result, pl.DataFrame)


def test_scorer_aggregation_componentwise(y_truth_pred):
    """Test componentwise aggregation returns per-timestep DataFrame."""
    y_truth, y_pred = y_truth_pred
    metric = MyMetric(aggregation_method="componentwise")

    result = metric.score(y_truth, y_pred)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(y_truth)


def test_scorer_with_time_weight(y_truth_pred):
    """Test scoring with time weights."""
    y_truth, y_pred = y_truth_pred

    # Create time weight DataFrame
    time_weight = pl.DataFrame({
        "time": y_truth["time"],
        "weight": np.linspace(0.5, 1.0, len(y_truth)),
    })

    metric = MyMetric()
    score_weighted = metric.score(y_truth, y_pred, time_weight=time_weight)
    score_unweighted = metric.score(y_truth, y_pred)

    # Weighted score should differ
    assert score_weighted != score_unweighted


def test_scorer_clone(y_truth_pred):
    """Test that scorer can be cloned."""
    metric = MyMetric(aggregation_method="timewise")
    cloned = clone(metric)

    assert cloned.aggregation_method == metric.aggregation_method


def test_scorer_get_set_params():
    """Test get_params and set_params work correctly."""
    metric = MyMetric(aggregation_method="all")

    params = metric.get_params()
    assert params["aggregation_method"] == "all"

    metric.set_params(aggregation_method="timewise")
    assert metric.get_params()["aggregation_method"] == "timewise"

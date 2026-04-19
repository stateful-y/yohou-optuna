"""Tests for <Name> splitter.

Tests <Name> using both the check generator pattern and splitter-specific tests.
"""

import pytest
from sklearn.base import clone
from yohou.model_selection import MySplitter
from yohou.testing import _yield_yohou_splitter_checks


@pytest.mark.parametrize(
    "splitter,expected_failures",
    [
        (MySplitter(n_splits=3), []),
        (MySplitter(n_splits=5, test_size=10), []),
    ],
    ids=["default", "custom_test_size"],
)
def test_splitter_systematic_checks(
    splitter,
    expected_failures,
    time_series_factory,
):
    """Run all applicable checks for splitter."""
    y = time_series_factory(length=100, n_components=1, seed=42)
    X = time_series_factory(length=100, n_components=2, seed=43)

    # Run all checks from generator
    expected_failures_set = set(expected_failures)
    tags = {"splitter_type": "expanding", "supports_panel_data": False}

    for check_name, check_func, check_kwargs in _yield_yohou_splitter_checks(splitter, y, X, tags=tags):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(**check_kwargs)


def test_splitter_basic_split(time_series_factory):
    """Test basic split behavior."""
    y = time_series_factory(length=100)
    splitter = MySplitter(n_splits=3, test_size=10)

    splits = list(splitter.split(y))
    assert len(splits) == 3

    for train_idx, test_idx in splits:
        assert len(test_idx) == 10
        assert len(train_idx) > 0
        # No overlap
        assert len(set(train_idx) & set(test_idx)) == 0


def test_splitter_temporal_order(time_series_factory):
    """Test that splits maintain temporal order."""
    y = time_series_factory(length=100)
    splitter = MySplitter(n_splits=3)

    for train_idx, test_idx in splitter.split(y):
        # All train indices should be before all test indices
        assert max(train_idx) < min(test_idx)


def test_splitter_non_overlapping_tests(time_series_factory):
    """Test that test sets across folds don't overlap."""
    y = time_series_factory(length=100)
    splitter = MySplitter(n_splits=3)

    test_sets = [set(test_idx) for _, test_idx in splitter.split(y)]
    for i, test1 in enumerate(test_sets):
        for test2 in test_sets[i + 1 :]:
            assert len(test1 & test2) == 0


def test_splitter_get_n_splits(time_series_factory):
    """Test get_n_splits matches actual split count."""
    y = time_series_factory(length=100)
    splitter = MySplitter(n_splits=3)

    expected = splitter.get_n_splits(y)
    actual = len(list(splitter.split(y)))
    assert expected == actual


def test_splitter_clone():
    """Test that splitter can be cloned."""
    splitter = MySplitter(n_splits=3, test_size=10)
    cloned = clone(splitter)

    assert cloned.n_splits == 3
    assert cloned.test_size == 10


def test_splitter_get_set_params():
    """Test get_params and set_params work correctly."""
    splitter = MySplitter(n_splits=3)

    params = splitter.get_params()
    assert params["n_splits"] == 3

    splitter.set_params(n_splits=5)
    assert splitter.get_params()["n_splits"] == 5

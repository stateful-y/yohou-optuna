"""Tests for my_dataset loading and data quality."""

import polars as pl
from yohou.datasets import load_my_dataset


def test_load_my_dataset():
    """Test that load_my_dataset returns a valid Polars DataFrame."""
    df = load_my_dataset()

    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0


def test_my_dataset_has_time_column():
    """Test that the dataset has 'time' column with datetime type."""
    df = load_my_dataset()

    assert "time" in df.columns
    assert df["time"].dtype.is_temporal()


def test_my_dataset_columns():
    """Test expected column names."""
    df = load_my_dataset()

    expected_columns = ["time", "value"]
    assert df.columns == expected_columns


def test_my_dataset_sorted_by_time():
    """Test that data is sorted by time."""
    df = load_my_dataset()

    assert df["time"].is_sorted()


def test_my_dataset_no_null_time():
    """Test no null values in time column."""
    df = load_my_dataset()

    assert df["time"].null_count() == 0


def test_my_dataset_no_duplicate_times():
    """Test no duplicate timestamps."""
    df = load_my_dataset()

    assert df["time"].n_unique() == len(df)


def test_my_dataset_consistent_intervals():
    """Test that time intervals are consistent."""
    df = load_my_dataset()

    from yohou.utils.validation import check_interval_consistency

    interval = check_interval_consistency(df)
    assert interval is not None


def test_my_dataset_numeric_columns():
    """Test that non-time columns are numeric."""
    df = load_my_dataset()

    for col in df.columns:
        if col != "time":
            assert df[col].dtype.is_numeric(), f"Column {col} is not numeric"


# def test_my_dataset_panel_naming():
#     """Test panel data naming convention (prefix__suffix)."""
#     df = load_my_dataset()
#
#     panel_cols = [c for c in df.columns if "__" in c]
#     assert len(panel_cols) > 0
#
#     # Verify consistent prefixes
#     prefixes = {c.split("__")[0] for c in panel_cols}
#     assert len(prefixes) >= 1

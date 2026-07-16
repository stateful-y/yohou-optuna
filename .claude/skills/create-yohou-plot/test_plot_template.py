"""Tests for <Module> plotting functions.

Tests plotting functions for correct figure output and parameter handling.
"""

import plotly.graph_objects as go
import polars as pl
import pytest
from yohou.plotting import plot_my_visualization


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "value1": range(12),
        "value2": [x * 2 for x in range(12)],
    })


@pytest.fixture
def panel_data():
    """Generate sample panel data."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "sales__store_1": range(12),
        "sales__store_2": [x * 2 for x in range(12)],
    })


def test_plot_returns_figure(sample_data):
    """Test that plot returns a Plotly figure."""
    fig = plot_my_visualization(sample_data, columns="value1")

    assert isinstance(fig, go.Figure)
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_plot_with_single_column(sample_data):
    """Test plotting a single column."""
    fig = plot_my_visualization(sample_data, columns="value1")

    assert len(fig.data) == 1
    assert fig.data[0].name == "value1"


def test_plot_with_multiple_columns(sample_data):
    """Test plotting multiple columns."""
    fig = plot_my_visualization(sample_data, columns=["value1", "value2"])

    assert len(fig.data) == 2


def test_plot_with_none_columns(sample_data):
    """Test plotting all numeric columns when columns=None."""
    fig = plot_my_visualization(sample_data, columns=None)

    # Should plot all numeric columns except "time"
    assert len(fig.data) == 2  # value1 and value2


def test_plot_custom_title(sample_data):
    """Test custom title is applied."""
    fig = plot_my_visualization(sample_data, columns="value1", title="Custom Title")

    assert fig.layout.title.text == "Custom Title"


def test_plot_custom_dimensions(sample_data):
    """Test custom width and height."""
    fig = plot_my_visualization(sample_data, columns="value1", width=800, height=600)

    assert fig.layout.width == 800
    assert fig.layout.height == 600


def test_plot_hide_legend(sample_data):
    """Test legend can be hidden."""
    fig = plot_my_visualization(sample_data, columns="value1", show_legend=False)

    assert all(trace.showlegend is False for trace in fig.data)


def test_plot_custom_colors(sample_data):
    """Test custom color palette."""
    custom_palette = ["#FF0000", "#00FF00"]
    fig = plot_my_visualization(
        sample_data,
        columns=["value1", "value2"],
        color_palette=custom_palette,
    )

    assert fig.data[0].line.color == "#FF0000"


def test_plot_panel_data(panel_data):
    """Test plotting panel data with panel_group_names."""
    fig = plot_my_visualization(panel_data, panel_group_names=["sales"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_plot_panel_invalid_group(panel_data):
    """Test ValueError for non-existent panel group."""
    with pytest.raises(ValueError, match="No panel columns found"):
        plot_my_visualization(panel_data, panel_group_names=["nonexistent"])


def test_plot_raises_on_invalid_type():
    """Test error when input is not a DataFrame."""
    with pytest.raises(TypeError):
        plot_my_visualization([1, 2, 3])


def test_plot_raises_on_empty_dataframe():
    """Test error when DataFrame is empty."""
    empty_df = pl.DataFrame({"time": []})
    with pytest.raises(ValueError, match="empty"):
        plot_my_visualization(empty_df)


def test_plot_raises_on_missing_time_column():
    """Test error when 'time' column is missing."""
    df = pl.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ValueError, match="time"):
        plot_my_visualization(df)


def test_plot_raises_on_nonexistent_column(sample_data):
    """Test error when specified column doesn't exist."""
    with pytest.raises(ValueError, match="not found"):
        plot_my_visualization(sample_data, columns="nonexistent")

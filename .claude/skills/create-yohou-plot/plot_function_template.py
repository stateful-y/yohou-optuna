"""Module docstring (e.g., 'Core time series plotting functions')."""

import plotly.graph_objects as go
import polars as pl
from yohou.plotting._utils import (
    apply_default_layout,
    resolve_color_palette,
)
from yohou.utils import validate_plotting_data


def plot_my_visualization(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    facet_scales: str = "free_y",
    dropdown: bool = False,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Create custom time series visualization.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, plots all numeric columns except 'time'.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to include.  If None, plot as global.
    facet_n_cols : int, default=2
        Number of columns in the facet grid when using panel groups.
    facet_scales : str, default="free_y"
        Subplot axis sharing ("free_y", "free_x", "free", "fixed").
    dropdown : bool, default=False
        If True, render panel groups in a dropdown instead of facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show legend.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to "time".
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters (line_width, marker_size, etc.).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If df is not a Polars DataFrame.
    ValueError
        If DataFrame is empty, missing 'time' column, or specified columns don't exist.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_my_visualization

    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "value": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })

    >>> fig = plot_my_visualization(df, columns="value")
    >>> len(fig.data)  # One trace
    1
    """
    # 1. Validate input DataFrame and resolve columns
    columns_to_plot = validate_plotting_data(df, columns=columns, exclude=["time"])

    # 2. Get color palette
    colors = resolve_color_palette(color_palette, n=len(columns_to_plot))

    # 4. Create figure
    fig = go.Figure()

    # 5. Add traces for each column
    for i, col in enumerate(columns_to_plot):
        line_width = kwargs.get("line_width", 2.0)
        line_dash = kwargs.get("line_dash", "solid")

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[col],
                mode="lines",
                name=col,
                line={
                    "color": colors[i],
                    "width": line_width,
                    "dash": line_dash,
                },
                showlegend=show_legend,
            )
        )

    # 6. Apply default layout
    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label or "time",
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig

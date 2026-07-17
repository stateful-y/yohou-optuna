---
name: create-yohou-plot
description: "Guide for creating new Plotly-based plotting functions in Yohou. Covers the plotting module structure, validate_plotting_data/resolve_color_palette/validate_plotting_params/apply_default_layout utilities, panel data faceting/dropdown patterns, consistent parameter naming conventions, and test templates for figure validation. Use when adding any new visualization function to src/yohou/plotting/."
---

# Creating New Plotting Functions

## Overview

Yohou's plotting module uses **Plotly** for interactive visualizations. All plotting functions:
- Accept polars DataFrames as input
- Return `plotly.graph_objects.Figure` objects
- Follow a unified parameter naming convention
- Support panel data via `panel_group_names` parameter
- Use the yohou color palette by default

**Location**: `src/yohou/plotting/`

---

## Module Organization

```text
src/yohou/plotting/
├── __init__.py           # Public API exports (__all__ with 26 symbols)
├── _utils.py             # Validation, color, layout, panel helpers (private)
├── exploration.py        # plot_time_series, plot_rolling_statistics, plot_boxplot, plot_missing_data
├── diagnostics.py        # plot_autocorrelation, plot_partial_autocorrelation, plot_correlation_heatmap, plot_seasonality, plot_subseasonality, plot_lag_scatter, plot_scatter_matrix, plot_cross_correlation, plot_stl_components
├── forecasting.py        # plot_forecast, plot_components, plot_time_weight
├── evaluation.py         # plot_calibration, plot_model_comparison_bar, plot_residual_time_series, plot_score_distribution, plot_score_per_horizon, plot_score_time_series
├── model_selection.py    # plot_cv_results_scatter, plot_splits
└── signal.py             # plot_spectrum, plot_phase
```

---

## Templates

- [Plot function template](./plot_function_template.py) — `plot_my_visualization()` with full parameter set + validation + layout
- [Test file template](./test_plot_template.py) — Full test with basic/params/panel/error tests

---

## Key Utilities

All helpers live in `src/yohou/plotting/_utils.py`.

### Input Validation

```python
from yohou.utils import validate_plotting_data

# Validates DataFrame structure AND resolves columns in one call
columns_to_plot = validate_plotting_data(df, exclude=["time"])  # None → all numeric (excl. "time")
columns_to_plot = validate_plotting_data(df, columns="y")       # Specific column(s)
columns_to_plot = validate_plotting_data(df, panel_group_names=["sales"])  # Panel columns
```

### Parameter Validation

```python
from yohou.plotting._utils import validate_plotting_params

validate_plotting_params(kind=kind, valid_kinds={"line", "bar"})  # Validates kind
validate_plotting_params(facet_n_cols=facet_n_cols, n_bins=n_bins)  # Validates integers
```

### Prediction Normalization

```python
from yohou.plotting._utils import _normalize_y_pred

y_pred_dict = _normalize_y_pred(y_pred)  # DataFrame|dict → dict[str, DataFrame]
```

### Color Management

```python
from yohou.plotting._utils import resolve_color_palette

colors = resolve_color_palette(color_palette, n=5)  # None → Yohou palette, or cycles custom
colors = resolve_color_palette(None, n=3)            # Always returns exactly n colors
```

### Layout Styling

```python
from yohou.plotting._utils import apply_default_layout

fig = apply_default_layout(fig, title="My Plot", x_label="Time", y_label="Value")
```

---

## Panel Data Support Pattern

Use the `panel_facet_figure()` helper with a render callback:

```python
from yohou.plotting._utils import (
    panel_facet_figure,
    resolve_color_palette,
    validate_plotting_params,
)
from yohou.utils import validate_plotting_data

def plot_panel_aware(
    df,
    *,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    facet_scales: str = "free_y",
    dropdown: bool = False,
    **kwargs,
):
    columns_to_plot = validate_plotting_data(df, exclude=["time"])

    if panel_group_names is not None:
        # Resolve matching panel columns (raises ValueError if none found)
        panel_cols = validate_plotting_data(df, panel_group_names=panel_group_names)

        def _render(fig, sub_df, display_name, panel_idx, row, col):
            """Render callback – called once per panel column."""
            fig.add_trace(
                go.Scatter(x=sub_df["time"], y=sub_df[display_name], name=display_name),
                row=row,
                col=col,
            )

        return panel_facet_figure(
            df,
            _render,
            panel_group_names=panel_group_names,
            facet_n_cols=facet_n_cols,
            title="My Panel Plot",
        )

    # Non-panel (global) case
    ...
```

**Key points**:
- Parameter name is `panel_group_names` (plural, list of group prefixes)
- `panel_facet_figure()` handles subplot grid creation and iteration
- The render callback receives pre-inspected `sub_df`, `display_name`, and grid position
- Use `validate_plotting_data(df, panel_group_names=...)` to validate groups exist before plotting

---

## Styling Conventions

```python
# Line plots
line_width: float = 2.0
line_dash: str = "solid"  # "solid", "dash", "dot", "dashdot"

# Markers
marker_size: float = 8.0
marker_symbol: str = "circle"

# Colors
color_palette: list[str] | None = None  # None = yohou palette

# Hover templates
hovertemplate="<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:.2f}<br><extra></extra>"
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/plotting/<file>.py`
2. `uvx ruff format src/yohou/plotting/<file>.py`
3. `uvx ty check src/yohou/plotting/<file>.py`
4. `uvx interrogate src/yohou/plotting/<file>.py` (docstring coverage)
5. `uv run pytest tests/plotting/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/plotting/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Missing time column check**: Always use `validate_plotting_data()` instead of manual checks
- **Not using yohou palette**: Use `resolve_color_palette()` for consistent colors
- **Inconsistent parameter names**: Follow convention (e.g., `show_legend`, not `display_legend`)
- **No panel data support**: Consider adding `panel_group_names` parameter
- **Mutating input DataFrame**: Always operate on copies or return new DataFrames
- **Missing doctest examples**: All public functions need runnable examples
- **Not applying default layout**: Use `apply_default_layout()` for consistent styling
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

---

## Real-World Examples to Study

**Core visualizations**:
- `src/yohou/plotting/exploration.py` — `plot_time_series()`, `plot_rolling_statistics()`, `plot_boxplot()`
- `src/yohou/plotting/forecasting.py` — `plot_forecast()`, `plot_components()`, `plot_time_weight()`
- `src/yohou/plotting/diagnostics.py` — `plot_autocorrelation()`, `plot_seasonality()`, `plot_stl_components()`
- `src/yohou/plotting/evaluation.py` — `plot_calibration()`, `plot_model_comparison_bar()`, `plot_score_distribution()`
- `src/yohou/plotting/model_selection.py` — `plot_cv_results_scatter()`, `plot_splits()`
- `src/yohou/plotting/signal.py` — `plot_spectrum()`, `plot_phase()`

**Testing**:
- `tests/plotting/test_exploration.py`, `test_diagnostics.py`, `test_forecasting.py`, `test_evaluation.py`, `test_model_selection.py`
- `tests/plotting/test_panel.py` — Dedicated panel-data tests
- `tests/plotting/conftest.py` — Shared panel fixtures

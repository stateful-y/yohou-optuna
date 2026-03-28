"""MyDataset - Brief Description of What Makes It Interesting.

One-line summary of the dataset's key characteristics.

Dataset: <frequency> <subject>, <start>-<end>
Demonstrates: plot_time_series, plot_rolling_statistics, <other plotting functions>
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from yohou.datasets import load_my_dataset
    from yohou.plotting import plot_time_series

    return load_my_dataset, mo, plot_time_series


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # MyDataset Dataset

    ## What You'll Learn

    - <First learning objective, e.g., Visualize raw time series data>
    - <Second learning objective>
    - <Third learning objective>

    ## Prerequisites

    None -- this is a standalone dataset exploration.
    """)


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "yohou"])


@app.cell
def _(load_my_dataset):
    df = load_my_dataset()
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Section Title

    Brief description of what this section demonstrates and why it matters.
    """)


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        title="MyDataset",
        x_label="Time",
        y_label="Value",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Section Title

    Brief description of the analysis or visualization in this section.
    """)


@app.cell
def _():
    # Code for section 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - <Summarize insight 1>
    - <Summarize insight 2>
    - <Summarize insight 3>
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next Steps

    - <Link to related notebook or topic 1>
    - <Link to related notebook or topic 2>
    """)


if __name__ == "__main__":
    app.run()

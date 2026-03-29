# Examples

Interactive [marimo](https://marimo.io) notebooks organized by purpose: **Tutorials** walk through concepts step-by-step, while **How-to Guides** solve specific tasks.

<!-- GALLERY -->

## Running Examples Locally

All examples are [marimo](https://marimo.io) reactive notebooks stored as `.py` files. Run them interactively or as scripts:

=== "just"

    ```bash
    # Open a specific example for interactive editing
    just example optuna_search.py
    ```

=== "marimo"

    ```bash
    # Interactive editing (recommended)
    uv run marimo edit examples/optuna_search.py

    # Run as a non-interactive script
    uv run marimo run examples/optuna_search.py
    ```

Replace `optuna_search.py` with any example filename from the gallery above.

## Next Steps

- Work through the [Getting Started](getting-started.md) tutorial if you are new to Yohou-Optuna
- Read [About OptunaSearchCV](../explanation/concepts.md) to understand how the search works
- Browse the [API Reference](../reference/api.md) for full parameter documentation

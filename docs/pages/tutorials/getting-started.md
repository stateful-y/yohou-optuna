# Getting Started

In this tutorial, we will install Yohou-Optuna and run a first example.

## Installation

Choose your preferred package manager:

=== "pip"

    ```bash
    pip install yohou_optuna
    ```

=== "uv"

    ```bash
    uv add yohou_optuna
    ```

=== "conda"

    ```bash
    conda install -c conda-forge yohou_optuna
    ```

=== "mamba"

    ```bash
    mamba install -c conda-forge yohou_optuna
    ```

> **Note**: For conda/mamba, ensure the package is published to conda-forge first.

Verify the installation:

```python
import yohou_optuna
print(yohou_optuna.__version__)
```

## Your First Example

```python
from yohou_optuna.example import hello

result = hello("World")
print(result)  # Output: Hello, World!
```


## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page.

Run locally:

=== "just"

    ```bash
    just example
    ```

=== "uv run"

    ```bash
    uv run marimo edit examples/hello.py
    ```

## Next Steps

- **Learn the concepts**: Read [Concepts](../explanation/concepts.md) to understand the design
- **Explore examples**: Check out the [Examples](examples.md) for interactive notebooks
- **Dive into the API**: Browse the [API Reference](../reference/api.md) for detailed documentation
- **Get help**: Visit [GitHub Discussions](https://github.com/stateful-y/yohou-optuna/discussions) or [open an issue](https://github.com/stateful-y/yohou-optuna/issues)

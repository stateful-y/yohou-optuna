# Getting Started

This guide will help you install and start using Yohou-Optuna in minutes.

## Installation

### Step 1: Install the package

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

### Step 2: Verify installation

```python
import yohou_optuna
print(yohou_optuna.__version__)
```

## Basic Usage

### Step 1: [Import and initialize]

```python
from yohou_optuna.example import hello

# Basic usage example
result = hello("World")
print(result)  # Output: Hello, World!
```

### Step 2: [Configure your setup]

Create a configuration file or set up options:

```yaml
# config.yaml (if your package supports config files)
yohou_optuna:
  option_1: value     # Description of what this controls
  option_2: true      # Description of what this controls
  option_3: 10        # Description of what this controls
```

Or configure in code:

```python
from yohou_optuna import [MainClass]

# Initialize with custom configuration
instance = [MainClass](
    option_1="value",   # Description
    option_2=True,      # Description
    option_3=10,        # Description
)
```

### Step 3: [Use the main functionality]

```python
# [Add realistic example showing actual usage]
# For example:
# result = instance.process(data)
# output = instance.transform(input_data)

# Example with the provided function
greeting = hello("Python")
print(greeting)
```

## Complete Example

Here's a complete working example:

```python
from yohou_optuna.example import hello

# [Replace with realistic multi-step example]
# Step 1: Initialize
names = ["Alice", "Bob", "Charlie"]

# Step 2: Process
greetings = [hello(name) for name in names]

# Step 3: Display results
for greeting in greetings:
    print(greeting)
```

## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page where you can:

- Run code directly in your browser
- Experiment with different parameters
- See visual outputs in real-time
- Download standalone HTML versions

Or run locally:

=== "just"

    ```bash
    just example
    ```

=== "uv run"

    ```bash
    uv run marimo edit examples/hello.py
    ```

## Next Steps

Now that you have Yohou-Optuna installed and running:

- **Learn the concepts**: Read the [User Guide](user-guide.md) to understand core concepts and capabilities
- **Explore examples**: Check out the [Examples](examples.md) for real-world use cases
- **Dive into the API**: Browse the [API Reference](api-reference.md) for detailed documentation
- **Get help**: Visit [GitHub Discussions](https://github.com/stateful-y/yohou-optuna/discussions) or [open an issue](https://github.com/stateful-y/yohou-optuna/issues)

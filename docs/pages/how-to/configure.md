# How to Configure Yohou-Optuna

This guide shows you how to set up and customize Yohou-Optuna for your project.

## Basic Configuration

```python
from yohou_optuna import [MainClass]

instance = [MainClass](
    option_1="value",
    option_2=True,
)
```

## Configuration File

```yaml
# config.yaml
yohou_optuna:
  option_1: value
  option_2: true
```

## Advanced Configuration

Describe advanced configuration patterns here: environment variables, multiple environments, runtime overrides.

## See Also

- [Concepts](../explanation/concepts.md) - understand the design
- [API Reference](../reference/api.md) - full list of options

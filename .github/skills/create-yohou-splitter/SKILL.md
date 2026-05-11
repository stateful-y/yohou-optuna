---
name: create-yohou-splitter
description: "Step-by-step guide for implementing new time series cross-validation splitters in Yohou. Covers expanding window, sliding window, and gap patterns extending BaseSplitter. Includes split/get_n_splits/_iter_test_indices templates, panel data row-index splitting, and test templates with _yield_yohou_splitter_checks (8 systematic checks). Use when creating or testing custom CV splitters."
---

# Creating New Splitters

## Quick Decision Tree

- **Expanding window** (training set grows) → Extend `ExpandingWindowSplitter` pattern
- **Sliding window** (fixed training size) → Extend `SlidingWindowSplitter` pattern
- **Custom logic** → Extend `BaseSplitter` directly

---

## Templates

- [Splitter class template](./splitter_template.py) — `MySplitter(BaseSplitter)` with split/_iter_test_indices/get_n_splits
- [Test file template](./test_splitter_template.py) — Full test with `_yield_yohou_splitter_checks` + specific tests

---

## Expanding vs. Sliding Window Patterns

### Expanding Window (Training Set Grows)

```python
def split(self, y, X_actual=None):
    n_samples = len(y)
    test_size = self.test_size or n_samples // (self.n_splits + 1)

    for i in range(self.n_splits):
        test_start = n_samples - (self.n_splits - i) * test_size
        test_end = test_start + test_size
        train_indices = np.arange(0, test_start)  # Grows with each split
        test_indices = np.arange(test_start, test_end)
        yield train_indices, test_indices
```

### Sliding Window (Fixed Training Size)

```python
def split(self, y, X_actual=None):
    n_samples = len(y)
    test_size = self.test_size or n_samples // (self.n_splits + 1)
    train_size = self.train_size or n_samples - (self.n_splits * test_size)

    for i in range(self.n_splits):
        test_start = train_size + (i * test_size)
        test_end = test_start + test_size
        train_start = max(0, test_start - train_size)
        train_indices = np.arange(train_start, test_start)
        test_indices = np.arange(test_start, test_end)
        yield train_indices, test_indices
```

---

## Gap Between Train and Test

```python
_parameter_constraints: dict = {
    **BaseSplitter._parameter_constraints,
    "gap": [Interval(numbers.Integral, 0, None, closed="left"), None],
}

def split(self, y, X_actual=None):
    gap = self.gap or 0
    for i in range(self.n_splits):
        test_start = ...
        train_indices = np.arange(0, test_start - gap)
        test_indices = np.arange(test_start, test_end)
        yield train_indices, test_indices
```

---

## Panel Data Support

Splitters work with panel data automatically (row indices apply across all panel groups). No special handling needed.

---

## Testing Splitters

### Systematic Checks

`_yield_yohou_splitter_checks` (8 checks):
- Tags accessible before/after fit
- Tags match capabilities
- Valid train/test indices
- get_n_splits() matches actual
- Non-overlapping test sets
- Panel data handling
- Parameter constraints enforced

### Tags for Generator

```python
tags = {
    "splitter_type": "expanding",  # "expanding", "sliding", "gap"
    "supports_panel_data": False,
    "produces_non_overlapping_tests": True,
}
```

### Available Test Fixtures

```python
def test_my_func(time_series_factory):
    y = time_series_factory(length=100, n_components=1, seed=42)

def test_panel(panel_time_series_factory):
    y = panel_time_series_factory(length=100, n_series=2, n_groups=2)
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/model_selection/split.py`
2. `uvx ruff format src/yohou/model_selection/split.py`
3. `uvx ty check src/yohou/model_selection/split.py`
4. `uvx interrogate src/yohou/model_selection/split.py` (docstring coverage)
5. `uv run pytest tests/model_selection/test_split.py -v`
6. `uv run pytest --doctest-modules src/yohou/model_selection/split.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Train/test overlap**: Ensure `test_start > train_end` (or `test_start >= train_end + gap`)
- **Non-temporal order**: Never shuffle indices — time series must maintain order
- **Off-by-one errors**: Use `np.arange(start, end)` carefully (end is exclusive)
- **Empty splits**: Validate that `n_samples` is sufficient for `n_splits * test_size + train_size`
- **Inconsistent test sizes**: Last split may have different size — document or adjust
- **Panel groups split differently**: Splitters use row indices, not per-group logic (by design)
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

---

## Real-World Examples to Study

**Built-in splitters**:
- `src/yohou/model_selection/split.py`:
  - `ExpandingWindowSplitter` - Training set grows
  - `SlidingWindowSplitter` - Fixed training window

**Testing**:
- `tests/model_selection/test_split.py` - Comprehensive splitter tests
- `src/yohou/testing/splitter.py` - Check functions (8 checks)

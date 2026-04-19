---
name: create-yohou-dataset
description: "Guide for adding new bundled datasets to Yohou. Covers the full workflow: data preparation with polars, Parquet export (zstd compression), Git LFS tracking, loader function creation in loaders.py, schema conventions (univariate/multivariate/panel with __ separator), Marimo example notebook creation, and test templates. Includes file size guidelines, licensing metadata, and external hosting patterns (Zenodo, HuggingFace) for large datasets."
---

# Creating New Datasets

## Overview

Yohou includes bundled datasets for examples, testing, and benchmarking. All datasets:
- Stored as **Parquet files** with **zstd compression**
- Tracked via **Git LFS** (large file storage)
- Have **loader functions** in `src/yohou/datasets/loaders.py`
- Follow **consistent schema conventions**
- Include **comprehensive metadata** in docstrings

**Location**: `src/yohou/datasets/data/`

---

## Templates

- [Loader function template](./loader_template.py) — `load_my_dataset()` function with full metadata docstring
- [Test file template](./test_dataset_template.py) — Test with basic/quality/panel tests
- [Example notebook template](./dataset_example_template.py) — Marimo notebook skeleton

---

## Adding a New Dataset (Step-by-Step)

### 1. Obtain and Prepare Data

```python
import polars as pl

df = pl.read_csv("raw_data.csv")

# CRITICAL: Ensure 'time' column with datetime type
df = df.with_columns(
    pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("time")
)

# For panel data: Use "__" separator (prefix__suffix)
df = df.rename({"store_1_sales": "sales__store_1", "store_2_sales": "sales__store_2"})

# Sort by time
df = df.sort("time")

# Validate time consistency
from yohou.utils.validation import check_interval_consistency
interval = check_interval_consistency(df)
```

### 2. Export to Parquet

```python
df.write_parquet(
    "src/yohou/datasets/data/my_dataset.parquet",
    compression="zstd",
    compression_level=9,
)
```

### 3. Git LFS Tracking

```bash
git lfs install
cat .gitattributes | grep parquet  # Verify *.parquet tracked
git add src/yohou/datasets/data/my_dataset.parquet
git commit -m "Add my_dataset to bundled datasets"
```

### 4. Create Loader Function

See [loader template](./loader_template.py). Add to `src/yohou/datasets/loaders.py`.

### 5. Update Exports

```python
# src/yohou/datasets/__init__.py
from .loaders import load_my_dataset
__all__ = [..., "load_my_dataset"]
```

### 6. Create Example Notebook (Optional)

See [example template](./dataset_example_template.py). Create as `examples/my_dataset.py`.

---

## Dataset Schema Conventions

### Univariate

```python
df = pl.DataFrame({"time": [...], "value": [...]})
```

### Multivariate

```python
df = pl.DataFrame({"time": [...], "target": [...], "feature_1": [...], "feature_2": [...]})
```

### Panel Data

```python
df = pl.DataFrame({
    "time": [...],
    "sales__store_1": [...],  # prefix__suffix with double underscore
    "sales__store_2": [...],
})
```

---

## File Size Guidelines

- **Small** (<1 MB): No concern
- **Medium** (1-10 MB): Acceptable for examples
- **Large** (10-100 MB): Consider sampling or external hosting
- **Very large** (>100 MB): External hosting required (Zenodo, HuggingFace)

---

## Checklist Before Committing

1. Data exported to `src/yohou/datasets/data/my_dataset.parquet` with zstd compression
2. Git LFS tracking verified (`git lfs ls-files`)
3. Loader function added to `loaders.py` with complete docstring metadata
4. Exports updated in `__init__.py`
5. Test added to `tests/datasets/test_loaders.py`
6. Example notebook created in examples/datasets
7. File size checked (<10 MB compressed preferred)
8. License/citation included in docstring
9. `uvx nox -s fix` passes
10. `uv run pytest tests/datasets/ -v` passes

---

## Common Pitfalls

- **Missing time column**: All datasets MUST have `"time"` column
- **Wrong time type**: Must be `pl.Datetime` or `pl.Date` (temporal type required, not string)
- **Unsorted time**: Always sort by `"time"` column
- **Not using Git LFS**: Parquet files must be tracked by LFS
- **No compression**: Always use `compression="zstd"`
- **Panel data naming**: Must use double underscore `"prefix__suffix"`
- **Missing metadata**: Loader docstring MUST include all Notes fields
- **File too large**: Check compressed size, consider sampling if >10 MB
- **No test**: Every loader needs a test
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

---

## Real-World Examples to Study

**Loader functions**:
- `src/yohou/datasets/loaders.py` — All 10 built-in datasets

**Testing**:
- `tests/datasets/test_loaders.py` — Dataset loading tests

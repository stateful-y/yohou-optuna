# Update Yohou Documentation

## Purpose

Guide for maintaining and extending Yohou's MkDocs documentation site, which follows an sklearn-inspired structure.

## Documentation Structure

```text
docs/
├── index.md                        # Landing page with grid cards
├── pages/
│   ├── getting-started/
│   │   ├── index.md                    # Quick start tutorial
│   │   ├── installation.md             # Installation guide
│   │   └── overview.md                 # Capabilities overview
│   ├── user-guide/
│   │   ├── index.md                    # Chapter listing
│   │   ├── core-concepts.md            # Ch1: Data format, lifecycle, tags
│   │   ├── forecasting.md              # Ch2: Point forecasting
│   │   ├── interval-forecasting.md     # Ch3: Interval forecasting
│   │   ├── preprocessing.md            # Ch4: Transformers & pipelines
│   │   ├── stationarity.md             # Ch5: Trend, seasonality, decomposition
│   │   ├── model-selection.md          # Ch6: CV splitters, search
│   │   ├── visualization.md            # Ch7: Plotting functions
│   │   └── advanced.md                 # Ch8: Panel data, metadata routing, weighting
│   ├── api/
│   │   ├── index.md                    # Module index table
│   │   ├── base.md                     # yohou.base (6 symbols)
│   │   ├── compose.md                  # yohou.compose (6 symbols)
│   │   ├── point.md                    # yohou.point (3 symbols)
│   │   ├── interval.md                 # yohou.interval (5 symbols)
│   │   ├── metrics.md                  # yohou.metrics (22 symbols)
│   │   ├── model-selection.md          # yohou.model_selection (5 symbols)
│   │   ├── preprocessing.md            # yohou.preprocessing (28 symbols)
│   │   ├── stationarity.md             # yohou.stationarity (10 symbols)
│   │   ├── plotting.md                 # yohou.plotting (33 symbols)
│   │   ├── datasets.md                 # yohou.datasets (7 symbols)
│   │   ├── utils.md                    # yohou.utils (38 symbols)
│   │   └── testing.md                  # yohou.testing (98 symbols)
│   ├── examples/
│   │   ├── index.md                    # Examples overview
│   │   ├── quickstart.md               # Top-level notebook links
│   │   ├── datasets.md                 # Dataset exploration notebooks
│   │   ├── point.md                    # Point forecaster notebooks
│   │   ├── interval.md                 # Interval forecaster notebooks
│   │   ├── preprocessing.md            # Preprocessing notebooks
│   │   ├── stationarity.md             # Stationarity notebooks
│   │   ├── metrics.md                  # Metrics notebooks
│   │   ├── model-selection.md          # Model selection notebooks
│   │   └── plotting.md                 # Plotting notebooks
├── development/
│   ├── index.md                    # Development overview
│   ├── contributing.md             # Full contributing guide (migrated)
│   ├── developing-estimators.md    # Custom estimator guide (stub)
│   └── changelog.md                # Links to CHANGELOG.md
└── integrations/
    └── index.md                    # yohou-optuna, yohou-nixtla
```

All page content lives under `docs/pages/`. The `docs/examples/` directory (without `pages/`) is used by hooks.py for exported notebook HTML.

## How to Add a New Public Symbol to API Docs

1. Identify the submodule the symbol belongs to (e.g., `yohou.preprocessing`).
2. Open the corresponding API page (e.g., `docs/pages/api/preprocessing.md`).
3. Add a mkdocstrings directive under the appropriate `##` heading:

```markdown
::: yohou.preprocessing.MyNewTransformer
    options:
      show_root_heading: true
      show_source: true
      members_order: source
```

4. For classes, use `show_source: true`. For functions, use `show_source: false`.
5. Ensure the symbol is exported in the submodule's `__init__.py`.

## How to Add a New User Guide Chapter

1. Create `docs/pages/user-guide/<chapter-slug>.md` with an "Under Development" admonition.
2. Add it to `docs/pages/user-guide/index.md` in the chapter listing.
3. Add it to the `nav` section in `mkdocs.yml` under "User Guide" (prefix path with `pages/`).

## How to Add a New Example Page

1. Create the marimo notebook in `examples/<subdir>/<name>.py`.
2. Add a link row to the corresponding `docs/pages/examples/<subdir>.md` page.
3. The `docs/hooks.py` `on_pre_build` hook automatically discovers and exports all notebooks recursively from `examples/` (excluding `__marimo__/`, `bugs/`, and `__init__.py` files).

## How to Add a New API Submodule Page

1. Create `docs/pages/api/<module>.md` following the pattern of existing pages.
2. Add it to `docs/pages/api/index.md` in the module table.
3. Add it to the `nav` section in `mkdocs.yml` under "API Reference" (prefix path with `pages/`).

## Configuration Files

- **`mkdocs.yml`**: Site config, nav, theme, plugins, extensions
- **`docs/hooks.py`**: Pre-build hook for marimo notebook export + markdown link rewriting
- **`docs/material/overrides/main.html`**: Theme overrides

## Build Commands

```bash
uvx nox -s build_docs    # Build docs (includes notebook export)
uvx nox -s serve_docs    # Serve locally at localhost:8080
```

## Key Conventions

- API pages use mkdocstrings `::: yohou.<module>.<Symbol>` directives
- All docstrings are NumPy style (enforced by `interrogate`)
- User guide stubs have `!!! note "Under Development"` admonitions
- Example pages link to exported HTML notebooks at `/examples/<path>/`
- Notebooks export to HTML-WASM via marimo (edit mode)
- Navigation uses Material theme tabs with `navigation.indexes` for section index pages
- **Never use Sphinx cross-links**: We use mkdocs, not Sphinx. Never use `:class:`, `:func:`, `:meth:`, `:mod:`, `:obj:`, `:ref:`, `:attr:`, or `:term:` directives in docstrings or documentation. Use backtick references instead (e.g., `` `ClassName` `` not `:class:\`ClassName\``). Also never use mkdocs cross-references like `[ClassName][]` in docstrings — those only render in `.md` files, not in Python help or IDEs. For hyperlinks in docstrings, always use Markdown syntax `[text](url)`, never RST syntax ``text <url>`_`
- **Never use inline comment separators**: Do not use `# --------`, `# ========`, section name headers, or any decorative comment dividers in code

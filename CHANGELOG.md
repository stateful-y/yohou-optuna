# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0-alpha.9] - 2026-08-10

This **minor release** includes 20 commits.


### Bug Fixes
- Pin exact uv version in setup-uv steps (template v0.29.6)  ([#46](https://github.com/stateful-y/yohou-optuna/pull/46)) by @gtauzin
- Pin ossf/scorecard-action to v2.4.4 (@v2 tag does not exist) by @gtauzin
- Pin the nightly Codecov upload and discover every upload to check  ([#63](https://github.com/stateful-y/yohou-optuna/pull/63)) by @gtauzin

### Refactoring
- Move build output to .artifacts/ and CODEOWNERS to .github/  ([#62](https://github.com/stateful-y/yohou-optuna/pull/62)) by @gtauzin

### Miscellaneous Tasks
- Update from python-package-copier v0.27.3  ([#38](https://github.com/stateful-y/yohou-optuna/pull/38)) by @gtauzin
- Update from python-package-copier v0.28.1  ([#41](https://github.com/stateful-y/yohou-optuna/pull/41)) by @gtauzin
- Update from python-package-copier v0.28.3  ([#42](https://github.com/stateful-y/yohou-optuna/pull/42)) by @gtauzin
- Update from python-package-copier v0.28.4  ([#43](https://github.com/stateful-y/yohou-optuna/pull/43)) by @gtauzin
- Update from python-package-copier v0.29.3  ([#44](https://github.com/stateful-y/yohou-optuna/pull/44)) by @gtauzin
- Update from template v0.31.1 (Renovate replaces Dependabot)  ([#48](https://github.com/stateful-y/yohou-optuna/pull/48)) by @gtauzin
- Update to v0.32.1 (pre-push gates + CI roll-up)  ([#50](https://github.com/stateful-y/yohou-optuna/pull/50)) by @gtauzin
- Sync to v0.35.0  ([#51](https://github.com/stateful-y/yohou-optuna/pull/51)) by @gtauzin
- Sync to v0.36.0 (Codecov OIDC + scorecard pin)  ([#52](https://github.com/stateful-y/yohou-optuna/pull/52)) by @gtauzin
- Sync to v0.37.0 (gitsign tag-signing docs)  ([#53](https://github.com/stateful-y/yohou-optuna/pull/53)) by @gtauzin
- Update from python-package-copier v0.38.0  ([#54](https://github.com/stateful-y/yohou-optuna/pull/54)) by @gtauzin
- Update from template v0.39.0  ([#55](https://github.com/stateful-y/yohou-optuna/pull/55)) by @gtauzin
- Update from template v0.39.1  ([#56](https://github.com/stateful-y/yohou-optuna/pull/56)) by @gtauzin
- Update from template v0.40.0  ([#57](https://github.com/stateful-y/yohou-optuna/pull/57)) by @gtauzin
- Update from python-package-copier v0.40.1  ([#61](https://github.com/stateful-y/yohou-optuna/pull/61)) by @gtauzin

### Build
- Migrate to the Zensical documentation engine (template v0.30.1)  ([#47](https://github.com/stateful-y/yohou-optuna/pull/47)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.8] - 2026-07-20

This **minor release** includes 3 commits.


### Breaking Changes
- Follow yohou rename of `feature_transformer` to `actual_transformer`  ([#34](https://github.com/stateful-y/yohou-optuna/pull/34)) by @gtauzin
  - No deprecation alias: `param_distributions` keys become `actual_transformer__*`
  - Transformer slots are keyword-only, and a third slot `forecast_transformer` joins them
  - yohou pin narrows from `>=0.1.0a6` to `>=0.1.0a11,<0.2.0`
  - Only docs, examples, tests and the bundled forecaster skill changed; this package's source never named the slot
  - `create-yohou-forecaster` template switches `BaseTransformer` to `BaseActualTransformer`

### Miscellaneous Tasks
- Update from template v0.26.1  ([#32](https://github.com/stateful-y/yohou-optuna/pull/32)) by @gtauzin
  - Examples become a top-level docs section; companion notebook cards replace hand-rolled tips
  - All five example notebooks execute again: `yohou[plotting]` added, and the never-existing `plot_model_comparison_bar` replaced with a plotly grouped bar
  - Notebook tests no longer skip on `ImportError`, so a broken notebook fails instead of passing silently
  - Custom skills move from `.github/skills/` to `.claude/skills/` as the tracked source of truth
  - `uv.lock` added as the single source of truth for lint tooling
  - `docs/hooks.py` unforked and back byte-identical to the template
- Update from template v0.27.0  ([#35](https://github.com/stateful-y/yohou-optuna/pull/35)) by @gtauzin
  - `pre-commit` replaced by `prek`; run `just install` to rewire the git hooks
  - Builtin hooks now pinned by prek's version in `uv.lock` instead of a frozen rev
  - `default_install_hook_types` installs the `commit-msg` hook that was declared but never wired up
  - New Validate Commit Message workflow checks the squash title of single-commit PRs
  - Changelog workflow now fails loudly when the hook runner cannot run, instead of `|| true`
  - Ruff gains `T10` (flake8-debugger), replacing the dropped `debug-statements` hook
  - Notebook tests get a 300s timeout and closed stdin, so a stray `breakpoint()` fails instead of hanging

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.7] - 2026-07-07

This **minor release** includes 1 commit.


### Bug Fixes
- Predict before scoring for the refactored yohou _score API  ([#30](https://github.com/stateful-y/yohou-optuna/pull/30)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.6] - 2026-05-11

This **minor release** includes 2 commits.


### Features
- Update API from X to X_actual, add X_future and X_forecast support  ([#24](https://github.com/stateful-y/yohou-optuna/pull/24)) by @gtauzin

### Documentation
- Update concepts.md for X_actual, X_future, X_forecast API by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.5] - 2026-04-22

This **minor release** includes 1 commit.


### Bug Fixes
- Infer metric names from all completed trials  ([#22](https://github.com/stateful-y/yohou-optuna/pull/22)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.4] - 2026-04-22

This **minor release** includes 1 commit.


### Features
- Upgrade to yohou 0.1.0a5 with class proba support  ([#20](https://github.com/stateful-y/yohou-optuna/pull/20)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.3] - 2026-04-19

This **minor release** includes 2 commits.


### Features
- Add interval forecaster support to OptunaSearchCV  ([#12](https://github.com/stateful-y/yohou-optuna/pull/12)) by @gtauzin

### Miscellaneous Tasks
- Sync with template v0.18.0 and rework docs  ([#13](https://github.com/stateful-y/yohou-optuna/pull/13)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.2] - 2026-03-01

This **minor release** includes 2 commits.


### Miscellaneous Tasks
- Apply copier template v0.14.0 changes  ([#5](https://github.com/stateful-y/yohou-optuna/pull/5)) by @gtauzin
- Apply copier template v0.15.0 changes  ([#6](https://github.com/stateful-y/yohou-optuna/pull/6)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.1] - 2026-02-20

This **minor release** includes 1 commit.

- Initial commit

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [Unreleased]

### Added
- Initial project setup

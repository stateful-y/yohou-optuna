"""Search train scores follow yohou's train-score definition."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import optuna
import polars as pl
import pytest
from optuna.distributions import CategoricalDistribution
from yohou.interval import SplitConformalForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.metrics.interval import IntervalScore
from yohou.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter, cross_validate
from yohou.model_selection import utils as ms_utils
from yohou.point import SeasonalNaive

import yohou_optuna.objective as objective_module
from yohou_optuna import OptunaSearchCV, Sampler
from yohou_optuna.objective import _Objective

FH = 24
COVERAGE = [0.9]


def _hourly(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pl.DataFrame({
        "time": [datetime(2024, 1, 1) + timedelta(hours=int(i)) for i in t],
        "y_0": np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(n),
    })


def _conformal() -> SplitConformalForecaster:
    return SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24), calibration_size=168)


def _search(cv, **kwargs) -> OptunaSearchCV:
    return OptunaSearchCV(
        forecaster=kwargs.pop("forecaster", _conformal()),
        param_distributions={"point_forecaster__seasonality": CategoricalDistribution([24])},
        scoring=kwargs.pop("scoring", IntervalScore(coverage_rates=COVERAGE)),
        sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=0),
        n_trials=1,
        cv=cv,
        refit=False,
        return_train_score=True,
        **kwargs,
    )


def _split_train_scores(search: OptunaSearchCV, n_splits: int) -> list[float]:
    return [search.cv_results_[f"split{i}_train_score"][0] for i in range(n_splits)]


class TestAgreementWithCrossValidate:
    """A trial's per-split train score equals cross-validation's."""

    def test_split_conformal_forecaster(self):
        y = _hourly(1168)
        cv = ExpandingWindowSplitter(n_splits=2, test_size=168)
        search = _search(cv).fit(y, forecasting_horizon=FH)
        expected = cross_validate(
            _conformal(),
            y,
            forecasting_horizon=FH,
            scoring=IntervalScore(coverage_rates=COVERAGE),
            cv=cv,
            return_train_score=True,
        )["train_score"].to_list()
        assert _split_train_scores(search, 2) == pytest.approx(expected)
        # The first split trains on 832 rows: the scored stretch is rows 496-663,
        # which ends before the 168 calibration rows 664-831.
        assert all(np.isfinite(expected))

    def test_sliding_window(self):
        y = _hourly(1600)
        cv = SlidingWindowSplitter(n_splits=2, train_size=1000, test_size=168)
        search = _search(cv).fit(y, forecasting_horizon=FH)
        expected = cross_validate(
            _conformal(),
            y,
            forecasting_horizon=FH,
            scoring=IntervalScore(coverage_rates=COVERAGE),
            cv=cv,
            return_train_score=True,
        )["train_score"].to_list()
        assert _split_train_scores(search, 2) == pytest.approx(expected)


class TestShortTrainingWindow:
    """A window with no room before the held-back rows scores NaN, never other rows."""

    def test_nan_and_warning(self):
        # First split: 500 - 2 * 168 = 164 training rows, fewer than 168 test + 168 held back.
        y = _hourly(500)
        cv = ExpandingWindowSplitter(n_splits=2, test_size=168)
        search = _search(cv, forecaster=SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=48))
        with pytest.warns(UserWarning, match="Train score is unavailable"):
            search.fit(y, forecasting_horizon=FH)
        train = _split_train_scores(search, 2)
        assert np.isnan(train[0])
        assert np.isfinite(search.cv_results_["split0_test_score"][0])


class TestScoreParams:
    """Score params reach train scoring sliced to the scored rows."""

    def test_sliced_to_scored_rows(self, monkeypatch):
        y = _hourly(1000)
        cv = ExpandingWindowSplitter(n_splits=2, test_size=168)
        captured: list[np.ndarray] = []

        def capture_train(forecaster, y_train, y_test, y_pred, scorer, score_params, error_score):
            captured.append(score_params["row_id"])
            return 0.0

        # Test scoring goes through the objective's own reference; train scoring through yohou's.
        monkeypatch.setattr(objective_module, "_score", lambda *args, **kwargs: 0.0)
        monkeypatch.setattr(ms_utils, "_score", capture_train)
        objective = _Objective(
            forecaster=SeasonalNaive(seasonality=24),
            param_distributions={"seasonality": CategoricalDistribution([24])},
            y=y,
            X_actual=None,
            X_future=None,
            X_forecast=None,
            forecasting_horizon=FH,
            cv=cv,
            scorers=MeanAbsoluteError(),
            fit_params={},
            predict_func_params={},
            score_params={"row_id": np.arange(len(y))},
            split_params={},
            return_train_score=True,
            error_score="raise",
        )
        optuna.create_study(direction="maximize").optimize(objective, n_trials=1)
        # Split 0 trains on rows 0-663 and scores 496-663; split 1 trains on 0-831 and scores 664-831.
        assert [ids.tolist() for ids in captured] == [list(range(496, 664)), list(range(664, 832))]


class TestTrainScoreFailure:
    """A failure while train scoring is recorded like any other failure in the split."""

    def _break_train_scoring(self, monkeypatch):
        def fail(*args, **kwargs):
            raise ValueError("intentional train-score failure")

        monkeypatch.setattr(objective_module, "_train_window_predictions", fail)

    def test_numeric_error_score(self, monkeypatch):
        self._break_train_scoring(monkeypatch)
        y = _hourly(1000)
        search = _search(
            ExpandingWindowSplitter(n_splits=2, test_size=168),
            forecaster=SeasonalNaive(seasonality=24),
            scoring=MeanAbsoluteError(),
            error_score=np.nan,
        )
        search.param_distributions = {"seasonality": CategoricalDistribution([24])}
        search.fit(y, forecasting_horizon=FH)

        trial = search.trials_[0]
        assert trial.user_attrs["failed_splits"] == [0, 1]
        assert np.isnan(_split_train_scores(search, 2)).all()
        # Exactly one test and one train entry per split, the test scores intact.
        assert "split2_test_score" not in search.cv_results_
        assert "split2_train_score" not in search.cv_results_
        assert np.isfinite([search.cv_results_[f"split{i}_test_score"][0] for i in range(2)]).all()

    def test_raise_error_score(self, monkeypatch):
        self._break_train_scoring(monkeypatch)
        y = _hourly(1000)
        search = _search(
            ExpandingWindowSplitter(n_splits=2, test_size=168),
            forecaster=SeasonalNaive(seasonality=24),
            scoring=MeanAbsoluteError(),
            error_score="raise",
        )
        search.param_distributions = {"seasonality": CategoricalDistribution([24])}
        with pytest.raises(ValueError, match="intentional train-score failure"):
            search.fit(y, forecasting_horizon=FH)

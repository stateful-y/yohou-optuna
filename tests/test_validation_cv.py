"""Tests for ``validation="cv"``: shared-round early stopping in OptunaSearchCV."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import optuna
import polars as pl
import pytest
from optuna.distributions import CategoricalDistribution
from sklearn.base import clone
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer

from shared_round_stubs import CurveEarlyStoppingAdapter, CurveRegressor
from yohou_optuna import OptunaSearchCV, Sampler

N_SPLITS = 3
TEST_SIZE = 12
HORIZON = 3


def _series(n: int = 132) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    times = pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(hours=n - 1), "1h", eager=True)
    return pl.DataFrame({"time": times, "value": 10.0 + np.arange(n) * 0.2 + rng.normal(0, 0.3, n)})


def _cv() -> ExpandingWindowSplitter:
    return ExpandingWindowSplitter(n_splits=N_SPLITS, test_size=TEST_SIZE)


def _point(estimator=None, **kwargs) -> PointReductionForecaster:
    return PointReductionForecaster(
        estimator=estimator if estimator is not None else CurveRegressor(),
        reduction_strategy=kwargs.pop("reduction_strategy", "direct"),
        actual_transformer=LagTransformer(lag=[1, 2]),
        **kwargs,
    )


def _search(forecaster=None, distributions=None, **kwargs) -> OptunaSearchCV:
    return OptunaSearchCV(
        forecaster=forecaster if forecaster is not None else _point(),
        param_distributions=(
            distributions if distributions is not None else {"estimator__patience": CategoricalDistribution([6])}
        ),
        scoring=kwargs.pop("scoring", MeanAbsoluteError()),
        sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=0),
        n_trials=kwargs.pop("n_trials", 1),
        cv=_cv(),
        validation=kwargs.pop("validation", "cv"),
        early_stopping_adapter=kwargs.pop("early_stopping_adapter", CurveEarlyStoppingAdapter()),
        **kwargs,
    )


class TestAgreementWithGridSearch:
    """One trial evaluates its candidate exactly as GridSearchCV(validation="cv") does."""

    def test_rounds_scores_and_train_scores_match(self):
        y = _series()
        optuna_search = _search(return_train_score=True).fit(y, forecasting_horizon=HORIZON)
        grid = GridSearchCV(
            forecaster=_point(),
            param_grid={"estimator__patience": [6]},
            scoring=MeanAbsoluteError(),
            cv=_cv(),
            validation="cv",
            early_stopping_adapter=CurveEarlyStoppingAdapter(),
            return_train_score=True,
        ).fit(y, forecasting_horizon=HORIZON)

        ours, theirs = optuna_search.cv_results_, grid.cv_results_
        assert ours["rounds"][0] == theirs["rounds"][0]
        assert bool(ours["rounds_at_boundary"][0]) == bool(theirs["rounds_at_boundary"][0])
        for i in range(N_SPLITS):
            assert ours[f"split{i}_curve_length"][0] == theirs[f"split{i}_curve_length"][0]
            assert ours[f"split{i}_test_score"][0] == pytest.approx(theirs[f"split{i}_test_score"][0])
            assert ours[f"split{i}_train_score"][0] == pytest.approx(theirs[f"split{i}_train_score"][0])
        assert ours["mean_test_score"][0] == pytest.approx(theirs["mean_test_score"][0])
        assert optuna_search.best_rounds_ == grid.best_rounds_


class TestRefit:
    """The refit trains the best trial's chosen rounds with early stopping off."""

    def test_refit_trains_the_chosen_rounds(self):
        y = _series()
        adapter = CurveEarlyStoppingAdapter()
        search = _search(early_stopping_adapter=adapter).fit(y, forecasting_horizon=HORIZON)

        assert set(search.best_rounds_) == {"step_1", "step_2", "step_3"}
        positions = dict(search.best_forecaster_._fitted_estimator_positions())
        for position, estimator in positions.items():
            assert estimator.rounds_used_ == search.best_rounds_[position]
            # The refit trained up to the largest chosen round, not to the template's ceiling.
            assert estimator.n_rounds == max(search.best_rounds_.values())
        # The search's own adapter configures the refit, then cuts every position to its round.
        refit_calls = adapter.calls[-(1 + len(positions)) :]
        assert refit_calls[0] == ("prepare_refit", max(search.best_rounds_.values()))
        assert sorted(refit_calls[1:]) == sorted(("truncate", search.best_rounds_[p]) for p in positions)

    def test_refit_false_leaves_best_rounds_unset(self):
        y = _series()
        search = _search(refit=False).fit(y, forecasting_horizon=HORIZON)
        assert "rounds" in search.cv_results_
        assert not hasattr(search, "best_forecaster_")


class TestResultsShape:
    """The round columns and trial attributes."""

    def test_no_round_columns_without_validation(self):
        y = _series()
        search = _search(validation=None, early_stopping_adapter=None).fit(y, forecasting_horizon=HORIZON)
        assert not {"rounds", "rounds_at_boundary", "split0_curve_length"} & set(search.cv_results_)
        assert not hasattr(search, "best_rounds_")

    def test_trial_attributes_are_json_serialisable(self):
        y = _series()
        search = _search(n_trials=3, distributions={"estimator__patience": CategoricalDistribution([4, 6, 8])}).fit(
            y, forecasting_horizon=HORIZON
        )
        for trial in search.trials_:
            json.dumps(trial.user_attrs)
            assert {"rounds", "rounds_at_boundary", "boundary_positions", "curve_lengths"} <= set(trial.user_attrs)
        assert len(search.cv_results_["rounds"]) == len(search.cv_results_["params"])

    def test_parameters_round_trip(self):
        search = _search()
        params = search.get_params(deep=False)
        assert params["validation"] == "cv"
        assert isinstance(params["early_stopping_adapter"], CurveEarlyStoppingAdapter)
        cloned = clone(search)
        assert cloned.validation == "cv"

    def test_round_at_the_curve_end_is_flagged_and_warned(self):
        y = _series()
        search = _search(forecaster=_point(estimator=CurveRegressor(n_rounds=8)), refit=False)
        with pytest.warns(UserWarning, match=r"'step_1'.*last round every fold trained"):
            search.fit(y, forecasting_horizon=HORIZON)
        assert list(search.cv_results_["rounds_at_boundary"]) == [True]


class TestFailures:
    """Configuration errors stop the search; fold failures follow error_score."""

    def test_non_reduction_forecaster_is_rejected_before_any_trial(self):
        search = _search(forecaster=SeasonalNaive(), distributions={"seasonality": CategoricalDistribution([1])})
        with pytest.raises(ValueError, match="requires a reduction forecaster"):
            search.fit(_series(), forecasting_horizon=HORIZON)
        assert not hasattr(search, "study_")

    @pytest.mark.parametrize("key", ["eval_set", "y_val"])
    def test_evaluation_data_in_fit_params_is_rejected(self, key):
        with pytest.raises(ValueError, match=key):
            _search()._check_shared_round_setup({key: object()})

    def test_sampled_unusable_configuration_stops_the_search(self):
        search = _search(distributions={"reduction_strategy": CategoricalDistribution(["dir-rec"])})
        with pytest.raises(ValueError, match="earlier steps' predictions"):
            search.fit(_series(), forecasting_horizon=HORIZON)

    def test_one_failing_fold_is_recorded(self):
        y = _series()
        smallest_train = len(next(iter(_cv().split(y)))[0]) - HORIZON
        search = _search(
            forecaster=_point(estimator=CurveRegressor(fail_below_train_rows=smallest_train + 1)), refit=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(y, forecasting_horizon=HORIZON)

        trial = search.trials_[0]
        assert trial.user_attrs["failed_splits"] == [0]
        assert trial.user_attrs["exception_type"] == "RuntimeError"
        assert "refuses" in trial.user_attrs["exception"]
        results = search.cv_results_
        assert np.isnan(results["split0_test_score"][0])
        assert np.isfinite(results["split1_test_score"][0]) and np.isfinite(results["split2_test_score"][0])
        assert results["split0_curve_length"][0] is None
        assert set(results["rounds"][0]) == {"step_1", "step_2", "step_3"}

    def test_error_score_raise_propagates_the_fold_error(self):
        y = _series()
        smallest_train = len(next(iter(_cv().split(y)))[0]) - HORIZON
        search = _search(
            forecaster=_point(estimator=CurveRegressor(fail_below_train_rows=smallest_train + 1)), error_score="raise"
        )
        with pytest.raises(RuntimeError, match="refuses"):
            search.fit(y, forecasting_horizon=HORIZON)

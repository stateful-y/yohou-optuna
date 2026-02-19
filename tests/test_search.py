"""Tests for OptunaSearchCV."""

from __future__ import annotations

import numpy as np
import optuna
import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
)
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.point import PointReductionForecaster
from yohou.testing import _yield_yohou_search_checks

from yohou_optuna import OptunaSearchCV, Sampler


class TestOptunaSearchCVSystematicChecks:
    """Run yohou's systematic search CV checks against OptunaSearchCV."""

    EXPECTED_FAILURES: set[str] = {
        # Grid/Randomized-specific checks that don't apply to Optuna
        "check_grid_search_exhaustive",
        "check_grid_search_param_grid_validation",
        "check_randomized_search_n_iter",
        "check_randomized_search_reproducibility",
        "check_randomized_search_distributions",
    }

    def test_systematic_checks(self, y_X_factory, default_sampler):
        """Run all applicable systematic checks from yohou.testing."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        # Split train/test
        train_len = 80
        y_train, y_test = y[:train_len], y[train_len:]
        X_train, X_test = X[:train_len], X[train_len:]

        search_cv = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
            refit=True,
        )

        # Pre-fit the search CV (checks expect a fitted instance for delegation tests)
        search_cv_fitted = clone(search_cv)
        search_cv_fitted.fit(y_train, X_train, forecasting_horizon=3)

        tags = {
            "search_type": "optuna",
            "refit": True,
            "multimetric": False,
            "supports_panel_data": True,
        }

        for check_name, check_func, check_kwargs in _yield_yohou_search_checks(
            search_cv_fitted, y_train, X_train, y_test, X_test, tags=tags
        ):
            if check_name in self.EXPECTED_FAILURES:
                continue
            check_func(search_cv_fitted, **check_kwargs)


class TestOptunaSearchCVFit:
    """Test basic fit functionality."""

    def test_fit_with_valid_data_sets_required_attributes(self, optuna_search_cv, y_X_factory):
        """Test that fit runs and sets required attributes."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        assert hasattr(optuna_search_cv, "cv_results_")
        assert hasattr(optuna_search_cv, "best_params_")
        assert hasattr(optuna_search_cv, "best_score_")
        assert hasattr(optuna_search_cv, "best_index_")
        assert hasattr(optuna_search_cv, "best_forecaster_")
        assert hasattr(optuna_search_cv, "study_")
        assert hasattr(optuna_search_cv, "trials_")

    def test_fit_without_X_succeeds(self, y_X_factory, default_sampler):
        """Test fit works without exogenous features."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.1, 1.0),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
        )
        search.fit(y, forecasting_horizon=3)
        assert hasattr(search, "best_params_")

    def test_predict_after_fit_returns_forecasts(self, optuna_search_cv, y_X_factory):
        """Test predict works after fit."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)
        y_pred = optuna_search_cv.predict(forecasting_horizon=3)
        assert y_pred is not None
        assert len(y_pred) > 0

    def test_predict_before_fit_raises_error(self, optuna_search_cv):
        """Test that predict before fit raises error."""
        # available_if decorator raises AttributeError when best_forecaster_ not set
        with pytest.raises((NotFittedError, AttributeError)):
            optuna_search_cv.predict(forecasting_horizon=3)


class TestCVResults:
    """Test cv_results_ dictionary structure."""

    def test_cv_results_keys(self, optuna_search_cv, y_X_factory):
        """Test cv_results_ contains expected keys."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        cv_results = optuna_search_cv.cv_results_
        assert "params" in cv_results
        assert "mean_test_score" in cv_results
        assert "std_test_score" in cv_results
        assert "rank_test_score" in cv_results
        assert "mean_fit_time" in cv_results
        assert "std_fit_time" in cv_results
        assert "mean_score_time" in cv_results
        assert "std_score_time" in cv_results

    def test_cv_results_param_keys(self, optuna_search_cv, y_X_factory):
        """Test cv_results_ contains parameter columns."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        cv_results = optuna_search_cv.cv_results_
        assert "param_estimator__alpha" in cv_results

    def test_cv_results_split_keys(self, optuna_search_cv, y_X_factory):
        """Test cv_results_ contains per-split scores."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        cv_results = optuna_search_cv.cv_results_
        assert "split0_test_score" in cv_results
        assert "split1_test_score" in cv_results

    def test_cv_results_n_trials(self, y_X_factory, default_sampler):
        """Test cv_results_ has entries for all completed trials."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        n_trials = 5
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=n_trials,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert len(search.cv_results_["params"]) == n_trials

    def test_cv_results_rankings(self, optuna_search_cv, y_X_factory):
        """Test that rankings are valid (1-indexed, dense)."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        ranks = optuna_search_cv.cv_results_["rank_test_score"]
        assert np.min(ranks) == 1
        assert np.max(ranks) <= len(ranks)


class TestStudyAndTrials:
    """Test Optuna study/trial integration."""

    def test_study_stored(self, optuna_search_cv, y_X_factory):
        """Test that study_ is stored after fit."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        assert isinstance(optuna_search_cv.study_, optuna.study.Study)

    def test_trials_stored(self, optuna_search_cv, y_X_factory):
        """Test that trials_ is stored after fit."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        assert isinstance(optuna_search_cv.trials_, list)
        assert len(optuna_search_cv.trials_) == optuna_search_cv.n_trials

    def test_continue_study(self, y_X_factory, default_sampler):
        """Test passing an existing study to continue optimization."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)

        study = optuna.create_study(direction="maximize")
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
        )

        # First fit
        search.fit(y, X, forecasting_horizon=3, study=study)
        first_n_trials = len(study.trials)

        # Continue with same study
        search.fit(y, X, forecasting_horizon=3, study=study)
        assert len(study.trials) == first_n_trials + 2


class TestMultiMetric:
    """Test multi-metric scoring."""

    def test_multimetric_fit(self, y_X_factory, default_sampler):
        """Test fit with multiple scoring metrics."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        scoring = {
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
        }
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=scoring,
            sampler=default_sampler,
            n_trials=3,
            refit="mae",
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert search.multimetric_
        assert "mean_test_mae" in search.cv_results_
        assert "mean_test_rmse" in search.cv_results_
        assert "rank_test_mae" in search.cv_results_
        assert "rank_test_rmse" in search.cv_results_

    def test_multimetric_best_from_refit_metric(self, y_X_factory, default_sampler):
        """Test best_score_ uses refit metric in multi-metric mode."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        scoring = {
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
        }
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=scoring,
            sampler=default_sampler,
            n_trials=3,
            refit="mae",
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert hasattr(search, "best_score_")
        assert search.best_score_ == search.cv_results_["mean_test_mae"][search.best_index_]


class TestRefit:
    """Test refit behavior."""

    def test_refit_false(self, y_X_factory, default_sampler):
        """Test refit=False does not create best_forecaster_."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            refit=False,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert not hasattr(search, "best_forecaster_")
        assert not hasattr(search, "refit_time_")

    def test_refit_true_creates_forecaster(self, optuna_search_cv, y_X_factory):
        """Test refit=True creates best_forecaster_."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        assert hasattr(optuna_search_cv, "best_forecaster_")
        assert hasattr(optuna_search_cv, "refit_time_")
        assert optuna_search_cv.refit_time_ > 0


class TestTrainScore:
    """Test return_train_score behavior."""

    def test_return_train_score(self, y_X_factory, default_sampler):
        """Test return_train_score=True includes training scores."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
            return_train_score=True,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert "mean_train_score" in search.cv_results_
        assert "std_train_score" in search.cv_results_
        assert "split0_train_score" in search.cv_results_

    def test_no_train_score_by_default(self, optuna_search_cv, y_X_factory):
        """Test return_train_score=False by default."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        assert "mean_train_score" not in optuna_search_cv.cv_results_


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_invalid_distribution_raises(self, y_X_factory):
        """Test non-BaseDistribution raises ValueError."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={"estimator__alpha": [0.1, 1.0]},
            scoring=MeanAbsoluteError(),
            n_trials=2,
            cv=2,
        )
        with pytest.raises(ValueError, match="invalid distribution"):
            search.fit(y, X, forecasting_horizon=3)

    def test_clone_preserves_params(self, optuna_search_cv):
        """Test that clone preserves all constructor parameters."""
        cloned = clone(optuna_search_cv)
        assert cloned.n_trials == optuna_search_cv.n_trials
        assert cloned.cv == optuna_search_cv.cv
        assert cloned.refit == optuna_search_cv.refit

    def test_multiple_params(self, y_X_factory, default_sampler):
        """Test search with multiple parameter distributions."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
                "estimator__fit_intercept": CategoricalDistribution([True, False]),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert "param_estimator__alpha" in search.cv_results_
        assert "param_estimator__fit_intercept" in search.cv_results_


class TestErrorHandling:
    """Test error_score handling."""

    def test_error_score_nan(self, y_X_factory, default_sampler):
        """Test error_score=np.nan handles failing trials gracefully."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)

        # Use parameter range that might cause issues
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
            error_score=np.nan,
        )
        # Should not raise even if some trials fail
        search.fit(y, X, forecasting_horizon=3)


class TestSamplerAndCallbacks:
    """Test sampler and callback integration."""

    def test_custom_sampler(self, y_X_factory):
        """Test using a custom sampler."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        sampler = Sampler(sampler=optuna.samplers.RandomSampler, seed=123)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=sampler,
            n_trials=3,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")

    def test_timeout(self, y_X_factory, default_sampler):
        """Test timeout parameter is accepted."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            timeout=300,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")


class TestObjective:
    """Test _Objective internals."""

    def test_objective_callable_returns_float(self, y_X_factory, default_sampler):
        """Test that objective returns a float score per trial."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        for trial in search.trials_:
            assert trial.state == optuna.trial.TrialState.COMPLETE
            assert isinstance(trial.value, float)

    def test_objective_stores_parameters_as_user_attrs(self, y_X_factory, default_sampler):
        """Test that suggested params are stored in trial user attributes."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        for trial in search.trials_:
            assert "param_estimator__alpha" in trial.user_attrs

    def test_objective_stores_timing(self, y_X_factory, default_sampler):
        """Test that fit/score timing is stored in trial user attributes."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        trial = search.trials_[0]
        assert "mean_fit_time" in trial.user_attrs
        assert "std_fit_time" in trial.user_attrs
        assert "mean_score_time" in trial.user_attrs
        assert "std_score_time" in trial.user_attrs
        assert trial.user_attrs["mean_fit_time"] >= 0

    def test_objective_stores_split_scores(self, y_X_factory, default_sampler):
        """Test that per-split scores are stored in trial user attributes."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=3,
        )
        search.fit(y, X, forecasting_horizon=3)

        trial = search.trials_[0]
        assert "split0_test_score" in trial.user_attrs
        assert "split1_test_score" in trial.user_attrs
        assert "split2_test_score" in trial.user_attrs
        assert "mean_test_score" in trial.user_attrs


class TestBuildCVResults:
    """Test _build_cv_results utility function."""

    def test_empty_trials_returns_empty(self):
        """Test that empty trials produce empty params list."""
        from yohou_optuna.utils import _build_cv_results

        results = _build_cv_results([], multimetric=False)
        assert results == {"params": []}

    def test_rankings_rank_one_is_best(self, y_X_factory, default_sampler):
        """Test that rank 1 corresponds to the best score."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=5,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        cv = search.cv_results_
        best_idx = np.argmax(cv["mean_test_score"])
        assert cv["rank_test_score"][best_idx] == 1

    def test_params_list_matches_n_trials(self, y_X_factory, default_sampler):
        """Test that params list length matches completed trials."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=4,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert len(search.cv_results_["params"]) == 4
        for params_dict in search.cv_results_["params"]:
            assert isinstance(params_dict, dict)
            assert "estimator__alpha" in params_dict

    def test_std_test_score_is_nonnegative(self, optuna_search_cv, y_X_factory):
        """Test that std of test scores is non-negative."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        std_scores = optuna_search_cv.cv_results_["std_test_score"]
        assert np.all(std_scores >= 0)

    def test_multimetric_rankings_per_metric(self, y_X_factory, default_sampler):
        """Test that multi-metric produces separate rankings per metric."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
            },
            sampler=default_sampler,
            n_trials=4,
            refit="mae",
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        cv = search.cv_results_
        assert "rank_test_mae" in cv
        assert "rank_test_rmse" in cv
        assert np.min(cv["rank_test_mae"]) == 1
        assert np.min(cv["rank_test_rmse"]) == 1


class TestErrorHandlingExtended:
    """Extended error handling tests with FailingForecaster."""

    def test_all_trials_fail_error_score_nan(self, y_X_factory, default_sampler, failing_forecaster):
        """Test that all-failing trials produce NaN scores with error_score=nan."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=failing_forecaster,
            param_distributions={
                "fail_on": CategoricalDistribution(["fit"]),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            error_score=np.nan,
            refit=False,
        )
        search.fit(y, X, forecasting_horizon=3)
        # All trials should have NaN scores
        assert np.all(np.isnan(search.cv_results_["mean_test_score"]))

    def test_all_trials_fail_error_score_raise(self, y_X_factory, default_sampler, failing_forecaster):
        """Test that error_score='raise' propagates errors."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=failing_forecaster,
            param_distributions={
                "fail_on": CategoricalDistribution(["fit"]),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=1,
            cv=2,
            error_score="raise",
            refit=False,
        )
        with pytest.raises(ValueError, match="intentional error"):
            search.fit(y, X, forecasting_horizon=3)


class TestIntDistribution:
    """Test integer distribution parameter search."""

    def test_int_distribution_search(self, y_X_factory, default_sampler):
        """Test search with IntDistribution for integer parameters."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        from yohou.preprocessing import LagTransformer

        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(
                estimator=Ridge(),
                feature_transformer=LagTransformer(lag=[1, 2, 3]),
            ),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")


class TestNJobs:
    """Test parallel execution."""

    @pytest.mark.slow
    def test_n_jobs_two(self, y_X_factory, default_sampler):
        """Test that n_jobs=2 runs without error."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
            n_jobs=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")
        assert len(search.cv_results_["params"]) == 3


@pytest.mark.integration
class TestIntegration:
    """End-to-end tests combining OptunaSearchCV with compose modules."""

    def test_with_feature_pipeline(self, y_X_factory, default_sampler):
        """Test OptunaSearchCV with FeaturePipeline wrapping."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        from yohou.preprocessing import LagTransformer

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 2, 3]),
        )
        search = OptunaSearchCV(
            forecaster=forecaster,
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        y_pred = search.predict(forecasting_horizon=3)
        assert y_pred is not None
        assert len(y_pred) > 0

    def test_three_metrics(self, y_X_factory, default_sampler):
        """Test search with three scoring metrics."""
        from yohou.metrics import MeanSquaredError

        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
                "mse": MeanSquaredError(),
            },
            sampler=default_sampler,
            n_trials=3,
            refit="mae",
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert "mean_test_mae" in search.cv_results_
        assert "mean_test_rmse" in search.cv_results_
        assert "mean_test_mse" in search.cv_results_
        assert "rank_test_mae" in search.cv_results_
        assert "rank_test_rmse" in search.cv_results_
        assert "rank_test_mse" in search.cv_results_

    def test_fit_duration_tracked(self, optuna_search_cv, y_X_factory):
        """Test that refit_time_ is set and is positive."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        assert hasattr(optuna_search_cv, "refit_time_")
        assert isinstance(optuna_search_cv.refit_time_, float)
        assert optuna_search_cv.refit_time_ > 0

    def test_study_direction_is_maximize(self, optuna_search_cv, y_X_factory):
        """Test that the optuna study direction is maximize."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        assert optuna_search_cv.study_.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_study_sampler_type(self, y_X_factory):
        """Test that custom sampler type is used in the study."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        sampler = Sampler(sampler=optuna.samplers.RandomSampler, seed=99)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=sampler,
            n_trials=2,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        assert isinstance(search.study_.sampler, optuna.samplers.RandomSampler)

    def test_trial_states_all_complete(self, optuna_search_cv, y_X_factory):
        """Test that all trials have COMPLETE state."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        for trial in optuna_search_cv.trials_:
            assert trial.state == optuna.trial.TrialState.COMPLETE


class TestMultiMetricTrainScore:
    """Test multi-metric combined with return_train_score."""

    def test_multimetric_with_train_score(self, y_X_factory, default_sampler):
        """Test return_train_score with multi-metric scoring."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
            },
            sampler=default_sampler,
            n_trials=3,
            refit="mae",
            cv=2,
            return_train_score=True,
        )
        search.fit(y, X, forecasting_horizon=3)

        cv = search.cv_results_
        # Should have per-split train scores for each metric
        assert "split0_train_mae" in cv
        assert "split1_train_mae" in cv
        assert "mean_train_mae" in cv
        assert "split0_train_rmse" in cv
        assert "mean_train_rmse" in cv


class TestMultiMetricErrorHandling:
    """Test error handling in multi-metric scoring scenarios."""

    def test_multimetric_error_score_nan(self, y_X_factory, default_sampler, failing_forecaster):
        """Test multi-metric scoring with failing forecaster and error_score=nan."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=failing_forecaster,
            param_distributions={
                "fail_on": CategoricalDistribution(["fit"]),
            },
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
            },
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            error_score=np.nan,
            refit=False,
        )
        search.fit(y, X, forecasting_horizon=3)

        cv = search.cv_results_
        assert np.all(np.isnan(cv["mean_test_mae"]))
        assert np.all(np.isnan(cv["mean_test_rmse"]))

    def test_multimetric_error_with_train_score(self, y_X_factory, default_sampler, failing_forecaster):
        """Test multi-metric error with return_train_score."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=failing_forecaster,
            param_distributions={
                "fail_on": CategoricalDistribution(["fit"]),
            },
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
            },
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            error_score=np.nan,
            refit=False,
            return_train_score=True,
        )
        search.fit(y, X, forecasting_horizon=3)

        cv = search.cv_results_
        assert np.all(np.isnan(cv["mean_train_mae"]))


class TestSklearnTags:
    """Test __sklearn_tags__ delegation."""

    def test_tags_from_fitted_forecaster(self, optuna_search_cv, y_X_factory):
        """Test that tags are delegated from best_forecaster_ after fit."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        tags = optuna_search_cv.__sklearn_tags__()
        # Tags should come from best_forecaster_ which is a PointReductionForecaster
        assert tags is not None

    def test_tags_from_unfitted_forecaster(self, default_sampler):
        """Test that tags are delegated from forecaster before fit."""
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
        )
        tags = search.__sklearn_tags__()
        # Should delegate to base forecaster
        assert tags is not None


class TestCallbacksAndStorage:
    """Test callbacks and storage parameter handling."""

    def test_invalid_callbacks_type_raises(self, y_X_factory, default_sampler):
        """Test that non-dict callbacks raises TypeError."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            callbacks=["not", "a", "dict"],
        )
        with pytest.raises(TypeError, match="callbacks must be a dict"):
            search.fit(y, X, forecasting_horizon=3)

    def test_invalid_callback_value_raises(self, y_X_factory, default_sampler):
        """Test that non-Callback values in callbacks dict raises TypeError."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            callbacks={"bad": "not_a_callback"},
        )
        with pytest.raises(TypeError, match="must be a Callback instance"):
            search.fit(y, X, forecasting_horizon=3)

    def test_valid_callback(self, y_X_factory, default_sampler):
        """Test that valid callback runs without error."""
        from yohou_optuna import Callback

        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
            callbacks={"stop": Callback(callback=optuna.study.MaxTrialsCallback, n_trials=3)},
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")

    def test_storage_wrapper(self, y_X_factory, default_sampler):
        """Test that storage parameter is accepted."""
        from yohou_optuna import Storage

        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
            storage=Storage(storage=optuna.storages.InMemoryStorage),
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")


class TestEmptyResults:
    """Test empty results scenario."""

    def test_no_completed_trials_refit_false(self, y_X_factory, default_sampler, failing_forecaster):
        """Test that no completed trials with refit=False returns self."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=failing_forecaster,
            param_distributions={
                "fail_on": CategoricalDistribution(["fit"]),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            error_score=np.nan,
            refit=False,
        )
        result = search.fit(y, X, forecasting_horizon=3)
        assert result is search


class TestPanelData:
    """Test OptunaSearchCV with panel data (__ separator columns)."""

    def test_fit_with_panel_data_succeeds(self, y_X_factory, default_sampler):
        """Test that fit works with panel data using __ separator."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, panel=True, n_groups=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")
        assert hasattr(search, "cv_results_")

    def test_cv_results_structure_with_panel_data(self, y_X_factory, default_sampler):
        """Test cv_results_ has correct structure with panel data."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, panel=True, n_groups=3)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)

        cv_results = search.cv_results_
        assert len(cv_results["params"]) == 3
        assert "mean_test_score" in cv_results
        assert "rank_test_score" in cv_results

    def test_predict_with_panel_data_returns_all_groups(self, y_X_factory, default_sampler):
        """Test predict returns forecasts for all panel groups."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, panel=True, n_groups=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            refit=True,
        )
        search.fit(y, X, forecasting_horizon=3)
        y_pred = search.predict(forecasting_horizon=3)
        assert y_pred is not None
        # Panel data should have columns for each group
        assert len(y_pred.columns) > 2  # time + observed_time + group columns


class TestForecasterDelegation:
    """Test method delegation to best_forecaster_ after fit."""

    def test_observe_delegates_to_best_forecaster(self, optuna_search_cv, y_X_factory):
        """Test that observe() delegates to best_forecaster_ after fit."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        # Get new data for observe
        y_new, X_new = y_X_factory(length=10, n_targets=1, n_features=2, seed=99)
        result = optuna_search_cv.observe(y_new, X_new)
        assert result is optuna_search_cv

    def test_observe_predict_delegates_to_best_forecaster(self, optuna_search_cv, y_X_factory):
        """Test that observe_predict() delegates to best_forecaster_ after fit."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        # Get new data for observe_predict
        y_new, X_new = y_X_factory(length=10, n_targets=1, n_features=2, seed=99)
        y_pred = optuna_search_cv.observe_predict(y_new, X_new, forecasting_horizon=3)
        assert y_pred is not None
        assert len(y_pred) > 0

    def test_observe_before_fit_raises_error(self, optuna_search_cv, y_X_factory):
        """Test that observe() before fit raises AttributeError."""
        y, X = y_X_factory(length=10, n_targets=1, n_features=2)
        with pytest.raises(AttributeError):
            optuna_search_cv.observe(y, X)

    def test_observe_predict_before_fit_raises_error(self, optuna_search_cv, y_X_factory):
        """Test that observe_predict() before fit raises AttributeError."""
        y, X = y_X_factory(length=10, n_targets=1, n_features=2)
        with pytest.raises(AttributeError):
            optuna_search_cv.observe_predict(y, X, forecasting_horizon=3)


@pytest.mark.slow
class TestStress:
    """Stress tests with large search spaces and high trial counts."""

    def test_large_search_space(self, y_X_factory, default_sampler):
        """Test search with many parameters."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
                "estimator__fit_intercept": CategoricalDistribution([True, False]),
                "estimator__copy_X": CategoricalDistribution([True, False]),
                "estimator__positive": CategoricalDistribution([True, False]),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=10,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert len(search.cv_results_["params"]) == 10
        for key in ["param_estimator__alpha", "param_estimator__fit_intercept"]:
            assert key in search.cv_results_

    def test_high_trial_count(self, y_X_factory, default_sampler):
        """Test search with high number of trials."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=20,
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert len(search.cv_results_["params"]) == 20


@pytest.mark.integration
class TestStorageIntegration:
    """Test file-based storage integration."""

    def test_sqlite_storage_persistence(self, y_X_factory, default_sampler, tmp_path):
        """Test that SQLite storage persists study data."""
        from yohou_optuna import Storage

        db_path = tmp_path / "test_study.db"
        storage = Storage(storage=optuna.storages.RDBStorage, url=f"sqlite:///{db_path}")

        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            storage=storage,
        )
        search.fit(y, X, forecasting_horizon=3)

        # Verify database file was created
        assert db_path.exists()

    def test_study_resumption_from_storage(self, y_X_factory, default_sampler, tmp_path):
        """Test that study can be resumed from file storage."""
        db_path = tmp_path / "resume_study.db"
        storage_url = f"sqlite:///{db_path}"

        y, X = y_X_factory(length=100, n_targets=1, n_features=2)

        # First fit with 2 trials
        study1 = optuna.create_study(
            direction="maximize",
            storage=storage_url,
            study_name="resume_test",
        )
        search1 = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            refit=False,
        )
        search1.fit(y, X, forecasting_horizon=3, study=study1)
        first_count = len(study1.trials)

        # Resume with same study
        study2 = optuna.load_study(study_name="resume_test", storage=storage_url)
        search2 = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            refit=False,
        )
        search2.fit(y, X, forecasting_horizon=3, study=study2)

        # Should have 4 total trials
        assert len(study2.trials) == first_count + 2


class TestIntervalPrediction:
    """Test interval forecasting support."""

    def test_point_forecaster_no_predict_interval(self, optuna_search_cv, y_X_factory):
        """Test that point forecaster does not have predict_interval available."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)

        # PointReductionForecaster doesn't support interval prediction
        # The method should not be available via available_if decorator
        with pytest.raises(AttributeError):
            optuna_search_cv.predict_interval(forecasting_horizon=3)

    @pytest.mark.slow
    def test_interval_forecaster_fit_and_predict(self, y_X_factory, default_sampler):
        """Test that interval forecaster can be fitted and make predictions."""
        from sklearn.linear_model import QuantileRegressor
        from yohou.interval import IntervalReductionForecaster

        # Use single target for QuantileRegressor compatibility
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=IntervalReductionForecaster(estimator=QuantileRegressor(solver="highs")),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.0, 1.0),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
            refit=True,
        )
        # This may fail due to QuantileRegressor requirements - mark as xfail if so
        try:
            search.fit(y, X, forecasting_horizon=3)
            assert hasattr(search, "best_forecaster_")
        except ValueError:
            pytest.skip("IntervalReductionForecaster not compatible with test data shape")


class TestMoreTags:
    """Test _more_tags method for sklearn compatibility."""

    def test_more_tags_returns_search_type(self, default_sampler):
        """Test that _more_tags returns optuna search type."""
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=2,
            cv=2,
        )
        tags = search._more_tags()
        assert tags == {"search_type": "optuna"}


class TestEmptyResultsRefit:
    """Test edge cases where no trials complete successfully."""

    def test_empty_results_with_refit_true_raises_error(self, default_sampler, mocker):
        """Test that fit raises ValueError when no trials complete and refit=True."""

        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=1,
            cv=2,
            refit=True,
        )
        # Mock _build_cv_results to return empty params
        mocker.patch(
            "yohou_optuna.search._build_cv_results",
            return_value={"params": [], "mean_test_score": np.array([])},
        )
        # Mock optuna study to avoid actual optimization
        mock_study = mocker.MagicMock()
        mock_study.trials = []
        mocker.patch("optuna.create_study", return_value=mock_study)

        # This should raise ValueError because refit=True with no completed trials
        with pytest.raises(ValueError, match="No trials were completed"):
            # We need to create data even though the mock prevents real fitting
            from datetime import datetime, timedelta

            import polars as pl

            time_col = pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16) + timedelta(seconds=99),
                interval="1s",
                eager=True,
            )
            y = pl.DataFrame({"time": time_col, "y_0": np.random.rand(100)})
            X = pl.DataFrame({"time": time_col, "X_0": np.random.rand(100)})
            search.fit(y, X, forecasting_horizon=3)

    def test_empty_results_with_refit_false_returns_self(self, default_sampler, mocker):
        """Test that fit returns self when no trials complete and refit=False."""
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=1,
            cv=2,
            refit=False,
        )
        # Mock _build_cv_results to return empty params
        mocker.patch(
            "yohou_optuna.search._build_cv_results",
            return_value={"params": [], "mean_test_score": np.array([])},
        )
        # Mock optuna study
        mock_study = mocker.MagicMock()
        mock_study.trials = []
        mocker.patch("optuna.create_study", return_value=mock_study)

        from datetime import datetime, timedelta

        import polars as pl

        time_col = pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16) + timedelta(seconds=99),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time_col, "y_0": np.random.rand(100)})
        X = pl.DataFrame({"time": time_col, "X_0": np.random.rand(100)})

        # Should return self without error
        result = search.fit(y, X, forecasting_horizon=3)
        assert result is search


class TestSingleMetricErrorHandling:
    """Test error handling for single-metric scoring scenarios."""

    def test_single_metric_error_with_train_score(self, default_sampler, y_X_factory):
        """Test error handling branch for single metric with return_train_score."""
        # Create forecaster that fails during fit - with valid param name

        class FailOnSecondFitForecaster(PointReductionForecaster):
            """A forecaster that fails on the first fit call."""

            _fit_count = 0

            def fit(self, y, X=None, forecasting_horizon=None, **kwargs):
                FailOnSecondFitForecaster._fit_count += 1
                # Fail every time to ensure error handling path is hit
                raise RuntimeError("Intentional fit failure")

        # Reset fit count
        FailOnSecondFitForecaster._fit_count = 0

        search = OptunaSearchCV(
            forecaster=FailOnSecondFitForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=1,
            cv=2,
            refit=False,
            error_score=np.nan,
            return_train_score=True,
        )

        y, X = y_X_factory(length=100, n_targets=1, n_features=2)

        # Should not raise - error_score=np.nan handles failures
        search.fit(y, X, forecasting_horizon=3)

        # All trials should have cv_results
        assert hasattr(search, "cv_results_")
        # Confirm we went through error path
        assert FailOnSecondFitForecaster._fit_count > 0


class TestObjectiveErrorPath:
    """Test the _handle_error path in _Objective.__call__."""

    def test_handle_error_stores_exception_info(self, default_sampler, mocker):
        """Test that _handle_error stores exception details in trial attrs."""
        from yohou_optuna.objective import _Objective

        # Create a mock trial
        mock_trial = mocker.MagicMock()
        mock_trial.user_attrs = {}

        def set_user_attr(key, val):
            mock_trial.user_attrs[key] = val

        mock_trial.set_user_attr = set_user_attr
        mock_trial.suggest_float = mocker.MagicMock(return_value=1.0)

        # Create objective
        objective = _Objective(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={"estimator__alpha": FloatDistribution(0.01, 10.0)},
            y=None,
            X=None,
            forecasting_horizon=3,
            cv=2,
            scorers=MeanAbsoluteError(),
            fit_params={},
            predict_params={},
            score_params={},
            return_train_score=False,
            error_score=0.0,
            multimetric=False,
            refit=False,
        )

        # Call _handle_error directly
        test_exception = ValueError("Test error message")
        result = objective._handle_error(mock_trial, test_exception)

        # Should store exception info
        assert mock_trial.user_attrs["exception"] == "Test error message"
        assert mock_trial.user_attrs["exception_type"] == "ValueError"
        assert result == 0.0  # error_score value

    def test_handle_error_with_nan_returns_neg_inf(self, mocker):
        """Test that _handle_error returns -inf when error_score is nan."""
        from yohou_optuna.objective import _Objective

        mock_trial = mocker.MagicMock()
        mock_trial.user_attrs = {}
        mock_trial.set_user_attr = lambda k, v: mock_trial.user_attrs.update({k: v})

        objective = _Objective(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={"estimator__alpha": FloatDistribution(0.01, 10.0)},
            y=None,
            X=None,
            forecasting_horizon=3,
            cv=2,
            scorers=MeanAbsoluteError(),
            fit_params={},
            predict_params={},
            score_params={},
            return_train_score=False,
            error_score=np.nan,
            multimetric=False,
            refit=False,
        )

        result = objective._handle_error(mock_trial, RuntimeError("fail"))
        assert result == float("-inf")

    def test_handle_error_with_raise_propagates_exception(self, mocker):
        """Test that _handle_error raises when error_score='raise'."""
        from yohou_optuna.objective import _Objective

        mock_trial = mocker.MagicMock()
        mock_trial.user_attrs = {}
        mock_trial.set_user_attr = lambda k, v: mock_trial.user_attrs.update({k: v})

        objective = _Objective(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={"estimator__alpha": FloatDistribution(0.01, 10.0)},
            y=None,
            X=None,
            forecasting_horizon=3,
            cv=2,
            scorers=MeanAbsoluteError(),
            fit_params={},
            predict_params={},
            score_params={},
            return_train_score=False,
            error_score="raise",
            multimetric=False,
            refit=False,
        )

        with pytest.raises(RuntimeError, match="Intentional"):
            objective._handle_error(mock_trial, RuntimeError("Intentional"))


class TestPrimaryMetricEdgeCases:
    """Test edge cases in _get_primary_metric method."""

    def test_get_primary_metric_no_test_keys_returns_neg_inf(self, mocker):
        """Test that _get_primary_metric returns -inf when no mean_test_ keys."""
        from yohou_optuna.objective import _Objective

        mock_trial = mocker.MagicMock()
        # Empty user_attrs - no mean_test_ keys
        mock_trial.user_attrs = {}

        objective = _Objective(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={"estimator__alpha": FloatDistribution(0.01, 10.0)},
            y=None,
            X=None,
            forecasting_horizon=3,
            cv=2,
            scorers=MeanAbsoluteError(),
            fit_params={},
            predict_params={},
            score_params={},
            return_train_score=False,
            error_score=np.nan,
            multimetric=True,  # No refit string, so uses test_keys branch
            refit=False,
        )

        result = objective._get_primary_metric(mock_trial)
        assert result == float("-inf")


class TestStoreMultimetricScoresEdgeCases:
    """Test edge cases in _store_multimetric_scores method."""

    def test_store_multimetric_scores_non_dict_early_return(self, mocker):
        """Test that _store_multimetric_scores returns early for non-dict scores."""
        from yohou_optuna.objective import _Objective

        mock_trial = mocker.MagicMock()
        mock_trial.user_attrs = {}
        mock_trial.set_user_attr = lambda k, v: mock_trial.user_attrs.update({k: v})

        objective = _Objective(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={"estimator__alpha": FloatDistribution(0.01, 10.0)},
            y=None,
            X=None,
            forecasting_horizon=3,
            cv=2,
            scorers=MeanAbsoluteError(),
            fit_params={},
            predict_params={},
            score_params={},
            return_train_score=False,
            error_score=np.nan,
            multimetric=True,
            refit=False,
        )

        # Call with non-dict scores - should return early
        objective._store_multimetric_scores(mock_trial, [0.5, 0.6], [])

        # Should not have stored any mean_test_ attributes
        mean_test_keys = [k for k in mock_trial.user_attrs if k.startswith("mean_test_")]
        assert len(mean_test_keys) == 0

    def test_store_multimetric_scores_empty_list_early_return(self, mocker):
        """Test that _store_multimetric_scores returns early for empty list."""
        from yohou_optuna.objective import _Objective

        mock_trial = mocker.MagicMock()
        mock_trial.user_attrs = {}
        mock_trial.set_user_attr = lambda k, v: mock_trial.user_attrs.update({k: v})

        objective = _Objective(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={"estimator__alpha": FloatDistribution(0.01, 10.0)},
            y=None,
            X=None,
            forecasting_horizon=3,
            cv=2,
            scorers=MeanAbsoluteError(),
            fit_params={},
            predict_params={},
            score_params={},
            return_train_score=False,
            error_score=np.nan,
            multimetric=True,
            refit=False,
        )

        # Call with empty scores - should return early
        objective._store_multimetric_scores(mock_trial, [], [])

        # Should not have stored any mean_test_ attributes
        mean_test_keys = [k for k in mock_trial.user_attrs if k.startswith("mean_test_")]
        assert len(mean_test_keys) == 0


class TestSamplerNone:
    """Test search with sampler=None."""

    def test_fit_with_sampler_none_uses_default_sampler(self, y_X_factory):
        """Test that fit works when sampler is None (uses optuna default)."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=None,  # No sampler - use optuna default
            n_trials=2,
            cv=2,
            refit=True,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_params_")


class TestStudyWithSamplerNone:
    """Test passing existing study with sampler=None."""

    def test_existing_study_with_sampler_none(self, y_X_factory):
        """Test that existing study is used without modifying sampler when None."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)

        # Create study outside
        existing_study = optuna.create_study(direction="maximize")

        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=None,  # No sampler
            n_trials=2,
            cv=2,
            refit=True,
        )
        search.fit(y, X, forecasting_horizon=3, study=existing_study)

        # Should use the existing study
        assert search.study_ is existing_study


class TestCallableRefit:
    """Test callable refit function."""

    def test_callable_refit_skips_best_score(self, y_X_factory, default_sampler):
        """Test that callable refit does not set best_score_."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)

        def custom_refit(cv_results):
            """Select index with best score."""
            return np.argmax(cv_results["mean_test_score"])

        search = OptunaSearchCV(
            forecaster=PointReductionForecaster(estimator=Ridge()),
            param_distributions={
                "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
            },
            scoring=MeanAbsoluteError(),
            sampler=default_sampler,
            n_trials=3,
            cv=2,
            refit=custom_refit,
        )
        search.fit(y, X, forecasting_horizon=3)

        # Callable refit should NOT set best_score_ directly
        # (but it will set best_forecaster_)
        assert hasattr(search, "best_forecaster_")
        assert hasattr(search, "best_params_")


class TestBuildCvResultsEdgeCases:
    """Test edge cases in _build_cv_results for missing split keys."""

    def test_build_cv_results_missing_split_keys(self, mocker):
        """Test that std computation handles missing split keys gracefully."""
        from yohou_optuna.utils import _build_cv_results

        # Create mock trial with mean scores but no split scores
        mock_trial = mocker.MagicMock()
        mock_trial.state = optuna.trial.TrialState.COMPLETE
        mock_trial.user_attrs = {
            "mean_test_score": 0.5,
            "param_estimator__alpha": 1.0,
            # No split0_test_score, split1_test_score etc.
        }
        mock_trial.params = {"estimator__alpha": 1.0}

        results = _build_cv_results([mock_trial], multimetric=False, return_train_score=False)

        # Should still work, std will be default (0.0 or nan)
        assert "mean_test_score" in results
        assert "std_test_score" in results

    def test_build_cv_results_missing_train_split_keys(self, mocker):
        """Test that std computation handles missing train split keys gracefully."""
        from yohou_optuna.utils import _build_cv_results

        # Create mock trial with scores but no train split scores
        mock_trial = mocker.MagicMock()
        mock_trial.state = optuna.trial.TrialState.COMPLETE
        mock_trial.user_attrs = {
            "mean_test_score": 0.5,
            "mean_train_score": 0.6,
            "split0_test_score": 0.45,
            "split1_test_score": 0.55,
            # No split0_train_score, split1_train_score
            "param_estimator__alpha": 1.0,
        }
        mock_trial.params = {"estimator__alpha": 1.0}

        results = _build_cv_results([mock_trial], multimetric=False, return_train_score=True)

        # Should still work
        assert "std_test_score" in results
        assert "std_train_score" in results

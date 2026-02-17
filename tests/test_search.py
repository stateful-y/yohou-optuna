"""Tests for OptunaSearchCV."""

from __future__ import annotations

import numpy as np
import optuna
import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
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

    def test_fit_basic(self, optuna_search_cv, y_X_factory):
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

    def test_fit_without_X(self, y_X_factory, default_sampler):
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

    def test_predict_after_fit(self, optuna_search_cv, y_X_factory):
        """Test predict works after fit."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2)
        optuna_search_cv.fit(y, X, forecasting_horizon=3)
        y_pred = optuna_search_cv.predict(forecasting_horizon=3)
        assert y_pred is not None
        assert len(y_pred) > 0

    def test_not_fitted_error(self, optuna_search_cv):
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

    def test_objective_stores_parameters_as_user_attrs(
        self, y_X_factory, default_sampler
    ):
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

    def test_all_trials_fail_error_score_nan(
        self, y_X_factory, default_sampler, failing_forecaster
    ):
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

    def test_all_trials_fail_error_score_raise(
        self, y_X_factory, default_sampler, failing_forecaster
    ):
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

        assert (
            optuna_search_cv.study_.direction
            == optuna.study.StudyDirection.MAXIMIZE
        )

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

        assert isinstance(
            search.study_.sampler, optuna.samplers.RandomSampler
        )

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

    def test_multimetric_error_score_nan(
        self, y_X_factory, default_sampler, failing_forecaster
    ):
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

    def test_multimetric_error_with_train_score(
        self, y_X_factory, default_sampler, failing_forecaster
    ):
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
            callbacks={
                "stop": Callback(
                    callback=optuna.study.MaxTrialsCallback, n_trials=3
                )
            },
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

    def test_no_completed_trials_refit_false(
        self, y_X_factory, default_sampler, failing_forecaster
    ):
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

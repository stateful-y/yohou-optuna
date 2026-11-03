"""Tests for fold-level trial pruning and fold evaluation order."""

import logging
import threading
import warnings
from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import optuna
import polars as pl
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import MedianPruner, NopPruner, ThresholdPruner
from optuna.trial import TrialState
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from yohou.metrics import MeanAbsoluteError, MeanSquaredError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

from conftest import MarkerRecordingSplitter, ThresholdFailingForecaster
from shared_round_stubs import CurveEarlyStoppingAdapter, CurveRegressor
from yohou_optuna import OptunaSearchCV, Pruner, Sampler, Storage

# The default Pruner wraps Optuna's WilcoxonPruner, which Optuna marks experimental and
# warns about on every instantiation; the suite turns warnings into errors.
pytestmark = pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")

N_FOLDS = 3
TEST_SIZE = 5
GOOD, BAD = 0.0, 5.0
FLOOR = -2.0  # below every GOOD fold score (about -0.4 to -0.9), above every BAD one (about -4 to -6)


@pytest.fixture(autouse=True)
def _quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _series(length=100, seed=42):
    """A one-column series with a time column, one row per second."""
    rng = np.random.default_rng(seed)
    start = datetime(2021, 12, 16)
    time = pl.datetime_range(start, start + timedelta(seconds=length - 1), interval="1s", eager=True)
    return pl.DataFrame({"time": time, "y_0": rng.random(length)})


def _build(case, **extra):
    """One of the characterisation searches: single metric, multi-metric, or one failing fold."""
    common = {
        "param_distributions": {"estimator__alpha": FloatDistribution(0.01, 10.0, log=True)},
        "sampler": Sampler(sampler=optuna.samplers.RandomSampler, seed=0),
        "n_trials": 4,
        "cv": ExpandingWindowSplitter(n_splits=N_FOLDS, test_size=TEST_SIZE),
    }
    common.update(extra)
    if case == "single":
        forecaster = PointReductionForecaster(estimator=Ridge())
        return OptunaSearchCV(forecaster, scoring=MeanAbsoluteError(), return_train_score=True, **common)
    if case == "multi":
        forecaster = PointReductionForecaster(estimator=Ridge())
        scoring = {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}
        return OptunaSearchCV(forecaster, scoring=scoring, refit="mae", **common)
    # The three folds train on 85, 90 and 95 rows, so min_length=88 fails fold 0 only.
    forecaster = ThresholdFailingForecaster(estimator=Ridge(), min_length=88)
    common.setdefault("error_score", 0.0)
    return OptunaSearchCV(forecaster, scoring=MeanAbsoluteError(), **common)


def _snapshot(search):
    """The score columns of cv_results_ and each trial's failure marks."""
    results = search.cv_results_
    columns = {
        key: [float(value) for value in results[key]]
        for key in sorted(results)
        if key.startswith(("split", "mean_test", "mean_train"))
    }
    trials = [(t.user_attrs.get("failed_splits"), t.user_attrs.get("exception_type")) for t in search.trials_]
    return {"params": [p["estimator__alpha"] for p in results["params"]], "cols": columns, "trials": trials}


# Produced by the code before fold-level pruning existed. Every completed trial must
# still record exactly these values, under every fold order.
CHARACTERISATION = {
    "failing": {
        "cols": {
            "mean_test_score": [-0.16208607928455074, -0.1617668118037044, -0.16201227311813204, -0.1620905877538514],
            "split0_test_score": [0.0, 0.0, 0.0, 0.0],
            "split1_test_score": [
                -0.17590649160670208,
                -0.17590544297478433,
                -0.17590624810816127,
                -0.1759065065021002,
            ],
            "split2_test_score": [-0.3103517462469501, -0.30939499243632884, -0.3101305712462349, -0.31036525675945403],
        },
        "params": [0.44303752452182665, 1.3981961408994048, 0.6431172050131991, 0.4311710058685491],
        "trials": [([0], "ValueError"), ([0], "ValueError"), ([0], "ValueError"), ([0], "ValueError")],
    },
    "multi": {
        "cols": {
            "mean_test_mae": [-0.24377896881259886, -0.24153802415498885, -0.24325572670937298, -0.24381103401361368],
            "mean_test_mse": [-0.0780392717036094, -0.07716401162237062, -0.07782880415150582, -0.07805229066721255],
            "split0_test_mae": [-0.24507866858414432, -0.23931363705385333, -0.24373036077372273, -0.24516133877928672],
            "split0_test_mse": [-0.07777603475847077, -0.07571020371989934, -0.07727564390711109, -0.07780705823049806],
            "split1_test_mae": [-0.17590649160670208, -0.17590544297478433, -0.17590624810816127, -0.1759065065021002],
            "split1_test_mse": [-0.0409425707537795, -0.04094248396621443, -0.04094255060047086, -0.0409425719866206],
            "split2_test_mae": [-0.3103517462469501, -0.30939499243632884, -0.3101305712462349, -0.31036525675945403],
            "split2_test_mse": [-0.1153992095985779, -0.11483934718099809, -0.11526821794693554, -0.11540724178451897],
        },
        "params": [0.44303752452182665, 1.3981961408994048, 0.6431172050131991, 0.4311710058685491],
        "trials": [(None, None), (None, None), (None, None), (None, None)],
    },
    "single": {
        "cols": {
            "mean_test_score": [-0.24377896881259886, -0.24153802415498885, -0.24325572670937298, -0.24381103401361368],
            "mean_train_score": [-0.2193570615682272, -0.21875166762857426, -0.21921793588364236, -0.21936554347400924],
            "split0_test_score": [
                -0.24507866858414432,
                -0.23931363705385333,
                -0.24373036077372273,
                -0.24516133877928672,
            ],
            "split0_train_score": [
                -0.25441306877341563,
                -0.25704907417596906,
                -0.2550295695945997,
                -0.2543752686191788,
            ],
            "split1_test_score": [
                -0.17590649160670208,
                -0.17590544297478433,
                -0.17590624810816127,
                -0.1759065065021002,
            ],
            "split1_train_score": [-0.23082721538757128, -0.22632313340483856, -0.229781340880142, -0.2308911940754884],
            "split2_test_score": [-0.3103517462469501, -0.30939499243632884, -0.3101305712462349, -0.31036525675945403],
            "split2_train_score": [
                -0.17283090054369477,
                -0.17288279530491518,
                -0.17284289717618537,
                -0.17283016772736043,
            ],
        },
        "params": [0.44303752452182665, 1.3981961408994048, 0.6431172050131991, 0.4311710058685491],
        "trials": [(None, None), (None, None), (None, None), (None, None)],
    },
}


@pytest.mark.parametrize("case", ["single", "multi", "failing"])
def test_characterisation_unchanged(case, caplog):
    """The restructured fold loop records what the original loop recorded."""
    caplog.set_level(logging.ERROR)
    search = _build(case)
    search.fit(_series(), forecasting_horizon=TEST_SIZE)
    got, want = _snapshot(search), CHARACTERISATION[case]
    assert got["params"] == pytest.approx(want["params"])
    assert got["trials"] == [tuple(t) for t in want["trials"]]
    assert sorted(got["cols"]) == sorted(want["cols"])
    for key, values in want["cols"].items():
        assert got["cols"][key] == pytest.approx(values, rel=1e-9, abs=1e-12), key


# ---------------------------------------------------------------------------
# Re-export
# ---------------------------------------------------------------------------


def test_pruner_is_the_sklearn_optuna_class():
    import sklearn_optuna

    import yohou_optuna

    assert yohou_optuna.Pruner is sklearn_optuna.Pruner
    assert "Pruner" in yohou_optuna.__all__


def test_inert_pruner_helper_still_exists_upstream():
    """The search reuses this private sklearn-optuna helper; a rename upstream fails here first."""
    from sklearn_optuna.search import _warn_if_pruner_inert

    assert callable(_warn_if_pruner_inert)


# ---------------------------------------------------------------------------
# Building blocks: a forecaster whose score is set by one parameter
# ---------------------------------------------------------------------------


class LevelRegressor(RegressorMixin, BaseEstimator):
    """Predicts ``level`` for every output; records each fit, since folds clone it."""

    fits: ClassVar[list[float]] = []
    lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, level=0.0):
        self.level = level

    def fit(self, X, y):
        with LevelRegressor.lock:
            LevelRegressor.fits.append(self.level)
        y = np.asarray(y)
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        return self

    def predict(self, X):
        shape = len(X) if self.n_outputs_ == 1 else (len(X), self.n_outputs_)
        return np.full(shape, self.level)


class PrunedFitForecaster(PointReductionForecaster):
    """A reduction forecaster whose fit raises Optuna's TrialPruned."""

    def fit(self, y, X_actual=None, forecasting_horizon=1, **fit_params):
        raise optuna.TrialPruned("pruned from inside fit")


@pytest.fixture(autouse=True)
def _reset_fits():
    LevelRegressor.fits = []
    yield
    LevelRegressor.fits = []


def _sine(length=120):
    """A series centred at zero, so predicting 0 is good and predicting 5 is bad on every fold."""
    start = datetime(2021, 12, 16)
    time = pl.datetime_range(start, start + timedelta(seconds=length - 1), interval="1s", eager=True)
    return pl.DataFrame({"time": time, "y_0": np.sin(np.arange(length) / 3.0)})


def _level_search(levels=(GOOD, BAD), n_splits=5, forecaster=None, **kwargs):
    """A search over the level of a LevelRegressor; five folds train on 95 to 115 rows."""
    kwargs.setdefault("n_trials", 8)
    kwargs.setdefault("refit", False)
    return OptunaSearchCV(
        forecaster if forecaster is not None else PointReductionForecaster(estimator=LevelRegressor()),
        {"estimator__level": CategoricalDistribution(list(levels))},
        scoring=kwargs.pop("scoring", MeanAbsoluteError()),
        sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=0),
        cv=ExpandingWindowSplitter(n_splits=n_splits, test_size=TEST_SIZE),
        **kwargs,
    )


def _threshold(**params):
    return Pruner(pruner=ThresholdPruner, lower=FLOOR, **params)


def _fit_collecting(search, y, **fit_params):
    """Fit, and return the messages of the UserWarnings raised (Optuna's experimental notice excluded)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(y, forecasting_horizon=TEST_SIZE, **fit_params)
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and not issubclass(w.category, ExperimentalWarning)
    ]


def _with_level(search, level):
    return [t for t in search.trials_ if t.params["estimator__level"] == level]


@pytest.fixture
def report_log(monkeypatch):
    """Every (trial number, step) reported, in reporting order, across threads."""
    log, lock = [], threading.Lock()
    original = optuna.trial.Trial.report

    def report(self, value, step):
        with lock:
            log.append((self.number, step))
        return original(self, value, step)

    monkeypatch.setattr(optuna.trial.Trial, "report", report)
    return log


def _orders(log):
    orders = {}
    for number, step in log:
        orders.setdefault(number, []).append(step)
    return orders


# ---------------------------------------------------------------------------
# Folds are fixed for the whole fit
# ---------------------------------------------------------------------------


class TestFoldsFixed:
    def test_one_split_per_fit(self):
        splitter = MarkerRecordingSplitter(n_splits=4, test_size=TEST_SIZE)
        search = _level_search(n_trials=3)
        search.set_params(cv=splitter)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert len(splitter.split_calls_) == 1
        assert search.n_splits_ == 4
        assert all(sorted(t.intermediate_values) == [0, 1, 2, 3] for t in search.trials_)

    def test_splitter_failure_raises_before_any_trial(self):
        search = _level_search(n_splits=5)
        with pytest.raises(ValueError, match=r"."):
            search.fit(_sine(length=12), forecasting_horizon=TEST_SIZE)
        assert not hasattr(search, "study_")


# ---------------------------------------------------------------------------
# Finite fold scores are reported
# ---------------------------------------------------------------------------


class TestReporting:
    def test_reported_values_without_a_pruner(self):
        search = _level_search(n_trials=2)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        for trial in search.trials_:
            assert sorted(trial.intermediate_values) == [0, 1, 2, 3, 4]
            for step, value in trial.intermediate_values.items():
                assert value == trial.user_attrs[f"split{step}_test_score"]

    def test_reported_metric_when_refit_names_a_metric(self):
        search = _level_search(n_trials=2, scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}, refit="mse")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        for trial in search.trials_:
            for step, value in trial.intermediate_values.items():
                assert value == trial.user_attrs[f"split{step}_test_mse"]

    @pytest.mark.parametrize("refit", [False, lambda cv_results: 0], ids=["refit-false", "refit-callable"])
    def test_reported_metric_when_refit_does_not_name_a_metric(self, refit):
        search = _level_search(n_trials=2, scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}, refit=refit)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        for trial in search.trials_:
            assert len(trial.intermediate_values) == 5
            for step, value in trial.intermediate_values.items():
                assert value == trial.user_attrs[f"split{step}_test_mae"]

    def test_failed_fold_with_numeric_error_score_is_reported(self, caplog):
        caplog.set_level(logging.ERROR)
        search = _build("failing", n_trials=1, fold_order="chronological")
        search.fit(_series(), forecasting_horizon=TEST_SIZE)
        trial = search.trials_[0]
        assert trial.intermediate_values[0] == 0.0
        assert sorted(trial.intermediate_values) == [0, 1, 2]

    def test_failed_fold_with_nan_error_score_is_not_reported(self, caplog):
        caplog.set_level(logging.ERROR)
        search = _build("failing", n_trials=1, error_score=np.nan)
        search.fit(_series(), forecasting_horizon=TEST_SIZE)
        trial = search.trials_[0]
        assert sorted(trial.intermediate_values) == [1, 2]
        assert np.isnan(trial.user_attrs["split0_test_score"])

    def test_shared_round_validation_without_a_pruner_reports_every_fold(self):
        search = _shared_round_search()
        search.fit(_trend(), forecasting_horizon=3)
        trial = search.trials_[0]
        assert sorted(trial.intermediate_values) == [0, 1, 2]
        assert all(np.isfinite(v) for v in trial.intermediate_values.values())


def _trend(n=132):
    rng = np.random.default_rng(11)
    start = datetime(2023, 1, 1)
    times = pl.datetime_range(start, start + timedelta(hours=n - 1), "1h", eager=True)
    return pl.DataFrame({"time": times, "value": 10.0 + np.arange(n) * 0.2 + rng.normal(0, 0.3, n)})


def _shared_round_search(**kwargs):
    """A validation="cv" search over the shared-round stubs, as in test_validation_cv."""
    forecaster = PointReductionForecaster(
        estimator=CurveRegressor(), reduction_strategy="direct", actual_transformer=LagTransformer(lag=[1, 2])
    )
    return OptunaSearchCV(
        forecaster=forecaster,
        param_distributions={"estimator__patience": CategoricalDistribution([6])},
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=0),
        n_trials=1,
        cv=ExpandingWindowSplitter(n_splits=3, test_size=12),
        validation="cv",
        early_stopping_adapter=CurveEarlyStoppingAdapter(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Pruner parameter
# ---------------------------------------------------------------------------


class TestPrunerParameter:
    def test_new_study_uses_the_given_pruner(self):
        search = _level_search(n_trials=1, pruner=Pruner(pruner=MedianPruner), fold_order="chronological")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert isinstance(search.study_.pruner, MedianPruner)

    def test_passed_study_takes_the_given_pruner(self):
        study = optuna.create_study(direction="maximize", pruner=NopPruner())
        search = _level_search(n_trials=1, pruner=Pruner(pruner=MedianPruner), fold_order="chronological")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE, study=study)
        assert isinstance(study.pruner, MedianPruner)

    def test_passed_study_keeps_its_pruner_when_none_given(self):
        own = NopPruner()
        study = optuna.create_study(direction="maximize", pruner=own)
        search = _level_search(n_trials=1)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE, study=study)
        assert study.pruner is own

    def test_raw_pruner_instance_is_rejected_before_any_study(self):
        search = _level_search(n_trials=1, pruner=MedianPruner())
        with pytest.raises(TypeError, match="pruner must be a Pruner instance"):
            search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert not hasattr(search, "study_")


# ---------------------------------------------------------------------------
# Stopping trials
# ---------------------------------------------------------------------------


class TestStopping:
    def test_no_pruning_without_a_pruner(self):
        study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=0))
        search = _level_search()
        search.fit(_sine(), forecasting_horizon=TEST_SIZE, study=study)
        assert _with_level(search, BAD), "the sampler never drew the bad level"
        assert all(t.state == TrialState.COMPLETE for t in search.trials_)

    def test_pruning_with_a_pruner(self):
        search = _level_search(pruner=_threshold(), fold_order="chronological")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        bad, good = _with_level(search, BAD), _with_level(search, GOOD)
        assert bad and good
        assert all(t.state == TrialState.PRUNED for t in bad)
        assert all(t.state == TrialState.COMPLETE for t in good)
        # Each bad trial fitted its first fold only; each good trial all five.
        assert LevelRegressor.fits.count(BAD) == len(bad)
        assert LevelRegressor.fits.count(GOOD) == 5 * len(good)

    def test_pruned_trial_is_not_an_error(self):
        search = _level_search(pruner=_threshold(), fold_order="chronological", error_score="raise")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        pruned = [t for t in search.trials_ if t.state == TrialState.PRUNED]
        assert pruned
        assert all("exception" not in t.user_attrs for t in pruned)

    def test_user_code_raising_trial_pruned_under_raise(self):
        search = _level_search(
            n_trials=1, forecaster=PrunedFitForecaster(estimator=LevelRegressor()), error_score="raise"
        )
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert [t.state for t in search.trials_] == [TrialState.PRUNED]
        assert search.n_pruned_ == 1

    def test_user_code_raising_trial_pruned_under_numeric_error_score(self, caplog):
        caplog.set_level(logging.ERROR)
        search = _level_search(n_trials=1, forecaster=PrunedFitForecaster(estimator=LevelRegressor()), error_score=0.0)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        trial = search.trials_[0]
        assert trial.state == TrialState.COMPLETE
        assert trial.user_attrs["failed_splits"] == [0, 1, 2, 3, 4]
        assert all(trial.user_attrs[f"split{i}_test_score"] == 0.0 for i in range(5))


# ---------------------------------------------------------------------------
# Pruned trials: partial results, counts, refit
# ---------------------------------------------------------------------------


class TestPrunedTrials:
    def test_partial_results_after_three_folds(self):
        search = _level_search(pruner=_threshold(n_warmup_steps=2), fold_order="chronological")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        pruned = [t for t in search.trials_ if t.state == TrialState.PRUNED]
        assert pruned
        for trial in pruned:
            split_keys = {k for k in trial.user_attrs if k.startswith("split")}
            assert split_keys == {
                f"split{i}_{kind}" for i in range(3) for kind in ("test_score", "fit_time", "score_time")
            }
            assert trial.user_attrs["param_estimator__level"] == BAD
            assert "mean_test_score" not in trial.user_attrs

    def test_failed_fold_before_the_prune_keeps_its_failure_record(self, caplog):
        caplog.set_level(logging.ERROR)
        # Folds train on 95, 100, ... rows, so min_length=98 fails fold 0 only.
        forecaster = ThresholdFailingForecaster(estimator=LevelRegressor(), min_length=98)
        search = _level_search(forecaster=forecaster, pruner=_threshold(), fold_order="chronological", error_score=0.0)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        pruned = [t for t in search.trials_ if t.state == TrialState.PRUNED]
        assert pruned
        for trial in pruned:
            assert trial.user_attrs["failed_splits"] == [0]
            assert trial.user_attrs["exception_type"] == "ValueError"
            assert sorted(trial.intermediate_values) == [0, 1]

    def test_count_with_pruning(self):
        search = _level_search(pruner=_threshold(), fold_order="chronological")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        n_pruned = sum(t.state == TrialState.PRUNED for t in search.study_.trials)
        assert search.n_pruned_ == n_pruned > 0
        assert len(search.cv_results_["params"]) == search.n_completed_ == len(search.trials_) - n_pruned

    def test_count_without_pruning(self):
        search = _level_search(n_trials=2)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert search.n_pruned_ == 0

    def test_best_parameters_come_from_a_completed_trial(self):
        search = _level_search(pruner=_threshold(), fold_order="chronological", refit=True)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert search.best_params_ == {"estimator__level": GOOD}

    def test_everything_pruned_with_refit(self):
        search = _level_search(levels=(BAD,), n_trials=3, pruner=_threshold(), fold_order="chronological", refit=True)
        with pytest.raises(ValueError, match="No trials were completed"):
            search.fit(_sine(), forecasting_horizon=TEST_SIZE)

    def test_everything_pruned_without_refit(self):
        search = _level_search(levels=(BAD,), n_trials=3, pruner=_threshold(), fold_order="chronological")
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert search.cv_results_ == {"params": []}
        assert search.n_pruned_ == 3
        assert not hasattr(search, "best_params_")


# ---------------------------------------------------------------------------
# Warnings and refusals
# ---------------------------------------------------------------------------

INERT = "cannot stop any trial"
ORDER = "highest-numbered fold"


class TestWarnings:
    def test_wilcoxon_with_too_few_folds(self):
        search = _level_search(n_splits=3, pruner=Pruner())
        messages = _fit_collecting(search, _sine())
        inert = [m for m in messages if INERT in m]
        assert len(inert) == 1
        assert "at least 4 folds" in inert[0]
        assert search.n_pruned_ == 0

    def test_median_warmup_covers_every_fold(self):
        search = _level_search(pruner=Pruner(pruner=MedianPruner, n_warmup_steps=5), fold_order="chronological")
        messages = _fit_collecting(search, _sine())
        assert [m for m in messages if INERT in m and "n_warmup_steps=5" in m]

    def test_no_inert_warning_on_the_default_splitter(self):
        search = OptunaSearchCV(
            PointReductionForecaster(estimator=LevelRegressor()),
            {"estimator__level": CategoricalDistribution([GOOD])},
            scoring=MeanAbsoluteError(),
            n_trials=1,
            refit=False,
            pruner=Pruner(),
        )
        messages = _fit_collecting(search, _sine())
        assert search.n_splits_ == 5
        assert not [m for m in messages if INERT in m]

    def test_order_sensitive_pruner_with_the_default_order_warns(self):
        search = _level_search(n_trials=1, pruner=Pruner(pruner=MedianPruner))
        messages = [m for m in _fit_collecting(search, _sine()) if ORDER in m]
        assert len(messages) == 1
        assert "MedianPruner" in messages[0]
        assert "recent_first" in messages[0]

    def test_order_sensitive_pruner_in_chronological_order_does_not_warn(self):
        search = _level_search(n_trials=1, pruner=Pruner(pruner=MedianPruner), fold_order="chronological")
        assert not [m for m in _fit_collecting(search, _sine()) if ORDER in m]

    def test_wilcoxon_under_any_order_does_not_warn(self):
        search = _level_search(n_trials=1, pruner=Pruner(), fold_order="random", random_state=0)
        assert not [m for m in _fit_collecting(search, _sine()) if ORDER in m]

    def test_pruner_with_shared_round_validation_is_refused(self):
        search = _shared_round_search(pruner=Pruner())
        with pytest.raises(ValueError, match=r"pruner cannot be combined with validation='cv'"):
            search.fit(_trend(), forecasting_horizon=3)
        assert not hasattr(search, "study_")


# ---------------------------------------------------------------------------
# Fold evaluation order
# ---------------------------------------------------------------------------


class TestFoldOrder:
    def test_default_is_recent_first(self):
        assert _level_search().get_params()["fold_order"] == "recent_first"

    def test_invalid_value_is_rejected_before_any_study(self):
        search = _level_search(fold_order="newest")
        with pytest.raises(ValueError, match="'recent_first', 'chronological', 'random'"):
            search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert not hasattr(search, "study_")

    @pytest.mark.parametrize(
        ("fold_order", "expected"), [("recent_first", [3, 2, 1, 0]), ("chronological", [0, 1, 2, 3])]
    )
    def test_deterministic_orders(self, report_log, fold_order, expected):
        search = _level_search(n_splits=4, n_trials=2, levels=(GOOD,), fold_order=fold_order)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert _orders(report_log) == {0: expected, 1: expected}

    def test_random_order(self, report_log):
        search = _level_search(n_trials=10, levels=(GOOD,), fold_order="random", random_state=0)
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        orders = _orders(report_log)
        assert len(orders) == 10
        assert all(sorted(order) == [0, 1, 2, 3, 4] for order in orders.values())
        assert len({tuple(order) for order in orders.values()}) > 1

    def test_same_results_under_every_order(self, caplog):
        caplog.set_level(logging.ERROR)
        snapshots = []
        for fold_order in ("chronological", "recent_first", "random"):
            search = _build("failing", fold_order=fold_order, random_state=0)
            search.fit(_series(), forecasting_horizon=TEST_SIZE)
            snapshots.append(_snapshot(search))
        assert snapshots[0] == snapshots[1] == snapshots[2]
        assert all(t == ([0], "ValueError") for t in snapshots[0]["trials"])

    def test_fold_identity_under_recent_first(self):
        chronological = _build("single", fold_order="chronological")
        chronological.fit(_series(), forecasting_horizon=TEST_SIZE)
        recent = _build("single", fold_order="recent_first")
        recent.fit(_series(), forecasting_horizon=TEST_SIZE)
        for i in range(N_FOLDS):
            key = f"split{i}_test_score"
            assert list(recent.cv_results_[key]) == list(chronological.cv_results_[key])

    def test_random_order_is_reproducible(self, report_log):
        first = _level_search(n_trials=6, levels=(GOOD,), fold_order="random", random_state=0)
        first.fit(_sine(), forecasting_horizon=TEST_SIZE)
        orders_first = _orders(report_log)
        report_log.clear()
        second = _level_search(n_trials=6, levels=(GOOD,), fold_order="random", random_state=0)
        second.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert _orders(report_log) == orders_first

    def test_random_order_does_not_depend_on_threads(self, report_log):
        serial = _level_search(n_trials=6, levels=(GOOD,), fold_order="random", random_state=0, n_jobs=1)
        serial.fit(_sine(), forecasting_horizon=TEST_SIZE)
        orders_serial = _orders(report_log)
        report_log.clear()
        parallel = _level_search(n_trials=6, levels=(GOOD,), fold_order="random", random_state=0, n_jobs=2)
        parallel.fit(_sine(), forecasting_horizon=TEST_SIZE)
        orders_parallel = _orders(report_log)
        assert set(orders_parallel) == set(orders_serial)
        assert all(orders_parallel[k] == orders_serial[k] for k in orders_serial)

    def test_random_state_is_ignored_by_other_orders(self, report_log):
        _level_search(n_trials=2, levels=(GOOD,), random_state=0).fit(_sine(), forecasting_horizon=TEST_SIZE)
        orders_a = _orders(report_log)
        report_log.clear()
        _level_search(n_trials=2, levels=(GOOD,), random_state=7).fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert _orders(report_log) == orders_a

    def test_random_order_with_a_failing_fold_on_database_storage(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        storage = Storage(storage=optuna.storages.RDBStorage, url=f"sqlite:///{tmp_path / 'study.db'}")
        forecaster = ThresholdFailingForecaster(estimator=LevelRegressor(), min_length=98)
        search = _level_search(
            n_trials=2,
            levels=(GOOD,),
            forecaster=forecaster,
            error_score=0.0,
            fold_order="random",
            random_state=0,
            storage=storage,
        )
        search.fit(_sine(), forecasting_horizon=TEST_SIZE)
        assert all(t.user_attrs["failed_splits"] == [0] for t in search.trials_)

    def test_any_order_under_shared_round_validation(self):
        chronological = _shared_round_search(fold_order="chronological")
        chronological.fit(_trend(), forecasting_horizon=3)
        shuffled = _shared_round_search(fold_order="random", random_state=0)
        messages = _fit_collecting_shared(shuffled)
        assert not [m for m in messages if "fold" in m.lower() and "order" in m.lower()]
        for key in chronological.cv_results_:
            if key.startswith(("split", "mean_test")):
                assert list(shuffled.cv_results_[key]) == list(chronological.cv_results_[key])


def _fit_collecting_shared(search):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(_trend(), forecasting_horizon=3)
    return [str(w.message) for w in caught if not issubclass(w.category, ExperimentalWarning)]


# ---------------------------------------------------------------------------
# Failure record when several folds fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fold_order", ["chronological", "recent_first"])
def test_lowest_failed_fold_names_the_exception_under_any_order(fold_order, caplog):
    """Folds 0 and 1 fail with different messages; the record names fold 0 whichever ran first."""
    caplog.set_level(logging.ERROR)
    # Folds train on 95, 100, 105, ... rows, so min_length=101 fails folds 0 and 1.
    forecaster = ThresholdFailingForecaster(estimator=LevelRegressor(), min_length=101)
    search = _level_search(n_trials=1, levels=(GOOD,), forecaster=forecaster, error_score=0.0, fold_order=fold_order)
    search.fit(_sine(), forecasting_horizon=TEST_SIZE)
    attrs = search.trials_[0].user_attrs
    assert attrs["failed_splits"] == [0, 1]
    assert attrs["exception_type"] == "ValueError"
    assert "length 95" in attrs["exception"]


def test_multimetric_fold_without_a_score_dict_writes_nothing(mocker):
    """A multi-metric fold whose scores are not a dict has no metric names to store under."""
    from yohou_optuna.objective import _Objective

    trial = mocker.MagicMock()
    trial.user_attrs = {}
    trial.set_user_attr = lambda key, value: trial.user_attrs.update({key: value})
    objective = _Objective(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={},
        y=None,
        X_actual=None,
        X_future=None,
        X_forecast=None,
        forecasting_horizon=3,
        splits=[],
        scorers=MeanAbsoluteError(),
        fit_params={},
        predict_func_params={},
        score_params={},
        multimetric=True,
    )
    objective._store_fold_scores(trial, 0, "test", 0.5)
    assert trial.user_attrs == {}

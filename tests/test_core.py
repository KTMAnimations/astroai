import numpy as np
import pandas as pd
import pytest

from astrogyro.baselines import fit, predict
from astrogyro.data import cluster_split, equal_cluster_weights, select_sample
from astrogyro.metrics import macro_rmse, paired_cluster_interval, rmse
from astrogyro.symbolic import evaluate, select_candidates


def test_missing_predictions_cannot_improve_error():
    assert rmse([10, 10], [0, 0]) == 10
    assert np.isinf(rmse([10, 10], [0, np.nan]))
    assert np.isinf(rmse([10, 10], [0, np.inf]))


def test_cluster_weights_and_metric_do_not_give_big_clusters_extra_votes():
    groups = np.array(["a"] * 100 + ["b"])
    weights = equal_cluster_weights(groups)
    assert weights[groups == "a"].sum() == pytest.approx(weights[groups == "b"].sum())
    assert macro_rmse(np.zeros(101), np.r_[np.ones(100), 3], groups) == 2


def test_known_power_law_recovery_on_unseen_inputs():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (100, 2))
    target = 1.2 + .45*X[:, 0] + .8*X[:, 1]
    model = fit(X, target, "power_law")
    test = rng.uniform(-1, 1, (50, 2))
    assert np.max(np.abs(predict(model, test) - (1.2+.45*test[:, 0]+.8*test[:, 1]))) < 1e-12


def test_split_has_no_cluster_or_row_overlap():
    df = pd.DataFrame({"cluster": np.repeat(list("abcdefghij"), 5),
                       "fiducial_age_Myr": np.repeat(np.arange(10)+100, 5)})
    masks = cluster_split(df, "j", 12)
    assert np.all(np.sum(masks, axis=0) == 1)
    sets = [set(df.loc[mask, "cluster"]) for mask in masks]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sets[2] == {"j"}


def test_validation_selection_uses_only_validation_scores():
    table = pd.DataFrame({"candidate_id": [0, 1, 2], "complexity": [1, 5, 2],
                          "expression": ["0", "log_age + 1", "log(log_age)"],
                          "default_selected": [True, False, False]})
    X = np.array([[-1., 0.], [-.5, .1]])
    selected = select_candidates(table, X, X[:, 0]+1)
    assert selected.loc[selected.validation_selected, "candidate_id"].item() == 1
    assert np.isinf(selected.loc[2, "validation_rmse"])


def test_constant_expression_broadcasts():
    assert np.array_equal(evaluate("1.5", np.zeros((3, 2))), [1.5]*3)


def test_paired_interval_preserves_direction():
    result = paired_cluster_interval([-.2, -.1, -.3], n_boot=1000)
    assert result["ci95_high"] < 0


def test_selection_does_not_exclude_long_periods():
    df = pd.DataFrame({"GaiaDR3_ID": ["1", "2"], "cluster": ["a", "a"],
                       "cmd_exclude": [0, 0], "dered_exclude": [0, 0], "phot_exclude": [0, 0],
                       "dr3_ruwe": [1., 1.], "Prot": [10., 159.], "BPRP_0": [1., 1.],
                       "fiducial_age_Myr": [1000., 1000.], "membership": [.9, .9]})
    selected, _ = select_sample(df, min_cluster_size=1)
    assert len(selected) == 2

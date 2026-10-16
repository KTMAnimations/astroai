"""Check scientific bookkeeping in a completed result bundle, without rerunning PySR."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astrogyro.data import CATALOG, ROOT, sha256
from astrogyro.metrics import rmse
from astrogyro.symbolic import evaluate

RESULTS = ROOT / "project/results/research_v1"


def completed_bundle():
    if not (RESULTS / "observational_summary.json").exists():
        pytest.skip("Main experiments are not yet complete")
    return RESULTS


def test_sample_matches_declared_source_and_counts():
    if not (RESULTS / "manifest.json").exists():
        pytest.skip("Study has not been prepared")
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    sample = pd.read_csv(RESULTS / "selected_stars.csv", dtype={"GaiaDR3_ID": str})
    assert sha256(CATALOG) == manifest["catalog_sha256"]
    assert len(sample) == manifest["n_stars"]
    assert sample.cluster.nunique() == manifest["n_clusters"]
    assert not sample.GaiaDR3_ID.duplicated().any()
    assert sample.groupby("cluster").size().min() >= 10
    assert sample.membership.min() >= .8


def test_complete_experiment_counts():
    output = completed_bundle()
    config = json.loads((output / "manifest.json").read_text())["config"]
    cfg = config["synthetic"]
    expected = len(cfg["truths"])*len(cfg["designs"])*len(cfg["noise_dex"])*cfg["replicates"]
    scores = pd.read_csv(output / "synthetic_scores.csv")
    assert scores.trial.nunique() == expected
    assert (scores.groupby("trial").size() == 5).all()
    assert not scores.duplicated(["trial", "model"]).any()
    obs = pd.read_csv(output / "observational_scores.csv")
    assert not obs.duplicated(["cluster", "seed", "model"]).any()
    assert (obs.groupby(["cluster", "seed"]).size() == 5).all()
    assert len(obs) == 16*len(config["observational"]["seeds"])*5


def test_saved_test_predictions_reproduce_scores_and_disjoint_splits():
    output = completed_bundle()
    sample = pd.read_csv(output / "selected_stars.csv")
    all_clusters = set(sample.cluster)
    for path in (output / "observational").glob("*/result.json"):
        result = json.loads(path.read_text())
        split = result["split"]
        groups = [set(split[name]) for name in ["discovery_clusters", "validation_clusters", "test_clusters"]]
        assert set.union(*groups) == all_clusters
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        predictions = pd.read_csv(path.with_name("predictions.csv"))
        assert set(predictions.cluster) == groups[2]
        for score in result["scores"]:
            assert rmse(predictions.log_period, predictions[score["model"]]) == pytest.approx(score["rmse_dex"], abs=1e-12)


def test_validation_candidate_is_selected_without_test_scores():
    output = completed_bundle()
    for path in (output / "observational").glob("*/candidates.csv"):
        table = pd.read_csv(path)
        assert table.default_selected.sum() == 1
        assert table.validation_selected.sum() == 1
        chosen = table.loc[table.validation_selected].iloc[0]
        assert chosen.validation_rmse == table.validation_rmse.min()
        assert not any("test" in c for c in table.columns)


def test_every_held_out_star_gets_every_method_and_seed():
    output = completed_bundle()
    predictions = pd.read_csv(output / "observational_predictions.csv", dtype={"GaiaDR3_ID": str})
    sample = pd.read_csv(output / "selected_stars.csv", dtype={"GaiaDR3_ID": str})
    assert set(predictions.GaiaDR3_ID) == set(sample.GaiaDR3_ID)
    assert (predictions.groupby("GaiaDR3_ID").size() == 3).all()
    assert not predictions.duplicated(["GaiaDR3_ID", "seed"]).any()


def test_summary_averages_seeds_before_clusters():
    output = completed_bundle()
    scores = pd.read_csv(output / "observational_scores.csv")
    expected = scores.groupby(["cluster", "model"]).rmse_dex.mean().groupby("model").mean()
    saved = json.loads((output / "observational_summary.json").read_text())
    for name, value in saved["mean_cluster_rmse_dex"].items():
        assert value == pytest.approx(expected[name])
    assert saved["paired_cluster_intervals"]["validation_minus_default"]["n_clusters"] == 16


def test_synthetic_summary_preserves_domain_failures():
    output = completed_bundle()
    scores = pd.read_csv(output / "synthetic_scores.csv")
    summary = pd.read_csv(output / "synthetic_summary.csv")
    diagnostics = json.loads((output / "selection_diagnostics.json").read_text())
    assert not scores.truth_grid_rmse.isna().any()
    assert (~np.isfinite(scores.truth_grid_rmse)).sum() == summary.n_invalid.sum()
    assert not scores.loc[~np.isfinite(scores.truth_grid_rmse), "recovered"].any()
    assert sum(diagnostics[k] for k in ["validation_lower_error", "default_lower_error",
                                       "numerically_tied", "both_invalid"]) == diagnostics["n_trials"]
    for path in (output / "synthetic").glob("*/result.json"):
        for result in json.loads(path.read_text())["models"]:
            row = scores[(scores.trial == path.parent.name) & (scores.model == result["model"])].iloc[0]
            expected = np.inf if result["truth_grid_rmse"] is None else result["truth_grid_rmse"]
            assert row.truth_grid_rmse == pytest.approx(expected)


def test_computational_sources_match_both_run_manifests():
    for name in ["research_v1", "domain_confirmation"]:
        manifest = json.loads((ROOT / "project/results" / name / "manifest.json").read_text())
        for filename, expected in manifest["code_sha256"].items():
            assert sha256(ROOT / "src/astrogyro" / filename) == expected


def test_confirmation_is_complete_and_uses_fresh_seeds():
    main = completed_bundle()
    fresh = ROOT / "project/results/domain_confirmation"
    main_seeds = {json.loads(p.read_text())["seed"] for p in (main / "synthetic").glob("*/result.json")}
    new_seeds = {json.loads(p.read_text())["seed"] for p in (fresh / "synthetic").glob("*/result.json")}
    assert len(new_seeds) == 30
    assert not main_seeds & new_seeds
    scores = pd.read_csv(fresh / "synthetic_scores.csv")
    assert len(scores) == 150
    assert (scores.groupby(["truth", "model"]).size() == 10).all()
    followup = pd.read_csv(fresh / "domain_followup.csv")
    assert len(followup) == 30
    assert set(followup.trial) == set(scores.trial)
    summary = json.loads((fresh / "domain_followup_summary.json").read_text())
    assert summary["guarded_recoveries"] == followup.recovered.sum()
    assert summary["guarded_invalid"] == (~np.isfinite(followup.guarded_truth_grid_rmse)).sum()


def test_domain_guard_ignores_target_based_diagnostics():
    import importlib.util
    spec = importlib.util.spec_from_file_location("domain_followup", ROOT / "scripts/domain_followup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A candidate wins sampled validation but is undefined at negative x.
    table = pd.DataFrame({"candidate_id": [0, 1, 2], "complexity": [2, 1, 3],
                          "expression": ["sqrt(log_age)", "1.0", "log_age + 1"],
                          "validation_rmse": [.01, .03, .02],
                          "truth_grid_rmse": [0., 0., 1e6]})
    X = np.array([[-1., .1], [1., .1]])
    assert module.choose_with_domain_guard(table, X).candidate_id == 2
    table.truth_grid_rmse = [1e9, 1e9, 0.]
    assert module.choose_with_domain_guard(table, X).candidate_id == 2

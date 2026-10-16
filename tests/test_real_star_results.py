import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from astrogyro.data import CATALOG, ROOT, features, sha256
from astrogyro.distributions import log_density

RESULTS = ROOT/"project/results/real_stars_v2"


def test_frozen_sources_and_group_splits_match():
    manifest = json.loads((RESULTS/"manifest.json").read_text())
    for name, checksum in manifest["code_sha256"].items():
        assert sha256(ROOT/name) == checksum
    for path in (RESULTS/"folds").glob("*/result.json"):
        saved = json.loads(path.read_text())
        original = ROOT/"project/results/research_v1/observational"/path.parent.name/"result.json"
        assert sha256(original) == manifest["split_sha256"][path.parent.name]
        assert saved["split"] == json.loads(original.read_text())["split"]
        assert saved["models"]["validation_selected"]["validation_nll"] == min(m["validation_nll"] for m in saved["models"].values())


def test_every_real_star_gets_all_models_and_scores_reproduce():
    pred = pd.read_csv(RESULTS/"predictions.csv", dtype={"GaiaDR3_ID": str})
    scores = pd.read_csv(RESULTS/"scores.csv")
    assert len(pred) == 721*3*7
    assert not pred.duplicated(["GaiaDR3_ID", "model", "seed"]).any()
    assert np.isfinite(pred[["nll", "crps", "mean_log_period", "q05", "q95"]]).all().all()
    assert (pred.crps >= 0).all()
    for _, row in scores.iterrows():
        group = pred[(pred.cluster == row.cluster) & (pred.seed == row.seed) & (pred.model == row.model)]
        assert len(group) == row.n_test
        assert group.nll.mean() == pytest.approx(row.nll)
        assert group.crps.mean() == pytest.approx(row.crps)
        assert np.sqrt(np.mean((group.mean_log_period-group.log_period)**2)) == pytest.approx(row.rmse_dex)


def test_power_law_point_predictions_match_previous_study():
    old = pd.read_csv(ROOT/"project/results/research_v1/observational_scores.csv")
    new = pd.read_csv(RESULTS/"scores.csv")
    merged = new[new.model == "gaussian_power"].merge(old[old.model == "power_law"], on=["cluster", "seed"])
    assert len(merged) == 48
    assert merged.rmse_dex_x.to_numpy() == pytest.approx(merged.rmse_dex_y.to_numpy(), abs=1e-10)


def test_external_stars_never_overlap_original_catalogue():
    external = pd.read_csv(RESULTS/"external_stars.csv", dtype={"GaiaDR3_ID": str})
    original = pd.read_csv(CATALOG, dtype={"GaiaDR3_ID": str})
    assert len(external) == 25
    assert not set(external.GaiaDR3_ID) & set(original.GaiaDR3_ID)
    metadata = json.loads((RESULTS/"external_summary.json").read_text())
    for fold, checksum in metadata["frozen_model_sha256"].items():
        path = RESULTS/"folds"/fold/"result.json"
        assert sha256(path) == checksum
        split = json.loads(path.read_text())["split"]
        assert "M67" not in split["discovery_clusters"] + split["validation_clusters"]


def test_saved_age_maxima_match_profiles():
    for path in (RESULTS/"folds").glob("*/result.json"):
        saved = json.loads(path.read_text())
        profiles = pd.read_csv(path.with_name("age_profiles.csv"))
        for row in saved["ages"]:
            maximum = profiles[row["model"]].idxmax()
            assert row["inferred_age_myr"] == pytest.approx(profiles.age_myr.iloc[maximum])


def test_age_command_matches_external_result_without_reference_age(tmp_path):
    external = pd.read_csv(RESULTS/"external_stars.csv", dtype={"GaiaDR3_ID": str})
    external = external[["GaiaDR3_ID", "Prot", "BPRP_0"]]
    external["fiducial_age_Myr"] = 999999  # The estimator must not consult an input reference age.
    path = tmp_path/"stars.csv"
    external.to_csv(path, index=False)
    result = subprocess.run([sys.executable, str(ROOT/"scripts/infer_cluster_age.py"), str(path),
                             "--holdout-cluster", "M67"], capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)
    expected = json.loads((RESULTS/"external_summary.json").read_text())
    assert output["inferred_cluster_age_myr"] == pytest.approx(expected["inferred_age_myr"]["mixture_quadratic"])
    assert output["n_stars"] == 25
    assert output["id_overlap_checked"]


def test_age_command_rejects_training_overlap_and_invalid_rows(tmp_path):
    sample = pd.read_csv(ROOT/"project/results/research_v1/selected_stars.csv", dtype={"GaiaDR3_ID": str}).iloc[:1]
    path = tmp_path/"stars.csv"
    sample.to_csv(path, index=False)
    command = [sys.executable, str(ROOT/"scripts/infer_cluster_age.py"), str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "overlaps fitted/validation" in result.stderr
    sample["Prot"] = -1
    sample.to_csv(path, index=False)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "no rows are silently removed" in result.stderr

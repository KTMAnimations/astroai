#!/usr/bin/env python3
"""Evaluate frozen M67-excluded models on non-overlapping published stars."""
import json

import numpy as np
import pandas as pd

from astrogyro.data import CATALOG, ROOT, features, sha256
from astrogyro.distributions import evaluate_distribution
from astrogyro.experiments import write_json
from run_real_star_study import age_profile


def main():
    root = ROOT/"project/results/real_stars_v2"
    data = ROOT/"project/data/external"
    metadata = json.loads((data/"gruner2023_manifest.json").read_text())
    assert sha256(data/"gruner2023_m67.csv") == metadata["sha256"]
    source = pd.read_csv(data/"gruner2023_m67.csv", dtype={"dr3_source_id": str})
    original = pd.read_csv(CATALOG, dtype={"GaiaDR3_ID": str})
    source["overlaps_original"] = source.dr3_source_id.isin(original.GaiaDR3_ID)
    source["in_color_domain"] = source["(BP-RP)0"].between(.65, 1.5, inclusive="neither")
    source["selected"] = source.in_color_domain & ~source.overlaps_original & np.isfinite(source.period) & (source.period > 0)
    source.to_csv(root/"external_selection_ledger.csv", index=False)
    sample = source[source.selected].rename(columns={"dr3_source_id": "GaiaDR3_ID", "period": "Prot", "(BP-RP)0": "BPRP_0"}).copy()
    sample["cluster"] = "M67_Gruner2023"
    sample["log_period"] = np.log10(sample.Prot)
    sample["log_color"] = np.log10(sample.BPRP_0)
    sample.to_csv(root/"external_stars.csv", index=False)
    predictions, scores, ages = [], [], []
    age_grid = np.geomspace(80, 5000, 401)
    models_used = {}
    for path in sorted((root/"folds").glob("holdout_M67_seed*/result.json")):
        saved = json.loads(path.read_text())
        assert "M67" not in saved["split"]["discovery_clusters"] + saved["split"]["validation_clusters"]
        models_used[path.parent.name] = sha256(path)
        for name, model in saved["models"].items():
            profile = age_profile(model, sample, age_grid)
            index = int(np.argmax(profile))
            ages.append({"model": name, "seed": saved["seed"], "inferred_age_myr": float(age_grid[index]), "boundary_hit": index in [0, 400]})
            for reference_age in [4000., 3870.]:
                sample["log_age_gyr"] = np.log10(reference_age/1000)
                prediction = evaluate_distribution(model, features(sample), sample.log_period.to_numpy())
                frame = sample[["GaiaDR3_ID", "BPRP_0", "Prot", "log_period"]].copy()
                frame["model"], frame["seed"], frame["reference_age_myr"] = name, saved["seed"], reference_age
                for key, value in prediction.items(): frame[key] = value
                predictions.append(frame)
                z = sample.log_period.to_numpy()
                scores.append({"model": name, "seed": saved["seed"], "reference_age_myr": reference_age,
                               "nll": float(frame.nll.mean()), "crps": float(frame.crps.mean()),
                               "rmse_dex": float(np.sqrt(np.mean((frame.mean_log_period-z)**2))),
                               "mae_dex": float(np.mean(np.abs(frame.median_log_period-z))),
                               "coverage68": float(np.mean((z >= frame.q16) & (z <= frame.q84))),
                               "coverage90": float(np.mean((z >= frame.q05) & (z <= frame.q95)))})
    pd.concat(predictions, ignore_index=True).to_csv(root/"external_predictions.csv", index=False)
    scores = pd.DataFrame(scores);scores.to_csv(root/"external_scores.csv", index=False)
    ages = pd.DataFrame(ages);ages.to_csv(root/"external_age_estimates.csv", index=False)
    ages["log_age"] = np.log10(ages.inferred_age_myr)
    inferred = 10**ages.groupby("model").log_age.mean()
    aggregate = scores[scores.reference_age_myr == 4000].groupby("model").mean(numeric_only=True).drop(columns=["seed", "reference_age_myr"])
    write_json(root/"external_summary.json", {"n_stars": len(sample), "reference_age_myr": 4000,
               "data_sha256": metadata["sha256"], "frozen_model_sha256": models_used,
               "scores": aggregate.to_dict("index"), "inferred_age_myr": inferred.to_dict(),
               "boundary_hits": ages.groupby("model").boundary_hit.sum().to_dict()})
    print(aggregate.round(4).to_string())
    print("External cluster inferred ages (Myr):", inferred.round(1).to_dict())


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Replay all 528 observational candidate fits without changing frozen results."""
import json
import math
from pathlib import Path

import pandas as pd

from astrogyro.data import ROOT, sha256
from astrogyro.experiments import write_json
from run_real_star_study import fit_candidates


def compare(saved, replayed, differences):
    if isinstance(saved, dict):
        assert saved.keys() == replayed.keys()
        for key in saved:
            compare(saved[key], replayed[key], differences)
    elif isinstance(saved, list):
        assert len(saved) == len(replayed)
        for a, b in zip(saved, replayed):
            compare(a, b, differences)
    elif isinstance(saved, (float, int)) and not isinstance(saved, bool):
        assert math.isclose(saved, replayed, abs_tol=1e-10, rel_tol=1e-10), (saved, replayed)
        differences.append(abs(saved-replayed))
    else:
        assert saved == replayed


def main():
    root = ROOT/"project/results/real_stars_v2"
    manifest = json.loads((root/"manifest.json").read_text())
    for name, checksum in manifest["code_sha256"].items():
        assert sha256(ROOT/name) == checksum, name
    sample_path = ROOT/"project/results/research_v1/selected_stars.csv"
    assert sha256(sample_path) == manifest["sample_sha256"]
    sample = pd.read_csv(sample_path, dtype={"GaiaDR3_ID": str})
    differences, count = [], 0
    paths = sorted((root/"folds").glob("*/result.json"))
    assert len(paths) == 48
    for i, path in enumerate(paths):
        saved = json.loads(path.read_text())
        train, val = [sample[sample.cluster.isin(saved["split"][key])].copy()
                      for key in ["discovery_clusters", "validation_clusters"]]
        models, candidates = fit_candidates(manifest["config"], train, val)
        compare(saved["models"], models, differences)
        compare(saved["candidates"], candidates, differences)
        count += len(candidates)
        if (i+1) % 12 == 0:
            print(f"Reproduced {i+1}/48 folds", flush=True)
    report = {"folds_replayed": len(paths), "candidate_fits_replayed": count,
              "numeric_values_compared": len(differences), "maximum_absolute_difference": max(differences),
              "absolute_tolerance": 1e-10, "relative_tolerance": 1e-10,
              "manifest_sha256": sha256(root/"manifest.json"),
              "verification_script_sha256": sha256(Path(__file__)),
              "all_models_and_candidates_match": True}
    write_json(root/"reproducibility.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

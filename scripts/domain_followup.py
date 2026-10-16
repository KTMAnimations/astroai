#!/usr/bin/env python3
"""Post-hoc input-domain guard; never use true targets to select a candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrogyro.data import ROOT
from astrogyro.experiments import domain, truth_grid, truth_log_period, write_json
from astrogyro.metrics import rmse
from astrogyro.symbolic import evaluate


def choose_with_domain_guard(table, feature_mesh):
    # Deliberately exclude diagnostic columns containing true-grid errors.
    candidates = table[["candidate_id", "expression", "complexity", "validation_rmse"]].copy()
    candidates["finite_domain"] = [np.isfinite(evaluate(e, feature_mesh)).all() for e in candidates.expression]
    eligible = candidates[candidates.finite_domain & np.isfinite(candidates.validation_rmse)]
    if eligible.empty:
        raise RuntimeError("No candidate is valid on the specified input mesh")
    return eligible.sort_values(["validation_rmse", "complexity", "candidate_id"]).iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "project/results/research_v1")
    parser.add_argument("--fresh-trials", action="store_true")
    args = parser.parse_args()
    root = args.results
    config = json.loads((root / "manifest.json").read_text())["config"]
    lo, hi = domain(config)
    x, y = np.meshgrid(np.linspace(lo[0], hi[0], 81), np.linspace(lo[1], hi[1], 51))
    feature_mesh = np.column_stack([x.ravel(), y.ravel()])
    evaluation_mesh = truth_grid(config)
    scores = pd.read_csv(root / "synthetic_scores.csv")
    rows = []
    for path in sorted((root / "synthetic").glob("*/candidates.csv")):
        table = pd.read_csv(path)
        original = scores[(scores.trial == path.parent.name) & (scores.model == "pysr_validation")].iloc[0]
        chosen = choose_with_domain_guard(table, feature_mesh)
        # Only after freezing the selection, read true targets for evaluation.
        error = rmse(truth_log_period(evaluation_mesh, original.truth), evaluate(chosen.expression, evaluation_mesh))
        rows.append({"trial": path.parent.name, "truth": original.truth, "design": original.design,
                     "noise_dex": original.noise_dex, "candidate_id": int(chosen.candidate_id),
                     "expression": chosen.expression, "validation_rmse": float(chosen.validation_rmse),
                     "changed_selection": chosen.expression != original.expression,
                     "original_truth_grid_rmse": original.truth_grid_rmse,
                     "guarded_truth_grid_rmse": error,
                     "recovered": error < config["synthetic"]["recovery_threshold_dex"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "domain_followup.csv", index=False)
    output = {"status": "fresh_seed_confirmation" if args.fresh_trials else "post_hoc_exploratory_same_trials", "n_trials": len(frame),
              "feature_only_mesh_shape": [81, 51], "changed_selections": int(frame.changed_selection.sum()),
              "original_invalid": int((~np.isfinite(frame.original_truth_grid_rmse)).sum()),
              "guarded_invalid": int((~np.isfinite(frame.guarded_truth_grid_rmse)).sum()),
              "original_recoveries": int((frame.original_truth_grid_rmse < config["synthetic"]["recovery_threshold_dex"]).sum()),
              "guarded_recoveries": int(frame.recovered.sum())}
    write_json(root / "domain_followup_summary.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

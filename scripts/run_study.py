#!/usr/bin/env python3
"""Run a resumable study; each cached trial is tied to code/data/config hashes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrogyro.data import ROOT
from astrogyro.experiments import initialize, run_observational, run_sensitivity, run_synthetic, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["all", "synthetic", "observational", "sensitivity", "prepare", "summarize"], default="all")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be between zero and shard-count minus one")
    config = json.loads(args.config.read_text())
    if args.pilot:
        config["study_id"] = "pilot"
        config["synthetic"].update(truths=["interaction"], designs=["balanced"], noise_dex=[0.05], replicates=1)
        config["search"].update(iterations=20, populations=6, cycles=100)
    output = args.output or ROOT / "project/results" / config["study_id"]
    sample = initialize(config, output)
    print(f"Sample: {len(sample)} stars / {sample.cluster.nunique()} clusters; output {output}", flush=True)
    if args.stage in ["all", "synthetic"]:
        run_synthetic(config, output, sample, args.shard_index, args.shard_count)
    if args.stage in ["all", "observational"]:
        run_observational(config, output, sample, args.shard_index, args.shard_count)
    if args.stage in ["all", "sensitivity"] and args.shard_index == 0:
        run_sensitivity(config, output, sample)
    if args.stage == "summarize":
        summarize(config, output, sample)
    # Cached strict-JSON failures are null. Preserve the infinite-loss penalty
    # in the public score table even when this driver is used without the
    # richer reporting script. This does not change any search or selection.
    scores_path = output / "synthetic_scores.csv"
    if args.shard_count == 1 and args.stage in ["all", "synthetic", "summarize"] and scores_path.exists():
        scores = pd.read_csv(scores_path)
        scores["truth_grid_rmse"] = scores.truth_grid_rmse.fillna(np.inf)
        scores["invalid_grid_prediction"] = ~np.isfinite(scores.truth_grid_rmse)
        scores.to_csv(scores_path, index=False)


if __name__ == "__main__":
    main()

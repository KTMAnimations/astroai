#!/usr/bin/env python3
"""Repeat one saved search exactly and compare its candidate equations/losses."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrogyro.data import ROOT, features
from astrogyro.experiments import domain, truth_log_period, write_json
from astrogyro.symbolic import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "project/results/research_v1")
    args = parser.parse_args()
    output = args.results
    config = json.loads((output / "manifest.json").read_text())["config"]
    trial = "interaction_balanced_noise0.05_seed00"
    saved = json.loads((output / "synthetic" / trial / "result.json").read_text())
    seed = saved["seed"]
    rng = np.random.default_rng(seed)
    lo, hi = domain(config)
    n_train, n_val = config["synthetic"]["n_train"], config["synthetic"]["n_validation"]
    X = rng.uniform(lo, hi, size=(n_train+n_val, 2))
    target = truth_log_period(X, "interaction") + rng.normal(0, .05, len(X))
    replay = search(X[:n_train], target[:n_train], None, seed, config["search"], ROOT / "tmp/reproducibility")
    original = pd.read_csv(output / "synthetic" / trial / "candidates.csv")
    same_equations = replay.expression.tolist() == original.expression.tolist()
    same_losses = len(replay) == len(original) and np.allclose(replay.loss, original.loss, rtol=1e-12, atol=1e-14)
    result = {"trial": trial, "seed": seed, "same_equations": same_equations, "same_losses": bool(same_losses),
              "n_candidates_original": len(original), "n_candidates_replay": len(replay)}
    write_json(output / "reproducibility_check.json", result)
    print(json.dumps(result, indent=2))
    if not same_equations or not same_losses:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare proper predictive distributions on the frozen research_v1 splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from astrogyro.data import ROOT, equal_cluster_weights, features, sha256
from astrogyro.distributions import KINDS, evaluate_distribution, fit_distribution, log_density
from astrogyro.experiments import write_json
from astrogyro.metrics import paired_cluster_interval


def group_nll(model, X, z, groups):
    losses = -log_density(model, X, z)
    if not np.isfinite(losses).all():
        return np.inf
    return float(np.mean([losses[groups == group].mean() for group in np.unique(groups)]))


def fit_candidates(config, train, val):
    X, z = features(train), train.log_period.to_numpy()
    weights = equal_cluster_weights(train.cluster)
    models, candidates = {}, []
    for kind in config["models"]:
        penalties = (config["mixture_quadratic_penalties"] if kind == "mixture_quadratic" else
                     config["rbf_penalties"] if kind == "gaussian_rbf" else [0.])
        fitted = []
        for penalty in penalties:
            model = fit_distribution(X, z, weights, kind, penalty)
            if not model["converged"]:
                raise RuntimeError(f"Nonconverged model: {kind}")
            loss = group_nll(model, features(val), val.log_period.to_numpy(), val.cluster.to_numpy())
            fitted.append((loss, penalty, model))
            candidates.append({"kind": kind, "penalty": penalty, "validation_nll": loss, "model": model})
        loss, penalty, model = min(fitted, key=lambda item: (item[0], item[1]))
        models[kind] = {**model, "validation_nll": loss}
    chosen = min(config["models"], key=lambda kind: models[kind]["validation_nll"])
    models["validation_selected"] = {**models[chosen], "selected_family": chosen}
    return models, candidates


def age_profile(model, test, age_grid):
    n = len(test)
    X = np.column_stack([np.repeat(np.log10(age_grid/1000), n), np.tile(test.log_color, len(age_grid))])
    z = np.tile(test.log_period, len(age_grid))
    mean_log_likelihood = log_density(model, X, z).reshape(len(age_grid), n).mean(axis=1)
    return mean_log_likelihood


def summarize(root, config):
    paths = sorted(root.glob("folds/*/result.json"))
    if len(paths) != 48:
        raise RuntimeError(f"Expected 48 completed folds, found {len(paths)}")
    scores = pd.DataFrame([row for p in paths for row in json.loads(p.read_text())["scores"]])
    scores.to_csv(root/"scores.csv", index=False)
    predictions = pd.concat([pd.read_csv(p.with_name("predictions.csv"), dtype={"GaiaDR3_ID": str}) for p in paths], ignore_index=True)
    predictions.to_csv(root/"predictions.csv", index=False)
    ages = pd.DataFrame([row for p in paths for row in json.loads(p.read_text())["ages"]])
    ages.to_csv(root/"age_estimates.csv", index=False)
    metrics = ["nll", "crps", "rmse_dex", "mae_dex", "coverage68", "coverage90", "width68_dex", "width90_dex"]
    cluster = scores.groupby(["cluster", "model"])[metrics].mean()
    cluster.to_csv(root/"cluster_scores.csv")
    summary = {"n_stars": predictions.GaiaDR3_ID.nunique(), "n_clusters": 16, "n_folds": 48,
               "mean_cluster_scores": cluster.groupby("model").mean().to_dict("index"), "paired_intervals": {}}
    for metric in ["nll", "crps", "rmse_dex", "mae_dex"]:
        frame = cluster[metric].unstack()
        for kind in [*config["models"][1:], "validation_selected"]:
            delta = frame[kind] - frame.gaussian_power
            summary["paired_intervals"][f"{kind}_minus_gaussian_power_{metric}"] = paired_cluster_interval(delta)
    ages["log_inferred_myr"] = np.log10(ages.inferred_age_myr)
    by_cluster = ages.groupby(["cluster", "model"]).agg(log_inferred_myr=("log_inferred_myr", "mean"),
                  reference_age_myr=("reference_age_myr", "first"), boundary_hits=("boundary_hit", "sum"))
    by_cluster["inferred_age_myr"] = 10**by_cluster.log_inferred_myr
    by_cluster["absolute_log_age_error"] = np.abs(by_cluster.log_inferred_myr-np.log10(by_cluster.reference_age_myr))
    by_cluster["absolute_fractional_error"] = np.abs(by_cluster.inferred_age_myr/by_cluster.reference_age_myr-1)
    by_cluster.to_csv(root/"cluster_age_estimates.csv")
    summary["age_scores"] = by_cluster.groupby("model").agg(mean_absolute_log_age_error=("absolute_log_age_error", "mean"),
                         median_absolute_fractional_error=("absolute_fractional_error", "median"),
                         boundary_hits=("boundary_hits", "sum")).to_dict("index")
    for kind in [*config["models"][1:], "validation_selected"]:
        frame = by_cluster.absolute_log_age_error.unstack()
        summary["paired_intervals"][f"{kind}_minus_gaussian_power_age_error"] = paired_cluster_interval(frame[kind]-frame.gaussian_power)
    write_json(root/"summary.json", summary)
    print(pd.DataFrame(summary["mean_cluster_scores"]).T.round(4).to_string())
    print(pd.DataFrame(summary["age_scores"]).T.round(4).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["run", "summarize"], default="run")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    config_path = ROOT/"configs/real_stars.json"
    config = json.loads(config_path.read_text())
    root = ROOT/"project/results"/config["study_id"]
    root.mkdir(parents=True, exist_ok=True)
    reference = ROOT/"project/results"/config["reference_study"]
    sample_path = reference/"selected_stars.csv"
    paths = sorted((reference/"observational").glob("*/result.json"))
    signature = {"config": config, "sample_sha256": sha256(sample_path),
                 "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                  [Path(__file__), ROOT/"src/astrogyro/distributions.py", ROOT/"src/astrogyro/data.py", ROOT/"src/astrogyro/metrics.py"]},
                 "split_sha256": {p.parent.name: sha256(p) for p in paths}}
    if (root/"manifest.json").exists():
        if json.loads((root/"manifest.json").read_text()) != signature:
            raise RuntimeError("Study definition changed; do not mix cached runs")
    else:
        write_json(root/"manifest.json", signature)
    if args.stage == "summarize":
        summarize(root, config); return
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard")
    sample = pd.read_csv(sample_path, dtype={"GaiaDR3_ID": str})
    age_grid = np.geomspace(*config["age_grid_myr"][:2], int(config["age_grid_myr"][2]))
    for i, path in enumerate(paths):
        if i % args.shard_count != args.shard_index:
            continue
        destination = root/"folds"/path.parent.name
        if (destination/"result.json").exists():
            continue
        started = time.monotonic()
        original = json.loads(path.read_text())
        split = original["split"]
        train, val, test = [sample[sample.cluster.isin(split[k])].copy() for k in ["discovery_clusters", "validation_clusters", "test_clusters"]]
        models, candidates = fit_candidates(config, train, val)
        destination.mkdir(parents=True, exist_ok=True)
        predictions, scores, ages, profiles = [], [], [], {"age_myr": age_grid}
        cluster = test.cluster.iloc[0]
        seed = int(path.parent.name.rsplit("seed", 1)[1])
        for name, model in models.items():
            pred = evaluate_distribution(model, features(test), test.log_period.to_numpy())
            frame = test[["GaiaDR3_ID", "cluster", "fiducial_age_Myr", "BPRP_0", "Prot", "log_period"]].copy()
            frame["model"], frame["seed"] = name, seed
            for key, values in pred.items(): frame[key] = values
            predictions.append(frame)
            z = test.log_period.to_numpy()
            scores.append({"cluster": cluster, "seed": seed, "model": name, "n_test": len(test),
                           "nll": float(np.mean(pred["nll"])), "crps": float(np.mean(pred["crps"])),
                           "rmse_dex": float(np.sqrt(np.mean((pred["mean_log_period"]-z)**2))),
                           "mae_dex": float(np.mean(np.abs(pred["median_log_period"]-z))),
                           "coverage68": float(np.mean((z >= pred["q16"]) & (z <= pred["q84"]))),
                           "coverage90": float(np.mean((z >= pred["q05"]) & (z <= pred["q95"]))),
                           "width68_dex": float(np.mean(pred["q84"]-pred["q16"])),
                           "width90_dex": float(np.mean(pred["q95"]-pred["q05"]))})
            profile = age_profile(model, test, age_grid)
            index = int(np.argmax(profile))
            profiles[name] = profile
            ages.append({"cluster": cluster, "seed": seed, "model": name,
                         "reference_age_myr": float(test.fiducial_age_Myr.iloc[0]),
                         "inferred_age_myr": float(age_grid[index]), "boundary_hit": index in [0, len(age_grid)-1]})
        pd.concat(predictions, ignore_index=True).to_csv(destination/"predictions.csv", index=False)
        pd.DataFrame(profiles).to_csv(destination/"age_profiles.csv", index=False)
        write_json(destination/"result.json", {"split": split, "seed": seed, "models": models,
                   "candidates": candidates, "scores": scores, "ages": ages, "elapsed_seconds": time.monotonic()-started})
        print(f"{i+1}/48 {path.parent.name}: completed ({time.monotonic()-started:.1f}s)", flush=True)
    if args.shard_count == 1:
        summarize(root, config)


if __name__ == "__main__":
    main()

"""Controlled recovery trials and group-separated observational evaluation."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines
from .data import CATALOG, ROOT, cluster_split, equal_cluster_weights, features, load_catalog, select_sample, sha256
from .metrics import macro_rmse, paired_cluster_interval, rmse
from .symbolic import evaluate, search, select_candidates


def write_json(path, obj):
    """Strict JSON: nonfinite values are explicit nulls, not invalid JSON tokens."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.bool_):
            return bool(value)
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample, ledger = select_sample(load_catalog(), **config["sample"])
    computational_files = [Path(__file__).with_name(n + ".py") for n in
                           ["data", "baselines", "metrics", "symbolic", "experiments"]]
    signature_data = {"config": config, "catalog_sha256": sha256(CATALOG),
                      "code_sha256": {p.name: sha256(p) for p in computational_files}}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    manifest_file = output / "manifest.json"
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if old["signature"] != signature:
            raise RuntimeError("Configuration/data/computational code changed. Use a new output directory; do not mix runs.")
    else:
        write_json(manifest_file, {**signature_data, "signature": signature,
                   "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "python": sys.version, "platform": platform.platform(),
                   "versions": {n: version(n) for n in ["pysr", "numpy", "pandas", "scipy", "sympy", "matplotlib"]},
                   "n_stars": len(sample), "n_clusters": sample.cluster.nunique()})
    sample.to_csv(output / "selected_stars.csv", index=False)
    ledger.to_csv(output / "sample_ledger.csv", index=False)
    sample.groupby("cluster").agg(n_stars=("Prot", "size"), age_Myr=("fiducial_age_Myr", "first"),
                                  median_membership=("membership", "median")).to_csv(output / "clusters.csv")
    return sample


def truth_log_period(X, truth):
    x, y = np.asarray(X).T
    basic = np.log10(0.344*np.sqrt(1000)) + .5*x + y
    if truth == "skumanich":
        return basic
    if truth == "interaction":
        return basic + 1.5*(x+.5)*y + .6*y*y
    if truth == "saturation":
        return basic - .5*np.log10(1+10**x)
    raise ValueError(truth)


def domain(config):
    s = config["sample"]
    return np.array([np.log10(s["min_age"]/1000), np.log10(s["color_lo"])]), np.array([np.log10(3.87), np.log10(s["color_hi"])])


def truth_grid(config):
    lo, hi = domain(config)
    x, y = np.meshgrid(np.linspace(lo[0], hi[0], 80), np.linspace(lo[1], hi[1], 50))
    return np.column_stack([x.ravel(), y.ravel()])


def run_synthetic(config, output, sample, shard_index=0, shard_count=1):
    cfg = config["synthetic"]
    combinations = list(itertools.product(cfg["truths"], cfg["designs"], cfg["noise_dex"], range(cfg["replicates"])))
    grid = truth_grid(config)
    lo, hi = domain(config)
    X_catalog = features(sample)
    output = Path(output)
    rows = []
    for counter, (truth, design, sigma, replicate) in enumerate(combinations):
        if counter % shard_count != shard_index:
            continue
        trial = f"{truth}_{design}_noise{sigma:.2f}_seed{replicate:02d}"
        trial_dir = output / "synthetic" / trial
        result_file = trial_dir / "result.json"
        if result_file.exists():
            rows.extend(json.loads(result_file.read_text())["models"])
            continue
        t0 = time.monotonic()
        seed = config["seed"] + 10000*cfg["truths"].index(truth) + 1000*cfg["noise_dex"].index(sigma) + replicate
        rng = np.random.default_rng(seed)
        n_train, n_val = cfg["n_train"], cfg["n_validation"]
        if design == "catalog":
            X = X_catalog[rng.integers(len(X_catalog), size=n_train+n_val)]
        else:
            X = rng.uniform(lo, hi, size=(n_train+n_val, 2))
        target = truth_log_period(X, truth) + rng.normal(0, sigma, len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = target[:n_train], target[n_train:]
        y_grid = truth_log_period(grid, truth)
        table = search(X_train, y_train, None, seed, config["search"], output / "raw_search" / trial)
        table = select_candidates(table, X_val, y_val)
        trial_dir.mkdir(parents=True, exist_ok=True)
        # Evaluation happens after both choices are frozen.
        table["truth_grid_rmse"] = [rmse(y_grid, evaluate(e, grid)) for e in table.expression]
        table.to_csv(trial_dir / "candidates.csv", index=False)
        model_rows = []
        for selector in ["default", "validation"]:
            selected = table.loc[table[f"{selector}_selected"]].iloc[0]
            model_rows.append({"model": f"pysr_{selector}", "expression": selected.expression,
                               "complexity": int(selected.complexity), "truth_grid_rmse": float(selected.truth_grid_rmse),
                               "validation_rmse": float(selected.validation_rmse)})
        for name in baselines.NAMES:
            model = baselines.fit(X_train, y_train, name)
            model_rows.append({"model": name, "coefficients": model["coefficients"],
                               "truth_grid_rmse": rmse(y_grid, baselines.predict(model, grid)),
                               "validation_rmse": rmse(y_val, baselines.predict(model, X_val))})
        for row in model_rows:
            row.update({"trial": trial, "truth": truth, "design": design, "noise_dex": sigma,
                        "replicate": replicate, "recovered": row["truth_grid_rmse"] < cfg["recovery_threshold_dex"]})
        write_json(result_file, {"seed": seed, "elapsed_seconds": time.monotonic()-t0, "models": model_rows})
        rows.extend(model_rows)
        if shard_count == 1:
            pd.DataFrame(rows).to_csv(output / "synthetic_scores.csv", index=False)
        print(f"Synthetic {counter+1}/{len(combinations)} {trial}: " +
              ", ".join(f"{r['model']}={r['truth_grid_rmse']:.4f}" for r in model_rows[:2]) +
              f" ({time.monotonic()-t0:.1f}s)", flush=True)
    if shard_count == 1:
        pd.DataFrame(rows).to_csv(output / "synthetic_scores.csv", index=False)


def run_observational(config, output, sample, shard_index=0, shard_count=1):
    output = Path(output)
    X, target, groups = features(sample), sample.log_period.to_numpy(), sample.cluster.to_numpy()
    clusters = sample.groupby("cluster").fiducial_age_Myr.first().sort_values().index
    all_scores, all_predictions = [], []
    for cluster_index, held_out in enumerate(clusters):
        for seed_index in config["observational"]["seeds"]:
            task_index = cluster_index * len(config["observational"]["seeds"]) + config["observational"]["seeds"].index(seed_index)
            if task_index % shard_count != shard_index:
                continue
            trial = f"holdout_{held_out}_seed{seed_index}"
            trial_dir = output / "observational" / trial
            result_file = trial_dir / "result.json"
            if result_file.exists():
                all_scores.extend(json.loads(result_file.read_text())["scores"])
                all_predictions.append(pd.read_csv(trial_dir / "predictions.csv", dtype={"GaiaDR3_ID": str}))
                continue
            t0 = time.monotonic()
            seed = config["seed"] + 100*cluster_index + seed_index
            train, val, test = cluster_split(sample, held_out, seed)
            weights = equal_cluster_weights(groups[train])
            table = search(X[train], target[train], weights, seed, config["search"], output / "raw_search" / trial)
            table = select_candidates(table, X[val], target[val], groups[val])
            trial_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(trial_dir / "candidates.csv", index=False)
            models, predictions = {}, {}
            for selector in ["default", "validation"]:
                chosen = table.loc[table[f"{selector}_selected"]].iloc[0]
                name = f"pysr_{selector}"
                models[name] = {"expression": chosen.expression, "complexity": int(chosen.complexity),
                                "validation_rmse": float(chosen.validation_rmse)}
                predictions[name] = evaluate(chosen.expression, X[test])
            for name in baselines.NAMES:
                models[name] = baselines.fit(X[train], target[train], name, weights)
                predictions[name] = baselines.predict(models[name], X[test])
            scores = []
            pred_frame = sample.loc[test, ["GaiaDR3_ID", "cluster", "fiducial_age_Myr", "BPRP_0", "Prot", "log_period"]].copy()
            pred_frame["seed"] = seed_index
            for name, pred in predictions.items():
                scores.append({"cluster": held_out, "seed": seed_index, "model": name,
                               "n_test": int(test.sum()), "n_discovery": int(train.sum()), "n_validation": int(val.sum()),
                               "age_Myr": float(sample.loc[test, "fiducial_age_Myr"].iloc[0]),
                               "age_extrapolation": bool(X[test, 0].min() < X[train, 0].min() or X[test, 0].max() > X[train, 0].max()),
                               "rmse_dex": rmse(target[test], pred),
                               "mae_dex": float(np.mean(np.abs(target[test]-pred))) if np.isfinite(pred).all() else float("inf")})
                pred_frame[name] = pred
            split = {"discovery_clusters": sorted(set(groups[train])), "validation_clusters": sorted(set(groups[val])),
                     "test_clusters": [held_out]}
            pred_frame.to_csv(trial_dir / "predictions.csv", index=False)
            write_json(result_file, {"seed": seed, "split": split, "models": models, "scores": scores,
                                    "elapsed_seconds": time.monotonic()-t0})
            all_scores.extend(scores)
            all_predictions.append(pred_frame)
            if shard_count == 1:
                pd.DataFrame(all_scores).to_csv(output / "observational_scores.csv", index=False)
            print(f"Observed {cluster_index+1}/{len(clusters)} {trial}: " +
                  ", ".join(f"{s['model']}={s['rmse_dex']:.3f}" for s in scores) +
                  f" ({time.monotonic()-t0:.1f}s)", flush=True)
    if shard_count != 1:
        return
    summarize_observational(config, output, all_scores, all_predictions)


def summarize_observational(config, output, all_scores, all_predictions):
    scores_df = pd.DataFrame(all_scores)
    scores_df[["rmse_dex", "mae_dex"]] = scores_df[["rmse_dex", "mae_dex"]].fillna(float("inf"))
    scores_df.to_csv(output / "observational_scores.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "observational_predictions.csv", index=False)
    per_cluster = scores_df.groupby(["cluster", "model"]).rmse_dex.mean().unstack()
    intervals = {}
    for name in ["pysr_default", "pysr_validation", "fixed_skumanich", "quadratic"]:
        delta = per_cluster[name]-per_cluster.power_law
        intervals[f"{name}_minus_power_law"] = (paired_cluster_interval(delta)
            if np.isfinite(delta).all() else {"status": "not_computed_due_to_invalid_predictions"})
    delta = per_cluster.pysr_validation-per_cluster.pysr_default
    intervals["validation_minus_default"] = (paired_cluster_interval(delta)
        if np.isfinite(delta).all() else {"status": "not_computed_due_to_invalid_predictions"})
    write_json(output / "observational_summary.json", {"mean_cluster_rmse_dex": per_cluster.mean().to_dict(),
               "paired_cluster_intervals": intervals, "n_clusters": len(per_cluster),
               "seeds_per_cluster": len(config["observational"]["seeds"])})


def summarize(config, output, sample):
    output = Path(output)
    cfg = config["synthetic"]
    expected_syn = len(cfg["truths"])*len(cfg["designs"])*len(cfg["noise_dex"])*cfg["replicates"]
    syn = list((output / "synthetic").glob("*/result.json"))
    expected_obs = sample.cluster.nunique()*len(config["observational"]["seeds"])
    obs = list((output / "observational").glob("*/result.json"))
    if len(syn) != expected_syn or len(obs) != expected_obs:
        raise RuntimeError(f"Incomplete runs: synthetic {len(syn)}/{expected_syn}; observational {len(obs)}/{expected_obs}")
    pd.DataFrame([r for path in sorted(syn) for r in json.loads(path.read_text())["models"]]).to_csv(output / "synthetic_scores.csv", index=False)
    scores = [r for path in sorted(obs) for r in json.loads(path.read_text())["scores"]]
    predictions = [pd.read_csv(path.with_name("predictions.csv"), dtype={"GaiaDR3_ID": str}) for path in sorted(obs)]
    summarize_observational(config, output, scores, predictions)
    print(f"Summarized {len(syn)} synthetic and {len(obs)} observational searches", flush=True)


def run_sensitivity(config, output, sample):
    raw = load_catalog()
    rows = []
    for membership, color_hi in itertools.product([.5, .8, .9], [1.5, 2.]):
        choices = {**config["sample"], "membership": membership, "color_hi": color_hi}
        candidate, _ = select_sample(raw, **choices)
        for held_out in sample.cluster.unique():
            train = candidate[candidate.cluster != held_out]
            test = sample[sample.cluster == held_out]
            for name in baselines.NAMES:
                model = baselines.fit(features(train), train.log_period.to_numpy(), name, equal_cluster_weights(train.cluster))
                rows.append({"membership_cut": membership, "color_hi": color_hi, "cluster": held_out,
                             "model": name, "n_train": len(train), "n_test": len(test),
                             "rmse_dex": rmse(test.log_period, baselines.predict(model, features(test)))})
    pd.DataFrame(rows).to_csv(Path(output) / "sensitivity_scores.csv", index=False)

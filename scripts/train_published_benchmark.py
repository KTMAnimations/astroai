#!/usr/bin/env python3
"""Retrain ChronoFlow's architecture on the frozen comparison splits."""
import argparse
import copy
import json
import math
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from astrogyro.data import ROOT, equal_cluster_weights, sha256
from astrogyro.experiments import write_json
from astrogyro.published import contexts, new_flow, predict_flow, tensor_log_density


def signature(config):
    reference = ROOT/"project/results/research_v1"
    upstream = ROOT/"chronoflow"
    files = [ROOT/"src/astrogyro/published.py", Path(__file__), ROOT/"configs/published_benchmark.json",
             upstream/"chronoflow/ChronoFlow.py", upstream/"chronoflow/model_artifacts/weights.pth",
             ROOT/"src/astrogyro/data.py"]
    return {"config": config, "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
            "sample_sha256": sha256(reference/"selected_stars.csv"),
            "split_sha256": {p.parent.name: sha256(p) for p in sorted((reference/"observational").glob("*/result.json"))},
            "upstream_commit": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
            "versions": {n: version(n) for n in ["numpy", "pandas", "scipy", "torch", "zuko"]},
            "python": platform.python_version(), "platform": platform.platform()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    config = json.loads((ROOT/"configs/published_benchmark.json").read_text())
    root = ROOT/"project/results"/config["study_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = signature(config)
    assert manifest["upstream_commit"] == config["upstream_commit"]
    if (root/"manifest.json").exists():
        if json.loads((root/"manifest.json").read_text()) != manifest:
            raise RuntimeError("Changed study definition: do not mix cached fits")
    else:
        write_json(root/"manifest.json", manifest)
    if args.initialize_only:
        print("Initialized immutable benchmark manifest");return
    assert 0 <= args.shard_index < args.shard_count
    sample = pd.read_csv(ROOT/"project/results/research_v1/selected_stars.csv", dtype={"GaiaDR3_ID": str})
    paths = sorted((ROOT/"project/results/research_v1/observational").glob("*/result.json"))
    jobs = [(path, weighting) for path in paths for weighting in config["training_weightings"]]
    for i, (path, weighting) in enumerate(jobs):
        if i % args.shard_count != args.shard_index:
            continue
        dest = root/"fits"/path.parent.name/weighting
        if (dest/"result.json").exists():
            continue
        started = time.monotonic();dest.mkdir(parents=True, exist_ok=True)
        split = json.loads(path.read_text())["split"]
        train, val = [sample[sample.cluster.isin(split[key])].copy() for key in ["discovery_clusters", "validation_clusters"]]
        # Pair the initial network across the two weighting variants of a split.
        seed = config["seed"]+i//2
        model = new_flow(seed)
        X = torch.tensor(contexts(train), dtype=torch.float32)
        z = torch.tensor(train.log_period.to_numpy(), dtype=torch.float32)
        mem = torch.tensor(train.membership.to_numpy(), dtype=torch.float32)
        w = equal_cluster_weights(train.cluster) if weighting == "equal_cluster" else np.ones(len(train))
        w = torch.tensor(w/w.sum(), dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["initial_learning_rate"])
        val_X, val_z, val_mem = contexts(val), val.log_period.to_numpy(), val.membership.to_numpy()
        best, best_epoch, best_state = math.inf, None, None
        history = []
        for epoch in range(1, config["epochs"]+1):
            fraction = (epoch-1)/(config["epochs"]-1)
            exp_lr = config["initial_learning_rate"]*(config["final_exponential_rate"]/config["initial_learning_rate"])**fraction
            lr = exp_lr*.5*(1+math.cos(2*math.pi*(epoch-1)/config["cosine_period_epochs"]))
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad()
            loss = -(w*tensor_log_density(model, X, z, mem, config["outlier_probability"])).sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {path.parent.name}, {weighting}, epoch {epoch}")
            loss.backward();optimizer.step()
            if epoch % config["validation_every"] == 0:
                val_loss = -predict_flow(model, val_X, val_z, val_mem)
                value = float(np.mean([val_loss[val.cluster.to_numpy() == c].mean() for c in val.cluster.unique()]))
                history.append({"epoch": epoch, "learning_rate": lr, "training_nll": float(loss.detach()), "validation_nll": value})
                if value < best:
                    best, best_epoch, best_state = value, epoch, copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), dest/"final.pt")
        torch.save(best_state, dest/"selected.pt")
        pd.DataFrame(history).to_csv(dest/"learning_curve.csv", index=False)
        write_json(dest/"result.json", {"split": split, "seed": seed, "weighting": weighting,
                   "n_discovery": len(train), "n_validation": len(val), "epochs": config["epochs"],
                   "selected_epoch": best_epoch, "selected_validation_nll": best,
                   "final_validation_nll": history[-1]["validation_nll"],
                   "n_parameters": sum(p.numel() for p in model.parameters()),
                   "weight_sha256": {name: sha256(dest/(name+".pt")) for name in ["final", "selected"]},
                   "elapsed_seconds": time.monotonic()-started})
        print(f"{i+1}/96 {path.parent.name} {weighting}: complete in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

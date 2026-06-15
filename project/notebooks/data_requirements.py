"""
Block 6.2: Data requirements prediction for PLATO/TESS.

Estimates how many old clusters (age > 2 Gyr) would be needed for PySR
to distinguish a complex gyrochronology form from Skumanich, using
synthetic data with increasing numbers of old clusters.

Produces:
  - project/results/data_requirements.json
  - project/figures/data_requirements.png
"""

import json
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonicalize import canonicalize

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

if args.test:
    N_OLD_CLUSTERS_GRID = [0, 2, 5]
    N_SEEDS = 2
    NITER = 5
    NPOP = 5
    POP_SIZE = 20
else:
    N_OLD_CLUSTERS_GRID = [0, 1, 2, 5, 10, 20]
    N_SEEDS = 10
    NITER = 50
    NPOP = 30
    POP_SIZE = 50

NOISE_STD = 1.9
STARS_PER_CLUSTER = 100

# ── Load real data distribution ─────────────────────────────────────────────
df_real = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df_real["BPRP_0"] > 0.5) & (df_real["BPRP_0"] < 2.0) & (df_real["fiducial_age_Myr"] > 10.0) & (df_real["Prot"] > 0.1)
df_real = df_real[valid].reset_index(drop=True)
print(f"Real data: {len(df_real)} stars, {df_real['cluster'].nunique()} clusters")

# ── Physics loss ────────────────────────────────────────────────────────────
mono_grid_setup = """
const MONO_GRID_DR = let
    grid_ages = Float32[50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0]
    grid_colors = Float32[0.55, 0.70, 0.85, 1.00, 1.20, 1.50, 1.90]
    n_grid = length(grid_ages) * length(grid_colors)
    grid = Matrix{Float32}(undef, 2, n_grid)
    k = 1
    for c in grid_colors
        for a in grid_ages
            grid[1, k] = a
            grid[2, k] = c
            k += 1
        end
    end
    Ref(grid)
end
"""

physics_loss = """
function physics_loss_dr(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    diffs = prediction .- dataset.y
    if dataset.weights !== nothing
        mse = sum(dataset.weights .* diffs .^ 2) / sum(dataset.weights)
    else
        mse = sum(diffs .^ 2) / length(diffs)
    end
    neg_penalty = sum(max.(0, -prediction)) / length(prediction)
    grid_pred, grid_flag = eval_tree_array(tree, MONO_GRID_DR[], options)
    mono_violations = zero(L)
    if grid_flag
        for ci in 1:7
            offset = (ci - 1) * 7
            for ai in 1:6
                mono_violations += max(zero(L), grid_pred[offset + ai] - grid_pred[offset + ai + 1])
            end
        end
    end
    return mse + 10.0 * neg_penalty + 1.0 * mono_violations
end
"""

from pysr import PySRRegressor
from juliacall import Main as jl

jl.seval("using SymbolicRegression: eval_tree_array, Dataset")
jl.seval(mono_grid_setup)
print("Julia initialized\n")

# ── Generate synthetic data with complex true form ──────────────────────────
# True form: piecewise Skumanich that stalls at 1 Gyr (WMB-like)
# P = 0.344 * sqrt(min(age, 1000)) * color + 0.005 * max(age-1000, 0) * color^0.3
# This has a Skumanich component + an old-age deviation
rng = np.random.default_rng(seed=999)


def true_form(age, color):
    base = 0.344 * np.sqrt(np.minimum(age, 1000.0)) * color
    old_deviation = 0.005 * np.maximum(age - 1000.0, 0.0) * np.power(color, 0.3)
    return base + old_deviation


def generate_dataset(n_old_clusters, rng):
    """Generate synthetic dataset with real young data + synthetic old clusters."""
    # Keep all real data
    df_syn = df_real.copy()

    # Add synthetic old clusters
    old_ages = rng.uniform(2000, 8000, size=n_old_clusters)
    old_rows = []
    for i, age in enumerate(old_ages):
        # Sample colors from real data distribution
        color_sample = rng.choice(df_real["BPRP_0"].values, size=STARS_PER_CLUSTER)
        for color in color_sample:
            old_rows.append({
                "cluster": f"synth_old_{i}",
                "fiducial_age_Myr": age,
                "BPRP_0": color,
                "Prot": 0.0,  # placeholder
            })

    if old_rows:
        df_old = pd.DataFrame(old_rows)
        df_syn = pd.concat([df_syn, df_old], ignore_index=True)

    # Generate periods from true form
    ages = df_syn["fiducial_age_Myr"].values
    colors = df_syn["BPRP_0"].values
    noise = rng.normal(0, NOISE_STD, size=len(df_syn))
    df_syn["Prot"] = np.maximum(true_form(ages, colors) + noise, 0.1)

    return df_syn


# ── Run PySR on each configuration ─────────────────────────────────────────
results = {}
total_t0 = time.time()

for n_old in N_OLD_CLUSTERS_GRID:
    print(f"\n{'='*60}")
    print(f"N_OLD_CLUSTERS = {n_old}")
    print(f"{'='*60}")

    forms_found = []

    for seed in range(N_SEEDS):
        gen_rng = np.random.default_rng(seed=seed * 100 + n_old)
        df_syn = generate_dataset(n_old, gen_rng)

        # Cluster weights
        counts = df_syn.groupby("cluster")["Prot"].transform("count").values.astype(float)
        w = 1.0 / counts
        w = w * len(w) / w.sum()

        X = np.column_stack([df_syn["fiducial_age_Myr"].values, df_syn["BPRP_0"].values])
        y = df_syn["Prot"].values

        outdir = os.path.join(RESULTS, "data_requirements", f"nold{n_old}_seed{seed}")
        os.makedirs(outdir, exist_ok=True)

        model = PySRRegressor(
            niterations=NITER,
            binary_operators=["+", "-", "*", "/", "pow"],
            unary_operators=["log", "exp", "sqrt"],
            populations=NPOP,
            population_size=POP_SIZE,
            maxsize=25,
            batching=True,
            batch_size=500,
            complexity_of_operators={"pow": 2, "log": 1, "exp": 1, "sqrt": 1},
            nested_constraints={
                "exp": {"exp": 0, "log": 0},
                "log": {"log": 0, "exp": 0},
                "sqrt": {"sqrt": 1},
            },
            constraints={"pow": (-1, 1)},
            progress=False,
            deterministic=False,
            parallelism="multithreading",
            loss_function=physics_loss,
            output_directory=outdir,
        )

        t0 = time.time()
        model.fit(X, y, variable_names=["age", "color"], weights=w)
        elapsed = time.time() - t0

        best = model.get_best()
        try:
            canon = canonicalize(str(best.equation))
        except Exception:
            canon = "ERROR"

        forms_found.append(canon)
        print(f"  seed={seed}: {canon[:40]:40s}  ({elapsed:.0f}s)")

    # Count form frequency
    form_counts = {}
    for f in forms_found:
        form_counts[f] = form_counts.get(f, 0) + 1

    top_form, top_count = max(form_counts.items(), key=lambda x: x[1])
    is_skumanich = "sqrt(age)" in top_form and "color" in top_form and top_form.count("C") <= 2

    results[n_old] = {
        "n_old_clusters": n_old,
        "n_total_stars": int(len(df_real) + n_old * STARS_PER_CLUSTER),
        "n_seeds": N_SEEDS,
        "dominant_form": top_form,
        "dominant_count": top_count,
        "dominant_pct": round(100 * top_count / N_SEEDS, 1),
        "n_unique_forms": len(form_counts),
        "is_skumanich_dominant": is_skumanich,
        "form_distribution": form_counts,
    }

    print(f"\n  Result: {top_form} at {top_count}/{N_SEEDS} ({100*top_count/N_SEEDS:.0f}%)")
    print(f"  Skumanich still dominant: {is_skumanich}")

total_elapsed = time.time() - total_t0

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"DATA REQUIREMENTS SUMMARY")
print(f"{'='*70}")
print(f"True form: piecewise Skumanich with stall at 1 Gyr + old-age deviation\n")

print(f"{'N_old':>5s}  {'N_total':>7s}  {'Dominant form':>30s}  {'Pct':>5s}  {'Sku?':>5s}")
print("-" * 60)
for n_old in N_OLD_CLUSTERS_GRID:
    r = results[n_old]
    print(f"{n_old:5d}  {r['n_total_stars']:7d}  {r['dominant_form'][:30]:>30s}  "
          f"{r['dominant_pct']:5.1f}%  {'YES' if r['is_skumanich_dominant'] else 'NO':>5s}")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = N_OLD_CLUSTERS_GRID
y = [results[n]["dominant_pct"] for n in x]
sku = [results[n]["is_skumanich_dominant"] for n in x]

bars = ax.bar(range(len(x)), y, color=["steelblue" if s else "coral" for s in sku],
              edgecolor="white")
ax.set_xticks(range(len(x)))
ax.set_xticklabels([str(n) for n in x])
ax.set_xlabel("Number of old clusters (age > 2 Gyr)")
ax.set_ylabel("Dominant form frequency (%)")
ax.set_title("How Many Old Clusters to Detect Non-Skumanich Physics?")
ax.axhline(80, color="gray", ls="--", alpha=0.5, label="80% stability threshold")
ax.legend()

# Color legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="steelblue", label="Skumanich dominant"),
    Patch(color="coral", label="Non-Skumanich dominant"),
    plt.Line2D([0], [0], color="gray", ls="--", label="80% threshold"),
], loc="upper right", fontsize=9)

plt.tight_layout()
fig_path = os.path.join(FIG, "data_requirements.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved: {fig_path}")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS, "data_requirements.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {out_path}")
print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

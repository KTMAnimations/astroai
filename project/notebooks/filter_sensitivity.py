"""
Task 4.2: Slow-sequence filter sensitivity analysis.

Tests whether the Skumanich form survives across different filter thresholds.
Grid: young thresholds × old thresholds, plus a no-filter ablation.

If PySR finds the same form regardless of threshold, the result is robust.
If the form changes, the filter is creating the signal.

Usage:
    python filter_sensitivity.py             # full grid (26 configurations)
    python filter_sensitivity.py --test      # smoke test (4 configurations)
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonicalize import canonicalize
from filter_slow_seq import apply_slow_seq_filter

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

if args.test:
    YOUNG_GRID = [0.30, 0.50]
    OLD_GRID = [0.20, 0.40]
    NITER = 5
    NPOP = 5
    POP_SIZE = 20
else:
    YOUNG_GRID = [0.20, 0.30, 0.40, 0.50, 0.60]
    OLD_GRID = [0.15, 0.20, 0.30, 0.40, 0.50]
    NITER = 50
    NPOP = 30
    POP_SIZE = 50

# ── Load unfiltered data ───────────────────────────────────────────────────
df_all = pd.read_csv(os.path.join(DATA, "clean_all.csv"))
print(f"Unfiltered: {len(df_all)} stars from {df_all['cluster'].nunique()} clusters")

# ── Physics loss ────────────────────────────────────────────────────────────
mono_grid_setup = """
const MONO_GRID_FS = let
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
function physics_loss_fs(tree, dataset::Dataset{T,L}, options) where {T,L}
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
    grid_pred, grid_flag = eval_tree_array(tree, MONO_GRID_FS[], options)
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


def fit_skumanich(ages, colors, periods, weights=None):
    template = np.sqrt(ages) * colors
    if weights is not None:
        C = np.sum(weights * periods * template) / np.sum(weights * template ** 2)
    else:
        C = np.sum(periods * template) / np.sum(template ** 2)
    return C


# ── Run grid ────────────────────────────────────────────────────────────────
results = []
total_t0 = time.time()

# Configuration list: no-filter + grid
configs = [(None, None)]  # no-filter ablation
for yf in YOUNG_GRID:
    for of in OLD_GRID:
        configs.append((yf, of))

for i, (yf, of) in enumerate(configs):
    if yf is None:
        label = "NO_FILTER"
        df_filt = df_all.copy()
    else:
        label = f"young={yf:.0%}_old={of:.0%}"
        df_filt = apply_slow_seq_filter(df_all, young_frac=yf, old_frac=of)

    # Apply FGK domain cut
    valid = (df_filt["BPRP_0"] > 0.5) & (df_filt["BPRP_0"] < 2.0) & (df_filt["fiducial_age_Myr"] > 10.0) & (df_filt["Prot"] > 0.1)
    df_fgk = df_filt[valid].reset_index(drop=True)
    n_stars = len(df_fgk)

    if n_stars < 100:
        print(f"[{i+1}/{len(configs)}] {label}: SKIPPED (n={n_stars} < 100)")
        continue

    print(f"[{i+1}/{len(configs)}] {label} (n={n_stars}) ", end="", flush=True)

    # Cluster weights
    counts = df_fgk.groupby("cluster")["Prot"].transform("count").values.astype(float)
    w = 1.0 / counts
    w = w * len(w) / w.sum()

    # 1-param Skumanich
    C_sku = fit_skumanich(df_fgk["fiducial_age_Myr"].values,
                          df_fgk["BPRP_0"].values,
                          df_fgk["Prot"].values, w)
    pred_sku = C_sku * np.sqrt(df_fgk["fiducial_age_Myr"].values) * df_fgk["BPRP_0"].values
    rmse_sku = float(np.sqrt(np.mean((df_fgk["Prot"].values - pred_sku) ** 2)))

    # PySR
    X = np.column_stack([df_fgk["fiducial_age_Myr"].values, df_fgk["BPRP_0"].values])
    y = df_fgk["Prot"].values

    outdir = os.path.join(RESULTS, "filter_sensitivity", f"config_{i:02d}")
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

    result = {
        "young_frac": yf,
        "old_frac": of,
        "label": label,
        "n_stars": n_stars,
        "n_clusters": int(df_fgk["cluster"].nunique()),
        "skumanich_C": float(C_sku),
        "skumanich_rmse": rmse_sku,
        "pysr_equation": str(best.equation),
        "pysr_complexity": int(best.complexity),
        "pysr_canonical": canon,
        "wall_clock_seconds": elapsed,
    }
    results.append(result)
    print(f"canon={canon[:35]:35s}  C={C_sku:.4f}  ({elapsed:.0f}s)")

total_elapsed = time.time() - total_t0

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"FILTER SENSITIVITY SUMMARY — {len(results)} configurations")
print(f"{'='*70}")
print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

print(f"{'Label':>25s}  {'N':>5s}  {'C':>7s}  {'RMSE':>7s}  {'Canonical form':>30s}")
print("-" * 85)
for r in results:
    print(f"{r['label']:>25s}  {r['n_stars']:5d}  {r['skumanich_C']:7.4f}  "
          f"{r['skumanich_rmse']:7.3f}  {r['pysr_canonical'][:30]:>30s}")

# Count how many configs find Skumanich
n_sku = sum(1 for r in results if "sqrt(age)" in r["pysr_canonical"] and "color" in r["pysr_canonical"])
print(f"\nSkumanich found in {n_sku}/{len(results)} configurations ({100*n_sku/max(len(results),1):.0f}%)")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS, "filter_sensitivity.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

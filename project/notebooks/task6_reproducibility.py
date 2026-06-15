"""
Task 6: PySR reproducibility check (3 runs).

Runs PySR 3 times with the same configuration but different random seeds
on the restricted FGK dataset (color < 2.0). Compares best equations,
RMSE, and canonical forms across runs.

Usage:
    python task6_reproducibility.py          # full 3 runs (200 iters each)
    python task6_reproducibility.py --test   # smoke test (5 iters each)
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

# ── Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Smoke test (5 iterations)")
args = parser.parse_args()

NITER = 5 if args.test else 200
NPOP = 5 if args.test else 30
POP_SIZE = 20 if args.test else 50
N_RUNS = 3

print(f"Config: {N_RUNS} runs, niterations={NITER}, populations={NPOP}")

# ── Load restricted FGK data ─────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
print(f"Dataset: {len(df)} FGK stars")

ages = df["fiducial_age_Myr"].values
colors = df["BPRP_0"].values
periods = df["Prot"].values

# Cluster-level weights
cluster_counts = df.groupby("cluster")["Prot"].transform("count").values.astype(float)
sample_weights = 1.0 / cluster_counts
sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

X = np.column_stack([ages, colors])
y = periods

# ── Physics loss (same as phase4_pysr_search.py) ─────────────────────────
mono_grid_setup = """
const MONO_GRID = let
    grid_ages = Float32[100.0, 200.0, 350.0, 500.0, 700.0, 1000.0]
    grid_colors = Float32[0.65, 0.75, 0.85, 0.95, 1.05]
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
function physics_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
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
    grid_pred, grid_flag = eval_tree_array(tree, MONO_GRID[], options)
    mono_violations = zero(L)
    if grid_flag
        for ci in 1:5
            offset = (ci - 1) * 6
            for ai in 1:5
                mono_violations += max(zero(L), grid_pred[offset + ai] - grid_pred[offset + ai + 1])
            end
        end
    end
    return mse + 10.0 * neg_penalty + 1.0 * mono_violations
end
"""

shared_kwargs = dict(
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
    progress=True,
    deterministic=False,
    parallelism="multithreading",
)

# ── Initialize Julia ─────────────────────────────────────────────────────
from pysr import PySRRegressor
from juliacall import Main as jl

jl.seval("using SymbolicRegression: eval_tree_array, Dataset")
jl.seval(mono_grid_setup)
print("Julia initialized, monotonicity grid ready")

# ── Run PySR N_RUNS times ────────────────────────────────────────────────
run_results = []

for run_i in range(N_RUNS):
    print(f"\n{'='*60}")
    print(f"RUN {run_i + 1}/{N_RUNS}")
    print(f"{'='*60}")

    outdir = os.path.join(RESULTS, f"repro_run_{run_i + 1}")
    os.makedirs(outdir, exist_ok=True)

    model = PySRRegressor(
        loss_function=physics_loss,
        output_directory=outdir,
        **shared_kwargs,
    )

    t0 = time.time()
    model.fit(X, y, variable_names=["age", "color"], weights=sample_weights)
    elapsed = time.time() - t0

    best = model.get_best()
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    # Save Pareto frontier
    model.equations_.to_csv(os.path.join(outdir, f"pareto_run_{run_i + 1}.csv"), index=False)

    # Canonicalize best equation
    try:
        canon = canonicalize(str(best.equation))
    except Exception as e:
        canon = f"ERROR: {e}"

    result = {
        "run": run_i + 1,
        "best_equation": str(best.equation),
        "best_rmse": rmse,
        "best_complexity": int(best.complexity),
        "best_loss": float(best.loss),
        "canonical_form": canon,
        "n_pareto": len(model.equations_),
        "wall_clock_seconds": elapsed,
    }
    run_results.append(result)

    print(f"\nRun {run_i + 1}: RMSE={rmse:.3f} d, cx={best.complexity}, "
          f"time={elapsed:.0f}s")
    print(f"  Equation: {best.equation}")
    print(f"  Canonical: {canon}")

# ── Save all results ─────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "reproducibility_results.json"), "w") as f:
    json.dump(run_results, f, indent=2)

# ── Comparison table ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"REPRODUCIBILITY SUMMARY")
print(f"{'='*60}")

print(f"\n{'Run':>4s}  {'RMSE':>7s}  {'Cx':>3s}  {'Time':>7s}  Equation")
print("─" * 90)
for r in run_results:
    eq_short = r["best_equation"][:55]
    print(f"  {r['run']:2d}   {r['best_rmse']:7.3f}  {r['best_complexity']:3d}  "
          f"{r['wall_clock_seconds']:6.0f}s  {eq_short}")

# Canonical form comparison
print(f"\nCanonical forms:")
canon_counts = {}
for r in run_results:
    form = r["canonical_form"]
    canon_counts[form] = canon_counts.get(form, 0) + 1
    print(f"  Run {r['run']}: {form}")

print(f"\nUnique canonical forms: {len(canon_counts)}")
for form, count in sorted(canon_counts.items(), key=lambda x: -x[1]):
    print(f"  {count}x: {form}")

# RMSE range
rmses = [r["best_rmse"] for r in run_results]
rmse_range = max(rmses) - min(rmses)
print(f"\nRMSE range: {rmse_range:.3f} d (min={min(rmses):.3f}, max={max(rmses):.3f})")

# Decision
same_form_count = max(canon_counts.values())
if rmse_range < 0.3 and same_form_count >= 2:
    print(f"\n→ DECISION: Proceed with 200 iterations for bootstrap")
    print(f"  (RMSE range < 0.3 d AND same canonical form in {same_form_count}/{N_RUNS} runs)")
elif rmse_range < 0.5:
    print(f"\n→ DECISION: Increase to 300 iterations for bootstrap")
    print(f"  (RMSE range {rmse_range:.3f} d is between 0.3-0.5)")
else:
    print(f"\n→ DECISION: Increase to 400 iterations and rerun this check")
    print(f"  (RMSE range {rmse_range:.3f} d > 0.5)")

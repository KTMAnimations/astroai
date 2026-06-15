"""
Block 2: PySR convergence study.

Tests whether 50 iterations is sufficient for stable results by running PySR
at [25, 50, 100, 200] iterations × 5 random seeds. Records both get_best()
and top-1-by-RMSE canonical forms.

If Skumanich appears at 25 iterations as readily as at 200, the bootstrap
(which used 50 iterations) is validated. If complex forms only appear at
higher iteration counts, the 83% figure is an artifact.

Usage:
    python convergence_study.py             # full run
    python convergence_study.py --test      # smoke test (2 seeds, [25,50] iters)
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

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

if args.test:
    ITER_GRID = [25, 50]
    N_SEEDS = 2
    NPOP = 5
    POP_SIZE = 20
else:
    ITER_GRID = [25, 50, 100, 200]
    N_SEEDS = 5
    NPOP = 30
    POP_SIZE = 50

# ── Load FGK data ───────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
N_STARS = len(df)
print(f"Dataset: {N_STARS} FGK stars from {df['cluster'].nunique()} clusters")

ages = df["fiducial_age_Myr"].values
colors = df["BPRP_0"].values
periods = df["Prot"].values
clusters = df["cluster"].values

# Cluster weights
cluster_counts = df.groupby("cluster")["Prot"].transform("count").values.astype(float)
sample_weights = 1.0 / cluster_counts
sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

X = np.column_stack([ages, colors])

# ── Physics loss (same as phase4_pysr_search.py) ───────────────────────────
mono_grid_setup = """
const MONO_GRID_CONV = let
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
function physics_loss_conv(tree, dataset::Dataset{T,L}, options) where {T,L}
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
    grid_pred, grid_flag = eval_tree_array(tree, MONO_GRID_CONV[], options)
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

# ── Initialize Julia ────────────────────────────────────────────────────────
from pysr import PySRRegressor
from juliacall import Main as jl

jl.seval("using SymbolicRegression: eval_tree_array, Dataset")
jl.seval(mono_grid_setup)
print("Julia initialized\n")


def unweighted_rmse(y_true, y_pred):
    mask = np.isfinite(y_pred)
    if mask.sum() < len(y_true) * 0.5:
        return np.inf
    ss = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    return float(np.sqrt(ss / len(y_true)))


# ── Run convergence grid ────────────────────────────────────────────────────
OUTDIR = os.path.join(RESULTS, "convergence")
os.makedirs(OUTDIR, exist_ok=True)

results = []
total_t0 = time.time()

for niter in ITER_GRID:
    for seed in range(N_SEEDS):
        label = f"iter={niter:3d} seed={seed}"
        print(f"[{label}] ", end="", flush=True)

        run_dir = os.path.join(OUTDIR, f"iter{niter}_seed{seed}")
        os.makedirs(run_dir, exist_ok=True)

        model = PySRRegressor(
            niterations=niter,
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
            output_directory=run_dir,
            random_state=seed * 1000 + niter,
        )

        t0 = time.time()
        model.fit(X, periods, variable_names=["age", "color"], weights=sample_weights)
        elapsed = time.time() - t0

        # get_best() canonical form
        best = model.get_best()
        try:
            canon_best = canonicalize(str(best.equation))
        except Exception:
            canon_best = "ERROR"

        # top-1 by RMSE
        pareto = model.equations_.copy()
        import sympy as sp
        age_sym, color_sym = sp.symbols("age color", positive=True)

        best_rmse_eq = None
        best_rmse_val = np.inf
        for _, row in pareto.iterrows():
            try:
                expr_str = str(row["sympy_format"]).replace("^", "**")
                expr = sp.sympify(expr_str, locals={"age": age_sym, "color": color_sym})
                f = sp.lambdify((age_sym, color_sym), expr, modules=["numpy"])
                pred = f(ages, colors)
                pred = np.where(np.isfinite(pred), pred, np.nan)
                rmse = unweighted_rmse(periods, pred)
                if rmse < best_rmse_val:
                    best_rmse_val = rmse
                    best_rmse_eq = str(row["equation"])
            except Exception:
                pass

        try:
            canon_rmse = canonicalize(best_rmse_eq) if best_rmse_eq else "ERROR"
        except Exception:
            canon_rmse = "ERROR"

        result = {
            "niterations": niter,
            "seed": seed,
            "get_best_equation": str(best.equation),
            "get_best_complexity": int(best.complexity),
            "get_best_canonical": canon_best,
            "top1_rmse_equation": best_rmse_eq,
            "top1_rmse_value": best_rmse_val,
            "top1_rmse_canonical": canon_rmse,
            "n_pareto": len(pareto),
            "wall_clock_seconds": elapsed,
        }
        results.append(result)

        print(f"get_best={canon_best[:30]:30s}  top1_rmse={canon_rmse[:30]:30s}  ({elapsed:.0f}s)")

total_elapsed = time.time() - total_t0

# ── Analysis ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"CONVERGENCE STUDY — {len(results)} runs")
print(f"{'='*70}")
print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

print(f"{'Iters':>5s}  {'get_best form':>40s}  {'Count':>5s}  {'Frac':>5s}")
print("-" * 65)
for niter in ITER_GRID:
    runs = [r for r in results if r["niterations"] == niter]
    forms = {}
    for r in runs:
        f = r["get_best_canonical"]
        forms[f] = forms.get(f, 0) + 1
    for form, count in sorted(forms.items(), key=lambda x: -x[1]):
        print(f"{niter:5d}  {form:>40s}  {count:5d}  {count/len(runs):5.1%}")

print(f"\n{'Iters':>5s}  {'top-1 RMSE form':>40s}  {'Count':>5s}  {'Frac':>5s}")
print("-" * 65)
for niter in ITER_GRID:
    runs = [r for r in results if r["niterations"] == niter]
    forms = {}
    for r in runs:
        f = r["top1_rmse_canonical"]
        forms[f] = forms.get(f, 0) + 1
    for form, count in sorted(forms.items(), key=lambda x: -x[1]):
        print(f"{niter:5d}  {form[:40]:>40s}  {count:5d}  {count/len(runs):5.1%}")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS, "convergence_study.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

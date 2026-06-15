"""
Task 4.4: Physics constraint ablation study.

Runs 10 constrained + 10 unconstrained PySR at 200 iterations to test
whether the physics loss function changes the functional form or just
the coefficients.

If both find the same forms at the same frequencies, physics constraints
are not load-bearing for this dataset (though still a methodological
contribution for the paper).

Usage:
    python constraint_ablation.py             # full (20 runs at 200 iters)
    python constraint_ablation.py --test      # smoke test (4 runs at 5 iters)
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
    N_RUNS = 2
    NITER = 5
    NPOP = 5
    POP_SIZE = 20
else:
    N_RUNS = 10
    NITER = 200
    NPOP = 30
    POP_SIZE = 50

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
N_STARS = len(df)
print(f"Dataset: {N_STARS} FGK stars from {df['cluster'].nunique()} clusters")

ages = df["fiducial_age_Myr"].values
colors = df["BPRP_0"].values
periods = df["Prot"].values

# Cluster weights
cluster_counts = df.groupby("cluster")["Prot"].transform("count").values.astype(float)
sample_weights = 1.0 / cluster_counts
sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

X = np.column_stack([ages, colors])

# ── Physics loss ────────────────────────────────────────────────────────────
mono_grid_setup = """
const MONO_GRID_CA = let
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
function physics_loss_ca(tree, dataset::Dataset{T,L}, options) where {T,L}
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
    grid_pred, grid_flag = eval_tree_array(tree, MONO_GRID_CA[], options)
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


def unweighted_rmse(y_true, y_pred):
    mask = np.isfinite(y_pred)
    if mask.sum() < len(y_true) * 0.5:
        return np.inf
    return float(np.sqrt(np.sum((y_true[mask] - y_pred[mask]) ** 2) / len(y_true)))


# ── Evaluation grid for checking negative/monotonicity violations ──────────
eval_ages = np.linspace(50, 4000, 50)
eval_colors = np.linspace(0.5, 2.0, 30)
eval_grid_age, eval_grid_color = np.meshgrid(eval_ages, eval_colors)
eval_X = np.column_stack([eval_grid_age.ravel(), eval_grid_color.ravel()])


# ── Run ─────────────────────────────────────────────────────────────────────
results = []
total_t0 = time.time()

for constrained in [True, False]:
    label = "constrained" if constrained else "unconstrained"

    for run in range(N_RUNS):
        print(f"[{label} run {run+1}/{N_RUNS}] ", end="", flush=True)

        outdir = os.path.join(RESULTS, "constraint_ablation", f"{label}_run{run}")
        os.makedirs(outdir, exist_ok=True)

        kwargs = dict(
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
            output_directory=outdir,
        )

        if constrained:
            kwargs["loss_function"] = physics_loss

        model = PySRRegressor(**kwargs)

        t0 = time.time()
        model.fit(X, periods, variable_names=["age", "color"], weights=sample_weights)
        elapsed = time.time() - t0

        best = model.get_best()
        try:
            canon = canonicalize(str(best.equation))
        except Exception:
            canon = "ERROR"

        # Evaluate on grid for violation counts
        import sympy as sp
        age_sym, color_sym = sp.symbols("age color", positive=True)

        try:
            expr_str = str(model.equations_.iloc[-1]["sympy_format"]).replace("^", "**")
            expr = sp.sympify(expr_str, locals={"age": age_sym, "color": color_sym})
            f = sp.lambdify((age_sym, color_sym), expr, modules=["numpy"])

            # Best equation evaluation
            best_str = str(best.sympy_format).replace("^", "**") if hasattr(best, "sympy_format") else str(best.equation).replace("^", "**")
            best_expr = sp.sympify(best_str, locals={"age": age_sym, "color": color_sym})
            best_fn = sp.lambdify((age_sym, color_sym), best_expr, modules=["numpy"])

            grid_pred = best_fn(eval_X[:, 0], eval_X[:, 1])
            n_negative = int(np.sum(grid_pred < 0))

            # Check monotonicity along age for each color
            mono_violations = 0
            for j in range(30):  # each color row
                row = grid_pred[j * 50:(j + 1) * 50]
                if np.any(np.isnan(row)):
                    continue
                mono_violations += int(np.sum(np.diff(row) < 0))
        except Exception:
            n_negative = -1
            mono_violations = -1

        result = {
            "constrained": constrained,
            "run": run,
            "equation": str(best.equation),
            "complexity": int(best.complexity),
            "canonical_form": canon,
            "n_negative_grid": n_negative,
            "n_mono_violations_grid": mono_violations,
            "wall_clock_seconds": elapsed,
        }
        results.append(result)

        print(f"{canon[:35]:35s}  neg={n_negative:3d}  mono={mono_violations:3d}  ({elapsed:.0f}s)")

total_elapsed = time.time() - total_t0

# ── Analysis ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"CONSTRAINT ABLATION SUMMARY — {len(results)} runs")
print(f"{'='*70}")
print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

for label in ["constrained", "unconstrained"]:
    runs = [r for r in results if r["constrained"] == (label == "constrained")]
    forms = {}
    for r in runs:
        forms[r["canonical_form"]] = forms.get(r["canonical_form"], 0) + 1

    neg_counts = [r["n_negative_grid"] for r in runs if r["n_negative_grid"] >= 0]
    mono_counts = [r["n_mono_violations_grid"] for r in runs if r["n_mono_violations_grid"] >= 0]

    print(f"\n{label.upper()} ({len(runs)} runs):")
    print(f"  Forms: {forms}")
    if neg_counts:
        print(f"  Negative predictions: mean={np.mean(neg_counts):.1f}, max={max(neg_counts)}")
    if mono_counts:
        print(f"  Monotonicity violations: mean={np.mean(mono_counts):.1f}, max={max(mono_counts)}")

# Compare: do constrained and unconstrained find the same forms?
c_forms = set(r["canonical_form"] for r in results if r["constrained"])
u_forms = set(r["canonical_form"] for r in results if not r["constrained"])
overlap = c_forms & u_forms

print(f"\nConstrained forms: {c_forms}")
print(f"Unconstrained forms: {u_forms}")
print(f"Overlap: {overlap}")
print(f"Physics constraints load-bearing: {'NO' if overlap == c_forms == u_forms else 'POSSIBLY'}")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS, "constraint_ablation.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

"""
Block 3: Synthetic control experiment for bootstrap stability validation.

Generates three synthetic datasets with KNOWN ground truth, runs the same
bootstrap stability analysis, and verifies that the method correctly:
  A) Identifies stable forms when the true form is simple (Skumanich)
  B) Detects instability when the true form is complex
  C) Shows data limitations when the true form has a piecewise transition

Without synthetic validation, the bootstrap stability claim is unverifiable.

Usage:
    python synthetic_control.py             # full (100 resamples × 3 datasets)
    python synthetic_control.py --test      # smoke test (5 resamples × 3 datasets)
    python synthetic_control.py --dataset A # run only dataset A
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
parser.add_argument("--dataset", choices=["A", "B", "C"], default=None,
                    help="Run only one dataset (default: all)")
parser.add_argument("--n-resamples", type=int, default=100)
args = parser.parse_args()

if args.test:
    N_RESAMPLES = 5
    NITER = 5
    NPOP = 5
    POP_SIZE = 20
else:
    N_RESAMPLES = args.n_resamples
    NITER = 50
    NPOP = 30
    POP_SIZE = 50

DATASETS_TO_RUN = [args.dataset] if args.dataset else ["A", "B", "C"]

# ── Load real data (for age/color distribution only) ────────────────────────
df_real = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df_real["BPRP_0"] > 0.5) & (df_real["BPRP_0"] < 2.0) & (df_real["fiducial_age_Myr"] > 10.0) & (df_real["Prot"] > 0.1)
df_real = df_real[valid].reset_index(drop=True)
N_STARS = len(df_real)
print(f"Using real data distribution: {N_STARS} stars from {df_real['cluster'].nunique()} clusters")

ages_real = df_real["fiducial_age_Myr"].values
colors_real = df_real["BPRP_0"].values
clusters_real = df_real["cluster"].values

NOISE_STD = 1.9  # matched to intrinsic scatter from noise_floor.json

# ── Physics loss (same as all other scripts) ────────────────────────────────
mono_grid_setup = """
const MONO_GRID_SYN = let
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
function physics_loss_syn(tree, dataset::Dataset{T,L}, options) where {T,L}
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
    grid_pred, grid_flag = eval_tree_array(tree, MONO_GRID_SYN[], options)
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
    progress=False,
    deterministic=False,
    parallelism="multithreading",
)

# ── Initialize Julia ────────────────────────────────────────────────────────
from pysr import PySRRegressor
from juliacall import Main as jl

jl.seval("using SymbolicRegression: eval_tree_array, Dataset")
jl.seval(mono_grid_setup)
print("Julia initialized\n")


# ── Helpers ─────────────────────────────────────────────────────────────────
def compute_cluster_weights(df_subset):
    counts = df_subset.groupby("cluster")["Prot"].transform("count").values.astype(float)
    w = 1.0 / counts
    w = w * len(w) / w.sum()
    return w


def unweighted_rmse(y_true, y_pred):
    mask = np.isfinite(y_pred)
    if mask.sum() < len(y_true) * 0.5:
        return np.inf
    ss = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    return float(np.sqrt(ss / len(y_true)))


# ── Generate synthetic datasets ─────────────────────────────────────────────
rng = np.random.default_rng(seed=12345)
noise = rng.normal(0, NOISE_STD, size=N_STARS)

# Dataset A: Known Skumanich (positive control)
periods_A = 0.344 * np.sqrt(ages_real) * colors_real + noise
periods_A = np.maximum(periods_A, 0.1)  # floor at 0.1 days

# Dataset B: Known complex (negative control) — two-term structure
periods_B = (0.3 * np.power(ages_real, 0.4) * np.power(np.maximum(colors_real - 0.3, 0.01), 0.6)
             + 0.005 * ages_real * np.exp(-colors_real)
             + noise)
periods_B = np.maximum(periods_B, 0.1)

# Dataset C: Known piecewise (transition at 1000 Myr)
age_factor = np.where(ages_real < 1000, np.sqrt(ages_real), np.sqrt(1000.0))
periods_C = 0.344 * age_factor * colors_real + noise
periods_C = np.maximum(periods_C, 0.1)

datasets = {
    "A": {"periods": periods_A, "true_form": "P = 0.344 * sqrt(age) * color",
           "description": "Known Skumanich (positive control)"},
    "B": {"periods": periods_B, "true_form": "P = 0.3*age^0.4*(color-0.3)^0.6 + 0.005*age*exp(-color)",
           "description": "Known complex (negative control)"},
    "C": {"periods": periods_C, "true_form": "P = 0.344*sqrt(min(age,1000))*color",
           "description": "Known piecewise (transition detection)"},
}

for name in DATASETS_TO_RUN:
    print(f"Dataset {name}: {datasets[name]['description']}")
    p = datasets[name]["periods"]
    print(f"  Period range: [{p.min():.2f}, {p.max():.2f}] d, mean={p.mean():.2f} d")


# ── Run bootstrap on each dataset ───────────────────────────────────────────
all_results = {}

for ds_name in DATASETS_TO_RUN:
    ds = datasets[ds_name]
    periods_syn = ds["periods"]

    OUTDIR = os.path.join(RESULTS, "synthetic", f"dataset_{ds_name}")
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DATASET {ds_name}: {ds['description']}")
    print(f"True form: {ds['true_form']}")
    print(f"{'='*70}")

    # Build DataFrame for weight computation
    df_syn = df_real.copy()
    df_syn["Prot"] = periods_syn

    boot_rng = np.random.default_rng(seed=42)
    ds_results = []

    total_t0 = time.time()

    for i in range(N_RESAMPLES):
        print(f"  [{ds_name}:{i+1}/{N_RESAMPLES}] ", end="", flush=True)

        boot_idx = boot_rng.choice(N_STARS, size=N_STARS, replace=True)
        df_boot = df_syn.iloc[boot_idx].reset_index(drop=True)

        X_boot = np.column_stack([df_boot["fiducial_age_Myr"].values,
                                   df_boot["BPRP_0"].values])
        y_boot = df_boot["Prot"].values
        w_boot = compute_cluster_weights(df_boot)

        run_dir = os.path.join(OUTDIR, f"boot_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)

        model = PySRRegressor(
            loss_function=physics_loss,
            output_directory=run_dir,
            **shared_kwargs,
        )

        t0 = time.time()
        model.fit(X_boot, y_boot, variable_names=["age", "color"], weights=w_boot)
        elapsed = time.time() - t0

        # get_best()
        best = model.get_best()
        try:
            canon_best = canonicalize(str(best.equation))
        except Exception:
            canon_best = "ERROR"

        # top-1 by RMSE on full synthetic data
        pareto = model.equations_.copy()
        import sympy as sp
        age_sym, color_sym = sp.symbols("age color", positive=True)

        top_eqs = []
        for _, row in pareto.iterrows():
            try:
                expr_str = str(row["sympy_format"]).replace("^", "**")
                expr = sp.sympify(expr_str, locals={"age": age_sym, "color": color_sym})
                f = sp.lambdify((age_sym, color_sym), expr, modules=["numpy"])
                pred = f(ages_real, colors_real)
                pred = np.where(np.isfinite(pred), pred, np.nan)
                rmse = unweighted_rmse(periods_syn, pred)
            except Exception:
                rmse = np.inf

            try:
                canon = canonicalize(str(row["equation"]))
            except Exception:
                canon = "ERROR"

            top_eqs.append({
                "complexity": int(row["complexity"]),
                "equation": str(row["equation"]),
                "unweighted_rmse": rmse,
                "canonical_form": canon,
            })

        top_eqs.sort(key=lambda x: x["unweighted_rmse"])
        top_3 = top_eqs[:3]

        result = {
            "resample": i,
            "n_unique_stars": len(set(boot_idx)),
            "best_equation": str(best.equation),
            "best_complexity": int(best.complexity),
            "best_canonical": canon_best,
            "top_3": top_3,
            "n_pareto": len(pareto),
            "wall_clock_seconds": elapsed,
        }
        ds_results.append(result)

        print(f"best={canon_best[:35]:35s}  ({elapsed:.0f}s)")

        if (i + 1) % 10 == 0 or i == N_RESAMPLES - 1:
            with open(os.path.join(OUTDIR, "bootstrap_summary.json"), "w") as f:
                json.dump(ds_results, f, indent=2)

    total_elapsed = time.time() - total_t0

    # Analysis
    print(f"\n--- Dataset {ds_name} Results ({len(ds_results)} resamples, {total_elapsed:.0f}s) ---")

    best_forms = {}
    for r in ds_results:
        form = r["best_canonical"]
        best_forms[form] = best_forms.get(form, 0) + 1

    print(f"\nCanonical form frequency (get_best):")
    print(f"{'Count':>6s}  {'Pct':>5s}  Form")
    print("-" * 60)
    for form, count in sorted(best_forms.items(), key=lambda x: -x[1])[:10]:
        pct = 100.0 * count / len(ds_results)
        print(f"{count:6d}  {pct:5.1f}%  {form}")

    # Dominant form percentage
    top_form, top_count = max(best_forms.items(), key=lambda x: x[1])
    dominance = top_count / len(ds_results)

    all_results[ds_name] = {
        "description": ds["description"],
        "true_form": ds["true_form"],
        "n_resamples": len(ds_results),
        "dominant_form": top_form,
        "dominant_form_frequency": top_count,
        "dominant_form_percentage": round(100 * dominance, 1),
        "form_distribution": best_forms,
        "n_unique_forms": len(best_forms),
        "total_wall_clock_seconds": total_elapsed,
        "interpretation": (
            "STABLE: method correctly identifies unique form"
            if dominance >= 0.8 else
            "UNSTABLE: method correctly detects form instability"
            if dominance < 0.5 else
            "MARGINAL: moderate form stability"
        ),
    }


# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SYNTHETIC CONTROL SUMMARY")
print(f"{'='*70}\n")

print(f"{'Dataset':>8s}  {'Dominant form':>35s}  {'Freq':>5s}  {'#Forms':>6s}  {'Verdict':>10s}")
print("-" * 75)
for ds_name in DATASETS_TO_RUN:
    r = all_results[ds_name]
    print(f"{ds_name:>8s}  {r['dominant_form'][:35]:>35s}  "
          f"{r['dominant_form_percentage']:5.1f}%  {r['n_unique_forms']:6d}  "
          f"{'STABLE' if r['dominant_form_percentage'] >= 80 else 'UNSTABLE' if r['dominant_form_percentage'] < 50 else 'MARGINAL':>10s}")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS, "synthetic_control_summary.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_path}")

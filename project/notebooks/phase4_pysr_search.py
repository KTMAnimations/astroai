"""
Phase 4: PySR symbolic regression search for gyrochronology.

Runs PySR twice:
  1. Physics-constrained loss (negativity + monotonicity penalties)
  2. Unconstrained (plain weighted MSE)

Both use inverse-frequency age weighting to mitigate the severe age
imbalance in the training data (73% of stars are <500 Myr).

Usage:
  python phase4_pysr_search.py              # full run (200 iterations)
  python phase4_pysr_search.py --test       # smoke test (5 iterations)
  python phase4_pysr_search.py --unconstrained-only  # skip constrained run

Produces:
  - project/results/pysr_constrained/       (constrained Pareto frontier)
  - project/results/pysr_unconstrained/     (unconstrained Pareto frontier)
  - project/results/pysr_summary.json       (best equations + RMSE)
  - project/figures/pysr_pareto.png         (Pareto frontier comparison)
  - project/figures/pysr_pred_vs_obs.png    (predicted vs observed)
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# ── Parse args ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Smoke test with 5 iterations")
parser.add_argument("--unconstrained-only", action="store_true")
parser.add_argument("--constrained-only", action="store_true")
args = parser.parse_args()

NITER = 5 if args.test else 200
NPOP = 5 if args.test else 30
POP_SIZE = 20 if args.test else 50
print(f"PySR config: niterations={NITER}, populations={NPOP}, population_size={POP_SIZE}")

# ── Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))

# Filter to valid gyrochronology domain
valid = (df["BPRP_0"] > 0.5) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
print(f"Loaded {len(df)} stars (valid domain: color>0.5, age>10 Myr)")

ages = df["fiducial_age_Myr"].values
colors = df["BPRP_0"].values
periods = df["Prot"].values

# ── Compute inverse-frequency age weights ───────────────────────────────────

log_ages = np.log10(ages)
age_bins = np.arange(log_ages.min(), log_ages.max() + 0.5, 0.5)  # 0.5-dex bins
bin_indices = np.digitize(log_ages, age_bins)
bin_counts = np.bincount(bin_indices, minlength=len(age_bins) + 1)
sample_weights = 1.0 / bin_counts[bin_indices]
sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

print(f"Age weight range: {sample_weights.min():.3f} – {sample_weights.max():.3f}")

X = np.column_stack([ages, colors])
y = periods

# ── Physics-constrained loss function (Julia) ───────────────────────────────

physics_loss = """
function physics_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end

    # Weighted MSE — custom loss_function does NOT auto-apply dataset.weights
    diffs = prediction .- dataset.y
    if dataset.weights !== nothing
        mse = sum(dataset.weights .* diffs .^ 2) / sum(dataset.weights)
    else
        mse = sum(diffs .^ 2) / length(diffs)
    end

    # Penalty: negative periods
    neg_penalty = sum(max.(0, -prediction)) / length(prediction)

    # Penalty: non-monotonic with age, checked within 0.1-mag color bins.
    # Only applied in the regime where monotonic spin-down is expected:
    #   ages 100–1000 Myr, colors 0.6–1.1 (late F through early K).
    ages = dataset.X[1, :]
    colors = dataset.X[2, :]
    mono_violations = zero(L)
    for color_lo in 0.6:0.1:1.0
        color_hi = color_lo + L(0.1)
        mask = (ages .> 100.0) .& (ages .< 1000.0) .& (colors .>= color_lo) .& (colors .< color_hi)
        if sum(mask) < 5
            continue
        end
        bin_ages = ages[mask]
        bin_pred = prediction[mask]
        sorted_idx = sortperm(bin_ages)
        sorted_pred = bin_pred[sorted_idx]
        mono_violations += sum(max.(0, sorted_pred[1:end-1] .- sorted_pred[2:end]))
    end

    return mse + 10.0 * neg_penalty + 1.0 * mono_violations
end
"""

# Unconstrained: weighted MSE only (no physics penalties)
unconstrained_loss = """
function unconstrained_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    diffs = prediction .- dataset.y
    if dataset.weights !== nothing
        return sum(dataset.weights .* diffs .^ 2) / sum(dataset.weights)
    else
        return sum(diffs .^ 2) / length(diffs)
    end
end
"""

# ── Shared PySR config ──────────────────────────────────────────────────────

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
    # Constrain pow: allow arbitrary base (-1) but only simple powers like
    # a constant or single variable (1). Prevents pow(age, color).
    constraints={"pow": (-1, 1)},
    progress=True,
    deterministic=False,
    parallelism="multithreading",
)

# ── Run PySR ────────────────────────────────────────────────────────────────

from pysr import PySRRegressor

results_summary = {}

# --- Constrained run ---
if not args.unconstrained_only:
    print("\n" + "=" * 60)
    print("CONSTRAINED RUN (physics penalties)")
    print("=" * 60)

    outdir_c = os.path.join(RESULTS, "pysr_constrained")
    os.makedirs(outdir_c, exist_ok=True)

    model_c = PySRRegressor(
        loss_function=physics_loss,
        output_directory=outdir_c,
        **shared_kwargs,
    )
    model_c.fit(X, y, variable_names=["age", "color"], weights=sample_weights)

    print("\n--- Constrained Pareto frontier ---")
    print(model_c.equations_[["complexity", "loss", "equation"]].to_string())

    best_c = model_c.get_best()
    pred_c = model_c.predict(X)
    rmse_c = float(np.sqrt(np.mean((y - pred_c) ** 2)))
    print(f"\nBest constrained equation: {best_c.equation}")
    print(f"Constrained RMSE: {rmse_c:.3f} days")

    model_c.equations_.to_csv(os.path.join(outdir_c, "pareto_constrained.csv"), index=False)

    results_summary["constrained"] = {
        "best_equation": str(best_c.equation),
        "best_loss": float(best_c.loss),
        "best_complexity": int(best_c.complexity),
        "rmse": rmse_c,
        "n_pareto": len(model_c.equations_),
    }

# --- Unconstrained run ---
if not args.constrained_only:
    print("\n" + "=" * 60)
    print("UNCONSTRAINED RUN (weighted MSE only)")
    print("=" * 60)

    outdir_u = os.path.join(RESULTS, "pysr_unconstrained")
    os.makedirs(outdir_u, exist_ok=True)

    model_u = PySRRegressor(
        loss_function=unconstrained_loss,
        output_directory=outdir_u,
        **shared_kwargs,
    )
    model_u.fit(X, y, variable_names=["age", "color"], weights=sample_weights)

    print("\n--- Unconstrained Pareto frontier ---")
    print(model_u.equations_[["complexity", "loss", "equation"]].to_string())

    best_u = model_u.get_best()
    pred_u = model_u.predict(X)
    rmse_u = float(np.sqrt(np.mean((y - pred_u) ** 2)))
    print(f"\nBest unconstrained equation: {best_u.equation}")
    print(f"Unconstrained RMSE: {rmse_u:.3f} days")

    model_u.equations_.to_csv(os.path.join(outdir_u, "pareto_unconstrained.csv"), index=False)

    results_summary["unconstrained"] = {
        "best_equation": str(best_u.equation),
        "best_loss": float(best_u.loss),
        "best_complexity": int(best_u.complexity),
        "rmse": rmse_u,
        "n_pareto": len(model_u.equations_),
    }

# ── Save summary ────────────────────────────────────────────────────────────

with open(os.path.join(RESULTS, "pysr_summary.json"), "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved: pysr_summary.json")

# ── Figures ─────────────────────────────────────────────────────────────────

# Load baseline results for comparison
baseline_path = os.path.join(RESULTS, "baseline_params.json")
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baselines = json.load(f)
else:
    baselines = {}

# --- Pareto frontier ---
fig, ax = plt.subplots(figsize=(10, 6))

if "constrained" in results_summary:
    eq_c = pd.read_csv(os.path.join(RESULTS, "pysr_constrained", "pareto_constrained.csv"))
    ax.plot(eq_c["complexity"], eq_c["loss"], "o-", label="PySR constrained", color="#1f77b4", markersize=6)

if "unconstrained" in results_summary:
    eq_u = pd.read_csv(os.path.join(RESULTS, "pysr_unconstrained", "pareto_unconstrained.csv"))
    ax.plot(eq_u["complexity"], eq_u["loss"], "s-", label="PySR unconstrained", color="#ff7f0e", markersize=6)

# Mark baselines as horizontal lines
baseline_colors = {"Barnes2007": "#2ca02c", "Angus2019": "#d62728", "DeadSimple_CurveFit": "#9467bd"}
for name, color in baseline_colors.items():
    if name in baselines:
        rmse = baselines[name]["rmse_train"]
        n_params = baselines[name]["n_params"]
        ax.axhline(rmse**2, color=color, linestyle="--", alpha=0.6, label=f"{name} (MSE={rmse**2:.1f})")

ax.set_xlabel("Complexity")
ax.set_ylabel("Loss")
ax.set_title("Pareto frontier: complexity vs loss")
ax.legend(fontsize=8)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "pysr_pareto.png"), dpi=150)
print(f"Saved: pysr_pareto.png")
plt.close(fig)

# --- Predicted vs observed ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, label, outdir in [
    (axes[0], "Constrained", os.path.join(RESULTS, "pysr_constrained")),
    (axes[1], "Unconstrained", os.path.join(RESULTS, "pysr_unconstrained")),
]:
    if label.lower() not in results_summary:
        ax.set_visible(False)
        continue
    r = results_summary[label.lower()]
    # Re-predict using the model if available, otherwise note
    try:
        model_ref = model_c if label == "Constrained" else model_u
        pred = model_ref.predict(X)
    except NameError:
        ax.text(0.5, 0.5, f"{label}\n(not run)", transform=ax.transAxes, ha="center")
        continue

    ax.scatter(y, pred, s=1, alpha=0.3, c="steelblue")
    lims = [0, min(50, max(y.max(), pred.max()) * 1.05)]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"{label}\n{r['best_equation'][:60]}\nRMSE={r['rmse']:.2f} d", fontsize=9)
    ax.set_xlabel("Observed $P_{\\rm rot}$ (days)")
    ax.set_ylabel("Predicted $P_{\\rm rot}$ (days)")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "pysr_pred_vs_obs.png"), dpi=150)
print(f"Saved: pysr_pred_vs_obs.png")
plt.close(fig)

# ── Final summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 4 SUMMARY")
print("=" * 60)
for label, r in results_summary.items():
    print(f"  {label:15s}  RMSE={r['rmse']:.3f} d  complexity={r['best_complexity']}  "
          f"n_pareto={r['n_pareto']}")
    print(f"                  {r['best_equation']}")

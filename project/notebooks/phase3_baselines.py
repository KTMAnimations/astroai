"""
Phase 3: Classical gyrochronology baselines.

Fits four baseline models to the slow-sequence-filtered data and evaluates
each with leave-one-cluster-out (LOO) cross-validation.

Models:
  1. Barnes 2007:           P = a * (color - c0)^b * age^n
  2. Mamajek-Hillenbrand:   P = a * (color - c0)^b * age^n   (separate fit)
  3. Angus 2019:            log10(P) = cA*log10(t) + poly(log10(color), deg=4)
  4. Dead-simple curve_fit: Same form as Barnes, 4 free params

Barnes and MH have the same functional form but different initial guesses,
so the optimizer may find different local minima. The dead-simple baseline
is identical in form to Barnes — it exists to confirm that scipy.optimize
can match PySR if the true form is a power law.

Produces:
  - project/results/baseline_params.json
  - project/results/baseline_loo_results.csv
  - project/figures/baselines_pred_vs_obs.png
  - project/figures/baselines_loo_by_cluster.png
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# ── Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
print(f"Loaded {len(df)} slow-sequence stars from {df['cluster'].nunique()} clusters")

ages = df["fiducial_age_Myr"].values
colors = df["BPRP_0"].values
periods = df["Prot"].values
clusters = df["cluster"].values

# ── Cluster-level weights ─────────────────────────────────────────────────
# Equal total weight per cluster, divided among members. This gives each
# cluster one effective vote regardless of membership count — matching the
# weighting used in phase4_pysr_search.py (lines 74-76).
# For curve_fit, sigma = 1/sqrt(weight) because it minimizes sum((r/sigma)^2).

cluster_counts = df.groupby("cluster")["Prot"].transform("count").values.astype(float)
sample_weights = 1.0 / cluster_counts
sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

print(f"Cluster weight range: {sample_weights.min():.3f} – {sample_weights.max():.3f}")

# ── Model definitions ──────────────────────────────────────────────────────


def barnes_model(X, a, b, c0, n):
    """P = a * (color - c0)^b * age^n"""
    age, color = X
    dc = np.clip(color - c0, 1e-6, None)
    return a * dc**b * age**n


def angus_model(X, cA, c0, c1, c2, c3, c4):
    """log10(P) = cA*log10(age) + c0 + c1*lc + c2*lc^2 + c3*lc^3 + c4*lc^4
    where lc = log10(color)
    """
    age, color = X
    log_age = np.log10(np.clip(age, 1e-3, None))
    log_color = np.log10(np.clip(color, 1e-3, None))
    log_p = (
        cA * log_age
        + c0
        + c1 * log_color
        + c2 * log_color**2
        + c3 * log_color**3
        + c4 * log_color**4
    )
    return 10.0**log_p


# ── Fitting helper ─────────────────────────────────────────────────────────


def fit_model(model_func, X, y, p0, bounds, label, weights=None):
    """Fit a model with optional cluster-level weighting.

    Parameters
    ----------
    weights : array-like or None
        Per-point importance weights (higher = more important).
        Converted to sigma = 1/sqrt(weight) for curve_fit.
    """
    try:
        sigma = None
        if weights is not None:
            sigma = 1.0 / np.sqrt(np.clip(weights, 1e-10, None))
        popt, pcov = curve_fit(
            model_func, X, y, p0=p0, bounds=bounds, maxfev=20000,
            sigma=sigma, absolute_sigma=False,
        )
        y_pred = model_func(X, *popt)
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        return popt, rmse, mae, y_pred
    except Exception as e:
        print(f"  WARNING: {label} fit failed: {e}")
        return None, np.inf, np.inf, None


# ── Define models with initial guesses and bounds ──────────────────────────

# Filter to valid domain: color > 0.5 (above Kraft break) and age > 10 Myr
# to avoid PMS complications. This matches where gyrochronology is expected
# to work.
valid = (colors > 0.5) & (colors < 2.0) & (ages > 10.0) & (periods > 0.1)
X_valid = (ages[valid], colors[valid])
y_valid = periods[valid]
w_valid = sample_weights[valid]
print(f"Valid domain (color>0.5, age>10 Myr): {valid.sum()} / {len(df)} stars")

# NOTE: Barnes, MH, and DeadSimple share the same functional form
# (P = a*(color-c0)^b * age^n). They converge to nearly identical parameters
# from different starting points (RMSE within 0.005 d), confirming a single
# global minimum. Only Barnes2007 and Angus2019 are reported in the paper.
# MH and DeadSimple are retained as convergence validation.
models = {
    "Barnes2007": {
        "func": barnes_model,
        # a, b, c0, n — Barnes original: a=0.7725, b=0.601, c0=0.4(B-V), n=0.5189
        # c0 shifted for BP-RP: ~0.4 in B-V maps to roughly 0.4-0.5 in BP-RP
        "p0": [0.5, 0.6, 0.4, 0.52],
        "bounds": ([0.01, 0.01, -0.5, 0.1], [50.0, 3.0, 1.5, 1.5]),
        "n_params": 4,
    },
    "MamajekHillenbrand2008": {
        "func": barnes_model,  # Same functional form, different starting point
        # MH2008: a=0.407, b=0.325, c0=0.495(B-V), n=0.566
        "p0": [0.3, 0.35, 0.5, 0.57],
        "bounds": ([0.01, 0.01, -0.5, 0.1], [50.0, 3.0, 1.5, 1.5]),
        "n_params": 4,
    },
    "Angus2019": {
        "func": angus_model,
        # cA ~ 0.65, polynomial coefficients from the paper
        # Bounds widened from ±20 to ±100 for c1-c4 — previous c4=-20.0 was
        # hitting the lower bound, indicating a pathological fit (checklist A3).
        "p0": [0.55, 0.3, 1.5, 0.5, -0.3, 0.1],
        "bounds": ([0.0, -100, -100, -100, -100, -100], [2.0, 100, 100, 100, 100, 100]),
        "n_params": 6,
    },
    "DeadSimple_CurveFit": {
        "func": barnes_model,
        # Deliberately naive starting point — let the optimizer find it
        "p0": [1.0, 0.5, 0.4, 0.5],
        "bounds": ([0.001, 0.01, -1.0, 0.01], [100.0, 5.0, 2.0, 2.0]),
        "n_params": 4,
    },
}

# ── Fit all models on the full training set ────────────────────────────────

print("\n=== Full-dataset fits ===")
results = {}

for name, spec in models.items():
    popt, rmse, mae, y_pred = fit_model(
        spec["func"], X_valid, y_valid, spec["p0"], spec["bounds"], name,
        weights=w_valid,
    )
    results[name] = {
        "params": popt.tolist() if popt is not None else None,
        "rmse_train": float(rmse),
        "mae_train": float(mae),
        "n_params": spec["n_params"],
    }
    if popt is not None:
        param_names = {
            "barnes_model": ["a", "b", "c0", "n"],
            "angus_model": ["cA", "c0", "c1", "c2", "c3", "c4"],
        }
        pnames = param_names.get(spec["func"].__name__, [f"p{i}" for i in range(len(popt))])
        param_str = ", ".join(f"{pn}={pv:.4f}" for pn, pv in zip(pnames, popt))
        print(f"  {name:30s}  RMSE={rmse:.3f} d  MAE={mae:.3f} d  params: {param_str}")
    else:
        print(f"  {name:30s}  FAILED")

# ── Leave-one-cluster-out cross-validation ─────────────────────────────────

print("\n=== Leave-one-cluster-out cross-validation ===")
unique_clusters = sorted(df["cluster"].unique())
loo_rows = []

for held_out in unique_clusters:
    train_mask = valid & (clusters != held_out)
    test_mask = valid & (clusters == held_out)

    if test_mask.sum() < 3:
        continue

    X_train = (ages[train_mask], colors[train_mask])
    y_train = periods[train_mask]
    # Recompute cluster weights for the training subset (held-out cluster removed)
    train_clusters = clusters[train_mask]
    train_cluster_counts = pd.Series(train_clusters).groupby(train_clusters).transform("count").values.astype(float)
    w_train = 1.0 / train_cluster_counts
    w_train = w_train * len(w_train) / w_train.sum()

    X_test = (ages[test_mask], colors[test_mask])
    y_test = periods[test_mask]
    test_age = df.loc[test_mask, "fiducial_age_Myr"].median()

    row = {"cluster": held_out, "age_Myr": test_age, "n_test": int(test_mask.sum())}

    for name, spec in models.items():
        popt, _, _, _ = fit_model(
            spec["func"], X_train, y_train, spec["p0"], spec["bounds"], name,
            weights=w_train,
        )
        if popt is not None:
            try:
                y_pred = spec["func"](X_test, *popt)
                resid = y_test - y_pred
                rmse = float(np.sqrt(np.mean(resid**2)))
                mae = float(np.mean(np.abs(resid)))
            except Exception:
                rmse = mae = np.nan
        else:
            rmse = mae = np.nan

        row[f"{name}_rmse"] = rmse
        row[f"{name}_mae"] = mae

    loo_rows.append(row)

loo_df = pd.DataFrame(loo_rows)

# Summary statistics
print(f"\n{'Model':30s}  {'Mean RMSE':>10s}  {'Std RMSE':>10s}  {'Worst cluster':>20s}")
print("-" * 80)
for name in models:
    col = f"{name}_rmse"
    mean_rmse = loo_df[col].mean()
    std_rmse = loo_df[col].std()
    worst_idx = loo_df[col].idxmax()
    worst_cluster = loo_df.loc[worst_idx, "cluster"]
    worst_rmse = loo_df.loc[worst_idx, col]
    print(f"  {name:30s}  {mean_rmse:10.3f}  {std_rmse:10.3f}  {worst_cluster} ({worst_rmse:.2f})")

# ── Save results ───────────────────────────────────────────────────────────

with open(os.path.join(RESULTS, "baseline_params.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: baseline_params.json")

loo_df.to_csv(os.path.join(RESULTS, "baseline_loo_results.csv"), index=False)
print(f"Saved: baseline_loo_results.csv")

# ── Figure 1: Predicted vs observed (full training set) ────────────────────

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)

for ax, (name, spec) in zip(axes, models.items()):
    popt = results[name]["params"]
    if popt is None:
        ax.set_title(f"{name}\n(fit failed)")
        continue
    y_pred = spec["func"](X_valid, *popt)
    ax.scatter(y_valid, y_pred, s=1, alpha=0.3, c="steelblue")
    lims = [0, min(50, max(y_valid.max(), y_pred.max()) * 1.05)]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"{name}\nRMSE={results[name]['rmse_train']:.2f} d", fontsize=10)
    ax.set_xlabel("Observed $P_{\\rm rot}$ (days)")

axes[0].set_ylabel("Predicted $P_{\\rm rot}$ (days)")
fig.suptitle("Baseline models — predicted vs observed (training set)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "baselines_pred_vs_obs.png"), dpi=150, bbox_inches="tight")
print(f"Saved: baselines_pred_vs_obs.png")
plt.close(fig)

# ── Figure 2: LOO RMSE by cluster age ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(models.keys())
markers = ["o", "s", "^", "D"]
colors_plot = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for name, marker, color in zip(model_names, markers, colors_plot):
    col = f"{name}_rmse"
    ax.scatter(loo_df["age_Myr"], loo_df[col], marker=marker, c=color,
               s=40, alpha=0.8, label=name, edgecolors="k", linewidths=0.3)

ax.set_xscale("log")
ax.set_xlabel("Cluster age (Myr)")
ax.set_ylabel("LOO RMSE (days)")
ax.set_title("Leave-one-cluster-out RMSE by cluster age")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "baselines_loo_by_cluster.png"), dpi=150)
print(f"Saved: baselines_loo_by_cluster.png")
plt.close(fig)

# ── Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)
for name in models:
    r = results[name]
    col = f"{name}_rmse"
    print(f"  {name:30s}  {r['n_params']} params  "
          f"train RMSE={r['rmse_train']:.3f}  "
          f"LOO RMSE={loo_df[col].mean():.3f} ± {loo_df[col].std():.3f}")

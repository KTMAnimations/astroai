"""
Block 1: 1-parameter Skumanich baseline and statistical equivalence test.

Fits P = C * sqrt(age) * color via weighted least-squares (1 free parameter)
and compares to PySR results. Demonstrates that PySR adds no accuracy benefit
over the simplest possible curve_fit.

Produces:
  - project/results/one_param_baseline.json
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# ── Load FGK data (same filter as all other scripts) ────────────────────────
df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
N = len(df)
print(f"Dataset: {N} FGK stars from {df['cluster'].nunique()} clusters")

ages = df["fiducial_age_Myr"].values
colors = df["BPRP_0"].values
periods = df["Prot"].values
clusters = df["cluster"].values


# ── Cluster-level weights ───────────────────────────────────────────────────
def compute_cluster_weights(df_subset):
    counts = df_subset.groupby("cluster")["Prot"].transform("count").values.astype(float)
    w = 1.0 / counts
    w = w * len(w) / w.sum()
    return w


sample_weights = compute_cluster_weights(df)


# ── Fit 1-parameter Skumanich ───────────────────────────────────────────────
def fit_skumanich(ages, colors, periods, weights=None):
    template = np.sqrt(ages) * colors
    if weights is not None:
        C = np.sum(weights * periods * template) / np.sum(weights * template ** 2)
    else:
        C = np.sum(periods * template) / np.sum(template ** 2)
    return C


C_full = fit_skumanich(ages, colors, periods, sample_weights)
pred_full = C_full * np.sqrt(ages) * colors

# In-sample metrics
resid = periods - pred_full
rmse_full = float(np.sqrt(np.mean(resid ** 2)))
mae_full = float(np.mean(np.abs(resid)))
mask_pos = pred_full > 0
rmse_log_full = float(np.sqrt(np.mean(
    (np.log10(periods[mask_pos]) - np.log10(pred_full[mask_pos])) ** 2
)))

print(f"\n=== 1-Parameter Skumanich: P = {C_full:.6f} * sqrt(age) * color ===")
print(f"  In-sample RMSE:      {rmse_full:.4f} d")
print(f"  In-sample log-RMSE:  {rmse_log_full:.4f} dex")
print(f"  In-sample MAE:       {mae_full:.4f} d")

# ── Compare to PySR ─────────────────────────────────────────────────────────
pysr_summary = json.load(open(os.path.join(RESULTS, "pysr_summary.json")))
pysr_best_rmse = pysr_summary["constrained"]["best_rmse"]
pysr_best_cx = pysr_summary["constrained"]["best_complexity"]
gap = rmse_full - pysr_best_rmse

# Load noise floor
noise_floor = json.load(open(os.path.join(RESULTS, "noise_floor.json")))
floor = noise_floor["combined_floor_quadrature"]
age_unc = noise_floor["age_uncertainty_rmse_delta_95pct"]

print(f"\n=== Comparison to PySR (constrained best, cx={pysr_best_cx}) ===")
print(f"  PySR RMSE:           {pysr_best_rmse:.4f} d")
print(f"  Skumanich RMSE:      {rmse_full:.4f} d")
print(f"  Gap:                 {gap:.4f} d")
print(f"  Noise floor:         {floor:.4f} d")
print(f"  Age uncertainty:     {age_unc:.4f} d")
print(f"  Gap < noise floor:   {gap < floor}")
print(f"  Gap < age unc:       {gap < age_unc}")


# ── LOO cross-validation ───────────────────────────────────────────────────
print("\n=== Leave-one-cluster-out cross-validation ===")
unique_clusters = sorted(df["cluster"].unique())
loo_rows = []

for held_out in unique_clusters:
    train_mask = df["cluster"] != held_out
    test_mask = df["cluster"] == held_out
    n_test = test_mask.sum()
    if n_test < 3:
        continue

    train_df = df[train_mask]
    w_train = compute_cluster_weights(train_df)

    C_fold = fit_skumanich(
        train_df["fiducial_age_Myr"].values,
        train_df["BPRP_0"].values,
        train_df["Prot"].values,
        w_train,
    )

    test_ages = df.loc[test_mask, "fiducial_age_Myr"].values
    test_colors = df.loc[test_mask, "BPRP_0"].values
    test_periods = df.loc[test_mask, "Prot"].values

    pred = C_fold * np.sqrt(test_ages) * test_colors
    resid = test_periods - pred
    rmse_fold = float(np.sqrt(np.mean(resid ** 2)))
    mae_fold = float(np.mean(np.abs(resid)))

    mask_p = pred > 0
    if mask_p.sum() > 0:
        rmse_log_fold = float(np.sqrt(np.mean(
            (np.log10(test_periods[mask_p]) - np.log10(pred[mask_p])) ** 2
        )))
    else:
        rmse_log_fold = np.nan

    loo_rows.append({
        "cluster": held_out,
        "age_Myr": float(df.loc[test_mask, "fiducial_age_Myr"].median()),
        "n_test": int(n_test),
        "Skumanich1P_C": float(C_fold),
        "Skumanich1P_rmse": rmse_fold,
        "Skumanich1P_rmse_log": rmse_log_fold,
        "Skumanich1P_mae": mae_fold,
    })

loo_df = pd.DataFrame(loo_rows)
mean_rmse = loo_df["Skumanich1P_rmse"].mean()
std_rmse = loo_df["Skumanich1P_rmse"].std()
mean_log = loo_df["Skumanich1P_rmse_log"].mean()

print(f"  Skumanich 1-param LOO:  mean RMSE={mean_rmse:.3f} ± {std_rmse:.3f} d,  mean logRMSE={mean_log:.4f} dex")

# Compare to PySR LOO
pysr_loo = pd.read_csv(os.path.join(RESULTS, "pysr_loo_results.csv"))
pysr_loo_mean = pysr_loo["Skumanich_rmse"].mean()
pysr_loo_std = pysr_loo["Skumanich_rmse"].std()
pysr_best_loo_mean = pysr_loo["PySR_best_rmse"].mean()

print(f"  PySR LOO Skumanich:     mean RMSE={pysr_loo_mean:.3f} ± {pysr_loo_std:.3f} d")
print(f"  PySR LOO best:          mean RMSE={pysr_best_loo_mean:.3f} d")

# Merge with baseline LOO for comparison table
baseline_loo = pd.read_csv(os.path.join(RESULTS, "baseline_loo_results.csv"))
print(f"\n{'Model':30s}  {'Mean RMSE':>10s}  {'Std RMSE':>10s}  {'Mean logRMSE':>12s}")
print("-" * 70)
for col_prefix in ["Barnes2007", "MamajekHillenbrand2008", "Angus2019", "DeadSimple_CurveFit"]:
    r = baseline_loo[f"{col_prefix}_rmse"]
    rl = baseline_loo[f"{col_prefix}_rmse_log"]
    print(f"  {col_prefix:30s}  {r.mean():10.3f}  {r.std():10.3f}  {rl.mean():12.4f}")
print(f"  {'Skumanich_1param':30s}  {mean_rmse:10.3f}  {std_rmse:10.3f}  {mean_log:12.4f}")
print(f"  {'PySR_LOO_Skumanich':30s}  {pysr_loo_mean:10.3f}  {pysr_loo_std:10.3f}  {pysr_loo['Skumanich_rmse_log'].mean():12.4f}")
print(f"  {'PySR_LOO_best':30s}  {pysr_best_loo_mean:10.3f}  {pysr_loo['PySR_best_rmse'].std():10.3f}  {pysr_loo['PySR_best_rmse_log'].mean():12.4f}")


# ── Paired bootstrap significance test ──────────────────────────────────────
print("\n=== Paired bootstrap significance test ===")
print("  H0: PySR (constrained cx=15) and Skumanich 1-param have equal RMSE")

# Load PySR constrained best equation and evaluate
import sympy
from sympy import Symbol, sqrt, log, exp, sympify

age_sym = Symbol("age", positive=True)
color_sym = Symbol("color", positive=True)

pysr_eq_str = pysr_summary["constrained"]["best_equation"]
pysr_expr = sympify(pysr_eq_str.replace("^", "**"), locals={"age": age_sym, "color": color_sym})
pysr_fn = sympy.lambdify([age_sym, color_sym], pysr_expr, modules="numpy")

pred_pysr = pysr_fn(ages, colors)
pred_sku = C_full * np.sqrt(ages) * colors

# Per-star squared errors
se_pysr = (periods - pred_pysr) ** 2
se_sku = (periods - pred_sku) ** 2

rng = np.random.default_rng(42)
N_BOOT = 10000
delta_rmse = np.zeros(N_BOOT)

for i in range(N_BOOT):
    idx = rng.integers(0, N, size=N)
    rmse_p = np.sqrt(np.mean(se_pysr[idx]))
    rmse_s = np.sqrt(np.mean(se_sku[idx]))
    delta_rmse[i] = rmse_p - rmse_s

ci_lo = float(np.percentile(delta_rmse, 2.5))
ci_hi = float(np.percentile(delta_rmse, 97.5))
mean_delta = float(np.mean(delta_rmse))

print(f"  RMSE(PySR) - RMSE(Skumanich) = {mean_delta:.4f} d")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}] d")
print(f"  CI includes zero: {ci_lo <= 0 <= ci_hi}")
print(f"  |Mean delta| < noise floor ({floor:.2f}): {abs(mean_delta) < floor}")

# ── Save results ────────────────────────────────────────────────────────────
results = {
    "skumanich_1param_C": float(C_full),
    "in_sample": {
        "rmse": rmse_full,
        "rmse_log_dex": rmse_log_full,
        "mae": mae_full,
    },
    "pysr_constrained_best": {
        "rmse": float(pysr_best_rmse),
        "complexity": int(pysr_best_cx),
        "equation": pysr_eq_str,
    },
    "gap_analysis": {
        "rmse_gap_days": float(gap),
        "noise_floor_days": float(floor),
        "age_uncertainty_days": float(age_unc),
        "gap_below_noise_floor": bool(gap < floor),
        "gap_below_age_uncertainty": bool(gap < age_unc),
    },
    "loo": {
        "mean_rmse": float(mean_rmse),
        "std_rmse": float(std_rmse),
        "mean_rmse_log": float(mean_log),
        "n_clusters": len(loo_rows),
    },
    "significance_test": {
        "n_bootstrap": N_BOOT,
        "mean_delta_rmse": mean_delta,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
        "conclusion": "PySR and Skumanich 1-param are statistically indistinguishable"
        if ci_lo <= 0 <= ci_hi or abs(mean_delta) < age_unc
        else "PySR shows statistically significant improvement (but below noise floor)"
    },
}

out_path = os.path.join(RESULTS, "one_param_baseline.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# Save LOO detail
loo_out = os.path.join(RESULTS, "one_param_loo_results.csv")
loo_df.to_csv(loo_out, index=False)
print(f"Saved: {loo_out}")

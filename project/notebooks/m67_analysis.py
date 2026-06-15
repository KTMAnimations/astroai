"""
Block 5.2: M67 analysis — weakened magnetic braking diagnostic.

M67 (3870 Myr) is the only cluster above 3 Gyr in the training data.
All models fail on it (RMSE ~30d). This script characterizes the failure
and connects it to weakened magnetic braking (WMB).

Key diagnostic: if M67 periods are SHORTER than Skumanich predicts,
stars are spinning faster than expected — consistent with WMB stalling
spin-down at some critical age/Rossby number.

Produces:
  - project/results/m67_analysis.json
  - project/figures/m67_pred_vs_obs.png
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from sympy import Symbol, sympify, lambdify

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)

m67 = df[df["cluster"] == "M67"].copy()
print(f"M67: {len(m67)} stars, age={m67['fiducial_age_Myr'].iloc[0]:.0f} Myr")

ages = m67["fiducial_age_Myr"].values
colors = m67["BPRP_0"].values
periods = m67["Prot"].values

# ── Load models ─────────────────────────────────────────────────────────────
# 1-param Skumanich
baseline = json.load(open(os.path.join(RESULTS, "one_param_baseline.json")))
C_sku = baseline["skumanich_1param_C"]

# Barnes 2007
params = json.load(open(os.path.join(RESULTS, "baseline_params.json")))
barnes_p = params["Barnes2007"]["params"]  # a, b, c0, n

# PySR constrained best
pysr_summary = json.load(open(os.path.join(RESULTS, "pysr_summary.json")))
pysr_eq_str = pysr_summary["constrained"]["best_equation"]
age_sym = Symbol("age", positive=True)
color_sym = Symbol("color", positive=True)
pysr_expr = sympify(pysr_eq_str.replace("^", "**"), locals={"age": age_sym, "color": color_sym})
pysr_fn = lambdify([age_sym, color_sym], pysr_expr, modules="numpy")


# ── Predictions ─────────────────────────────────────────────────────────────
def barnes_model(age, color, a, b, c0, n):
    return a * np.power(np.maximum(color - c0, 0.01), b) * np.power(age, n)


pred_sku = C_sku * np.sqrt(ages) * colors
pred_barnes = barnes_model(ages, colors, *barnes_p)
pred_pysr = pysr_fn(ages, colors)

models = {
    "Skumanich": pred_sku,
    "Barnes2007": pred_barnes,
    "PySR_best": pred_pysr,
}

print(f"\n{'Model':>15s}  {'RMSE':>8s}  {'MAE':>8s}  {'Mean resid':>10s}  {'Median resid':>12s}")
print("-" * 65)
for name, pred in models.items():
    resid = periods - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    print(f"{name:>15s}  {rmse:8.2f}  {mae:8.2f}  {resid.mean():10.2f}  {np.median(resid):12.2f}")


# ── Effective age diagnostic ────────────────────────────────────────────────
# If P = C * sqrt(age_eff) * color, then age_eff = (P / (C * color))^2
age_eff = (periods / (C_sku * colors)) ** 2
age_actual = ages[0]  # M67 age

ratio = age_eff / age_actual
mean_ratio = float(np.mean(ratio))
median_ratio = float(np.median(ratio))

print(f"\n=== Effective Age Diagnostic ===")
print(f"  M67 actual age: {age_actual:.0f} Myr")
print(f"  Mean effective age:   {np.mean(age_eff):.0f} Myr  (ratio: {mean_ratio:.3f})")
print(f"  Median effective age: {np.median(age_eff):.0f} Myr  (ratio: {median_ratio:.3f})")
print(f"  Stars spinning faster than Skumanich predicts: {np.sum(periods < pred_sku)}/{len(periods)} ({100*np.mean(periods < pred_sku):.0f}%)")

# Interpretation
if median_ratio < 0.8:
    interpretation = (
        f"M67 stars appear {(1 - median_ratio)*100:.0f}% younger than their actual age "
        f"based on rotation. This is consistent with weakened magnetic braking: "
        f"spin-down stalled at an effective age of ~{np.median(age_eff):.0f} Myr."
    )
elif median_ratio < 1.0:
    interpretation = (
        f"M67 stars appear moderately younger ({(1 - median_ratio)*100:.0f}%) than expected. "
        f"Marginal evidence for reduced braking efficiency."
    )
else:
    interpretation = "M67 stars rotate as expected or slower — no WMB evidence."

print(f"\n  Interpretation: {interpretation}")


# ── Color-dependent analysis ────────────────────────────────────────────────
print(f"\n=== Effective Age by Color Bin ===")
print(f"{'Color bin':>15s}  {'N':>4s}  {'Med age_eff':>12s}  {'Ratio':>6s}")
print("-" * 45)

color_bins = [(0.5, 0.8), (0.8, 1.1), (1.1, 1.4), (1.4, 2.0)]
color_results = []
for lo, hi in color_bins:
    mask = (colors >= lo) & (colors < hi)
    if mask.sum() < 3:
        continue
    ae = age_eff[mask]
    r = float(np.median(ae) / age_actual)
    print(f"  [{lo:.1f}, {hi:.1f})     {mask.sum():4d}  {np.median(ae):12.0f}  {r:6.3f}")
    color_results.append({
        "color_lo": lo, "color_hi": hi, "n_stars": int(mask.sum()),
        "median_age_eff": float(np.median(ae)), "ratio": r,
    })


# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Pred vs Obs
ax = axes[0]
for name, pred in models.items():
    ax.scatter(periods, pred, alpha=0.4, s=15, label=name)
lim = [0, max(periods.max(), pred_sku.max()) * 1.1]
ax.plot(lim, lim, "k--", alpha=0.5, label="1:1")
ax.set_xlabel("Observed P_rot (d)")
ax.set_ylabel("Predicted P_rot (d)")
ax.set_title("M67: Predicted vs Observed")
ax.legend(fontsize=8)
ax.set_xlim(lim)
ax.set_ylim(lim)

# Panel 2: Effective age vs color
ax = axes[1]
ax.scatter(colors, age_eff, alpha=0.4, s=15, c="steelblue")
ax.axhline(age_actual, color="red", ls="--", label=f"True age ({age_actual:.0f} Myr)")
ax.set_xlabel("(BP-RP)_0")
ax.set_ylabel("Effective gyro age (Myr)")
ax.set_title("M67: Effective Age from Skumanich Inversion")
ax.legend(fontsize=8)
ax.set_ylim(0, age_actual * 1.5)

plt.tight_layout()
fig_path = os.path.join(FIG, "m67_pred_vs_obs.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved: {fig_path}")


# ── Save results ────────────────────────────────────────────────────────────
results = {
    "n_stars": len(m67),
    "age_Myr": float(age_actual),
    "skumanich_C": float(C_sku),
    "rmse": {name: float(np.sqrt(np.mean((periods - pred) ** 2))) for name, pred in models.items()},
    "mean_residual": {name: float(np.mean(periods - pred)) for name, pred in models.items()},
    "effective_age": {
        "mean": float(np.mean(age_eff)),
        "median": float(np.median(age_eff)),
        "mean_ratio": mean_ratio,
        "median_ratio": median_ratio,
    },
    "fraction_faster_than_skumanich": float(np.mean(periods < pred_sku)),
    "color_dependence": color_results,
    "interpretation": interpretation,
}

out_path = os.path.join(RESULTS, "m67_analysis.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_path}")

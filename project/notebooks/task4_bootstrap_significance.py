"""
Task 4: Bootstrap significance test for RMSE differences.

10,000 bootstrap resamples to test whether PySR improves on Angus/Barnes.
Also computes RMSE in log10(P) space (standard gyrochronology metric).
"""

import json
import os

import numpy as np
import pandas as pd
import sympy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# ── Load restricted dataset ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
age = df["fiducial_age_Myr"].values
color = df["BPRP_0"].values
y = df["Prot"].values
N = len(df)
print(f"Dataset: {N} FGK stars")

# ── Load baseline parameters ─────────────────────────────────────────────
with open(os.path.join(RESULTS, "baseline_params.json")) as f:
    baselines = json.load(f)

# ── Evaluate equation via sympy ──────────────────────────────────────────
age_sym, color_sym = sympy.symbols("age color")


def eval_eq(expr_str, a, c):
    expr_str = expr_str.replace("^", "**")
    expr = sympy.sympify(expr_str, locals={"age": age_sym, "color": color_sym})
    f = sympy.lambdify((age_sym, color_sym), expr, modules=["numpy"])
    pred = f(a, c)
    if isinstance(pred, (int, float)):
        pred = np.full_like(a, pred, dtype=float)
    return np.asarray(pred, dtype=float)


# ── Compute predictions for all models ───────────────────────────────────

# Barnes refitted
a_b, b_b, c0_b, n_b = baselines["Barnes2007"]["params"]
dc = np.clip(color - c0_b, 1e-6, None)
pred_barnes = a_b * dc**b_b * age**n_b

# Angus refitted
cA, c0, c1, c2, c3, c4 = baselines["Angus2019"]["params"]
log_age = np.log10(np.clip(age, 1e-3, None))
log_color = np.log10(np.clip(color, 1e-3, None))
log_p = cA * log_age + c0 + c1*log_color + c2*log_color**2 + c3*log_color**3 + c4*log_color**4
pred_angus = 10.0**log_p

# PySR constrained cx=22
with open(os.path.join(RESULTS, "pysr_summary.json")) as f:
    pysr = json.load(f)

eq_cx22 = pysr["constrained"]["best_equation"]
pred_cx22 = eval_eq(eq_cx22, age, color)

# PySR constrained cx=14
eq_cx14 = pysr["constrained"]["representatives"]["cx_14"]["equation"]
pred_cx14 = eval_eq(eq_cx14, age, color)

# ChronoFlow mode (from phase5 output)
cf_path = os.path.join(DATA, "clean_slow_seq_with_residuals.csv")
if os.path.exists(cf_path):
    df_cf = pd.read_csv(cf_path)
    # Match by GaiaDR3_ID since the file may have different row order
    cf_merged = df.merge(df_cf[["GaiaDR3_ID", "chronoflow_pred_mode"]], on="GaiaDR3_ID", how="left")
    pred_cf = cf_merged["chronoflow_pred_mode"].values
    has_cf = np.isfinite(pred_cf).sum() > N * 0.5
else:
    has_cf = False

models = {
    "Barnes": pred_barnes,
    "Angus": pred_angus,
    "PySR_cx22": pred_cx22,
    "PySR_cx14": pred_cx14,
}
if has_cf:
    models["ChronoFlow"] = pred_cf

# ── Compute residuals ────────────────────────────────────────────────────
residuals = {}
for name, pred in models.items():
    residuals[name] = y - pred

# ── Linear-space RMSE ────────────────────────────────────────────────────
print(f"\n=== RMSE (linear P, days) ===")
rmse = {}
for name, resid in residuals.items():
    mask = np.isfinite(resid)
    rmse[name] = np.sqrt(np.mean(resid[mask]**2))
    print(f"  {name:15s}: {rmse[name]:.3f} d")

# ── log10(P) RMSE ────────────────────────────────────────────────────────
print(f"\n=== RMSE (log10 P, dex) ===")
log_y = np.log10(np.clip(y, 1e-3, None))
rmse_log = {}
for name, pred in models.items():
    log_pred = np.log10(np.clip(pred, 1e-3, None))
    mask = np.isfinite(log_pred)
    rmse_log[name] = np.sqrt(np.mean((log_y[mask] - log_pred[mask])**2))
    print(f"  {name:15s}: {rmse_log[name]:.4f} dex")

# ── Bootstrap (10,000 resamples) ─────────────────────────────────────────
print(f"\n=== Bootstrap significance test (10,000 resamples) ===")

rng = np.random.default_rng(42)
n_boot = 10000

# Pairwise comparisons
pairs = [
    ("PySR_cx22", "Angus"),
    ("PySR_cx22", "Barnes"),
    ("PySR_cx14", "Angus"),
]
if has_cf:
    pairs.extend([
        ("ChronoFlow", "Angus"),
        ("ChronoFlow", "PySR_cx22"),
    ])

boot_diffs = {pair: np.zeros(n_boot) for pair in pairs}
boot_diffs_log = {pair: np.zeros(n_boot) for pair in pairs}

for i in range(n_boot):
    idx = rng.integers(0, N, size=N)

    for (m1, m2) in pairs:
        r1 = residuals[m1][idx]
        r2 = residuals[m2][idx]
        mask1 = np.isfinite(r1)
        mask2 = np.isfinite(r2)
        rmse1 = np.sqrt(np.mean(r1[mask1]**2))
        rmse2 = np.sqrt(np.mean(r2[mask2]**2))
        boot_diffs[(m1, m2)][i] = rmse1 - rmse2

        # log space
        lp1 = np.log10(np.clip(models[m1][idx], 1e-3, None))
        lp2 = np.log10(np.clip(models[m2][idx], 1e-3, None))
        ly = log_y[idx]
        lr1 = ly - lp1
        lr2 = ly - lp2
        mask_l1 = np.isfinite(lr1)
        mask_l2 = np.isfinite(lr2)
        rl1 = np.sqrt(np.mean(lr1[mask_l1]**2))
        rl2 = np.sqrt(np.mean(lr2[mask_l2]**2))
        boot_diffs_log[(m1, m2)][i] = rl1 - rl2

# ── Print results ────────────────────────────────────────────────────────
print(f"\n{'Comparison':>30s}  {'Δ RMSE':>8s}  {'95% CI':>18s}  {'Signif':>6s}")
print("─" * 75)

for (m1, m2) in pairs:
    d = boot_diffs[(m1, m2)]
    mean_d = np.mean(d)
    ci_lo = np.percentile(d, 2.5)
    ci_hi = np.percentile(d, 97.5)
    sig = "YES" if ci_hi < 0 else ("YES*" if ci_lo > 0 else "NO")
    # m1 < m2 means m1 is better (lower RMSE)
    label = f"{m1} vs {m2}"
    print(f"  {label:>28s}  {mean_d:+8.3f}  [{ci_lo:+.3f}, {ci_hi:+.3f}]  {sig:>6s}")

print(f"\n{'Comparison (log10 P)':>30s}  {'Δ RMSE':>8s}  {'95% CI':>18s}  {'Signif':>6s}")
print("─" * 75)

for (m1, m2) in pairs:
    d = boot_diffs_log[(m1, m2)]
    mean_d = np.mean(d)
    ci_lo = np.percentile(d, 2.5)
    ci_hi = np.percentile(d, 97.5)
    sig = "YES" if ci_hi < 0 else ("YES*" if ci_lo > 0 else "NO")
    label = f"{m1} vs {m2}"
    print(f"  {label:>28s}  {mean_d:+8.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig:>6s}")

print(f"\n  Legend: YES = m1 significantly better, YES* = m1 significantly worse, NO = no significant difference")
print(f"  Significance: 95% CI excludes zero")

"""
Task 3: Wide-grid physical sanity check on all PySR equations.

Evaluates every Pareto equation on a 50×30 grid of (age, color) values
and checks for negative predictions, NaN/Inf, and monotonicity violations.
"""

import os

import numpy as np
import pandas as pd
import sympy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# ── Equation evaluator ──────────────────────────────────────────────────
age_sym, color_sym = sympy.symbols("age color")


def safe_eval(expr_str, age_vals, color_vals):
    expr_str = expr_str.replace("^", "**")
    if "PySRFunction" in expr_str:
        expr_str = expr_str.split("=>", 1)[1].rstrip(")")
    expr = sympy.sympify(expr_str, locals={"age": age_sym, "color": color_sym})
    f = sympy.lambdify((age_sym, color_sym), expr, modules=["numpy"])
    pred = f(age_vals, color_vals)
    if isinstance(pred, (int, float)):
        pred = np.full_like(age_vals, pred, dtype=float)
    return np.asarray(pred, dtype=float)


# ── Build evaluation grid ────────────────────────────────────────────────
grid_ages = np.geomspace(50, 5000, 50)
grid_colors = np.linspace(0.5, 2.0, 30)
age_mesh, color_mesh = np.meshgrid(grid_ages, grid_colors)
age_flat = age_mesh.ravel()
color_flat = color_mesh.ravel()
n_grid = len(age_flat)

# Solar point
SOLAR_AGE = np.array([4567.0])
SOLAR_COLOR = np.array([0.82])

print(f"Grid: {len(grid_ages)} ages × {len(grid_colors)} colors = {n_grid} points")
print(f"Age range: {grid_ages[0]:.0f} – {grid_ages[-1]:.0f} Myr")
print(f"Color range: {grid_colors[0]:.2f} – {grid_colors[-1]:.2f}")

# ── Load Pareto frontiers ────────────────────────────────────────────────
pareto_c = pd.read_csv(os.path.join(RESULTS, "pysr_constrained", "pareto_constrained.csv"))
pareto_u = pd.read_csv(os.path.join(RESULTS, "pysr_unconstrained", "pareto_unconstrained.csv"))

# ── Also load restricted data for RMSE ───────────────────────────────────
df = pd.read_csv(os.path.join(ROOT, "data", "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
data_age = df["fiducial_age_Myr"].values
data_color = df["BPRP_0"].values
data_y = df["Prot"].values

# ── Evaluate each equation ───────────────────────────────────────────────
results_rows = []

for run_label, pareto_df in [("C", pareto_c), ("U", pareto_u)]:
    for _, row in pareto_df.iterrows():
        cx = int(row["complexity"])
        eq_str = row["sympy_format"]
        eq_display = str(row["equation"])

        try:
            # Grid evaluation
            pred_grid = safe_eval(eq_str, age_flat, color_flat)
            n_negative = int(np.sum(pred_grid < 0))
            n_nan = int(np.sum(~np.isfinite(pred_grid)))

            # Monotonicity with age (at each fixed color)
            mono_age_violations = 0
            for j, c in enumerate(grid_colors):
                preds_at_c = safe_eval(eq_str, grid_ages, np.full_like(grid_ages, c))
                diffs = np.diff(preds_at_c)
                mono_age_violations += int(np.sum(diffs < 0))

            # Monotonicity with color at fixed age (only for age > 200 Myr)
            mono_color_violations = 0
            for i, a in enumerate(grid_ages):
                if a < 200:
                    continue
                preds_at_a = safe_eval(eq_str, np.full_like(grid_colors, a), grid_colors)
                diffs = np.diff(preds_at_a)
                mono_color_violations += int(np.sum(diffs < 0))

            # Solar point
            p_sun = safe_eval(eq_str, SOLAR_AGE, SOLAR_COLOR)[0]

            # RMSE on restricted data
            pred_data = safe_eval(eq_str, data_age, data_color)
            mask = np.isfinite(pred_data)
            rmse = np.sqrt(np.mean((data_y[mask] - pred_data[mask])**2)) if mask.sum() > 0 else np.inf

            results_rows.append({
                "run": run_label,
                "cx": cx,
                "rmse": rmse,
                "n_neg": n_negative,
                "n_nan": n_nan,
                "mono_age": mono_age_violations,
                "mono_color": mono_color_violations,
                "p_sun": p_sun,
                "equation": eq_display[:55],
            })
        except Exception as e:
            results_rows.append({
                "run": run_label,
                "cx": cx,
                "rmse": np.inf,
                "n_neg": -1,
                "n_nan": -1,
                "mono_age": -1,
                "mono_color": -1,
                "p_sun": np.nan,
                "equation": f"EVAL FAILED: {e}",
            })

# ── Print results table ──────────────────────────────────────────────────
print(f"\n{'Run':>3s} {'Cx':>3s} {'RMSE':>7s} {'N_neg':>6s} {'N_NaN':>6s} "
      f"{'Mono↑A':>7s} {'Mono↑C':>7s} {'P_sun':>7s}  Equation")
print("─" * 120)

flagged = []
for r in results_rows:
    flag = ""
    if r["n_neg"] > 0:
        flag += "!"
    if r["mono_age"] > 0:
        flag += "A"
    if r["mono_color"] > 0:
        flag += "C"
    if r["n_nan"] > 0:
        flag += "N"

    print(f" {r['run']:>2s} {r['cx']:3d} {r['rmse']:7.2f} {r['n_neg']:6d} {r['n_nan']:6d} "
          f"{r['mono_age']:7d} {r['mono_color']:7d} {r['p_sun']:7.1f}  {flag:4s} {r['equation']}")

    if flag and r["run"] == "C" and r["cx"] in (22,):
        flagged.append(("BEST CONSTRAINED cx=22", r, flag))
    if flag and r["run"] == "U" and r["cx"] in (18, 20):
        flagged.append((f"BEST UNCONSTRAINED cx={r['cx']}", r, flag))

# ── Flag best equations ──────────────────────────────────────────────────
if flagged:
    print(f"\n⚠ FLAGGED BEST EQUATIONS:")
    for label, r, flag in flagged:
        print(f"  {label}: flags={flag}")
        if "!" in flag:
            print(f"    → {r['n_neg']} negative predictions on the grid")
        if "A" in flag:
            print(f"    → {r['mono_age']} age monotonicity violations")
        if "C" in flag:
            print(f"    → {r['mono_color']} color monotonicity violations")
else:
    print(f"\n✓ No violations in best constrained (cx=22) or best unconstrained (cx=18/20)")

# ── Summary statistics ───────────────────────────────────────────────────
print(f"\n=== Summary ===")
n_total = len(results_rows)
n_with_neg = sum(1 for r in results_rows if r["n_neg"] > 0)
n_with_mono_age = sum(1 for r in results_rows if r["mono_age"] > 0)
n_with_mono_color = sum(1 for r in results_rows if r["mono_color"] > 0)
print(f"  Total equations checked: {n_total}")
print(f"  With negative predictions: {n_with_neg}")
print(f"  With age monotonicity violations: {n_with_mono_age}")
print(f"  With color monotonicity violations: {n_with_mono_color}")

# Solar predictions for all models
print(f"\n  Solar predictions (expected ≈25.4 d):")
for r in results_rows:
    if r["run"] == "C" and r["cx"] in (5, 14, 22):
        print(f"    Constrained cx={r['cx']:2d}: P_sun={r['p_sun']:6.1f} d")
    if r["run"] == "U" and r["cx"] in (5, 18, 20):
        print(f"    Unconstr.   cx={r['cx']:2d}: P_sun={r['p_sun']:6.1f} d")

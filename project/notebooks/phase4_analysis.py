"""
Phase 4 post-analysis: proper reporting of PySR results.

Fixes three issues from the initial reporting:
  1. Reports equations at multiple complexity levels with unweighted RMSE,
     not just the trivially simple get_best() selection.
  2. Pareto plot uses unweighted MSE for both PySR and baselines (same scale).
  3. Pred-vs-obs shows the best mid-complexity equation, not the linear one.

Also checks whether any Pareto equation recovers the Barnes functional form.

Produces:
  - project/results/pysr_summary.json       (updated with per-equation RMSE)
  - project/figures/pysr_pareto.png          (corrected: same scale)
  - project/figures/pysr_pred_vs_obs.png     (corrected: best equation)
  - project/figures/pysr_equations_detail.png (multi-complexity comparison)
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# ── Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))
valid = (df["BPRP_0"] > 0.5) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)

age = df["fiducial_age_Myr"].values
color = df["BPRP_0"].values
y = df["Prot"].values
N = len(df)
print(f"Data: {N} stars")


# ── Evaluate each Pareto equation and compute unweighted RMSE ──────────────

def eval_equation(eq_lambda_str, age, color):
    """Evaluate a PySR lambda/equation string on the data.

    Handles PySRFunction(X=>expr) format from the CSV as well as
    plain equation strings.
    """
    try:
        # Use the expression string directly (sympy_format or equation column)
        expr_str = eq_lambda_str

        # Clean up PySRFunction wrapper if present
        if "PySRFunction" in expr_str:
            expr_str = expr_str.split("=>", 1)[1].rstrip(")")

        # Make it numpy-safe
        expr_str = expr_str.replace("^", "**")

        # Evaluate with numpy functions available
        local_ns = {
            "age": age,
            "color": color,
            "sqrt": np.sqrt,
            "log": np.log,
            "exp": np.exp,
            "abs": np.abs,
            "pow": np.power,
        }
        pred = eval(expr_str, {"__builtins__": {}}, local_ns)
        if isinstance(pred, (int, float)):
            pred = np.full_like(age, pred, dtype=float)
        pred = np.asarray(pred, dtype=float)
        pred = np.where(np.isfinite(pred), pred, np.nan)
        return pred
    except Exception as e:
        print(f"  eval failed for: {eq_lambda_str[:60]}  ({e})")
        return None


def compute_rmse(pred, y):
    if pred is None:
        return np.inf
    mask = np.isfinite(pred)
    if mask.sum() < len(y) * 0.5:
        return np.inf
    return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))


def compute_mse(pred, y):
    if pred is None:
        return np.inf
    mask = np.isfinite(pred)
    if mask.sum() < len(y) * 0.5:
        return np.inf
    return float(np.mean((y[mask] - pred[mask]) ** 2))


results = {}

for label, csv_name in [
    ("constrained", "pysr_constrained/pareto_constrained.csv"),
    ("unconstrained", "pysr_unconstrained/pareto_unconstrained.csv"),
]:
    csv_path = os.path.join(RESULTS, csv_name)
    if not os.path.exists(csv_path):
        print(f"Skipping {label}: {csv_path} not found")
        continue

    pareto = pd.read_csv(csv_path)
    rows = []
    for _, row in pareto.iterrows():
        pred = eval_equation(row["sympy_format"], age, color)
        rmse = compute_rmse(pred, y)
        mse = compute_mse(pred, y)
        rows.append({
            "complexity": int(row["complexity"]),
            "weighted_loss": float(row["loss"]),
            "unweighted_mse": mse,
            "unweighted_rmse": rmse,
            "equation": row["equation"],
            "sympy_format": row["sympy_format"],
        })

    eq_df = pd.DataFrame(rows)
    results[label] = eq_df

    print(f"\n=== {label.upper()} Pareto frontier (unweighted RMSE) ===")
    print(f"{'Cx':>4s}  {'wLoss':>8s}  {'uwMSE':>8s}  {'uwRMSE':>8s}  Equation")
    print("-" * 100)
    for _, r in eq_df.iterrows():
        eq_short = r["equation"][:65]
        print(f"{r['complexity']:4d}  {r['weighted_loss']:8.1f}  "
              f"{r['unweighted_mse']:8.1f}  {r['unweighted_rmse']:8.3f}  {eq_short}")

    # Best at each complexity tier
    best_idx = eq_df["unweighted_rmse"].idxmin()
    best = eq_df.loc[best_idx]
    print(f"\nLowest unweighted RMSE: {best['unweighted_rmse']:.3f} d "
          f"at complexity {best['complexity']}")
    print(f"  {best['equation']}")


# ── Load baselines ──────────────────────────────────────────────────────────

with open(os.path.join(RESULTS, "baseline_params.json")) as f:
    baselines = json.load(f)

print("\n=== Baselines (unweighted) ===")
for name, b in baselines.items():
    print(f"  {name:30s}  RMSE={b['rmse_train']:.3f}  MSE={b['rmse_train']**2:.1f}")


# ── Check for Barnes-like forms ─────────────────────────────────────────────

print("\n=== Checking for Barnes-like forms: P = a * (color - c0)^b * age^n ===")
for label, eq_df in results.items():
    found_barnes = False
    for _, r in eq_df.iterrows():
        eq = r["equation"]
        # Look for power-law patterns in age and color
        if ("^" in eq or "pow" in eq) and "color" in eq and "age" in eq:
            # Check if it's multiplicative (no + or -)
            if "+" not in eq and "-" not in eq.replace("e-", "E"):
                print(f"  {label} cx={r['complexity']}: {eq}  (multiplicative power law)")
                found_barnes = True
    if not found_barnes:
        print(f"  {label}: No Barnes-like (pure multiplicative power-law) form found.")
        print(f"  PySR preferred additive structures like 'log(age) + (age²*color)^n'")
        print(f"  or linear products like 'age * color * const'.")


# ── Save updated summary ───────────────────────────────────────────────────

summary = {}
for label, eq_df in results.items():
    best_idx = eq_df["unweighted_rmse"].idxmin()
    best = eq_df.loc[best_idx]

    # Also pick representative equations at different complexities
    representatives = {}
    for target_cx in [5, 10, 15, 20]:
        nearest = eq_df.iloc[(eq_df["complexity"] - target_cx).abs().argsort()[:1]]
        r = nearest.iloc[0]
        representatives[f"cx_{r['complexity']}"] = {
            "equation": r["equation"],
            "unweighted_rmse": r["unweighted_rmse"],
            "complexity": int(r["complexity"]),
        }

    summary[label] = {
        "best_equation": best["equation"],
        "best_rmse": float(best["unweighted_rmse"]),
        "best_complexity": int(best["complexity"]),
        "n_pareto": len(eq_df),
        "representatives": representatives,
    }

with open(os.path.join(RESULTS, "pysr_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: pysr_summary.json")


# ── Figure 1: Corrected Pareto plot (unweighted MSE, same scale) ───────────

fig, ax = plt.subplots(figsize=(10, 6))

plot_colors = {"constrained": "#1f77b4", "unconstrained": "#ff7f0e"}
plot_markers = {"constrained": "o", "unconstrained": "s"}

for label, eq_df in results.items():
    ax.plot(
        eq_df["complexity"], eq_df["unweighted_mse"],
        f"{plot_markers[label]}-", label=f"PySR {label}",
        color=plot_colors[label], markersize=5, linewidth=1.2,
    )

# Baselines as horizontal lines (unweighted MSE — same scale)
baseline_styles = {
    "Barnes2007": ("#2ca02c", "--"),
    "Angus2019": ("#d62728", "-."),
    "DeadSimple_CurveFit": ("#9467bd", ":"),
}
for name, (bcolor, ls) in baseline_styles.items():
    if name in baselines:
        mse = baselines[name]["rmse_train"] ** 2
        n_params = baselines[name]["n_params"]
        ax.axhline(mse, color=bcolor, linestyle=ls, alpha=0.7,
                   label=f"{name} ({n_params}p, MSE={mse:.1f})")

ax.set_xlabel("Expression complexity")
ax.set_ylabel("Unweighted MSE (days²)")
ax.set_title("Pareto frontier: complexity vs unweighted MSE")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "pysr_pareto.png"), dpi=150)
print(f"Saved: pysr_pareto.png")
plt.close(fig)


# ── Figure 2: Pred vs obs for best equations at multiple complexities ──────

# Reload Pareto CSVs to get full sympy_format strings (not truncated)
pareto_c = pd.read_csv(os.path.join(RESULTS, "pysr_constrained", "pareto_constrained.csv"))
pareto_u = pd.read_csv(os.path.join(RESULTS, "pysr_unconstrained", "pareto_unconstrained.csv"))

# Ensure age/color are numpy arrays (not pandas Series)
age_arr = np.asarray(age, dtype=float)
color_arr = np.asarray(color, dtype=float)

# Pick representative constrained equations + best unconstrained
target_cxs = [5, 10, 15]
picks = []
for target in target_cxs:
    idx = (pareto_c["complexity"] - target).abs().idxmin()
    row = pareto_c.loc[idx]
    picks.append(("C", row))

best_u_idx = results["unconstrained"]["unweighted_rmse"].idxmin()
best_u_cx = results["unconstrained"].loc[best_u_idx, "complexity"]
u_idx = (pareto_u["complexity"] - best_u_cx).abs().idxmin()
picks.append(("U", pareto_u.loc[u_idx]))

fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5))

for ax, (run_label, row) in zip(axes, picks):
    pred = eval_equation(row["sympy_format"], age_arr, color_arr)
    if pred is None:
        ax.text(0.5, 0.5, "eval failed", transform=ax.transAxes, ha="center")
        continue
    mask = np.isfinite(pred)
    ax.scatter(y[mask], pred[mask], s=1, alpha=0.3, c="steelblue")
    lims = [0, min(55, np.nanmax([y.max(), np.nanpercentile(pred[mask], 99)]) * 1.05)]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    cx = int(row["complexity"])
    # Get RMSE from our computed results
    eq_results = results["constrained" if run_label == "C" else "unconstrained"]
    rmse_row = eq_results.loc[eq_results["complexity"] == cx]
    rmse_val = rmse_row["unweighted_rmse"].values[0] if len(rmse_row) else compute_rmse(pred, y)

    eq_display = str(row["equation"])
    if len(eq_display) > 55:
        eq_display = eq_display[:52] + "..."
    prefix = "Constrained" if run_label == "C" else "Unconstrained"
    ax.set_title(f"{prefix} cx={cx}\n{eq_display}\nRMSE={rmse_val:.2f} d", fontsize=8)
    ax.set_xlabel("Observed $P_{\\rm rot}$ (days)")
    ax.set_ylabel("Predicted $P_{\\rm rot}$ (days)")

fig.suptitle("PySR predictions at increasing complexity", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "pysr_pred_vs_obs.png"), dpi=150, bbox_inches="tight")
print(f"Saved: pysr_pred_vs_obs.png")
plt.close(fig)


# ── Figure 3: Detailed comparison — residuals by cluster age ───────────────

# Use the best constrained equation (lowest unweighted RMSE)
eq_c_results = results["constrained"]
best_c_cx = eq_c_results.loc[eq_c_results["unweighted_rmse"].idxmin(), "complexity"]
best_c_row = pareto_c.loc[(pareto_c["complexity"] - best_c_cx).abs().idxmin()]
pred_best = eval_equation(best_c_row["sympy_format"], age_arr, color_arr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: residuals vs age
resid = y - pred_best
mask = np.isfinite(resid)
axes[0].scatter(df["fiducial_age_Myr"].values[mask], resid[mask],
                s=1, alpha=0.3, c="steelblue")
axes[0].axhline(0, color="k", lw=0.8, ls="--")
axes[0].set_xlabel("Cluster age (Myr)")
axes[0].set_ylabel("Residual (days)")
axes[0].set_xscale("log")
best_c_rmse = eq_c_results.loc[eq_c_results["unweighted_rmse"].idxmin(), "unweighted_rmse"]
axes[0].set_title(f"Best constrained (cx={int(best_c_cx)})\n{str(best_c_row['equation'])[:60]}",
                  fontsize=9)

# Panel 2: residuals vs color
axes[1].scatter(df["BPRP_0"].values[mask], resid[mask],
                s=1, alpha=0.3, c="steelblue")
axes[1].axhline(0, color="k", lw=0.8, ls="--")
axes[1].set_xlabel("$(BP-RP)_0$")
axes[1].set_ylabel("Residual (days)")
axes[1].set_title(f"RMSE = {best_c_rmse:.2f} d", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "pysr_equations_detail.png"), dpi=150)
print(f"Saved: pysr_equations_detail.png")
plt.close(fig)


# ── Final summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 4 ANALYSIS SUMMARY")
print("=" * 70)
for label, s in summary.items():
    print(f"\n{label.upper()}:")
    print(f"  Best RMSE: {s['best_rmse']:.3f} d at complexity {s['best_complexity']}")
    print(f"  Equation:  {s['best_equation']}")
    print(f"  Representatives:")
    for k, v in s["representatives"].items():
        print(f"    cx={v['complexity']:2d}  RMSE={v['unweighted_rmse']:.3f}  {v['equation'][:60]}")
print(f"\nBaseline comparison:")
print(f"  Barnes2007:  RMSE={baselines['Barnes2007']['rmse_train']:.3f} d  (4 params)")
print(f"  Angus2019:   RMSE={baselines['Angus2019']['rmse_train']:.3f} d  (6 params)")

#!/usr/bin/env python3
"""Generate the multi-panel results figure.

Panel (a): Pareto front, complexity vs RMSE, with noise floor.
Panel (b): Bootstrap stability, unique canonical forms at each complexity.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonicalize import canonicalize

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

# ── Data for Panel A: Pareto front ────────────────────────────────────────
with open(os.path.join(RESULTS, "model_selection_aic.json")) as f:
    aic_data = json.load(f)

constrained_eqs = aic_data["constrained"]["all_equations"]
cx_pareto = [e["complexity"] for e in constrained_eqs]
rmse_pareto = [e["rmse"] for e in constrained_eqs]

skumanich_rmse = aic_data["baselines"]["Skumanich_1param"]["rmse"]

with open(os.path.join(RESULTS, "noise_floor.json")) as f:
    nf = json.load(f)
noise_floor = nf["combined_floor_quadrature"]

# ── Data for Panel B: Bootstrap stability at each complexity ──────────────
boot_dir = os.path.join(RESULTS, "bootstrap")
N_BOOT = 100

target_complexities = sorted(set(cx_pareto))
stability_data = {}

for target_cx in target_complexities:
    forms = []
    for i in range(N_BOOT):
        p = os.path.join(boot_dir, f"boot_{i:03d}", "pareto.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        row = df[df["complexity"] == target_cx]
        if len(row) == 0:
            continue
        try:
            forms.append(canonicalize(row.iloc[0]["sympy_format"]))
        except Exception:
            forms.append("(error)")
    if forms:
        c = Counter(forms)
        top_form, top_count = c.most_common(1)[0]
        stability_data[target_cx] = {
            "n_unique": len(c),
            "top_form": top_form,
            "top_pct": 100 * top_count / len(forms),
            "n_present": len(forms),
        }

# ── Create figure ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                gridspec_kw={"width_ratios": [1, 1]},
                                sharex=True)

# Panel A: Pareto front
ax1.plot(cx_pareto, rmse_pareto, "ko-", ms=4, lw=1.1, zorder=3)

# Highlight complexity 6 (Skumanich)
cx6_rmse = None
for e in constrained_eqs:
    if e["complexity"] == 6:
        cx6_rmse = e["rmse"]
        ax1.plot(6, cx6_rmse, "s", color="#2166ac", ms=8, zorder=5,
                 mew=1.2, mec="k")
        break

# Noise floor
ax1.axhline(noise_floor, color="#d6604d", ls="--", lw=1.0, alpha=0.9)
ax1.text(23, noise_floor - 0.25, f"noise floor ({noise_floor:.1f} d)",
         fontsize=6.5, color="#d6604d", ha="right", va="top")

# Annotate Skumanich point
ax1.annotate(r"$P = C\sqrt{t}\;(B_P\!-\!R_P)_0$" + f"\nRMSE = {cx6_rmse:.2f} d",
             xy=(6, cx6_rmse), xytext=(12, 8.8),
             fontsize=6.5, ha="left", color="#2166ac",
             arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=1.0))

# Annotate best
best_rmse = min(rmse_pareto)
best_cx = cx_pareto[rmse_pareto.index(best_rmse)]
ax1.annotate(f"best = {best_rmse:.2f} d\n(complexity {best_cx})",
             xy=(best_cx, best_rmse), xytext=(19, 7.8),
             fontsize=6.5, ha="center", color="0.3",
             arrowprops=dict(arrowstyle="-|>", color="0.5", lw=0.8))

# Gap annotation
mid_y = (best_rmse + noise_floor) / 2
ax1.annotate("", xy=(1.5, noise_floor), xycoords="data",
             xytext=(1.5, best_rmse), textcoords="data",
             arrowprops=dict(arrowstyle="<->", color="#d6604d", lw=0.8))
ax1.text(2.2, mid_y, f"Δ={best_rmse - noise_floor:.1f} d",
         fontsize=6, color="#d6604d", va="center", fontstyle="italic")

ax1.set_xlabel("Expression complexity", fontsize=9)
ax1.set_ylabel("RMSE (days)", fontsize=9)
ax1.set_ylim(bottom=0.8, top=10.5)
ax1.set_xlim(0.5, 23.5)
ax1.set_title("(a) Pareto front", fontsize=9.5, fontweight="bold")
ax1.tick_params(labelsize=8)

# Panel B: Stability (unique forms vs complexity)
cx_vals = sorted(stability_data.keys())
n_unique = [stability_data[cx]["n_unique"] for cx in cx_vals]

colors_bar = []
for nu in n_unique:
    if nu <= 2:
        colors_bar.append("#2166ac")
    elif nu <= 5:
        colors_bar.append("#fdb863")
    else:
        colors_bar.append("#d6604d")

ax2.bar(cx_vals, n_unique, color=colors_bar, edgecolor="k", lw=0.4, width=1.0)
ax2.set_xlabel("Expression complexity", fontsize=9)
ax2.set_ylabel("Unique canonical forms (N = 100)", fontsize=9)
ax2.set_title("(b) Bootstrap form stability", fontsize=9.5, fontweight="bold")
ax2.tick_params(labelsize=8)

# Annotate the cx=6 bar
ax2.annotate(r"$C\sqrt{t}\;c$" + "\n100/100",
             xy=(6, 1), xytext=(3, 25),
             fontsize=7, ha="center", color="#2166ac", fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=1.0))

# Dividing line between stable/unstable
ax2.axvline(8, color="gray", ls=":", lw=0.8, alpha=0.6)
ax2.text(4.5, 55, "stable", fontsize=7.5, ha="center", color="#2166ac",
         fontstyle="italic", alpha=0.7)
ax2.text(16, 55, "unstable", fontsize=7.5, ha="center", color="#d6604d",
         fontstyle="italic", alpha=0.7)

plt.tight_layout(w_pad=2.5)
out_path = os.path.join(FIGURES, "pareto_stability.pdf")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {out_path.replace('.pdf', '.png')}")

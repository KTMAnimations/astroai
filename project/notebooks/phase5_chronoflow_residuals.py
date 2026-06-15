"""
Phase 5: Run ChronoFlow forward model and compute residuals.

*** WARNING: ALL RESULTS IN THIS SCRIPT ARE IN-SAMPLE (TRAINING DATA). ***
ChronoFlow was trained on this exact dataset. Residuals reflect ChronoFlow's
smoothing behavior, NOT genuine astrophysical systematics. Do NOT use these
residuals for downstream analysis (e.g., fitting PySR on residuals would be
fitting noise). For out-of-sample evaluation, see Phase 9 (LOO cross-validation).
Phase 6 (PySR on ChronoFlow residuals) has been CANCELLED for this reason.

For each star, evaluates ChronoFlow's conditional distribution
P(log P_rot | log age, color) at the star's known cluster age and color,
extracts the mode as the point prediction, and computes residuals.

Uses vectorized torch evaluation (~1 second for all 5269 stars).

Produces:
  - project/data/clean_slow_seq_with_residuals.csv
  - project/figures/chronoflow_residuals_by_age.png
  - project/figures/chronoflow_residuals_by_color.png
  - project/figures/chronoflow_residuals_by_cluster.png
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

sys.path.insert(0, os.path.join(ROOT, "..", "chronoflow"))
from chronoflow.ChronoFlow import ChronoFlow

# ── Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv(os.path.join(DATA, "clean_slow_seq.csv"))

# Merge per-star photometric color error from the original catalogue.
# Previously used a fixed logCerr=-1.55 for all stars (checklist A6).
cat = pd.read_csv(os.path.join(DATA, "prot_catalogue.csv"), usecols=["GaiaDR3_ID", "e_BPRP_0"])
df = df.merge(cat, on="GaiaDR3_ID", how="left")
# Fill any missing errors with the median
median_err = df["e_BPRP_0"].median()
df["e_BPRP_0"] = df["e_BPRP_0"].fillna(median_err)
print(f"Color error range: {df['e_BPRP_0'].min():.4f} – {df['e_BPRP_0'].max():.4f} "
      f"(median={median_err:.4f})")

valid = (df["BPRP_0"] > 0.5) & (df["BPRP_0"] < 2.0) & (df["fiducial_age_Myr"] > 10.0) & (df["Prot"] > 0.1)
df = df[valid].reset_index(drop=True)
n = len(df)
print(f"Loaded {n} stars")

# ── Load ChronoFlow ─────────────────────────────────────────────────────────

weights_path = os.path.join(ROOT, "..", "chronoflow", "chronoflow", "model_artifacts", "weights.pth")
cf = ChronoFlow(weights_path)
model = cf.model
model.eval()
print("ChronoFlow model loaded")

# ── Vectorized prediction ──────────────────────────────────────────────────

logA = torch.tensor(np.log10(df["fiducial_age_Myr"].values), dtype=torch.float32)
color_t = torch.tensor(df["BPRP_0"].values, dtype=torch.float32)
# Use per-star photometric color error instead of fixed -1.55 for all stars.
# logCerr = log10(e_BPRP_0) — ChronoFlow expects log10 of the color error.
logCerr = torch.tensor(np.log10(np.clip(df["e_BPRP_0"].values, 1e-6, None)),
                        dtype=torch.float32)
print(f"logCerr range: {logCerr.min().item():.3f} – {logCerr.max().item():.3f}")
context = torch.stack([logA, color_t, logCerr], dim=1)  # (n, 3)

logP_grid = torch.linspace(-1.5, 2.3, 300)  # log10(P/days)
n_grid = len(logP_grid)

print(f"Evaluating {n} stars × {n_grid} grid points...")
t0 = time.time()
with torch.no_grad():
    dist = model(context)
    ll = torch.zeros(n, n_grid)
    for j in range(n_grid):
        x = logP_grid[j].expand(n, 1)
        ll[:, j] = dist.log_prob(x).squeeze(-1)

# Mode (peak of distribution)
mode_idx = ll.argmax(dim=1)
pred_logP_mode = logP_grid[mode_idx].numpy()

# Median (50th percentile of distribution)
probs = torch.exp(ll - ll.max(dim=1, keepdim=True).values)
probs_norm = probs / probs.sum(dim=1, keepdim=True)
cdf = probs_norm.cumsum(dim=1)
median_idx = (cdf >= 0.5).to(torch.int).argmax(dim=1)
pred_logP_median = logP_grid[median_idx].numpy()

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

pred_P_mode = 10.0 ** pred_logP_mode
pred_P_median = 10.0 ** pred_logP_median
obs_P = df["Prot"].values
residual_mode = obs_P - pred_P_mode
residual_median = obs_P - pred_P_median

print(f"Mode RMSE:   {np.sqrt(np.mean(residual_mode**2)):.3f} d")
print(f"Median RMSE: {np.sqrt(np.mean(residual_median**2)):.3f} d")

# ── Save ────────────────────────────────────────────────────────────────────

df["chronoflow_pred_mode"] = pred_P_mode
df["chronoflow_pred_median"] = pred_P_median
df["residual_mode"] = residual_mode
df["residual_median"] = residual_median
df.to_csv(os.path.join(DATA, "clean_slow_seq_with_residuals.csv"), index=False)
print(f"Saved: clean_slow_seq_with_residuals.csv")

# ── Figure 1: Residuals by cluster age ──────────────────────────────────────

cluster_stats = df.groupby("cluster").agg(
    age=("fiducial_age_Myr", "first"),
    mean_resid=("residual_mode", "mean"),
    std_resid=("residual_mode", "std"),
    n=("residual_mode", "count"),
).sort_values("age")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df["fiducial_age_Myr"], residual_mode, s=1, alpha=0.3, c="steelblue")
axes[0].axhline(0, color="k", lw=0.8, ls="--")
axes[0].set_xscale("log")
axes[0].set_xlabel("Cluster age (Myr)")
axes[0].set_ylabel("Residual: observed - predicted (days)")
axes[0].set_title("ChronoFlow residuals vs age")
axes[0].set_ylim(-30, 30)

axes[1].errorbar(cluster_stats["age"], cluster_stats["mean_resid"],
                 yerr=cluster_stats["std_resid"] / np.sqrt(cluster_stats["n"]),
                 fmt="o", ms=4, capsize=2, color="steelblue")
axes[1].axhline(0, color="k", lw=0.8, ls="--")
axes[1].set_xscale("log")
axes[1].set_xlabel("Cluster age (Myr)")
axes[1].set_ylabel("Mean residual (days)")
axes[1].set_title("Per-cluster mean residual ± SE")

fig.tight_layout()
fig.text(0.5, -0.02, "Note: IN-SAMPLE residuals (training data). See Phase 9 for out-of-sample.",
         ha="center", fontsize=9, style="italic", color="gray")
fig.savefig(os.path.join(FIG, "chronoflow_residuals_by_age.png"), dpi=150, bbox_inches="tight")
print(f"Saved: chronoflow_residuals_by_age.png")
plt.close(fig)

# ── Figure 2: Residuals by color ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df["BPRP_0"], residual_mode, s=1, alpha=0.3, c="steelblue")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("$(BP-RP)_0$")
ax.set_ylabel("Residual (days)")
ax.set_title("ChronoFlow residuals vs color")
ax.set_ylim(-30, 30)

color_bins = np.arange(0.5, 3.5, 0.2)
color_centers = color_bins[:-1] + 0.1
bin_means, bin_ses = [], []
for lo, hi in zip(color_bins[:-1], color_bins[1:]):
    mask = (df["BPRP_0"].values >= lo) & (df["BPRP_0"].values < hi)
    if mask.sum() > 5:
        bin_means.append(np.nanmean(residual_mode[mask]))
        bin_ses.append(np.nanstd(residual_mode[mask]) / np.sqrt(mask.sum()))
    else:
        bin_means.append(np.nan)
        bin_ses.append(np.nan)
ax.errorbar(color_centers, bin_means, yerr=bin_ses, fmt="s-", color="red",
            ms=4, capsize=2, label="Binned mean ± SE")
ax.legend()
fig.tight_layout()
fig.text(0.5, -0.02, "Note: IN-SAMPLE residuals (training data). See Phase 9 for out-of-sample.",
         ha="center", fontsize=9, style="italic", color="gray")
fig.savefig(os.path.join(FIG, "chronoflow_residuals_by_color.png"), dpi=150, bbox_inches="tight")
print(f"Saved: chronoflow_residuals_by_color.png")
plt.close(fig)

# ── Figure 3: Per-cluster bar chart ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
clusters_sorted = cluster_stats.index.tolist()
x = range(len(clusters_sorted))
ax.bar(x, cluster_stats["mean_resid"],
       yerr=cluster_stats["std_resid"] / np.sqrt(cluster_stats["n"]),
       capsize=2, color="steelblue", alpha=0.7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{c}\n({cluster_stats.loc[c, 'age']:.0f})" for c in clusters_sorted],
                   rotation=90, fontsize=7)
ax.set_ylabel("Mean residual (days)")
ax.set_title("ChronoFlow mean residual per cluster (with SE)")
fig.tight_layout()
fig.text(0.5, -0.02, "Note: IN-SAMPLE residuals (training data). See Phase 9 for out-of-sample.",
         ha="center", fontsize=9, style="italic", color="gray")
fig.savefig(os.path.join(FIG, "chronoflow_residuals_by_cluster.png"), dpi=150, bbox_inches="tight")
print(f"Saved: chronoflow_residuals_by_cluster.png")
plt.close(fig)

# ── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 5 SUMMARY")
print("⚠ CAVEAT: All metrics below are IN-SAMPLE (training data).")
print("  ChronoFlow was trained on this dataset. These residuals")
print("  reflect training error, not generalization. Use LOO (Phase 9)")
print("  for out-of-sample evaluation before citing patterns as findings.")
print("=" * 60)
print(f"Stars:          {n}")
print(f"Mode RMSE:      {np.sqrt(np.mean(residual_mode**2)):.3f} d")
print(f"Median RMSE:    {np.sqrt(np.mean(residual_median**2)):.3f} d")
print(f"Mean residual:  {np.mean(residual_mode):.3f} d")
print(f"Std residual:   {np.std(residual_mode):.3f} d")
print(f"\nClusters with largest |mean residual|:")
top = cluster_stats.reindex(cluster_stats["mean_resid"].abs().sort_values(ascending=False).index).head(5)
for c in top.index:
    r = top.loc[c]
    print(f"  {c:20s} (age={r['age']:.0f} Myr):  mean={r['mean_resid']:+.2f} ± {r['std_resid']/np.sqrt(r['n']):.2f} d  (N={int(r['n'])})")

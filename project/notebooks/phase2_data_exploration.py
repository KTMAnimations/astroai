"""
Phase 2: Data exploration, quality filtering, and slow-sequence filtering.

Produces:
  - project/figures/period_vs_color_by_age.png        (quality check scatter plot)
  - project/figures/period_vs_color_per_cluster.png    (per-cluster panels for slow-seq inspection)
  - project/figures/slow_seq_filtering_per_cluster.png (before/after slow-seq filtering)
  - project/figures/age_distribution.png               (training data skew)
  - project/data/clean_all.csv                         (quality-filtered, no slow-seq filter)
  - project/data/clean_slow_seq.csv                    (quality-filtered + slow-seq filtered)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import median_filter
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

# ── 1. Load and apply quality cuts (per ChronoFlow README) ──────────────────

df_raw = pd.read_csv(os.path.join(DATA, "prot_catalogue.csv"))
print(f"Raw catalogue: {len(df_raw)} rows")

quality_mask = (
    (df_raw["cmd_exclude"] == 0)
    & (df_raw["dered_exclude"] == 0)
    & (df_raw["phot_exclude"] == 0)
    & (df_raw["dr3_ruwe"] < 1.4)
)
df = df_raw[quality_mask].copy()
print(f"After quality cuts: {len(df)} rows ({len(df_raw) - len(df)} removed)")

# ── 2. Basic data summary ───────────────────────────────────────────────────

print("\n=== Key variable ranges ===")
print(f"  Prot:             {df['Prot'].min():.3f} – {df['Prot'].max():.1f} days")
print(f"  BPRP_0:           {df['BPRP_0'].min():.3f} – {df['BPRP_0'].max():.3f}")
print(f"  fiducial_age_Myr: {df['fiducial_age_Myr'].min():.1f} – {df['fiducial_age_Myr'].max():.1f} Myr")
print(f"  Clusters:         {df['cluster'].nunique()}")

# ── 3. Training data skew ───────────────────────────────────────────────────

cluster_summary = (
    df.groupby("cluster")
    .agg(age_Myr=("fiducial_age_Myr", "first"), n_stars=("Prot", "count"))
    .sort_values("age_Myr")
)
print("\n=== Cluster summary ===")
print(cluster_summary.to_string())

n_lt500 = cluster_summary.loc[cluster_summary["age_Myr"] < 500, "n_stars"].sum()
n_gt1000 = cluster_summary.loc[cluster_summary["age_Myr"] > 1000, "n_stars"].sum()
n_gt2000 = cluster_summary.loc[cluster_summary["age_Myr"] > 2000, "n_stars"].sum()
n_total = len(df)
print(f"\n=== Age skew ===")
print(f"  <500 Myr:  {n_lt500} stars ({100*n_lt500/n_total:.1f}%)")
print(f"  >1000 Myr: {n_gt1000} stars ({100*n_gt1000/n_total:.1f}%)")
print(f"  >2000 Myr: {n_gt2000} stars ({100*n_gt2000/n_total:.1f}%)")

# ── 4. Quality-check scatter plot: P_rot vs (BP-RP)_0, colored by age ──────

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    df["BPRP_0"],
    df["Prot"],
    c=df["fiducial_age_Myr"],
    s=1,
    alpha=0.5,
    cmap="coolwarm",
    norm=LogNorm(vmin=df["fiducial_age_Myr"].min(), vmax=df["fiducial_age_Myr"].max()),
)
cbar = fig.colorbar(sc, ax=ax, label="Cluster age (Myr)")
ax.set_xlabel("$(BP - RP)_0$")
ax.set_ylabel("$P_{\\rm rot}$ (days)")
ax.set_title("Rotation period vs color — quality-filtered (N = %d)" % len(df))
ax.set_xlim(0.2, 3.5)
ax.set_ylim(0, 50)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "period_vs_color_by_age.png"), dpi=150)
print(f"\nSaved: period_vs_color_by_age.png")
plt.close(fig)

# ── 5. Age distribution histogram ──────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ages = df["fiducial_age_Myr"].values
axes[0].hist(ages, bins=50, edgecolor="k", linewidth=0.3)
axes[0].set_xlabel("Age (Myr)")
axes[0].set_ylabel("Number of stars")
axes[0].set_title("Age distribution (linear)")

axes[1].hist(np.log10(ages), bins=30, edgecolor="k", linewidth=0.3)
axes[1].set_xlabel("log₁₀(Age / Myr)")
axes[1].set_ylabel("Number of stars")
axes[1].set_title("Age distribution (log)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "age_distribution.png"), dpi=150)
print(f"Saved: age_distribution.png")
plt.close(fig)

# ── 6. Per-cluster period–color diagrams ────────────────────────────────────

clusters_sorted = cluster_summary.index.tolist()
n_clusters = len(clusters_sorted)
ncols = 6
nrows = int(np.ceil(n_clusters / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows), sharex=True)
axes_flat = axes.flatten()

for i, cluster_name in enumerate(clusters_sorted):
    ax = axes_flat[i]
    sub = df[df["cluster"] == cluster_name]
    age = sub["fiducial_age_Myr"].iloc[0]
    ax.scatter(sub["BPRP_0"], sub["Prot"], s=2, alpha=0.5, c="k")
    ax.set_title(f"{cluster_name}\n{age:.0f} Myr (N={len(sub)})", fontsize=8)
    ax.set_ylim(0, min(55, sub["Prot"].quantile(0.99) * 1.3))
    ax.set_xlim(0.3, 3.2)

for i in range(n_clusters, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.supxlabel("$(BP - RP)_0$")
fig.supylabel("$P_{\\rm rot}$ (days)")
fig.suptitle("Period vs color per cluster (quality-filtered)", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "period_vs_color_per_cluster.png"), dpi=150, bbox_inches="tight")
print(f"Saved: period_vs_color_per_cluster.png")
plt.close(fig)

# ── 7. Slow-sequence filtering ─────────────────────────────────────────────
#
# Strategy: for each cluster, in sliding color bins, identify the slow-sequence
# ridge as the UPPER mode of the period distribution and remove stars that are
# far below it (fast rotators).
#
# The slow sequence is the population of stars that have converged onto
# Skumanich-like spin-down. In young clusters (<500 Myr), there is a clear
# bimodal split: fast rotators (short P) and slow rotators (long P). We want
# to keep the slow-sequence stars.
#
# Approach per cluster:
#   1. Bin stars by color in overlapping bins.
#   2. In each bin, compute the 75th percentile of P_rot as a robust estimate
#      of the slow-sequence ridge.
#   3. Keep stars with P_rot > ridge * threshold_fraction.
#   4. For old clusters (>600 Myr) where the sequence has converged, use a
#      gentler cut (keep more stars).
#
# This is a heuristic. The plan calls for also running PySR on unfiltered data
# as an ablation, so both datasets are saved.


def filter_slow_sequence(sub_df, age_myr, color_col="BPRP_0", period_col="Prot"):
    """
    Filter a single cluster's data to keep slow-sequence rotators.

    For young clusters (<600 Myr): keep stars with P > 40% of the local
    75th-percentile ridge. This removes the fast-sequence population which
    sits well below the slow sequence.

    For old clusters (>=600 Myr): the sequences have mostly converged, so
    use a gentler cut (30% of the local median) to only remove extreme
    outliers.

    Returns a boolean mask (True = keep).
    """
    colors = sub_df[color_col].values
    periods = sub_df[period_col].values
    keep = np.ones(len(sub_df), dtype=bool)

    # Color range for this cluster
    color_min, color_max = np.nanpercentile(colors, [2, 98])
    bin_width = 0.3  # mag
    bin_step = 0.1   # overlap for smoothness

    if age_myr >= 600:
        # Old cluster: sequence mostly converged, gentle outlier removal
        bin_centers = np.arange(color_min + bin_width / 2, color_max, bin_step)
        for bc in bin_centers:
            in_bin = (colors >= bc - bin_width / 2) & (colors < bc + bin_width / 2)
            if in_bin.sum() < 5:
                continue
            median_p = np.median(periods[in_bin])
            # Remove stars with P < 30% of the local median (extreme outliers)
            cut = median_p * 0.30
            keep[in_bin & (periods < cut)] = False
    else:
        # Young cluster: bimodal, need to separate slow from fast
        bin_centers = np.arange(color_min + bin_width / 2, color_max, bin_step)
        for bc in bin_centers:
            in_bin = (colors >= bc - bin_width / 2) & (colors < bc + bin_width / 2)
            if in_bin.sum() < 5:
                continue
            # Use 75th percentile as the slow-sequence ridge estimate
            p75 = np.percentile(periods[in_bin], 75)
            # Keep stars with P > 40% of the ridge
            # This removes the fast-sequence clump which typically sits at
            # <25% of the slow-sequence period
            cut = p75 * 0.40
            keep[in_bin & (periods < cut)] = False

    return keep


# Apply slow-sequence filter to each cluster
# Reset index so positional indexing works cleanly
df = df.reset_index(drop=True)
slow_seq_mask = np.zeros(len(df), dtype=bool)

for cluster_name in clusters_sorted:
    cluster_mask = df["cluster"] == cluster_name
    sub = df[cluster_mask]
    age = sub["fiducial_age_Myr"].iloc[0]
    keep = filter_slow_sequence(sub, age)
    slow_seq_mask[sub.index] = keep
    n_removed = (~keep).sum()
    if n_removed > 0:
        print(f"  {cluster_name} ({age:.0f} Myr): removed {n_removed}/{len(sub)} fast rotators")

df_slow = df[slow_seq_mask].copy()
print(f"\nAfter slow-sequence filtering: {len(df_slow)} / {len(df)} "
      f"({len(df) - len(df_slow)} removed, {100*(len(df)-len(df_slow))/len(df):.1f}%)")

# ── 8. Visualization: before/after slow-sequence filtering ──────────────────

# Show a selection of clusters where filtering matters most (young ones)
show_clusters = [c for c in clusters_sorted
                 if cluster_summary.loc[c, "age_Myr"] < 500
                 and cluster_summary.loc[c, "n_stars"] >= 50][:12]

fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
axes_flat = axes.flatten()

for i, cluster_name in enumerate(show_clusters):
    ax = axes_flat[i]
    sub_all = df[df["cluster"] == cluster_name]
    sub_slow = df_slow[df_slow["cluster"] == cluster_name]
    removed = sub_all[~sub_all.index.isin(sub_slow.index)]
    age = sub_all["fiducial_age_Myr"].iloc[0]

    ax.scatter(removed["BPRP_0"], removed["Prot"], s=3, alpha=0.4, c="red", label="removed")
    ax.scatter(sub_slow["BPRP_0"], sub_slow["Prot"], s=3, alpha=0.5, c="steelblue", label="kept")
    ax.set_title(f"{cluster_name} ({age:.0f} Myr)\n"
                 f"kept {len(sub_slow)}/{len(sub_all)}", fontsize=9)
    ax.set_ylim(0, min(40, sub_all["Prot"].quantile(0.98) * 1.3))
    ax.set_xlim(0.3, 3.2)
    if i == 0:
        ax.legend(fontsize=7, markerscale=3)

for i in range(len(show_clusters), len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.supxlabel("$(BP - RP)_0$")
fig.supylabel("$P_{\\rm rot}$ (days)")
fig.suptitle("Slow-sequence filtering (young clusters)", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "slow_seq_filtering_per_cluster.png"), dpi=150, bbox_inches="tight")
print(f"Saved: slow_seq_filtering_per_cluster.png")
plt.close(fig)

# ── 9. Save cleaned datasets ───────────────────────────────────────────────

# Columns to keep for PySR
keep_cols = ["GaiaDR3_ID", "cluster", "Prot", "BPRP_0", "fiducial_age_Myr"]

df[keep_cols].to_csv(os.path.join(DATA, "clean_all.csv"), index=False)
print(f"Saved: clean_all.csv ({len(df)} rows)")

df_slow[keep_cols].to_csv(os.path.join(DATA, "clean_slow_seq.csv"), index=False)
print(f"Saved: clean_slow_seq.csv ({len(df_slow)} rows)")

# ── 10. Final summary ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2 SUMMARY")
print("=" * 60)
print(f"Raw catalogue:              {len(df_raw)} stars")
print(f"After quality cuts:         {len(df)} stars (used by ChronoFlow)")
print(f"After slow-seq filtering:   {len(df_slow)} stars")
print(f"")
print(f"Age range:  {df['fiducial_age_Myr'].min():.1f} – {df['fiducial_age_Myr'].max():.1f} Myr")
print(f"Color range: {df['BPRP_0'].min():.3f} – {df['BPRP_0'].max():.3f}")
print(f"Period range: {df['Prot'].min():.3f} – {df['Prot'].max():.1f} days")
print(f"Clusters: {df['cluster'].nunique()}")
print(f"Ages in Myr (linear, not log-transformed)")
print(f"")
print(f"Saved datasets:")
print(f"  clean_all.csv       — quality-filtered only (for PySR ablation)")
print(f"  clean_slow_seq.csv  — quality + slow-seq filtered (primary PySR input)")
print(f"")
print(f"Saved figures:")
print(f"  period_vs_color_by_age.png")
print(f"  period_vs_color_per_cluster.png")
print(f"  slow_seq_filtering_per_cluster.png")
print(f"  age_distribution.png")

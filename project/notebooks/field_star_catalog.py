"""
Block 6.1: Field-star age catalog via Skumanich inversion.

Applies the validated 1-parameter Skumanich equation (P = C * sqrt(age) * color)
to McQuillan et al. 2014 Kepler rotation periods, producing gyrochronological
age estimates for ~34,000 field stars.

Cross-matches with Berger et al. 2020 for Gaia DR3 (BP-RP)_0 colors.

Produces:
  - project/results/field_star_ages.csv
  - project/results/field_star_catalog_summary.json
  - project/figures/field_star_age_distribution.png
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# ── Load Skumanich coefficient and bootstrap uncertainty ────────────────────
baseline = json.load(open(os.path.join(RESULTS, "one_param_baseline.json")))
C_sku = baseline["skumanich_1param_C"]
print(f"Skumanich coefficient: C = {C_sku:.6f}")

# Bootstrap coefficient spread (from bootstrap_summary.json)
boot_path = os.path.join(RESULTS, "bootstrap", "bootstrap_summary.json")
boot_data = json.load(open(boot_path))

# Extract C values from Skumanich-form bootstrap results
boot_C_values = []
for entry in boot_data:
    eq = entry["best_equation"]
    canon = entry["best_canonical"]
    if "sqrt(age)" in canon and "color" in canon:
        # Extract coefficient from equations like "color * (sqrt(age) * 0.3512394)"
        # or "sqrt(age * 0.12) * color" where C = sqrt(0.12)
        import re
        # Try pattern: C * sqrt(age) * color
        m = re.search(r'(\d+\.?\d*)\s*\*?\s*sqrt', eq)
        if m:
            boot_C_values.append(float(m.group(1)))
        else:
            # Try pattern: sqrt(age * X) * color -> C = sqrt(X)
            m = re.search(r'sqrt\(age\s*\*\s*(\d+\.?\d*)\)', eq)
            if m:
                boot_C_values.append(np.sqrt(float(m.group(1))))

if boot_C_values:
    C_std = float(np.std(boot_C_values))
    print(f"Bootstrap C spread: {np.mean(boot_C_values):.6f} ± {C_std:.6f} (N={len(boot_C_values)})")
else:
    C_std = 0.01  # fallback
    print(f"Could not extract bootstrap C values, using C_std={C_std}")

# Intrinsic scatter from noise floor
noise_floor = json.load(open(os.path.join(RESULTS, "noise_floor.json")))
sigma_intrinsic = noise_floor["intrinsic_scatter_median_days"]
print(f"Intrinsic scatter: {sigma_intrinsic:.2f} d")


# ── Download McQuillan 2014 catalog ────────────────────────────────────────
mcq_cache = os.path.join(DATA, "mcquillan2014.csv")

if os.path.exists(mcq_cache):
    print(f"\nLoading cached McQuillan catalog: {mcq_cache}")
    mcq = pd.read_csv(mcq_cache)
else:
    print("\nDownloading McQuillan et al. 2014 from VizieR...")
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    tables = Vizier.get_catalogs("J/ApJS/211/24/table1")
    mcq = tables[0].to_pandas()
    mcq.to_csv(mcq_cache, index=False)
    print(f"Saved: {mcq_cache}")

print(f"McQuillan catalog: {len(mcq)} stars")


# ── Download Berger et al. 2020 (Gaia cross-match) ─────────────────────────
berger_cache = os.path.join(DATA, "berger2020.csv")

if os.path.exists(berger_cache):
    print(f"Loading cached Berger catalog: {berger_cache}")
    berger = pd.read_csv(berger_cache)
else:
    print("Downloading Berger et al. 2020 from VizieR...")
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    tables = Vizier.get_catalogs("J/AJ/159/280/table1")
    berger = tables[0].to_pandas()
    berger.to_csv(berger_cache, index=False)
    print(f"Saved: {berger_cache}")

print(f"Berger catalog: {len(berger)} stars")


# ── Cross-match ─────────────────────────────────────────────────────────────
# McQuillan uses KIC, Berger uses KIC — direct join
# Find the KIC column names
mcq_kic_col = [c for c in mcq.columns if "KIC" in c.upper() or c == "KIC"][0]
berger_kic_col = [c for c in berger.columns if "KIC" in c.upper() or c == "KIC"][0]

print(f"\nMcQuillan KIC column: {mcq_kic_col}")
print(f"Berger KIC column: {berger_kic_col}")

# Find relevant columns in Berger (BP-RP or Teff)
print(f"Berger columns: {list(berger.columns)[:20]}")

merged = mcq.merge(berger, left_on=mcq_kic_col, right_on=berger_kic_col, how="inner",
                   suffixes=("_mcq", "_berger"))
print(f"Cross-matched: {len(merged)} stars")

# Find period column
prot_col = [c for c in merged.columns if "Prot" in c or "prot" in c.lower() or "Per" in c][0]
print(f"Period column: {prot_col}")

# Find BP-RP column — try several naming conventions
bp_rp_candidates = [c for c in merged.columns if "bp" in c.lower() and "rp" in c.lower()]
teff_candidates = [c for c in merged.columns if "teff" in c.lower() or "Teff" in c]

print(f"BP-RP candidates: {bp_rp_candidates}")
print(f"Teff candidates: {teff_candidates}")

# If BP-RP not directly available, compute from Teff using empirical relation
if bp_rp_candidates:
    bp_rp_col = bp_rp_candidates[0]
    merged["BPRP_0"] = pd.to_numeric(merged[bp_rp_col], errors="coerce")
    print(f"Using BP-RP column: {bp_rp_col}")
elif teff_candidates:
    teff_col = teff_candidates[0]
    teff = pd.to_numeric(merged[teff_col], errors="coerce")
    # Approximate Teff -> BP-RP conversion for MS stars (Pecaut & Mamajek 2013)
    # BP-RP ≈ 8350/Teff - 0.35 (rough empirical fit for 3500-7000 K)
    merged["BPRP_0"] = 8350.0 / teff - 0.35
    print(f"Converted Teff to BP-RP using empirical relation")
else:
    raise ValueError("Cannot find BP-RP or Teff column in Berger catalog")

# Extract period and clean
merged["Prot"] = pd.to_numeric(merged[prot_col], errors="coerce")

# Filter to gyrochronology-valid domain
valid = (
    merged["BPRP_0"].notna()
    & merged["Prot"].notna()
    & (merged["BPRP_0"] > 0.5)
    & (merged["BPRP_0"] < 2.0)
    & (merged["Prot"] > 0.5)  # exclude very fast rotators
    & (merged["Prot"] < 50.0)  # exclude likely spurious long periods
)
catalog = merged[valid].copy()
print(f"\nAfter filtering to FGK domain: {len(catalog)} stars")


# ── Compute gyrochronological ages ─────────────────────────────────────────
# P = C * sqrt(age) * color  →  age = (P / (C * color))^2
catalog["gyro_age_Myr"] = (catalog["Prot"] / (C_sku * catalog["BPRP_0"])) ** 2

# Propagate uncertainties
# δage/age = 2 * sqrt( (δP/P)^2 + (δC/C)^2 + (δcolor/color)^2 + (σ_intrinsic/P)^2 )
# Typical δP ~ 0.5 d (McQuillan), δcolor ~ 0.01 mag (Gaia)
delta_P = 0.5  # conservative period uncertainty
delta_color = 0.01  # Gaia color uncertainty

catalog["gyro_age_err_Myr"] = catalog["gyro_age_Myr"] * 2 * np.sqrt(
    (delta_P / catalog["Prot"]) ** 2
    + (C_std / C_sku) ** 2
    + (delta_color / catalog["BPRP_0"]) ** 2
    + (sigma_intrinsic / catalog["Prot"]) ** 2
)

# Flag candidate WMB stars: those whose periods are SHORT relative to
# what Skumanich predicts for their estimated age bracket
# (This is circular if we use the Skumanich age, so we use a comparison approach:
# flag stars where P < C * sqrt(4000) * color, i.e., they'd need to be <4 Gyr
# but their age estimate places them >4 Gyr)
catalog["gyro_age_gt_4Gyr"] = catalog["gyro_age_Myr"] > 4000
catalog["possible_wmb"] = False  # placeholder — need external age estimates for real WMB flagging

print(f"\n=== Age Distribution ===")
print(f"  Median age: {catalog['gyro_age_Myr'].median():.0f} Myr")
print(f"  Mean age:   {catalog['gyro_age_Myr'].mean():.0f} Myr")
print(f"  Age range:  [{catalog['gyro_age_Myr'].min():.0f}, {catalog['gyro_age_Myr'].max():.0f}] Myr")
print(f"  < 1 Gyr:    {(catalog['gyro_age_Myr'] < 1000).sum()} ({100*(catalog['gyro_age_Myr'] < 1000).mean():.1f}%)")
print(f"  1-4 Gyr:    {((catalog['gyro_age_Myr'] >= 1000) & (catalog['gyro_age_Myr'] < 4000)).sum()}")
print(f"  > 4 Gyr:    {(catalog['gyro_age_Myr'] >= 4000).sum()} ({100*(catalog['gyro_age_Myr'] >= 4000).mean():.1f}%)")
print(f"  Median uncertainty: ±{catalog['gyro_age_err_Myr'].median():.0f} Myr ({100*catalog['gyro_age_err_Myr'].median()/catalog['gyro_age_Myr'].median():.0f}%)")


# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Age histogram
ax = axes[0]
ages_gyr = catalog["gyro_age_Myr"] / 1000
ax.hist(ages_gyr, bins=50, range=(0, 15), alpha=0.7, color="steelblue", edgecolor="white")
ax.axvline(4.567, color="orange", ls="--", lw=2, label="Solar age")
ax.set_xlabel("Gyro Age (Gyr)")
ax.set_ylabel("Count")
ax.set_title(f"Gyrochronological Ages ({len(catalog):,} Kepler FGK Stars)")
ax.legend()

# Panel 2: Period vs color, colored by age
ax = axes[1]
sc = ax.scatter(catalog["BPRP_0"], catalog["Prot"], c=np.log10(catalog["gyro_age_Myr"]),
                cmap="viridis", s=1, alpha=0.3, vmin=1.5, vmax=4.5)
plt.colorbar(sc, ax=ax, label="log₁₀(Age / Myr)")
ax.set_xlabel("(BP-RP)₀")
ax.set_ylabel("P_rot (d)")
ax.set_title("Period-Color Diagram Colored by Gyro Age")

plt.tight_layout()
fig_path = os.path.join(FIG, "field_star_age_distribution.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved: {fig_path}")


# ── Save catalog ────────────────────────────────────────────────────────────
out_cols = [mcq_kic_col, "Prot", "BPRP_0", "gyro_age_Myr", "gyro_age_err_Myr", "gyro_age_gt_4Gyr"]
# Only keep columns that exist
out_cols = [c for c in out_cols if c in catalog.columns]
out_df = catalog[out_cols].copy()
out_df = out_df.rename(columns={mcq_kic_col: "KIC"})

cat_path = os.path.join(RESULTS, "field_star_ages.csv")
out_df.to_csv(cat_path, index=False)
print(f"Saved catalog: {cat_path} ({len(out_df)} stars)")

# Summary JSON
summary = {
    "n_mcquillan": int(len(mcq)),
    "n_berger": int(len(berger)),
    "n_crossmatch": int(len(merged)),
    "n_valid_fgk": int(len(catalog)),
    "skumanich_C": float(C_sku),
    "skumanich_C_err": float(C_std),
    "age_stats": {
        "median_Myr": float(catalog["gyro_age_Myr"].median()),
        "mean_Myr": float(catalog["gyro_age_Myr"].mean()),
        "min_Myr": float(catalog["gyro_age_Myr"].min()),
        "max_Myr": float(catalog["gyro_age_Myr"].max()),
        "lt_1Gyr": int((catalog["gyro_age_Myr"] < 1000).sum()),
        "1_to_4Gyr": int(((catalog["gyro_age_Myr"] >= 1000) & (catalog["gyro_age_Myr"] < 4000)).sum()),
        "gt_4Gyr": int((catalog["gyro_age_Myr"] >= 4000).sum()),
    },
    "median_age_err_Myr": float(catalog["gyro_age_err_Myr"].median()),
    "median_fractional_err": float(catalog["gyro_age_err_Myr"].median() / catalog["gyro_age_Myr"].median()),
}

summary_path = os.path.join(RESULTS, "field_star_catalog_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary: {summary_path}")

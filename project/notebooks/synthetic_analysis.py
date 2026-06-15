"""
Block 3.2: Analyze synthetic control experiment results.

Reads bootstrap_summary.json from each synthetic dataset (A, B, C) and
produces comparison tables and figures.

Usage:
    python synthetic_analysis.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# ── Load results ────────────────────────────────────────────────────────────
datasets = {}
for ds_name in ["A", "B", "C"]:
    path = os.path.join(RESULTS, "synthetic", f"dataset_{ds_name}", "bootstrap_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            datasets[ds_name] = json.load(f)
        print(f"Dataset {ds_name}: {len(datasets[ds_name])} resamples loaded")
    else:
        print(f"Dataset {ds_name}: NOT FOUND at {path}")

if not datasets:
    print("No synthetic results found. Run synthetic_control.py first.")
    exit(1)

# ── Analyze each dataset ───────────────────────────────────────────────────
descriptions = {
    "A": "Known Skumanich (positive control)",
    "B": "Known complex (negative control)",
    "C": "Known piecewise (transition)",
}

true_forms = {
    "A": "P = 0.344 * sqrt(age) * color",
    "B": "P = 0.3*age^0.4*(color-0.3)^0.6 + 0.005*age*exp(-color)",
    "C": "P = 0.344*sqrt(min(age,1000))*color",
}

summary = {}

for ds_name, data in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset {ds_name}: {descriptions.get(ds_name, '?')}")
    print(f"True form: {true_forms.get(ds_name, '?')}")
    print(f"{'='*60}")

    # Count canonical forms from get_best()
    best_forms = {}
    for r in data:
        form = r["best_canonical"]
        best_forms[form] = best_forms.get(form, 0) + 1

    # Count canonical forms from top-1 by RMSE
    rmse_forms = {}
    for r in data:
        if r.get("top_3"):
            form = r["top_3"][0]["canonical_form"]
            rmse_forms[form] = rmse_forms.get(form, 0) + 1

    # Dominant form
    top_best_form, top_best_count = max(best_forms.items(), key=lambda x: x[1])

    print(f"\nget_best() form frequency (top 5):")
    print(f"{'Count':>6s}  {'Pct':>5s}  Form")
    print("-" * 60)
    for form, count in sorted(best_forms.items(), key=lambda x: -x[1])[:5]:
        pct = 100.0 * count / len(data)
        print(f"{count:6d}  {pct:5.1f}%  {form}")

    if rmse_forms:
        top_rmse_form, top_rmse_count = max(rmse_forms.items(), key=lambda x: x[1])
        print(f"\ntop-1 by RMSE form frequency (top 5):")
        print(f"{'Count':>6s}  {'Pct':>5s}  Form")
        print("-" * 60)
        for form, count in sorted(rmse_forms.items(), key=lambda x: -x[1])[:5]:
            pct = 100.0 * count / len(data)
            print(f"{count:6d}  {pct:5.1f}%  {form}")

    dominance = top_best_count / len(data)
    is_skumanich = "sqrt(age)" in top_best_form and "color" in top_best_form

    if dominance >= 0.8:
        verdict = "STABLE"
    elif dominance < 0.5:
        verdict = "UNSTABLE"
    else:
        verdict = "MARGINAL"

    summary[ds_name] = {
        "description": descriptions.get(ds_name, "?"),
        "true_form": true_forms.get(ds_name, "?"),
        "n_resamples": len(data),
        "dominant_form": top_best_form,
        "dominant_count": top_best_count,
        "dominant_pct": round(100 * dominance, 1),
        "n_unique_forms": len(best_forms),
        "is_skumanich": is_skumanich,
        "verdict": verdict,
        "form_distribution": best_forms,
    }

# ── Comparison table ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SYNTHETIC CONTROL COMPARISON TABLE")
print(f"{'='*70}\n")

print(f"{'Dataset':>8s}  {'Description':>35s}  {'Dom. form':>25s}  {'Freq':>5s}  {'#Forms':>6s}  {'Verdict':>10s}")
print("-" * 100)
for ds_name in ["A", "B", "C"]:
    if ds_name in summary:
        r = summary[ds_name]
        print(f"{ds_name:>8s}  {r['description']:>35s}  {r['dominant_form'][:25]:>25s}  "
              f"{r['dominant_pct']:5.1f}%  {r['n_unique_forms']:6d}  {r['verdict']:>10s}")

# Validation check
print(f"\n--- Method Validation ---")
if "A" in summary:
    if summary["A"]["verdict"] == "STABLE" and summary["A"]["is_skumanich"]:
        print("  [PASS] Dataset A: correctly identifies stable Skumanich form")
    else:
        print("  [FAIL] Dataset A: should be stable Skumanich")

if "B" in summary:
    if summary["B"]["verdict"] == "UNSTABLE":
        print("  [PASS] Dataset B: correctly detects form instability for complex truth")
    elif summary["B"]["verdict"] == "MARGINAL":
        print("  [WARN] Dataset B: marginal instability detection")
    else:
        print("  [FAIL] Dataset B: should show instability for complex truth")

if "C" in summary:
    print(f"  [INFO] Dataset C: {summary['C']['verdict']} — expected behavior depends on data distribution")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
if len(datasets) == 1:
    axes = [axes]

for ax, (ds_name, r) in zip(axes, summary.items()):
    forms = r["form_distribution"]
    # Top 5 forms
    top_forms = sorted(forms.items(), key=lambda x: -x[1])[:5]
    labels = [f[:25] for f, _ in top_forms]
    counts = [c for _, c in top_forms]
    if len(forms) > 5:
        labels.append("Other")
        counts.append(sum(c for f, c in forms.items() if f not in dict(top_forms)))

    colors_list = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.barh(range(len(labels)), counts, color=colors_list)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Count")
    ax.set_title(f"Dataset {ds_name}: {descriptions.get(ds_name, '?')}\n"
                 f"Verdict: {r['verdict']}", fontsize=10)
    ax.invert_yaxis()

plt.tight_layout()
fig_path = os.path.join(FIG, "synthetic_control_comparison.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved: {fig_path}")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS, "synthetic_control_analysis.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {out_path}")

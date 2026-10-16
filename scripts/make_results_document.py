#!/usr/bin/env python3
"""Generate the detailed research tables directly from the saved study bundle."""
import json

import numpy as np
import pandas as pd

from astrogyro.data import ROOT


def markdown_table(frame):
    # Avoid an optional tabulate dependency for a simple, inspectable table.
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def main():
    root = ROOT / "project/results/research_v1"
    summary = json.loads((root / "observational_summary.json").read_text())
    diag = json.loads((root / "selection_diagnostics.json").read_text())
    guard = json.loads((root / "domain_followup_summary.json").read_text())
    fresh_root = ROOT / "project/results/domain_confirmation"
    fresh = json.loads((fresh_root / "domain_followup_summary.json").read_text())
    fresh_scores = pd.read_csv(fresh_root / "domain_followup.csv")
    fresh_table = fresh_scores.groupby("truth").agg(
        n=("trial", "size"), original_invalid=("original_truth_grid_rmse", lambda v: int((~np.isfinite(v)).sum())),
        guarded_invalid=("guarded_truth_grid_rmse", lambda v: int((~np.isfinite(v)).sum())),
        original_recovered=("original_truth_grid_rmse", lambda v: int((v < .02).sum())),
        guarded_recovered=("recovered", "sum")).reset_index()
    counts = pd.read_csv(root / "recovery_counts.csv")
    scores = pd.read_csv(root / "observational_scores.csv")
    cluster = scores.groupby(["cluster", "model"]).rmse_dex.mean().unstack()
    age = scores.groupby("cluster").age_Myr.first().sort_values()
    names = {"fixed_skumanich": "Fixed exponent", "power_law": "Power law", "quadratic": "Quadratic",
             "pysr_default": "Symbolic default", "pysr_validation": "Symbolic validation"}
    recovered = counts[["truth", "design", "noise_dex", *names, "any_candidate"]].rename(columns={
        **names, "truth": "Known function", "design": "Sampling", "noise_dex": "Noise (dex)",
        "any_candidate": "Any search candidate"})
    means = pd.DataFrame([{"Method": names[k], "Mean cluster RMSE (dex)": f"{summary['mean_cluster_rmse_dex'][k]:.6f}"} for k in names])
    intervals = pd.DataFrame([{"Comparison": k, "Mean difference": f"{v['mean_difference']:+.6f}",
                               "Descriptive 95% interval": f"[{v['ci95_low']:+.6f}, {v['ci95_high']:+.6f}]"}
                              for k, v in summary["paired_cluster_intervals"].items()])
    per_cluster = cluster.loc[age.index, list(names)].copy()
    per_cluster = per_cluster.map(lambda x: f"{x:.4f}").rename(columns=names)
    per_cluster.insert(0, "Age (Myr)", age.values)
    per_cluster = per_cluster.reset_index().rename(columns={"cluster": "Test cluster"})
    sensitivities = pd.read_csv(root / "sensitivity_scores.csv").groupby(["membership_cut", "color_hi", "model"]).rmse_dex.mean().unstack()
    sensitivities = sensitivities.map(lambda x: f"{x:.4f}").rename(columns=names).reset_index()
    delta = cluster.pysr_validation - cluster.pysr_default
    omission = delta.drop(["NGC1750", "NGC6819"]).mean()
    doc = f"""# Results: equation selection and stellar rotation

Study `research_v1`, completed 15 October 2026. Generated from saved tables by
`scripts/make_results_document.py`. The [protocol](PROTOCOL.md) defines the main
experiment; the [research log](RESEARCH_LOG.md) records pilots, review and follow-up.

## Research contribution

The experiment quantifies a distinction between finding an accurate candidate
equation and selecting it. For this fixed benchmark, independent validation
substantially improves functional recovery over PySR's default accuracy/complexity
rule. Validation is an established method; these results are an empirical study
of its behavior in this stellar-rotation setting, not a claim to invent it.

The observational comparison does not establish an advantage over a fitted power
law. This is a useful limit on the evidence, not proof that no more complex
stellar-rotation relation exists.

## Controlled experiment: all outcomes

There are three known functions, two sampling designs, three noise levels and
ten trials per cell: 180 searches. Each search receives 600 discovery and 200
validation observations. Both selectors see the same candidate set. A separate
80-by-50 noise-free grid evaluates predictions after selection. Recovery means
RMSE below 0.02 dex; it does not require literal algebraic identity.

Every count below has denominator **10**. The final column is a diagnostic using
known truth after evaluation, never a real-data selection method.

{markdown_table(recovered)}

Across the configured mix, recovery totals are 60/180 for the fixed-exponent
baseline, 59/180 for the fitted power law, 164/180 for the quadratic, 76/180 for
default symbolic selection and 157/180 for validation selection. The quadratic
is especially well matched to these examples: it represents the interaction law
exactly and approximates the smooth saturation law over the limited domain.
These counts do not imply that symbolic regression beats suitable parametric models.

Validation has lower true-grid error in {diag['validation_lower_error']} trials,
default selection has lower error in {diag['default_lower_error']}, and
{diag['numerically_tied']} are tied within 1e-10 dex. In
{diag['default_selection_misses_despite_accurate_candidate']} trials the default
selector misses recovery despite an accurate candidate being available. In
{diag['no_candidate_recovers']} trials no candidate meets the recovery threshold.

Two validation-selected functions are undefined at some grid points. They retain
infinite error and count as failures, rather than disappearing from averages.
Both occur in the saturation/catalogue/0.15-dex cell. Its mean validation error
against truth is therefore infinite; its finite median does not mean all runs
succeeded. See [continuous errors](../project/figures/research_v1/synthetic_recovery.png)
and [all summary statistics](../project/results/research_v1/synthetic_summary.csv).

### Why the default can miss an available equation

In installed PySR 2.7.0, `model_selection='best'` first keeps equations with
training loss no more than 1.5 times the minimum, then selects the largest score,
which measures improvement in log loss per unit complexity. It does not directly
minimize independent predictive error. With noise, the loss floor can make a
simpler, less accurate function eligible. This is a tradeoff in the configured
selection rule, not evidence that the search library is broken.

For the specified example `interaction_catalog_noise0.05_seed00`, default
selection chooses complexity 9, with 0.03287 dex error against the known law.
Validation selects complexity 15 from the same set, with 0.00451 dex error.
The corresponding validation errors are 0.05235 and 0.04536 dex. The candidate
table preserves the complete expressions, training losses and selector flags.

The injected functions, transforms, operators, finite budget and sampling
distribution all constrain this result. Ten trials per cell give limited
precision. High-noise catalogue sampling has lower validation recovery than
balanced coverage in all three functions here; more independent trials would
be needed to estimate that difference precisely.

## Observational experiment: withheld clusters

The selected sample contains 721 stars and 16 clusters. Each test cluster is
excluded from fitting and validation. Three of the remaining clusters form
age-stratified validation data; the other 12 form discovery data. Every method
uses the same discovery rows, with equal total training weight per cluster.
Three split/search seeds are averaged within each test cluster before averaging
across clusters. The 48 runs are not 48 independent astronomical populations.

{markdown_table(means)}

Paired differences are in dex; negative favors the first method. The intervals
use 20,000 resamples of the 16 paired cluster scores. They are descriptive and
do not fully account for overlapping training sets, catalogue selection or
systematic age error. Multiple comparisons are shown without a multiplicity
correction; no thresholded significance claim is made.

{markdown_table(intervals)}

Validation-selected symbolic regression is close to the power law, with an
interval spanning zero. The quadratic has the lowest point estimate, but its
interval against the power law also spans zero. Neither interval proves
equivalence. A result near 0.22 dex is substantial scatter in log period; it
cannot be converted directly into a calibrated individual-star age uncertainty.

{markdown_table(per_cluster)}

Validation beats default selection in {int((delta < 0).sum())}/16 cluster averages.
The mean difference is {delta.mean():+.5f} dex and the median is {delta.median():+.5f}
dex. As a post-hoc influence diagnostic, omitting NGC1750 and NGC6819 changes the
mean to {omission:+.5f} dex. These two difficult cases contribute much of the
observed mean improvement; they remain in every primary result. This omission
is a diagnostic, not a revised test sample.

M67 is the oldest cluster and tests age extrapolation. All 25 selected stars are
shown in its holdout plot. Better performance for one model on M67 does not
isolate weakened magnetic braking; color dependence, membership, rotation
scatter and the adopted ages also matter. The study does not infer field-star
ages from these fits.

## Training-sample sensitivity

These separate baseline fits use all other eligible clusters for training and
always test the **same primary stars**. They do not use the inner validation
split and should not be compared as another row in the primary model table.

{markdown_table(sensitivities)}

Extending the training color range to 2.0 increases mean error on the common
narrower-color test sample for every listed baseline and membership threshold.
Changing membership from 0.8 to 0.9 has a small effect here. This measures the
interaction between population selection and these model families; it does not
show that redder stars or a particular membership score are intrinsically bad data.

## Exploratory follow-up: domain validity

After observing two invalid functions, a feature-only guard checks candidates
on an 81-by-51 mesh across the declared input domain and rejects nonfinite
functions before choosing by validation error. The rule is recorded in the
research log and implemented separately. It never consults true targets to choose.

On the already inspected trials it changes {guard['changed_selections']} selections,
reduces invalid selected functions from {guard['original_invalid']} to
{guard['guarded_invalid']}, and changes recovery from {guard['original_recoveries']}/180
to {guard['guarded_recoveries']}/180. This is post-hoc reuse of the same trials,
not independent confirmation. A finite mesh also cannot prove validity at every
point. Original results remain the headline comparison.

### Fresh-seed confirmation

A fixed 30-search follow-up uses fresh seeds, all three functions, catalogue
sampling and 0.15-dex noise. It retains the original search budget and the guard
specified above. This tests a failure-prone condition, not the full benchmark.

{markdown_table(fresh_table)}

The guard changes {fresh['changed_selections']}/{fresh['n_trials']} choices.
Invalid selected functions change from {fresh['original_invalid']} to
{fresh['guarded_invalid']}; recoveries change from {fresh['original_recoveries']}
to {fresh['guarded_recoveries']}. These fresh trials provide a limited check of
the fixed procedure, not proof of continuous-domain safety. All outcomes are
retained in `project/results/domain_confirmation/` with their own manifest.

Validation recovers 21/30 functions in this fresh batch, versus 0/30 for default
selection. This supports the primary selection finding in this high-noise
condition. Because no invalid selection occurred, it does not demonstrate an
additional benefit from the domain guard. The fixed follow-up ended after all
30 trials; no further batch was added to seek a different outcome.

## Reproducibility and boundaries

The original catalogue was checked against its pinned upstream source. The run
manifest records the data hash, computational-source hashes, exact settings,
versions and seeds. All 228 searches completed. One nontrivial replay reproduced
the same 17 candidate expressions and losses. Tests verify data selection,
group separation, candidate selection, complete predictions, score calculations
and explicit accounting for failed functions. Only that one search was replayed
as an exact numerical check; the whole study was not independently replicated
on a second machine.

The raw catalogue supplies observed periods and estimated cluster ages. Quality
and membership cuts leave a selected population with rotational scatter and
possible contaminants. The model predicts period given age and color; inverting
it requires a separate uncertainty model and age-validation study.

The next scientifically useful extension would broaden the domain-guard test
to additional functions, expand independent old-cluster/field-star validation, and compare
probabilistic rotation models. More searches on the same catalogue alone cannot
establish a new physical law. The fixed experiment was completed regardless of
which method won, rather than extended until a favorable headline appeared.

## Sources and assistance

- Van-Lane et al. (2025), [ChronoFlow](https://arxiv.org/abs/2412.12244), supplies
  the catalogue and the distributional view of stellar rotation.
- Cranmer (2023), [PySR](https://arxiv.org/abs/2305.01582), supplies symbolic-regression
  software; see the [API](https://pysr.ai/api) and pinned installed source.
- Barnes et al. (2016), [M67 rotation](https://arxiv.org/abs/1603.09179), and
  Dungee et al. (2022), [cool stars in M67](https://arxiv.org/abs/2211.01377),
  provide relevant observational context; neither is directly replicated here.

This revision used AI assistance for code, research design, execution, figures
and writing. Observational measurements were taken from the cited catalogue.
The repository preserves the work for inspection, correction and independent replay.
"""
    (ROOT / "docs/RESULTS.md").write_text(doc)
    print(ROOT / "docs/RESULTS.md")


if __name__ == "__main__":
    main()

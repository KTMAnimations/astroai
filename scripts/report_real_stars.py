#!/usr/bin/env python3
"""Build observational figures and a complete results document from frozen fits."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from astrogyro.data import ROOT
from astrogyro.distributions import components
from make_figures import save, style
from make_results_document import markdown_table

LABELS = {"gaussian_power": "Power law + Gaussian", "gaussian_quadratic": "Quadratic + Gaussian",
          "gaussian_hetero": "Quadratic + variable scatter", "gaussian_rbf": "Smooth residual model",
          "mixture_power": "Power law + mixture", "mixture_quadratic": "Quadratic + mixture",
          "validation_selected": "Validation-selected family"}
COLORS = {k: "#929ba9" for k in LABELS}
COLORS.update(gaussian_power="#25344b", mixture_quadratic="#226cb5", validation_selected="#39956b")


def ensemble_quantiles(models, X, probabilities):
    arrays = [components(m, X) for m in models]
    mu = np.concatenate([a[0] for a in arrays], axis=1)
    sig = np.concatenate([a[1] for a in arrays], axis=1)
    weight = np.concatenate([a[2]/len(models) for a in arrays], axis=1)
    output = []
    for p in probabilities:
        lo, hi = np.min(mu-12*sig, axis=1), np.max(mu+12*sig, axis=1)
        for _ in range(50):
            mid = (lo+hi)/2
            below = (weight*ndtr((mid[:, None]-mu)/sig)).sum(axis=1) < p
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        output.append((lo+hi)/2)
    return np.column_stack(output)


def make_figures(root, out, summary):
    style()
    names = [k for k in LABELS if k != "gaussian_power"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, title in zip(axes, ["nll", "crps"], ["Negative log likelihood (nats)", "CRPS (dex)"]):
        for i, name in enumerate(names):
            s = summary["paired_intervals"][f"{name}_minus_gaussian_power_{metric}"]
            ax.errorbar(s["mean_difference"], i, xerr=[[s["mean_difference"]-s["ci95_low"]], [s["ci95_high"]-s["mean_difference"]]],
                        fmt="o", color=COLORS[name], capsize=4, ms=7)
        ax.axvline(0, ls="--", color="#aeb5bf");ax.grid(axis="x")
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names], fontsize=9);ax.invert_yaxis()
        ax.set(title=title, xlabel="Difference from power-law Gaussian\nLeft: better predictive distribution")
    fig.suptitle("Predictive distributions improve on withheld real clusters\nPaired means and descriptive 95% intervals; 16 clusters, three seeds each", fontsize=15)
    save(fig, out, "distribution_scores")

    chosen = ["gaussian_power", "mixture_quadratic", "validation_selected"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(2)
    for i, name in enumerate(chosen):
        m = summary["mean_cluster_scores"][name]
        axes[0].bar(x+(i-1)*.23, [100*m["coverage68"], 100*m["coverage90"]], .21, color=COLORS[name], label=LABELS[name])
    for xx, level in zip(x, [68, 90]):
        axes[0].plot([xx-.4, xx+.4], [level, level], "--", color="#d56a3a", lw=2)
    axes[0].set_xticks(x, ["68% interval", "90% interval"]);axes[0].set_ylim(0, 100)
    axes[0].set(ylabel="Observed held-out coverage (%)", title="Average across the 16 clusters")
    axes[0].legend(fontsize=9, loc="lower right")
    widths = [summary["mean_cluster_scores"][n]["width68_dex"] for n in chosen]
    axes[1].barh(range(3), widths, color=[COLORS[n] for n in chosen], height=.6)
    axes[1].set_yticks(range(3), [LABELS[n] for n in chosen], fontsize=10);axes[1].invert_yaxis()
    axes[1].set(xlabel="Mean 68% prediction-interval width (dex)", title="Narrower intervals, closer coverage", xlim=(0, .55))
    for i, v in enumerate(widths):axes[1].text(v+.012, i, f"{v:.3f}", va="center")
    fig.suptitle("Model the population's spread, not just one average period\nDashed orange marks nominal coverage; cluster-specific behavior still varies", fontsize=14)
    save(fig, out, "coverage")

    ages = pd.read_csv(root/"cluster_age_estimates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    for ax, name in zip(axes, ["gaussian_power", "validation_selected"]):
        d = ages[ages.model == name]
        ax.plot([80, 5000], [80, 5000], "--", color="#aeb5bf", lw=1)
        ax.scatter(d.reference_age_myr, d.inferred_age_myr, s=55, color=COLORS[name])
        for _, r in d.iterrows():
            if abs(np.log10(r.inferred_age_myr/r.reference_age_myr)) > .2 or r.cluster == "M67":
                offset = (-5, 5) if r.cluster == "M67" else (5, 5)
                ax.annotate(r.cluster, (r.reference_age_myr, r.inferred_age_myr), xytext=offset,
                            textcoords="offset points", fontsize=8, ha="right" if r.cluster == "M67" else "left")
        value = summary["age_scores"][name]["median_absolute_fractional_error"]*100
        ax.set(xscale="log", yscale="log", xlim=(80, 5500), ylim=(80, 5500), xlabel="Catalogue reference age (Myr)",
               ylabel="Inferred withheld-cluster age (Myr)", title=f"{LABELS[name]}\nMedian absolute fractional error: {value:.1f}%")
        ax.grid(alpha=.6)
    fig.suptitle("Age recovery from the withheld stars' periods and colors\nAll 16 clusters retained; point estimates averaged in log age over three splits", fontsize=14)
    save(fig, out, "age_recovery")

    external = pd.read_csv(root/"external_stars.csv")
    colors = np.linspace(.65, 1.5, 250)
    X = np.column_stack([np.full(len(colors), np.log10(4.)), np.log10(colors)])
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["gaussian_power", "mixture_quadratic"]:
        models = [json.loads(p.read_text())["models"][name] for p in sorted((root/"folds").glob("holdout_M67_seed*/result.json"))]
        q = 10**ensemble_quantiles(models, X, [.05, .5, .95])
        ax.plot(colors, q[:, 1], color=COLORS[name], lw=2, label=LABELS[name]+" median")
        if name == "mixture_quadratic":
            ax.fill_between(colors, q[:, 0], q[:, 2], color=COLORS[name], alpha=.15, label="Mixture 90% predictive band")
    ax.errorbar(external.BPRP_0, external.Prot, yerr=external.period_err, fmt="o", mfc="white", mec="#25344b",
                color="#25344b", capsize=2, ms=5, label="25 external Gruner et al. stars", zorder=5)
    ax.set(xlabel="Dereddened Gaia DR3 BP-RP", ylabel="Rotation period (days)", ylim=(0, None),
           title="New stars, with every M67 star excluded from model fitting\nThree frozen models combined equally; reference age 4.0 Gyr")
    ax.legend(fontsize=9, loc="upper left");ax.grid(alpha=.6)
    save(fig, out, "external_m67")


def make_document(root, summary):
    external = json.loads((root/"external_summary.json").read_text())
    rows = []
    for name in LABELS:
        m = summary["mean_cluster_scores"][name]
        rows.append({"Model": LABELS[name], "NLL": f"{m['nll']:.4f}", "CRPS": f"{m['crps']:.4f}",
                     "Mean RMSE": f"{m['rmse_dex']:.4f}", "Median MAE": f"{m['mae_dex']:.4f}",
                     "68% coverage": f"{100*m['coverage68']:.1f}%", "90% coverage": f"{100*m['coverage90']:.1f}%"})
    age_rows = pd.read_csv(root/"cluster_age_estimates.csv")
    ages = age_rows[age_rows.model == "gaussian_power"][["cluster", "reference_age_myr", "inferred_age_myr"]].rename(columns={"inferred_age_myr": "Power law"})
    for name in ["mixture_quadratic", "validation_selected"]:
        ages = ages.merge(age_rows[age_rows.model == name][["cluster", "inferred_age_myr"]].rename(columns={"inferred_age_myr": LABELS[name]}), on="cluster")
    ages = ages.round(0).sort_values("reference_age_myr")
    external_table = pd.DataFrame(external["scores"]).T.rename(index=LABELS).round(4).reset_index().rename(columns={"index": "Model"})
    primary = summary["paired_intervals"]["mixture_quadratic_minus_gaussian_power_nll"]
    crps_ci = summary["paired_intervals"]["mixture_quadratic_minus_gaussian_power_crps"]
    age_ci = summary["paired_intervals"]["validation_selected_minus_gaussian_power_age_error"]
    text = f"""# Real-star results: rotation distributions and cluster ages

The new model improves prediction on real stellar observations. Across the same
721 stars, 16 held-out clusters and 48 splits as the initial study, the quadratic
mixture improves both negative log likelihood (NLL) and continuous ranked
probability score (CRPS) over a Gaussian power-law baseline. It also improves
median-prediction absolute error. Its mean-prediction RMSE does not improve.

This follow-up models a distribution of rotation periods instead of expecting
one curve to represent both the narrow slow sequence and the broader tail. The
mean of the slow component remains an explicit quadratic in log age and color.
A second Gaussian lies at shorter log periods; its weight depends on age and
color. These statistical components do not identify individual stars as binaries
or assign physical evolutionary states.

[Protocol](REAL_STAR_PROTOCOL.md) · [Prediction tool](../scripts/predict_rotation.py) ·
[Complete result bundle](../project/results/real_stars_v2/)

## Fixed comparison on real observations

Every row uses the same discovery, validation and test clusters. Test-cluster
periods never select parameters, penalties or model families. Regularization and
the final family selector use only the three validation clusters in each fold.
All stars and difficult clusters are retained. The Gaussian power-law point
predictions reproduce the initial study's fitted-power-law scores.

Lower NLL, CRPS, RMSE and MAE are better. NLL is a log-density score in nats and
can be negative; the other errors are in dex of rotation period. Coverage is
averaged within seeds, then equally over clusters. MAE evaluates the predicted
median; RMSE evaluates the predicted mean of log period.

{markdown_table(pd.DataFrame(rows))}

The predeclared primary difference, quadratic mixture minus Gaussian power law,
is **{primary['mean_difference']:+.4f} nats**, with descriptive paired 95% interval
**[{primary['ci95_low']:+.4f}, {primary['ci95_high']:+.4f}]**. The mixture improves
NLL in 15 of 16 cluster averages. CRPS improves by approximately 10.3%, with paired
difference {crps_ci['mean_difference']:+.4f} dex and interval
[{crps_ci['ci95_low']:+.4f}, {crps_ci['ci95_high']:+.4f}]. This second proper score
supports the distributional improvement rather than a claim based only on a
sharper density peak.

![Distribution score comparisons](../project/figures/real_stars_v2/distribution_scores.png)

The mixture's 68% and 90% prediction intervals achieve 68.4% and 90.0% coverage
on average. The baseline's corresponding coverage is 80.7% and 92.1%. The mixture's
mean 68% interval is narrower (0.291 versus 0.458 dex). This is a useful observed
calibration result, not a guarantee for every color, cluster or future population.
These intervals describe population spread, not uncertainty on the mean relation.

![Prediction interval coverage](../project/figures/real_stars_v2/coverage.png)

## Recovering withheld cluster ages

Each test cluster's periods and colors are evaluated over the fixed 80-5000 Myr
grid; the maximizing likelihood gives a point estimate. Its catalogue age is
used only afterward for scoring. Across three seeds, inferred log ages are
averaged before comparison. Catalogue ages are the original study's averages
of published isochrone estimates, not new gyrochronological labels.

Median absolute fractional error is **18.3%** for the Gaussian power law,
**12.5%** for the quadratic mixture, and **9.9%** for the family selected using
validation data. The corresponding mean absolute log-age errors are 0.1126,
0.0952 and 0.0921 dex. All 16 clusters remain in these calculations.

{markdown_table(ages)}

![Age recovery](../project/figures/real_stars_v2/age_recovery.png)

The median improvement is encouraging, but the paired interval on the selected
model's change in mean absolute log-age error is [{age_ci['ci95_low']:+.4f},
{age_ci['ci95_high']:+.4f}] dex and spans zero. M34, NGC1750 and NGC2516 remain
difficult. One of the 48 selected age fits reaches the grid boundary. The saved
tables include every boundary flag. The product likelihood does not account for
shared systematic age errors or all model-parameter uncertainty, so no calibrated
cluster-age confidence intervals or 10%-accurate individual-star ages are claimed.

## External observations: 25 new M67 stars

The external sample comes from Gruner, Barnes & Weingrill (2023), via a pinned
gyro-interp data file. It supplies Gaia DR3 identifiers, dereddened BP-RP, periods
and period uncertainties. The selected 25 stars share **no IDs with the entire
original 8,412-entry catalogue**. All M67 stars are excluded from both discovery
and validation in the three frozen models used here. There is no refitting or
model selection on external errors.

At the paper's nominal 4.0 Gyr age:

{markdown_table(external_table)}

Mean-prediction RMSE falls from **0.1537 to 0.1174 dex**, approximately **23.6%**,
for the quadratic mixture versus the Gaussian power law. CRPS falls from 0.0913
to 0.0575 dex. The ordinary quadratic Gaussian is competitive (0.1249 dex RMSE)
and has lower median MAE than the mixture; the mixture does not win every metric.

![External M67 predictions](../project/figures/real_stars_v2/external_m67.png)

The quadratic mixture infers 3.38 Gyr for this external group, versus 4.75 Gyr for
the power law and 4.02 Gyr for the ordinary quadratic Gaussian. The adopted 4.0 Gyr
age is approximate; the source also discusses a 3.6 Gyr isochrone fit. A saved
3.87 Gyr sensitivity evaluates the same stars and frozen models. We do not choose
an age scale based on which makes a method look best.

This is a separate-star check within one old cluster, not a new cluster-age
calibration or field-star validation. The authors' single-main-sequence selection
differs from the broader primary sample. Its 68% mixture intervals cover 92.0%
of these external observations, showing overcoverage for this selected population.
The published half-period-sequence corrections are retained unchanged. Every
selected star is plotted, including the fast-rotation outlier. Source and
reddening details are in the [external data record](../project/data/external/README.md).

## Use the fitted model

`application_model.json` fits the quadratic mixture to all 721 primary stars
after the evaluation. It uses the most frequent inner-validation penalty (ties
favor stronger regularization). This application fit is separate from all
held-out/external metrics. It predicts the rotation distribution at a supplied
age and dereddened Gaia color:

```bash
.venv/bin/python scripts/predict_rotation.py --age-myr 1000 --bp-rp 1.0
```

For that example, the predicted median is about 11.1 days, with a 68% population
interval of 7.5-12.9 days. The script enforces the observed 135-3870 Myr and
0.65 < BP-RP < 1.5 application domain. This is not an individual-star age calculator.

For a coeval group with measured periods and colors, the cluster-age tool needs
`Prot` (days), `BPRP_0` and optionally `GaiaDR3_ID` columns. The external example
uses fits that exclude all M67 training and validation stars:

```bash
.venv/bin/python scripts/infer_cluster_age.py project/results/real_stars_v2/external_stars.csv --holdout-cluster M67
```

It returns 3.38 Gyr. The input catalogue age is never used to estimate this age.
Omitting the holdout option uses the full-data application fit for a new group.
Known training-ID overlap is rejected when identifiers are supplied. The output
reports age-grid boundary hits; variation between fits is not a confidence interval.

## Reproduce and inspect

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_real_star_study.py
.venv/bin/python scripts/evaluate_external_m67.py
.venv/bin/python scripts/fit_rotation_model.py
.venv/bin/python scripts/report_real_stars.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_real_star_reproducibility.py
.venv/bin/python -m pytest -q
```

The runner resumes completed folds, checks source and split hashes, and saves all
candidate parameters and validation choices. Its manifest and the earlier study
remain intact. Tests verify likelihood gradients, normalized densities, CRPS,
quantiles, split separation, complete predictions, unchanged baseline scores,
external non-overlap, frozen model hashes and age-grid maxima.
An independent replay refitted all 528 candidate configurations across 48 folds;
all 17,421 compared numerical values matched exactly in the recorded environment.
The saved `reproducibility.json` records this check. Cluster-age command tests
also verify that reference-age columns cannot supply the inferred age and known
training-ID overlap is rejected.

These experiments were designed after inspecting the earlier results, and the
same observational catalogue remains an exploratory development benchmark.
Descriptive bootstrap intervals treat clusters as the resampling units but do
not include all dependence from overlapping training sets or systematic age
uncertainty. No result establishes a new physical braking law. The concrete
progress is a better tested probability model on real observations, an external
sample check, and a usable rotation-prediction tool.

## Scientific context and contributions

[ChronoFlow](https://arxiv.org/abs/2412.12244) and
[gyro-interp](https://arxiv.org/abs/2303.08830) motivate modeling rotational
dispersion; this study implements a simpler parametric comparison, not their
full published methods. [Gruner et al. (2023)](https://arxiv.org/abs/2305.16997)
provide the external observations. Published data and existing statistical
methods are credited to their sources. This revision used AI assistance for
research, implementation, experiment execution, diagnosis, figures and writing.
"""
    (ROOT/"docs/REAL_STAR_RESULTS.md").write_text(text)


def main():
    root = ROOT/"project/results/real_stars_v2"
    out = ROOT/"project/figures/real_stars_v2";out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/"summary.json").read_text())
    make_figures(root, out, summary)
    make_document(root, summary)
    print("Saved four figures and docs/REAL_STAR_RESULTS.md")


if __name__ == "__main__": main()

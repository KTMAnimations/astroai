# Research log

## 2026-10-15: Audit and redesign

The initial audit reproduced the legacy fits but found unsupported scientific
claims, a failed complex-truth control, a NaN-scoring error, broken verification,
missing producer scripts, inconsistent sample descriptions, and clipped M67
plots. The current study uses a new package and output directory so those
historical results remain distinguishable from the verified follow-up.

Checked the original catalogue against the upstream ChronoFlow file at commit
`f0b766f7d057c4e8f9a64a867414d25c5d985cd6`: the files match byte for byte.
The upstream instructions confirm the four photometric/astrometric quality cuts.
The paper specifies HDBScan membership with a source-probability fallback.

The primary selection produces 721 stars in 16 clusters. The restriction to
100 Myr and older, 0.65 < BP-RP < 1.5, membership >= 0.8, and >= 10 stars per
cluster was chosen before running the new model comparisons. It supports a
limited population-level prediction experiment, not a universal gyrochronology law.

## Execution pilots (excluded from results)

A short, balanced-input interaction trial checked the end-to-end code: 20 search
iterations, 6 populations, 100 mutation cycles. It completed in 20.9 seconds
including initialization. A separate-seed trial with the planned full budget
(100 iterations, 12 populations, 200 cycles) completed in 26.7 seconds.
The pilot supports running the fixed main budget on this machine. Pilot outputs
live in ignored `tmp/` directories and do not enter the study tables.

The main configuration has 180 controlled searches and 48 cluster-holdout
searches (16 clusters times 3 split/search seeds). Independent searches may run
in separate processes; every individual search uses deterministic serial PySR.
The protocol, seeds and selection criteria are fixed before the main execution.

The study tests recovery in function space, because algebraically different
expressions can make equivalent predictions. It reports continuous errors and
an explicit recovery threshold rather than claiming expression recurrence proves
physical truth.

## Main-run completion and reporting review

All 180 synthetic and 48 observational searches completed. Replaying the
nontrivial `interaction_balanced_noise0.05_seed00` trial reproduced the same
17 candidate equations and losses. The source hashes and all original trial
outputs are retained.

The reporting review found two validation-selected synthetic equations that
were finite on the sampled validation inputs but undefined elsewhere on the
full evaluation grid. Both arose in noisy, catalogue-sampled saturation trials.
The experiment correctly assigned infinite error and marked them unrecovered.
Strict JSON encoded these scores as null; a first draft of the summary then
treated the nulls as missing. The reporting script now restores infinite error,
counts invalid functions explicitly, and includes those trials in comparisons.
The trial results and selection rules were not changed. A regression check
ensures that every synthetic comparison, including failures, is accounted for.

## Exploratory domain-check follow-up (rule specified before evaluation)

The two failures motivate a separate, post-hoc reliability check. For each saved
candidate set, first reject functions with any nonfinite prediction on a
feature-only 81-by-51 mesh spanning the declared age/color domain. Then choose
the remaining function with minimum saved validation error, breaking ties by
complexity and candidate ID. The selector never receives true grid targets or
test errors. Evaluate the frozen choice on the original 80-by-50 truth grid.

This reuses already inspected trials and is an exploratory demonstration, not
an independent confirmation. It does not replace the primary results. A finite
mesh cannot prove that a function is defined everywhere in a continuous domain.

The reused-trial follow-up changed two choices, removed the two observed domain
failures, and raised recovery from 157/180 to 159/180. No main result was replaced.

## Fresh-seed domain-check confirmation (fixed before running)

To test the guard beyond the trials that motivated it, run 30 fresh searches:
all three known functions, catalogue sampling, 0.15 dex noise, ten trials each.
Use seed base 20361015, outside the main study's seed range. Retain the same
operators, budget, training/validation sizes and feature-only guard. Compare
invalid counts, recovery counts and continuous errors before and after the
guard. This condition targets the sampling/noise regime where the failures
occurred; it does not estimate an all-condition average. Complete all 30 trials
regardless of outcome and report failures as well as improvements.

### Confirmation outcomes

All 30 searches completed (247 seconds of summed search time). Default selection
recovered 0/30 functions, validation recovered 21/30, and a recovering candidate
was present in 28/30 searches. No selected function was invalid on the evaluation
grid. The domain guard changed no choices, leaving 21/30 recoveries. The fresh
trials support the main selection finding in this high-noise catalogue condition,
but do not demonstrate an extra benefit from the domain guard. No further seed
batch was added to seek a different result.

The full study now contains 258 searches, plus two excluded execution pilots and
one exact-seed replay. The seven current figures were visually inspected, and
the four-page PDF was rendered page by page. Legacy claims are labeled historical;
the root README now reports the verified results and their limitations.

## 2026-10-15: Distribution models on real stars

After the initial study, the follow-up protocol specified six distribution
families and reused all 48 cluster splits. No star, cluster, period or residual
was removed to improve the new scores. The comparison fits Gaussian power-law,
quadratic and variable-scatter models, two slow/fast mixture models, and a smooth
residual model. Three mixture penalties and four residual-model penalties are
selected only on validation clusters. All 528 candidate configurations converged.
The primary comparison was fixed as quadratic-mixture minus power-law-Gaussian
held-out negative log likelihood. This is an exploratory extension informed by
the initial benchmark, not an independent preregistration.

The quadratic mixture improves mean cluster NLL by 0.5021 nats, with paired
descriptive difference interval [-0.6694, -0.3346], and improves 15 of 16 cluster
averages. CRPS decreases by 10.3%. Its 68% and 90% intervals achieve average
coverage of 68.4% and 90.0%, with a narrower mean 68% interval than the power law.
Mean-log-period RMSE remains slightly worse: 0.2251 versus 0.2195 dex. Every model,
metric and cluster remains in the saved tables.

Likelihood age recovery reduces median absolute fractional cluster-age error
from 18.3% to 12.5% for the quadratic mixture and 9.9% for validation family
selection. The paired interval on mean log-age-error improvement spans zero.
Large errors for several clusters and one selected boundary hit remain visible.
Catalogue reference ages are averaged published isochrone estimates; their
uncertainty and shared systematics are not removed by the number of stars.

## External M67 observations and usable predictions

Source research identified Gruner et al. (2023) Table C.2, with periods,
uncertainties, Gaia DR3 IDs and dereddened Gaia colors. The pinned gyro-interp
copy has 47 rows. The fixed color domain leaves 28; removing three IDs present
anywhere in the original 8,412-entry catalogue leaves 25. The selection uses no
period/residual cut. Published half-period-sequence corrections are already in
the source periods and are retained unchanged. The provenance record specifies
the source's reddening convention and the exact upstream commit and file hash.

The external scoring specification was recorded before computing its errors.
All M67 members are absent from discovery and validation in the three frozen
fits used for this check. At the nominal 4.0 Gyr reference, the quadratic mixture
reduces mean-log-period RMSE from 0.1537 to 0.1174 dex, a 23.6% reduction. Its
inferred group age is 3.38 Gyr. The ordinary quadratic Gaussian is competitive
at 0.1249 dex RMSE and gives a lower median-prediction MAE and a 4.02 Gyr age.
The 3.87 Gyr sensitivity is saved separately; no age scale is chosen to favor a
model. This is an additional-star check in one cluster, not independent field-star
validation. Nominal 68% mixture intervals cover 92% of the selected external
stars, exposing overcoverage for this different population.

A separate application mixture was fitted to all 721 primary stars using the
most frequent validation-selected penalty (0.1). It does not supply held-out
metrics. The rotation command predicts a median of 11.1 days with a 7.5-12.9 day
68% population interval at 1,000 Myr and BP-RP = 1.0. The cluster-age command
reproduces the 3.38 Gyr external estimate without consulting a reference-age
column. It rejects known training-ID overlap and invalid rows instead of silently
removing them. Neither tool provides calibrated individual-star age uncertainty.

## Follow-up verification and presentation

A full independent numerical replay refitted all 528 candidates across 48 folds.
All 17,421 compared numerical values, including parameters and validation scores,
matched exactly in the recorded environment; the maximum absolute difference was
zero. `real_stars_v2/reproducibility.json` records the comparison and source hash.
The 29-test suite passed, including analytic likelihood-gradient checks, density
normalization, CRPS and quantiles, complete star/model coverage, original baseline
agreement, split/source hashes, external non-overlap, age-grid maxima and CLI
checks against age-label leakage and training overlap. Dependency checks passed.

The four new figures were inspected, and the updated four-page PDF was rendered
and visually checked page by page. The GitHub README and science-fair guide now
lead with the real-star follow-up, with both favorable and unfavorable metrics
visible. The initial study, its report, figures and numerical results remain
unchanged. New commit messages describe the scientific work; historical commit
messages are preserved.

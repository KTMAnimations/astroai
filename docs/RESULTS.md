# Results: equation selection and stellar rotation

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

| Known function | Sampling | Noise (dex) | Fixed exponent | Power law | Quadratic | Symbolic default | Symbolic validation | Any search candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interaction | balanced | 0.0 | 0 | 0 | 10 | 10 | 10 | 10 |
| interaction | balanced | 0.05 | 0 | 0 | 10 | 1 | 10 | 10 |
| interaction | balanced | 0.15 | 0 | 0 | 9 | 0 | 8 | 8 |
| interaction | catalog | 0.0 | 0 | 0 | 10 | 10 | 10 | 10 |
| interaction | catalog | 0.05 | 0 | 0 | 10 | 0 | 10 | 10 |
| interaction | catalog | 0.15 | 0 | 0 | 7 | 0 | 4 | 8 |
| saturation | balanced | 0.0 | 0 | 0 | 10 | 10 | 10 | 10 |
| saturation | balanced | 0.05 | 0 | 0 | 10 | 9 | 10 | 10 |
| saturation | balanced | 0.15 | 0 | 0 | 7 | 0 | 7 | 8 |
| saturation | catalog | 0.0 | 0 | 0 | 10 | 10 | 10 | 10 |
| saturation | catalog | 0.05 | 0 | 0 | 10 | 6 | 10 | 10 |
| saturation | catalog | 0.15 | 0 | 0 | 6 | 0 | 3 | 7 |
| skumanich | balanced | 0.0 | 10 | 10 | 10 | 10 | 10 | 10 |
| skumanich | balanced | 0.05 | 10 | 10 | 10 | 0 | 10 | 10 |
| skumanich | balanced | 0.15 | 10 | 10 | 8 | 0 | 8 | 10 |
| skumanich | catalog | 0.0 | 10 | 10 | 10 | 10 | 10 | 10 |
| skumanich | catalog | 0.05 | 10 | 10 | 10 | 0 | 10 | 10 |
| skumanich | catalog | 0.15 | 10 | 9 | 7 | 0 | 7 | 9 |

Across the configured mix, recovery totals are 60/180 for the fixed-exponent
baseline, 59/180 for the fitted power law, 164/180 for the quadratic, 76/180 for
default symbolic selection and 157/180 for validation selection. The quadratic
is especially well matched to these examples: it represents the interaction law
exactly and approximates the smooth saturation law over the limited domain.
These counts do not imply that symbolic regression beats suitable parametric models.

Validation has lower true-grid error in 125 trials,
default selection has lower error in 5, and
50 are tied within 1e-10 dex. In
94 trials the default
selector misses recovery despite an accurate candidate being available. In
10 trials no candidate meets the recovery threshold.

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

| Method | Mean cluster RMSE (dex) |
| --- | --- |
| Fixed exponent | 0.218218 |
| Power law | 0.219500 |
| Quadratic | 0.214377 |
| Symbolic default | 0.234485 |
| Symbolic validation | 0.216981 |

Paired differences are in dex; negative favors the first method. The intervals
use 20,000 resamples of the 16 paired cluster scores. They are descriptive and
do not fully account for overlapping training sets, catalogue selection or
systematic age error. Multiple comparisons are shown without a multiplicity
correction; no thresholded significance claim is made.

| Comparison | Mean difference | Descriptive 95% interval |
| --- | --- | --- |
| pysr_default_minus_power_law | +0.014985 | [-0.004026, +0.038193] |
| pysr_validation_minus_power_law | -0.002520 | [-0.008452, +0.003882] |
| fixed_skumanich_minus_power_law | -0.001282 | [-0.006686, +0.004305] |
| quadratic_minus_power_law | -0.005123 | [-0.011010, +0.001397] |
| validation_minus_default | -0.017505 | [-0.038591, -0.000138] |

Validation-selected symbolic regression is close to the power law, with an
interval spanning zero. The quadratic has the lowest point estimate, but its
interval against the power law also spans zero. Neither interval proves
equivalence. A result near 0.22 dex is substantial scatter in log period; it
cannot be converted directly into a calibrated individual-star age uncertainty.

| Test cluster | Age (Myr) | Fixed exponent | Power law | Quadratic | Symbolic default | Symbolic validation |
| --- | --- | --- | --- | --- | --- | --- |
| M50 | 135.0 | 0.3829 | 0.3871 | 0.3789 | 0.3809 | 0.3788 |
| M34 | 161.0 | 0.2637 | 0.2862 | 0.2717 | 0.2559 | 0.2641 |
| M35 | 188.0 | 0.3376 | 0.3394 | 0.3288 | 0.3525 | 0.3453 |
| NGC2516 | 263.0 | 0.2842 | 0.2836 | 0.2890 | 0.3012 | 0.2934 |
| NGC1750 | 322.0 | 0.3392 | 0.3162 | 0.2941 | 0.4434 | 0.3359 |
| NGC3532 | 398.0 | 0.0936 | 0.0997 | 0.0928 | 0.0972 | 0.0869 |
| M37 | 446.0 | 0.1618 | 0.1607 | 0.1572 | 0.1689 | 0.1601 |
| NGC2548 | 462.0 | 0.1109 | 0.1206 | 0.1077 | 0.0887 | 0.0994 |
| NGC2281 | 608.0 | 0.1844 | 0.1877 | 0.2015 | 0.1899 | 0.1923 |
| Praesepe | 718.0 | 0.1412 | 0.1387 | 0.1337 | 0.1561 | 0.1370 |
| NGC6866 | 735.0 | 0.1146 | 0.1127 | 0.0940 | 0.1611 | 0.1033 |
| Hyades | 775.0 | 0.2869 | 0.2860 | 0.2876 | 0.2973 | 0.2867 |
| NGC6811 | 1004.0 | 0.1371 | 0.1466 | 0.1706 | 0.1355 | 0.1702 |
| NGC6819 | 2310.0 | 0.2870 | 0.3064 | 0.3167 | 0.4133 | 0.3008 |
| Ruprecht147 | 2454.0 | 0.2150 | 0.2065 | 0.1885 | 0.1902 | 0.1934 |
| M67 | 3870.0 | 0.1514 | 0.1340 | 0.1175 | 0.1196 | 0.1241 |

Validation beats default selection in 10/16 cluster averages.
The mean difference is -0.01750 dex and the median is -0.00750
dex. As a post-hoc influence diagnostic, omitting NGC1750 and NGC6819 changes the
mean to -0.00429 dex. These two difficult cases contribute much of the
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

| membership_cut | color_hi | Fixed exponent | Power law | Quadratic |
| --- | --- | --- | --- | --- |
| 0.5 | 1.5 | 0.2218 | 0.2221 | 0.2198 |
| 0.5 | 2.0 | 0.2329 | 0.2347 | 0.2240 |
| 0.8 | 1.5 | 0.2179 | 0.2193 | 0.2150 |
| 0.8 | 2.0 | 0.2274 | 0.2312 | 0.2189 |
| 0.9 | 1.5 | 0.2185 | 0.2198 | 0.2162 |
| 0.9 | 2.0 | 0.2292 | 0.2327 | 0.2193 |

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

On the already inspected trials it changes 2 selections,
reduces invalid selected functions from 2 to
0, and changes recovery from 157/180
to 159/180. This is post-hoc reuse of the same trials,
not independent confirmation. A finite mesh also cannot prove validity at every
point. Original results remain the headline comparison.

### Fresh-seed confirmation

A fixed 30-search follow-up uses fresh seeds, all three functions, catalogue
sampling and 0.15-dex noise. It retains the original search budget and the guard
specified above. This tests a failure-prone condition, not the full benchmark.

| truth | n | original_invalid | guarded_invalid | original_recovered | guarded_recovered |
| --- | --- | --- | --- | --- | --- |
| interaction | 10 | 0 | 0 | 8 | 8 |
| saturation | 10 | 0 | 0 | 5 | 5 |
| skumanich | 10 | 0 | 0 | 8 | 8 |

The guard changes 0/30 choices.
Invalid selected functions change from 0 to
0; recoveries change from 21
to 21. These fresh trials provide a limited check of
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

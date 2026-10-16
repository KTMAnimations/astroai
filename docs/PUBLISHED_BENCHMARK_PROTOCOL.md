# Published-model reproduction and matched benchmark

Specified 16 October 2026 before computing the new paper age reproductions or
held-out comparison scores. The earlier catalogue results are known; this is an
exploratory comparison with a fixed analysis, not an untouched confirmatory study.

## Questions and source implementation

1. Does the actual released ChronoFlow network reproduce its tutorial and the
   numerical stellar/cluster age examples reported in Van-Lane et al. (2025)?
2. How does this repository's quadratic mixture compare with the ChronoFlow
   architecture when both use the same discovery, validation and test stars?

Use the authors' code and trained weights at commit
`f0b766f7d057c4e8f9a64a867414d25c5d985cd6`, already pinned by the repository's
submodule. Record file hashes and versions. The upstream release contains one
final checkpoint and inference code, but no individual cross-validation weights
or complete training script. Therefore distinguish **released-model reproduction**
from **matched-data retraining of the published architecture**. Neither is a
claim to reproduce the paper's entire cross-validation result.

Sources: [paper, Sections IV–V](https://arxiv.org/html/2412.12244v2),
[released implementation](https://github.com/philvanlane/chronoflow/tree/f0b766f7d057c4e8f9a64a867414d25c5d985cd6),
[tutorial](https://github.com/philvanlane/chronoflow/blob/f0b766f7d057c4e8f9a64a867414d25c5d985cd6/tutorials/01_stellar_age_inference.ipynb).

## Released-model reproduction

Reproduce the upstream tutorial likelihood and priors using unmodified inference
code. Compare a vectorized adapter against that scalar implementation before
using the adapter for larger grids. Use the published zuko version 1.1.0 and
record compatibility differences in Python, PyTorch and numerical libraries.

Evaluate the Sun at period 26.09 days and color 0.82 with the released default
log color error -1.55, membership 1 and outlier probability 0.05. The published
reference median and 68% interval are 5.1 Gyr and [3.7, 6.8] Gyr. The paper does
not specify the solar color uncertainty; label the default as an assumption.
Reproduce the tutorial's 20-day, color-1.5 example with color error 0.03.

Apply the documented four quality cuts to the original catalogue (7,615 rows),
then use color > 0.5 for the four numerically reported excluded clusters: M34,
NGC2516, NGC1647 and NGC1750. Use each star's observed color error and the stated
HDBScan membership with source fallback. No additional membership/period cuts.
Their reference medians are 245, 132, 87 and 99 Myr. All four are explicitly
excluded from the final network according to Section V.1.1. This is stronger
separation than evaluating an ordinary training cluster with that checkpoint.

Use 2,001 points in log10(age/Myr) from 0 to 4.14. Form group likelihoods in log
space. Compute deterministic CDF quantiles, rather than Monte Carlo summaries.
Report absolute/log differences from published medians, raw posterior intervals,
and intervals after convolution with the paper's 0.06-dex intrinsic scatter.
Evaluate a 4,001-point grid to check numerical convergence. A within-5% median
match is the predeclared numerical reproduction target; retain failures.

The released code uses a uniform **log-age** prior, although Equation 6 describes
a uniform age prior. Match the released code in the primary reproduction and
report a uniform-linear-age sensitivity. The code's background likelihood is
constant without an explicit support check; our vectorized version restricts the
uniform background to the paper's [-1.75, 2.5] log-period bounds when computing
whole-distribution scores. It must match the upstream code inside those bounds.

## Matched-data training comparison

Reuse all 721 primary stars and all 48 frozen discovery/validation/test splits.
No new period/residual selection, cluster exclusions or change to catalogue ages.
Load the already frozen repository models. Train new neural spline flows on the
same discovery rows, keeping the entire test cluster out of fitting and selection.

Use the released `zuko.flows.NSF(1, 3, transforms=3)` architecture: three spline
transforms, eight bins and the library's two 64-unit hidden layers. Inputs are
log10(age/Myr), dereddened Gaia BP-RP and log10(color uncertainty); the target is
log10(period/day). Our earlier mixture uses age and color but does not use the
available color uncertainty. Report this modeling difference explicitly.

Train each flow for 5,000 full-batch Adam steps on CPU in float32. Use the paper's
flow/background likelihood with weight membership × (1 - 0.05). Two prespecified
weightings are compared: equal total weight per discovery cluster (primary, as
in our earlier study) and equal weight per star (paper-style sensitivity). Retain
both final and validation-selected checkpoints for each fit. Validation considers
only checkpoints at epochs 100, 200, …, 5,000 and uses mean validation-cluster NLL.
No test scores or inferred test ages select epochs or weightings.

Use exponential learning-rate decay from 0.001 to 0.0000005, multiplied by
0.5 × [1 + cos(2π × epoch/1000)]. The paper describes exponential decay and cosine
annealing but does not release enough code to reconstruct its optimizer, batching,
seed or exact scheduler convention. Our explicit choices are documented
retraining choices, not an assertion of identical original training. Save the
learning curves, weights, seed, elapsed time, data/split hashes and all choices.
No optimization of architecture or training settings on the outer results.

A 200-step synthetic-only timing pilot took about 0.74 seconds on this machine;
it was used to assess execution cost, not choose settings on observed outcomes.

## Scores, inference and decision rule

Primary difference: mean cluster NLL of the frozen quadratic mixture minus the
equal-cluster, validation-selected ChronoFlow architecture. Lower is better.
Also report the paper-style weighting and fixed-final-epoch results, the earlier
power-law and quadratic Gaussians, and the earlier validation-selected family.

Use identical observed periods, ages and colors for test scoring. Preserve native
model likelihoods: the flow includes its published membership/background model,
while the mixture describes all selected stars directly. Both are normalized
distributions on log period. Report CRPS, mean-log-period RMSE, median-prediction
MAE, 68%/90% interval coverage and widths, and every cluster. Compute flow CDFs
from its monotonic transform and integrate on a fixed [-4, 4] dex grid with 4,001
points; check finer-grid numerical accuracy before interpreting small differences.
Use analytic mixture scores, checked against the same numerical integration.

For cluster ages, give every method the same 80–5,000 Myr grid of 401 points and
maximize the group period likelihood. Reference age enters only subsequent
scoring. Average three inferred log ages within each cluster, retain boundary
hits and all difficult clusters. Compare paired cluster errors and bootstrap
16 cluster means, with 20,000 draws. These descriptive intervals omit shared
training dependence and systematic age calibration errors. Do not interpret them
as new physical-law evidence or calibrated individual-star age uncertainties.

Finally evaluate the existing 25 non-overlapping external M67 stars with the
three freshly trained M67-excluded flow pairs and the three frozen mixture fits.
Use 4,000 Myr, retaining 3,870 Myr as the existing sensitivity. The source lacks
per-star color errors; use the released default log error -1.55 explicitly, with
membership 1 and the same 0.05 outlier probability. The external data cannot
select checkpoints or change settings. These stars were already inspected in
the prior study; this is a fixed external comparison, not a new blind test.

Complete all 96 neural fits and report all outcomes. A better score than our
simple baseline does not imply beating the published model. A reproduction
mismatch must be diagnosed and retained; do not change inputs merely to match a
quoted number. Preserve all earlier studies and their recorded conclusions.

## Post-hoc solar mismatch diagnostic

After observing the primary solar mismatch, inspect the following fixed color
uncertainties: 0.001, 0.003, 0.01, 10^-1.55, 0.05 and 0.1, each with outlier
probability 0 or 0.05. Keep period, color, membership and prior fixed. Report the
entire grid as an input-sensitivity diagnostic. No cell replaces the primary
reproduction, and proximity to the published age does not establish which
unstated uncertainty the authors used.

## Direct released-checkpoint comparison

As a supplementary comparison specified before computing these scores, evaluate
both the unchanged released network and our frozen holdout models on the exact
same primary-sample stars in M34, NGC2516 and NGC1750. These are all clusters
shared by our 16-cluster benchmark and the paper's explicitly excluded final-model
clusters. Use measured color errors/membership for ChronoFlow and the same periods,
colors, reference ages and 80–5,000 Myr age grid for both methods. Average our
three split results; the released network is evaluated once per cluster.

This directly tests the actual released weights, but the models' calibration
samples differ. Label it **matched test inputs, unequal training data**, report
every eligible cluster, and draw no broad superiority conclusion from these
three deliberately atypical clusters. The 48-split retraining comparison remains
the controlled comparison with identical discovery and validation rows.

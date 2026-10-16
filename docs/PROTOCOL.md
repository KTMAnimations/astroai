# Experiment protocol, 15 October 2026

This protocol replaces the claims in the legacy notebooks. It records the
questions and decision rules before the new main runs. A small pilot checks
execution and runtime; pilot outputs do not enter the reported results.
The existing catalogue and legacy results have already been inspected, so this
is an exploratory follow-up, not a preregistered blind replication.

## Questions and hypotheses

1. When a known rotation law generates the data, how do noise, sampling coverage,
   and equation selection affect recovery of its predictions?
2. On observed stars, does choosing a symbolic equation with independent
   validation clusters improve predictions for a further, withheld cluster?
3. Does symbolic regression outperform fixed and fitted power-law baselines
   under the same data splits and scoring rule?

The hypothesis is that independent validation will help when extra structure
improves prediction, but may add selection noise when the simple law is adequate.
All outcomes, including failure to beat a baseline, will be reported.
Repeated selection of an expression is not a test that it is a physical law.

## Observations and scope

Use the checked-in ChronoFlow catalogue with its SHA-256 recorded in the run
manifest. Preserve Gaia IDs as strings. Apply the catalogue's photometric
exclusion flags and RUWE < 1.4. Require finite positive periods, age >= 100 Myr,
0.65 < dereddened Gaia BP-RP < 1.5, membership >= 0.8, and at least 10 selected
stars per cluster. Use HDBScan membership when available, otherwise source
membership, as described by Van-Lane et al. (2025).

These choices define an older, narrower-color sample for a tractable comparison.
They do not establish a pure slow sequence or remove all binaries/nonmembers.
There is no upper period cut, no clipping of plotted observations, and no cut
based on agreement with a model. Cluster ages are catalogue estimates, not exact
ground truth. The study predicts log rotation period at a supplied cluster age;
it does not deliver calibrated individual-star ages or detect weakened braking.

## Models and score

Let x = log10(age / Gyr), y = log10(dereddened BP-RP), z = log10(period / day).
Fit by squared error in z, with equal total weight for each discovery cluster.
The primary observational score is the mean of held-out cluster RMSEs in z.
One dex means a factor of ten; 0.1 dex means a factor of about 1.26.

- Fixed Skumanich-inspired baseline: z = a + 0.5x + y (one fitted constant).
- Fitted power law: z = a + bx + cy (three coefficients).
- Quadratic baseline: z = a + bx + cy + dx^2 + exy + fy^2 (six coefficients).
- PySR default selection: the library's accuracy/complexity rule.
- PySR validation selection: minimum mean validation-cluster RMSE among the
  same discovered candidates; ties resolve toward lower complexity.

Invalid predictions receive infinite loss; they never disappear from a score.
PySR uses serial deterministic searches with saved seeds, package versions,
operator set, budget and all candidate equations. No candidate selection uses
outer-test outcomes. The log transforms encode a power-law hypothesis; the
search does not start without scientific assumptions.

## Controlled experiment

Three known functions generate z:

- Skumanich: a + 0.5x + y, with a = log10(0.344 sqrt(1000)).
- Interaction: Skumanich + 1.5(x + 0.5)y + 0.6y^2.
- Smooth saturation: Skumanich - 0.5 log10(1 + 10^x).

These are controlled mathematical examples, not claimed descriptions of all
stars. The interaction is a representable nonlinear alternative. Saturation
smoothly reduces the age slope beyond 1 Gyr without an abrupt break.

Two input designs use the same age/color range: independent draws from the
selected catalogue's coordinates, or uniform coverage in log age and log color.
For each design, function and Gaussian log-period noise level (0, 0.05, 0.15 dex),
run 10 independent trials with 600 discovery and 200 validation observations.
Catalogue-coordinate draws may repeat inputs, but have independently generated
measurement errors. This is an experiment about sampling coverage, not a
bootstrap confidence interval on real stars.

After selection, evaluate each fitted model against the exact, noise-free
function on the same 80-by-50 grid across the full specified domain. This
includes interpolation through regions sparse in the catalogue. Define
functional recovery as grid RMSE < 0.02 dex (about a 4.7% period factor).
Report continuous errors alongside this threshold and counts with denominators.
The grid supplies evaluation only; it is never available to the selector.

## Observational experiment

Leave one entire cluster out for testing. Divide the remaining clusters into
three age strata and draw one validation cluster from each using a fixed seed.
Fit all models on the remaining discovery clusters, choose the validation
equation, and evaluate on the untouched outer cluster. Do not refit on validation
data. All methods receive exactly the same discovery observations.
Repeat with three split/search seeds and average scores within each test cluster
before inference. Report every cluster and whether its age lies outside the
discovery age range. A paired cluster bootstrap (20,000 draws) supplies a
descriptive 95% interval on mean score differences. This is not a full treatment
of shared training-set dependence, catalogue selection, or systematic age error.
Do not describe a confidence interval containing zero as proof of equivalence.

## Sensitivity and stopping rules

Report a baseline-only sensitivity analysis for membership thresholds 0.5, 0.8,
0.9 and upper color limits 1.5, 2.0. Always evaluate on the same primary-sample
held-out stars, exclude their entire cluster from training, and label this
separate full-training analysis. Report how its conclusions change.

Complete the fixed configuration regardless of which method wins. Debugging
fixes invalidate affected cached outputs and are logged. A larger-budget or
alternative-model follow-up must be labeled exploratory and retain the first
results. No result is called a discovery merely because repeated searches agree.

## Sources

- Van-Lane et al. (2025), [ChronoFlow](https://arxiv.org/abs/2412.12244), including
  membership handling and the probabilistic nature of stellar rotation.
- [PySR API](https://pysr.ai/api), model selection and deterministic execution.
- Barnes et al. (2016), [M67 rotation](https://arxiv.org/abs/1603.09179).
- Dungee et al. (2022), [cool stars in M67](https://arxiv.org/abs/2211.01377).

The M67 papers do not justify treating every M67 residual as evidence for
weakened magnetic braking. Their samples and color regimes matter.

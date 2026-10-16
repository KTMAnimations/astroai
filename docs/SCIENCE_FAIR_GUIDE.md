# Science fair presentation guide

## Working title

**Can a model predict how real stars rotate—and recover a cluster's age?**

Subtitle: Testing rotation distributions on withheld star clusters and additional observations.

## Abstract

Stars with similar ages and colors can have different rotation periods. This
study tested whether predicting that spread improves stellar-rotation models.
A published catalogue supplied 721 stars in 16 clusters after documented quality,
membership, age and color cuts. Six model families were evaluated using identical
splits: one whole cluster for testing, three for validation and twelve for fitting.
Three splits per test cluster produced 48 comparisons. A two-component Gaussian
mixture with a quadratic slow-rotation relation improved negative log likelihood
in 15 of 16 cluster averages and reduced continuous ranked probability score by
10.3% versus a Gaussian power law. Its 68% and 90% prediction intervals contained
68.4% and 90.0% of held-out observations on average. Mean-log-period RMSE did not
improve across the primary sample. In an external check using 25 additional M67
stars, absent from the original catalogue and never used for fitting or selection,
the mixture reduced RMSE by 23.6%. Withheld-cluster age estimates also showed a
lower median fractional error, although the paired interval on mean log-age-error
improvement included zero. The results support better population probability
predictions within the tested domain, while showing the need for more independent
clusters and explicit treatment of age uncertainty.

## Question, hypothesis and variables

**Question:** Does modeling the distribution of rotation periods improve
predictions for stars in an unseen cluster, and can that likelihood recover the
cluster's age from periods and colors?

**Hypothesis:** A narrow slow-rotation component plus a broader fast component
will predict the observed distribution better than a single Gaussian around a
power law. The follow-up protocol fixed the comparison before its new test scores
were computed. It followed inspection of the initial study, so the shared
catalogue remains an exploratory development benchmark.

| Element | Definition |
|---|---|
| Inputs | Cluster age and dereddened Gaia BP-RP for period prediction; periods and colors for age recovery |
| Model comparison | Six fixed distribution families, plus a family selected on validation data |
| Primary measurement | Difference in held-out negative log likelihood: quadratic mixture minus power-law Gaussian |
| Secondary measurements | CRPS, mean-log-period RMSE, median-prediction MAE, interval coverage/width and age error |
| Controls | Same stars, whole-cluster splits, equal training weight per cluster, fixed parameter grids and optimization starts |
| Replication unit | Test cluster, after averaging its three split results; 16 cluster units |
| External check | 25 additional M67 stars; all M67 members excluded from fitting and selection |

## Explain the astronomy and the model

Rotation period is the time a star takes to spin once. Magnetic winds can remove
angular momentum as stars age. Color is a rough indicator of temperature and
mass. Cluster members have an estimated shared age, providing a way to test
relations among age, color and rotation period. The reference ages come from
published isochrone estimates and retain calibration uncertainty.

The initial project searched for one equation. The follow-up predicts a
probability distribution: a narrow component at longer periods and a broader
component at shorter periods. Their relative weight depends on age and color.
The slow-component mean is a quadratic in logarithmic age and color. The model
is a statistical description; assigning a component does not identify a star's
physical state or prove why it rotates quickly.

[ChronoFlow](https://arxiv.org/abs/2412.12244) and
[gyro-interp](https://arxiv.org/abs/2303.08830) motivate modeling dispersion. This
project implements a simpler comparison, not a complete replication of either
published method. The external measurements come from
[Gruner et al. (2023)](https://arxiv.org/abs/2305.16997).

## Procedure for the display

1. Verify the original catalogue against its pinned source. Recorded quality,
   membership, age/color and cluster-size cuts reduce 8,412 entries to 721 stars
   in 16 clusters. Do not cut by period or model residual.
2. Reuse the initial study's 48 whole-cluster discovery/validation/test splits.
   Exclude the entire test cluster from fitting and model selection.
3. Fit six families with equal discovery-cluster weights. Use only validation
   clusters to choose regularization and the separate family selector.
4. Freeze the fits. Save predictions and scores for every test star, including
   fast rotators and difficult clusters. Average seeds within each cluster.
5. Compare proper distribution scores, point errors, coverage and width. Use
   paired cluster resampling for descriptive intervals.
6. Estimate each withheld cluster's age by maximizing its period likelihood over
   a fixed age grid. Consult the catalogue age only afterward to score it.
7. Check external M67 identifiers against the entire original catalogue. Predict
   all 25 eligible non-overlapping stars using the three frozen M67-excluded
   fits. Retain published period corrections and the fast-rotation outlier.
8. Fit a separate application model to all 721 primary stars only after the
   evaluation. Its in-sample fit does not supply any reported held-out score.

The [protocol](REAL_STAR_PROTOCOL.md) records the fixed choices. The
[complete results](REAL_STAR_RESULTS.md) include every family and cluster, the
3.87 Gyr external age sensitivity, and limitations.

## What the figures show

| Display panel | Figure and claim | Evidence to keep visible |
|---|---|---|
| Question and method | Inputs, six model families, discovery → validation → test diagram | 721 stars, 16 clusters, identical splits, no residual exclusions |
| Main result | `distribution_scores.svg` and `coverage.svg` | NLL improves in 15/16 clusters; CRPS improves 10.3%; mean-log-period RMSE does not improve |
| Astronomy checks | `age_recovery.svg` and `external_m67.svg` | All 16 ages and all 25 external stars; external RMSE decreases 23.6%; one external cluster |

Use SVG files from `project/figures/real_stars_v2/` for large printing. The
[four-page handout](../output/pdf/real_star_research_brief.pdf) is a compact summary.
Keep the full score table available. The ordinary quadratic Gaussian is
competitive on external M67 and has lower median-prediction absolute error there.

**NLL** asks how much probability density the model gives to the observed
period. Lower is better; a density score can be negative. **CRPS** compares the
whole predicted cumulative distribution with the observation and penalizes both
misplaced and overly broad predictions. It also favors the mixture here.
**RMSE** evaluates the predicted mean of log period and penalizes large misses
strongly. **MAE** evaluates the predicted median. Coverage alone is insufficient:
very broad intervals can contain nearly everything, so show widths too.

A dex is a base-ten logarithmic unit. A 0.1-dex difference is a factor of about
1.26 in period. Error averages are not guarantees for individual stars.

## A one-minute explanation

“This project tests whether a model can predict how real stars rotate. A single
curve misses the spread: stars of similar age and color can have very different
periods. I compared six models while withholding entire star clusters. A model
with a narrow slow-rotation component and a broader fast component gave better
probability predictions in 15 of 16 clusters. Its prediction intervals also
matched the observed spread more closely. I then tested 25 additional published
M67 stars that were absent from the original catalogue. Rotation-period error
fell by about 24% compared with a power law. The model can also estimate a
cluster's age, but some clusters remain difficult and the age improvement is
uncertain. The result is a reproducible improvement in population predictions,
with all observations, competing models and failures available for inspection.”

Adapt first-person statements to the work you personally performed and can
explain. Do not describe AI-executed experiments as unaided manual work.

## Questions to prepare for

**Why withhold entire clusters?** Members share a reference age and observational
context. A random star split could place members of the same cluster in both
fitting and testing. The test here asks about an unseen cluster.

**Did you just change the score to make the model win?** The follow-up specified
NLL as its primary score before computing its new comparisons and reported every
planned metric. CRPS independently supports the distribution improvement. The
original squared-error result remains: mean-log-period RMSE does not improve
across the primary sample. The earlier catalogue had already informed the new
question, so this is openly exploratory.

**How strong is the main result?** The mean paired NLL difference is -0.5021 nats,
with descriptive 95% interval [-0.6694, -0.3346]. All models share training data
across folds, and the calculation does not include every systematic uncertainty.
It supports a measured improvement on this benchmark, not a universal guarantee.

**Are the 25 external stars independent?** Their Gaia IDs do not occur anywhere
in the 8,412-entry original catalogue, and all M67 members are withheld from the
models used to predict them. They still share one cluster and its age/reddening
uncertainties. They are additional observations, not 25 independent age anchors.

**Are the intervals calibrated everywhere?** No. Average primary-sample coverage
is close to nominal, but the external, more selectively single-star sample has
92% coverage for nominal 68% intervals. That overcoverage is reported.

**How accurate are the ages?** Median absolute fractional cluster-age error is
18.3% for the power law, 12.5% for the quadratic mixture and 9.9% for validation
family selection. Some clusters have much larger errors, one selected fit reaches
the age-grid boundary, and the paired interval on mean log-age-error improvement
spans zero. These are not 10%-accurate individual-star ages or calibrated age
confidence intervals.

**Did you discover a new law of stellar aging?** No. Mixture models, validation
and gyrochronology are established ideas. The contribution is the implemented
comparison, reproducible diagnosis, external observation check and usable tools.
A statistical component does not prove a distinct physical population.

**What happened to the equation search?** The initial 258-search study is retained.
On controlled synthetic functions, validation selected a recovering equation in
157/180 main trials versus 76/180 for the default selector; a quadratic baseline
recovered 164/180. Symbolic regression showed no established real-star advantage.
That result motivated modeling rotational dispersion. See the
[initial results](RESULTS.md) and [research log](RESEARCH_LOG.md).

**What would you test next?** Freeze this model before evaluating additional
clusters, propagate age/color measurement uncertainty, check residuals within
color and age ranges, and compare against full published gyrochronology methods.
Field-star validation needs independent age measurements, not ages derived from
the same rotation relation being tested.

## Contribution and reproducibility record

Published observations and reference ages belong to their credited sources.
The scientific Python/Julia libraries implement established numerical methods.
AI assistance supported source research, design, coding, execution, diagnosis,
plots and writing. The repository preserves protocols, source hashes, candidate
fits, raw predictions and tests. Add an accurate record of the student's own
question development, checks, code changes, interpretation and presentation work.

Before presenting, trace one plotted star to its saved prediction, explain every
data cut, distinguish fitting from validation, and demonstrate the two command-line
tools. The handout summarizes a reproducible experiment; it is not external peer
review or an assurance of a particular competition result.

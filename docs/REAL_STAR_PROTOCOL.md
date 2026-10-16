# Real-star follow-up protocol

Fixed before the new held-out model comparisons, 15 October 2026. The earlier
study and its difficult clusters have already been inspected, so this is an
exploratory extension, not an untouched confirmatory catalogue.

## Aim

Predict the distribution of stellar rotation periods, then use that likelihood
to recover the age of a completely withheld cluster. A single curve fitted by
squared error cannot represent the narrow slow-rotation sequence and the broader
fast-rotation population simultaneously. Mixture models are a scientifically
motivated extension, following the distributional treatment in ChronoFlow and
gyro-interp; this implementation is its own simpler comparison.

Sources: [ChronoFlow](https://arxiv.org/abs/2412.12244),
[Bouma et al. 2023](https://arxiv.org/abs/2303.08830), and the
[gyro-interp model documentation](https://gyro-interp.readthedocs.io/en/latest/gyrointerp.html).

## Data and separation

Retain all 721 primary stars, all 16 clusters, and the exact 48 discovery,
validation and test splits from research_v1. Do not remove difficult stars or
clusters. Test-cluster periods and ages cannot choose model parameters, starts,
regularization or model families. All methods receive identical discovery data.
Keep each discovery cluster's total training weight equal.

## Fixed model comparison

All distributions describe z = log10(period/day), conditional on age and color.
Use scaled versions of the same logarithmic age/color features as research_v1.

1. Power-law mean with one fitted Gaussian scatter.
2. Quadratic mean with one fitted Gaussian scatter.
3. Quadratic mean with log scatter linear in age and color.
4. Two-Gaussian mixture with a power-law slow-component mean.
5. Two-Gaussian mixture with a quadratic slow-component mean.
6. Smooth radial-basis residual model around a power law, with Gaussian scatter.

For each mixture, the faster component has mean offset below the slow component,
its own fitted width, and an age/color-dependent logistic mixing weight. Component
names are statistical descriptions; they do not identify a physical cause for
an individual star. Fit by weighted likelihood with deterministic multiple
initializations. Minimum component widths prevent likelihood collapse.

Quadratic mixture curvature penalties are 0, 0.01 and 0.1; radial-basis penalties
are 0.001, 0.01, 0.1 and 1. Choose these settings using only mean validation-cluster
negative log likelihood (NLL). Also report a model-family selector using the same
validation criterion. Save all choices and optimization status. Pilot fits on
synthetic data may verify numerical correctness; do not tune on outer outcomes.

## Scores and success criteria

Primary comparison: quadratic mixture minus Gaussian power-law mean NLL on
held-out clusters. Lower is better; an improvement would demonstrate a better
predictive distribution under this score. Secondary checks: continuous ranked
probability score (CRPS), mean-prediction RMSE, median-prediction absolute error,
68%/90% prediction-interval coverage and widths. NLL and CRPS are proper scoring
rules and penalize overly broad as well as overly narrow predictions. Do not
claim point-prediction improvement if only distribution scores improve.

Average the three seeds within each cluster before aggregation. Report every
cluster and paired descriptive bootstrap intervals; retain the earlier caveat
about overlapping training sets and uncertain catalogue ages. Include all
predeclared models, not only the best result. No multiplicity-adjusted discovery
claim is planned.

## Age recovery

For each withheld cluster, evaluate the fitted period likelihood on 401 logarithmic
age points spanning 80-5000 Myr. Sum individual-star log likelihoods and report
the maximizing age. Average inferred log ages over the three seeds before
comparing with the catalogue age. Report median absolute fractional age error,
mean absolute log-age error, every cluster, and boundary hits. These are cluster
point estimates; do not present the product likelihood as calibrated age
uncertainty or imply that hundreds of stars remove systematic calibration error.

## External observations and scope

Investigate the Gruner et al. (2023) M67 catalogue, which was not a source in the
primary selected sample. Before scoring, document its color/reddening scale,
quality selection and Gaia-ID overlap. Test only within the declared color domain,
remove overlap with the entire original catalogue, and train with all M67 rows
excluded. Any source lacking a defensible color conversion or IDs cannot be
called independent external validation. Freeze the model fitting/selection rule
before inspecting its external prediction errors. Published slow-sequence
selection means this benchmark may differ from the broad primary population.

## Stopping and reporting

Complete the fixed 48-fold comparison and the external check if its inputs are
usable. Keep the full baseline study. Investigate numerical failures, but retain
unfavorable scientific results. A later model redesign requires a separately
recorded protocol and output bundle. The purpose is useful prediction and age
recovery, not making a predetermined model win.

### External-data specification before scoring

The pinned gyro-interp copy of Gruner et al. (2023) Table C.2 contains 47 stars,
Gaia DR3 IDs and directly supplied dereddened Gaia BP-RP. The table caption states
that HPS-flagged periods are already doubled; retain them unchanged. Apply only
the fixed 0.65 < color < 1.5 domain, finite positive measurements and removal of
all IDs present in the full 8,412-entry original catalogue. This leaves 25 stars.
Use the paper's approximate 4,000 Myr reference age, with 3,870 Myr as a named
age-scale sensitivity, and the three already frozen M67-holdout models. Do not
refit or select models on this external sample. Report all seven methods, all
stars, and cluster age recovery. This is a separate-star check in one cluster,
not a new independent cluster-age calibration or a field-star validation.

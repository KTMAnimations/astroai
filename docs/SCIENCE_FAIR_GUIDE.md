# Science fair presentation guide

## Working title

**Can an equation search learn stellar rotation?**

Subtitle: How noise, sampling and validation affect recovery of a known function.

## Abstract

Symbolic regression searches for equations that describe data, but discovering
an accurate candidate and selecting it are separate problems. This study tested
that distinction using stellar rotation. Three known functions generated
synthetic data under two sampling designs and three noise levels. Ten trials per
condition produced 180 searches. Each search used 600 discovery and 200
validation observations; predictions were then evaluated against the exact
function on a separate grid. Recovery required an error below 0.02 dex. The
default selector recovered the function in 76 trials, compared with 157 for
independent validation. In 94 trials, default selection missed an accurate
candidate already present in the search output. A quadratic baseline recovered
164 times, showing that suitable parametric models remained competitive. A
second experiment tested 721 observed stars in 16 clusters, withholding each
cluster from fitting and validation. Validation-selected symbolic regression
scored 0.2170 dex, compared with 0.2195 dex for a fitted power law; the descriptive
95% interval on their difference spanned zero. The results show a selection
failure in this controlled benchmark and no established symbolic-regression
advantage on the observational sample. They do not establish a new law of
stellar aging.

## Question, hypothesis and variables

**Question:** When does a search find a useful equation but select a poorer one,
and does independent validation improve predictions for unseen star clusters?

**Hypothesis:** Choosing an equation on independent validation data will improve
recovery when a useful nonlinear term is present, but may add selection noise
when a simpler model is adequate. This hypothesis was recorded before the main
runs. The catalogue and earlier experiments had already been examined.

| Element | Definition |
|---|---|
| Synthetic independent variables | Known function, sampling design, noise level, selection rule |
| Observational comparison | Five methods trained on identical discovery clusters |
| Synthetic dependent variables | Error against the known function; recovery below 0.02 dex; invalid predictions |
| Observational dependent variable | Mean held-out cluster RMSE in log10 period |
| Controls | Fixed search budget, operators, transforms, sample sizes, evaluation domain and seeds |
| Baselines | Fixed age exponent, fitted power law, quadratic relation |
| Replication unit | Synthetic trial; observational test cluster after averaging its three seeds |

## Explain the astronomy accurately

Rotation period is the time a star takes to spin once. For the population where
rotation-age relations apply, magnetic winds can remove angular momentum as
stars age. Color is a rough indicator of stellar temperature and mass. The
catalogue groups stars into clusters with estimated shared ages, allowing an
age-color-period relation to be tested.

The project starts from a square-root age dependence inspired by Skumanich.
Its extra linear color dependence is a chosen approximation. It uses the
ChronoFlow catalogue, but does not reproduce ChronoFlow's probabilistic model.
Stars of the same age and color can have different periods, and the adopted
cluster ages are uncertain. A fitted curve cannot explain all of that variation.
See the sources in the [protocol](PROTOCOL.md) and [results](RESULTS.md).

## Procedure to put on the display

1. Verify the published catalogue and apply recorded quality, membership and
   age/color criteria: 8,412 catalogue entries become 721 stars in 16 clusters.
2. Generate artificial periods from three specified functions. Vary the sampling
   pattern and noise; retain the exact function for evaluation only.
3. Run PySR on discovery data. Choose one candidate using the library default
   and another using separate validation data. Fit three parametric baselines
   on the same discovery observations.
4. Freeze the choices, then measure error on a separate noise-free input grid.
   Repeat ten times per condition. Retain invalid predictions as failures.
5. Test the real catalogue by leaving an entire cluster out. Select three other
   clusters for validation and fit on the remaining 12. Repeat three split/search
   seeds and average within each test cluster.
6. Compare paired cluster errors, inspect every cluster, and vary training-sample
   cuts while keeping the test stars fixed.
7. Report domain-check follow-ups separately from the primary experiment. Their
   designs and outcomes are in the research log and full results.

The log variables are x = log10(age/Gyr), y = log10(BP-RP), and z = log10(period/day).
The three baselines fit one, three and six coefficients respectively. Log
transforms are scientific assumptions, so the search does not start from nothing.

## Suggested display layout

| Left panel: question and controls | Center: controlled result | Right panel: real-data test and limits |
|---|---|---|
| Title, research question, hypothesis | Recovery-count heatmaps | Held-out model comparison |
| Sample plot and data source | One clearly labeled example: interaction/catalogue/0.05 dex | 721 stars, 16 clusters, three seeds |
| Discovery → validation → test workflow | Default 0/10, validation 10/10 in that example | Difference from power law: -0.0025 dex |
| Known functions and fixed controls | All-scenario table available beside the display | Interval [-0.0085, +0.0039]; no established advantage |
| Definitions of RMSE and dex | Baselines and failures included | Limitations, next experiment, sources and assistance |

Use the SVG files in `project/figures/research_v1/` for large printing. The
four-page PDF is a handout. Keep the complete results table accessible so the
highlighted example does not stand in for every condition. Include the quadratic
baseline: it performed well and is part of the scientific finding.

## A one-minute explanation

“This project asks whether a computer can choose a useful equation for stellar
rotation. First, I examined synthetic experiments where the correct function was
known. Each search produced several equations. One rule chose by training fit
and complexity; another used separate validation observations. In 94 of 180
trials, the default rule missed an accurate equation the search had already
found. Validation improved recovery, but a simple quadratic model also did very
well. Then I tested real stars by withholding entire clusters. Symbolic
regression and a power law had similar measured errors, and the uncertainty did
not establish an advantage for symbolic regression. The main lesson is that
finding an equation, choosing it and validating a physical law are separate
steps.”

Adapt the first-person wording to the work you personally performed and can
explain. Do not describe AI-executed experiments as unaided manual work.

## Questions to prepare for

**Why use artificial data?** It provides a known answer. With observations alone,
a repeated equation could reflect a bias in the search or selection rule.

**Why withhold whole clusters?** Stars from the same cluster share an adopted
age and observational context. A random star split would place members of the
same cluster in training and testing and answer an easier question.

**What is dex?** It is a base-ten logarithmic unit. A 0.1-dex difference is a
factor of about 1.26 in period. A 0.02-dex recovery threshold corresponds to a
factor of about 1.047, but an RMSE is not a per-star error guarantee.

**Why did the default rule fail?** It trades fit against expression complexity
within a training-loss threshold. With noisy data, omitting useful structure can
still meet that threshold. The study tests this configured rule, not every
possible symbolic-regression method or setting.

**Why did the quadratic do well?** One synthetic truth is exactly quadratic in
the transformed inputs; another can be approximated by a quadratic over the
limited domain. A flexible equation search is not automatically better than a
suitable fixed family.

**Does the real-data interval prove the models are equal?** No. It leaves both
small improvements and small degradations plausible under the descriptive
resampling calculation. Shared training sets and age-systematic errors impose
further limitations.

**What about the two invalid equations?** They worked on sampled validation
inputs but contained operations such as square roots that were undefined
elsewhere in the domain. They remain failures in the primary result. The
separate domain guard checks inputs without consulting true test targets.

**Did you discover weakened magnetic braking?** No. This experiment does not
isolate that mechanism. The M67 plot is a held-out prediction check within a
selected sample, with all its selected observations shown.

**Is this a new method?** Independent validation and domain checks are established
ideas. The contribution is a reproducible comparison and diagnosis in a defined
stellar-rotation experiment.

**What would you do next?** Check independent astronomical populations, propagate
cluster-age uncertainty, and compare models that predict the distribution of
periods. More search time on the same data cannot replace independent evidence.

## Contribution record

The source observations and cluster ages come from published work, credited in
`project/data/README.md`. PySR and the scientific Python/Julia libraries implement
existing numerical methods. AI assistance was used in this revision for source
research, experiment design, coding, execution, diagnosis, plots and writing.
The repository retains the protocol, settings, source hashes and raw candidate
and prediction tables. A student should add an accurate account of their own
question development, code changes, checks, interpretation and presentation work.

Before presenting, be able to trace one plotted point back to its saved trial,
explain each data cut, and explain why a null or negative result can be useful.
The handout is a research summary, not evidence of external peer review or an
assurance of a particular competition result.

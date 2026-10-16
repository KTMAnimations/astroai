# Science fair presentation guide

## Working title

**Reproducing and comparing models of stellar rotation**

Research question: How does a compact probability model compare with a published
neural model when both predict stars in completely withheld clusters?

## Abstract

Stellar rotation can constrain age, but stars with similar ages and colors can
have different periods. This project first reproduced numerical examples from
the published ChronoFlow model using its released code and weights. Four reported
cluster-age medians agreed within 0.2%; a solar example did not reproduce within
a prespecified 5% tolerance under the released defaults. For a controlled model
comparison, 96 neural networks using the published architecture were trained on
48 fixed discovery, validation and test splits covering 721 stars in 16 clusters.
Every test cluster was excluded from fitting and selection. A 12-parameter
quadratic mixture had lower negative log likelihood than the prespecified
17,733-parameter, equal-cluster flow in 11 of 16 cluster averages. The mean paired
difference was -0.165 nats, with a descriptive 95% interval of [-0.308, -0.033].
Other error comparisons were less conclusive, and per-star weighting made the
flow competitive. Tests on 25 additional M67 stars favored the mixture for
period error but a flow variant for age recovery. The results support partial
numerical reproducibility and a useful compact comparator on this restricted
sample, while showing sensitivity to inputs and training choices.

## What the project is

This is a **partial numerical reproduction plus a controlled model comparison**.
It reproduces the actual released ChronoFlow network's tutorial and four cluster
examples. It also conducts a new experiment with the published architecture on
identical data splits. It does not reproduce the paper's entire training process,
full cross-validation benchmark or global uncertainty estimate. It has not
established a new physical law or broad superiority over ChronoFlow.

The earlier work remains useful context: symbolic regression recovered known
synthetic functions more often with independent validation, but did not establish
a real-star advantage over a power law. The subsequent mixture study improved
probability predictions against simple baselines. The latest experiment asks
whether that comparison survives a stronger published-model benchmark.

## Explain the astronomy

A rotation period is the time a star takes to spin once. Magnetic winds can
remove angular momentum as a star ages. Color roughly describes temperature and
mass. A cluster provides stars with an estimated shared age, allowing a model to
relate age, color and rotation. The reference ages remain uncertain estimates.

A probability model predicts a range of periods rather than one value. Our
mixture describes a narrow slow-rotation component and a broader fast component;
these statistical components do not identify physical stellar classes. The
neural spline flow learns a flexible probability distribution, including color
uncertainty and the published background-contamination likelihood.

## Keep the two experiments distinct

| Experiment | Question | Evidence | What it does not show |
|---|---|---|---|
| Released-model reproduction | Can we recover the authors' numerical outputs? | Four cluster medians within 0.2%; tutorial likelihood matches | Agreement is not independent proof of true cluster ages |
| Matched training comparison | How do models perform on the same withheld stars? | 721 stars, 16 clusters, 48 splits, 96 neural fits | This restricted benchmark does not recreate the full published training study |
| External M67 check | How do the frozen fits predict additional observations? | Same 25 non-overlapping stars for every model | One cluster is not broad field-star validation |
| Direct released-weight check | How do the actual released weights compare on eligible common test stars? | Same 80 stars in three excluded clusters | Calibration data differ; the check is supplementary |

## Controls and procedure

1. Verify the catalogue and upstream code/weights against pinned hashes. Record
   all quality cuts and retain the source identifiers.
2. Reproduce the native tutorial. Compare a vectorized implementation against
   unmodified scalar calls, then check probability normalization.
3. Infer ages for the four reported clusters explicitly excluded from the final
   network. Keep the solar mismatch and the documented prior/input assumptions.
4. Reuse the earlier 48 cluster splits. One cluster is for testing, three for
   validation and twelve for fitting. A test cluster never enters fitting or
   model/checkpoint selection.
5. Train the released neural architecture for 5,000 steps per fit. Evaluate two
   fixed weightings: equal total weight per cluster and equal weight per star.
   Save final and validation-selected checkpoints. Select only on validation NLL.
6. Score every selected star with all eight reported methods. Average three
   splits within each cluster, then give all 16 clusters equal weight.
7. Recover each withheld cluster's age from its periods/colors on a common age
   grid; use its reference age only afterward for evaluation.
8. Predict the 25 external M67 stars with models that exclude every M67 member.
   Preserve all selected observations, including the fast rotator.
9. Replay a complete neural fit, check finer grids and wider integration bounds,
   verify score tables, and inspect all figures.

No periods or residuals determine which stars enter the comparison. There are
no post-hoc removals of difficult clusters. The protocol was fixed before the
new comparisons, but earlier inspection of the same catalogue means the research
is exploratory rather than an untouched confirmation.

## Results to show together

| Measurement | Primary retrained flow | Quadratic mixture |
|---|---:|---:|
| Mean cluster NLL | -0.3310 | -0.4960 |
| Mean cluster CRPS, dex | 0.1127 | 0.1080 |
| Mean-log-period RMSE, dex | 0.2324 | 0.2251 |
| Nominal 68% interval coverage | 61.3% | 68.4% |
| Median absolute fractional cluster-age error | 15.1% | 12.5% |
| External M67 RMSE, dex | 0.1685 | 0.1174 |

The primary NLL interval excludes zero. The corresponding CRPS, point-error and
age-error intervals include zero. Against the paper-style per-star flow, NLL
intervals also include zero and CRPS is nearly tied. Show that sensitivity beside
the main result; do not choose a different headline comparator after seeing the
scores. The per-star flow's external M67 age is about 4.00 Gyr, versus 3.38 Gyr
for the mixture and a nominal 4.0 Gyr reference.

**NLL** rewards probability density assigned to the observed period; lower is
better, and negative values are possible for densities. **CRPS** evaluates the
whole cumulative distribution and penalizes overly broad as well as misplaced
predictions. **RMSE** scores the mean of log period and emphasizes large misses.
**MAE** scores the predicted median. Coverage must be considered with interval
width: an unnecessarily broad interval can contain nearly everything.

A dex is a base-ten logarithmic unit; 0.1 dex is a factor of about 1.26. An average
error is not a guarantee for an individual star.

## Display and one-minute explanation

Use the four SVG figures in `project/figures/published_benchmark_v1/` for printing:
`reproduction`, `matched_scores`, `matched_ages` and `external_comparison`.
Put the research question, source data and split diagram on the left, reproduction
and matched scores in the center, and age/external checks with limitations on the
right. Keep the complete tables accessible. The older PDFs summarize earlier
stages and should be labeled that way.

“This project asks how a small probability model compares with a published
neural model of stellar rotation. First, the released code reproduced four
published cluster-age estimates very closely, but its solar example differed
under the default inputs. Then I compared models on the same 721 stars, always
withholding whole clusters. The small mixture model predicted the observed
period probabilities better than the prespecified equal-cluster neural model.
However, changing to the paper-style per-star weighting made the neural model
competitive. On additional M67 observations, the mixture had lower period error,
while a neural variant gave a better group-age estimate. The contribution is a
reproducible comparison that shows where the methods work, and where conclusions
depend on the data and training choices.”

Adapt first-person statements to the work you personally performed and can
explain. Do not describe AI-executed work as unaided manual experimentation.

## Questions to prepare for

**Why not simply use the released network on all test clusters?** It was trained
on many of them. That would measure performance partly on training data. The
controlled comparison trains new networks excluding each test cluster.

**Is the new network exactly the published model?** It uses the released
architecture, zuko version and likelihood. The original complete training script
is unavailable, so optimizer, batching and scheduler conventions are explicitly
specified choices. Actual released weights are used separately for reproduction.

**Did the small model beat ChronoFlow?** It beat the prespecified equal-cluster
retraining variant on NLL in this restricted benchmark. Other metrics were less
conclusive, and the paper-style weighting was competitive. A universal or
state-of-the-art superiority claim would be unsupported.

**Why did the solar reproduction differ?** The paper does not specify its color
uncertainty. A recorded diagnostic shows the released model's solar age is
sensitive to that input. This does not prove the authors' exact input or explain
away the failed reproduction. Both the mismatch and all sensitivity cells remain.

**What about the age prior?** The released code is uniform in log age; the paper
describes uniform age. Those priors place different weight on old ages. The
primary reproduction follows the code and reports the other choice separately.

**Why are the cluster-age errors still uncertain?** Reference ages have systematic
uncertainty, some clusters have large discrepancies, and some estimates reach the
fixed age-grid boundary. Shared training sets also limit a simple bootstrap's
interpretation. No 10%-accurate individual-star age claim is made.

**What is original here?** The controlled comparison and documented reproducibility
audit are the additional experiments. Mixture models, spline flows, validation and
gyrochronology are established methods. The result is not a new physical law.

**What would strengthen the result?** Freeze the analysis before acquiring more
independent clusters, test the original full training population, resolve missing
training/input details, and propagate age/color uncertainties. A more flexible
model may benefit from the larger dataset on which it was developed.

## Sources and contribution record

Read the [complete report](PUBLISHED_BENCHMARK_RESULTS.md),
[protocol](PUBLISHED_BENCHMARK_PROTOCOL.md) and [research log](RESEARCH_LOG.md).
[ChronoFlow](https://arxiv.org/abs/2412.12244) supplies the published model and
catalogue; [Gruner et al. (2023)](https://arxiv.org/abs/2305.16997) supply external
measurements. The scientific software implements established methods.

AI assistance supported research, design, implementation, execution, checking,
plots and writing. Record the student's own contributions accurately. Before
presenting, trace one plotted star to a saved prediction, explain every data cut,
show how model selection avoids test data, and reproduce one numerical check.
A repository report is not external peer review or a competition guarantee.

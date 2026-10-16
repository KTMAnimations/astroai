# Stellar rotation: reproducing and comparing gyrochronology models

This repository is a **partial numerical reproduction of ChronoFlow plus a
controlled comparison of stellar-rotation models**. It also includes working
tools to predict rotation-period distributions and estimate a cluster's age.

The latest experiment runs the authors' actual released network, then retrains
its architecture on the same data splits as our simpler mixture model. It
preserves the original equation-search study and the earlier real-star results.

[Published-model results](docs/PUBLISHED_BENCHMARK_RESULTS.md) ·
[Comparison protocol](docs/PUBLISHED_BENCHMARK_PROTOCOL.md) ·
[Science fair guide](docs/SCIENCE_FAIR_GUIDE.md) ·
[Research log](docs/RESEARCH_LOG.md)

## Reproducing the published implementation

The released ChronoFlow weights reproduce four reported cluster-age medians:

| Cluster | Published age, Myr | Reproduced age, Myr |
|---|---:|---:|
| M34 | 245 | 244.5 |
| NGC2516 | 132 | 131.9 |
| NGC1647 | 87 | 87.0 |
| NGC1750 | 99 | 99.1 |

All four agree within **0.2%**. These clusters were excluded from the final
network's training according to the paper. The native tutorial also reproduces
its printed likelihood. This verifies numerical outputs; agreement with a
paper's output is distinct from accuracy against independently known ages.

The solar example **does not reproduce within the specified 5% tolerance**:
the released defaults give 5.99 Gyr versus the paper's 5.1 Gyr. Its solar color
uncertainty is unspecified, and the saved sensitivity analysis shows a substantial
effect. The code also uses a uniform log-age prior where the paper describes a
uniform age prior. Both issues remain documented; no input was silently changed
to obtain the quoted result.

![Reproduction of published numerical results](project/figures/published_benchmark_v1/reproduction.png)

## A controlled comparison on real stars

The released network has already seen many of our ordinary test clusters, so it
cannot serve as a held-out comparator for all of them. We instead trained **96
new neural networks**, each for **5,000 steps**, using the published architecture
and likelihood on our exact **48 discovery/validation/test splits**: **721 stars
in 16 clusters**, with each test cluster excluded from fitting and selection.

The primary flow comparison uses equal discovery-cluster weights, matching our
earlier protocol, and validation-selected checkpoints. A paper-style per-star
weighting and fixed-final-epoch fits are also reported.

| Held-out metric | Primary retrained flow | Quadratic mixture |
|---|---:|---:|
| NLL, nats (lower is better) | -0.3310 | **-0.4960** |
| CRPS, dex | 0.1127 | **0.1080** |
| Mean-log-period RMSE, dex | 0.2324 | **0.2251** |
| Median-prediction MAE, dex | 0.1402 | **0.1339** |
| Coverage of nominal 68% intervals | 61.3% | 68.4% |
| Median absolute fractional cluster-age error | 15.1% | 12.5% |

The mixture improves NLL in **11 of 16 clusters**. Its primary paired NLL difference
is **-0.1650 nats**, with descriptive 95% interval **[-0.3077, -0.0327]**. The
intervals on CRPS, point-error and age-error differences include zero.

The result depends on training choices: the paper-style per-star flow is
competitive, with CRPS essentially tied and an NLL-difference interval that
includes zero. The mixture has **12 fitted parameters**, versus **17,733** in the
flow, making it a useful compact comparator for this dataset. These experiments
do not establish broad superiority over ChronoFlow on its larger population.

![Matched comparison with retrained ChronoFlow architecture](project/figures/published_benchmark_v1/matched_scores.png)

This reproduces the published architecture, not the paper's complete training
process or full cross-validation benchmark. The original training script and
individual holdout weights are not released. Optimizer, batching and scheduler
choices are explicitly recorded. Our restricted catalogue had already informed
earlier development, so the comparison is exploratory.

## Additional observations and direct released-weight checks

On the same **25 external M67 stars**, with all M67 members excluded from fitting
and selection, the mixture has **30.3% lower log-period RMSE** than the primary
retrained flow. The per-star flow gives an age estimate closer to the nominal
4.0 Gyr reference. Interval calibration also differs: nominal 68% coverage is
46.7% for the primary flow and 92.0% for the mixture on this selected population.
The complete tables retain every method and every star.

A separate comparison evaluates the unchanged released ChronoFlow weights and
our frozen models on the same **80 stars in three explicitly excluded clusters**.
Its training datasets differ, so it is labeled supplementary and is not used to
claim controlled superiority. See the [full report](docs/PUBLISHED_BENCHMARK_RESULTS.md).

## Reproduce the benchmark

The original study uses `.venv`; the published-model comparison uses an isolated
`.venv-published` environment. See [environment notes](environment/README.md).

```bash
bash scripts/setup_published_environment.sh
PYTHONPATH=src .venv-published/bin/python scripts/reproduce_chronoflow.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-published/bin/python scripts/train_published_benchmark.py
PYTHONPATH=src .venv-published/bin/python scripts/evaluate_published_benchmark.py
PYTHONPATH=src .venv-published/bin/python scripts/evaluate_published_benchmark.py --stage summarize
PYTHONPATH=src .venv-published/bin/python scripts/summarize_published_sensitivities.py
PYTHONPATH=src .venv-published/bin/python scripts/evaluate_published_external.py
PYTHONPATH=src .venv-published/bin/python scripts/compare_released_chronoflow.py
PYTHONPATH=src .venv-published/bin/python scripts/diagnose_solar_reproduction.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-published/bin/python scripts/verify_published_benchmark.py
PYTHONPATH=src .venv-published/bin/python scripts/report_published_benchmark.py
```

The runner resumes completed fits and checks source, data, configuration and
environment hashes. All weights, learning curves, validation choices, individual
predictions and age profiles are saved. A full independent 5,000-step replay
reproduced the saved neural weights exactly. Native-call comparisons, density
normalization, grid refinement, wider integration bounds, split separation,
training-ID exclusions and score reconstruction are checked.

For a fresh run in another environment, use a separate checkout and a new
`study_id` in `configs/published_benchmark.json`. Reporting defaults to the saved
reference bundle. Tests:

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv-published/bin/python -m pytest tests/test_published_model.py tests/test_published_results.py -q
```

## Use the existing prediction tools

The compact application model predicts rotation periods for a supplied age and
dereddened Gaia DR3 color. It is fitted after evaluation and does not supply the
held-out scores:

```bash
bash scripts/setup_environment.sh
.venv/bin/python scripts/predict_rotation.py --age-myr 1000 --bp-rp 1.0
.venv/bin/python scripts/infer_cluster_age.py project/results/real_stars_v2/external_stars.csv --holdout-cluster M67
```

The first example gives a median period of about 11.1 days and a 68% population
interval of 7.5–12.9 days. Its domain is 135–3870 Myr and 0.65 < BP-RP < 1.5.
The second estimates a coeval group's age using M67-excluded fits. These tools
do not provide calibrated individual-star ages or age confidence intervals.

## Earlier experiments and attribution

- [Real-star mixture study](docs/REAL_STAR_RESULTS.md): the original 721-star
  comparison against simpler baselines and the first external M67 check.
- [Earlier real-star handout](output/pdf/real_star_research_brief.pdf).
- [Equation-search study](docs/RESULTS.md): 258 searches, including synthetic
  controls, cluster holdouts and a fresh-seed follow-up. Validation helped choose
  useful synthetic equations; symbolic regression showed no established real-star
  advantage over a power law.
- [Initial equation-search handout](output/pdf/stellar_rotation_research_brief.pdf).

[ChronoFlow](https://arxiv.org/abs/2412.12244) supplies the main catalogue and
published model being reproduced. [gyro-interp](https://arxiv.org/abs/2303.08830)
provides scientific context for modeling dispersion; its full implementation is
not benchmarked here. [Gruner et al. (2023)](https://arxiv.org/abs/2305.16997)
supply the external observations, and [PySR](https://arxiv.org/abs/2305.01582)
powers the initial equation searches. [Data provenance](project/data/README.md)
records sources, quality cuts and hashes. No result establishes a new physical
braking law.

Active code lives in `src/astrogyro/`. The three named result directories preserve
the successive studies; other notebooks/results are historical exploration.
These revisions used AI assistance for research, design, implementation, execution,
figures and writing. Student presenters should record their own contributions
and be able to explain and reproduce the analysis. Historical commit messages
are preserved; new commits describe the concrete scientific work.

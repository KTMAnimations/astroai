# Recorded environment

The study ran with Python 3.11.8, PySR 2.7.0, SymbolicRegression.jl 2.7.1,
and Julia 1.13.2 on macOS arm64. The Python lock and Julia project/manifest record
the installed dependencies. Searches use 64-bit arithmetic, deterministic serial
execution and explicit random seeds. The result manifest also records versions,
platform, source hashes, data hash and the complete experiment configuration.

From the repository root, run `bash scripts/setup_environment.sh`. The first
PySR import may download Julia and compile its dependencies. This creates only
the local `.venv` environment plus Julia's standard package cache.

The experiments do not require the ChronoFlow or GPgyro submodules: they use the
checked-in, attributed catalogue. Submodule mappings are restored for users who
want to inspect the historical integrations.

Exact numerical replay is checked on one nontrivial saved trial. Reproduction on
another operating system or processor may introduce floating-point differences;
compare equations and predictive metrics as well as the recorded versions.

The `real_stars_v2` follow-up uses the same recorded environment, with NumPy/SciPy
for distribution fitting. `scripts/verify_real_star_reproducibility.py` independently
refits all 528 candidate configurations across all 48 splits and compares every
saved parameter and selection score; it does not rerun PySR. Set
`OPENBLAS_NUM_THREADS=1` for the recorded numerical replay.

The Python lock also includes ReportLab and pypdf for report generation and PDF
inspection. Poppler's `pdftoppm` was used for visual inspection; it is a separate
system dependency and is not needed to run the numerical study or build the PDF.

## Published-model benchmark

`scripts/setup_published_environment.sh` creates a separate `.venv-published`
environment from `published-requirements.lock.txt` and initializes the pinned
ChronoFlow submodule. The recorded benchmark uses Python 3.11.8, PyTorch 2.2.2,
zuko 1.1.0, NumPy 1.26.4 and SciPy 1.13.1. The paper developed against older
Python/PyTorch versions; the native tutorial and four published cluster medians
provide explicit numerical compatibility checks. This environment does not
replace the original equation-study environment.

Run benchmark scripts with `PYTHONPATH=src .venv-published/bin/python`. Training
uses one CPU thread per process and deterministic operations. The pinned upstream
weights and the newly trained weights are distinct; only newly trained holdout
weights enter the controlled 48-split comparison. The unchanged upstream model
is used for numerical reproduction and the clearly labeled supplementary check.

The reference training manifest also checks environment/platform metadata.
For a fresh run on a different machine or environment, use a separate checkout
and a new `study_id` in `configs/published_benchmark.json`; do not overwrite the
reference bundle. The fitting and evaluation scripts follow that setting. The
report generator reads the checked-in `published_benchmark_v1` reference bundle.

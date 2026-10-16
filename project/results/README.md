# Result provenance

`published_benchmark_v1/` contains the published-model follow-up: reproduction
with the authors' unchanged ChronoFlow weights, 96 independently trained neural
fits on the same 48 earlier splits, matched scores and ages, external M67
comparisons, and a supplementary released-weight comparison on three excluded
clusters. Its protocol distinguishes numerical reproduction from retraining of
the published architecture. All weights, learning curves, choices and source
hashes are saved. See [the benchmark report](../../docs/PUBLISHED_BENCHMARK_RESULTS.md).

`real_stars_v2/` contains the current observational follow-up: six distribution
families, a validation-selected family, 48 frozen cluster splits, per-star scores,
age-likelihood profiles, an external M67 check, and a separate application fit.
Its manifest records source, data, configuration and split hashes. External
results record the exact frozen model hashes and the original-catalogue overlap
ledger. See [the results](../../docs/REAL_STAR_RESULTS.md).

`research_v1/` retains the initial reproducible equation study, with 180 synthetic
and 48 observational searches. `domain_confirmation/` contains its 30 fresh-seed
follow-up searches. Candidate tables and per-star held-out predictions make
these comparisons inspectable; their conclusions have not been replaced.

Other files and directories are historical exploratory outputs. They are retained
for audit and do not support the current project's conclusions. In particular,
legacy equation-stability counts, field-star ages and M67 residuals should not be
interpreted as established physical discoveries. See `../../docs/RESEARCH_LOG.md`.

In the equation-study per-trial JSON, a null error means an invalid prediction with infinite
penalty. The reporting script restores `inf` in CSV scores and counts the failures;
it never omits these trials. `raw_search/` contains ignored transient checkpoints;
the saved text candidate tables are the portable research record.

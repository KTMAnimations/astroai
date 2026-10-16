# Result provenance

`research_v1/` contains the current, reproducible study. Its manifest records
source/data hashes, versions, settings and seeds. Candidate tables and per-star
held-out predictions make the reported comparisons inspectable.

Other files and directories are historical exploratory outputs. They are retained
for audit and do not support the current project's conclusions. In particular,
legacy equation-stability counts, field-star ages and M67 residuals should not be
interpreted as established physical discoveries. See `../../docs/RESEARCH_LOG.md`.

In current per-trial JSON, a null error means an invalid prediction with infinite
penalty. The reporting script restores `inf` in CSV scores and counts the failures;
it never omits these trials. `raw_search/` contains ignored transient checkpoints;
the saved text candidate tables are the portable research record.

# Research log

## 2026-10-15: Audit and redesign

The initial audit reproduced the legacy fits but found unsupported scientific
claims, a failed complex-truth control, a NaN-scoring error, broken verification,
missing producer scripts, inconsistent sample descriptions, and clipped M67
plots. The current study uses a new package and output directory so those
historical results remain distinguishable from the verified follow-up.

Checked the original catalogue against the upstream ChronoFlow file at commit
`f0b766f7d057c4e8f9a64a867414d25c5d985cd6`: the files match byte for byte.
The upstream instructions confirm the four photometric/astrometric quality cuts.
The paper specifies HDBScan membership with a source-probability fallback.

The primary selection produces 721 stars in 16 clusters. The restriction to
100 Myr and older, 0.65 < BP-RP < 1.5, membership >= 0.8, and >= 10 stars per
cluster was chosen before running the new model comparisons. It supports a
limited population-level prediction experiment, not a universal gyrochronology law.

## Execution pilots (excluded from results)

A short, balanced-input interaction trial checked the end-to-end code: 20 search
iterations, 6 populations, 100 mutation cycles. It completed in 20.9 seconds
including initialization. A separate-seed trial with the planned full budget
(100 iterations, 12 populations, 200 cycles) completed in 26.7 seconds.
The pilot supports running the fixed main budget on this machine. Pilot outputs
live in ignored `tmp/` directories and do not enter the study tables.

The main configuration has 180 controlled searches and 48 cluster-holdout
searches (16 clusters times 3 split/search seeds). Independent searches may run
in separate processes; every individual search uses deterministic serial PySR.
The protocol, seeds and selection criteria are fixed before the main execution.

The study tests recovery in function space, because algebraically different
expressions can make equivalent predictions. It reports continuous errors and
an explicit recovery threshold rather than claiming expression recurrence proves
physical truth.

## Main-run completion and reporting review

All 180 synthetic and 48 observational searches completed. Replaying the
nontrivial `interaction_balanced_noise0.05_seed00` trial reproduced the same
17 candidate equations and losses. The source hashes and all original trial
outputs are retained.

The reporting review found two validation-selected synthetic equations that
were finite on the sampled validation inputs but undefined elsewhere on the
full evaluation grid. Both arose in noisy, catalogue-sampled saturation trials.
The experiment correctly assigned infinite error and marked them unrecovered.
Strict JSON encoded these scores as null; a first draft of the summary then
treated the nulls as missing. The reporting script now restores infinite error,
counts invalid functions explicitly, and includes those trials in comparisons.
The trial results and selection rules were not changed. A regression check
ensures that every synthetic comparison, including failures, is accounted for.

## Exploratory domain-check follow-up (rule specified before evaluation)

The two failures motivate a separate, post-hoc reliability check. For each saved
candidate set, first reject functions with any nonfinite prediction on a
feature-only 81-by-51 mesh spanning the declared age/color domain. Then choose
the remaining function with minimum saved validation error, breaking ties by
complexity and candidate ID. The selector never receives true grid targets or
test errors. Evaluate the frozen choice on the original 80-by-50 truth grid.

This reuses already inspected trials and is an exploratory demonstration, not
an independent confirmation. It does not replace the primary results. A finite
mesh cannot prove that a function is defined everywhere in a continuous domain.

The reused-trial follow-up changed two choices, removed the two observed domain
failures, and raised recovery from 157/180 to 159/180. No main result was replaced.

## Fresh-seed domain-check confirmation (fixed before running)

To test the guard beyond the trials that motivated it, run 30 fresh searches:
all three known functions, catalogue sampling, 0.15 dex noise, ten trials each.
Use seed base 20361015, outside the main study's seed range. Retain the same
operators, budget, training/validation sizes and feature-only guard. Compare
invalid counts, recovery counts and continuous errors before and after the
guard. This condition targets the sampling/noise regime where the failures
occurred; it does not estimate an all-condition average. Complete all 30 trials
regardless of outcome and report failures as well as improvements.

### Confirmation outcomes

All 30 searches completed (247 seconds of summed search time). Default selection
recovered 0/30 functions, validation recovered 21/30, and a recovering candidate
was present in 28/30 searches. No selected function was invalid on the evaluation
grid. The domain guard changed no choices, leaving 21/30 recoveries. The fresh
trials support the main selection finding in this high-noise catalogue condition,
but do not demonstrate an extra benefit from the domain guard. No further seed
batch was added to seek a different result.

The full study now contains 258 searches, plus two excluded execution pilots and
one exact-seed replay. The seven current figures were visually inspected, and
the four-page PDF was rendered page by page. Legacy claims are labeled historical;
the root README now reports the verified results and their limitations.

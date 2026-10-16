# Data provenance

`prot_catalogue.csv` comes from the [ChronoFlow catalogue](https://github.com/philvanlane/chronoflow/blob/f0b766f7d057c4e8f9a64a867414d25c5d985cd6/data/prot_catalogue.csv),
described by [Van-Lane et al. (2025)](https://arxiv.org/abs/2412.12244).
It contains 8,412 entries, including sources excluded from ChronoFlow training.
The local file was verified byte for byte against that pinned upstream version.

SHA-256: `577e4366d164a2c7d9f77320cf86c1da3fbd94ead92ddb2a63d5c10661674238`.

The active pipeline reads this original file. It retains identifiers, period
sources, membership probabilities and quality flags. Its generated selection and
exclusion ledger are in `../results/research_v1/`. See the repository's
`docs/PROTOCOL.md` for exact choices and their limitations.

The `clean_*.csv`, McQuillan, Berger and residual tables belong to the historical
analysis. The new study does not use the legacy field-star age catalogue as
validation. Catalogue measurements and cluster ages retain their original
observational and calibration uncertainties.

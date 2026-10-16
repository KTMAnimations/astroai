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

The Python lock also includes ReportLab and pypdf for report generation and PDF
inspection. Poppler's `pdftoppm` was used for visual inspection; it is a separate
system dependency and is not needed to run the numerical study or build the PDF.

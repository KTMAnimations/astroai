# Research Plan: PySR Symbolic Regression for Gyrochronology

**Proposal 4 from top_5_proposals.md**

This document is a step-by-step guide from an empty folder to a completed paper. Every step, parameter choice, and evaluation criterion is explained without specialist vocabulary.

---

## What is being created that is new

The naive version of this project — "install PySR, call model.fit(), report the best equation" — is not publishable. Running a turnkey tool on someone else's dataset is not a contribution. Miles Cranmer, PySR's creator, explicitly warns on his blog that "many ML applications in astrophysics aren't necessary" and that "a simpler algorithm will often do just as well." This project must pass Cranmer's own test.

**Critical prior work that this project must position against:** Lu & Angus 2024 (AJ 167, 159, arXiv:2310.14990) published a Gaussian Process-based gyrochronology calibration valid from 0.67 to 14 Gyr that already captures the weakened magnetic braking transition and ages ~100,000 stars. Their GP is flexible and data-driven — conceptually the same motivation as this project. **This project does NOT claim to replace the GP.** Note: Lu & Angus explicitly exclude stars with Rossby number > 1.866 (the critical WMB onset value) from their model's validity domain, so their GP does NOT capture the weakened magnetic braking transition itself — it covers the age range but avoids the regime. The contribution here is different: a **closed-form, interpretable equation** that can be written on a whiteboard, inspected for physical meaning, and tested for uniqueness via bootstrap stability analysis. The GP is a black box; PySR produces an equation. A reviewer will ask "why not just use the GP?" — the answer is interpretability and the ability to test whether the data uniquely constrains the functional form.

The novel contribution has three parts. Crucially, the **scientific question** comes first: **Does the functional form of gyrochronology change at old ages, and can data-driven equation discovery detect the transition to weakened magnetic braking?** The PySR machinery is the tool used to answer this question, not the contribution itself.

1. **Physics-constrained symbolic regression.** Instead of letting PySR search freely, define a custom loss function that penalizes physically impossible equations: negative periods, non-monotonic age dependence (period must increase with age for main-sequence stars in the 100 Myr–3 Gyr range where standard magnetic braking applies), and singularities at valid input values. PySR supports custom Julia loss functions — you write one that encodes gyrochronology physics. This transforms the project from "I ran PySR" to "I developed a physics-constrained symbolic regression framework for stellar age-dating." **Note on prior work:** PhySO (Tenachi et al. 2023, ApJ 959, 99, arXiv:2303.03192) is a separate symbolic regression framework that uses dimensional analysis as physics constraints and was published in ApJ. This project's approach is different (penalty-based soft constraints on physical behavior rather than hard dimensional-analysis constraints) and uses PySR rather than PhySO. Custom physics-penalty loss functions in PySR are now established practice: Salim et al. (2025, arXiv:2505.04681, co-authored by Cranmer) used a custom Julia loss for star formation parameterization. The novelty here is the **application domain** (stellar rotation / gyrochronology), not the technique of custom losses itself. **Note:** PySR has been applied to many astronomy problems beyond neutron stars — including galaxy-halo connection (Cranmer et al. NeurIPS 2020), cluster SZ flux-mass relations (Wadekar et al. PNAS 2023), black hole mass scaling relations, star/galaxy/quasar classification (arXiv:2602.24022), Sersic profile emulation (arXiv:2508.20266), and more. See the PySR papers page (ai.damtp.cam.ac.uk/pysr/papers/) for the full list. The novelty claim here is specifically: **no PySR application to stellar rotation or age-dating with physics-constrained loss functions.**

2. **Bootstrap stability of the functional form.** Run PySR 100 times on bootstrap resamples of the training data. For each run, record the top-5 Pareto-optimal equations. Count how often each functional FORM appears across the 100 runs (ignoring exact coefficient values). If the same form appears 80+ times, it is robust. If 10 different forms appear, the data does not uniquely constrain the functional form — and that itself is a publishable finding about the information content of the data. To our knowledge, no PySR paper has reported systematic functional-form stability analysis at this scale (100 resamples with canonicalization). Related concepts exist in the broader SR literature (ARGOS uses bootstrapped confidence intervals; a 2023 ACS Omega paper introduced a stability metric), but the specific combination of PySR + bootstrap + canonical form counting appears novel. This methodology would be reusable by future PySR studies.

3. **Out-of-distribution extrapolation to the stalled-spin-down regime.** The ChronoFlow training set only goes up to ~4 Gyr. Use 15-25 well-characterized old stars with *surface* rotation periods AND asteroseismic ages to test whether the PySR equation extrapolates correctly into the regime where magnetic braking weakens. The existence of weakened magnetic braking (WMB) is now well-established (Hall et al. 2021 found WMB 98.4% more likely than standard braking; Metcalfe et al. 2025 ApJ 991 L17 confirmed it "unambiguously"). What remains debated is the mechanism, mass dependence, and critical Rossby number. If PySR's equation — discovered from young-star data alone — naturally captures or fails to capture the braking transition, that is a genuine astrophysical finding about what the young-star rotation data implies about old-star physics.

The ChronoFlow residual analysis is still done, but it is a secondary experiment with explicit physical interpretation (not just RMSE reporting).

---

## Phase 1: Environment setup

**Time: 30 minutes**

1. Create a Python environment:
   ```
   python -m venv pysr_gyro_env
   source pysr_gyro_env/bin/activate
   ```

2. Install PySR (this also installs Julia automatically):
   ```
   pip install pysr numpy pandas matplotlib scipy sympy
   ```
   Current PySR versions install Julia dependencies automatically on first import (`import pysr`). The older `pysr.install()` call may still work but is not the canonical setup path. Verify against the current README at github.com/MilesCranmer/PySR. First import takes 2-5 minutes.

3. Install ChronoFlow dependencies (to run the neural model for comparison):
   ```
   pip install torch zuko
   ```

4. Clone the ChronoFlow repo:
   ```
   git clone https://github.com/philvanlane/chronoflow.git
   ```

5. Create your project structure:
   ```
   mkdir -p project/{data,results,figures,notebooks}
   ```

---

## Phase 2: Download and understand the data

**Time: 2-3 hours**

1. The training data lives in the ChronoFlow repo under `data/`. Copy it to your project:
   ```
   cp -r chronoflow/data/* project/data/
   ```

2. Open the data in a notebook and inspect it. The dataset contains **7,615** unique Gaia DR3 main-sequence sources from 30 open clusters and associations (Van-Lane et al. 2025, ApJ, arXiv:2412.12244). Each star has:
   - **Rotation period** (days) — measured from photometric variability
   - **De-reddened Gaia DR3 color** (BP-RP)_0 — a proxy for mass/temperature (redder = cooler = lower mass)
   - **Cluster membership** — which cluster the star belongs to
   - **Cluster age** (log years or Myr) — the age assigned to the entire cluster from isochrone fitting

3. Understand the variables you will use:
   - **Input features for PySR:** age (in Myr or log10(age/yr)), (BP-RP)_0 color
   - **Target for PySR:** rotation period (days)
   - You are searching for an equation: `P_rot = f(age, color)`

4. **Filter to slow-sequence rotators (MANDATORY before running PySR).**

   Young open clusters (<500 Myr) show a bimodal period distribution: a "fast sequence" (C sequence, disk-locked/rapid rotators) and a "slow sequence" (I sequence, the converged spin-down sequence that gyrochronology targets). The slow sequence doesn't fully converge until ~500-700 Myr for G dwarfs and ~2-3 Gyr for K dwarfs (Spada & Lanzafame 2020). PySR fits a single deterministic function P = f(age, color) and **cannot represent bimodality**. If fast rotators dominate the young data (81% of training stars are <500 Myr), PySR will either find a compromise equation that fits neither sequence, or fit the dominant fast rotators and fail on the slow sequence.

   **Fix:** ChronoFlow models the full bimodal distribution using the normalizing flow — it does NOT perform explicit slow/fast sequence separation or provide per-star slow-sequence labels. The student must implement slow-sequence filtering independently using published slow-sequence fits from Curtis et al. (2019, 2020), Dungee et al. (2022), or Bouma et al. (2023). Practically: for each cluster, plot period vs color and visually identify the slow-sequence ridge, or apply sigma-clipping to remove outlier fast rotators above the ridge. **Allocate 2-3 days in Week 1 for this curation.** Also run PySR on the unfiltered data and compare — this ablation shows whether filtering matters.

5. **Understand the training data skew.** The age distribution is severely unbalanced:
   - Clusters >2 Gyr: 3 clusters (NGC 6819, Ruprecht 147, M67) — ~483 stars (~6%)
   - Clusters >1 Gyr: 6 clusters — ~833 stars (~11%)
   - Clusters <500 Myr: 24 clusters — ~6,152 stars (~81%)
   - M67 (3.87 Gyr) is the ONLY cluster above 3 Gyr
   PySR's equation will be well-constrained below 500 Myr and poorly constrained above 1 Gyr. Any exponent-level claim is driven almost entirely by young clusters. **Acknowledge this in the paper and interpret results accordingly.** The statement "ages 1.5 Myr to ~4 Gyr" is literally true but seriously misleading about sampling density.

5. **Quality check:** Make a scatter plot of rotation period vs (BP-RP)_0, colored by cluster age. You should see that younger clusters (blue dots) have shorter periods and older clusters (red dots) have longer periods, with a clear color-dependence. If this pattern is not visible, something is wrong with the data loading.

5. **Unit decisions:** Convert age to Myr (not log). Using linear age values (1–4000) rather than log-transformed values allows PySR to choose between `log(age)`, `age^n`, `sqrt(age)`, and other functional forms. Note: PySR can only discover logarithmic relationships if `log` is included in the operator set (Phase 4 includes it). Transcendental functions like `log` cannot be exactly composed from algebraic operators (+, ×, ^). In practice, PySR struggles to approximate log-like relationships without `log` in the operator set (see PySR GitHub issue #7 for an example). The practical recommendation: always include `log` in the operator set when log relationships are plausible.

---

## Phase 3: Implement the classical baselines

**Time: 3-4 hours**

Before running PySR, implement the three classical gyrochronology relations so you have baselines to compare against.

### Barnes 2007

The relation has the form:
```
P = a × (color - 0.4)^b × age^c
```
where `a`, `b`, `c` are fitted constants. The original values are approximately a=0.7725, b=0.601, c=0.5189 (check the paper for exact values). The color here is (B-V); you will need to convert Gaia (BP-RP)_0 to (B-V) using a standard transformation or refit the constants to Gaia colors.

Actually, the simplest approach: **refit** Barnes-form, Mamajek-Hillenbrand-form, and Angus-form equations to the ChronoFlow training data using `scipy.optimize.curve_fit`. This gives each classical relation the best possible chance on this specific dataset.

### Mamajek-Hillenbrand 2008

```
P = a × (color - c_0)^b × age^n
```
Similar form to Barnes but with slightly different constants and a different zero-point color c_0.

### Angus et al. 2019

Uses a significantly different functional form from Barnes — additive in log-space with a polynomial color dependence:
```
log₁₀(P_rot) = c_A × log₁₀(t) + Σ(c_n × [log₁₀(BP-RP)]^n, n=0..4)
```
where c_A ≈ 0.65 (approximately Skumanich-like but not forced to 0.5) and the 5th-order polynomial captures the complex color dependence. This has more free parameters than Barnes but is more flexible. Extract the coefficients from the paper (AJ 158, 173) directly. **Do NOT use the `stardate` package** — it has had no release since September 2019, is not pip-installable from PyPI (the `stardate` on PyPI is a different package for date formatting), and has a complex dependency chain.

### The dead-simple baseline (MANDATORY — recommended by Huppenkothen et al. 2023 white paper)

The Huppenkothen best-practices checklist requires comparison to "the simplest model possible without sacrificing performance." Before running PySR, fit the Barnes functional form `P = a × (color - c0)^b × age^c` using `scipy.optimize.curve_fit`. This is a 4-parameter nonlinear least-squares fit that takes less than 1 second.

If PySR's best equation has the same functional form as Barnes (a power law in age and color), the PySR contribution is NOT the equation — it is the bootstrap stability analysis proving that this form is uniquely optimal. If PySR finds a structurally different equation (e.g., with a logarithmic age term or a color-age interaction), compare its RMSE and cross-validation performance to the refitted Barnes. The improvement must exceed the uncertainty from cross-validation to be meaningful. A 0.1% RMSE gain does not justify added complexity (per the Buchner & Fotopoulou, Nature Reviews Physics 6, 535-545, 2024 guidance on ML in astronomy).

### Lu & Angus 2024 GP baseline

Lu & Angus 2024 (AJ 167, 159) is the current state-of-the-art data-driven gyrochronology model. It uses a GP calibrated on cluster data from 0.67 to 14 Gyr. If their trained GP or code is publicly available, include it as a baseline. If not, note in the paper that the GP exists and outperforms parametric relations at old ages — PySR's value proposition is interpretability, not accuracy.

### For each baseline:
1. Fit the free parameters to the training data using `scipy.optimize.curve_fit`
2. Compute RMSE on the training data AND on held-out clusters (leave-one-out)
3. Store the fitted parameters
4. Record the number of free parameters (for fair complexity comparison with PySR)

---

## Phase 4: Run PySR — the direct search

**Time: 1-2 days of compute (CPU), hands-off**

This is the core experiment. PySR will search through combinations of mathematical operators to find an equation that predicts rotation period from age and color.

### Setting up the physics-constrained PySR search

The key modification from a naive PySR run: a custom loss function written in Julia that penalizes physically impossible equations.

```python
import pysr
import numpy as np

X = np.column_stack([ages_myr, colors_bprp0])  # shape (N, 2)
y = periods_days  # shape (N,)

# Physics-constrained loss function (written in Julia syntax, passed as a string)
# Penalizes: negative predictions, and predictions that decrease with age
physics_loss = """
function physics_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    # Standard MSE
    diffs = prediction .- dataset.y
    mse = sum(diffs .^ 2) / length(diffs)

    # Penalty: negative periods
    neg_penalty = sum(max.(0, -prediction)) / length(prediction)

    # Penalty: non-monotonic with age — HEAVILY RESTRICTED.
    #
    # IMPORTANT: Standard magnetic braking (monotonic spin-down) applies ONLY to:
    #   - Ages 100 Myr to ~1 Gyr (before stalled spin-down in K dwarfs)
    #   - Colors (BP-RP)_0 ~ 0.6 to 1.1 (late F through early K dwarfs)
    #
    # It does NOT apply to:
    #   - Ages < 100 Myr: PMS spin-up during contraction
    #   - Ages > ~1 Gyr for K dwarfs: stalled spin-down (Curtis et al. 2019, ApJ 879, 49
    #     showed K dwarfs in NGC 6811 at 1 Gyr rotate at same rate as Praesepe at 670 Myr)
    #   - Colors < 0.6: above the Kraft break, weak/absent magnetic braking
    #   - Colors > 1.1: late K/M dwarfs with extended stalled phases
    #
    # RECOMMENDED: Run PySR TWICE — once with this restricted penalty, once WITHOUT
    # any monotonicity penalty. Compare the Pareto frontiers. If the unconstrained
    # search finds physically sensible equations, the penalty is unnecessary.
    # If it finds pathological equations, the penalty is justified.
    ages = dataset.X[1, :]
    colors = dataset.X[2, :]
    mask = (ages .> 100.0) .& (ages .< 1000.0) .& (colors .> 0.6) .& (colors .< 1.1)
    if sum(mask) > 10
        masked_ages = ages[mask]
        masked_pred = prediction[mask]
        sorted_idx = sortperm(masked_ages)
        sorted_pred = masked_pred[sorted_idx]
        mono_violations = sum(max.(0, sorted_pred[1:end-1] .- sorted_pred[2:end]))
    else
        mono_violations = zero(L)
    end

    return mse + 10.0 * neg_penalty + 1.0 * mono_violations
end
"""

model = pysr.PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/", "pow"],
    unary_operators=["log", "exp", "sqrt"],  # abs REMOVED: creates non-differentiable points, no known physical mechanism produces |x| dependence in rotation
    populations=30,
    population_size=50,
    maxsize=25,
    batching=True,
    batch_size=500,
    loss_function=physics_loss,  # NOTE: Both 'loss_function' and 'full_objective' may work as aliases depending on PySR version. Verify against your installed version: `python -c "import pysr; help(pysr.PySRRegressor)"`. Current docs (v1.5.9) list 'loss_function'.
    complexity_of_operators={
        "pow": 2, "log": 1, "exp": 1, "sqrt": 1
    },
    # select_k_features removed — with exactly 2 input features, selecting top 2 of 2 is a no-op
    progress=True,
    output_directory="project/results/pysr_constrained",  # saves results to this permanent path
)

model.fit(X, y, variable_names=["age", "color"])
```

Run this TWICE: once with the physics-constrained loss, once with plain MSE. Compare the Pareto frontiers. The constrained search should produce equations that are physically sensible everywhere, while the unconstrained search may find equations that fit better on the training data but blow up at extreme ages or colors.

### What PySR does internally

PySR uses an evolutionary algorithm (genetic programming). It maintains a "population" of candidate equations, each represented as a tree of mathematical operations. In each generation:
1. It randomly mutates some equations (change an operator, add a term, remove a term)
2. It evaluates how well each equation fits the data
3. It keeps the best equations and discards the worst
4. It also tracks a "Pareto frontier" — the set of equations that are the best at each complexity level

After the search finishes, PySR returns a table of equations ranked by the trade-off between accuracy and simplicity. The simplest equation might be `P = 3.2 * age^0.5`, and the most complex might be a 20-term expression. The interesting ones are in the middle: equations that are simple enough to interpret but accurate enough to be useful.

### Interpreting the output

```python
print(model)  # shows the Pareto frontier: complexity vs loss
best = model.get_best()  # the equation PySR recommends
print(best.equation)  # the symbolic expression as a string
print(best.loss)       # the MSE on the training data
```

Save the full Pareto frontier as a CSV:
```python
model.equations_.to_csv("project/results/pareto_direct.csv")
```

### Running time

With 8,000 data points, 200 iterations, and 30 populations, this takes approximately 6-24 hours on a modern laptop CPU (4-8 cores). PySR parallelizes across cores automatically. You can start with 50 iterations to verify it works, then increase to 200 for the final run.

---

## Phase 5: Run ChronoFlow and collect residuals

**Time: 3-4 hours**

1. Follow the ChronoFlow tutorials in `chronoflow/tutorials/` to load the pre-trained model and generate age predictions for each star in the training set.

2. For each star, compute the ChronoFlow-predicted rotation period at the known cluster age and color. **NOTE:** ChronoFlow is a normalizing flow that models P(log P_rot | log age, (BP-RP)₀) — it returns probability distributions, not deterministic point predictions. The `predict_period()` function shown below may not exist as written. **Examine the ChronoFlow tutorials in `chronoflow/tutorials/` for the actual API** (likely `calcProbsAtAge()` or similar). Extract the median or mode of the conditional distribution as the point prediction:
   ```python
   # PSEUDO-CODE — verify against actual ChronoFlow API
   # The flow returns a distribution; extract the median as the point estimate
   P_chronoflow_median = np.median(chronoflow_model.sample(age=age, color=color, n=1000))
   ```

3. Compute the residual for each star:
   ```python
   residual = P_observed - P_chronoflow
   ```

4. Store the residuals alongside the original features. You now have a new dataset: `(age, color) → residual`.

---

## Phase 6: Run PySR on the residuals — with physical interpretation

**Time: 1-2 days of compute (CPU), hands-off**

Repeat the PySR search from Phase 4, but now the target variable is the residual instead of the raw period:

```python
model_resid = pysr.PySRRegressor(
    # same settings as before, but WITHOUT the physics-constrained loss
    # (residuals can be positive or negative, and need not be monotonic)
    ...
)
model_resid.fit(X, residuals, variable_names=["age", "color"])
```

The goal here is different: you are looking for a simple correction term that explains the systematic errors in ChronoFlow. The ideal outcome is an equation like:
```
correction = 0.3 * log(age) * color^(-0.5)
```
which would tell astronomers that ChronoFlow underestimates periods for old, blue stars (or whatever the pattern turns out to be).

### Physical interpretation of residuals (MANDATORY — do not just report RMSE)

Before running PySR on residuals, first LOOK at the residuals:

1. **Bin residuals by color (BP-RP)_0 in 0.2 mag bins.** Plot mean residual and standard deviation per bin. Are there systematic offsets for M dwarfs (red, (BP-RP)_0 > 1.5) vs F/G stars (blue, (BP-RP)_0 < 0.8)? If yes, this connects to the Kraft break — the transition between convective and radiative envelopes.

2. **Bin residuals by cluster age.** Plot mean residual vs age. Is there a systematic age trend? If ChronoFlow underpredicts periods for old clusters, that connects to the stalled-spin-down problem.

3. **Bin residuals by cluster.** Are there specific clusters where ChronoFlow fails badly? If Praesepe or NGC 6811 shows large residuals, check whether those clusters have known peculiarities (binary contamination, differential reddening).

4. **Interpret the PySR correction equation physically.** If PySR finds a correction term, does it make physical sense? A correction that depends on color but not age suggests a mass-dependent systematic in ChronoFlow. A correction that depends on age but not color suggests a time-dependent systematic (perhaps related to spin-down law assumptions). Report the physical interpretation, not just the equation.

Even if the best residual equation has a large RMSE (meaning ChronoFlow's errors are mostly random, not systematic), that is a publishable finding — it means ChronoFlow has captured the deterministic component of the rotation-age-mass relationship and the remaining scatter is intrinsic.

**CIRCULARITY WARNING:** The residual analysis above uses ChronoFlow's predictions on its OWN training data. Residual patterns may reflect ChronoFlow's finite capacity/inductive bias rather than genuine systematic errors on unseen stars. **To make the physical interpretation honest, run Phase 6 on leave-one-cluster-out (LOO) residuals instead:** for each cluster, use the ChronoFlow prediction from the model that was NOT trained on that cluster (from Phase 9's LOO runs). This requires running ChronoFlow LOO first (adds compute), but makes the residual analysis genuinely out-of-sample.

---

## Phase 7: Bootstrap stability analysis (MANDATORY)

**Time: 2-3 days of compute (parallelizable)**

This experiment answers: "Is the functional form PySR finds robust, or does it change every time you resample the data?"

**IMPORTANT METHODOLOGICAL NOTE:** PySR is inherently stochastic (multi-population evolutionary algorithm with separate islands on different cores). The bootstrap analysis will capture BOTH data sampling uncertainty AND PySR's algorithmic randomness, conflated into a single measure. To properly claim "the data uniquely supports form X," you must disentangle these two sources of variability:

**Step A — Isolate algorithmic noise (MANDATORY, do FIRST):** Run PySR 10 times on the SAME full dataset with different random seeds. If the same functional form appears in 8+/10 runs, PySR's algorithmic noise is low and the bootstrap analysis will primarily reflect data uncertainty. If different forms appear each time, PySR has not converged and more iterations or a larger population are needed before the bootstrap is meaningful. Use `deterministic=False` (default) but log the random seed for each run.

**Step B — Bootstrap for data uncertainty:** Only after Step A confirms algorithmic convergence:

1. Generate 100 bootstrap samples of the training data (sample N stars with replacement from the N-star dataset). **If compute is tight, reduce to 50 resamples** — still unprecedented in the PySR literature.
2. Run PySR on each bootstrap sample with the physics-constrained loss, using 50 iterations (shorter than the main run). **Realistic compute estimate — CORRECTED:** PySR uses ALL available CPU cores for its multi-population island model by default. This means bootstrap runs must be **sequential** (each using all cores), NOT parallel across cores. 100 sequential runs × 6-12 hours each = **600-1200 hours wall-clock on a single machine (25-50 days)**. This is far longer than previously estimated.

   **Mitigations (pick one):**
   - Reduce to **50 resamples** (still unprecedented) → 12-25 days
   - Use `procs=2` to limit PySR to 2 cores, then run 4 jobs in parallel on an 8-core machine → ~6-12 days (but each run takes longer with fewer cores — test timing first)
   - Use a university computing cluster to run jobs truly in parallel across nodes
   - Reduce to **30 iterations** per run (check Step A first to verify convergence at 30 iterations)

   **Allocate 2-4 weeks for this phase, not "2-3 days."** This is the largest computational phase and the timeline has no slack if runs take longer than expected.
3. For each run, record the top-3 equations on the Pareto frontier.
4. **Canonicalize functional forms before counting.** This is harder than it sounds — budget 50-100 lines, not 20. Three issues must be handled:

   **(a) Commutativity:** PySR can produce `a * age^0.5 * color^0.6` and `color^0.6 * age^0.5 * a`. Use PySR's built-in `model.equations_['sympy_format']` column (already-parsed SymPy objects) rather than calling `sympy.sympify()` on string representations — this avoids parsing failures on custom operators. Then call `sympy.simplify()`.

   **(b) Algebraic equivalence and coefficient variation — use the PLACEHOLDER approach ONLY.** Previous versions of this plan suggested `sympy.powsimp(force=True)` and "round to 1 decimal place." Testing reveals powsimp does NOT combine `a * x^0.5 * y^0.5` into `a * (x*y)^0.5` even with `force=True` and `positive=True` symbols. And rounding creates sharp boundaries (age^0.54 rounds to 0.5 but age^0.55 rounds to 0.6).

   **The correct approach:** Replace ALL numerical constants (both coefficients and exponents) with ordered placeholder symbols (`C0, C1, C2, ...`), ordered by first appearance in a depth-first traversal of the expression tree. Then call `sympy.simplify()` on the placeholder expression and hash the result. This cleanly separates "functional form" (the operator tree with variable names and placeholder constants) from "coefficient values" (the actual numbers). Two equations with the same operator structure but different coefficients hash to the same form.

   Example: `0.78 * age^0.51 * color^0.62` and `0.83 * age^0.49 * color^0.58` both become `C0 * age^C1 * color^C2` → same hash.

   This is ~50-80 lines of careful Python. Test on toy examples before running 100 bootstrap jobs.

   Test the canonicalization pipeline on toy examples before running 100 bootstrap jobs. If `a * age^0.5 * color^0.6` and `b * color^0.6 * age^0.5` don't hash to the same value, the pipeline is broken.

   **Also consider:** Run Step A (10 same-data runs) both WITH and WITHOUT batching (`batching=True` vs `batching=False`). Batching adds noise (evaluating fitness on random 500-point subsets). If Step A variability is high, you won't know if it's PySR's evolutionary algorithm or the batching. This diagnostic costs only 10 extra runs.
5. **Check whether different forms have distinguishable loss.** If forms A, B, and C all appear with training MSEs within 0.1% of each other, they are functionally equivalent and the bootstrap is measuring PySR's stochastic search, not data information content. Report the loss distribution per canonical form. Only claim "the data uniquely supports form X" if form X has statistically lower loss than alternatives.

6. Report:
   - The most common functional form and how often it appeared
   - The second and third most common forms
   - Whether the constrained and unconstrained searches converge to the same form
   - Whether the top forms have distinguishable or indistinguishable loss

This analysis is the methodological contribution that no other PySR paper has done. If the same form appears 80+ times out of 100, you can claim the data strongly supports that functional form. If many forms appear, that is equally interesting — it means the data does not uniquely determine how rotation, age, and color are related.

## Phase 8: Out-of-distribution extrapolation to the stalled-spin-down regime (MANDATORY)

**Time: 2-3 days**

This experiment is what connects the project to a major question in stellar physics and elevates it beyond a methods demonstration. The ChronoFlow training data spans ages 1.5 Myr to ~4 Gyr. Stars older than ~4 Gyr show "weakened magnetic braking" (WMB) — their rotation periods stop increasing as fast as the classical power-law relations predict. The existence of WMB is now well-established (van Saders et al. 2016, Nature 529, 181; Hall et al. 2021, Nature Astronomy 5, 707, who found WMB 98.4% more likely; Metcalfe et al. 2025, ApJ 991, L17, who confirmed it "unambiguously"). What remains debated is the physical mechanism — Metcalfe et al. (2025) propose dynamo collapse, but Levesque & Charbonneau (2025, arXiv:2510.14164) argue the coronal wind collapse hypothesis requires physically unrealistic parameters. The mass dependence and abruptness of the transition are also debated (Yang et al. 2025, A&A, propose a four-segment model).

### Assemble a systematic old-star test sample (realistic target: 15-25 stars)

Do NOT use just 3-5 test points. Assemble a proper sample. **Critical distinction:** use only stars with **surface** rotation periods (from spot modulation or chromospheric activity), not asteroseismic rotation rates (from rotational splitting of oscillation modes). Asteroseismic rotation measures a depth-averaged or core rotation rate that differs from the surface period gyrochronology uses. Mixing the two is a methodological error.

- **The Sun:** age = 4567 Myr, P_rot = 25.4 days, (BP-RP)_0 ≈ 0.82. The most precisely known calibration point.
- **~~16 Cyg A and B~~ — REMOVED.** Davies et al. (2015) explicitly state "we cannot detect surface rotation with the current Kepler data" for these stars. García et al. (2014) found no photometric variability. Their rotation periods (~23 and ~24 days) are from asteroseismic rotational splitting only. Including them violates this plan's own stated methodology. Their asteroseismic age (7.0 ± 0.3 Gyr, Metcalfe et al. 2015 ApJ 811 L37) can be noted for context but they cannot serve as gyrochronology test points.
- **van Saders et al. (2016, Nature 529, 181) sample:** 21 total Kepler stars with asteroseismic properties and rotation periods. **Carefully verify which stars have photometric surface periods** from McQuillan et al. (2014) or García et al. (2014) vs. only asteroseismic rotation rates. Only include stars with surface period measurements. NOT all have ages >4 Gyr — the sample spans a range. Expect ~8-12 stars genuinely in the stalled-braking regime (ages >4 Gyr) with confirmed surface rotation periods. Download from the van Saders et al. supplementary tables and cross-match with McQuillan.
- **Metcalfe et al. (2023, ApJL 948, L6):** This paper contains only **2 stars** (61 UMa at ~1.4 Gyr and tau Cet at ~9 Gyr). It is NOT a large sample. Tau Cet is valuable as an old calibration point; 61 UMa is within the training range and less useful.
- **Hall et al. (2021, Nature Astronomy 5, 707):** 91 main-sequence Kepler stars. **WARNING:** These have *asteroseismic* rotation rates (from rotational splitting), NOT photometric surface rotation periods. Use these for comparison/discussion ONLY, not as primary test points for gyrochronology. If used, clearly note the methodological difference. A subset may have independently measured surface periods from McQuillan et al. — cross-match to identify these.
- **García et al. (2014) / McQuillan et al. (2014):** Cross-match these photometric rotation catalogs with APOKASC asteroseismic ages (Pinsonneault et al. 2025) to identify additional old stars with surface periods. **COMPLETENESS WARNING:** McQuillan et al. detected periods for 34,030 stars with P = 0.2–70 days, but detection completeness drops dramatically for P > 30 days due to Kepler's quarterly spacecraft reorientations (~90-day segments). Stars >4 Gyr have periods >30 days — precisely where the catalog is least complete. The cross-match will yield some old stars, but biased toward the more active (shorter-period, anomalous) ones. Report this selection bias in the paper.
- **Pezzotti et al. (2026, A&A 706, A257, arXiv:2512.14517):** Revisits activity-rotation-age with Kepler LEGACY asteroseismic ages. May provide additional calibration points.

### Analysis

For each old star, compute:
- Barnes 2007 predicted period (extrapolated beyond its training range)
- Mamajek-Hillenbrand 2008 predicted period
- PySR predicted period (from the best physics-constrained equation)
- True measured period

Plot predicted vs true for all old stars, with each method as a different symbol/color. The key diagnostic:

1. **If PySR's equation (trained on data up to ~4 Gyr) naturally predicts the stalled spin-down** — i.e., it predicts shorter-than-classical periods for old stars — then the functional form captures the braking transition implicitly. This would mean the young-star data already contains information about the old-star regime, which is a genuine scientific discovery.

2. **If PySR's equation fails to predict the stalled spin-down** (predicts periods that are too long, matching the Barnes extrapolation), then the braking transition is NOT encoded in the young-star data. This is an equally publishable null result — it tells the community that you cannot learn the braking transition from cluster rotation data alone.

3. **If PySR finds a functional form with a natural saturation mechanism** — e.g., `P = a * log(age) * f(color)` or `P = a * age^b * (1 - exp(-age/τ))` — that form itself provides physical insight about the nature of the braking transition. Compare to the theoretical predictions of van Saders et al. (Rossby-number-dependent braking) and Metcalfe et al. (critical Rossby number).

### Connect to the physical debate

In the discussion section, interpret the extrapolation result in the context of:
- The critical Rossby number hypothesis (Ro_crit ≈ 2.16 on the van Saders et al. convective turnover timescale calibration, or Ro_crit ≈ 0.91 ± 0.03 on the Metcalfe et al. asteroseismically calibrated scale — these use different τ_cz calibrations, not contradictory values). Specify which scale when citing a number.
- The mass dependence of the transition (does PySR's equation predict different transition ages for different stellar masses/colors?)
- Whether the functional form discovered by PySR is consistent with any proposed physical mechanism

## Phase 9: Cross-validation across clusters (MANDATORY)

**Time: 1-2 days**

This is the most important verification step. Gyrochronology equations are only useful if they generalize to clusters not in the training set.

1. **Leave-one-cluster-out cross-validation:** For each of the 30 clusters:
   - Remove all stars in that cluster from the training data
   - Refit Barnes 2007, Mamajek-Hillenbrand 2008, and Angus 2019 on the remaining 29 clusters
   - Rerun PySR on the remaining 29 clusters (use fewer iterations, e.g., 50, to keep compute manageable)
   - Evaluate all methods on the held-out cluster
   - Record the per-cluster RMSE for each method

2. **Summary statistics:** For each method, report:
   - Mean RMSE across all 30 held-out clusters
   - Standard deviation of RMSE
   - Worst-case cluster (highest RMSE) — this reveals where each method fails

3. **Cluster age dependence:** Plot per-cluster RMSE vs cluster age for each method. Classical relations are known to degrade for old clusters (>1 Gyr). If PySR's equation also degrades, report this honestly.

---

## Phase 10: Verification, sanity checks, and failure modes

**Time: 2-3 days**

### Verification

1. **Physical sanity of the discovered equation:**
   - Does period increase with age? (It should, for all ages > ~100 Myr.)
   - Does period increase with color (redder = lower mass = longer period at fixed age, due to smaller moment of inertia — NOT "faster spin-down"; M dwarfs actually have weaker magnetic braking torques but end up with longer periods because they are smaller; Curtis et al. 2020, ApJ 904, 140)? (It should, at least for the converged slow-rotator sequence.)
   - Are there unphysical features like negative periods or division by zero at certain age/color values?

2. **Comparison to known functional forms:** Is the PySR-discovered equation recognizable? If PySR finds something like `P = a * age^0.52 * (color - 0.4)^0.6`, that is essentially Barnes 2007 rediscovered — and the contribution is the bootstrap proof that this form is uniquely stable, not the equation itself. If PySR finds something structurally different — say, with a logarithmic age dependence or an interaction term — that is scientifically interesting and the comparison to Barnes must be rigorous (cross-validated RMSE, not just training RMSE).

3. **Sensitivity to PySR hyperparameters:**
   - Run PySR three times with different random seeds. Do you get the same top equation? If yes, the result is robust. If the best equation changes each time, report the ensemble of equations on the Pareto frontier.
   - Increase `maxsize` from 25 to 35. Does a significantly better equation appear at higher complexity?

4. **Numerical stability:** Evaluate the equation on the full age range (1 Myr to 14 Gyr) and color range (0.3 to 3.0). Check for overflow, underflow, or NaN values.

### Where the method fails (MANDATORY — recommended by Huppenkothen et al. 2023 white paper)

5. **Color boundaries:** The discovered equation will likely diverge or become unphysical near (BP-RP)₀ < ~0.55 (hot stars above the Kraft break at Teff ~ 6250–6550 K with ~200 K width (Beyer & White 2024), where magnetic braking is not the dominant spin-down mechanism; the Kraft break is at (BP-RP)₀ ~ 0.54–0.60 per Beyer & White 2024, NOT at 0.4 — the 0.4 value is the B-V zero-point in Barnes 2007, a different color system) and (BP-RP)₀ > 2.5 (very low-mass M dwarfs that remain fully convective). Report these boundaries explicitly. Plot the equation's predictions across the full color range and mark where it becomes unphysical.

6. **Age boundaries:** The training data spans 1.5 Myr to ~4 Gyr. Report the equation's behavior at 0.1 Myr (pre-main-sequence) and 10+ Gyr (old disk/halo). If the equation predicts absurd values outside the training range, mark the extrapolation boundary clearly.

7. **PySR convergence failures:** If certain bootstrap resamples produce no valid equation (PySR fails to converge or all equations violate the physics constraints), report the failure rate. If 15/100 bootstraps fail, that is informative — it suggests the data is marginally constraining.

8. **The honest comparison to curve_fit:** If `scipy.optimize.curve_fit` with the Barnes functional form achieves the same cross-validated RMSE as PySR's best equation, say so plainly. The contribution in this case is the bootstrap stability analysis and the out-of-distribution test, not the equation itself. Do not oversell.

9. **Age-uncertainty sensitivity analysis (MANDATORY).** Cluster ages from isochrone fitting carry uncertainties of ~0.06-0.08 dex (15-20%), per ChronoFlow's own error budget. Different stellar evolution codes (PARSEC, MIST, BASTI) give systematically different ages at the ~0.1 dex level. This means PySR cannot meaningfully distinguish between age^0.50 and age^0.52 — the difference produces ~1% change in predicted period at 1 Gyr, far below intrinsic scatter. **To test whether PySR's functional form is robust to age uncertainty:** perturb all cluster ages by draws from a Gaussian with σ = 0.1 dex (the systematic uncertainty between evolution codes). Rerun PySR on the perturbed data 10 times. If the same functional form appears, it is robust to age uncertainty. If the form changes, the exponent-level claims are fitting noise in the ages, not discovering physics. Report this analysis in the paper — it is the most important sanity check for any symbolic regression result where inputs have known systematic uncertainties.

10. **RMSE noise floor.** State explicitly in the paper: the minimum achievable RMSE is set by the combination of (a) intrinsic scatter in stellar rotation periods at fixed age and mass (which is large, especially for M dwarfs), and (b) cluster age uncertainties (~15-20%). If PySR and Barnes differ by less than this floor, the difference is not statistically significant. Compute this floor and report it alongside all RMSE comparisons.

---

## Phase 11: Paper writing plan

### RNAAS version (1500 words, 1 figure)

**Title:** "Does Stellar Spin-Down Follow a Unique Functional Form? A Symbolic Regression Test"

Note: the title frames the astrophysical question, not the tool. "PySR" does not appear in the title.

1. **Abstract** (150 words)
   - Problem: gyrochronology relations are either hand-crafted (Barnes) or black-box (ChronoFlow)
   - Method: PySR symbolic regression on 7,615 cluster rotators from ChronoFlow's training set
   - Result: the best PySR equation, its RMSE vs Barnes/MH/Angus, and whether it recovers a known functional form or discovers a new one
   - Implication: either the classical functional form is (approximately) optimal, or there is a better simple equation that the community should adopt

2. **Body** (~1200 words)
   - Paragraph 1: What gyrochronology is and why closed-form relations matter
   - Paragraph 2: What PySR is and how symbolic regression works (one paragraph, no math)
   - Paragraph 3: Data — ChronoFlow training set, 30 clusters, 7,615 rotators
   - Paragraph 4: Results — the Pareto frontier, the best equation in LaTeX, comparison table
   - Paragraph 5: Cross-validation results — per-cluster RMSE
   - Paragraph 6: Residual analysis — what PySR finds in ChronoFlow's errors

3. **Figure**
   - A 2-panel figure:
     - Left: the Pareto frontier (complexity vs loss), with Barnes, MH, and Angus marked as horizontal lines at their respective complexities and losses
     - Right: observed vs predicted rotation period for the best PySR equation, colored by cluster age

4. **References** (~10 references: Barnes 2007, MH 2008, Angus 2019, Van-Lane 2025, Cranmer 2024 PySR, Cranmer 2020 NeurIPS, Udrescu-Tegmark 2020, OGLE catalog if used)

### Full MNRAS / ApJL version (8-12 pages)

**Title:** "Does Stellar Spin-Down Follow a Unique Functional Form? Physics-Constrained Symbolic Regression for Gyrochronology and the Stalled-Braking Transition"

**Sections:**

1. **Introduction** (1.5 pages)
   - The gyrochronology problem and its importance for stellar age-dating
   - Classical analytic relations (Barnes, MH, Angus) and their limitations
   - Neural approaches (ChronoFlow, Van-Lane et al. ApJ 2025) — flexible but uninterpretable
   - The stalled-spin-down problem (van Saders et al. 2016, Metcalfe et al. 2016) — do existing functional forms break at old ages?
   - This paper's contribution: physics-constrained symbolic regression to test whether a unique functional form exists and whether it captures the braking transition. Cite Huppenkothen et al. (2023) best practices.

2. **Data** (0.5 page)
   - ChronoFlow training set: 30 clusters, 7,615 rotators (after slow-sequence filtering: fewer), age range, color range, period range
   - Out-of-distribution test sample: 10-20 old stars with confirmed **surface** rotation periods, primarily from van Saders et al. (cross-matched with McQuillan 2014)
   - Table of cluster properties (name, age, number of members)

3. **Methods** (2 pages)
   - 3.1 PySR symbolic regression: how it works, operator set, hyperparameters
   - 3.2 Physics-constrained loss function (custom Julia implementation) — the methodological contribution
   - 3.3 Bootstrap stability analysis (100 resamples, functional form counting)
   - 3.4 Classical baselines: Barnes, MH, Angus (refitted to Gaia colors) AND scipy.optimize.curve_fit with Barnes form (the simplest-possible baseline)
   - 3.5 ChronoFlow: brief description of the neural model
   - 3.6 Evaluation: leave-one-cluster-out cross-validation, RMSE, MAE

4. **Results** (3-4 pages)
   - 4.1 The Pareto frontier from direct PySR (figure + table of top equations)
   - 4.2 The best PySR equation in full (LaTeX, with uncertainties on constants)
   - 4.3 Physics-constrained vs unconstrained search comparison
   - 4.4 Bootstrap stability: which functional form(s) are uniquely supported by the data?
   - 4.5 Cross-validation comparison table (PySR vs Barnes vs MH vs Angus vs ChronoFlow vs curve_fit)
   - 4.6 Out-of-distribution extrapolation to old stars: does PySR predict the stalled spin-down? (**Frame as exploratory** — 10-20 heterogeneous test points cannot support strong statistical claims. Use language like "suggestive" and "consistent with," not "demonstrates" or "proves.")
   - 4.7 ChronoFlow residual analysis: physical interpretation of where ChronoFlow succeeds and fails

5. **Where the method fails** (0.5 page) — Color and age boundaries of validity. Bootstrap convergence failures. Honest comparison to curve_fit.

6. **Discussion** (1.5 pages)
   - Is the discovered equation physically interpretable? Connection to magnetic braking theory.
   - Does the functional form capture or miss the braking transition? Connection to van Saders et al. critical Rossby number hypothesis.
   - Mass dependence of the transition (different colors = different transition ages?)
   - Implications for stellar age-dating: should the community update Barnes 2007?
   - What we tried that did not work (recommended by Huppenkothen et al. 2023)

7. **Conclusions** (0.5 page)
   - Summary of the best equation and its stability across bootstraps
   - Whether the stalled-spin-down is or is not encoded in young-star data
   - The equation as a Python function on GitHub with Zenodo DOI

8. **References** (~30-35 references)

9. **Data and code availability** — PySR search results (all 100 bootstrap runs), physics-constrained loss function code, the best equation as a callable Python function, and the old-star test sample with references. All on GitHub + Zenodo.

10. **Appendix**
   - Full Pareto frontier table (all 100 bootstrap runs)
   - Functional form frequency histogram across bootstraps
   - Per-cluster cross-validation RMSE table
   - Python code snippet for the best equation

---

## Timeline

| Week | Activity |
| --- | --- |
| 1 | Environment setup, data download, understand ChronoFlow data, make scatter plots, assemble old-star OOD sample |
| 2 | Implement Barnes/MH/Angus/curve_fit baselines, refit to Gaia colors, verify against published values |
| 3 | Run PySR direct search (constrained + unconstrained, let it run 1-2 days), inspect Pareto frontier |
| 4 | Run ChronoFlow forward model, compute and visualize residuals, run PySR on residuals |
| 5-7 | Bootstrap stability analysis: Step A (10 same-data runs, ~3-5 days) then Step B (50-100 resamples, sequential, 2-4 weeks — see compute note in Phase 7) |
| 8 | Out-of-distribution extrapolation test, cross-validation (leave-one-cluster-out) |
| 9 | Verification, failure mode characterization, generate all figures and tables |
| 10-11 | Write the paper (RNAAS first, then full version) |
| 12 | Revise, advisor feedback |

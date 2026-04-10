# Research Plan: PySR Symbolic Regression for Gyrochronology

**Proposal 4 from top_5_proposals.md**

This document is a step-by-step guide from an empty folder to a completed paper. Every step, parameter choice, and evaluation criterion is explained without specialist vocabulary.

---

## What is being created that is new

The naive version of this project — "install PySR, call model.fit(), report the best equation" — is not publishable. Running a turnkey tool on someone else's dataset is not a contribution. Miles Cranmer, PySR's creator, explicitly warns on his blog that "many ML applications in astrophysics aren't necessary" and that "a simpler algorithm will often do just as well." This project must pass Cranmer's own test.

The novel contribution has three parts. Crucially, the **scientific question** comes first: **Does the functional form of gyrochronology change at old ages, and can data-driven equation discovery detect the transition to weakened magnetic braking?** The PySR machinery is the tool used to answer this question, not the contribution itself.

1. **Physics-constrained symbolic regression.** Instead of letting PySR search freely, define a custom loss function that penalizes physically impossible equations: negative periods, non-monotonic age dependence (period must increase with age for main-sequence stars past ~100 Myr), and singularities at valid input values. PySR supports custom Julia loss functions — you write one that encodes gyrochronology physics. This transforms the project from "I ran PySR" to "I developed a physics-constrained symbolic regression framework for stellar age-dating." No existing PySR application in astronomy uses physics constraints.

2. **Bootstrap stability of the functional form.** Run PySR 100 times on bootstrap resamples of the training data. For each run, record the top-5 Pareto-optimal equations. Count how often each functional FORM appears across the 100 runs (ignoring exact coefficient values). If the same form appears 80+ times, it is robust. If 10 different forms appear, the data does not uniquely constrain the functional form — and that itself is a publishable finding about the information content of the data. No PySR paper in any domain has reported this kind of functional-form stability analysis. This methodology is reusable by every future PySR study in any scientific domain.

3. **Out-of-distribution extrapolation to the stalled-spin-down regime.** The ChronoFlow training set only goes up to ~4 Gyr. Use 20-30 well-characterized old stars (the Sun, 16 Cyg A/B, and asteroseismically-dated Kepler stars from van Saders et al. 2016 and Metcalfe et al. 2023) to test whether the PySR equation extrapolates correctly into the regime where magnetic braking weakens. If PySR's equation — discovered from young-star data alone — naturally captures or fails to capture the stalled spin-down, that is a genuine astrophysical finding about what the young-star rotation data implies about old-star physics. This connects the project to an active debate in stellar physics, not just a methods demonstration.

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
   pip install pysr numpy pandas matplotlib scipy
   python -c "import pysr; pysr.install()"
   ```
   The `pysr.install()` step downloads Julia and the SymbolicRegression.jl backend. It takes 2-5 minutes on first run.

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

2. Open the data in a notebook and inspect it. The dataset contains approximately 8,000 stars from 30 open clusters and associations. Each star has:
   - **Rotation period** (days) — measured from photometric variability
   - **De-reddened Gaia DR3 color** (BP-RP)_0 — a proxy for mass/temperature (redder = cooler = lower mass)
   - **Cluster membership** — which cluster the star belongs to
   - **Cluster age** (log years or Myr) — the age assigned to the entire cluster from isochrone fitting

3. Understand the variables you will use:
   - **Input features for PySR:** age (in Myr or log10(age/yr)), (BP-RP)_0 color
   - **Target for PySR:** rotation period (days)
   - You are searching for an equation: `P_rot = f(age, color)`

4. **Quality check:** Make a scatter plot of rotation period vs (BP-RP)_0, colored by cluster age. You should see that younger clusters (blue dots) have shorter periods and older clusters (red dots) have longer periods, with a clear color-dependence. If this pattern is not visible, something is wrong with the data loading.

5. **Unit decisions:** Convert age to Myr (not log). PySR works better with numbers in a reasonable range (1-4000) than with log-transformed values, because it can discover logarithmic relationships on its own.

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

Uses a more flexible functional form with additional terms. Check their `stardate` package for the exact parameterization.

### The dead-simple baseline (MANDATORY — required by Huppenkothen et al. 2023)

The Huppenkothen best-practices checklist requires comparison to "the simplest model possible without sacrificing performance." Before running PySR, fit the Barnes functional form `P = a × (color - c0)^b × age^c` using `scipy.optimize.curve_fit`. This is a 4-parameter nonlinear least-squares fit that takes less than 1 second.

If PySR's best equation has the same functional form as Barnes (a power law in age and color), the PySR contribution is NOT the equation — it is the bootstrap stability analysis proving that this form is uniquely optimal. If PySR finds a structurally different equation (e.g., with a logarithmic age term or a color-age interaction), compare its RMSE and cross-validation performance to the refitted Barnes. The improvement must exceed the uncertainty from cross-validation to be meaningful. A 0.1% RMSE gain does not justify added complexity (per the Nature Reviews Physics 2025 guidance on ML in astronomy).

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

    # Penalty: non-monotonic with age (column 1)
    # Check that increasing age → increasing period for fixed color
    ages = dataset.X[1, :]
    sorted_idx = sortperm(ages)
    sorted_pred = prediction[sorted_idx]
    mono_violations = sum(max.(0, sorted_pred[1:end-1] .- sorted_pred[2:end]))

    return mse + 10.0 * neg_penalty + 1.0 * mono_violations
end
"""

model = pysr.PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/", "pow"],
    unary_operators=["log", "exp", "sqrt", "abs"],
    populations=30,
    population_size=50,
    maxsize=25,
    batching=True,
    batch_size=500,
    loss_function=physics_loss,
    complexity_of_operators={
        "pow": 2, "log": 1, "exp": 1, "sqrt": 1
    },
    select_k_features=2,
    progress=True,
    temp_equation_file=True,
    tempdir="project/results/pysr_constrained",
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

2. For each star, compute the ChronoFlow-predicted rotation period at the known cluster age and color:
   ```python
   # Using ChronoFlow's forward model
   P_chronoflow = chronoflow_model.predict_period(age=age, color=color)
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

---

## Phase 7: Bootstrap stability analysis (MANDATORY)

**Time: 2-3 days of compute (parallelizable)**

This experiment answers: "Is the functional form PySR finds robust, or does it change every time you resample the data?"

1. Generate 100 bootstrap samples of the training data (sample N stars with replacement from the N-star dataset).
2. Run PySR on each bootstrap sample with the physics-constrained loss, using 50 iterations (shorter than the main run, to keep compute feasible — 100 runs × 50 iterations ≈ 50 hours on 8 cores).
3. For each run, record the top-3 equations on the Pareto frontier.
4. Across all 100 runs, identify the unique functional FORMS that appear (ignore exact coefficient values — two equations are "the same form" if they have the same operator tree structure).
5. Report:
   - The most common functional form and how often it appeared
   - The second and third most common forms
   - Whether the constrained and unconstrained searches converge to the same form

This analysis is the methodological contribution that no other PySR paper has done. If the same form appears 80+ times out of 100, you can claim the data strongly supports that functional form. If many forms appear, that is equally interesting — it means the data does not uniquely determine how rotation, age, and color are related.

## Phase 8: Out-of-distribution extrapolation to the stalled-spin-down regime (MANDATORY)

**Time: 2-3 days**

This experiment is what connects the project to an active astrophysical debate and elevates it beyond a methods demonstration. The ChronoFlow training data spans ages 1.5 Myr to ~4 Gyr. Stars older than ~4 Gyr show "weakened magnetic braking" — their rotation periods stop increasing as fast as the classical power-law relations predict (van Saders et al. 2016, Nature 529, 181; Metcalfe et al. 2016, ApJL 826, L2). This is one of the most active debates in stellar physics.

### Assemble a systematic old-star test sample (target: 20-30 stars)

Do NOT use just 3-5 test points. Assemble a proper sample:

- **The Sun:** age = 4567 Myr, P_rot = 25.4 days, (BP-RP)_0 ≈ 0.82. The most precisely known calibration point.
- **16 Cyg A and B:** asteroseismic age ~6.8 Gyr, P_rot ~23 and ~24 days. Well-studied binary with precise parameters.
- **van Saders et al. (2016) sample:** ~20 Kepler asteroseismic targets with ages 4-10 Gyr and measured rotation periods from McQuillan et al. (2014) or García et al. (2014). These are the stars that first revealed the stalled-spin-down phenomenon. Download from the van Saders et al. supplementary tables.
- **Metcalfe et al. (2023) sample:** Updated asteroseismic ages and rotation periods for evolved solar analogs. Cross-reference with the van Saders sample to avoid duplicates.
- **Hall et al. (2021) chromospheric activity sample:** Stars with activity-derived rotation periods and asteroseismic ages, providing an independent rotation-period measurement method.

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
- The critical Rossby number hypothesis (Ro_crit ≈ 2, beyond which magnetic braking weakens)
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
   - Does period increase with color (redder = lower mass = faster spin-down)? (It should.)
   - Are there unphysical features like negative periods or division by zero at certain age/color values?

2. **Comparison to known functional forms:** Is the PySR-discovered equation recognizable? If PySR finds something like `P = a * age^0.52 * (color - 0.4)^0.6`, that is essentially Barnes 2007 rediscovered — and the contribution is the bootstrap proof that this form is uniquely stable, not the equation itself. If PySR finds something structurally different — say, with a logarithmic age dependence or an interaction term — that is scientifically interesting and the comparison to Barnes must be rigorous (cross-validated RMSE, not just training RMSE).

3. **Sensitivity to PySR hyperparameters:**
   - Run PySR three times with different random seeds. Do you get the same top equation? If yes, the result is robust. If the best equation changes each time, report the ensemble of equations on the Pareto frontier.
   - Increase `maxsize` from 25 to 35. Does a significantly better equation appear at higher complexity?

4. **Numerical stability:** Evaluate the equation on the full age range (1 Myr to 14 Gyr) and color range (0.3 to 3.0). Check for overflow, underflow, or NaN values.

### Where the method fails (MANDATORY — required by Huppenkothen et al. 2023 checklist)

5. **Color boundaries:** The discovered equation will likely diverge or become unphysical near color < 0.4 (hot stars above the Kraft break, where magnetic braking is not the dominant spin-down mechanism) and color > 2.5 (very low-mass M dwarfs that remain fully convective). Report these boundaries explicitly. Plot the equation's predictions across the full color range and mark where it becomes unphysical.

6. **Age boundaries:** The training data spans 1.5 Myr to ~4 Gyr. Report the equation's behavior at 0.1 Myr (pre-main-sequence) and 10+ Gyr (old disk/halo). If the equation predicts absurd values outside the training range, mark the extrapolation boundary clearly.

7. **PySR convergence failures:** If certain bootstrap resamples produce no valid equation (PySR fails to converge or all equations violate the physics constraints), report the failure rate. If 15/100 bootstraps fail, that is informative — it suggests the data is marginally constraining.

8. **The honest comparison to curve_fit:** If `scipy.optimize.curve_fit` with the Barnes functional form achieves the same cross-validated RMSE as PySR's best equation, say so plainly. The contribution in this case is the bootstrap stability analysis and the out-of-distribution test, not the equation itself. Do not oversell.

---

## Phase 11: Paper writing plan

### RNAAS version (1500 words, 1 figure)

**Title:** "Does Stellar Spin-Down Follow a Unique Functional Form? A Symbolic Regression Test"

Note: the title frames the astrophysical question, not the tool. "PySR" does not appear in the title.

1. **Abstract** (150 words)
   - Problem: gyrochronology relations are either hand-crafted (Barnes) or black-box (ChronoFlow)
   - Method: PySR symbolic regression on ~8,000 cluster rotators from ChronoFlow's training set
   - Result: the best PySR equation, its RMSE vs Barnes/MH/Angus, and whether it recovers a known functional form or discovers a new one
   - Implication: either the classical functional form is (approximately) optimal, or there is a better simple equation that the community should adopt

2. **Body** (~1200 words)
   - Paragraph 1: What gyrochronology is and why closed-form relations matter
   - Paragraph 2: What PySR is and how symbolic regression works (one paragraph, no math)
   - Paragraph 3: Data — ChronoFlow training set, 30 clusters, ~8,000 rotators
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
   - ChronoFlow training set: 30 clusters, ~8,000 rotators, age range, color range, period range
   - Out-of-distribution test sample: 20-30 old stars from van Saders et al. and Metcalfe et al.
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
   - 4.6 Out-of-distribution extrapolation to old stars: does PySR predict the stalled spin-down?
   - 4.7 ChronoFlow residual analysis: physical interpretation of where ChronoFlow succeeds and fails

5. **Where the method fails** (0.5 page) — Color and age boundaries of validity. Bootstrap convergence failures. Honest comparison to curve_fit.

6. **Discussion** (1.5 pages)
   - Is the discovered equation physically interpretable? Connection to magnetic braking theory.
   - Does the functional form capture or miss the braking transition? Connection to van Saders et al. critical Rossby number hypothesis.
   - Mass dependence of the transition (different colors = different transition ages?)
   - Implications for stellar age-dating: should the community update Barnes 2007?
   - What we tried that did not work (required by Huppenkothen checklist)

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
| 5 | Bootstrap stability analysis (100 resamples — start Monday, runs ~50 hours across cores) |
| 6 | Out-of-distribution extrapolation test, cross-validation (leave-one-cluster-out) |
| 7 | Verification, failure mode characterization, generate all figures and tables |
| 8 | Write the paper (RNAAS first, then full version) |

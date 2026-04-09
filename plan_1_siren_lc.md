# Research Plan: SIREN for Variable-Star Light Curves

**Proposal 1 from top_5_proposals.md**

This document is a step-by-step guide from an empty folder to a completed paper. It assumes no prior astronomy or ML infrastructure. Every command, architecture decision, and evaluation metric is explained without assuming specialist knowledge.

---

## What is being created that is new

No one has used an implicit neural representation — specifically SIREN, a neural network with sine-wave activation functions — to reconstruct astronomical light curves. Every existing deep-learning method for irregular time-series in astronomy (ASTROMER, AstroCo, ATAT, ATCAT, Astroconformer) treats observations as a sequence and processes them with attention or convolution. SIREN is fundamentally different: it treats time as a continuous coordinate and learns a function from that coordinate to brightness. The network IS the light curve — you can query it at any time, including times with no observation.

The novel artifact is: a trained SIREN network per star that acts as a continuous, differentiable reconstruction of the light curve from sparse irregular data, plus a systematic benchmark showing how SIREN compares to Gaussian process regression and Fourier decomposition across a range of data sparsity levels.

---

## Phase 1: Environment setup

**Time: 1 hour**

1. Create a new Python environment. Use conda or venv:
   ```
   python -m venv siren_lc_env
   source siren_lc_env/bin/activate
   ```

2. Install core packages:
   ```
   pip install torch numpy scipy matplotlib pandas astropy
   pip install celerite2 george
   ```

3. Clone the reference SIREN implementation:
   ```
   git clone https://github.com/vsitzmann/siren.git
   ```
   You will not use this repo directly as a library. You will read the file `siren/modules.py` to understand the architecture, then write your own simplified version (about 40 lines of PyTorch).

4. Create your project directory structure:
   ```
   mkdir -p project/{data,models,results,figures,notebooks}
   ```

---

## Phase 2: Download and prepare the OGLE data

**Time: 2-3 hours**

1. Download LMC Cepheid light curves:
   ```
   cd project/data
   wget https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/phot.tar.gz
   tar xzf phot.tar.gz
   ```
   This creates `phot/I/` and `phot/V/` directories. Each file is one star, named like `OGLE-LMC-CEP-0001.dat`. Each file has three columns: time (HJD - 2450000), magnitude, magnitude error.

2. Download the Cepheid parameter catalog:
   ```
   wget https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/cepF.dat
   ```
   This file contains period, mean magnitude, amplitude, and Fourier coefficients for each fundamental-mode Cepheid.

3. Repeat for RR Lyrae:
   ```
   wget https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/rrlyr/phot.tar.gz
   tar xzf phot.tar.gz
   wget https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/rrlyr/rrab.dat
   ```

4. Parse the data into a standard format. Write a Python script that:
   - Reads each `.dat` file (whitespace-separated: time, mag, mag_err)
   - Reads the catalog file to get the known period for each star
   - Stores everything in a single pandas DataFrame or a list of dictionaries: `{star_id, times, mags, errors, period, amplitude}`
   - Filters to well-sampled stars: keep only stars with at least 100 observations in I-band

5. **Quality check:** Plot 5 random light curves (time vs magnitude) to verify the data looks right. Magnitudes should decrease upward (brighter = smaller number). Periods should be in the range 0.2–100 days.

**Expected yield:** ~4,700 fundamental-mode Cepheids and ~39,000 RRab stars in the LMC, each with hundreds of I-band and V-band observations.

---

## Phase 3: Build the SIREN architecture

**Time: 3-4 hours**

The SIREN network is a standard fully-connected neural network (also called a multi-layer perceptron) with one change: every activation function is `sin(x)` instead of `ReLU(x)`.

### Architecture details

```
Input:  1 number (time, normalized to [-1, 1])
Layer 1: Linear(1 → 256) → sin(ω₀ × output)
Layer 2: Linear(256 → 256) → sin(output)
Layer 3: Linear(256 → 256) → sin(output)
Layer 4: Linear(256 → 1) → no activation (raw output = predicted magnitude)
```

Key design decisions:
- **ω₀ (omega_0):** The first layer multiplies its output by a constant ω₀ before applying the sine. Sitzmann et al. 2020 use ω₀ = 30 for image fitting. For light curves, start with ω₀ = 30 and tune later. This parameter controls how quickly the network can oscillate — higher values let it represent higher-frequency features.
- **Weight initialization:** The first layer weights are drawn from Uniform(-1/input_dim, 1/input_dim). All other layers use Uniform(-√(6/n) / ω₀, √(6/n) / ω₀) where n is the number of input features to that layer. This initialization is critical — it ensures the sine activations start in their useful range. Copy this exactly from Sitzmann et al.
- **No batch normalization, no dropout.** SIREN is a pure coordinate network.

### Implementation

Write a PyTorch `nn.Module` class called `SIREN` with:
- `__init__(self, hidden_dim=256, num_layers=3, omega_0=30.0)`
- `forward(self, t)` where `t` is a tensor of shape `(N, 1)` — N query times, each a single number

The forward pass is: input → first sine layer (with ω₀ scaling) → middle sine layers → final linear layer → output magnitude.

### Time normalization

Before feeding times into the network, normalize them to the range [-1, 1]:
```
t_normalized = 2 * (t - t_min) / (t_max - t_min) - 1
```
Similarly, normalize magnitudes to roughly [-1, 1]:
```
mag_normalized = (mag - mag_mean) / mag_std
```
De-normalize the output when reporting results.

---

## Phase 4: Build the two baselines

**Time: 3-4 hours**

### Baseline 1: Gaussian process regression with celerite2

Use a quasi-periodic kernel. In `celerite2`, this is built from a `SHOTerm` (simple harmonic oscillator) that naturally captures periodic signals.

```python
import celerite2
from celerite2 import terms

# Quasi-periodic kernel: a damped SHO at the known period
kernel = terms.SHOTerm(sigma=amp_guess, rho=period, tau=10*period)
kernel += terms.SHOTerm(sigma=0.1, rho=0.5)  # short-term variability
gp = celerite2.GaussianProcess(kernel, mean=mag_mean)
gp.compute(times, yerr=errors)
```

Optimize the kernel hyperparameters by maximizing the log-likelihood using `scipy.optimize.minimize`. Then predict at a dense grid of times using `gp.predict()`.

### Baseline 2: 7th-order Fourier decomposition

Phase-fold the light curve using the known period:
```
phase = (time % period) / period
```

Fit a truncated Fourier series:
```
mag(phase) = a_0 + Σ_{k=1}^{7} [a_k * cos(2π k phase) + b_k * sin(2π k phase)]
```

This is a linear regression problem. Use `numpy.linalg.lstsq` or `scipy.optimize.curve_fit`. The output is the 15 Fourier coefficients (a_0 plus 7 pairs of a_k, b_k).

**Important:** The Fourier baseline uses the known period, which is an advantage SIREN does not have. This is intentional — it shows how SIREN compares to a method that has extra information.

---

## Phase 5: The core experiment — sparse reconstruction

**Time: 1-2 weeks**

This is the central experiment of the paper. For each star:

1. **Start with the full light curve** (all observations).
2. **Randomly remove a fraction of the observations.** Test fractions: 0% (full data), 30%, 50%, 70%, 90%.
3. **Fit SIREN, GP, and Fourier to the remaining observations.**
4. **Evaluate reconstruction quality on the removed observations** — the held-out points the methods never saw.

### Training SIREN on one star

For each star, training is:
```python
model = SIREN(hidden_dim=256, num_layers=3, omega_0=30.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2000):
    pred = model(t_train_normalized)
    loss = ((pred - mag_train_normalized) ** 2 / err_train_normalized ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Training 2000 steps for one star takes about 2-5 seconds on a CPU. For 200 stars × 5 sparsity levels = 1000 fits, total training time is about 1-2 hours on a laptop.

### Evaluation metrics

For each (star, sparsity level, method) combination, compute:
- **RMSE** on held-out points: `sqrt(mean((predicted - true)^2))`
- **MAE** on held-out points: `mean(|predicted - true|)`
- **Period recovery:** For SIREN and GP (which do not use the period), compute the power spectrum of the reconstruction and check whether the dominant frequency matches the known period.
- **Amplitude recovery:** Compare the range (max - min) of the reconstruction to the known amplitude from the OGLE catalog.

### How to structure the experiment

Select a sample of 200 stars: 100 Cepheids spanning a range of periods (short-period 1-3 days, medium 5-15 days, long 20-80 days) and 100 RR Lyrae (periods 0.2-0.9 days). For each star and each sparsity level, run all three methods. Store results in a table: `(star_id, type, period, sparsity, method, rmse, mae, period_recovered, amplitude_recovered)`.

### Repeat with random seeds

For each sparsity level, the random removal of points should be repeated 5 times with different random seeds. Report the mean and standard deviation of the metrics across seeds.

---

## Phase 6: Additional experiments

**Time: 1 week**

### Experiment A: ω₀ sensitivity

Fix one well-sampled Cepheid. Train SIREN with ω₀ = 5, 10, 15, 20, 25, 30, 40, 50, 60. Plot reconstruction RMSE versus ω₀. This tells you how sensitive the result is to this hyperparameter.

### Experiment B: Network depth and width

Fix ω₀ = 30. Train SIREN with:
- 2 layers × 128 hidden units
- 3 layers × 128 hidden units
- 3 layers × 256 hidden units (default)
- 4 layers × 256 hidden units
- 5 layers × 256 hidden units

Plot RMSE versus number of parameters. This is the ablation study reviewers will expect.

### Experiment C: Derivative quality

SIREN produces a differentiable function. Compute the time-derivative of the SIREN output using PyTorch autograd:
```python
t.requires_grad_(True)
mag_pred = model(t)
dmag_dt = torch.autograd.grad(mag_pred.sum(), t, create_graph=True)[0]
```

Compare the rate of brightness change (dmag/dt) at maximum and minimum brightness to the values astronomers measure from Fourier fits. This is a bonus result — if SIREN derivatives are physically sensible, it adds scientific value to the paper.

### Experiment D (optional, for the long paper): Hypernetwork

Instead of training a separate SIREN per star, train a second network (the "hypernetwork") that takes a few observations as input and outputs the weights of the SIREN. This lets you reconstruct a new light curve in a single forward pass without per-star optimization. This is the extension that upgrades the project from an RNAAS note to a full AJ/MNRAS paper.

The hypernetwork architecture:
```
Input:  A set of (time, magnitude) pairs, encoded as a fixed-length vector
        using a small set-encoder (e.g., DeepSets: mean-pool over per-point MLPs)
Output: All weights and biases of the SIREN network, flattened into a single vector
```

This is more complex and is optional for the first paper.

---

## Phase 7: Verification and sanity checks

**Time: 2-3 days**

Before writing the paper, verify:

1. **Overfitting check:** For a well-sampled star with no data removed, SIREN should fit the training data nearly perfectly (RMSE < 0.01 mag). If it does not, there is a bug.

2. **Extrapolation check:** Train SIREN on the first 80% of the time range. Evaluate on the last 20%. Periodic stars should extrapolate well if the network learned the periodicity. Non-periodic behavior in the extrapolation zone is expected and interesting to report.

3. **GP sanity check:** The GP should produce good reconstructions at low sparsity (0-30% removed) since GPs are well-suited to this problem. If the GP performs poorly everywhere, the kernel choice or hyperparameter optimization is wrong.

4. **Known-answer test:** For one star, manually compute the Fourier coefficients from the OGLE catalog file. Verify that your Fourier baseline code produces the same coefficients to within rounding.

5. **Baseline fairness:** SIREN and GP do NOT use the known period. Fourier DOES use the known period. Report this clearly in the paper. It is not unfair — it is informative. If SIREN beats Fourier despite not knowing the period, that is a strong result.

---

## Phase 8: Paper writing plan

Target: RNAAS (short format, 1500 words + 1 figure) for the initial result. If the results are strong, extend to AJ/MNRAS (full paper, 10-15 pages) with the hypernetwork extension and full ablation.

### RNAAS version (1500 words, 1 figure, 1 table)

**Title:** "SIREN: Implicit Neural Representations for Variable-Star Light-Curve Reconstruction"

**Structure:**

1. **Abstract** (150 words)
   - State the problem: reconstructing light curves from sparse irregular data
   - State the method: SIREN, a coordinate-based neural network with sine activations
   - State the main result: SIREN matches/beats GP regression on held-out points across a range of sparsity levels for OGLE Cepheids and RR Lyrae
   - State the implication: a new family of methods for light-curve analysis that produces continuous, differentiable reconstructions

2. **Body** (~1200 words)
   - Paragraph 1: The problem and why it matters (distance ladder, variable-star classification)
   - Paragraph 2: What SIREN is and why it is suited (periodic activations, continuous output, no hand-tuned kernel)
   - Paragraph 3: Data (OGLE LMC Cepheids and RR Lyrae, number of stars, number of observations)
   - Paragraph 4: Experiment setup (sparsity levels, hold-out evaluation, baselines)
   - Paragraph 5: Results (RMSE comparison, reference to figure and table)
   - Paragraph 6: Discussion — when SIREN wins (high sparsity), when GP wins (low sparsity, very short light curves), and what this means for future surveys like LSST

3. **Figure** (the one allowed figure)
   - A 2×3 panel grid:
     - Top row: reconstruction examples for one Cepheid at 30%, 50%, 90% sparsity (SIREN in color, GP in gray, data points as black dots, held-out points as red circles)
     - Bottom row: RMSE vs sparsity fraction, averaged over all Cepheids (left), all RR Lyrae (middle), and a combined plot (right), with error bars from the 5 random seeds

4. **Table** (optional, can replace figure panels)
   - Rows: method (SIREN, GP, Fourier)
   - Columns: RMSE at each sparsity level, period recovery rate, amplitude recovery rate

5. **References** (~10-15 references: Sitzmann 2020, Naul 2018, OGLE catalog papers, celerite2, GP regression references)

### Full AJ/MNRAS version (10-15 pages)

**Sections:**

1. **Introduction** (1.5 pages)
   - The sparse light-curve reconstruction problem
   - Current methods: GP regression, Fourier decomposition, transformer-based approaches
   - Implicit neural representations: what they are, why sine activations help periodic signals
   - This paper's contribution

2. **Data** (1 page)
   - OGLE OCVS LMC Cepheids and RR Lyrae
   - Sample selection criteria
   - Table of sample statistics

3. **Methods** (2-3 pages)
   - 3.1 SIREN architecture (with architecture diagram)
   - 3.2 Training procedure (loss function, optimizer, normalization)
   - 3.3 Hypernetwork extension (set-encoder → SIREN weights)
   - 3.4 GP baseline setup
   - 3.5 Fourier baseline setup
   - 3.6 Evaluation protocol (sparsity levels, hold-out, metrics)

4. **Results** (3-4 pages)
   - 4.1 Reconstruction quality vs sparsity (main result table and figure)
   - 4.2 Period and amplitude recovery
   - 4.3 Derivative quality
   - 4.4 Ablation: ω₀ sensitivity
   - 4.5 Ablation: depth and width
   - 4.6 Hypernetwork performance (if implemented)
   - 4.7 Computational cost comparison

5. **Discussion** (1-2 pages)
   - When SIREN wins and when it loses
   - Implications for LSST and other sparse surveys
   - Limitations: current approach fits one star at a time (unless hypernetwork is used)
   - Connection to other INR applications in science

6. **Conclusions** (0.5 page)
   - Summary of main finding
   - Recommended use cases
   - Future directions

7. **References** (~30-40 references)

8. **Appendix** (optional)
   - Full table of per-star results
   - Additional reconstruction examples

---

## Timeline

| Week | Activity |
| --- | --- |
| 1 | Environment setup, data download, parse OGLE files, verify with plots |
| 2 | Implement SIREN architecture, train on 5 test stars, debug |
| 3 | Implement GP and Fourier baselines, verify on same 5 stars |
| 4 | Run the full sparse-reconstruction experiment (200 stars × 5 sparsity levels × 3 methods) |
| 5 | Run ablation experiments (ω₀, depth/width, derivative quality) |
| 6 | Generate all figures and tables, verify numbers |
| 7 | Write the paper (RNAAS or full version) |
| 8 | Revise, get advisor feedback, submit |

# Research Plan: Simulation-Based Inference for FRB Dispersion Measures

**Proposal 3 from top_5_proposals.md**

---

## What is being created that is new

The naive version — "call sbi.SNPE() on simulated FRBs" — is not publishable. Calling a library function is not a contribution. The community paper by Huppenkothen et al. (2023, arXiv:2310.12528) identifies "machine learning for machine learning's sake" as the primary failure mode of astro-ML papers, and multiple SBI-specific critiques (Hermans et al. 2021 on overconfident posteriors; Hahn et al. A&A 2025 on variance inflation; domain-adaptation failures in Villaescusa-Navarro et al.) set a high bar for validation rigor.

**Critical prior work:** Rajabi et al. (2025, arXiv:2512.24003) already built and publicly released (GPL-2.0) a forward simulator that includes dispersion, scattering (with ν⁻⁴ frequency dependence), scintillation, spectral index, and instrumental noise, and generated 180,000 simulations with it for training a CNN-LSTM point estimator. **This project does NOT claim to be the first FRB simulator.** The Rajabi et al. simulator exists and is public.

The novel contribution has three parts, ordered by importance:

1. **Calibrated joint posteriors, not point estimates.** This is the primary contribution. Rajabi et al. produce a single DM number with no uncertainty. This project produces the full joint posterior over three correlated parameters: DM, scattering timescale (τ_s), and spectral index (α). The correlations are physically meaningful — DM and τ_s both increase with distance, and their ratio constrains the turbulence spectrum. No existing FRB DM estimator (classical or ML) provides calibrated multi-parameter posteriors. This is the gap. Crucially, the posteriors must be validated against known SBI failure modes: we include TARP diagnostics (Lemos et al. 2023), model misspecification tests, and 500 injection tests — matching or exceeding the validation standard set by Gebhard et al. (A&A 2025).

2. **An improved and validated forward model.** The Rajabi et al. simulator is a starting point, but it was validated only indirectly (via the CNN-LSTM's DM accuracy). This project extends and validates the simulator more rigorously: (a) quantitative statistical comparison of simulated vs real CHIME dynamic spectra (pixel intensity histograms, pulse profiles, bandpass shapes — not just visual inspection), (b) misspecification tests showing how posteriors degrade when specific simulator components are disabled. The simulator improvement over Rajabi et al. must be explicitly demonstrated, not assumed. If the Rajabi simulator is already sufficient, use it as-is and contribute the validation and SBI inference instead.

3. **Explicit characterization of where the method fails.** For which bursts (low S/N, high scattering, narrow bandwidth) do the posteriors become uninformative or miscalibrated? Reporting failure modes builds trust with reviewers who know the SBI literature.

---

## Phase 1: Environment setup

**Time: 1-2 hours**

```
python -m venv fmpe_frb_env
source fmpe_frb_env/bin/activate
pip install torch numpy scipy matplotlib pandas astropy h5py
pip install sbi           # simulation-based inference toolkit
pip install scintools      # scintillation modeling — NOTE: scintools is designed for pulsar
                           # scintillation on timescales of seconds to minutes. For FRB timescales
                           # (microseconds to milliseconds), the scintillation PATTERN generation
                           # (correlated Gaussian fields in time-frequency) is reusable, but the
                           # dynamic spectrum simulation pipeline may need custom code. Investigate
                           # in week 2 whether scintools.Simulation can generate patterns at FRB
                           # timescales or whether a custom implementation is needed.
```

Project structure:
```
mkdir -p project/{data,simulator,posteriors,results,figures}
```

---

## Phase 2: Build the forward simulator

**Time: 2-3 weeks** (this is the largest phase — the simulator IS the contribution)

The simulator takes a set of physical parameters and produces a synthetic dynamic spectrum — a 2D array with axes (time, frequency) representing the burst as a radio telescope would see it.

### Input parameters (what you sample from)

| Parameter | Symbol | Prior range | Physical meaning |
| --- | --- | --- | --- |
| Dispersion measure | DM | 50–3000 pc/cm³ | Total electron column density along the line of sight |
| Scattering timescale | τ_s | 0.01–100 ms | Temporal broadening from multipath propagation |
| Spectral index | α | -5 to 0 | How burst brightness changes with frequency (power law) |
| Intrinsic width | w_i | 0.1–30 ms | Duration of the burst at the source |
| Scintillation bandwidth | Δν_scint | 0.01–10 MHz | Frequency scale of scintillation modulation |
| Fluence | F | — | Overall brightness (normalized out during processing) |

### Simulation steps (for one burst)

1. **Generate the intrinsic pulse:** a Gaussian in time with width w_i, modulated by a power-law spectrum with index α across the frequency band.

2. **Apply dispersion:** shift each frequency channel in time by the cold-plasma dispersion relation:
   ```
   Δt(ν) = 4.149 × DM × (ν_lo^(-2) - ν^(-2)) ms
   ```
   where ν is in GHz. The constant 4.149 matches Lorimer & Kramer (2004): K_DM = 4.148808 ± 0.000003 GHz² pc⁻¹ cm³ ms. This is the cold-plasma limit, valid when ν >> ν_plasma — an excellent approximation at radio frequencies.

3. **Apply scattering:** convolve the time profile at each frequency with a one-sided exponential with timescale τ_s × (ν / ν_ref)^(-4). The frequency scaling exponent -4 is the thin-screen approximation of Kolmogorov scattering (the full uniform-medium Kolmogorov prediction is -4.4; see Lorimer & Kramer 2004). Measured exponents for individual FRBs scatter broadly; for extragalactic FRBs the dominant scattering screen is often in the host galaxy, not the IGM (Macquart & Koay 2013). The exponent can be varied for the misspecification test (Phase 4).

4. **Apply scintillation:** multiply the dynamic spectrum by a scintillation pattern generated using `scintools`. The pattern has characteristic bandwidth Δν_scint and characteristic timescale Δt_scint (derived from Δν_scint via the scintillation arc relation). Use `scintools.Simulation` to generate correlated Gaussian fields in the (time, frequency) plane.

5. **Apply instrumental response:** downsample to CHIME's channelization (1024 frequency channels across 400–800 MHz, time resolution ~1 ms), add Gaussian noise at a specified signal-to-noise ratio (sample S/N from 8 to 200 to cover the CHIME detection range).

6. **Output:** a 2D array of shape (N_time, 1024) representing the simulated dynamic spectrum, plus the ground-truth parameter vector (DM, τ_s, α, w_i, Δν_scint).

### Validation of the simulator

Before using it for training, verify:
- At τ_s = 0 and Δν_scint → ∞ (no scattering, no scintillation), the output should be a simple dispersed pulse. Check that de-dispersing at the true DM recovers a clean Gaussian.
- At high τ_s, the pulse should visibly broaden at low frequencies.
- At small Δν_scint, the dynamic spectrum should show "patchy" bright and dark frequency channels (the scintillation pattern).
- Generate 100 spectra and visually compare to real CHIME Catalog 2 total-intensity dynamic spectra (waterfall plots). The simulated spectra should look qualitatively similar.

### Simulator-vs-real statistical comparison (MANDATORY — do this BEFORE training the flow)

Before using the simulator for training, quantitatively compare simulated and real CHIME dynamic spectra. This directly addresses the sim-to-real gap criticism that is the dominant concern in the SBI literature (see domain-adaptation failures in Villaescusa-Navarro et al., and the Hahn et al. A&A 2025 finding that model misspecification inflates posterior variance).

Generate 1,000 simulated spectra at the DM and S/N distribution matching the CHIME Catalog 2. For both simulated and real spectra, compute:
- Pixel intensity histograms (should overlap)
- DM-dependent arrival-time curves (the dispersion sweep shape should match)
- Frequency-averaged pulse profiles (temporal shape comparison)
- Time-averaged spectra (bandpass shape comparison)
- Scintillation bandwidth distribution (for bursts where scintillation is visible)

Present these as overlay plots. If any distribution diverges visibly, diagnose the cause (missing instrumental effect, incorrect noise model, etc.) and fix the simulator before proceeding. Do not train the flow on a simulator that produces visibly different spectra from real data — a reviewer will catch this.

### How many simulations

Generate 500,000 training simulations and 50,000 validation simulations as a starting point. At ~0.5 seconds per simulation (the dispersion step is the bottleneck), this takes about 3 days on a laptop. The simulations can be run in parallel across CPU cores.

**Note on simulation budget:** 500K simulations for a 5-parameter problem with 2D observations is on the low end of what published SBI papers use (Gebhard et al. A&A 2025 used 33.5M for 16 parameters). Run SBC/TARP diagnostics (Phase 4) after training to check whether the posterior has converged. If calibration diagnostics reveal underfitting or poor coverage, increase to 1–2M simulations. The compute cost scales linearly.

Store simulations in HDF5 format: one file per batch of 10,000 spectra.

---

## Phase 3: Build the neural posterior estimator

**Time: 1 week**

### What simulation-based inference does

You have a simulator that maps parameters θ = (DM, τ_s, α) → dynamic spectrum x. You want the inverse: given an observed spectrum x, what is the posterior distribution p(θ | x)?

The `sbi` package trains a neural network to approximate this posterior directly. You feed it (θ, x) pairs from the simulator, and it learns a conditional density estimator q(θ | x) that approximates p(θ | x).

### Architecture choice: Neural Posterior Estimation

**METHOD NAME CLARIFICATION:** The `sbi` package supports two relevant inference methods:
- **SNPE** (Sequential Neural Posterior Estimation) with a MAF (Masked Autoregressive Flow) — a discrete bijective flow trained with maximum likelihood. Simpler, faster to train, well-tested.
- **FMPE** (Flow Matching Posterior Estimation) — a continuous normalizing flow trained with the flow-matching objective from Lipman et al. 2023. Newer, potentially more stable for complex posteriors, but requires ODE solving at inference time.

**Start with SNPE+MAF** (simpler, better-documented in `sbi`). If calibration diagnostics reveal poor coverage, try FMPE as an alternative. Report which method was used. Do NOT call the paper "Flow-Matching" if you used SNPE — this is a different method and reviewers will catch the mismatch immediately.

```python
from sbi.inference import SNPE  # or: from sbi.inference import FMPE
from sbi.utils import posterior_nn

# Define the embedding network (compresses the 2D spectrum to a vector)
embedding_net = torch.nn.Sequential(
    # Treat the spectrum as a 1-channel image
    torch.nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d((8, 8)),
    torch.nn.Flatten(),
    torch.nn.Linear(32 * 8 * 8, 128),
    torch.nn.ReLU(),
)

# Option A: SNPE with MAF (recommended starting point)
density_estimator = posterior_nn(
    model="maf",          # Masked Autoregressive Flow
    embedding_net=embedding_net,
    hidden_features=128,
    num_transforms=5,
)
inference = SNPE(prior=prior, density_estimator=density_estimator)

# Option B: Genuine FMPE (try if SNPE calibration is poor)
# from sbi.inference import FMPE
# inference = FMPE(prior=prior, density_estimator=density_estimator)

inference.append_simulations(theta_train, x_train)
posterior = inference.train(training_batch_size=256, max_num_epochs=100)
```

### The embedding network is critical

The dynamic spectrum is a 2D array (~100 × 1024). The embedding network compresses it to a 128-dimensional vector. The flow then maps from this vector to the 3D posterior over (DM, τ_s, α).

Try two embedding architectures:
- **Small CNN** (as above): 3 conv layers + pooling + linear
- **1D CNN along frequency axis** after collapsing time: sum the spectrum along the time axis to get a 1D dedispersion curve, then apply 1D convolutions. This is physically motivated because the DM information is in the frequency-dependent arrival time.

Compare both and use the one with better posterior calibration.

---

## Phase 4: Posterior calibration, coverage diagnostics, and misspecification tests

**Time: 5-7 days**

This is the verification step that distinguishes a real SBI paper from a toy demo. Given the known SBI failure modes documented in Hermans et al. (2021, overconfident posteriors), Hahn et al. (A&A 2025, variance inflation), and the domain-adaptation literature, this phase must be thorough enough to preemptively address every criticism a knowledgeable reviewer will raise.

### Expected coverage (simulation-based calibration)

For each credible level α (e.g., 50%, 90%, 95%), compute the fraction of held-out simulations where the true parameter falls inside the α-credible region of the predicted posterior. If the posterior is well-calibrated, the 90% credible region should contain the truth 90% of the time.

```python
from sbi.analysis import run_sbc  # simulation-based calibration

ranks = run_sbc(posterior, prior, x_test, theta_test, num_simulations=10000)
# Plot rank histograms — they should be uniform if calibrated
```

### TARP diagnostics (MANDATORY)

Use the Tests of Accuracy with Random Points (TARP) diagnostic from Lemos et al. (2023, arXiv:2302.03026), which is becoming the community standard for SBI posterior validation. TARP tests whether the estimated posteriors have correct coverage properties by computing expected coverage probabilities at random reference points in parameter space, providing a more sensitive calibration check than standard SBC rank histograms.

```python
# TARP is available in the sbi package or as a standalone implementation
# Plot the TARP coverage curve — it should lie on the diagonal
```

### Per-parameter calibration

Report calibration separately for DM, τ_s, and α. It is common for SBI posteriors to be well-calibrated for some parameters and poorly calibrated for others (especially nuisance parameters). If α is poorly calibrated, discuss whether it is identifiable from the data at all.

### Conditional calibration

Check whether calibration degrades at specific parameter values. Bin by DM (low, medium, high) and by S/N (low, medium, high) and report coverage in each bin. If the simulator is biased at extreme DMs, the posterior will be miscalibrated there. This directly addresses the Hahn et al. finding about conditional miscalibration.

### Model misspecification test (MANDATORY)

This test directly addresses the dominant SBI criticism: what happens when the forward model is wrong?

1. **Scintillation-off test:** Retrain the posterior estimator on simulations generated WITHOUT scintillation (set Δν_scint → ∞). Apply this misspecified estimator to the full test set (which includes scintillation). Measure how coverage degrades. If the misspecified model produces overconfident posteriors (coverage below nominal levels), this demonstrates that scintillation modeling matters — justifying the complexity of the full simulator. If coverage is unaffected, scintillation does not matter for DM estimation and the simulator can be simplified — report this honestly.

2. **Scattering-exponent test:** The default forward model uses a scattering frequency exponent of -4 (Kolmogorov thin screen). Retrain on simulations with exponent -4 but test on simulations generated with exponent -3.5 (a plausible alternative). Does the posterior remain calibrated? This tests sensitivity to a specific physical assumption.

3. **Report results of all misspecification tests in the paper.** The Huppenkothen checklist requires sharing what did NOT work. If misspecification causes catastrophic failure, that defines the method's scope. If the method is robust to misspecification, that builds trust.

---

## Phase 5: Apply to real CHIME data

**Time: 1 week**

### Download CHIME Catalog 2 dynamic spectra

**CRITICAL DATA CORRECTION:** The CHIME/FRB Catalog 2 (arXiv:2601.09399, submitted January 2026) provides **total-intensity dynamic spectra** (waterfall plots) at 0.983 ms time resolution and 24.4 kHz frequency resolution for ~4,539 bursts from 3,641 sources. It does **NOT** provide baseband (raw voltage) data for all events. The Catalog 2 paper explicitly states that baseband analysis will be presented "in a forthcoming paper." For Catalog 1, baseband data was released for only 140 out of 536 events.

**The project uses intensity waterfalls, not baseband.** This is the actual data product available. The simulator must produce output in the same format: total-intensity dynamic spectra at CHIME's channelization (1024 frequency channels across 400–800 MHz, ~1 ms time resolution). The intensity waterfalls from Catalog 2 are sufficient for DM, scattering, and spectral index inference — baseband is not needed for these parameters.

**Week 1 checkpoint:** Download 10 real CHIME Catalog 2 waterfalls and verify they are in the expected format (2D intensity arrays with time and frequency axes). Confirm the exact resolution, normalization, and any preprocessing steps applied by the CHIME pipeline.

### Preprocess to match simulator output

The neural posterior estimator expects inputs in the same format as the simulator output. Ensure:
- Same frequency range (400–800 MHz, 1024 channels)
- Same time resolution (~1 ms)
- Same normalization (divide by the median, take log, z-score normalize)

### Run inference

For each of the 4,545 CHIME bursts:
```python
posterior_samples = posterior.sample((10000,), x=real_spectrum)
```

This gives 10,000 samples from the approximate posterior p(DM, τ_s, α | observed spectrum). Compute summary statistics:
- Posterior median for each parameter
- 68% and 95% credible intervals
- Pairwise correlation between DM and τ_s

### Validation on real data (no ground truth available)

You cannot check coverage on real data (you do not know the true DM perfectly). Instead:
- **Compare DM posterior medians to the CHIME catalog's published DM values.** They should be consistent. Plot (our DM) vs (CHIME DM) — it should be a tight diagonal.
- **Check posterior width vs S/N.** Higher S/N bursts should have tighter posteriors.
- **DM–τ_s correlation.** Plot the posterior correlation coefficient between DM and τ_s across all bursts. Theory predicts a positive correlation (more distant bursts have both higher DM and more scattering). Report whether this is observed.

---

## Phase 6: The joint DM–scattering–spectral-index catalog

**Time: 3-5 days**

Compile the posteriors for all 4,545 bursts into a catalog:

| Burst ID | DM_median | DM_68lo | DM_68hi | tau_median | tau_68lo | tau_68hi | alpha_median | alpha_68lo | alpha_68hi | DM_tau_corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

This catalog is the second major deliverable (after the simulator). Release it as a supplementary table.

### Science analysis from the catalog

- **DM–τ_s relation:** Plot median τ_s vs median DM. Fit a power law τ_s ∝ DM^β. Theory predicts β ≈ 2 for a Kolmogorov turbulence spectrum in a single thin screen. If β differs, discuss what this means for the distribution of scattering material.
- **Spectral index distribution:** Plot the histogram of α values. The mean spectral index of FRBs is debated; your catalog provides per-burst measurements with uncertainties.
- **Outliers:** Identify bursts with unusually wide posteriors or strong DM–τ_s anti-correlations. These may be pathological cases where the forward model breaks.

---

## Phase 7: Verification and injection tests

**Time: 5-7 days**

1. **Simulator self-consistency:** Generate a burst with known parameters. Run the posterior estimator. The true parameters should fall within the 68% credible region.

2. **Injection test (500 bursts — matching the Gebhard A&A standard).** Inject 500 simulated bursts into real CHIME noise (not simulated noise — use actual telescope data segments known to contain no burst). Run the estimator on each injection. Check coverage: does the 90% credible interval contain the true DM 90% of the time? Report coverage separately for:
   - Low S/N (8-20), medium S/N (20-50), high S/N (>50)
   - Low DM (<300), medium DM (300-1000), high DM (>1000)
   - Low scattering (τ_s < 1 ms), high scattering (τ_s > 10 ms)
   This is the closest analog to a true coverage test on real data and the number (500) matches the validation standard in the FMPE exoplanet paper by Gebhard et al. (A&A 2025).

3. **Comparison to CHIME catalog:** For bursts where CHIME published uncertainty estimates, check that our posteriors are consistent (overlapping credible intervals).

4. **Ablation: scintillation on vs off.** Compare posteriors from the full model vs the scintillation-off model on the same real bursts. If scintillation makes a big difference for wide bursts but not narrow ones, report this — it justifies the simulator complexity. If scintillation makes no difference, report that honestly — it simplifies the method for future users.

5. **Report what does NOT work.** Required by the Huppenkothen checklist. If certain embedding architectures failed (e.g., the 1D collapsed CNN was worse than the 2D CNN), report this. If certain prior ranges caused training instabilities, report this. If posteriors for α are uninformative for low-S/N bursts, report this. Normalize failure.

---

## Phase 8: Paper writing plan

**Title:** "Calibrated Joint Posteriors on Dispersion, Scattering, and Spectral Index for ~4,500 CHIME/FRB Bursts via Simulation-Based Inference"

Note: "Calibrated Joint Posteriors" leads the title — this is the actual novelty (posteriors, not point estimates). Do NOT use "Flow Matching" in the title unless you actually use FMPE rather than SNPE. The forward model is discussed prominently but is an extension of Rajabi et al.'s existing simulator, not the sole contribution.

### Sections (full MNRAS / ApJ paper, 12-16 pages — target the full paper, not a Letter):

A Letter (5-8 pages) would not accommodate the validation rigor required to preempt SBI criticisms. Target a full paper.

1. **Abstract** (200 words) — Scintillation-aware forward model for FRB dynamic spectra. Calibrated 3D posteriors via flow matching posterior estimation. 500-injection coverage test. Catalog of DM, τ_s, α posteriors for 4,545 CHIME bursts. DM–scattering correlation measured.

2. **Introduction** (1.5 pages) — FRB DM measurement, limitations of point estimates, the scintillation problem, SBI as a solution. Cite the SBI best-practices literature (Huppenkothen et al. 2023, Hermans et al. 2021) to signal methodological awareness.

3. **Forward Model** (2 pages) — This is the primary contribution. Describe each simulation step (intrinsic pulse, dispersion, scattering, scintillation, instrumental response). Include a figure showing a simulated burst at each stage of the pipeline. Describe the prior ranges. Include the simulator-vs-real statistical comparison (pixel histograms, pulse profiles, bandpass shapes). THIS SECTION IS WHAT THE PAPER WILL BE CITED FOR.

4. **Neural Posterior Estimation** (1 page) — Embedding network, flow architecture, training details.

5. **Calibration and Validation** (2 pages) — This is the section that preempts SBI criticisms.
   - 5.1 SBC rank histograms and TARP diagnostics on held-out simulations
   - 5.2 Per-parameter and conditional calibration (by DM, S/N, τ_s)
   - 5.3 Model misspecification tests (scintillation-off, scattering-exponent variation)
   - 5.4 500-injection test in real CHIME noise (coverage by S/N, DM, τ_s bin)

6. **Results** (2-3 pages)
   - 6.1 Comparison to CHIME catalog DMs
   - 6.2 Posterior width vs S/N
   - 6.3 DM–τ_s power law relation
   - 6.4 Spectral index distribution
   - 6.5 Ablation: scintillation on vs off

7. **Where the method fails** (0.5-1 page) — For which bursts are posteriors uninformative (posterior width > X% of prior width)? At what S/N threshold? For what scattering timescales does the single-screen assumption break? Report explicitly.

8. **Discussion** (1 page) — What the DM–τ_s relation tells us about the intergalactic turbulence spectrum. Comparison to Konar et al. (2025, OJAp) population-level SBI results. Connection to Rajabi et al. (2025, arXiv:2512.24003) CNN-LSTM point estimates. Limitations.

9. **Conclusions** (0.5 page) — Catalog released as machine-readable supplementary table. Forward model code released as standalone Python package with Zenodo DOI. Trained posterior estimator weights released.

10. **References** (~35)

11. **Data and code availability** — Forward simulator: standalone GitHub repository + Zenodo DOI. Posterior estimator: separate repository. Catalog: CDS/VizieR supplementary table. All dependencies pinned (pip freeze or Docker).

---

## Timeline

| Week | Activity |
| --- | --- |
| 1 | Setup, understand CHIME data format, read Rajabi et al. 2025 |
| 2-3 | Build forward simulator step by step (dispersion → scattering → scintillation → noise) |
| 4 | Validate simulator against real CHIME spectra, generate 500K training sims |
| 5-6 | Build and train neural posterior estimator, run calibration diagnostics |
| 7 | Apply to real CHIME data, compile catalog |
| 8 | Science analysis (DM–τ_s relation, spectral index distribution) |
| 9 | Injection test, ablation (scintillation on/off), verification |
| 10-11 | Write paper |
| 12 | Revise, submit |

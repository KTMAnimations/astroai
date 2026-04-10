# Research Plan: The Observation-Duration Scaling Law for Asteroseismic Classification

**Proposal 2 from top_5_proposals.md**

---

## What is being created that is new

The naive version of this project — "swap CNN for Mamba and report accuracy" — is not publishable beyond RNAAS. Swapping architectures is not a scientific contribution. The community paper by Huppenkothen et al. (2023, arXiv:2310.12528) explicitly identifies "machine learning for machine learning's sake" as the primary failure mode of astro-ML papers, and the Nature Astronomy perspective by Peiris (2026) argues that if an LLM can easily replicate your contribution, the work was too routine.

This project avoids both traps by answering a **scientific question first**: **How does asteroseismic classification accuracy degrade with observation duration, what drives the degradation, and does architecture choice matter?**

**Critical prior work that this project extends (not replaces):** Hon et al. (2018, MNRAS 476, 3233, arXiv:1802.07260) already measured classification accuracy at 27, 82, 356, and 1460 days using their CNN, finding 93.2% accuracy at 27 days. Kuszlewicz et al. (2020, MNRAS 497, 4843, arXiv:2007.10921) measured >91% at 27 days with a time-domain method. The basic scaling law therefore exists. **This project does NOT claim to be the first measurement of duration-dependent accuracy.**

What this project provides that Hon 2018 and Kuszlewicz 2020 did not:

1. **A multi-architecture benchmark at each duration.** Hon 2018 tested only their CNN. Kuszlewicz 2020 used only their time-domain method. No study has compared CNN, Transformer, state-space model (Mamba), AND a simple non-neural baseline (Random Forest on hand-crafted features) at controlled parameter budgets across a systematic duration sweep. This answers: does architecture choice matter, and if so, at which durations?
2. **A non-neural baseline.** No prior study compared deep learning to a simple feature-engineered classifier at each duration. If Random Forest on (ν_max, Δν) matches all neural methods, the community should know that deep learning is unnecessary for this task.
3. **Cross-survey transfer evaluation.** A model trained on Kepler, evaluated on TESS without retraining, measuring the domain gap for asteroseismology across surveys. No prior study tested this.
4. **Interpretability and failure mode characterization.** Which types of stars become unclassifiable at short durations, and why? Which parts of the spectrum do different architectures attend to? No prior study provided this analysis. This is the aspect that provides physically interpretable guidance for TESS/PLATO/Roman target selection.
5. **A finer duration grid.** Hon 2018 tested 4 durations. This project tests 7 (27, 60, 90, 180, 365, 730, 1460 days), providing a more complete picture including the 60-day and 90-day points relevant to TESS extended missions and PLATO.

---

## Phase 1: Environment setup

**Time: 1-2 hours**

```
python -m venv mamba_rg_env
source mamba_rg_env/bin/activate
pip install torch numpy scipy matplotlib pandas astropy lightkurve
pip install mamba-ssm   # REQUIRES CUDA 11.6+ and Linux — no pure-PyTorch fallback exists in this package
# If no CUDA available, use ONE of these alternatives:
#   pip install transformers   # HuggingFace Mamba with use_mambapy=True in config (slow but works on CPU)
#   OR clone https://github.com/johnma2006/mamba-minimal (pure PyTorch, single file)
#   OR clone https://github.com/Xinguang/MiniMamba (pure PyTorch, CPU/CUDA/Apple Silicon)
# If Mamba install fails entirely, fall back to S4:
#   git clone https://github.com/state-spaces/s4  # has genuine pure-PyTorch fallbacks (cauchy_naive)
```

Project structure:
```
mkdir -p project/{data,spectra,models,results,figures}
```

---

## Phase 2: Data acquisition

**Time: 1-2 days**

### Kepler power spectra

Download pre-computed power spectra from MAST HLSP KEPSEISMIC:
```
https://archive.stsci.edu/prepds/kepseismic/
```

Each file is one star's power spectrum, computed via FFT on gap-filled (inpainted) data using the KADACS pipeline (García et al. 2011, MNRAS 414, L6 for the correction pipeline; García et al. 2014, A&A 568, 10 for the inpainting addition). Alternatively, download raw Kepler light curves with `lightkurve` and compute power spectra yourself using Lomb-Scargle. **Note:** If you use KEPSEISMIC spectra for the full-duration benchmark but compute truncated spectra via Lomb-Scargle, the methodology differs (FFT on inpainted data vs. Lomb-Scargle on gapped data). For Kepler's near-uniform sampling the results will be similar, but note this in the paper.

```python
import lightkurve as lk
lc = lk.search_lightcurve("KIC 12345678", mission="Kepler").download_all().stitch()
pg = lc.to_periodogram(method="lombscargle", minimum_frequency=1, maximum_frequency=283)
```

### Labels

Download the APOKASC-3 catalog. **DATA ACCESS NOTE:** Some ADS listings indicate the data tables may be "available on request" from the authors (pinsonneault.1@osu.edu) rather than freely downloadable from VizieR or MAST. **Verify access in week 1** — email the authors if needed. This is a potential week-one blocker. Each star has:
- Evolutionary state: RGB (red giant branch) or RC (red clump)
- ν_max (frequency of maximum oscillation power, in μHz)
- Δν (large frequency separation, in μHz)

Keep stars with confident evolutionary state labels and measured ν_max and Δν. Expected yield: **~11,371 stars with evolutionary state labels** (4,755 RC + 6,616 RGB/AGB; evolutionary state classifications from Vrard et al. 2025, A&A 697, A165, cross-matched to APOKASC-3) out of 15,808 total stars in APOKASC-3 (Pinsonneault et al. 2025, ApJS 276, 69, arXiv:2410.00102). The remaining ~4,400 stars lack confident RGB/RC classifications. Cite both papers.

**AGB contamination warning:** The "RGB/AGB" category is a combined label — asteroseismology cannot reliably distinguish RGB from AGB stars, especially at the most evolved stages. Vrard et al. notes that methods are consistent near the red clump (low Δν), but agreement degrades for the most evolved AGB stars beyond the clump. An unknown fraction of "RGB" training labels are actually AGB stars with different internal structure. **Consider restricting training to Δν > 2 μHz** where labels are more reliable, or at minimum flag luminous giants where AGB/RGB confusion is highest. Note in the paper that misclassifications at the luminous end may reflect label noise, not model failure.

### TESS held-out set

**TESS cross-survey test set — DATA AVAILABILITY WARNING:**

Sreenivas et al. 2026 (arXiv:2604.00498; first author K.R. Sreenivas) describes a 19,151-star TESS CVZ red-giant catalog. However, as of April 2026, the associated GitHub repository (redeaglekr/TESS_CVZ) contains only a LICENSE, README, and an animated GIF — **the catalog tables, power spectra, and classification labels have not been uploaded yet.** The README says "more updates to follow."

**Consequences for this project:**
- The cross-survey transfer experiment (Phase 6) is blocked until the data is released.
- **Workaround:** Compute TESS CVZ power spectra independently using `lightkurve` for the subset of stars overlapping with the Kepler APOKASC-3 sample (which provides ground-truth labels). This avoids dependence on the Sreenivas catalog.
- **Label quality concern:** If using the Sreenivas catalog's RGB/CHeB labels when they become available, note that these labels are themselves **CNN-derived** (from the Hon et al. architecture). The "transfer test" would then measure model-to-model agreement, not accuracy against independent ground truth. To avoid this, use only the Kepler-TESS overlap stars where APOKASC-3 provides asteroseismic ground-truth labels. This limits the TESS test sample but provides honest validation.

This cross-survey transfer is one of four contributions, not the primary one. The core project (duration scaling law on Kepler data) is not affected by this data availability issue.

---

## Phase 3: The observation-duration truncation experiment

**Time: 2-3 days of data preparation**

This is the scientific core that makes the project non-derivative.

For each Kepler star:
1. Take the full ~4-year light curve
2. Truncate it to the first T days, for T = 27, 60, 90, 180, 365, 730, 1460 days
3. Compute a power spectrum from each truncated light curve
4. The result is 7 versions of each star's spectrum, at different observation durations

The truncation must be done on the light curve BEFORE computing the power spectrum, not by truncating the spectrum itself. This is because the frequency resolution of the power spectrum depends on the total time baseline: Δf = 1/T.

Store all truncated spectra. For 15,000 stars × 7 durations, this is 105,000 spectra. Each spectrum is a 1D array, typically 10,000–100,000 frequency bins (shorter durations produce fewer bins). Total disk: ~5-15 GB.

---

## Phase 4: Build the Mamba architecture

**Time: 3-5 days**

### Architecture

```
Input:  1D power spectrum, shape (L,) where L varies by duration
        → Linear projection to hidden_dim

Mamba Block (repeated N times):
    → Layer norm
    → Mamba selective state-space layer (from mamba_ssm)
    → Residual connection

Global pooling:
    → Mean over the sequence dimension → single vector of size hidden_dim

Output heads:
    → Linear(hidden_dim, 2)  for RGB/RC classification
    → Linear(hidden_dim, 1)  for ν_max regression
    → Linear(hidden_dim, 1)  for Δν regression
```

Design choices:
- **hidden_dim = 256.** A Mamba block with d_model=D, expansion factor 2, and d_state=16 has approximately 3 × 2 × D² parameters. With D=64 and 4 blocks, total is only ~100K parameters — far below the ~2M target for a fair comparison. With D=256 and 4 blocks, total is ~1.5-2M parameters, matching the CNN and Transformer budgets. **Previous versions of this plan used hidden_dim=64, which gave ~100K parameters — a 50x mismatch with the claimed ~5M budget. This has been corrected.**
- **N = 4 Mamba blocks.** Mamba blocks are lightweight; 4 is typical for sequence classification tasks.
- **Input normalization:** take log10 of the power spectrum, then z-score normalize (subtract mean, divide by std) per spectrum.
- **Variable-length handling:** Since the plan trains a separate model per duration, all spectra within a single training run have similar lengths (determined by T). No variable-length handling is needed within a training run. Pad to the maximum length within each duration bucket if minor length variations exist. **Note:** Mamba does NOT natively handle variable-length inputs without padding (see mamba-ssm GitHub issue #236). `pack_padded_sequence` is incompatible with Mamba (it is an RNN-specific PyTorch feature). This is not a problem for this project because we train per-duration.
- **Also cite:** TSCMamba (Ahamed & Cheng 2024, arXiv:2406.04419, Information Fusion 2025) — a Mamba variant specifically designed for time series classification. A reviewer will ask why vanilla Mamba was used instead of TSCMamba. Justify: simplicity, interpretability of single-view approach, fair comparison to single-view CNN/Transformer. Or include it as a fifth architecture.
- **Architecture version:** Use **Mamba-2 or Mamba-3** (arXiv:2603.15569, ICLR 2026) rather than the original Mamba-1 (2023). Mamba-3 introduces more expressive recurrence (complex-valued state updates, trapezoidal discretization, MIMO formulation). The "Achilles' Heel of Mamba" paper (arXiv:2509.17514) documented asymmetry bias on synthetic sequence-matching tasks — these specific limitations may have low relevance to spectral classification where frequency ordering is physically meaningful, but using the latest architecture avoids the question. A pure-PyTorch implementation exists (mamba3-minimal on GitHub). A reviewer will ask why the 2023 architecture was used when the improved version was published before the project started. If Mamba-3 proves too difficult to install, use Mamba-2 (from the official `mamba-ssm` package) and acknowledge the limitation.

### Comparison architectures (same parameter budget)

**CNN baseline — two variants to resolve the echelle circularity:**

The echelle diagram requires knowing Δν before constructing the input, but Δν is one of the regression targets. Hon et al. 2018 resolved this by using Δν values pre-computed from the SYD pipeline (Huber et al. 2009), making their CNN a two-stage pipeline, not end-to-end. To ensure a fair comparison:

- **CNN-1D (fair comparison to Mamba):** A 1D CNN applied directly to the raw log power spectrum, with the same three output heads (classification, ν_max, Δν). This is a fair apples-to-apples comparison because both Mamba and this CNN receive the same input. Use a standard architecture: 5 conv layers with kernel size 7, stride 2 for downsampling, batch normalization, ReLU, and global average pooling. **Note:** No published asteroseismology paper uses a 1D CNN on raw power spectra (Hon et al. used folded 2D spectra). This is intentionally a generic baseline — its performance relative to the specialized 2D CNN tests whether the echelle folding step adds value.
- **CNN-2D (reimplementation of Hon et al. 2018):** Fold the spectrum at a pre-computed Δν from the SYD pipeline to produce an echelle diagram. 3-layer 2D CNN with batch normalization and max-pooling. This is a two-stage pipeline (SYD + CNN) and is NOT directly comparable to single-stage methods like Mamba. Include it for reference and comparison to published results, but clearly note in the paper that it receives additional information (pre-computed Δν) that the other methods do not.

**Transformer baseline:**
- Same input projection
- 4 Transformer encoder layers with 4 attention heads
- Same global mean pooling and output heads
- **Use FlashAttention** via PyTorch 2.0+'s `torch.nn.functional.scaled_dot_product_attention`, which auto-selects FlashAttention-2 when inputs are CUDA float16/bfloat16 tensors. To force a specific backend for benchmarking:
  ```python
  from torch.nn.attention import SDPBackend, sdpa_kernel
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
      output = F.scaled_dot_product_attention(query, key, value)
  ```
  **Note:** There is NO `enable_flash=True` parameter on `scaled_dot_product_attention` — the deprecated `torch.backends.cuda.sdp_kernel(enable_flash=True)` context manager existed briefly but is replaced by the above API. FlashAttention reduces memory from O(L²) to O(L) but compute remains O(L²d). The practical crossover where Mamba outperforms FlashAttention Transformers is ~4K tokens for training and ~8K for inference. **Critical nuance:** At 27 days, spectra have ~2,300 bins. At 60 days, ~5,200 bins. These are BELOW the crossover where Mamba outperforms FlashAttention Transformers. Mamba's computational advantage only materializes at 365+ days (~30K bins), where classification accuracy is already >98% and the scientific question is uninteresting. **Do not frame the paper around Mamba's efficiency — it is irrelevant at the durations that matter most.** Frame it around the multi-architecture comparison and failure-mode analysis instead. Report timing at all durations honestly, including cases where Mamba has no advantage.

**Random Forest / XGBoost baseline (non-neural, feature-engineered):**
This baseline follows the recommendation from the Huppenkothen et al. (2023) best-practices white paper (arXiv:2310.12528, published in the Bulletin of the AAS — a non-peer-reviewed community bulletin, but widely cited), which advises comparison to "the simplest model possible without sacrificing performance." Extract hand-crafted features from each power spectrum:
- ν_max estimate (frequency of maximum power, from a smoothed envelope)
- Δν estimate (autocorrelation of the spectrum)
- Background level (median power in a ν_max-dependent window: [0.1×ν_max, 0.5×ν_max]). **NOT 1–10 μHz** — granulation frequencies scale with ν_max (Kallinger et al. 2014, A&A 570, A41). For typical red giants with ν_max ~ 30–50 μHz, 1–10 μHz captures instrumental noise, not granulation. Only for the most luminous giants (ν_max ~ 3–5 μHz) does that range contain granulation.
- Spectral flatness (ratio of geometric mean to arithmetic mean of power) — borrowed from audio signal processing. **Hypothesized to be discriminative** between RGB and RC, but the direction is an empirical question: RGB stars have more theoretical mixed modes per Δν (smaller ΔΠ₁) but most are damped below detectability due to high inertia, leaving few visible peaks. RC stars have fewer theoretical modes but more are observable (Elsworth et al. 2017, MNRAS 466, 3344). The mapping from mode density to spectral flatness is therefore non-obvious — **do not predict the direction a priori; measure it and report what you find.** This feature has no asteroseismology precedent.
- Total integrated power in the ν_max ± 3Δν window

Train a Random Forest (scikit-learn, 500 trees) and XGBoost classifier on these ~10 features. These baselines answer a critical question: **does a deep learning model learn anything from the raw spectrum that hand-crafted features miss?** If Random Forest matches the neural architectures at all durations, the answer is no — and that is itself an important finding that tells the community feature engineering still suffices.

Match all neural architectures to approximately the same number of trainable parameters (~1.5-2.5M). **Note:** Parameter count is a standard but imperfect proxy for model capacity — different architectures have different inductive biases and effective capacity at the same parameter count. Acknowledge this limitation in the paper. The Random Forest is not parameter-matched (it has far fewer effective parameters) and that is intentional — it represents the simplest reasonable approach.

**Also cite:** He et al. 2022 (MNRAS 512, 1710) used XGBoost on LAMOST spectral features for RGB/RC classification, achieving >90% purity. Different domain (spectroscopic vs. asteroseismic), but a reviewer will find it. Acknowledge and distinguish in the introduction.

---

## Phase 5: Training

**Time: 1-2 weeks**

### Training strategy

For each observation duration T:
1. Use the truncated spectra at duration T
2. Split into 80% train, 10% validation, 10% test by star (not by truncation)
3. Train Mamba, CNN, and Transformer independently
4. Record test-set accuracy (RGB/RC), ν_max RMSE, Δν RMSE
5. Also record wall-clock training time and inference time per star

### Loss function

**WARNING: The naive multi-task loss is severely imbalanced.** Cross-entropy for binary classification is O(1). MSE on ν_max (range ~1–280 μHz) for moderate errors is O(100). MSE on Δν (range ~0.5–18 μHz) is O(1-100). The ν_max term will dominate training, starving the classification head. This is a well-known multi-task learning problem (Kendall et al. 2018, CVPR).

**Fix:** Normalize regression targets to zero mean and unit variance BEFORE computing MSE, so all loss terms are O(1). Alternatively, use learned task weights (homoscedastic uncertainty weighting per Kendall et al.). At minimum, include a **single-task ablation**: train classification-only and regression-only models and compare to the multi-task model. Hon et al. 2018 trained separate models for classification and parameter estimation — if the multi-task approach hurts classification accuracy, report this honestly and use separate models.

```python
# Normalized targets (zero mean, unit variance)
nu_max_norm = (nu_max - nu_max_mean) / nu_max_std
delta_nu_norm = (delta_nu - delta_nu_mean) / delta_nu_std
loss = cross_entropy(class_pred, class_true) + mse(nu_max_pred, nu_max_norm) + mse(delta_nu_pred, delta_nu_norm)
```

### Hyperparameters

- Optimizer: **AdamW** (not Adam) with weight decay 0.1, betas=(0.9, 0.95), gradient clipping at 1.0. This follows Mamba training conventions. Add a **linear warmup** phase (10 epochs) before cosine annealing.
- Batch size: 32
- Epochs: 100 with early stopping on validation loss (patience=15)
- **Pooling ablation for Mamba:** Test both mean pooling (standard for spectra where position = frequency = physical meaning) AND last-token pooling (more standard for Mamba since it's a directional model where the last state "sees" the full sequence). Report which works better.
- **Data augmentation:** Try at minimum (a) noise injection (jittering — add Gaussian noise scaled to ~10% of local power), (b) amplitude scaling (multiply by a random factor in [0.8, 1.2] to simulate different S/N conditions). Hon et al. 2018 themselves augmented by splitting Kepler time series into multiple shorter segments. **(c) Frequency masking (SpecAugment-style) — use with EXTREME CAUTION.** No published asteroseismology paper uses SpecAugment on power spectra. Masking frequency bins could destroy exactly the mixed-mode features that distinguish RGB from RC. If the masked region covers the dipole mixed-mode pattern, the augmented example becomes a star that looks like RGB but is labeled RC. Test with and without; if it hurts classification, drop it and report why. Report results of all augmentation experiments — including failures.

### The key output: the scaling law

After training all models at all durations, you have a table:

| Duration (days) | Mamba acc | CNN acc | Transformer acc | Mamba ν_max RMSE | ... |
| --- | --- | --- | --- | --- | --- |
| 27 | ? | ? | ? | ? | ... |
| 60 | ? | ? | ? | ? | ... |
| ... | | | | | |
| 1460 | ? | ? | ? | ? | ... |

Plot accuracy vs duration for all architectures on the same axes. The plot IS the main figure of the paper. It answers: "at what observation duration does each architecture fail, and does the choice of architecture matter?"

**Statistical note:** The same test stars appear at all 7 durations (different-duration models are independent, so no data leakage). But accuracy confidence intervals across durations are correlated (measured on the same stars). Cross-duration comparisons are paired measurements — use McNemar's test or paired bootstrap, not independent-sample tests.

---

## Phase 6: Cross-survey transfer (Kepler → TESS)

**Time: 3-5 days**

Take the Mamba model trained on full-length Kepler spectra. Evaluate it — without any retraining — on TESS CVZ power spectra (27-day observations from a different telescope with different noise properties).

This measures the "sim-to-real gap" for asteroseismology across surveys. Report:
- Classification accuracy on TESS stars with known evolutionary states
- ν_max and Δν errors compared to literature values

If performance degrades badly, also try fine-tuning the last layer only on 100 labeled TESS stars and report the improvement. This tells astronomers how much labeled data they need to adapt to a new survey.

---

## Phase 7: Verification, interpretability, and failure modes

**Time: 4-5 days**

### Verification

1. **Known-answer test:** For a handful of well-studied stars (e.g., KIC 8410637, a well-known RC star), verify that the model's classification and parameter estimates match published values.
2. **Confusion analysis:** Which stars does Mamba misclassify? Are they the same stars that Hon et al. 2018 misclassified, or different ones? Plot the misclassified stars in the ν_max–Δν plane and check whether they cluster near the RGB/RC boundary.
3. **Timing benchmark:** Measure wall-clock time for Mamba vs Transformer inference on spectra of length 10K, 50K, 100K. Mamba should scale linearly; Transformer quadratically. Plot this.

### Interpretability (MANDATORY — recommended by Huppenkothen et al. 2023 best-practices white paper, arXiv:2310.12528)

Reviewers will reject a paper that treats all models as black boxes. For each architecture, provide at least one interpretability analysis:

4. **Transformer attention maps:** Extract and visualize the attention weights from the final Transformer encoder layer for correctly-classified and misclassified stars. Do the attention heads focus on the ν_max region? On the Δν-spaced oscillation peaks? On the granulation background? If attention aligns with known asteroseismic features, this provides astrophysical insight — the model is learning real physics, not artifacts. If attention focuses on noise or instrumental features, that is a diagnostic of fragility.

5. **Mamba interpretability:** Use the "hidden attention" formulation from Ali et al. (ACL 2025, "The Hidden Attention of Mamba Models"), which reformulates Mamba as an implicit attention mechanism and provides code for visualizing attention-equivalent matrices. This is directly comparable to the Transformer attention maps above and is a published, validated method. As a simpler alternative, plot the Δ (step size) values — the closest analog to an LSTM forget gate — as a function of frequency for representative RGB and RC stars. **Treat this as a stretch goal**, not mandatory: the interpretability literature for Mamba is newer and less standardized than for Transformers.

6. **Random Forest feature importance:** Report the Gini importance ranking of the hand-crafted features. At which duration does ν_max become uninformative (importance drops below noise)? This directly measures the information content vs observation length.

### Failure mode characterization (MANDATORY)

This section follows the Huppenkothen et al. (2023) recommendation to "Explore model limits and scope" and is what elevates the paper from an architecture comparison to a scientific contribution.

7. **Duration-dependent failure map:** For each observation duration, identify the region of the ν_max–Δν plane where ALL architectures fall below 80% accuracy. Plot these "failure zones" as shaded regions that grow as duration decreases.

   **Physics context for interpreting the failure map:** The physical discriminant between RGB and RC is the asymptotic gravity-mode period spacing ΔΠ₁: ~50 s for RGB, ~100–300 s for RC (Bedding et al. 2011, Nature 471, 608). This manifests as different mixed-mode patterns in the power spectrum. The actual RGB/RC confusion zone is at **ν_max ~ 20–110 μHz** (where the primary red clump at ν_max ~20–60 μHz and secondary clump at ~40–110 μHz overlap with the RGB). At ν_max > 110 μHz, very few RC stars exist — classification is EASIER, not harder. The failure zone should grow from the 20–110 μHz confusion region outward as duration decreases.

   **At 27 days** (Δf = 1/T ≈ 0.43 μHz), individual mixed modes are unresolved for RGB stars (mixed-mode spacings ~0.08 μHz near ν_max ~ 40 μHz) and barely resolved for RC stars (~0.48 μHz). The classifier must therefore rely on broader features: the mixed-mode forest envelope shape, granulation background, or spectral shape — not individual mode frequencies. This is a testable prediction the interpretability analysis (attention maps, gate dynamics) can verify.

   **Three degradation mechanisms at short durations:** (1) frequency resolution loss (Δf = 1/T), (2) S/N reduction (fewer photons), (3) stochastic mode excitation — solar-like oscillation amplitudes vary temporally, so a 27-day window may catch an atypical realization. Consider testing this third mechanism by comparing multiple independent 27-day windows from the same 4-year Kepler light curve.

8. **Architecture-dependent failures:** Are there stars that Mamba classifies correctly but CNN does not (or vice versa)? If yes, characterize them. If the architectures fail on the same stars, the limitation is in the data, not the model — and that is the most useful finding for mission planning.

9. **Report what does NOT work.** The Huppenkothen checklist explicitly requires sharing failed approaches. If you tried data augmentation and it didn't help, say so. If a particular Mamba configuration diverged during training, say so. Normalize reporting failure.

---

## Phase 8: Paper writing plan

**Title:** "How Long Must You Observe? The Observation-Duration Scaling Law for Red-Giant Asteroseismic Classification"

Note: Mamba does not appear in the title. This signals to reviewers that the paper answers a scientific question, not that it promotes an architecture. Mamba appears in the methods section.

### Sections (full MNRAS / A&A paper, 12-16 pages):

1. **Abstract** (200 words) — The scaling law result: at X days, classification accuracy drops below Y%; at Z days, architecture choice begins to matter. Cross-survey transfer from Kepler to TESS achieves W% without retraining. The failure zone in the ν_max–Δν plane is characterized.

2. **Introduction** (1.5 pages) — The asteroseismology observation-planning problem. Why duration matters for TESS, PLATO, and Roman. Acknowledge Hon et al. (2018) duration measurements (93.2% at 27 days) and Kuszlewicz et al. (2020) as prior work. The gap: no multi-architecture comparison exists, no non-neural baseline has been tested, no interpretability or failure-mode analysis has been done, and no cross-survey (Kepler→TESS) transfer test exists. Cite Huppenkothen et al. (2023) best practices.

3. **Data** (1 page) — Kepler APOKASC-3, truncation procedure, TESS CVZ test set.

4. **Methods** (2.5 pages) — Four approaches at controlled parameter budgets:
   - 4.1 Random Forest on hand-crafted features (the simple baseline)
   - 4.2 CNN reimplementation of Hon et al. 2018
   - 4.3 Transformer encoder
   - 4.4 Mamba selective state-space model
   - 4.5 Parameter matching and training protocol

5. **Results** (4-5 pages)
   - 5.1 The scaling law (THE main figure: accuracy vs duration for all 4 methods)
   - 5.2 Does architecture matter? At which duration does the choice affect accuracy?
   - 5.3 Does deep learning beat feature engineering? Random Forest vs neural approaches
   - 5.4 Cross-survey transfer (Kepler → TESS)
   - 5.5 Computational cost comparison
   - 5.6 Interpretability: what do the models learn? (attention maps, gate dynamics, feature importance)

6. **Where the method fails** (1 page) — The failure zone in ν_max–Δν space. Which stars are unclassifiable at 27 days? Architecture-dependent vs data-dependent failures. Concrete recommendations for TESS/PLATO target selection.

7. **Discussion** (1.5 pages) — Implications for TESS (27 days), PLATO (months), and Roman. What minimum observation duration should mission planners target? How the scaling law connects to frequency resolution physics (Δf = 1/T) and mixed-mode resolvability (ΔΠ₁). Note: Astroconformer (Pan et al. 2024) performed ν_max regression, not RGB/RC classification — their 30-day test is relevant to the regression heads but not directly comparable to the classification results. Note also that 283.2 μHz (= 1/(2 × 29.4244 min)) is the Kepler long-cadence Nyquist frequency, not a stellar physics boundary — stars can oscillate at higher frequencies. Limitations and what we tried that did not work.

8. **Conclusions** (0.5 page)

9. **References** (~35)

10. **Data and code availability** — All code released on GitHub with a Zenodo DOI. Trained model weights released for community use. The scaling law table released as a machine-readable supplementary file.

---

## Timeline

| Week | Activity |
| --- | --- |
| 1 | Setup, download Kepler spectra and labels, verify data |
| 2 | Build truncation pipeline, generate all 7-duration variants |
| 3 | Implement Mamba, CNN, Transformer, Random Forest/XGBoost architectures |
| 4-5 | Train all models at all durations, collect metrics (the scaling law table) |
| 6 | Cross-survey transfer experiment (Kepler → TESS) |
| 7 | Interpretability analysis (attention maps, Mamba gates, RF feature importance) |
| 8 | Failure mode characterization (duration-dependent failure map in ν_max–Δν space) |
| 9 | Timing benchmarks, verification, known-answer tests |
| 10-11 | Figures, tables, write paper |
| 12 | Revise, submit |

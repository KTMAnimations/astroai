# Research Plan: The Observation-Duration Scaling Law for Asteroseismic Classification

**Proposal 2 from top_5_proposals.md**

---

## What is being created that is new

The naive version of this project — "swap CNN for Mamba and report accuracy" — is not publishable beyond RNAAS. Swapping architectures is not a scientific contribution. The community paper by Huppenkothen et al. (2023, arXiv:2310.12528) explicitly identifies "machine learning for machine learning's sake" as the primary failure mode of astro-ML papers, and the Nature Astronomy perspective by Peiris (2026) argues that if an LLM can easily replicate your contribution, the work was too routine.

This project avoids both traps by answering a **scientific question first**: **How short can a stellar observation be before asteroseismic evolutionary-state classification becomes unreliable?** This question has direct operational value for TESS (27-day sectors), PLATO (3-month observation chunks), and future mission planning. No systematic measurement of this scaling law exists in the literature.

The architecture comparison (Mamba vs CNN vs Transformer) is a secondary tool used to measure the scaling law, not the primary contribution. Even if all three architectures perform identically at every duration, the scaling law itself is the publishable result — it tells mission planners something they currently do not know.

The novel artifacts are:
1. **A scaling law for asteroseismic classification:** accuracy as a function of observation duration from 27 days to 4 years, measured empirically on real Kepler data truncated to progressively shorter windows. This is the primary contribution.
2. **Cross-survey transfer evaluation:** a model trained on Kepler, evaluated on TESS without retraining, measuring the domain gap for asteroseismology across surveys.
3. **An architecture comparison at each duration:** Mamba vs CNN vs Transformer vs Random Forest, showing at which observation length (if any) architecture choice starts to matter. This is the secondary contribution.
4. **Failure mode characterization:** which types of stars become unclassifiable at short durations, and why? This provides physically interpretable guidance, not just accuracy numbers.

---

## Phase 1: Environment setup

**Time: 1-2 hours**

```
python -m venv mamba_rg_env
source mamba_rg_env/bin/activate
pip install torch numpy scipy matplotlib pandas astropy lightkurve
pip install mamba-ssm   # requires CUDA; on CPU-only machines, use mamba_ssm's pure-PyTorch fallback
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
https://archive.stsci.edu/hlsp/kepseismic
```

Each file is one star's power spectrum. Alternatively, download raw Kepler light curves with `lightkurve` and compute power spectra yourself using Lomb-Scargle:

```python
import lightkurve as lk
lc = lk.search_lightcurve("KIC 12345678", mission="Kepler").download_all().stitch()
pg = lc.to_periodogram(method="lombscargle", minimum_frequency=1, maximum_frequency=283)
```

### Labels

Download the APOKASC-3 catalog from VizieR or the associated MAST HLSP. Each star has:
- Evolutionary state: RGB (red giant branch) or RC (red clump)
- ν_max (frequency of maximum oscillation power, in μHz)
- Δν (large frequency separation, in μHz)

Keep stars with confident evolutionary state labels and measured ν_max and Δν. Expected yield: ~15,000 stars.

### TESS held-out set

Download the Stokholm et al. 2025 TESS CVZ red-giant catalog (arXiv:2604.00498). Compute power spectra from TESS 2-minute cadence light curves for stars overlapping with the Kepler field (if any) and for the independent TESS CVZ sample. This is the cross-survey test set — never used during training.

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
- **hidden_dim = 64.** Small enough for fast training. Scale up to 128 if results are noisy.
- **N = 4 Mamba blocks.** Mamba blocks are lightweight; 4 is typical for sequence classification tasks.
- **Input normalization:** take log10 of the power spectrum, then z-score normalize (subtract mean, divide by std) per spectrum.
- **Variable-length handling:** since shorter durations produce shorter spectra, the model must handle variable-length inputs. Mamba handles this natively — no padding needed. Use PyTorch's `pack_padded_sequence` or simply batch stars of similar length.

### Comparison architectures (same parameter budget)

**CNN baseline (reimplementation of Hon et al. 2018):**
- Convert power spectrum to an echelle diagram (2D image): fold the spectrum at Δν and stack
- 3-layer CNN with batch normalization and max-pooling
- Same output heads

**Transformer baseline:**
- Same input projection
- 4 standard Transformer encoder layers with 4 attention heads
- Same global mean pooling and output heads
- This is the O(L²) comparison — it should be slower than Mamba on long sequences

**Random Forest / XGBoost baseline (non-neural, feature-engineered):**
This baseline is required by the Huppenkothen et al. (2023) best-practices checklist, which mandates comparison to "the simplest model possible without sacrificing performance." Extract hand-crafted features from each power spectrum:
- ν_max estimate (frequency of maximum power, from a smoothed envelope)
- Δν estimate (autocorrelation of the spectrum)
- Background noise level (median power in the 1–10 μHz range)
- Spectral flatness (ratio of geometric mean to arithmetic mean of power)
- Total integrated power in the ν_max ± 3Δν window

Train a Random Forest (scikit-learn, 500 trees) and XGBoost classifier on these ~10 features. These baselines answer a critical question: **does a deep learning model learn anything from the raw spectrum that hand-crafted features miss?** If Random Forest matches the neural architectures at all durations, the answer is no — and that is itself an important finding that tells the community feature engineering still suffices.

Match all neural architectures to approximately the same number of trainable parameters (~2-5M). The Random Forest is not parameter-matched (it has far fewer effective parameters) and that is intentional — it represents the simplest reasonable approach.

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

```python
loss = cross_entropy(class_pred, class_true) + mse(nu_max_pred, nu_max_true) + mse(delta_nu_pred, delta_nu_true)
```

### Hyperparameters

- Optimizer: Adam, lr=1e-3 with cosine annealing
- Batch size: 32
- Epochs: 100 with early stopping on validation loss (patience=15)
- No data augmentation (power spectra are not naturally augmentable)

### The key output: the scaling law

After training all models at all durations, you have a table:

| Duration (days) | Mamba acc | CNN acc | Transformer acc | Mamba ν_max RMSE | ... |
| --- | --- | --- | --- | --- | --- |
| 27 | ? | ? | ? | ? | ... |
| 60 | ? | ? | ? | ? | ... |
| ... | | | | | |
| 1460 | ? | ? | ? | ? | ... |

Plot accuracy vs duration for all three architectures on the same axes. The plot IS the main figure of the paper. It answers: "at what observation duration does each architecture fail, and does Mamba's advantage over CNN/Transformer grow as observations get shorter?"

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

### Interpretability (MANDATORY — required by Huppenkothen et al. 2023 best-practices checklist)

Reviewers will reject a paper that treats all models as black boxes. For each architecture, provide at least one interpretability analysis:

4. **Transformer attention maps:** Extract and visualize the attention weights from the final Transformer encoder layer for correctly-classified and misclassified stars. Do the attention heads focus on the ν_max region? On the Δν-spaced oscillation peaks? On the granulation background? If attention aligns with known asteroseismic features, this provides astrophysical insight — the model is learning real physics, not artifacts. If attention focuses on noise or instrumental features, that is a diagnostic of fragility.

5. **Mamba state dynamics:** For Mamba, plot the selective gate values (the data-dependent selection mechanism) as a function of frequency for representative RGB and RC stars. Do the gates open at oscillation peaks and close in noise regions? This is the Mamba-specific analog of attention maps and has not been visualized for any astronomical application.

6. **Random Forest feature importance:** Report the Gini importance ranking of the hand-crafted features. At which duration does ν_max become uninformative (importance drops below noise)? This directly measures the information content vs observation length.

### Failure mode characterization (MANDATORY)

This section is required by the Huppenkothen checklist ("Explore model limits and scope") and is what elevates the paper from an architecture comparison to a scientific contribution.

7. **Duration-dependent failure map:** For each observation duration, identify the region of the ν_max–Δν plane where ALL architectures fall below 80% accuracy. Plot these "failure zones" as shaded regions that grow as duration decreases. The physical interpretation: which stellar parameter ranges become unresolvable at short observation baselines? If the failure zone maps onto a specific evolutionary phase (e.g., low-luminosity RGB stars with ν_max > 100 μHz), report this as a concrete recommendation for TESS/PLATO target selection.

8. **Architecture-dependent failures:** Are there stars that Mamba classifies correctly but CNN does not (or vice versa)? If yes, characterize them. If the architectures fail on the same stars, the limitation is in the data, not the model — and that is the most useful finding for mission planning.

9. **Report what does NOT work.** The Huppenkothen checklist explicitly requires sharing failed approaches. If you tried data augmentation and it didn't help, say so. If a particular Mamba configuration diverged during training, say so. Normalize reporting failure.

---

## Phase 8: Paper writing plan

**Title:** "How Long Must You Observe? The Observation-Duration Scaling Law for Red-Giant Asteroseismic Classification"

Note: Mamba does not appear in the title. This signals to reviewers that the paper answers a scientific question, not that it promotes an architecture. Mamba appears in the methods section.

### Sections (full MNRAS / A&A paper, 12-16 pages):

1. **Abstract** (200 words) — The scaling law result: at X days, classification accuracy drops below Y%; at Z days, architecture choice begins to matter. Cross-survey transfer from Kepler to TESS achieves W% without retraining. The failure zone in the ν_max–Δν plane is characterized.

2. **Introduction** (1.5 pages) — The asteroseismology observation-planning problem. Why duration matters for TESS, PLATO, and Roman. Why current benchmarks (Hon 2018) only test full-length Kepler. The gap: no systematic duration-accuracy relationship exists. Cite Huppenkothen et al. (2023) best practices to signal methodological rigor to reviewers.

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

7. **Discussion** (1.5 pages) — Implications for TESS (27 days), PLATO (months), and Roman. What minimum observation duration should mission planners target? How the scaling law connects to frequency resolution physics (Δf = 1/T). Comparison to Astroconformer's brief 30-day test. Limitations and what we tried that did not work.

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

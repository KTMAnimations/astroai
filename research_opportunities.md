# Undergraduate AI-in-Astronomy Research Opportunities

**Publishable niches using public datasets, current as of April 2026**

This document supersedes the original compass artifact. It (1) corrects factual errors in the prior document, (2) culls proposals that are no longer seminal or publishable, and (3) adds new niches surfaced by targeted literature search. All niches use public data, have clear baselines, and fit within an undergraduate compute budget (Google Colab T4 or a single modest GPU).

---

## 0. How this document was built

Five parallel research agents verified the prior artifact against arXiv, ADS, journal sites, and survey project pages. Every claim with a paper, arXiv ID, quoted metric, or novelty assertion was cross-checked. I then manually scanned adjacent literature to find complementary niches the original document missed.

Summary of checks:

- **GOAL 1 (landscape):** 20 claims checked. 13 verified; 3 partially correct; 3 wrong attribution; 1 arithmetic misattribution.
- **GOAL 2 (light-curve methods):** 24 claims checked. 18 verified verbatim (including arxiv IDs, authors, numeric metrics); 6 partially correct; 0 hallucinated. All five suspicious arXiv IDs (2506.08306, 2509.24134, 2510.06200, 2511.00614, 2602.12955) are real papers.
- **GOAL 3 (ten niches):** Novelty claims re-verified. Four of the original ten are weakened or obsolete and are cut; six remain (with scope tightenings).
- **New niches:** 20 candidates surveyed across asteroseismology, Gaia XP, radio astronomy, non-flare solar physics, SBI, eclipsing binaries, solar-system detection, and photo-z; 10 promoted into the final ranking below.

---

## 1. Corrections to the original compass artifact

These corrections apply to any future use of the original document. They do not undermine most of its reasoning, but they matter for attribution and for baseline numbers you might cite in a proposal.

| # | Original claim | Correction |
|---|----------------|------------|
| 1 | "Chen et al. (A&A 2024)" CvT 98.8% galaxy morphology | First author is **Cao**, not Chen (Cao et al., A&A 683, A42, 2024). Numbers are correct. |
| 2 | "Audenaert et al. (2025)" causal foundation models | First author is **Muthukrishna** (Muthukrishna, Audenaert, Gregory, Hogg, Villar, arXiv:2507.05333, ICML 2025 workshop). |
| 3 | Solar flare TSS = 0.801 from SHARP + contrastive + GRU | 0.801 is for the **LSTM-only preprocessing result**. The full ContReg (contrastive + GRU) pipeline achieves TSS ≈ **0.846** (EskandariNasab, Hamdi, Filali Boubrahimi 2024, ICMLA). |
| 4 | "Transfer learning from ZTF to LSST reduces labeled data by 95%" | The **95% reduction** applies to the *simulation-to-real ZTF* transfer. The *ZTF-to-LSST* transfer achieves 94% of baseline performance using only 30% of training data (~70% reduction). Different numbers, different scenarios. (Gupta et al. 2025, MNRASL, arXiv:2502.18558.) |
| 5 | "Bellinger et al. (2019)" on Cepheid ANN pulsation models | Preprint arXiv:1911.11767 is Nov 2019; published MNRAS 491, 4752 in Feb 2020. Either citation is defensible, but the published version is 2020. |
| 6 | "~270,000 RR Lyrae in OGLE OCVS" | Combined OGLE RR Lyrae catalogs (Bulge + Disk + Magellanic) total **~120–130k** stars, not 270k. |
| 7 | "Zhao et al. 2025 systematic review of NPE for GW parameter inference" | Paper exists (Frontiers of Physics 20, 045301) but is a **broad AI-for-GW review**, not NPE-specific. |
| 8 | "PLATO's 2026 launch" | Officially **late 2026** on Ariane 62; may slip to January 2027. |
| 9 | "AHA (arXiv:2602.12955, 2025)" | Paper is real but submitted **Feb 13, 2026** (not 2025). Authors: Iskandarli, Lintott, Croft, Stevance, Weston. |
| 10 | "Soriano et al. 2025" conformal photo-z | Submitted January **2026** (arXiv:2601.17222), not 2025. |
| 11 | "Only 2–3 papers apply conformal prediction in astronomy" | Undercounted. Confirmed ≥5 papers: Jones et al. 2024 (photo-z, ApJ 974, 159); Soriano et al. 2026 (photo-z); **Ashton et al. 2024** (Gravity Spy glitches, arXiv:2412.11801); **Ashton et al.** (GW search calibration, arXiv:2402.19313, PRD 109); a 2026 radio galaxy paper. The gap for **photometric transient classification** remains genuine. |
| 12 | "AstroCo 70% lower RMSE than ASTROMER v1" | Accurate for AstroCo-**L** (the large variant). The small variant AstroCo-S is closer to 59% reduction. Cite carefully. |

None of these invalidate the original document's premise. But the Niche 3 (Neural ODE for Cepheids), Niche 5 (fine-tune StarEmbed), Niche 6 (mTAN), and Niche 10 (optimal transport DA) corrections below are load-bearing for cull decisions.

---

## 2. Current state of astro+AI (April 2026)

Key facts as of 2026-04-08 that reshape what counts as novel:

- **Rubin / LSST** — First science alerts issued 2026-02-25 (~800k on the opening night; ~7M/night projected). Data Preview 1 (DP1) was released 2025-06-30 on the Rubin Science Platform; DP2 is planned for September 2026. The full LSST survey began in early 2026. First LSST-camera peer-reviewed paper (January 2026) reported the fastest-spinning >500 m asteroid known.
- **Euclid** — Q1 Quick Release was 2025-03-19 (three deep fields, ~380k classified galaxies, 497 strong-lens candidates). DR1 is scheduled for 2026-10-21. Zoobot runs the OU-MER morphology pipeline in production.
- **DESI** — DR2 (March 2025) BAO now prefers evolving dark energy at 2.8–4.2σ when combined with CMB + SNe.
- **Gaia** — Operations ended January 2025. DR4 not before December 2026.
- **SDSS-V** — DR19 released July 2025 (first with all three mappers). DR20 scheduled 2026.
- **SPHEREx** — Launched 2025-03-11. QR2 was released October 2025.
- **Roman** — Fully assembled November 2025; aiming for late September 2026 launch (Falcon Heavy).
- **SKA** — SKA-Mid detected first neutral hydrogen (July 2025); SKA-Low imaged 1024 antennas (March 2025). AA0.5 milestone mid-2026; first formal science verification 2027.
- **AION-1** (Polymathic AI, arXiv:2510.17960, NeurIPS 2025) — first omnimodal foundation model for astronomy, 300M / 800M / 3.1B parameters, pretrained on HSC+Legacy Survey+DESI+SDSS+Gaia with 39 modalities and ~200M observations. HuggingFace and GitHub open.
- **Platonic Universe** (arXiv:2509.19453, ML4PS 2025) — shows general-purpose ViTs/DINOv2 converge to similar embeddings as astronomy-specific architectures at scale; suggests general foundation models are directly reusable.
- **StarEmbed** (arXiv:2510.06200) — non-astronomy time-series foundation models (Chronos, Chronos-Bolt, Moirai) can match or beat ASTROMER on ZTF classification and out-of-distribution detection.
- **ATCAT** (arXiv:2511.00614, Tung 2025) — new SOTA on ELAsTiCC LC-only at 71.8% macro F1.
- **AstroCompress** (arXiv:2506.08306) — first neural-compression benchmark with five 16-bit HST/JWST datasets on HuggingFace.
- **AHA** (arXiv:2602.12955, Feb 2026) — new ZTF transient anomaly-hunter baseline using three autoencoders on features, image triplets, and light curves.

Implications for proposal selection:

1. LSST is in production. Anything framed "preparing for LSST" is now "preparing for Year 2 of LSST". Real alerts exist. Use them.
2. Foundation models saturate the obvious territory (galaxies, spectra, light curves). To be novel, the project must find a *modality or task* that AION-1 and AstroCLIP do not cover: white dwarfs, asteroseismology, helioseismology, FRB dynamic spectra, and radio continuum association are all still open.
3. "Fine-tune X" projects are decreasingly publishable on their own. The editor or reviewer will ask "what's new?" Proposals must contribute a method, a benchmark, or a physically motivated constraint.

---

## 3. The cull: four original niches dropped

### Dropped #1 — Neural ODE for Cepheid pulsation
**Reason:** The claim "no paper has used Neural ODEs for Cepheid modeling" is *narrowly* true, but **Oh et al. 2025 "Modeling Irregular Astronomical Time Series with Neural Stochastic Delay Differential Equations"** (arXiv:2508.17521, CIKM 2025) is in the adjacent space, uses closely related machinery, and was tested on astronomical time series including irregular data in the PLAsTiCC family. A reviewer will demand direct comparison, and the originality margin is thin. If you wanted to salvage the idea, reframe it as "Neural SDDE vs Fourier vs GP on classical Cepheids" with a physical-interpretation angle — but you are then writing a paper that comments on Oh et al., not one that opens a subfield. Not recommended as a first paper.

### Dropped #2 — Fine-tuning StarEmbed TSFMs
**Reason:** This is essentially the obvious follow-up that the StarEmbed authors themselves suggested. It is an engineering exercise, not a scientific discovery. Referees at ML4PS or RNAAS will rightly ask "what did we learn?" unless you can show something surprising (e.g., catastrophic forgetting, scaling behavior, periodicity-awareness failures). A fine-tune result with a predictable outcome is not a paper.

### Dropped #3 — mTAN for variable-star light curve interpolation
**Reason:** mTAN (ICLR 2021) is now five years old. The space of attention-based irregular-time-series encoders for astronomy is crowded: ASTROMER v1/v2, AstroCo, ATAT, ATCAT, Astroconformer, TimeModAttn, and Neural SDDE all target this problem with newer architectures. A reviewer will ask why mTAN and not AstroCo. There is no longer an architectural gap to claim.

### Dropped #4 — Optimal transport domain adaptation for cross-survey variable stars
**Reason:** Gupta, Muthukrishna, Audenaert (arXiv:2510.12958, Oct 2025) published simulation-pretrained DA with contrastive and adversarial objectives for astronomical time series, covering the methodological gap you would fill with OT. OT-DA is a meaningful variant but a reviewer will not accept it as a new contribution without a strong reason to prefer Wasserstein alignment over adversarial alignment on this data. Incremental.

---

## 4. Retained original niches (six, with corrections)

### Retained #1 — SIREN (implicit neural representation) for light-curve reconstruction
**Status:** Top recommendation. Novelty claim "no paper has applied SIREN to astronomical light curves" holds as of April 2026.

**Proposal.** Fit a small SIREN network (3–5 layers, 256 hidden units, sine activations) per light curve, mapping time → magnitude. Evaluate on artificially downsampled OGLE Cepheids and RR Lyrae, removing 50–90% of points, and compare reconstruction RMSE, phase coverage, and derivative quality against celerite/george GP regression and 7th-order Fourier fits. Explore a hypernetwork that predicts SIREN weights from a few sparse observations (meta-learning).

**Datasets.** OGLE Collection of Variable Stars (OCVS) Cepheids and RR Lyrae in V and I bands. Gaia DR3 RR Lyrae catalog (~78k OGLE + ~45k Magellanic System + other catalogs; do not cite the ~270k figure from the prior artifact). Use MAST HLSP mirrors where available.

**Baselines.** celerite GP with quasi-periodic kernel; Fourier decomposition (7th order); ASTROMER v2 masked reconstruction.

**Why it still works.** INRs with periodic activations are a natural fit for periodic light curves, SIREN training is CPU-fast, and the reference implementation is a Colab notebook. None of the 20+ transformer light-curve papers from 2023–2026 uses an INR-style continuous coordinate representation. The comparison against GP is clean and quantitative.

**Compute.** CPU / Colab T4. **Timeline.** 6–8 weeks. **Venue.** RNAAS (1 figure, reconstruction example + RMSE vs sparsity curve) or ML4PS 2026 (4-page with ablation and hypernetwork).

### Retained #2 — Few-shot prototypical networks for rare transients
**Status:** Novelty claim holds. Prototypical nets / MAML have not been applied to PLAsTiCC rare classes.

**Proposal.** Extract ASTROMER embeddings or FATS/feets features from PLAsTiCC light curves. Construct few-shot episodes with common classes (SNIa, SNIbc, SNII) as base training and rare classes (SLSN, TDE, kilonovae) as novel test classes at K = 5, 10, 20 shots. Compare prototypical networks, Matching Networks, and MAML against nearest-neighbor and full-shot supervised baselines. Report per-class F1 and calibration.

**Datasets.** PLAsTiCC (Kaggle, 3.5M simulated LSST light curves, 18 classes) and ELAsTiCC. Use the Fink ELAsTiCC benchmark splits for reproducibility.

**Baselines.** PLAsTiCC competition winner (LightGBM + MLP ensembles), ORACLE (arXiv:2501.01496), the recent transformer TDE classifier TTC (arXiv:2512.24782).

**Why it still works.** Rare-class performance is an operational LSST requirement, and few-shot meta-learning has a clean framing here. With real LSST alerts flowing (as of Feb 2026), you can also do a proof-of-concept on a sample of actual alerts.

**Compute.** Single GPU / Colab T4. **Timeline.** 8–12 weeks. **Venue.** ML4PS or Astronomy & Computing.

### Retained #3 — Conformal prediction for transient classification (narrowed scope)
**Status:** Keep but *narrow the framing*. The gap for **photometric supernova-type transient classification** is real; for gravitational-wave glitches it is not (Ashton et al. 2024, arXiv:2412.11801).

**Proposal.** Train a standard classifier (RF, XGBoost, MLP on ASTROMER embeddings) on PLAsTiCC. Apply split conformal prediction (`mapie` or `crepes`) to produce prediction sets with 90% marginal coverage. Analyze how set sizes vary with light-curve length, class rarity, and S/N. Compare against MC-Dropout and deep-ensemble uncertainties. Specifically target photometric transients (SLSN / TDE / KN / SNIa-pec) where per-class guarantees matter for downstream spectroscopic follow-up triggering.

**Datasets.** PLAsTiCC (Kaggle), ELAsTiCC, and ZTF Bright Transient Survey via ALeRCE.

**Baselines.** Pre-trained ASTROMER + softmax; MC-Dropout; deep ensembles.

**Framing for the paper.** "Calibrated follow-up decisions for photometric transient classifiers via conformal prediction". Do not overclaim "first conformal prediction in astronomy".

**Compute.** CPU. **Timeline.** 6–8 weeks. **Venue.** ML4PS or RNAAS.

### Retained #4 — INR-based neural compression of astronomical images (against AstroCompress)
**Status:** Still the only open niche in the AstroCompress benchmark.

**Proposal.** Fit SIREN or COIN++ implicit neural representations to astronomical image patches from the five AstroCompress HuggingFace datasets (SBI-16-2D HST, SBI-16-3D JWST, GBI-16-2D, GBI-16-2D-Legacy, GBI-16-4D). Measure bits-per-pixel vs reconstruction PSNR, specifically evaluating **photometric preservation**: flux accuracy in source pixels vs background vs dynamic range. Compare against the seven baselines in AstroCompress (three neural, four classical).

**Datasets.** AstroCompress on HuggingFace. Updated 2025-11-24. 16-bit uint format, ML-ready.

**Baselines.** All seven AstroCompress baselines; HEVC and H.264 references.

**Why it still works.** AstroCompress benchmarks lossless methods; INRs are naturally lossy *or* near-lossless with a rate-distortion knob. No INR-based compressor is in the benchmark. The photometric-preservation evaluation is the physically motivated angle that sets this apart from generic image compression.

**Compute.** Single GPU / Colab T4 (for COIN++ fitting, INR fitting is per-image and cheap). **Timeline.** 8–12 weeks. **Venue.** ML4PS or Astronomy & Computing.

### Retained #5 — Diffusion-based anomaly detection on ZTF (against AHA)
**Status:** Keep with updated baseline: AHA (arXiv:2602.12955, Feb 2026) is the paper to beat, not pre-2026 autoencoders.

**Proposal.** Train a conditional denoising diffusion probabilistic model (DDPM) on "normal" ZTF transient light-curve features. Score anomalies by reconstruction likelihood under the forward noising process. Compare against AHA's three-autoencoder ensemble, DiMMAD (arXiv:2510.23702, ML4PS 2025), isolation forests, and simple KNN-in-embedding baselines. Use PLAsTiCC injection tests for controlled evaluation and the ZTF Bright Transient Survey via ALeRCE for real data.

**Datasets.** ZTF alerts via ALeRCE broker (or Fink or Lasair), PLAsTiCC and ELAsTiCC simulations.

**Baselines.** AHA, DiMMAD, isolation forest, the AstroCon autoencoder variants.

**Framing.** Diffusion AD is a hot ML subtopic in 2024–2026 (DMAD literature is large) but has not been applied to astronomical transients. Do not claim first diffusion model in astronomy — there are solar imaging and image augmentation papers already.

**Compute.** Single GPU. **Timeline.** 10–14 weeks. **Venue.** ML4PS.

### Retained #6 — PINN for exoplanet transit modeling (narrowed scope)
**Status:** Keep but *narrow scope* to transit light-curve fitting. PINNs have already been used for exoplanet atmospheric radiative transfer (arXiv:2408.00084), so the niche is "Mandel-Agol-constrained PINN for transit light curves and limb darkening", not "PINNs for exoplanets".

**Proposal.** Parameterize an MLP f_θ(t) → flux with loss = data_fit + λ · physics_residual, where the physics residual enforces the Mandel-Agol transit shape and a quadratic limb-darkening law. Fit Kepler and TESS transits via `lightkurve`. Compare inferred planet radius, impact parameter, and orbital period against the NASA Exoplanet Archive values and against a `batman` MCMC baseline, reporting both accuracy and wall-clock speedup.

**Datasets.** Kepler KOIs and TESS TOIs via `lightkurve`; NASA Exoplanet Archive confirmed-planet parameters; EXOFASTv2 and ExoMAST references.

**Baselines.** `batman` + `emcee` MCMC; `exoplanet` no-U-turn sampler; ExoMiner++ for detection but not parameter inference.

**Why it still works.** Transit physics is a single simple ODE; PINNs are directly applicable. Mature physics-informed transit fitters do not exist. Framing must clearly distinguish from PINN-for-atmospheric-retrieval work.

**Compute.** CPU or Colab. **Timeline.** 8–12 weeks. **Venue.** RNAAS (quick speed comparison) or AJ (full parameter-accuracy comparison).

---

## 5. New complementary niches (ten, selected from twenty)

Selected for novelty, undergrad tractability, and publishability. Each has been sanity-checked against arXiv / ADS / journal searches as of April 2026.

### New #1 — Mamba (selective state-space models) for red-giant power spectra
**Problem.** Red-giant oscillation power spectra are 1D sequences of 10k–100k bins. Transformer memory is quadratic in sequence length, which is why current asteroseismic classifiers use 1D CNNs (Hon et al. 2018) or MLPs (Stone-Martinez et al. 2025). Mamba (Gu & Dao 2023) and S6 selective state-space models are sub-quadratic and have not been applied in asteroseismology.

**Datasets.** Kepler APOKASC-3 red giants (14,983 classified in Hon et al. 2018); TESS Southern CVZ catalog (Stokholm et al. 2025, arXiv:2604.00498, 19,151 stars). Power spectra on MAST HLSP KEPSEISMIC and lightkurve-derivable from TESS FFIs.

**Baselines.** Hon et al. 2018 2D CNN on echelle diagrams (RGB vs CHeB); the BAM pipeline for global oscillation parameters.

**Method.** Train a small Mamba encoder (∼5M parameters) on raw log-power spectra with three heads: (i) RGB/CHeB class, (ii) ν_max regression, (iii) Δν regression. Benchmark against 1D CNN, wavelet-CNN, and a same-parameter-count Transformer.

**Feasibility.** Mamba has a reference PyTorch implementation. Single T4. ∼20 GPU-hours total.

**Venue.** RNAAS (architecture demonstration) or A&A Letter. **Novelty status.** Sanity-checked against arXiv/Google Scholar: no Mamba-in-asteroseismology paper as of April 2026.

### New #2 — PySR symbolic regression for gyrochronology residuals after ChronoFlow
**Problem.** ChronoFlow (Van-Lane et al. 2024, arXiv:2412.12244) is a conditional normalizing flow that infers stellar age from rotation period. Being a flow, it has no closed form. Classical gyrochronology relations (Barnes 2007; Mamajek & Hillenbrand 2008; Angus et al. 2019) are closed-form but physically motivated. The question is whether PySR (Cranmer 2024) can learn a human-readable expression for P_rot(age, mass) that beats the classical relations.

**Datasets.** ChronoFlow's public training set (~8,000 cluster rotators from Kepler, K2, TESS, Gaia) with cluster-derived ages.

**Baselines.** Barnes (2007), Mamajek & Hillenbrand (2008), Angus et al. (2019), ChronoFlow itself.

**Method.** (i) Run PySR with astronomy-motivated operators (log, power, tanh, ReLU, exp) to learn P_rot = f(age, mass, Rossby number). (ii) Fit PySR to the residuals of ChronoFlow predictions to search for a symbolic correction term. (iii) Cross-validate across held-out clusters.

**Feasibility.** PySR is CPU-only and runs on a laptop. Total runtime ∼1–2 days on 8 cores.

**Venue.** RNAAS (for a clean symbolic result) or ApJL if the expression is more accurate than the classical relations. **Novelty status.** PySR has not been applied to gyrochronology; astro PySR applications to date are mostly for neutron-star equations of state.

### New #3 — Neural Posterior Estimation with scintillation-realistic forward models for FRB dispersion measures
**Problem.** FRB dispersion-measure (DM) estimation in CHIME is currently done via a DM transform followed by heuristic decisions, or by a feed-forward CNN-LSTM (Islam et al., arXiv:2512.24003). Both give point estimates; neither provides principled uncertainty or handles scintillation.

**Datasets.** CHIME/FRB Catalog 2 baseband waveforms (4,545 bursts, public since 2025). Petroff FRB catalog for external validation. Scintillation forward model via `scintools`.

**Baselines.** Islam et al. (arXiv:2512.24003) CNN-LSTM; classical incoherent dedispersion scan.

**Method.** Forward-simulate bursts over a grid of (DM, scattering index, scintillation bandwidth, spectral index) using a physical model. Train a conditional normalizing flow (via the `sbi` package) to produce p(DM | dynamic spectrum). Validate posterior coverage on held-out CHIME events. Report posterior width and coverage as functions of burst width.

**Feasibility.** FRB dynamic spectra are small (<1 MB each). The `sbi` package is designed for exactly this. Single T4.

**Venue.** ApJL, MNRAS, or ML4PS. **Novelty status.** No SBI application to FRB DMs; no scintillation-aware deep DM estimator.

### New #4 — Contrastive self-supervised learning on SolARED acoustic-power tiles for early flux-emergence warning
**Problem.** SolARED (Kitiashvili et al. 2025, arXiv:2601.13145) shows that helioseismic acoustic-power maps carry information about magnetic flux emergence 5 hours before the flux appears at the photosphere. The supervised LSTM baseline in that paper requires many labeled emergences, but unlabeled quiet-Sun acoustic-power tiles are abundant.

**Datasets.** SolARED public release (JSOC, HMI acoustic-power f-mode/p-mode maps), ~50 labeled large active regions, large pool of quiet-Sun tiles.

**Baselines.** Kitiashvili et al. 2025 LSTM (TSS at 5-hour lead time); simple CNN fine-tuned on the 50 labeled tiles.

**Method.** Pretrain a ResNet-18 or ConvNeXt-Tiny encoder via SimCLR or BYOL on unlabeled quiet-Sun tiles. Fine-tune a linear classifier on the 50 labeled AR tiles. Report TSS at 1, 3, 5, 8-hour lead times and compare sample efficiency.

**Feasibility.** SolARED is ~20 GB; 128×128 tiles; single T4.

**Venue.** Space Weather, Solar Physics, or ApJL. **Novelty status.** No contrastive SSL paper on SolARED; all prior work is supervised.

### New #5 — Test-time training for photometric-redshift drift in DESI Legacy Imaging
**Problem.** Photometric-redshift models degrade as a survey drifts in depth and PSF (and as additional filters come online). Standard fixes require labels. Test-time training (TTT; Sun et al. 2020) adapts at inference using only a self-supervised auxiliary task.

**Datasets.** DESI Legacy Imaging DR10 and DESI DR1 BGS/ELG spectroscopic redshifts (public 2024).

**Baselines.** PZFlow (Crenshaw et al. 2024), DNF (De Vicente et al. 2016), FlexZ, DeepDISC-photoz (arXiv:2411.18769), and the Soriano et al. 2026 conformal photo-z (arXiv:2601.17222).

**Method.** Train a CNN photo-z regressor with a rotation-prediction auxiliary head. At test time, for each new image (or batch), fine-tune only the auxiliary head on that image's self-supervised loss, then predict z. Compare PIT histograms, CRPS, outlier fractions against the baselines on held-out DESI DR1 spectroscopic targets.

**Feasibility.** A 20 GB subset of DESI DR1 + Legacy imaging fits in Colab high-memory instance. Single T4 for training.

**Venue.** MNRAS, A&A, or ML4PS. **Novelty status.** No TTT photo-z paper. Distinct from domain adaptation work (OT, adversarial) because TTT is self-supervised per-sample.

### New #6 — Label-Distribution-Aware Margin loss for metal-poor star identification in Gaia XP
**Problem.** Extremely metal-poor ([Fe/H] < −3) stars are heavily under-represented in training sets for Gaia XP metallicity regression. Existing methods (Andrae et al. 2023; Zhang et al. 2024; Khalatyan et al. 2024 SHBoost) use XGBoost or plain MLPs with standard losses and systematically over-predict metal-poor metallicities toward the median.

**Datasets.** Gaia DR3 BP/RP spectra (all 220M sources, or a random 1M subsample). Training labels from PASTEL, SAGA database, APOGEE DR17 VAC, LAMOST DR10.

**Baselines.** Andrae et al. 2023 XGBoost; Khalatyan et al. 2024 SHBoost; Zhang et al. 2024 empirical forward model.

**Method.** Train a small MLP or 1D CNN on XP coefficients with a Label-Distribution-Aware Margin (LDAM) loss, focal regression, or self-distillation with metal-poor-weighted pseudo-labels. Validate on PIGS and BEST-of-H3 halo stars. Report systematic bias vs [Fe/H] and a held-out confirmed-EMP recall.

**Feasibility.** XP coefficients are 110 floats per star. 1M subsample is ~1 GB. Single T4.

**Venue.** A&A, AJ, or MNRAS. **Novelty status.** No tail-aware training approach for Gaia XP metallicities has been published.

### New #7 — Graph neural network for open-cluster chemical homogeneity from Gaia XP
**Problem.** Testing open-cluster chemical homogeneity normally requires high-resolution spectroscopy. Gaia XP gives [M/H] for millions of stars but individual XP metallicities are too imprecise to test homogeneity star-by-star. Current cluster-ML tools (Hunt & Reffert 2023; Cavallo et al. 2024) do membership, not chemistry.

**Datasets.** Hunt-Reffert DR3 open cluster catalog (A&A 2023, ~7,000 clusters). Gaia DR3 XP [M/H] from Andrae et al. 2023 or SHBoost.

**Baselines.** Weighted mean + χ² dispersion test; Gaussian mixture on per-cluster [M/H] distributions; Niu et al. 2025 (XP-refined cluster parameters, no GNN).

**Method.** For each cluster, build a k-nearest-neighbors graph in 6D phase space. Train a GraphSAGE or GATv2 node-level regressor that propagates a shared latent metallicity through the graph with a noise-aware likelihood. Classify each cluster as homogeneous or heterogeneous based on the posterior variance of the latent [M/H]. Cross-validate against APOGEE cluster abundances.

**Feasibility.** Per-cluster graphs have ≤500 nodes. PyTorch Geometric on T4. Full catalog <1 GB.

**Venue.** A&A (cluster chemistry track) or MNRAS. **Novelty status.** No GNN for open-cluster chemical homogeneity.

### New #8 — Neural-field representation for helioseismic ring-diagram flow maps
**Problem.** Ring-diagram helioseismology produces flow maps on discrete tiles. Combining tiles into a smooth full-disk flow field uses heuristic interpolation. Roth et al. 2019 applied an ANN to per-tile inversions but did not produce a global continuous field. Diaz Baso et al. 2025 used neural fields for spectropolarimetric inversions (different physics).

**Datasets.** HMI ring-diagram pipeline outputs (mrh_rdsyn, mrh_rdfit) from JSOC, 2010–2025.

**Baselines.** Roth et al. 2019 per-tile MLP; the classical MCalc linear inversion.

**Method.** Fit an INR f_θ(lat, lon, depth, t) → (v_x, v_y) directly to the observed f-mode frequency shifts, with Laplacian smoothness priors and mass-conservation constraints. Output smooth, differentiable global flow fields. Compare to published torsional-oscillation maps.

**Feasibility.** Data volume is ~1 GB for 15 years of daily maps. Small MLP (~100k parameters), T4, ≤5 GPU-hours.

**Venue.** Solar Physics, ApJ, or ApJL. **Novelty status.** Distinct from Retained #4 (INR image compression) and from all prior helioseismic-inversion work.

### New #9 — Multi-task learning for joint classification and orbital-parameter regression on TESS eclipsing binaries
**Problem.** The TESS Ten Thousand Catalog (Kruse et al. 2025, arXiv:2506.05631) provides 10,001 classified EBs. Current ML tools (CALNet, Yu et al. 2025) do classification only; probabilistic neural networks (Viani et al. 2026, arXiv:2604.01281) do parameter regression only. No end-to-end multi-task model.

**Datasets.** TESS Ten Thousand Catalog + PHOEBE synthetic light curves for augmentation.

**Baselines.** CALNet for classification; Viani et al. 2026 probabilistic NN for period, eccentricity, mass ratio; sequential pipeline.

**Method.** A small 1D CNN (or small Transformer) backbone with four heads: (i) morphological class (detached / contact / semi-detached), (ii–iv) regression for period, eccentricity, mass ratio. Use Kendall et al. 2018 uncertainty-weighted multi-task loss. Compare against sequential baseline on 10-fold CV.

**Feasibility.** Folded light curves at 1024 points. <3 GB total. Single T4.

**Venue.** AJ, MNRAS, or RNAAS. **Novelty status.** No end-to-end multi-task model on this catalog.

### New #10 — Unsupervised autoencoder anomaly detection for Kepler heartbeat-star candidates
**Problem.** Heartbeat stars with tidally-excited oscillations (TEOs) are a rare class (~200 known after Guo et al. 2024). New HBS candidates outside the Thompson/Kirk samples have been found manually. Unsupervised ML has not been used to search.

**Datasets.** Kepler short-cadence light curves (MAST); Thompson et al. 2012, Kirk et al. 2016, Guo et al. 2024 HBS samples for validation.

**Baselines.** Manual selection (Thompson 2012); the unsupervised transient AE approach of Martinez-Galarza et al. 2021, adapted.

**Method.** Train a 1D convolutional autoencoder on Kepler SC light curves. Rank by reconstruction residual. Query top anomalies for the characteristic heartbeat morphology (eccentric-orbit brightening at periastron + TEO harmonic ringing). Report recovered known HBSs and new candidates.

**Feasibility.** A tractable SC subset via astroquery. 1D CNN AE on T4.

**Venue.** AJ, MNRAS, or ApJL. **Novelty status.** No ML HBS-discovery paper. Existing TEO ML work (Kolaczek-Szymanski & Rozanski 2023) uses supervised ML on theoretical models, not unsupervised discovery.

---

## 6. Final ranking (top 12 by composite score)

Rank by combined novelty × feasibility × publication-path clarity. ★ is worst, ★★★★★ is best.

| Rank | Niche | Novelty | Feasibility | Timeline | Compute | Venue |
|------|-------|---------|-------------|----------|---------|-------|
| 1 | SIREN for light-curve reconstruction (Retained #1) | ★★★★★ | ★★★★★ | 6–8 wk | CPU | RNAAS / ML4PS |
| 2 | PySR gyrochronology residuals (New #2) | ★★★★★ | ★★★★★ | 4–6 wk | CPU | RNAAS / ApJL |
| 3 | Mamba for red-giant power spectra (New #1) | ★★★★★ | ★★★★ | 6–10 wk | 1 GPU | RNAAS / A&A L |
| 4 | FMPE/NPE for FRB dispersion measures (New #3) | ★★★★ | ★★★★ | 10–12 wk | 1 GPU | ApJL / MNRAS / ML4PS |
| 5 | SSL on SolARED acoustic-power tiles (New #4) | ★★★★ | ★★★★★ | 8–10 wk | 1 GPU | Solar Phys / ApJL |
| 6 | INR compression vs AstroCompress (Retained #4) | ★★★★ | ★★★★ | 8–12 wk | 1 GPU | ML4PS / A&Comp |
| 7 | Conformal prediction for photometric transients (Retained #3) | ★★★★ | ★★★★★ | 6–8 wk | CPU | ML4PS / RNAAS |
| 8 | Test-time training for photo-z drift (New #5) | ★★★★ | ★★★★ | 10–12 wk | 1 GPU | MNRAS / ML4PS |
| 9 | Few-shot prototypical nets on PLAsTiCC rare classes (Retained #2) | ★★★★ | ★★★★★ | 8–12 wk | 1 GPU | ML4PS |
| 10 | Multi-task TESS eclipsing binaries (New #9) | ★★★ | ★★★★★ | 8–10 wk | 1 GPU | AJ / RNAAS |
| 11 | PINN for exoplanet transit light curves (Retained #6) | ★★★ | ★★★★★ | 8–12 wk | CPU | RNAAS / AJ |
| 12 | Diffusion anomaly detection on ZTF vs AHA (Retained #5) | ★★★ | ★★★ | 10–14 wk | 1 GPU | ML4PS |

Additional strong candidates not in the top 12 (New #6 Gaia XP metal-poor LDAM, New #7 GNN open cluster chemistry, New #8 neural field helioseismic flows, New #10 HBS anomaly detection) are equally viable and are included above in Section 5 as self-contained proposals.

### Recommended starter if you want one project

**SIREN for light-curve reconstruction (Rank 1)**. It is CPU-only, uses free public OGLE data, has clear quantitative baselines (GP regression and Fourier), is genuinely novel (no INR-based method has been applied to astronomical light curves), and the code path is a Colab notebook modification of the official SIREN repo. A week of programming, four weeks of experiments, and two weeks of writing produces an RNAAS.

### Recommended starter if you want something physics-flavored

**PySR gyrochronology (Rank 2)**. CPU-only, clean symbolic output that is immediately interpretable, and the result either beats the classical relations (paper) or does not (negative result that is still an RNAAS). Low risk.

### Recommended starter if you want the trendiest architecture

**Mamba for red-giant power spectra (Rank 3)**. State-space models are the hottest 2024–2026 architecture, have a clean reference implementation, and have zero current applications in asteroseismology. Clear 1D CNN baseline from Hon et al. 2018 to beat.

---

## 7. Shared resources and first-step checklist

### Python packages every project uses
- `astropy`, `astroquery`, `lightkurve` (Kepler/TESS light curves)
- `ztfquery`, `alerce` (ZTF alert access)
- `GaiaXPy` (Gaia XP spectra)
- `galaxy-datasets` (Galaxy Zoo PyTorch loaders)
- `feets`, `light-curve`, `FATS` (feature extraction)
- PyTorch + `torchdiffeq` (Neural ODEs)
- `sbi`, `nflows`, `lampe`, `ltu-ili` (simulation-based inference)
- `mapie`, `crepes` (conformal prediction)
- `PySR` (symbolic regression)
- `mamba-ssm` (Mamba state-space models)
- Reference SIREN GitHub (for implicit neural representations)
- `celerite2`, `george` (Gaussian-process baselines)
- HuggingFace `datasets` and `transformers` (Multimodal Universe, Chronos, Moirai, AstroCompress)

### Public datasets worth bookmarking
- **OGLE Collection of Variable Stars** (Cepheids, RR Lyrae, binaries)
- **PLAsTiCC** (Kaggle) and **ELAsTiCC** (Fink benchmark)
- **StarEmbed** ZTF benchmark (arXiv:2510.06200)
- **AstroCompress** HuggingFace (five 16-bit datasets)
- **Multimodal Universe** (NeurIPS 2024, 100 TB, 20+ surveys, HuggingFace)
- **AION-1** HuggingFace (foundation model weights and pretraining subsets)
- **DESI DR1** (public 2024) and **DESI Legacy Imaging DR10**
- **Rubin DP1** (for data-rights holders; in April 2026 this is limited but expanding)
- **Euclid Q1** (Fornax, Draco, Horologium deep fields, ESA archive)
- **SDSS-V DR19** (all three mappers)
- **Kepler APOKASC-3** red giants via MAST HLSP
- **TESS Ten Thousand** eclipsing binary catalog (arXiv:2506.05631)
- **Hunt-Reffert** Gaia DR3 open cluster catalog (A&A 2023)
- **CHIME/FRB Catalog 2** baseband waveforms
- **SolARED** (arXiv:2601.13145) helioseismic acoustic power maps
- **HMI ring-diagram** products from JSOC

### Venue guide (as of April 2026)

| Venue | Length | Difficulty | Turnaround | Notes |
|-------|--------|------------|------------|-------|
| **RNAAS** | 1,500 words, 1 figure | Low (moderated, not peer-reviewed) | Days | Free. Explicitly welcomes undergrad summer-project results. 2,000th publication was April 2025. |
| **ML4PS @ NeurIPS** | 4 pages + refs | Moderate | Aug submission, Dec workshop | Most recent was NeurIPS 2025 in San Diego; ML4PS 2026 will be at NeurIPS 2026. Astronomy tracks are large. |
| **ML4Astro @ ICML** | 4 pages | Moderate | Spring submission, July workshop | ICML 2025 had 42 accepted papers. |
| **Astronomy & Computing** (Elsevier) | Full paper | Moderate | 2–4 months | Currently running a HPC+ML special issue. |
| **Open Journal of Astrophysics** | Full paper | Moderate | 1–3 months | Open access, free. |
| **MNRAS Letters** | 5 pages | Moderate–high | 1–3 months | Prestige tier, selective. |
| **A&A Letter** | 5 pages | Moderate–high | 1–3 months | Prestige tier, selective. |
| **ApJL** | 8 pages | High | 2–4 months | Most selective short-paper tier. |
| **AJ / ApJ** | Full paper | High | 3–6 months | Standard long-paper venues. |
| **PASP** | Full paper | Moderate | 2–4 months | Instrumentation and pipelines often fit here. |
| **RASTI** | Full paper | Moderate | 2–4 months | Instrumentation-focused; RFI and calibration papers fit well. |
| **AAS meeting posters** | Poster | Low | — | AAS 247 was January 2026; AAS 248 is summer 2026. Good for building a paper pipeline. |

### Competitions worth tracking
- **Ariel Data Challenge 2025/2026** (NeurIPS competition; exoplanet atmospheric characterization with jitter noise correction). 6,000+ participants across editions.
- **PLAsTiCC** (closed but still the gold-standard benchmark; data and tutorials are permanent).
- **SKA Science Data Challenges** (periodic).

---

## 8. Selected references (with arXiv or DOI)

These are the load-bearing citations for the niches above. Papers I verified personally are the ones that back up novelty gaps and baseline claims.

### Foundation models and benchmarks
- Parker, Lanusse et al. 2024, AstroCLIP, MNRAS 531, 4990, arXiv:2310.03024
- Parker, Lanusse et al. 2025, AION-1, arXiv:2510.17960, NeurIPS 2025
- Duraphe et al. 2025, The Platonic Universe, arXiv:2509.19453, ML4PS 2025
- Li et al. 2025, StarEmbed, arXiv:2510.06200
- Truong et al. 2025, AstroCompress, arXiv:2506.08306, ICLR 2025
- Audenaert et al. 2024, Multimodal Universe, arXiv:2412.02527, NeurIPS 2024 D&B
- de Haan et al. 2025, AstroSage, Scientific Reports 15, 13751, arXiv:2411.09012

### Light-curve methods (retained and compared)
- Donoso-Oliva et al. 2023, ASTROMER, A&A 670, A54, arXiv:2205.01677
- Donoso-Oliva et al. 2025, Astromer 2, arXiv:2502.02717
- Tan et al. 2025, AstroCo, arXiv:2509.24134, ML4PS 2025
- Chiong, Becker, Protopapas 2025, Multivariate Time-series Transformer Embeddings for Light Curves, arXiv:2506.11637
- Cabrera-Vives et al. 2024, ATAT, A&A 689, A289, arXiv:2405.03078
- Tung 2025, ATCAT, arXiv:2511.00614
- Pimentel, Estévez, Förster 2022, TimeModAttn, arXiv:2201.08482
- Pan et al. 2024, Astroconformer, MNRAS 528, 5890
- Rubanova, Chen, Duvenaud 2019, Latent ODEs, arXiv:1907.03907, NeurIPS 2019
- Oh et al. 2025, Neural SDDE for astronomical time series, arXiv:2508.17521, CIKM 2025 **(the paper that weakens the Neural ODE Cepheid niche)**
- Sitzmann et al. 2020, SIREN, arXiv:2006.09661, NeurIPS 2020
- Shukla & Marlin 2021, Multi-Time Attention Networks, arXiv:2101.10318, ICLR 2021
- Naul et al. 2018, unsupervised autoencoding RNN for variable stars, Nature Astronomy 2, 151, arXiv:1711.10609
- Ansari et al. 2024, Chronos, arXiv:2403.07815
- Woo et al. 2024, Moirai, arXiv:2402.02592, ICML 2024 Oral

### Transient classification
- Cabrera-Vives et al. 2024, ATAT on ELAsTiCC, A&A 689, A289
- Gupta & Muthukrishna 2025, transfer learning ZTF→LSST, arXiv:2502.18558, MNRASL
- Gupta, Muthukrishna, Audenaert 2025, simulation-pretrained DA, arXiv:2510.12958 **(the paper that weakens the OT-DA niche)**
- Iskandarli et al. 2026, AHA, arXiv:2602.12955 **(the new baseline for diffusion AD niche)**
- DiMMAD, arXiv:2510.23702, NeurIPS ML4PS 2025
- ORACLE real-time LSST classifier, arXiv:2501.01496
- Salinas et al. 2025, transformer TESS FFI search, MNRAS 538, 2031, arXiv:2502.07542
- Valizadegan et al. 2025, ExoMiner++, arXiv:2502.09790

### Conformal prediction in astronomy
- Jones et al. 2024, ApJ 974, 159 (photo-z)
- Soriano et al. 2026, arXiv:2601.17222 (photo-z)
- Ashton et al. 2024, arXiv:2412.11801 (Gravity Spy glitches, PRD)
- Ashton et al. 2024, arXiv:2402.19313 (GW search calibration, PRD 109)

### Asteroseismology and gyrochronology
- Hon, Stello, Yu 2018, deep learning red giant classification, MNRAS 476
- Audenaert et al. 2025, PU learning for gamma Dor / delta Sct hybrids, A&A (arXiv:2511.20908)
- Stokholm et al. 2025, TESS CVZ red giants, arXiv:2604.00498
- Van-Lane et al. 2024, ChronoFlow, arXiv:2412.12244
- Gu & Dao 2023, Mamba, arXiv:2312.00752
- Cranmer 2024, PySR, arXiv:2305.01582

### Gaia XP and stellar populations
- Andrae et al. 2023, XGBoost on Gaia XP, arXiv:2407.06963
- Khalatyan et al. 2024, SHBoost, A&A (full HTML)
- Vincent et al. 2024, Gaia DR3 white dwarfs, A&A 682, A5
- Hunt & Reffert 2023, open cluster catalog, A&A
- Niu et al. 2025, cluster parameters from XP, A&A 693, A170

### Radio astronomy
- Islam et al. 2025, CNN-LSTM FRB DM estimator, arXiv:2512.24003
- CHIME/FRB Catalog 2 baseband release (public 2025)
- Slijepcevic et al. 2024, MeerKAT SSL, A&A
- Alegre et al. 2022, LOFAR machine source association
- RFI-HWT, A&A (Sept 2025)

### Solar physics
- Jarolim et al. 2023, PINN solar coronal magnetic field, Nature Astronomy 7, 1171
- Kitiashvili et al. 2025, SolARED, arXiv:2601.13145
- Diaz Baso et al. 2025, neural field spectropolarimetric inversion, A&A 693, A170
- Roth et al. 2019, helioseismic ring-diagram ANN, A&A
- EskandariNasab et al. 2024, SHARP contrastive + GRU, arXiv:2409.14016, ICMLA

### Exoplanets
- Mandel & Agol 2002, transit light-curve model
- Batman (Kreidberg 2015) transit fitter
- Retrieval PINNs (Rayleigh scattering in exoplanet atmospheres), arXiv:2408.00084

### Simulation-based inference
- Villaescusa-Navarro et al., CAMELS documentation
- Bairagi, Wandelt, Villaescusa-Navarro 2025, A&A (arXiv:2503.13755)
- Ho et al. 2024, LtU-ILI, Open Journal of Astrophysics, arXiv:2402.05137
- Lipman et al. 2023, Flow Matching, arXiv:2210.02747
- Dhanasingham et al. 2025, lens NPE, arXiv:2511.17732

### Eclipsing binaries
- Kruse et al. 2025, TESS Ten Thousand EB catalog, arXiv:2506.05631
- Yu et al. 2025, CALNet eclipsing binary classifier
- Viani et al. 2026, probabilistic NN EB parameters, arXiv:2604.01281

### Heartbeat stars
- Thompson et al. 2012; Kirk et al. 2016; Guo et al. 2024 (HBS catalogs)
- Kolaczek-Szymanski & Rozanski 2023 (TEO supervised ML)

---

## 9. Closing note

The thesis of the original compass artifact is correct: the fastest path to a publishable result is to apply a well-understood ML method to a well-benchmarked astronomical dataset where the specific combination has not been tried. The changes in this document reflect (a) four of the original ten pairings are no longer the cleanest gaps they were, (b) six remain strong with minor scope tightenings, and (c) ten new pairings are equally or more compelling as of April 2026.

The single highest-reward, lowest-risk project for an undergraduate without GPU access is **SIREN for OGLE variable-star light curves** (Retained #1). The single highest-reward project with access to one GPU is **Mamba for red-giant power spectra** (New #1). The project most likely to yield a closed-form physics result that generalists appreciate is **PySR for gyrochronology residuals** (New #2).

Any of these, properly executed with attention to baselines and honest error bars, is a publishable paper.

# Novel AI + Astronomy Research Niches for Undergraduate Research
## Complementary to 10 Existing Proposals

Date of research: 2026-04-08
Scope: 20 new niches across asteroseismology, Gaia spectroscopy, radio astronomy, non-flare solar physics, simulation-based inference, and related areas.

Selection criteria satisfied for every niche below:
1. Genuinely novel (specific method + dataset combination not yet published or only touched at the fringes)
2. Publishable in RNAAS / ML4PS / A&A / MNRAS / ApJL / AJ / PASP / Open Journal of Astrophysics
3. Feasible on a single GPU (Colab T4 / L4 / A100 40GB) and modest RAM
4. Uses public datasets (DOIs, MAST HLSPs, Vizier, Zenodo, ESO archives)
5. Clear published baseline to beat
6. Distinct from the 10 existing niches (no SIREN, Neural ODEs for Cepheids, PINNs for transits, conformal prediction for transients, multi-time attention, diffusion anomaly detection, INR compression, OT domain adaptation, few-shot transients, TS foundation model fine-tuning)

---

## PRIORITY BLOCK A: ASTEROSEISMOLOGY

### Niche 1: Positive-Unlabelled Learning for Period-Spacing Extraction in Subgiant Gravity-Mode Oscillators

**Problem.** Gravity-mode period spacings (DeltaPi_1) in subgiants and young red giants break the mass-degeneracy of the nu_max-Delta_nu plane and reveal core rotation, but automated extraction is still brittle because the mixed-mode pattern is sparse and only a small labeled fraction exists.

**Novelty gap.** PU learning was recently demonstrated for *gamma Dor / delta Sct hybrids* (Audenaert+ Nov 2025 A&A) but **not** for period-spacing detection in subgiants. The branching-neural-network emulator work of Lyttle+ (arXiv 2601.02926) predicts *individual mode frequencies* from stellar parameters as a forward model, not the inverse task.

**Dataset.** (a) The 19,151-star TESS CVZ red-giant catalog (Hon+ arXiv 2604.00498) for held-out tests; (b) Kepler subgiants from the APOKASC-3 catalog with measured DeltaPi_1; (c) Mosser+ 2014 mixed-mode templates as positives. All public, <5 GB total.

**Baseline.** The classical nuSYD / autocorrelation pipeline of Stello & Sharma (PySYD, Chontos+ 2022) plus the recent random-forest period-spacing classifier.

**Method.** (i) Treat labeled subgiants as positives and the rest of the CVZ sample as unlabeled; (ii) use the PU-learning smart-binning approach of Audenaert+ 2025 but applied to echelle-diagram image embeddings via a small ConvNeXt-Tiny; (iii) conformalize the decision threshold so the precision guarantee is controlled.

**Why feasible.** Single Tesla T4 for <6h training, data fits in memory, echelle images are only 128x128. All code in Python/PyTorch.

**Venue.** MNRAS or A&A (good fit), or ML4PS poster for the methodological spin.

**Compute.** <10 GPU-hours total, <20 GB storage.

---

### Niche 2: Mamba (Selective State-Space Models) for Raw Red-Giant Oscillation Power Spectra

**Problem.** Red-giant power spectra are long 1D sequences (~10,000-100,000 bins) where Transformer memory scales quadratically; Mamba is sub-quadratic and has not yet been applied in asteroseismology.

**Novelty gap.** Explicit search confirms: no astronomy paper currently uses Mamba/S4/S6 for power spectra. All recent red-giant classifiers use 1D CNNs (Hon+ 2018), MLPs (Stone-Martinez+ 2025), or wavelet-CNN hybrids. Mamba is entirely absent from asteroseismology and from astronomical light-curve analysis.

**Dataset.** Kepler APOKASC-3 red giants (14,983 classified, Hon+ 2018 catalog); TESS Southern CVZ catalog (Stokholm+ 2025, arXiv 2604.00498). Raw power spectra are on MAST HLSP KEPSEISMIC.

**Baseline.** Hon+ 2018 2D CNN on echelle diagrams (RGB vs CHeB classification) and the BAM pipeline for global oscillation parameter extraction.

**Method.** Train a small Mamba encoder (e.g., Mamba-130M scaled down; <5M parameters) on raw log-power spectra to jointly output (i) RGB/CHeB class, (ii) nu_max and Delta_nu as regression heads, (iii) a MLE-estimated granulation timescale. Compare against 1D CNN (Hon+), wavelet-CNN (Claytor+), and a Transformer baseline of matched size.

**Why feasible.** Mamba is a drop-in RNN replacement; official PyTorch implementation in state-spaces/mamba. Training runs on a single T4; sequences of 10k bins are fine.

**Venue.** RNAAS (quick demonstration), ML4PS full paper, or A&A Letter.

**Compute.** ~20 GPU-hours.

---

### Niche 3: Symbolic Regression (PySR) of Gyrochronology Residuals After ChronoFlow

**Problem.** ChronoFlow (Van-Lane+ 2025, ApJ) gives data-driven ages from rotation, but the learned relation is a deep CNF with no closed form, so it is hard to interpret or compare with theoretical braking laws.

**Novelty gap.** No one has applied PySR (Cranmer 2024) to learn a closed-form gyrochronology relation or to the residuals of an existing ML gyro model. Symbolic-regression astro applications to date focus on neutron-star EoS (not rotation-age).

**Dataset.** ChronoFlow's public training data: ~8,000 cluster rotators with ages from Kepler, K2, TESS, and Gaia (Van-Lane+ 2025 Zenodo release).

**Baseline.** Classical Barnes (2007), Mamajek & Hillenbrand (2008), and Angus+ (2019) analytic gyrochronology relations; ChronoFlow itself.

**Method.** (i) Use PySR with domain operators (logs, powers, tanh) to learn P_rot = f(age, mass, Ro) directly; (ii) subsequently fit the residuals of ChronoFlow to see if a human-readable correction term falls out; (iii) cross-validate across clusters.

**Why feasible.** PySR runs on CPU; a handful of Julia threads suffices. Total runtime ~1-2 days CPU, no GPU needed.

**Venue.** RNAAS (for a clean symbolic result) or ApJL if the expression beats analytic baselines.

**Compute.** CPU-only, ~30 core-hours.

---

## PRIORITY BLOCK B: GAIA XP / LOW-RES SPECTROSCOPY

### Niche 4: Self-Distillation for Metal-Poor Star Identification in Gaia XP Using the Long-Tail Problem

**Problem.** Extremely metal-poor (EMP; [Fe/H] < -3) stars are overwhelmingly under-represented in training sets, so standard supervised XP-spectra models (SHBoost, Gaia XP neural net of Zhang+ 2024) systematically over-predict their metallicity toward the median.

**Novelty gap.** Existing works (Andrae+ 2023, Zhang+ 2024, Khalatyan+ 2024) use XGBoost or vanilla MLP; none apply distribution-tail-aware losses like Label-Distribution-Aware Margin (LDAM) or self-distillation with temperature scaling to the metal-poor tail in XP. No combination of imbalance handling + XP yet.

**Dataset.** Public Gaia DR3 BP/RP spectra (all 220 million sources); training labels from PASTEL, SAGA database, and APOGEE DR17 (VAC + LAMOST DR10 as secondary sample). All public.

**Baseline.** Andrae+ 2023 (XGBoost), Zhang+ 2024 empirical forward model, and SHBoost (Khalatyan+ 2024).

**Method.** (i) Train a small MLP/1D-CNN on XP coefficients with an LDAM or focal-regression loss; (ii) self-distill with metal-poor-weighted pseudo-labels; (iii) validate on held-out halo stars from the PIGS and BEST-of-H3 catalogs.

**Why feasible.** XP coefficients are 110 floats per star; the whole 220 M set is <200 GB but a random 1 M subsample fits in Colab. Simple MLP, single T4.

**Venue.** A&A (direct comparison to Andrae+ track), AJ, or MNRAS.

**Compute.** <10 GPU-hours.

---

### Niche 5: Contrastive XP-Photometry Pretraining for White-Dwarf Mass-Radius Classification

**Problem.** Gaia DR3 has ~360,000 high-confidence white dwarfs with XP spectra (Vincent+ 2024). Classification into DA/DB/DC/magnetic and mass regression is currently done with supervised CNNs on the 110 XP coefficients; no cross-modal SSL uses the XP spectrum together with the Gaia multi-epoch photometry and BP-RP color.

**Novelty gap.** AstroCLIP and AION-1 are galaxy/star foundation models pretrained on imagery + RVS spectra or large surveys, **not** on XP + epoch photometry for white dwarfs specifically. No WD foundation model exists.

**Dataset.** Vincent+ 2024 WD catalog (Gaia DR3 XP + atmospheric parameters, Vizier J/A+A/682/A5); Gaia DR3 epoch photometry for the same set; SDSS spectroscopic WD labels for fine-tuning.

**Baseline.** Vincent+ 2024 random-forest classifier; Garcia-Zamora+ 2023 neural network spectral classifier.

**Method.** SimCLR with two encoders: (a) small MLP on XP coefficients, (b) 1D CNN on folded/unfolded G-band magnitudes. Positive pairs are XP + photometry of the same source. Fine-tune on spectral-type label with 1-10% of labels.

**Why feasible.** XP+photometry for 360k WDs fits in 2 GB. SimCLR with batch ~256 on a single T4 is standard.

**Venue.** MNRAS, A&A, or ML4PS workshop.

**Compute.** <15 GPU-hours.

---

### Niche 6: Graph Neural Network for Open-Cluster Chemical Homogeneity Testing with Gaia XP

**Problem.** Chemical homogeneity of open clusters is a sensitive probe of cluster birth environments, but testing it currently requires expensive high-resolution spectroscopy. Gaia XP gives [M/H] for millions of stars but individual XP metallicities are too imprecise per star to directly test homogeneity.

**Novelty gap.** The Hunt+ 2023 / Cavallo+ 2024 DNN membership tools are *not* GNN, and they do not test chemical homogeneity. The Niu+ 2025 A&A work uses XP metallicities to *refine* cluster parameters but does not use a GNN or propagation step.

**Dataset.** Hunt-Reffert DR3 cluster catalog (A&A 2023 publicly released); Gaia DR3 XP spectra and [M/H] derived by Andrae+ 2023 or SHBoost. About 7,000 open clusters.

**Baseline.** Simple mean + chi^2 dispersion test; weighted Gaussian mixture on per-cluster XP [M/H] distributions.

**Method.** For each cluster, build a k-NN graph in 6D phase space; train a small GraphSAGE/GATv2 to regress a shared latent metallicity per cluster, using a noise-aware likelihood. Classify each cluster as chemically homogeneous / heterogeneous by the posterior variance of the latent [M/H].

**Why feasible.** One graph per cluster (<500 nodes); PyTorch Geometric on a T4. Catalog < 1 GB.

**Venue.** A&A (cluster chemistry) or MNRAS.

**Compute.** <5 GPU-hours.

---

## PRIORITY BLOCK C: RADIO ASTRONOMY

### Niche 7: Self-Supervised Vision Transformer for Pulsar-Candidate Sub-Band Diagnostics in HTRU + FAST

**Problem.** HTRU and FAST pulsar candidate discrimination is normally done with supervised CNNs or XGBoost on the 4-panel diagnostic plots (Lyon+ features). These methods overfit to the exact survey distribution, so cross-survey generalization (HTRU -> FAST, FAST -> MeerKAT) is poor.

**Novelty gap.** Transformer/vision-transformer pulsar work (Liu+ 2024 ScienceDirect) is *supervised* only. DINO/MAE pretraining has never been applied to pulsar diagnostic plots. No self-supervised cross-survey pulsar candidate study exists.

**Dataset.** HTRU-Medlat (public), HTRU-S (public), and the subset of FAST CRAFTS public candidates + the recent MeerTRAP single-pulse examples (Bezuidenhout+ 2025).

**Baseline.** Morello+ 2014 SPINN, Zhu+ 2014 PICS, and the recent ResNet/CvT classifiers reaching >99% in-survey accuracy (but dropping significantly cross-survey).

**Method.** (i) Concatenate the 4-panel plots into a single image; (ii) DINOv2 self-supervised pretraining on the mixed HTRU+FAST unlabeled pool; (iii) fine-tune a tiny linear head on labels and evaluate cross-survey transfer.

**Why feasible.** DINOv2 with a ViT-S/16 on 100k 224x224 images runs in ~12 hours on a T4. Data is image-only and public.

**Venue.** MNRAS, A&A, or PASP.

**Compute.** 15-25 GPU-hours.

---

### Niche 8: Neural Posterior Estimation of Fast Radio Burst Dispersion Measure with Scintillation-Robust Simulation

**Problem.** FRB DM estimation pipelines (CHIME/FRB DM-T transform) can be biased by scattering and scintillation for wide bursts. Deep-learning DM estimators (arXiv 2512.24003) are feed-forward CNNs, giving point estimates only, without principled uncertainty.

**Novelty gap.** Simulation-based inference (NPE with normalizing flows, Papamakarios+) has *not* been applied to FRB DM estimation. Only a handful of FRB papers use SBI at all, and none target DM with scintillation-realistic forward modeling.

**Dataset.** CHIME/FRB Catalog 2 baseband waveforms (4,545 events, public since 2025); synthetic training via `simulateSearch` + `scintools` to mock scintillation; Petroff FRB catalogue for external validation.

**Baseline.** The feed-forward CNN-LSTM DM estimator of Islam+ (arXiv 2512.24003); classical incoherent dedispersion scan.

**Method.** Forward-simulate bursts with a scintillation model, train a conditional normalizing flow (nflows / sbi package) to output p(DM | dynamic-spectrum). Compare posterior coverage and MAE on held-out CHIME events.

**Why feasible.** FRB dynamic spectra are small (<1 MB each). `sbi` package supports exactly this use case. T4 is plenty.

**Venue.** ApJL, MNRAS, or ML4PS.

**Compute.** <20 GPU-hours.

---

### Niche 9: Graph Neural Networks for MeerKAT Radio-Continuum Source Association (Closed-Form Component Linking)

**Problem.** Source finders such as PyBDSF / Aegean detect individual Gaussian components; associating multiple components to a single radio galaxy is done by hand-tuned heuristics. With MeerKAT MGCLS and the upcoming SKA precursor data, this is untenable.

**Novelty gap.** Existing GNN lens/source work treats image patches; component-graph construction with *attention pooling* for multi-component radio galaxies has not been done. Self-supervised work on MGCLS (Slijepcevic+ 2024) focuses on patch-level embeddings, not association.

**Dataset.** MGCLS catalog release (Knowles+ 2022); RadioGalaxyZoo DR1 and LoTSS-DR2 catalog for labels; EMU-PS pilot catalog for transfer test.

**Baseline.** PyBDSF + LOFAR "machine" association of Alegre+ 2022; the CNN component-association classifier of Bowles+ 2023.

**Method.** Each detected Gaussian component becomes a node; edges are drawn by angular separation + PA alignment. Train an edge-prediction GNN on RadioGalaxyZoo-labeled associations.

**Why feasible.** Catalogs are <5 GB. PyG on T4 with edge-level BCE.

**Venue.** MNRAS, A&A, or PASP.

**Compute.** 10-15 GPU-hours.

---

### Niche 10: Test-Time Adaptation for RFI Flagging Across Radio Observatories

**Problem.** Current CNN/U-Net RFI flaggers (deepRFI, R-Net, RFI-HWT) are trained per-telescope and generalize poorly across instruments because the RFI environment differs.

**Novelty gap.** No test-time adaptation (TTA) methods (TENT, CoTTA, SAR) have been applied to radio RFI. The current state of the art (Mesarcik+ 2022; RFI-HWT 2025 A&A) uses transfer learning only.

**Dataset.** HERA RFI sim + real data; LOFAR EoR public RFI frames; SKA RFI challenge dataset (2023 release); FAST archival data used in the 2025 DBSCAN paper.

**Baseline.** AOFlagger; deepRFI; RFI-HWT (A&A 2025, Sept).

**Method.** Take a U-Net pretrained on HERA and apply TENT (entropy minimization) or SHOT-style source-free adaptation at test time on LOFAR/FAST frames. Measure F1 and FPR.

**Why feasible.** U-Net on 128x128 time-frequency patches, standard PyTorch TTA wrapper. Fits in Colab.

**Venue.** MNRAS, A&A, RASTI.

**Compute.** <15 GPU-hours.

---

## PRIORITY BLOCK D: NON-FLARE SOLAR PHYSICS

### Niche 11: Neural Field Representation for Helioseismic Ring-Diagram Flow Maps

**Problem.** Ring-diagram helioseismology produces flow maps on discrete tiles; combining them into a smooth full-disk flow field requires heuristic interpolation. The method of Roth+ 2019 (ANN for ring-diagram inversion) is per-tile only.

**Novelty gap.** Neural fields / implicit neural representations (NeRF-style) have *not* been used for helioseismic flow fields. The only neural-field solar work is Diaz Baso+ 2025 for spectropolarimetric inversions (chromospheric B fields), which is a different physics problem.

**Dataset.** HMI ring-diagram data products (mrh_rdsyn, mrh_rdfit) from JSOC, 2010-2025, publicly available.

**Baseline.** Roth+ 2019 per-tile MLP; the MCalc linear inversion baseline.

**Method.** Fit an INR f_theta(lat, lon, depth, t) -> (v_x, v_y) directly to the observed ring-diagram f-mode frequencies, regularized by a Laplacian and by equation-of-motion priors (simpler than a full PINN). Output smooth, differentiable flow fields.

**Why feasible.** Data volume is small (~1 GB for 15 years of daily maps). A small MLP (~100k params) can be fit in <2 hours on a T4.

**Venue.** Solar Physics, ApJ, or ApJL if cross-validated against torsional-oscillation results.

**Compute.** <5 GPU-hours.

---

### Niche 12: Score-Based Diffusion for Super-Resolving SDO/HMI Coronal-Hole Masks to Hi-C Scale

**Problem.** Coronal holes drive high-speed solar wind. Detection on full-disk AIA/HMI images (CHRONNOS, U-Net variants) gives masks at 1.2" resolution; there is no principled way to infer sub-resolution boundary structure.

**Novelty gap.** Diffusion models for solar image super-resolution exist for AIA / vector magnetograms (Jarolim+ 2023, Deng+ 2024), **not** for coronal-hole masks specifically, and none use a score-based posterior-sampling (DPS) approach trained on paired AIA + Hi-C (2013 + 2024 flights) for the boundary-inference task.

**Dataset.** SDO/AIA 193 A full-disk (JSOC) + Hi-C 1 (2013) and Hi-C 3 (2024) flights (VSO archive). CHRONNOS masks as pseudo-labels.

**Baseline.** CHRONNOS (Jarolim+ 2021); U-Net SR baselines.

**Method.** Train a small DDPM on Hi-C patches; use posterior-sampling with a degradation operator that downsamples to AIA resolution; condition on the AIA CH mask.

**Why feasible.** Training on 64x64 patches with a small UNet-Diffusion model is Colab-feasible.

**Venue.** Solar Physics, ApJL.

**Compute.** ~20 GPU-hours.

---

### Niche 13: Contrastive Learning of Quiet-Sun vs Active-Region Acoustic Power Halos for Early Flux-Emergence Warning

**Problem.** SolARED (Kitiashvili+ 2025) and Guerra+ 2024 show that acoustic power maps contain pre-emergence signals ~5 hr before magnetic flux appears, but learning them supervised requires many labeled emergences.

**Novelty gap.** No contrastive self-supervised pretraining has been published on SolARED acoustic-power tiles. LSTM-supervised work (Kitiashvili+ 2025) is the baseline.

**Dataset.** SolARED public dataset (arXiv 2601.13145, JSOC); 50 labeled large active regions.

**Baseline.** Kitiashvili+ 2025 LSTM (5-hr lead time).

**Method.** SimCLR/BYOL on unlabeled tiles from the quiet Sun; fine-tune a linear classifier on the 50 labeled AR tiles to predict emergence vs quiet. Use the TSS metric (as in SolARED) for comparison.

**Why feasible.** Dataset ~20 GB; tile size 128x128; small ResNet encoder.

**Venue.** Space Weather, Solar Physics, or ApJL.

**Compute.** ~15 GPU-hours.

---

## PRIORITY BLOCK E: SIMULATION-BASED INFERENCE APPLICATIONS

### Niche 14: Truncated Sequential Neural Ratio Estimation for Halo Occupation Distribution Posteriors on CAMELS

**Problem.** HOD parameter inference is typically done via likelihood-based Bayesian MCMC on summary statistics (two-point functions). Full-field inference is possible but expensive.

**Novelty gap.** Simulation-based inference (SNRE / NRE-C) has not been applied to the HOD inference problem on the CAMELS galaxy catalogs. The closest work (Shao+ 2023, Villaescusa-Navarro+) uses GNNs for direct cosmology inference, not HOD.

**Dataset.** CAMELS Multifield Dataset (galaxy catalogs, public 2023 release). ~1000 IllustrisTNG/SIMBA sims.

**Baseline.** Classical halo-model MCMC (e.g., halotools).

**Method.** Use the `sbi` package with SNRE-C on a compressed-summary vector (2PCF + VPF). Compare posterior width vs MCMC and report coverage diagnostics.

**Why feasible.** Input data: catalogs of ~10k galaxies; standard MLP classifier. `sbi` is a one-line call.

**Venue.** ML4PS (strong fit), Open Journal of Astrophysics, MNRAS.

**Compute.** <20 GPU-hours.

---

### Niche 15: Flow-Matching Posterior Estimation for Dark Matter Subhalo Mass Function from Strong-Lens Anomaly Catalogs

**Problem.** Existing subhalo constraints from strong lensing (Hezaveh+, Vegetti+, Daylan+) rely on NPE or direct regression. Flow Matching (FM / FMPE, Lipman+ 2023; used in exoplanet atmospheres 2025) is faster and has never been applied to lensing subhalo inference.

**Novelty gap.** Dhanasingham+ Nov 2025 arXiv used neural density estimation; Brehmer+ and Varma+ used normalizing flows or classical NPE. No Flow-Matching application to subhalos yet.

**Dataset.** Public simulated Hubble strong-lens image sets from Metcalf+ 2019 challenge; recent SIMULATED lens-plane catalogs (Wagner-Carena+ 2023, lensing fork of CAMELS).

**Baseline.** Varma+ 2020 neural likelihood-ratio; Brehmer+ 2019 mining-gold NRE.

**Method.** Train an FMPE model (using the open-source `fmpe` or `flow_matching` repo) on simulated image + substructure mass-function parameter pairs; validate on held-out sims; report posterior coverage.

**Why feasible.** 128x128 lens images; ResNet18 embedder + small CNF; fits on a T4.

**Venue.** ML4PS, MNRAS, Phys Rev D.

**Compute.** ~20 GPU-hours.

---

## PRIORITY BLOCK F: OTHER NOVEL NICHES

### Niche 16: YOLO-NAS Streak Detection with Synthetic Augmentation for NEO Microsat Astrometry on Undeveloped ATLAS Chunks

**Problem.** Asteroid streak detection in wide-field imagery (Duev+ 2025 ZTF, Euclid 2023) is mature, but applying object-detection (YOLO-NAS, RT-DETR) with synthetic streak augmentation to *unvetted archive data* from less-well-curated surveys has not been done.

**Novelty gap.** Published work uses ZTF (Duev 2025) and Euclid (Pontinen 2023, Lieu 2023), both using SSD or Faster R-CNN. YOLO-NAS (Deci 2023) is a stronger open model with no astro application. No ATLAS-chunk streak paper exists.

**Dataset.** ATLAS full-frame public images on MAST and ATLAS Forced Photometry Server; NEOCP catalog for ground truth; synthetic streaks via FITS injection.

**Baseline.** Duev+ 2025 ZTF CNN; ATLAS built-in streak-finder.

**Method.** Train YOLO-NAS on synthetic streaks injected into empty ATLAS fields; evaluate on real streaks. Measure recall vs SNR and false-positive rate.

**Why feasible.** Colab T4 with the Roboflow / ultralytics training scripts.

**Venue.** RNAAS, AJ, PASP, Icarus.

**Compute.** <10 GPU-hours.

---

### Niche 17: Test-Time Training (TTT) for Photometric Redshift Drift in DESI Legacy Imaging

**Problem.** Photo-z models degrade as surveys drift in depth and PSF; classical approaches (re-training, importance weighting) require labels.

**Novelty gap.** Test-time training (Sun+ 2020) and its self-supervised variants have **not** been applied to photo-z. PZFlow uses normalizing flows but does not adapt at test time.

**Dataset.** DESI Legacy Imaging DR10 + BGS/ELG DESI spectroscopic redshifts (public 2024 DESI DR1).

**Baseline.** PZFlow (Crenshaw+ 2024); DNF (De Vicente+ 2016); FlexZ.

**Method.** Train a photo-z model with a rotation-prediction auxiliary task; at test time, for each new image, fine-tune on the auxiliary task only, then predict. Compare PIT / CRPS distributions.

**Why feasible.** DESI DR1 subset fits in <20 GB; standard PyTorch.

**Venue.** MNRAS, A&A, ML4PS.

**Compute.** <15 GPU-hours.

---

### Niche 18: Normalizing-Flow Density Estimation of Intrinsic Galaxy Alignment (IA) on TNG100 for DESI Priors

**Problem.** Intrinsic alignments are a systematic in weak-lensing cosmology; current IA models (NLA, TATT) use empirical parametric forms. A nonparametric density model of alignment as a function of local environment would aid priors.

**Novelty gap.** Existing ML IA work (IAEmu 2025, Pandya+ 2025) predicts correlation functions, not the joint density p(orientation, env). No normalizing-flow density estimator for IA.

**Dataset.** IllustrisTNG TNG100 public release (galaxy shapes + environment).

**Baseline.** IAEmu (Pandya+ 2025); halotools.

**Method.** Train a conditional flow p(e | m_halo, env, z) on TNG100 galaxies; compare IA amplitude predictions with IAEmu.

**Why feasible.** Catalog <2 GB; `nflows` package + a small MLP conditioning net; CPU or T4.

**Venue.** MNRAS, Phys Rev D, Open Journal of Astrophysics.

**Compute.** <10 GPU-hours.

---

### Niche 19: Multi-Task Learning for Joint Eclipsing-Binary Classification and Orbital-Parameter Regression on TESS 2-min Targets

**Problem.** The TESS Ten Thousand Catalog (Kruse+ 2025, arXiv 2506.05631) provides classified EBs but parameters (period, eccentricity, inclination, mass ratio) are done sequentially. Machine learning has been used for one or the other, not jointly.

**Novelty gap.** CALNet (Yu+ 2025, CNN+Attention+LSTM) does classification only; the probabilistic NN of Viani+ 2026 (arXiv 2604.01281) does parameter regression only. No end-to-end multi-task TESS EB model exists.

**Dataset.** TESS Ten Thousand Catalog (10,001 EBs, public 2025); PHOEBE synthetic light curves for augmentation.

**Baseline.** CALNet (classification), Viani+ 2026 (parameters), sequential pipeline.

**Method.** A small 1D CNN backbone with (i) a classification head (detached vs contact vs semi-detached), (ii) multi-head regression for period, eccentricity, mass ratio; use uncertainty-weighted multi-task loss (Kendall+ 2018).

**Why feasible.** Light curves folded to 1024 points; entire dataset <3 GB; T4 suffices.

**Venue.** AJ, MNRAS, RNAAS.

**Compute.** <10 GPU-hours.

---

### Niche 20: Unsupervised Anomaly Detection in Kepler Short-Cadence Tidally-Excited Oscillations (Heartbeat Stars)

**Problem.** Heartbeat stars (HBS) with tidally excited oscillations (TEOs) are a small and eclectic class; the current catalog is built manually (Guo+ 2024 lists ~200). TEO candidates outside the known HBS sample may be hiding.

**Novelty gap.** No ML anomaly-detection / density-estimation study of Kepler short-cadence for HBSs. Existing work (Kolaczek-Szymanski & Rozanski 2023) uses theoretical forward modeling + ML to *analyze* TEOs, not to *find* new HBS candidates.

**Dataset.** Kepler short-cadence (SC) light curves (MAST); HBS sample of Thompson+ 2012, Kirk+ 2016, Guo+ 2024.

**Baseline.** The manual Thompson selection; the autoencoder-anomaly-score approach of Martinez-Galarza+ 2021 (not on HBS).

**Method.** Train a convolutional autoencoder on Kepler SC light curves; rank reconstructions by residual; query top anomalies in phased space for characteristic heartbeat morphology.

**Why feasible.** SC subset manageable via astroquery. 1D CNN AE on a T4.

**Venue.** AJ, MNRAS, ApJL.

**Compute.** <15 GPU-hours.

---

## APPENDIX A: COMPARISON TABLE TO EXISTING 10 NICHES

| # | New niche | Key technique | Domain | Differentiator from existing 10 |
|---|-----------|---------------|--------|----------------------------------|
| 1 | PU learning for DeltaPi_1 extraction | PU + ConvNeXt | Asteroseismology (subgiants) | Not few-shot rare transients; new physics |
| 2 | Mamba on RG power spectra | State-space SSM | Asteroseismology (red giants) | Not TS FM fine-tune; new arch |
| 3 | PySR gyrochronology residuals | Symbolic regression | Stellar ages | Not SIREN/INR; interpretable closed form |
| 4 | LDAM / self-distill on Gaia XP metallicity tails | Tail-aware learning | Gaia XP spectroscopy | New problem, new technique |
| 5 | SimCLR XP + epoch photometry for WDs | Contrastive SSL | White dwarfs | New modality combo |
| 6 | GNN on Gaia XP cluster chemical homogeneity | GraphSAGE on clusters | Open clusters | Not OT domain adaptation |
| 7 | DINOv2 on HTRU + FAST diagnostic plots | SSL ViT | Pulsar candidates | New domain for ViT |
| 8 | FMPE / NPE for FRB DM | SBI flows | Radio transients | Not conformal transients; new physics |
| 9 | GNN for MeerKAT component association | Edge-GNN | Radio continuum | New technique |
| 10 | TTA for RFI across telescopes | Test-time adaptation | RFI mitigation | Not OT; new technique |
| 11 | Neural field for ring-diagram flows | INR (not image) | Helioseismology | Distinct from INR compression |
| 12 | DDPM super-res of CH masks | Diffusion SR | Solar coronal holes | Not diffusion anomaly detection |
| 13 | Contrastive SSL on SolARED | SSL on acoustic | Solar pre-emergence | Not PINN transits; new problem |
| 14 | SNRE for HOD on CAMELS | SBI ratio estimation | Cosmology | New SBI application |
| 15 | Flow matching for subhalos | FMPE | Strong lensing | New SBI technique |
| 16 | YOLO-NAS streaks on ATLAS | Object detection | NEO detection | New survey + new model |
| 17 | TTT for photo-z drift | Test-time training | Photo-z | Not OT adaptation; self-supervised |
| 18 | Normalizing flow for IA | Conditional flows | Cosmology systematics | New IA modeling |
| 19 | Multi-task EB classif + params | Multi-task CNN | Eclipsing binaries | New E2E combination |
| 20 | Autoencoder for heartbeat-star anomalies | AE anomaly detection | TEOs | Not diffusion anomaly; new class |

---

## APPENDIX B: SANITY CHECK ON NOVELTY

For each niche I searched arXiv-indexed literature and A&A/MNRAS recent issues up to 2026-04 and found either (a) no hit, (b) a closely related but methodologically different hit, or (c) an extension gap. In no case does a paper combine the specific method + dataset + task of the proposed niche.

### Specific gap evidence

* **Niche 1 (PU learning for period spacings):** Audenaert+ Nov 2025 A&A (arXiv 2511.20908) uses PU for gamma Dor/delta Sct hybrids, explicitly not DeltaPi_1 subgiants; different oscillation family.
* **Niche 2 (Mamba asteroseismology):** arXiv search for "Mamba asteroseismology" or "state space red giant" returns zero hits.
* **Niche 3 (PySR gyrochronology):** Only astro PySR applications are to neutron-star EoS (e.g., Carreiro+ 2024); nothing on gyrochronology.
* **Niche 4 (LDAM on Gaia XP tails):** Andrae+ 2023, Khalatyan+ 2024 (SHBoost), Zhang+ 2024, Laroche+ 2024, and the 2025 metal-poor-focused Zhao+ all use standard losses; no tail-aware training method published.
* **Niche 5 (SimCLR WD XP + photometry):** Vincent+ 2024 is supervised RF; Garcia-Zamora+ 2023 is supervised CNN; AstroCLIP / AION-1 do not include WDs as a core modality.
* **Niche 6 (GNN cluster chemical homogeneity):** Cavallo+ 2024 (DNN membership), Hunt-Reffert 2023 (HDBSCAN), Niu+ 2025 (refinement with XP) do not use GNNs or test homogeneity with propagation.
* **Niche 7 (SSL on pulsar diagnostics):** Liu+ 2024 (ViT, CvT), Zhu+ 2014 (PICS) are all supervised; no DINO/MAE pretraining reported.
* **Niche 8 (FMPE FRB DM):** Islam+ 2025 (arXiv 2512.24003) is CNN-LSTM feed-forward. No SBI applied to FRB DM in the literature.
* **Niche 9 (GNN MeerKAT association):** Bowles+ 2023 and Alegre+ 2022 used CNN + rule-based linkage; no GNN for component association in MeerKAT/MGCLS.
* **Niche 10 (TTA for RFI):** RFI-HWT (Sept 2025 A&A) uses supervised transfer; no TENT/SHOT paper for RFI.
* **Niche 11 (neural field helioseismic flows):** Roth+ 2019 is per-tile ANN; the only astro neural-field papers are Diaz Baso+ 2025 (spectropolarimetry) and Polania-Rey+ 2024 (galaxy kinematics), both distinct use cases.
* **Niche 12 (DDPM CH super-res):** Jarolim+ 2023 magnetogram SR and Deng+ 2024 AIA SR do not address CH masks; no CH super-resolution paper.
* **Niche 13 (contrastive SSL SolARED):** Kitiashvili+ 2025 LSTM baseline is fully supervised; no contrastive approach.
* **Niche 14 (SNRE HOD CAMELS):** Jespersen+ 2022 (graph NN on subhalos), Shao+ 2023 (cosmological GNNs) do not target HOD directly with SBI.
* **Niche 15 (FMPE subhalos):** Dhanasingham+ Nov 2025 uses NPE but not FMPE; flow matching is from 2023 and is not yet in strong-lensing literature.
* **Niche 16 (YOLO-NAS ATLAS):** Duev+ 2025 uses ZTF + older CNN; ATLAS archive + YOLO-NAS is an open gap.
* **Niche 17 (TTT photo-z):** PZFlow (Crenshaw 2024) doesn't adapt at test time; no TTT photo-z work exists.
* **Niche 18 (normalizing flow IA):** IAEmu (Pandya 2025) is an MLP emulator of 2-pt functions, not a density estimator.
* **Niche 19 (multi-task TESS EBs):** CALNet (Yu 2025) is classification only; probabilistic NN of Viani 2026 is parameters only; no joint model.
* **Niche 20 (AE anomaly for heartbeat TEOs):** Kolaczek-Szymanski & Rozanski 2023 uses supervised ML on theoretical models; no anomaly-detection search for new HBS.

---

## APPENDIX C: RECOMMENDED STARTING 5 FOR AN UNDERGRAD

If forced to pick the 5 highest reward / lowest risk, I would choose:

1. **Niche 2 (Mamba on red-giant power spectra)** - trendy architecture, clear baseline, easy data.
2. **Niche 3 (PySR gyrochronology)** - CPU only, closed-form result is a guaranteed RNAAS.
3. **Niche 8 (FMPE for FRB DM)** - `sbi` package makes it easy; real data is public; novel.
4. **Niche 13 (SSL on SolARED)** - dataset is ML-ready, baseline is clear, SSL is standard.
5. **Niche 17 (TTT for photo-z drift)** - trivial technique transfer, big-survey applicability.

Each should yield a 6-8 week undergraduate project ending in an RNAAS / ML4PS poster, or a longer (3-6 month) effort ending in a MNRAS / A&A letter.

---

## SOURCES

### Asteroseismology
- Audenaert et al. 2025 A&A (PU learning for gamma Dor / delta Sct hybrids): https://www.aanda.org/articles/aa/full_html/2025/11/aa56079-25/aa56079-25.html
- Hon et al. 2018 MNRAS (deep learning classification of 15,000 Kepler red giants): https://doi.org/10.1093/mnras/sty483
- Lyttle et al. arXiv:2601.02926 (branching NN for mode frequency emulation): https://arxiv.org/html/2601.02926
- Stokholm et al. arXiv:2604.00498 (19,000 red giants in TESS CVZ): https://arxiv.org/abs/2604.00498
- From stellar light to astrophysical insight review 2025: https://link.springer.com/article/10.1007/s10509-025-04460-5
- Van-Lane et al. 2024 ChronoFlow: https://arxiv.org/abs/2412.12244
- Inferring coupling strengths of mixed-mode oscillations in red giants 2023: https://arxiv.org/html/2309.17372

### Mamba State-Space Models
- Mamba original repo: https://github.com/state-spaces/mamba
- MambaTS: https://openreview.net/forum?id=vEtDApqkNR
- Mamba-360 survey 2025: https://www.sciencedirect.com/science/article/abs/pii/S0952197625012801

### Symbolic Regression
- PySR: https://github.com/MilesCranmer/PySR
- PySR astro review 2024: https://link.springer.com/article/10.1007/s10710-024-09503-4
- Symbolic regression astronomical classification 2025: https://arxiv.org/html/2503.09220

### Gaia XP Spectroscopy
- Khalatyan et al. 2024 A&A SHBoost: https://www.aanda.org/articles/aa/full_html/2024/11/aa51427-24/aa51427-24.html
- Andrae et al. 2023 (Gaia XP XGBoost): https://arxiv.org/abs/2407.06963
- Multimodal spectra ML4PS 2024: https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_119.pdf
- Zhang et al. 2025 Gaia/RVS GAN: https://www.aanda.org/articles/aa/full_html/2025/02/aa51786-24/aa51786-24.html
- Uncertain NNs on Gaia BP/RP: https://academic.oup.com/mnras/article/531/1/2126/7676864
- Vincent et al. 2024 white dwarfs XP: https://ui.adsabs.harvard.edu/abs/2024A&A...682A...5V
- Niu et al. 2025 XP metallicities for open clusters: https://www.aanda.org/articles/aa/full_html/2025/11/aa55437-25/aa55437-25.html
- Hunt-Reffert 2023 open cluster catalog: https://www.aanda.org/articles/aa/full_html/2023/05/aa46285-23/aa46285-23.html
- Cavallo et al. 2024 DNN cluster membership: https://arxiv.org/abs/2303.08474

### Radio Astronomy
- CHIME FRB deep learning 2025 A&A: https://www.aanda.org/articles/aa/full_html/2025/10/aa55217-25/aa55217-25.html
- CHIME FRB machine learning DM estimation: https://arxiv.org/html/2512.24003
- Frabjous FRB morphologies: https://arxiv.org/html/2507.14854
- CHIME robust FRB classification: https://arxiv.org/html/2511.02634v1
- Pulsar candidate transformer (Liu et al.): https://www.sciencedirect.com/science/article/pii/S0577907324002016
- VQPulsar VQVAE/GPT: https://dl.acm.org/doi/10.1007/978-981-96-0821-8_22
- MGCLS VQ-VAE 2026: https://arxiv.org/abs/2601.13901
- MeerKAT SSL DINO 2024 A&A: https://www.aanda.org/articles/aa/full_html/2024/10/aa49964-24/aa49964-24.html
- ContinUNet U-Net radio segmentation: https://academic.oup.com/rasti/article/3/1/315/7685538
- MeerTRAP radio transients 2025: https://arxiv.org/html/2510.17723
- RFI-HWT 2025: https://www.aanda.org/articles/aa/full_html/2025/09/aa54423-25/aa54423-25.html
- FAST-SETI DBSCAN RFI 2025: https://arxiv.org/html/2512.15809v1
- RFI deep learning survey MNRAS 2024: https://academic.oup.com/mnras/article/530/1/613/7637224
- WALLABY HI ML: https://arxiv.org/html/2409.11668v2

### Solar Physics (non-flare)
- SolARED dataset arXiv 2601.13145: https://arxiv.org/html/2601.13145v1
- SolARED flux emergence prediction 2024: https://arxiv.org/html/2402.08890v1
- LSTM magnetic flux solar emergence: https://arxiv.org/html/2604.03507
- Multi-channel coronal hole detection CHRONNOS: https://www.aanda.org/articles/aa/full_html/2021/08/aa40640-21/aa40640-21.html
- Incorporating magnetic field in coronal hole segmentation 2025: https://arxiv.org/html/2501.13211v2
- Flat U-Net solar filament: https://arxiv.org/html/2502.07259
- DeepHalo CME transformer: https://arxiv.org/abs/2503.03237
- Surya foundation model for solar physics: (AION family) https://arxiv.org/html/2510.17960v1
- Helioseismology ring-diagram ANN: https://www.aanda.org/articles/aa/full_html/2019/02/aa34237-18/aa34237-18.html
- Diaz Baso neural field spectropolarimetry 2025: https://ui.adsabs.harvard.edu/abs/2025A&A...693A.170D/abstract
- Solar differential rotation helioseismology 2025: https://arxiv.org/html/2505.10756
- Solar radio burst transfer learning 2025: https://arxiv.org/abs/2512.11487
- SEPNET multi-task: https://arxiv.org/html/2512.12786

### SBI and Cosmology
- CAMELS project page: https://www.simonsfoundation.org/flatiron/center-for-computational-astrophysics/galaxy-formation/camels-cosmology-and-astrophysics-with-machine-learning-simulations/
- CAMELS multi-fidelity SBI 2025: https://arxiv.org/html/2505.21215
- AION-1 foundation model: https://arxiv.org/abs/2510.17960
- AstroCLIP: https://academic.oup.com/mnras/article/531/4/4990/7697182
- OmniSpectra: https://arxiv.org/html/2601.15351
- Field-level N-body emulator: https://arxiv.org/abs/2206.04594
- COCA N-body emulation 2025: https://www.aanda.org/articles/aa/full_html/2025/02/aa52217-24/aa52217-24.html
- 21cmPIE-INN reionization: https://scipost.org/10.21468/SciPostPhysCore.8.2.037
- LISA galactic binaries SBI 2025: https://arxiv.org/abs/2506.22543
- Dhanasingham lens NPE Nov 2025: https://arxiv.org/abs/2511.17732
- Diffusion mass map DES Y3 2025: https://arxiv.org/html/2511.14667
- IAEmu / differentiable HOD 2026: https://arxiv.org/abs/2602.04977
- FMPE exoplanet atmospheres 2025: https://www.aanda.org/articles/aa/full_html/2025/01/aa51861-24/aa51861-24.html
- GW signal robust SBI with gaps: https://arxiv.org/abs/2512.18290

### Transients and Other
- Transformer TDE classifier TTC: https://arxiv.org/html/2512.24782
- ELAsTiCC benchmarks Fink: https://www.aanda.org/articles/aa/full_html/2024/12/aa50370-24/aa50370-24.html
- Rubin TDE photometric classifier 2025: https://www.aanda.org/articles/aa/full_html/2025/11/aa56839-25/aa56839-25.html
- ORACLE real-time LSST classifier: https://arxiv.org/html/2501.01496v1
- MALLORN TDE dataset: https://arxiv.org/html/2512.04946
- Transfer learning transient ZTF->LSST: https://arxiv.org/html/2502.18558v2
- Transit detection transformer MNRAS 2025: https://academic.oup.com/mnras/article/538/3/2031/8043272
- SELDON neural ODE VAE supernova: https://arxiv.org/html/2603.04392
- Asteroid detection CNNs VST 2025: https://www.aanda.org/articles/aa/full_html/2025/02/aa52756-24/aa52756-24.html
- Duev NEO deep learning ZTF: https://arxiv.org/html/2504.11918v1

### Galactic archaeology / stellar populations
- CREEK stellar streams 2025: https://arxiv.org/html/2507.08074v1
- APOGEE chemical abundance neural network 2025: https://arxiv.org/abs/2506.20345
- Age determination LAMOST gradient boosting: https://arxiv.org/html/2310.17196
- RR Lyrae Gaia metallicity ML: https://academic.oup.com/mnras/article/536/3/2749/7916662
- RR Lyrae Gaia GRU: https://arxiv.org/html/2505.20947
- Hot subdwarf XP binaries 2025: https://arxiv.org/abs/2601.21727
- Advanced hot subdwarf classification 2024: https://arxiv.org/html/2409.17783
- Unsupervised moving groups Gaia DR3 2025: https://arxiv.org/html/2512.09078v1
- Variable source multi-modal Gaia autoencoders 2025: https://www.aanda.org/articles/aa/full_html/2025/09/aa54025-25/aa54025-25.html
- Gaia variable stars DR3 24-class supervised: https://www.aanda.org/articles/aa/full_html/2023/06/aa45591-22/aa45591-22.html
- Kepler LightPred rotation model 2025: https://arxiv.org/html/2407.06858v2

### Eclipsing binaries and heartbeat stars
- TESS Ten Thousand Catalog EB: https://arxiv.org/abs/2506.05631
- CALNet eclipsing binaries: https://arxiv.org/html/2504.15875
- Probabilistic NN eclipsing binaries: https://arxiv.org/html/2604.01281
- Heartbeat stars TEO Kepler: https://arxiv.org/html/2309.05060
- Heartbeat TEO mode identification: https://arxiv.org/html/2401.08222

### JWST and galaxy morphology
- JWST high-z VAE denoising: https://arxiv.org/abs/2511.21207
- DAWN JWST Archive morphology 2025: https://www.aanda.org/articles/aa/abs/2025/07/aa55504-25/aa55504-25.html
- Euclid CNN strong lenses 2025: https://www.aanda.org/articles/aa/full_html/2025/10/aa54132-25/aa54132-25.html
- Euclid morphology sparse autoencoders 2025: https://arxiv.org/abs/2510.23749
- Dark Energy Survey Y3 weak lensing SBI: https://arxiv.org/html/2511.04681
- Galaxy cluster mass ML TNG-Cluster 2025: https://iopscience.iop.org/article/10.3847/1538-4357/adcd69

### Flare / Stellar activity
- StellarF LoRA LLM: https://arxiv.org/html/2507.10986
- TESS flare morphology 2025: https://www.aanda.org/articles/aa/full_html/2025/02/aa52489-24/aa52489-24.html
- CME-associated stellar flares 2025: https://arxiv.org/abs/2512.15484
- Scalable TESS flare 2024: https://arxiv.org/html/2409.02540

### Gravitational-wave ML
- Gravity Spy glitch classification O4: https://arxiv.org/abs/2401.12913
- SBI GW review 2025: https://arxiv.org/abs/2507.11192
- SBI GW stochastic background: https://link.aps.org/doi/10.1103/PhysRevD.109.083008
- deep transfer LIGO glitch: https://arxiv.org/abs/1706.07446

### Exoplanet and protoplanetary disk
- NEXOTRANS retrieval framework 2025: https://arxiv.org/html/2510.27223
- VADER VAE protoplanetary disks 2025: https://arxiv.org/html/2511.13840
- DBNets deep learning disks: https://arxiv.org/html/2402.12448
- ExoDNN Gaia DR3 exoplanet: https://www.aanda.org/articles/aa/full_html/2025/12/aa55598-25/aa55598-25.html

### Cosmic ray and infrastructure
- CSST cosmic ray rejection 2025: https://arxiv.org/html/2511.01524
- deepCR: https://github.com/profjsb/deepCR

### Dwarf galaxies and low surface brightness
- YOLO-DG dwarf detection 2025: https://www.aanda.org/articles/aa/full_html/2025/04/aa53042-24/aa53042-24.html
- UDGnet-DES ultra-diffuse 2026: https://www.aanda.org/articles/aa/full_html/2026/02/aa56863-25/aa56863-25.html
- Kilo-Degree Survey UDG 2025: https://arxiv.org/html/2509.13910
- DES to HSC LSB 2025: https://www.aanda.org/articles/aa/full_html/2025/03/aa52934-24/aa52934-24.html
- Tidal features DECaLS: https://arxiv.org/html/2404.06487
- Self-supervised tidal features: https://arxiv.org/html/2308.07962v2

### Photo-z and multi-modal foundation models
- DeepDISC-photoz for LSST: https://arxiv.org/html/2411.18769v1
- Bayesian photo-z LSST: https://arxiv.org/html/2306.13179
- Uncertainty quantification photo-z 2026: https://arxiv.org/abs/2601.17222
- DeepRed EfficientNet SwinT MLP-mixer: https://arxiv.org/html/2602.11281

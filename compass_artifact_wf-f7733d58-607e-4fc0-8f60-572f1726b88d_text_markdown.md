# Astronomy meets AI: a research landscape and roadmap for undergraduate publication

**The intersection of astronomy and artificial intelligence is experiencing explosive growth, with foundation models, transformers, and simulation-based inference reshaping every subfield from galaxy classification to gravitational wave detection.** For an undergraduate seeking a publishable project, the most promising niches combine a genuinely novel ML method with a well-benchmarked astronomical dataset—and several such opportunities remain wide open. This report maps the full landscape, dives deep into light-curve reconstruction methods, and identifies concrete, tractable research projects with public data, established baselines, and clear paths to publication at venues like RNAAS, ML4PS at NeurIPS, or AAS meeting posters.

---

## GOAL 1 — The broad landscape of AI in astronomy

### Galaxy morphology, exoplanets, and gravitational waves

**Galaxy morphology classification** is one of the oldest ML applications in astronomy. Convolutional neural networks trained on Galaxy Zoo data (GZ1, GZ2, GZ DECaLS) dominated through 2022, but Vision Transformers now achieve **>98% accuracy** on five-class morphology tasks. The Convolutional Vision Transformer (CvT) applied by Chen et al. (A&A 2024) set the current benchmark at 98.8% overall accuracy. Zoobot, a pre-trained model from Walmsley et al., provides transfer-ready embeddings from 100M human annotations. The field is shifting toward unsupervised and self-supervised methods to prepare for the billions of galaxies expected from LSST/Rubin Observatory and Euclid.

**Exoplanet detection** has moved from classification/validation to direct discovery. Transformers now process raw TESS Full-Frame Image light curves without requiring phase folding—a 2025 MNRAS paper found 214 new transit candidates from TESS sectors 1–26. ExoMiner++ (NASA) continues to improve automated vetting, and the community is preparing for PLATO's 2026 launch. Meanwhile, **gravitational wave detection** uses deep learning for real-time signal identification in LIGO O4/O5 data. A 2025 systematic review by Zhao et al. documents the rapid maturation of neural posterior estimation for parameter inference, with normalizing flows enabling Bayesian posteriors in seconds rather than hours.

### Variable stars, photometric redshifts, and lensing

**Variable star classification** is a thriving ML subfield driven by massive survey catalogs. Random forests with hand-crafted features remain competitive, but self-supervised transformers—particularly the ASTROMER family—now provide state-of-the-art representations for light curves. The ATAT transformer achieves **macro F1 = 82.9%** on the 20-class ELAsTiCC challenge, outperforming the 429-feature Balanced Hierarchical Random Forest baseline (F1 = 79.4%). Foundation models that disentangle physics from instrument properties are the cutting edge, exemplified by the causal models from Audenaert et al. (2025).

**Photometric redshift estimation** is critical for LSST cosmology. The field has matured from template fitting versus ML debates into hybrid approaches: Bayesian Neural Networks with conformal prediction for calibrated uncertainties, direct image-based estimation outperforming catalog-based methods, and Mixture Density Networks for full probability distributions. **Strong gravitational lensing detection** is preparing for LSST's expected ~100,000 lenses, with Vision Transformers competitive with CNNs and unsupervised domain adaptation bridging the simulation-to-real gap. DeepLense (ML4Sci/CERN) provides a modular pipeline.

### Solar physics, cosmological inference, and radio astronomy

**Solar flare prediction** remains challenged by severe class imbalance (M/X flares are rare). The best methods combine physics-based SHARP magnetic field features with contrastive learning + GRU classifiers, achieving **TSS = 0.801**. The SDO/HMI SHARP parameter dataset is freely available. **Simulation-based inference (SBI)** is rapidly maturing: Bairagi et al. (A&A 2025) established scaling laws for how many simulations are needed, and LtU-ILI provides an open-source toolkit. SBI is being applied to real DES and KiDS data. **Radio astronomy** ML focuses on RFI classification and pulsar candidate vetting, with the Square Kilometre Array driving the need for exabyte-scale processing.

### Anomaly detection, transient classification, and spectroscopy

**Anomaly detection in sky surveys** is one of the most active areas, driven entirely by LSST preparation. Isolation forests, autoencoders (AHA pipeline), and multi-class anomaly scoring (MCIF) compete on ZTF alert streams and ELAsTiCC simulations. DiMMAD, presented at NeurIPS ML4PS 2025, uses multi-metric distance ensembles for out-of-distribution detection. **Transient classification** is led by transformer architectures (ATAT, ATCAT) and increasingly multimodal approaches combining photometry, images, spectra, and metadata (AppleCiDEr, AstroM3). Transfer learning from ZTF to LSST reduces labeled data needs by **95%**.

**Spectroscopic analysis** has evolved from The Cannon (data-driven, generative) through StarNet (CNN) to SpecTE (transformer encoder with denoising pretraining), achieving MAE of ~45K in Teff on LAMOST spectra. Cycle-StarNet uses GANs to bridge the synthetic-real spectral gap.

### Generative models and foundation models are reshaping the field

**Diffusion models are overtaking GANs** for cosmological simulation emulation, offering better mode coverage without mode collapse. The Psi-GAN and νGAN generate conditional cosmological fields across parameter spaces and redshifts, while 3D denoising diffusion enables super-resolution of dark matter volumes.

**Foundation models represent the dominant trend of 2024–2026.** AstroCLIP (Polymathic AI, MNRAS 2024) embeds galaxy images and spectra into a shared latent space via contrastive learning, enabling zero-shot photo-z estimation and morphology classification. AstroLLaMA evolved through multiple iterations—the original 7B model suffered catastrophic forgetting, but AstroSage-LLaMA-3.1-8B (Nature Scientific Reports 2025) achieves **80.9%** on AstroMLab-1, matching GPT-4o. The **Multimodal Universe** dataset (NeurIPS 2024 Datasets & Benchmarks) provides 100TB of cross-matched images, spectra, time series, and metadata from 20+ surveys on HuggingFace. AION-1 represents the frontier: an omnimodal model combining HSC, Legacy Survey, DESI, SDSS, and Gaia data.

### Key venues and workshops

The primary ML-for-astronomy venues span both communities. **ML4PS at NeurIPS** is the premier workshop (4-page extended abstracts, ~30–50% acceptance, double-blind). **RNAAS** (Research Notes of the AAS) accepts 1,500-word contributions with one figure, is free to publish, and explicitly welcomes undergraduate summer project results—acceptance is near-universal. Major journals include **ApJ/AJ** (AAS), **MNRAS** (RAS), and **A&A** (European). The arXiv astro-ph.IM section is where most ML papers first appear. Additional venues include ICML AI4Science workshops, Astronomy & Computing (Elsevier), and Frontiers in Astronomy and Space Sciences.

---

## GOAL 2 — Deep dive on AI methods for light-curve reconstruction from sparse data

### The reconstruction problem and why traditional methods dominate

Reconstructing continuous light curves from sparse, irregularly sampled observations is fundamental to time-domain astronomy. Cepheid and RR Lyrae variables—critical standard candles for the cosmic distance ladder—are typically observed at irregular cadences with gaps spanning days to months. Traditional approaches rely on **Lomb-Scargle periodograms** for period finding, **Fourier decomposition** (typically 7th order) for shape fitting, and **Gaussian process regression** with hand-designed kernels for interpolation. The user seeks purely computational alternatives: end-to-end learned methods that avoid explicit basis transformations.

### Transformer architectures purpose-built for astronomical time series

The **ASTROMER family** represents the most mature purely-learned approach for astronomical light curves. ASTROMER v1 (Donoso-Oliva et al., A&A 2023, arXiv:2205.01677) is an encoder-only transformer with sinusoidal positional encodings adapted for irregular observation times. Pre-trained via self-supervised masked reconstruction on ~1.5M MACHO R-band light curves, it outperforms RNN baselines when labeled data is limited. **ASTROMER 2** (arXiv:2502.02717, 2025) improved the architecture with BERT-style masking (50% positions probed: 30% masked, 10% randomized, 10% unchanged). **AstroCo** (NeurIPS ML4PS 2025, arXiv:2509.24134) introduced a Conformer-style encoder combining multi-head self-attention with depthwise convolutions and gating, achieving **70% lower reconstruction RMSE** than ASTROMER v1. **Multiband Astromer** (arXiv:2506.11637, 2025) extends to asynchronous multi-band observations via fusion mechanisms.

**ATAT** (A&A 2024) and **ATCAT** (arXiv:2511.00614, 2025) use novel time modulation and quantile feature tokenizers for irregular light curve classification. ATCAT achieves **71.8% F1 on ELAsTiCC** (new SOTA), outperforming all previous methods at every early-detection cutoff. **TimeModAttn** (arXiv:2201.08482) applies temporal modulation to supernova classification without explicit imputation. **Astroconformer** (MNRAS 2024) combines self-attention with convolutions for Kepler stellar parameter inference, achieving 0.017 dex RMSE for surface gravity.

### Time-series foundation models show surprising zero-shot transfer

The **StarEmbed benchmark** (arXiv:2510.06200, 2025) is the first systematic evaluation of time-series foundation models on astronomical light curves. Testing Chronos, Chronos-Bolt, Moirai, and ASTROMER on ~40,000 expert-labeled ZTF light curves across 7 classes, it found that **Chronos models—trained entirely on non-astronomical data—can outperform domain-specific ASTROMER baselines** in some tasks, particularly out-of-distribution detection. This motivates fine-tuning TSFMs for astronomy. However, current TSFMs handle irregular timestamps by treating observations in measurement order, discarding absolute timestamp information—a significant limitation for period-sensitive applications like Cepheid analysis.

**Chronos** (Amazon, arXiv:2403.07815) tokenizes time series via scaling and quantization into discrete vocabulary, using a T5 architecture. **Moirai** (Salesforce, ICML 2024 Oral) uses masked encoder-based universal forecasting with multi-patch-size projections. **TimesFM** (Google) and **Lag-Llama** are decoder-only architectures. None were designed for irregular sampling, limiting their direct applicability to astronomical reconstruction without adaptation.

### Neural ODEs offer continuous-time dynamics but remain untapped for variable stars

**Latent ODEs** (Rubanova et al., NeurIPS 2019, arXiv:1907.03907) generalize RNNs to continuous-time hidden dynamics, naturally handling arbitrary time gaps. The decoder generates continuous trajectories at any queried time—exactly what light-curve reconstruction requires. **Neural SDDE** (arXiv:2508.17521, CIKM 2025) extends this with stochastic diffusion terms and explicit time delays, and has been **directly tested on PLAsTiCC data**, achieving the highest performance across all scenarios. **GRU-ODE-Bayes** (NeurIPS 2019) combines GRU with neural ODEs for sporadically-observed time series. Despite this promise, **no paper has applied Neural ODEs specifically to Cepheid or RR Lyrae light-curve modeling**—a clear gap.

### Implicit neural representations are completely unexplored in astronomy

**SIREN** (Sitzmann et al., NeurIPS 2020, arXiv:2006.09661) uses MLPs with sinusoidal activation functions to map coordinates to signal values continuously. For light curves, a SIREN mapping time → magnitude would produce a continuous, differentiable reconstruction from sparse samples. The periodic activations are naturally suited for periodic variable stars. SIREN achieves **>10 dB PSNR improvement** over ReLU networks for signal fitting. **HyperTime** extends INRs specifically to time-series generation. Context-aware INRs for field reconstruction from sparse sensor data (arXiv:2401.11611) demonstrate the paradigm's applicability. Yet **no paper has applied INRs to astronomical light curves**—this is the single largest unexplored opportunity identified in this survey.

### Attention-based methods for irregular timestamps

**Multi-Time Attention Networks (mTAN)** (Shukla & Marlin, ICLR 2021, arXiv:2101.10318) learn continuous-time embeddings and use attention to produce fixed-length representations from variable-number observations. The interpolation capability—conditioning on sparse irregular observations and reconstructing at any time point—is exactly what light-curve reconstruction needs. mTAN achieves **AUC 0.858** on PhysioNet while being **85× faster than ODE-RNN** and **100× faster than Latent ODE**. **SeFT** (Set Functions for Time Series) treats observations as unordered sets with attention-based aggregation. **STraTS** uses observation triplets (time, variable, value) with continuous value embeddings. None have been applied to astronomical data.

### The seminal predecessor and what it accomplished

**Naul et al. (Nature Astronomy 2018, arXiv:1711.10609)** remains the seminal paper for end-to-end learned light-curve representation. Their unsupervised autoencoding RNN explicitly uses sampling times and heteroskedastic noise, achieving **98.8% classification accuracy** on ASAS data. Autoencoded features rival hand-crafted features. However, this approach requires period-folding for best performance and does not produce truly continuous reconstructions.

### Benchmarks and current performance baselines

| Dataset | Classes | Best Method | Performance | Metric |
|---------|---------|-------------|-------------|--------|
| OGLE (5 classes) | Variable stars | CNN (phase-folded) | 88% / F1=0.89 | Accuracy/F1 |
| EROS-2 LMC (5 superclasses) | Periodic variables | Random Forest (22 features) | 99% | Precision/Recall |
| ELAsTiCC (20 classes) | Transients + variables | ATCAT transformer | **F1 = 71.8%** | Macro F1 |
| Kepler (5 classes) | Variable stars | Swin Transformer | **99%** | Accuracy |
| MACHO (6 classes) | Variable stars | AstroCo conformer | Best reconstruction RMSE | RMSE |
| ZTF (7 classes) | Variable stars | Chronos (zero-shot) | SOTA on OOD detection | ARI/F1 |
| PLAsTiCC (18 classes) | Transients | Ensemble (LightGBM+NN) | Competition winner | Weighted log-loss |

Key datasets with Cepheid/RR Lyrae labels include **OGLE OCVS** (~270,000 RR Lyrae, ~15,000 Cepheids in Gaia DR3), **ZTF CPVS** (781,602 periodic variables), **ASAS-SN** (687,695 variables), and **MACHO** (~19,000 classified variables). All are freely downloadable.

---

## GOAL 3 — Ten publishable niches for an undergraduate

This section identifies concrete research projects ranked by a composite score of novelty, feasibility, and publication potential. Each includes specific datasets, baselines to beat, suggested approaches, and realistic timelines.

### 1. SIREN for light-curve reconstruction — the top recommendation

**Problem statement**: Can implicit neural representations (specifically SIREN networks) reconstruct continuous Cepheid and RR Lyrae light curves from sparse, irregularly sampled observations, matching or exceeding Gaussian process regression quality?

**Datasets**: OGLE Collection of Variable Stars (ogle.astrouw.edu.pl/~ogle/OCVS/) — thousands of well-sampled Cepheid and RR Lyrae light curves in V and I bands. Gaia DR3 RR Lyrae catalog (~270,000 sources with G/BP/RP photometry).

**Current SOTA/baseline**: Gaussian process regression with Matérn or quasi-periodic kernels (using `celerite` or `george` Python packages). Fourier decomposition (typically 7th order) for phase-folded curves.

**Suggested approach**: Fit a small SIREN network (3–5 layers, 256 units, sine activations) per light curve, mapping time → magnitude. Test on artificially downsampled well-observed curves (remove 50–90% of points). Compare reconstruction RMSE, phase coverage, and derivative quality against GP regression. Explore meta-learned hypernetworks that predict SIREN weights from a few observations.

**Why this works for an undergrad**: SIREN runs on CPU. The official SIREN GitHub includes a Colab notebook. OGLE data downloads in minutes. The concept is simple—fit a neural network to each curve—but **no paper has tried this for astronomical light curves**, making it genuinely novel. A comparison of SIREN vs. GP on 100+ Cepheids, with reconstruction quality as a function of data sparsity, is a complete, publishable result.

**Venue**: RNAAS (1,500 words + 1 figure showing reconstruction examples and RMSE comparison) or ML4PS at NeurIPS 2026 (4-page paper with ablation study). **Difficulty**: Low. **Compute**: CPU/Colab free tier. **Timeline**: 6–8 weeks.

### 2. Few-shot classification of rare transients with prototypical networks

**Problem statement**: Can prototypical networks classify rare transient types (superluminous supernovae, tidal disruption events, kilonovae) from only 5–20 labeled examples, outperforming standard classifiers trained on imbalanced data?

**Datasets**: PLAsTiCC challenge data on Kaggle (kaggle.com/c/PLAsTiCC-2018) — 3.5M simulated LSST light curves in 6 bands, 18 classes with extreme imbalance. ELAsTiCC simulation data.

**Current SOTA/baseline**: PLAsTiCC competition winner used LightGBM + MLP ensembles. ASTROMER 2 tested classification with 20–500 samples per class. No paper has applied prototypical networks or MAML to PLAsTiCC-style data.

**Suggested approach**: Extract features using FATS/feets or ASTROMER embeddings. Construct few-shot episodes: common classes (SNIa, SNIbc) as base training set, rare classes (SLSN, TDE, KN) as novel test classes with K=5,10,20 shots. Compare prototypical networks against nearest-neighbor on ASTROMER embeddings and standard classifiers trained on full imbalanced data.

**Why this works for an undergrad**: PLAsTiCC data is preprocessed and tutorial-rich from the 2018 Kaggle challenge. Prototypical networks are simple to implement (many PyTorch implementations exist). The rare-class focus addresses a real LSST operational need.

**Venue**: ML4PS at NeurIPS (ideal fit) or Astronomy & Computing. **Difficulty**: Low-moderate. **Compute**: Single GPU/Colab. **Timeline**: 8–12 weeks.

### 3. Neural ODE dynamics for Cepheid pulsation light curves

**Problem statement**: Can Neural ODEs learn continuous dynamical systems that generate Cepheid light curves, and does this learned dynamics capture physically meaningful pulsation properties?

**Datasets**: OGLE Cepheid catalog (classical Cepheids in LMC/SMC, thousands of sources with multi-hundred-epoch VI photometry).

**Current SOTA/baseline**: 7th-order Fourier decomposition is the standard parameterization. Bellinger et al. (2019) used ANNs on theoretical pulsation models to predict physical parameters. No paper has used Neural ODEs for Cepheid light-curve modeling.

**Suggested approach**: Use `torchdiffeq` to define a latent ODE: dh/dt = f_θ(h, t), where h is a latent state and f_θ is a small MLP. Encode sparse observations into initial conditions, then solve the ODE to generate the light curve at arbitrary times. Compare fit quality against Fourier decomposition. Test whether the learned dynamics extrapolate beyond the observed time range.

**Why this works for an undergrad**: The `torchdiffeq` package handles all ODE-solving complexity. OGLE Cepheids have clean, periodic, well-sampled light curves—ideal for testing. The physics angle (learning pulsation dynamics) adds scientific interest beyond mere curve fitting.

**Venue**: RNAAS or ApJ Letters. **Difficulty**: Moderate (requires understanding Neural ODEs). **Compute**: Colab sufficient. **Timeline**: 10–14 weeks.

### 4. Conformal prediction for calibrated transient classification uncertainty

**Problem statement**: Can conformal prediction provide distribution-free uncertainty guarantees for astronomical transient classifiers, and how do prediction set sizes compare across different base models?

**Datasets**: PLAsTiCC (Kaggle), ZTF Bright Transient Survey data (via ALeRCE broker).

**Current SOTA/baseline**: Only 2–3 papers apply conformal prediction in astronomy (Jones et al. 2024 for photo-z, Soriano et al. 2025 for photo-z, one weak lensing paper). **No paper applies conformal prediction to transient classification.**

**Suggested approach**: Train standard classifiers (Random Forest, ATAT-style transformer, or use pre-trained ASTROMER embeddings + MLP) on PLAsTiCC training set. Apply split conformal prediction to produce prediction sets with guaranteed coverage (e.g., 90% marginal coverage). Analyze how set sizes vary with light-curve length, class rarity, and signal-to-noise. Compare against MC-Dropout and deep ensemble uncertainty.

**Why this works for an undergrad**: Conformal prediction is a post-hoc calibration method—**no retraining required**. The `mapie` or `conformal-prediction` Python packages implement it in a few lines. The novelty is the application domain, not the method complexity. Conformal prediction is a hot topic in ML, and astronomy applications are nearly virgin territory.

**Venue**: ML4PS at NeurIPS (strong fit—bridges ML theory and astronomy) or RNAAS. **Difficulty**: Low. **Compute**: CPU sufficient. **Timeline**: 6–8 weeks.

### 5. Extending StarEmbed with fine-tuned time-series foundation models

**Problem statement**: Does fine-tuning time-series foundation models (Chronos, Moirai) on astronomical light curves improve over the zero-shot performance reported in StarEmbed?

**Datasets**: StarEmbed benchmark (~40,000 labeled ZTF light curves, 7 classes). Code and data are publicly available.

**Current SOTA/baseline**: StarEmbed (arXiv:2510.06200) showed Chronos achieves SOTA on out-of-distribution detection but only tested zero-shot. No fine-tuning results have been published.

**Suggested approach**: Reproduce StarEmbed zero-shot results. Fine-tune Chronos-tiny and Moirai-small on the ZTF training split. Compare fine-tuned vs. zero-shot performance on classification (F1), clustering (ARI), and OOD detection. Test cross-survey transfer: fine-tune on ZTF, evaluate on ASAS-SN or OGLE.

**Why this works for an undergrad**: StarEmbed provides all code, data, and evaluation infrastructure. The student only needs to add fine-tuning loops. Chronos-tiny is small enough for a single GPU. The result—fine-tuned TSFMs vs. zero-shot—is a clear, publishable comparison.

**Venue**: RNAAS (comparison note) or short ML4PS contribution. **Difficulty**: Low. **Compute**: Single GPU/Colab. **Timeline**: 4–6 weeks.

### 6. Multi-Time Attention Networks for variable star light-curve interpolation

**Problem statement**: Can mTAN (Multi-Time Attention Networks), which achieved SOTA on clinical irregular time-series interpolation, provide better light-curve reconstruction than GPs when trained on large astronomical datasets?

**Datasets**: OGLE variable stars (millions of light curves), ASAS-SN (687,695 variables), ZTF (781,602 periodic variables).

**Current SOTA/baseline**: GP regression (celerite, george) is the standard interpolation method. ASTROMER's masked reconstruction provides a deep learning baseline. mTAN has never been applied to astronomical data.

**Suggested approach**: Adapt the official mTAN codebase (available on GitHub) for astronomical light curves. Train an mTAN encoder-decoder on OGLE multi-class variable stars. Evaluate interpolation quality by masking random subsets of well-sampled light curves and measuring reconstruction RMSE. Compare against celerite GP and ASTROMER masked reconstruction.

**Why this works for an undergrad**: mTAN's interpolation capability directly maps to the light-curve reconstruction problem. The codebase exists. The comparison against GPs and ASTROMER is clean and quantitative.

**Venue**: A&A or ML4PS. **Difficulty**: Moderate. **Compute**: Single GPU. **Timeline**: 10–12 weeks.

### 7. INR-based neural compression of astronomical images

**Problem statement**: Can implicit neural representations compress astronomical images more efficiently than traditional codecs while preserving photometric accuracy?

**Datasets**: **AstroCompress** benchmark (arXiv:2506.08306, June 2025) — 5 datasets of 16-bit astronomical imaging on HuggingFace, with established baselines for neural lossless compression.

**Current SOTA/baseline**: AstroCompress benchmarks several neural lossless compression algorithms against HEVC and H.264. The benchmark is brand new (June 2025), meaning any new result is immediately relevant.

**Suggested approach**: Fit SIREN or COIN-style (Dupont et al., 2021) implicit neural representations to astronomical image patches. Measure bits-per-pixel vs. reconstruction PSNR, specifically evaluating photometric preservation (flux accuracy in source pixels vs. background).

**Why this works for an undergrad**: AstroCompress provides ready-made data and evaluation infrastructure. INR-based compression is an active ML research area with no astronomy applications yet. Small networks fit on CPU.

**Venue**: ML4PS or Astronomy & Computing. **Difficulty**: Moderate. **Compute**: Single GPU. **Timeline**: 8–12 weeks.

### 8. Applying diffusion-based anomaly detection to ZTF alert streams

**Problem statement**: Can denoising diffusion probabilistic models detect anomalous transients in ZTF data more effectively than autoencoder-based methods (AHA pipeline)?

**Datasets**: ZTF alert stream (via ALeRCE broker API or `ztfquery` package). PLAsTiCC/ELAsTiCC simulations for controlled evaluation.

**Current SOTA/baseline**: AHA (arXiv:2602.12955, 2025) uses autoencoders. DiMMAD (NeurIPS ML4PS 2025) uses multi-metric distance ensembles. Isolation forests are widely used. **No paper has applied diffusion-based anomaly detection to astronomical transients.**

**Suggested approach**: Train a small diffusion model on "normal" light-curve features (common SNIa, variable stars). Score anomalies by reconstruction likelihood. Compare against isolation forest and autoencoder baselines on PLAsTiCC injection tests.

**Why this works for an undergrad**: Diffusion-based anomaly detection is a hot ML topic (2024–2025) with code available. PLAsTiCC provides controlled ground truth for evaluating anomaly detection.

**Venue**: ML4PS. **Difficulty**: Moderate-high. **Compute**: Single GPU. **Timeline**: 10–14 weeks.

### 9. Physics-informed neural network for exoplanet transit modeling

**Problem statement**: Can a PINN that encodes Mandel-Agol transit equations as physics-informed loss terms fit transit light curves faster than MCMC while maintaining parameter accuracy?

**Datasets**: Kepler/TESS light curves via `lightkurve` Python package (pip install lightkurve). Confirmed exoplanet parameters from NASA Exoplanet Archive.

**Current SOTA/baseline**: MCMC fitting with `batman` transit model is standard. ExoMiner uses CNNs for detection but not parameter inference. PINNs have been applied to solar physics (Nature Astronomy 2023) but **not to exoplanet transit fitting**.

**Suggested approach**: Parameterize a small MLP to predict flux as a function of time, with loss = data_fit + λ × physics_residual (where physics_residual enforces the Mandel-Agol transit shape). Compare inferred planet radius, impact parameter, and orbital period against catalog values.

**Why this works for an undergrad**: Transit equations are simple (undergraduate physics level). `lightkurve` makes data access trivial. PINNs for astronomy is an emerging area with high publication interest.

**Venue**: RNAAS or AJ. **Difficulty**: Moderate. **Compute**: CPU/Colab. **Timeline**: 8–12 weeks.

### 10. Cross-survey domain adaptation for periodic variable classification using optimal transport

**Problem statement**: Can optimal transport-based domain adaptation transfer a variable star classifier from OGLE (well-labeled, Southern hemisphere) to ZTF (Northern hemisphere, different cadence/filters) without target-domain labels?

**Datasets**: OGLE OCVS (source domain, thousands of labeled Cepheids/RR Lyrae/EBs with VI photometry). ZTF CPVS (target domain, 781,602 periodic variables with gr photometry).

**Current SOTA/baseline**: Kim et al. (2021) demonstrated DNN transfer from OGLE/EROS-2 to ASAS. Gupta & Muthukrishna (2025) showed contrastive/adversarial transfer for transients. **Optimal transport-based domain adaptation has not been applied to variable star cross-survey transfer.**

**Suggested approach**: Extract FATS features from both surveys. Use Python Optimal Transport (POT) library for Wasserstein-distance-based alignment of feature distributions. Compare against DANN (Domain-Adversarial Neural Networks) and naive transfer.

**Why this works for an undergrad**: The POT library is well-documented. FATS feature extraction standardizes different survey formats. The cross-survey problem is operationally important for LSST.

**Venue**: ML4PS or A&A. **Difficulty**: Moderate. **Compute**: Single GPU. **Timeline**: 10–12 weeks.

### Comparative summary of all ten niches

| Rank | Niche | Novelty | Feasibility | Compute | Timeline | Best Venue |
|------|-------|---------|-------------|---------|----------|------------|
| 1 | SIREN for light curves | ★★★★★ | ★★★★★ | CPU | 6–8 wk | RNAAS / ML4PS |
| 2 | Few-shot rare transients | ★★★★ | ★★★★★ | 1 GPU | 8–12 wk | ML4PS |
| 3 | Neural ODE for Cepheids | ★★★★★ | ★★★★ | Colab | 10–14 wk | RNAAS / ApJL |
| 4 | Conformal prediction | ★★★★ | ★★★★★ | CPU | 6–8 wk | ML4PS |
| 5 | Fine-tune StarEmbed TSFMs | ★★★ | ★★★★★ | 1 GPU | 4–6 wk | RNAAS |
| 6 | mTAN for light curves | ★★★★ | ★★★★ | 1 GPU | 10–12 wk | A&A / ML4PS |
| 7 | INR compression (AstroCompress) | ★★★★ | ★★★★ | 1 GPU | 8–12 wk | ML4PS |
| 8 | Diffusion anomaly detection | ★★★★ | ★★★ | 1 GPU | 10–14 wk | ML4PS |
| 9 | PINN for transit fitting | ★★★★ | ★★★★ | CPU | 8–12 wk | RNAAS / AJ |
| 10 | Optimal transport DA | ★★★ | ★★★ | 1 GPU | 10–12 wk | ML4PS |

### Existing competitions worth knowing about

**PLAsTiCC** (2018, Kaggle, 1,094 teams) remains the gold-standard classification challenge—data is still available and tutorial-rich. **ELAsTiCC** tested real-time broker readiness in 2022. The **Ariel Data Challenge** (Kaggle, annual since 2020) focuses on exoplanet atmospheric characterization from noisy spectra. The **SKA Science Data Challenge** covers radio continuum source finding. These competitions provide preprocessed datasets with clear evaluation metrics, making them ideal starting points.

### Practical tools and first steps for any project

Every recommended project can run on **Google Colab free tier** (NVIDIA T4, 16GB VRAM). Essential Python packages: `astropy`, `astroquery`, `lightkurve` (Kepler/TESS), `ztfquery` (ZTF), `GaiaXPy` (Gaia XP spectra), `galaxy-datasets` (Galaxy Zoo PyTorch datasets), `FATS`/`feets`/`light-curve` (feature extraction). For ML: `pytorch`, `torchdiffeq` (Neural ODEs), `mapie` (conformal prediction), and the official SIREN repository. The Multimodal Universe dataset on HuggingFace provides a single access point for cross-matched data from 20+ surveys.

---

## Conclusion

The astronomy-AI intersection in 2026 is defined by three converging forces: the imminent flood of data from LSST/Rubin Observatory, the rise of foundation models trained across astronomical modalities, and the maturation of simulation-based inference for cosmological parameter estimation. For an undergraduate seeking a first publication, the optimal strategy is to apply a well-understood ML method to a well-benchmarked astronomical dataset where that specific combination has not been tried. **Implicit neural representations for light-curve reconstruction stands out as the single best opportunity**—it is genuinely novel, computationally trivial, uses freely available OGLE data, and has clear quantitative baselines to beat. Conformal prediction for transient classification and few-shot learning for rare transients are close runners-up with similarly low barriers. The key insight across all recommendations is that the most publishable undergraduate projects are not the ones requiring the most sophisticated methods, but the ones identifying the right gap between an existing ML technique and an existing astronomical benchmark where nobody has yet connected the dots.
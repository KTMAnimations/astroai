# Undergraduate Research Proposals — Astronomy + AI

**Last updated: 2026-04-09**

**Status:** Originally five proposals. After independent literature review, scope calibration against published papers, and scrutiny through community best-practices frameworks (Huppenkothen et al. 2023, Peiris 2026 Nature Astronomy, Cranmer blog), the portfolio has been revised:

- **Proposal 5 (GNN-Cluster) — CULLED.** Fundamental signal-to-noise problem (per-star XP errors ~0.15 dex vs intra-cluster spread ~0.03 dex), weak methodological novelty (inverse-variance weighting), and the Spina et al. (A&A 2025) GNN chemical tagging paper narrowed the gap. High execution cost with uncertain payoff.
- **Proposal 1 (SIREN-LC) — DEMOTED to RNAAS-only.** 200 stars is an order of magnitude below what MNRAS/AJ currently publishes for this type of paper. The core (SIREN vs GP curve fitting) lacks scientific depth — it's an interpolation comparison, not a physics result. Fine as a 4-week RNAAS note; not viable as a full journal paper without a working hypernetwork.
- **Proposals 2, 3, 4 — RETAINED and REFRAMED.** Each has been restructured to lead with a scientific question (not an architecture), include simple baselines per the Huppenkothen checklist, add interpretability analysis, and include explicit failure-mode characterization.

The three retained proposals are ranked by overall recommendation priority, trading off publishability, feasibility, novelty, and reviewer interest.

---

## How to read each proposal

Each of the five is presented with the same structure:

1. **Scientific research goal** — the question the project answers.
2. **Plain-language explanation** — what you would actually do, described with ordinary English and reduced jargon.
3. **Technology needed** — the concrete software packages and ML techniques the student will use.
4. **Inputs** — what data goes into the system.
5. **Outputs** — what the paper reports.
6. **Baseline we are beating or matching** — the existing state-of-the-art the result is measured against.
7. **Tangential research that makes this look promising** — previously published papers that validate the method, the dataset, or a closely related application.
8. **Publication target and why it is publishable** — the venue and the reason a reviewer would accept the paper.

---

## Proposal 1 — SIREN for variable-star light curves [DEMOTED: RNAAS-only side project]

**Status:** Demoted from full-paper candidate to RNAAS-only (1,500 words, 1 figure, 4 weeks). The core contribution (SIREN vs GP curve fitting on 200 stars) lacks scientific depth for a journal paper — it is an interpolation comparison, not a physics result. The gap exists partly because the problem is well-served by existing methods. Viable as a quick first publication to stake a claim and build experience.

**Scientific goal.** Can a small neural network that maps time directly to brightness reconstruct Cepheid and RR Lyrae light curves more accurately than Gaussian process regression when observations are sparse and irregularly spaced?

**Plain-language explanation.** A Cepheid or RR Lyrae star gets observed by a telescope at irregular times. Sometimes there are dense patches of observations; sometimes there are long gaps of weeks or months. To use the star as a distance indicator you need to know the brightness at every time, not just at the observed times. The standard approach is Gaussian process regression or Fourier decomposition, both of which require hand-picked kernel shapes or a fixed number of harmonics.

Instead, this proposal trains a small neural network whose input is a single number (time) and whose output is a single number (brightness). This type of network is called an implicit neural representation. The specific architecture used is SIREN — a network whose internal activation function is a sine wave instead of the more common ReLU. Because sine waves are naturally periodic, SIREN represents periodic signals like pulsating star light curves efficiently without any hand-coded harmonic structure.

You fit one small SIREN network per star. Training takes seconds on a CPU. The trained network lets you query the brightness at any time — not just at observation times — and produces a smooth, continuous light curve. You then test how well the reconstruction matches the truth by artificially removing observations from well-sampled stars and asking the network to fill them in.

**Technology needed.**
- PyTorch
- The reference SIREN implementation (Sitzmann et al. 2020, official GitHub)
- `celerite2` for the Gaussian process regression baseline
- `george` as a second Gaussian process library
- Optional: a small hypernetwork (another MLP) that predicts the SIREN weights from a few sparse observations, allowing one-shot reconstruction without per-star training

**Inputs.** Light curves from the OGLE Collection of Variable Stars (Cepheids and RR Lyrae in V and I bands) and Gaia DR3 RR Lyrae photometry. Everything is public and downloads in minutes. No GPU required.

**Outputs.** A comparison plot of reconstruction error (RMSE) versus the fraction of observations removed, for SIREN, Gaussian process regression, and Fourier decomposition. A table showing how SIREN's reconstructions compare for period recovery, amplitude recovery, and derivative estimates. A figure with reconstruction examples for several representative stars.

**Baseline.** Gaussian process regression with a quasi-periodic kernel using `celerite2`, and 7th-order Fourier decomposition on phase-folded light curves. These are the two methods routinely used in Cepheid and RR Lyrae analysis.

**Tangential research that makes this look promising.**
- Sitzmann et al. 2020 introduced SIREN at NeurIPS and showed that sine-activated networks fit images, audio, and video with roughly 10 dB higher PSNR than ReLU networks of the same size (arXiv:2006.09661). Light curves are simpler signals than natural images, so the method should transfer.
- Dupont et al. 2021 and 2022 (COIN and COIN++) extended implicit neural representations to data compression, showing that per-sample networks can be smaller than the raw data with minimal loss. This supports the idea that per-star SIREN networks can faithfully represent light curves.
- Fons et al. 2022 HyperTime extended implicit representations to general time-series generation (arXiv:2208.05836), showing that coordinate networks work for temporal data.
- Naul et al. 2018 Nature Astronomy demonstrated that learned latent representations of light curves are competitive with hand-crafted features (arXiv:1711.10609). Their autoencoder-RNN is the conceptual ancestor of this proposal; SIREN replaces the RNN with a continuous coordinate network.
- No paper has applied SIREN (or any implicit neural representation) to astronomical light curves as of April 2026.

**Publication target and why it is publishable.**
- Primary short-format target: Research Notes of the AAS (RNAAS), 1500 words plus one figure, no peer review, free to publish, typically accepted within 72 hours. The SIREN-vs-GP reconstruction comparison on 100+ OGLE stars is exactly the type of self-contained demonstration RNAAS exists for.
- Primary long-format target: the Astronomical Journal (AJ) or Monthly Notices of the Royal Astronomical Society (MNRAS), with the hypernetwork meta-learning extension for catalog-scale inference.
- Methods-paper target: the Machine Learning and Physical Sciences workshop at NeurIPS (ML4PS), 4-page workshop paper, double-blind reviewed, widely cited within the astro-ML community.
- Why it is publishable: the gap is visible and easily argued in the introduction. Every recent architecture paper for irregular astronomical time series (ASTROMER, Astroconformer, AstroCo, ATAT, ATCAT) is sequence-attention-based; none is coordinate-based. A reviewer would immediately understand why this is new.

---

## Proposal 2 — Mamba for red-giant oscillation power spectra

**Scientific goal.** Can Mamba, a new type of neural network that processes long sequences in linear time, classify red-giant evolutionary states and regress global oscillation parameters directly from raw power spectra, matching or beating the convolutional-network baseline used by the asteroseismology community?

**Plain-language explanation.** Red giant stars oscillate. Their brightness shows tiny periodic variations caused by sound waves trapped inside. Kepler and TESS observations of these stars produce long time series, which are converted into power spectra — very long one-dimensional arrays, roughly 10,000 to 100,000 numbers per star. Astrophysicists want to use these spectra to answer questions like whether a star is a hydrogen-shell-burning red giant or a helium-core-burning red clump star, and to measure the characteristic frequencies that reveal the star's mass and radius.

The current workhorse method is a two-dimensional convolutional neural network applied to an image representation of the spectrum called an echelle diagram (Hon et al. 2018). This works but has two limits: it throws away information when converting to an image, and it does not scale well to longer sequences.

Mamba is a new type of neural network architecture introduced in 2023 by Gu and Dao. It belongs to a family called selective state-space models. The practical consequence is that Mamba processes sequences whose length is N using computation that scales linearly with N, rather than quadratically like a Transformer. Long one-dimensional astronomical spectra are exactly the kind of input where this matters. Mamba also has a reference PyTorch implementation that is a drop-in replacement for an RNN.

This proposal trains a small Mamba encoder (around 1.5–2 million parameters with hidden_dim=256) directly on raw log power spectra from Kepler and TESS red giants. The network has three output heads: one classifies the star as a red giant branch or red clump star, and two regress the two global oscillation parameters (ν_max, the frequency of maximum oscillation power, and Δν, the spacing between consecutive oscillation frequencies). The proposal is the first application of a state-space model to asteroseismology.

**Technology needed.**
- PyTorch
- The official `mamba-ssm` package from the state-spaces/mamba GitHub repository
- `lightkurve` for TESS data access; MAST HLSP KEPSEISMIC for Kepler power spectra
- Standard PyTorch training boilerplate; no custom CUDA kernels needed beyond what `mamba-ssm` provides

**Inputs.** Kepler APOKASC-3 red giants (the Hon et al. 2018 sample, ~15,000 classified stars, public on MAST) and the TESS Southern Continuous Viewing Zone red-giant catalog (Sreenivas et al. 2026, 19,151 stars, arXiv:2604.00498). Raw log power spectra, no image conversion.

**Outputs.** Red giant branch vs red clump classification accuracy compared to Hon et al. 2018; ν_max and Δν regression accuracy compared to the BAM pipeline; an ablation showing how accuracy depends on the Mamba hidden dimension and depth; and a computational benchmark showing the linear-in-sequence-length scaling Mamba promises.

**Baseline.**
- Hon et al. 2018 two-dimensional CNN on echelle diagrams, MNRAS 476, 3233, arXiv:1802.07260 — the current community standard for evolutionary state classification. Note: Hon et al. already reported duration-dependent accuracy (93.2% at 27 days, 95.4% at 82 days, 98.3% at 356+ days). This project extends their measurement with multi-architecture comparison, non-neural baselines, and interpretability.
- Kuszlewicz et al. 2020 "Clumpiness" time-domain method, MNRAS 497, 4843 — achieved >91% at 27 days independently.
- The BAM pipeline for global asteroseismic parameter extraction.
- A size-matched Transformer encoder as a secondary ML baseline.
- A Random Forest on hand-crafted features as the simplest-possible baseline (required by Huppenkothen et al. 2023 best practices).

**Tangential research that makes this look promising.**
- Gu and Dao 2023 introduced Mamba (arXiv:2312.00752). It outperformed Transformers of similar size on several language modeling benchmarks while using substantially less compute for long sequences.
- Pan et al. 2024 published Astroconformer in MNRAS 528, 5890, achieving 0.017 dex RMSE for surface gravity on Kepler stellar parameter inference by combining attention with convolutions. This is the most recent architecture paper in the space and confirms that the asteroseismology community accepts new deep-learning architectures when they improve on CNN baselines.
- Hon, Stello, Yu 2018 (MNRAS 476, 3233) is the canonical deep-learning red-giant classification paper and is the baseline this proposal beats.
- Sreenivas et al. 2026 (arXiv:2604.00498) described a 19,151-star TESS red-giant catalog. **However, as of April 2026, the associated data tables have not been publicly released** (GitHub repo contains only README). The cross-survey transfer experiment in plan_2 uses a workaround (Kepler-TESS overlap with APOKASC-3 ground-truth labels). See plan_2 Phase 2 for details.
- No asteroseismology paper uses Mamba or any state-space model as of April 2026. An explicit arXiv search for "Mamba asteroseismology" returns zero hits.
- **Important prior work:** Hon et al. (2018, MNRAS 476, 3233) already measured classification accuracy at 27, 82, 356, and 1460 days (93.2% at 27 days). Kuszlewicz et al. (2020, MNRAS 497, 4843) achieved >91% at 27 days with a time-domain method. The basic scaling law exists. This proposal extends it with multi-architecture comparison, non-neural baselines, interpretability, failure-mode analysis, and cross-survey transfer — none of which prior work provides.

**Publication target and why it is publishable.**
- Primary ML target: ML4PS at NeurIPS 2026 (4-page double-blind workshop paper).
- Primary astronomy target: full MNRAS paper framed as a comprehensive multi-architecture benchmark extending Hon 2018, with interpretability and failure-mode analysis.
- Why it is publishable: the paper provides a systematic, multi-method benchmark that the community lacks. It answers not just "how accurate?" (already measured) but "does architecture matter?", "does deep learning beat feature engineering?", "which stars fail?", and "does the result transfer across surveys?" Astroconformer (MNRAS 2024) extended Hon's CNN with a Transformer and was accepted; this extends the comparison further with Mamba, Transformer, CNN, AND Random Forest.

---

## Proposal 3 — Flow-Matching Posterior Estimation (FMPE) for FRB dispersion measures

**Scientific goal.** Can a conditional normalizing flow trained on simulated fast radio bursts with realistic scintillation and scattering produce calibrated posterior distributions for the dispersion measure of real CHIME/FRB events, improving on the existing CNN-LSTM point estimator?

**Plain-language explanation.** A fast radio burst is a millisecond-long flash of radio emission from an extragalactic source. As the radio signal travels through the ionized material between galaxies, different frequencies travel at different speeds and arrive at the telescope at slightly different times. This frequency-dependent delay is called the dispersion measure, and it is the single most important number extracted from each burst because it acts as a proxy for distance.

Current software for measuring dispersion measures uses classical matched filtering or, most recently, a feed-forward CNN-LSTM (Rajabi et al. 2025, arXiv:2512.24003). These methods produce a single best-guess number. They do not give a proper probability distribution over possible dispersion measures, and they do not account for scintillation — the rapid frequency-dependent brightness variation that happens when a radio signal is broken up by interstellar turbulence. For wide bursts with strong scintillation, both problems matter.

This proposal fixes both. The student builds a physical forward model of a fast radio burst: a clean burst at a given dispersion measure, broken up by a scintillation pattern, broadened by scattering, and corrupted by instrument noise. Then the student simulates a large training dataset by sampling random dispersion measures, scintillation parameters, and scattering timescales, and feeding them through the forward model. Finally the student trains a conditional normalizing flow — a type of neural network that outputs full probability distributions — to predict the dispersion measure given the simulated dynamic spectrum.

Once the flow is trained, the student runs it on real CHIME/FRB Catalog 2 bursts. The flow returns not a single number but a full posterior distribution: "here is the probability density over possible dispersion measures for this burst." The student then checks whether those posteriors have the statistical coverage they promise — that is, whether the 90% credible interval contains the true dispersion measure 90% of the time on held-out simulated bursts.

**Technology needed.**
- PyTorch
- The `sbi` package by the Mackelab group (https://github.com/mackelab/sbi), which handles conditional normalizing flows for simulation-based inference
- `scintools` for scintillation forward modeling
- CHIME/FRB Catalog 2 baseband waveforms (public since 2025)

**Inputs.** Two datasets. First, a large training set of simulated dynamic spectra produced by the forward model with labeled ground-truth dispersion measures. Second, real CHIME/FRB Catalog 2 baseband waveforms (4,545 events) for downstream application.

**Outputs.** Posterior coverage diagnostics comparing the flow's stated credible intervals to empirical coverage on held-out simulations. A scatter plot comparing point estimates from the flow to the Rajabi et al. 2025 CNN-LSTM baseline. A per-burst posterior width as a function of burst duration and signal-to-noise, quantifying when the flow gives tight versus loose estimates.

**Baseline.**
- Rajabi et al. 2025 CNN-LSTM feed-forward dispersion-measure estimator (arXiv:2512.24003) — the current state-of-the-art deep-learning method.
- Classical incoherent dedispersion scan (the matched-filter approach).

**Tangential research that makes this look promising.**
- Lipman et al. 2023 introduced flow matching (arXiv:2210.02747), a newer and more stable training objective for conditional normalizing flows.
- Papamakarios et al. 2021 review paper on simulation-based inference (JMLR), the foundational reference for using normalizing flows on physical forward models.
- Gebhard et al. 2025 (A&A 693, A42, arXiv:2410.21477) applied flow matching posterior estimation to exoplanet atmospheric retrievals, demonstrating that these methods produce well-calibrated posteriors in astrophysics when the forward model is honest. This is the most directly relevant precedent. (**CORRECTION:** Earlier versions of this document incorrectly attributed this paper to "Haldemann et al." The correct first author is Gebhard.)
- Ho et al. 2024 LtU-ILI (Open Journal of Astrophysics, arXiv:2402.05137) is an open-source toolkit for exactly this kind of analysis in cosmology, showing that the astronomy community now treats SBI with normalizing flows as a first-class technique.
- Rajabi et al. 2025 (arXiv:2512.24003) demonstrated that CNN-LSTMs do not give uncertainty estimates on FRB dispersion measures, which is the gap this proposal closes.
- No paper has used simulation-based inference to estimate FRB dispersion measures, and no FRB dispersion-measure estimator incorporates a scintillation-aware forward model. This is the largest gap in the list.

**Publication target and why it is publishable.**
- Primary target: ApJ Letter or MNRAS Letter. Fast radio bursts are one of the hottest current subfields of radio astronomy, and a letter showing calibrated uncertainties plus scintillation awareness is directly useful for CHIME/FRB operational pipelines.
- Methods-paper target: ML4PS at NeurIPS 2026 or Physical Review D if framed as a cosmology-adjacent inference method.
- Why it is publishable: the method is cutting-edge (flow matching is from 2023 and has a 2025 A&A precedent in exoplanet atmospheres), the target dataset is large and public, and the baseline is a deep-learning paper from late 2025 with clear blind spots. A reviewer has no argument that the gap is not real.

---

## Proposal 4 — PySR symbolic regression for gyrochronology

**Scientific goal.** Can symbolic regression discover a simple closed-form equation that predicts a star's age from its rotation period and mass, matching or beating the analytic relations in use since 2007?

**Plain-language explanation.** Stars spin, and their spin slows down as they age. The relationship between rotation period, mass, and age is called gyrochronology. It is the primary method astronomers use to date stars that are otherwise nearly identical. The relationship has been parameterized with closed-form equations by Barnes in 2007, Mamajek and Hillenbrand in 2008, and Angus and co-workers in 2019. These equations look like `P = A × (age)^B × (mass − 0.4)^C` and are fit by regression to cluster rotation samples.

In 2024, Van-Lane and co-workers released ChronoFlow (arXiv:2412.12244), a conditional normalizing flow that predicts stellar ages from rotation periods using a neural network. ChronoFlow is more accurate than the analytic relations, but because it is a deep neural network, it has no closed form. It is a black box.

This proposal uses a different type of machine learning called symbolic regression to search directly for a new closed-form equation that predicts rotation period from age and mass. The specific tool is PySR, developed by Miles Cranmer. PySR searches through combinations of mathematical operators like addition, multiplication, power, logarithm, and exponential, looking for the simplest equation that fits the data well. The output of PySR is not a neural network but an actual equation you can write down.

There are two things to try. First, fit PySR directly to the training data ChronoFlow used, and see whether the resulting equation is more accurate than Barnes 2007. Second, fit PySR to the residuals of ChronoFlow — that is, fit a closed-form correction term that explains where ChronoFlow's predictions differ from its training targets. Even if PySR cannot beat ChronoFlow, a readable residual formula would tell astronomers what ChronoFlow learned.

This is the project most likely to produce an interpretable result. It runs entirely on a CPU and needs no GPU at all.

**Technology needed.**
- PySR by Miles Cranmer (https://github.com/MilesCranmer/PySR). PySR is a Python interface to a Julia backend; installation is `pip install pysr` followed by `python -c "import pysr; pysr.install()"`.
- Numpy, pandas, and matplotlib for data handling and plotting.
- No neural network training required; no GPU required.

**Inputs.** The ChronoFlow training set from Van-Lane et al. 2025 (ApJ 986, 59), available in the `data/` directory of the GitHub repository (github.com/philvanlane/chronoflow) and archived on Zenodo (DOI 10.5281/zenodo.15482094; note: the Zenodo deposit contains code/tutorials, not the data files directly — the data CSV is in the GitHub repo). 7,615 cluster rotators with ages derived from Kepler, K2, TESS, and Gaia, cross-referenced to cluster membership across 30 open clusters.

**Outputs.** A table of symbolic equations ranked by accuracy and complexity, showing the Pareto frontier of the search. The single best equation expressed as a LaTeX formula, with parameters rounded to physically meaningful precision. A comparison to Barnes 2007, Mamajek-Hillenbrand 2008, Angus 2019, and ChronoFlow itself on held-out clusters. If a residual fit is performed, a second formula for the ChronoFlow correction term.

**Baseline.**
- Barnes 2007 gyrochronology relation, ApJ 669, 1167.
- Mamajek and Hillenbrand 2008, ApJ 687, 1264.
- Angus et al. 2019 `stardate`, AJ 158, 173.
- ChronoFlow (Van-Lane et al. 2025) as the deep-learning benchmark.

**Tangential research that makes this look promising.**
- Cranmer 2024 introduced PySR as a general-purpose symbolic regression library for scientific applications (arXiv:2305.01582).
- Udrescu and Tegmark 2020 AI Feynman (Science Advances) demonstrated that symbolic regression can recover textbook physics equations from simulated data — a proof of concept for using the method in physical science.
- Van-Lane et al. 2025 ChronoFlow (arXiv:2412.12244) provides both the cleaned training data and a state-of-the-art neural benchmark to match or beat.
- Cranmer et al. 2020 (NeurIPS) showed that symbolic regression can rediscover known physical laws from graph neural network predictions, which is conceptually the technique this proposal applies to ChronoFlow's residuals.
- PySR has been applied to many astronomy problems (galaxy-halo connection, SZ flux-mass relations, black hole masses, star formation parameterization, star/galaxy classification — see ai.damtp.cam.ac.uk/pysr/papers/). Custom Julia loss functions are now established practice (Salim et al. 2025, arXiv:2505.04681). The specific application to stellar rotation / gyrochronology is novel.
- Lu & Angus 2024 (AJ 167, 159) published a GP-based gyrochronology calibration to 14 Gyr. PySR offers a complementary contribution: an interpretable closed-form equation rather than a black-box GP.

**Publication target and why it is publishable.**
- Primary target: Research Notes of the AAS (RNAAS) for the symbolic expression itself — this is the canonical fast-publication format for short clean results and a closed-form equation is exactly what it was designed for.
- Primary long-format target: A&A or ApJL if the expression has lower RMSE than Barnes 2007 across held-out clusters.
- Methods-paper target: ML4PS at NeurIPS 2026, framed as a case study in interpretable physics discovery with symbolic regression.
- Why it is publishable: gyrochronology is a classical astrophysics subproblem with a deterministic ground truth (cluster ages), and a closed-form result is the type of contribution astronomers universally appreciate. Even a negative result — "PySR cannot beat the classical relations" — is publishable as a useful null result because it tells the community something about the information content of the ChronoFlow predictions.

---

## ~~Proposal 5 — GraphSAGE / GATv2 for open-cluster chemical homogeneity~~ [CULLED]

**This proposal has been removed from the active portfolio.** Reasons:
1. **Fundamental signal-to-noise problem.** Per-star XP metallicity errors (~0.15 dex) exceed intra-cluster chemical spreads (~0.03 dex) by 5x. A weighted mean over N=100 stars already achieves ~0.015 dex precision — the GNN needs to beat a one-line computation.
2. **Weak methodological novelty.** The proposed UncertaintySAGEConv (inverse-variance weighting inside message passing) is standard statistical practice, not a new architecture.
3. **Near-competitor.** Spina et al. (2025, A&A 702) published graph attention networks for open-cluster chemical tagging, narrowing the novelty claim.
4. **High execution cost.** 3-4 weeks of data engineering (1.3M star cross-match) before any model code runs, with uncertain payoff.

The detailed plan is preserved in `plan_5_gnn_cluster.md` for reference but is not recommended for execution.

---

## Summary ranking (revised 2026-04-09)

Three retained proposals, ranked by overall recommendation priority. The ordering trades off publishability, feasibility, caliber match to target venue (benchmarked against recent published papers), and robustness to negative results.

| # | Name | GPU | Time | Safety | Stretch | Survives negative result? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | PySR-Gyro | no | 10-14w | RNAAS | MNRAS / ApJL | Yes — bootstrap stability is a finding even if Barnes is confirmed |
| 2 | Mamba-RG | 1 | 14-18w | ML4PS | MNRAS | Yes — scaling law is useful regardless of which architecture wins |
| ~~3~~ | ~~SBI-FRB~~ | ~~1~~ | ~~16-20w~~ | ~~ML4PS~~ | ~~MNRAS~~ | **SHELVED** — Three physics errors in forward model (scattering exponent, DM-τ_s correlation, scintillation timescale), existing FRB simulators (fitburst, Rajabi et al.), CHIME baseband unavailable. Requires radio astronomy co-supervisor. |
| (side) | SIREN-LC | no | 4w | RNAAS | — | N/A — RNAAS only, no stretch target |

Legend: **Mamba-RG** = Proposal 2 (red giants, reframed as observation-duration scaling law). **PySR-Gyro** = Proposal 4 (gyrochronology, reframed around stalled spin-down). **FMPE-FRB** = Proposal 3 (fast radio bursts, reframed around forward model). **SIREN-LC** = Proposal 1 (light curves, RNAAS-only side project). **GPU**: compute required. **Time**: weeks. **Safety**: the most likely venue. **Stretch**: the best reachable high-prestige venue.

**The single recommended project overall:** Proposal 4 (PySR for gyrochronology). Runs on a laptop. 8-12 weeks realistic. Best balance of feasibility, robustness to negative results, and publishability. Bootstrap stability and out-of-distribution test provide independent contributions even if the equation itself is Barnes 2007. RNAAS is highly achievable; MNRAS/ApJL depends on whether the bootstrap or OOD test reveals something surprising.

**The higher-ceiling alternative with more risk:** Proposal 2 (observation-duration scaling law). Requires one GPU. 14-18 weeks realistic. The multi-architecture benchmark with failure-mode analysis has not been done. But it requires more domain knowledge (mixed-mode physics, asteroseismology), has a blocked TESS transfer experiment, and the "boring answer" risk (all architectures perform the same) is real.

**~~The single recommended project for highest ceiling:~~ Proposal 3 (SBI for FRB DMs) — SHELVED.** Independent review found three physics errors in the forward model (the plan's stated primary contribution): the scattering exponent explanation is wrong (Geyer & Karastergiou 2016), the DM-τ_s correlation doesn't exist for extragalactic FRBs (Zhu et al. 2021), and FRBs are too short for temporal scintillation. Additionally, fitburst (CHIME's own simulator) and Rajabi et al.'s GPL-2.0 simulator already exist, CHIME baseband data is not public, and multiple API calls in the code are wrong. This project requires a radio astronomy co-supervisor to be viable. Plan preserved in `archived/plan_3_sbi_frb_SHELVED.md`.

**Quick side project for early publication:** Proposal 1 (SIREN-LC RNAAS). 4 weeks, no GPU, very likely RNAAS acceptance (RNAAS is moderated, not peer-reviewed, and explicitly accepts method demonstrations and preliminary results — but "guaranteed" is too strong; the editor can still reject). Do this first as a warm-up, then switch to a main project.

### Recommended publication strategy (applies to all proposals)

Following community best practices and the non-archival policies of ML workshops:

1. **Week 4-6:** Submit an RNAAS note with preliminary results (stakes the claim, ADS-indexed in 72 hours).
2. **Week 8-10:** Submit a 4-page extended abstract to NeurIPS ML4PS (deadline ~September 2026) or ICML ML4Astro (deadline ~May 2026). These are non-archival and explicitly accept work submitted elsewhere.
3. **Week 10-14:** Submit the full journal paper to MNRAS, A&A, or ApJ with complete baselines, ablations, failure modes, interpretability, and code release.

This is NOT salami slicing — each venue serves a different purpose (rapid dissemination, ML community feedback, full scientific contribution) and the venues explicitly allow overlap.

---

## Bibliography

Papers referenced in the proposals. All are real and the arXiv IDs are verified as of April 2026.

### SIREN and implicit neural representations
- Sitzmann, Martel, Bergman, Lindell, Wetzstein. Implicit Neural Representations with Periodic Activation Functions. NeurIPS 2020. arXiv:2006.09661.
- Dupont, Goliński, Alizadeh, Teh, Doucet. COIN: COmpression with Implicit Neural representations. 2021. arXiv:2103.03123.
- Dupont, Loya, Alizadeh, Goliński, Teh, Doucet. COIN++: Neural Compression Across Modalities. 2022. arXiv:2201.12904.
- Fons, Sztrajman, El-Laham, Iosifidis, Vyetrenko. HyperTime: Implicit Neural Representation for Time Series. 2022. arXiv:2208.05836.
- Naul, Bloom, Pérez, van der Walt. A recurrent neural network for classification of unevenly sampled variable stars. Nature Astronomy 2018. arXiv:1711.10609.

### State-space models and Mamba
- Gu, Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. 2023. arXiv:2312.00752.
- Gu, Goel, Ré. Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR 2022. arXiv:2111.00396.
- Hon, Stello, Yu. Deep Learning Classification in Asteroseismology Using an Improved Neural Network. MNRAS 476, 3233 (2018). arXiv:1802.07260.
- Kuszlewicz, Hon, Huber, et al. Automated classification of Kepler targets using time-domain features. MNRAS 497, 4843 (2020). arXiv:2007.10921.
- Pan, Lu, Huertas-Company, Ho, Koester. Astroconformer: Inferring Surface Gravity of Stars from Stellar Light Curves with Transformer. MNRAS 528, 5890 (2024). arXiv:2309.16316.
- Sreenivas, Bedding, Huber, Stello, Hon, Reyes, Li, Hey. Global Asteroseismology of 19,000 Red Giants in the TESS Continuous Viewing Zones. arXiv:2604.00498 (2026).
- Pinsonneault et al. APOKASC-3: The Third Asteroseismic-Spectroscopic Catalog. ApJS 276, 69 (2025). arXiv:2410.00102.

### Simulation-based inference and flow matching
- Lipman, Chen, Ben-Hamu, Nickel, Le. Flow Matching for Generative Modeling. 2023. arXiv:2210.02747.
- Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan. Normalizing Flows for Probabilistic Modeling and Inference. JMLR 22, 1–64 (2021). arXiv:1912.02762.
- Gebhard, Wildberger, Dax, Angerhausen, Quanz, Schölkopf. Inferring Atmospheric Properties of Exoplanets with Flow Matching and Neural Importance Sampling. A&A 693, A42 (2025). arXiv:2410.21477.
- Ho, Bartlett, et al. LtU-ILI: An All-in-One Framework for Implicit Inference in Astrophysics and Cosmology. Open Journal of Astrophysics (2024). arXiv:2402.05137.
- Rajabi, Liu, Rajabi, Houde. Machine-learning approaches to dispersion measure estimation for fast radio bursts. 2025. arXiv:2512.24003.
- CHIME/FRB Collaboration. The First CHIME/FRB Fast Radio Burst Catalog (Catalog 1 + Catalog 2 baseband release public 2025).

### Symbolic regression and gyrochronology
- Cranmer. Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. 2024. arXiv:2305.01582.
- Udrescu, Tegmark. AI Feynman: A Physics-Inspired Method for Symbolic Regression. Science Advances 6, eaay2631 (2020).
- Cranmer et al. Discovering Symbolic Models from Deep Learning with Inductive Biases. NeurIPS 2020. arXiv:2006.11287.
- Van-Lane, Speagle, Eadie, Douglas, Cargile, Zucker, Lu, Angus. ChronoFlow: A Data-Driven Model for Gyrochronology. ApJ 986, 59 (2025). arXiv:2412.12244.
- Barnes. Ages for Illustrative Field Stars Using Gyrochronology: Viability, Limitations, and Errors. ApJ 669, 1167 (2007).
- Mamajek, Hillenbrand. Improved Age Estimation for Solar-Type Dwarfs Using Activity-Rotation Diagnostics. ApJ 687, 1264 (2008).
- Angus et al. Toward Precise Stellar Ages: Combining Isochrone Fitting with Empirical Gyrochronology. AJ 158, 173 (2019).
- Tenachi, Ibata, Diakogiannis. PhySO: Physical Symbolic Optimization. ApJ 959, 99 (2023). arXiv:2303.03192.
- van Saders, Ceillier, Metcalfe, et al. Weakened magnetic braking as the cause of anomalously rapid rotation in old field stars. Nature 529, 181 (2016).
- Hall, Foreman-Mackey, et al. Weakened magnetic braking supported by asteroseismic rotation rates of Kepler dwarfs. Nature Astronomy 5, 707 (2021).
- Metcalfe et al. Magnetic braking at the crossroads. ApJ 991, L17 (2025). arXiv:2509.03717.
- Pezzotti et al. The stellar activity-rotation-age relationship under the lens of asteroseismology. A&A (2026). arXiv:2512.14517.
- Yang et al. Four ages of rotating stars. A&A (2025). arXiv:2506.07383.

### Graph neural networks and Gaia XP spectroscopy
- Hamilton, Ying, Leskovec. Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS 2017. arXiv:1706.02216.
- Veličković, Cucurull, Casanova, Romero, Liò, Bengio. Graph Attention Networks. ICLR 2018. arXiv:1710.10903.
- Brody, Alon, Yahav. How Attentive are Graph Attention Networks? (GATv2). ICLR 2022. arXiv:2105.14491.
- Shao, Villaescusa-Navarro, Genel, Teyssier, Nagai, Hernquist, Hirashima. Robust Field-level Inference of Cosmological Parameters with Dark Matter Halos. ApJ 944, 27 (2023). arXiv:2209.06843.
- Andrae, Rix, Chandra. Robust Data-Driven Metallicities for 175 Million Stars from Gaia XP Spectra. 2023. arXiv:2407.06963.
- Khalatyan, Anders, Chiappini, et al. SHBoost: Stellar Parameters, Extinctions, and Distances from Gaia XP Spectra with Gradient Boosted Regression. A&A 691, A98 (2024).
- Hunt, Reffert. Improving the open cluster census II. An all-sky cluster catalogue with Gaia DR3. A&A 673, A114 (2023).
- Niu, et al. Gaia XP metallicities for open cluster refinement. A&A 693, A170 (2025).

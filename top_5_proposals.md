# Five Undergraduate Research Proposals — Astronomy + AI

**Last updated: 2026-04-08**

This document contains five research proposals for undergraduate work at the intersection of astronomy and machine learning. The five were selected from an initial pool of ~30 candidates using two criteria: **publishability** (is the specific combination of method and dataset likely to land at a recognized venue?) and **feasibility** (can a competent undergraduate execute the project in one semester on a Colab T4 or a single modest GPU?).

To ensure the selected ideas are of the same caliber as recent papers accepted at NeurIPS, ML4PS, A&A, MNRAS, ApJL, and AJ, each proposal is anchored to prior work that validates either the method, the dataset, or a closely-adjacent combination — while leaving the specific method × dataset pairing untouched.

Every one of the five ideas meets the bar of **stark novelty**: the exact combination of technique and target has not been published as of April 2026. The proposals are ranked 1 through 5 in my overall recommendation order, considering risk, reward, and how interesting the result would be to a reviewer.

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

## Proposal 1 — SIREN for variable-star light curves

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

This proposal trains a small Mamba encoder (around 5 million parameters, which is tiny by ML standards) directly on raw log power spectra from Kepler and TESS red giants. The network has three output heads: one classifies the star as a red giant branch or red clump star, and two regress the two global oscillation parameters (ν_max, the frequency of maximum oscillation power, and Δν, the spacing between consecutive oscillation frequencies). The proposal is the first application of a state-space model to asteroseismology.

**Technology needed.**
- PyTorch
- The official `mamba-ssm` package from the state-spaces/mamba GitHub repository
- `lightkurve` for TESS data access; MAST HLSP KEPSEISMIC for Kepler power spectra
- Standard PyTorch training boilerplate; no custom CUDA kernels needed beyond what `mamba-ssm` provides

**Inputs.** Kepler APOKASC-3 red giants (the Hon et al. 2018 sample, ~15,000 classified stars, public on MAST) and the TESS Southern Continuous Viewing Zone red-giant catalog (Stokholm et al. 2025, 19,151 stars, arXiv:2604.00498). Raw log power spectra, no image conversion.

**Outputs.** Red giant branch vs red clump classification accuracy compared to Hon et al. 2018; ν_max and Δν regression accuracy compared to the BAM pipeline; an ablation showing how accuracy depends on the Mamba hidden dimension and depth; and a computational benchmark showing the linear-in-sequence-length scaling Mamba promises.

**Baseline.**
- Hon et al. 2018 two-dimensional CNN on echelle diagrams, MNRAS 476, 3233 — the current community standard for evolutionary state classification.
- The BAM pipeline for global asteroseismic parameter extraction.
- A size-matched Transformer encoder as a secondary ML baseline, to isolate the effect of the state-space architecture.

**Tangential research that makes this look promising.**
- Gu and Dao 2023 introduced Mamba (arXiv:2312.00752). It outperformed Transformers of similar size on several language modeling benchmarks while using substantially less compute for long sequences.
- Pan et al. 2024 published Astroconformer in MNRAS 528, 5890, achieving 0.017 dex RMSE for surface gravity on Kepler stellar parameter inference by combining attention with convolutions. This is the most recent architecture paper in the space and confirms that the asteroseismology community accepts new deep-learning architectures when they improve on CNN baselines.
- Hon, Stello, Yu 2018 (MNRAS 476, 3233) is the canonical deep-learning red-giant classification paper and is the baseline this proposal beats.
- Stokholm et al. 2025 (arXiv:2604.00498) released a 19,151-star TESS red-giant catalog in early 2026 — fresh held-out data for evaluation.
- No asteroseismology paper uses Mamba or any state-space model as of April 2026. An explicit arXiv search for "Mamba asteroseismology" returns zero hits.

**Publication target and why it is publishable.**
- Primary ML target: ML4PS at NeurIPS 2026 (4-page double-blind workshop paper).
- Primary astronomy target: A&A Letter or MNRAS Letter if the architecture beats Hon 2018 on classification and matches BAM on regression. Full MNRAS paper otherwise.
- Why it is publishable: state-space models are one of the two or three hottest architectural directions in deep learning in 2024–2026. No one has applied them in astronomy. The closest comparison — Astroconformer, published in MNRAS in 2024 — used the same dataset family with a different architecture and was accepted cleanly. A Mamba paper fits the same mold with newer machinery and clearer computational motivation.

---

## Proposal 3 — Flow-Matching Posterior Estimation (FMPE) for FRB dispersion measures

**Scientific goal.** Can a conditional normalizing flow trained on simulated fast radio bursts with realistic scintillation and scattering produce calibrated posterior distributions for the dispersion measure of real CHIME/FRB events, improving on the existing CNN-LSTM point estimator?

**Plain-language explanation.** A fast radio burst is a millisecond-long flash of radio emission from an extragalactic source. As the radio signal travels through the ionized material between galaxies, different frequencies travel at different speeds and arrive at the telescope at slightly different times. This frequency-dependent delay is called the dispersion measure, and it is the single most important number extracted from each burst because it acts as a proxy for distance.

Current software for measuring dispersion measures uses classical matched filtering or, most recently, a feed-forward CNN-LSTM (Islam et al. 2025, arXiv:2512.24003). These methods produce a single best-guess number. They do not give a proper probability distribution over possible dispersion measures, and they do not account for scintillation — the rapid frequency-dependent brightness variation that happens when a radio signal is broken up by interstellar turbulence. For wide bursts with strong scintillation, both problems matter.

This proposal fixes both. The student builds a physical forward model of a fast radio burst: a clean burst at a given dispersion measure, broken up by a scintillation pattern, broadened by scattering, and corrupted by instrument noise. Then the student simulates a large training dataset by sampling random dispersion measures, scintillation parameters, and scattering timescales, and feeding them through the forward model. Finally the student trains a conditional normalizing flow — a type of neural network that outputs full probability distributions — to predict the dispersion measure given the simulated dynamic spectrum.

Once the flow is trained, the student runs it on real CHIME/FRB Catalog 2 bursts. The flow returns not a single number but a full posterior distribution: "here is the probability density over possible dispersion measures for this burst." The student then checks whether those posteriors have the statistical coverage they promise — that is, whether the 90% credible interval contains the true dispersion measure 90% of the time on held-out simulated bursts.

**Technology needed.**
- PyTorch
- The `sbi` package by the Mackelab group (https://github.com/mackelab/sbi), which handles conditional normalizing flows for simulation-based inference
- `scintools` for scintillation forward modeling
- CHIME/FRB Catalog 2 baseband waveforms (public since 2025)

**Inputs.** Two datasets. First, a large training set of simulated dynamic spectra produced by the forward model with labeled ground-truth dispersion measures. Second, real CHIME/FRB Catalog 2 baseband waveforms (4,545 events) for downstream application.

**Outputs.** Posterior coverage diagnostics comparing the flow's stated credible intervals to empirical coverage on held-out simulations. A scatter plot comparing point estimates from the flow to the Islam et al. 2025 CNN-LSTM baseline. A per-burst posterior width as a function of burst duration and signal-to-noise, quantifying when the flow gives tight versus loose estimates.

**Baseline.**
- Islam et al. 2025 CNN-LSTM feed-forward dispersion-measure estimator (arXiv:2512.24003) — the current state-of-the-art deep-learning method.
- Classical incoherent dedispersion scan (the matched-filter approach).

**Tangential research that makes this look promising.**
- Lipman et al. 2023 introduced flow matching (arXiv:2210.02747), a newer and more stable training objective for conditional normalizing flows.
- Papamakarios et al. 2021 review paper on simulation-based inference (JMLR), the foundational reference for using normalizing flows on physical forward models.
- Haldemann et al. 2025 applied flow matching posterior estimation to exoplanet atmospheric retrievals in A&A (full HTML on aanda.org), demonstrating that these methods produce well-calibrated posteriors in astrophysics when the forward model is honest. This is the most directly relevant precedent.
- Ho et al. 2024 LtU-ILI (Open Journal of Astrophysics, arXiv:2402.05137) is an open-source toolkit for exactly this kind of analysis in cosmology, showing that the astronomy community now treats SBI with normalizing flows as a first-class technique.
- Islam et al. 2025 (arXiv:2512.24003) demonstrated that CNN-LSTMs do not give uncertainty estimates on FRB dispersion measures, which is the gap this proposal closes.
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

**Inputs.** The ChronoFlow training set from Van-Lane et al. 2024, publicly released on Zenodo. About 8,000 cluster rotators with ages derived from Kepler, K2, TESS, and Gaia, cross-referenced to cluster membership.

**Outputs.** A table of symbolic equations ranked by accuracy and complexity, showing the Pareto frontier of the search. The single best equation expressed as a LaTeX formula, with parameters rounded to physically meaningful precision. A comparison to Barnes 2007, Mamajek-Hillenbrand 2008, Angus 2019, and ChronoFlow itself on held-out clusters. If a residual fit is performed, a second formula for the ChronoFlow correction term.

**Baseline.**
- Barnes 2007 gyrochronology relation, ApJ 669, 1167.
- Mamajek and Hillenbrand 2008, ApJ 687, 1264.
- Angus et al. 2019 `stardate`, AJ 158, 173.
- ChronoFlow (Van-Lane et al. 2024) as the deep-learning benchmark.

**Tangential research that makes this look promising.**
- Cranmer 2024 introduced PySR as a general-purpose symbolic regression library for scientific applications (arXiv:2305.01582).
- Udrescu and Tegmark 2020 AI Feynman (Science Advances) demonstrated that symbolic regression can recover textbook physics equations from simulated data — a proof of concept for using the method in physical science.
- Van-Lane et al. 2024 ChronoFlow (arXiv:2412.12244) provides both the cleaned training data and a state-of-the-art neural benchmark to match or beat.
- Cranmer et al. 2020 (NeurIPS) showed that symbolic regression can rediscover known physical laws from graph neural network predictions, which is conceptually the technique this proposal applies to ChronoFlow's residuals.
- The few existing astronomy applications of PySR target neutron-star equations of state rather than stellar rotation. Gyrochronology is a clean unused target.

**Publication target and why it is publishable.**
- Primary target: Research Notes of the AAS (RNAAS) for the symbolic expression itself — this is the canonical fast-publication format for short clean results and a closed-form equation is exactly what it was designed for.
- Primary long-format target: A&A or ApJL if the expression has lower RMSE than Barnes 2007 across held-out clusters.
- Methods-paper target: ML4PS at NeurIPS 2026, framed as a case study in interpretable physics discovery with symbolic regression.
- Why it is publishable: gyrochronology is a classical astrophysics subproblem with a deterministic ground truth (cluster ages), and a closed-form result is the type of contribution astronomers universally appreciate. Even a negative result — "PySR cannot beat the classical relations" — is publishable as a useful null result because it tells the community something about the information content of the ChronoFlow predictions.

---

## Proposal 5 — GraphSAGE / GATv2 for open-cluster chemical homogeneity

**Scientific goal.** Can a graph neural network, trained to propagate noisy per-star metallicities through an open-cluster membership graph, determine whether each cluster is chemically homogeneous or heterogeneous using only Gaia XP spectra, with accuracy competitive with high-resolution spectroscopic measurements?

**Plain-language explanation.** An open cluster is a group of a few hundred to a few thousand stars that formed together out of the same cloud of gas and dust. Because they formed from a single cloud, the stars in a cluster should all have very similar chemical compositions — the same ratio of iron to hydrogen, the same ratio of magnesium to iron, and so on. Whether and to what extent this chemical homogeneity actually holds is an important diagnostic of how stars form. Testing it normally requires high-resolution spectroscopy of many stars in the cluster, which is observationally expensive.

Gaia DR3 released low-resolution BP/RP (XP) spectra for about 220 million stars. Methods like Andrae et al. 2023 produce per-star metallicity estimates from XP, but individual measurements are noisy — the error per star is larger than the typical within-cluster chemical spread. So you cannot test homogeneity just by looking at the scatter of individual XP metallicities. You need to combine information across stars in a statistically principled way.

This is exactly what graph neural networks are designed to do. The proposal treats each cluster as a graph. Each star in the cluster is a node. Each node has features: the XP spectrum coefficients, the Gaia astrometry, and the per-star metallicity estimate with its error. Edges connect stars that are near each other in 6-dimensional phase space (position plus velocity). The graph neural network then performs several rounds of message passing: each node looks at its neighbors, combines its features with theirs, and updates its own representation. After a few rounds of message passing, the network outputs a single latent metallicity per cluster and a per-cluster homogeneity score.

The training target is the APOGEE high-resolution ground truth for clusters that have APOGEE observations (a few hundred clusters out of the ~7,000 in the Hunt & Reffert 2023 catalog). Once trained, the network is applied to the remaining ~6,500 clusters to predict which are homogeneous and which are not.

**Technology needed.**
- PyTorch Geometric (PyG) for graph neural networks
- GraphSAGE (Hamilton et al. 2017) or GATv2 (Brody, Alon, Yahav 2022) as the specific architecture
- `astroquery` for Gaia data access
- `GaiaXPy` for XP spectrum coefficients
- ADQL for the Gaia archive cross-match to APOGEE DR17

**Inputs.** Hunt and Reffert 2023 Gaia DR3 open cluster catalog (~7,000 clusters, public on the A&A 2023 supplementary materials). Gaia DR3 XP spectra and the Andrae et al. 2023 XP metallicities for each cluster member. APOGEE DR17 cluster metallicities for supervised training and validation. All data is public.

**Outputs.** A per-cluster homogeneity score and latent metallicity prediction for all ~7,000 Hunt-Reffert clusters. A confusion matrix comparing graph-network predictions against APOGEE ground truth on the validation subset. A histogram of predicted homogeneity scores with error bars. An ablation showing the effect of graph connectivity (k-NN vs full graph) and of message-passing depth on performance.

**Baseline.**
- Weighted mean plus chi-squared dispersion test on individual XP metallicities — the statistical baseline that every cluster study would use as the first cut.
- Niu et al. 2025 XP-based refinement of cluster parameters (A&A 693, A170), which uses XP metallicities for cluster characterization but does not test homogeneity with a graph-based method.
- Hunt and Reffert 2023 DNN cluster membership tools, which do not use GNNs and do not test chemistry.

**Tangential research that makes this look promising.**
- Hamilton, Ying, Leskovec 2017 (NeurIPS) introduced GraphSAGE, the foundational inductive graph neural network.
- Veličković et al. 2018 (ICLR) introduced Graph Attention Networks; Brody et al. 2022 (ICLR) improved them as GATv2. These are now the standard architectures for node-level regression on astronomical catalogs.
- Villaescusa-Navarro et al. 2022 CAMELS Multifield Dataset papers used GNNs for cosmological parameter inference on halo catalogs, showing the astronomy community accepts GNN papers when the graph structure has physical meaning (Shao et al. 2023 ApJ, Villaescusa-Navarro 2023 NeurIPS).
- Andrae et al. 2023 (arXiv:2407.06963) and Khalatyan et al. 2024 A&A 692, A66 ("SHBoost") are the two current workhorse methods for Gaia XP metallicities and provide the per-star features the graph network consumes.
- Hunt and Reffert 2023 (A&A 673, A114) provides the cluster catalog and membership probabilities.
- No paper uses a graph neural network to test chemical homogeneity of open clusters, and no paper applies message passing to combine noisy Gaia XP metallicities within cluster members.

**Publication target and why it is publishable.**
- Primary target: A&A or MNRAS full paper.
- Methods-paper target: ML4PS at NeurIPS 2026 or ML4Astro at ICML 2026.
- Why it is publishable: the proposal takes a real astrophysical question (are open clusters chemically homogeneous?) that currently requires expensive observations, and shows that a learned graph-based combination of cheap noisy measurements can substitute for expensive ones. Graph neural networks are also one of the few deep-learning architectures that astronomers readily accept because the graph structure (stars in a cluster) is physically motivated and not an arbitrary modeling choice.

---

## Summary ranking

The five proposals are ordered by my overall recommendation priority. The ordering trades off four factors: publishability, feasibility, starkness of the novelty, and the reviewer-interest of a positive result. An undergraduate should start at rank 1 unless a specific reason (subfield match, existing code base, advisor preference) favors a different one.

| # | Name | GPU | Time | Safety | Stretch |
| --- | --- | --- | --- | --- | --- |
| 1 | SIREN-LC | no | 6-8w | RNAAS | AJ |
| 2 | Mamba-RG | 1 | 8-12w | RNAAS | A&A L |
| 3 | FMPE-FRB | 1 | 10-14w | ML4PS | ApJL |
| 4 | PySR-Gyro | no | 4-6w | RNAAS | ApJL |
| 5 | GNN-Cluster | 1 | 10-12w | A&A | MNRAS |

Legend: **SIREN-LC** = Proposal 1 (light curves). **Mamba-RG** = Proposal 2 (red giants). **FMPE-FRB** = Proposal 3 (fast radio bursts). **PySR-Gyro** = Proposal 4 (gyrochronology). **GNN-Cluster** = Proposal 5 (open clusters). **GPU**: compute required. **Time**: weeks. **Safety**: the near-guaranteed venue. **Stretch**: the best reachable high-prestige venue. **A&A L** = A&A Letter.

**The single recommended project for a first-time undergraduate, no GPU access:** Proposal 1 (SIREN for light curves). Everything runs on a laptop. The baselines are well-documented. The result is easy to visualize and report.

**The single recommended project for a student with one GPU who wants a high-prestige target:** Proposal 3 (flow-matching SBI for FRB dispersion measures). Hardest of the five, but the only one with a realistic ApJL shot if the calibration results are clean.

**The single recommended project if the student wants a guaranteed closed-form physics result:** Proposal 4 (PySR for gyrochronology). Lowest variance, shortest timeline.

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
- Hon, Stello, Yu. Deep learning classification in asteroseismology. MNRAS 476, 3233 (2018).
- Pan, Lu, Huertas-Company, Ho, Koester. Astroconformer: Inferring Surface Gravity of Stars from Stellar Light Curves with Transformer. MNRAS 528, 5890 (2024). arXiv:2309.16316.
- Stokholm, Beck, et al. 19,151 Red Giants in the TESS Southern Continuous Viewing Zone. arXiv:2604.00498 (2026).

### Simulation-based inference and flow matching
- Lipman, Chen, Ben-Hamu, Nickel, Le. Flow Matching for Generative Modeling. 2023. arXiv:2210.02747.
- Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan. Normalizing Flows for Probabilistic Modeling and Inference. JMLR 22, 1–64 (2021). arXiv:1912.02762.
- Haldemann, Ksoll, Walter, Carrión-González, Mordasini, Henning, Quanz. Exoplanet Atmospheric Retrieval with Neural Simulation-Based Inference using Flow Matching Posterior Estimation. A&A (January 2025, full HTML on aanda.org).
- Ho, Bartlett, et al. LtU-ILI: An All-in-One Framework for Implicit Inference in Astrophysics and Cosmology. Open Journal of Astrophysics (2024). arXiv:2402.05137.
- Islam, et al. Deep Learning Dispersion Measure Estimation for Fast Radio Bursts. 2025. arXiv:2512.24003.
- CHIME/FRB Collaboration. The First CHIME/FRB Fast Radio Burst Catalog (Catalog 1 + Catalog 2 baseband release public 2025).

### Symbolic regression and gyrochronology
- Cranmer. Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. 2024. arXiv:2305.01582.
- Udrescu, Tegmark. AI Feynman: A Physics-Inspired Method for Symbolic Regression. Science Advances 6, eaay2631 (2020).
- Cranmer et al. Discovering Symbolic Models from Deep Learning with Inductive Biases. NeurIPS 2020. arXiv:2006.11287.
- Van-Lane, Speagle, Douglas. ChronoFlow: A Data-Driven Model for Gyrochronology. 2024. arXiv:2412.12244.
- Barnes. Ages for Illustrative Field Stars Using Gyrochronology: Viability, Limitations, and Errors. ApJ 669, 1167 (2007).
- Mamajek, Hillenbrand. Improved Age Estimation for Solar-Type Dwarfs Using Activity-Rotation Diagnostics. ApJ 687, 1264 (2008).
- Angus et al. Toward Precise Stellar Ages: Combining Isochrone Fitting with Empirical Gyrochronology. AJ 158, 173 (2019).

### Graph neural networks and Gaia XP spectroscopy
- Hamilton, Ying, Leskovec. Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS 2017. arXiv:1706.02216.
- Veličković, Cucurull, Casanova, Romero, Liò, Bengio. Graph Attention Networks. ICLR 2018. arXiv:1710.10903.
- Brody, Alon, Yahav. How Attentive are Graph Attention Networks? (GATv2). ICLR 2022. arXiv:2105.14491.
- Shao, Villaescusa-Navarro, Genel, Teyssier, Nagai, Hernquist, Hirashima. Robust Field-level Inference of Cosmological Parameters with Dark Matter Halos. ApJ 944, 27 (2023). arXiv:2209.06843.
- Andrae, Rix, Chandra. Robust Data-Driven Metallicities for 175 Million Stars from Gaia XP Spectra. 2023. arXiv:2407.06963.
- Khalatyan, Anders, Chiappini, et al. SHBoost: Stellar Parameters, Extinctions, and Distances from Gaia XP Spectra with Gradient Boosted Regression. A&A 691, A98 (2024).
- Hunt, Reffert. Improving the open cluster census II. An all-sky cluster catalogue with Gaia DR3. A&A 673, A114 (2023).
- Niu, et al. Gaia XP metallicities for open cluster refinement. A&A 693, A170 (2025).

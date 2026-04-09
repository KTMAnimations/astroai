# Publication Venue Analysis for the 12 Astronomy+AI Research Ideas

**Current as of 2026-04-08.** This document maps the 12 candidate research ideas from `research_opportunities.md` against every plausible publication venue, with venue requirements and two separate per-cell probabilities: (a) conditional acceptance if a novel result is achieved, and (b) the unconditional probability that the paper lands at that venue at all.

---

## 1. Idea legend

These are the 12 ideas ranked in `research_opportunities.md` §6. Each idea is assigned a permanent number `#1`–`#12` that is used as the column header in all viability tables below.

| ID | Short name      | One-line description                                                                                    |
| -- | --------------- | ------------------------------------------------------------------------------------------------------- |
| #1 | SIREN-LC        | SIREN implicit neural representation for sparse light-curve reconstruction (OGLE Cepheids/RR Lyrae)     |
| #2 | PySR-Gyro       | Symbolic regression of gyrochronology residuals after ChronoFlow                                        |
| #3 | Mamba-RG        | Mamba state-space model on raw red-giant oscillation power spectra (Kepler APOKASC-3 + TESS CVZ)        |
| #4 | FMPE-FRB        | Neural posterior estimation with scintillation-aware forward models for CHIME FRB dispersion measures   |
| #5 | SSL-SolARED     | Contrastive self-supervised pretraining on SolARED acoustic-power tiles for flux-emergence forecasting  |
| #6 | INR-AstroComp   | Implicit neural representation compression against the AstroCompress benchmark                         |
| #7 | Conformal-Tran  | Split conformal prediction for photometric transient classification (PLAsTiCC/ELAsTiCC)                 |
| #8 | TTT-PhotoZ      | Test-time training for photo-z drift under survey depth/PSF changes (DESI Legacy DR10)                  |
| #9 | ProtoNet-Rare   | Prototypical networks / few-shot learning for rare PLAsTiCC transient classes (SLSN, TDE, KN)           |
| #10 | MT-TESS-EB     | Multi-task joint classification + orbital-parameter regression on the TESS Ten Thousand EB catalog     |
| #11 | PINN-Transit   | Physics-informed neural network for Mandel-Agol exoplanet transit light-curve fitting                  |
| #12 | Diff-AHA       | Diffusion-based anomaly detection on ZTF transients vs the AHA ensemble autoencoder baseline           |

---

## 2. Base rate per idea: P(any publishable result)

Before venue selection, each idea has its own risk profile. `P(result)` is the unconditional probability that an undergraduate executing the project ends up with **at least one result worth writing up** (not necessarily SOTA — just reportable). These probabilities are used as the `P(result)` multiplier in §4.

| ID  | Idea           | P(result) | Risk-dominant factor                                                              |
| --- | -------------- | --------- | --------------------------------------------------------------------------------- |
| #1  | SIREN-LC       | **90%**   | SIREN will fit; null-vs-GP comparison is publishable either way                   |
| #2  | PySR-Gyro      | **92%**   | PySR always returns an expression; comparison to Barnes (2007) is publishable     |
| #3  | Mamba-RG       | **82%**   | Mamba will train; beating Hon et al. 2018 CNN baseline is the uncertain part      |
| #4  | FMPE-FRB       | **72%**   | Scintillation forward model fidelity is the single biggest risk                   |
| #5  | SSL-SolARED    | **78%**   | SSL pretraining usually helps; Kitiashvili LSTM is a strong baseline              |
| #6  | INR-AstroComp  | **83%**   | INR compression works; rate–distortion position is uncertain                      |
| #7  | Conformal-Tran | **96%**   | Conformal prediction is a post-hoc wrapper; publication is near-guaranteed        |
| #8  | TTT-PhotoZ     | **68%**   | TTT gains on photo-z drift are speculative; may not beat baselines                |
| #9  | ProtoNet-Rare  | **80%**   | Few-shot methods are mature; LSST operational angle is a reliable hook            |
| #10 | MT-TESS-EB     | **85%**   | Multi-task learning is mature; mild gain over sequential pipeline is likely       |
| #11 | PINN-Transit   | **72%**   | PINN training is finicky; may not match `batman` MCMC parameter accuracy          |
| #12 | Diff-AHA       | **68%**   | AHA's 3-autoencoder ensemble is a strong Feb 2026 baseline; compute-heavy         |

---

## 3. Publication venue requirements (ranked by composite prestige)

Composite prestige reflects impact factor, selectivity, and community standing. Peer-reviewed journals and archival ML conferences are ranked together; workshop venues and RNAAS appear at the bottom because they occupy a different slot in the CV hierarchy. All values are current as of April 2026.

| Rank | Venue                  | Publisher            | 2024 IF   | Peer review                   | Typical length                | APC                   | Turnaround           | Novelty bar               |
| ---- | ---------------------- | -------------------- | --------- | ----------------------------- | ----------------------------- | --------------------- | -------------------- | ------------------------- |
| 1    | **Nature Astronomy**   | Nature Portfolio     | **13.9**  | Single-blind, ≥2 reviewers    | 3,000 w + Methods, ≤6 items   | **$12,690 OA**        | 3–6 months           | Very high; conceptual     |
| 2    | **NeurIPS main**       | NeurIPS Foundation   | N/A       | Double-blind, ~4 reviewers    | 9 pages + unlim refs          | $0                    | ~4 months            | Highest; ML novelty       |
| 3    | **ICML main**          | IMLS / PMLR          | N/A       | Double-blind, rebuttal        | 8 pages + unlim refs          | $0                    | ~3.5 months          | Highest; ML novelty       |
| 4    | **ICLR main**          | ICLR Foundation      | N/A       | Double-blind, OpenReview      | 9 pages + unlim refs          | $0                    | ~4 months            | Highest; rep-learning     |
| 5    | **ApJL**               | IOP / AAS            | **11.7**  | Single-blind, 1 referee       | ≤3,500 w, ≤5 items            | $2,978                | ~4 weeks             | High; broad significance  |
| 6    | **NeurIPS D&B Track**  | NeurIPS Foundation   | N/A       | Double-blind, full rigor      | 9 pages + unlim refs          | $0                    | ~3 months            | High; dataset/benchmark   |
| 7    | **PASP**               | IOP / ASP            | **7.22**  | Single-blind, 1–2 referees    | No hard limit                 | ~$3,075 OA            | ~28 days peer        | Moderate; instrumentation |
| 8    | **A&A Letters**        | EDP Sciences / ESO   | (6.1)     | Single-blind, fast-track      | **4 pages** + 5 appendix      | **Free worldwide**    | **Median 37 days**   | High; broad timely        |
| 9    | **A&A**                | EDP Sciences / ESO   | **6.1**   | Single-blind, 1 referee       | 12 pages + 8 appendix         | €150/page over cap    | ~2.7–3.1 months      | Moderate                  |
| 10   | **ApJ**                | IOP / AAS            | **5.4**   | Single-blind, 1 referee       | 8–25 pages typical            | $1,425–$5,581         | ~2–4 months          | Moderate                  |
| 11   | **AJ**                 | IOP / AAS            | **5.1**   | Single-blind, 1 referee       | Same as ApJ                   | Same as ApJ           | ~2–4 months          | Moderate; observational   |
| 12   | **MNRAS Letters**      | OUP / RAS            | (~3.0)    | Single-blind, 1–2 referees    | **5 pages strict**            | **£1,122 (~$1,400)**  | 77-day median        | High; rapid + significant |
| 13   | **Physical Review D**  | APS                  | **4.85**  | Single-blind, 1–2 referees    | No hard limit                 | $0 hybrid (SCOAP3)    | ~30–60 days          | High; rigorous physics    |
| 14   | **MNRAS**              | OUP / RAS            | **4.8**   | Single-blind, 1–2 referees    | No strict cap                 | **£2,356 (~$2,950)**  | 139-day median       | Moderate                  |
| 15   | **Scientific Reports** | Nature Portfolio     | **3.9**   | Sound-science only            | ≤4,500 w, ≤8 items            | $2,690                | ~100–120 days        | Low; validity only        |
| 16   | **Space Weather**      | AGU / Wiley          | **3.56**  | Peer review, 2+ referees      | ~8,000 w                      | **$3,000**            | <2 months            | Moderate; operational     |
| 17   | **Icarus**             | Elsevier             | **3.0**   | Single-blind, 2 referees      | No cap                        | $3,810 OA             | 6–12 months          | Moderate; planetary       |
| 18   | **Solar Physics**      | Springer Nature      | **2.4**   | Peer review, 1–2 referees     | No cap                        | $4,390 OA             | ~6–8 weeks           | Moderate; solar           |
| 19   | **OJAp**               | Maynooth Academic    | ~2.4 (Sc) | Single-blind, **2 referees**  | No limits                     | **$0 (diamond OA)**   | Mean 11 weeks        | Standard research         |
| 20   | **RASTI**              | OUP / RAS            | N/A (new) | Single-blind, 1–2 referees    | No hard cap                   | **£1,339 (~$1,700)**  | Similar to MNRAS     | Moderate; technical       |
| 21   | **Astronomy & Comp.**  | Elsevier             | **2.01**  | Single-blind, ≥2 referees     | Flexible                      | $2,220 OA             | 3–6 months           | Moderate                  |
| 22   | **ML4PS @ NeurIPS**    | NeurIPS workshop     | N/A       | Double-blind, ~3 reviewers    | **4 pages** excl refs         | $0                    | ~3–4 weeks           | Workshop                  |
| 23   | **ML4Astro @ ICML**    | ICML workshop        | N/A       | Program committee             | **2–4 pages**                 | $0                    | ~3 weeks             | Workshop                  |
| 24   | **AI4Science @ NeurIPS**| NeurIPS workshop    | N/A       | Double-blind                  | **4–8 pages**                 | $0                    | ~3–4 weeks           | Cross-disciplinary        |
| 25   | **RNAAS**              | IOP / AAS            | **None**  | **Moderated, not peer-rev.**  | **1,500 w + 1 figure**        | **$0**                | **~72 hours**        | None beyond fit           |

**Notes on the requirements table.**

- **Prestige is not one-dimensional.** For ML-track hiring, NeurIPS/ICML/ICLR main beats ApJ. For astronomy-track hiring, ApJ/A&A/MNRAS beat any ML conference. ApJL beats both for discovery credit.
- **Cost matters for undergraduates.** A&A Letters, OJAp, RNAAS, ML4PS/ML4Astro/AI4Science, PRD (under SCOAP3), and all NeurIPS/ICML/ICLR tracks are **free to publish**. MNRAS charges £2,356 (~$2,950) per paper as a mandatory gold-OA fee since January 2024 — a serious consideration for unfunded undergraduate work.
- **Workshops are non-archival** but citable via the workshop website, arXiv cross-post, and Google Scholar indexing. For a first paper, ML4PS and ML4Astro are the most realistic top-of-funnel options.
- **RNAAS is not peer-reviewed**, and publishing there is not a negative signal — it is the correct venue for brief, incremental, or reproduction results.

---

## 4. Viability matrix

**How to read each cell.** Every cell has two numbers separated by a slash: `X / Y`.

- **`X` = P(venue accepts | you produce a novel result)** — conditional, assuming you have the goods.
- **`Y` = P(accept ∧ result) = X × P(result)** — unconditional joint probability, accounting for both venue selectivity and project execution risk.

The gap between `X` and `Y` measures execution risk. When the two numbers are close, the project is low-risk. When `Y` is much smaller than `X`, execution risk dominates.

Cells marked **—** are out of scope (wrong subfield). Bold cells mark the strongest fit per row.

Column headers refer to the idea IDs from §1: `#1`=SIREN-LC, `#2`=PySR-Gyro, `#3`=Mamba-RG, `#4`=FMPE-FRB, `#5`=SSL-SolARED, `#6`=INR-AstroComp, `#7`=Conformal-Tran, `#8`=TTT-PhotoZ, `#9`=ProtoNet-Rare, `#10`=MT-TESS-EB, `#11`=PINN-Transit, `#12`=Diff-AHA.

### 4A. Apex venues (Nature Astronomy + top-3 ML conferences)

These are long shots for undergraduate work. Include only if the result turns out to be exceptional.

| Venue             | #1    | #2    | #3      | #4    | #5    | #6      | #7    | #8      | #9    | #10   | #11   | #12   |
| ----------------- | ----- | ----- | ------- | ----- | ----- | ------- | ----- | ------- | ----- | ----- | ----- | ----- |
| Nature Astronomy  | 2/2   | 2/2   | 2/2     | 5/4   | 3/2   | 1/1     | 1/1   | 1/1     | 4/3   | 2/2   | 2/1   | 5/3   |
| NeurIPS main      | 8/7   | 8/7   | **15/12** | 10/7 | 8/6   | 12/10   | 8/8   | 12/8    | 10/8  | 5/4   | 5/4   | 10/7  |
| ICML main         | 8/7   | 10/9  | **15/12** | 10/7 | 8/6   | 12/10   | 8/8   | 12/8    | 10/8  | 5/4   | 5/4   | 10/7  |
| ICLR main         | 10/9  | 7/6   | 12/10   | 8/6   | 8/6   | **15/12** | 7/7 | 10/7    | 8/6   | 4/3   | 5/4   | 8/5   |

### 4B. High-prestige journals (ApJL, PASP, A&A family, ApJ, AJ, MNRAS family, PRD)

These are the primary target tier. Bold = strongest fit per row.

| Venue              | #1        | #2        | #3        | #4        | #5        | #6        | #7        | #8        | #9        | #10       | #11       | #12       |
| ------------------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| ApJL               | 15/14     | 22/20     | 18/15     | **28/20** | 15/12     | 8/7       | 12/12     | 12/8      | 22/18     | 15/13     | 18/13     | 22/15     |
| NeurIPS D&B Track  | 10/9      | 10/9      | 15/12     | 18/13     | 18/14     | **30/25** | 15/14     | 12/8      | 22/18     | 15/13     | 10/7      | 22/15     |
| PASP               | 62/56     | 55/51     | 58/48     | 52/37     | 45/35     | **78/65** | 66/63     | 62/42     | 55/44     | 60/51     | 55/40     | 58/39     |
| A&A Letters        | 20/18     | 28/26     | 22/18     | 32/23     | 20/16     | 10/8      | 15/14     | 15/10     | 25/20     | 20/17     | 22/16     | 25/17     |
| A&A                | 68/61     | **72/66** | 70/57     | **72/52** | 60/47     | 60/50     | 68/65     | 66/45     | 68/54     | **72/61** | 66/48     | 62/42     |
| ApJ                | 65/59     | 66/61     | 66/54     | 68/49     | 58/45     | 55/46     | 62/60     | 62/42     | 66/53     | 68/58     | 66/48     | 60/41     |
| AJ                 | **72/65** | 66/61     | 65/53     | 60/43     | 48/37     | 60/50     | 66/63     | **72/49** | **72/58** | **78/66** | **76/55** | 60/41     |
| MNRAS Letters      | 25/23     | 32/29     | 26/21     | 36/26     | 24/19     | 10/8      | 18/17     | 18/12     | 32/26     | 25/21     | 25/18     | 28/19     |
| Physical Review D  | 5/5       | 4/4       | 5/4       | **30/22** | —         | 3/2       | 5/5       | 22/15     | 10/8      | 2/2       | 5/4       | 8/5       |
| MNRAS              | 70/63     | 70/64     | 70/57     | 70/50     | 58/45     | 58/48     | 66/63     | 65/44     | 70/56     | 70/60     | 66/48     | 60/41     |

### 4C. Specialty and mid-tier venues

Subfield-specific or methods-focused journals. Best fits are subfield-gated.

| Venue              | #1        | #2        | #3        | #4        | #5        | #6        | #7        | #8        | #9        | #10       | #11       | #12       |
| ------------------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Scientific Reports | **76/68** | **76/70** | **76/62** | 72/52     | 70/55     | 68/56     | **76/73** | 70/48     | 74/59     | **76/65** | **72/52** | **70/48** |
| Space Weather      | —         | —         | —         | —         | **80/62** | —         | —         | —         | —         | —         | —         | —         |
| Icarus             | —         | —         | —         | —         | —         | —         | —         | —         | —         | —         | 15/11     | —         |
| Solar Physics      | —         | —         | —         | —         | **80/62** | —         | —         | —         | —         | —         | —         | —         |
| OJAp               | 60/54     | 62/57     | 62/51     | 65/47     | 55/43     | 56/46     | 60/58     | 60/41     | 62/50     | 60/51     | 60/43     | 56/38     |
| RASTI              | **76/68** | 70/64     | **76/62** | **76/55** | 64/50     | **82/68** | **76/73** | **76/52** | 70/56     | 66/56     | 70/50     | **72/49** |
| Astronomy & Comp.  | 65/59     | 60/55     | 66/54     | 66/48     | 55/43     | **76/63** | 66/63     | 66/45     | 60/48     | 60/51     | 60/43     | 66/45     |

### 4D. Workshop venues and RNAAS (safety net)

Non-archival ML workshops + the editorially-moderated RNAAS. High base-rate venues — the first paper pipeline.

| Venue                   | #1        | #2        | #3        | #4        | #5        | #6        | #7        | #8        | #9        | #10       | #11       | #12       |
| ----------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| ML4PS @ NeurIPS         | **76/68** | 66/61     | **78/64** | **74/53** | **72/56** | 74/61     | **78/75** | **74/50** | **76/61** | 65/55     | 66/48     | 72/49     |
| ML4Astro @ ICML         | 74/67     | 66/61     | 74/61     | 70/50     | 70/55     | 70/58     | 72/69     | 70/48     | 72/58     | 65/55     | 66/48     | 68/46     |
| AI4Science @ NeurIPS    | 48/43     | 46/42     | 52/43     | 52/37     | 50/39     | 52/43     | 50/48     | 50/34     | 52/42     | 46/39     | 50/36     | 50/34     |
| RNAAS                   | **98/88** | **98/90** | **97/80** | **97/70** | **97/76** | **97/81** | **98/94** | **97/66** | **97/78** | **98/83** | **97/70** | **97/66** |

**Color guide (mental).** Cells where the second number is ≥60 are realistic primary targets. Cells with second number 40–60 are reasonable backups. Cells with second number 20–40 are stretch goals with meaningful reject risk. Cells <20 are long shots; only submit if the project's nominal result is exceptional.

---

## 5. Recommended submission strategy per idea

For each idea, the pipeline is a **primary target** (best joint probability), a **stretch target** (higher prestige, lower probability), a **safety net** (~95%+ joint probability even with a mediocre result), and an **archival home** if the primary is non-archival. Workshop + journal dual submission is standard because ML4PS/ML4Astro/AI4Science are non-archival.

### Idea #1 — SIREN-LC (base rate 90%)

- **Primary:** RNAAS (98/88) for the headline, simultaneously ML4PS 2026 (76/68) for the methods writeup.
- **Stretch:** AJ (72/65) or MNRAS (70/63) as a full-length paper with a hypernetwork / meta-learning extension.
- **Safety net:** RNAAS is the guarantee.
- **Strategy:** Start with a 1,500-word RNAAS showing SIREN vs GP on 100 OGLE stars, then extend to AJ with hypernetwork meta-learning. Dual-submit the extended version to ML4PS.

### Idea #2 — PySR-Gyro (base rate 92%)

- **Primary:** RNAAS (98/90) — PySR produces closed-form expressions, the canonical RNAAS result type.
- **Stretch:** A&A (72/66) or ApJL (22/20) if the derived expression beats Barnes/Mamajek-Hillenbrand/Angus in cross-cluster validation.
- **Safety net:** RNAAS is guaranteed.
- **Strategy:** RNAAS first for the symbolic expression; if the expression is non-trivially better than analytic baselines across held-out clusters, submit the full characterization to ApJL or A&A.

### Idea #3 — Mamba-RG (base rate 82%)

- **Primary:** ML4PS (78/64) — the ML community cares about state-space-model applications.
- **Stretch:** A&A Letters (22/18) or MNRAS (70/57) as the full architecture study.
- **Safety net:** RNAAS (97/80) for a quick result note.
- **Strategy:** ML4PS methods paper with benchmarks against Hon et al. 2018 CNN, Astroconformer, and a same-parameter Transformer. Parallel track: extend to a full MNRAS paper including ν_max and Δν regression.

### Idea #4 — FMPE-FRB (base rate 72%)

- **Primary:** ApJL (28/20) or MNRAS Letters (36/26). FRBs carry enough community interest to justify a Letter if posterior coverage beats the CNN-LSTM baseline.
- **Stretch:** PRD (30/22) if the physics angle (scintillation-aware SBI) is framed tightly.
- **Safety net:** RNAAS (97/70) and ML4PS (74/53).
- **Strategy:** Target ApJL first for impact; fall back to MNRAS or A&A if selectivity bites. Highest upside and highest execution risk; scope down to a subset of CHIME Catalog 2 bursts to manage risk.

### Idea #5 — SSL-SolARED (base rate 78%)

- **Primary:** **Solar Physics (80/62) or Space Weather (80/62)** — exact domain fit.
- **Stretch:** ApJL (15/12) if the lead time improvement is operationally significant.
- **Safety net:** RNAAS (97/76) and ML4PS (72/56).
- **Strategy:** Solar Physics is the default target; Space Weather as co-pilot depending on framing. SSL vs supervised LSTM is the publishable hook regardless of magnitude.

### Idea #6 — INR-AstroComp (base rate 83%)

- **Primary:** **RASTI (82/68) or PASP (78/65)** — the instrumentation/methods pair.
- **Stretch:** NeurIPS D&B (30/25) as a benchmark paper that extends AstroCompress.
- **Safety net:** RNAAS (97/81) for a quick-result note.
- **Strategy:** RASTI is the ideal fit because it is methods-first, ML-friendly, and hungry for content. Dual-submit an extended benchmark to NeurIPS D&B if the INR produces a new Pareto frontier.

### Idea #7 — Conformal-Tran (base rate 96%)

- **Primary:** **ML4PS (78/75)** and **RASTI (76/73)** — both fit perfectly.
- **Stretch:** ApJL (12/12) is unlikely but viable if the paper demonstrates a real follow-up decision improvement.
- **Safety net:** RNAAS (98/94) is essentially guaranteed.
- **Strategy:** Highest base-rate safety of any idea. Ideal project if the undergraduate needs a publication on a deadline. ML4PS + RNAAS is the default pipeline.

### Idea #8 — TTT-PhotoZ (base rate 68%)

- **Primary:** ML4PS (74/50) and RASTI (76/52).
- **Stretch:** PRD (22/15) only if TTT measurably improves cosmology-relevant photo-z calibration.
- **Safety net:** Scientific Reports (70/48) — TTT is a clean hypothesis test and Sci. Rep. accepts sound-science papers without novelty judgments.
- **Strategy:** Highest execution risk in the set. Frame the paper as "TTT for photo-z drift: what works, what doesn't, and why" and submit to ML4PS.

### Idea #9 — ProtoNet-Rare (base rate 80%)

- **Primary:** **AJ (72/58) or ML4PS (76/61)**.
- **Stretch:** ApJL (22/18) if the few-shot method recovers actual rare-transient candidates in a real LSST alert stream snapshot.
- **Safety net:** RNAAS (97/78).
- **Strategy:** AJ is the natural fit (observational + ML + operational); ML4PS captures the ML-community credit. LSST operational framing matters for both.

### Idea #10 — MT-TESS-EB (base rate 85%)

- **Primary:** **AJ (78/66)** — exoplanets and eclipsing binaries are core AJ scope.
- **Stretch:** A&A (72/61) as the European counterpart.
- **Safety net:** RNAAS (98/83) for a quick result.
- **Strategy:** AJ for the full paper; RNAAS for any quick model-card / weight release on HuggingFace. Most straightforward undergraduate project in the list.

### Idea #11 — PINN-Transit (base rate 72%)

- **Primary:** **AJ (76/55)** and ApJ (66/48). Exoplanet transit fitting is AJ/ApJ core scope.
- **Stretch:** ApJL (18/13) if the PINN produces a speedup used to fit a catalog-scale problem.
- **Safety net:** RNAAS (97/70) for a speed-comparison note.
- **Strategy:** Start with the RNAAS speed comparison (`batman` MCMC vs PINN on 50 Kepler/TESS targets). If competitive parameter accuracy, extend to AJ with a catalog-scale analysis.

### Idea #12 — Diff-AHA (base rate 68%)

- **Primary:** ML4PS (72/49) and RASTI (72/49).
- **Stretch:** ApJL (22/15) if the diffusion model discovers a new real anomaly in the ZTF alert archive.
- **Safety net:** RNAAS (97/66) for a benchmarking note even with a null result.
- **Strategy:** High execution risk, high compute. Without a >16GB GPU for a week, deprioritize. If pursued, frame against AHA (Feb 2026 baseline), DiMMAD (ML4PS 2025), and isolation forests rigorously.

---

## 6. Summary: best single recommendation per idea

This consolidates the recommendation column into one table: the single best archival venue, best workshop, and safety net per idea.

| ID  | Idea           | P(result) | Best primary    | Joint | Best workshop | Joint | Safety (RNAAS) |
| --- | -------------- | --------- | --------------- | ----- | ------------- | ----- | -------------- |
| #1  | SIREN-LC       | 90%       | AJ              | 65%   | ML4PS         | 68%   | 88%            |
| #2  | PySR-Gyro      | 92%       | A&A             | 66%   | ML4PS         | 61%   | 90%            |
| #3  | Mamba-RG       | 82%       | MNRAS / A&A     | 57%   | ML4PS         | 64%   | 80%            |
| #4  | FMPE-FRB       | 72%       | A&A             | 52%   | ML4PS         | 53%   | 70%            |
| #5  | SSL-SolARED    | 78%       | Solar Physics   | 62%   | ML4PS         | 56%   | 76%            |
| #6  | INR-AstroComp  | 83%       | RASTI           | 68%   | ML4PS         | 61%   | 81%            |
| #7  | Conformal-Tran | 96%       | RASTI           | 73%   | ML4PS         | 75%   | 94%            |
| #8  | TTT-PhotoZ     | 68%       | RASTI           | 52%   | ML4PS         | 50%   | 66%            |
| #9  | ProtoNet-Rare  | 80%       | AJ              | 58%   | ML4PS         | 61%   | 78%            |
| #10 | MT-TESS-EB     | 85%       | AJ              | 66%   | ML4PS         | 55%   | 83%            |
| #11 | PINN-Transit   | 72%       | AJ              | 55%   | ML4PS         | 48%   | 70%            |
| #12 | Diff-AHA       | 68%       | RASTI           | 49%   | ML4PS         | 49%   | 66%            |

**Top-5 joint-probability projects (excluding RNAAS):**

1. #7 Conformal-Tran → ML4PS: **75%**
2. #7 Conformal-Tran → RASTI: **73%**
3. #6 INR-AstroComp → RASTI: **68%**
4. #1 SIREN-LC → ML4PS: **68%**
5. #10 MT-TESS-EB → AJ: **66%**

**Top-5 joint-probability projects (RNAAS included):**

1. #7 Conformal-Tran → RNAAS: **94%**
2. #2 PySR-Gyro → RNAAS: **90%**
3. #1 SIREN-LC → RNAAS: **88%**
4. #10 MT-TESS-EB → RNAAS: **83%**
5. #6 INR-AstroComp → RNAAS: **81%**

**Highest prestige feasibly reachable per idea.**

- Real shot at a top journal (≥15% joint at ApJL or ApJ+A&A+MNRAS): **#1, #2, #3, #4, #9, #10, #11, #12**.
- Best kept at workshop + RNAAS (high reliability, low prestige ceiling): **#7, #8**.
- Domain-locked venue: **#5** → Solar Physics / Space Weather; **#6** → RASTI / PASP / NeurIPS D&B.

---

## 7. Caveats and reading notes

- **All probability numbers are estimates**, not measured frequencies. They encode editorial taste, topical fit, and observed acceptance patterns for comparable ML-in-astronomy papers. Actual outcomes can deviate by ±10–15 percentage points.
- **Acceptance rates are not published** for most astronomy journals. The timings reported (~139 days for MNRAS, ~2.7–3.1 months for A&A, ~60 days median for ApJ) are journal-reported turnarounds, not accept/reject counts.
- **First-author undergraduate papers** face additional hurdles at selective journals; the probabilities assume a competent advisor co-author. Without a recognized co-author, subtract ~10 percentage points from the conditional values at ApJL, A&A Letters, MNRAS Letters, and Nature Astronomy.
- **Dual submission.** ML4PS, ML4Astro, and AI4Science are non-archival and explicitly permit simultaneous submission to archival journals. Archival conferences (NeurIPS/ICML/ICLR main, NeurIPS D&B) forbid dual submission.
- **Nature Astronomy** is included for completeness; no project in the list is a realistic target unless it makes an unexpected discovery.
- **Physical Review D** becomes viable only for #4 (strong physics angle), possibly #8 (if framed for cosmology), and marginally #12 (if a GW-related anomaly is found).
- **The joint probabilities (second number) multiply conditional acceptance by base rate.** They assume that if the base rate fires, the paper is written to the venue's standards. Writing quality is a separate axis not modeled here.
- **Cost matters.** MNRAS (£2,356) and MNRAS Letters (£1,122) are mandatory gold OA since January 2024; without a grant, these may be out of reach without a waiver. A&A Letters, OJAp, RASTI, RNAAS, and all NeurIPS/ICML/ICLR venues are free or subscription-funded.
- **Prestige ≠ best fit.** Submitting #7 (Conformal-Tran) to ApJL would be a mistake even if the probability were higher — the paper is not a discovery paper and would be desk-rejected for scope. Submit to the venue whose editorial identity matches the paper, not the highest-IF venue you think might accept it.

---

## 8. Sources

Venue details drawn from:

- [AAS Journals impact factors announcement (June 2025)](https://aas.org/posts/news/2025/06/aas-journals-again-receive-strong-impact-factors)
- [A&A 2025 paper-length policy](https://www.aanda.org/2025-press-releases/3045-shaping-a-sustainable-future-for-a-a-new-policy-on-paper-length)
- [A&A for Authors](https://www.aanda.org/for-authors)
- [MNRAS Instructions to Authors](https://academic.oup.com/mnras/pages/general_instructions)
- [MNRAS fully OA announcement](https://academic.oup.com/mnras/pages/mnras-open-access)
- [MNRAS author charges](https://academic.oup.com/mnras/pages/Author_Charges)
- [MNRAS Letters charges](https://academic.oup.com/mnrasl/pages/Charges)
- [RAS publication statistics (timing data)](https://ras.ac.uk/journals/publication-statistics)
- [AAS journals article charges and licensing](https://journals.aas.org/article-charges-and-copyright/)
- [AAS manuscript preparation](https://journals.aas.org/manuscript-preparation/)
- [ApJL requirements](https://journals.aas.org/the-astrophysical-journal-letters/)
- [Nature Astronomy submission guidelines](https://www.nature.com/natastron/submission-guidelines)
- [Nature Astronomy peer review policy](https://www.nature.com/natastron/editorial-policies/peer-review)
- [Nature Portfolio APC support](https://support.nature.com/en/support/solutions/articles/6000211135-article-processing-charges-apc-)
- [RNAAS homepage + preparation guidelines](https://journals.aas.org/research-notes/)
- [PASP via IOP Publishing](https://iopscience.iop.org/journal/1538-3873)
- [RASTI general instructions](https://academic.oup.com/rasti/pages/general-instructions)
- [Astronomy & Computing author guide](https://www.sciencedirect.com/journal/astronomy-and-computing/publish/guide-for-authors)
- [Open Journal of Astrophysics](https://astro.theoj.org/)
- [Solar Physics submission guidelines](https://link.springer.com/journal/11207/submission-guidelines)
- [Space Weather (AGU/Wiley)](https://agupubs.onlinelibrary.wiley.com/journal/15427390)
- [Icarus guide for authors](https://www.sciencedirect.com/journal/icarus/publish/guide-for-authors)
- [Physical Review D authors](https://journals.aps.org/prd/authors)
- [Scientific Reports submission guidelines](https://www.nature.com/srep/author-instructions/submission-guidelines)
- [ML4PS NeurIPS 2025](https://ml4physicalsciences.github.io/2025/) and [2025 guidelines](https://ml4physicalsciences.github.io/2025/guidelines.html)
- [ML4Astro ICML 2025](https://ml4astro.github.io/icml2025/)
- [AI4Science NeurIPS 2025](https://ai4sciencecommunity.github.io/neurips25)
- [NeurIPS 2025 Datasets & Benchmarks CFP](https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks)
- [NeurIPS 2025 statistics](https://papercopilot.com/statistics/neurips-statistics/neurips-2025-statistics/)
- [ICML 2025 statistics](https://papercopilot.com/statistics/icml-statistics/icml-2025-statistics/)
- [ICLR 2025 statistics](https://papercopilot.com/statistics/iclr-statistics/iclr-2025-statistics/)
- [Astrophysics journal impact-factor meta-analysis (arXiv:2410.03342)](https://arxiv.org/html/2410.03342v2)

# How old is that star? Ask it to spin. 🌀⭐

> A weekend-rabbit-hole side project. I read a paper, got nerd-sniped, and ended up writing a tiny astronomy note about it.

Stars slow down as they age — like a spinning top losing steam. A young star whips around in a few days; an old one takes weeks. So if you can measure how fast a star spins, you can guess its age. Astronomers call this **gyrochronology** (literally "rotation-dating"), and it's one of the few ways we have to age-date a random star that isn't the Sun.

I stumbled onto **[ChronoFlow (Van-Lane et al. 2025)](https://arxiv.org/abs/2412.12244)** — a slick neural network that does exactly this. It works great, but it's a black box: it gives you an age, not an *equation*. That bugged me. We've had a hand-written formula for stellar spin-down since **1972** (Skumanich's famous "period grows like √age" law). So I wanted to know:

> **If you hand a machine the raw data and let it invent its own equation from scratch — does it beat the 1972 formula, or just rediscover it?**

This repo is me answering that question over a few evenings.

---

## The approach (in plain English)

Instead of a neural network, I used **symbolic regression** ([PySR](https://github.com/MilesCranmer/PySR)) — an algorithm that evolves actual math equations to fit data, like natural selection for formulas. I fed it ~5,300 stars from 29 star clusters (clusters are great because every star in one is the same known age) and asked: *find me `rotation_period = f(age, color)`.*

Then I did the part I'm actually proud of — a **"am I fooling myself?" test**. Symbolic regression will *always* hand you a fancy equation that fits a little better. The real question is whether that equation is *real* or just memorizing noise. So I re-ran the whole search on 100 bootstrap-resampled copies of the data and counted how often each equation *shape* came back. A real law shows up every time. Noise-fitting shows up once and never again.

---

## The punchline 📉

![The money figure](project/figures/rnaas_figure.png)

**Left:** every equation PySR found, plotted as accuracy vs. complexity. The error drops fast, then flatlines way above the "noise floor" (the best anyone could possibly do given measurement scatter). **Right:** how stable each equation was across 100 resamples.

The result was almost funny:

- The **simple equation is rock-solid** — `P = C·√t·(B_P − R_P)₀` came back in **100 out of 100** resamples. That's just Skumanich's 1972 √age law with a color term. The machine rediscovered the 50-year-old textbook answer.
- The **fancier equations that "won" on accuracy were different every single time** (30–58 distinct forms at high complexity). They were fitting noise, not physics.
- A **one-line `scipy.curve_fit`** matched the fancy ML pipeline in cross-validation. Sometimes the boring tool *is* the answer.

So: open-cluster spin data genuinely can't support anything fancier than the √age law. Not a sexy "AI discovers new physics" headline — but an honest, useful negative result, and a reusable recipe for catching yourself before you over-claim.

---

## Some visuals I liked along the way

**5,300 stars, colored by age** — younger clusters (blue) spin fast (short periods, low on the plot); older ones (red) spin slow. The whole story in one scatter plot:

![Period vs color by age](project/figures/period_vs_color_by_age.png)

**Bonus round:** I inverted the relation to age-date 31,000 random Kepler field stars, and stress-tested it on M67 — an old cluster where the simple law is *known* to break (stars over ~4 Gyr stop slowing down as expected). It breaks exactly where it should:

![M67 breakdown](project/figures/m67_pred_vs_obs.png)

---

## What's in here

| Path | What it is |
|------|-----------|
| `project/paper/rnaas.tex` | The write-up — a short [RNAAS](https://journals.aas.org/research-notes/)-style note |
| `project/notebooks/` | The actual pipeline: data cleaning, baselines, PySR runs, bootstrap, validation |
| `project/results/` | Everything the runs spat out (Pareto fronts, 100 bootstraps, cross-validation, synthetic controls) |
| `project/figures/` | All the plots |
| `project/data/` | Cleaned star catalogs |
| `chronoflow/`, `GPgyro/` | The external packages I leaned on (not my code) |

## The one equation to remember

$$P_\mathrm{rot} = C\,\sqrt{t}\,\,(B_P - R_P)_0$$

A star's spin period ≈ a constant × √(its age) × its color. Skumanich basically had it in 1972. The data still agrees, and now I've got a bootstrap test to prove it's the *only* thing the data agrees with.

---

*Built out of curiosity, not affiliation. Data from the ChronoFlow catalog; equation-hunting by PySR. — Kai Vaid*

# Gyrochronology

Stars slow down as they get older — a young one spins around in a couple of days, an old one takes weeks. So if you can measure how fast a star turns, you can guess its age. This is me checking whether a machine handed nothing but the data lands on the same equation astronomers have leaned on since 1972, or talks itself into a fancier one that isn't really there.

I got into this after reading the [ChronoFlow paper](https://arxiv.org/abs/2412.12244). It's a neural network that dates stars from their spin, and it works, but it hands you an age and never an equation. I wanted the equation. There's been a clean little formula for this since Skumanich wrote it down in 1972 — period grows like the square root of age — and I was curious whether a from-scratch search would beat it or just walk back to it.

So instead of a neural net I used [symbolic regression](https://github.com/MilesCranmer/PySR), which evolves actual algebra to fit data, and pointed it at about 5,300 stars across 29 clusters. Clusters are the trick here: every star in one is the same known age, so they pin the relationship down.

![Around 5,300 stars. The young clusters (blue) sit low and spin fast; the old ones (red) ride high and spin slow.](project/figures/period_vs_color_by_age.png)

The catch with letting a machine invent equations is that it will always find one that fits a hair better. The interesting question is whether that equation is real or just memorizing the noise. So I re-ran the entire search on a hundred resampled copies of the data and counted how often each equation's *shape* came back. A real law turns up every time. Noise turns up once and never again.

![Left: every equation it found, error against complexity, flattening out well above the noise floor. Right: how many distinct equation shapes survived across a hundred resamples.](project/figures/rnaas_figure.png)

It came back to Skumanich. The simple form — period as the square root of age times color — showed up in all hundred runs. The fancier equations that edged it out on paper were different every single time; they were fitting scatter. And a one-line `scipy.curve_fit` matched the whole symbolic-regression pipeline in cross-validation. The data just doesn't hold enough to support anything more elaborate than the 1972 answer.

As a sanity check I inverted the relation to age a few tens of thousands of Kepler field stars, and pushed it onto M67 — an old cluster where the simple law is known to fall apart, because stars past about four billion years stop slowing down on schedule. It falls apart right where it should.

![M67. Predicted against observed; the simple law overshoots once the stars get old.](project/figures/m67_pred_vs_obs.png)

## What's here

The write-up is a short research note in `project/paper`. The pipeline — cleaning the catalog, fitting the classical baselines, the symbolic-regression runs, the resampling test, the cross-validation and synthetic controls — lives in `project/notebooks`, with everything it produced under `project/results` and the plots in `project/figures`. The `chronoflow` and `GPgyro` folders are the external packages I borrowed from, not my code.

## Made by

Kai Vaid. Built out of curiosity, no affiliation. Data from the ChronoFlow catalog, equation-hunting by PySR.

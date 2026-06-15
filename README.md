# Gyrochronology

Stars slow down as they age. A young one spins around in a couple of days; an old one takes weeks, so measuring a star's spin gives you its age. There has been a clean formula for this since 1972, when Skumanich found that period grows like the square root of age.

I got into this after reading the [ChronoFlow paper](https://arxiv.org/abs/2412.12244), a neural network that dates stars from their spin. It works, but it hands you an age and never an equation, and I wanted the equation. So I used [symbolic regression](https://github.com/MilesCranmer/PySR), which evolves actual algebra to fit data, on about 5,300 stars across 29 clusters. Clusters are the trick: every star in one is the same known age, so they pin the relationship down. The question was whether a search starting from nothing would beat Skumanich or walk straight back to him.

![Around 5,300 stars. Young clusters (blue) sit low and spin fast; old ones (red) ride high and spin slow.](project/figures/period_vs_color_by_age.png)

A machine inventing equations will always find one that fits a hair better, so the real question is whether that equation is real or just fitting noise. I re-ran the whole search on a hundred resampled copies of the data and counted how often each equation's shape came back. A real law turns up every time; noise turns up once and never again.

![Left: every equation found, error against complexity, flattening out above the noise floor. Right: how many distinct equation shapes survived across a hundred resamples.](project/figures/rnaas_figure.png)

It came back to Skumanich. The simple form, period as the square root of age times color, showed up in all hundred runs, while the fancier equations that edged it out on paper were different every time. A one-line `scipy.curve_fit` matched the whole pipeline in cross-validation. The data just doesn't hold enough to support anything more elaborate than the 1972 answer.

As a last check I inverted the relation to age tens of thousands of Kepler field stars and pushed it onto M67, an old cluster where the law is known to break. Stars past about four billion years stop slowing on schedule, and the predictions overshoot them exactly there.

![M67. Predicted against observed; the law overshoots once the stars get old.](project/figures/m67_pred_vs_obs.png)

## What's here

The write-up is a short note in `project/paper`. The pipeline in `project/notebooks` cleans the catalog, fits the baselines, runs the search and its hundred resamples, and does the cross-validation, writing results to `project/results` and plots to `project/figures`. The `chronoflow` and `GPgyro` folders are external packages, not my code.

Star catalog from ChronoFlow, equation search with PySR. By Roy Vaid.

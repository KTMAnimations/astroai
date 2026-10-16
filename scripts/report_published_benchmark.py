#!/usr/bin/env python3
"""Generate comparison figures and a report from the frozen benchmark tables."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astrogyro.data import ROOT
from make_figures import save,style
from make_results_document import markdown_table

LABELS={
 'gaussian_power':'Power law + Gaussian',
 'gaussian_quadratic':'Quadratic + Gaussian',
 'mixture_quadratic':'Quadratic mixture',
 'validation_selected':'Earlier family selector',
 'cf_equal_cluster_final':'Flow: equal-cluster, final',
 'cf_equal_cluster_selected':'Flow: equal-cluster, selected',
 'cf_per_star_final':'Flow: per-star, final',
 'cf_per_star_selected':'Flow: per-star, selected'}
COLORS={k:'#909aa8' for k in LABELS}
COLORS.update(mixture_quadratic='#226cb5',cf_equal_cluster_selected='#39956b',gaussian_power='#25344b')
REFERENCE='cf_equal_cluster_selected'


def figures(root,out,summary,reproduction):
    style()
    rows=reproduction['benchmarks'];clusters=[r for r in rows if r['object']!='Sun'];sun=rows[0]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.3,1]})
    ax=axes[0];x=np.arange(4)
    p=np.array([r['reference_published_median_myr'] for r in clusters])
    r=np.array([r['coarse_log_prior']['median_myr'] for r in clusters])
    ax.scatter(x-.12,p,s=65,color='#929ba9',label='Published median')
    ax.scatter(x+.12,r,s=55,color='#226cb5',label='Released-code reproduction')
    for i,(a,b) in enumerate(zip(p,r)):ax.annotate(f'{b:.1f} / {a:.0f}',(i,max(a,b)),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks(x,[r['object'] for r in clusters]);ax.set(ylabel='Cluster age (Myr)',ylim=(50,300),title='All four cluster medians agree within 0.2%')
    ax.legend(fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.5)
    ax=axes[1]
    y=[0,1];median=[5.1,sun['coarse_log_prior']['median_myr']/1000]
    lower=[3.7,sun['coarse_log_prior']['q16_myr']/1000];upper=[6.8,sun['coarse_log_prior']['q84_myr']/1000]
    for i in y:ax.errorbar(median[i],i,xerr=[[median[i]-lower[i]],[upper[i]-median[i]]],fmt='o',capsize=5,color=['#929ba9','#d56a3a'][i])
    ax.set_yticks(y,['Published Sun','Released defaults']);ax.invert_yaxis();ax.set(xlabel='Solar age (Gyr); 68% interval',title='Solar median fails the 5% reproduction target',ylim=(1.6,-.6))
    ax.axvline(4.6,color='#aeb5bf',ls=':',label='Reference solar age 4.6 Gyr');ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.5)
    fig.suptitle('Reproducing published numbers with the actual released ChronoFlow weights\nThe solar color uncertainty is unspecified in the paper; the default assumption is retained',fontsize=14)
    save(fig,out,'reproduction')

    names=[k for k in LABELS if k!=REFERENCE]
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,metric,title in zip(axes,['nll','crps'],['Negative log likelihood (nats)','CRPS (dex)']):
        for i,name in enumerate(names):
            s=summary['paired_intervals'][f'{name}_minus_{REFERENCE}_{metric}']
            ax.errorbar(s['mean_difference'],i,xerr=[[s['mean_difference']-s['ci95_low']],[s['ci95_high']-s['mean_difference']]],fmt='o',capsize=4,color=COLORS[name])
        ax.axvline(0,color='#39956b',ls='--');ax.set_yticks(range(len(names)),[LABELS[n] for n in names],fontsize=9);ax.invert_yaxis()
        ax.set(title=title,xlabel='Difference from equal-cluster, selected flow\nLeft: lower error; right: higher error');ax.grid(axis='x',alpha=.5)
    fig.suptitle('Matched training comparison: 721 stars, 16 withheld clusters, three splits each\nChronoFlow architecture retrained on identical discovery rows; paired descriptive 95% intervals',fontsize=14)
    save(fig,out,'matched_scores')

    ages=pd.read_csv(root/'cluster_age_estimates.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    for ax,name in zip(axes,[REFERENCE,'mixture_quadratic']):
        d=ages[ages.model==name];ax.plot([80,5500],[80,5500],'--',color='#aeb5bf',lw=1)
        ax.scatter(d.reference_age_myr,d.inferred_age_myr,s=50,color=COLORS[name])
        for _,row in d.iterrows():
            if abs(np.log10(row.inferred_age_myr/row.reference_age_myr))>.25:
                offset=(-5,10) if row.cluster=='NGC2516' else (4,4)
                ax.annotate(row.cluster,(row.reference_age_myr,row.inferred_age_myr),xytext=offset,
                            textcoords='offset points',fontsize=8,ha='right' if row.cluster=='NGC2516' else 'left')
        value=summary['age_scores'][name]['median_absolute_fractional_error']*100
        ax.set(xscale='log',yscale='log',xlim=(80,5500),ylim=(80,6000),xlabel='Catalogue reference age (Myr)',ylabel='Inferred withheld-cluster age (Myr)',title=f'{LABELS[name]}\nMedian absolute fractional error: {value:.1f}%');ax.grid(alpha=.5)
    fig.suptitle('Cluster-age recovery under the same input data and age grid\nAll 16 clusters retained; reference ages remain uncertain; boundary hits are reported',fontsize=14)
    save(fig,out,'matched_ages')

    external=pd.read_csv(root/'external_predictions.csv')
    d=external[(external.reference_age_myr==4000)&(external.model==REFERENCE)].groupby('GaiaDR3_ID').first()
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name in zip(axes,[REFERENCE,'mixture_quadratic']):
        p=external[(external.reference_age_myr==4000)&(external.model==name)].groupby('GaiaDR3_ID').mean(numeric_only=True).sort_values('BPRP_0')
        ax.scatter(p.BPRP_0,p.Prot,s=28,facecolor='white',edgecolor='#25344b',label='All 25 external observations',zorder=5)
        ax.plot(p.BPRP_0,10**p.median_log_period,color=COLORS[name],label='Median prediction')
        ax.fill_between(p.BPRP_0,10**p.q05,10**p.q95,color=COLORS[name],alpha=.16,label='Mean log 90% interval bounds')
        ax.set(xlabel='Dereddened Gaia DR3 BP-RP',ylabel='Rotation period (days)',ylim=(0,55),title=LABELS[name]);ax.grid(alpha=.5);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Same 25 external M67 stars; no M67 members used to fit or select either method\nThree fits averaged in log space for display; scores evaluate each fit separately at 4.0 Gyr',fontsize=13)
    save(fig,out,'external_comparison')


def document(root,summary,reproduction,external):
    rows=[]
    for name in LABELS:
        m=summary['mean_cluster_scores'][name]
        rows.append({'Method':LABELS[name],'NLL':f"{m['nll']:.4f}",'CRPS':f"{m['crps']:.4f}",'RMSE':f"{m['rmse_dex']:.4f}",'MAE':f"{m['mae_dex']:.4f}",'68% coverage':f"{m['coverage68']*100:.1f}%",'90% coverage':f"{m['coverage90']*100:.1f}%"})
    reproduction_table=pd.read_csv(root/'reproduction/comparison.csv').round(3)
    age_rows=[]
    for name in LABELS:
        a=summary['age_scores'][name]
        age_rows.append({'Method':LABELS[name],'Mean absolute log-age error':f"{a['mean_absolute_log_age_error']:.4f}",
                         'Median absolute fractional error':f"{a['median_absolute_fractional_error']*100:.1f}%",'Boundary hits / 48':int(a['boundary_hits'])})
    ext=[]
    for name in LABELS:
        m=external['scores'][name]
        ext.append({'Method':LABELS[name],'NLL':f"{m['nll']:.4f}",'CRPS':f"{m['crps']:.4f}",'RMSE':f"{m['rmse_dex']:.4f}",'MAE':f"{m['mae_dex']:.4f}",
                    'Inferred age (Gyr)':f"{external['inferred_age_myr'][name]/1000:.3f}"})
    primary=summary['paired_intervals'][f'mixture_quadratic_minus_{REFERENCE}_nll']
    crps=summary['paired_intervals'][f'mixture_quadratic_minus_{REFERENCE}_crps']
    age_ci=summary['paired_intervals'][f'mixture_quadratic_minus_{REFERENCE}_age_error']
    secondary=json.loads((root/'secondary_paired_comparisons.json').read_text())['comparisons']
    secondary_rows=[]
    for name,values in secondary.items():
        n,c=values['nll'],values['crps']
        secondary_rows.append({'Flow comparator':LABELS[name],
             'Mixture minus flow NLL':f"{n['mean_difference']:+.4f}",
             'NLL 95% interval':f"[{n['ci95_low']:+.4f}, {n['ci95_high']:+.4f}]",
             'Mixture minus flow CRPS':f"{c['mean_difference']:+.4f}"})
    direct=pd.read_csv(root/'released_comparison/cluster_scores.csv')[['cluster','model','n_test','nll','crps','rmse_dex']].round(4)
    ages=pd.read_csv(root/'cluster_age_estimates.csv')
    ages=ages[ages.model.isin([REFERENCE,'mixture_quadratic'])].pivot(index=['cluster','reference_age_myr'],columns='model',values='inferred_age_myr').reset_index().round(1)
    conclusion=('The mixture has lower mean NLL in this benchmark.' if primary['mean_difference']<0 else 'The retrained flow has lower mean NLL in this benchmark.')
    text=f'''# Published-model reproduction and matched comparison

This study runs the actual released ChronoFlow network, reproduces four published
cluster-age examples, and retrains its architecture for a controlled comparison
with this repository's models. It does not reproduce the paper's entire
cross-validation study or establish a new physical braking law.

The four reproduced cluster medians agree within 0.2%. The solar example does
not meet the predeclared 5% agreement target under the released defaults.
In the matched comparison, the primary NLL difference (quadratic mixture minus
validation-selected, equal-cluster flow) is **{primary['mean_difference']:+.4f}
nats**, with descriptive 95% interval **[{primary['ci95_low']:+.4f},
{primary['ci95_high']:+.4f}]**. {conclusion}

[Protocol](PUBLISHED_BENCHMARK_PROTOCOL.md) ·
[Configuration](../configs/published_benchmark.json) ·
[Complete result bundle](../project/results/published_benchmark_v1/)

## What was reproduced

The [authors' released code and weights](https://github.com/philvanlane/chronoflow/tree/f0b766f7d057c4e8f9a64a867414d25c5d985cd6)
are pinned at the existing submodule commit. The unmodified tutorial likelihood
reproduces -3.46 to its printed precision; the vectorized adapter agrees with
native scalar calls. Its CDF and likelihood pass normalization checks.

For the numerical examples in [Van-Lane et al. (2025)](https://arxiv.org/html/2412.12244v2),
we apply the documented quality cuts, the age-inference color threshold, measured
color errors and membership probabilities. The four cluster populations were
excluded from the released model's training according to the paper. They are
not ordinary training-set age fits. The table gives raw posterior intervals;
separate saved summaries also include the reported 0.06-dex intrinsic scatter.

{markdown_table(reproduction_table)}

![Released-model reproduction](../project/figures/published_benchmark_v1/reproduction.png)

The solar period and color match the paper, but its color uncertainty is not
specified. Using the released default gives 5.99 Gyr rather than 5.1 Gyr. The
post-hoc sensitivity table retains all tested color-error/outlier settings and
shows substantial input sensitivity. No alternative input replaces the failed
primary reproduction or proves what the authors originally used.

Another reproducibility issue is prior measure: the released implementation is
uniform in log age, whereas the paper describes a uniform age prior. The primary
run follows the code; the saved uniform-linear-age sensitivity is separate.
Doubling the age-grid resolution checks numerical convergence. These are
partial numerical reproductions, not a recreation of the paper's full training
process, global precision claim or uncertainty budget.

## Comparison with identical training and test rows

The upstream repository releases one final checkpoint, but no cluster-specific
cross-validation weights or complete training loop. Evaluating that checkpoint
on our ordinary test clusters would leak training information. Instead, the
controlled comparison retrains the published zuko neural-spline-flow architecture
on every frozen discovery split: **721 stars, 16 test clusters, three splits each**.
The entire test cluster is excluded from both fitting and validation.

Each of the **96 neural fits** receives 5,000 full-batch Adam steps. Half use equal
cluster weights, matching our existing study; half use the paper-style per-star
weighting. Both final and validation-selected checkpoints are retained. Selection
uses mean validation-cluster NLL at the 50 prescribed checkpoints. Architecture,
learning-rate schedule and stopping budget never use outer-test scores.

This is retraining of the published architecture under a documented benchmark
protocol. Missing original optimizer/batching/scheduler details prevent a claim
of identical original training. The source's 17,733-parameter architecture,
three spline layers, eight bins and membership/background likelihood are retained.
All methods receive the same discovery, validation and test rows. The flow uses
the available color uncertainties; our earlier mixture ignores that input.
Native likelihoods differ: the flow has its published uniform contaminant
component, while our mixture models the selected population directly.

Lower NLL, CRPS, RMSE and MAE are better. NLL is measured in nats for a density on
log10 period. Other period errors are in dex. RMSE evaluates mean log period;
MAE evaluates the predicted median. Average three splits within each cluster,
then weight the 16 clusters equally.

{markdown_table(pd.DataFrame(rows))}

The quadratic mixture has lower NLL in **{primary['clusters_better']} of 16**
cluster averages relative to the primary flow comparator. Its paired CRPS
difference is **{crps['mean_difference']:+.4f} dex**, with descriptive interval
**[{crps['ci95_low']:+.4f}, {crps['ci95_high']:+.4f}]**. All weighting/checkpoint
variants are shown; the headline comparator was fixed before scoring.

The secondary comparisons qualify that result:

{markdown_table(pd.DataFrame(secondary_rows))}

Against the paper-style per-star flows, the NLL intervals include zero and CRPS
is nearly tied. Thus the strongest result is relative to the equal-cluster
training variant; the experiment does not establish an advantage over every
reasonable training recipe. The quadratic mixture has 12 fitted parameters,
versus 17,733 in the flow. This makes it a useful compact comparator on this
restricted sample, without establishing state-of-the-art performance.

![Matched probability scores](../project/figures/published_benchmark_v1/matched_scores.png)

These intervals resample cluster means. They do not include all dependence from
overlapping training sets, systematic reference-age errors or the earlier reuse
of this catalogue for model development. A favorable difference on this restricted
sample would not establish broad superiority over the published system trained
on its larger population.

## Cluster ages on the same age grid

Every method maximizes the withheld group's period likelihood on the same 401
points between 80 and 5,000 Myr. Reference age enters only subsequent evaluation.
Inferred log ages are averaged across the three splits. The table retains all
clusters and every boundary hit; it does not remove difficult age calibrations.

{markdown_table(pd.DataFrame(age_rows))}

{markdown_table(ages)}

![Matched cluster ages](../project/figures/published_benchmark_v1/matched_ages.png)

These are cluster point estimates, with no calibrated individual-star age or
cluster-age confidence-interval claim. Accurate recovery of a paper's numerical
output is a different question from accuracy against an independently known age.
The paired change in mean absolute log-age error (mixture minus primary flow)
is {age_ci['mean_difference']:+.4f} dex with interval
[{age_ci['ci95_low']:+.4f}, {age_ci['ci95_high']:+.4f}], which includes zero.

## External M67 comparison

All methods predict the same 25 published stars absent from the entire original
catalogue. Every M67 member is excluded from the three discovery/validation fits.
Use a reference age of 4.0 Gyr; the existing 3.87 Gyr sensitivity is also saved.
The external table lacks individual color errors, so the flow uses the explicit
released default, log10(error) = -1.55. These stars were inspected in the earlier
study and cannot be described as a new blind test.

{markdown_table(pd.DataFrame(ext))}

The mixture's external RMSE is about 30.3% lower than the primary retrained flow's,
but the per-star, validation-selected flow estimates an age of 4.00 Gyr versus
3.38 Gyr for the mixture. The primary flow's nominal 68% intervals cover 46.7%
of these observations, while the mixture's cover 92.0%; both miss nominal
coverage in opposite directions. The comparison therefore does not have one
winner for every scientific task.

![Matched external observations](../project/figures/published_benchmark_v1/external_comparison.png)

For display, curves average the three per-fit predicted log medians and interval
bounds. They are not an ensemble's exact quantiles. Numerical scores evaluate
each model separately, then average. This is one external cluster population,
with a different single-star selection and uncertain age/reddening calibration.

## Direct comparison with the released weights

A supplementary check uses the unchanged released network on every cluster that
is both in our primary sample and explicitly excluded from its training: M34,
NGC1750 and NGC2516. Our frozen cluster-holdout fits predict those exact same test
stars. The released network is evaluated once; our three split scores are averaged.

{markdown_table(direct)}

Test observations match, but **training datasets differ**. These three clusters
are deliberately atypical in the paper and are not a representative benchmark.
This table supplies a direct released-weight comparison; it does not replace the
48-split controlled experiment or support a broad superiority claim.

## Reproducibility and implementation checks

The benchmark runs in a separate environment: Python 3.11.8, PyTorch 2.2.2,
zuko 1.1.0, NumPy 1.26.4 and SciPy 1.13.1. The zuko version matches the release;
Python/PyTorch differ from the paper's development environment. Matching the
native tutorial and four cluster examples provides a concrete compatibility
check. The prior study's environment and numerical results remain unchanged.

```bash
bash scripts/setup_published_environment.sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-published/bin/python scripts/reproduce_chronoflow.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-published/bin/python scripts/train_published_benchmark.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-published/bin/python scripts/evaluate_published_benchmark.py
PYTHONPATH=src .venv-published/bin/python scripts/evaluate_published_benchmark.py --stage summarize
PYTHONPATH=src .venv-published/bin/python scripts/summarize_published_sensitivities.py
PYTHONPATH=src .venv-published/bin/python scripts/evaluate_published_external.py
PYTHONPATH=src .venv-published/bin/python scripts/compare_released_chronoflow.py
PYTHONPATH=src .venv-published/bin/python scripts/diagnose_solar_reproduction.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-published/bin/python scripts/verify_published_benchmark.py
PYTHONPATH=src .venv-published/bin/python scripts/report_published_benchmark.py
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv-published/bin/python -m pytest tests/test_published_model.py tests/test_published_results.py -q
```

Training resumes completed fits and rejects changed source/configuration hashes.
An independent 5,000-step replay reproduced both saved checkpoints exactly.
Numerical grid refinement changed tested scores/quantiles by less than 0.00003 dex.
The bundle contains all weights, learning curves, choices, per-star scores, age
profiles, source hashes, failed reproduction outcomes and sensitivity results.

The repository is now a **partial numerical reproduction plus a controlled model
comparison**. Any extension claim must name the additional experiment and its
measured outcome. Existing statistical methods and source observations remain
credited to their authors; AI assistance supported research, implementation,
execution, checking and writing.
'''
    (ROOT/'docs/PUBLISHED_BENCHMARK_RESULTS.md').write_text(text)


def main():
    root=ROOT/'project/results/published_benchmark_v1';out=ROOT/'project/figures/published_benchmark_v1';out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/'summary.json').read_text());reproduction=json.loads((root/'reproduction/summary.json').read_text());external=json.loads((root/'external_summary.json').read_text())
    figures(root,out,summary,reproduction);document(root,summary,reproduction,external)
    print('Saved four figures and docs/PUBLISHED_BENCHMARK_RESULTS.md')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Reproduce released ChronoFlow examples, retaining numerical mismatches."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid

from astrogyro.data import ROOT, load_catalog, sha256
from astrogyro.experiments import write_json
from astrogyro.published import contexts, flow_age_profile, load_flow, posterior_summary, predict_flow


def main():
    torch.set_num_threads(1)
    config = json.loads((ROOT/'configs/published_benchmark.json').read_text())
    root = ROOT/'project/results'/config['study_id']/'reproduction';root.mkdir(parents=True,exist_ok=True)
    weight = ROOT/'chronoflow/chronoflow/model_artifacts/weights.pth'
    model = load_flow(weight)
    sys.path.insert(0,str(ROOT/'chronoflow/chronoflow'))
    from ChronoFlow import ChronoFlow
    native = ChronoFlow(str(weight))
    tutorial_ll = native.calcLogLikelihood(2,.5,1,logCerr=-1)
    native_post = native.calcPostAge(np.log10(20),1.5,logCerr=np.log10(.03))
    native_grid = np.linspace(0,4.14,1000)
    native_cdf = cumulative_trapezoid(native_post,native_grid,initial=0)
    tutorial_quantiles = np.interp([.16,.5,.84],native_cdf,native_grid)
    report = {'weights_sha256':sha256(weight),'script_sha256':sha256(Path(__file__)),
              'upstream_commit':config['upstream_commit'],
              'tutorial':{'log_likelihood':tutorial_ll,'published_rounded_log_likelihood':-3.46,
                         'log_likelihood_matches_rounding':round(tutorial_ll,2)==-3.46,
                         'age_prior_at_100_myr':native.calcAgePrior(2),
                         'color_prior_at_color_1_error_0_028':native.calcColourPrior(1,.028),
                         'posterior_log_age_q16_median_q84':tutorial_quantiles.tolist(),
                         'published_sampled_median_log_age':3.34},'benchmarks':[]}
    sample = load_catalog()
    quality = sample[(sample.cmd_exclude==0)&(sample.dered_exclude==0)&(sample.phot_exclude==0)&(sample.dr3_ruwe<1.4)]
    assert len(quality)==7615
    cases = [('Sun',pd.DataFrame({'Prot':[26.09],'BPRP_0':[.82],'e_BPRP_0':[10**-1.55],'membership':[1.]}),5100.)]
    for cluster,target in config['reproduction_clusters'].items():
        cases.append((cluster,quality[(quality.cluster==cluster)&(quality.BPRP_0>.5)].copy(),target))
    grid = np.linspace(0,4.14,4001)
    profile_rows=[]
    for name,frame,target in cases:
        ll=flow_age_profile(model,frame,10**grid)
        coarse,p=posterior_summary(grid[::2],ll[::2])
        fine,_=posterior_summary(grid,ll)
        broadened,_=posterior_summary(grid,ll,scatter_dex=0.06 if name!='Sun' else 0.)
        linear,_=posterior_summary(grid,ll,linear_age_prior=True)
        difference=coarse['median_myr']/target-1
        row={'object':name,'n_stars':len(frame),'reference_published_median_myr':target,
             'coarse_log_prior':coarse,'fine_log_prior':fine,'with_intrinsic_scatter':broadened,
             'linear_age_prior_sensitivity':linear,'fractional_median_difference':difference,
             'within_5_percent_target':abs(difference)<=.05,
             'grid_median_relative_difference':coarse['median_myr']/fine['median_myr']-1}
        report['benchmarks'].append(row)
        for g,l in zip(grid,ll):profile_rows.append({'object':name,'log_age_myr':g,'log_likelihood':l})
        frame.to_csv(root/f'{name}_stars.csv',index=False)
        print(f'{name}: {coarse["median_myr"]:.2f} Myr; published {target:.2f}; difference {100*difference:+.2f}%',flush=True)
    pd.DataFrame(profile_rows).to_csv(root/'age_profiles.csv',index=False)
    pd.DataFrame([{'object':r['object'],'n_stars':r['n_stars'],'reference_median_myr':r['reference_published_median_myr'],
                   **r['coarse_log_prior'],'fractional_difference':r['fractional_median_difference'],
                   'within_5_percent':r['within_5_percent_target']} for r in report['benchmarks']]).to_csv(root/'comparison.csv',index=False)
    write_json(root/'summary.json',report)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Secondary paired comparisons with every predeclared flow variant."""
import json
from pathlib import Path

import pandas as pd

from astrogyro.data import ROOT,sha256
from astrogyro.experiments import write_json
from astrogyro.metrics import paired_cluster_interval


def main():
    root=ROOT/'project/results/published_benchmark_v1'
    table=pd.read_csv(root/'cluster_scores.csv');ages=pd.read_csv(root/'cluster_age_estimates.csv')
    results={}
    for flow in sorted(n for n in table.model.unique() if n.startswith('cf_')):
        results[flow]={}
        for metric in ['nll','crps','rmse_dex','mae_dex']:
            p=table.pivot(index='cluster',columns='model',values=metric)
            results[flow][metric]=paired_cluster_interval(p.mixture_quadratic-p[flow],seed=20261016)
        p=ages.pivot(index='cluster',columns='model',values='absolute_log_age_error')
        results[flow]['age_error']=paired_cluster_interval(p.mixture_quadratic-p[flow],seed=20261016)
    write_json(root/'secondary_paired_comparisons.json',{'interpretation':'Quadratic mixture minus each flow; secondary comparisons, no multiplicity adjustment',
               'script_sha256':sha256(Path(__file__)),'comparisons':results})
    for flow,values in results.items():print(flow,values['nll'],values['crps'])


if __name__=='__main__':main()

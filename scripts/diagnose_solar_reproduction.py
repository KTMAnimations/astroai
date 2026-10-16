#!/usr/bin/env python3
"""Post-hoc sensitivity for the solar mismatch; never replaces the primary run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from astrogyro.data import ROOT, sha256
from astrogyro.experiments import write_json
from astrogyro.published import load_flow, posterior_summary, predict_flow


def main():
    torch.set_num_threads(1)
    model=load_flow(ROOT/'chronoflow/chronoflow/model_artifacts/weights.pth')
    g=np.linspace(0,4.14,4001);rows=[]
    for sigma in [.001,.003,.01,10**-1.55,.05,.1]:
        X=np.column_stack([g,np.full(len(g),.82),np.full(len(g),np.log10(sigma))])
        for outlier in [0.,.05]:
            ll=predict_flow(model,X,np.full(len(g),np.log10(26.09)),outlier=outlier)
            stats,_=posterior_summary(g,ll)
            rows.append({'color_error':sigma,'outlier_probability':outlier,**stats})
    root=ROOT/'project/results/published_benchmark_v1/reproduction'
    pd.DataFrame(rows).to_csv(root/'solar_input_sensitivity.csv',index=False)
    write_json(root/'solar_sensitivity_manifest.json',{'script_sha256':sha256(Path(__file__)),
               'status':'Post-hoc diagnostic after observing the default-input mismatch; primary reproduction unchanged',
               'fixed_inputs':{'period_days':26.09,'color':.82,'membership':1,'prior':'uniform log age'}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__=='__main__':main()

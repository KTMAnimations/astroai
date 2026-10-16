#!/usr/bin/env python3
"""Direct released-weight comparison on clusters excluded from both model fits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from astrogyro.data import ROOT, sha256
from astrogyro.experiments import write_json
from astrogyro.published import contexts,evaluate_flow,flow_age_profile,load_flow
from evaluate_published_benchmark import scores


def main():
    torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/published_benchmark.json').read_text())
    root=ROOT/'project/results'/config['study_id']/'released_comparison';root.mkdir(parents=True,exist_ok=True)
    source=ROOT/'project/results/real_stars_v2'
    sample=pd.read_csv(ROOT/'project/results/research_v1/selected_stars.csv',dtype={'GaiaDR3_ID':str})
    clusters=sorted(set(sample.cluster)&set(config['released_model_excluded_clusters']))
    assert clusters==['M34','NGC1750','NGC2516']
    weights=ROOT/'chronoflow/chronoflow/model_artifacts/weights.pth';model=load_flow(weights)
    grid=np.geomspace(80,5000,401)
    predictions=[];rows=[];ages=[];profiles=[]
    for cluster in clusters:
        test=sample[sample.cluster==cluster].copy();z=test.log_period.to_numpy()
        pred=evaluate_flow(model,contexts(test),z,test.membership.to_numpy())
        frame=test[['GaiaDR3_ID','cluster','Prot','BPRP_0','e_BPRP_0','log_period']].copy()
        frame['model']='chronoflow_released';frame['seed']=-1
        for k,v in pred.items():frame[k]=v
        predictions.append(frame)
        rows.append({'cluster':cluster,'model':'chronoflow_released','seed':-1,'n_test':len(test),**scores(pred,z)})
        ll=flow_age_profile(model,test,grid);index=int(np.argmax(ll))
        ages.append({'cluster':cluster,'model':'chronoflow_released','seed':-1,'reference_age_myr':float(test.fiducial_age_Myr.iloc[0]),
                     'inferred_age_myr':float(grid[index]),'boundary_hit':index in [0,400]})
        profiles.extend({'cluster':cluster,'age_myr':float(a),'log_likelihood':float(l)} for a,l in zip(grid,ll))
    old_pred=pd.read_csv(source/'predictions.csv',dtype={'GaiaDR3_ID':str})
    old_pred=old_pred[old_pred.cluster.isin(clusters)&old_pred.model.isin(['gaussian_power','mixture_quadratic'])]
    predictions.append(old_pred)
    old_scores=pd.read_csv(source/'scores.csv')
    rows.extend(old_scores[old_scores.cluster.isin(clusters)&old_scores.model.isin(['gaussian_power','mixture_quadratic'])].to_dict('records'))
    old_ages=pd.read_csv(source/'age_estimates.csv')
    ages.extend(old_ages[old_ages.cluster.isin(clusters)&old_ages.model.isin(['gaussian_power','mixture_quadratic'])].to_dict('records'))
    pd.concat(predictions,ignore_index=True).to_csv(root/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(root/'scores.csv',index=False)
    pd.DataFrame(ages).to_csv(root/'age_estimates.csv',index=False)
    pd.DataFrame(profiles).to_csv(root/'released_age_profiles.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['cluster','model']).mean(numeric_only=True).drop(columns='seed')
    summary.to_csv(root/'cluster_scores.csv')
    write_json(root/'manifest.json',{'status':'Matched test observations; different training datasets. Supplementary, not primary.',
               'clusters':clusters,'n_stars':int(sample.cluster.isin(clusters).sum()),'upstream_weights_sha256':sha256(weights),
               'script_sha256':sha256(Path(__file__)),'training_exclusion_evidence':'Van-Lane et al. 2025 Section V.1.1',
               'own_model_hashes':{p.parent.name:sha256(p) for cluster in clusters for p in (source/'folds').glob(f'holdout_{cluster}_seed*/result.json')}})
    print(summary[['nll','crps','rmse_dex']].round(4).to_string())


if __name__=='__main__':main()

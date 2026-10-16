#!/usr/bin/env python3
"""Compare the same M67-excluded models on the existing 25 external stars."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from astrogyro.data import ROOT, sha256
from astrogyro.experiments import write_json
from astrogyro.published import contexts, evaluate_flow, flow_age_profile, load_flow
from evaluate_published_benchmark import OWN, scores


def main():
    torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/published_benchmark.json').read_text())
    root=ROOT/'project/results'/config['study_id'];old=ROOT/'project/results/real_stars_v2'
    sample=pd.read_csv(old/'external_stars.csv',dtype={'GaiaDR3_ID':str})
    original=pd.read_csv(ROOT/'project/data/prot_catalogue.csv',dtype={'GaiaDR3_ID':str})
    assert not set(sample.GaiaDR3_ID)&set(original.GaiaDR3_ID)
    grid=np.geomspace(*config['comparison_age_grid_myr'][:2],config['comparison_age_grid_myr'][2])
    predictions=[];rows=[];ages=[];profiles=[];weight_hashes={}
    for fold in sorted((root/'fits').glob('holdout_M67_seed*')):
        seed=int(fold.name.rsplit('seed',1)[1])
        for weight in config['training_weightings']:
            fit=json.loads((fold/weight/'result.json').read_text())
            assert 'M67' not in fit['split']['discovery_clusters']+fit['split']['validation_clusters']
            for checkpoint in ['final','selected']:
                name=f'cf_{weight}_{checkpoint}';path=fold/weight/f'{checkpoint}.pt'
                assert sha256(path)==fit['weight_sha256'][checkpoint]
                weight_hashes[str(path.relative_to(root))]=sha256(path)
                model=load_flow(path)
                ll=flow_age_profile(model,sample,grid,membership=1.,default_log_error=config['external_default_log_color_error'])
                index=int(np.argmax(ll))
                ages.append({'model':name,'seed':seed,'inferred_age_myr':float(grid[index]),'boundary_hit':index in [0,len(grid)-1]})
                profiles.extend({'model':name,'seed':seed,'age_myr':float(a),'log_likelihood':float(l)} for a,l in zip(grid,ll))
                for age in config['external_reference_ages_myr']:
                    X=contexts(sample,age_myr=age,default_log_error=config['external_default_log_color_error'])
                    pred=evaluate_flow(model,X,sample.log_period.to_numpy(),grid=np.linspace(*config['period_score_grid']))
                    frame=sample[['GaiaDR3_ID','BPRP_0','Prot','log_period']].copy()
                    frame['model'],frame['seed'],frame['reference_age_myr']=name,seed,age
                    for k,v in pred.items():frame[k]=v
                    predictions.append(frame)
                    rows.append({'model':name,'seed':seed,'reference_age_myr':age,**scores(pred,sample.log_period.to_numpy())})
    old_pred=pd.read_csv(old/'external_predictions.csv',dtype={'GaiaDR3_ID':str})
    old_pred=old_pred[old_pred.model.isin(OWN)]
    predictions.append(old_pred)
    for (name,seed,age),frame in old_pred.groupby(['model','seed','reference_age_myr']):
        rows.append({'model':name,'seed':int(seed),'reference_age_myr':float(age),**scores(frame,frame.log_period.to_numpy())})
    old_ages=pd.read_csv(old/'external_age_estimates.csv')
    ages.extend(old_ages[old_ages.model.isin(OWN)].to_dict('records'))
    pd.concat(predictions,ignore_index=True).to_csv(root/'external_predictions.csv',index=False)
    table=pd.DataFrame(rows);table.to_csv(root/'external_scores.csv',index=False)
    a=pd.DataFrame(ages);a.to_csv(root/'external_age_estimates.csv',index=False)
    pd.DataFrame(profiles).to_csv(root/'external_age_profiles.csv',index=False)
    a['log_age']=np.log10(a.inferred_age_myr)
    result={'n_stars':len(sample),'default_log_color_error':config['external_default_log_color_error'],
            'reference_age_myr':4000,'sample_sha256':sha256(old/'external_stars.csv'),
            'script_sha256':sha256(Path(__file__)),'weight_sha256':weight_hashes,
            'scores':table[table.reference_age_myr==4000].groupby('model').mean(numeric_only=True).drop(columns=['seed','reference_age_myr']).to_dict('index'),
            'inferred_age_myr':(10**a.groupby('model').log_age.mean()).to_dict(),
            'boundary_hits':a.groupby('model').boundary_hit.sum().to_dict()}
    assert len(weight_hashes)==12
    write_json(root/'external_summary.json',result)
    print(pd.DataFrame(result['scores']).T.round(4).to_string())
    print(result['inferred_age_myr'])


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Score trained published architectures on identical withheld observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from astrogyro.data import ROOT, sha256
from astrogyro.experiments import write_json
from astrogyro.metrics import paired_cluster_interval
from astrogyro.published import contexts, evaluate_flow, flow_age_profile, load_flow

OWN = ['gaussian_power','gaussian_quadratic','mixture_quadratic','validation_selected']
FLOW = [f'cf_{weight}_{checkpoint}' for weight in ['equal_cluster','per_star'] for checkpoint in ['final','selected']]


def scores(pred, z):
    return {'nll':float(np.mean(pred['nll'])), 'crps':float(np.mean(pred['crps'])),
            'rmse_dex':float(np.sqrt(np.mean((pred['mean_log_period']-z)**2))),
            'mae_dex':float(np.mean(np.abs(pred['median_log_period']-z))),
            'coverage68':float(np.mean((z>=pred['q16'])&(z<=pred['q84']))),
            'coverage90':float(np.mean((z>=pred['q05'])&(z<=pred['q95']))),
            'width68_dex':float(np.mean(pred['q84']-pred['q16'])),
            'width90_dex':float(np.mean(pred['q95']-pred['q05']))}


def evaluate(config,root,args):
    sample=pd.read_csv(ROOT/'project/results/research_v1/selected_stars.csv',dtype={'GaiaDR3_ID':str})
    paths=sorted((ROOT/'project/results/real_stars_v2/folds').glob('*/result.json'))
    grid=np.geomspace(*config['comparison_age_grid_myr'][:2],config['comparison_age_grid_myr'][2])
    period_grid=np.linspace(*config['period_score_grid'])
    for i,path in enumerate(paths):
        if i%args.shard_count!=args.shard_index:continue
        dest=root/'evaluation'/path.parent.name
        if (dest/'result.json').exists():continue
        fitroot=root/'fits'/path.parent.name
        if not all((fitroot/w/'result.json').exists() for w in config['training_weightings']):
            if args.available_only:continue
            raise RuntimeError(f'Incomplete training: {path.parent.name}')
        old=json.loads(path.read_text())
        test=sample[sample.cluster.isin(old['split']['test_clusters'])].copy()
        dest.mkdir(parents=True,exist_ok=True)
        predictions=[];rows=[];ages=[];profiles={'age_myr':grid};weights={}
        seed=old['seed'];cluster=test.cluster.iloc[0];z=test.log_period.to_numpy()
        for weight in config['training_weightings']:
            fit=json.loads((fitroot/weight/'result.json').read_text())
            assert fit['split']==old['split']
            for checkpoint in ['final','selected']:
                name=f'cf_{weight}_{checkpoint}';file=fitroot/weight/f'{checkpoint}.pt'
                assert sha256(file)==fit['weight_sha256'][checkpoint]
                weights[name]=sha256(file)
                model=load_flow(file)
                pred=evaluate_flow(model,contexts(test),z,test.membership.to_numpy(),grid=period_grid)
                frame=test[['GaiaDR3_ID','cluster','fiducial_age_Myr','BPRP_0','e_BPRP_0','Prot','log_period']].copy()
                frame['model'],frame['seed']=name,seed
                for k,v in pred.items():frame[k]=v
                predictions.append(frame)
                rows.append({'cluster':cluster,'seed':seed,'model':name,'n_test':len(test),**scores(pred,z)})
                ll=flow_age_profile(model,test,grid)
                profiles[name]=ll
                idx=int(np.argmax(ll))
                ages.append({'cluster':cluster,'seed':seed,'model':name,'reference_age_myr':float(test.fiducial_age_Myr.iloc[0]),
                             'inferred_age_myr':float(grid[idx]),'boundary_hit':idx in [0,len(grid)-1]})
        old_pred=pd.read_csv(path.with_name('predictions.csv'),dtype={'GaiaDR3_ID':str})
        predictions.append(old_pred[old_pred.model.isin(OWN)])
        rows.extend(r for r in old['scores'] if r['model'] in OWN)
        ages.extend(r for r in old['ages'] if r['model'] in OWN)
        old_profiles=pd.read_csv(path.with_name('age_profiles.csv'))
        for name in OWN:profiles[name]=old_profiles[name].to_numpy()*len(test)
        pd.concat(predictions,ignore_index=True).to_csv(dest/'predictions.csv',index=False)
        pd.DataFrame(profiles).to_csv(dest/'age_profiles.csv',index=False)
        write_json(dest/'result.json',{'split':old['split'],'seed':seed,'scores':rows,'ages':ages,
                   'weight_sha256':weights,'reference_result_sha256':sha256(path)})
        print(f'Evaluated {path.parent.name}: {len(test)} stars, eight methods',flush=True)


def summarize(config,root):
    paths=sorted((root/'evaluation').glob('*/result.json'))
    assert len(paths)==48,f'Expected 48, found {len(paths)}'
    rows=pd.DataFrame([r for p in paths for r in json.loads(p.read_text())['scores']])
    rows.to_csv(root/'scores.csv',index=False)
    predictions=pd.concat([pd.read_csv(p.with_name('predictions.csv'),dtype={'GaiaDR3_ID':str}) for p in paths],ignore_index=True)
    assert len(predictions)==721*3*8
    predictions.to_csv(root/'predictions.csv',index=False)
    metrics=['nll','crps','rmse_dex','mae_dex','coverage68','coverage90','width68_dex','width90_dex']
    cluster=rows.groupby(['cluster','model'])[metrics].mean()
    cluster.to_csv(root/'cluster_scores.csv')
    summary={'n_stars':721,'n_clusters':16,'n_splits':48,'neural_fits':96,
             'mean_cluster_scores':cluster.groupby('model').mean().to_dict('index'),'paired_intervals':{}}
    reference='cf_equal_cluster_selected'
    for model in [*OWN,*FLOW]:
        if model==reference:continue
        for metric in ['nll','crps','rmse_dex','mae_dex']:
            pivot=cluster[metric].unstack();delta=pivot[model]-pivot[reference]
            result=paired_cluster_interval(delta,seed=config['seed'],n_boot=config['bootstrap_replicates'])
            result['clusters_better']=int((delta<0).sum())
            summary['paired_intervals'][f'{model}_minus_{reference}_{metric}']=result
    ages=pd.DataFrame([r for p in paths for r in json.loads(p.read_text())['ages']])
    ages.to_csv(root/'age_estimates.csv',index=False)
    ages['log_age']=np.log10(ages.inferred_age_myr)
    by=ages.groupby(['cluster','model']).agg(log_age=('log_age','mean'),reference_age_myr=('reference_age_myr','first'),boundary_hits=('boundary_hit','sum'))
    by['inferred_age_myr']=10**by.log_age
    by['absolute_log_age_error']=np.abs(by.log_age-np.log10(by.reference_age_myr))
    by['absolute_fractional_error']=np.abs(by.inferred_age_myr/by.reference_age_myr-1)
    by.to_csv(root/'cluster_age_estimates.csv')
    summary['age_scores']=by.groupby('model').agg(mean_absolute_log_age_error=('absolute_log_age_error','mean'),
                  median_absolute_fractional_error=('absolute_fractional_error','median'),boundary_hits=('boundary_hits','sum')).to_dict('index')
    for model in [*OWN,*FLOW]:
        if model==reference:continue
        pivot=by.absolute_log_age_error.unstack()
        summary['paired_intervals'][f'{model}_minus_{reference}_age_error']=paired_cluster_interval(pivot[model]-pivot[reference],seed=config['seed'])
    summary['maximum_period_grid_tail_mass']=float(predictions.tail_mass.max())
    write_json(root/'summary.json',summary)
    print(pd.DataFrame(summary['mean_cluster_scores']).T.round(4).to_string())
    print(pd.DataFrame(summary['age_scores']).T.round(4).to_string())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['evaluate','summarize'],default='evaluate')
    parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--available-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/published_benchmark.json').read_text())
    root=ROOT/'project/results'/config['study_id']
    manifest=json.loads((root/'manifest.json').read_text())
    for name,checksum in manifest['code_sha256'].items():assert sha256(ROOT/name)==checksum,name
    signature={'training_manifest_sha256':sha256(root/'manifest.json'),
               'evaluation_script_sha256':sha256(Path(__file__)),
               'reference_manifest_sha256':sha256(ROOT/'project/results/real_stars_v2/manifest.json')}
    file=root/'evaluation_manifest.json'
    if file.exists():assert json.loads(file.read_text())==signature,'Evaluation definition changed'
    else:write_json(file,signature)
    if args.stage=='evaluate':evaluate(config,root,args)
    else:summarize(config,root)


if __name__=='__main__':main()

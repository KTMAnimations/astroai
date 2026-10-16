#!/usr/bin/env python3
"""Replay one full neural fit and check numerical scoring convergence."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from astrogyro.data import ROOT, equal_cluster_weights, sha256
from astrogyro.experiments import write_json
from astrogyro.published import contexts, evaluate_flow, load_flow, new_flow, predict_flow, tensor_log_density


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    config=json.loads((ROOT/'configs/published_benchmark.json').read_text())
    root=ROOT/'project/results'/config['study_id'];fold='holdout_Hyades_seed0'
    path=root/'fits'/fold/'equal_cluster';saved=json.loads((path/'result.json').read_text())
    sample=pd.read_csv(ROOT/'project/results/research_v1/selected_stars.csv',dtype={'GaiaDR3_ID':str})
    train,val=[sample[sample.cluster.isin(saved['split'][k])] for k in ['discovery_clusters','validation_clusters']]
    X=torch.tensor(contexts(train),dtype=torch.float32);z=torch.tensor(train.log_period.to_numpy(),dtype=torch.float32)
    mem=torch.tensor(train.membership.to_numpy(),dtype=torch.float32)
    w=equal_cluster_weights(train.cluster);w=torch.tensor(w/w.sum(),dtype=torch.float32)
    model=new_flow(saved['seed']);optimizer=torch.optim.Adam(model.parameters(),lr=config['initial_learning_rate'])
    best=math.inf;best_state=None;best_epoch=None
    for epoch in range(1,config['epochs']+1):
        fraction=(epoch-1)/(config['epochs']-1)
        exp_lr=config['initial_learning_rate']*(config['final_exponential_rate']/config['initial_learning_rate'])**fraction
        optimizer.param_groups[0]['lr']=exp_lr*.5*(1+math.cos(2*math.pi*(epoch-1)/config['cosine_period_epochs']))
        optimizer.zero_grad();loss=-(w*tensor_log_density(model,X,z,mem)).sum();loss.backward();optimizer.step()
        if epoch%config['validation_every']==0:
            perstar=-predict_flow(model,contexts(val),val.log_period.to_numpy(),val.membership.to_numpy())
            value=float(np.mean([perstar[val.cluster.to_numpy()==c].mean() for c in val.cluster.unique()]))
            if value<best:best,best_state,best_epoch=value,copy.deepcopy(model.state_dict()),epoch
    differences={}
    for name,state in [('final',model.state_dict()),('selected',best_state)]:
        original=torch.load(path/f'{name}.pt',map_location='cpu',weights_only=True)
        differences[name]=max(float(torch.max(torch.abs(state[k]-original[k]))) for k in state)
        assert differences[name]<=1e-6
    assert best_epoch==saved['selected_epoch']
    checks=[]
    for name,file in [('released',ROOT/'chronoflow/chronoflow/model_artifacts/weights.pth'),('retrained',path/'selected.pt')]:
        m=load_flow(file)
        a,c,e=np.meshgrid([135,600,1000,3870],[.7,1.,1.45],[.015,.03,.1],indexing='ij')
        context=np.column_stack([np.log10(a.ravel()),c.ravel(),np.log10(e.ravel())])
        observations=np.full(len(context),1.0)
        coarse=evaluate_flow(m,context,observations,membership=.9,grid=np.linspace(-4,4,4001))
        fine=evaluate_flow(m,context,observations,membership=.9,grid=np.linspace(-4,4,8001))
        delta={k:float(np.max(np.abs(coarse[k]-fine[k]))) for k in ['mean_log_period','crps','median_log_period','q05','q95']}
        assert max(delta.values())<.001,delta
        checks.append({'model':name,'feature_grid_points':len(context),'maximum_absolute_differences':delta})
    tail_checks=[]
    predictions=pd.read_csv(root/'predictions.csv',dtype={'GaiaDR3_ID':str})
    for name,group in predictions[predictions.model.str.startswith('cf_')].groupby('model'):
        worst=group.loc[group.tail_mass.idxmax()]
        weighting,checkpoint=name[3:].rsplit('_',1)
        file=root/'fits'/f'holdout_{worst.cluster}_seed{int(worst.seed)}'/weighting/f'{checkpoint}.pt'
        m=load_flow(file);frame=sample[sample.GaiaDR3_ID==worst.GaiaDR3_ID]
        kwargs={'membership':frame.membership.to_numpy()}
        narrow=evaluate_flow(m,contexts(frame),frame.log_period.to_numpy(),grid=np.linspace(-4,4,4001),**kwargs)
        wide=evaluate_flow(m,contexts(frame),frame.log_period.to_numpy(),grid=np.linspace(-6,6,12001),**kwargs)
        delta={k:float(np.max(np.abs(narrow[k]-wide[k]))) for k in ['mean_log_period','crps','median_log_period','q05','q95']}
        assert max(delta.values())<.001,delta
        tail_checks.append({'model':name,'maximum_narrow_tail_mass':float(worst.tail_mass),
                            'maximum_absolute_differences':delta})
    report={'fold_replayed':fold,'epochs':config['epochs'],'selected_epoch':best_epoch,
            'maximum_weight_differences':differences,'numeric_convergence':checks,'tail_domain_checks':tail_checks,
            'training_manifest_sha256':sha256(root/'manifest.json'),'script_sha256':sha256(Path(__file__))}
    write_json(root/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()

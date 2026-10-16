"""Audit the saved comparison without requiring the optional PyTorch environment."""
import json

import numpy as np
import pandas as pd
import pytest

from astrogyro.data import ROOT, sha256

RESULTS=ROOT/'project/results/published_benchmark_v1'


def test_published_source_hashes_and_every_training_choice():
    manifest=json.loads((RESULTS/'manifest.json').read_text())
    for name,checksum in manifest['code_sha256'].items():
        assert sha256(ROOT/name)==checksum
    paths=sorted((RESULTS/'fits').glob('*/*/result.json'))
    assert len(paths)==96
    for path in paths:
        fit=json.loads(path.read_text());curve=pd.read_csv(path.with_name('learning_curve.csv'))
        assert fit['epochs']==5000 and len(curve)==50
        best=curve.loc[curve.validation_nll.idxmin()]
        assert fit['selected_epoch']==best.epoch
        assert fit['selected_validation_nll']==pytest.approx(best.validation_nll)
        split=fit['split']
        assert not set(split['test_clusters']) & set(split['discovery_clusters']+split['validation_clusters'])
        assert not set(split['discovery_clusters']) & set(split['validation_clusters'])
        original=ROOT/'project/results/research_v1/observational'/path.parent.parent.name/'result.json'
        assert sha256(original)==manifest['split_sha256'][path.parent.parent.name]
        assert split==json.loads(original.read_text())['split']
        for checkpoint,checksum in fit['weight_sha256'].items():
            assert sha256(path.with_name(checkpoint+'.pt'))==checksum


def test_all_comparison_stars_scores_and_original_baselines_preserved():
    predictions=pd.read_csv(RESULTS/'predictions.csv',dtype={'GaiaDR3_ID':str})
    assert len(predictions)==721*3*8
    assert predictions.GaiaDR3_ID.nunique()==721
    assert not predictions.duplicated(['GaiaDR3_ID','model','seed']).any()
    assert np.isfinite(predictions[['nll','crps','mean_log_period','q05','q95']]).all().all()
    assert (predictions.crps>=0).all()
    scores=pd.read_csv(RESULTS/'scores.csv')
    for _,row in scores.iterrows():
        p=predictions[(predictions.cluster==row.cluster)&(predictions.seed==row.seed)&(predictions.model==row.model)]
        assert len(p)==row.n_test
        assert p.nll.mean()==pytest.approx(row.nll)
        assert p.crps.mean()==pytest.approx(row.crps)
    old=pd.read_csv(ROOT/'project/results/real_stars_v2/scores.csv')
    merged=scores.merge(old,on=['cluster','seed','model'])
    assert len(merged)==48*4
    assert merged.nll_x.to_numpy()==pytest.approx(merged.nll_y.to_numpy(),abs=1e-12)


def test_external_same_stars_and_m67_never_fitted():
    table=pd.read_csv(RESULTS/'external_predictions.csv',dtype={'GaiaDR3_ID':str})
    assert len(table)==25*3*8*2
    original=pd.read_csv(ROOT/'project/data/prot_catalogue.csv',dtype={'GaiaDR3_ID':str})
    assert not set(table.GaiaDR3_ID)&set(original.GaiaDR3_ID)
    for p in (RESULTS/'fits').glob('holdout_M67_seed*/*/result.json'):
        split=json.loads(p.read_text())['split']
        assert 'M67' not in split['discovery_clusters']+split['validation_clusters']
    summary=json.loads((RESULTS/'external_summary.json').read_text())
    assert len(summary['weight_sha256'])==12
    for path,checksum in summary['weight_sha256'].items():assert sha256(RESULTS/path)==checksum


def test_age_estimates_match_saved_likelihood_maxima():
    for p in (RESULTS/'evaluation').glob('*/result.json'):
        result=json.loads(p.read_text());profiles=pd.read_csv(p.with_name('age_profiles.csv'))
        for row in result['ages']:
            index=profiles[row['model']].idxmax()
            assert row['inferred_age_myr']==pytest.approx(profiles.age_myr.iloc[index])


def test_reproduction_keeps_solar_mismatch_and_every_cluster():
    result=json.loads((RESULTS/'reproduction/summary.json').read_text())
    assert result['tutorial']['log_likelihood_matches_rounding']
    rows={r['object']:r for r in result['benchmarks']}
    assert set(rows)=={'Sun','M34','NGC2516','NGC1647','NGC1750'}
    assert not rows['Sun']['within_5_percent_target']
    for cluster in ['M34','NGC2516','NGC1647','NGC1750']:
        assert rows[cluster]['within_5_percent_target']
        assert abs(rows[cluster]['grid_median_relative_difference'])<.001


def test_released_comparison_only_uses_documented_excluded_clusters():
    manifest=json.loads((RESULTS/'released_comparison/manifest.json').read_text())
    assert manifest['clusters']==['M34','NGC1750','NGC2516']
    table=pd.read_csv(RESULTS/'released_comparison/predictions.csv',dtype={'GaiaDR3_ID':str})
    original=table[table.model=='chronoflow_released']
    assert len(original)==80 and not original.GaiaDR3_ID.duplicated().any()
    for seed in [0,1,2]:
        ours=table[(table.model=='mixture_quadratic')&(table.seed==seed)]
        assert set(ours.GaiaDR3_ID)==set(original.GaiaDR3_ID)
    for fold,checksum in manifest['own_model_hashes'].items():
        path=ROOT/'project/results/real_stars_v2/folds'/fold/'result.json'
        assert sha256(path)==checksum

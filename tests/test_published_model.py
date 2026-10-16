import sys

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import ndtr

pytest.importorskip('torch')
pytest.importorskip('zuko')
import torch

from astrogyro.data import ROOT
from astrogyro.distributions import cdf, crps
from astrogyro.published import flow_age_profile, grid_scores, load_flow, posterior_summary, predict_flow

WEIGHTS = ROOT/'chronoflow/chronoflow/model_artifacts/weights.pth'


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    return load_flow(WEIGHTS)


def test_adapter_matches_unmodified_upstream(model):
    sys.path.insert(0, str(ROOT/'chronoflow/chronoflow'))
    from ChronoFlow import ChronoFlow
    native = ChronoFlow(str(WEIGHTS))
    X = np.array([[2, .5, -1], [3, 1, -1.55], [3.6, 1.4, -2], [2.2, .8, -1.8]])
    z = np.array([1, 1.1, 1.5, .7]);mem = np.array([1, .85, .9, .8])
    expected = [native.calcLogLikelihood(*x[:2], zz, logCerr=x[2], P_clmem=m) for x, zz, m in zip(X, z, mem)]
    assert predict_flow(model, X, z, mem) == pytest.approx(expected, abs=3e-5)
    assert expected[0] == pytest.approx(-3.46, abs=.005)


def test_flow_cdf_and_density_form_a_normalized_distribution(model):
    g = np.linspace(-6, 6, 24001)
    X = np.tile([3, 1, -1.55], (len(g), 1))
    density = np.exp(predict_flow(model, X, g, .9))
    F = predict_flow(model, X, g, .9, cdf=True)
    assert trapezoid(density, g) == pytest.approx(1, abs=1e-4)
    assert F[0] < 1e-7 and F[-1] > 1-1e-7
    assert np.min(np.diff(F)) >= -1e-6
    mask = (np.abs(g+1.75)>.005)&(np.abs(g-2.5)>.005)
    assert np.quantile(np.abs(np.gradient(F,g)[mask]-density[mask]), .99) < .001


def test_grid_crps_matches_analytic_gaussian_mixture():
    model = {'kind':'mixture_power', 'beta':[.8,.2,.1], 'delta':.5,
             'sigma_slow':.08,'sigma_fast':.35,'gate_coef':[1,.3,-.2]}
    X = np.array([[-.5,-.1],[0,0],[.5,.1]])
    z = np.array([.3, .9, 1.3]);g = np.linspace(-4,4,4001)
    F = cdf(model,np.repeat(X,len(g),axis=0),np.tile(g,len(X))).reshape(len(X),len(g))
    scores = grid_scores(g,F,z)
    assert scores['crps'] == pytest.approx(crps(model,X,z),abs=2e-5)


def test_posterior_prior_measure_is_explicit():
    g = np.linspace(0,4,4001);ll = np.zeros_like(g)
    log_summary,_ = posterior_summary(g,ll)
    linear_summary,_ = posterior_summary(g,ll,linear_age_prior=True)
    assert log_summary['median_myr'] == pytest.approx(100,abs=.001)
    assert linear_summary['median_myr'] == pytest.approx(5000.5,rel=2e-6)


def test_flow_age_inference_does_not_read_reference_age(model):
    import pandas as pd
    frame=pd.DataFrame({'Prot':[5.,10.],'BPRP_0':[.8,1.2],'e_BPRP_0':[.03,.02],
                        'membership':[.9,1.],'fiducial_age_Myr':[100.,200.]})
    grid=np.geomspace(80,5000,31)
    before=flow_age_profile(model,frame,grid)
    frame['fiducial_age_Myr']=999999.
    assert np.array_equal(before,flow_age_profile(model,frame,grid))

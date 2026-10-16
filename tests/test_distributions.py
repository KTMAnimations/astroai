import numpy as np
import pytest
from scipy.optimize import check_grad

from astrogyro.distributions import (basis, _mixture_objective, _hetero_objective,
                                    cdf, crps, fit_distribution, log_density, quantiles)


def test_likelihood_gradients_match_finite_differences():
    rng = np.random.default_rng(13)
    X = rng.uniform([-.8, -.15], [.5, .15], size=(80, 2))
    B, H = basis(X, True), basis(X)
    z, w = rng.normal(1, .2, len(X)), np.full(len(X), 1/len(X))
    theta = np.r_[1., .2, .1, .01, .01, .01, .6, np.log(.1), np.log(.3), 1., .2, .1]
    args = (B, H, z, w, .01)
    assert check_grad(lambda t: _mixture_objective(t, *args)[0], lambda t: _mixture_objective(t, *args)[1], theta) < 1e-4
    theta = np.r_[theta[:6], np.log(.2), .1, .1]
    args = (B, H, z, w)
    assert check_grad(lambda t: _hetero_objective(t, *args)[0], lambda t: _hetero_objective(t, *args)[1], theta) < 1e-4


def test_gaussian_crps_quantiles_and_density():
    model = {"kind": "gaussian_power", "beta": [0, 0, 0], "sigma": 1.}
    X = np.zeros((1, 2))
    assert crps(model, X, [0])[0] == pytest.approx((np.sqrt(2)-1)/np.sqrt(np.pi))
    q = quantiles(model, X, [.025, .5, .975])[0]
    assert q == pytest.approx([-1.9599639845, 0., 1.9599639845], abs=1e-8)
    grid = np.linspace(-8, 8, 4001)
    density = np.exp(log_density(model, np.zeros((len(grid), 2)), grid))
    assert np.trapezoid(density, grid) == pytest.approx(1., abs=1e-8)


def test_mixture_is_normalized_and_has_ordered_quantiles():
    model = {"kind": "mixture_power", "beta": [1., .2, .1], "delta": .7,
             "sigma_slow": .08, "sigma_fast": .35, "gate_coef": [1.2, .3, -.2]}
    grid = np.linspace(-5, 5, 6001)
    X = np.zeros((len(grid), 2))
    assert np.trapezoid(np.exp(log_density(model, X, grid)), grid) == pytest.approx(1., abs=1e-8)
    X = np.array([[-1., -.1], [0., .1]])
    q = quantiles(model, X)
    assert (np.diff(q, axis=1) > 0).all()
    for j, probability in enumerate([.05, .16, .5, .84, .95]):
        assert cdf(model, X, q[:, j]) == pytest.approx([probability]*2, abs=1e-8)


def test_mixture_recovers_synthetic_slow_sequence_without_dropping_tail():
    rng = np.random.default_rng(14)
    X = rng.uniform([-.8, -.15], [.5, .15], (800, 2))
    slow = 1. + .4*X[:, 0] + .7*X[:, 1]
    tail = rng.random(len(X)) < .2
    z = slow + rng.normal(0, .06, len(X))
    z[tail] -= .65 + rng.normal(0, .25, tail.sum())
    model = fit_distribution(X, z, np.ones(len(z)), "mixture_power")
    assert model["converged"]
    assert np.sqrt(np.mean((basis(X) @ model["beta"] - slow)**2)) < .03
    assert model["n_discovery"] == 800

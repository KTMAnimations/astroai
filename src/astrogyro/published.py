"""Adapters for the pinned ChronoFlow network; no changes to upstream code."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import zuko
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logsumexp

from .data import ROOT

BOUNDS = (-1.75, 2.5)


def new_flow(seed=0):
    torch.manual_seed(seed)
    return zuko.flows.NSF(1, 3, transforms=3)


def load_flow(path):
    model = new_flow()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def contexts(frame, age_myr=None, default_log_error=None):
    age = frame.fiducial_age_Myr.to_numpy() if age_myr is None else np.broadcast_to(age_myr, len(frame))
    error = (np.log10(frame.e_BPRP_0.to_numpy()) if default_log_error is None
             else np.full(len(frame), default_log_error))
    X = np.column_stack([np.log10(age), frame.BPRP_0.to_numpy(), error])
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite age, color or color uncertainty")
    return X


def tensor_log_density(model, X, z, membership, outlier=.05):
    log_flow = model(X).log_prob(z.reshape(-1, 1))
    weight = membership*(1-outlier)
    background = torch.where((z >= BOUNDS[0]) & (z <= BOUNDS[1]),
                             torch.full_like(z, -np.log(BOUNDS[1]-BOUNDS[0])), -torch.inf)
    return torch.logaddexp(torch.log(weight)+log_flow, torch.log1p(-weight)+background)


def predict_flow(model, X, z, membership=1., outlier=.05, cdf=False, batch=8192):
    X, z = np.asarray(X), np.asarray(z)
    membership = np.broadcast_to(membership, len(z))
    results = []
    with torch.no_grad():
        for start in range(0, len(z), batch):
            stop = start+batch
            xx = torch.as_tensor(X[start:stop], dtype=torch.float32)
            zz = torch.as_tensor(z[start:stop], dtype=torch.float32)
            mm = torch.tensor(membership[start:stop].copy(), dtype=torch.float32)
            if cdf:
                transformed = model(xx).transform(zz[:, None]).squeeze(-1)
                flow_cdf = torch.special.ndtr(transformed)
                bg = torch.clamp((zz-BOUNDS[0])/(BOUNDS[1]-BOUNDS[0]), 0, 1)
                weight = mm*(1-outlier)
                value = weight*flow_cdf+(1-weight)*bg
            else:
                value = tensor_log_density(model, xx, zz, mm, outlier)
            results.append(value.numpy().astype(float))
    return np.concatenate(results)


def grid_scores(grid, F, observed):
    """CDF integration; split at the observation to avoid a step-grid CRPS bias."""
    grid, F, observed = np.asarray(grid), np.asarray(F), np.asarray(observed)
    if np.any(np.diff(F, axis=1) < -2e-6):
        raise ValueError("Nonmonotonic CDF")
    if np.any((observed < grid[0]) | (observed > grid[-1])):
        raise ValueError("Observed period outside integration domain")
    integral_F = cumulative_trapezoid(F, grid, axis=1, initial=0)
    integral_F_at_z = np.array([np.interp(z, grid, integral_F[i]) for i, z in enumerate(observed)])
    crps = trapezoid(F**2, grid, axis=1)-2*(integral_F[:, -1]-integral_F_at_z)+grid[-1]-observed
    qs = np.array([[np.interp(p, f, grid) for p in [.05, .16, .5, .84, .95]] for f in F])
    return {"mean_log_period": grid[0]+trapezoid(1-F, grid, axis=1),
            "q05": qs[:, 0], "q16": qs[:, 1], "median_log_period": qs[:, 2],
            "q84": qs[:, 3], "q95": qs[:, 4], "crps": crps,
            "tail_mass": F[:, 0]+1-F[:, -1]}


def evaluate_flow(model, X, z, membership=1., grid=None):
    grid = np.linspace(-4, 4, 4001) if grid is None else np.asarray(grid)
    n = len(z)
    F = predict_flow(model, np.repeat(X, len(grid), axis=0), np.tile(grid, n),
                     np.repeat(np.broadcast_to(membership, n), len(grid)), cdf=True).reshape(n, len(grid))
    pred = grid_scores(grid, F, z)
    pred.update(nll=-predict_flow(model, X, z, membership), pit=predict_flow(model, X, z, membership, cdf=True))
    return pred


def flow_age_profile(model, frame, age_grid, membership=None, default_log_error=None):
    X = np.tile(contexts(frame, age_myr=1., default_log_error=default_log_error), (len(age_grid), 1))
    X[:, 0] = np.repeat(np.log10(age_grid), len(frame))
    z = np.tile(np.log10(frame.Prot.to_numpy()), len(age_grid))
    mem = frame.membership.to_numpy() if membership is None else np.broadcast_to(membership, len(frame))
    return predict_flow(model, X, z, np.tile(mem, len(age_grid))).reshape(len(age_grid), len(frame)).sum(axis=1)


def posterior_summary(log_age_grid, log_likelihood, linear_age_prior=False, scatter_dex=0.):
    from scipy.ndimage import gaussian_filter1d
    g = np.asarray(log_age_grid)
    logp = np.asarray(log_likelihood)+(g*np.log(10) if linear_age_prior else 0)
    p = np.exp(logp-logsumexp(logp))
    if scatter_dex:
        p = gaussian_filter1d(p, scatter_dex/(g[1]-g[0]), mode="constant")
    p /= trapezoid(p, g)
    cdf = cumulative_trapezoid(p, g, initial=0)
    q = np.interp([.16, .5, .84], cdf, g)
    return {"q16_myr": float(10**q[0]), "median_myr": float(10**q[1]), "q84_myr": float(10**q[2])}, p

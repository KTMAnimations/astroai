"""Metrics with explicit invalid-prediction and independent-unit handling."""
from __future__ import annotations

import numpy as np


def rmse(y, prediction, weights=None) -> float:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or y.size == 0:
        raise ValueError("Predictions and observations must have the same nonempty shape")
    if not np.isfinite(y).all():
        raise ValueError("Observations must be finite")
    if not np.isfinite(prediction).all():
        return float("inf")
    return float(np.sqrt(np.average((y - prediction) ** 2, weights=weights)))


def macro_rmse(y, prediction, groups) -> float:
    """Mean of within-cluster RMSEs; each cluster has one vote."""
    y, prediction, groups = np.asarray(y), np.asarray(prediction), np.asarray(groups)
    if not np.isfinite(prediction).all():
        return float("inf")
    return float(np.mean([rmse(y[groups == g], prediction[groups == g]) for g in np.unique(groups)]))


def paired_cluster_interval(differences, seed=20261015, n_boot=20000):
    """Percentile interval for mean paired cluster scores.

    Inputs must contain one value per cluster (average algorithm seeds first).
    This describes variation across observed clusters, not systematic age errors.
    """
    delta = np.asarray(differences, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("Need at least two finite paired cluster differences")
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"mean_difference": float(delta.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_clusters": len(delta), "n_bootstrap": n_boot}

"""Transparent baselines fitted with weighted linear least squares in log space."""
from __future__ import annotations

import numpy as np

NAMES = ("fixed_skumanich", "power_law", "quadratic")


def design(X, name):
    x, y = np.asarray(X).T
    if name == "fixed_skumanich":
        return np.ones((len(X), 1))
    if name == "power_law":
        return np.column_stack([np.ones(len(X)), x, y])
    if name == "quadratic":
        return np.column_stack([np.ones(len(X)), x, y, x*x, x*y, y*y])
    raise ValueError(name)


def offset(X, name):
    return 0.5 * X[:, 0] + X[:, 1] if name == "fixed_skumanich" else np.zeros(len(X))


def fit(X, target, name, weights=None):
    X, target = np.asarray(X), np.asarray(target)
    weights = np.ones(len(target)) if weights is None else np.asarray(weights)
    A = design(X, name)
    coef = np.linalg.lstsq(A * np.sqrt(weights[:, None]),
                         (target - offset(X, name)) * np.sqrt(weights), rcond=None)[0]
    return {"name": name, "coefficients": coef.tolist(), "n_parameters": len(coef)}


def predict(model, X):
    X = np.asarray(X)
    return design(X, model["name"]) @ np.asarray(model["coefficients"]) + offset(X, model["name"])

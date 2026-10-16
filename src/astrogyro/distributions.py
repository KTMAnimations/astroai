"""Conditional rotation distributions with explicit likelihoods and proper scores."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtr

KINDS = ("gaussian_power", "gaussian_quadratic", "gaussian_hetero",
         "mixture_power", "mixture_quadratic", "gaussian_rbf")
LOG2PI = np.log(2 * np.pi)


def basis(X, quadratic=False):
    X = np.asarray(X, dtype=float)
    u, v = (X[:, 0] + .2) / .6, X[:, 1] / .15
    cols = [np.ones(len(X)), u, v]
    if quadratic:
        cols += [u*u, u*v, v*v]
    return np.column_stack(cols)


def rbf_basis(X):
    linear = basis(X)
    # Fixed feature-only centres; no held-out periods enter this representation.
    a, c = np.meshgrid(np.linspace(-1, np.log10(3.87), 5), np.linspace(np.log10(.65), np.log10(1.5), 4))
    centers = np.column_stack([a.ravel(), c.ravel()])
    rbf = np.exp(-.5 * np.sum(((np.asarray(X)[:, None, :] - centers[None]) / [.4, .12])**2, axis=2))
    return np.column_stack([linear, rbf])


def _mixture_objective(theta, B, H, z, w, penalty):
    p = B.shape[1]
    beta, delta, l1, l2, gate = theta[:p], theta[p], theta[p+1], theta[p+2], theta[p+3:]
    mu = B @ beta
    e1, e2 = z - mu, z - mu + delta
    v1, v2 = np.exp(2*l1), np.exp(2*l2)
    g = H @ gate
    lp1 = -np.logaddexp(0, -g) - l1 - .5*LOG2PI - .5*e1*e1/v1
    lp2 = -np.logaddexp(0, g) - l2 - .5*LOG2PI - .5*e2*e2/v2
    lp = np.logaddexp(lp1, lp2)
    r1, r2 = np.exp(lp1 - lp), np.exp(lp2 - lp)
    loss = -np.sum(w * lp)
    grad = np.r_[-B.T @ (w * (r1*e1/v1 + r2*e2/v2)),
                 np.sum(w*r2*e2/v2), np.sum(w*r1*(1-e1*e1/v1)),
                 np.sum(w*r2*(1-e2*e2/v2)), H.T @ (w*(expit(g)-r1))]
    if p > 3:
        loss += penalty * np.sum(beta[3:]**2)
        grad[3:p] += 2*penalty*beta[3:]
    loss += .001*np.sum(gate[1:]**2)
    grad[p+4:] += .002*gate[1:]
    return float(loss), grad


def _hetero_objective(theta, B, H, z, w):
    p = B.shape[1]
    residual = z - B @ theta[:p]
    logsigma = H @ theta[p:]
    invvar = np.exp(-2*logsigma)
    loss = np.sum(w*(logsigma + .5*LOG2PI + .5*residual**2*invvar))
    grad = np.r_[-B.T @ (w*residual*invvar), H.T @ (w*(1-residual**2*invvar))]
    return float(loss), grad


def fit_distribution(X, z, weights, kind, penalty=0.):
    if kind not in KINDS:
        raise ValueError(kind)
    z, w = np.asarray(z), np.asarray(weights, dtype=float)
    w = w/w.sum()
    quadratic = kind in ("gaussian_quadratic", "gaussian_hetero", "mixture_quadratic")
    B = rbf_basis(X) if kind == "gaussian_rbf" else basis(X, quadratic)
    H = basis(X)
    ridge = np.zeros(B.shape[1])
    if kind == "gaussian_rbf":
        ridge[3:] = penalty
    beta = np.linalg.lstsq(B*np.sqrt(w[:, None]), z*np.sqrt(w), rcond=None)[0]
    if np.any(ridge):
        beta = np.linalg.solve(B.T @ (w[:, None]*B) + np.diag(ridge), B.T @ (w*z))
    sigma = max(.025, float(np.sqrt(np.sum(w*(z - B @ beta)**2))))
    result = {"kind": kind, "penalty": penalty, "beta": beta.tolist(), "sigma": sigma,
              "converged": True, "n_discovery": len(z)}
    if kind == "gaussian_hetero":
        initial = np.r_[beta, np.log(sigma), 0., 0.]
        optimized = minimize(_hetero_objective, initial, args=(B, H, z, w), jac=True,
                             method="L-BFGS-B", bounds=[(None, None)]*len(beta) + [(-4, 0), (-2, 2), (-2, 2)],
                             options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-7})
        result.update(beta=optimized.x[:len(beta)].tolist(), log_sigma_coef=optimized.x[len(beta):].tolist(),
                      converged=bool(optimized.success), objective=float(optimized.fun), optimizer_message=str(optimized.message))
    elif kind.startswith("mixture"):
        attempts = []
        for delta, shift in [(.35, .08), (.7, .12), (1., .2)]:
            start = beta.copy(); start[0] += shift
            initial = np.r_[start, delta, np.log(.08), np.log(.35), 1.5, .4, 0.]
            bounds = [(None, None)]*len(beta) + [(.15, 1.5), (np.log(.025), np.log(.3)),
                       (np.log(.12), np.log(1.)), (-6, 6), (-6, 6), (-6, 6)]
            fit = minimize(_mixture_objective, initial, args=(B, H, z, w, penalty), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-7})
            attempts.append(fit)
        valid = [r for r in attempts if r.success and np.isfinite(r.fun)]
        if not valid:
            raise RuntimeError("No mixture initialization converged")
        best = min(valid, key=lambda r: r.fun)
        p = len(beta)
        result.update(beta=best.x[:p].tolist(), delta=float(best.x[p]), sigma_slow=float(np.exp(best.x[p+1])),
                      sigma_fast=float(np.exp(best.x[p+2])), gate_coef=best.x[p+3:].tolist(),
                      objective=float(best.fun), converged=True,
                      attempts=[{"converged": bool(r.success), "objective": float(r.fun), "iterations": int(r.nit)} for r in attempts])
    return result


def components(model, X):
    kind = model["kind"]
    B = rbf_basis(X) if kind == "gaussian_rbf" else basis(X, len(model["beta"]) == 6)
    mu = B @ np.asarray(model["beta"])
    if kind.startswith("mixture"):
        gate = expit(basis(X) @ np.asarray(model["gate_coef"]))
        means = np.column_stack([mu, mu-model["delta"]])
        sigmas = np.tile([model["sigma_slow"], model["sigma_fast"]], (len(X), 1))
        weights = np.column_stack([gate, 1-gate])
    else:
        sigma = np.exp(basis(X) @ np.asarray(model["log_sigma_coef"])) if kind == "gaussian_hetero" else np.full(len(X), model["sigma"])
        means, sigmas, weights = mu[:, None], sigma[:, None], np.ones((len(X), 1))
    return means, sigmas, weights


def log_density(model, X, z):
    mu, sigma, weight = components(model, X)
    return logsumexp(np.log(weight) - np.log(sigma) - .5*LOG2PI - .5*((np.asarray(z)[:, None]-mu)/sigma)**2, axis=1)


def cdf(model, X, z):
    mu, sigma, weight = components(model, X)
    return np.sum(weight*ndtr((np.asarray(z)[:, None]-mu)/sigma), axis=1)


def quantiles(model, X, probabilities=(.05, .16, .5, .84, .95)):
    mu, sigma, weight = components(model, X)
    lower, upper = np.min(mu-12*sigma, axis=1), np.max(mu+12*sigma, axis=1)
    output = []
    for probability in probabilities:
        lo, hi = lower.copy(), upper.copy()
        for _ in range(50):
            mid = (lo+hi)/2
            below = np.sum(weight*ndtr((mid[:, None]-mu)/sigma), axis=1) < probability
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        output.append((lo+hi)/2)
    return np.column_stack(output)


def crps(model, X, z):
    """Closed-form CRPS for a univariate Gaussian mixture (units: dex)."""
    mu, sigma, w = components(model, X)
    def A(delta, scale):
        t = delta/scale
        return 2*scale*np.exp(-.5*t*t)/np.sqrt(2*np.pi) + delta*(2*ndtr(t)-1)
    first = np.sum(w*A(np.asarray(z)[:, None]-mu, sigma), axis=1)
    second = np.zeros(len(X))
    for j in range(mu.shape[1]):
        for k in range(mu.shape[1]):
            second += w[:, j]*w[:, k]*A(mu[:, j]-mu[:, k], np.sqrt(sigma[:, j]**2+sigma[:, k]**2))
    return first - .5*second


def evaluate_distribution(model, X, z):
    mu, sigma, weights = components(model, X)
    mean = np.sum(weights*mu, axis=1)
    q = quantiles(model, X)
    return {"mean_log_period": mean, "q05": q[:, 0], "q16": q[:, 1], "median_log_period": q[:, 2],
            "q84": q[:, 3], "q95": q[:, 4], "nll": -log_density(model, X, z),
            "crps": crps(model, X, z), "pit": cdf(model, X, z)}

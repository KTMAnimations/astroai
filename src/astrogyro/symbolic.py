"""Small, deterministic PySR searches with saved candidates and honest selection."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import sympy as sp

from .metrics import macro_rmse, rmse

VARIABLES = ["log_age", "log_color"]


def evaluate(expression, X):
    symbols = sp.symbols("log_age log_color")
    expr = sp.sympify(str(expression), locals=dict(zip(VARIABLES, symbols)))
    fn = sp.lambdify(symbols, expr, modules="numpy")
    with np.errstate(all="ignore"):
        values = np.asarray(fn(*np.asarray(X).T), dtype=float)
    if values.ndim == 0:
        values = np.full(len(X), float(values))
    if values.shape != (len(X),):
        raise ValueError("Unexpected prediction shape")
    return values


def search(X, y, weights, seed, config, output_dir):
    from pysr import PySRRegressor
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PySRRegressor(
        niterations=config["iterations"], populations=config["populations"],
        population_size=config["population_size"], ncycles_per_iteration=config["cycles"],
        maxsize=config["maxsize"], binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt"],
        nested_constraints={"exp": {"exp": 0}, "log": {"log": 0}},
        model_selection="best", batching=False, deterministic=True,
        parallelism="serial", random_state=int(seed), precision=64,
        progress=False, verbosity=0, output_directory=str(output_dir),
    )
    model.fit(np.asarray(X), np.asarray(y), weights=weights, variable_names=VARIABLES)
    table = model.equations_.copy()
    table["candidate_id"] = np.arange(len(table))
    table["expression"] = table.sympy_format.astype(str)
    default_index = int(model.get_best().name)
    table["default_selected"] = table.index == default_index
    return table[["candidate_id", "complexity", "loss", "score", "equation", "expression", "default_selected"]].reset_index(drop=True)


def select_candidates(table, X_val, y_val, groups=None):
    result = table.copy()
    errors = []
    for expression in result.expression:
        try:
            pred = evaluate(expression, X_val)
            score = rmse(y_val, pred) if groups is None else macro_rmse(y_val, pred, groups)
        except (ValueError, TypeError, OverflowError):
            score = float("inf")
        errors.append(score)
    result["validation_rmse"] = errors
    eligible = result[np.isfinite(result.validation_rmse)]
    if eligible.empty:
        raise RuntimeError("Every candidate failed on validation data")
    selected = eligible.sort_values(["validation_rmse", "complexity", "candidate_id"]).iloc[0]
    result["validation_selected"] = result.candidate_id == selected.candidate_id
    return result

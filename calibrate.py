"""Probability recalibration (Platt scaling on the logit, fit per stat).

The simulator's raw probabilities are systematically a bit too confident on overs. We fit
    p_cal = sigmoid(a * logit(p_raw) + b)
per stat on training weeks and apply it to live output. a < 1 shrinks extreme probabilities
toward the middle; b < 0 shifts overs down. Stored in calibration.json.
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit(df):
    """df: backtest rows with columns stat, p, hit."""
    params = {}
    for stat, g in df.groupby("stat"):
        x, y = _logit(g["p"].values), g["hit"].values

        def nll(ab):
            q = 1 / (1 + np.exp(-(ab[0] * x + ab[1])))
            q = np.clip(q, EPS, 1 - EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        a, b = minimize(nll, [1.0, 0.0], method="Nelder-Mead").x
        params[stat] = {"a": round(float(a), 4), "b": round(float(b), 4)}
    return params


def save(params):
    json.dump(params, open(PATH, "w"), indent=2)


def load():
    return json.load(open(PATH)) if os.path.exists(PATH) else {}


def apply(p, stat, params=None):
    params = params if params is not None else load()
    key = "anytime_td" if stat == "anytime_td" else stat
    if key not in params:
        return p
    a, b = params[key]["a"], params[key]["b"]
    return float(1 / (1 + np.exp(-(a * _logit(np.asarray(p)) + b))))

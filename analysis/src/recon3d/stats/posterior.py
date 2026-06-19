"""Posterior summary helpers shared by Bayesian LMM scripts."""

from __future__ import annotations

from typing import Any, Callable

import arviz as az
import numpy as np
import pandas as pd


IntervalFunc = Callable[[np.ndarray, float], tuple[float, float]]


def prob_direction(draws: np.ndarray) -> float:
    samples = np.asarray(draws, dtype=float).reshape(-1)
    return float(max((samples > 0).mean(), (samples < 0).mean()))


def hdi_interval(draws: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    hdi = az.hdi(np.asarray(draws, dtype=float).reshape(-1), hdi_prob=prob)
    return float(hdi[0]), float(hdi[1])


def draw_summary(
    draws: np.ndarray,
    *,
    prob: float = 0.95,
    interval_func: IntervalFunc = hdi_interval,
    center_name: str = "beta",
) -> dict[str, float]:
    samples = np.asarray(draws, dtype=float).reshape(-1)
    lower, upper = interval_func(samples, prob)
    return {
        center_name: float(np.mean(samples)),
        "median": float(np.median(samples)),
        "HDI_lower": lower,
        "HDI_upper": upper,
        "prob_greater": float((samples > 0).mean()),
        "prob_less": float((samples < 0).mean()),
        "prob_direction": prob_direction(samples),
    }


def posterior_diagnostics(idata: Any, *, model: str) -> pd.DataFrame:
    summary = az.summary(idata)
    cols = [col for col in ["mean", "sd", "ess_bulk", "ess_tail", "r_hat"] if col in summary.columns]
    diagnostics = summary[cols].reset_index(names="term")
    diagnostics.insert(0, "model", model)
    return diagnostics

"""Thin wrappers around R emmeans/emtrends outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import rpy2.robjects as ro
from pandas.api.types import CategoricalDtype
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import rpy2py
from rpy2.robjects.packages import importr
from scipy import stats

from recon3d.stats.frequentist_lmm import LMMHelper


DenormFunc = Callable[[float, pd.DataFrame], float]


def _factor_levels_from_data(data: pd.DataFrame, factor_var: str, fallback: list[str]) -> list[str]:
    values = data[factor_var]
    if isinstance(values.dtype, CategoricalDtype):
        return list(values.cat.categories)
    observed = [str(v) for v in pd.unique(values)]
    ordered = [level for level in fallback if level in observed]
    extras = [level for level in observed if level not in ordered]
    return ordered + extras


def factor_levels_from_bayesian_model(
    result: dict[str, Any],
    formula: str,
    factor_var: str,
    fallback: list[str],
) -> list[str]:
    model = result.get("models", {}).get(formula)
    data = getattr(model, "data", None)
    if data is not None and factor_var in data.columns:
        return _factor_levels_from_data(data, factor_var, fallback)
    return fallback


def estimate_emtrends(
    fitted: dict[str, Any],
    *,
    x_var: str,
    specs: str,
    denorm_slope: DenormFunc,
    by: str | None = None,
    one_sided: str = ">",
) -> pd.DataFrame:
    trends = LMMHelper.estimate_emtrends(
        fitted,
        var=x_var,
        specs=specs,
        by=by,
        side=one_sided,
    )
    trend_col = f"{x_var}.trend"
    if trend_col not in trends.columns:
        raise ValueError(f"Expected emtrends column '{trend_col}' not found.")
    trends = trends.copy()
    trends["slope_z"] = pd.to_numeric(trends[trend_col], errors="coerce")
    trends["slope_orig"] = trends["slope_z"].map(lambda value: denorm_slope(float(value), fitted["data_used"]))
    for col in ["lower.CL", "upper.CL"]:
        if col in trends.columns:
            trends[f"{col}_orig"] = pd.to_numeric(trends[col], errors="coerce").map(
                lambda value: denorm_slope(float(value), fitted["data_used"])
            )
    return trends


def pairwise_emtrends_contrasts(
    fitted: dict[str, Any],
    *,
    x_var: str,
    specs: str,
    by: str | None,
    side: str | None,
    denorm_slope: DenormFunc,
    ci_level: float,
    fallback_by_levels: dict[str, list[str]],
    fallback_specs_levels: dict[str, list[str]],
) -> pd.DataFrame:
    pandas2ri.activate()
    importr("emmeans")
    model_name = fitted["r_model_name"]
    spec_expr = f"~ {specs}"
    if by:
        spec_expr = f"{spec_expr} | {by}"
    ro.globalenv["emtrends_contrast_model"] = ro.r(model_name)
    ro.r(f'tr <- emmeans::emtrends(emtrends_contrast_model, specs = {spec_expr}, var = "{x_var}")')
    by_arg = f', by = "{by}"' if by else ""
    if side in (">", "<"):
        ro.r(f'con <- pairs(tr{by_arg}, side = "{side}")')
    else:
        ro.r(f"con <- pairs(tr{by_arg})")
    contrasts = rpy2py(ro.r("as.data.frame(summary(con, infer = c(TRUE, TRUE)))")).copy()
    contrasts = _restore_pairwise_contrast_labels(
        contrasts,
        fitted=fitted,
        specs=specs,
        by=by,
        fallback_by_levels=fallback_by_levels,
        fallback_specs_levels=fallback_specs_levels,
    )
    for col in ["estimate", "lower.CL", "upper.CL"]:
        if col in contrasts.columns:
            contrasts[f"{col}_orig"] = pd.to_numeric(contrasts[col], errors="coerce").map(
                lambda value: denorm_slope(float(value), fitted["data_used"])
            )
    if side in (">", "<"):
        contrasts = _replace_contrast_ci_with_two_sided_ci(
            contrasts,
            fitted=fitted,
            denorm_slope=denorm_slope,
            ci_level=ci_level,
        )
    return contrasts


def _replace_contrast_ci_with_two_sided_ci(
    contrasts: pd.DataFrame,
    *,
    fitted: dict[str, Any],
    denorm_slope: DenormFunc,
    ci_level: float,
) -> pd.DataFrame:
    out = contrasts.copy()
    required = {"estimate", "SE", "df"}
    if not required.issubset(out.columns):
        return out

    estimate = pd.to_numeric(out["estimate"], errors="coerce")
    se = pd.to_numeric(out["SE"], errors="coerce")
    df = pd.to_numeric(out["df"], errors="coerce")
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=df)
    out["lower.CL"] = estimate - t_crit * se
    out["upper.CL"] = estimate + t_crit * se
    out["lower.CL_orig"] = out["lower.CL"].map(lambda value: denorm_slope(float(value), fitted["data_used"]))
    out["upper.CL_orig"] = out["upper.CL"].map(lambda value: denorm_slope(float(value), fitted["data_used"]))
    return out


def _restore_pairwise_contrast_labels(
    contrasts: pd.DataFrame,
    *,
    fitted: dict[str, Any],
    specs: str,
    by: str | None,
    fallback_by_levels: dict[str, list[str]],
    fallback_specs_levels: dict[str, list[str]],
) -> pd.DataFrame:
    out = contrasts.copy()
    data_used = fitted["data_used"]

    if by and by in out.columns:
        by_levels = _factor_levels_from_data(data_used, by, fallback_by_levels.get(by, []))
        numeric_by = pd.to_numeric(out[by], errors="coerce")
        if numeric_by.notna().all():
            by_map = {idx + 1: level for idx, level in enumerate(by_levels)}
            out[by] = numeric_by.astype(int).map(by_map).fillna(out[by].astype(str))

    if "contrast" in out.columns and specs in fallback_specs_levels:
        spec_levels = _factor_levels_from_data(data_used, specs, fallback_specs_levels[specs])
        if len(spec_levels) == 2:
            contrast_label = f"{spec_levels[0]} - {spec_levels[1]}"
            numeric_contrast = pd.to_numeric(out["contrast"], errors="coerce")
            if numeric_contrast.notna().all() or set(out["contrast"].astype(str)) == {"1"}:
                out["contrast"] = contrast_label
    return out


def treatment_vs_control_emtrends(
    fitted: dict[str, Any],
    *,
    x_var: str,
    factor: str,
    reference_level: str,
    denorm_slope: DenormFunc,
) -> pd.DataFrame:
    pandas2ri.activate()
    importr("emmeans")
    model_name = fitted["r_model_name"]
    ro.globalenv["roi_contrast_model"] = ro.r(model_name)
    ro.r(
        f"""
        tr <- emmeans::emtrends(roi_contrast_model, specs = ~ {factor}, var = "{x_var}")
        factor_levels <- as.character(as.data.frame(tr)$`{factor}`)
        reference_index <- match("{reference_level}", factor_levels)
        if (is.na(reference_index)) {{
          stop("Reference level is not present in emtrends levels")
        }}
        con <- emmeans::contrast(tr, method = "trt.vs.ctrl", ref = reference_index, adjust = "none")
        """
    )
    contrasts = rpy2py(ro.r('as.data.frame(summary(con, infer = c(TRUE, TRUE), adjust = "none"))')).copy()
    for col in ["estimate", "lower.CL", "upper.CL"]:
        if col in contrasts.columns:
            contrasts[f"{col}_orig"] = pd.to_numeric(contrasts[col], errors="coerce").map(
                lambda value: denorm_slope(float(value), fitted["data_used"])
            )
    return contrasts

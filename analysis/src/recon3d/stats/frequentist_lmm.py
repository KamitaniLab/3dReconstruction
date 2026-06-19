import re
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd
from rpy2.robjects import globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter, rpy2py
import rpy2.robjects as ro
import numpy as np
from scipy import stats as spstats

class LMMHelper:
    @staticmethod
    def _interval_quantile(interval_prob: float, sided: str) -> float:
        side = str(sided).strip().lower().replace("_", "-")
        alpha = 1.0 - float(interval_prob)
        if side in ("two-sided", "two sided", "two", "2"):
            return 1.0 - alpha / 2.0
        if side in ("one-sided", "one sided", "one", "1"):
            return 1.0 - alpha
        raise ValueError(f"sided must be 'two-sided' or 'one-sided', got {sided!r}")

    @staticmethod
    def _zscore_series(s: pd.Series, ddof: int = 0) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        mu = s_num.mean()
        sd = s_num.std(ddof=ddof)
        if pd.isna(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (s_num - mu) / sd

    @staticmethod
    def _apply_zscore(
        df: pd.DataFrame,
        cols: List[str],
        *,
        by: Optional[List[str]] = None,
        inplace: bool = True,
        ddof: int = 0,
    ) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                continue
            if by:
                z = out.groupby(by)[c].transform(lambda s: LMMHelper._zscore_series(s, ddof=ddof))
            else:
                z = LMMHelper._zscore_series(out[c], ddof=ddof)
            if inplace:
                out[c] = z
            else:
                out[f"{c}_z"] = z
        return out

    @staticmethod
    def _has_random_effects(formula: str) -> bool:
        """Judge by whether ( ... | ... ) is included (lme4 style)."""
        return bool(re.search(r"\([^()]*\|[^()]*\)", formula))

    @staticmethod
    def find_column(df: pd.DataFrame, candidates) -> Optional[str]:
        """Return the first candidate column name present in df."""
        for column in candidates:
            if column in df.columns:
                return column
        return None

    @staticmethod
    def _add_onesided_pvalues(coef_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add one-sided p-values to coefficient dataframe using two-sided p-values.

        Args:
            coef_df: Coefficient dataframe with 't value' and 'Pr(>|t|)' columns

        Returns:
            Modified coefficient dataframe with added columns:
            - 'Pr(>t)': Right-sided p-value (H1: β > 0)
            - 'Pr(<t)': Left-sided p-value (H1: β < 0)
        """
        coef_df = coef_df.copy()

        t_values = coef_df['t value'].values
        p_two_sided = coef_df['Pr(>|t|)'].values

        # Right-sided test: H1: β > 0
        # If t > 0: p_right = p_two_sided / 2
        # If t <= 0: p_right = 1 - p_two_sided / 2
        p_right = np.where(t_values > 0,
                          p_two_sided / 2,
                          1 - p_two_sided / 2)

        # Left-sided test: H1: β < 0
        # If t < 0: p_left = p_two_sided / 2
        # If t >= 0: p_left = 1 - p_two_sided / 2
        p_left = np.where(t_values < 0,
                         p_two_sided / 2,
                         1 - p_two_sided / 2)

        # Add to dataframe
        coef_df['Pr(>t)'] = p_right
        coef_df['Pr(<t)'] = p_left

        return coef_df

    @staticmethod
    def _add_confidence_interval(
        df: pd.DataFrame,
        *,
        interval_prob: float = 0.95,
        sided: str = "two-sided",
    ) -> pd.DataFrame:
        """Add CI_lower and CI_upper when estimate and SE columns are available."""
        out = df.copy()
        est_col = LMMHelper.find_column(out, ("Estimate", "estimate", "emmean", "lsmean"))
        se_col = LMMHelper.find_column(
            out,
            ("Std. Error", "Std.Error", "Std.Err.", "SE", "std.error"),
        )
        df_col = LMMHelper.find_column(out, ("df", "DF", "DenDF", "d.f.", "df.residual"))
        if est_col is None or se_col is None:
            out["CI_lower"] = np.nan
            out["CI_upper"] = np.nan
            return out

        interval_q = LMMHelper._interval_quantile(interval_prob=interval_prob, sided=sided)
        zcrit = float(spstats.norm.ppf(interval_q))
        est = pd.to_numeric(out[est_col], errors="coerce")
        se = pd.to_numeric(out[se_col], errors="coerce")
        crit = pd.Series(zcrit, index=out.index, dtype=float)
        if df_col is not None:
            dff = pd.to_numeric(out[df_col], errors="coerce")
            mask = dff.notna() & (dff > 0)
            if mask.any():
                crit.loc[mask] = dff.loc[mask].map(
                    lambda value: float(spstats.t.ppf(interval_q, df=value))
                )
        out["CI_lower"] = est - crit * se
        out["CI_upper"] = est + crit * se
        return out

    @staticmethod
    def fit_lmm_r(
        data: pd.DataFrame,
        formula: str,
        *,
        reml: bool = True,
        backend: str = "lmerTest",        # "lme4" or "lmerTest" (KR requires lmerTest)
        try_broom: bool = True,
        ensure_factor_cols: Optional[List[str]] = None,
        verbose: bool = False,
        zscore_cols: Optional[List[str]] = None,
        zscore_by: Optional[List[str]] = None,
        zscore_inplace: bool = True,
        zscore_ddof: int = 0,
        df_method: str = "Kenward-Roger",  # "Satterthwaite" / "Kenward-Roger" / "lme4"
        anova_type: Optional[int] = None,  # 2 or 3 (only for LMM, lmerTest)
        ref_levels: Optional[Dict[str, str]] = None,
        contrast_options: Optional[List[str]] = None,
        model_name: Optional[str] = None,  # 追加: モデル名を指定可能に
    ) -> Dict[str, Any]:
        """
        Automatically fit formula using R's lme4/lmerTest (with random effects) or stats::lm (without random effects).
        Also supports z-scoring of specified columns.
        """
        # 1) Preprocessing (z-scoring)
        df_use = data.copy()
        if zscore_cols:
            df_use = LMMHelper._apply_zscore(
                df_use, zscore_cols, by=zscore_by, inplace=zscore_inplace, ddof=zscore_ddof
            )

        # 2) Enable conversion
        pandas2ri.activate()

        # 3) Determine which model to use
        is_lmm = LMMHelper._has_random_effects(formula)

        # 4) Common R packages
        importr("base")
        utils = importr("utils")
        if contrast_options is not None:
            if len(contrast_options) != 2:
                raise ValueError("contrast_options must contain unordered and ordered contrast names.")
            unordered, ordered = contrast_options
            ro.r(f'options(contrasts = c("{unordered}", "{ordered}"))')

        # 5) Pass data to R side (use temporary name to avoid collision)
        r_df_name = f"df_r_{uuid.uuid4().hex[:8]}"
        globalenv[r_df_name] = df_use

        # 6) Specify factors
        if ensure_factor_cols:
            for col in ensure_factor_cols:
                if col in df_use.columns:
                    ro.r(f'{r_df_name}${col} <- base::factor({r_df_name}${col})')

        # ▼ Additional: Set reference category (relevel)
        if ref_levels:
            for col, ref in ref_levels.items():
                if col not in df_use.columns:
                    continue
                # Factorize in R (safe even if already factor)
                ro.r(f'{r_df_name}${col} <- base::factor({r_df_name}${col})')
                # Check existence in Python side (raise error if not present)
                if ref not in pd.Categorical(df_use[col]).categories:
                    raise ValueError(f'ref_levels["{col}"] reference level "{ref}" does not exist in data.')
                # Set reference in R
                ro.r(f'{r_df_name}${col} <- stats::relevel({r_df_name}${col}, ref = "{ref}")')

        # 7) Fit model
        if is_lmm:
            # If KR is requested but backend is lme4, switch to lmerTest
            if df_method.lower().startswith("kenward") and backend.lower() != "lmertest":
                backend = "lmerTest"

            if backend.lower() == "lmertest":
                importr("lmerTest")
                # pbkrtest may be needed for KR, load if available (will be called automatically if not)
                try:
                    importr("pbkrtest")
                except Exception:
                    pass
                lmer_ns = "lmerTest::lmer"
            else:
                importr("lme4")
                lmer_ns = "lme4::lmer"

            reml_str = "TRUE" if reml else "FALSE"
            ro.r(f'model <- {lmer_ns}({formula}, data = {r_df_name}, REML = {reml_str})')

            # ▼ Key point: use lmerTest::summary(..., ddf=) to create coefficient table
            if backend.lower() == "lmertest":
                # Normalize ddf candidates
                ddf_norm = ("Kenward-Roger" if df_method.lower().startswith("kenward")
                            else "Satterthwaite" if df_method.lower().startswith("satter")
                            else "lme4")
                ro.r(f'sum_obj <- summary(model, ddf = "{ddf_norm}")')
                summary_text = "\n".join(list(utils.capture_output(ro.r("sum_obj"))))
                # Get coefficient table from sum_obj (ddf setting is reflected)
                coef_df = rpy2py(ro.r("as.data.frame(coef(sum_obj))"))
            else:
                # For lme4, as before (ddf not calculated)
                summary_text = "\n".join(list(utils.capture_output(ro.r("summary(model)"))))
                coef_df = rpy2py(ro.r("as.data.frame(coef(summary(model)))"))

            # Add one-sided p-values
            coef_df = LMMHelper._add_onesided_pvalues(coef_df)
            coef_df = LMMHelper._add_confidence_interval(coef_df)

            if verbose:
                print(summary_text)

            varcorr_df = rpy2py(ro.r("as.data.frame(lme4::VarCorr(model))"))
            fixef = rpy2py(ro.r("lme4::fixef(model)"))
            ranef = rpy2py(ro.r("lme4::ranef(model)"))
            ngrps_names = list(rpy2py(ro.r("names(lme4::ngrps(model))")))
            ngrps_vals  = list(map(int, rpy2py(ro.r("as.integer(lme4::ngrps(model))"))))
            ngrps = dict(zip(ngrps_names, ngrps_vals))

            logLik = float(ro.r("as.numeric(logLik(model))")[0])
            AIC = float(ro.r("AIC(model)")[0])
            BIC = float(ro.r("BIC(model)")[0])
            sigma = float(ro.r("sigma(model)")[0])
            nobs = int(ro.r("stats::nobs(model)")[0])

            results: Dict[str, Any] = {
                "model_type": "LMM",
                "coef_df": coef_df,
                "varcorr_df": varcorr_df,
                "fixef": fixef,
                "ranef": ranef,
                "ngrps": ngrps,
                "stats": {
                    "logLik": logLik, "AIC": AIC, "BIC": BIC,
                    "sigma": sigma, "nobs": nobs, "REML": bool(reml),
                },
                "summary_text": summary_text,
                "data_used": df_use,
                # Additionally, return the actual df method used (None for LM)
                "df_method_used": (ddf_norm if backend.lower()=="lmertest" else "lme4"),
            }

            # ▼ (Optional) Also return ANOVA table: Type=2/3 + ddf
            if anova_type in (2, 3) and backend.lower() == "lmertest":
                ro.r(f'anv <- anova(model, ddf = "{ddf_norm}", type = {anova_type})')
                try:
                    results["anova_table"] = rpy2py(ro.r("as.data.frame(anv)"))
                except Exception:
                    pass

            # broom.mixed (if available)
            if try_broom:
                try:
                    importr("broom.mixed")
                    results["tidy"] = rpy2py(ro.r("broom.mixed::tidy(model)"))
                    results["glance"] = rpy2py(ro.r("broom.mixed::glance(model)"))
                except Exception:
                    pass

        else:
            # OLS: stats::lm
            importr("stats")
            ro.r(f"model <- stats::lm({formula}, data = {r_df_name})")
            summary_text = "\n".join(list(utils.capture_output(ro.r("summary(model)"))))
            if verbose:
                print(summary_text)

            coef_df = rpy2py(ro.r("as.data.frame(coef(summary(model)))"))

            # Add one-sided p-values
            coef_df = LMMHelper._add_onesided_pvalues(coef_df)
            coef_df = LMMHelper._add_confidence_interval(coef_df)

            fixef = rpy2py(ro.r("stats::coef(model)"))

            # Return empty for LMM-specific values
            varcorr_df = pd.DataFrame()
            ranef: Dict[str, pd.DataFrame] = {}
            ngrps: Dict[str, int] = {}

            logLik = float(ro.r("as.numeric(logLik(model))")[0])
            AIC = float(ro.r("AIC(model)")[0])
            BIC = float(ro.r("BIC(model)")[0])
            sigma = float(ro.r("sigma(model)")[0])      # Residual standard deviation
            nobs = int(ro.r("stats::nobs(model)")[0])

            results = {
                "model_type": "LM",
                "coef_df": coef_df,          # Columns: Estimate, Std. Error, t value, Pr(>|t|)
                "varcorr_df": varcorr_df,    # Empty
                "fixef": fixef,              # Coefficient vector
                "ranef": ranef,              # Empty
                "ngrps": ngrps,              # Empty
                "stats": {
                    "logLik": logLik, "AIC": AIC, "BIC": BIC,
                    "sigma": sigma, "nobs": nobs, "REML": None,
                },
                "summary_text": summary_text,
                "data_used": df_use,
            }

            # broom (if available)
            if try_broom:
                try:
                    importr("broom")
                    tidy_df = rpy2py(ro.r("broom::tidy(model)"))
                    glance_df = rpy2py(ro.r("broom::glance(model)"))
                    results["tidy"] = tidy_df
                    results["glance"] = glance_df
                except Exception:
                    pass

        # Actually used factor levels (for checking)
        factor_levels_used: Dict[str, List[str]] = {}
        cols_to_report = set(ensure_factor_cols or [])
        if ref_levels:
            cols_to_report |= set(ref_levels.keys())
        for col in cols_to_report:
            if col in df_use.columns:
                lv = list(rpy2py(ro.r(f'levels({r_df_name}${col})')))
                factor_levels_used[col] = lv
        results["factor_levels_used"] = factor_levels_used

        # model info
        if model_name is None:
            model_name = f"model_{uuid.uuid4().hex[:8]}"
        ro.r(f"{model_name} <- model")  # Save to R global environment
        results["r_model_name"] = model_name
        results["formula"] = formula
        results["backend"] = backend
        results["reml"] = bool(reml) if is_lmm else None
        results["df_method_requested"] = df_method
        results["preproc"] = {
            "zscore_cols": zscore_cols or [],
            "zscore_by": zscore_by or [],
            "zscore_inplace": zscore_inplace,
            "zscore_ddof": zscore_ddof,
            "ensure_factor_cols": ensure_factor_cols or [],
            "ref_levels": ref_levels or {},
            "contrast_options": contrast_options or [],
            "factor_levels_used": factor_levels_used,  # Already created
        }

        # 8) Cleanup
        try:
            ro.r(f"rm({r_df_name})")
        except Exception:
            pass

        return results

    @staticmethod
    def estimate_emtrends(
        fitted: Dict[str, Any],
        *,
        var: str,
        specs: Optional[str] = None,
        by: Optional[str] = None,
        side: Optional[str] = None,
        interval_prob: float = 0.95,
    ) -> pd.DataFrame:
        """
        Estimate marginal trends for a fitted R model using emmeans::emtrends.

        Args:
            fitted: Result dictionary returned by fit_lmm_r().
            var: Continuous predictor whose trend should be estimated.
            specs: Factor to estimate trends over. Use None for a single overall trend.
            by: Optional grouping factor for simple trends.
            side: Optional one-sided test direction, either ">" or "<".
            interval_prob: Confidence interval probability for the two-sided CI columns.
        """
        model_name = fitted.get("r_model_name")
        if not model_name:
            raise ValueError("'r_model_name' is missing in fitted result.")

        with localconverter(ro.default_converter + pandas2ri.converter):
            if not bool(ro.r(f'exists("{model_name}")')[0]):
                raise RuntimeError(f"R model object '{model_name}' not found.")
            importr("emmeans")
            ro.r("suppressPackageStartupMessages(library(emmeans))")
            ro.globalenv["emtrends_model"] = ro.r(model_name)

            if specs:
                spec_expr = f"~ {specs}"
                if by:
                    spec_expr = f"{spec_expr} | {by}"
            else:
                spec_expr = "~ 1"

            ro.r(f'tr <- emmeans::emtrends(emtrends_model, specs = {spec_expr}, var = "{var}")')
            if side in (">", "<"):
                ro.r(f'tr_side <- as.data.frame(summary(tr, infer = c(TRUE, TRUE), side = "{side}"))')
                trends = rpy2py(ro.r("tr_side"))
                ro.r("tr_ci <- as.data.frame(summary(tr, infer = c(TRUE, TRUE)))")
                trends_ci = rpy2py(ro.r("tr_ci"))
                for col in ("lower.CL", "upper.CL"):
                    if col in trends_ci.columns:
                        trends[col] = pd.to_numeric(trends_ci[col], errors="coerce").to_numpy()
            else:
                trends = rpy2py(ro.r("as.data.frame(summary(tr, infer = c(TRUE, TRUE)))"))

        return trends

    @staticmethod
    def test_emmeans_contrasts(
        fitted: Dict[str, Any],
        *,
        specs: str,
        by: Optional[str] = None,
        method: str = "pairwise",
        reverse: bool = False,
        interval_prob: float = 0.95,
        sided: str = "two-sided",
    ) -> pd.DataFrame:
        """
        Compute emmeans contrasts for a fitted R model.

        Args:
            fitted: Result dictionary returned by fit_lmm_r().
            specs: Factor whose marginal means should be contrasted.
            by: Optional grouping factor for simple contrasts.
            method: emmeans contrast method, e.g. "pairwise".
            reverse: Use revpairwise instead of pairwise when method is pairwise.
            interval_prob: Confidence interval probability.
            sided: "two-sided" or "one-sided" for CI width.
        """
        model_name = fitted.get("r_model_name")
        if not model_name:
            raise ValueError("'r_model_name' is missing in fitted result.")

        contrast_method = "revpairwise" if reverse and method == "pairwise" else method
        with localconverter(ro.default_converter + pandas2ri.converter):
            if not bool(ro.r(f'exists("{model_name}")')[0]):
                raise RuntimeError(f"R model object '{model_name}' not found.")
            importr("emmeans")
            ro.r("suppressPackageStartupMessages(library(emmeans))")
            ro.globalenv["emmeans_model"] = ro.r(model_name)
            spec_expr = f"~ {specs}"
            if by:
                spec_expr = f"{spec_expr} | {by}"
            ro.r(f"emm <- emmeans::emmeans(emmeans_model, specs = {spec_expr})")
            ro.r(f'con <- emmeans::contrast(emm, method = "{contrast_method}")')
            contrasts = rpy2py(ro.r("as.data.frame(summary(con))"))

        return LMMHelper._add_confidence_interval(
            contrasts,
            interval_prob=interval_prob,
            sided=sided,
        )

def predict_lmm_r(
    fitted: Dict[str, Any],
    newdata: pd.DataFrame,
    *,
    include_random_effects: bool = True,   # 条件付き予測（RE込み）/ 辺際予測（RE除外）
    allow_new_levels: bool = True,         # 未出被験者などを許可
    se_fit: bool = False,                  # LMMでは lme4::predict に se.fit は無い; OLSのみ対応
    interval: Optional[str] = None,        # OLS の "confidence" / "prediction" に対応
    level: float = 0.95
) -> pd.DataFrame:
    """
    Predicts outcomes using a fitted R-side model (LMM or OLS) from Python.

    For LMM (Linear Mixed Model): Calls lme4::predict(..., re.form=...) in R. Note that standard errors are not returned by default; use merTools::predictInterval if needed.
    For OLS (Ordinary Least Squares): Calls stats::predict(..., se.fit=, interval=) in R, which can return standard errors and confidence/prediction intervals.

    Args:
        fitted (Dict[str, Any]): The fitted model object returned by fit_lmm_r, containing R model handle and preprocessing info.
        newdata (pd.DataFrame): New data for prediction. Preprocessing (z-scoring, factorization, releveling) will be applied as in training.
        include_random_effects (bool): If True, include random effects in prediction (conditional prediction). If False, exclude them (marginal prediction).
        allow_new_levels (bool): If True, allow new levels in random effects (e.g., unseen subjects). Only relevant for LMM.
        se_fit (bool): If True, return standard errors of fit (only supported for OLS).
        interval (Optional[str]): If set to "confidence" or "prediction", return confidence or prediction intervals (only supported for OLS).
        level (float): Confidence level for intervals (default 0.95).

    Returns:
        pd.DataFrame: DataFrame with predictions (.pred for LMM, .fit/.lwr/.upr/.se.fit for OLS) and any additional columns from newdata.
    """


    # Required meta information
    model_name = fitted.get("r_model_name")
    if not model_name:
        raise ValueError("'r_model_name' is missing in results. Please save the model handle name during fitting.")

    pre = fitted.get("preproc", {})
    z_cols = pre.get("zscore_cols", [])
    z_by   = pre.get("zscore_by", [])
    z_inpl = pre.get("zscore_inplace", True)
    z_ddof = pre.get("zscore_ddof", 0)
    ensure_factors = pre.get("ensure_factor_cols", [])
    ref_levels = pre.get("ref_levels", {})

    # 1) Apply the same preprocessing to newdata (z-score, factorize, relevel)
    df_new = newdata.copy()
    if z_cols:
        df_new = LMMHelper._apply_zscore(df_new, z_cols, by=z_by or None, inplace=z_inpl, ddof=z_ddof)
    # Factorize categorical columns
    for col in ensure_factors:
        if col in df_new.columns:
            df_new[col] = pd.Categorical(df_new[col])
    # Set reference levels for categorical columns
    for col, ref in ref_levels.items():
        if col in df_new.columns:
            df_new[col] = pd.Categorical(df_new[col])
            # If the reference level does not exist, add it (safe for prediction only)
            if ref not in df_new[col].cat.categories:
                df_new[col] = df_new[col].cat.add_categories([ref])
            # Only change order in pandas; final relevel is done on R side
    pandas2ri.activate()

    # 2) Pass newdata to R
    nd_name = f"newdata_{uuid.uuid4().hex[:8]}"
    ro.globalenv[nd_name] = df_new

    # 3) Ensure relevel is also applied on R side
    for col, ref in ref_levels.items():
        if col in df_new.columns:
            ro.r(f'{nd_name}${col} <- base::factor({nd_name}${col})')
            ro.r(f'{nd_name}${col} <- stats::relevel({nd_name}${col}, ref = "{ref}")')

    # 4) Prediction
    (fitted.get("backend") or "").lower()
    is_lmm = (fitted.get("model_type") == "LMM")

    if is_lmm:
        # Note: lme4::predict does not return se.fit (use merTools::predictInterval if needed)
        re_form = "NULL" if include_random_effects else "~0"
        allow = "TRUE" if allow_new_levels else "FALSE"
        ro.r(f"""
            preds <- stats::predict({model_name},
                                    newdata={nd_name},
                                    re.form={re_form},
                                    allow.new.levels={allow})
        """)
        # import numpy as np  # re-import numpy
        pred = np.asarray(ro.r("as.numeric(preds)"), dtype=float).reshape(-1)
        out = df_new.copy()
        out[".pred"] = pred
        # Example for confidence intervals (LMM): use merTools optionally
        # Uncomment if needed
        # if interval is not None:
        #     importr("merTools")
        #     ro.r(f"""
        #         pi <- merTools::predictInterval({model_name}, newdata={nd_name},
        #                                        level={level}, n.sims=1000,
        #                                        include.resid.var=TRUE, which="full")
        #     """)
        #     pi_df = pandas2ri.rpy2py(ro.r("as.data.frame(pi)"))  # fit, lwr, upr
        #     out[".lwr"] = pi_df["lwr"]
        #     out[".upr"] = pi_df["upr"]
        return out

    else:
        # OLS: stats::predict can return SE and intervals
        interval_arg = f'"{interval}"' if interval in ("confidence", "prediction") else "NULL"
        se_flag = "TRUE" if se_fit or interval else "FALSE"
        lvl = level
        ro.r(f"""
            pr <- stats::predict({model_name}, newdata={nd_name},
                                 se.fit={se_flag}, interval={interval_arg},
                                 level={lvl})
        """)
        # The result of predict is a matrix/list -> convert to data.frame
        ro.r("pr_df <- as.data.frame(pr)")
        pr_df = pandas2ri.rpy2py(ro.r("pr_df"))
        out = df_new.copy()
        # Columns include 'fit', 'lwr', 'upr', 'se.fit', etc.
        for col in pr_df.columns:
            out[f".{col}"] = pr_df[col].values
        return out

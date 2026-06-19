import re
from typing import List, Optional, Dict, Any, Union, Tuple

import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import warnings
import xarray as xr
from scipy.special import expit

from pandas.api.types import is_categorical_dtype


class BayesianLMMHelper:
    """Utilities for Bayesian LMM (Bambi/PyMC): data prep, fitting, predictions."""

    @staticmethod
    def _prep_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """Cast key columns to categorical and drop unused levels without changing existing order."""
        df = df.copy()
        for col in ["roi", "dataset", "subject", "stimulus"]:
            if col not in df.columns:
                continue
            if is_categorical_dtype(df[col]):
                df[col] = df[col].cat.remove_unused_categories()
            else:
                df[col] = pd.Categorical(df[col])
        return df

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        mean_val = float(s.mean()) if hasattr(s, "mean") else 0.0
        std_val = float(s.std(ddof=0)) if hasattr(s, "std") else 1.0
        if std_val == 0:
            std_val = 1.0
        return (s - mean_val) / std_val

    @classmethod
    def prepare_bayes_dataframe(
        cls,
        data: pd.DataFrame,
        *,
        response_col: str,
        continuous_cols: Optional[List[str]] = None,
        alias_stimulus_from_dataset: bool = True,
        roi_ref: Optional[str] = None,
        dataset_ref: Optional[str] = None,
    ) -> pd.DataFrame:
        """Prepare dataframe: categoricals, optional stimulus aliasing, optional baseline releveling, z-scoring."""
        df = cls._prep_categoricals(data)

        if alias_stimulus_from_dataset and ("stimulus" not in df.columns) and ("dataset" in df.columns):
            if is_categorical_dtype(df["dataset"]):
                df["stimulus"] = df["dataset"].copy().cat.remove_unused_categories()
            else:
                df["stimulus"] = pd.Categorical(df["dataset"])
                if is_categorical_dtype(df["stimulus"]):
                    df["stimulus"] = df["stimulus"].cat.remove_unused_categories()

        if ("roi" in df.columns) and is_categorical_dtype(df["roi"]) and (roi_ref is not None):
            cats = list(df["roi"].cat.categories)
            if roi_ref in cats:
                new_cats = [roi_ref] + [c for c in cats if c != roi_ref]
                df["roi"] = df["roi"].cat.reorder_categories(new_cats, ordered=True)

        if ("dataset" in df.columns) and is_categorical_dtype(df["dataset"]) and (dataset_ref is not None):
            cats = list(df["dataset"].cat.categories)
            if dataset_ref in cats:
                new_cats = [dataset_ref] + [c for c in cats if c != dataset_ref]
                df["dataset"] = df["dataset"].cat.reorder_categories(new_cats, ordered=True)

        if continuous_cols is not None:
            for c in continuous_cols:
                if c in df.columns and isinstance(df[c], pd.Series):
                    df[f"{c}_z"] = cls._zscore(df[c])

        if response_col not in df.columns:
            raise ValueError(f"response_col '{response_col}' not found in dataframe.")
        if "subject" not in df.columns:
            raise ValueError("column 'subject' is required for grouping.")

        return df

    @classmethod
    def fit_lmm_bayesian(
        cls,
        data: pd.DataFrame,
        formulas: List[str],
        *,
        response_col: str = "error",
        roi_ref: str = "WholeVC",
        dataset_ref: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        alias_stimulus_from_dataset: bool = True,
        draws: int = 2000,
        tune: int = 2000,
        target_accept: float = 0.9,
        max_treedepth: int = 10,
        random_seed: int = 42,
        chains: Optional[int] = None,
        cores: Optional[int] = None,
        store_ll: bool = True,
        store_loo: bool = False,
        return_waic: bool = True,
        family: Union[str, "bmb.Family"] = "gaussian",
        priors: Optional[Dict[str, "bmb.Prior"]] = None,
        use_bambi_default_priors: bool = False,
        beta_auto_squeeze: bool = True,
        beta_epsilon: float = 1e-6,
    ):
        """Fit one or more Bayesian LMMs and optionally compute WAIC/LOO."""
        df = cls.prepare_bayes_dataframe(
            data,
            response_col=response_col,
            continuous_cols=continuous_cols,
            alias_stimulus_from_dataset=alias_stimulus_from_dataset,
            roi_ref=roi_ref,
            dataset_ref=dataset_ref,
        ).copy()

        fam_name: str = family.lower() if isinstance(family, str) else getattr(family, "name", "custom").lower()

        if fam_name == "beta":
            if (df[response_col] <= 0).any() or (df[response_col] >= 1).any():
                if not beta_auto_squeeze:
                    raise ValueError(
                        f"Beta family requires {response_col} in (0,1). Found 0/1 and beta_auto_squeeze=False."
                    )
                df[response_col] = (df[response_col] * (1 - 2 * beta_epsilon) + beta_epsilon)

        if fam_name == "bernoulli":
            bad = ~df[response_col].isin([0, 1])
            if bad.any():
                raise ValueError(f"Bernoulli expects 0/1 in {response_col}. Bad rows: {int(bad.sum())}")

        if priors is None and not use_bambi_default_priors:
            if fam_name == "gaussian":
                priors = {
                    "Common": bmb.Prior("Normal", mu=0, sigma=2.5),
                    "Intercept": bmb.Prior("Normal", mu=0, sigma=5),
                    "Sigma": bmb.Prior("HalfCauchy", beta=2.5),
                    "GroupSpecific": bmb.Prior("HalfCauchy", beta=2.5),
                }
            elif fam_name == "beta":
                priors = {
                    "Common": bmb.Prior("Normal", mu=0, sigma=2.5),
                    "Intercept": bmb.Prior("Normal", mu=0, sigma=5),
                    "kappa": bmb.Prior("Gamma", alpha=2, beta=0.1),
                    "GroupSpecific": bmb.Prior("HalfCauchy", beta=2.5),
                }
            elif fam_name == "bernoulli":
                priors = {
                    "Common": bmb.Prior("Normal", mu=0, sigma=2.5),
                    "Intercept": bmb.Prior("Normal", mu=0, sigma=5),
                    "GroupSpecific": bmb.Prior("HalfCauchy", beta=2.5),
                }
            else:
                priors = {
                    "Common": bmb.Prior("Normal", mu=0, sigma=2.5),
                    "Intercept": bmb.Prior("Normal", mu=0, sigma=5),
                    "GroupSpecific": bmb.Prior("HalfCauchy", beta=2.5),
                }

        idata_kwargs = {"log_likelihood": bool(store_ll)}
        models: Dict[str, bmb.Model] = {}
        idatas: Dict[str, az.InferenceData] = {}

        for name, fml in zip(formulas, formulas):
            model_kwargs = {"family": family}
            if priors is not None:
                model_kwargs["priors"] = priors
            model = bmb.Model(fml, data=df, **model_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_kwargs = {
                    "draws": draws,
                    "tune": tune,
                    "target_accept": target_accept,
                    "random_seed": random_seed,
                    "idata_kwargs": idata_kwargs,
                    "nuts": {"max_treedepth": max_treedepth},
                }
                if chains is not None:
                    fit_kwargs["chains"] = chains
                if cores is not None:
                    fit_kwargs["cores"] = cores
                idata = model.fit(
                    **fit_kwargs,
                )
            models[name] = model
            idatas[name] = idata

        result: Dict[str, Any] = {"idata": idatas, "models": models}

        if store_ll:
            try:
                _ = az.extract(next(iter(idatas.values())), group="log_likelihood")[response_col]
            except Exception as e:
                raise RuntimeError(
                    f"log_likelihood('{response_col}') not found in InferenceData. "
                    "Check idata_kwargs={'log_likelihood': True} and response name."
                ) from e

        n_models = len(idatas)

        if store_ll and store_loo:
            if n_models > 1:
                result["loo_table"] = az.compare(idatas, ic="loo", scale="deviance", var_name=response_col)
            else:
                only_name, only_idata = next(iter(idatas.items()))
                result["loo"] = {only_name: az.loo(only_idata, var_name=response_col)}

        if return_waic:
            if n_models > 1:
                result["waic_table"] = az.compare(idatas, ic="waic", scale="deviance", var_name=response_col)
            else:
                only_name, only_idata = next(iter(idatas.items()))
                result["waic"] = {only_name: az.waic(only_idata, var_name=response_col)}

        if store_loo and ("loo_table" in result):
            print("\n=== LOO comparison (deviance; lower is better) ===")
            print(result["loo_table"].round(3))
        if return_waic and ("waic_table" in result):
            print("\n=== WAIC comparison (deviance; lower is better) ===")
            print(result["waic_table"].round(3))

        return result

    @staticmethod
    def predict_mu_draws_robust(
        model: "bmb.Model",
        idata: "az.InferenceData",
        new_df: pd.DataFrame,
        *,
        include_random_effects: bool = True,
    ) -> tuple:
        base = dict(idata=idata, data=new_df, kind="mean", inplace=False)
        try_kw = []
        if include_random_effects:
            try_kw.append({**base, "include_group_specific": True})
            try_kw.append(base.copy())
        else:
            try_kw.append({**base, "include_group_specific": False})
            try_kw.append(base.copy())

        da = None
        last_exc = None
        for kw in try_kw:
            try:
                xr_pred = model.predict(**kw)
                da = BayesianLMMHelper._to_dataarray(xr_pred)
                if da is not None:
                    break
            except Exception as e:
                last_exc = e

        if da is None:
            try:
                xr_pred = model.predict(idata=idata, data=new_df, kind="linpred", inplace=False)
                da = BayesianLMMHelper._to_dataarray(xr_pred)
            except Exception as e:
                last_exc = e

        if da is None:
            raise RuntimeError("Could not interpret predict() output. Check category levels / Bambi version.") from last_exc

        n_obs_expected = new_df.shape[0]
        candidates = [d for d in ["observation", "__obs__", "obs", "rows"] if d in da.dims]
        obs_dim = None
        for d in candidates:
            if da.sizes.get(d, -1) == n_obs_expected:
                obs_dim = d
                break
        if obs_dim is None:
            obs_dim = "observation" if "observation" in da.dims else da.dims[-1]

        sample_dims = [d for d in da.dims if d != obs_dim]
        if len(sample_dims) == 0:
            da = da.expand_dims({"sample": [0]})
        elif set(sample_dims) == {"chain", "draw"}:
            da = da.stack(sample=("chain", "draw"))
        else:
            da = da.stack(sample=tuple(sample_dims))

        if obs_dim != "observation":
            da = da.rename({obs_dim: "observation"})
        da = da.transpose("sample", "observation")

        draws = da.values
        obs_index = (
            da.coords["observation"].to_pandas().to_numpy()
            if "observation" in da.coords
            else np.arange(draws.shape[1], dtype=int)
        )
        return draws, obs_index

    @staticmethod
    def predict_from_idata_only(
        idata: "az.InferenceData",
        new_df: pd.DataFrame,
        formula: str,
        *,
        include_random_effects: bool = True,
    ) -> tuple:
        if not hasattr(idata, "posterior"):
            raise ValueError("InferenceData object does not have posterior samples")
        posterior = idata.posterior
        n_draws = posterior.sizes["draw"]
        n_chains = posterior.sizes["chain"]
        total_samples = n_draws * n_chains
        posterior_stacked = posterior.stack(sample=("chain", "draw"))

        n_obs = len(new_df)
        predictions = np.zeros((total_samples, n_obs))

        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(f"Cannot parse formula: {formula}")
        predictors_part = parts[1].strip()

        fixed_part = re.sub(r"\([^)]+\|[^)]+\)", "", predictors_part)
        fixed_terms = [t.strip() for t in fixed_part.split("+") if t.strip()]

        if "Intercept" in posterior_stacked.data_vars:
            intercept = posterior_stacked["Intercept"].values
            predictions += intercept[:, np.newaxis]

        for term in fixed_terms:
            if term == "1":
                continue
            if term in posterior_stacked.data_vars and term in new_df.columns:
                coef = posterior_stacked[term].values
                x_values = new_df[term].values
                predictions += coef[:, np.newaxis] * x_values[np.newaxis, :]

        if include_random_effects:
            random_matches = re.findall(r"\(([^)]+)\|([^)]+)\)", predictors_part)
            for random_term, group_var in random_matches:
                random_term = random_term.strip()
                group_var = group_var.strip()
                if group_var not in new_df.columns:
                    continue

                groups = new_df[group_var].values
                unique_groups = np.unique(np.asarray(groups))

                ri_name = f"1|{group_var}"
                if ri_name in posterior_stacked.data_vars:
                    ri_data = posterior_stacked[ri_name]
                    for group in unique_groups:
                        try:
                            group_coef = ri_data.sel(subject__factor_dim=group).values
                            mask = groups == group
                            predictions[:, mask] += group_coef[:, np.newaxis]
                        except Exception:
                            continue

                if random_term != "1":
                    rs_name = f"{random_term}|{group_var}"
                    if rs_name in posterior_stacked.data_vars and random_term in new_df.columns:
                        rs_data = posterior_stacked[rs_name]
                        for group in unique_groups:
                            try:
                                group_coef = rs_data.sel(subject__factor_dim=group).values
                                mask = groups == group
                                x_values = new_df.loc[mask, random_term].values
                                predictions[:, mask] += group_coef[:, np.newaxis] * x_values[np.newaxis, :]
                            except Exception:
                                continue

        obs_index = np.arange(n_obs, dtype=int)
        return predictions, obs_index

    @staticmethod
    def _to_dataarray(xr_pred):
        if xr_pred is None:
            return None
        if isinstance(xr_pred, xr.DataArray):
            return xr_pred
        if isinstance(xr_pred, xr.Dataset):
            varname = list(xr_pred.data_vars)[0]
            return xr_pred[varname]
        if hasattr(xr_pred, "posterior"):
            ds = xr_pred.posterior
            for cand in ["mean", "linpred", "mu", "fitted", "y_hat"]:
                if cand in ds.data_vars:
                    return ds[cand]
            if len(ds.data_vars):
                return ds[list(ds.data_vars)[0]]
        return None

    @staticmethod
    def _get_posterior_draws(idata, name_candidates):
        """Return posterior draws (chain*draw,) for the first existing var name."""
        post = idata.posterior
        for nm in name_candidates:
            if nm in post.data_vars:
                return post[nm].values.reshape(-1)
        raise KeyError(f"None of these posterior vars exist: {name_candidates}")

    @staticmethod
    def _draws_1d(idata, var, level=None):
        """Return posterior draws (chain*draw,) for scalar or one/two-indexed parameters."""
        da = idata.posterior[var]

        if level is None:
            return da.values.reshape(-1)

        extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
        if len(extra_dims) == 0:
            raise ValueError(f"{var} has no level dims, but level={level} was provided.")

        if len(extra_dims) == 1:
            dim = extra_dims[0]
            coord_vals = list(da.coords[dim].values) if dim in da.coords else None

            if isinstance(level, str):
                if coord_vals is not None and level in coord_vals:
                    da_sel = da.sel({dim: level})
                else:
                    raise KeyError(f"{var}: level '{level}' not found. Available: {list(da.coords.get(dim, []))}")
            elif isinstance(level, (tuple, list)) and len(level) == 2:
                # Common case in Bambi: interaction terms encoded in one coord like "A, B"
                if coord_vals is None:
                    raise KeyError(f"{var}: tuple level {level} provided but dim '{dim}' has no coordinates.")

                a, b = str(level[0]), str(level[1])
                candidates = [
                    f"{a}, {b}",
                    f"{a},{b}",
                    f"{a}:{b}",
                    f"{a}*{b}",
                    str((a, b)),
                ]

                matched = next((c for c in candidates if c in coord_vals), None)
                if matched is None:
                    # Final fallback: normalized string match (ignores spaces)
                    norm_target = f"{a},{b}".replace(" ", "")
                    for c in coord_vals:
                        if str(c).replace(" ", "") == norm_target:
                            matched = c
                            break

                if matched is None:
                    raise KeyError(
                        f"{var}: tuple level {level} not found in coords for dim '{dim}'. "
                        f"Examples: {coord_vals[:10]}"
                    )

                da_sel = da.sel({dim: matched})
            else:
                da_sel = da.isel({dim: int(level)})
            return da_sel.values.reshape(-1)

        if len(extra_dims) == 2:
            dim1, dim2 = extra_dims
            l1, l2 = level
            da_sel = da
            da_sel = da_sel.sel({dim1: l1}) if isinstance(l1, str) else da_sel.isel({dim1: int(l1)})
            da_sel = da_sel.sel({dim2: l2}) if isinstance(l2, str) else da_sel.isel({dim2: int(l2)})
            return da_sel.values.reshape(-1)

        raise ValueError(f"{var} has >2 level dims: {extra_dims}")

    @staticmethod
    def simple_effect(
        idata,
        factor1: str,
        level1: str,
        level2: str,
        cond_factor: str,
        cond_level: str,
        baseline_factor1: str,
        baseline_cond: str,
        prob: float = 0.95,
        sided: str = "two-sided",
    ):
        """Compute fixed-only simple effect: (factor1[level1]-factor1[level2]) | cond_factor=cond_level."""
        b0 = BayesianLMMHelper._draws_1d(idata, "Intercept", None)

        def b_main(factor, level, baseline):
            if level == baseline:
                return np.zeros_like(b0)
            return BayesianLMMHelper._draws_1d(idata, factor, level)

        def b_int(level_f1, level_cond):
            if (level_f1 == baseline_factor1) or (level_cond == baseline_cond):
                return np.zeros_like(b0)

            # Support either interaction naming order: factor1:cond_factor or cond_factor:factor1
            post_vars = set(idata.posterior.data_vars)
            cand_direct = f"{factor1}:{cond_factor}"
            cand_rev = f"{cond_factor}:{factor1}"
            if cand_direct in post_vars:
                return BayesianLMMHelper._draws_1d(idata, cand_direct, (level_f1, level_cond))
            if cand_rev in post_vars:
                return BayesianLMMHelper._draws_1d(idata, cand_rev, (level_cond, level_f1))

            # If interaction term is absent in posterior, treat as zero contribution.
            return np.zeros_like(b0)

        mu1 = b0 + b_main(factor1, level1, baseline_factor1) + b_main(cond_factor, cond_level, baseline_cond) + b_int(level1, cond_level)
        mu2 = b0 + b_main(factor1, level2, baseline_factor1) + b_main(cond_factor, cond_level, baseline_cond) + b_int(level2, cond_level)

        diff = mu1 - mu2
        lo, hi = BayesianLMMHelper._interval_bounds_1d(diff, prob=prob, sided=sided)

        return {
            "beta": float(np.median(diff)),
            "HDI_lower": lo,
            "HDI_upper": hi,
            "prob_greater": float((diff > 0).mean()),
            "prob_less": float((diff < 0).mean()),
        }

    @staticmethod
    def _interval_bounds_1d(x: np.ndarray, prob: float, sided: str = "two-sided") -> Tuple[float, float]:
        if not (0.0 < float(prob) < 1.0):
            raise ValueError(f"prob must be in (0,1), got {prob}")

        side = str(sided).strip().lower().replace("_", "-")
        alpha = 1.0 - float(prob)

        if side in ("two-sided", "two sided", "two", "2"):
            hdi = az.hdi(np.asarray(x, dtype=float), hdi_prob=float(prob))
            return float(hdi[0]), float(hdi[1])

        if side in ("one-sided", "one sided", "one", "1"):
            # Return central equal-tailed interval corresponding to one-sided alpha.
            lo = float(np.quantile(x, alpha))
            hi = float(np.quantile(x, 1.0 - alpha))
            return lo, hi

        raise ValueError(f"sided must be 'two-sided' or 'one-sided', got {sided!r}")

class BayesianLMMSummary:
    """Summaries for fixed effects, random effects, and EMMs."""

    @staticmethod
    def _hdi_1d(x: np.ndarray, prob: float, sided: str = "two-sided") -> Tuple[float, float]:
        return BayesianLMMHelper._interval_bounds_1d(x, prob=prob, sided=sided)

    @staticmethod
    def fixed_effects(idata, prob: float = 0.95, sided: str = "two-sided") -> pd.DataFrame:
        """
        Robust fixed-effects summary for Bambi/PyMC InferenceData.

        Handles common Bambi layouts:
        - posterior['beta'] with a term dimension (vector of coefficients)
        - separate scalar variables like 'Intercept'
        Fallback: summarize any scalar (chain,draw) variables that look like fixed effects,
                  but avoids mixing in sigma/sd/chol/etc.
        """
        posterior = idata.posterior
        rows = []

        def add_row(term: str, draws_1d: np.ndarray):
            draws_1d = np.asarray(draws_1d, dtype=float).reshape(-1)
            lo, hi = BayesianLMMSummary._hdi_1d(draws_1d, prob, sided=sided)
            mean_val = float(np.mean(draws_1d))
            rows.append({
                "term": term,
                "mean": mean_val,
                "beta": mean_val,
                "median": float(np.median(draws_1d)),
                "HDI_lower": lo,
                "HDI_upper": hi,
                "significant": (lo > 0) or (hi < 0),
            })

        # 1) Preferred: vectorized coefficients (common in Bambi)
        # Typical var names: 'beta', sometimes 'b'
        for coef_var in ["beta", "b"]:
            if coef_var in posterior.data_vars:
                da = posterior[coef_var]
                # find the coefficient/term dimension (not chain/draw)
                term_dims = [d for d in da.dims if d not in ("chain", "draw")]
                if len(term_dims) == 1:
                    term_dim = term_dims[0]
                    term_coords = list(da.coords[term_dim].values) if term_dim in da.coords else list(range(da.sizes[term_dim]))
                    for t in term_coords:
                        try:
                            draws = da.sel({term_dim: t}).values
                            add_row(str(t), draws)
                        except Exception:
                            # if coord selection fails, fallback to isel by index
                            idx = term_coords.index(t)
                            draws = da.isel({term_dim: idx}).values
                            add_row(str(t), draws)
                    # If beta/b exists, we consider fixed effects done (Intercept often included there)
                    if len(rows) > 0:
                        return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)

        # 2) Scalar Intercept
        for cand in ["Intercept", "b_Intercept", "beta_Intercept", "const"]:
            if cand in posterior.data_vars and posterior[cand].ndim == 2:
                add_row("Intercept", posterior[cand].values)
                break

        # Shared exclusion rules for paths 3 and 4
        blacklist_prefix = ("sd_", "sigma", "chol", "L_", "kappa", "lp__", "r_", "z_", "eps", "nu")
        blacklist_contains = ("|", "subject", "GroupSpecific", "Var", "cor", "Corr")

        def _is_random(var: str) -> bool:
            vlow = var.lower()
            return vlow.startswith(blacklist_prefix) or any(s in vlow for s in blacklist_contains)

        # 3) Vector fixed effects: shape (chain, draw, n_levels) — main effects & interactions
        #    Bambi stores these as separate named variables with a coordinate dimension.
        seen = {r["term"] for r in rows}
        for var in sorted(posterior.data_vars):
            if var in ("beta", "b") or var in seen:
                continue
            da = posterior[var]
            extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
            if len(extra_dims) != 1:
                continue  # scalar handled in path 2 / step 4; >1 extra dim is random
            if _is_random(var):
                continue
            term_dim = extra_dims[0]
            coord_vals = (
                list(da.coords[term_dim].values)
                if term_dim in da.coords
                else list(range(da.sizes[term_dim]))
            )
            for cv in coord_vals:
                try:
                    draws = da.sel({term_dim: cv}).values.reshape(-1)
                except Exception:
                    draws = da.isel({term_dim: coord_vals.index(cv)}).values.reshape(-1)
                label = f"{var}[{cv}]" if len(coord_vals) > 1 else var
                if label not in seen:
                    add_row(label, draws)
                    seen.add(label)

        # 4) Remaining scalar (chain, draw) fixed effects not yet captured
        for var in sorted(posterior.data_vars):
            if var in ("beta", "b") or var in seen:
                continue
            da = posterior[var]
            if da.ndim != 2:
                continue
            if _is_random(var):
                continue
            add_row(var, da.values)
            seen.add(var)

        return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)

    @staticmethod
    def random_effects(idata, df: pd.DataFrame, prob: float = 0.95, sided: str = "two-sided") -> pd.DataFrame:
        posterior = idata.posterior
        subject_levels = pd.Index(df["subject"].unique()).sort_values()
        n_subject = len(subject_levels)

        if "subject" in posterior.coords and len(posterior.coords["subject"]) == n_subject:
            subject_dim = "subject"
        else:
            candidate_dims = []
            for v in posterior.data_vars:
                for d in posterior[v].dims:
                    if d in ("chain", "draw"):
                        continue
                    if posterior[v].sizes[d] == n_subject:
                        candidate_dims.append(d)
            candidate_dims = list(dict.fromkeys(candidate_dims))
            subject_dim = candidate_dims[0]

        subject_coord = posterior.coords[subject_dim].values if subject_dim in posterior.coords else np.arange(posterior.sizes[subject_dim])

        random_vars = []
        for var in posterior.data_vars:
            da = posterior[var]
            if subject_dim in da.dims and not set(da.dims).issubset({"chain", "draw"}):
                random_vars.append(var)

        rows = []
        for var in random_vars:
            da = posterior[var]
            non_cd_dims = [d for d in da.dims if d not in ("chain", "draw")]

            if subject_dim not in non_cd_dims:
                continue

            shape = [da.sizes[d] for d in non_cd_dims]

            for idx_tuple in np.ndindex(*shape):
                indexer = {dim: i for dim, i in zip(non_cd_dims, idx_tuple)}
                sub_idx = indexer[subject_dim]
                subj_label = subject_coord[sub_idx]

                term_bits = []
                for dim, i in zip(non_cd_dims, idx_tuple):
                    if dim == subject_dim:
                        continue
                    coord_val = da.coords[dim].values[i] if dim in da.coords else i
                    term_bits.append(f"{dim}={coord_val}")

                term_name = f"{var}|{','.join(term_bits)}" if term_bits else f"{var}|Intercept"

                vals = da.isel(**indexer).values.flatten()
                lower, upper = BayesianLMMSummary._hdi_1d(vals, prob=prob, sided=sided)

                rows.append(
                    {
                        "group": "subject",
                        "level": str(subj_label),
                        "term": term_name,
                        "mean": float(np.mean(vals)),
                        "beta": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "HDI_lower": float(lower),
                        "HDI_upper": float(upper),
                        "significant": (lower > 0) or (upper < 0),
                    }
                )

        df_re = pd.DataFrame(rows)
        return df_re.sort_values(["group", "level", "term"]).reset_index(drop=True)

    @staticmethod
    def emmeans(
        idata,
        df: pd.DataFrame,
        factor1: str,
        factor2: str,
        link: str = "identity",
        prob: float = 0.95,
        sided: str = "two-sided",
        return_draws: bool = False,
    ) -> pd.DataFrame:
        posterior = idata.posterior

        if hasattr(df[factor1].dtype, "categories"):
            levels1 = list(df[factor1].cat.categories)
        else:
            levels1 = sorted(pd.unique(df[factor1]))

        if hasattr(df[factor2].dtype, "categories"):
            levels2 = list(df[factor2].cat.categories)
        else:
            levels2 = sorted(pd.unique(df[factor2]))

        intercept = None
        for cand in ["Intercept", "b_Intercept", "beta_Intercept", "const"]:
            if cand in posterior.data_vars and posterior[cand].ndim == 2:
                intercept = posterior[cand].values
                break
        if intercept is None:
            raise ValueError("No intercept found in idata.posterior.")

        def find_main_effect(posterior, factor_levels):
            for var in posterior.data_vars:
                da = posterior[var]
                if not np.issubdtype(da.dtype, np.floating):
                    continue
                non_cd_dims = [d for d in da.dims if d not in ("chain", "draw")]
                if len(non_cd_dims) != 1:
                    continue
                dim = non_cd_dims[0]
                if dim not in da.coords:
                    continue
                coord_vals = list(da.coords[dim].values)
                if set(coord_vals).issubset(set(factor_levels)) and len(coord_vals) > 0:
                    return var, dim, coord_vals
            return None, None, []

        f1_var, f1_dim, f1_lv = find_main_effect(posterior, levels1)
        f1_vals = posterior[f1_var].transpose("chain", "draw", f1_dim).values if f1_var is not None else None

        f2_var, f2_dim, f2_lv = find_main_effect(posterior, levels2)
        f2_vals = posterior[f2_var].transpose("chain", "draw", f2_dim).values if f2_var is not None else None

        def find_interaction_effect(posterior, lv1, lv2):
            for var in posterior.data_vars:
                da = posterior[var]
                if not np.issubdtype(da.dtype, np.floating):
                    continue
                non_cd_dims = [d for d in da.dims if d not in ("chain", "draw")]
                if len(non_cd_dims) != 2:
                    continue
                d1, d2 = non_cd_dims
                if d1 not in da.coords or d2 not in da.coords:
                    continue
                coord1 = list(da.coords[d1].values)
                coord2 = list(da.coords[d2].values)

                if set(coord1).issubset(set(lv1)) and set(coord2).issubset(set(lv2)):
                    return var, d1, d2, coord1, coord2
                if set(coord1).issubset(set(lv2)) and set(coord2).issubset(set(lv1)):
                    return var, d2, d1, coord2, coord1
            return None, None, None, [], []

        inter_var, i_f1_dim, i_f2_dim, i_f1_lv, i_f2_lv = find_interaction_effect(posterior, levels1, levels2)
        inter_vals = (
            posterior[inter_var].transpose("chain", "draw", i_f1_dim, i_f2_dim).values
            if inter_var is not None
            else None
        )

        rows = []
        for lv1 in levels1:
            for lv2 in levels2:
                eta = intercept.copy()

                if f1_vals is not None and lv1 in f1_lv:
                    eta = eta + f1_vals[:, :, f1_lv.index(lv1)]

                if f2_vals is not None and lv2 in f2_lv:
                    eta = eta + f2_vals[:, :, f2_lv.index(lv2)]

                if inter_vals is not None and (lv1 in i_f1_lv) and (lv2 in i_f2_lv):
                    eta = eta + inter_vals[:, :, i_f1_lv.index(lv1), i_f2_lv.index(lv2)]

                eta_flat = eta.reshape(-1)

                if link == "logit":
                    resp = expit(eta_flat)
                elif link == "identity":
                    resp = eta_flat
                else:
                    raise ValueError("Unsupported link function.")

                lower, upper = BayesianLMMSummary._hdi_1d(resp, prob=prob, sided=sided)

                row = {
                    factor1: lv1,
                    factor2: lv2,
                    "mean": float(np.mean(resp)),
                    "beta": float(np.mean(resp)),
                    "median": float(np.median(resp)),
                    "HDI_lower": lower,
                    "HDI_upper": upper,
                    "significant": (lower > 0) or (upper < 0),
                }
                if return_draws:
                    row["draws"] = resp

                rows.append(row)

        df_emm = pd.DataFrame(rows)
        return df_emm.sort_values([factor1, factor2]).reset_index(drop=True)

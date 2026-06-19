"""Reproduce the contour-matched slant LMM analyses used in the paper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from recon3d.config import load_yaml_config
from recon3d.contour_slant import (
    HORIZONTAL_SHAPE_DATASET,
    HV_STIMULI,
    ROI_REF,
    STIMULI,
    THIN_TILT_DATASET,
    build_contour_slant_dataframe,
)
from recon3d.metadata import (
    CONTOUR_SLANT_ROIS,
    SUBROIS,
    WHOLE_VISUAL_ROI,
)
from recon3d.stats.bayesian_lmm import BayesianLMMHelper, BayesianLMMSummary
from recon3d.stats.emmeans import (
    estimate_emtrends,
    factor_levels_from_bayesian_model,
    pairwise_emtrends_contrasts,
    treatment_vs_control_emtrends,
)
from recon3d.stats.frequentist_lmm import LMMHelper
from recon3d.stats.io import write_csv, write_json, write_text
from recon3d.stats.posterior import hdi_interval, prob_direction
from recon3d.stats.slope_posterior import PosteriorSlopeExtractor
from recon3d.subjects import (
    PUBLIC_SUBJECTS,
    apply_bayesian_subject_aliases,
    restore_bayesian_subject_labels,
)


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parent
DEFAULT_CONFIG = ANALYSIS_DIR / "configs" / "atlasnet.yaml"
DEFAULT_OUTPUT_ROOT = ANALYSIS_DIR / "outputs" / "statistics" / "contour_matched_lmm"
X_VAR = "true_deg_z"
FORMULA_SLOPE_WHOLEVC = f"pred_deg_z ~ {X_VAR} + ({X_VAR} | subject)"
FORMULA_SLOPE_SUBROI = f"pred_deg_z ~ {X_VAR}*roi + ({X_VAR} | subject)"
FORMULA_HV_WHOLEVC = f"pred_deg_z ~ {X_VAR}*stimulus + ({X_VAR} | subject)"
FORMULA_HV_SUBROI = f"pred_deg_z ~ {X_VAR}*roi*stimulus + ({X_VAR} | subject)"
CONTRAST_OPTIONS = ["contr.sum", "contr.poly"]
PREDICTION_LINE_POINTS = 100
HORIZONTAL_VERTICAL_SUBJECTS = list(PUBLIC_SUBJECTS)
CI_LEVEL = 0.95


@dataclass(frozen=True)
class ContourSlantLMMContext:
    """Resolved config values shared by the two LMM analyses."""

    representation_name: str
    reconstruction_name: str
    data_root: Path
    output_dir: Path
    horizontal_vertical_subjects: list[str]
    include_largest_slant: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ContourSlantLMMContext":
        config = load_yaml_config(args.config)
        representation_name = config["representation"]["name"]
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
        else:
            output_name = (
                f"{representation_name}_include_largest_slant"
                if args.include_largest_slant
                else representation_name
            )
            output_dir = DEFAULT_OUTPUT_ROOT / output_name
        return cls(
            representation_name=representation_name,
            reconstruction_name=config["representation"]["reconstruction_name"],
            data_root=Path(args.data_root),
            output_dir=output_dir,
            horizontal_vertical_subjects=HORIZONTAL_VERTICAL_SUBJECTS,
            include_largest_slant=args.include_largest_slant,
        )

    def slope_dataframe(self) -> pd.DataFrame:
        return build_contour_slant_dataframe(
            data_root=self.data_root,
            model_name=self.reconstruction_name,
            subjects=PUBLIC_SUBJECTS,
            rois=CONTOUR_SLANT_ROIS,
            stimuli=STIMULI,
            use_thin_tilt_horizontal=False,
            include_largest_slant=self.include_largest_slant,
        )

    def horizontal_vertical_dataframe(self) -> pd.DataFrame:
        return build_contour_slant_dataframe(
            data_root=self.data_root,
            model_name=self.reconstruction_name,
            subjects=self.horizontal_vertical_subjects,
            rois=CONTOUR_SLANT_ROIS,
            stimuli=HV_STIMULI,
            use_thin_tilt_horizontal=True,
            include_largest_slant=self.include_largest_slant,
        )


@dataclass
class ContourSlantLMMAnalysis:
    context: ContourSlantLMMContext
    args: argparse.Namespace

    def run(self) -> None:
        self.context.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_metadata()

        if self.args.analysis in {"all", "slope"}:
            self.run_slope()
        if self.args.analysis in {"all", "horizontal-vertical"}:
            self.run_horizontal_vertical()

    def run_slope(self) -> None:
        # Build the shared analysis table once, then pass the same rows to the
        # requested model backends.
        slope_df = self.context.slope_dataframe()
        write_csv(slope_df, self.context.output_dir / "lmm_input_slope_by_stimulus.csv")
        if self.args.backend in {"all", "frequentist"}:
            run_slope_frequentist(slope_df, self.context.output_dir / "frequentist")
        if self.args.backend in {"all", "bayesian"}:
            run_slope_bayesian(slope_df, self.context.output_dir / "bayesian", self.args)

    def run_horizontal_vertical(self) -> None:
        # The horizontal-vs-vertical comparison uses a separate input table
        # because the horizontal stimulus comes from the thin-tilt set.
        hv_df = self.context.horizontal_vertical_dataframe()
        write_csv(hv_df, self.context.output_dir / "lmm_input_horizontal_vs_vertical.csv")
        if self.args.backend in {"all", "frequentist"}:
            run_horizontal_vertical_frequentist(hv_df, self.context.output_dir / "frequentist")
        if self.args.backend in {"all", "bayesian"}:
            run_horizontal_vertical_bayesian(hv_df, self.context.output_dir / "bayesian", self.args)

    def write_metadata(self) -> None:
        write_json(
            {
                "representation": self.context.representation_name,
                "model_name": self.context.reconstruction_name,
                "subjects": PUBLIC_SUBJECTS,
                "subjects_horizontal_vertical": self.context.horizontal_vertical_subjects,
                "rois": CONTOUR_SLANT_ROIS,
                "sub_rois": SUBROIS,
                "wholevc": WHOLE_VISUAL_ROI,
                "stimuli": STIMULI,
                "horizontal_vertical_stimuli": HV_STIMULI,
                "include_largest_slant": self.context.include_largest_slant,
                "datasets": {
                    "horizontal_shape": HORIZONTAL_SHAPE_DATASET,
                    "thin_tilt": THIN_TILT_DATASET,
                },
                "reference_levels": {
                    "roi": ROI_REF,
                    "stimulus": HV_STIMULI[0],
                },
                "formulas": {
                    "slope_wholevc": FORMULA_SLOPE_WHOLEVC,
                    "slope_subroi": FORMULA_SLOPE_SUBROI,
                    "horizontal_vertical_wholevc": FORMULA_HV_WHOLEVC,
                    "horizontal_vertical_subroi": FORMULA_HV_SUBROI,
                },
                "test_directions": {
                    "slope_vs_zero": "one-sided greater",
                    "earlyvc_slope_contrasts": "two-sided",
                    "horizontal_vertical_slope_contrasts": "one-sided greater",
                },
            },
            self.context.output_dir / "metadata.json",
        )


def _fit_frequentist_model(df: pd.DataFrame, formula: str, model_name: str, ref_levels: dict[str, str] | None):
    return LMMHelper.fit_lmm_r(
        df,
        formula,
        backend="lmerTest",
        df_method="Kenward-Roger",
        zscore_cols=["true_deg", "pred_deg"],
        zscore_inplace=False,
        ensure_factor_cols=["subject", "roi", "stimulus"],
        ref_levels=ref_levels,
        contrast_options=CONTRAST_OPTIONS,
        model_name=model_name,
    )


def _fit_bayesian_model(
    df: pd.DataFrame,
    formula: str,
    *,
    roi_ref: str | None,
    args: argparse.Namespace,
):
    model_df = df.copy()
    alias_to_subject = apply_bayesian_subject_aliases(model_df)
    for factor in ("subject", "stimulus"):
        if factor in model_df.columns:
            model_df[factor] = model_df[factor].astype(str)
    result = BayesianLMMHelper.fit_lmm_bayesian(
        model_df,
        formulas=[formula],
        response_col="pred_deg_z",
        continuous_cols=["true_deg", "pred_deg"],
        roi_ref=roi_ref,
        alias_stimulus_from_dataset=False,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        random_seed=args.random_seed,
        chains=args.chains,
        cores=args.cores,
        store_ll=False,
        return_waic=False,
        use_bambi_default_priors=True,
    )
    restore_bayesian_subject_labels(result, alias_to_subject)
    return result


def _norm_stats(data_used: pd.DataFrame) -> dict[str, float]:
    return {
        "mu_x": float(data_used["true_deg"].mean()),
        "sd_x": float(data_used["true_deg"].std(ddof=0)),
        "mu_y": float(data_used["pred_deg"].mean()),
        "sd_y": float(data_used["pred_deg"].std(ddof=0)),
    }


def _denorm_slope(value_z: float, data_used: pd.DataFrame) -> float:
    stats = _norm_stats(data_used)
    return value_z * stats["sd_y"] / stats["sd_x"]


def _posterior_summary(
    samples: np.ndarray,
    *,
    data_used: pd.DataFrame,
    prefix: str,
    credible_mass: float = 0.95,
) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    ci_low, ci_high = hdi_interval(samples, prob=credible_mass)
    return {
        f"{prefix}_mean_z": float(np.mean(samples)),
        f"{prefix}_median_z": float(np.median(samples)),
        f"{prefix}_ci_low_z": float(ci_low),
        f"{prefix}_ci_high_z": float(ci_high),
        f"{prefix}_mean_orig": _denorm_slope(float(np.mean(samples)), data_used),
        f"{prefix}_median_orig": _denorm_slope(float(np.median(samples)), data_used),
        f"{prefix}_ci_low_orig": _denorm_slope(float(ci_low), data_used),
        f"{prefix}_ci_high_orig": _denorm_slope(float(ci_high), data_used),
        "prob_positive": float((samples > 0).mean()),
        "prob_negative": float((samples < 0).mean()),
        "prob_direction": prob_direction(samples),
        "n_samples": int(samples.size),
    }


def _slope_summary(
    samples: np.ndarray,
    *,
    data_used: pd.DataFrame,
    credible_mass: float = 0.95,
) -> dict[str, Any]:
    return _posterior_summary(
        samples,
        data_used=data_used,
        prefix="slope",
        credible_mass=credible_mass,
    )


def _draw_summary(
    samples: np.ndarray,
    *,
    data_used: pd.DataFrame,
    credible_mass: float = 0.95,
) -> dict[str, Any]:
    return _posterior_summary(
        samples,
        data_used=data_used,
        prefix="estimate",
        credible_mass=credible_mass,
    )


def _prediction_line(
    fitted: dict[str, Any],
    *,
    stimulus: str,
    rois: list[str],
    model_group: str,
    n_points: int = PREDICTION_LINE_POINTS,
) -> pd.DataFrame:
    data_used = fitted["data_used"]
    stats = _norm_stats(data_used)
    x_min = float(pd.to_numeric(data_used["true_deg"], errors="coerce").min())
    x_max = float(pd.to_numeric(data_used["true_deg"], errors="coerce").max())
    x_values = np.linspace(x_min, x_max, n_points)
    rows = []
    subject = str(data_used["subject"].iloc[0])
    for roi in rois:
        for x in x_values:
            rows.append(
                {
                    "model_group": model_group,
                    "stimulus": stimulus,
                    "roi": roi,
                    "subject": subject,
                    "true_deg": float(x),
                    "true_deg_z": float((x - stats["mu_x"]) / stats["sd_x"]),
                }
            )
    newdata = pd.DataFrame(rows)

    pandas2ri.activate()
    model_name = fitted["r_model_name"]
    newdata_name = "prediction_line_newdata"
    pred_name = "prediction_line_pred"
    ro.globalenv[newdata_name] = newdata
    ro.globalenv["prediction_line_model"] = ro.r(model_name)
    ro.r(
        f"""
        prediction_line_model_frame <- stats::model.frame(prediction_line_model)
        for (column in intersect(names(prediction_line_model_frame), names({newdata_name}))) {{
          if (is.factor(prediction_line_model_frame[[column]])) {{
            {newdata_name}[[column]] <- factor(
              {newdata_name}[[column]],
              levels = levels(prediction_line_model_frame[[column]])
            )
          }}
        }}
        {pred_name} <- stats::predict(
          prediction_line_model,
          newdata = {newdata_name},
          re.form = ~0,
          allow.new.levels = TRUE
        )
        """
    )
    pred_z = np.asarray(ro.r(pred_name), dtype=float)
    out = newdata.copy()
    out["pred_deg_fit_z"] = pred_z
    out["pred_deg_fit"] = out["pred_deg_fit_z"] * stats["sd_y"] + stats["mu_y"]
    ro.r(f"rm({newdata_name}, {pred_name}, prediction_line_model, prediction_line_model_frame)")
    return out.drop(columns=["subject"])


def _save_frequentist_model_outputs(fitted: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(fitted["data_used"], output_dir / "data_used.csv")
    write_csv(fitted["coef_df"], output_dir / "fixed_effects.csv", index=True)
    write_csv(fitted["varcorr_df"], output_dir / "random_effects.csv")
    write_csv(pd.DataFrame([fitted["stats"]]), output_dir / "model_stats.csv")
    if fitted.get("anova_table") is not None:
        write_csv(fitted["anova_table"], output_dir / "anova.csv", index=True)
    if fitted.get("tidy") is not None:
        write_csv(fitted["tidy"], output_dir / "tidy.csv")
    if fitted.get("glance") is not None:
        write_csv(fitted["glance"], output_dir / "glance.csv")
    write_text(fitted.get("summary_text", ""), output_dir / "summary.txt")
    write_json(_frequentist_metadata(fitted), output_dir / "metadata.json")
    ro.r(f'saveRDS({fitted["r_model_name"]}, file="{output_dir / (fitted["r_model_name"] + ".rds")}")')


def _frequentist_metadata(fitted: dict[str, Any]) -> dict[str, Any]:
    return {
        "formula": fitted.get("formula"),
        "model_type": fitted.get("model_type"),
        "backend": fitted.get("backend"),
        "df_method_requested": fitted.get("df_method_requested"),
        "df_method_used": fitted.get("df_method_used"),
        "preproc": fitted.get("preproc"),
    }


def _bayesian_metadata(formula: str, data_used: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "formula": formula,
        "backend": "bayesian",
        "package": "bambi",
        "response_col": "pred_deg_z",
        "continuous_cols": ["true_deg", "pred_deg"],
        "sampler": {
            "draws": args.draws,
            "tune": args.tune,
            "target_accept": args.target_accept,
            "max_treedepth": args.max_treedepth,
            "random_seed": args.random_seed,
            "chains": args.chains,
            "cores": args.cores,
        },
        "normalization": _norm_stats(data_used),
    }


def _save_bayesian_model_outputs(
    result: dict[str, Any],
    formula: str,
    output_dir: Path,
    *,
    model_group: str,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    idata = result["idata"][formula]
    model = result["models"][formula]
    data_used = model.data.copy()
    write_csv(data_used, output_dir / "data_used.csv")
    write_csv(BayesianLMMSummary.fixed_effects(idata), output_dir / "fixed_effects.csv")
    idata.to_netcdf(output_dir / "idata.nc")
    write_json(_bayesian_metadata(formula, data_used, args) | {"model_group": model_group}, output_dir / "metadata.json")
    try:
        diagnostics = az.summary(idata, kind="diagnostics").reset_index(names="term")
        write_csv(diagnostics, output_dir / "sampler_diagnostics.csv")
    except Exception as exc:
        write_text(f"Could not compute sampler diagnostics: {exc}\n", output_dir / "sampler_diagnostics_warning.txt")


def run_slope_frequentist(df: pd.DataFrame, output_dir: Path) -> None:
    slope_root = output_dir / "slope_by_stimulus"
    slope_tables = []
    contrast_tables = []
    # Fit one WholeVC model and one joint sub-ROI model for each stimulus.
    for stimulus in STIMULI:
        df_stim = df[df["stimulus"] == stimulus].copy()

        df_wholevc = df_stim[df_stim["roi"] == WHOLE_VISUAL_ROI].copy()
        wholevc = _fit_frequentist_model(df_wholevc, FORMULA_SLOPE_WHOLEVC, f"model_wholevc_{stimulus}", ref_levels=None)
        wholevc_dir = slope_root / "wholevc" / stimulus
        _save_frequentist_model_outputs(wholevc, wholevc_dir)
        slopes = estimate_emtrends(
            wholevc,
            x_var=X_VAR,
            specs="1",
            denorm_slope=_denorm_slope,
            one_sided=">",
        )
        slopes["roi"] = WHOLE_VISUAL_ROI
        slopes["stimulus"] = stimulus
        slopes["model_group"] = "wholevc"
        write_csv(slopes, wholevc_dir / "slopes.csv")
        write_csv(
            _prediction_line(wholevc, stimulus=stimulus, rois=[WHOLE_VISUAL_ROI], model_group="wholevc"),
            wholevc_dir / "prediction_line.csv",
        )
        slope_tables.append(slopes)

        df_subroi = df_stim[df_stim["roi"].isin(SUBROIS)].copy()
        subroi = _fit_frequentist_model(df_subroi, FORMULA_SLOPE_SUBROI, f"model_subroi_joint_{stimulus}", ref_levels={"roi": ROI_REF})
        subroi_dir = slope_root / "subroi" / stimulus
        _save_frequentist_model_outputs(subroi, subroi_dir)
        slopes = estimate_emtrends(
            subroi,
            x_var=X_VAR,
            specs="roi",
            denorm_slope=_denorm_slope,
            one_sided=">",
        )
        slopes["stimulus"] = stimulus
        slopes["model_group"] = "subroi"
        write_csv(slopes, subroi_dir / "slopes.csv")
        roi_contrasts = treatment_vs_control_emtrends(
            subroi,
            x_var=X_VAR,
            factor="roi",
            reference_level=ROI_REF,
            denorm_slope=_denorm_slope,
        )
        roi_contrasts["stimulus"] = stimulus
        roi_contrasts["model_group"] = "subroi"
        write_csv(roi_contrasts, subroi_dir / "earlyvc_slope_contrasts.csv")
        write_csv(
            _prediction_line(subroi, stimulus=stimulus, rois=SUBROIS, model_group="subroi"),
            subroi_dir / "prediction_line.csv",
        )
        slope_tables.append(slopes)
        contrast_tables.append(roi_contrasts)

    write_csv(pd.concat(slope_tables, ignore_index=True), slope_root / "slopes_all.csv")
    write_csv(pd.concat(contrast_tables, ignore_index=True), slope_root / "earlyvc_slope_contrasts_all.csv")


def run_horizontal_vertical_frequentist(df: pd.DataFrame, output_dir: Path) -> None:
    hv_root = output_dir / "horizontal_vs_vertical"
    hv_df = df[df["stimulus"].isin(HV_STIMULI)].copy()
    hv_df["stimulus"] = hv_df["stimulus"].cat.remove_unused_categories()

    # Fit the comparison once for WholeVC and once for the four sub-ROIs.
    df_wholevc = hv_df[hv_df["roi"] == WHOLE_VISUAL_ROI].copy()
    wholevc = _fit_frequentist_model(df_wholevc, FORMULA_HV_WHOLEVC, "model_wholevc_horizontal_vs_vertical", ref_levels=None)
    wholevc_dir = hv_root / "wholevc"
    _save_frequentist_model_outputs(wholevc, wholevc_dir)
    wholevc_slopes = estimate_emtrends(
        wholevc,
        x_var=X_VAR,
        specs="stimulus",
        denorm_slope=_denorm_slope,
        one_sided=">",
    )
    wholevc_slopes["roi"] = WHOLE_VISUAL_ROI
    write_csv(wholevc_slopes, wholevc_dir / "slopes.csv")
    wholevc_contrasts = pairwise_emtrends_contrasts(
        wholevc,
        x_var=X_VAR,
        specs="stimulus",
        by=None,
        side=">",
        denorm_slope=_denorm_slope,
        ci_level=CI_LEVEL,
        fallback_by_levels={"roi": SUBROIS, "stimulus": HV_STIMULI},
        fallback_specs_levels={"stimulus": HV_STIMULI, "roi": SUBROIS},
    )
    wholevc_contrasts["roi"] = WHOLE_VISUAL_ROI
    write_csv(wholevc_contrasts, wholevc_dir / "horizontal_vertical_slope_contrasts.csv")

    df_subroi = hv_df[hv_df["roi"].isin(SUBROIS)].copy()
    subroi = _fit_frequentist_model(df_subroi, FORMULA_HV_SUBROI, "model_subroi_with_3way_interaction", ref_levels={"roi": ROI_REF})
    subroi_dir = hv_root / "subroi"
    _save_frequentist_model_outputs(subroi, subroi_dir)
    subroi_slopes = estimate_emtrends(
        subroi,
        x_var=X_VAR,
        specs="stimulus",
        by="roi",
        denorm_slope=_denorm_slope,
        one_sided=">",
    )
    write_csv(subroi_slopes, subroi_dir / "slopes.csv")
    subroi_contrasts = pairwise_emtrends_contrasts(
        subroi,
        x_var=X_VAR,
        specs="stimulus",
        by="roi",
        side=">",
        denorm_slope=_denorm_slope,
        ci_level=CI_LEVEL,
        fallback_by_levels={"roi": SUBROIS, "stimulus": HV_STIMULI},
        fallback_specs_levels={"stimulus": HV_STIMULI, "roi": SUBROIS},
    )
    write_csv(subroi_contrasts, subroi_dir / "horizontal_vertical_slope_contrasts.csv")

    write_csv(pd.concat([wholevc_slopes, subroi_slopes], ignore_index=True), hv_root / "slopes_all.csv")
    write_csv(
        pd.concat([wholevc_contrasts, subroi_contrasts], ignore_index=True),
        hv_root / "horizontal_vertical_slope_contrasts_all.csv",
    )


def run_slope_bayesian(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    slope_root = output_dir / "slope_by_stimulus"
    slope_tables = []
    contrast_tables = []
    # Use the same model split as the frequentist analysis: WholeVC separately,
    # then the four sub-ROIs in a single interaction model.
    for stimulus in STIMULI:
        df_stim = df[df["stimulus"] == stimulus].copy()

        df_wholevc = df_stim[df_stim["roi"] == WHOLE_VISUAL_ROI].copy()
        wholevc = _fit_bayesian_model(df_wholevc, FORMULA_SLOPE_WHOLEVC, roi_ref=None, args=args)
        wholevc_dir = slope_root / "wholevc" / stimulus
        _save_bayesian_model_outputs(wholevc, FORMULA_SLOPE_WHOLEVC, wholevc_dir, model_group="wholevc", args=args)
        idata = wholevc["idata"][FORMULA_SLOPE_WHOLEVC]
        data_used = wholevc["models"][FORMULA_SLOPE_WHOLEVC].data
        extractor = PosteriorSlopeExtractor(idata, x_var=X_VAR)
        slopes = pd.DataFrame(
            [
                {
                    "roi": WHOLE_VISUAL_ROI,
                    "stimulus": stimulus,
                    "model_group": "wholevc",
                    "formula": FORMULA_SLOPE_WHOLEVC,
                    **_slope_summary(extractor.base_slope(), data_used=data_used),
                }
            ]
        )
        write_csv(slopes, wholevc_dir / "slopes.csv")
        slope_tables.append(slopes)

        df_subroi = df_stim[df_stim["roi"].isin(SUBROIS)].copy()
        subroi = _fit_bayesian_model(df_subroi, FORMULA_SLOPE_SUBROI, roi_ref=ROI_REF, args=args)
        subroi_dir = slope_root / "subroi" / stimulus
        _save_bayesian_model_outputs(subroi, FORMULA_SLOPE_SUBROI, subroi_dir, model_group="subroi", args=args)
        idata = subroi["idata"][FORMULA_SLOPE_SUBROI]
        data_used = subroi["models"][FORMULA_SLOPE_SUBROI].data
        extractor = PosteriorSlopeExtractor(idata, x_var=X_VAR)
        roi_levels = factor_levels_from_bayesian_model(subroi, FORMULA_SLOPE_SUBROI, "roi", SUBROIS)
        roi_samples = {roi: extractor.roi_slope(roi, roi_levels) for roi in roi_levels}
        slopes = pd.DataFrame(
            [
                {
                    "roi": roi,
                    "stimulus": stimulus,
                    "model_group": "subroi",
                    "formula": FORMULA_SLOPE_SUBROI,
                    **_slope_summary(samples, data_used=data_used),
                }
                for roi, samples in roi_samples.items()
            ]
        )
        write_csv(slopes, subroi_dir / "slopes.csv")
        contrasts = pd.DataFrame(
            [
                {
                    "roi": roi,
                    "reference_roi": ROI_REF,
                    "contrast": f"{roi} - {ROI_REF}",
                    "stimulus": stimulus,
                    "model_group": "subroi",
                    "formula": FORMULA_SLOPE_SUBROI,
                    **_draw_summary(roi_samples[roi] - roi_samples[ROI_REF], data_used=data_used),
                }
                for roi in roi_levels
                if roi != ROI_REF
            ]
        )
        write_csv(contrasts, subroi_dir / "earlyvc_slope_contrasts.csv")
        slope_tables.append(slopes)
        contrast_tables.append(contrasts)

    write_csv(pd.concat(slope_tables, ignore_index=True), slope_root / "slopes_all.csv")
    write_csv(pd.concat(contrast_tables, ignore_index=True), slope_root / "earlyvc_slope_contrasts_all.csv")


def run_horizontal_vertical_bayesian(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    hv_root = output_dir / "horizontal_vs_vertical"
    hv_df = df[df["stimulus"].isin(HV_STIMULI)].copy()
    hv_df["stimulus"] = hv_df["stimulus"].cat.remove_unused_categories()

    # Summaries are computed from posterior slope draws so the reported
    # intervals are HDIs on the denormalized slope scale.
    df_wholevc = hv_df[hv_df["roi"] == WHOLE_VISUAL_ROI].copy()
    wholevc = _fit_bayesian_model(df_wholevc, FORMULA_HV_WHOLEVC, roi_ref=None, args=args)
    wholevc_dir = hv_root / "wholevc"
    _save_bayesian_model_outputs(wholevc, FORMULA_HV_WHOLEVC, wholevc_dir, model_group="wholevc", args=args)
    idata = wholevc["idata"][FORMULA_HV_WHOLEVC]
    data_used = wholevc["models"][FORMULA_HV_WHOLEVC].data
    extractor = PosteriorSlopeExtractor(idata, x_var=X_VAR)
    stimulus_levels = factor_levels_from_bayesian_model(wholevc, FORMULA_HV_WHOLEVC, "stimulus", HV_STIMULI)
    stim_samples = {stimulus: extractor.stimulus_slope(stimulus, stimulus_levels) for stimulus in stimulus_levels}
    wholevc_slopes = pd.DataFrame(
        [
            {
                "roi": WHOLE_VISUAL_ROI,
                "stimulus": stimulus,
                "model_group": "wholevc",
                "formula": FORMULA_HV_WHOLEVC,
                **_slope_summary(samples, data_used=data_used),
            }
            for stimulus, samples in stim_samples.items()
        ]
    )
    write_csv(wholevc_slopes, wholevc_dir / "slopes.csv")
    wholevc_contrasts = _bayesian_stimulus_contrasts(stim_samples, data_used=data_used)
    wholevc_contrasts["roi"] = WHOLE_VISUAL_ROI
    write_csv(wholevc_contrasts, wholevc_dir / "horizontal_vertical_slope_contrasts.csv")

    df_subroi = hv_df[hv_df["roi"].isin(SUBROIS)].copy()
    subroi = _fit_bayesian_model(df_subroi, FORMULA_HV_SUBROI, roi_ref=ROI_REF, args=args)
    subroi_dir = hv_root / "subroi"
    _save_bayesian_model_outputs(subroi, FORMULA_HV_SUBROI, subroi_dir, model_group="subroi", args=args)
    idata = subroi["idata"][FORMULA_HV_SUBROI]
    data_used = subroi["models"][FORMULA_HV_SUBROI].data
    extractor = PosteriorSlopeExtractor(idata, x_var=X_VAR)
    roi_levels = factor_levels_from_bayesian_model(subroi, FORMULA_HV_SUBROI, "roi", SUBROIS)
    stimulus_levels = factor_levels_from_bayesian_model(subroi, FORMULA_HV_SUBROI, "stimulus", HV_STIMULI)
    roi_stim_samples = {
        (roi, stimulus): extractor.roi_stimulus_slope(
            roi=roi,
            stimulus=stimulus,
            roi_levels=roi_levels,
            stimulus_levels=stimulus_levels,
        )
        for roi in roi_levels
        for stimulus in stimulus_levels
    }
    subroi_slopes = pd.DataFrame(
        [
            {
                "roi": roi,
                "stimulus": stimulus,
                "model_group": "subroi",
                "formula": FORMULA_HV_SUBROI,
                **_slope_summary(samples, data_used=data_used),
            }
            for (roi, stimulus), samples in roi_stim_samples.items()
        ]
    )
    write_csv(subroi_slopes, subroi_dir / "slopes.csv")
    subroi_contrasts = pd.concat(
        [
            _bayesian_stimulus_contrasts(
                {stimulus: roi_stim_samples[(roi, stimulus)] for stimulus in stimulus_levels},
                data_used=data_used,
            ).assign(roi=roi)
            for roi in roi_levels
        ],
        ignore_index=True,
    )
    write_csv(subroi_contrasts, subroi_dir / "horizontal_vertical_slope_contrasts.csv")

    write_csv(pd.concat([wholevc_slopes, subroi_slopes], ignore_index=True), hv_root / "slopes_all.csv")
    write_csv(
        pd.concat([wholevc_contrasts, subroi_contrasts], ignore_index=True),
        hv_root / "horizontal_vertical_slope_contrasts_all.csv",
    )


def _bayesian_stimulus_contrasts(samples_by_stimulus: dict[str, np.ndarray], *, data_used: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stimuli = list(samples_by_stimulus)
    for i, stim1 in enumerate(stimuli):
        for stim2 in stimuli[i + 1 :]:
            diff = samples_by_stimulus[stim1] - samples_by_stimulus[stim2]
            rows.append(
                {
                    "contrast": f"{stim1} - {stim2}",
                    **_draw_summary(diff, data_used=data_used),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    context = ContourSlantLMMContext.from_args(args)
    ContourSlantLMMAnalysis(context, args).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Analysis config YAML. Default: configs/atlasnet.yaml",
    )
    parser.add_argument(
        "--analysis",
        choices=["all", "slope", "horizontal-vertical"],
        default="all",
        help="Which paper analysis to reproduce.",
    )
    parser.add_argument(
        "--backend",
        choices=["all", "frequentist", "bayesian"],
        default="all",
        help="Which model family to fit.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Public data directory containing reconstruction outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "outputs/statistics/contour_matched_lmm/<representation> "
            "or <representation>_include_largest_slant when "
            "--include-largest-slant is set."
        ),
    )
    parser.add_argument(
        "--include-largest-slant",
        action="store_true",
        help="Include the largest nominal slants (+/-60 deg) in the slant LMM input tables.",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=5000)
    parser.add_argument("--target-accept", type=float, default=0.995)
    parser.add_argument("--max-treedepth", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

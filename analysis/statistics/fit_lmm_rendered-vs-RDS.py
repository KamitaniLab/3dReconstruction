"""Reproduce reconstruction-identification LMM summary tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recon3d.config import load_yaml_config, resolve_path
from recon3d.evaluation.reconstruction import compute_identification_accuracy
from recon3d.metadata import WHOLE_VISUAL_ROI
from recon3d.stats.bayesian_lmm import BayesianLMMHelper, BayesianLMMSummary
from recon3d.stats.frequentist_lmm import LMMHelper
from recon3d.stats.io import write_csv, write_json, write_text
from recon3d.stats.posterior import draw_summary, posterior_diagnostics
from recon3d.subjects import (
    PUBLIC_SUBJECTS,
    apply_bayesian_subject_aliases,
    restore_bayesian_subject_labels,
)


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ANALYSIS_DIR / "configs" / "atlasnet.yaml"
DEFAULT_OUTPUT_ROOT = ANALYSIS_DIR / "outputs" / "statistics" / "reconstruction_lmm"

RENDERED_DATASET = "test-3d-artificial-objects-image_rep8"
RDS_DATASET = "test-3d-artificial-objects-rds_rep8"
DATASETS = [RENDERED_DATASET, RDS_DATASET]

FORMULA_4ROI = "accuracy ~ roi * dataset + (dataset|subject)"
FORMULA_WHOLEVC = "accuracy ~ dataset + (dataset|subject)"
ROI_REF = "EarlyVC"
RECONSTRUCTION_ROIS = ["EarlyVC", "MTVC", "DorsalVC", "VentralVC", WHOLE_VISUAL_ROI]


@dataclass(frozen=True)
class ReconstructionLMMContext:
    """Resolved config values shared by the rendered-vs-RDS LMMs."""

    representation_name: str
    results_root: Path
    output_dir: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ReconstructionLMMContext":
        config = load_yaml_config(args.config)
        representation_name = config["representation"]["name"]
        output_dir = Path(args.output_dir) if args.output_dir is not None else DEFAULT_OUTPUT_ROOT / representation_name
        return cls(
            representation_name=representation_name,
            results_root=resolve_path(
                config["reconstruction_evaluation"]["results_dir"],
                base=ANALYSIS_DIR,
            ),
            output_dir=output_dir,
        )

    def lmm_dataframe(self) -> pd.DataFrame:
        accuracy = compute_identification_accuracy(
            results_root=self.results_root,
            representation_name=self.representation_name,
            stimulus_sets=DATASETS,
            subjects=PUBLIC_SUBJECTS,
            rois=RECONSTRUCTION_ROIS,
            num_lures=1,
        )

        rows = []
        for dataset in DATASETS:
            raw = accuracy[dataset]["identification accuracy"]["raw"]
            for subject in PUBLIC_SUBJECTS:
                for roi in RECONSTRUCTION_ROIS:
                    rows.extend(
                        {
                            "accuracy": float(value),
                            "subject": subject,
                            "roi": roi,
                            "dataset": dataset,
                        }
                        for value in raw[roi][subject]
                    )
        return _ordered_categories(pd.DataFrame(rows))


@dataclass
class RenderedVsRDSLMMAnalysis:
    context: ReconstructionLMMContext
    args: argparse.Namespace

    def run(self) -> None:
        output_dir = self.context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build the input table once, split it into the two model tables, then
        # run whichever statistical backends were requested.
        df = self.context.lmm_dataframe()
        df_4roi, df_wholevc = _split_dataframes(df)

        write_csv(df, output_dir / "lmm_input.csv")
        self.write_metadata()

        if self.args.backend in ("all", "frequentist"):
            run_frequentist(df_4roi, df_wholevc, output_dir / "frequentist")

        if self.args.backend in ("all", "bayesian"):
            run_bayesian(df_4roi, df_wholevc, output_dir / "bayesian", self.args)

    def write_metadata(self) -> None:
        write_json(
            {
                "representation": self.context.representation_name,
                "datasets": {
                    "rendered": RENDERED_DATASET,
                    "rds": RDS_DATASET,
                },
                "subjects": PUBLIC_SUBJECTS,
                "rois": RECONSTRUCTION_ROIS,
                "formulas": {
                    "4roi": FORMULA_4ROI,
                    "wholevc": FORMULA_WHOLEVC,
                },
                "reference_levels": {
                    "dataset": RENDERED_DATASET,
                    "roi": ROI_REF,
                },
                "note": "Accuracy is object-identification accuracy computed from reconstruction Chamfer-distance matrices.",
            },
            self.context.output_dir / "metadata.json",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the reconstruction-identification LMMs and write CSV summaries."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Analysis config YAML. Default: configs/atlasnet.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/statistics/reconstruction_lmm/<representation>",
    )
    parser.add_argument(
        "--backend",
        choices=("all", "frequentist", "bayesian"),
        default="all",
        help="Which model family to fit.",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=5000)
    parser.add_argument("--target-accept", type=float, default=0.995)
    parser.add_argument("--max-treedepth", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None)
    return parser.parse_args()


def _ordered_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Keep the intended level order for summaries, but use unordered categoricals.
    # R's stats::relevel() refuses ordered factors after rpy2 conversion.
    df["subject"] = pd.Categorical(df["subject"], categories=PUBLIC_SUBJECTS, ordered=False)
    df["roi"] = pd.Categorical(df["roi"], categories=RECONSTRUCTION_ROIS, ordered=False)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=False)
    return df


def _split_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The Bayesian fit is sensitive to factor level order; this order reproduces
    # the ROI coding used for the reported Bayesian estimates.
    model_rois = ["EarlyVC", "DorsalVC", "MTVC", "VentralVC"]
    df_4roi = df[df["roi"].isin(model_rois)].copy()
    df_4roi["roi"] = pd.Categorical(df_4roi["roi"], categories=model_rois, ordered=True)
    df_4roi["dataset"] = pd.Categorical(df_4roi["dataset"], categories=DATASETS, ordered=True)

    df_wholevc = df[df["roi"] == WHOLE_VISUAL_ROI].copy().drop(columns=["roi"])
    df_wholevc["dataset"] = pd.Categorical(df_wholevc["dataset"], categories=DATASETS, ordered=True)
    return df_4roi, df_wholevc


def _with_term_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.name is None:
        out.index.name = "term"
    return out.reset_index()


def _model_stats_row(name: str, fitted: dict[str, Any]) -> dict[str, Any]:
    row = {"model": name, "formula": fitted["formula"]}
    row.update(fitted.get("stats", {}))
    row["backend"] = fitted.get("backend")
    row["df_method"] = fitted.get("df_method_used")
    return row


def _fit_frequentist_model(
    df: pd.DataFrame,
    *,
    formula: str,
    model_name: str,
    roi_ref: str | None,
) -> dict[str, Any]:
    df_for_r = df.copy()
    for col in ["subject", "dataset", "roi"]:
        if col in df_for_r.columns and isinstance(df_for_r[col].dtype, pd.CategoricalDtype):
            df_for_r[col] = pd.Categorical(
                df_for_r[col],
                categories=list(df_for_r[col].cat.categories),
                ordered=False,
            )
    ref_levels = {"dataset": RENDERED_DATASET}
    factor_cols = ["subject", "dataset"]
    if roi_ref is not None:
        ref_levels["roi"] = roi_ref
        factor_cols.append("roi")
    return LMMHelper.fit_lmm_r(
        df_for_r,
        formula,
        backend="lmerTest",
        df_method="Kenward-Roger",
        ensure_factor_cols=factor_cols,
        ref_levels=ref_levels,
        model_name=model_name,
    )


def _scalar(row: pd.Series, candidates: tuple[str, ...]) -> float:
    for column in candidates:
        if column in row.index:
            return float(row[column])
    raise KeyError(f"None of the candidate columns were found: {candidates}")


def _p_column(df: pd.DataFrame) -> str:
    column = LMMHelper.find_column(df, ("p.value", "Pr(>|t|)", "Pr(>F)", "p"))
    if column is None:
        raise KeyError(f"No p-value column found in columns: {list(df.columns)}")
    return column


def _contrast_direction(row: pd.Series) -> int:
    contrast = str(row.get("contrast", ""))
    compact = contrast.replace(" ", "").replace("(", "").replace(")", "")
    rendered = RENDERED_DATASET
    rds = RDS_DATASET
    if compact == f"{rds}-{rendered}":
        return 1
    if compact == f"{rendered}-{rds}":
        return -1
    if "RDS-rendered" in compact or "rds-image" in compact:
        return 1
    if "rendered-RDS" in compact or "image-rds" in compact:
        return -1
    raise ValueError(f"Could not determine contrast direction from {contrast!r}.")


def _extract_simple_effect(row: pd.Series, *, model: str, roi: str) -> dict[str, Any]:
    sign = _contrast_direction(row)
    estimate = sign * _scalar(row, ("estimate", "Estimate", "contrast"))
    lower = sign * _scalar(row, ("CI_lower", "lower.CL", "asymp.LCL"))
    upper = sign * _scalar(row, ("CI_upper", "upper.CL", "asymp.UCL"))
    if lower > upper:
        lower, upper = upper, lower
    return {
        "model": model,
        "roi": roi,
        "contrast": "RDS - rendered",
        "estimate": estimate,
        "CI_lower": lower,
        "CI_upper": upper,
        "p_value": _scalar(row, (_p_column(pd.DataFrame([row])),)),
    }


def _frequentist_simple_effects(fitted_4roi: dict[str, Any], fitted_wholevc: dict[str, Any]) -> pd.DataFrame:
    rows = []
    table_rois = ["EarlyVC", "MTVC", "DorsalVC", "VentralVC"]
    contrast_4roi = LMMHelper.test_emmeans_contrasts(
        fitted_4roi,
        specs="dataset",
        by="roi",
        method="pairwise",
        reverse=True,
    )
    for roi in table_rois:
        roi_rows = contrast_4roi[contrast_4roi["roi"] == roi]
        if len(roi_rows) != 1:
            raise RuntimeError(f"Expected one contrast row for {roi}, got {len(roi_rows)}.")
        rows.append(_extract_simple_effect(roi_rows.iloc[0], model="4roi", roi=roi))

    contrast_wholevc = LMMHelper.test_emmeans_contrasts(
        fitted_wholevc,
        specs="dataset",
        method="pairwise",
        reverse=True,
    )
    if len(contrast_wholevc) != 1:
        raise RuntimeError(f"Expected one WholeVC contrast row, got {len(contrast_wholevc)}.")
    rows.append(_extract_simple_effect(contrast_wholevc.iloc[0], model="wholevc", roi=WHOLE_VISUAL_ROI))
    return pd.DataFrame(rows)


def _term_contains(term: str, *needles: str) -> bool:
    return all(needle in term for needle in needles)


def _frequentist_interactions(fitted_4roi: dict[str, Any]) -> pd.DataFrame:
    coef_df = _with_term_index(fitted_4roi["coef_df"])
    p_col = _p_column(coef_df)
    rows = []
    interaction_rois = ["MTVC", "DorsalVC", "VentralVC"]
    for roi in interaction_rois:
        matches = coef_df[
            coef_df["term"].map(lambda term: _term_contains(str(term), ":", roi, RDS_DATASET))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one interaction coefficient for {roi}, got {len(matches)}.")
        row = matches.iloc[0]
        rows.append(
            {
                "model": "4roi",
                "roi": roi,
                "reference_roi": ROI_REF,
                "contrast": f"({roi} - {ROI_REF}) x (RDS - rendered)",
                "estimate": _scalar(row, ("Estimate", "estimate")),
                "CI_lower": _scalar(row, ("CI_lower", "lower.CL")),
                "CI_upper": _scalar(row, ("CI_upper", "upper.CL")),
                "p_value": float(row[p_col]),
            }
        )
    return pd.DataFrame(rows)


def _format_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _format_prob(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def _frequentist_summary_simple(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in effects.iterrows():
        rows.append(
            {
                "roi": row["roi"],
                "contrast": row["contrast"],
                "estimate_percent": row["estimate"] * 100,
                "CI_lower_percent": row["CI_lower"] * 100,
                "CI_upper_percent": row["CI_upper"] * 100,
                "p_value": row["p_value"],
                "p_value_text": _format_p(row["p_value"]),
            }
        )
    return pd.DataFrame(rows)


def _frequentist_summary_interactions(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in effects.iterrows():
        rows.append(
            {
                "roi": row["roi"],
                "reference_roi": row["reference_roi"],
                "contrast": row["contrast"],
                "estimate_percent": row["estimate"] * 100,
                "CI_lower_percent": row["CI_lower"] * 100,
                "CI_upper_percent": row["CI_upper"] * 100,
                "p_value": row["p_value"],
                "p_value_text": _format_p(row["p_value"]),
            }
        )
    return pd.DataFrame(rows)


def _save_frequentist_model_outputs(name: str, fitted: dict[str, Any], out_dir: Path) -> None:
    write_csv(fitted["data_used"], out_dir / name / "data_used.csv")
    write_csv(_with_term_index(fitted["coef_df"]), out_dir / name / "fixed_effects.csv")
    write_csv(pd.DataFrame([_model_stats_row(name, fitted)]), out_dir / name / "model_stats.csv")
    write_text(fitted["summary_text"], out_dir / name / "summary.txt")


def run_frequentist(df_4roi: pd.DataFrame, df_wholevc: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fit the four-sub-ROI interaction model and the WholeVC model from the
    # same identification-accuracy table.
    fitted_4roi = _fit_frequentist_model(
        df_4roi,
        formula=FORMULA_4ROI,
        model_name="reconstruction_lmm_4roi",
        roi_ref=ROI_REF,
    )
    fitted_wholevc = _fit_frequentist_model(
        df_wholevc,
        formula=FORMULA_WHOLEVC,
        model_name="reconstruction_lmm_wholevc",
        roi_ref=None,
    )

    _save_frequentist_model_outputs("4roi", fitted_4roi, out_dir)
    _save_frequentist_model_outputs("wholevc", fitted_wholevc, out_dir)

    fixed = pd.concat(
        [
            _with_term_index(fitted_4roi["coef_df"]).assign(model="4roi"),
            _with_term_index(fitted_wholevc["coef_df"]).assign(model="wholevc"),
        ],
        ignore_index=True,
    )
    model_stats = pd.DataFrame(
        [
            _model_stats_row("4roi", fitted_4roi),
            _model_stats_row("wholevc", fitted_wholevc),
        ]
    )

    simple = _frequentist_simple_effects(fitted_4roi, fitted_wholevc)
    interactions = _frequentist_interactions(fitted_4roi)

    write_csv(fixed, out_dir / "fixed_effects.csv")
    write_csv(model_stats, out_dir / "model_stats.csv")
    write_csv(simple, out_dir / "simple_effects_rds_minus_rendered.csv")
    write_csv(interactions, out_dir / "interaction_effects_roi_by_rds.csv")
    write_csv(_frequentist_summary_simple(simple), out_dir / "summary_table_simple_effects.csv")
    write_csv(_frequentist_summary_interactions(interactions), out_dir / "summary_table_interactions.csv")


def _fit_bayesian_model(
    df: pd.DataFrame,
    *,
    formula: str,
    roi_ref: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_df, alias_to_subject = _bayesian_fit_dataframe(df)
    result = BayesianLMMHelper.fit_lmm_bayesian(
        model_df,
        formulas=[formula],
        response_col="accuracy",
        roi_ref=roi_ref,
        dataset_ref=RENDERED_DATASET,
        family="gaussian",
        beta_auto_squeeze=True,
        store_ll=True,
        store_loo=False,
        return_waic=False,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        random_seed=args.random_seed,
        chains=args.chains,
        cores=args.cores,
    )
    restore_bayesian_subject_labels(result, alias_to_subject)
    return result


def _bayesian_fit_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    model_df = df.copy()
    alias_to_subject = apply_bayesian_subject_aliases(model_df)
    return model_df, alias_to_subject


def _only_idata(result: dict[str, Any], formula: str):
    return result["idata"][formula]


def _two_sided_hdi(draws: np.ndarray, prob: float) -> tuple[float, float]:
    return BayesianLMMHelper._interval_bounds_1d(draws, prob=prob, sided="two-sided")


def _bayesian_simple_effects(idata_4roi, idata_wholevc) -> pd.DataFrame:
    rows = []
    table_rois = ["EarlyVC", "MTVC", "DorsalVC", "VentralVC"]
    for roi in table_rois:
        effect = BayesianLMMHelper.simple_effect(
            idata_4roi,
            factor1="dataset",
            level1=RDS_DATASET,
            level2=RENDERED_DATASET,
            cond_factor="roi",
            cond_level=roi,
            baseline_factor1=RENDERED_DATASET,
            baseline_cond=ROI_REF,
            prob=0.95,
            sided="two-sided",
        )
        rows.append(
            {
                "model": "4roi",
                "roi": roi,
                "contrast": "RDS - rendered",
                "beta": float(effect["beta"]),
                "median": float(effect["beta"]),
                "HDI_lower": float(effect["HDI_lower"]),
                "HDI_upper": float(effect["HDI_upper"]),
                "prob_greater": float(effect["prob_greater"]),
                "prob_less": float(effect["prob_less"]),
                "prob_direction": float(max(effect["prob_greater"], effect["prob_less"])),
            }
        )

    draws = BayesianLMMHelper._draws_1d(idata_wholevc, "dataset", RDS_DATASET)
    rows.append(
        {
            "model": "wholevc",
            "roi": WHOLE_VISUAL_ROI,
            "contrast": "RDS - rendered",
            **draw_summary(draws, interval_func=_two_sided_hdi),
        }
    )
    return pd.DataFrame(rows)


def _interaction_draws(idata, roi: str) -> np.ndarray:
    posterior_vars = set(idata.posterior.data_vars)
    if "roi:dataset" in posterior_vars:
        return BayesianLMMHelper._draws_1d(idata, "roi:dataset", (roi, RDS_DATASET))
    if "dataset:roi" in posterior_vars:
        return BayesianLMMHelper._draws_1d(idata, "dataset:roi", (RDS_DATASET, roi))
    raise KeyError("No roi-by-dataset interaction term found in Bayesian posterior.")


def _bayesian_interactions(idata_4roi) -> pd.DataFrame:
    rows = []
    interaction_rois = ["MTVC", "DorsalVC", "VentralVC"]
    for roi in interaction_rois:
        rows.append(
            {
                "model": "4roi",
                "roi": roi,
                "reference_roi": ROI_REF,
                "contrast": f"({roi} - {ROI_REF}) x (RDS - rendered)",
                **draw_summary(_interaction_draws(idata_4roi, roi), interval_func=_two_sided_hdi),
            }
        )
    return pd.DataFrame(rows)


def _bayesian_summary_simple(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in effects.iterrows():
        rows.append(
            {
                "roi": row["roi"],
                "contrast": row["contrast"],
                "beta_percent": row["beta"] * 100,
                "median_percent": row["median"] * 100 if "median" in row else np.nan,
                "HDI_lower_percent": row["HDI_lower"] * 100,
                "HDI_upper_percent": row["HDI_upper"] * 100,
                "prob_direction": row["prob_direction"],
                "prob_direction_text": _format_prob(row["prob_direction"]),
            }
        )
    return pd.DataFrame(rows)


def _bayesian_summary_interactions(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in effects.iterrows():
        rows.append(
            {
                "roi": row["roi"],
                "reference_roi": row["reference_roi"],
                "contrast": row["contrast"],
                "beta_percent": row["beta"] * 100,
                "median_percent": row["median"] * 100,
                "HDI_lower_percent": row["HDI_lower"] * 100,
                "HDI_upper_percent": row["HDI_upper"] * 100,
                "prob_direction": row["prob_direction"],
                "prob_direction_text": _format_prob(row["prob_direction"]),
            }
        )
    return pd.DataFrame(rows)


def run_bayesian(
    df_4roi: pd.DataFrame,
    df_wholevc: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fit the same two model formulas as the frequentist analysis and summarize
    # posterior contrasts on the original accuracy scale.
    result_4roi = _fit_bayesian_model(df_4roi, formula=FORMULA_4ROI, roi_ref=ROI_REF, args=args)
    result_wholevc = _fit_bayesian_model(df_wholevc, formula=FORMULA_WHOLEVC, roi_ref=None, args=args)

    idata_4roi = _only_idata(result_4roi, FORMULA_4ROI)
    idata_wholevc = _only_idata(result_wholevc, FORMULA_WHOLEVC)

    fixed = pd.concat(
        [
            BayesianLMMSummary.fixed_effects(idata_4roi).assign(model="4roi"),
            BayesianLMMSummary.fixed_effects(idata_wholevc).assign(model="wholevc"),
        ],
        ignore_index=True,
    )
    simple = _bayesian_simple_effects(idata_4roi, idata_wholevc)
    interactions = _bayesian_interactions(idata_4roi)
    diagnostics = pd.concat(
        [
            posterior_diagnostics(idata_4roi, model="4roi"),
            posterior_diagnostics(idata_wholevc, model="wholevc"),
        ],
        ignore_index=True,
    )

    write_csv(fixed, out_dir / "fixed_effects.csv")
    write_csv(simple, out_dir / "simple_effects_rds_minus_rendered.csv")
    write_csv(interactions, out_dir / "interaction_effects_roi_by_rds.csv")
    write_csv(diagnostics, out_dir / "posterior_diagnostics.csv")
    write_csv(_bayesian_summary_simple(simple), out_dir / "summary_table_simple_effects.csv")
    write_csv(_bayesian_summary_interactions(interactions), out_dir / "summary_table_interactions.csv")
    idata_4roi.to_netcdf(out_dir / "idata_4roi.nc")
    idata_wholevc.to_netcdf(out_dir / "idata_wholevc.nc")


def run(args: argparse.Namespace) -> None:
    context = ReconstructionLMMContext.from_args(args)
    RenderedVsRDSLMMAnalysis(context, args).run()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

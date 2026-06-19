"""Plotting helpers for contour-matched slant LMM outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recon3d.metadata import ROI_BASE_COLORS


STIMULUS_ORDER = [
    "horizontal_thin_bar",
    "horizontal_thick_bar",
    "horizontal_cylinder",
    "vertical_thin_bar",
]
SUBROI_ORDER = ["EarlyVC", "MTVC", "DorsalVC", "VentralVC"]
WHOLEVC = "WholeVC"

STIMULUS_LABELS = {
    "horizontal_thin_bar": "Thin bar",
    "horizontal_thick_bar": "Thick bar",
    "horizontal_cylinder": "Cylinder",
    "vertical_thin_bar": "Thin bar (vertical)",
}
ROI_LABELS = {
    "EarlyVC": "Early VC",
    "MTVC": "MT & neighbors",
    "DorsalVC": "Dorsal VC",
    "VentralVC": "Ventral VC",
    "WholeVC": "Whole VC",
}
ROI_COLORS = {
    **ROI_BASE_COLORS,
    "WholeVC": "#7F8CA3",
}
SUBJECT_LABELS = {subject: subject for subject in ["S1", "S2", "S3", "S4", "S5"]}
SUBJECT_COLORS = ["#888888", "#888888", "#888888", "#888888", "#888888"]
PREDICTION_LINE_COLOR = "#888888"


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5


def slope_model_dir(statistics_dir: Path, model_group: str, stimulus: str) -> Path:
    return statistics_dir / "slope_by_stimulus" / model_group / stimulus


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def normalize_slope_columns(slopes: pd.DataFrame) -> pd.DataFrame:
    out = slopes.copy()
    rename_map = {}
    if "slope_orig" in out.columns:
        rename_map["slope_orig"] = "slope"
    if "slope_median_orig" in out.columns:
        rename_map["slope_median_orig"] = "slope"
    if "lower.CL_orig" in out.columns:
        rename_map["lower.CL_orig"] = "ci_low"
    if "upper.CL_orig" in out.columns:
        rename_map["upper.CL_orig"] = "ci_high"
    if "slope_ci_low_orig" in out.columns:
        rename_map["slope_ci_low_orig"] = "ci_low"
    if "slope_ci_high_orig" in out.columns:
        rename_map["slope_ci_high_orig"] = "ci_high"
    return out.rename(columns=rename_map)


def p_to_star(p_value: float) -> str | None:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return None


def format_slant_axes(
    ax: plt.Axes,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    label_font_size: int = 8,
    tick_font_size: int = 7,
) -> None:
    ax.set_xlim(-88, 88)
    ax.set_ylim(-88, 88)
    ax.set_xticks([-60, -30, 0, 30, 60])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("Experimental slant (deg)" if show_xlabel else "", fontsize=label_font_size)
    ax.set_ylabel("Reconstructed slant (deg)" if show_ylabel else "", fontsize=label_font_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.axvline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")


def plot_observed_points_and_prediction(
    ax: plt.Axes,
    *,
    data: pd.DataFrame,
    prediction: pd.DataFrame,
    roi: str,
    point_size: float = 16,
    point_alpha: float = 0.35,
    line_width: float = 2.0,
) -> None:
    data_roi = data[data["roi"].astype(str) == roi].copy()
    pred_roi = prediction[prediction["roi"].astype(str) == roi].copy()
    if data_roi.empty:
        raise ValueError(f"No observed data found for ROI {roi!r}.")
    if pred_roi.empty:
        raise ValueError(f"No prediction line found for ROI {roi!r}.")

    subjects = list(pd.unique(data_roi["subject"].astype(str)))
    subject_color = {
        subject: SUBJECT_COLORS[idx % len(SUBJECT_COLORS)]
        for idx, subject in enumerate(subjects)
    }
    for subject in subjects:
        d = data_roi[data_roi["subject"].astype(str) == subject]
        ax.scatter(
            pd.to_numeric(d["true_deg"], errors="coerce"),
            pd.to_numeric(d["pred_deg"], errors="coerce"),
            s=point_size,
            alpha=point_alpha,
            color=subject_color[subject],
            label=SUBJECT_LABELS.get(subject, subject),
            edgecolors="none",
        )

    pred_roi = pred_roi.sort_values("true_deg")
    ax.plot(
        pd.to_numeric(pred_roi["true_deg"], errors="coerce"),
        pd.to_numeric(pred_roi["pred_deg_fit"], errors="coerce"),
        color=PREDICTION_LINE_COLOR,
        linewidth=line_width,
    )

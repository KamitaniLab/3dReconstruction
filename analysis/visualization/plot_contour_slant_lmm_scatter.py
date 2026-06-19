"""Plot observed slant and fitted LMM prediction lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from recon3d.config import (
    ANALYSIS_OUTPUT_ROOT,
    load_yaml_config,
    visualization_output_dir,
)
from recon3d.plotting.contour_slant_lmm import (
    ROI_LABELS,
    STIMULUS_LABELS,
    STIMULUS_ORDER,
    SUBROI_ORDER,
    WHOLEVC,
    configure_matplotlib,
    format_slant_axes,
    normalize_slope_columns,
    p_to_star,
    plot_observed_points_and_prediction,
    read_required_csv,
    slope_model_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/atlasnet.yaml"))
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=None,
        help="Directory containing outputs from statistics/fit_lmm_slope.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_OUTPUT_ROOT / "visualization",
        help="Root directory for generated figures.",
    )
    parser.add_argument(
        "--include-largest-slant",
        action="store_true",
        help=(
            "Read LMM outputs that include the largest nominal slants (+/-60 deg). "
            "Used only when --statistics-dir is not specified."
        ),
    )
    parser.add_argument(
        "--group",
        choices=["all", "wholevc", "subroi"],
        default="all",
        help="Which model group to plot.",
    )
    parser.add_argument(
        "--stimulus",
        choices=STIMULUS_ORDER,
        action="append",
        help="Stimulus to plot. May be passed multiple times. Defaults to all.",
    )
    return parser.parse_args()


def _add_slope_label(ax: plt.Axes, slopes, *, roi: str, stimulus: str) -> None:
    row = slopes[(slopes["roi"].astype(str) == roi) & (slopes["stimulus"].astype(str) == stimulus)]
    if row.empty or "slope" not in row.columns:
        return
    slope = float(row["slope"].iloc[0])
    ax.text(0, 68, f"slope = {slope:.2f}", ha="center", va="bottom", fontsize=7)


def _add_significance_label(ax: plt.Axes, fixed_effects, *, x_term: str = "true_deg_z") -> None:
    if x_term not in fixed_effects.index or "Pr(>t)" not in fixed_effects.columns:
        return
    star = p_to_star(float(fixed_effects.loc[x_term, "Pr(>t)"]))
    if star is not None:
        ax.text(0, 77, star, ha="center", va="bottom", fontsize=8)


def plot_wholevc(statistics_dir: Path, output_dir: Path, stimuli: list[str]) -> None:
    out_dir = output_dir / "wholevc"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stimulus in stimuli:
        model_dir = slope_model_dir(statistics_dir, "wholevc", stimulus)
        data = read_required_csv(model_dir / "data_used.csv")
        prediction = read_required_csv(model_dir / "prediction_line.csv")
        slopes = normalize_slope_columns(read_required_csv(model_dir / "slopes.csv"))
        fixed_effects = read_required_csv(model_dir / "fixed_effects.csv")
        fixed_effects = fixed_effects.set_index(fixed_effects.columns[0])

        fig, ax = plt.subplots(figsize=(1.8, 1.8))
        plot_observed_points_and_prediction(ax, data=data, prediction=prediction, roi=WHOLEVC)
        format_slant_axes(ax)
        ax.set_title(STIMULUS_LABELS.get(stimulus, stimulus), fontsize=8)
        _add_slope_label(ax, slopes, roi=WHOLEVC, stimulus=stimulus)
        _add_significance_label(ax, fixed_effects)

        save_path = out_dir / f"contour_slant_lmm_scatter_wholevc_{stimulus}.svg"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_subroi(statistics_dir: Path, output_dir: Path, stimuli: list[str]) -> None:
    out_dir = output_dir / "subroi"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stimulus in stimuli:
        model_dir = slope_model_dir(statistics_dir, "subroi", stimulus)
        data = read_required_csv(model_dir / "data_used.csv")
        prediction = read_required_csv(model_dir / "prediction_line.csv")
        slopes = normalize_slope_columns(read_required_csv(model_dir / "slopes.csv"))

        fig, axes = plt.subplots(1, len(SUBROI_ORDER), figsize=(7.2, 1.9), sharex=True, sharey=True)
        for idx, roi in enumerate(SUBROI_ORDER):
            ax = axes[idx]
            plot_observed_points_and_prediction(ax, data=data, prediction=prediction, roi=roi)
            format_slant_axes(
                ax,
                show_xlabel=idx == len(SUBROI_ORDER) // 2,
                show_ylabel=idx == 0,
            )
            title = ROI_LABELS.get(roi, roi)
            if idx == 0:
                title = f"{STIMULUS_LABELS.get(stimulus, stimulus)}\n{title}"
            else:
                title = f"\n{title}"
            ax.set_title(title, fontsize=8)
            _add_slope_label(ax, slopes, roi=roi, stimulus=stimulus)

        save_path = out_dir / f"contour_slant_lmm_scatter_subroi_{stimulus}.svg"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close(fig)
        print(f"Saved: {save_path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    config = load_yaml_config(args.config)
    representation_name = config["representation"]["name"]
    statistics_name = (
        f"{representation_name}_include_largest_slant"
        if args.include_largest_slant
        else representation_name
    )
    statistics_dir = args.statistics_dir or (
        ANALYSIS_OUTPUT_ROOT
        / "statistics"
        / "contour_matched_lmm"
        / statistics_name
        / "frequentist"
    )
    output_dir = visualization_output_dir(args.output_dir, config, __file__)
    if args.include_largest_slant:
        output_dir = output_dir / "include_largest_slant"
    stimuli = args.stimulus or STIMULUS_ORDER
    if args.group in {"all", "wholevc"}:
        plot_wholevc(statistics_dir, output_dir, stimuli)
    if args.group in {"all", "subroi"}:
        plot_subroi(statistics_dir, output_dir, stimuli)


if __name__ == "__main__":
    main()

"""Plot reconstruction identification accuracy."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recon3d.config import (
    ANALYSIS_OUTPUT_ROOT,
    load_yaml_config,
    resolve_path,
    visualization_output_dir,
)
from recon3d.evaluation.reconstruction import compute_identification_accuracy
from recon3d.metadata import (
    SUBJECT_COLORS,
    SUBROIS,
    WHOLE_VISUAL_ROI,
    roi_subject_colors,
)
from recon3d.stats import calc_ci
from recon3d.subjects import PUBLIC_SUBJECTS


DEFAULT_STIMULUS_SET_LABELS = (
    "natural\nwithin",
    "natural\nout-of",
    "artificial\nrendered",
    "artificial\nRDS",
)
STIMULUS_SETS = (
    "test-3d-natural-objects_rep8_insample",
    "test-3d-natural-objects_rep8_outsample",
    "test-3d-artificial-objects-image_rep8",
    "test-3d-artificial-objects-rds_rep8",
)
ARTIFICIAL_IMAGE_STIMULUS_SET = "test-3d-artificial-objects-image_rep8"
ARTIFICIAL_RDS_STIMULUS_SET = "test-3d-artificial-objects-rds_rep8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/atlasnet.yaml"))
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_OUTPUT_ROOT / "visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = visualization_output_dir(args.output_dir, config, __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, stimulus_sets = load_results(config)
    natural_stimulus_sets = stimulus_sets[:2]

    plot_whole_visual_cortex_results(results, stimulus_sets, output_dir=output_dir)
    plot_stimulus_setwise_roi_results(
        results,
        natural_stimulus_sets,
        output_dir=output_dir,
    )
    plot_artificial_comparison(config, results, output_dir=output_dir)


def load_results(config: dict) -> tuple[dict, list[str]]:
    results_root = resolve_path(
        config["reconstruction_evaluation"]["results_dir"],
        base=Path.cwd(),
    )
    stimulus_sets = list(STIMULUS_SETS)
    rois = [WHOLE_VISUAL_ROI, *SUBROIS]
    results = compute_identification_accuracy(
        results_root,
        config["representation"]["name"],
        stimulus_sets=stimulus_sets,
        subjects=PUBLIC_SUBJECTS,
        rois=rois,
        num_lures=1,
    )
    return results, stimulus_sets


def plot_whole_visual_cortex_results(
    results: dict,
    stimulus_sets: list[str],
    *,
    output_dir: Path,
) -> None:
    fig = plot_stimulus_set_identification_accuracy(
        results,
        stimulus_sets=stimulus_sets,
        subjects=PUBLIC_SUBJECTS,
        roi=WHOLE_VISUAL_ROI,
        stimulus_set_labels=DEFAULT_STIMULUS_SET_LABELS,
    )
    save_path = output_dir / "reconstruction_identification_accuracy_wholevc_2-way.svg"
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_stimulus_setwise_roi_results(
    results: dict,
    stimulus_sets: list[str],
    *,
    output_dir: Path,
) -> None:
    for stimulus_set in stimulus_sets:
        fig = plot_roi_identification_accuracy(
            results,
            stimulus_sets=[stimulus_set],
            subjects=PUBLIC_SUBJECTS,
            rois=SUBROIS,
        )
        save_path = (
            output_dir
            / f"reconstruction_roi_{stimulus_set}_identification_accuracy_2-way.svg"
        )
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_artificial_comparison(
    config: dict,
    results: dict,
    *,
    output_dir: Path,
) -> None:
    rds_stimulus_set = ARTIFICIAL_RDS_STIMULUS_SET
    image_stimulus_set = ARTIFICIAL_IMAGE_STIMULUS_SET
    paired_results = collect_paired_stimulus_set_values(
        {
            image_stimulus_set: results[image_stimulus_set],
            rds_stimulus_set: results[rds_stimulus_set],
        }
    )
    fig = plot_paired_stimulus_set_comparison(
        paired_results,
        subjects=PUBLIC_SUBJECTS,
        rois=SUBROIS,
        stimulus_set_label_a="rendered",
        stimulus_set_label_b="RDS",
        figsize=(4, 2),
    )
    save_path = (
        output_dir
        / "reconstruction_roi_artificial_objects_identification_accuracy_2-way.svg"
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_roi_identification_accuracy(
    result_data: dict,
    stimulus_sets: list[str],
    subjects: list[str],
    rois: list[str],
    *,
    num_lures: int = 1,
    y_ticks: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    y_ticklabels: tuple[str, ...] = ("40", "50", "60", "70", "80", "90", "100"),
    subject_colors: list[str] | None = None,
    roi_colors: dict[str, list[str]] | None = None,
    main_font_size: int = 8,
    sub_font_size: int = 6,
):
    subject_colors = subject_colors or SUBJECT_COLORS
    roi_colors = roi_colors or roi_subject_colors(subject_colors)
    metric = "identification accuracy"
    fig_width = max(0.9, 0.6 * len(rois)) * len(stimulus_sets) / 1.7
    fig, axes = plt.subplots(1, len(stimulus_sets), figsize=(fig_width, 1.2))
    if len(stimulus_sets) == 1:
        axes = [axes]

    x_positions = [index * 0.1 for index in range(len(rois))]
    x_pad = 0.05
    for stimulus_set_index, stimulus_set in enumerate(stimulus_sets):
        ax = axes[stimulus_set_index]
        ax.set_xlim([x_positions[0] - x_pad, x_positions[-1] + x_pad])
        ax.set_ylim([min(y_ticks), max(y_ticks)])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(rois, fontsize=main_font_size, rotation=45, ha="right", va="top")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels, fontsize=sub_font_size)
        for y in y_ticks:
            ax.axhline(y, color="lightgray", linewidth=0.2, zorder=0)
        if stimulus_set_index == 0:
            ax.set_ylabel("Identification accuracy (%)", fontsize=main_font_size)
        ax.axhline(1 / (num_lures + 1), color="black", linewidth=0.5, linestyle="--")

        for (roi_index, roi), (subject_index, subject) in product(
            enumerate(rois),
            enumerate(subjects),
        ):
            values = np.asarray(
                result_data[stimulus_set][metric]["raw"][roi][subject],
                dtype=float,
            )
            ci = result_data[stimulus_set][metric]["ci"][roi][subject]
            x = roi_index * 0.1 + (subject_index - len(subjects) // 2) * 0.013
            palette = roi_colors.get(roi, subject_colors)
            color = palette[subject_index % len(palette)]
            mean_value = float(np.mean(values))
            yerr = np.array(
                [
                    [max(0.0, mean_value - float(ci[0]))],
                    [max(0.0, float(ci[1]) - mean_value)],
                ]
            )
            ax.errorbar(
                x,
                mean_value,
                yerr=yerr,
                fmt="none",
                ecolor="gray",
                elinewidth=0.6,
                capsize=0,
                zorder=2,
            )
            ax.plot(x, mean_value, marker="o", color=color, markersize=2.5, zorder=4)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig


def plot_stimulus_set_identification_accuracy(
    result_data: dict,
    stimulus_sets: list[str],
    subjects: list[str],
    roi: str,
    *,
    num_lures: int = 1,
    y_ticks: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    y_ticklabels: tuple[str, ...] = ("40", "50", "60", "70", "80", "90", "100"),
    stimulus_set_labels: tuple[str, ...] | list[str] | None = None,
    subject_colors: list[str] | None = None,
    stimulus_set_spacing: float = 0.6,
    group_spacing: float = 0.3,
    group_spacing_after_idx: int | None = 1,
    subject_spacing: float = 0.075,
    main_font_size: int = 8,
    sub_font_size: int = 7,
):
    subject_colors = subject_colors or SUBJECT_COLORS
    stimulus_set_labels = stimulus_set_labels or stimulus_sets
    if len(stimulus_set_labels) != len(stimulus_sets):
        raise ValueError(
            "stimulus_set_labels must have the same length as stimulus_sets."
        )
    metric = "identification accuracy"

    x_stimulus_set_pos = []
    current_x = 0.0
    for stimulus_set_index in range(len(stimulus_sets)):
        x_stimulus_set_pos.append(current_x)
        current_x += stimulus_set_spacing
        if group_spacing_after_idx is not None and stimulus_set_index == group_spacing_after_idx:
            current_x += group_spacing

    fig_width = (x_stimulus_set_pos[-1] + stimulus_set_spacing + 0.5) / 1.5
    fig, ax = plt.subplots(figsize=(fig_width, 1.5))
    for y_value in y_ticks:
        ax.axhline(y_value, color="lightgray", linewidth=0.2)

    center_offset = subject_spacing * (len(subjects) - 1) / 2.0
    for stimulus_set_index, stimulus_set in enumerate(stimulus_sets):
        mean_dict = result_data[stimulus_set][metric]["mean"][roi]
        ci_dict = result_data[stimulus_set][metric]["ci"][roi]
        for subject_index, subject in enumerate(subjects):
            x = (
                x_stimulus_set_pos[stimulus_set_index]
                + subject_index * subject_spacing
                - center_offset
            )
            y = mean_dict[subject]
            y_low, y_high = ci_dict[subject]
            color = subject_colors[subject_index % len(subject_colors)]
            ax.errorbar(
                x,
                y,
                yerr=[[y - y_low], [y_high - y]],
                fmt="none",
                ecolor="gray",
                elinewidth=0.5,
                capsize=0,
            )
            ax.plot(x, y, marker="o", markersize=3, color=color)

    ax.set_ylim(min(y_ticks), max(y_ticks))
    ax.set_xticks(x_stimulus_set_pos)
    ax.set_xticklabels(
        stimulus_set_labels,
        fontsize=sub_font_size,
        rotation=0,
        ha="center",
        va="top",
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=sub_font_size)
    ax.set_ylabel("Identification accuracy (%)", fontsize=main_font_size)
    ax.axhline(1 / (num_lures + 1), linestyle="--", color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_xlim(-0.5, x_stimulus_set_pos[-1] + stimulus_set_spacing - 0.3)
    return fig


def collect_paired_stimulus_set_values(results_pair: dict) -> dict:
    metric = "identification accuracy"
    stimulus_set_a, stimulus_set_b = list(results_pair.keys())
    data_a = results_pair[stimulus_set_a][metric]["raw"]
    data_b = results_pair[stimulus_set_b][metric]["raw"]
    rois = sorted(set(data_a) & set(data_b))
    subjects = sorted(
        set.intersection(*(set(data_a[roi]) & set(data_b[roi]) for roi in rois))
    )
    results = {}

    for roi in rois:
        results[roi] = {"subjects": {}}
        for subject in subjects:
            vals_a = np.asarray(data_a[roi][subject], dtype=np.float64)
            vals_b = np.asarray(data_b[roi][subject], dtype=np.float64)
            if vals_a.shape != vals_b.shape:
                raise ValueError(f"Shape mismatch for {roi}-{subject}.")
            results[roi]["subjects"][subject] = {
                "vals_a": vals_a.tolist(),
                "vals_b": vals_b.tolist(),
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
            }
    return results


def plot_paired_stimulus_set_comparison(
    paired_results: dict,
    *,
    subjects: list[str],
    rois: list[str],
    stimulus_set_label_a: str = "RDS",
    stimulus_set_label_b: str = "rendered",
    figsize: tuple[float, float] = (4, 2),
    roi_spacing: float = 0.9,
    condition_gap: float = 0.35,
    subject_jitter: float = 0.04,
    subject_colors: list[str] | None = None,
    roi_colors: dict[str, list[str]] | None = None,
    y_ticks: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    y_ticklabels: tuple[str, ...] = ("40", "50", "60", "70", "80", "90", "100"),
    main_font_size: int = 8,
    sub_font_size: int = 6,
):
    subject_colors = subject_colors or SUBJECT_COLORS
    roi_colors = roi_colors or roi_subject_colors(subject_colors)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    base_x = np.arange(len(rois), dtype=float) * roi_spacing
    xticks = []
    xticklabels = []

    for roi_index, roi in enumerate(rois):
        roi_data = paired_results.get(roi, {}).get("subjects", {})
        center = base_x[roi_index]
        x_a_center = center - condition_gap / 2.0
        x_b_center = center + condition_gap / 2.0
        xticks.extend([x_a_center, x_b_center])
        xticklabels.extend(
            [f"{roi}\n{stimulus_set_label_a}", f"{roi}\n{stimulus_set_label_b}"]
        )
        palette = roi_colors.get(roi, subject_colors)

        for subject_index, subject in enumerate(subjects):
            result = roi_data.get(subject)
            if result is None:
                continue
            vals_a = np.asarray(result["vals_a"], dtype=float)
            vals_b = np.asarray(result["vals_b"], dtype=float)
            offset = (subject_index - (len(subjects) - 1) / 2.0) * subject_jitter
            color = palette[subject_index % len(palette)]

            for x, values in (
                (x_a_center + offset, vals_a),
                (x_b_center + offset, vals_b),
            ):
                mean = float(values.mean())
                ci_low, ci_high = calc_ci(values)
                err_low = max(0.0, mean - float(ci_low))
                err_high = max(0.0, float(ci_high) - mean)
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[err_low], [err_high]],
                    fmt="none",
                    ecolor="0.5",
                    elinewidth=0.5,
                    capsize=0,
                )
                ax.plot(x, mean, marker="o", color=color, markersize=2.5, linewidth=0)

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xticklabels,
        fontsize=sub_font_size,
        rotation=45,
        ha="right",
        va="top",
    )
    ax.set_xlim(base_x[0] - roi_spacing / 2.0, base_x[-1] + roi_spacing / 2.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=sub_font_size)
    for y in y_ticks:
        ax.axhline(y, color="lightgray", linewidth=0.2, zorder=0)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Identification accuracy (%)", fontsize=main_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(axis="y", labelsize=sub_font_size)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()

"""Plot feature-decoding evaluation metrics for WholeVC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from recon3d.config import (
    ANALYSIS_OUTPUT_ROOT,
    load_yaml_config,
    resolve_path,
    visualization_output_dir,
)
from recon3d.evaluation.feature_decoding import (
    FEATURE_DECODING_STIMULUS_SETS,
    compute_feature_decoding_metric,
)
from recon3d.metadata import SUBJECT_COLORS, WHOLE_VISUAL_ROI
from recon3d.subjects import PUBLIC_SUBJECTS


METRIC_CONFIGS = {
    "profile correlation": {
        "y_ticks": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        "y_ticklabels": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        "output_file": "feature_decoding_profile_correlation.svg",
    },
    "identification accuracy": {
        "y_ticks": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        "y_ticklabels": (40, 50, 60, 70, 80, 90, 100),
        "output_file": "feature_decoding_identification_accuracy_2-way.svg",
    },
}

DEFAULT_STIMULUS_SET_LABELS = (
    "natural\nwithin",
    "natural\nout-of",
    "artificial\nrendered",
    "artificial\nRDS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/atlasnet.yaml"))
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_OUTPUT_ROOT / "visualization")
    return parser.parse_args()


def plot_one_metric(
    metric: str,
    *,
    config: dict,
    results_root: Path,
    output_dir: Path,
) -> None:
    metric_config = METRIC_CONFIGS[metric]
    results = compute_feature_decoding_metric(
        results_root,
        config["representation"]["name"],
        FEATURE_DECODING_STIMULUS_SETS,
        metric=metric,
    )
    fig = plot_feature_decoding_metric(
        results,
        FEATURE_DECODING_STIMULUS_SETS,
        metric=metric,
        stimulus_set_labels=DEFAULT_STIMULUS_SET_LABELS,
        y_ticks=metric_config["y_ticks"],
        y_ticklabels=metric_config["y_ticklabels"],
    )
    save_path = output_dir / metric_config["output_file"]
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_decoding_metric(
    result_data: dict,
    stimulus_sets: list[str],
    *,
    metric: str,
    subjects: list[str] = PUBLIC_SUBJECTS,
    rois: list[str] | None = None,
    roi: str = WHOLE_VISUAL_ROI,
    y_ticks: tuple[float, ...],
    y_ticklabels: tuple[float | int, ...],
    stimulus_set_labels: tuple[str, ...] | list[str] | None = None,
    subject_colors: list[str] | None = None,
    stimulus_set_spacing: float = 0.6,
    subject_spacing: float = 0.075,
    group_spacing: float = 0.3,
    group_spacing_after_idx: int | None = 1,
    main_font_size: int = 8,
    sub_font_size: int = 7,
):
    rois = rois or [WHOLE_VISUAL_ROI]
    subject_colors = subject_colors or SUBJECT_COLORS
    stimulus_set_labels = stimulus_set_labels or stimulus_sets
    if len(stimulus_set_labels) != len(stimulus_sets):
        raise ValueError(
            "stimulus_set_labels must have the same length as stimulus_sets."
        )
    roi_index = rois.index(roi)

    x_stimulus_set_pos = []
    current_x = 0.0
    for stimulus_set_index in range(len(stimulus_sets)):
        x_stimulus_set_pos.append(current_x)
        current_x += stimulus_set_spacing
        if group_spacing_after_idx is not None and stimulus_set_index == group_spacing_after_idx:
            current_x += group_spacing

    fig_width = (x_stimulus_set_pos[-1] + stimulus_set_spacing + 0.5) / 1.5
    fig, ax = plt.subplots(figsize=(fig_width, 1.0))
    for y_value in y_ticks:
        ax.axhline(y_value, color="lightgray", linewidth=0.2)

    center_offset = subject_spacing * (len(subjects) - 1) / 2.0
    for stimulus_set_index, stimulus_set in enumerate(stimulus_sets):
        mean = result_data[stimulus_set]["mean"][roi_index, :]
        ci = result_data[stimulus_set]["ci"][roi_index, :]

        for subject_index, _subject in enumerate(subjects):
            x = (
                x_stimulus_set_pos[stimulus_set_index]
                + subject_index * subject_spacing
                - center_offset
            )
            y = mean[subject_index]
            y_low, y_high = ci[subject_index]
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

    chance_level = 0.5 if metric == "identification accuracy" else 0.0
    ax.axhline(chance_level, linestyle="--", color="black", linewidth=0.5)
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
    ax.set_ylabel(metric.capitalize(), fontsize=main_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, x_stimulus_set_pos[-1] + stimulus_set_spacing - 0.3)
    return fig


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    feature_config = config["feature_decoding"]
    results_root = resolve_path(feature_config["results_dir"], base=Path.cwd())

    output_dir = visualization_output_dir(args.output_dir, config, __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRIC_CONFIGS:
        plot_one_metric(
            metric,
            config=config,
            results_root=results_root,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()

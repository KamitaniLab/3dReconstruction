"""Plot forest plots for contour-matched slant LMM slopes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recon3d.config import (
    ANALYSIS_OUTPUT_ROOT,
    load_yaml_config,
    visualization_output_dir,
)
from recon3d.plotting.contour_slant_lmm import (
    ROI_COLORS,
    ROI_LABELS,
    STIMULUS_LABELS,
    STIMULUS_ORDER,
    SUBROI_ORDER,
    configure_matplotlib,
    normalize_slope_columns,
    read_required_csv,
)


FOREST_YLIM = (-0.5, 2.1)


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
        "--stimulus",
        choices=STIMULUS_ORDER,
        action="append",
        help="Stimulus to plot. May be passed multiple times. Defaults to all.",
    )
    return parser.parse_args()


def plot_forest(slopes: pd.DataFrame, output_dir: Path, stimuli: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slopes = normalize_slope_columns(slopes)
    required = {"roi", "stimulus", "model_group", "slope", "ci_low", "ci_high"}
    missing = required - set(slopes.columns)
    if missing:
        raise ValueError(f"Slope table is missing required columns: {sorted(missing)}")

    df = slopes[
        (slopes["model_group"].astype(str) == "subroi")
        & slopes["roi"].astype(str).isin(SUBROI_ORDER)
    ].copy()
    for stimulus in stimuli:
        d = df[df["stimulus"].astype(str) == stimulus].copy()
        if d.empty:
            print(f"[WARN] No rows found for stimulus={stimulus}")
            continue
        d["roi"] = pd.Categorical(d["roi"], categories=SUBROI_ORDER, ordered=True)
        d = d.sort_values("roi").reset_index(drop=True)

        x = np.arange(len(d))
        y = pd.to_numeric(d["slope"], errors="coerce").to_numpy()
        lo = pd.to_numeric(d["ci_low"], errors="coerce").to_numpy()
        hi = pd.to_numeric(d["ci_high"], errors="coerce").to_numpy()

        fig, ax = plt.subplots(figsize=(1.68, 2.75))
        for idx, row in d.iterrows():
            roi = str(row["roi"])
            color = ROI_COLORS.get(roi, "#4C4C4C")
            yerr = np.array([[y[idx] - lo[idx]], [hi[idx] - y[idx]]])
            ax.errorbar(
                x[idx],
                y[idx],
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.13,
                capsize=0,
                markersize=3.15,
            )
            ax.text(
                x[idx],
                hi[idx] + 0.08,
                f"{y[idx]:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
            print(f"{stimulus} - {roi}: slope = {y[idx]:.4f}, CI = [{lo[idx]:.4f}, {hi[idx]:.4f}]")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [ROI_LABELS.get(str(roi), str(roi)) for roi in d["roi"]],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("Slope", fontsize=8)
        ax.set_title(STIMULUS_LABELS.get(stimulus, stimulus), fontsize=8, y=1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.set_xlim(-0.5, len(d) - 0.5)
        ax.set_ylim(*FOREST_YLIM)
        fig.tight_layout()

        save_path = output_dir / f"contour_slant_lmm_forest_{stimulus}.svg"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        plt.close(fig)
        print(f"Saved: {save_path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    config = load_yaml_config(args.config)
    representation_name = config["representation"]["name"]
    statistics_dir = args.statistics_dir or (
        ANALYSIS_OUTPUT_ROOT
        / "statistics"
        / "contour_matched_lmm"
        / representation_name
        / "frequentist"
    )
    output_dir = visualization_output_dir(args.output_dir, config, __file__)
    slopes = read_required_csv(statistics_dir / "slope_by_stimulus" / "slopes_all.csv")
    plot_forest(slopes, output_dir, args.stimulus or STIMULUS_ORDER)


if __name__ == "__main__":
    main()

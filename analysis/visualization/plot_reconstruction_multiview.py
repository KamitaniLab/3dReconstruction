"""Plot qualitative multi-view reconstruction examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv
from PIL import Image

from recon3d.config import (
    ANALYSIS_OUTPUT_ROOT,
    REPO_ROOT,
    load_yaml_config,
    visualization_output_dir,
)
from recon3d.metadata import (
    QUALITATIVE_EXAMPLE_SUBJECT,
    WHOLE_VISUAL_ROI,
    natural_object_split_label,
    public_stimulus_label,
)
from recon3d.plotting.visualize_objects import (
    MeshVisualizer,
    PointCloudVisualizer,
    rotated_camera_position,
    save_as_svg,
)


SUBJECT = QUALITATIVE_EXAMPLE_SUBJECT
ROI = WHOLE_VISUAL_ROI
TRAIN_DATASET = "train-3d-natural-objects_rep3"
IMAGE_SIZE = (224, 224)
VIEW_ANGLES = (0.0, 120.0, 240.0)
POINT_SIZE = 5
POINT_COLOR = (0.6, 0.6, 0.6)
LIGHTING = "default"
DIFFUSION_POINT_CMAP = LinearSegmentedColormap.from_list(
    "gray_soft",
    ["#CCCCCC", "#262626"],
)

FIGURE_SPECS = [
    ("test-3d-natural-objects_rep8", (1, 7, 30, 59), "natural"),
    ("test-3d-artificial-objects-image_rep8", (6, 13), "artificial-image"),
    ("test-3d-artificial-objects-rds_rep8", (6, 13), "artificial-rds"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/atlasnet.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    representation = config["representation"]
    visualization_object_type = representation["visualization_object_type"]
    output_dir = visualization_output_dir(
        ANALYSIS_OUTPUT_ROOT / "visualization",
        config,
        __file__,
    )

    try:
        pv.start_xvfb()
    except OSError:
        pass

    for dataset, stimulus_indices, figure_name in FIGURE_SPECS:
        reconstructed_objects = load_objects(
            decoded_object_dir(config, dataset),
            visualization_object_type=visualization_object_type,
        )
        true_pointclouds = (
            load_true_pointclouds(dataset)
            if visualization_object_type == "pointcloud"
            else {}
        )
        for stimulus_index in stimulus_indices:
            canvas = make_multiview_figure(
                reconstructed_objects,
                true_pointclouds,
                dataset=dataset,
                stimulus_index=stimulus_index,
                visualization_object_type=visualization_object_type,
            )
            path = output_dir / output_name(
                representation_name=representation["name"],
                dataset=dataset,
                figure_name=figure_name,
                stimulus_index=stimulus_index,
            )
            save_as_svg(canvas, path)
            print(f"Saved: {path}")


def make_multiview_figure(
    reconstructed_objects: dict[str, Any],
    true_pointclouds: dict[str, Any],
    *,
    dataset: str,
    stimulus_index: int,
    visualization_object_type: str,
) -> Image.Image:
    labels = sorted(reconstructed_objects)
    stimulus_label = labels[stimulus_index]
    reconstructed_object = reconstructed_objects[stimulus_label]
    true_pointcloud = true_pointclouds.get(stimulus_label)

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1] * len(VIEW_ANGLES)),
        "white",
    )

    for row, view_angle in enumerate(VIEW_ANGLES):
        y = IMAGE_SIZE[1] * row

        if visualization_object_type == "pointcloud" and true_pointcloud is not None:
            true_image = render_object(
                true_pointcloud,
                visualization_object_type,
                rotated_camera_position(view_angle),
            )
        else:
            true_image = load_true_rendered_image(dataset, stimulus_label, view_angle)
        if true_image is not None:
            canvas.paste(true_image, (0, y))

        reconstructed_image = render_object(
            reconstructed_object,
            visualization_object_type,
            rotated_camera_position(view_angle),
        )
        canvas.paste(reconstructed_image, (IMAGE_SIZE[0], y))

    return canvas


def load_true_rendered_image(dataset: str, label: str, view_angle: float) -> Image.Image | None:
    angle_name = f"angle{int(view_angle)}"
    subdir = stimulus_dataset_name(dataset)
    image_path = REPO_ROOT / "data" / "rendered" / "true" / subdir / f"{label}_{angle_name}.png"
    if not image_path.exists():
        return None
    return Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)


def load_true_pointclouds(dataset: str) -> dict[str, Any]:
    pointcloud_dir = REPO_ROOT / "data" / "pointcloud" / stimulus_dataset_name(dataset)
    if not pointcloud_dir.exists():
        return {}
    return load_objects(pointcloud_dir, visualization_object_type="pointcloud")


def render_object(
    obj: Any,
    visualization_object_type: str,
    cpos: list[tuple[float, float, float]],
) -> Image.Image:
    if visualization_object_type == "mesh":
        return MeshVisualizer.render(
            obj,
            cpos=cpos,
            image_size=IMAGE_SIZE,
            off_screen=True,
            lighting=LIGHTING,
        )
    if visualization_object_type == "pointcloud":
        return PointCloudVisualizer.render(
            obj,
            cpos=cpos,
            image_size=IMAGE_SIZE,
            off_screen=True,
            point_size=POINT_SIZE,
            point_color=POINT_COLOR,
            lighting=LIGHTING,
            z_color=True,
            z_cmap=DIFFUSION_POINT_CMAP,
        )
    raise ValueError(f"Unknown visualization_object_type: {visualization_object_type}")


def load_objects(
    object_dir: Path,
    visualization_object_type: str,
) -> dict[str, Any]:
    if not object_dir.exists():
        raise FileNotFoundError(f"Object directory not found: {object_dir}")

    if visualization_object_type == "mesh":
        return {
            public_stimulus_label(path.name): MeshVisualizer.load(path)
            for path in sorted(object_dir.glob("*.ply"))
        }
    if visualization_object_type == "pointcloud":
        return {
            public_stimulus_label(path.name): PointCloudVisualizer.load(path)
            for path in sorted(object_dir.glob("*.npy"))
        }
    raise ValueError(f"Unknown visualization_object_type: {visualization_object_type}")


def decoded_object_dir(config: dict[str, Any], dataset: str) -> Path:
    representation = config["representation"]
    run_name = (
        f"{TRAIN_DATASET}_test-"
        f"{dataset.removeprefix('test-')}_fmap_{config['decoding']['option']}"
    )
    return (
        REPO_ROOT
        / "data"
        / "reconstruction"
        / representation["reconstruction_name"]
        / "decoded"
        / run_name
        / SUBJECT
        / ROI
        / representation["visualization_object_type"]
    )


def dataset_name(dataset: str) -> str:
    return dataset.removesuffix("_rep8")


def stimulus_dataset_name(dataset: str) -> str:
    name = dataset_name(dataset)
    if name in {
        "test-3d-artificial-objects-image",
        "test-3d-artificial-objects-rds",
    }:
        return "test-3d-artificial-objects"
    return name


def output_name(
    *,
    representation_name: str,
    dataset: str,
    figure_name: str,
    stimulus_index: int,
) -> str:
    stem = f"reconstruction_{representation_name}_{figure_name}_multiview"
    if dataset == "test-3d-natural-objects_rep8":
        stem = (
            f"reconstruction_{representation_name}_{figure_name}_"
            f"{natural_object_split_label(stimulus_index)}_multiview"
        )
    return (
        f"{stem}_sub-{SUBJECT}_roi-{ROI}_"
        f"stim-{dataset}-{stimulus_index:02d}.svg"
    )


if __name__ == "__main__":
    main()

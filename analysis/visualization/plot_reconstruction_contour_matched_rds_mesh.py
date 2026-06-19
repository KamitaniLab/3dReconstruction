"""Visualize contour-matched RDS meshes with brain reconstructions."""

from __future__ import annotations

import argparse
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageDraw

from recon3d.config import ANALYSIS_OUTPUT_ROOT, REPO_ROOT
from recon3d.contour_slant import HORIZONTAL_SHAPE_DATASET, reconstruction_run_name
from recon3d.metadata import (
    CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT,
    QUALITATIVE_EXAMPLE_SUBJECT,
    WHOLE_VISUAL_ROI,
)
from recon3d.plotting.visualize_objects import save_as_svg

DEFAULT_SUBJECTS: list[str] = [QUALITATIVE_EXAMPLE_SUBJECT]
DEFAULT_ROIS: list[str] = [WHOLE_VISUAL_ROI]
STIMULUS = "horizontal_thin_bar"
STIMULUS_LABEL = "bar-h"
SHAPE_SET = "horizontal-shape-variants"

_DATA_ROOT = REPO_ROOT / "data"

STIM_MESH_DIR = _DATA_ROOT / "mesh" / HORIZONTAL_SHAPE_DATASET
STIM_PC_DIR = _DATA_ROOT / "pointcloud" / HORIZONTAL_SHAPE_DATASET

_EXP = reconstruction_run_name(HORIZONTAL_SHAPE_DATASET)
RECON_BASE_DIR = _DATA_ROOT / "reconstruction" / "atlasnet_encoder_bn5" / "decoded" / _EXP

OUTPUT_DIR = ANALYSIS_OUTPUT_ROOT / "visualization" / "atlasnet" / "contour_matched_rds"

TRIALS = CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT[STIMULUS]
NOMINAL_ANGLES = [float(trial["nominal_deg"]) for trial in TRIALS]
STIMULUS_INDEX = [int(trial["stimulus_index"]) for trial in TRIALS]
TRUE_ANGLES = [float(trial["exp_true_deg"]) for trial in TRIALS]

PANEL_SIZE = (300, 300)


@dataclass
class ShapeData:
    mesh:         Any
    pointcloud:   np.ndarray | None = None
    eigen_vector: np.ndarray | None = None
    eigen_value:  float | None      = None


def calculate_first_pc(pc: np.ndarray):
    centered = pc - np.mean(pc, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx      = np.argsort(eigenvalues)[::-1]
    first_pc = eigenvectors[:, idx[0]]
    first_ev = eigenvalues[idx[0]]
    evr      = first_ev / np.sum(eigenvalues)
    return first_pc, first_ev, evr


def _load_mesh_pv(path: Path) -> Any:
    ms = trimesh.load_mesh(str(path))
    mesh_list = ms.dump() if "Scene" in type(ms).__name__ else [ms]  # type: ignore[attr-defined]
    if len(mesh_list) > 1:
        verts   = np.vstack([m.vertices for m in mesh_list])
        offsets = np.insert(np.cumsum([m.vertices.shape[0] for m in mesh_list]), 0, 0)[:-1]
        faces   = np.vstack([m.faces + o for m, o in zip(mesh_list, offsets)])
        mesh_list = [trimesh.Trimesh(verts, faces)]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "mesh.ply"
        mesh_list[0].export(tmp)  # type: ignore[attr-defined]
        return pv.read(tmp)


def _find_pointcloud_by_id(pc_dir: Path, shape_id: str, *, prefix: str = "") -> Path | None:
    if not pc_dir.exists():
        return None
    return next(pc_dir.glob(f"{prefix}{shape_id}-*.npy"), None)


def load_shape_db(subjects: list[str], rois: list[str]) -> dict:
    """Build shape_db keyed by condition tuples."""
    shape_db: dict = {}

    true_shapes: dict[str, ShapeData] = {}
    for obj in sorted(STIM_MESH_DIR.glob(f"*{STIMULUS_LABEL}*.obj")):
        shape_id = re.match(r"^(\d+)", obj.stem).group(1)
        mesh     = _load_mesh_pv(obj)

        pc_file = _find_pointcloud_by_id(STIM_PC_DIR, shape_id)
        pc      = np.load(pc_file).squeeze() if pc_file is not None else np.array(mesh.points)

        pc_vec, ev, _ = calculate_first_pc(pc)
        true_shapes[shape_id] = ShapeData(
            mesh=mesh, pointcloud=pc,
            eigen_vector=pc_vec.real, eigen_value=float(ev.real),
        )

    shape_db[("true", SHAPE_SET)] = dict(sorted(true_shapes.items()))

    for sub in subjects:
        for roi in rois:
            mesh_dir = RECON_BASE_DIR / sub / roi / "mesh"
            pc_dir   = RECON_BASE_DIR / sub / roi / "pointcloud"

            if not mesh_dir.exists():
                print(f"[warn] not found: {mesh_dir}")
                continue

            recon_shapes: dict[str, ShapeData] = {}
            for ply in sorted(mesh_dir.glob(f"*{STIMULUS_LABEL}*.ply")):
                shape_id = re.match(r"rds_(\d+)", ply.stem).group(1)
                mesh     = _load_mesh_pv(ply)

                pc_file = _find_pointcloud_by_id(pc_dir, shape_id, prefix="rds_")
                pc      = np.load(pc_file).squeeze() if pc_file is not None else np.array(mesh.points)

                pc_vec, ev, _ = calculate_first_pc(pc)
                recon_shapes[shape_id] = ShapeData(
                    mesh=mesh, pointcloud=pc,
                    eigen_vector=pc_vec.real, eigen_value=float(ev.real),
                )

            shape_db[("recon", SHAPE_SET, sub, roi)] = dict(sorted(recon_shapes.items()))

    return shape_db


def render_mesh_image(
    mesh: Any,
    cpos: list,
    image_size: tuple = PANEL_SIZE,
    parallel: bool = True,
    scale: float = 1.4,
    off_screen: bool = True,
) -> Image.Image:
    pl = pv.Plotter(off_screen=off_screen)
    pl.camera.SetParallelProjection(parallel)
    pl.camera.parallel_scale = scale
    pl.set_background("white")
    pl.add_mesh(mesh, style="surface", color="gray", show_edges=False)
    _, img_arr = pl.show(
        window_size=image_size, cpos=cpos,
        screenshot=True,
        return_cpos=True, return_img=True,
    )
    return Image.fromarray(img_arr)


class ReconstructionVisualizerRotated:
    """Render horizontal contour-matched stimuli and reconstructions."""

    def __init__(self, shape_db: dict):
        self._shape_db           = shape_db
        self.stimulus_index:     list  = []
        self.conditions:         list  = []
        self._panel_size:        tuple = PANEL_SIZE
        self._condition_dir:     str   = "col"
        self.cpos                      = [(0, 5, 0), (0, 0, 0), (-1, 0, 0)]
        self.proj_h_index:       int   = 2
        self.proj_v_index:       int   = 0
        self.proj_h_dir:         float = -1.0
        self.proj_v_dir:         float = -1.0
        self.nominal_angles_deg: list  = []
        self.true_angles_deg:    list  = []

    def _rotated_up(self, alpha_rad: float) -> np.ndarray:
        return np.array([-np.cos(alpha_rad), 0.0, -np.sin(alpha_rad)])

    def draw(self) -> Image.Image:
        rot_size = (self._panel_size[1], self._panel_size[0])
        canvas   = self._make_canvas(rot_size)

        for i, condition in enumerate(self.conditions):
            shapes = self._shape_db[condition]

            for j, stim_idx in enumerate(self.stimulus_index):
                shape_id = list(shapes.keys())[stim_idx]
                shape    = shapes[shape_id]

                if self.nominal_angles_deg and self.true_angles_deg:
                    alpha      = np.deg2rad(self.nominal_angles_deg[j] - self.true_angles_deg[j])
                    local_cpos = [self.cpos[0], self.cpos[1], tuple(self._rotated_up(alpha))]
                else:
                    local_cpos = self.cpos

                img = render_mesh_image(shape.mesh, cpos=local_cpos,
                                        image_size=self._panel_size,
                                        parallel=True, scale=1.4, off_screen=True)

                pa_rot = (self.nominal_angles_deg[j] - self.true_angles_deg[j]) \
                         if (self.nominal_angles_deg and self.true_angles_deg) else 0

                if condition[0] == "true":
                    pa_img = self._eigv_to_image(shape.eigen_vector, color="black", broken=True)
                    pa_img = pa_img.rotate(pa_rot, expand=False)
                else:
                    pa_img = self._eigv_to_image(shape.eigen_vector, color="red")
                img = Image.alpha_composite(img.convert("RGBA"), pa_img)

                if condition[0] == "recon":
                    stim_set   = condition[1]
                    true_shape = self._shape_db[("true", stim_set)][shape_id]
                    pa_img     = self._eigv_to_image(true_shape.eigen_vector, color="black", broken=True)
                    pa_img     = pa_img.rotate(pa_rot, expand=False)
                    img = Image.alpha_composite(img.convert("RGBA"), pa_img)

                img = img.rotate(-90, expand=True)

                x = i * rot_size[0] if self._condition_dir == "col" else j * rot_size[0]
                y = j * rot_size[1] if self._condition_dir == "col" else i * rot_size[1]
                canvas.paste(img, (x, y))

        return canvas

    def _make_canvas(self, panel_size: tuple) -> Image.Image:
        if self._condition_dir == "col":
            w = len(self.conditions)     * panel_size[0]
            h = len(self.stimulus_index) * panel_size[1]
        else:
            w = len(self.stimulus_index) * panel_size[0]
            h = len(self.conditions)     * panel_size[1]
        return Image.new("RGB", (w, h), (255, 255, 255))

    def _eigv_to_image(
        self, eigv: np.ndarray,
        image_size: tuple = PANEL_SIZE,
        color: str = "black", width: int = 6, length: int = 300, broken: bool = False,
    ) -> Image.Image:
        img  = Image.new("RGBA", image_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        gt = np.array([self.proj_h_dir * eigv[self.proj_h_index],
                       self.proj_v_dir * eigv[self.proj_v_index]])
        gt = gt / np.linalg.norm(gt)
        cx, cy = image_size[0] / 2, image_size[1] / 2
        x1, y1 = cx + length / 2 * gt[0], cy - length / 2 * gt[1]
        x2, y2 = cx - length / 2 * gt[0], cy + length / 2 * gt[1]
        if not broken:
            draw.line((x1, y1, x2, y2), fill=color, width=width)
        else:
            total = math.hypot(x2 - x1, y2 - y1)
            dx, dy = (x2 - x1) / total, (y2 - y1) / total
            cur, dash, gap = 0, 10, 10
            while cur < total:
                s = (x1 + cur * dx,                    y1 + cur * dy)
                e = (x1 + dx * min(cur + dash, total), y1 + dy * min(cur + dash, total))
                draw.line((s, e), fill=color, width=6)
                cur += dash + gap
        return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = args.subjects
    rois = args.rois

    pv.start_xvfb()

    print("Loading shape data...")
    shape_db = load_shape_db(subjects, rois)
    print("Done loading.")

    reconvis = ReconstructionVisualizerRotated(shape_db)
    reconvis.stimulus_index     = STIMULUS_INDEX
    reconvis.nominal_angles_deg = NOMINAL_ANGLES
    reconvis.true_angles_deg    = TRUE_ANGLES
    reconvis.conditions = [
        ("true", SHAPE_SET),
        *[("recon", SHAPE_SET, sub, roi) for sub in subjects for roi in rois],
    ]
    reconvis.cpos         = [(0, 5, 0), (0, 0, 0), (-1, 0, 0)]
    reconvis.proj_h_index = 2
    reconvis.proj_v_index = 0
    reconvis.proj_h_dir   = -1.0
    reconvis.proj_v_dir   = -1.0

    image = reconvis.draw()

    subs_tag = "-".join(subjects)
    rois_tag = "-".join(rois)
    out = OUTPUT_DIR / f"contour_matched_rds_bar-h_sub-{subs_tag}_roi-{rois_tag}.svg"
    save_as_svg(image, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

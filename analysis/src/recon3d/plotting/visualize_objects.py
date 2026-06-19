"""Utilities for visualizing 3D mesh and point-cloud objects."""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image


class MeshVisualizer:
    """Load and render mesh objects."""

    @staticmethod
    def load(mesh_file: str | Path) -> Any:
        """Load a mesh file and convert it to a PyVista object."""
        mesh_or_scene = trimesh.load_mesh(mesh_file)

        if "Scene" in str(type(mesh_or_scene)):
            mesh_list = mesh_or_scene.dump()  # type: ignore
        else:
            mesh_list = [mesh_or_scene]

        if len(mesh_list) > 1:
            vertices = [mesh.vertices for mesh in mesh_list]  # type: ignore
            faces = [mesh.faces for mesh in mesh_list]  # type: ignore
            faces_offset = np.cumsum([vertex.shape[0] for vertex in vertices])
            faces_offset = np.insert(faces_offset, 0, 0)[:-1]
            vertices = np.vstack(vertices)
            faces = np.vstack([face + offset for face, offset in zip(faces, faces_offset)])
            mesh_list = [trimesh.Trimesh(vertices, faces)]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "mesh.ply"
            mesh_list[0].export(tmp_file)  # type: ignore
            return pv.read(tmp_file)

    @staticmethod
    def render(
        mesh: Any,
        initial_rot: tuple[float, float, float] | None = None,
        additional_rot: tuple[float, float, float] | None = None,
        color: str = "gray",
        style: str = "surface",
        cpos: list[tuple[float, float, float]] | None = None,
        lighting: str = "default",
        image_size: tuple[int, int] = (500, 500),
        parallel: bool = False,
        scale: float = 1,
        off_screen: bool = False,
        background_color: str | tuple[float, float, float] = "white",
        transparent_background: bool = False,
    ) -> Image.Image:
        """Render a mesh as a PIL image."""
        if cpos is None:
            cpos = [(5, 0, 0), (0, 0, 0), (0, 1, 0)]

        plotter = _new_plotter(lighting=lighting, off_screen=off_screen)
        plotter.camera.SetParallelProjection(parallel)
        plotter.camera.parallel_scale = scale
        _add_light(plotter, lighting)

        pv.global_theme.transparent_background = transparent_background
        if not transparent_background:
            plotter.set_background(background_color)  # type: ignore

        if initial_rot is not None:
            mesh = mesh.rotate_x(initial_rot[0], inplace=False)
            mesh = mesh.rotate_y(initial_rot[1], inplace=False)
            mesh = mesh.rotate_z(initial_rot[2], inplace=False)

        if additional_rot is not None:
            mesh = mesh.rotate_x(additional_rot[0], inplace=False)
            mesh = mesh.rotate_y(additional_rot[1], inplace=False)
            mesh = mesh.rotate_z(additional_rot[2], inplace=False)

        plotter.add_mesh(mesh, style=style, color=color, show_edges=False)
        return _show(plotter, cpos, image_size)


class PointCloudVisualizer:
    """Load and render point-cloud objects."""

    @staticmethod
    def load(pointcloud_file: str | Path) -> np.ndarray:
        """Load a point cloud from an .npy file as an (N, 3) array."""
        pointcloud = np.load(pointcloud_file).squeeze()
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
            pointcloud = np.asarray(pointcloud).reshape(-1, 3)
        return pointcloud.astype(np.float64)

    @staticmethod
    def render(
        pointcloud: np.ndarray,
        cpos: list[tuple[float, float, float]] | None = None,
        image_size: tuple[int, int] = (500, 500),
        off_screen: bool = False,
        point_size: int = 3,
        point_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
        lighting: str = "default",
        z_color: bool = False,
        z_cmap: str = "plasma",
        transparent_background: bool = False,
    ) -> Image.Image:
        """Render a point cloud as a PIL image."""
        if cpos is None:
            cpos = [(5, 0, 0), (0, 0, 0), (0, 1, 0)]

        plotter = _new_plotter(lighting=lighting, off_screen=off_screen)
        pv.global_theme.transparent_background = transparent_background
        if not transparent_background:
            plotter.set_background("white")  # type: ignore

        scalars = pointcloud[:, 2] if z_color else None
        color = None if z_color else point_color
        plotter.add_points(
            pointcloud,
            color=color,
            scalars=scalars,
            cmap=z_cmap if z_color else None,
            style="points",
            point_size=point_size,
            show_edges=False,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        return _show(plotter, cpos, image_size)


def save_as_svg(image: Image.Image, path: str | Path) -> None:
    """Save a raster PIL image embedded in an SVG wrapper."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    width, height = image.size
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image x="0" y="0" width="{width}" height="{height}" '
        f'xlink:href="data:image/png;base64,{png_b64}"/>\n'
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def camera_info(param: list[float]) -> tuple[float, float, float]:
    """Calculate a camera position from the old figure-script parameter format."""
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])
    cam_y = param[3] * np.sin(phi) * param[6]
    temp = param[3] * np.cos(phi) * param[6]
    cam_x = temp * np.cos(theta)
    cam_z = temp * np.sin(theta)
    return cam_x, cam_y, cam_z


def rotated_camera_position(
    view_angle: float,
    *,
    camera_position: tuple[float, float, float] = (4, 0, 0),
    focal_point: tuple[float, float, float] = (0, 0, 0),
    view_up: tuple[float, float, float] = (0, 1, 0),
    rotation_axis: tuple[float, float, float] = (0, 1, 0),
) -> list[tuple[float, float, float]]:
    """Return a PyVista camera position rotated around the focal point."""
    rotation = trimesh.transformations.rotation_matrix(
        np.deg2rad(view_angle),
        rotation_axis,
        focal_point,
    )
    camera_position_h = np.append(np.asarray(camera_position, dtype=float), 1.0)
    rotated_position = np.dot(rotation, camera_position_h)[:3]
    return [
        tuple(float(x) for x in rotated_position.ravel()[:3]),
        tuple(float(x) for x in focal_point),
        tuple(float(x) for x in view_up),
    ]


def _new_plotter(lighting: str, off_screen: bool) -> pv.Plotter:
    if lighting == "default":
        return pv.Plotter(off_screen=off_screen)
    if lighting == "three lights":
        return pv.Plotter(lighting="three lights", off_screen=off_screen)
    return pv.Plotter(lighting="none", off_screen=off_screen)


def _add_light(plotter: pv.Plotter, lighting: str) -> None:
    if lighting == "headlight":
        plotter.add_light(pv.Light(light_type="headlight"))
    elif lighting == "ceiling light":
        plotter.add_light(pv.Light(light_type="scene light", position=(0, 3, 0)))
    elif lighting == "camera light":
        plotter.add_light(pv.Light(light_type="camera light", position=(0, 3, 0)))


def _show(
    plotter: pv.Plotter,
    cpos: list[tuple[float, float, float]],
    image_size: tuple[int, int],
) -> Image.Image:
    _, image_array = plotter.show(
        window_size=image_size,
        cpos=cpos,
        screenshot=True,
        return_cpos=True,
        return_img=True,
    )
    return Image.fromarray(image_array)

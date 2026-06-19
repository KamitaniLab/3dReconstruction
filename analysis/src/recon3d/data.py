from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import trimesh

from recon3d.plotting.visualize_objects import MeshVisualizer


@dataclass
class ShapeData:
    pointcloud_file:   Path | str
    mesh_file:         Path | str

    pointcloud: np.ndarray      = field(init=False)
    mesh:       trimesh.Trimesh = field(init=False)

    eigen_vector:   np.ndarray | None = None
    eigen_value:    float      | None = None
    principle_axis: np.ndarray | None = None

    def __post_init__(self):
        self.pointcloud = self.__load_pointcloud(self.pointcloud_file)
        self.mesh       = self.__load_mesh(self.mesh_file)

    def __load_mesh(self, file):
        return MeshVisualizer.load(file)

    def __load_pointcloud(self, file):
        return np.load(file).squeeze()

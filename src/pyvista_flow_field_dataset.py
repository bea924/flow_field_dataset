import os
import pyvista as pv
from typing import Literal
import numpy as np

VolumeFieldType = Literal["Velocity", "Pressure", "Temperature"]


# TODO: Put the real ones here
SurfaceFieldType = Literal["Pressure", "Temperature", "WallShearStressMagnitude", "WallShearStress_0", "WallShearStress_1", "WallShearStress_2"]


class PyvistaSample:
    def __init__(self, volume_path: str, surface_path: str):
        self.volume_path = volume_path
        self.surface_path = surface_path
        self._volume_data: pv.MultiBlock | None = None
        self._surface_data: pv.MultiBlock | None = None

    @property
    def surface_data(self):
        if self._surface_data is None:
            self._surface_data = pv.read(self.surface_path)
        # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._surface_data

    @property
    def volume_data(self):
        if self._volume_data is None:
            self._volume_data = pv.read(self.volume_path)
        # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._volume_data

    def plot_surface(self, field: SurfaceFieldType):
        self.surface_data.plot(scalars=field)

    def plot_volume(self, field: VolumeFieldType):
        self.volume_data[0][0][0].plot(scalars=field, opacity=0.7)

    def get_points(self) -> np.ndarray:
        """
        Returns the points of the dataset as a numpy array.

        Returns:
        --------
        np.ndarray: The points of the dataset. Shape: (n_points, 3)
        """
        # TODO check if same works for non cgns
        block = self.volume_data[0][0]
        return block.cell_centers().points


    def get_surface_points(self, block_index: int) -> np.ndarray:
        """
        Returns the points of the surface dataset as a numpy array.

        Returns:
        --------
        np.ndarray: The points of the surface dataset. Shape: (n_points, 3)
        """
        block = self.surface_data[0][block_index]
        return block.points
    

    def get_labeled_surface_points(self) -> np.ndarray:
        """
        Returns the surface points of the dataset with their block index as a numpy array.

        Returns:
        --------
        np.ndarray: The labeled points of the dataset. Shape: (n_points, 4)
        """
        labeled_points = []
        for i, block in enumerate(self.surface_data[0]):
            labeled_points.append(np.hstack((block.points, np.full((block.n_points, 1), i))))
        return np.vstack(labeled_points)

    def load(self):
        self.volume_data
        self.surface_data

    def unload(self):
        # TODO: Test if this frees the memory
        self._volume_data = None
        self._surface_data = None
    
    def compute_aggregate_force(self, block_index: int) -> np.ndarray:
        """
        Compute the aggregate force acting on the surface of the block with the given index. This is done by integrating the force acting on each point of the surface.

        Parameters:
        -----------
        block_index: int
            The index of the block to compute the aggregate force for.

        Returns:
        --------
        np.ndarray: The aggregate force acting on the surface of the block.
        """
        block = self.surface_data[0][block_index]
        raise NotImplementedError("Implement this method")


class PyvistaFlowFieldDataset:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        possible_extensions = [
            ".ply",
            ".vtp",
            ".stl",
            ".vtk",
            ".geo",
            ".obj",
            ".iv",
            ".vtu",
            ".cgns",
        ]
        files = os.listdir(self.data_dir)
        files = [f for f in files if any(f.endswith(ext) for ext in possible_extensions)]
        surface_files = [f for f in files if "surface" in f]
        volume_files = [f for f in files if "volume" in f]
        surface_files=sorted(surface_files)
        volume_files=sorted(volume_files)
        self.samples = [PyvistaSample(os.path.join(data_dir,v), os.path.join(data_dir,s)) for v, s in zip(volume_files, surface_files)]
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

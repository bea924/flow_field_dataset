import os
import pyvista as pv
from typing import Literal
import numpy as np

from huggingface_hub import list_repo_files, hf_hub_download
import shutil
import os

VolumeFieldType = Literal["Velocity", "Pressure", "Temperature"]


SurfaceFieldType = Literal["Pressure", "Temperature", "WallShearStressMagnitude", "WallShearStress_0", "WallShearStress_1", "WallShearStress_2", "CellArea", "Normal_0","Normal_1", "Normal_2"]


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
        block = self.volume_data[0][0][0]
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

    def get_bounding_box(self):
        """
        Returns the bounding box of the volume data.
        The bounding box is a six-tuple (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        return self.volume_data.bounds


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
    
    @classmethod
    def load_from_huggingface(cls,path: str="datasets/ds_huggingface", hub_repo: str="peteole/CoolMucSmall", num_samples=3)->"PyvistaFlowFieldDataset":
        """
        Download all files from the specified Hugging Face repository to a given local path and load them as a PyvistaFlowFieldDataset.

        Args:
            path (str): The local directory where the data will be saved.
            hub_repo (str): The Hugging Face repository ID (e.g., 'bert-base-uncased').
        
        Returns:
            PyvistaFlowFieldDataset
        """
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            # Get the list of files in the repository
            files = [file for file in list_repo_files(hub_repo,repo_type="dataset") if file.endswith('.cgns')]
        except Exception as e:
            raise ValueError(f"Error getting the list of files in repository '{hub_repo}': {e}")
        volume_files = [f for f in files if "volume" in f]
        surface_files = [f for f in files if "surface" in f]
        volume_files=sorted(volume_files)
        surface_files=sorted(surface_files)
        assert len(volume_files)==len(surface_files), "Number of volume and surface files must be equal"
        total_samples = len(volume_files)
        print(f"Found {total_samples} files in repository '{hub_repo}'.")
        num_samples = min(num_samples, total_samples)
        surface_files = surface_files[:num_samples]
        volume_files = volume_files[:num_samples]
        print(f"Downloading {len(files)} files from repository '{hub_repo}' to '{path}'.")

        files_to_download = volume_files + surface_files
        for i, file in enumerate(files_to_download, start=1):
            try:
                #Test if file exists
                if os.path.exists(os.path.join(path, file)):
                    print(f"File {i}/{len(files_to_download)}: {file} already exists. Skipping download.")
                    continue
                # Download each file and copy it to the target path
                print(f"Downloading file {i}/{len(files_to_download)}: {file}")
                hf_hub_download(repo_id=hub_repo, filename=file, local_dir=path, repo_type="dataset")
            except Exception as e:
                print(f"Error downloading file '{file}': {e}")

        print(f"All files have been downloaded to '{path}'.")
        return cls(path)
    def get_bounds(self):
        """
        Returns the bounding box of the volume data.
        The bounding box is a six-tuple (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        bounds = (np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf)
        for sample in self.samples:
            sample_bounds = sample.get_bounding_box()
            bounds = (
                min(bounds[0], sample_bounds[0]),
                max(bounds[1], sample_bounds[1]),
                min(bounds[2], sample_bounds[2]),
                max(bounds[3], sample_bounds[3]),
                min(bounds[4], sample_bounds[4]),
                max(bounds[5], sample_bounds[5]),
            )
        return bounds
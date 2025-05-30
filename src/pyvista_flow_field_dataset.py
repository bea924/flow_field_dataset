import os
from pathlib import Path
import pyvista as pv
from typing import Literal
import numpy as np

import huggingface_hub as hf #type: ignore
import shutil

VolumeFieldType = Literal["Velocity", "Pressure", "Temperature"]


SurfaceFieldType = Literal["Pressure", "Temperature", "WallShearStressMagnitude", "WallShearStress_0", "WallShearStress_1", "WallShearStress_2", "CellArea", "Normal_0","Normal_1", "Normal_2"]


class PyvistaSample:
    def __init__(self, volume_path: str | Path, surface_path: str | Path):
        self.volume_path = Path(volume_path)
        self.surface_path = Path(surface_path)
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

    def get_bounding_box(self):
        """
        Returns the bounding box of the volume data.
        The bounding box is a six-tuple (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        return self.volume_data.bounds


class PyvistaFlowFieldDataset:
    def __init__(self, samples: list[PyvistaSample]):
        self.samples = samples
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
    
    @classmethod
    def try_from_directory(cls, data_dir: str | Path, num_samples: int) -> "None | PyvistaFlowFieldDataset":
        data_dir = os.path.abspath(data_dir)
        data_dir = Path(data_dir)
        volume_dir = data_dir / "volume"
        surface_dir = data_dir / "surface"
        os.makedirs(volume_dir, exist_ok=True)
        os.makedirs(surface_dir, exist_ok=True)
        volume_files = list(volume_dir.glob("*.cgns"))
        surface_files = list(surface_dir.glob("*.cgns"))
        volume_indices = [int(f.stem.split("_")[-1]) for f in volume_files]
        surface_indices = [int(f.stem.split("_")[-1]) for f in surface_files]
        volume_indices.sort()
        surface_indices.sort()
        if volume_indices != surface_indices:
            return None
        volume_files =sorted(volume_files, key=lambda x: int(x.stem.split("_")[-1]))
        surface_files =sorted(surface_files, key=lambda x: int(x.stem.split("_")[-1]))
        samples = [PyvistaSample(v, s) for v, s in zip(volume_files, surface_files)]
        if len(samples) < num_samples:
            return None
        samples = samples[:num_samples]
        return cls(samples)
    @classmethod
    def load_from_huggingface(cls,data_dir: str | Path, num_samples=3)->"PyvistaFlowFieldDataset":
        """
        Download all files from the specified Hugging Face repository to a given local path and load them as a PyvistaFlowFieldDataset.

        Args:
            path (str): The local directory where the data will be saved.
            hub_repo (str): The Hugging Face repository ID (e.g., 'bert-base-uncased').
        
        Returns:
            PyvistaFlowFieldDataset
        """
        loaded = cls.try_from_directory(data_dir, num_samples)
        if loaded is not None:
            print(f"Loaded {len(loaded)} samples from '{data_dir}'.")
            return loaded
        
        data_dir = os.path.abspath(data_dir)
        data_dir = Path(data_dir)
        volume_dir = data_dir / "volume"
        surface_dir = data_dir / "surface"
        tmp_dir = data_dir / "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        
        # remove existing files
        if os.path.exists(volume_dir):
            shutil.rmtree(volume_dir)
        if os.path.exists(surface_dir):
            shutil.rmtree(surface_dir)
        os.makedirs(volume_dir, exist_ok=True)
        os.makedirs(surface_dir, exist_ok=True)
        
        repo_id = "datasets/bgce/cooldata"
        fs = hf.HfFileSystem()
        runs =fs.glob(f"{repo_id}/production_run*", detail=False)
        samples: list[PyvistaSample] = []
        runs = sorted(runs, key=lambda x: int(x.split("/")[-1].removeprefix("production_run")))
        for run in runs:
            run_name = os.path.basename(run)
            zip_files_in_run = list(fs.glob(f"{run}/batch_*.zip", detail=False))
            zip_files_in_run = sorted(zip_files_in_run, key=lambda x: int(x.split("/")[-1].removeprefix("batch_").removesuffix(".zip")))
            for zip_file in zip_files_in_run:
                local_path = os.path.join(tmp_dir,run_name, os.path.basename(zip_file))
                fs.download(zip_file, local_path)
                # Extract the zip file
                unzip_dir = os.path.join(tmp_dir, run_name, os.path.basename(zip_file).removesuffix(".zip"))
                os.makedirs(unzip_dir, exist_ok=True)
                shutil.unpack_archive(local_path, unzip_dir)
                # Match indices
                volume_files = list(Path(unzip_dir).glob("volume_design_*.cgns"))
                surface_files = list(Path(unzip_dir).glob("surface_design_*.cgns"))
                volume_indices = [int(f.stem.split("_")[-1]) for f in volume_files]
                surface_indices = [int(f.stem.split("_")[-1]) for f in surface_files]
                volume_indices.sort()
                surface_indices.sort()
                if volume_indices != surface_indices:
                    print(f"Skipping {run_name} due to mismatched indices.")
                    continue
                volume_files = sorted(volume_files, key=lambda x: int(x.stem.split("_")[-1]))
                surface_files = sorted(surface_files, key=lambda x: int(x.stem.split("_")[-1]))
                for v, s in zip(volume_files, surface_files):
                    # Copy the files to the new directory
                    shutil.copy(v, volume_dir)
                    shutil.copy(s, surface_dir)
                    moved_volume_path = volume_dir / v.name
                    moved_surface_path = surface_dir / s.name
                    samples.append(PyvistaSample(moved_volume_path, moved_surface_path))
                    if len(samples) >= num_samples:
                        shutil.rmtree(tmp_dir)
                        return cls(samples)
        # Clean up temporary directory
        shutil.rmtree(tmp_dir)
                
        return cls(samples)
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
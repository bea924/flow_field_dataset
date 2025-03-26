from dataclasses import dataclass
import json
import os
from typing import Literal, Union
import torch
from tensordict import TensorDict
import numpy as np
import pyvista as pv

from src.pyvista_flow_field_dataset import PyvistaFlowFieldDataset, PyvistaSample

VoxelField = Literal["Pressure", "Temperature", "Velocity", "Position"]
voxel_fields: list[VoxelField] = ["Pressure", "Temperature", "Velocity", "Position"]

Normalization = dict[VoxelField, Union[tuple[list[float], list[float]],tuple[float,float]]]
"""
A dictionary that maps field names to tuples of (mean, std) for normalization.
"""
class VoxelFlowFieldSample:
    def __init__(self, path: str, bounding_box: tuple[float, float, float, float, float, float], resolution: tuple[int, int, int], normalization: Normalization | None = None):
        """A sample of a voxelized flow field. The data is stored in a file in the TensorDict format on disk in an unnormalized form. A normalization can be applied dynamically when getting data.
        
        Args:
        - path: Path to the file containing the voxelized flow field.
        - bounding_box: Tuple (xmin, xmax, ymin, ymax, zmin, zmax) defining the bounding box of the flow field.
        - resolution: Tuple (nx, ny, nz) defining the resolution of the voxel grid.
        - normalization: A normalization that is applied dynamically when getting data. A dictionary that maps field names to tuples of (mean, std) for normalization. The
        """
        
        self.path = path
        self.bounding_box = bounding_box
        self.resolution = resolution
        self._data: TensorDict | None = None
        self.normalization = normalization


    @property
    def data(self):
        if self._data is None:
            self._data = TensorDict.load(self.path)
            # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._data
    
    def get_field(self, field: VoxelField) -> torch.Tensor:
        """
        Returns the specified field of the flow field as a torch.Tensor.
        Shape:
        - Pressure, Temperature: (resolution_x, resolution_y, resolution_z)
        - Velocity, Position: (resolution_x, resolution_y, resolution_z, 3)
        """
        if self.normalization is not None:
            mean, std = self.normalization[field]
            return (self.data[field] - torch.tensor(mean,device=self.data.device)) / torch.tensor(std,device=self.data.device)
        return self.data[field]
    
    @property
    def Y(self):
        return torch.cat([
            self.get_field("Velocity"),
            self.get_field("Pressure").unsqueeze(-1),
            self.get_field("Temperature").unsqueeze(-1)
        ],dim=-1)
    
    @property
    def mask(self):
        return self.data['Mask']
    
    @classmethod 
    def from_mask_y(cls,mask: torch.Tensor,Y:torch.Tensor, bounding_box: tuple[float, float, float, float, float, float], resolution: tuple[int, int, int],  save_path: str, normalization: Normalization | None = None) -> "VoxelFlowFieldSample":
        
        velocity = Y[:,:,:,:3]
        pressure = Y[:,:,:,3]
        temperature = Y[:,:,:,4]
        # get position from bounding box and resolution
        xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
        x = np.linspace(xmin, xmax, resolution[0])
        y = np.linspace(ymin, ymax, resolution[1])
        z = np.linspace(zmin, zmax, resolution[2])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        position = torch.tensor(np.stack([x,y,z],axis=-1),dtype=torch.float32)
        data = TensorDict({
            "Pressure": pressure,
            "Temperature": temperature,
            "Velocity": velocity,
            "Mask": mask,
            "Position": position
        })
        # denormalize the data
        if normalization is not None:
            for field in data:
                if field in normalization:
                    mean, std = normalization[field]
                    data[field] = data[field] * std + mean
        data.save(save_path)
        return cls(save_path, bounding_box, resolution, normalization)

    @classmethod
    def from_pyvista(
        cls,
        sample: PyvistaSample,
        save_path: str,
        resolution: tuple[int, int, int],
        bounding_box: tuple[float,float,float,float,float,float],
    ) -> "VoxelFlowFieldSample":
        """
        Interpolates the volume data from the sample to a voxel grid and saves it to a file.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
        x = np.linspace(xmin, xmax, resolution[0])
        y = np.linspace(ymin, ymax, resolution[1])
        z = np.linspace(zmin, zmax, resolution[2])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        grid = pv.StructuredGrid(x, y, z)
        volume_data = sample.volume_data[0][0][0]
        interpolated = grid.sample(volume_data)
        data = TensorDict({
            "Pressure": torch.tensor(interpolated["Pressure"].reshape(resolution,order="F"),dtype=torch.float32),
            "Temperature": torch.tensor(interpolated["Temperature"].reshape(resolution,order="F"),dtype=torch.float32),
            "Velocity": torch.tensor(interpolated["Velocity"].reshape(resolution + (3,),order="F"),dtype=torch.float32),
            "Mask": torch.tensor(interpolated["vtkValidPointMask"].reshape(resolution,order="F"),dtype=torch.bool),
            "Position": torch.tensor(np.stack([x,y,z],axis=-1).reshape(resolution + (3,),order="F"),dtype=torch.float32)
        })
        data.save(save_path)
        return cls(save_path, bounding_box, resolution)

    def load(self):
        self._data
        return self

    def unload(self):
        self._data = None
        return self

    def to_pyvista(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_box
        x, y, z = np.mgrid[
            xmin:xmax:complex(self.resolution[0]),
            ymin:ymax:complex(self.resolution[1]),
            zmin:zmax:complex(self.resolution[2]),
        ]
        grid = pv.StructuredGrid(x, y, z)
        grid["Pressure"] = self.data["Pressure"].numpy().flatten(order="F")
        grid["Temperature"] = self.data["Temperature"].numpy().flatten(order="F")
        grid["Velocity"] = self.data["Velocity"].numpy().reshape(-1, 3, order="F")
        grid["vtkValidPointMask"] = self.data["Mask"].numpy().flatten(order="F").astype(np.int8)
        grid['vtkGhostType'] = np.zeros(len(grid.points), dtype=np.uint8) 
        grid['vtkGhostType'][~ self.data["Mask"].numpy().flatten(order="F")]=32
        grid.cell_data['vtkGhostType'] = np.zeros(grid.n_cells, dtype=np.uint8)
        grid.cell_data['vtkGhostType'][~ self.data["Mask"].numpy()[1:,1:,1:].flatten(order="F")]= 32
        return grid

    def plot(self, field: VoxelField):
        grid = self.to_pyvista()
        grid.plot(scalars=field, cmap="viridis")


@dataclass
class VoxelFlowFieldDatasetConfig:
    """Configuration for creating a VoxelFlowFieldDataset from a PyvistaFlowFieldDataset."""
    pyvista_dataset: PyvistaFlowFieldDataset
    resolution: tuple[int, int, int] = (32, 32, 32)

class VoxelFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str,
        config: VoxelFlowFieldDatasetConfig | None = None,
    ):
        """
        Dataset of voxelized flow fields. The constructor either loads the dataset from a cache directory or converts a
        PyvistaFlowFieldDataset to a DGLFlowFieldDataset.
        """
        self.cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.samples: list[VoxelFlowFieldSample] = []
        if config is not None:
            # clear the cache directory
            for root, dirs, files in os.walk(self.cache_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            bounding_box = config.pyvista_dataset.get_bounds()
            for i in range(len(config.pyvista_dataset)):
                sample = config.pyvista_dataset[i]
                sample = VoxelFlowFieldSample.from_pyvista(
                    sample,
                    os.path.join(self.cache_dir, f"{i}.pt"),
                    config.resolution,
                    bounding_box,
                )
                self.samples.append(sample)
            self.normalization = self.compute_normalization()
            json.dump(
                {"resolution": config.resolution, "bounding_box": bounding_box, "normalization": self.normalization},
                open(os.path.join(self.cache_dir, "metadata.json"), "w"),
            )
        else:
            metadata = json.load(open(os.path.join(self.cache_dir, "metadata.json")))
            resolution = metadata["resolution"]
            bounding_box = metadata["bounding_box"]
            self.normalization = metadata["normalization"]
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pt"):
                    self.samples.append(
                        VoxelFlowFieldSample(os.path.join(self.cache_dir, file), bounding_box, resolution)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def compute_normalization(self) -> Normalization:
        normalization: Normalization = {}
        for field in voxel_fields:
            sample_means = []
            sample_stds = []
            for sample in self.samples:
                sample_means.append(sample.get_field(field).mean(dim=(0, 1, 2)))
                sample_stds.append(sample.get_field(field).std(dim=(0, 1, 2)))
            mean = torch.stack(sample_means).mean(dim=0).tolist()
            std = torch.stack(sample_stds).mean(dim=0).tolist()
            normalization[field] = (mean, std)
        return normalization
    
    def normalize(self):
        """Normalizes the dataset in place."""
        for sample in self.samples:
            sample.normalization = self.normalization
        return self
    
    def unnormalize(self):
        """Unnormalizes the dataset in place."""
        for sample in self.samples:
            sample.normalization = None
        return self

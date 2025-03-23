import json
import os
from typing import Literal
import torch
from tensordict import TensorDict
import numpy as np
import pyvista as pv

from src.pyvista_flow_field_dataset import PyvistaFlowFieldDataset, PyvistaSample

VoxelField = Literal["Pressure", "Temperature", "Velocity", "Mask", "Position"]


class VoxelFlowFieldSample:
    def __init__(self, path: str, bounding_box: tuple[float, float, float, float, float, float], resolution: tuple[int, int, int]):
        self.path = path
        self.bounding_box = bounding_box
        self.resolution = resolution
        self._data: TensorDict | None = None


    @property
    def data(self):
        if self._data is None:
            self._data = TensorDict.load(self.path)
            # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._data
    
    def get_field(self, field: VoxelField):
        """
        Returns the specified field of the flow field as a torch.Tensor.
        Shape:
        - Pressure, Temperature: (resolution_x, resolution_y, resolution_z)
        - Velocity, Position: (resolution_x, resolution_y, resolution_z, 3)
        """
        return self.data[field]
    
    @property
    def Y(self):
        return self.data['Y']
    
    @property
    def mask(self):
        return self.data['Mask']

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
        x, y, z = np.mgrid[
            xmin:xmax:complex(resolution[0]),
            ymin:ymax:complex(resolution[1]),
            zmin:zmax:complex(resolution[2]),
        ]
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
        data['Y'] = torch.cat([data['Velocity'],data['Pressure'].unsqueeze(-1),data['Temperature'].unsqueeze(-1)],dim=-1)
        data.save(save_path)
        return cls(save_path, bounding_box, resolution)

    def load(self):
        self._data
        return self

    def unload(self):
        self._data = None
        return self

    def plot(self, field: VoxelField):
        raise NotImplementedError("Implement this method")


class VoxelFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str,
        pyvista_dataset: PyvistaFlowFieldDataset | None = None,
        resolution: tuple[int, int, int] = (32, 32, 32),
    ):
        """
        Dataset of voxelized flow fields. The constructor either loads the dataset from a cache directory or converts a
        PyvistaFlowFieldDataset to a DGLFlowFieldDataset.
        """
        self.cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.samples: list[VoxelFlowFieldSample] = []
        if pyvista_dataset is not None:
            # clear the cache directory
            for root, dirs, files in os.walk(self.cache_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            bounding_box = pyvista_dataset.get_bounds()
            for i in range(len(pyvista_dataset)):
                sample = pyvista_dataset[i]
                sample = VoxelFlowFieldSample.from_pyvista(
                    sample,
                    os.path.join(self.cache_dir, f"{i}.pt"),
                    resolution,
                    bounding_box,
                )
                self.samples.append(sample)
            json.dump(
                {"resolution": resolution, "bounding_box": bounding_box},
                open(os.path.join(self.cache_dir, "metadata.json"), "w"),
            )
        else:
            metadata = json.load(open(os.path.join(self.cache_dir, "metadata.json")))
            resolution = metadata["resolution"]
            bounding_box = metadata["bounding_box"]
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pt"):
                    self.samples.append(
                        VoxelFlowFieldSample(os.path.join(self.cache_dir, file))
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

import os
from typing import Literal
import torch
from tensordict import TensorDict

from pyvista_flow_field_dataset import PyvistaFlowFieldDataset, PyvistaSample

VoxelField = Literal["velocity", "pressure", "temperature"]


class VoxelFlowFieldSample:
    def __init__(self, path: str):
        self.path = path
        self._data: TensorDict | None = None

    @property
    def data(self):
        if self._data is None:
            self._data = TensorDict.load(self.path)
            # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._data

    @property
    def points(self) -> torch.Tensor:
        return self.data["points"]

    @property
    def velocity(self) -> torch.Tensor:
        return self.data["velocity"]

    @property
    def pressure(self) -> torch.Tensor:
        return self.data["pressure"]

    @property
    def temperature(self) -> torch.Tensor:
        return self.data["temperature"]

    @property
    def mask(self) -> torch.Tensor:
        return self.data["mask"]

    @property
    def bounding_box(self):
        points = self.points
        return points.min(dim=0).values, points.max(dim=0).values

    @classmethod
    def from_pyvista(
        cls,
        sample: PyvistaSample,
        save_path: str,
        resolution: tuple[int, int, int],
        bounding_box: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
    ):
        """
        Interpolates the volume data from the sample to a voxel grid and saves it to a file.
        """
        raise NotImplementedError("Implement this method")

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
        bounding_box: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ] = ((0, 1), (0, 1), (0, 1)),
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
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            for i in range(len(pyvista_dataset)):
                sample = pyvista_dataset[i]
                sample = VoxelFlowFieldSample.from_pyvista(
                    sample,
                    os.path.join(self.cache_dir, f"{i}.pt"),
                    resolution,
                    bounding_box,
                )
                self.samples.append(sample)
        else:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pt"):
                    self.samples.append(
                        VoxelFlowFieldSample(os.path.join(self.cache_dir, file))
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

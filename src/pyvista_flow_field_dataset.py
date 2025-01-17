import os
import pyvista as pv
from typing import Iterable, Iterator, Literal
import numpy as np

VolumeFieldType = Literal["Velocity", "Pressure", "Temperature"]
SurfaceFieldType = Literal["pressure", "temperature", "shear_stress", "heat_flux"]

class PyvistaSample:
    def __init__(self, data: pv.MultiBlock):
        self.data = data
        #self.surface_data = data[0][0].extract_surface()
        # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
    
    def save(self, filename: str):
        """
        Save the dataset to a file. The file format is determined by the file extension. Supported formats include:
        '.ply', '.vtp', '.stl', '.vtk, '.geo', '.obj', '.iv'
        """
        self.data.save(filename)
        
    def plot_surface(self, field: SurfaceFieldType):
        self.data.plot(field)
    def plot_volume(self, field: VolumeFieldType):
        self.data[0][0].plot(field, opacity=0.5)
        
    def get_points(self)->np.ndarray:
        """
        Returns the points of the dataset as a numpy array.
        
        Returns:
        --------
        np.ndarray: The points of the dataset. Shape: (n_points, 3)
        """
        return self.data.points
    
    def get_surface_points(self)->np.ndarray:
        """
        Returns the points of the surface dataset as a numpy array.
        
        Returns:
        --------
        np.ndarray: The points of the surface dataset. Shape: (n_points, 3)
        """
        return self.surface_data.points
    
    
    @classmethod
    def from_file(cls, filename: str):
        data = pv.read(filename)
        # TODO not sure it works like this for pv.polydata objects as well as strucutresGrid
        return cls(data)
        

class PyvistaFlowFieldDataset:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        possible_extensions = [".ply", ".vtp", ".stl", ".vtk", ".geo", ".obj", ".iv", ".vtu",".cgns"]
        self.files = [f for f in os.listdir(self.data_dir) if os.path.splitext(f)[-1] in possible_extensions]
        self._index = 0  # Track iteration state
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int):
        return PyvistaSample.from_file(os.path.join(self.data_dir, self.files[idx]))
    
    
    
        
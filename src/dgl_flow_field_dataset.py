import os
import dgl
import torch
import pyvista_flow_field_dataset as pvffd
import pyvista as pv


class DGLVolumeFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str):
        """
        Parameters:
        -----------
        cache_dir: str
            The directory where the dataset converted to DGLGraphs will be stored.
        """
        self.cache_dir = cache_dir
        self.files = [f for f in os.listdir(self.cache_dir) if f.endswith(".dgl")]
        
        # TODO: Check if the cache_dir contains valid DGLGraphs and check each of the graphs.
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = os.path.join(self.cache_dir, self.files[idx])
        return dgl.load_graphs(filename)[0][0]
    
    @classmethod
    def pv_to_volume_dgl(cls,polydata: pv.PolyData) -> dgl.DGLGraph:
        """
        Convert a Pyvista PolyData object to a DGLGraph of the volume flow field.
        
        Parameters:
        -----------
        polydata: pv.PolyData
            The PolyData object to convert.
        
        Returns:
        --------
        DGLGraph: The converted graph.
        """
        raise NotImplementedError("Implement this method")
    
    def volume_dgl_to_pv(self, graph: dgl.DGLGraph) -> pv.PolyData:
        """
        Convert a DGLGraph of the volume flow field to a Pyvista PolyData object.
        
        Parameters:
        -----------
        graph: dgl.DGLGraph
            The DGLGraph to convert.
        
        Returns:
        --------
        pv.PolyData: The converted PolyData object.
        """
        raise NotImplementedError("Implement this method")
    
    @classmethod
    def from_pyvista_dataset(cls, pyvista_dataset: pvffd.PyvistaFlowFieldDataset, cache_dir: str):
        """
        Convert a PyvistaFlowFieldDataset to a DGLFlowFieldDataset.
        
        Parameters:
        -----------
        pyvista_dataset: pvffd.PyvistaFlowFieldDataset
            The PyvistaFlowFieldDataset to convert.
        cache_dir: str
            The directory where the dataset converted to DGLGraphs will be stored.
        
        Returns:
        --------
        DGLFlowFieldDataset: The converted dataset.
        """
        os.makedirs(cache_dir, exist_ok=True)
        for i in range(len(pyvista_dataset)):
            sample = pyvista_dataset[i]
            g = cls.pv_to_volume_dgl(sample.data)
            dgl.save_graphs(os.path.join(cache_dir, f"{i}.dgl"), g)
        
        return cls(cache_dir)

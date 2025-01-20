import os
import dgl
import torch
import pyvista_flow_field_dataset as pvffd
import pyvista as pv
from pyvista_flow_field_dataset import PyvistaFlowFieldDataset, PyvistaSample


class DGLVolumeFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self, cache_dir: str, pyvista_dataset: PyvistaFlowFieldDataset | None = None
    ):
        """
        Creates a new DGLVolumeFlowFieldDataset. If a PyvistaFlowFieldDataset is provided, it will be converted to DGLGraphs and stored in the cache directory. If not, the dataset will be loaded from the cache directory.
        Parameters:
        -----------
        cache_dir: str
            The directory where the dataset converted to DGLGraphs is stored.
        polydata: pv.PolyData
            The directory where the dataset converted to DGLGraphs is stored. Default None.
        """
        self.cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if pyvista_dataset is not None:
            # clear the cache directory
            for f in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, f))
            for i in range(len(pyvista_dataset)):
                sample = pyvista_dataset[i]
                g = self.pv_to_volume_dgl(sample.volume_data)
                dgl.save_graphs(os.path.join(self.cache_dir, f"{i}.dgl"), g)
        self.files = [f for f in os.listdir(self.cache_dir) if f.endswith(".dgl")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.cache_dir, self.files[idx])
        if filename.endswith(".dgl"):
            g = dgl.load_graphs(filename)[0][0]
            return self.normalize(g)
        else:
            gpv = PyvistaSample.from_file(filename)
            g = self.pv_to_volume_dgl(gpv.data)
            return self.normalize(g)

    @classmethod
    def pv_to_volume_dgl(cls, polydata: pv.PolyData) -> dgl.DGLGraph:
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
        # TODO do we need to convert explicitly from polydata? when will we need polydata specifically?
        raise NotImplementedError("Implement this method")

    @classmethod
    def pyvista_to_volume_dgl(cls, sample: PyvistaSample) -> dgl.DGLGraph:
        """
        Convert a Pyvista UnstructuredGrid object to a DGLGraph of the volume flow field.

        Parameters:
        -----------
        grid: pv.UnstructuredGrid
            The UnstructuredGrid object to convert.

        Returns:
        --------
        DGLGraph: The converted graph.
        """
        
        grid = sample.volume_data
        edges_from = []
        edges_to = []

        # TODO: Speed up this loop
        for i in range(grid.n_points):
            neighbors = grid.point_neighbors(i)
            edges_from.extend([i] * len(neighbors))
            edges_to.extend(neighbors)

        graph = dgl.graph((edges_from, edges_to), num_nodes=grid.n_points)
        # graph.ndata['velocity'] = torch.tensor(grid.point_data['Velocity'], dtype=torch.float32)
        # TODO add the attributes to the nodes from the grid
        return graph

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

    def normalize(cls, graph: dgl.DGLGraph):
        """
        Normalize the features of the graph.

        Parameters:
        -----------
        graph: dgl.DGLGraph
            The graph to normalize.
        """
        raise NotImplementedError("Implement this method")

    def denormalize(cls, graph: dgl.DGLGraph):
        """
        Denormalize the features of the graph.

        Parameters:
        -----------
        graph: dgl.DGLGraph
            The graph to denormalize.
        """
        raise NotImplementedError("Implement this method")

    @classmethod
    def l2_loss(cls, graph1: dgl.DGLGraph, graph2: dgl.DGLGraph):
        """
        Compute the L2 loss between two DGLGraphs.

        Parameters:
        -----------
        graph1: dgl.DGLGraph
            The first graph.
        graph2: dgl.DGLGraph
            The second graph.

        Returns:
        --------
        float: The L2 loss between the two graphs.
        """
        raise NotImplementedError("Implement this method")


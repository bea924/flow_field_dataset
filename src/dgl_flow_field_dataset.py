import os
import dgl
import torch
import pyvista_flow_field_dataset as pvffd
import pyvista as pv
from pyvista_flow_field_dataset import PyvistaFlowFieldDataset, PyvistaSample


class DGLVolumeFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, pyvista_dataset: PyvistaFlowFieldDataset = None):
        """
        Parameters:
        -----------
        cache_dir: str
            The directory where the dataset converted to DGLGraphs is stored.
        polydata: pv.PolyData
            The directory where the dataset converted to DGLGraphs is stored. Default None.
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.files = [f for f in os.listdir(self.cache_dir) if f.endswith(".dgl")]
        # TODO should we make the cache_dir optional if we provide pyvista_dataset

        # TODO: Check if the cache_dir contains valid DGLGraphs and check each of the graphs.
        if not self.files and pyvista_dataset == None:
            raise ValueError("No valid DGLGraphs found in the cache directory and no PyvistaFlowFieldDataset instance provided.")
        if pyvista_dataset:
            self.files.append(pyvista_dataset.files)

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
    def UnstructuredGrid_to_volume_dgl(cls, grid: pv.UnstructuredGrid) -> dgl.DGLGraph:
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
        edges_from = []
        edges_to = []

        for i in range(grid.n_points):
            neighbors = grid.point_neighbors(i)
            edges_from.extend([i]*len(neighbors))
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

    @classmethod
    def from_pyvista_dataset(
        cls, pyvista_dataset: pvffd.PyvistaFlowFieldDataset, cache_dir: str
    ):
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
            dgl.save_graphs(os.path.join(cache_dir, f"{i}.dgl"))

        return cls(cache_dir)

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


class DGLSurfaceFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str):
        """
        Parameters:
        -----------
        cache_dir: str
            The directory where the dataset converted to DGLGraphs will be stored.
        """
        self.cache_dir = cache_dir
        self.files = [f for f in os.listdir(self.cache_dir) if f.endswith(".dgl")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.cache_dir, self.files[idx])
        return dgl.load_graphs(filename)[0][0]

    @classmethod
    def pv_to_surface_dgl(cls, polydata: pv.PolyData) -> dgl.DGLGraph:
        """
        Convert a Pyvista PolyData object to a DGLGraph of the surface flow field.

        Parameters:
        -----------
        polydata: pv.PolyData
            The PolyData object to convert.

        Returns:
        --------
        DGLGraph: The converted graph.
        """
        raise NotImplementedError("Implement this method")

    def surface_dgl_to_pv(self, graph: dgl.DGLGraph) -> pv.PolyData:
        """
        Convert a DGLGraph of the surface flow field to a Pyvista PolyData object.

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
    def from_pyvista_dataset(
        cls, pyvista_dataset: pvffd.PyvistaFlowFieldDataset, cache_dir: str
    ):
        """
        Convert a PyvistaFlowFieldDataset to a DGLSurfaceFlowFieldDataset.

        Parameters:
        -----------
        pyvista_dataset: pvffd.PyvistaFlowFieldDataset
            The PyvistaFlowFieldDataset to convert.
        cache_dir: str
            The directory where the dataset converted to DGLGraphs will be stored.

        Returns:
        --------
        DGLSurfaceFlowFieldDataset: The converted dataset.
        """
        os.makedirs(cache_dir, exist_ok=True)
        for i in range(len(pyvista_dataset)):
            sample = pyvista_dataset[i]
            g = cls.pv_to_surface_dgl(sample.surface_data)
            dgl.save_graphs(os.path.join(cache_dir, f"{i}.dgl"), g)

        return cls(cache_dir)

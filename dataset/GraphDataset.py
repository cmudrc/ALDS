import os
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import torch
import numpy as np
import meshio
import scipy.io
import ctypes
import h5py
import shutil
import scipy.sparse as sp
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph


class GenericGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, partition=False, **kwargs):
        super(GenericGraphDataset, self).__init__(root, transform, pre_transform)
        self.raw_dir = os.path.join(root, 'raw')
        self.processed_dir = os.path.join(root, 'processed')
        # check if the raw data & processed data directories are empty
        if len(os.listdir(self.raw_dir)) == 0:
            raise RuntimeError('Raw data directory is empty. Please download the dataset first.')
        if len(os.listdir(self.processed_dir)) == 0:
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])
        if partition:
            self.sub_size = kwargs['sub_size']

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in range(1, 3):
            data = torch.load('data/processed/data_{}.pt'.format(i))
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def _download(self):
        pass

    def _process(self):
        pass

    def get_partition_domain(self, data, mode):
        """
        returns a full partitioned collection of subdomains of the original domain
        
        :param data: the original domain stored in a torch_geometric.data.Data object. geometry is stored in data.pos
        """
        # get domain geometry bounds
        x_min, x_max = data.pos[:, 0].min(), data.pos[:, 0].max()
        y_min, y_max = data.pos[:, 1].min(), data.pos[:, 1].max()
        z_min, z_max = data.pos[:, 2].min(), data.pos[:, 2].max()

        # divide the domain into subdomains according to self.sub_size
        subdomains = []
        for x in np.arange(x_min, x_max, self.sub_size):
            for y in np.arange(y_min, y_max, self.sub_size):
                for z in np.arange(z_min, z_max, self.sub_size):
                    # find nodes within the subdomain
                    mask = (data.pos[:, 0] >= x) & (data.pos[:, 0] < x + self.sub_size) & \
                           (data.pos[:, 1] >= y) & (data.pos[:, 1] < y + self.sub_size) & \
                           (data.pos[:, 2] >= z) & (data.pos[:, 2] < z + self.sub_size)
                    subdomain = subgraph(mask, data)

                    # add boundary information to the subdomain. boundary information is applied as vector on the boundary nodes
                    # indentify boundary nodes
                    boundary_mask = self.get_graph_boundary_edges(subdomain)
                    boundary_nodes = subdomain.edge_index[0][boundary_mask].unique()
                    boundary_nodes = torch.cat([boundary_nodes, subdomain.edge_index[1][boundary_mask].unique()])
                    boundary_nodes = boundary_nodes.unique()
                    boundary_nodes = boundary_nodes[boundary_nodes != -1]

                    # add boundary information to the subdomain
                    boundary_info = torch.zeros((boundary_nodes.size(0), 3))
                    # compute boundary vector
                    # get all edges connected to the boundary nodes
                    boundary_edges = subdomain.edge_index[:, boundary_mask]
                    # for every node on the boundary, compute Neumann boundary condition by averaging the 'x' property of the connected nodes
                    for i, node in enumerate(boundary_nodes):
                        connected_nodes = boundary_edges[1][boundary_edges[0] == node]

                        # compute magnitude & direction of the boundary vector
                        boundary_vector = data.pos[node] - data.pos[connected_nodes]
                        boundary_magnitude = data.x[node] - data.x[connected_nodes]
                        # compute Neumann boundary condition
                        boundary_info[i] = boundary_magnitude / boundary_vector.norm()

                    # add boundary information to the subdomain
                    subdomain.bc = boundary_info

                    subdomains.append(subdomain)

        return subdomains
    
    @staticmethod
    def get_graph_boundary_edges(data, dimension=3):
        """
        returns the boundary edges of a graph
        
        :param data: the graph stored in a torch_geometric.data.Data
        :param dimension: the defined geometrical dimension of the graph
        """
        # get adjacency matrix
        adj = pyg.utils.to_dense_adj(data.edge_index).squeeze()
        # get boundary edges as edgws with only one cell assignments
        boundary_edges = []
        boundary_edges = torch.where(adj.sum(dim=0) == 1)[0]

        return boundary_edges

        

    def reconstruct_from_partition(self, subdomains):
        """
        reconstructs the original domain from a partitioned collection of subdomains
        
        :param subdomains: a list of subdomains, each stored in a torch_geometric.data.Data object
        """
        # concatenate all subdomains
        data = Data()
        data.x = torch.cat([subdomain.x for subdomain in subdomains], dim=0)
        data.edge_index = torch.cat([subdomain.edge_index for subdomain in subdomains], dim=1)
        data.edge_attr = torch.cat([subdomain.edge_attr for subdomain in subdomains], dim=0)
        data.pos = torch.cat([subdomain.pos for subdomain in subdomains], dim=0)
        data.bc = torch.cat([subdomain.bc for subdomain in subdomains], dim=0)
        return data


class CoronaryArteryDataset(GenericGraphDataset):
    def __init__(self, root, transform=None, pre_transform=None, partition=False, **kwargs):
        super(CoronaryArteryDataset, self).__init__(root, transform, pre_transform, partition, **kwargs)
        self.raw_file_names = ["all_results_0{}0.vtu".format(i) for i in range(258, 345)]

    def download(self):
        pass

    def process(self):
        data = torch.load('data/processed/data_1.pt')
        data_list = self.get_partition_domain(data, 'train')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def _process_file(path):
        raw_solution = meshio.read(path)
        

    def _download(self):
        pass

    def _process(self):
        pass

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
import os
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import torch
import numpy as np
import meshio
import scipy.io
import ctypes
import h5py
import shutil
import multiprocessing as mp
from threading import Thread
import scipy.sparse as sp
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph


class GenericGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, partition=False, **kwargs):
        super(GenericGraphDataset, self).__init__(root, transform, pre_transform)
        # self.raw_dir = os.path.join(root, 'raw')
        # self.processed_dir = os.path.join(root, 'processed')
        # check if the raw data & processed data directories are empty
        if len(os.listdir(self.raw_dir)) == 0:
            raise RuntimeError('Raw data directory is empty. Please download the dataset first.')
        if not os.path.exists(self.processed_dir) or len(os.listdir(self.processed_dir)) == 0:
            print('Processing data...')
            self.process()

        self.data = torch.load(self.processed_paths[0])
        if partition:
            self.sub_size = kwargs['sub_size']
            self.data = self.get_partition_domain(self.data, 'train')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    # def process(self):
    #     data_list = []
    #     for i in range(1, 3):
    #         data = torch.load('data/processed/data_{}.pt'.format(i))
    #         data_list.append(data)
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])

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
        os.makedirs(os.path.join(self.root, 'partition'), exist_ok=True)
        if os.path.exists(os.path.join(self.root, 'partition', 'data.pt')):
            subdomains = torch.load(os.path.join(self.root, 'partition', 'data.pt'))
        else:
            if mode == 'train':
                num_processes = mp.cpu_count()
                len_single_process = max(len(data) // (num_processes - 1), 1)
                data_list = [(data[i * len_single_process:(i + 1) * len_single_process], self.sub_size) for i in range(0, len(data), len_single_process)]
                with mp.Pool(num_processes) as pool:
                    # self.data_test = self._get_partiton_domain(data_list[0])
                    subdomains = pool.map(self._get_partiton_domain, data_list)
                torch.save(subdomains, os.path.join(self.root, 'partition', 'data.pt'))
        return subdomains
    
    @staticmethod
    def _get_partiton_domain(data):
        """
        returns a full partitioned collection of subdomains of the original domain
        
        :param data: the original domain stored in a torch_geometric.data.Data
        :param sub_size: the size of the subdomains
        """
        data_batch, sub_size = data
        subdomains = []
        for data in data_batch:
            data = data[0]
        # get domain geometry bounds
            x_min, x_max = data.pos[:, 0].min(), data.pos[:, 0].max()
            y_min, y_max = data.pos[:, 1].min(), data.pos[:, 1].max()
            z_min, z_max = data.pos[:, 2].min(), data.pos[:, 2].max()
            # temporary fix to the device issue
            # data.edge_index = torch.Tensor(data.edge_index)

            # divide the domain into subdomains according to self.sub_size
            for x in np.arange(x_min, x_max, sub_size):
                for y in np.arange(y_min, y_max, sub_size):
                    for z in np.arange(z_min, z_max, sub_size):
                        # find nodes within the subdomain
                        mask = (data.pos[:, 0] >= x) & (data.pos[:, 0] < x + sub_size) & \
                            (data.pos[:, 1] >= y) & (data.pos[:, 1] < y + sub_size) & \
                            (data.pos[:, 2] >= z) & (data.pos[:, 2] < z + sub_size)
                        subdomain, _ = subgraph(mask, data.edge_index)

                        ########################## TBD: fix boundary information ##########################
                        '''
                        # add boundary information to the subdomain. boundary information is applied as vector on the boundary nodes
                        # indentify boundary nodes
                        boundary_mask = GenericGraphDataset.get_graph_boundary_edges(subdomain)
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
                        '''
                        ####################################################################################

                        subdomain = Data(x=data.x[mask], pos=data.pos[mask], edge_index=subdomain)
                        subdomains.append(subdomain)

        return subdomains
    
    def get_graph_boundary_edges(data, dimension=3):
        """
        returns the boundary edges of a graph
        
        :param data: the graph stored in a torch_geometric.data.Data
        :param dimension: the defined geometrical dimension of the graph
        """
        # get adjacency matrix
        adj = pyg.utils.to_dense_adj(data).squeeze()
        # get boundary edges as edgws with only one cell assignments
        boundary_edges = []
        boundary_edges = torch.where(adj.sum(dim=0) == 1)[0]

        return boundary_edges
    
    # @staticmethod
    # def get_graph_boundary_edges(data, dimension=3):
    #     """
    #     returns the boundary edges of a graph
        
    #     :param data: the graph stored in a torch_geometric.data.Data
    #     :param dimension: the defined geometrical dimension of the graph
    #     """
    #     # get adjacency matrix
    #     adj = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    #     # get boundary edges as edgws with only one cell assignments
    #     boundary_edges = []
    #     boundary_edges = torch.where(adj.sum(dim=0) == 1)[0]

    #     return boundary_edges

        

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
        # self.raw_file_names = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith('.vtu')]

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith('.vtu')]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        num_processes = mp.cpu_count()
        len_single_process = max(len(self.raw_file_names) // (num_processes - 1), 1)
        raw_data_list = [self.raw_file_names[i:i + len_single_process] for i in range(0, len(self.raw_file_names), len_single_process)]
        with mp.Pool(num_processes) as pool:
            # data_list_test = CoronaryArteryDataset._process_file(raw_data_list[0])
            data_list = pool.map(CoronaryArteryDataset._process_file, raw_data_list)
        # data, slices = self.collate(data_list)
        torch.save(data_list, self.processed_paths[0])

    @staticmethod
    def _process_file(path_list):
        data_list = []
        for path in path_list:
            raw_solution = meshio.read(path)
            # export mesh physics
            velocity = raw_solution.point_data['velocity']
            pressure = raw_solution.point_data['pressure']
            # physics_node_id = raw_solution.point_data['GlobalNodeID']

            # rearrange the physics data according to the node id
            # velocity = np.array([velocity[physics_node_id == i+1] for i in range(velocity.size)])
            # pressure = np.array([pressure[physics_node_id == i+1] for i in range(pressure.size)])

            # export mesh geometry
            pos = torch.tensor(raw_solution.points, dtype=torch.float)
            cells = raw_solution.cells_dict['tetra']
            edge_index = CoronaryArteryDataset._cell_to_connectivity(cells)

            # identify mesh wall nodes
            wall_node = CoronaryArteryDataset._get_boundary_nodes(raw_solution)
            wall_idx = torch.zeros(pos.size(0), dtype=torch.float)
            wall_idx[wall_node] = 1

            # create a torch_geometric.data.Data object
            data = Data(x=torch.cat([torch.tensor(velocity), torch.tensor(pressure).unsqueeze(1), wall_idx.unsqueeze(1)], dim=1), pos=pos, edge_index=torch.Tensor(edge_index))
            data_list.append(data)
        return data_list


    @staticmethod
    def _cell_to_connectivity(cells):
        all_edges = []
        for cell in cells:
            for i in range(cell.size):
                all_edges.append([cell[i], cell[(i + 1) % cell.size]])
                # add the reverse edge
                all_edges.append([cell[(i + 1) % cell.size], cell[i]])
            # all_edges.append(edges)

        return torch.tensor(all_edges).t().contiguous().view(2, -1)
    
    @staticmethod
    def _get_boundary_nodes(data):
        """
        returns the boundary nodes of a graph
        
        :param edge_index: the edge index of the graph
        :param num_nodes: the total number of nodes in the graph
        """
        # get boundary nodes as nodes with WSS but no velocity
        velocity = data.point_data['velocity']
        wss = data.point_data['vWSS']
        boundary_nodes = np.where((np.sum(velocity, axis=1) == 0) & (np.sum(wss, axis=1) != 0))[0]
        return boundary_nodes


    def _download(self):
        pass

    def _process(self):
        pass

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
import os
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import torch
import numpy as np
import meshio
import pandas as pd
# import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
# from threading import Thread
import scipy.sparse as sp
from operator import itemgetter
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
# from fenics import *
# from dolfin import *
# from fenicstools.Interpolation import interpolate_nonmatching_mesh
from scipy.spatial import Delaunay


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

        data = torch.load(self.processed_paths[0], map_location=torch.device('cpu'))
        if partition:
            self.sub_size = kwargs['sub_size']
            self.data = self.get_partition_domain(data, 'train')
        else:
            self.data = data

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
    

class DuctAnalysisDataset(GenericGraphDataset):
    def __init__(self, root, transform=None, pre_transform=None, partition=False, **kwargs):
        super(DuctAnalysisDataset, self).__init__(root, transform, pre_transform, partition, **kwargs)
        # self.raw_file_names = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith('.mat')]

    def download(self):
        pass

    def len(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]

    @property
    def raw_file_names(self):
        return ["Mesh_Output_High.msh", "Mesh_Output_Med.msh", "Mesh_Output_Low.msh", "Output_Summary_High_100", "Output_Summary_Med_100", "Output_Summary_Low_100", "Output_Summary_High_25", "Output_Summary_Med_25", "Output_Summary_Low_25"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        # num_processes = mp.cpu_count()
        # len_single_process = max(len(self.raw_file_names) // (num_processes - 1), 1)
        raw_data_list = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
        # raw_data_list = [raw_data_list[i:i + len_single_process] for i in range(0, len(raw_data_list), len_single_process)]
        # with mp.Pool(num_processes) as pool:
        #     # data_list_test = CoronaryArteryDataset._process_file(raw_data_list[0])
        #     data_list = pool.map(DuctAnalysisDataset._process_file, raw_data_list)
        # data, slices = self.collate(data_list)
        data_list = self._process_file(raw_data_list)
        torch.save(data_list, self.processed_paths[0])

    @staticmethod
    def _process_file(path_list):
        data_list = []
        # mesh_idx = ['High', 'Med', 'Low']
        # process mesh files
        for idx, path in enumerate(path_list[:2]):
            mesh = meshio.read(path)
            pos = torch.tensor(mesh.points, dtype=torch.float)
            cells = [mesh.cells_dict['quad'], mesh.cells_dict['triangle']]
            edge_index = DuctAnalysisDataset._cell_to_connectivity(cells)

            # edge attr is the length of the edge
            # edge_attr = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(1)
            
            # process physics files
            # print(path_list[idx+3])
            physics = pd.read_csv(path_list[idx+3], sep=',')
            # print(physics)
            velocity_x = torch.tensor(physics['      x-velocity'], dtype=torch.float).unsqueeze(1)
            velocity_y = torch.tensor(physics['      y-velocity'], dtype=torch.float).unsqueeze(1)
            velocity_z = torch.tensor(physics['      z-velocity'], dtype=torch.float).unsqueeze(1)
            velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=1)
            # normalize the velocity to be in the range of [0, 1]
            velocity = velocity / torch.max(velocity)

            pressure = torch.tensor(physics['        pressure'], dtype=torch.float).unsqueeze(1)
            # normalize the pressure
            pressure = pressure / torch.max(pressure)
            # check if nan exists in the physics data
            if torch.isnan(velocity).sum() > 0 or torch.isnan(pressure).sum() > 0:
                print('nan exists in original physics data')

            # identify mesh wall nodes as nodes that satisfy the following conditions:
            # 1. part of triangle cells
            # 2. has no velocity
            wall_idx = []
            for cell in mesh.cells_dict['triangle']:
                for node in cell:
                    if torch.sum(velocity[node]) == 0:
                        wall_idx.append(node)
            wall_idx = torch.tensor(wall_idx, dtype=torch.float)
            wall_idx = torch.unique(wall_idx)

            # create a torch_geometric.data.Data object if the mesh is of high resolution
            if idx == 0:
                data = Data(x=torch.tensor([0]), y=torch.cat([velocity, pressure], dim=1), pos=pos, edge_index=torch.Tensor(edge_index), wall_idx=wall_idx)
                data_list.append(data)
            else:
                # call lagrangian interpolation to interpolate the physics data to the high resolution mesh
                pos_high = data_list[0].pos
                pos_low = pos
                velocity_x_high = DuctAnalysisDataset._lagrangian_interpolation(pos_low, velocity_x, pos_high)
                velocity_y_high = DuctAnalysisDataset._lagrangian_interpolation(pos_low, velocity_y, pos_high)
                velocity_z_high = DuctAnalysisDataset._lagrangian_interpolation(pos_low, velocity_z, pos_high)
                pressure_high = DuctAnalysisDataset._lagrangian_interpolation(pos_low, pressure, pos_high)

                velocity_high = torch.cat([velocity_x_high, velocity_y_high, velocity_z_high], dim=1)
                velocity_high = torch.tensor(velocity_high, dtype=torch.float)
                # normalize the velocity
                velocity_high = velocity_high / torch.max(velocity_high)
                pressure_high = torch.tensor(pressure_high, dtype=torch.float)
                # normalize the pressure
                pressure_high = pressure_high / torch.max(pressure_high)
                # check if nan exists in the interpolated physics data
                if torch.isnan(velocity_high).sum() > 0 or torch.isnan(pressure_high).sum() > 0:
                    print('nan exists in interpolated physics data')

                data_list[0].x = torch.cat([velocity_high, pressure_high], dim=1)

        return data_list
    
    @staticmethod
    def _cell_to_connectivity(cells):
        all_edges = []
        for cell_group in cells:
            for cell in cell_group:
                for i in range(cell.size):
                    all_edges.append([cell[i], cell[(i + 1) % cell.size]])
                    # add the reverse edge
                    all_edges.append([cell[(i + 1) % cell.size], cell[i]])
            # all_edges.append(edges)
        return torch.tensor(all_edges).t().contiguous().view(2, -1)
    
    @staticmethod
    def _lagrangian_interpolation(points, physics, new_points):
        """
        Perform 1st-order Lagrangian interpolation of physics properties 
        at a new set of points based on provided 3D points and physics information.

        Args:
            points (np.ndarray): Array of shape (num_points, 3) representing the 3D positions of the points.
            physics (np.ndarray): Array of shape (num_points, 1) representing the physics information at the points.
            new_points (np.ndarray): Array of shape (num_new_points, 3) representing the new points for interpolation.

        Returns:
            np.ndarray: Interpolated physics values at the new points of shape (num_new_points, 1).
        """
        # Ensure correct input shapes
        points = np.asarray(points)
        physics = np.asarray(physics)
        new_points = np.asarray(new_points)
        
        if points.shape[1] != 3 or new_points.shape[1] != 3:
            raise ValueError("Points and new_points must have shape (*, 3).")
        if physics.shape[0] != points.shape[0] or physics.shape[1] != 1:
            raise ValueError("Physics must have shape (num_points, 1).")

        # Create a Delaunay triangulation of the input points
        delaunay = Delaunay(points)

        # Find the tetrahedrons (simplices) containing each new point
        simplex_indices = delaunay.find_simplex(new_points)

        # Find the simplices containing each new point
        simplex_indices = delaunay.find_simplex(new_points)

        # Preallocate result
        interpolated_physics = np.zeros((new_points.shape[0], 1))

        # Efficient computation of barycentric coordinates
        for i, simplex in enumerate(simplex_indices):
            if simplex == -1:  # Point outside convex hull
                # interpolated_physics[i] = np.nan
                interpolated_physics[i] = 0
                continue

            # Get the transform for the simplex
            transform = delaunay.transform[simplex]
            coords = new_points[i] - transform[3]
            barycentric_coords = np.dot(transform[:3], coords)
            barycentric_coords = np.append(barycentric_coords, 1 - np.sum(barycentric_coords))

            # Interpolate physics
            vertices = delaunay.simplices[simplex]
            interpolated_physics[i] = np.sum(physics[vertices].flatten() * barycentric_coords)
            # check if nan in the interpolated physics
            # if np.isnan(interpolated_physics[i]):
            #     print('nan exists in interpolated physics')

        return torch.tensor(interpolated_physics)
    
    def get_partition_domain(self, data, mode):
        # because the current dataset only contains one data object, only perform partitioning on the first data object
        if os.path.exists(os.path.join(self.root, 'partition', 'data.pt')):
            subdomains = torch.load(os.path.join(self.root, 'partition', 'data.pt'), map_location=torch.device('cpu'))
        else:
            os.makedirs(os.path.join(self.root, 'partition'), exist_ok=True)
            if mode == 'train':
                data = data[0]
                subdomains = self._get_partition_domain(data, self.sub_size)
                torch.save(subdomains, os.path.join(self.root, 'partition', 'data.pt'))
            else:
                return data   
        return subdomains  
    
    # @staticmethod
    # def _get_partition_domain(data, sub_size=0.001):
    #     """
    #     returns a full partitioned collection of subdomains of the original domain
        
    #     :param data: the original domain stored in a torch_geometric.data.Data
    #     """
    #     subdomains = []
    #     # get domain geometry bounds
    #     x_min, x_max = data.pos[:, 0].min(), data.pos[:, 0].max()
    #     y_min, y_max = data.pos[:, 1].min(), data.pos[:, 1].max()
    #     z_min, z_max = data.pos[:, 2].min(), data.pos[:, 2].max()
    #     # temporary fix to the device issue
    #     # data.edge_index = torch.Tensor(data.edge_index)
    #     print('x range: ', x_min, x_max)
    #     print('y range: ', y_min, y_max)
    #     print('z range: ', z_min, z_max)
    #     print('sub_size: ', sub_size)

    #     # subdomain_count = 0
    #     # divide the domain into subdomains according to self.sub_size
    #     for x in np.arange(x_min, x_max, sub_size):
    #         for y in np.arange(y_min, y_max, sub_size):
    #             for z in np.arange(z_min, z_max, sub_size):
    #                 # find nodes within the subdomain
    #                 mask = (data.pos[:, 0] >= x) & (data.pos[:, 0] < x + sub_size) & \
    #                     (data.pos[:, 1] >= y) & (data.pos[:, 1] < y + sub_size) & \
    #                     (data.pos[:, 2] >= z) & (data.pos[:, 2] < z + sub_size)
    #                 if mask.unique().size(0) == 1:
    #                     continue
    #                 else:
    #                     subdomain, _ = subgraph(mask, data.edge_index)

    #                 ########################## TBD: fix boundary information ##########################
    #                 '''
    #                 # add boundary information to the subdomain. boundary information is applied as vector on the boundary nodes
    #                 # indentify boundary nodes
    #                 boundary_mask = GenericGraphDataset.get_graph_boundary_edges(subdomain)
    #                 boundary_nodes = subdomain.edge_index[0][boundary_mask].unique()
    #                 boundary_nodes = torch.cat([boundary_nodes, subdomain.edge_index[1][boundary_mask].unique()])
    #                 boundary_nodes = boundary_nodes.unique()
    #                 boundary_nodes = boundary_nodes[boundary_nodes != -1]

    #                 # add boundary information to the subdomain
    #                 boundary_info = torch.zeros((boundary_nodes.size(0), 3))
    #                 # compute boundary vector
    #                 # get all edges connected to the boundary nodes
    #                 boundary_edges = subdomain.edge_index[:, boundary_mask]
    #                 # for every node on the boundary, compute Neumann boundary condition by averaging the 'x' property of the connected nodes
    #                 for i, node in enumerate(boundary_nodes):
    #                     connected_nodes = boundary_edges[1][boundary_edges[0] == node]

    #                     # compute magnitude & direction of the boundary vector
    #                     boundary_vector = data.pos[node] - data.pos[connected_nodes]
    #                     boundary_magnitude = data.x[node] - data.x[connected_nodes]
    #                     # compute Neumann boundary condition
    #                     boundary_info[i] = boundary_magnitude / boundary_vector.norm()

    #                 # add boundary information to the subdomain
    #                 subdomain.bc = boundary_info
    #                 '''
    #                 ####################################################################################
    #                 # check if nan exists in the subdomain
    #                 if torch.isnan(data.x[mask]).sum() > 0:
    #                     print('nan exists')
    #                     continue

    #                 edge_attr = torch.norm(data.pos[subdomain[0]] - data.pos[subdomain[1]], dim=1).unsqueeze(1)
    #                 # reorganize edge index to be start from 0
    #                 unique_nodes = torch.unique(subdomain)
    #                 node_map = dict(zip(unique_nodes.numpy(), range(unique_nodes.size(0))))
    #                 edge_index = torch.tensor([[node_map[edge[0].item()] for edge in subdomain.t()], [node_map[edge[1].item()] for edge in subdomain.t()]], dtype=torch.long)
    #                 edge_index = edge_index.view(2, -1)
    #                 subdomain = Data(x=data.x[mask], pos=data.pos[mask], y=data.y[mask], edge_index=edge_index, edge_attr=edge_attr, global_node_id=unique_nodes)
    #                 # subdomain = Data(x=data.x[mask], pos=data.pos[mask], y=data.y[mask], edge_index=subdomain, edge_attr=edge_attr)
    #                 subdomains.append(subdomain)
    #                 # print('subdomain created')
    #                 # subdomain_count += 1

    #     print('subdomain count: ', len(subdomains))

    #     return subdomains

    def _get_partition_domain(self, data, sub_size=0.001):
        # vectorized implementation of the previous function
        subdomains = []
        # get domain geometry bounds
        x_min, x_max = data.pos[:, 0].min(), data.pos[:, 0].max()
        y_min, y_max = data.pos[:, 1].min(), data.pos[:, 1].max()
        z_min, z_max = data.pos[:, 2].min(), data.pos[:, 2].max()
        # temporary fix to the device issue
        # data.edge_index = torch.Tensor(data.edge_index)
        print('x range: ', x_min, x_max)
        print('y range: ', y_min, y_max)
        print('z range: ', z_min, z_max)
        print('sub_size: ', sub_size)

        x_coords = torch.arange(x_min, x_max, sub_size)
        y_coords = torch.arange(y_min, y_max, sub_size)
        z_coords = torch.arange(z_min, z_max, sub_size)
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        grid_z = grid_z.flatten()
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)

        # find nodes within the subdomain
        # for x, y, z in grid:
        #     mask = (data.pos[:, 0] >= x) & (data.pos[:, 0] < x + sub_size) & \
        #         (data.pos[:, 1] >= y) & (data.pos[:, 1] < y + sub_size) & \
        #         (data.pos[:, 2] >= z) & (data.pos[:, 2] < z + sub_size)
        #     if mask.unique().size(0) == 1:
        #         continue
        #     else:
        #         subdomain, _ = subgraph(mask, data.edge_index)

        #     # check if nan exists in the subdomain
        #     if torch.isnan(data.x[mask]).sum() > 0:
        #         print('nan exists')
        #         continue

        #     edge_attr = torch.norm(data.pos[subdomain[0]] - data.pos[subdomain[1]], dim=1).unsqueeze(1)
        #     # reorganize edge index to be start from 0
        #     unique_nodes = torch.unique(subdomain)
        #     node_map = dict(zip(unique_nodes.numpy(), range(unique_nodes.size(0))))
        #     edge_index = torch.tensor([[node_map[edge[0].item()] for edge in subdomain.t()], [node_map[edge[1].item()] for edge in subdomain.t()]], dtype=torch.long)
        #     edge_index = edge_index.view(2, -1)
        #     subdomain = Data(x=data.x[mask], pos=data.pos[mask], y=data.y[mask], edge_index=edge_index, edge_attr=edge_attr, global_node_id=unique_nodes)
        #     subdomains.append(subdomain)

        # print('subdomain count: ', len(subdomains))
        
        # parallelize the process
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(DuctAnalysisDataset.get_subdomain, data, sub_size, x, y, z) for x, y, z in grid]
            wait(futures, return_when=ALL_COMPLETED)
            for future in futures:
                subdomain = future.result()
                if subdomain is not None:
                    subdomains.append(subdomain)

        return subdomains
    
    @staticmethod
    def get_subdomain(data, sub_size, x, y, z):
        mask = (data.pos[:, 0] >= x) & (data.pos[:, 0] < x + sub_size) & \
            (data.pos[:, 1] >= y) & (data.pos[:, 1] < y + sub_size) & \
            (data.pos[:, 2] >= z) & (data.pos[:, 2] < z + sub_size)
        if mask.unique().size(0) == 1:
            return None
        else:
            subdomain, _ = subgraph(mask, data.edge_index)

        # check if nan exists in the subdomain
        if torch.isnan(data.x[mask]).sum() > 0:
            print('nan exists')
            return None

        edge_attr = torch.norm(data.pos[subdomain[0]] - data.pos[subdomain[1]], dim=1).unsqueeze(1)
        # reorganize edge index to be start from 0
        unique_nodes = torch.unique(subdomain)
        node_map = dict(zip(unique_nodes.numpy(), range(unique_nodes.size(0))))
        edge_index = torch.tensor([[node_map[edge[0].item()] for edge in subdomain.t()], [node_map[edge[1].item()] for edge in subdomain.t()]], dtype=torch.long)
        edge_index = edge_index.view(2, -1)
        subdomain = Data(x=data.x[mask], pos=data.pos[mask], y=data.y[mask], edge_index=edge_index, edge_attr=edge_attr, global_node_id=unique_nodes)
        return subdomain
    
    @staticmethod
    def reconstruct_from_partition(subdomains):
        """
        reconstructs the original domain from a partitioned collection of subdomains
        
        :param subdomains: a list of subdomains, each stored in a torch_geometric.data.Data object
        """
        # concatenate all subdomains
        data = Data()
        # arrange data.x of subdomains according to the global node id
        global_node_id = torch.cat([subdomain.global_node_id for subdomain in subdomains], dim=0)
        global_node_id = global_node_id.unique()
        global_node_id = global_node_id[global_node_id != -1]
        # node_map = dict(zip(global_node_id.numpy(), range(global_node_id.size(0)))
        x = torch.cat([subdomain.x for subdomain in subdomains], dim=0)
        x = x[global_node_id]
        # arrange data.y of subdomains according to the global node id
        y = torch.cat([subdomain.y for subdomain in subdomains], dim=0)
        y = y[global_node_id]
        # arrange data.pos of subdomains according to the global node id
        data.x = x
        data.y = y

        return data
    
    def get_one_full_sample(self, idx):
        return self.data
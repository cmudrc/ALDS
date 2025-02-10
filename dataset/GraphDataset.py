import os
import meshio
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 
import time
import torch
import numpy as np
import random
import tqdm
import vtk
from vtk import vtkFLUENTReader
from collections import deque
import pandas as pd
# import multiprocessing as mp
from multiprocessing import Manager, Process
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import multiprocessing as mp


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
        return len(self._data)
    
    def get(self, idx):
        return self._data[idx]

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
    def extract_unstructured_grid(multi_block):
        """
        Extracts the first vtkUnstructuredGrid from a vtkMultiBlockDataSet.
        """
        for i in range(multi_block.GetNumberOfBlocks()):
            block = multi_block.GetBlock(i)
            if isinstance(block, vtk.vtkUnstructuredGrid):
                return block
        raise ValueError("No vtkUnstructuredGrid found in the .msh file.")
    
    @staticmethod
    def extract_wall_block(multi_block, target_name="wall:walls"):
        """
        Extracts the 'wall:walls' block from the vtkMultiBlockDataSet.
        Returns the unstructured grid corresponding to wall surfaces.
        """
        for i in range(multi_block.GetNumberOfBlocks()):
            block = multi_block.GetBlock(i)
            name = multi_block.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) if multi_block.GetMetaData(i) else None
            
            if isinstance(block, vtk.vtkUnstructuredGrid) and name and target_name in name:
                return block
        raise ValueError(f"No block named '{target_name}' found in the .msh file.")
    
    @staticmethod
    def vtk_to_pyg(data):
        """
        Converts a vtkUnstructuredGrid to a torch_geometric.data.Data object.
        """
        # Step 1: Extract vertex positions (nodes)
        num_points = data.GetNumberOfPoints()
        pos = np.array([data.GetPoint(i) for i in range(num_points)], dtype=np.float32)
        pos = torch.tensor(pos, dtype=torch.float)

        # Step 2: Extract edges from cell connectivity
        edge_set = set()
        for i in range(data.GetNumberOfCells()):
            cell = data.GetCell(i)
            num_cell_points = cell.GetNumberOfPoints()

            for j in range(num_cell_points):
                for k in range(j + 1, num_cell_points):
                    edge = (cell.GetPointId(j), cell.GetPointId(k))
                    edge_set.add(edge)
                    edge = (cell.GetPointId(k), cell.GetPointId(j))
                    edge_set.add(edge)

        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t()

        return Data(pos=pos, edge_index=edge_index)

    @staticmethod
    def _process_file(path_list):
        data_list = []
        # mesh_idx = ['High', 'Med', 'Low']
        # process mesh files
        for idx, path in enumerate(path_list[:2]):
            reader = vtkFLUENTReader()
            reader.SetFileName(path)
            reader.Update()

            # Extract mesh from VTK output
            dataset = reader.GetOutput()

            mesh = DuctAnalysisDataset.extract_unstructured_grid(dataset)
            num_points = mesh.GetNumberOfPoints()
            num_cells = mesh.GetNumberOfCells()

            if num_points == 0 or num_cells == 0:
                raise ValueError("No valid mesh data found in the .msh file.")

            try:
                wall_mesh = DuctAnalysisDataset.extract_wall_block(dataset)
                wall_indices = set()

                for i in range(wall_mesh.GetNumberOfCells()):
                    cell = wall_mesh.GetCell(i)
                    for j in range(cell.GetNumberOfPoints()):
                        wall_indices.add(cell.GetPointId(j))

                wall_index_tensor = torch.tensor(list(wall_indices), dtype=torch.long)
                print(f"Extracted {len(wall_indices)} wall node indices.")
            except ValueError:
                print("Wall block not found.")
                wall_index_tensor = torch.tensor([], dtype=torch.long)
            
            # process physics files
            # print(path_list[idx+3])
            physics = pd.read_csv(path_list[idx+3], sep=',')
            # print(physics)
            velocity_x = torch.tensor(physics['      x-velocity'], dtype=torch.float).unsqueeze(1)
            velocity_y = torch.tensor(physics['      y-velocity'], dtype=torch.float).unsqueeze(1)
            velocity_z = torch.tensor(physics['      z-velocity'], dtype=torch.float).unsqueeze(1)
            velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=1)
            # normalize the velocity to be in the range of [0, 1]
            velocity = velocity / torch.max(torch.abs(velocity))

            pressure = torch.tensor(physics['        pressure'], dtype=torch.float).unsqueeze(1)
            # normalize the pressure
            pressure = pressure / torch.max(pressure)
            # check if nan exists in the physics data
            if torch.isnan(velocity).sum() > 0 or torch.isnan(pressure).sum() > 0:
                print('nan exists in original physics data')

            # create a torch_geometric.data.Data object if the mesh is of high resolution
            if idx == 0:
                mesh_high = mesh
                data = DuctAnalysisDataset.vtk_to_pyg(mesh)
                data.y = torch.cat([velocity, pressure], dim=1)
                data.wall_idx = wall_index_tensor
                data_list.append(data)
            else:
                # call lagrangian interpolation to interpolate the physics data to the high resolution mesh
                velocity_x_high = DuctAnalysisDataset._lagrangian_interpolation(mesh, velocity_x, mesh_high)
                velocity_y_high = DuctAnalysisDataset._lagrangian_interpolation(mesh, velocity_y, mesh_high)
                velocity_z_high = DuctAnalysisDataset._lagrangian_interpolation(mesh, velocity_z, mesh_high)
                pressure_high = DuctAnalysisDataset._lagrangian_interpolation(mesh, pressure, mesh_high)

                velocity_high = torch.cat([velocity_x_high, velocity_y_high, velocity_z_high], dim=1)
                velocity_high = torch.tensor(velocity_high, dtype=torch.float)
                # normalize the velocity
                velocity_high = velocity_high / torch.max(torch.abs(velocity_high))
                pressure_high = torch.tensor(pressure_high, dtype=torch.float)
                # normalize the pressure
                pressure_high = pressure_high / torch.max(pressure_high)
                # check if nan exists in the interpolated physics data
                if torch.isnan(velocity_high).sum() > 0 or torch.isnan(pressure_high).sum() > 0:
                    print('nan exists in interpolated physics data')

                data_list[0].x = torch.cat([velocity_high, pressure_high], dim=1)

        return data_list
    
    @staticmethod
    def _lagrangian_interpolation(mesh, physics, new_mesh):
        """
        Perform 1st-order Lagrangian interpolation of physics properties 
        at a new set of points based on provided 3D points and physics information.

        Args:
            mesh (vtkUnstructuredGrid): Unstructured mesh loaded from an Ansys .msh file via vtkFLUENTReader.
            physics (np.ndarray): Array of shape (num_points, 1) representing the physics information at the points.
            new_mesh (vtkUnstructuredGrid): Unstructured mesh loaded from an Ansys .msh file for interpolation.

        Returns:
            np.ndarray: Interpolated physics values at the new points of shape (num_new_points, 1).
        """

        # Ensure physics array shape
        physics = np.asarray(physics).flatten()
        num_original_points = mesh.GetNumberOfPoints()
        
        if physics.shape[0] != num_original_points:
            raise ValueError("Mismatch: physics array length must match the number of points in the original mesh.")

        num_new_points = new_mesh.GetNumberOfPoints()
        if num_new_points == 0:
            raise ValueError("New mesh has no points to interpolate.")

        # Step 1: Attach physics data to the original mesh
        physics_array = vtk.vtkFloatArray()
        physics_array.SetName("PhysicsData")
        physics_array.SetNumberOfComponents(1)
        physics_array.SetNumberOfTuples(num_original_points)

        for i in range(num_original_points):
            physics_array.SetValue(i, physics[i])

        mesh.GetPointData().AddArray(physics_array)

        # Step 2: Use vtkProbeFilter for interpolation
        probe_filter = vtk.vtkProbeFilter()
        probe_filter.SetSourceData(mesh)  # Set original mesh as the source
        probe_filter.SetInputData(new_mesh)  # Set new mesh as the target for interpolation
        probe_filter.Update()

        # Step 3: Extract interpolated values
        interpolated_array = probe_filter.GetOutput().GetPointData().GetArray("PhysicsData")

        if interpolated_array is None:
            raise RuntimeError("Interpolation failed: No physics data found in the output.")

        # Convert to NumPy array
        interpolated_values = np.array([interpolated_array.GetValue(i) for i in range(num_new_points)], dtype=np.float32)

        return torch.tensor(interpolated_values.reshape(-1, 1), dtype=torch.float)
    
    def get_partition_domain(self, data, mode):
        """
        returns a full partitioned collection of subdomains of the original domain
        
        :param data: the original domain stored in a torch_geometric.data.Data object. 
        """
        if os.path.exists(os.path.join(self.root, 'partition', 'data.pt')):
            subdomains = torch.load(os.path.join(self.root, 'partition', 'data.pt'), map_location=torch.device('cpu'))
        else:
            os.makedirs(os.path.join(self.root, 'partition'), exist_ok=True)
            reader = vtkFLUENTReader()
            reader.SetFileName(os.path.join(self.raw_dir, self.raw_file_names[0]))
            reader.Update()
            mesh = reader.GetOutput().GetBlock(0)
            data = data[0]
            x, y, pos = data.x, data.y, data.pos
            num_subdomains = self.sub_size
            subdomains = self._get_partition_domain(mesh, x, y, pos, num_subdomains)

            torch.save(subdomains, os.path.join(self.root, 'partition', 'data.pt'))
        return subdomains
    
    @staticmethod
    def compute_centroid_chunk(mesh, cell_ids, progress_dict, process_id):
        """Computes the centroids for a chunk of cells."""
        chunk_results = []
        for idx, cell_id in enumerate(cell_ids):
            cell = mesh.GetCell(cell_id)
            num_cell_points = cell.GetNumberOfPoints()
            centroid = np.mean([mesh.GetPoint(cell.GetPointId(j)) for j in range(num_cell_points)], axis=0)
            chunk_results.append((cell_id, centroid))
            
            # Update progress every 100 cells
            if idx % 100 == 0:
                progress_dict[process_id] = idx

        return chunk_results
    
    @staticmethod
    def progress_monitor(progress_dict, total_cells):
        """Monitors the progress of parallel processing."""
        start_time = time.time()
        while True:
            completed = sum(progress_dict.values())
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / completed) * (total_cells - completed) if completed > 0 else 0
            print(f"\rProgress: {completed}/{total_cells} cells processed - Time elapsed: {elapsed_time:.2f}s - ETA: {eta:.2f}s", end="")
            time.sleep(2)  # Update every 2 seconds
            if completed >= total_cells:
                break
        print("\nProcessing complete!")
    
    def compute_cell_centroids(self, mesh, chunk_size=10000):
        """
        Computes the centroids of all cells in the VTK unstructured mesh using multiprocessing.
        
        Args:
            mesh (vtk.vtkUnstructuredGrid): The input unstructured mesh.
            chunk_size (int): Number of cells to process per batch to reduce memory load.

        Returns:
            np.ndarray: Shape (num_cells, 3), array of cell centroids.
        """
        num_cells = mesh.GetNumberOfCells()
        centroids = np.zeros((num_cells, 3), dtype=np.float32)
        cell_ids = list(range(num_cells))
        num_chunks = (num_cells // chunk_size) + 1
        max_workers = os.cpu_count() - 1

        # Shared dictionary to track progress
        with Manager() as manager:
            progress_dict = manager.dict({i: 0 for i in range(num_chunks)})

            # Start progress monitor in a separate process
            monitor_process = Process(target=self.progress_monitor, args=(progress_dict, num_cells))
            monitor_process.start()

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.compute_centroid_chunk, mesh, cell_ids[i:i + chunk_size], progress_dict, i): i
                    for i in range(0, num_cells, chunk_size)
                }

                for future in as_completed(futures):  # Dynamically process completed tasks
                    chunk_index = futures[future]
                    try:
                        chunk_results = future.result()
                        for cell_id, centroid in chunk_results:
                            centroids[cell_id] = centroid
                    except Exception as e:
                        print(f"Error processing chunk {chunk_index}: {e}")

            monitor_process.join()  # Wait for progress monitoring to complete

        return centroids

    def _get_partition_domain(self, mesh, x, y, pos, num_subdomains):
        """
        Perform domain decomposition on a VTK unstructured mesh and associated physics data.
        Uses METIS-based partitioning without explicit graph conversion.

        Args:
            mesh (vtkUnstructuredGrid): The unstructured mesh to be partitioned.
            physics (np.ndarray): Physics properties at mesh nodes, shape (num_points, physics_dim).
            num_subdomains (int): The number of decomposed subdomains.

        Returns:
            subdomain_meshes (list of vtkUnstructuredGrid): Decomposed sub-meshes.
            subdomain_physics (list of np.ndarray): Physics properties for each subdomain.
        """
        # Initialize VTK MPI controller (Required for distributed processing)
        controller = vtk.vtkMultiProcessController.GetGlobalController()
        if controller is None:
            print("Error: VTK MPI controller not initialized. Using single process.")
            controller = vtk.vtkDummyController()

        # Set up the distributed data filter
        distributed_filter = vtk.vtkDistributedDataFilter()
        distributed_filter.SetController(controller)
        distributed_filter.SetInputData(mesh)
        # set up the number of partitions
        kd_tree = distributed_filter.GetKdtree()
        kd_tree.SetNumberOfRegionsOrLess(num_subdomains)
        progress_observer = ProgressObserver()
        distributed_filter.AddObserver(vtk.vtkCommand.ProgressEvent, progress_observer)
        distributed_filter.SetBoundaryModeToAssignToAllIntersectingRegions()

        # Execute the partitioning
        distributed_filter.Update()

        # Retrieve partitioned mesh
        partitioned_mesh = distributed_filter.GetOutput()

        subdomain_dataset = []
        for i in tqdm.tqdm(range(num_subdomains), desc="Processing Subdomains"):
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(partitioned_mesh)
            threshold.SetUpperThreshold(i)
            threshold.SetLowerThreshold(i)
            threshold.Update()

            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(threshold.GetOutput())
            geometry_filter.Update()

            submesh = geometry_filter.GetOutput()
            sub_x = x[threshold.GetOutput().GetPointData().GetArray('GlobalNodeID')]
            sub_y = y[threshold.GetOutput().GetPointData().GetArray('GlobalNodeID')]
            sub_pos = pos[threshold.GetOutput().GetPointData().GetArray('GlobalNodeID')]

            subdomain_data = self.vtk_to_pyg(submesh)
            subdomain_data.x = torch.tensor(sub_x, dtype=torch.float).squeeze(1)
            subdomain_data.y = torch.tensor(sub_y, dtype=torch.float).squeeze(1)
            subdomain_data.pos = torch.tensor(sub_pos, dtype=torch.float).squeeze(1)

            subdomain_dataset.append(subdomain_data)

        return subdomain_dataset
    
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
        pos = torch.cat([subdomain.pos for subdomain in subdomains], dim=0)
        pos = pos[global_node_id]
        # arrange data.edge_index of subdomains according to the global node id
        # edge_index = torch.cat([subdomain.edge_index for subdomain in subdomains], dim=1)
        # edge_index = edge_index[:, edge_index[0].argsort()]
        # edge_index = edge_index[:, edge_index[1].argsort()]
        # edge_index = edge_index[:, edge_index[0] != -1]
        # edge_index = edge_index[:, edge_index[1] != -1]
        # edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        # # arrange data.edge_attr of subdomains according to the global node id
        # edge_attr = torch.cat([subdomain.edge_attr for subdomain in subdomains], dim=0)
        # edge_attr = edge_attr[edge_index[0]]
        
        data.x = x
        data.y = y
        data.pos = pos
        # data.edge_index = edge_index
        # data.edge_attr = edge_attr

        return data
    
    def get_one_full_sample(self, idx):
        return self.data
    

class SubGraphDataset(InMemoryDataset):
    # similar to Sub_jhtdb, creates a subset of the original dataset given the indices
    def __init__(self, root, indices):
        super(SubGraphDataset, self).__init__(None)
        self.indices = indices

        self.data = torch.load(os.path.join(self.root, 'processed', 'data.pt'))
        self.data = [self.data[i] for i in self.indices]


class ProgressObserver:
    """Observer class to track VTK filter progress."""
    def __init__(self):
        self.progress = 0

    def __call__(self, caller, event):
        """Handles progress update events."""
        if isinstance(caller, vtk.vtkDistributedDataFilter):
            self.progress = caller.GetProgress() * 100
            print(f"\rPartitioning Progress: {self.progress:.2f}%", end="")
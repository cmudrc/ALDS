import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import torch
import numpy as np
import scipy.io
import h5py
import pyJHTDB
from pyJHTDB import libJHTDB
import sklearn.metrics
# from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset
import torch.nn as nn


# class MatDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(MatDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return []
    
#     @property
#     def processed_file_names(self):
#         return ['data.pt']
    
#     def download(self):
#         pass

#     def process(self):
#         raise NotImplementedError
    
#     def extract_solution(self, h5_file, sim, res):
#         raise NotImplementedError
    
#     def construct_data_object(self, coords, connectivity, solution, k):
#         raise NotImplementedError
    

class BurgersDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.pre_transform = pre_transform
        # super(BurgersDataset, self).__init__(root, transform, pre_transform)
        self.raw_dir = os.path.join(root, 'raw')
        self.processed_dir = os.path.join(root, 'processed')
        self.processed_paths = [os.path.join(self.processed_dir, 'burgers_data.pt')]
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data = torch.load(self.processed_paths[0])
        
    
    @property
    def raw_file_names(self):
        return ['solution_10.h5', 'solution_20.h5', 'solution_40.h5', 'solution_80.h5']
    
    @property
    def mesh_file_names(self):
        return ['mesh_10.h5', 'mesh_20.h5', 'mesh_40.h5', 'mesh_80.h5']
    
    @property
    def processed_file_names(self):
        return ['burgers_data.pt']

    def process(self):
        data_list = []
        # load mesh
        with h5py.File(os.path.join(self.raw_dir, self.mesh_file_names[3]), 'r') as f:
            X = f['X'][:]

        for i in range(10):
            pos = torch.tensor(X, dtype=torch.float)
            pos_x = pos[:, 0].unsqueeze(1)
            pos_y = pos[:, 1].unsqueeze(1)
            
            x_values = np.unique(pos_x)
            y_values = np.unique(pos_y)

            # print('res: {}, i: {}'.format(res, i))
            with h5py.File(os.path.join(self.raw_dir, self.raw_file_names[3]), 'r') as f:  
                with h5py.File(os.path.join(self.raw_dir, self.raw_file_names[0]), 'r') as f_l:
                    data_array_group = f['{}'.format(i)]
                    
                    dset = data_array_group['u'][:]

                    data_array_group_l = f_l['{}'.format(i)]
                    dset_l = data_array_group_l['u'][:]
                    
                    # take two sequential time steps and form the input and label for the entire temporal sequence
                    for i in range(dset.shape[0] - 1):
                        y = torch.tensor(dset[i], dtype=torch.float)
                        y = torch.sqrt(y[0, :]**2 + y[1, :]**2).unsqueeze(1).reshape(len(x_values), len(y_values), 1)
                        # normalize the label
                        y = (y - y.min()) / (y.max() - y.min())
                        # y = np.concatenate((y, pos_x, pos_y), axis=1).reshape(len(x_values), len(y_values), 3)

                        x = torch.tensor(dset_l[i], dtype=torch.float)
                        x = torch.sqrt(x[0, :]**2 + x[1, :]**2).unsqueeze(1)
                        x = (x - x.min()) / (x.max() - x.min())
                        x = np.concatenate((x, pos_x, pos_y), axis=1).reshape(len(x_values), len(y_values), 3)
                        data = [x, y]

                        data_list.append(data)

        torch.save(data_list, self.processed_paths[0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class BurgersDatasetWhole(Dataset):
    """
    Same as BurgersDataset, but does not have the _get_window method
    """
    def __init__(self, root, transform=None, pre_transform=None):
        self.pre_transform = pre_transform
        # super(BurgersDataset, self).__init__(root, transform, pre_transform)
        self.raw_dir = os.path.join(root, 'raw')
        self.processed_dir = os.path.join(root, 'processed')
        self.processed_paths = [os.path.join(self.processed_dir, 'burgers_data.pt')]
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data = torch.load(self.processed_paths[0])
            
    
    @property
    def raw_file_names(self):
        return ['solution_10.h5', 'solution_20.h5', 'solution_40.h5', 'solution_80.h5']
    
    @property
    def mesh_file_names(self):
        return ['mesh_10.h5', 'mesh_20.h5', 'mesh_40.h5', 'mesh_80.h5']
    
    @property
    def processed_file_names(self):
        return ['burgers_data.pt']

    def process(self):
        data_list = []
        # load mesh
        with h5py.File(os.path.join(self.raw_dir, self.mesh_file_names[3]), 'r') as f:
            X = f['X'][:]
a
class JHTDB(Dataset):
    # def initialize_JHTDB():
    #     """
    #     Initialize the JHTDB object
    #     """
    #     # Parameters for the data download
    #     lJHTDB = libJHTDB()
    #     lJHTDB.initialize()
    #     lJHTDB.add_token('edu.cmu.zedaxu-f374fe6b')
    #     return lJHTDB
    def __init__(self, root, tstart, tend, fields, dataset):
        self.root = root
        self.tstart = tstart
        self.tend = tend
        self.fields = fields
        self.dataset = dataset
        self.data = self.process()

        self.jhtdb = pyJHTDB.libJHTDB()
        self.jhtdb.initialize()
        self.jhtdb.add_token('edu.cmu.zedaxu-f374fe6b')

    def _download(self):
        os.makedirs(self.root, exist_ok=True)

        result = self.jhtdb.getbigCutout(
            tstart=self.tstart,
            tend=self.tend,
            field=self.fields,
            dataset=self.dataset,
            start=np.array([0, 0, 512], dtype=np.int),
            end=np.array([1024, 1024, 512], dtype=np.int),
            step=np.array([1, 1, 1], dtype=np.int),
            filename=os.path.join(self.root, 'data')
        )

    def _process(self):
        data = h5py.File(os.path.join(self.root, 'data'), 'r')
        
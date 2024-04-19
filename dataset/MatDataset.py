import os
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import torch
import numpy as np
import scipy.io
import ctypes
import h5py
import shutil
import pyJHTDB
from pyJHTDB import libJHTDB
import sklearn.metrics
# from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
# from paraview.simple import *
# from dolfin import *


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
    def __init__(self, root, tstart, tend, fields, dataset, down_sample_rate=3, partition=False, **kwargs):
        self.root = root
        self.tstart = tstart
        self.tend = tend
        self.fields = fields
        self.down_sample_rate = down_sample_rate
        self.dataset = dataset
        self.jhtdb = pyJHTDB.libJHTDB()
        self.jhtdb.initialize()
        self.jhtdb.lib.turblibSetExitOnError(ctypes.c_int(0))
        self.jhtdb.add_token('edu.cmu.zedaxu-f374fe6b')
        if partition:
            self.sub_size = kwargs['sub_size']

        self.data = self.process(partition)

    def _download(self):
        os.makedirs(os.path.join(self.root, 'raw'), exist_ok=True)

        result = self.jhtdb.getbigCutout(
            t_start=self.tstart,
            t_end=self.tend,
            t_step=1,
            fields=self.fields,
            data_set=self.dataset,
            start=np.array([1, 1, 512], dtype=np.int),
            end=np.array([1024, 1024, 512], dtype=np.int),
            step=np.array([1, 1, 1], dtype=np.int),
            filename='data',
        )

        self.jhtdb.finalize()

        shutil.move('data.xmf', os.path.join(self.root, 'raw'))
        shutil.move('data.h5', os.path.join(self.root, 'raw'))

        print(result.shape)

    def _process(self, flag_partition=False):
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        u_list = []
        with h5py.File(os.path.join(self.root, 'raw', 'data.h5'), 'r') as f:
            for i in range(self.tend - self.tstart+1):
                u_idx = str(i+1).rjust(4, '0')
                u = f['Velocity_{}'.format(u_idx)][:].astype(np.float32)
                u_label = torch.tensor(u[0, :, :, :])
                u_label = torch.sqrt(u_label[:, :, 0]**2 + u_label[:, :, 1]**2 + u_label[:, :, 2]**2)
                # pool the u_label to create low resolution u_input
                u_input = self._pool(u_label, 3)

                if flag_partition:
                    u_input_list = self.get_partition_domain(u_input.unsqueeze(-1), mode='test')
                    u_label_list = self.get_partition_domain(u_label.unsqueeze(-1), mode='test')
                    for u_input, u_label in zip(u_input_list, u_label_list):
                        u_list.append([u_input, u_label])
                else:
                    u_list.append([u_input, u_label])

        torch.save(u_list, os.path.join(self.root, 'processed', 'data.pt'))

    def symmetric_padding(self, x, mode):
        # pad the domain symmetrically to make it divisible by sub_size
        # get pad size
        pad_size = (x.shape[1] % self.sub_size) // 2 + 1
        x = F.pad(x, (0, 0, pad_size, pad_size, pad_size, pad_size))
        # print(x[0, :, :, 0])
        if mode == 'train':
            # add one dimension to the tensor x, with 0 in the padded region and 1 in the original region
            x_pad_idx = torch.ones((x.shape[0], x.shape[1], 1))
            x_pad_idx[:pad_size, :, :] = 0
            x_pad_idx[-pad_size:, :, :] = 0
            x_pad_idx[:, -pad_size:, :] = 0
            x_pad_idx[:, :pad_size, :] = 0
            x = torch.cat((x, x_pad_idx), dim=-1)
            return x, pad_size
        elif mode == 'test':    
            return x, pad_size

    def get_partition_domain(self, x, mode, displacement=0):
        # pad the domain symmetrically to make it divisible by sub_size
        x, pad_size = self.symmetric_padding(x, mode)
        # partition the domain into num_partitions subdomains of the same size
        x_list = []
        num_partitions_dim = x.shape[1] // self.sub_size

        for i in range(num_partitions_dim):
            for j in range(num_partitions_dim):
                x_list.append(x[i*self.sub_size:(i+1)*self.sub_size, j*self.sub_size:(j+1)*self.sub_size, :])
        return x_list
    
    def reconstruct_from_partitions(self, x, x_list, displacement=0):
        # reconstruct the domain from the partitioned subdomains
        num_partitions_dim = int(np.sqrt(len(x_list)))
        x, pad_size = self.symmetric_padding(x, mode='test')
        x = torch.zeros_like(x[:, 1:-1, 1:-1, 0].unsqueeze(-1))
        # if the domain can be fully partitioned into subdomains of the same size
        # if len(x_list) == num_partitions_dim**2:
        for i in range(num_partitions_dim):
            for j in range(num_partitions_dim):
                x[:, i:i+self.sub_size-2, j:j+self.sub_size-2, :] = x_list[i*num_partitions_dim + j][:, 1:-1, 1:-1, :]

        if pad_size == 1:
            return x
        else:
            x = x[:, pad_size-1:-pad_size+1, pad_size-1:-pad_size+1, :]
            return x

    def _pool(self, u, factor):
        # average pooling on the input u, maintaining the same shape
        # pad the domain u so that the pooled u has the same shape as the original u
        u = torch.nn.functional.pad(u, (int((factor-1)/2), int((factor-1)/2), int((factor-1)/2), int((factor-1)/2)))
        u = u.unsqueeze(0).unsqueeze(0)
        u = torch.nn.functional.avg_pool2d(u, factor, stride=1).squeeze(0).squeeze(0)
        return u 

    def process(self, flag_partition=False):
        if not (os.path.exists(os.path.join(self.root, 'raw', 'data.h5')) or os.path.exists(os.path.join(self.root, 'processed', 'data.pt'))):
            self._download()
        if not os.path.exists(os.path.join(self.root, 'processed', 'data.pt')):
            self._process(flag_partition)
        return torch.load(os.path.join(self.root, 'processed', 'data.pt'))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    


class JHTDB_ICML(Dataset):
    # def initialize_JHTDB():
    #     """
    #     Initialize the JHTDB object
    #     """
    #     # Parameters for the data download
    #     lJHTDB = libJHTDB()
    #     lJHTDB.initialize()
    #     lJHTDB.add_token('edu.cmu.zedaxu-f374fe6b')
    #     return lJHTDB
    def __init__(self, root, tstart, tend, fields, dataset, partition=False, **kwargs):
        self.root = root
        self.tstart = tstart
        self.tend = tend
        self.fields = fields
        self.dataset = dataset
        self.flag_partition = partition
        self.jhtdb = pyJHTDB.libJHTDB()
        self.jhtdb.initialize()
        self.jhtdb.lib.turblibSetExitOnError(ctypes.c_int(0))
        self.jhtdb.add_token('edu.cmu.zedaxu-f374fe6b')
        
        if partition:
            self.sub_size = kwargs['sub_size']

        self.data = self.process(partition)

    def _download(self):
        os.makedirs(os.path.join(self.root, 'raw'), exist_ok=True)

        result = self.jhtdb.getbigCutout(
            t_start=self.tstart,
            t_end=self.tend,
            t_step=1,
            fields=self.fields,
            data_set=self.dataset,
            start=np.array([1, 1, 512], dtype=np.int),
            end=np.array([1024, 1024, 512], dtype=np.int),
            step=np.array([1, 1, 1], dtype=np.int),
            filename='data',
        )

        self.jhtdb.finalize()

        shutil.move('data.xmf', os.path.join(self.root, 'raw'))
        shutil.move('data.h5', os.path.join(self.root, 'raw'))

        print(result.shape)

    def _process(self, flag_partition=False):
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        u_list = []
        with h5py.File(os.path.join(self.root, 'raw', 'data.h5'), 'r') as f:
            for i in range(self.tend - self.tstart - 100):
                u_idx = str(i+1).rjust(4, '0')
                u_input = f['Velocity_{}'.format(u_idx)][:].astype(np.float32)
                u_input = torch.tensor(u_input[0, :, :, :])
                u_input = torch.sqrt(u_input[:, :, 0]**2 + u_input[:, :, 1]**2 + u_input[:, :, 2]**2)
                # u_label at the next time step
                u_label_idx = str(i+100).rjust(4, '0')
                u_label = f['Velocity_{}'.format(u_label_idx)][:].astype(np.float32)
                u_label = torch.tensor(u_label[0, :, :, :])
                u_label = torch.sqrt(u_label[:, :, 0]**2 + u_label[:, :, 1]**2 + u_label[:, :, 2]**2)

                if flag_partition:
                    u_input_list = self.get_partition_domain(u_input.unsqueeze(-1), mode='test')
                    u_label_list = self.get_partition_domain(u_label.unsqueeze(-1), mode='test')
                    for u_input, u_label in zip(u_input_list, u_label_list):
                        u_list.append([u_input, u_label])
                else:
                    u_list.append([u_input, u_label])

        torch.save(u_list, os.path.join(self.root, 'processed', 'data.pt'))

    def symmetric_padding(self, x, mode):
        # pad the domain symmetrically to make it divisible by sub_size
        # get pad size
        pad_size = (x.shape[1] % self.sub_size) // 2 + 1
        x = F.pad(x, (0, 0, pad_size, pad_size, pad_size, pad_size))
        # print(x[0, :, :, 0])
        if mode == 'train':
            # add one dimension to the tensor x, with 0 in the padded region and 1 in the original region
            x_pad_idx = torch.ones((x.shape[0], x.shape[1], 1))
            x_pad_idx[:pad_size, :, :] = 0
            x_pad_idx[-pad_size:, :, :] = 0
            x_pad_idx[:, -pad_size:, :] = 0
            x_pad_idx[:, :pad_size, :] = 0
            x = torch.cat((x, x_pad_idx), dim=-1)
            return x, pad_size
        elif mode == 'test':    
            return x, pad_size

    def get_partition_domain(self, x, mode, displacement=0):
        # pad the domain symmetrically to make it divisible by sub_size
        x, pad_size = self.symmetric_padding(x, mode)
        # partition the domain into num_partitions subdomains of the same size
        x_list = []
        num_partitions_dim = x.shape[1] // self.sub_size

        for i in range(num_partitions_dim):
            for j in range(num_partitions_dim):
                x_list.append(x[i*self.sub_size:(i+1)*self.sub_size, j*self.sub_size:(j+1)*self.sub_size, :])
        return x_list
    
    def reconstruct_from_partitions(self, x, x_list, displacement=0):
        # reconstruct the domain from the partitioned subdomains
        num_partitions_dim = int(np.sqrt(len(x_list)))
        x, pad_size = self.symmetric_padding(x, mode='test')
        x = torch.zeros_like(x)
        # if the domain can be fully partitioned into subdomains of the same size
        # if len(x_list) == num_partitions_dim**2:
        for i in range(num_partitions_dim):
            for j in range(num_partitions_dim):
                x[:, i*self.sub_size:(i+1)*self.sub_size, j*self.sub_size:(j+1)*self.sub_size, :] = x_list[i*num_partitions_dim + j].unsqueeze(0)

        if pad_size == 1:
            return x
        else:
            x = x[:, pad_size-1:-pad_size+1, pad_size-1:-pad_size+1, :]
            return x

    def process(self, flag_partition=False):
        if not (os.path.exists(os.path.join(self.root, 'raw', 'data.h5')) or os.path.exists(os.path.join(self.root, 'processed', 'data.pt'))):
            self._download()
        if not os.path.exists(os.path.join(self.root, 'processed', 'data.pt')):
            self._process(flag_partition)
        return torch.load(os.path.join(self.root, 'processed', 'data.pt'))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_one_full_sample(self, idx):
        if not self.flag_partition:
            return self.data[idx]
        else:
            with h5py.File(os.path.join(self.root, 'raw', 'data.h5'), 'r') as f:
                u_idx = str(idx+1).rjust(4, '0')
                u_input = f['Velocity_{}'.format(u_idx)][:].astype(np.float32)
                u_input = torch.tensor(u_input[0, :, :, :])
                u_input = torch.sqrt(u_input[:, :, 0]**2 + u_input[:, :, 1]**2 + u_input[:, :, 2]**2)
                # u_label at the next time step
                u_label_idx = str(idx+2).rjust(4, '0')
                u_label = f['Velocity_{}'.format(u_label_idx)][:].astype(np.float32)
                u_label = torch.tensor(u_label[0, :, :, :])
                u_label = torch.sqrt(u_label[:, :, 0]**2 + u_label[:, :, 1]**2 + u_label[:, :, 2]**2)
                u_input_list = self.get_partition_domain(u_input.unsqueeze(-1), mode='test')
                u_label_list = self.get_partition_domain(u_label.unsqueeze(-1), mode='test')
                return u_input.unsqueeze(-1), u_input_list, u_label_list
        

class Sub_JHTDB(Dataset):
    '''
    Includes a subset of the JHTDB dataset given the indices of the data
    '''
    def __init__(self, root, indices):
        self.root = root
        # verify that JHTDB data is correctly processed
        if not os.path.exists(os.path.join(self.root, 'processed', 'data.pt')):
            raise ValueError('JHTDB data is not processed yet')
        self.indices = indices

        self.data = torch.load(os.path.join(self.root, 'processed', 'data.pt'))
        self.data = [self.data[i] for i in self.indices]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
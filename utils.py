import os
import argparse
import time
from models.classifier import *
from models.encoder import *
from models.model import *
# from models.scheduler import *
# from deepxde.nn.pytorch import DeepONet
from dataset.MatDataset import *
from dataset.GraphDataset import *
# from torch_geometric.data import Data
import torch_geometric as pyg
from torch_geometric.nn import GraphSAGE
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
import wandb
from numba import jit
import vtk
from vtk import vtkUnstructuredGrid, vtkPoints, vtkCellArray, vtkXMLUnstructuredGridWriter, vtkTetra, vtkHexahedron
import networkx as nx


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_cur_time():
    return time.strftime('%m-%d-%H-%M', time.localtime())


def plot_prediction(y, y_pred, save_mode='wandb', **kwargs):
    window_size_x, window_size_y = y_pred.shape[2], y_pred.shape[1]
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size_x), np.linspace(0, 1, window_size_y))
    fig, axs = plt.subplots(3, 1, figsize=(5*window_size_x/window_size_y, 3*5))
    axs[0].contourf(xx, yy, y.cpu().detach().reshape(window_size_y, window_size_x), levels=np.linspace(0, 1, 100), cmap='plasma')
    axs[0].set_title('(a) Ground truth')
    axs[0].axis('off')
    axs[1].contourf(xx, yy, y_pred.cpu().reshape(window_size_y, window_size_x), levels=np.linspace(0, 1, 100), cmap='plasma')
    axs[1].set_title('(b) Prediction')
    axs[1].axis('off')
    axs[2].contourf(xx, yy, np.abs(y.cpu().reshape(window_size_y, window_size_x) - y_pred.cpu().reshape(window_size_y, window_size_x)) / y.cpu().reshape(window_size_y, window_size_x), levels=np.linspace(0, 1, 100), cmap='plasma')
    axs[2].set_title('(c) Absolute difference by percentage')
    axs[2].axis('off')
    # add colorbar and labels to the rightmost plot
    cbar = plt.colorbar(axs[2].collections[0], ax=axs[2], orientation='vertical')
    cbar.set_label('Velocity magnitude (normalized)')
    plt.tight_layout()

    # plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    if save_mode == 'wandb':
        wandb.log({'prediction': wandb.Image(plt)})
    elif save_mode == 'plt':
        plt.show()
    elif save_mode == 'save':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.pdf', format='pdf', dpi=1200)
    elif save_mode == 'save_png':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.png', format='png', dpi=300)
    plt.close()


def plot_partition(y, y_pred, labels, sub_size, save_mode='wandb', **kwargs):
    # cover a colored mask on the prediction indicating the partition
    window_size_x, window_size_y = y_pred.shape[2], y_pred.shape[1]
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size_x), np.linspace(0, 1, window_size_y))
    fig, axs = plt.subplots(3, 1, figsize=(5*window_size_x/window_size_y, 3*5))

    colormap = plt.cm.tab20

    mask = np.zeros((window_size_y, window_size_x))
    # for i in range(window_size_x - sub_size + 1):
    #     for j in range(window_size_y - sub_size + 1):
    #         mask[j:j + sub_size, i:i + sub_size] = labels[i * (window_size_y - sub_size + 1) + j]
    for i in range(window_size_y // sub_size):
        for j in range(window_size_x // sub_size):
            mask[j * sub_size:(j + 1) * sub_size, i * sub_size:(i + 1) * sub_size] = labels[i * (window_size_y // sub_size) + j]

    # revert y axis of mask
    mask = np.flip(mask, axis=0)

    # axs[0].contourf(xx, yy, y_pred.cpu().detach().reshape(window_size_y, window_size_x), levels=100, cmap='plasma')
    axs[0].contourf(xx, yy, y_pred.cpu().squeeze(0).squeeze(-1), levels=np.linspace(0, 1, 100), cmap='plasma')
    axs[0].set_title('(a) Prediction')
    axs[0].axis('off')
    # axs[0].imshow(mask, cmap='tab20', alpha=0.1, interpolation='none')
    # for i in range(int(window_size / sub_size)):
    #     for j in range(int(window_size / sub_size)):
    #         rect = mpatches.Rectangle((j * sub_size / window_size, i * sub_size / window_size), sub_size / window_size, sub_size / window_size, facecolor=colormap(labels[i * int(window_size / sub_size) + j]), edgecolor='none', alpha=0.2)
    #         axs[0].add_patch(rect)

    # axs[1].contourf(xx, yy, np.abs(y.cpu().reshape(window_size_y, window_size_x) - y_pred.cpu().reshape(window_size_y, window_size_x)) / y.cpu().reshape(window_size_y, window_size_x), levels=np.linspace(0, 1, 100), cmap='plasma')
    axs[1].contourf(xx, yy, np.abs(y.squeeze(0).squeeze(-1).cpu() - y_pred.cpu().squeeze(0).squeeze(-1)) / y.cpu().squeeze(0).squeeze(-1), levels=np.linspace(0, 1, 100), cmap='plasma')
    axs[1].set_title('(b) Absolute difference by percentage')
    axs[1].axis('off')
    # axs[1].imshow(mask, cmap='tab20', alpha=0.1, interpolation='none')
    # for i in range(int(window_size / sub_size)):
    #     for j in range(int(window_size / sub_size)):
    #         rect = mpatches.Rectangle((j * sub_size / window_size, i * sub_size / window_size), sub_size / window_size, sub_size / window_size, facecolor=colormap(labels[i * int(window_size / sub_size) + j]), edgecolor='none')
    #         axs[1].add_patch(rect)

    axs[2].imshow(mask, cmap='tab20', interpolation='none')
    # add legend to show which color corresponds to which partition
    patches = [mpatches.Patch(color=colormap(i), label=f'Partition {i}') for i in range(len(np.unique(labels)))]
    axs[2].legend(handles=patches, loc='upper right')
    
    # add colorbar and labels to the rightmost plot
    # cbar = plt.colorbar(axs[1].collections[0], ax=axs[1], orientation='vertical')
    # cbar.set_label('Absolute difference')
    # # plt.tight_layout()
    

    if save_mode == 'wandb':
        wandb.log({'prediction': wandb.Image(plt)})
    elif save_mode == 'plt':
        plt.show()
    elif save_mode == 'save':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] + '.pdf', format='pdf', dpi=1200)
    elif save_mode == 'save_png':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] + '.png', format='png', dpi=300)
    plt.close()
    

def init_encoder(type, n_components, **kwargs):
    if type == 'pca':
        return PCAEncoder(n_components=n_components)
    elif type == 'vae':
        return VAEEncoder(n_components=n_components, **kwargs)
    elif type == 'spectrum':
        return SpectrumEncoder(n_components=n_components, **kwargs)
    else:
        raise ValueError(f'Invalid encoder type: {type}')
    

def init_classifier(type, n_clusters, **kwargs):
    if type == 'kmeans':
        return KMeansClassifier(n_clusters=n_clusters)
    if type == 'mean_shift':
        return MeanShiftClassifier()
    if type == 'wasserstein':
        return WassersteinKMeansClassifier(n_clusters=n_clusters)
    else:
        raise ValueError(f'Invalid classifier type: {type}')
    

def init_model(type, in_channels, out_channels, **kwargs):
    if type == 'fno':
        return FNO2d(in_channels, out_channels, **kwargs)
    elif type == 'teecnet':
        return TEECNet(in_channels, out_channels=out_channels, **kwargs)
    elif type == 'beno':
        return HeteroGNS(in_channels, out_channels, **kwargs)
    elif type == 'deeponet':
        # return DeepONet(in_channels, kwargs['trunk_size'], activation=kwargs['activation'], \
        #                 kernel_initializer=kwargs['kernel_initializer'], num_outputs=out_channels)
        return DeepONet(in_channels, kwargs['trunk_size'], hidden_dim=kwargs['width'], output_dim=out_channels)
    elif type == 'graphsage':
        return GraphSAGE(in_channels, out_channels, num_layers=5)
    elif type == 'neuralop':
        return KernelNN(width=kwargs['width'], ker_width=kwargs['width'], depth=kwargs['num_layers'], in_width=in_channels, out_width=out_channels)
    else:
        raise ValueError(f'Invalid model type: {type}')
    

def init_dataset(name, **kwargs):
    if name == 'jhtdb':
        return JHTDB_ICML(**kwargs)
    elif name == 'transition_bl':
        return JHTDB_RECTANGULAR(**kwargs)
    elif name == 'transition_bc':
        return JHTDB_RECTANGULAR_BOUNDARY(**kwargs)
    elif name == 'jhtdb_bc':
        return JHTDB_BOUNDARY(**kwargs)
    elif name == 'duct':
        return DuctAnalysisDataset(**kwargs)
    else:
        raise ValueError(f'Invalid dataset name: {name}')


def plot_3d_prediction(y_pred, save_mode='wandb', **kwargs):
    position = y_pred.pos.cpu().detach().numpy()
    # projection 3d
    fig = plt.figure(figsize=(20, 5))
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.scatter(position[:, 0], position[:, 1], position[:, 2], c=torch.norm(y_pred.x[:, :1], dim=1).cpu().detach().numpy(), cmap='plasma')
    # ax0.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.x[:, 0].cpu().detach().numpy(), y_pred.x[:, 1].cpu().detach().numpy(), y_pred.x[:, 2].cpu().detach().numpy(), length=torch.norm(y_pred.x[:, :3], dim=1).cpu().detach().numpy(), normalize=True)
    ax0.set_title('Input')
    ax0.axis('off')
    plt.colorbar(ax0.collections[0], ax=ax0, orientation='vertical')

    ax1 = fig.add_subplot(132, projection='3d')
    ax1.scatter(position[:, 0], position[:, 1], position[:, 2], c=torch.norm(y_pred.y[:, :1], dim=1).cpu().detach().numpy(), cmap='plasma')
    # ax1.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.y[:, 0].cpu().detach().numpy(), y_pred.y[:, 1].cpu().detach().numpy(), y_pred.y[:, 2].cpu().detach().numpy(), length=torch.norm(y_pred.y[:, :3], dim=1).cpu().detach().numpy(), normalize=True)
    ax1.set_title('Ground truth')
    ax1.axis('off')
    plt.colorbar(ax1.collections[0], ax=ax1, orientation='vertical')

    ax2 = fig.add_subplot(133, projection='3d')
    ax2.scatter(position[:, 0], position[:, 1], position[:, 2], c=torch.norm(y_pred.pred[:, :1], dim=1).cpu().detach().numpy(), cmap='plasma')
    # ax2.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.pred[:, 0].cpu().detach().numpy(), y_pred.pred[:, 1].cpu().detach().numpy(), y_pred.pred[:, 2].cpu().detach().numpy(), length=torch.norm(y_pred.pred[:, :3], dim=1).cpu().detach().numpy(), normalize=True)
    ax2.set_title('Prediction')
    ax2.axis('off')
    plt.colorbar(ax2.collections[0], ax=ax2, orientation='vertical')

    # ax2 = fig.add_subplot(133, projection='3d')
    # ax2.scatter(position[:, 0], position[:, 1], position[:, 2], c=np.abs(torch.norm(y_pred.x, dim=1).cpu().detach().numpy() - torch.norm(y_pred.y, dim=1).cpu().detach().numpy()), cmap='plasma')
    # ax2.quiver(position[:, 0], position[:, 1], position[:, 2], y_pred.x[:, 0].cpu().detach().numpy() - y_pred.y[:, 0].cpu().detach().numpy(), y_pred.x[:, 1].cpu().detach().numpy() - y_pred.y[:, 1].cpu().detach().numpy(), y_pred.x[:, 2].cpu().detach().numpy() - y_pred.y[:, 2].cpu().detach().numpy(), length=0.1, normalize=True)
    # ax2.set_title('Absolute difference')

    if save_mode == 'wandb':
        wandb.log({'prediction': wandb.Image(plt)})
    elif save_mode == 'plt':
        plt.show()
    elif save_mode == 'save':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.pdf', format='pdf', dpi=300)
    elif save_mode == 'save_png':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.png', format='png', dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ALDS experiment')
    parser.add_argument('--dataset', type=str, default='duct', help='Name of the dataset')
    parser.add_argument('--encoder', type=str, default='pca', help='Name of the encoder')
    parser.add_argument('--classifier', type=str, default='kmeans', help='Name of the classifier')
    parser.add_argument('--model', type=str, default='neuralop', help='Name of the model')
    parser.add_argument('--exp_name', type=str, default='collection_duct_neuralop', help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train', help='Mode of the experiment')
    parser.add_argument('--exp_config', type=str, default='configs/exp_config/teecnet_duct.yaml', help='Path to the experiment configuration file')
    parser.add_argument('--train_config', type=str, default='configs/train_config/teecnet.yaml', help='Path to the training configuration file')
    args = parser.parse_args()
    return args


@jit(nopython=True)
def compute_tke_spectrum(u, lx, ly):
    """
    Given velocity fields u and v, computes the turbulent kinetic energy spectrum. The function computes in three steps:
    1. Compute velocity spectrum with fft, returns uf, vf.
    2. Compute the point-wise turbulent kinetic energy Ef=0.5*(uf, vf)*conj(uf, vf).
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).
    """
    nx = u.shape[0]
    ny = u.shape[1]

    nt = nx * ny
    # Compute velocity spectrum
    uf = torch.fft.fftn(u, norm='ortho')

    # Compute the point-wise turbulent kinetic energy
    Ef = 0.5 * (uf * torch.conj(uf)).real
    kx = 2 * torch.pi / lx 
    ky = 2 * torch.pi / ly
    knorm = np.sqrt(kx ** 2 + ky ** 2)
    kxmax = nx / 2
    kymax = ny / 2
    wave_numbers = knorm * torch.arange(0, nx)
    tke_spectrum = torch.zeros(nx)
    for i in range(nx):
        rkx = i
        if i > kxmax:
            rkx = rkx - nx
        for j in range(ny):
            rky = j
            if j > kymax:
                rky = rky - ny
            rk = np.sqrt(rkx * rkx + rky * rky)
            k_index = int(np.round(rk))
            tke_spectrum[k_index] += Ef[i, j]
    # k = torch.fft.fftfreq(nx, lx / nx)

    # plt.loglog(wave_numbers[1:], tke_spectrum[1:])
    # plt.savefig('tke_spectrum.png')
    return tke_spectrum[1:], wave_numbers[1:]

def plot_3d_partition(data, y_pred, labels, save_mode='wandb', **kwargs):
    position = data.pos.cpu().detach().numpy()
    # projection 3d
    fig, axs = plt.subplots(3, 1, projection='3d', figsize=(40, 15))
    colormap = plt.cm.tab20
    for i in range(len(np.unique(labels))):
        mask = labels == i
        axs[0].scatter(position[mask, 0], position[mask, 1], position[mask, 2], c=y_pred[mask].cpu().detach().numpy(), cmap='plasma')
        axs[1].scatter(position[mask, 0], position[mask, 1], position[mask, 2], c=data.y[mask].cpu().detach().numpy(), cmap='plasma')
        axs[2].scatter(position[mask, 0], position[mask, 1], position[mask, 2], c=np.abs(data.y[mask].cpu().detach().numpy() - y_pred[mask].cpu().detach().numpy()), cmap='plasma')

    if save_mode == 'wandb':
        wandb.log({'prediction': wandb.Image(plt)})
    elif save_mode == 'plt':
        plt.show()
    elif save_mode == 'save':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.pdf', format='pdf', dpi=300)


def save_pyg_to_vtk(data, mesh_path, save_path):
    # save the prediction data to vtk file
    # data: pytorch geometric data object
    # mesh_path: path to the original mesh data
    # save_path: path to save the vtk file
    reader = vtk.vtkFLUENTReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    mesh = reader.GetOutput().GetBlock(0)

    # create a new vtk unstructured grid
    grid = vtk.vtkUnstructuredGrid()
    grid.DeepCopy(mesh)

    # add the prediction data to the grid
    pred = data.pred.cpu().detach().numpy()
    pred = np.expand_dims(pred, axis=1)
    pred = np.concatenate([pred, pred, pred], axis=1)
    pred = pred.flatten()
    pred = np.ascontiguousarray(pred, dtype=np.float64)
    vtk_pred = vtk.vtkDoubleArray()
    vtk_pred.SetNumberOfComponents(3)
    vtk_pred.SetNumberOfTuples(len(pred) // 3)
    vtk_pred.SetArray(pred, len(pred), 1)
    vtk_pred.SetName('prediction')
    grid.GetPointData().AddArray(vtk_pred)

    # write the grid to vtk file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(grid)
    writer.Write()
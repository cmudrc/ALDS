import os
import argparse
from models.classifier import *
from models.encoder import *
from models.model import *
from models.scheduler import *
from dataset.MatDataset import *
import yaml
import matplotlib.pyplot as plt
import wandb


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def plot_prediction(y, y_pred, save_mode='wandb', **kwargs):
    window_size = y.shape[-1]
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].contourf(xx, yy, y.cpu().detach().numpy().reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[0].set_title('(a) Ground truth')
    axs[0].axis('off')
    axs[1].contourf(xx, yy, y_pred.reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[1].set_title('(b) Prediction')
    axs[1].axis('off')
    axs[2].contourf(xx, yy, np.abs(y.cpu().detach().numpy().reshape(window_size, window_size) - y_pred.reshape(window_size, window_size)), levels=100, cmap='plasma')
    axs[2].set_title('(c) Absolute difference by percentage')
    axs[2].axis('off')

    # plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    if save_mode == 'wandb':
        wandb.log({'prediction': wandb.Image(plt)})
    elif save_mode == 'plt':
        plt.show()
    elif save_mode == 'save':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'], format='svg', dpi=1200)
    plt.close()


def init_encoder(type, n_components, **kwargs):
    if type == 'pca':
        return PCAEncoder(n_components=n_components)
    elif type == 'vae':
        return VAEEncoder(n_components=n_components, **kwargs)
    else:
        raise ValueError(f'Invalid encoder type: {type}')
    

def init_classifier(type, n_clusters, **kwargs):
    if type == 'kmeans':
        return KMeansClassifier(n_clusters=n_clusters)
    else:
        raise ValueError(f'Invalid classifier type: {type}')
    

def init_model(type, in_channels, out_channels, **kwargs):
    if type == 'fno':
        return FNO2d(in_channels, out_channels, **kwargs)
    elif type == 'teecnet':
        return TEECNetConv(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f'Invalid model type: {type}')
    

def init_dataset(name, **kwargs):
    if name == 'jhtdb':
        return JHTDB_ICML(**kwargs)
    else:
        raise ValueError(f'Invalid dataset name: {name}')
    

def parse_args():
    parser = argparse.ArgumentParser(description='Run ALDS experiment')
    parser.add_argument('--dataset', type=str, default='jhtdb', help='Name of the dataset')
    parser.add_argument('--encoder', type=str, default='pca', help='Name of the encoder')
    parser.add_argument('--classifier', type=str, default='kmeans', help='Name of the classifier')
    parser.add_argument('--model', type=str, default='fno', help='Name of the model')
    parser.add_argument('--exp_name', type=str, default='fno_jhtdb', help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train', help='Mode of the experiment')
    parser.add_argument('--exp_config', type=str, default='configs/exp_config/fno_jhtdb.yaml', help='Path to the experiment configuration file')
    parser.add_argument('--train_config', type=str, default='configs/train_config/fno.yaml', help='Path to the training configuration file')
    args = parser.parse_args()
    return args
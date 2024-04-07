import os

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


def plot_prediction(y, y_pred):
    window_size = y.shape[-1]
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].contourf(xx, yy, y.cpu().detach().numpy().reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[0].set_title('(a) Ground truth')
    axs[0].axis('off')
    axs[1].contourf(xx, yy, y_pred.reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[1].set_title('(b) Prediction')
    axs[1].axis('off')
    axs[2].contourf(xx, yy, np.abs(y.cpu().detach().numpy().reshape(window_size, window_size) - y_pred.reshape(window_size, window_size)), levels=100, cmap='plasma')
    axs[2].set_title('(c) Absolute difference')
    axs[2].axis('off')

    # plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    wandb.log({'prediction': wandb.Image(plt)})
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
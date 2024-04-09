import os
# import h5py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from dataset.MatDataset import JHTDB_ICML
from models.scheduler import PartitionScheduler
from models.encoder import PCAEncoder
from models.classifier import KMeansClassifier
from models.model import FNO2d

from utils import load_yaml


def plot_prediction(y, y_pred):
    window_size = y.shape[0]
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].contourf(xx, yy, y.cpu().detach().numpy().reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[0].set_title('(a) Ground truth')
    axs[0].axis('off')
    axs[1].contourf(xx, yy, y_pred.reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[1].set_title('(b) Prediction')
    axs[1].axis('off')
    axs[2].contourf(xx, yy, np.abs(y.reshape(window_size, window_size) - y_pred.reshape(window_size, window_size)), levels=100, cmap='plasma')
    axs[2].set_title('(c) Absolute difference')
    axs[2].axis('off')

    # plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    plt.savefig('logs/jhtdb_icml.png')
    plt.close()

if __name__ == '__main__':
    train_config = load_yaml('configs/train_config.yaml')
    dataset = JHTDB_ICML(root='data/jhtdb_icml', tstart=1, tend=500, fields='u', dataset='isotropic1024coarse', partition=True, sub_size=64)
    encoder = PCAEncoder(n_components=8)
    classifier = KMeansClassifier(n_clusters=8)
    model = FNO2d(8, 8, 36)

    # scheduler = PartitionScheduler('fno_jhtdb', 8, dataset, encoder, classifier, model)
    # scheduler.train(train_config)

    ###################################################################################################
    # Test
    ###################################################################################################
    scheduler = PartitionScheduler('fno_jhtdb', 8, dataset, encoder, classifier, model, train=False)
    x, sub_x_list, sub_y_list = dataset.get_one_full_sample(100)
    sub_x_tensor = torch.stack(sub_x_list)
    pred_y_list = scheduler.predict(sub_x_tensor)
    
    pred_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), pred_y_list)
    sub_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), sub_y_list)

    plot_prediction(sub_y.squeeze(0), pred_y.squeeze(0))

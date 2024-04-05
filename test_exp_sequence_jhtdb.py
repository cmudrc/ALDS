import os
# import h5py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml

from dataset.MatDataset import JHTDB_ICML
from models.scheduler import PartitionScheduler
from models.encoder import PCAEncoder
from models.classifier import KMeansClassifier
from models.model import FNO2d


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    train_config = load_yaml('configs/train_config.yaml')
    dataset = JHTDB_ICML(root='data/jhtdb_icml', tstart=1, tend=500, fields='u', dataset='isotropic1024coarse', partition=True, sub_size=64)
    encoder = PCAEncoder(n_components=8)
    classifier = KMeansClassifier(n_clusters=8)
    model = FNO2d(8, 8, 36)
    
    scheduler = PartitionScheduler(8, dataset, encoder, classifier, model)
    scheduler.train(train_config)
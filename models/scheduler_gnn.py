import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn    
from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F
from joblib import dump, load
import wandb
from dataset.MatDataset import Sub_JHTDB
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from models.model import *


class GNNPartitionScheduler():
    def __init__(self, exp_name, num_partitons, dataset, encoder, classifier, model=None, train=True):
        super(GNNPartitionScheduler, self).__init__()
        self.name = exp_name
        self.num_partitions = num_partitons
        self.encoder = encoder
        self.classifier = classifier
        self.model = model
        self.dataset = dataset

        self.subsets = self._train_partitions(num_partitons, train)
        if not train:
            self.models = self._load_models()

    def get_sub_dataset(self):
        return self.subsets
    
    def _initialize_model(self, model_type, in_channels, out_channels, **kwargs):
        return self.model
    
    def _load_models(self):
        models = []
        for i in range(self.num_partitions):
            model = self._initialize_model(self.model, 8, 8, width=64)
            model.load_state_dict(torch.load('logs/models/collection_{}/partition_{}.pth'.format(self.name, i), map_location=torch.device('cpu')))
            models.append(model)
        return models
    
    def _train_partitions(self, num_partitions, train):
        
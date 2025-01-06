import os
import numpy as np
import matplotlib.pyplot as plt
# import torch
from torch.utils.data import random_split
import torch.nn as nn
# import torch_geometric as pyg
# import torch_geometric.nn as pyg_nn    
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset, uniform
# import torch.nn.functional as F
from joblib import dump, load
import wandb
from dataset.MatDataset import Sub_JHTDB
# from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from models.model import *
from utils import *


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
        # if num_partitions == 1: skip the clustering and directly train the model
        if num_partitions == 1:
            return [self.dataset]
        if train:
            os.makedirs('logs/models/collection_{}'.format(self.name), exist_ok=True)
            # train the encoder on the dataset
            self.encoder.train(self.dataset, save_model=True, path='logs/models/collection_{}'.format(self.name))
            # dump(self.encoder.model, 'logs/models/collection_{}/encoder.joblib'.format(self.name))
            latent_space = self.encoder.get_latent_space(self.dataset)
            print('Latent space shape:', latent_space.shape)
            # cluster the latent space into different groups
            self.classifier.train(latent_space, save_model=True, path='logs/models/collection_{}'.format(self.name))
            # dump(self.classifier.model, 'logs/models/collection_{}/classifier.joblib'.format(self.name))
            labels = self.classifier.cluster(latent_space)
            # print('Labels:', labels)
        else:
            # load the pre-trained encoder and classifier
            self.encoder.load_model('logs/models/collection_{}'.format(self.name))
            self.classifier.load_model('logs/models/collection_{}'.format(self.name))
            latent_space = self.encoder.get_latent_space(self.dataset)
            labels = self.classifier.cluster(latent_space)

        # partition the dataset into different subsets
        subsets = []
        for i in range(num_partitions):
            idx = np.where(labels == i)[0]
            # print(f'Partition {i}: {len(idx)} samples')
            subsets.append(Sub_JHTDB(self.dataset.root, idx))

        return subsets
    
    def _train_sub_models(self, train_config, device, subset_idx=None, is_parallel=False):
        models = []
        if subset_idx is not None:
            subsets = [self.subsets[idx] for idx in subset_idx]
        else:
            subsets = self.subsets
        for i, subset in enumerate(subsets):
            # print(len(subset))
            wandb.init(project='domain_partition_scheduler', group='partition_training', config=train_config)
            train_dataset, val_dataset = random_split(subset, [int(0.8 * len(subset)), len(subset) - int(0.8 * len(subset))])
            # print(train_config['batch_size'])
            train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
            model = self._initialize_model(self.model, 8, 8, width=64)

            # model = model.to(device)
            if is_parallel:
                model = nn.DataParallel(model)
                model = model.to(device)
            else:
                model = model.to(device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_config['step_size'], gamma=train_config['gamma'])
            epochs = train_config['epochs']
            log_interval = train_config['log_interval']
            val_interval = train_config['val_interval']
            best_loss = np.inf
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                wandb.log({'train_loss': train_loss})
                if epoch % log_interval == 0:
                    print(f'Epoch {epoch}: Train loss: {train_loss}')
                if epoch % val_interval == 0:
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = batch.to(device)
                            out = model(batch.x, batch.edge_index, batch.edge_attr)
                            loss = criterion(out, batch.y)
                            val_loss += loss.item()
                        val_loss /= len(val_loader)
                        wandb.log({'val_loss': val_loss})
                        plot_3d_prediction(batch[0], out, save_mode='wandb', path='logs/figures/{}'.format(self.name))
                        if val_loss < best_loss:
                            best_loss = val_loss
                            os.makedirs('logs/models/collection_{}'.format(self.name), exist_ok=True)
                            torch.save(model.state_dict(), 'logs/models/collection_{}/partition_{}.pth'.format(self.name, i))
                            print(f'Epoch {epoch}: Validation loss: {val_loss}')
                scheduler.step()

            models.append(model)
            wandb.finish()
        return models
    
    def train(self, train_config, subset_idx=None):
        # for parallel training on multiple gpus
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            self._train_sub_models(train_config, torch.device('cuda'), subset_idx, is_parallel=True)
        elif torch.cuda.device_count() == 1:
            print('Using single GPU')
            self._train_sub_models(train_config, torch.device('cuda'), subset_idx, is_parallel=False)
        else:
            print('Using CPU')
            self._train_sub_models(train_config, torch.device('cpu'), subset_idx, is_parallel=False)

    def predict(self, x):
        # pred_y_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                for i in range(len(x)):
                    pred_y = model(x[i].x, x[i].edge_index, x[i].edge_attr)
                    x[i].x = pred_y
        return x
                 

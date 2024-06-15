import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.nn.init import uniform_ as reset
# import torch_geometric.nn as pyg_nn
# from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F
from joblib import dump, load
import wandb
from dataset.MatDataset import Sub_JHTDB
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from models.model import *


class PartitionScheduler():
    def __init__(self, exp_name, num_partitons, dataset, encoder, classifier, model=None, train=True):
        super(PartitionScheduler, self).__init__()
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
    
    def _train_partitions(self, num_partitions, train=True):
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
            wandb.init(project='domain_partition_scheduler', group='partition_training', config=train_config)
            train_dataset, val_dataset = random_split(subset, [int(0.8 * len(subset)), len(subset) - int(0.8 * len(subset))])
            train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
            model = self._initialize_model(self.model, 8, 8, width=64)
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

            for epoch in range(epochs):
                model.train()
                loss_epoch = 0
                for batch_idx, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    try:
                        x, y = x.to(device), y.to(device)
                        pred = model(x)
                    except:
                        x, boundary, y = x[0].to(device), x[1].to(device), y[:, :, :, 0].unsqueeze(-1).to(device)
                        pred = model(x, boundary)
                    # print(x.shape, y.shape)
                    
                    loss = criterion(pred, y)

                    loss_epoch += loss.item()
                    loss.backward()
                    optimizer.step()

                loss_epoch /= len(train_loader)
                scheduler.step()
                wandb.log({'loss': loss_epoch})
                if epoch % log_interval == 0:
                    print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss_epoch}')
                    
                if epoch % val_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0
                        for x, y in val_loader:
                            try:
                                x, y = x.to(device), y.to(device) 
                                pred = model(x)
                            except:
                                x, boundary, y = x[0].to(device), x[1].to(device), y[:, :, :, 0].unsqueeze(-1).to(device)
                                pred = model(x, boundary)

                            val_loss += criterion(pred, y).item()
                        val_loss /= len(val_loader)
                        wandb.log({'val_loss': val_loss})
                        print(f'Epoach: {epoch + 1}/{epochs}, Validation Loss: {val_loss}')

                        # plot one sample from the validation set
                        x, y = val_dataset[0]
                        try:
                            x, y = x.to(device), y.to(device)
                            pred = model(x)
                        except:
                            x, boundary, y = x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device), y[:, :, 0].unsqueeze(0).unsqueeze(-1).to(device)
                            pred = model(x, boundary)
                        self._plot_prediction(y, pred)
                        
                        # torch.save(model.state_dict(), f'logs/models/partition_{i}_epoch_{epoch}.pth')
                
                # register the model in a model collection
                # os.makedirs('logs/models/collection_{}'.format({self.name}), exist_ok=True)
                # model_scripted = torch.jit.script(model)
                # model_scripted.save('logs/models/collection_{}/partition_{}.pt'.format(subset_idx, i, epoch))
                torch.save(model.state_dict(), 'logs/models/collection_{}/partition_{}.pth'.format(self.name, i))
                models.append(model)

            wandb.finish()
        self.models = models

    def _plot_prediction(self, y, y_pred):
        window_size = y.shape[1]
        xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].contourf(xx, yy, y.cpu().detach().reshape(window_size, window_size), levels=100, cmap='plasma')
        axs[0].set_title('(a) Ground truth')
        axs[0].axis('off')
        axs[1].contourf(xx, yy, y_pred.cpu().reshape(window_size, window_size), levels=100, cmap='plasma')
        axs[1].set_title('(b) Prediction')
        axs[1].axis('off')
        axs[2].contourf(xx, yy, np.abs(y.cpu().reshape(window_size, window_size) - y_pred.cpu().reshape(window_size, window_size)) / y.cpu().reshape(window_size, window_size), levels=100, cmap='plasma')
        axs[2].set_title('(c) Absolute difference by percentage')
        axs[2].axis('off')
        # add colorbar and labels to the rightmost plot
        cbar = plt.colorbar(axs[2].collections[0], ax=axs[2], orientation='vertical')
        cbar.set_label('Absolute difference')
        plt.tight_layout()

        wandb.log({'prediction': wandb.Image(plt)})

    def train(self, train_config, subset_idx=None):
        # for parallel training on multiple gpus
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            self._train_sub_models(train_config, torch.device('cuda'), subset_idx, is_parallel=True)
        else:
            print('Using single GPU')
            self._train_sub_models(train_config, torch.device('cuda'), subset_idx, is_parallel=False)

    def predict(self, x, max_subset_size=10000):
        # x = torch.tensor(x, dtype=torch.float32)
        # see if self.models is available
        if not hasattr(self, 'models'):
            raise ValueError('Models are not trained yet')
        latent_space = self.encoder.get_latent(x)
        labels = self.classifier.cluster(latent_space)
        predictions = torch.zeros_like(x)
        # get all subsets
        num_subsets = 0
        x_subsets = []
        subsets_idx_mask = []
        model_idx = []
        for i in range(self.num_partitions):
            idx = np.where(labels == i)[0]
            # check if the subset exceeds the maximum size
            if len(idx) > max_subset_size:
                num_cur_subsets = len(idx) // max_subset_size
                for j in range(num_cur_subsets):
                    x_subsets.append(x[idx[j*max_subset_size:(j+1)*max_subset_size]])
                    subsets_idx_mask.append(idx[j*max_subset_size:(j+1)*max_subset_size])
                    model_idx.append(i)
                num_subsets += num_cur_subsets
                if len(idx) % max_subset_size != 0:
                    x_subsets.append(x[idx[num_cur_subsets*max_subset_size:]])
                    subsets_idx_mask.append(idx[num_cur_subsets*max_subset_size:])
                    num_subsets += 1
                    model_idx.append(i)
            else:
                x_subsets.append(x[idx])
                subsets_idx_mask.append(idx)
                num_subsets += 1
                model_idx.append(i)
        # print(len(x_subsets), len(subsets_idx_mask))
        print(f'Predicting on {len(x_subsets)} subsets')
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]          
            # assign model and corresponding data to different gpus and predict in parallel
            for num_execs in range(num_subsets // torch.cuda.device_count() + 1):
                exec_list = []
                with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
                    for i, device in enumerate(device_list):
                        idx = num_execs * torch.cuda.device_count() + i
                        if idx >= num_subsets:
                            break 
                        cur_subset = x_subsets[idx].detach().clone().to(device)
                        # print(cur_subset.device)
                        cur_model = self.models[model_idx[idx]].to(device)
                        # print(cur_model.mlp0.mlp1.weight.device)
                        pred = executor.submit(self._predict_sub_model, cur_model, cur_subset, idx)
                        exec_list.append(pred)
                    
                    print(f'Waiting for {len(exec_list)} threads to finish')
                    # batch_idx = num_execs * torch.cuda.device_count()
                    complete_pred, incomplete_pred = wait(exec_list, return_when=ALL_COMPLETED)
                    for pred in complete_pred:
                        # idx = subsets_idx_mask[batch_idx]
                        cur_pred = pred.result().detach().clone()
                        cur_idx = int(cur_pred[:, 1].max().item())
                        # print(cur_idx)
                        cur_pred = cur_pred[:, 0]
                        idx = subsets_idx_mask[cur_idx]
                        predictions[idx] = cur_pred
                        # predictions[subsets_idx_mask[idx+1]] = cur_pred
                        # batch_idx += 1

                    # wait for threads to finish before submitting the next batch
                    # executor.shutdown(wait=True)

        else:
            print('Using single GPU')
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            for i in range(num_subsets):
                cur_subset = x_subsets[i].detach().clone().to(device)
                cur_model = self.models[model_idx[i]].to(device)
                pred = self._predict_sub_model(cur_model, cur_subset)
                cur_pred = pred.detach().cpu()
                idx = subsets_idx_mask[i]
                predictions[idx] = cur_pred

        return predictions, labels
    
    def recurrent_predict(self, x, x_list, num_iters):
        all_predictions = []
        all_labels = []
        predictions = x_list
        for i in range(num_iters):
            # print(i)
            pred, _ = self.predict(predictions)
            predictions = pred.cpu().clone()
            prediction_reconstructed = self.dataset.reconstruct_from_partitions(x.unsqueeze(0), predictions)
            # print(prediction_reconstructed.shape)
            all_predictions.append(predictions)
            all_labels.append(_)
            predictions = self.dataset.get_partition_domain(prediction_reconstructed.squeeze(0), mode='test')
            predictions = torch.stack(predictions)

        return all_predictions, all_labels
    
    def _predict_sub_model(self, model, x, idx=None):
        # print(f'Predicting on {device}, with tensor on {x.device}')
        model.eval()
        with torch.no_grad():
            pred = model(x).cpu()
        if idx is not None:
            # concatenate idx to the prediction
            pred = torch.stack([pred, torch.ones_like(pred) * idx], dim=1)
        return pred

    def evaluate_sub_models(self, subset_idx=None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if subset_idx is not None:
            subsets = self.subsets[subset_idx]
        else:
            subsets = self.subsets
        for j in range(len(subsets)):
            for i in range(self.num_partitions):
                subset = subsets[j]
                model = self.models[i].to(device)
                criterion = torch.nn.MSELoss()
                val_loader = DataLoader(subset, batch_size=1, shuffle=False)
                model.eval()
                val_loss = 0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    val_loss += criterion(pred, y).item()
                val_loss /= len(val_loader)
                print(f'Subset {j}, Model {i}, Validation Loss: {val_loss}')
            # wandb.log({'val_loss': val_loss})
            # wandb.finish() 
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
import wandb
from dataset.MatDataset import Sub_JHTDB


class PartitionScheduler():
    def __init__(self, num_partitons, dataset, encoder, classifier, model=None):
        super(PartitionScheduler, self).__init__()
        self.num_partitions = num_partitons
        self.encoder = encoder
        self.classifier = classifier
        self.model = model
        self.dataset = dataset

        self.subsets = self._train_partitions(num_partitons)

    def get_sub_dataset(self):
        return self.subsets
    
    def _train_partitions(self, num_partitions):
        # train the encoder on the dataset
        self.encoder.train(self.dataset)
        latent_space = self.encoder.get_latent_space(self.dataset)

        # cluster the latent space into different groups
        self.classifier.train(latent_space)
        labels = self.classifier.cluster(latent_space)

        # partition the dataset into different subsets
        subsets = []
        for i in range(num_partitions):
            idx = np.where(labels == i)[0]
            # print(f'Partition {i}: {len(idx)} samples')
            subsets.append(Sub_JHTDB(self.dataset.root, idx))

        return subsets

    def _train_sub_models(self, train_config, device, subset_idx=None, is_parallel=False):
        if subset_idx is not None:
            subsets = self.subsets[subset_idx]
        else:
            subsets = self.subsets
        for i, subset in enumerate(subsets):
            wandb.init(project='domain_partition_scheduler', group='partition_training', config=train_config)
            train_dataset, val_dataset = random_split(subset, [int(0.8 * len(subset)), len(subset) - int(0.8 * len(subset))])
            train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
            if is_parallel:
                model = nn.DataParallel(self.model)
                model = model.to(device)
            else:
                model = self.model.to(device)

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
                    x, y = x.to(device), y.to(device)
                    print(x.shape, y.shape)
                    optimizer.zero_grad()
                    pred = model(x)
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
                            x, y = x.to(device), y.to(device)
                            pred = model(x)
                            val_loss += criterion(pred, y).item()
                        val_loss /= len(val_loader)
                        wandb.log({'val_loss': val_loss})
                        print(f'Epoach: {epoch + 1}/{epochs}, Validation Loss: {val_loss}')

                        # plot one sample from the validation set
                        x, y = val_dataset[0]
                        x = x.unsqueeze(0).to(device)
                        pred = model(x).squeeze(0)
                        self._plot_prediction(64, y.cpu(), pred.cpu(), epoch, batch_idx, 'results')
                        torch.save(model.state_dict(), f'logs/models/partition_{i}_epoch_{epoch}.pth')
                
                # register the model in a model collection
                os.makedirs('logs/models/collection_{}'.format(subset_idx), exist_ok=True)
                model_scripted = torch.jit.script(model)
                model_scripted.save('logs/models/collection_{}/partition_{}.pt'.format(subset_idx, i, epoch))

    def _plot_prediction(self, window_size, y, y_pred, epoch, batch_idx, folder):
        xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
        fig = plt.figure()
        plt.contourf(xx, yy, y.cpu().detach().numpy().reshape(window_size, window_size), levels=100, cmap='plasma')
        plt.axis('off')
        plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close()

    def train(self, train_config, subset_idx=None):
        # for parallel training on multiple gpus
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            self._train_sub_models(train_config, torch.device('cuda'), subset_idx, is_parallel=True)
        else:
            print('Using single GPU')
            self._train_sub_models(train_config, torch.device('cuda'), subset_idx, is_parallel=False)

            
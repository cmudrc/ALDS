import os
# import h5py

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dataset.MatDataset import JHTDB
from models.scheduler import TBVAE


def plot_prediction(window_size, y, y_pred, epoch, batch_idx, folder):
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].contourf(xx, yy, y.reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[0].set_title('(a) Ground truth')
    axs[0].axis('off')
    axs[1].contourf(xx, yy, y_pred.reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[1].set_title('(b) Prediction')
    axs[1].axis('off')
    axs[2].contourf(xx, yy, np.abs(y.reshape(window_size, window_size) - y_pred.reshape(window_size, window_size)), levels=100, cmap='plasma')
    axs[2].set_title('(c) Absolute difference')
    axs[2].axis('off')

    # plt.colorbar(axs[2].contourf(xx, yy, np.abs(y.cpu().detach().numpy().reshape(window_size, window_size) - y_pred.reshape(window_size, window_size)), levels=100, cmap='plasma'), ax=axs[2], pad=0.01)

    # plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    wandb.log({'prediction': wandb.Image(plt)})
    plt.close()

def test_exp_sequence():
    # Set the random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the root directory
    root = os.path.join(os.getcwd(), 'data', 'jhtdb')
    
    # Set the dataset
    dataset = JHTDB(root, tstart=1, tend=500, fields='u', dataset='isotropic1024coarse', partition=True, sub_size=64)
    
    # Set the model
    model = TBVAE(input_dim=1, latent_dim=32, hidden_dim=256, num_classes=1, num_layers=3, dropout=0.5).to(device)

    # Set the loss function
    criterion = torch.nn.MSELoss()

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set the number of epochs
    epochs = 100

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        loss_epoch = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            y_pred = model.decode(z)
            loss = criterion(y_pred, y)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        loss_epoch /= len(train_loader)
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss_epoch}')
        wandb.log({'loss': loss_epoch})
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                x, y = dataset[0]
                x, y = x.to(device), y.to(device)
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar)
                y_pred = model.decode(z)
                plot_prediction(64, y.cpu(), y_pred.cpu(), epoch, batch_idx, 'results')
                # save model
                torch.save(model.state_dict(), 'logs/models/TBVAE_model.pth')
                

if __name__ == '__main__':
    wandb.init(project='domain_partition_scheduler', group='testings')
    test_exp_sequence()
    print('Done!')
import os
# import h5py

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dataset.MatDataset import JHTDB
from models.teecnet import TEECNetConv


def plot_prediction(window_size, y, y_pred, epoch, batch_idx, folder):
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
    dataset = JHTDB(root, tstart=1, tend=100, fields='u', dataset='isotropic1024coarse')
    
    # Set the model
    model = TEECNetConv(1, 32, 1, num_layers=6, retrieve_weights=False, num_powers=3).to(device)

    # Set the loss function
    criterion = torch.nn.MSELoss()

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set the number of epochs
    num_epochs = 20

    # Set the batch size
    batch_size = 32

    # Set the data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        loss_epoch = 0

        for i, data in enumerate(data_loader):
            sub_x_list, sub_y_list = model.get_partition_domain(data[0], mode='train'), model.get_partition_domain(data[1], mode='test')
            # output_list = []
            for sub_x, sub_y in zip(sub_x_list, sub_y_list):
                sub_x, sub_y = sub_x.to(device), sub_y.to(device)

                # Forward pass
                outputs = model(sub_x)
                # output_list.append(outputs)

                # Compute the loss
                loss = criterion(outputs, sub_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the loss
                loss_epoch += loss.item()

            # Print the loss
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], Loss: {loss.item()}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch / (len(data_loader) * len(sub_x_list))}')
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch / len(data_loader)}')
        wandb.log({'loss': loss_epoch / len(data_loader) * len(sub_x_list)})
        plot_prediction(81, sub_y, outputs, epoch, i, 'test')

    # Save the model
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == '__main__':
    # wandb.init(project='domain_partition_teecnet', group='testings')
    test_exp_sequence()
    print('Done!')
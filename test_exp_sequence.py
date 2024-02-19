import os
# import h5py

import torch
import numpy as np
# import wandb

from dataset.MatDataset import BurgersDataset
from models.teecnet import TEECNetConv


def test_exp_sequence():
    # Set the random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the root directory
    root = os.path.join(os.getcwd(), 'data', 'burgers')
    
    # Set the dataset
    dataset = BurgersDataset(root)
    
    # Set the model
    model = TEECNetConv(1, 16, 1, num_layers=6, retrieve_weights=False, num_powers=3).to(device)

    # Set the loss function
    criterion = torch.nn.MSELoss()

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set the number of epochs
    num_epochs = 10

    # Set the batch size
    batch_size = 32

    # Set the data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        loss_epoch = 0

        for i, data in enumerate(data_loader):
            sub_x_list, sub_y_list = model.get_partition_domain(data[0], mode='train'), model.get_partition_domain(data[1], mode='test')

            for sub_x, sub_y in zip(sub_x_list, sub_y_list):
                # inputs, labels = data
                inputs, labels = sub_x.to(device), sub_y.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                        
                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item()}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch / (len(data_loader) * len(sub_x_list))}')

    # Save the model
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == '__main__':
    test_exp_sequence()
    print('Done!')
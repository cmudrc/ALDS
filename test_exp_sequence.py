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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the number of epochs
    num_epochs = 1

    # Set the batch size
    batch_size = 8

    # Set the data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == '__main__':
    test_exp_sequence()
    print('Done!')
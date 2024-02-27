from dataset.MatDataset import BurgersDataset
import torch
from models.teecnet import TEECNetConv
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import os


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

    plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    # wandb.log({'prediction': wandb.Image(plt)})
    plt.close()


sub_x_total = torch.load('sub_x_total.pt')
# sub_y_total = torch.load('sub_y_total.pt')
fft_x_list = torch.load('fft_x_list.pt')

sub_x_total = torch.tensor(sub_x_total, dtype=torch.float).squeeze()
fft_max = torch.max(fft_x_list)

print('fft_max: {}'.format(fft_max))
print(fft_max.shape)
        
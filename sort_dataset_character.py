from dataset.MatDataset import BurgersDataset, JHTDB
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


def compute_tke_spectrum(u, v, lx, ly):
    """
    Given velocity fields u and v, computes the turbulent kinetic energy spectrum. The function computes in three steps:
    1. Compute velocity spectrum with fft, returns uf, vf.
    2. Compute the point-wise turbulent kinetic energy Ef=0.5*(uf, vf)*conj(uf, vf).
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).
    """
    nx = u.shape[0]
    ny = u.shape[1]

    nt = nx * ny
    # Compute velocity spectrum
    uf = torch.fft.fftn(u) / nt
    vf = torch.fft.fftn(v) / nt

    # Compute the point-wise turbulent kinetic energy
    Ef = 0.5 * (uf * torch.conj(uf) + vf * torch.conj(vf)).real
    kx = 2 * torch.pi / lx 
    ky = 2 * torch.pi / ly
    knorm = (kx + ky) / 2
    kxmax = nx / 2
    kymax = ny / 2
    wave_numbers = knorm * torch.arange(0, nx)
    tke_spectrum = torch.zeros(nx)
    for i in range(nx):
        rkx = i
        if i > kxmax:
            rkx = rkx - nx
        for j in range(ny):
            rky = j
            if j > kymax:
                rky = rky - ny
            rk = torch.sqrt(rkx * rkx + rky * rky)
            k_index = int(np.round(rk))
            tke_spectrum[k_index] += Ef[i, j]
    return tke_spectrum, wave_numbers


model = TEECNetConv(1, 32, 1, num_layers=6, retrieve_weights=False, num_powers=3, sub_size=8)
# dataset = BurgersDataset(root='data/burgers')
dataset = JHTDB(root='data/jhtdb', tstart=1, tend=100, fields='u', dataset='isotropic1024coarse')

fft_x_list = []
fft_y_list = []

sub_x_total = []
sub_y_total = []

for data in tqdm.tqdm(dataset):
    sub_x_list, sub_y_list = model.get_partition_domain(data[0], mode='train'), model.get_partition_domain(data[1], mode='test')
    # sub_x_total.append(sub_x_list)
    # sub_y_total.append(sub_y_list)
    for sub_x, sub_y in zip(sub_x_list, sub_y_list):
        # calculate fft of sub_x and sub_y
        sub_tke_spectrum_x, sub_wave_numbers_x = compute_tke_spectrum(sub_x[0], sub_x[1], 1, 1)
        sub_tke_spectrum_y, sub_wave_numbers_y = compute_tke_spectrum(sub_y[0], sub_y[1], 1, 1)
        fft_x_list.append(sub_wave_numbers_x[sub_tke_spectrum_x.argmax()])
        fft_y_list.append(sub_wave_numbers_y[sub_tke_spectrum_y.argmax()])

# plot the distribution of the dominant frequencies of the input and output
plt.hist(fft_x_list, bins=100, alpha=0.5, label='input')
plt.hist(fft_y_list, bins=100, alpha=0.5, label='output')
plt.legend(loc='upper right')
plt.savefig('dominant_frequencies.png')

# save the sub_x_total and sub_y_total lists and the fft_x_list and fft_y_list lists
# torch.save(sub_x_total, 'sub_x_total.pt')
# torch.save(sub_y_total, 'sub_y_total.pt')
torch.save(fft_x_list, 'fft_x_list.pt')
torch.save(fft_y_list, 'fft_y_list.pt')
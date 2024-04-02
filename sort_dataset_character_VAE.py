from dataset.MatDataset import BurgersDataset, JHTDB
import torch
from models.teecnet import TEECNetConv
from models.scheduler import TBVAE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import os


def plot_prediction(window_size, y, epoch, batch_idx, folder):
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
    fig = plt.figure()
    plt.contourf(xx, yy, y.cpu().detach().numpy().reshape(window_size, window_size), levels=100, cmap='plasma')
    # plt.set_title('(a) Ground truth')
    plt.axis('off')
    # plt.colorbar(axs[2].contourf(xx, yy, np.abs(y.cpu().detach().numpy().reshape(window_size, window_size) - y_pred.reshape(window_size, window_size)), levels=100, cmap='plasma'), ax=axs[2], pad=0.01)

    plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    # wandb.log({'prediction': wandb.Image(plt)})
    plt.close()


# model = TBVAE(input_dim=1, latent_dim=32, hidden_dim=256, num_classes=1, num_layers=3, dropout=0.5)
# model.load_state_dict(torch.load('logs/models/TBVAE_model.pth'))
# model.eval()
# dataset = BurgersDataset(root='data/burgers')
dataset = JHTDB(root='data/jhtdb', tstart=1, tend=100, fields='u', dataset='isotropic1024coarse', partition=True, sub_size=64)
pca = PCA(n_components=10)

data_space = []

# do pca on data to obtain the latent space
for data in tqdm.tqdm(dataset):
    x, y = data
    data_space.append(x.cpu().detach().numpy().reshape(-1))

latent_space = pca.fit_transform(np.array(data_space))

# cluster the data in the latent space into different groups
kmeans = KMeans(n_init='auto', n_clusters=8, random_state=0).fit(latent_space)
print(kmeans.labels_)

# plot the data in the latent space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
plt.scatter(latent_space[:, 0], latent_space[:, 1], c=kmeans.labels_, cmap='viridis', s=5)
plt.show()
# save figure and latent space
np.save('logs/latent_space.npy', latent_space)
np.save('logs/cluster_labels.npy', kmeans.labels_)

# plot data sample at each cluster
for i in range(8):
    idx = np.where(kmeans.labels_ == i)[0]
    data = dataset[idx[0]][0]
    plot_prediction(64, data, 0, i, 'logs/cluster_samples')

# save figure
fig.savefig('logs/latent_space.png')
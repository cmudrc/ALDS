import os
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
from joblib import dump, load
from multiprocessing import Pool


class Encoder():
    def __init__(self, n_components):
        self.n_components = n_components
        pass

    def train(self, dataset):
        pass

    def get_latent_space(self, dataset):
        pass


class TBVAE(nn.Module):
    """
    Variational Autoencoder for translating turbulent flow fields into a latent space. Such latent space is then used to classify the flow fields by their characteristics.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=3, dropout=0.5, **kwargs):
        super(TBVAE, self).__init__() 
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        for _ in range(num_layers - 1):
            self.encoder.add_module(
                f'hidden_{_}',
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)

        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        for _ in range(num_layers - 1):
            self.decoder.add_module(
                f'hidden_{_}',
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.decoder.add_module(
            'output',
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class PCAEncoder(Encoder):
    def __init__(self, n_components, **kwargs):
        super(PCAEncoder, self).__init__(n_components)
        self.model = PCA(n_components=n_components)

    def train(self, dataset, save_model=False, path=None):
        data_space = []
        for data in dataset:
            x, y = data
            data_space.append(x.cpu().detach().numpy().reshape(-1))
        self.model.fit(np.array(data_space))
        # dump(self.model, 'logs/models/collection_fno_jhtdb/pca_encoder.joblib')
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'pca_encoder.joblib'))

    def get_latent_space(self, dataset):
        data_space = []
        for data in dataset:
            x, y = data
            data_space.append(x.cpu().detach().numpy().reshape(-1))
        latent_space = self.model.transform(np.array(data_space))
        return latent_space
    
    def get_latent(self, x):
        data_space = []
        for sub_x in x:
            data_space.append(sub_x.cpu().detach().numpy().reshape(-1))
        latent_space = self.model.transform(np.array(data_space))
        return latent_space
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'pca_encoder.joblib'))
    

class VAEEncoder(Encoder):
    def __init__(self, n_components, **kwargs):
        super(VAEEncoder, self).__init__(n_components)
        self.model = TBVAE(input_dim=kwargs['input_dim'], latent_dim=kwargs['n_components'], hidden_dim=128, num_layers=kwargs['num_layers'], dropout=kwargs['dropout'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, dataset):
        def loss_function(x_hat, x, mu, logvar):
            BCE = nn.functional.mse_loss(x_hat, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD
        
        device = self.device
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        for data in dataset:
            x, y = data
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = self.model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()

    def get_latent_space(self, dataset):
        device = self.device
        self.model.to(device)
        self.model.eval()
        latent_space = []
        with torch.no_grad():
            for data in dataset:
                x, y = data
                x = x.to(device)
                mu, logvar = self.model.encode(x)
                z = self.model.reparameterize(mu, logvar)
                latent_space.append(z.cpu().detach().numpy())
        return np.array(latent_space)
    

class SpectrumEncoder(Encoder):
    def __init__(self, n_components, **kwargs):
        super(SpectrumEncoder, self).__init__(n_components)
        self.n_components = n_components
        self.domain_size = kwargs['domain_size']

    def train(self, dataset, save_model=False, path=None):
        pass
    
    @staticmethod
    def _compute_tke_spectrum(u):
        """
        Given velocity fields u and v, computes the turbulent kinetic energy spectrum. The function computes in three steps:
        1. Compute velocity spectrum with fft, returns uf, vf.
        2. Compute the point-wise turbulent kinetic energy Ef=0.5*(uf, vf)*conj(uf, vf).
        3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
        Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
        the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
        E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).
        """
        u, lx, ly = u
        # check if u is a data that contains x and y or just x
        if len(u) == 2:
            u = u[0]
        # print(u.shape)
        nx = u.shape[0]
        ny = u.shape[1]

        nt = nx * ny
        # Compute velocity spectrum
        uf = np.fft.fft2(u, axes=(0, 1))

        # Compute the point-wise turbulent kinetic energy
        Ef = 0.5 * (uf * np.conj(uf)).real
        # kx = 2 * np.pi / lx 
        # ky = 2 * np.pi / ly
        # knorm = np.sqrt(kx ** 2 + ky ** 2)
        kxmax = nx / 2
        kymax = ny / 2
        # wave_numbers = knorm * np.arange(0, nx)
        tke_spectrum = np.zeros(nx)
        for i in range(nx):
            rkx = i
            if i > kxmax:
                rkx = rkx - nx
            for j in range(ny):
                rky = j
                if j > kymax:
                    rky = rky - ny
                rk = np.sqrt(rkx * rkx + rky * rky)
                k_index = int(np.round(rk))
                tke_spectrum[k_index] += Ef[i, j]
        # k = torch.fft.fftfreq(nx, lx / nx)

        # plt.loglog(wave_numbers[1:], tke_spectrum[1:])
        # plt.savefig('tke_spectrum.png')
        return np.log(tke_spectrum[1:] + 1e-8)

    def get_latent_space(self, dataset):
        try:
            dataset = [[data[0][:, :, 0], self.domain_size, self.domain_size] for data in dataset]
        except:
            dataset = [[data[0], self.domain_size, self.domain_size] for data in dataset]
        with Pool() as p:
            latent_space = p.map(self._compute_tke_spectrum, dataset)
        return np.array(latent_space)

    def get_latent(self, x):
        x = [(data, self.domain_size, self.domain_size) for data in x]
        with Pool() as p:
            latent_space = p.map(self._compute_tke_spectrum, x)
        return np.array(latent_space)

    def load_model(self, path):
        pass
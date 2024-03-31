import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import uniform_ as reset
# import torch_geometric.nn as pyg_nn
# from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F


class TBVAE(nn.Module):
    """
    Variational Autoencoder for translating turbulent flow fields into a latent space. Such latent space is then used to classify the flow fields by their characteristics.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5, **kwargs):
        super(TBVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes


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


class PartitionScheduler(nn.Module):
    def __init__(self, num_partitons, model_library):
        super(PartitionScheduler, self).__init__()
        self.num_partitions = num_partitons
        self.model_library = model_library

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import uniform_ as reset
# import torch_geometric.nn as pyg_nn
# from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F
# from torch_scatter import scatter_softmax


# class TEECNet(torch.nn.Module):
#     r"""The Taylor-series Expansion Error Correction Network which consists of several layers of a Taylor-series Error Correction kernel.

#     Args:
#         in_channels (int): Size of each input sample.
#         width (int): Width of the hidden layers.
#         out_channels (int): Size of each output sample.
#         num_layers (int): Number of layers.
#         **kwargs: Additional arguments of :class:'torch_geometric.nn.conv.MessagePassing'
#     """
#     def __init__(self, in_channels, width, out_channels, num_layers=4, **kwargs):
#         super(TEECNet, self).__init__()
#         self.num_layers = num_layers

#         self.fc1 = nn.Linear(in_channels, width)
#         self.kernel = KernelConv(width, width, kernel=PowerSeriesKernel, in_edge=5, num_layers=5, **kwargs)
#         # self.kernel_out = KernelConv(width, out_channels, kernel=PowerSeriesKernel, in_edge=1, num_layers=2, **kwargs)
#         self.fc_out = nn.Linear(width, out_channels)

#     def forward(self, x, edge_index, edge_attr):
#         x = self.fc1(x)
#         for i in range(self.num_layers):
#             # x = F.relu(self.kernel(x, edge_index, edge_attr))
#             x = self.kernel(x, edge_index, edge_attr)
#         # x = self.kernel_out(x, edge_index, edge_attr)
#         x = self.fc_out(x)
#         return x


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
    

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    

class PowerSeriesConv(nn.Module):
    def __init__(self, in_channel, out_channel, num_powers, **kwargs):
        super(PowerSeriesConv, self).__init__()
        self.num_powers = num_powers
        # self.convs = torch.nn.ModuleList()
        # for i in range(num_powers):
        #     self.convs.append(nn.Linear(in_channel, out_channel))
        # self.conv = nn.Linear(in_channel, out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.activation = F.gelu
        self.root_param = nn.Parameter(torch.Tensor(num_powers, out_channel))

        # self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_powers):
            # reset(self.convs[i])
            reset(self.conv)
        # size = self.num_powers
        # uniform(size, self.root_param)

    def forward(self, x):
        x = torch.movedim(x, -1, 1)
        # x_full = None
        x_conv_ = self.conv(x)
        x_conv_ = torch.movedim(x_conv_, 1, -1)
        for i in range(self.num_powers):
            # x_conv = self.convs[i](x)
            if i == 0:
                x_full = self.root_param[i] * x_conv_
            else:
                x_full += self.root_param[i] * torch.pow(self.activation(x_conv_), i)
        # x_full = torch.movedim(x_full, -1, 1)
        return x_full
        # x_full = torch.sum(self.root_param * torch.pow(self.activation(x_conv_), torch.arange(self.num_powers).to(x.device).view(-1, 1, 1)), dim=0)
        # return x_full
    

class PowerSeriesKernel(nn.Module):
    def __init__(self, num_layers, num_powers, activation=nn.GELU, **kwargs):
        super(PowerSeriesKernel, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.conv0 = PowerSeriesConv(kwargs['in_channel'], 256, num_powers)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(PowerSeriesConv(256, 256, num_powers))
        self.norm = nn.BatchNorm2d(256)

        self.conv_out = PowerSeriesConv(256, kwargs['out_channel'], num_powers)
        self.activation = activation()

    def forward(self, x):
        x = self.conv0(x)
        for i in range(self.num_layers):
            # x = self.activation(self.convs[i](x))
            x = self.convs[i](x)
            x = torch.movedim(x, -1, 1)
            x = self.norm(x)
            x = torch.movedim(x, 1, -1)
        x = self.conv_out(x)
        return x


# class KernelConv(pyg_nn.MessagePassing):
#     r"""
#     The continuous kernel-based convolutional operator from the
#     `"Neural Message Passing for Quantum Chemistry"
#     <https://arxiv.org/abs/1704.01212>`_ paper.
#     This convolution is also known as the edge-conditioned convolution from the
#     `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
#     Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
#     :class:`torch_geometric.nn.conv.ECConv` for an alias):

#     .. math::
#         \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
#         \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
#         h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

#     where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
#     a MLP. In our implementation the kernel is combined via a Taylor expansion of 
#     graph edge attributes :math:`\mathbf{e}_{i,j}` and a typical neural operator implementation
#     of a DenseNet kernel.

#     Args:
#         in_channel (int): Size of each input sample (nodal values).
#         out_channel (int): Size of each output sample (nodal values).
#         kernel (torch.nn.Module): A kernel function that maps edge attributes to
#             edge weights.
#         in_edge (int): Size of each input edge attribute.
#         num_layers (int): Number of layers in the Taylor-series expansion kernel.
#     """
#     def __init__(self, in_channel, out_channel, kernel, in_edge=1, num_layers=3, **kwargs):
#         super(KernelConv, self).__init__(aggr='mean')
#         self.in_channels = in_channel
#         self.out_channels = out_channel
#         self.in_edge = in_edge
#         self.root_param = nn.Parameter(torch.Tensor(in_channel, out_channel))
#         self.bias = nn.Parameter(torch.Tensor(out_channel))

#         self.linear = nn.Linear(in_channel, out_channel)
#         self.kernel = kernel(in_channel=in_edge, out_channel=out_channel**2, num_layers=num_layers, **kwargs)
#         self.operator_kernel = DenseNet([in_edge, 64, 128, out_channel**2], nn.ReLU)
#         if kwargs['retrieve_weight']:
#             self.retrieve_weights = True
#             self.weight_k = None
#             self.weight_op = None
#         else:
#             self.retrieve_weights = False

#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.kernel)
#         reset(self.linear)
#         reset(self.operator_kernel)
#         size = self.in_channels
#         uniform(size, self.root_param)
#         uniform(size, self.bias)

#     def forward(self, x, edge_index, edge_attr):
#         x = x.unsqueeze(-1) if x.dim() == 1 else x
#         pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
#         return self.propagate(edge_index, x=x, pseudo=pseudo)
    
#     def message(self, x_i, x_j, pseudo):
#         weight_k = self.kernel(pseudo).view(-1, self.out_channels, self.out_channels)
#         weight_op = self.operator_kernel(pseudo).view(-1, self.out_channels, self.out_channels)
#         x_i = self.linear(x_i)
#         x_j = self.linear(x_j)
       
#         x_j_k = torch.matmul((x_j-x_i).unsqueeze(1), weight_k).squeeze(1)
#         # x_j_k = weight_k
#         x_j_op = torch.matmul(x_j.unsqueeze(1), weight_op).squeeze(1)

#         if self.retrieve_weights:
#             self.weight_k = weight_k
#             # self.weight_op = weight_op
#         return x_j_k + x_j_op
#         # return x_j_k
    
#     def update(self, aggr_out, x):
#         return aggr_out + torch.mm(x, self.root_param) + self.bias
    
#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class TEECNetConv(nn.Module):
    """
    The convolutional representation of TEECNet model that operates on matrixes instead of graphs.
    """
    def __init__(self, in_channels, width, out_channels, num_layers=4, **kwargs):
        super(TEECNetConv, self).__init__()
        self.num_layers = num_layers

        self.fc1 = nn.Linear(in_channels+2, width)
        self.kernel = PowerSeriesKernel(in_channel=width, out_channel=width, num_layers=num_layers, **kwargs)
        self.fc_out = nn.Linear(width, out_channels)
        try:
            self.sub_size = kwargs['sub_size']
        except KeyError:
            self.sub_size = 9

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def symmetric_padding(self, x, mode):
        # pad the domain symmetrically to make it divisible by sub_size
        # get input shape to determine if the input is batched or not
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # get pad size
        pad_size = (x.shape[1] % self.sub_size) // 2 + 1
        x = F.pad(x, (0, 0, pad_size, pad_size, pad_size, pad_size, 0, 0))
        # print(x[0, :, :, 0])
        if mode == 'train':
            # add one dimension to the tensor x, with 0 in the padded region and 1 in the original region
            x_pad_idx = torch.ones((x.shape[0], x.shape[1], x.shape[2], 1))
            x_pad_idx[:, :pad_size, :, :] = 0
            x_pad_idx[:, -pad_size:, :, :] = 0
            x_pad_idx[:, :, -pad_size:, :] = 0
            x_pad_idx[:, :, :pad_size, :] = 0
            x = torch.cat((x, x_pad_idx), dim=-1)
            return x, pad_size
        elif mode == 'test':    
            return x, pad_size

    def get_partition_domain(self, x, mode, displacement=0):
        # pad the domain symmetrically to make it divisible by sub_size
        x, pad_size = self.symmetric_padding(x, mode)
        # partition the domain into num_partitions subdomains of the same size
        x_list = []
        num_partitions_dim = x.shape[1] - self.sub_size + 1

        for i in range(num_partitions_dim):
            for j in range(num_partitions_dim):
                x_list.append(x[:, i:i+self.sub_size, j:j+self.sub_size, :])

        return x_list
    
    def reconstruct_from_partitions(self, x, x_list, displacement=0):
        # reconstruct the domain from the partitioned subdomains
        num_partitions_dim = int(np.sqrt(len(x_list)))
        x, pad_size = self.symmetric_padding(x, mode='test')
        x = torch.zeros_like(x[:, 1:-1, 1:-1, 0].unsqueeze(-1))
        # if the domain can be fully partitioned into subdomains of the same size
        # if len(x_list) == num_partitions_dim**2:
        for i in range(num_partitions_dim):
            for j in range(num_partitions_dim):
                x[:, i:i+self.sub_size-2, j:j+self.sub_size-2, :] = x_list[i*num_partitions_dim + j][:, 1:-1, 1:-1, :]

        if pad_size == 1:
            return x
        else:
            x = x[:, pad_size-1:-pad_size+1, pad_size-1:-pad_size+1, :]
            return x

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc1(x)
        # for i in range(self.num_layers):
        x = self.kernel(x)
        x = self.fc_out(x)
        return x
    

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, **kwargs):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])
        # print(self.conv0.weights1.device)
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
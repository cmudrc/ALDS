import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.tri import Triangulation
import wandb


def plot_prediction(y, y_pred, save_mode='wandb', **kwargs):
    window_size_x, window_size_y = y_pred.shape[2], y_pred.shape[1]
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size_x), np.linspace(0, 1, window_size_y))
    fig, axs = plt.subplots(3, 1, figsize=(20, 5))
    axs[0].contourf(xx, yy, y.cpu().detach().reshape(window_size_y, window_size_x), levels=100, cmap='plasma')
    axs[0].set_title('(a) Ground truth')
    axs[0].axis('off')
    axs[1].contourf(xx, yy, y_pred.cpu().reshape(window_size_y, window_size_x), levels=100, cmap='plasma')
    axs[1].set_title('(b) Prediction')
    axs[1].axis('off')
    axs[2].contourf(xx, yy, np.abs(y.cpu().reshape(window_size_y, window_size_x) - y_pred.cpu().reshape(window_size_y, window_size_x)) / y.cpu().reshape(window_size_y, window_size_x), levels=np.linspace(0, 1, 100), cmap='plasma')
    axs[2].set_title('(c) Absolute difference by percentage')
    axs[2].axis('off')
    # add colorbar and labels to the rightmost plot
    cbar = plt.colorbar(axs[2].collections[0], ax=axs[2], orientation='vertical')
    cbar.set_label('Absolute difference')
    plt.tight_layout()

    # plt.savefig(os.path.join(folder, f'epoch_{epoch}_batch_{batch_idx}.png'))
    if save_mode == 'wandb':
        wandb.log({'prediction': wandb.Image(plt)})
    elif save_mode == 'plt':
        plt.show()
    elif save_mode == 'save':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.pdf', format='pdf', dpi=1200)
    elif save_mode == 'save_png':
        os.makedirs(os.path.dirname(kwargs['path']), exist_ok=True)
        plt.savefig(kwargs['path'] +'.png', format='png', dpi=300)
    plt.close()


def visualize_prediction(writer, data, model, epoch, mode='writer', **kwargs):
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    x = x.to(kwargs['device'])
    edge_index = edge_index.to(kwargs['device'])
    edge_attr = edge_attr.to(kwargs['device'])

    pred = model(x, edge_index, edge_attr).detach().cpu().numpy().squeeze()
    pos_x = data.pos[:, 0].detach().cpu().numpy()
    pos_y = data.pos[:, 1].detach().cpu().numpy()

    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()

    # reconstruct the mesh
    tri = Triangulation(pos_x, pos_y, data.cells.detach().cpu().numpy())
    # for debug purpose print triangulation x and y array shape
    # print(tri.x.shape)
    # print(tri.y.shape)
    # print(pred.shape)
    # print(tri.triangles.shape)
    # plot the temepreture contour
    plt.tricontourf(tri, pred, levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Prediction')
    
    if mode == 'writer':
        writer.add_figure('Prediction', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'prediction_{}.png'.format(epoch)))
    
    plt.close()

    plt.tricontourf(tri, y, levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Ground Truth')
    if mode == 'writer':
        writer.add_figure('Ground Truth', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'ground_truth_{}.png'.format(epoch)))
    plt.close()

    plt.tricontourf(tri, np.abs(pred - y), levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Absolute Error')
    if mode == 'writer':
        writer.add_figure('Absolute Error', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'absolute_error_{}.png'.format(epoch)))

    plt.close()

    plt.tricontourf(tri, x, levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Low Resolution Temperature')
    if mode == 'writer':
        writer.add_figure('Low Resolution Temperature', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'low_res_temperature_{}.png'.format(epoch)))

    plt.close()

    # free cuda memory
    del x, edge_index, edge_attr, y, pred, pos_x, pos_y, tri
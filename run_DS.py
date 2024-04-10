import torch
import wandb
import os
import numpy as np
from sklearn.metrics import r2_score

from utils import *


def train_DS(exp_name, model, dataset, train_config, **kwargs):
    wandb.init(project='ALDS', name=exp_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    log_interval = train_config['log_interval']
    val_interval = train_config['val_interval']

    for epoch in range(train_config['epochs']):
        model.train()
        loss_epoch = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        loss_epoch /= len(train_loader)
        wandb.log({'loss': loss_epoch})
        if epoch % log_interval == 0:
            print(f'Epoch {epoch}, Loss: {loss_epoch}')

        if epoch % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_loss += criterion(y_pred, y).item()

            val_loss /= len(val_loader)
            wandb.log({'val_loss': val_loss})
            print(f'Epoch {epoch}, Validation Loss: {val_loss}')

            plot_prediction(y[0].cpu(), y_pred[0].cpu(), save_mode='wandb')

    os.makedirs(f'logs/models/collection_{exp_name}', exist_ok=True)
    torch.save(model.state_dict(), f'logs/models/collection_{exp_name}/model.pth')


def pred_DS(idxs, exp_name, model, dataset, save_mode, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(f'logs/models/collection_{exp_name}/model.pth'))
    model.eval()
    r2_scores = []
    for idx in idxs:
        pred_y_list = []
        x, sub_x_list, sub_y_list = dataset.get_one_full_sample(idx)
        for sub_x in sub_x_list:
            sub_x = sub_x.to(model.device)
            sub_pred_y = model(sub_x).cpu().detach().numpy()
            pred_y_list.append(sub_pred_y)

        pred_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), pred_y_list)
        sub_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), sub_y_list)

        plot_prediction(sub_y, pred_y, save_mode=save_mode, path=f'logs/figures/{exp_name}/idx_{idx}.pdf')

        r2_scores.append(r2_score(sub_y.flatten().cpu().detach().numpy(), pred_y.flatten().cpu().detach().numpy()))
        # save the prediction
        os.makedirs(f'logs/raw_data/{exp_name}', exist_ok=True)
        torch.save(pred_y, f'logs/raw_data/{exp_name}/pred_idx_{idx}.pth')
        torch.save(sub_y, f'logs/raw_data/{exp_name}/gt_idx_{idx}.pth')

        if save_mode == 'wandb':
            wandb.log({'r2_score': r2_scores[-1]})

    # save r2 scores
    np.save(f'logs/raw_data/{exp_name}/r2_scores.npy', r2_scores)

    return r2_scores


if __name__ == '__main__':
    args = parse_args()
    run_mode = args.mode
    model_name = args.model
    exp_name = args.exp_name
    dataset_name = args.dataset
    exp_config = args.exp_config
    train_config = args.train_config

    exp_config = load_yaml(exp_config)
    train_config = load_yaml(train_config)

    model = init_model(model_name, **exp_config)
    dataset = init_dataset(dataset_name, **exp_config)

    if run_mode == 'train':
        train_DS(exp_name, model, dataset, train_config)
        print('Training finished.')
    elif run_mode == 'pred':
        idxs = exp_config['idxs']
        save_mode = exp_config['save_mode']
        
        pred_DS(idxs, exp_name, model, dataset, save_mode)
        print('Prediction finished.')
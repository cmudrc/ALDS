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
            optimizer.zero_grad()
            try:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
            except:
                x, boundary, y = x[0].to(device), x[1].to(device), y[:, :, :, 0].to(device)
                y_pred = model(x, boundary).squeeze()
            # print(y_pred.shape, y.shape)
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
                    try:
                        x, y = x.to(device), y.to(device)
                        y_pred = model(x)
                    except:
                        x, boundary, y = x[0].to(device), x[1].to(device), y[:, :, :, 0].to(device)
                        y_pred = model(x, boundary).squeeze()
                    val_loss += criterion(y_pred, y).item()

            val_loss /= len(val_loader)
            wandb.log({'val_loss': val_loss})
            print(f'Epoch {epoch}, Validation Loss: {val_loss}')

            # plot_prediction(y[0].cpu(), y_pred[0].cpu(), save_mode='wandb')

    os.makedirs(f'logs/models/collection_{exp_name}', exist_ok=True)
    torch.save(model.state_dict(), f'logs/models/collection_{exp_name}/model.pth')

def recurrent_predict(dataset, x, sub_x_, model, sub_boundary_, num_iters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set maximum number of subdomains per time to predict to avoid memory issues
    sub_x_limit = 16
    if len(sub_x_) > sub_x_limit:
        num_iters = len(sub_x_) // sub_x_limit if len(sub_x_) % sub_x_limit == 0 else len(sub_x_) // sub_x_limit + 1
        sub_x = [sub_x_[i:i+sub_x_limit] for i in range(num_iters-1)]
        sub_x.append(sub_x_[(num_iters-1)*sub_x_limit:])
        sub_boundary = [sub_boundary_[i:i+sub_x_limit] for i in range(num_iters-1)]
        sub_boundary.append(sub_boundary_[(num_iters-1)*sub_x_limit:])
        all_pred_y_list = []
        for i in range(num_iters):
            predictions = []
            for sub_x_batch, sub_boundary_batch in zip(sub_x, sub_boundary):
                sub_x_batch = sub_x_batch.to(device)
                sub_boundary_batch = sub_boundary_batch.to(device)
                pred_y_batch_list = model(sub_x_batch, sub_boundary_batch).cpu().detach()
                predictions += pred_y_batch_list
            # predictions = torch.cat(pred_y_list, dim=0)
            prediction_reconstructed = dataset.reconstruct_from_partitions(x.unsqueeze(0), predictions).squeeze(0)

            all_pred_y_list.append(predictions)

            predictions, pred_boundary_list = dataset.get_partition_domain(prediction_reconstructed, mode='test')
            sub_x = [torch.stack(predictions[i:i+sub_x_limit], dim=0) for i in range(num_iters-1)]
            sub_boundary = [torch.stack(pred_boundary_list[i:i+sub_x_limit], dim=0) for i in range(num_iters-1)]
            sub_x.append(torch.stack(predictions[(num_iters-1)*sub_x_limit:], dim=0))
            sub_boundary.append(torch.stack(pred_boundary_list[(num_iters-1)*sub_x_limit:], dim=0))

    else:
        sub_x = sub_x.to(device)
        sub_boundary = sub_boundary.to(device)
        all_pred_y_list = []
        all_labels = []
        predictions = sub_x
        pred_boundary_list = sub_boundary
        for i in range(num_iters):
            pred_y_list, labels = model(sub_x, sub_boundary)
            predictions = pred_y_list.cpu().clone()
            prediction_reconstructed = dataset.reconstruct_from_partitions(x.unsqueeze(0), predictions)

            all_pred_y_list.append(predictions)

            predictions, pred_boundary_list = dataset.get_partition_domain(prediction_reconstructed, mode='test')
            sub_x = torch.stack(predictions)
            sub_boundary = torch.stack(pred_boundary_list)
        
    return all_pred_y_list, all_labels
    

def pred_DS(idxs, exp_name, model, dataset, save_mode, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(f'logs/models/collection_{exp_name}/model.pth'))
    model.eval()
    all_r2_scores = []
    if 'timesteps' in kwargs:
        for idx in idxs:
            r2_scores = []
            try:
                x, sub_x_list, _ = dataset.get_one_full_sample(idx)
                all_pred_y_list, all_labels = recurrent_predict(dataset, x, sub_x_tensor, model, num_iters=kwargs['timesteps'])
            except:
                x, sub_x_list, sub_boundary_list, _ = dataset.get_one_full_sample(idx)
                sub_x_tensor = torch.stack(sub_x_list)
                sub_boundary_tensor = torch.stack(sub_boundary_list)    

                all_pred_y_list, all_labels = recurrent_predict(dataset, x, sub_x_tensor, model, sub_boundary_tensor, num_iters=kwargs['timesteps'])
            
            timestep = idx
            for pred_y_list, labels in zip(all_pred_y_list, all_labels):
                try:
                    _, sub_y_list, _ = dataset.get_one_full_sample(timestep+1)
                except:
                    _, sub_y_list, _, _ = dataset.get_one_full_sample(timestep+1)
                pred_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), pred_y_list)
                sub_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), sub_y_list)

                plot_prediction(sub_y, pred_y, save_mode=save_mode, path=f'logs/figures/{exp_name}/timestep_{timestep}')
                
                plot_partition(sub_y, pred_y, labels, kwargs['sub_size']+2, save_mode=save_mode, path=f'logs/figures/{exp_name}/partition_timestep_{timestep}')

                r2_scores.append(r2_score(sub_y.flatten().cpu().detach().numpy(), pred_y.flatten().cpu().detach().numpy()))
                # save the prediction
                os.makedirs(f'logs/raw_data/{exp_name}', exist_ok=True)
                torch.save(pred_y, f'logs/raw_data/{exp_name}/pred_timestep_{timestep}.pth')
                torch.save(sub_y, f'logs/raw_data/{exp_name}/gt_timestep_{timestep}.pth')
                timestep += 1
                if save_mode == 'wandb':
                    wandb.log({'r2_score': r2_scores[-1]})
        all_r2_scores.append(r2_scores)

    else:
        r2_scores = []
        for idx in idxs:
            pred_y_list = []
            x, sub_x_list, sub_y_list = dataset.get_one_full_sample(idx)
            for sub_x in sub_x_list:
                sub_x = sub_x.unsqueeze(0).to(device)
                sub_pred_y = model(sub_x).cpu().detach()
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
    np.save(f'logs/raw_data/{exp_name}/r2_scores.npy', all_r2_scores)

    return all_r2_scores


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
        if 'timesteps' in exp_config:
            pred_DS(idxs, exp_name, model, dataset, save_mode, timesteps=exp_config['timesteps'])
        else:
            pred_DS(idxs, exp_name, model, dataset, save_mode)
        print('Prediction finished.')
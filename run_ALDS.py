import torch
import numpy as np
import wandb
from sklearn.metrics import r2_score
from utils import *

from utils import parse_args, plot_prediction
from models.scheduler import PartitionScheduler


def train_ALDS(exp_name, encoder, classifier, model, dataset, num_partitions, train_config, **kwargs):
    scheduler = PartitionScheduler(exp_name, num_partitions, dataset, encoder, classifier, model, train=True)
    scheduler.train(train_config)

def pred_ALDS(idxs, exp_name, encoder, classifier, model, dataset, num_partitions, save_mode, **kwargs):
    scheduler = PartitionScheduler(exp_name, num_partitions, dataset, encoder, classifier, model, train=False)
    
    r2_scores = []
    for idx in idxs:
        x, sub_x_list, sub_y_list = dataset.get_one_full_sample(idx)
        sub_x_tensor = torch.stack(sub_x_list)
        pred_y_list = scheduler.predict(sub_x_tensor)
        
        pred_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), pred_y_list)
        sub_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), sub_y_list)

        plot_prediction(sub_y, pred_y, save_mode=save_mode, path=f'logs/figures/{exp_name}/idx_{idx}.svg')

        r2_scores.append(r2_score(sub_y.flatten().cpu().detach().numpy(), pred_y.flatten().cpu().detach().numpy()))
        # save the prediction
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
    encoder_name = args.encoder
    classifier_name = args.classifier
    model_name = args.model
    exp_name = args.exp_name
    dataset_name = args.dataset
    exp_config = args.exp_config
    train_config = args.train_config

    exp_config = load_yaml(exp_config)
    train_config = load_yaml(train_config)

    n_clusters = exp_config['n_clusters']

    encoder = init_encoder(encoder_name, **exp_config)
    classifier = init_classifier(classifier_name, **exp_config)
    model = init_model(model_name, **exp_config)
    dataset = init_dataset(dataset_name, **exp_config)

    if exp_config['save_mode'] == 'wandb':
        wandb.init(project='ALDS', name=exp_name)

        if run_mode == 'train':
            train_ALDS(exp_name, encoder, classifier, model, dataset, n_clusters, train_config)
            print('Training done!')
        elif run_mode == 'pred':
            idxs = exp_config['idxs']
            save_mode = exp_config['save_mode']
            pred_ALDS(idxs, exp_name, encoder, classifier, model, dataset, n_clusters, save_mode)
            print('Prediction done!')

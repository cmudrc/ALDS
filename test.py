from models.scheduler_gnn import GNNPartitionScheduler
from utils import *
import torch
import os
from sklearn.metrics import r2_score
import wandb


def train_graph_ALDD(exp_name, encoder, classifier, model, dataset, num_partitions, train_config, **kwargs):
    scheduler = GNNPartitionScheduler(exp_name, num_partitions, dataset, encoder, classifier, model, train=True)
    scheduler.train(train_config, **kwargs)

def pred_graph_ALDD(idxs, exp_name, encoder, classifier, model, dataset, num_partitions, save_mode, **kwargs):
    scheduler = GNNPartitionScheduler(exp_name, num_partitions, dataset, encoder, classifier, model, train=False)
    for idx in idxs:
        x, sub_x_list, sub_y_list = dataset.get_one_full_sample(idx)
        sub_x_tensor = torch.stack(sub_x_list)
        pred_y_list, labels = scheduler.predict(sub_x_tensor)
        
        pred_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), pred_y_list)
        sub_y = dataset.reconstruct_from_partitions(x.unsqueeze(0), sub_y_list)

        plot_prediction(sub_y, pred_y, save_mode=save_mode, path=f'logs/figures/{exp_name}/timestep_{idx}')
        
        plot_partition(sub_y, pred_y, labels, kwargs['sub_size']+2, save_mode=save_mode, path=f'logs/figures/{exp_name}/partition_timestep_{idx}')

        r2_scores = r2_score(sub_y.flatten().cpu().detach().numpy(), pred_y.flatten().cpu().detach().numpy())
        # save the prediction
        os.makedirs(f'logs/raw_data/{exp_name}', exist_ok=True)
        torch.save(pred_y, f'logs/raw_data/{exp_name}/pred_timestep_{idx}.pth')
        torch.save(sub_y, f'logs/raw_data/{exp_name}/gt_timestep_{idx}.pth')
        if save_mode == 'wandb':
            wandb.log({'r2_score': r2_scores})
        print(f'Prediction for done!')


if __name__ == '__main__':
    # dataset = CoronaryArteryDataset(root='data/coronary', partition=True, sub_size=5)
    # dataset = DuctAnalysisDataset(root='data/Duct', partition=True, sub_size=0.03)
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
    print('Dataset loaded!')

    if run_mode == 'train':
        train_graph_ALDD(exp_name, encoder, classifier, model, dataset, n_clusters, train_config)

    elif run_mode == 'pred':
        pred_graph_ALDD([0], exp_name, encoder, classifier, model, dataset, n_clusters, 'wandb', sub_size=0.03)
        # pred_graph_ALDD([0, 1, 2, 3], exp_name, encoder, classifier, model, dataset, n_clusters, 'local', sub_size=0.03)
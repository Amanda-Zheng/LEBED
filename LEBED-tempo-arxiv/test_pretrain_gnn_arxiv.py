# coding=utf-8
import copy
import os
from argparse import ArgumentParser
from torch_geometric.nn import GraphSAGE, GCN, GAT, GIN, MLP
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import itertools
import datetime
import sys
import logging
from tensorboardX import SummaryWriter
from dataset import load_nc_dataset,NCDataset

def get_dataset(dataset, year=None):
    ### Load and preprocess data ###
    if dataset == 'ogb-arxiv':
        dataset = load_nc_dataset(args.data_dir, 'ogb-arxiv', year=year)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']

    return dataset


def main(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.dataset == 'ogb-arxiv':
        tr_year, val_year = [[1950, 2011]], [[2011, 2014]]
        # 2014, 2016, 2018, 2020, C43=6
        te_years = [[2014, 2016], [2016, 2018], [2018, 2020]]
        dataset_tr = get_dataset(dataset='ogb-arxiv', year=tr_year[0])
        dataset_val = get_dataset(dataset='ogb-arxiv', year=val_year[0])
    else:
        raise ValueError('Invalid dataname')

    log_dir_in = os.path.join('../data', 'ogbn_arxiv', 'gen')
    if not os.path.exists(os.path.join(log_dir_in)):
        raise ValueError('Invalid data folder')

    te_file_list = os.listdir(os.path.join('../data', 'ogbn_arxiv', 'gen'))
    te_filtered_file_list = [filename for filename in te_file_list if filename.endswith('.pth')]
    feat_num = dataset_tr.d
    class_num = dataset_tr.c

    if args.model == 'GCN':
        encoder = GCN(feat_num, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'SAGE':
        encoder = GraphSAGE(feat_num, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                            num_layers=args.num_layers).to(device)
    elif args.model == 'GAT':
        encoder = GAT(feat_num, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'GIN':
        encoder = GIN(feat_num, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'MLP':
        encoder = MLP(channel_list=[feat_num, args.hid_dim, args.encoder_dim]).to(device)

    # note that here is no softmax activation function
    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, class_num), ).to(device)
    models = [encoder, cls_model]
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt'), map_location=device))
    cls_model.load_state_dict(torch.load(os.path.join(args.model_path, 'cls_model.pt'), map_location=device))

    logging.info('Start with {}# num of graphs...'.format(len(te_filtered_file_list)))
    te_data_acc_ls = []
    for i in range(len(te_filtered_file_list)):
        dataset_te = torch.load(os.path.join(log_dir_in,te_filtered_file_list[i]))
        te_data_acc = test(dataset_te, models, encoder, cls_model)
        te_data_acc_ls.append(te_data_acc.item())
        logging.info(
            'Gen Test graphs ACC = {} in PATH = {} info num nodes= {} | num classes = {} | num node feats = {} | num_edges = {} | neg_num_edges = {}'.format(
                round(te_data_acc.item() * 100., 2),
                te_filtered_file_list[i], dataset_te.n, dataset_te.c, dataset_te.d,
                dataset_te.graph['edge_index'].shape[1],
                dataset_te.graph['neg_edge_index'].shape[1]))

    t_full_acc_np = np.array(te_data_acc_ls)

    logging.info(
        'Gen Test #num of {} graphs avg ACC = {} '.format(len(te_data_acc_ls),
                                                          np.mean(t_full_acc_np) * 100.))

    logging.info('TOTAL EVAL # num graph = {}'.format(t_full_acc_np.shape[0]))
    all_err_np = 1 - t_full_acc_np

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(all_err_np, bins=50, density=True, color='blue', alpha=0.7)
    plt.title('Error Distribution (Ground Truth)')
    plt.xlabel('Error Value')
    plt.ylabel('Density')
    plt.grid(True)

    # Generate the save path with modes included in the file name
    save_path = os.path.join(log_dir, f'error_distribution_all_gt.png')

    plt.savefig(save_path)
    plt.clf()


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()
    with torch.no_grad():
        if isinstance(encoder, MLP):
            emb_out = encoder(data.graph['node_feat'].to(device))
        else:
            emb_out = encoder(data.graph['node_feat'].to(device), data.graph['edge_index'].to(device))

        logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
        probs = F.log_softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        labels = data.label.squeeze().to(device) if mask is None else data.label.squeeze()[mask].to(device)
        corrects = preds.eq(labels)
        accuracy = corrects.float().mean()
    return accuracy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='ogb-arxiv')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()
    device = torch.device(args.device)
    # Generate the save path with modes included in the file name
    log_dir = 'logs/Gentest/{}-{}-{}-{}'.format(args.dataset, args.model,
                                                str(args.seed),
                                                datetime.datetime.now().strftime(
                                                    "%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'Gentest.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    writer = SummaryWriter(log_dir + '/tbx_log')
    logging.info(args)
    main(args, device)
    logging.info('Finish! This is the logdir = {}'.format(log_dir))

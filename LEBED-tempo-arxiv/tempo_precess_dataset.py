# coding=utf-8
import copy
import os
from argparse import ArgumentParser
import random
import numpy as np
import torch
import sys
import logging
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, negative_sampling
import torch_geometric as tg
from dataset import load_nc_dataset,NCDataset
import datetime

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


def NodeFeat(dataset,p, type):
    x = copy.deepcopy(dataset.graph['node_feat'])
    y = copy.deepcopy(dataset.label)
    edge_idx = copy.deepcopy(dataset.graph['edge_index'])
    n,d=x.shape[0], x.shape[1]
    if type == 'noise':
        mean, std = 0, p
        noise = torch.randn(x.size()) * std + mean
        x_new = x + noise
        new_dataset = NCDataset(name='node_feat_g_noise_' + str(round(p,2)))
    elif type == 'mkidx':
        idx = torch.empty((x.shape[-1],), dtype=torch.float32).uniform_(0, 1) < p
        x_new = x
        x_new[:, idx] = 0
        new_dataset = NCDataset(name='node_feat_g_mkidx_' + str(round(p,2)))

    neg_edge_index = negative_sampling(
        edge_index=edge_idx, num_nodes=n,
        num_neg_samples=edge_idx.size(1), method='sparse')

    new_dataset.graph = {'edge_index': edge_idx,
                     'node_feat': x_new,
                     'edge_feat': None,
                     'num_nodes': n,
                     'neg_edge_index': neg_edge_index}

    if len(y.shape) == 1:
        new_dataset.label = y.unsqueeze(1)
    else:
        new_dataset.label = y


    new_dataset.n = new_dataset.graph['num_nodes']
    new_dataset.c = max(y.max().item() + 1, y.shape[1])
    new_dataset.d = new_dataset.graph['node_feat'].shape[1]

    return new_dataset

def EdgeDrop(dataset,p):
    x = copy.deepcopy(dataset.graph['node_feat'])
    y = copy.deepcopy(dataset.label)
    edge_idx = copy.deepcopy(dataset.graph['edge_index'])
    n, d = x.shape
    new_edge_index, _ = dropout_adj(edge_idx, None,  p=p)

    neg_edge_index = negative_sampling(
        edge_index=new_edge_index, num_nodes=n,
        num_neg_samples=new_edge_index.size(1), method='sparse')
    new_dataset = NCDataset(name='edge_drop_g_' + str(round(p, 2)))

    new_dataset.graph = {'edge_index': new_edge_index,
                         'node_feat': x,
                         'edge_feat': None,
                         'num_nodes': n,
                         'neg_edge_index': neg_edge_index}

    if len(y.shape) == 1:
        new_dataset.label = y.unsqueeze(1)
    else:
        new_dataset.label = y

    new_dataset.n = new_dataset.graph['num_nodes']
    new_dataset.c = max(y.max().item() + 1, y.shape[1])
    new_dataset.d = new_dataset.graph['node_feat'].shape[1]

    return new_dataset


def random_extract(tensor, total_elements, num_elements):
    indices = torch.randperm(total_elements)[:num_elements]  # Randomly permuted indices
    extracted_elements = tensor.view(-1)[indices]  # Select elements using the indices
    return extracted_elements


def random_extract_class_dis(yin, test_idx, extract_num):
    y = yin.reshape(-1)
    unique_classes, class_counts = torch.unique(y[test_idx], return_counts=True)
    class_extract_nums = (class_counts * extract_num // torch.sum(class_counts)).tolist()

    extracted_indices = []
    for class_label, extract_count in zip(unique_classes, class_extract_nums):
        class_indices = test_idx[y[test_idx] == class_label]
        selected_indices = torch.randperm(class_indices.size(0))[:extract_count]
        selected_elements = class_indices[selected_indices]
        extracted_indices.append(selected_elements)

    extracted_indices = torch.cat(extracted_indices, dim=0)
    shuffled_indices = torch.randperm(extracted_indices.size(0))
    extracted_indices = extracted_indices[shuffled_indices]

    return extracted_indices


def subG(dataset,p):
    x = copy.deepcopy(dataset.graph['node_feat'])
    y = copy.deepcopy(dataset.label)
    edge_idx = copy.deepcopy(dataset.graph['edge_index'])
    n, d = x.shape
    total_test_num = n
    adj = to_scipy_sparse_matrix(edge_idx).tocsr()
    numin = int(total_test_num * p)
    sampled_idx = random_extract_class_dis(y, torch.arange(x.shape[0]), numin)
    x_new_sub = x[sampled_idx]
    y_new_G_sub = y[sampled_idx]
    valid_sub_idx = index_correct(adj, sampled_idx)
    edge_new_G_idx = from_scipy_sparse_matrix(adj[valid_sub_idx, :][:, valid_sub_idx])[0]
    neg_edge_index = negative_sampling(
        edge_index=edge_new_G_idx, num_nodes=x_new_sub.shape[0],
        num_neg_samples=edge_new_G_idx.size(1), method='sparse')
    new_dataset = NCDataset(name='sub_g_' + str(round(p, 2)))

    new_dataset.graph = {'edge_index': edge_new_G_idx,
                         'node_feat': x_new_sub,
                         'edge_feat': None,
                         'num_nodes': x_new_sub.shape[0],
                         'neg_edge_index': neg_edge_index}

    if len(y.shape) == 1:
        new_dataset.label = y_new_G_sub.unsqueeze(1)
    else:
        new_dataset.label = y_new_G_sub

    new_dataset.n = new_dataset.graph['num_nodes']
    new_dataset.c = max(y_new_G_sub.max().item() + 1, y_new_G_sub.shape[1])
    new_dataset.d = new_dataset.graph['node_feat'].shape[1]

    return new_dataset

def index_correct(infer_Gadj,infer_Gtest_idx):
    num_rows, num_cols = infer_Gadj.shape

    # Check if the indices are out of bounds
    out_of_bounds_rows = torch.logical_or(infer_Gtest_idx < 0, infer_Gtest_idx >= num_rows)
    out_of_bounds_cols = torch.logical_or(infer_Gtest_idx < 0, infer_Gtest_idx >= num_cols)

    # Find the valid indices
    out_of_bounds = torch.logical_or(out_of_bounds_rows, out_of_bounds_cols)
    valid_test_idx = infer_Gtest_idx[torch.logical_not(out_of_bounds)]

    return valid_test_idx

def main(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    #da_test_ls = []
    if args.dataset == 'ogb-arxiv':
        tr_year, val_year = [[1950, 2011]], [[2011, 2014]]
        #2014, 2016, 2018, 2020, C43=6
        te_years = [[2014, 2016], [2016, 2018], [2018, 2020]]
        dataset_tr = get_dataset(dataset='ogb-arxiv', year=tr_year[0])
        dataset_val = get_dataset(dataset='ogb-arxiv', year=val_year[0])
    else:
        raise ValueError('Invalid dataname')

    log_dir = os.path.join('../data', 'ogbn_arxiv', 'gen')
    if not os.path.exists(os.path.join(log_dir)):
        os.makedirs(os.path.join(log_dir))
    
    for sub in range(len(te_years)):
        dataset_te = get_dataset(dataset='ogb-arxiv', year=te_years[sub])
        logging.info(
            'Test graphs info num nodes= {} | num classes = {} | num node feats = {} | num_edges = {} in year scope = {}'.format(dataset_te.n, dataset_te.c, dataset_te.d, dataset_te.graph['edge_index'].shape[1],te_years[sub]))
        p_cnt = 0

        for p in np.random.uniform(0.1, 0.7, 30).tolist():
            node_feat_g_noise = NodeFeat(dataset_te, p, type='noise')
            node_feat_g_mkidx = NodeFeat(dataset_te, p, type='mkidx')
            sub_g = subG(dataset_te, p)
            edge_drop_g = EdgeDrop(dataset_te, p)

            p_cnt += 1
            logging.info(
                '{}-th Test graphs = {} info num nodes= {} | num classes = {} | num node feats = {} | num_edges = {} | neg_num_edges = {} in year scope = {}'.format(p_cnt, str('node_feat_g_noise'),
                    node_feat_g_noise.n, node_feat_g_noise.c, node_feat_g_noise.d, node_feat_g_noise.graph['edge_index'].shape[1], node_feat_g_noise.graph['neg_edge_index'].shape[1], te_years[sub]))

            logging.info(
                '{}-th Test graphs = {} info num nodes= {} | num classes = {} | num node feats = {} | num_edges = {} | neg_num_edges = {} in year scope = {}'.format(p_cnt,str('node_feat_g_mkidx'),
                    node_feat_g_mkidx.n, node_feat_g_mkidx.c, node_feat_g_mkidx.d, node_feat_g_mkidx.graph['edge_index'].shape[1], node_feat_g_mkidx.graph['neg_edge_index'].shape[1], te_years[sub]))

            logging.info(
                '{}-th Test graphs = {} info num nodes= {} | num classes = {} | num node feats = {} | num_edges = {} | neg_num_edges = {} in year scope = {}'.format(p_cnt,str('sub_g'),
                    sub_g.n, sub_g.c, sub_g.d, sub_g.graph['edge_index'].shape[1], sub_g.graph['neg_edge_index'].shape[1], te_years[sub]))
            logging.info(
                '{}-th Test graphs = {} info num nodes= {} | num classes = {} | num node feats = {} | num_edges = {} | neg_num_edges = {} in year scope = {}'.format(p_cnt,
                    str('edge_drop_g'),
                    edge_drop_g.n, edge_drop_g.c, edge_drop_g.d, edge_drop_g.graph['edge_index'].shape[1],
                    edge_drop_g.graph['neg_edge_index'].shape[1], te_years[sub]))

            if os.path.exists(os.path.join(log_dir, str(te_years[sub][0]) + '_node_feat_g_noise_' + str(p_cnt) + '.pth')):
                logging.info('careful overlapping!')
                break
            elif os.path.exists(os.path.join(log_dir, str(te_years[sub][0]) + '_node_feat_g_mkidx_' + str(p_cnt) + '.pth')):
                logging.info('careful overlapping!')
                break
            elif os.path.exists(os.path.join(log_dir, str(te_years[sub][0]) + '_sub_g_' + str(p_cnt) + '.pth')):
                logging.info('careful overlapping!')
                break
            elif os.path.exists(os.path.join(log_dir, str(te_years[sub][0]) + '_edge_drop_g_' + str(p_cnt) + '.pth')):
                logging.info('careful overlapping!')
                break
            else:
                torch.save(node_feat_g_noise,
                           os.path.join(log_dir, str(te_years[sub][0]) + '_node_feat_g_noise_' + str(p_cnt) + '.pth'))
                torch.save(node_feat_g_mkidx,
                           os.path.join(log_dir, str(te_years[sub][0]) + '_node_feat_g_mkidx_' + str(p_cnt) + '.pth'))
                torch.save(sub_g, os.path.join(log_dir, str(te_years[sub][0]) + '_sub_g_' + str(p_cnt) + '.pth'))
                torch.save(edge_drop_g, os.path.join(log_dir, str(te_years[sub][0]) + '_edge_drop_g_' + str(p_cnt) + '.pth'))
                logging.info('{} prob turb in {}'.format(p, os.path.join(log_dir,
                                                                         str(te_years[sub][0]) + '_node_feat_g_noise_' + str(
                                                                             p_cnt) + '.pth')))
                logging.info('{} prob turb in {}'.format(p, os.path.join(log_dir,
                                                                         str(te_years[sub][0]) + '_node_feat_g_mkidx_' + str(
                                                                             p_cnt) + '.pth')))
                logging.info(
                    '{} prob turb in {}'.format(p, os.path.join(log_dir, str(te_years[sub][0]) + '_sub_g_' + str(p_cnt) + '.pth')))
                logging.info('{} prob turb in {}'.format(p, os.path.join(log_dir, str(te_years[sub][0]) + '_edge_drop_g_' + str(
                    p_cnt) + '.pth')))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='ogb-arxiv')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    device = torch.device(args.device)

    #log_in_dir = '../data/ogb-arxiv-log-{}'.format(datetime.datetime.now().strftime(
    #                                                    "%Y%m%d-%H%M%S-%f"))
    log_in_dir = '../data/ogb-arxiv-log-20230910-105858-050994'
    if os.path.exists(log_in_dir):
        assert False

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_in_dir, 'Gprocess.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)
    main(args, device)
    logging.info('Finish, log = {}'.format(log_in_dir))

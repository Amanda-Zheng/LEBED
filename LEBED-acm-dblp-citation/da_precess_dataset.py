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

def NodeFeat(dataset,p, type):
    x = copy.deepcopy(dataset.x)
    y = copy.deepcopy(dataset.y)
    edge_idx = copy.deepcopy(dataset.edge_index)
    neg_edge_idx = copy.deepcopy(dataset.neg_edge_index)

    if type == 'noise':
        mean, std = 0, p
        noise = torch.randn(x.size()) * std + mean
        x_new = x + noise
    elif type == 'mkidx':
        idx = torch.empty((x.shape[-1],), dtype=torch.float32).uniform_(0, 1) < p
        x_new = x
        x_new[:, idx] = 0
    
    new_data = tg.data.Data(x=x_new, edge_index=edge_idx, y=y, neg_edge_index=neg_edge_idx)
    return new_data


def EdgeDrop(dataset,p):
    x = copy.deepcopy(dataset.x)
    y = copy.deepcopy(dataset.y)
    edge_idx = copy.deepcopy(dataset.edge_index)
    n, d = x.shape
    new_edge_index, _ = dropout_adj(edge_idx, None,  p=p)

    new_neg_edge_index = negative_sampling(
        edge_index=new_edge_index, num_nodes=n,
        num_neg_samples=new_edge_index.size(1), method='sparse')
    new_data = tg.data.Data(x=x, edge_index=new_edge_index, y=y, neg_edge_index=new_neg_edge_index)

    return new_data


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
    x = copy.deepcopy(dataset.x)
    y = copy.deepcopy(dataset.y)
    edge_idx = copy.deepcopy(dataset.edge_index)
    n, d = x.shape
    total_test_num = n
    adj = to_scipy_sparse_matrix(edge_idx).tocsr()
    numin = int(total_test_num * p)
    sampled_idx = random_extract_class_dis(y, torch.arange(x.shape[0]), numin)
    x_new_sub = x[sampled_idx]
    y_new_G_sub = y[sampled_idx]
    valid_sub_idx = index_correct(adj, sampled_idx)
    edge_new_G_idx = from_scipy_sparse_matrix(adj[valid_sub_idx, :][:, valid_sub_idx])[0]
    new_neg_edge_index = negative_sampling(
        edge_index=edge_new_G_idx, num_nodes=x_new_sub.shape[0],
        num_neg_samples=edge_new_G_idx.size(1), method='sparse')
    new_data = tg.data.Data(x=x_new_sub, edge_index=edge_new_G_idx, y=y_new_G_sub, neg_edge_index=new_neg_edge_index)

    return new_data

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
    for item in [args.source, args.target1, args.target2]:
        log_dir = os.path.join('../data', item, 'gen')
        if not os.path.exists(os.path.join(log_dir)):
            os.makedirs(os.path.join(log_dir))
        for sub in ['train_sub.pt', 'val_sub.pt','te_sub.pt','full_g.pt']:
            da_test_data = torch.load(os.path.join('../data',item,'induced', sub)).to(device)
            logging.info('DA Test graphs info = {} in path = {}'.format(da_test_data, os.path.join('../data',item,'induced', sub)))
            #da_test_ls.append(da_test_data)
            log_dir = os.path.join('../data',item,'gen')
            p_cnt=0
            for p in np.random.uniform(0.1, 0.7, 10).tolist():
                node_feat_g_noise = NodeFeat(da_test_data, p, type='noise')
                node_feat_g_mkidx = NodeFeat(da_test_data, p, type='mkidx')
                sub_g = subG(da_test_data, p)
                edge_drop_g = EdgeDrop(da_test_data, p)
                p_cnt+=1
                if os.path.exists(os.path.join(log_dir, sub[:-3]+'_node_feat_g_noise_' + str(p_cnt) + '.pth')):
                    logging.info('careful overlapping!')
                    break
                elif os.path.exists(os.path.join(log_dir, sub[:-3]+'_node_feat_g_mkidx_' + str(p_cnt) + '.pth')):
                    logging.info('careful overlapping!')
                    break
                elif os.path.exists(os.path.join(log_dir, sub[:-3]+'_sub_g_' + str(p_cnt) + '.pth')):
                    logging.info('careful overlapping!')
                    break
                elif os.path.exists(os.path.join(log_dir, sub[:-3]+'_edge_drop_g_' + str(p_cnt) + '.pth')):
                    logging.info('careful overlapping!')
                    break
                else:
                    torch.save(node_feat_g_noise,
                               os.path.join(log_dir, sub[:-3]+'_node_feat_g_noise_' + str(p_cnt) + '.pth'))
                    torch.save(node_feat_g_mkidx,
                               os.path.join(log_dir, sub[:-3]+'_node_feat_g_mkidx_' + str(p_cnt) + '.pth'))
                    torch.save(sub_g, os.path.join(log_dir, sub[:-3]+'_sub_g_' + str(p_cnt) + '.pth'))
                    torch.save(edge_drop_g, os.path.join(log_dir, sub[:-3]+'_edge_drop_g_' + str(p_cnt) + '.pth'))
                    logging.info('{} prob turb in {}'.format(p, os.path.join(log_dir, sub[:-3]+'_node_feat_g_noise_' + str(p_cnt) + '.pth')))
                    logging.info('{} prob turb in {}'.format(p, os.path.join(log_dir, sub[:-3]+'_node_feat_g_mkidx_' + str(p_cnt) + '.pth')))
                    logging.info(
                        '{} prob turb in {}'.format(p, os.path.join(log_dir, sub[:-3]+'_sub_g_' + str(p_cnt) + '.pth')))
                    logging.info('{} prob turb in {}'.format(p, os.path.join(log_dir, sub[:-3]+'_edge_drop_g_' + str(p_cnt) + '.pth')))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--target1", type=str, default='dblp')
    parser.add_argument("--target2", type=str, default='network')
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    device = torch.device(args.device)

    log_in_dir = '../' + 'data/ACM-DBLP-CITATION/'


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

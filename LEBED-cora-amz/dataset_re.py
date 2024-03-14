from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url

from os import path
import os
import pickle as pkl
import pickle
from torch_geometric.utils import negative_sampling
import logging

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_nc_dataset(data_dir, dataname, sub_dataname='', gen_model='gcn'):
    """ Loader for NCDataset
        Returns NCDataset
    """
    if dataname in  ('cora', 'amazon-photo'):
        dataset, feat_noise_ls, feat_mkidx_ls = load_synthetic_dataset(data_dir, dataname, sub_dataname, gen_model)
    else:
        raise ValueError('Invalid dataname')
    return dataset, feat_noise_ls, feat_mkidx_ls

def load_synthetic_dataset(data_dir, name, lang, gen_model='gcn'):
    dataset = NCDataset(lang)

    assert lang in range(0, 10), 'Invalid dataset'

    if name == 'cora':
        node_feat, y = pkl.load(open('{}/Planetoid/cora/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Planetoid(root='{}/Planetoid'.format(data_dir),
                              name='cora')
    elif name == 'amazon-photo':
        node_feat, y = pkl.load(open('{}/Amazon/Photo/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Amazon(root='{}/Amazon'.format(data_dir),
                                  name='Photo')
    data = torch_dataset[0]

    edge_index = data.edge_index
    # label = data.y
    label = y
    num_nodes = node_feat.size(0)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = label
    edge_label_index = dataset.graph['edge_index']

    if name == 'cora':
        filename = '{}/Planetoid/cora/gen/{}-{}-negedge.pkl'.format(data_dir, lang, gen_model)
        if os.path.exists(filename):
            logging.info('loadding from existing ....')
            with open(filename, 'rb') as f:
                neg_edge_index = pickle.load(f)
        else:
            neg_edge_index = negative_sampling(
                edge_index=dataset.graph['edge_index'], num_nodes=num_nodes,
                num_neg_samples=edge_label_index.size(1), method='sparse')
            with open(filename, 'wb') as f:
                pickle.dump(neg_edge_index, f)

        filename_feat_noise = '{}/Planetoid/cora/gen/{}-{}-feat_noise.pkl'.format(data_dir, lang, gen_model)
        filename_feat_mkidx = '{}/Planetoid/cora/gen/{}-{}-feat_mkidx.pkl'.format(data_dir, lang, gen_model)
        if os.path.exists(filename_feat_noise):
            logging.info('loadding feat_noise from existing ....')
            with open(filename_feat_noise, 'rb') as f:
                feat_noise_ls = pickle.load(f)
        if os.path.exists(filename_feat_mkidx):
            logging.info('loadding feat_mkidx from existing ....')
            with open(filename_feat_mkidx, 'rb') as f:
                feat_mkidx_ls = pickle.load(f)

        else:
            feat_noise_ls = []
            feat_mkidx_ls = []
            logging.info('creating feat transforms....')
            for p in np.random.uniform(0.1, 0.7, 20).tolist():
                feat_noise, feat_idx = NodeFeat_te(node_feat, p)
                feat_noise_ls.append(feat_noise)
                feat_mkidx_ls.append(feat_idx)
            try:
                with open(filename_feat_noise, 'wb') as f:
                    pickle.dump(feat_noise_ls, f)
            except Exception as e:
                logging.error(f"Error while saving feat_noise: {e}")

            try:
                with open(filename_feat_mkidx, 'wb') as f:
                    pickle.dump(feat_mkidx_ls, f)
            except Exception as e:
                logging.error(f"Error while saving feat_mkidx: {e}")

    elif name == 'amazon-photo':
        filename = '{}/Amazon/Photo/gen/{}-{}-negedge.pkl'.format(data_dir, lang, gen_model)
        if os.path.exists(filename):
            logging.info('loadding from existing ....')
            with open(filename, 'rb') as f:
                neg_edge_index = pickle.load(f)
        else:
            neg_edge_index = negative_sampling(
                edge_index=dataset.graph['edge_index'], num_nodes=num_nodes,
                num_neg_samples=edge_label_index.size(1), method='sparse')
            with open(filename, 'wb') as f:
                pickle.dump(neg_edge_index, f)

        filename_feat_noise = '{}/Amazon/Photo/gen/{}-{}-feat_noise.pkl'.format(data_dir, lang, gen_model)
        filename_feat_mkidx = '{}/Amazon/Photo/gen/{}-{}-feat_mkidx.pkl'.format(data_dir, lang, gen_model)
        if os.path.exists(filename_feat_noise):
            logging.info('loadding feat_noise from existing ....')
            with open(filename_feat_noise, 'rb') as f:
                feat_noise_ls = pickle.load(f)
        if os.path.exists(filename_feat_mkidx):
            logging.info('loadding feat_mkidx from existing ....')
            with open(filename_feat_mkidx, 'rb') as f:
                feat_mkidx_ls = pickle.load(f)

        else:
            feat_noise_ls = []
            feat_mkidx_ls = []
            logging.info('creating feat transforms....')
            for p in np.random.uniform(0.1, 0.7, 20).tolist():
                feat_noise, feat_idx = NodeFeat_te(node_feat, p)
                feat_noise_ls.append(feat_noise)
                feat_mkidx_ls.append(feat_idx)
            try:
                with open(filename_feat_noise, 'wb') as f:
                    pickle.dump(feat_noise_ls, f)
            except Exception as e:
                logging.error(f"Error while saving feat_noise: {e}")

            try:
                with open(filename_feat_mkidx, 'wb') as f:
                    pickle.dump(feat_mkidx_ls, f)
            except Exception as e:
                logging.error(f"Error while saving feat_mkidx: {e}")

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes,
                     'neg_edge_index':neg_edge_index}

    return dataset,feat_noise_ls,feat_mkidx_ls


def NodeFeat_te(feat, p):
    #for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #np.random.uniform(1e-5, 0.5, 10)
    mean, std = 0, p
    noise = torch.randn(feat.size()) * std + mean
    idx = torch.empty((feat.shape[-1],), dtype=torch.float32).uniform_(0, 1) < p
    #feat = feat + noise.to(feat.device)
    # x = x.clone()
    # x[:, idx] = 0
    return noise, idx


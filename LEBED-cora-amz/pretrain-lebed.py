# coding=utf-8
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
import copy
from tensorboardX import SummaryWriter
import scipy.sparse as sp
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops

from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits

def main(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'cora':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = args.gnn_gen
        dataset_tr = get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model)
        dataset_val = get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model)
        datasets_te = [get_dataset(dataset='cora', sub_dataset=te_subs[i], gen_model=gen_model) for i in
                       range(len(te_subs))]
    elif args.dataset == 'amazon-photo':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = args.gnn_gen
        dataset_tr = get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model)
        dataset_val = get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model)
        datasets_te = [get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i], gen_model=gen_model) for i in
                       range(len(te_subs))]
    else:
        raise ValueError('Invalid dataname')

    print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
    print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
    for i in range(len(te_subs)):
        dataset_te = datasets_te[i]
        print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

    # loss_func = nn.CrossEntropyLoss().to(device)
    train_data = dataset_tr
    val_data = dataset_val
    loss_func = nn.NLLLoss().to(device)
    feat_dim = train_data.d
    num_class = train_data.c
    if args.model == 'GCN':
        encoder = GCN(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'SAGE':
        encoder = GraphSAGE(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                            num_layers=args.num_layers).to(device)
    elif args.model == 'GAT':
        encoder = GAT(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'GIN':
        encoder = GIN(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'MLP':
        encoder = MLP(in_channels=feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, num_class), ).to(device)
    models = [encoder, cls_model]
    params = itertools.chain(*[model.parameters() for model in models])
    if args.dataset!='cora':
        if os.path.exists(os.path.join('init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim),'init_encoder.pt')):
            logging.info('Loading existing initial params...')
            encoder.load_state_dict(torch.load(os.path.join('init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_encoder.pt'), map_location=device))
            cls_model.load_state_dict(torch.load(os.path.join('init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_cls_model.pt'), map_location=device))
        else:
            os.makedirs(os.path.join('init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim)))
            # note that here is no softmax activation function
            logging.info('Params initialization...')
            nn.init.xavier_uniform_(cls_model[0].weight)
            nn.init.constant_(cls_model[0].bias, 0.0)
            torch.save(encoder.state_dict(), os.path.join('init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_encoder.pt'))
            torch.save(cls_model.state_dict(), os.path.join('init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_cls_model.pt'))
    elif args.dataset=='cora':
        if os.path.exists(os.path.join('init_params', str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim),'init_encoder.pt')):
            logging.info('Loading existing initial params...')
            encoder.load_state_dict(torch.load(os.path.join('init_params', str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_encoder.pt'), map_location=device))
            cls_model.load_state_dict(torch.load(os.path.join('init_params', str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_cls_model.pt'), map_location=device))
        else:
            os.makedirs(os.path.join('init_params', str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim)))
            # note that here is no softmax activation function
            logging.info('Params initialization...')
            nn.init.xavier_uniform_(cls_model[0].weight)
            nn.init.constant_(cls_model[0].bias, 0.0)
            torch.save(encoder.state_dict(), os.path.join('init_params', str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_encoder.pt'))
            torch.save(cls_model.state_dict(), os.path.join('init_params', str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_cls_model.pt'))
    #flattened_params_init = torch.cat([param.view(-1) for param in params])
    #torch.save(flattened_params_init, os.path.join('init_params', str(args.model)+'_init_params.pth'))
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc_ls = []
    best_epoch = 0.0

    for epoch in range(0, args.epochs):

        train_acc, train_loss = train(models, encoder, cls_model, optimizer, loss_func, train_data)
        val_acc = test(val_data, models, encoder, cls_model)
        test_acc_ls = []
        for i in range(len(te_subs)):
            dataset_te = datasets_te[i]
            test_data = dataset_te
            test_acc = test(test_data, models, encoder, cls_model)
            test_acc_ls.append(round(test_acc.cpu().detach().numpy()* 100.,2))
        #infer_test_acc = test(infer_data, models, encoder, cls_model, mask=infer_data.test_mask)
        logging.info(
            'Epoch: {}, train_loss = {:.4f}, train_acc = {:.2f}, val_acc = {:.2f}, test_acc_all = {}'.format(
                epoch,
                train_loss,
                train_acc * 100.,
                val_acc * 100.,
                test_acc_ls))

        writer.add_scalar('curve/acc_train_seed_' + str(args.seed), train_acc, epoch)
        writer.add_scalar('curve/acc_val_seed_' + str(args.seed), val_acc, epoch)
        writer.add_scalar('curve/acc_test_seed_' + str(args.seed), test_acc, epoch)
        writer.add_scalar('curve/loss_train_seed_' + str(args.seed), train_loss, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc_ls = test_acc_ls
            best_train_acc = train_acc
            best_epoch = epoch
            torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pt'))
            torch.save(cls_model.state_dict(), os.path.join(log_dir, 'cls_model.pt'))

    logging.info(
        'Best Epoch: {}, best_train_acc = {:.2f}, best_val_acc = {:.2f}, best_test_acc = {}'.format(
            best_epoch,
            best_train_acc * 100.,
            best_val_acc * 100.,
            best_test_acc_ls))


def get_dataset(dataset, sub_dataset=None, gen_model=None):
    ### Load and preprocess data ###
    if dataset == 'cora':
        dataset = load_nc_dataset(args.data_dir, 'cora', sub_dataset, gen_model)
    elif dataset == 'amazon-photo':
        dataset = load_nc_dataset(args.data_dir, 'amazon-photo', sub_dataset, gen_model)
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

def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()

    #edge_index_with_self_loop = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    #edge_index_in = torch.unique(edge_index_with_self_loop.T, dim=0, return_inverse=False).T

    if isinstance(encoder, MLP):
        emb_out = encoder(data.graph['node_feat'].to(device))
    else:
        emb_out = encoder(data.graph['node_feat'].to(device), data.graph['edge_index'].to(device))

    logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs)
    preds = probs.argmax(dim=1)
    labels = data.label.squeeze().to(device) if mask is None else data.label.squeeze()[mask].to(device)
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def train(models, encoder, cls_model, optimizer, loss_func, train_data):
    for model in models:
        model.train()


    #edge_index_with_self_loop = add_self_loops(train_data.edge_index, num_nodes=train_data.num_nodes)[0]
    #edge_index_in = torch.unique(edge_index_with_self_loop.T, dim=0, return_inverse=False).T
    if isinstance(encoder, MLP):
        emb_source = encoder(train_data.graph['node_feat'].to(device))
    else:
        emb_source = encoder(train_data.graph['node_feat'].to(device), train_data.graph['edge_index'].to(device))

    source_logits = cls_model(emb_source)
    source_probs = F.softmax(source_logits, dim=1)
    #log_source_probs = torch.log(source_probs)
    #if args.model == 'GIN':
    # to keep numerical stability
    max_logits, _ = source_logits.max(dim=1, keepdim=True)  # Find the maximum logits along each row
    log_source_probs = source_logits - max_logits  # Subtract the maximum logits to avoid overflow
    log_source_probs = log_source_probs - torch.logsumexp(log_source_probs, dim=1, keepdim=True)
    #else:
    #    log_source_probs = torch.log(source_probs)
    cls_loss = loss_func(log_source_probs, train_data.label.squeeze().to(device))

    preds = source_probs.argmax(dim=1)
    labels = train_data.label.squeeze().to(device)

    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()

    loss = cls_loss
    optimizer.zero_grad()
    loss.backward()
    #if args.model=='GIN':
    #    params = itertools.chain(*[model.parameters() for model in models])
    #    max_grad_norm = 0.5  # Set your desired maximum gradient norm
    #    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()

    return accuracy, loss.item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--gnn_gen', type=str, default='gcn', choices=['gcn', 'gat', 'sgc'],
                        help='random initialized gnn for data generation')
    # parser.add_argument("--version", type=str, required=True)
    # parser.add_argument("--version", type=str, default='debug')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--encoder_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()
    device = torch.device(args.device)
    log_dir = 'logs/Model_pretrain/{}-{}-{}-gen{}-{}'.format(args.dataset, args.model, str(args.seed), str(args.gnn_gen),datetime.datetime.now().strftime(
                                                        "%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train_model.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    writer = SummaryWriter(log_dir + '/tbx_log')
    logging.info(args)
    main(args, device)
    logging.info(args)
    logging.info('Finish!, this is the log dir = {}'.format(log_dir))

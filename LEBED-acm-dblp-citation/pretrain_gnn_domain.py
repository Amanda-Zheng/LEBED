# coding=utf-8
import copy
import os
from argparse import ArgumentParser
from torch_geometric.nn import GraphSAGE, GCN, GAT, GIN, MLP
from dual_gnn.dataset.DomainData import DomainData
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import itertools
import datetime
import sys
import logging
from scipy import linalg
from tensorboardX import SummaryWriter
import torch_geometric as tg
import grafog.transforms as T
from torch_geometric import seed
import torch_geometric as tg
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, negative_sampling


def param_diff(param1, param2):
    """
    Returns:
        the l2 norm difference the two networks
    """
    diff = (torch.norm(param1 - param2) ** 2).cpu().detach().numpy()
    return np.sqrt(diff)

def main(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    str_data = torch.load(os.path.join('../data',args.source,'induced', 'train_sub.pt')).to(device)
    sval_data = torch.load(os.path.join('../data',args.source,'induced', 'val_sub.pt')).to(device)
    ste_data = torch.load(os.path.join('../data', args.source, 'induced', 'te_sub.pt')).to(device)
    da_test_ls = []
    for item in [args.target1, args.target2]:
        for sub in ['train_sub.pt', 'val_sub.pt','te_sub.pt']:
            da_test_data = torch.load(os.path.join('../data',item,'induced', sub)).to(device)
            logging.info('DA Test graphs info = {} in path = {}'.format(da_test_data, os.path.join('../data',item,'induced', sub)))
            da_test_ls.append(da_test_data)

    loss_func = nn.NLLLoss().to(device)

    feat_num = str_data.x.shape[1]
    class_num = max(str_data.y)+1
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
        encoder = MLP(channel_list=[feat_num,args.hid_dim, args.encoder_dim]).to(device)

    #note that here is no softmax activation function
    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, class_num), ).to(device)
    models = [encoder, cls_model]
    params = itertools.chain(*[model.parameters() for model in models])

    if os.path.exists(os.path.join('init_params',
                                   str(args.source) + '_' + str(args.model) + '_' + str(args.num_layers) + '_' + str(args.hid_dim) + '_' + str(
                                           args.encoder_dim), 'init_encoder.pt')):
        logging.info('Loading existing initial params...')
        encoder.load_state_dict(torch.load(os.path.join('init_params',
                                                        str(args.source) + '_' + str(args.model) + '_' + str(args.num_layers) + '_' + str(
                                                            args.hid_dim) + '_' + str(args.encoder_dim),
                                                        'init_encoder.pt'), map_location=device))
        cls_model.load_state_dict(torch.load(os.path.join('init_params',
                                                          str(args.source) + '_' + str(args.model) + '_' + str(args.num_layers) + '_' +  str(
                                                              args.hid_dim) + '_' + str(args.encoder_dim),
                                                          'init_cls_model.pt'), map_location=device))
    else:
        os.makedirs(os.path.join('init_params',
                                 str(args.source) + '_' + str(args.model) + '_' + str(args.num_layers) + '_' + str(args.hid_dim) + '_' + str(
                                     args.encoder_dim)))
        # note that here is no softmax activation function
        logging.info('Params initialization...')
        nn.init.xavier_uniform_(cls_model[0].weight)
        nn.init.constant_(cls_model[0].bias, 0.0)
        torch.save(encoder.state_dict(), os.path.join('init_params',
                                                      str(args.source) + '_' + str(args.model) + '_' + str(args.num_layers) + '_' +  str(
                                                          args.hid_dim) + '_' + str(args.encoder_dim),
                                                      'init_encoder.pt'))
        torch.save(cls_model.state_dict(), os.path.join('init_params',
                                                        str(args.source) + '_' + str(args.model) + '_' + str(args.num_layers) + '_' +  str(
                                                            args.hid_dim) + '_' + str(args.encoder_dim),
                                                        'init_cls_model.pt'))

    #init_params_list = [model.parameters() for model in models]
    #init_flattened_params = torch.cat([param.view(-1) for param in itertools.chain(*init_params_list)])
    #diff_norm_0 = param_diff(init_flattened_params, 0)
    #logging.info('diff_norm_0 = {}'.format(diff_norm_0))
    #assert False
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    best_s_val_acc = 0.0
    best_s_test_acc = 0.0
    best_epoch = 0.0
    t_full_acc_ls =[]
    t_full_acc_ls2 = []
    #s_train_acc_ls = []
    #s_val_acc_ls = []
    #s_test_acc_ls =[]
    for epoch in range(0, args.epochs):
        s_train_acc, s_train_loss = train(models, encoder, cls_model, optimizer, loss_func, str_data)
        s_val_acc = test(sval_data, models, encoder, cls_model)
        s_test_acc = test(ste_data, models, encoder, cls_model)
        t_full_acc_ls = []
        opt_params_list = [model.parameters() for model in models]
        flattened_params_opt = torch.cat([param.view(-1) for param in itertools.chain(*opt_params_list)])
        #diff_norm = param_diff(init_flattened_params, flattened_params_opt)
        #logging.info('Epoch: {}, Diff_norm: {:.4f}'.format(epoch, diff_norm))

        for da_test_data in da_test_ls:
            t_full_acc = test(da_test_data, models, encoder, cls_model)
            t_full_acc_ls.append(round(t_full_acc.item(),4))
        t_full_acc_np = np.array(t_full_acc_ls)

        logging.info('Epoch: {}, source_train_loss: {:.4f}, source_train_acc: {:.2f}, source_val_acc: {:.2f}, source_test_acc:{:.2f}'. \
                     format(epoch, s_train_loss, s_train_acc.item()*100., s_val_acc.item()*100., s_test_acc.item()*100.))
        logging.info('Epoch: {}, AVG_target_full_acc: {:.2f}, target_full_acc_mean = {}'. \
                     format(epoch, np.mean(t_full_acc_np)*100.,t_full_acc_np*100.))
        writer.add_scalar('curve/acc_source_train_seed_' + str(args.seed), s_train_acc, epoch)
        writer.add_scalar('curve/acc_source_val_seed_' + str(args.seed), s_val_acc, epoch)
        writer.add_scalar('curve/acc_source_test_seed_' + str(args.seed), s_test_acc, epoch)
        writer.add_scalar('curve/loss_source_train_seed_' + str(args.seed), s_train_loss, epoch)
        if s_val_acc.item() > best_s_val_acc:
            best_s_val_acc = s_val_acc.item()
            best_s_test_acc = s_test_acc.item()
            best_epoch = epoch
            best_target_acc = t_full_acc_np
            torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pt'))
            torch.save(cls_model.state_dict(), os.path.join(log_dir, 'cls_model.pt'))

    line = "Best Epoch: {}, best_source_test_acc: {:.2f}, best_source_val_acc: {:.2f}, AVG_best_target_acc: {:.2f}, best_target_acc: {}" \
        .format(best_epoch, best_s_test_acc*100., best_s_val_acc*100., np.mean(best_target_acc)*100., best_target_acc*100.)
    logging.info(line)
    logging.info(args)
    logging.info('Finish!, this is the log dir = {}'.format(log_dir))


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()
    with torch.no_grad():
        if isinstance(encoder, MLP):
            emb_out = encoder(data.x)
        else:
            emb_out = encoder(data.x, data.edge_index)

        logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
        probs = F.log_softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        labels = data.y if mask is None else data.y[mask]
        corrects = preds.eq(labels)
        accuracy = corrects.float().mean()
    return accuracy


def train(models, encoder, cls_model, optimizer, loss_func, source_data):
    for model in models:
        model.train()
    if isinstance(encoder, MLP):
        emb_source = encoder(source_data.x)
    else: 
        emb_source = encoder(source_data.x, source_data.edge_index)

    source_logits = cls_model(emb_source)
    source_probs = F.log_softmax(source_logits,dim=1)

    cls_loss = loss_func(source_probs, source_data.y)
    preds = source_probs.argmax(dim=1)
    labels = source_data.y

    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()

    loss = cls_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return accuracy, loss.item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--target1", type=str, default='dblp')
    parser.add_argument("--target2", type=str, default='network')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--encoder_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()
    device = torch.device(args.device)
    log_dir = 'logs/Models_tra/{}-to-{}-{}-{}-{}-{}'.format(args.source, args.target1, args.target2, args.model,
                                                             str(args.seed),
                                                             datetime.datetime.now().strftime(
                                                                 "%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'pretrain.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    writer = SummaryWriter(log_dir + '/tbx_log')
    logging.info(args)
    main(args, device)

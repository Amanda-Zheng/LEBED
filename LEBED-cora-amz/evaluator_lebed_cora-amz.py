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
from itertools import product
from sklearn.metrics import roc_auc_score
import numpy as np
import datetime
import sys
import logging
import copy
import torch.nn as nn
import torch.sparse
import torch.nn.init as init
from tensorboardX import SummaryWriter
import scipy.sparse as sp
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, negative_sampling
from dataset_re import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, \
    to_sparse_tensor, load_fixed_splits


def main(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.dataset == 'cora':
        tr_sub, val_sub = [0], [1]
        te_subs = list(range(2, 10))
        gen_model = args.gnn_gen
        dataset_tr, _, _ = get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model)
        dataset_val, _, _ = get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model)
        datasets_te = []
        feats_noise = []
        feats_mkidx = []

        for i in range(len(te_subs)):
            dataset_te, feat_noise, feat_mkidx = get_dataset(dataset='cora', sub_dataset=te_subs[i],
                                                             gen_model=gen_model)
            datasets_te.append(dataset_te)
            feats_noise.append(feat_noise)
            feats_mkidx.append(feat_mkidx)

    elif args.dataset == 'amazon-photo':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = args.gnn_gen
        dataset_tr, _, _ = get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model)
        dataset_val, _, _ = get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model)
        datasets_te = []
        feats_noise = []
        feats_mkidx = []

        for i in range(len(te_subs)):
            dataset_te, feat_noise, feat_mkidx = get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i],
                                                             gen_model=gen_model)
            datasets_te.append(dataset_te)
            feats_noise.append(feat_noise)
            feats_mkidx.append(feat_mkidx)

    else:
        raise ValueError('Invalid dataname')

    print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
    print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
    for i in range(len(datasets_te)):
        dataset_te = datasets_te[i]
        print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

    # loss_func = nn.CrossEntropyLoss().to(device)
    train_data = dataset_tr
    # loss_func = nn.NLLLoss().to(device)
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

    # note that here is no softmax activation function
    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, num_class), ).to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt'), map_location=device))
    cls_model.load_state_dict(torch.load(os.path.join(args.model_path, 'cls_model.pt'), map_location=device))

    models = [encoder, cls_model]
    params_list = [model.parameters() for model in models]
    train_flattened_params_opt = torch.cat([param.view(-1) for param in itertools.chain(*params_list)])
    test_error_ls = []
    diff_norm_g_ls = []
    test_learned_error_ls = []
    ae_ls = []

    for i in range(len(datasets_te)):
        # for i in range(1):
        dataset_te = datasets_te[i]
        test_data = dataset_te
        test_g_idx = i
        for flag_type in ['noise', 'mkidx']:
            if flag_type == 'noise':
                for augidx in range(len(feats_noise[i])):
                    aug = feats_noise[i][augidx]
                    test_emb_out, test_probs, test_pseudo, test_acc = test(test_data, models, encoder, cls_model, aug,
                                                                           type='noise')
                    test_edge_label_index = test_data.graph['edge_index'].to(device)
                    test_edge_label = torch.ones(test_edge_label_index.shape[1]).to(device)
                    neg_edge_index = test_data.graph['neg_edge_index'].to(device)

                    new_edge_label_index = torch.cat([test_edge_label_index, neg_edge_index], dim=-1).to(device)
                    new_edge_label = torch.cat([test_edge_label, test_edge_label.new_zeros(neg_edge_index.size(1))],
                                               dim=0)
                    logits_lp = F.cosine_similarity(test_emb_out[new_edge_label_index[0]],
                                                    test_emb_out[new_edge_label_index[1]],
                                                    dim=-1)
                    metric_lp_ref = F.binary_cross_entropy_with_logits(logits_lp, new_edge_label)

                    # train_lp(test_emb_out, test_data, test_g_idx)

                    diff_norm_g, test_learned_acc = train_norm(feat_dim, num_class, test_data, test_pseudo,
                                                               train_flattened_params_opt, metric_lp_ref, test_g_idx,aug,
                                                                           type='noise')
                    AE = torch.abs(test_acc - test_learned_acc).item()
                    ae_ls.append(AE)
                    test_error_ls.append(1 - test_acc.item())
                    test_learned_error_ls.append(1 - test_learned_acc.item())
                    diff_norm_g_ls.append(diff_norm_g)
                    logging.info(
                        'Test_graph_id = {} in NO.{}-{}, Test-ERR = {:.2f}, Diff-norm = {:.4f}, Learned_Err = {:.2f}, AE = {:.2f}'.format(
                            test_g_idx, augidx, flag_type,
                            test_error_ls[
                                -1] * 100.,
                            diff_norm_g_ls[-1],
                            test_learned_error_ls[
                                -1] * 100.,
                            ae_ls[-1] * 100.))
            elif flag_type == 'mkidx':
                for augidx in range(len(feats_mkidx[i])):
                    aug = feats_mkidx[i][augidx]
                    test_emb_out, test_probs, test_pseudo, test_acc = test(test_data, models, encoder, cls_model, aug,
                                                                           type='mkidx')
                    test_edge_label_index = test_data.graph['edge_index'].to(device)
                    test_edge_label = torch.ones(test_edge_label_index.shape[1]).to(device)
                    neg_edge_index = test_data.graph['neg_edge_index'].to(device)

                    new_edge_label_index = torch.cat([test_edge_label_index, neg_edge_index], dim=-1).to(device)
                    new_edge_label = torch.cat([test_edge_label, test_edge_label.new_zeros(neg_edge_index.size(1))],
                                               dim=0)
                    logits_lp = F.cosine_similarity(test_emb_out[new_edge_label_index[0]],
                                                    test_emb_out[new_edge_label_index[1]],
                                                    dim=-1)
                    metric_lp_ref = F.binary_cross_entropy_with_logits(logits_lp, new_edge_label)

                    # train_lp(test_emb_out, test_data, test_g_idx)

                    diff_norm_g, test_learned_acc = train_norm(feat_dim, num_class, test_data, test_pseudo,
                                                               train_flattened_params_opt, metric_lp_ref, test_g_idx,aug,
                                                                           type='mkidx')
                    AE = torch.abs(test_acc - test_learned_acc).item()
                    ae_ls.append(AE)
                    test_error_ls.append(1 - test_acc.item())
                    test_learned_error_ls.append(1 - test_learned_acc.item())
                    diff_norm_g_ls.append(diff_norm_g)
                    logging.info(
                        'Test_graph_id = {} in NO.{}-{}, Test-ERR = {:.2f}, Diff-norm = {:.4f}, Learned_Err = {:.2f}, AE = {:.2f}'.format(
                            test_g_idx, augidx, flag_type,
                            test_error_ls[
                                -1] * 100.,
                            diff_norm_g_ls[-1],
                            test_learned_error_ls[
                                -1] * 100.,
                            ae_ls[-1] * 100.))
    mae_np = np.array(ae_ls)
    mae_value = np.mean(mae_np) * 100.
    std_value = np.std(mae_np) * 100.
    logging.info('Learned-acc MAE = {:.2f} with std = {:.2f}'.format(mae_value, std_value))

    from scipy.stats import spearmanr

    all_err_np = np.array(test_error_ls)
    all_norm_np = np.array(diff_norm_g_ls)
    all_prederr_np = np.array(test_learned_error_ls)
    np.savez(os.path.join(log_dir, "GT-err_arrays.npz"), all_err=all_err_np)
    np.savez(os.path.join(log_dir, "Learned_err_arrays.npz"), all_err=all_prederr_np)
    np.savez(os.path.join(log_dir, "diffnorm_arrays.npz"), all_norm=all_norm_np)


    # Calculate the Spearman correlation coefficient and p-value
    corcoef, p = spearmanr(all_err_np, all_norm_np)
    corcoef2, p2 = spearmanr(all_err_np, all_prederr_np)

    logging.info('Pnorm-Spearman correlation coefficient = {:.2f} under P-value = {}'.format(corcoef, p))
    logging.info('Learned-Spearman correlation coefficient = {:.2f} under P-value = {}'.format(corcoef2, p2))
    # Create a linear regression model
    np_ls_err = [all_err_np]
    np_ls_norm = [all_norm_np]
    np_ls_learn_err = [all_prederr_np]
    # type_ls = ['ALL']
    # for idx in range(len(type_ls)):
    linearR(np_ls_err[0], np_ls_norm[0], 'Pnorm')
    linearR(np_ls_err[0], np_ls_learn_err[0], 'Learned')


    import matplotlib.pyplot as plt
    # Create the scatter plot
    plt.scatter(all_err_np, all_norm_np, label="All Groups")

    # Set labels and title
    plt.xlabel("Test Error")
    plt.ylabel("Normalized Difference")
    plt.title("Scatter plot of Correlation between Accuracy and Difference")

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(log_dir, "scatter_plot.png"))
    plt.clf()

    # all_norm_np = np.load(os.path.join(log_dir, "diffnorm_arrays.npz"))
    # all_norm = all_norm_np['all_norm']

    # Create a histogram plot for error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_err_np, bins=50, density=True, color='blue', alpha=0.7)
    plt.title('Error Distribution (Ground Truth)')
    plt.xlabel('Error Value')
    plt.ylabel('Density')
    plt.grid(True)
    # Save the figure as an image
    save_path = os.path.join(log_dir, "error_distribution_gt.png")
    plt.savefig(save_path)
    plt.clf()
    # Show the plot (optional)
    # plt.show()




def get_dataset(dataset, sub_dataset=None, gen_model=None):
    ### Load and preprocess data ###
    if dataset == 'cora':
        dataset, feat_noise_ls, feat_mkidx_ls = load_nc_dataset(args.data_dir, 'cora', sub_dataset, gen_model)
    elif dataset == 'amazon-photo':
        dataset, feat_noise_ls, feat_mkidx_ls = load_nc_dataset(args.data_dir, 'amazon-photo', sub_dataset, gen_model)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.graph['neg_edge_index'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.graph['neg_edge_index']

    return dataset, feat_noise_ls, feat_mkidx_ls


def linearR(acc_np, norm_np, type):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    model = LinearRegression()
    # Fit the model
    model.fit(acc_np.reshape(-1, 1), norm_np)
    # Make predictions
    y_pred = model.predict(acc_np.reshape(-1, 1))
    # Calculate the R-squared score
    r_squared = r2_score(norm_np, y_pred)
    logging.info('{}-R-squared score = {:2f}'.format(str(type), r_squared))


def train_norm(feat_dim, num_class, input_data, input_pseudo, train_flattened_params_opt, metric_lp_ref, test_g_idx, aug=None, type=None):
    if args.model == 'GCN':
        encoder_init = GCN(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                           num_layers=args.num_layers).to(device)
    elif args.model == 'SAGE':
        encoder_init = GraphSAGE(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                                 num_layers=args.num_layers).to(device)
    elif args.model == 'GAT':
        encoder_init = GAT(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                           num_layers=args.num_layers).to(device)
    elif args.model == 'GIN':
        encoder_init = GIN(feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                           num_layers=args.num_layers).to(device)
    elif args.model == 'MLP':
        encoder_init = MLP(in_channels=feat_dim, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                           num_layers=args.num_layers).to(device)

    # note that here is no softmax activation function
    cls_model_init = nn.Sequential(nn.Linear(args.encoder_dim, num_class), ).to(device)
    if args.dataset!='cora':
        if os.path.exists(
                os.path.join('../init_cora_amazon/init_params', str(args.dataset) + '_' + str(args.model) + '_' + str(args.hid_dim) + '_' + str(args.encoder_dim),
                             'init_encoder.pt')):
            logging.info('Loading existing initial params...')
            encoder_init.load_state_dict(torch.load(
                os.path.join('../init_cora_amazon/init_params', str(args.dataset) + '_' + str(args.model) + '_' + str(args.hid_dim) + '_' + str(args.encoder_dim),
                             'init_encoder.pt'),
                map_location=device))
            cls_model_init.load_state_dict(
                torch.load(
                    os.path.join('../init_cora_amazon/init_params', str(args.dataset) + '_' + str(args.model) + '_' + str(args.hid_dim) + '_' + str(args.encoder_dim),
                                 'init_cls_model.pt'),
                    map_location=device))
        else:
            logging.info('ERROR! Params initialization does not exist!')
            assert False
    if args.dataset=='cora':
        if os.path.exists(
                os.path.join('../init_cora_amazon/init_params', str(args.model) + '_' + str(args.hid_dim) + '_' + str(args.encoder_dim),
                             'init_encoder.pt')):
            logging.info('Loading existing initial params...')
            encoder_init.load_state_dict(torch.load(
                os.path.join('../init_cora_amazon/init_params', str(args.model) + '_' + str(args.hid_dim) + '_' + str(args.encoder_dim),
                             'init_encoder.pt'),
                map_location=device))
            cls_model_init.load_state_dict(
                torch.load(
                    os.path.join('../init_cora_amazon/init_params', str(args.model) + '_' + str(args.hid_dim) + '_' + str(args.encoder_dim),
                                 'init_cls_model.pt'),
                    map_location=device))
        else:
            logging.info('ERROR! Params initialization does not exist!')
            assert False

    models_init = [encoder_init, cls_model_init]
    params = itertools.chain(*[model.parameters() for model in models_init])
    optimizer_init = torch.optim.Adam(params, lr=args.lr_frominit, weight_decay=args.wd_frominit)
    loss_func = nn.NLLLoss().to(device)

    best_input_acc = 0.0
    best_epoch = 0.0
    best_diff_norm = 0.0
    # patience = 0
    # best_loss = 1e10
    best_margin = args.lp_margin
    best_metric_lp = 1e10
    edge_label_index = input_data.graph['edge_index'].to(device)
    input_edge_label = torch.ones(edge_label_index.shape[1]).to(device)
    neg_edge_index = input_data.graph['neg_edge_index'].to(device)
    new_edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=-1).to(device)
    new_edge_label = torch.cat([input_edge_label, input_edge_label.new_zeros(neg_edge_index.size(1))], dim=0)
    for epoch in range(0, args.epochs_frominit):

        embedding, input_acc, input_loss = train_frominit(models_init, encoder_init, cls_model_init, optimizer_init,
                                                          loss_func,
                                                          input_data, input_pseudo,aug=aug, type=type)

        params_list = [model.parameters() for model in models_init]
        flattened_params_opt = torch.cat([param.view(-1) for param in itertools.chain(*params_list)])
        diff_norm = param_diff(train_flattened_params_opt, flattened_params_opt)

        # input = torch.cat([embedding[new_edge_label_index[0]], embedding[new_edge_label_index[1]]], axis=1)
        # logits_lp = link_evaltor_load(input).view(-1)
        logits_lp = F.cosine_similarity(embedding[new_edge_label_index[0]], embedding[new_edge_label_index[1]], dim=-1)
        metric_lp = F.binary_cross_entropy_with_logits(logits_lp, new_edge_label)
        # logging.info('Epoch = {}, Metric_lp = {},'.format(epoch, metric_lp.item()))
        margin = torch.abs(metric_lp - metric_lp_ref).item()
        writer.add_scalar('curve/acc_input_seed_' + str(args.seed), input_acc, epoch)
        writer.add_scalar('curve/loss_input_seed_' + str(args.seed), input_loss, epoch)
        writer.add_scalar('curve/pnorm_input_seed_' + str(args.seed), diff_norm, epoch)
        if epoch >= args.atleastepoch:
            if margin < best_margin:
                best_epoch = epoch
                best_input_acc = input_acc
                best_diff_norm = diff_norm
                best_metric_lp = metric_lp.item()
                break
            else:
                # using the last epoch results
                best_epoch = epoch
                best_input_acc = input_acc
                best_diff_norm = diff_norm
                best_metric_lp = metric_lp.item()


        # if input_loss < best_loss:
        #    best_input_acc = input_acc
        #    best_epoch = epoch
        #    best_loss = input_loss
        #    best_diff_norm = diff_norm
        # if metric_lp < best_metric_lp:
        #    best_metric_lp = metric_lp
        #   best_input_acc = input_acc
        #    best_epoch = epoch
        #    best_loss = input_loss
        #    best_diff_norm = diff_norm
        # else:
        #    patience += 1
        #    if patience == 20:
        #        logging.info('INFER: breaking at {}-th epoch for training overfitting...'.format(epoch))
        #        break

        if epoch % 10 == 0:
            logging.info(
                'TRAIN-FROM-INIT: Epoch: {}, input_data_loss = {:.4f}, input_data_acc = {:.2f}, p_norm = {:.4f}, metric = {:.4f}, metric_ref = {:.4f}'.format(
                    epoch,
                    input_loss,
                    input_acc * 100.,
                    diff_norm,
                    metric_lp.item(),
                    metric_lp_ref.item()))

    logging.info(
        'TRAIN-FROM-INIT: Test_graph_id = {}, Best Epoch: {}, best_train_acc = {:.2f}, BEST_best_p_norm = {:.4f}, BEST_metric_lp = {:.4f} with metric_ref = {:.4f}'.format(
            test_g_idx,
            best_epoch,
            best_input_acc * 100.,
            best_diff_norm,
            best_metric_lp,
            metric_lp_ref.item()))

    return best_diff_norm, best_input_acc


def test(data, models, encoder, cls_model, aug, type=None, mask=None):
    for model in models:
        model.eval()

    # edge_index_with_self_loop = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    # edge_index_in = torch.unique(edge_index_with_self_loop.T, dim=0, return_inverse=False).T
    if type == 'noise':
        input_feat = data.graph['node_feat'].to(device) + aug.to(device)
        logging.info('Generating pseudo with noise...')
    elif type == 'mkidx':
        input_feat = data.graph['node_feat'].clone().to(device)
        input_feat[:, aug] = 0
        logging.info('Generating pseudo with mask feat...')
    else:
        input_feat = data.graph['node_feat'].to(device)
        logging.info('Generating pseudo with raw feat...')

    # feat = feat + noise.to(feat.device)
    # x = x.clone()
    # x[:, idx] = 0

    if isinstance(encoder, MLP):
        emb_out = encoder(input_feat)
    else:
        emb_out = encoder(input_feat, data.graph['edge_index'].to(device))

    logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
    probs = F.softmax(logits, dim=1)
    #log_probs = torch.log(probs)
    preds = probs.argmax(dim=1)
    labels = data.label.squeeze().to(device) if mask is None else data.label.squeeze()[mask].to(device)
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return emb_out, probs, preds, accuracy


def param_diff(param1, param2):
    """
    Returns:
        the l2 norm difference the two networks
    """
    diff = (torch.norm(param1 - param2) ** 2).cpu().detach().numpy()
    return np.sqrt(diff)


def train_frominit(models, encoder, cls_model, optimizer, loss_func, input_data, input_pseudo, aug=None, type=None):
    for model in models:
        model.train()

    if type == 'noise':
        input_feat = input_data.graph['node_feat'].to(device) + aug.to(device)
        #logging.info('Training pseudo with noise...')
    elif type == 'mkidx':
        input_feat = input_data.graph['node_feat'].clone().to(device)
        input_feat[:, aug] = 0
        #logging.info('Training pseudo with mask feat...')
    else:
        input_feat = input_data.graph['node_feat'].to(device)
        #logging.info('Training pseudo with raw feat...')

    if isinstance(encoder, MLP):
        emb_source = encoder(input_feat)
    else:
        emb_source = encoder(input_feat, input_data.graph['edge_index'].to(device))

    source_logits = cls_model(emb_source)
    source_probs = F.softmax(source_logits, dim=1)
    #log_source_probs = torch.log(source_probs)
    max_logits, _ = source_logits.max(dim=1, keepdim=True)  # Find the maximum logits along each row
    log_source_probs = source_logits - max_logits  # Subtract the maximum logits to avoid overflow
    log_source_probs = log_source_probs - torch.logsumexp(log_source_probs, dim=1, keepdim=True)
    cls_loss = loss_func(log_source_probs, input_pseudo)
    labels = input_pseudo
    preds = source_probs.argmax(dim=1)
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()

    loss = cls_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return emb_source, accuracy, loss.item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--gnn_gen', type=str, default='sgc', choices=['gcn', 'gat', 'sgc'],
                        help='random initialized gnn for data generation')
    parser.add_argument("--model_path", type=str, default='logs/Model_pretrain/cora-GCN-0-20230822-115909-918747')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--encoder_dim", type=int, default=32)
    parser.add_argument("--lr_frominit", type=float, default=1e-3)
    parser.add_argument("--wd_frominit", type=float, default=5e-4)
    parser.add_argument("--epochs_frominit", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lp_margin", type=float, default=1e-2)
    parser.add_argument("--atleastepoch", type=int, default=5)

    args = parser.parse_args()
    device = torch.device(args.device)
    log_dir = 'logs/infer/PNorm-{}-{}-{}-{}'.format(args.dataset, args.model, str(args.seed),
                                                    datetime.datetime.now().strftime(
                                                        "%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'infer.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    writer = SummaryWriter(log_dir + '/tbx_log')
    logging.info(args)
    main(args, device)

    logging.info(args)
    logging.info('Finish!, this is the log dir = {}'.format(log_dir))

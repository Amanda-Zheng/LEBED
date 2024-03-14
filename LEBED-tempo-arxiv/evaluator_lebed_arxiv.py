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
import numpy as np
import datetime
import sys
import logging
import copy
import torch.nn as nn
import torch.sparse
import torch.nn.init as init
from tensorboardX import SummaryWriter
from dataset import load_nc_dataset


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
    feat_dim = dataset_tr.d
    num_class = dataset_tr.c
    
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
    logging.info('Start with {}# num of graphs...'.format(len(te_filtered_file_list)))
    for i in range(len(te_filtered_file_list)):
        dataset_te = torch.load(os.path.join(log_dir_in,te_filtered_file_list[i]))
        test_acc, test_pseudo, metric_lp_ref = learner_agent(dataset_te, models, encoder, cls_model)
        diff_norm_g, test_learned_acc = train_norm(feat_dim, num_class, dataset_te, test_pseudo,
                                                   train_flattened_params_opt, metric_lp_ref, te_filtered_file_list[i])
        # AE = torch.abs(test_acc - test_learned_acc).item()
        torch.cuda.empty_cache()
        test_error_ls.append(1 - test_acc.item())
        diff_norm_g_ls.append(diff_norm_g)
        logging.info(
            'GT-Test-ERR = {:.2f}, Diff-norm = {:.4f} for TEST_G_path = {}'.format(
                test_error_ls[
                    -1] * 100.,
                diff_norm_g_ls[-1],
            te_filtered_file_list[i]))


    from scipy.stats import spearmanr

    all_err_np = np.array(test_error_ls)
    all_norm_np = np.array(diff_norm_g_ls)

    np.savez(os.path.join(log_dir, "GT-err_arrays.npz"), all_err=all_err_np)
    np.savez(os.path.join(log_dir, "diffnorm_arrays.npz"), all_norm=all_norm_np)

    # Calculate the Spearman correlation coefficient and p-value
    corcoef, p = spearmanr(all_err_np, all_norm_np)
    logging.info('Pnorm-Spearman correlation coefficient = {:.2f} under P-value = {}'.format(corcoef, p))
    # Create a linear regression model
    np_ls_err = [all_err_np]
    np_ls_norm = [all_norm_np]

    linearR(np_ls_err[0], np_ls_norm[0], 'Pnorm')


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


def train_norm(feat_dim, num_class, input_data, input_pseudo, train_flattened_params_opt, metric_lp_ref, test_g_path):
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

    if os.path.exists(os.path.join('../init_arxiv/init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim),'init_encoder.pt')):
        logging.info('Loading existing initial params...')
        encoder_init.load_state_dict(torch.load(os.path.join('../init_arxiv/init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_encoder.pt'), map_location=device))
        cls_model_init.load_state_dict(torch.load(os.path.join('../init_arxiv/init_params', str(args.dataset)+ '_'+ str(args.model)+ '_'+ str(args.hid_dim)+'_'+str(args.encoder_dim), 'init_cls_model.pt'), map_location=device))
    else:
        logging.info('Error! Should have initialization models!')
        assert False

    models_init = [encoder_init, cls_model_init]
    params_list_init = [model.parameters() for model in models_init]
    flattened_params_init = torch.cat([param.view(-1) for param in itertools.chain(*params_list_init)])
    params = itertools.chain(*[model.parameters() for model in models_init])
    #diff_norm_0 = param_diff(flattened_params_init, 0)
    #logging.info('diff_norm_0 = {}'.format(diff_norm_0))
    #assert False
    optimizer_init = torch.optim.Adam(params, lr=args.lr_frominit, weight_decay=args.wd_frominit)
    loss_func = nn.NLLLoss().to(device)
    best_epoch = 0.0
    best_diff_norm = best_ref_norm = 0.0
    best_input_acc = 0.0
    best_margin = args.lp_margin_rate*metric_lp_ref
    best_metric_lp = 1e10
    edge_label_index = input_data.graph['edge_index'].to(device)
    input_edge_label = torch.ones(edge_label_index.shape[1]).to(device)
    neg_edge_index = input_data.graph['neg_edge_index'].to(device)
    new_edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=-1).to(device)
    new_edge_label = torch.cat([input_edge_label, input_edge_label.new_zeros(neg_edge_index.size(1))], dim=0)
    for epoch in range(0, args.epochs_frominit):

        embedding, input_acc, input_loss = train_frominit(models_init, encoder_init, cls_model_init, optimizer_init,
                                                          loss_func,
                                                          input_data, input_pseudo)
        torch.cuda.empty_cache()
        params_list = [model.parameters() for model in models_init]
        flattened_params_opt = torch.cat([param.view(-1) for param in itertools.chain(*params_list)])
        diff_norm = param_diff(train_flattened_params_opt, flattened_params_opt)
        ref_norm= param_diff(flattened_params_init,flattened_params_opt)

        logits_lp = F.cosine_similarity(embedding[new_edge_label_index[0]], embedding[new_edge_label_index[1]], dim=-1)
        metric_lp = F.binary_cross_entropy_with_logits(logits_lp, new_edge_label)
        margin = torch.abs(metric_lp - metric_lp_ref).item()
        writer.add_scalar('curve/acc_input_seed_' + str(args.seed), input_acc, epoch)
        writer.add_scalar('curve/loss_input_seed_' + str(args.seed), input_loss, epoch)
        writer.add_scalar('curve/pnorm_input_seed_' + str(args.seed), diff_norm, epoch)
        if epoch >= args.atleastepoch:
            if margin < best_margin:
                best_epoch = epoch
                best_diff_norm = diff_norm
                best_metric_lp = metric_lp.item()
                best_input_acc=input_acc.item()
                best_ref_norm = ref_norm
                break
            else:
                # using the last epoch results
                best_epoch = epoch
                best_diff_norm = diff_norm
                best_metric_lp = metric_lp.item()
                best_input_acc = input_acc.item()
                best_ref_norm = ref_norm

        if epoch % 10 == 0:
            logging.info(
                'TRAIN-FROM-INIT: Epoch: {}, input_data_loss = {:.4f}, input_data_acc = {:.2f}, ref_norm = {:.4f}, p_norm = {:.4f}, metric = {:.4f}, metric_ref = {:.4f}'.format(
                    epoch,
                    input_loss,
                    input_acc.item() * 100.,
                    ref_norm,
                    diff_norm,
                    metric_lp.item(),
                    metric_lp_ref.item()))

    logging.info(
        'TRAIN-FROM-INIT: Best Epoch: {}, BEST_input_acc = {:.2f}, BEST_ref_norm = {:.4f}, BEST_best_p_norm = {:.4f}, BEST_metric_lp = {:.4f} with metric_ref = {:.4f} in Test_graph_path = {}'.format(
            best_epoch,
            best_input_acc*100.,
            best_ref_norm,
            best_diff_norm,
            best_metric_lp,
            metric_lp_ref.item(),
            test_g_path)
    )

    return best_diff_norm,best_input_acc


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()

    with torch.no_grad():
        if isinstance(encoder, MLP):
            emb_out = encoder(data.graph['node_feat'].to(device))
        else:
            emb_out = encoder(data.graph['node_feat'].to(device), data.graph['edge_index'].to(device))

        logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
        preds = F.log_softmax(logits, dim=1)
        pseudo = preds.argmax(dim=1)
        y = data.label.squeeze().to(device)
        GT = y if mask is None else y[mask]
        corrects = pseudo.eq(GT)
        accuracy = corrects.float().mean()

    return emb_out, pseudo, accuracy


def param_diff(param1, param2):
    """
    Returns:
        the l2 norm difference the two networks
    """
    diff = (torch.norm(param1 - param2) ** 2).cpu().detach().numpy()
    return np.sqrt(diff)


def learner_agent(test_data, models, encoder, cls_model):
    test_emb_out, test_pseudo, test_acc = test(test_data, models, encoder, cls_model, None)
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
    return test_acc, test_pseudo, metric_lp_ref


def train_frominit(models, encoder, cls_model, optimizer, loss_func, input_data, input_pseudo):
    for model in models:
        model.train()

    if isinstance(encoder, MLP):
        emb_source = encoder(input_data.graph['node_feat'].to(device))
    else:
        emb_source = encoder(input_data.graph['node_feat'].to(device), input_data.graph['edge_index'].to(device))

    source_logits = cls_model(emb_source)
    preds = F.log_softmax(source_logits, dim=1)
    loss = loss_func(preds, input_pseudo)
    preds_acc = preds.argmax(dim=1)

    corrects = preds_acc.eq(input_pseudo)
    accuracy = corrects.float().mean()

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    return emb_source, accuracy, loss.item()


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
    parser.add_argument("--lr_frominit", type=float, default=1e-3)
    parser.add_argument("--wd_frominit", type=float, default=5e-4)
    parser.add_argument("--epochs_frominit", type=int, default=200)
    parser.add_argument("--lp_margin_rate", type=float, default=0.05)
    parser.add_argument("--atleastepoch", type=int, default=10)

    args = parser.parse_args()
    device = torch.device(args.device)
    log_dir = 'logs/infer/PNorm-{}-{}-{}-{}'.format(args.dataset, args.model,
                                                          str(args.seed),
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

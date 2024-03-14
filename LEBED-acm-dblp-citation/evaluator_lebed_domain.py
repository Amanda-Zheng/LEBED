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
import pickle


def load_data_path(mode,domain_lists):
    te_data_ls = []
    if mode=='0':
        # ste_data = torch.load(os.path.join('../data', args.source, 'induced', 'te_sub.pt')).to(device)
        te_data_ls.append(os.path.join('../data', args.source, 'induced', 'train_sub.pt'))
        logging.info(
            '#NUM = {}, 0-Source Train DA Test graphs in PATH = {}'.format(len(te_data_ls), te_data_ls[-1]))
    elif mode=='1':
        # ste_data = torch.load(os.path.join('../data', args.source, 'induced', 'te_sub.pt')).to(device)
        te_data_ls.append(os.path.join('../data', args.source, 'induced', 'te_sub.pt'))
        logging.info(
            '#NUM = {}, 1-Source RAW DA Test graphs in PATH = {}'.format(len(te_data_ls), te_data_ls[-1]))
    elif mode == '2':
        s_file_list = os.listdir(os.path.join('../data', args.source, 'gen'))
        s_filtered_file_list = [filename for filename in s_file_list if filename.endswith('.pth') and "te_sub" in filename]

        for si in range(len(s_filtered_file_list)):
            # s_test_data = torch.load(os.path.join('../data', args.source, 'gen', s_filtered_file_list[si])).to(device)
            te_data_ls.append(os.path.join('../data', args.source, 'gen', s_filtered_file_list[si]))
            logging.info(
                '#NUM = {},2-Source GEN DA Test graphs in PATH = {}'.format(len(te_data_ls), te_data_ls[-1]))
    elif mode == '3' or mode == '4':
        for item in domain_lists:
            if mode == '3':
                for raw_name in ['train_sub.pt', 'val_sub.pt', 'te_sub.pt', 'full_g.pt']:
                    # t_raw_data = torch.load(os.path.join('../data', item, 'induced', raw_name)).to(device)
                    te_data_ls.append(os.path.join('../data', item, 'induced', raw_name))
                    logging.info(
                        '#NUM = {}, 3-Target RAW DA Test graphs in PATH = {}'.format(len(te_data_ls), te_data_ls[-1]))
            elif mode == '4':
                file_list = os.listdir(os.path.join('../data', item, 'gen'))
                filtered_file_list = [filename for filename in file_list if filename.endswith('.pth')]
                te_subs = [filename for filename in filtered_file_list]
                for i in range(len(te_subs)):
                    # da_test_data = torch.load(os.path.join('../data', item, 'gen', te_subs[i])).to(device)
                    te_data_ls.append(os.path.join('../data', item, 'gen', te_subs[i]))
                    logging.info(
                        '#NUM = {}, 4-Target GEN DA Test graphs in PATH = {}'.format(len(te_data_ls), te_data_ls[-1]))

    return te_data_ls


def main(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    str_data = torch.load(os.path.join('../data', args.source, 'induced', 'train_sub.pt')).to(device)
    sval_data = torch.load(os.path.join('../data', args.source, 'induced', 'val_sub.pt')).to(device)
    feat_dim = str_data.x.shape[1]
    num_class = max(str_data.y) + 1
    te_file_path = []
    if args.domain_ind=='1':
        domain_lists = [args.target1]
    elif args.domain_ind=='2':
        domain_lists = [args.target2]
    elif args.domain_ind=='both':
        domain_lists = [args.target1, args.target2]
    for modek in args.mode:
        te_file_path_mid = load_data_path(modek,domain_lists)
        te_file_path+=te_file_path_mid

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
    logging.info('Start with {}# num of graphs...'.format(len(te_file_path)))
    for i in range(len(te_file_path)):
        dataset_te = torch.load(te_file_path[i]).to(device)
        test_acc, test_pseudo, metric_lp_ref = learner_agent(dataset_te, models, encoder, cls_model)
        diff_norm_g, test_learned_acc = train_norm(feat_dim, num_class, dataset_te, test_pseudo,
                                                   train_flattened_params_opt, metric_lp_ref, te_file_path[i])
        # AE = torch.abs(test_acc - test_learned_acc).item()
        torch.cuda.empty_cache()
        test_error_ls.append(1 - test_acc.item())
        diff_norm_g_ls.append(diff_norm_g)
        logging.info(
            'GT-Test-ERR = {:.2f}, Diff-norm = {:.4f} for TEST_G_path = {}'.format(
                test_error_ls[
                    -1] * 100.,
                diff_norm_g_ls[-1],
            te_file_path[i]))


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

    if os.path.exists(os.path.join('../init_domain_ACD/init_params',
                                   str(args.source) + '_' + str(args.model) + '_' + str(args.num_layers) + '_' + str(
                                       args.hid_dim) + '_' + str(
                                       args.encoder_dim), 'init_encoder.pt')):
        logging.info('Loading existing initial params...')
        encoder_init.load_state_dict(torch.load(os.path.join('../init_domain_ACD/init_params',
                                                             str(args.source) + '_' + str(args.model) + '_' + str(
                                                                 args.num_layers) + '_' + str(
                                                                 args.hid_dim) + '_' + str(args.encoder_dim),
                                                             'init_encoder.pt'), map_location=device))
        cls_model_init.load_state_dict(torch.load(os.path.join('../init_domain_ACD/init_params',
                                                               str(args.source) + '_' + str(args.model) + '_' + str(
                                                                   args.num_layers) + '_' + str(
                                                                   args.hid_dim) + '_' + str(args.encoder_dim),
                                                               'init_cls_model.pt'), map_location=device))
    else:
        logging.info('Error! Should have initialization models!')

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
    edge_label_index = input_data.edge_index.to(device)
    input_edge_label = torch.ones(edge_label_index.shape[1]).to(device)
    neg_edge_index = input_data.neg_edge_index.to(device)
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
            emb_out = encoder(data.x.to(device))
        else:
            emb_out = encoder(data.x.to(device), data.edge_index.to(device))

        logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
        preds = F.log_softmax(logits, dim=1)
        pseudo = preds.argmax(dim=1)
        y = data.y.to(device)
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
    test_edge_label_index = test_data.edge_index.to(device)
    test_edge_label = torch.ones(test_edge_label_index.shape[1]).to(device)
    neg_edge_index = test_data.neg_edge_index.to(device)

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
        emb_source = encoder(input_data.x.to(device))
    else:
        emb_source = encoder(input_data.x.to(device), input_data.edge_index.to(device))

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
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--target1", type=str, default='dblp')
    parser.add_argument("--target2", type=str, default='network')
    parser.add_argument("--model_path", type=str, default='logs/Models_tra/acm-to-dblp-network-GCN-0-20230904-220846-668903')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--encoder_dim", type=int, default=32)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr_frominit", type=float, default=1e-3)
    parser.add_argument("--wd_frominit", type=float, default=5e-4)
    parser.add_argument("--epochs_frominit", type=int, default=200)
    parser.add_argument("--lp_margin_rate", type=float, default=0.05)
    parser.add_argument("--atleastepoch", type=int, default=10)
    parser.add_argument('--mode', nargs='+', type=str, help='List of strings', default=['3', '4'])
    parser.add_argument('--domain_ind', type=str, help='evaluated domains', default='both', choices=['1', '2', 'both'])

    args = parser.parse_args()
    device = torch.device(args.device)
    modes_str = '_'.join(args.mode)
    log_dir = 'logs/infer/PNorm-{}-to-{}-{}-{}-{}-MODE-{}-DOMAIN-{}-{}'.format(args.source, args.target1, args.target2, args.model,
                                                          str(args.seed),modes_str,str(args.domain_ind),
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

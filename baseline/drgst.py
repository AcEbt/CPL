import argparse
import numpy as np
import torch
import torch.optim as optim
import random
import os
import sys

os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os_path)
from utils import *
from data import *
import torch.nn as nn
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Cora", help='dataset for training')
parser.add_argument('--hidden', type=int, default=128,help='Number of hidden units.')
parser.add_argument("--hid_dim_1", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--view", type=int, default=5, help="Number of extra view of augmentation")

parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--epochs_ft', type=int, default=2000, help='Number of epochs to finetuning.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers.')
parser.add_argument('--nb_heads', type=int, default=8)
parser.add_argument('--nb_out_heads', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in training")
parser.add_argument("--aug_drop", type=float, default=0.1, help="Attribute augmentation dropout rate")
parser.add_argument('--lr', type=float, default=0.001,help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--beta', type=float, default=1/3,help='coefficient for weighted CE loss')

parser.add_argument("--threshold", type=float, default=0.53, help="Threshold for pseudo labeling")
parser.add_argument("--iter", type=int, default=3, help="Number of pseudo labeling iteration")
parser.add_argument("--top", type=int, default=10, help="Number of pseudo label in each iteration")
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--multiview', action='store_true', default=False)
parser.add_argument("--random_pick", action="store_true", default=False, help="Indicator of random pseudo labeling")

parser.add_argument('--drop_method', type=str, default='dropout')

parser.add_argument("--seed", type=int, default=1024, help="Random seed")
parser.add_argument("--gpu", type=int, default=3, help="gpu id")
parser.add_argument("--device", type=str, default='cpu', help="device of the model")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

criterion = torch.nn.CrossEntropyLoss().cuda()
args.device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
device = args.device
print(device)

if args.dataset == 'Citeseer':
    args.threshold = 0.2
elif args.dataset in ['Cora', 'Pubmed']:
    args.threshold = 0.5
elif args.dataset in ['APh', 'CaCS']:
    args.threshold = 0.9


def train(args, model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g):
    sign = True
    nclass = labels.max().int().item() + 1
    # Model and optimizer
    model = get_models(args, features.shape[1], nclass, g=g)
    optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    best, bad_counter = 0, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        output = torch.softmax(output, dim=1)
        output = torch.mm(output, T)
        sign = False
        loss_train = weighted_cross_entropy(output[idx_train], pseudo_labels[idx_train], bald[idx_train], args.beta, nclass, sign)
        # loss_train = criterion(output[idx_train], pseudo_labels[idx_train])
        acc_train = accuracy(output[idx_train], pseudo_labels[idx_train])
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(features, adj)
            loss_val = criterion(output[idx_val], labels[idx_val])
            loss_test = criterion(output[idx_test], labels[idx_test])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])

        if acc_val > best:
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
            best = acc_val
            bad_counter = 0
            best_output = output
            best_print = [f'epoch: {epoch}',
                          f'loss_train: {loss_train.item():.4f}',
                          f'acc_train: {acc_train:.4f}',
                          f'loss_val: {loss_val.item():.4f}',
                          f'acc_val: {acc_val:.4f}',
                          f'loss_test: {loss_test.item():4f}',
                          f'acc_test: {acc_test:.4f}']
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            # print('early stop')
            break
        
        # if epoch % 100 == 0:
        # print(f'epoch: {epoch}',
        #     f'loss_train: {loss_train.item():.4f}',
        #     f'acc_train: {acc_train:.4f}',
        #     f'loss_val: {loss_val.item():.4f}',
        #     f'acc_val: {acc_val:.4f}',
        #     f'loss_test: {loss_test.item():4f}',
        #     f'acc_test: {acc_test:.4f}')
        
    # print("best result: ", best_print)
    return best_output


@torch.no_grad()
def test(adj, features, labels, idx_test, nclass, model_path, g, logger):
    nfeat = features.shape[1]
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    logger.info("Test set results: loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(),acc_test))

    return acc_test, loss_test


if __name__ == '__main__':
    model_path = './save_model/drgst-%s-%s-itr%s-seed%s-thr%.2f.pth' % (args.model, args.dataset, args.iter, args.seed, args.threshold)
    log_path = './log/drgst-%s-%s-itr%s-seed%s-thr%.2f.txt' % (args.model, args.dataset, args.iter, args.seed, args.threshold)
    log_time_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(levelname)s %(asctime)s - %(message)s'
    log_time_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        format=log_format,
        datefmt=log_time_format,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    g, adj, features, labels, idx_train, idx_val, idx_test, oadj = load_data(args.dataset)

    g = g.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    train_index = torch.where(idx_train)[0]
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    idx_pseudo = torch.zeros_like(idx_train)
    n_node = labels.size()[0]
    nclass = labels.max().int().item() + 1
    logger.info(f"DR-GST. model: {args.model}, dataset: {args.dataset}, N_node: {n_node}, N_class: {nclass}")

    if args.drop_method == 'dropedge':
        mc_adj = get_mc_adj(oadj, device, args.droprate)

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)
    bald = torch.ones(n_node).to(device)
    T = nn.Parameter(torch.eye(nclass, nclass).to(device)) # transition matrix
    T.requires_grad = False

    best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g)
    acc_test0, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)


    # generate pseudo labels
    # state_dict = torch.load(model_path)
    # model = get_models(args, features.shape[1], nclass, g=g)
    # model.load_state_dict(state_dict)
    # model.to(device)

    tests = []
    pl_acc = []
    N_pl_list = []
    tests.append(acc_test0)
    pl_acc.append(0)
    N_pl_list.append(-1)
    for itr in range(args.iter):
        T = update_T(best_output, idx_train, labels, T, device)
        idx_unlabeled = ~(idx_train_ag)
        if args.drop_method == 'dropout':
            bald = uncertainty_dropout(adj, features, g, nclass, model_path, args, device)
        elif args.drop_method == 'dropedge':
            bald = uncertainty_dropedge(mc_adj, adj, features, g, nclass, model_path, args, device)

        # model.eval()
        idx_train_ag, pseudo_labels, idx_pseudo = regenerate_pseudo_label(best_output, pseudo_labels, idx_train, idx_unlabeled,
                                                                        args.threshold, device)
        pred_diff =  pseudo_labels[idx_pseudo] == labels[idx_pseudo]
        n_pl = len(pred_diff)
        if n_pl > 0:
            best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g)
            pred_diff = len(pred_diff[pred_diff==True])/len(pred_diff)
            logger.info('itr {} summary: added {} pl labels , pl_acc: {}'.format(itr, n_pl,  pred_diff*100))

            acc_test, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)
            N_pl_list.append(n_pl)
            pl_acc.append(pred_diff)
            tests.append(acc_test)
    
    pl_acc = np.array(pl_acc)
    N_pl_list = np.array(N_pl_list)
    ave_pl_acc = np.sum(pl_acc*N_pl_list)/np.sum(N_pl_list)
    logger.info('original acc: {:.5f}, best test accuracy: {:.5f}, pl_acc: {:.5f}'.format(
        acc_test0, max(tests), pl_acc[np.argmax(tests)])) #ave_pl_acc
    
    # print("ENDS")
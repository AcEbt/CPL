import argparse
import numpy as np
import torch
import torch.optim as optim
import random
from utils import accuracy
from utils import *
from data import *
import torch.nn as nn
import sys
import os
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

parser.add_argument("--threshold", type=float, default=0.01, help="Threshold for pseudo labeling")
parser.add_argument("--iter", type=int, default=20, help="Number of pseudo labeling iteration")
parser.add_argument("--top", type=int, default=10, help="Number of pseudo label in each iteration")
parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--multiview', action='store_true', default=False)
parser.add_argument("--random_pick", action="store_true", default=False, help="Indicator of random pseudo labeling")


parser.add_argument("--seed", type=int, default=1024, help="Random seed")
parser.add_argument("--gpu", type=int, default=2, help="gpu id")
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


def train(args, model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, FT=False):

    nclass = labels.max().int().item() + 1
    # Model and optimizer
    model = get_models(args, features.shape[1], nclass, g=g)

    if not FT:
        epochs = args.epochs
    else:
        state_dict = torch.load('./save_model/tmp.pth')
        model.load_state_dict(state_dict)
        epochs = args.epochs_ft

    optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    best, bad_counter = 0, 0
    for epoch in range(epochs):
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

        if not FT and epoch == 100:
            torch.save(model.state_dict(), './save_model/tmp.pth', _use_new_zipfile_serialization=False)
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
        #     print(f'epoch: {epoch}',
        #         f'loss_train: {loss_train.item():.4f}',
        #         f'acc_train: {acc_train:.4f}',
        #         f'loss_val: {loss_val.item():.4f}',
        #         f'acc_val: {acc_val:.4f}',
        #         f'loss_test: {loss_test.item():4f}',
        #         f'acc_test: {acc_test:.4f}')
    
    print("best result: ", best_print)
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
    # logger.info("Test set results: loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(),acc_test))

    return acc_test, loss_test


if __name__ == '__main__':
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

    # args.top = int(n_node * 0.8 // 2000 * 100)

    model_path = './save_model/%s-%s-itr%s-top%s-seed%s-m%.0f.pth' % (
        args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
    log_path = './log/cautious-%s-%s-itr%s-top%s-seed%s-m%.0f.txt' % (
        args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
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

    logger.info(f"Cautious. model: {args.model}, dataset: {args.dataset}, N_node: {n_node}, N_class: {nclass}")

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)
    bald = torch.ones(n_node).to(device)
    T = nn.Parameter(torch.eye(nclass, nclass).to(device)) # transition matrix
    T.requires_grad = False
    best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g)
    acc_test0, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)


    # generate pseudo labels
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
    model.load_state_dict(state_dict)
    model.to(device)
    consistency = []
    PL_node = []
    tests = []
    pl_acc = []
    for itr in range(args.iter):
        model.eval()
        output_ave, confidence, predict_labels, consist = multiview_pred(model, features, adj, g, args)
        consistency.append(round(consist,5))
        confidence[idx_train_ag] = 0
        if args.random_pick:
            idxes = torch.where(confidence > 0.)[0]
            pl_idx = random.sample(idxes.tolist(), args.top)
            pl_idx = torch.Tensor(pl_idx).long().to(device)
            conf_score = confidence[pl_idx]
        else:
            pl_idx = torch.where(confidence > args.threshold)[0]
            conf_score = confidence[pl_idx]
            if len(conf_score) >= args.top:
                conf_score, pl_idx = torch.topk(confidence, args.top)
        if len(conf_score) > 0:
            pl_labels = predict_labels[pl_idx]
            idx_train_ag[pl_idx] = True
            pred_diff =  pseudo_labels[pl_idx] == pl_labels
            pred_diff = len(pred_diff[pred_diff==True])/len(pred_diff)

            pseudo_labels[pl_idx] = pl_labels
            PL_node += list(pl_idx.cpu().numpy())         
            pred_diff_sum =  pseudo_labels[idx_train_ag^idx_train] == labels[idx_train_ag^idx_train]
            pred_diff_sum = len(pred_diff_sum[pred_diff_sum==True])/len(pred_diff_sum)
            pl_acc.append(pred_diff_sum)

            best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, FT=True)
            # Testing
            acc_test, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)
            tests.append(acc_test)
            logger.info('itr {} summary: added {} pl labels with condifence {:.5f}, pl_acc: {}, consistency {:.5f}, {} pl labels in total, test_acc: {:.4f}'.format(
                itr, len(pl_idx), conf_score.min().item(), pred_diff*100, consist, len(PL_node), acc_test))
    
    logger.info('original acc: {:.5f}, best test accuracy: {:.5f}, consistency: {}, pl_acc: {}'.format(
        acc_test0, max(tests), consistency[np.argmax(tests)], pl_acc[np.argmax(tests)]))
    
    # print("ENDS")
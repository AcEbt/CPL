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
parser.add_argument('--patience', type=int, default=200)
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
    Acc_record = []
    for rate in [0.4, 0.3, 0.2, 0.1 , 0]:
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
        tests = []
        tests.append(acc_test0)
        for itr in range(args.iter):
            model.eval()
            idxes = ~(idx_train_ag)
            idxes = torch.where(idxes > 0)[0]
            pl_idx = random.sample(idxes.cpu().numpy().tolist(), args.top)
            false_pl_idx = random.sample(pl_idx, int(args.top*rate))
            false_labels = torch.argmin(best_output, axis=1)
            pseudo_labels[pl_idx] = labels[pl_idx]
            pseudo_labels[false_pl_idx] = false_labels[false_pl_idx]

            pl_idx = torch.Tensor(pl_idx).long().to(device)
            idx_train_ag[pl_idx] = True
            idx_test[pl_idx] = False
            best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, FT=True)
            # Testing
            acc_test, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)
            tests.append(acc_test)
            logger.info('itr {} summary:  test_acc: {:.4f}'.format(itr, acc_test))
        Acc_record.append(np.array(tests))

    

    from matplotlib import pyplot as plt
    n_pl = np.array(range(0, args.top*args.iter+1,args.top))
    plt.figure()
    plt.plot(n_pl, Acc_record[4],'r-',label='1.0')
    plt.plot(n_pl, Acc_record[3],'b-',label='0.9')
    plt.plot(n_pl, Acc_record[2],'k-',label='0.8')
    plt.plot(n_pl, Acc_record[1],'g-',label='0.7')
    plt.plot(n_pl, Acc_record[0],'m-',label='0.6')
    plt.grid()
    plt.legend()
    plt.xlabel('#PL label')
    plt.ylabel('Acc test')
    plt.savefig("tmp.png")

    print("ENDS")

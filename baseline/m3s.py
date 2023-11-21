'''
Once remarked, the unlabeled data will not be remarked in next stages
'''
import torch
import os
import sys
import argparse
import numpy as np
import torch.optim as optim
from utils_baseline import *

os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os_path)

from utils import accuracy
from utils import *
from data import *
import logging


# Training settings
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
parser.add_argument("--iter", type=int, default=20, help="Number of pseudo labeling iteration")
parser.add_argument("--top", type=int, default=10, help="Number of pseudo label in each iteration")
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--multiview', action='store_true')

parser.add_argument("--seed", type=int, default=1024, help="Random seed")
parser.add_argument("--gpu", type=int, default=3, help="gpu id")
parser.add_argument("--device", type=str, default='cpu', help="device of the model")

parser.add_argument('--n_cluster', type=int, default=200, help='Number of clusters')
parser.add_argument('--max_iter', type=int, default=20, help='Number of epochs for kmeans')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(args, model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, g, FT=False):

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
        
        loss_train = criterion(output[idx_train], pseudo_labels[idx_train])
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
        # 
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
    model_path = './save_model/m3s-%s-%s-itr%s-seed%s-thr%.2f.pth' % (args.model, args.dataset, args.iter, args.seed, args.threshold)
    log_path = './log/m3s-%s-%s-itr%s-seed%s-thr%.2f.txt' % (args.model, args.dataset, args.iter, args.seed, args.threshold)
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
    logger.info(f"M3S. model: {args.model}, dataset: {args.dataset}, N_node: {n_node}, N_class: {nclass}")

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)
    best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, g)
    acc_test0, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)
    PL_node = []
    pl_acc = []
    tests = []
    pl_acc.append(0)
    PL_node.append(0)
    tests.append(acc_test0)
    for itr in range(args.iter):
        idx_unlabeled = ~(idx_train_ag|idx_val)
        state_dict = torch.load(model_path)
        model = get_models(args, features.shape[1], nclass, g=g)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        # Centroids of class in labeled data
        centroid_train = Centroid_train(best_output[idx_train_ag], pseudo_labels[idx_train_ag], nclass, device)
        # Kmeans
        kmeans = Kmeans(args.n_cluster, args.max_iter, False, device)
        kmeans.fit(best_output[idx_unlabeled])
        # Generating the pseudo labels for kmeans
        kmeans_pseudo_labels = kmeans.aligning_labels(centroid_train)
        # Generating the pseudo labels
        idx_train_ag, pseudo_labels = generate_pseudo_label_after_aligning(best_output, pseudo_labels, idx_train_ag, idx_unlabeled,
                                                                          kmeans_pseudo_labels, args.threshold, device)
        pred_diff = pseudo_labels[ idx_train_ag^idx_train] == labels[ idx_train_ag^idx_train]
        if len(pred_diff) > 0:
            pred_diff = len(pred_diff[pred_diff==True])/len(pred_diff)
            logger.info('total number of pl sample: {}, pl_acc: {:.5f}'.format(
                len(idx_train_ag[idx_train_ag==True]) - len(idx_train[idx_train==True]), pred_diff))
        
            best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, g, FT=True)
            # Testing
            acc_test, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)
            tests.append(acc_test)
            pl_acc.append(pred_diff)


    logger.info('original acc: {:.5f}, best test accuracy: {:.5f}, pl_acc{:.5f}'.format(
        acc_test0, max(tests), pl_acc[np.argmax(tests)]))

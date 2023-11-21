import argparse
import os
from tqdm import tqdm
import copy
import logging
import sys
import warnings

import numpy as np
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score, average_precision_score

import torch_geometric.transforms as T
from torch_geometric.datasets import WikiCS, Actor, Twitch, Amazon, LastFMAsia, CitationFull
from torch_geometric.loader import DataLoader

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from utils import *
from model import *
from dataset import *


def train(model, train_loader, optimizer, device):
    model.train()

    # total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
    #     total_loss += loss.item() * data.num_graphs

    # return total_loss / len(train_dataset)


@torch.no_grad()
def tst(model, val_loader, test_loader, device):
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader, ncols=70):  # val_loader: #
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    # pos_val_pred = val_pred[val_true==1]
    # neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader, ncols=70):  # test_loader: #
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)

    # AUC & AP
    valid_auc = roc_auc_score(val_true, val_pred)
    valid_ap = average_precision_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    test_ap = average_precision_score(test_true, test_pred)

    return valid_auc, valid_ap, test_auc, test_ap


@torch.no_grad()
def pseudo_label(model, neg_loader, mask, split_edge, views, aug_drop, conf_threshold, top, pl_link, device):
    model.eval()

    y_pred, y_link = [], []
    for data in tqdm(neg_loader, ncols=70):  # neg_loader: #
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        probs = torch.sigmoid(logits)
        # Augmentation
        for _ in range(views):
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, aug_drop)
            probs += torch.sigmoid(logits)
        probs /= (views + 1)

        y_pred.append(probs.view(-1).cpu())
        link = torch.stack((data.node_id[data.ptr[:-1]], data.node_id[data.ptr[:-1] + 1])).t()
        y_link.append(link.cpu())
    y_pred, y_link = torch.cat(y_pred), torch.cat(y_link)

    # Add pseudo label
    row = torch.where(y_pred > conf_threshold)[0]
    while row.shape[0] < top:
        conf_threshold -= 0.5
        row = torch.where(y_pred > conf_threshold)[0]

    values = y_pred[row]
    _, idx = torch.topk(values, top)
    row = row[idx]
    pseudo_ones = y_link[row, :].t()

    # Update mask matrix and train edge index
    mask[pseudo_ones[0], pseudo_ones[1]] = False
    split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], pseudo_ones.t()), 0)

    print("{} pairs have been labeled.".format(len(row)))

    return mask, split_edge, pseudo_ones


warnings.filterwarnings("ignore")
for seeds in [4399, 1024, 4321]:
    print("current seed is: {}".format(seeds))
    if __name__ == '__main__':

        # Data settings
        parser = argparse.ArgumentParser(description='SEAL')
        parser.add_argument("--dataset", type=str, default="PT",
                            help="Actor/WikiCS/PT/Amazon_ph/CiteSeer")
        parser.add_argument('--fast_split', action='store_true',
                            help="for large custom datasets (not OGB), do a fast data split")
        parser.add_argument('--data_dir', type=str, default='datasets', help='Data directory')
        parser.add_argument('--out_dir', type=str, default='results', help='Results directory')
        parser.add_argument('--seed', type=int, default=1024, help='Random seed')
        parser.add_argument("--record", action="store_true", default=False, help="Indicator of recording results")

        parser.add_argument("--p_val", type=float, default=0.4, help="percentage of validation set")
        parser.add_argument("--p_test", type=float, default=0.5, help="percentage of test set")
        # GNN settings
        parser.add_argument('--model', type=str, default='DGCNN')
        parser.add_argument('--sortpool_k', type=float, default=0.6)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--hidden_channels', type=int, default=32)
        parser.add_argument('--batch_size', type=int, default=32)
        # Subgraph extraction settings
        parser.add_argument('--num_hops', type=int, default=1)
        parser.add_argument('--ratio_per_hop', type=float, default=1.0)
        parser.add_argument('--max_nodes_per_hop', type=int, default=None)
        parser.add_argument('--node_label', type=str, default='drnl',
                            help="which specific labeling trick to use")
        parser.add_argument('--use_feature', action='store_true',
                            help="whether to use raw node features as GNN input")
        parser.add_argument('--use_edge_weight', action='store_true',
                            help="whether to consider edge weight in GNN"
                                 "")
        # Training settings
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--epochs', type=int, default=2)
        parser.add_argument('--train_percent', type=float, default=100)
        parser.add_argument('--val_percent', type=float, default=20)
        parser.add_argument('--test_percent', type=float, default=100)
        parser.add_argument('--dynamic_train', action='store_true', default=True,
                            help="dynamically extract enclosing subgraphs on the fly")
        parser.add_argument('--dynamic_val', action='store_true', default=True)
        parser.add_argument('--dynamic_test', action='store_true', default=True)
        parser.add_argument('--num_workers', type=int, default=0,
                            help="number of workers for dynamic mode; 0 if not dynamic")
        parser.add_argument('--train_node_embedding', action='store_true',
                            help="also train free-parameter node embeddings together with GNN")
        parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                            help="load pretrained node embeddings as additional node features")
        # Testing settings
        parser.add_argument('--use_valedges_as_input', action='store_true')
        parser.add_argument('--eval_steps', type=int, default=1)
        parser.add_argument('--data_appendix', type=str, default='',
                            help="an appendix to the data directory")
        # Pseudo labeling settings
        parser.add_argument("--iter", type=int, default=50, help="Number of pseudo labeling iteration")
        parser.add_argument("--upbound", type=float, default=0.9, help="Threshold for pseudo labeling")
        parser.add_argument("--top", type=int, default=600, help="Number of maximum pseudo label in each iteration")
        parser.add_argument("--aug_drop", type=float, default=0.9, help="Attribute augmentation dropout rate")
        parser.add_argument("--view", type=int, default=2, help="Number of extra view of augmentation")
        parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of training edges for modeling")
        args = parser.parse_args()

        seed = args.seed
        p_val = args.p_val
        p_test = args.p_test

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset download & loading
        root = args.data_dir
        os.makedirs(root, exist_ok=True)
        if args.dataset.startswith('ogbl'):
            dataset = PygLinkPropPredDataset(name=args.dataset)
            evaluator = Evaluator(name=args.dataset)
            split_edge = dataset.get_edge_split()
            data = dataset[0]
        else:
            if "Amazon" in args.dataset:
                root = os.path.join(root, "amazon")
                if "com" in args.dataset:
                    dataset = Amazon(root, "Computers")
                elif "ph" in args.dataset:
                    dataset = Amazon(root, "Photo")

            if args.dataset in ["PubMed", "CiteSeer"]:
                dataset = CitationFull(root, args.dataset)
            else:
                root = os.path.join(root, args.dataset.lower())
                if args.dataset == "WikiCS":
                    dataset = WikiCS(root)
                elif args.dataset == "Actor":
                    dataset = Actor(root)
                elif args.dataset == "PT":
                    dataset = Twitch(root, "PT")
                elif args.dataset == "LastFM":
                    dataset = LastFMAsia(root)

            mask, split_edge = do_edge_split(dataset, seed=seed, val_ratio=p_val, test_ratio=p_test)
            data = dataset[0].to(device)
            data.full_edge_index = data.edge_index.cpu()
            data.edge_index = split_edge['train']['edge'].t()

        # Dataset preprocess folder suffix
        if args.data_appendix == '':
            args.data_appendix = '_h{}_{}_rph{}'.format(
                args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
            if args.max_nodes_per_hop is not None:
                args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
            if args.use_valedges_as_input:
                args.data_appendix += '_uvai'

        # Metric selection
        if args.dataset.startswith('ogbl-citation'):
            args.eval_metric = 'mrr'
            directed = True
        elif args.dataset.startswith('ogbl'):
            args.eval_metric = 'hits'
            directed = False
        else:  # assume other datasets are undirected
            args.eval_metric = 'auc'
            directed = False

        # Pseudo labeling parameters
        run = args.iter  # 20
        upbound = args.upbound
        aug_drop = args.aug_drop  # 0.95
        K = args.view  # 0
        top = args.top  # 10

        conf_threshold = upbound

        # Create output directory
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_dir = os.path.join(out_dir, "seal")
        os.makedirs(out_dir, exist_ok=True)

        out_raw = "{}_raw_seal_{}_{}_{}.txt".format(args.dataset, run, upbound, top)
        out_pl = "{}_pseudo_seal_{}_{}_{}.txt".format(args.dataset, run, upbound, top)

        out_raw = os.path.join(out_dir, out_raw)
        out_pl = os.path.join(out_dir, out_pl)

        # Logger file
        log_file = "log_{}_seal_{}_{}_{}.txt".format(args.dataset, run, upbound, top)
        log_file = os.path.join(out_dir, log_file)
        log_format = '%(levelname)s %(asctime)s - %(message)s'
        log_time_format = '%Y-%m-%d %H:%M:%S'
        logging.basicConfig(
            format=log_format,
            datefmt=log_time_format,
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger = logging.getLogger()

        # handler = logging.StreamHandler(sys.stdout)
        # handler.setLevel(logging.INFO)
        # handler.setFormatter(logging.Formatter(log_format))
        # logger.addHandler(handler)

        print("Dataset: {}, Pseudo Label Run: {}, Upper Bound: {}, Maximum Label Per Run: {}"
              .format(args.dataset, run, upbound, top))

        # SEAL.
        path = os.path.join(dataset.root, '{}_seal{}'.format(args.dataset, args.data_appendix))
        use_coalesce = True if args.dataset == 'ogbl-collab' else False

        train_dataset = SEALDynamicDataset(
            path,
            data,
            split_edge,
            num_hops=args.num_hops,
            seed=seed,
            percent=args.train_percent,
            split='train',
            use_coalesce=use_coalesce,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
        )
        val_dataset = SEALDynamicDataset(
            path,
            data,
            split_edge,
            num_hops=args.num_hops,
            seed=seed,
            percent=args.val_percent,
            split='valid',
            use_coalesce=use_coalesce,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
        )
        test_dataset = SEALDynamicDataset(
            path,
            data,
            split_edge,
            num_hops=args.num_hops,
            seed=seed,
            percent=args.test_percent,
            split='test',
            use_coalesce=use_coalesce,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
        )

        max_z = 1000  # set a large max_z so that every z has embeddings to look up

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

        # Record the best result and model
        best_auc = 0
        best_ap = 0
        best_val_auc = 0
        best_model = None

        pl_link = np.array([])

        set_all_seed(seed)

        if args.train_node_embedding:
            emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
        elif args.pretrained_node_embedding:
            weight = torch.load(args.pretrained_node_embedding)
            emb = torch.nn.Embedding.from_pretrained(weight)
            emb.weight.requires_grad = False
        else:
            emb = None

        set_all_seed(seed)

        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
                      train_dataset, args.dynamic_train, use_feature=args.use_feature,
                      node_embedding=emb).to(device)
        parameters = list(model.parameters())
        if args.train_node_embedding:
            torch.nn.init.xavier_uniform_(emb.weight)
            parameters += list(emb.parameters())

        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)

        # Training starts
        for epoch in range(args.epochs):
            # print("Model training epoch", epoch)
            train(model, train_loader, optimizer, device)

            if epoch % args.eval_steps == 0:
                auc_val, ap_val, auc, ap = tst(model, val_loader, test_loader, device)
                if auc_val > best_val_auc:
                    best_val_auc = auc_val
                    best_auc = auc
                    best_ap = ap
                    best_model = copy.deepcopy(model)

        print("Raw - Best AUC: {}, Best AP: {}".format(best_auc, best_ap))

        # Add pseudo label
        num_pos_edge = split_edge['train']['edge'].size(0)
        # Free CUDA memory
        del train_dataset, train_loader
        neg_dataset = SEALDynamicNegativeDataset(
            path,
            data,
            mask,
            num_hops=args.num_hops,
            seed=seed,
            base=num_pos_edge,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
        )
        neg_loader = DataLoader(neg_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers)
        mask, split_edge, pseudo_ones = pseudo_label(model, neg_loader, mask, split_edge,
                                                     K, aug_drop, upbound, top, pl_link, device)
        pl_link = pseudo_ones.cpu().numpy()
        # Update training edge label for subgraph extraction
        data.edge_index = split_edge['train']['edge'].t()
        # Free CUDA memory
        del neg_dataset, neg_loader

        # Pseudo labeling iteration
        best_run_auc = 0
        best_run_ap = 0
        m = -1
        n_label = 0
        while pl_link.shape[1] < run * top:
            m = m + 1
            set_all_seed(seed)
            # Adjust training dataset
            train_dataset = SEALDynamicDataset(
                path,
                data,
                split_edge,
                num_hops=args.num_hops,
                seed=seed,
                percent=args.train_percent,
                split='train',
                use_coalesce=use_coalesce,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)

            best_auc = 0
            best_ap = 0
            best_val_auc = 0
            best_model = None

            set_all_seed(seed)

            if args.train_node_embedding:
                emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
            elif args.pretrained_node_embedding:
                weight = torch.load(args.pretrained_node_embedding)
                emb = torch.nn.Embedding.from_pretrained(weight)
                emb.weight.requires_grad = False
            else:
                emb = None

            set_all_seed(seed)

            model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
                          train_dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb).to(device)
            parameters = list(model.parameters())
            if args.train_node_embedding:
                torch.nn.init.xavier_uniform_(emb.weight)
                parameters += list(emb.parameters())

            optimizer = torch.optim.Adam(params=parameters, lr=args.lr)

            for epoch in range(args.epochs):
                # print("Model training epoch", epoch)
                train(model, train_loader, optimizer, device)

                if epoch % args.eval_steps == 0:
                    auc_val, ap_val, auc, ap = tst(model, val_loader, test_loader, device)
                    if auc_val > best_val_auc:
                        best_val_auc = auc_val
                        best_auc = auc
                        best_ap = ap
                        best_model = copy.deepcopy(model)
            print("Pseudo run {} - AUC: {}, AP: {}, pseudo labeled:{}"
                  .format(m, best_auc, best_ap, pl_link.shape[1]))

            num_pos_edge = split_edge['train']['edge'].size(0)
            # Free CUDA memory
            del train_dataset, train_loader
            neg_dataset = SEALDynamicNegativeDataset(
                path,
                data,
                mask,
                num_hops=args.num_hops,
                seed=seed,
                base=num_pos_edge,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
            )
            neg_loader = DataLoader(neg_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
            mask, split_edge, pseudo_ones = pseudo_label(model, neg_loader, mask, split_edge,
                                                         K, aug_drop, upbound, top, pl_link, device)

            pl_link = np.concatenate((pl_link, pseudo_ones.cpu().numpy()), axis=1)
            print("{} pairs have been labeled with confidence threshold {}.".format(pl_link.shape[1], conf_threshold))

            # Update training edge label for subgraph extraction
            data.edge_index = split_edge['train']['edge'].t()
            # Free CUDA memory
            del neg_dataset, neg_loader

            if best_auc > best_run_auc:
                best_run_auc = best_auc
                best_run_ap = best_ap

        print("Best: {}, {}".format(best_run_auc, best_run_ap))
        print("Pseudo {} links: {}".format(pl_link.shape[1], pl_link))


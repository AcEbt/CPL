import os
import random
import numpy as np
import copy
import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.optim import Adam

from torch_geometric.datasets import WikiCS, Actor, Twitch, Amazon, LastFMAsia, CitationFull
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges, remove_self_loops, negative_sampling, dropout_adj
import torch.nn.functional as F

from model import DeepGAE
import warnings

warnings.filterwarnings("ignore")


def set_all_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


for seeds in [42, 4321, 4399, 128, 1024]: #
    print("current seed is: {}".format(seeds))
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='GAE')
        # General settings
        parser.add_argument("--dataset", type=str, default="Actor",
                            help="Actor/WikiCS/PT/Amazon_ph/CiteSeer")
        parser.add_argument("--data_dir", type=str, default="datasets", help="Data directory")
        parser.add_argument("--out_dir", type=str, default="results/", help="Result directory")
        parser.add_argument("--record", action="store_true", default=False, help="Indicator of recording results")

        # Training settings
        parser.add_argument("--seed", type=int, default=4399, help="Random seed")
        parser.add_argument("--hid_dim_1", type=int, default=32, help="Hidden layer dimension")
        parser.add_argument("--epoch", type=int, default=400, help="The max number of epochs")
        parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of optimizer")
        parser.add_argument("--p_val", type=float, default=0.4, help="percentage of validation set")
        parser.add_argument("--p_test", type=float, default=0.5, help="percentage of test set")

        parser.add_argument("--iter", type=int, default=100, help="Number of pseudo labeling iteration")
        parser.add_argument("--upbound", type=float, default=0.50, help="Threshold for pseudo labeling")
        parser.add_argument("--top", type=int, default=200, help="Number of pseudo label in each iteration")
        parser.add_argument("--early_stop", type=int, default=1, help="stop after consecutive 0 PL iteration")
        parser.add_argument("--dropout", type=float, default=0.9, help="Dropout rate in training")

        parser.add_argument("--feature_view", action="store_true", default=False, help="Indicator of feature view")
        parser.add_argument("--aug_drop", type=float, default=0.95, help="Attribute augmentation dropout rate")
        parser.add_argument("--structure_view", action="store_true", default=False, help="Indicator of feature view")
        parser.add_argument("--adj_drop", type=float, default=0.2, help="Adjacency augmentation dropout rate")
        parser.add_argument("--node_view", action="store_true", default=False, help="Indicator of feature view")
        parser.add_argument("--node_drop", type=float, default=0.05, help="Node augmentation dropout rate")
        parser.add_argument("--view", type=int, default=2, help="Number of extra view of augmentation")
        parser.add_argument("--random_pick", action="store_true", default=False, help="Indicator of random pseudo labeling")

        parser.add_argument("--losstype", type=str, default="ce", help="The type of loss function (ce/hinge)")

        args = parser.parse_args()

        seed = seeds  # 4321
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = args.dataset  # "Citeseer"
        losstype = args.losstype
        root = args.data_dir
        os.makedirs("datasets", exist_ok=True)
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

        data = dataset[0].to(device)
        # print("Data shape", data.x.shape)

        enc_in_channels = data.x.shape[1]
        enc_hidden_channels = args.hid_dim_1  # 32
        enc_out_channels = 16
        lr = args.lr  # 0.01
        epochs = args.epoch  # 400
        run = args.iter  # 20
        upbound = args.upbound  # 0.99
        dropout = args.dropout  # 0.9
        aug_drop = args.aug_drop
        early_stop = args.early_stop # 0.95
        K = args.view  # 0
        top = args.top  # 10
        p_val = args.p_val
        p_test = args.p_test
        conf_threshold = 0.995

        if args.dataset == "CiteSeer":
            conf_threshold = 0.95
        # print(aug_drop, K)

        # Create output directory
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_dir = os.path.join(out_dir, "gae")
        os.makedirs(out_dir, exist_ok=True)
        out_raw = "{}_raw_gae_{}_{}_{}.txt".format(args.dataset, run, upbound, top)
        out_raw = os.path.join(out_dir, out_raw)
        out_pl = "{}_pseudo_gae_{}_{}_{}.txt".format(args.dataset, run, upbound, top)
        out_pl = os.path.join(out_dir, out_pl)

        # Logger file
        log_file = "log_{}_gae_{}_{}_{}.txt".format(args.dataset, run, upbound, top)
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

        logger.info("Dataset: {}, Pseudo Label Run: {}, Upper Bound: {}, Maximum Label Per Run: {}, Training Percent:{}"
                    .format(args.dataset, run, upbound, top, 1-p_test-p_val))
        view_info = []
        if args.feature_view:
            view_info.append("Feature View: {}".format(args.aug_drop))
        if args.structure_view:
            view_info.append("Structure View: {}".format(args.adj_drop))
        if args.node_view:
            view_info.append("Node View: {}".format(args.node_drop))

        if len(view_info) != 0:
            logger.info(", ".join(view_info))

        set_all_seed(seed)

        all_edge_index = data.edge_index
        data = train_test_split_edges(data, p_val, p_test)
        train_negative_edge_index, _ = remove_self_loops(all_edge_index)
        data.train_negative_edges = train_negative_edge_index

        sample_size = data.train_pos_edge_index.shape[1]
        retain_sample_size = int(sample_size * aug_drop)

        mask = torch.ones(data.num_nodes, data.num_nodes) - torch.eye(data.num_nodes)
        mask[data.train_pos_edge_index[0], data.train_pos_edge_index[1]] = 0
        mask = mask.to(device)

        set_all_seed(seed)

        best_auc = 0
        best_ap = 0
        best_val_auc = 0
        best_model = None
        model = DeepGAE(enc_in_channels, enc_hidden_channels, enc_out_channels, dropout, losstype).to(device)
        optimizer = Adam(model.parameters(), lr=lr)

        loss_p = []
        pl_link = []
        consist_pre = []

        for epoch in range(epochs):  # epochs=400
            model.train()
            optimizer.zero_grad()
            loss = model.loss(data.x, data.train_pos_edge_index, data.train_negative_edges, seed)
            loss.backward()
            optimizer.step()
            if epoch % 1 == 0:
                model.eval()
                roc_auc, ap = model.single_test(data.x,
                                                data.train_pos_edge_index,
                                                data.test_pos_edge_index,
                                                data.test_neg_edge_index)
                roc_auc_val, ap_val = model.single_test(data.x,
                                                        data.train_pos_edge_index,
                                                        data.val_pos_edge_index,
                                                        data.val_neg_edge_index)
                # print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
                if roc_auc_val > best_val_auc:
                    best_val_auc = roc_auc_val
                    best_auc = roc_auc
                    best_ap = ap
                    best_model = copy.deepcopy(model)
        logger.info("Raw - AUC: {}, AP: {}".format(best_auc, best_ap))

        best_model.eval()
        adj_pred = best_model(data.x, data.train_pos_edge_index)

        if not args.random_pick:
            pp_edge = adj_pred[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
            pn_edge = adj_pred[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
            p_edge = torch.cat((pp_edge, pn_edge))
            adj_pp = adj_pred * 0
            consist_p = torch.ones(p_edge.shape).bool().to(device)

            K = 1

            if args.feature_view:
                x = F.dropout(data.x, p=aug_drop)
                adj_pp = best_model(x, data.train_pos_edge_index)
                adj_pred += adj_pp
                K += 1

                pp_edge = adj_pp[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
                pn_edge = adj_pp[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
                p_a_edge = torch.cat((pp_edge, pn_edge))
                consist_p *= (p_edge > 0.5) == (p_a_edge > 0.5)
                del adj_pp, pp_edge, pn_edge, p_a_edge

            if args.structure_view:
                if hasattr(data, 'train_edge_weight'):
                    edge_index, edge_weight = dropout_adj(
                        data.train_pos_edge_index,
                        edge_attr=data.train_edge_weight,
                        p=args.adj_drop,
                        num_nodes=data.num_nodes
                    )
                    edge_weight /= (1 - args.adj_drop)
                    adj_pp = best_model(data.x, edge_index, edge_weight)
                else:
                    edge_index, _ = dropout_adj(
                        data.train_pos_edge_index,
                        p=args.adj_drop,
                        num_nodes=data.num_nodes
                    )
                    adj_pp = best_model(data.x, edge_index)

                adj_pred += adj_pp
                K += 1

                pp_edge = adj_pp[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
                pn_edge = adj_pp[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
                p_a_edge = torch.cat((pp_edge, pn_edge))
                consist_p *= (p_edge > 0.5) == (p_a_edge > 0.5)
                del adj_pp, pp_edge, pn_edge, p_a_edge

            if args.node_view:
                drop_idx = np.random.choice(data.num_nodes, int(args.node_drop * data.num_nodes))
                x = data.x.clone()
                x[drop_idx] = 0

                adj_pp = best_model(x, data.train_pos_edge_index)
                adj_pred += adj_pp
                K += 1

                pp_edge = adj_pp[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
                pn_edge = adj_pp[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
                p_a_edge = torch.cat((pp_edge, pn_edge))
                consist_p *= (p_edge > 0.5) == (p_a_edge > 0.5)
                del adj_pp, pp_edge, pn_edge, p_a_edge

            consist_pre.append(consist_p.float().mean().item())
            # for _ in range(K):
            #     x = F.dropout(data.x, p=aug_drop)
            #     adj_pred += best_model(x, data.train_pos_edge_index)
            adj_pred = adj_pred / K
            row, col = torch.where((adj_pred * mask) > conf_threshold)
            values = adj_pred[row, col]
            conf_score = conf_threshold
            del adj_pred, best_model, consist_p
            if len(values) >= top:
                conf_score, idx = torch.topk(values, top)
                conf_score = conf_score[0].item()
                row, col = row[idx], col[idx]
            # print(len(row))
        else:
            row, col = torch.where((adj_pred * mask) > 0.5)
            rand_idx = random.sample(range(row.shape[0]), top)
            row, col = row[rand_idx], col[rand_idx]
            conf_score = 0

        pseudo_ones = torch.stack((row, col))
        mask[row, col] = 0
        data.train_pos_edge_index = torch.cat((pseudo_ones, data.train_pos_edge_index), 1)
        logger.info("{} pairs have been labeled with confidence {}.".format(len(row), conf_score))

        lossp = model.loss(data.x, data.train_pos_edge_index, data.train_negative_edges, seed)
        loss_p.append(lossp.cpu().item())
        pl_link = pseudo_ones.cpu().numpy()

        best_run_auc = 0
        best_run_ap = 0
        m = -1

        while pl_link.shape[1] < run*top:
            m = m+1
            set_all_seed(seed)

            best_auc = 0
            best_ap = 0
            best_val_auc = 0
            best_model = None
            model = DeepGAE(enc_in_channels, enc_hidden_channels, enc_out_channels, dropout, losstype).to(device)
            optimizer = Adam(model.parameters(), lr=lr)

            if pseudo_ones.shape[1] != 0:
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    loss = model.loss(data.x, data.train_pos_edge_index, all_edge_index, seed)
                    loss.backward()
                    optimizer.step()
                    if epoch % 1 == 0:
                        model.eval()
                        roc_auc, ap = model.single_test(data.x,
                                                        data.train_pos_edge_index,
                                                        data.test_pos_edge_index,
                                                        data.test_neg_edge_index)
                        roc_auc_val, ap_val = model.single_test(data.x,
                                                                data.train_pos_edge_index,
                                                                data.val_pos_edge_index,
                                                                data.val_neg_edge_index)

                    # print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
                    if roc_auc_val > best_val_auc:
                        best_val_auc = roc_auc_val
                        best_auc = roc_auc
                        best_ap = ap
                        best_model = copy.deepcopy(model)
                logger.info("Pseudo run {} - AUC: {}, AP: {}, pseudo labeled:{}".format(m, best_auc, best_ap, pl_link.shape[1]))

            best_model.eval()
            adj_pred = best_model(data.x, data.train_pos_edge_index)

            if not args.random_pick:

                pp_edge = adj_pred[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
                pn_edge = adj_pred[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
                p_edge = torch.cat((pp_edge, pn_edge))
                adj_pp = adj_pred * 0
                consist_p = torch.ones(p_edge.shape).bool().to(device)

                K = 1

                if args.feature_view:
                    x = F.dropout(data.x, p=aug_drop)
                    adj_pp = best_model(x, data.train_pos_edge_index)
                    adj_pred += adj_pp
                    K += 1

                    pp_edge = adj_pp[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
                    pn_edge = adj_pp[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
                    p_a_edge = torch.cat((pp_edge, pn_edge))
                    consist_p *= (p_edge > 0.5) == (p_a_edge > 0.5)
                    del adj_pp, pp_edge, pn_edge, p_a_edge

                if args.structure_view:
                    if hasattr(data, 'train_edge_weight'):
                        edge_index, edge_weight = dropout_adj(
                            data.train_pos_edge_index,
                            edge_attr=data.train_edge_weight,
                            p=args.adj_drop,
                            num_nodes=data.num_nodes
                        )
                        edge_weight /= (1 - args.adj_drop)
                        adj_pp = best_model(data.x, edge_index, edge_weight)
                    else:
                        edge_index, _ = dropout_adj(
                            data.train_pos_edge_index,
                            p=args.adj_drop,
                            num_nodes=data.num_nodes
                        )
                        adj_pp = best_model(data.x, edge_index)

                    adj_pred += adj_pp
                    K += 1

                    pp_edge = adj_pp[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
                    pn_edge = adj_pp[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
                    p_a_edge = torch.cat((pp_edge, pn_edge))
                    consist_p *= (p_edge > 0.5) == (p_a_edge > 0.5)
                    del adj_pp, pp_edge, pn_edge, p_a_edge

                if args.node_view:
                    drop_idx = np.random.choice(data.num_nodes, int(args.node_drop * data.num_nodes))
                    x = data.x.clone()
                    x[drop_idx] = 0

                    adj_pp = best_model(x, data.train_pos_edge_index)
                    adj_pred += adj_pp
                    K += 1

                    pp_edge = adj_pp[data.train_pos_edge_index[0, :], data.train_pos_edge_index[1, :]]
                    pn_edge = adj_pp[data.train_negative_edges[0, :], data.train_negative_edges[1, :]]
                    p_a_edge = torch.cat((pp_edge, pn_edge))
                    consist_p *= (p_edge > 0.5) == (p_a_edge > 0.5)
                    del adj_pp, pp_edge, pn_edge, p_a_edge

                consist_pre.append(consist_p.float().mean().item())
                # for _ in range(K):
                #     x = F.dropout(data.x, p=aug_drop)
                #     adj_pred += best_model(x, data.train_pos_edge_index)
                adj_pred = adj_pred / K
                row, col = torch.where((adj_pred * mask) > conf_threshold)
                while row.shape[0] < top:
                    conf_threshold -= 0.005
                    row, col = torch.where((adj_pred * mask) > conf_threshold)
                    if conf_threshold < upbound:
                        print("early stop, no proper pseudo labeling target")
                        break
                values = adj_pred[row, col]
                conf_score = conf_threshold
                del adj_pred, best_model, consist_p
                if run * top - pl_link.shape[1] <= len(values) and run * top - pl_link.shape[1] < top:

                    conf_score, idx = torch.topk(values, run * top - pl_link.shape[1])
                    conf_score = conf_score[0].item()
                    row, col = row[idx], col[idx]
                elif len(values) >= top:
                    conf_score, idx = torch.topk(values, top)
                    conf_score = conf_score[0].item()
                    row, col = row[idx], col[idx]
                # print(len(row))
            else:
                row, col = torch.where((adj_pred * mask) > 0.5)
                rand_idx = random.sample(range(row.shape[0]), top)
                row, col = row[rand_idx], col[rand_idx]
                conf_score = 0

            pseudo_ones = torch.stack((row, col))


            mask[row, col] = 0
            data.train_pos_edge_index = torch.cat((pseudo_ones, data.train_pos_edge_index), 1)
            data.train_negative_edges = torch.cat((pseudo_ones, data.train_negative_edges), 1)
            if hasattr(data,'train_edge_weight'):
                data.train_edge_weight = torch.ones(data.train_pos_edge_index.size(1), dtype=torch.float).to(device)

            logger.info("{} pairs have been labeled with confidence {}.".format(len(row), conf_score))
            lossp = model.loss(data.x, data.train_pos_edge_index, data.train_negative_edges, seed)
            loss_p.append(lossp.cpu().item())
            pl_link = np.concatenate((pl_link, pseudo_ones.cpu().numpy()), axis=1)

            if best_auc > best_run_auc:
                best_run_auc = best_auc
                best_run_ap = best_ap

        logger.info("Best: {}, {}".format(best_run_auc, best_run_ap))
        logger.info("Consistency: {}".format(consist_pre))
        logger.info("Pseudo {} links: {}".format(pl_link.shape[1], pl_link))
        logger.info("loss cautious pseudo: {}".format(loss_p))
        np.save(str(seed)+"_pl_links.npy", pl_link)


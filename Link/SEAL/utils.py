import math
import random
import numpy as np
import scipy.sparse as ssp

from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score

import torch
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    # dist_over_2, dist_mod_2 = dist // 2, dist % 2
    dist_over_2, dist_mod_2 = dist.div(2, rounding_mode='floor'), dist.remainder(2)

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    # elif node_label == 'hop':  # mininum distance to src and dst
    #     z = torch.tensor(dists)
    # elif node_label == 'zo':  # zero-one labeling trick
    #     z = (torch.tensor(dists)==0).to(torch.long)
    # elif node_label == 'de':  # distance encoding
    #     z = de_node_labeling(adj, 0, 1)
    # elif node_label == 'de+':
    #     z = de_plus_node_labeling(adj, 0, 1)
    # elif node_label == 'degree':  # this is technically not a valid labeling trick
    #     z = torch.tensor(adj.sum(axis=0)).squeeze(0)
    #     z[z>100] = 100  # limit the maximum label to 100
    # else:
    #     z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, 
                node_id=node_ids, num_nodes=num_nodes)
    return data


# def do_edge_split(dataset, fast_split=False, seed=4321, val_ratio=0.05, test_ratio=0.1):
#     data = dataset[0]

#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     if not fast_split:
#         data = train_test_split_edges(data, val_ratio, test_ratio)
#         mask = data.train_neg_adj_mask
#         # edge_index, _ = add_self_loops(data.train_pos_edge_index)
#         edge_index, _ = add_self_loops(dataset[0].edge_index)
#         data.train_neg_edge_index = negative_sampling(
#             edge_index, num_nodes=data.num_nodes,
#             num_neg_samples=data.train_pos_edge_index.size(1))
#     else:
#         num_nodes = data.num_nodes
#         row, col = data.edge_index
#         # Return upper triangular portion.
#         mask = row < col
#         row, col = row[mask], col[mask]
#         n_v = int(math.floor(val_ratio * row.size(0)))
#         n_t = int(math.floor(test_ratio * row.size(0)))
#         # Positive edges.
#         perm = torch.randperm(row.size(0))
#         row, col = row[perm], col[perm]
#         r, c = row[:n_v], col[:n_v]
#         data.val_pos_edge_index = torch.stack([r, c], dim=0)
#         r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
#         data.test_pos_edge_index = torch.stack([r, c], dim=0)
#         r, c = row[n_v + n_t:], col[n_v + n_t:]
#         data.train_pos_edge_index = torch.stack([r, c], dim=0)
#         # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
#         neg_edge_index = negative_sampling(
#             data.edge_index, num_nodes=num_nodes,
#             num_neg_samples=row.size(0))
#         data.val_neg_edge_index = neg_edge_index[:, :n_v]
#         data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
#         data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

#     split_edge = {'train': {}, 'valid': {}, 'test': {}}
#     split_edge['train']['edge'] = data.train_pos_edge_index.t()
#     split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
#     split_edge['valid']['edge'] = data.val_pos_edge_index.t()
#     split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
#     split_edge['test']['edge'] = data.test_pos_edge_index.t()
#     split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    
#     return mask, split_edge


def sample_edge(edge_index, seed, train_ratio):
    np.random.seed(seed)
    num = edge_index.size(1)
    perm = np.random.permutation(num)
    perm = perm[:int(train_ratio * num)]
    
    return edge_index[:, perm]


def do_edge_split(dataset, seed=4321, train_ratio=1.0, val_ratio=0.4, test_ratio=0.5):
    data = dataset[0]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    data = train_test_split_edges(data, val_ratio, test_ratio)
    mask = data.train_neg_adj_mask
    edge_index, _ = add_self_loops(dataset[0].edge_index)

    # Control edge ratio
    if train_ratio != 1.0:
        # np.random.seed(seed)
        # num_pos = data.train_pos_edge_index.size(1)
        # perm = np.random.permutation(num_pos)
        # perm = perm[:int(train_ratio * num_pos)]
        # data.train_pos_edge_index = data.train_pos_edge_index[:, perm]

        data.train_pos_edge_index = sample_edge(data.train_pos_edge_index, seed, train_ratio)
        data.val_pos_edge_index = sample_edge(data.val_pos_edge_index, seed, train_ratio)
        data.val_neg_edge_index = sample_edge(data.val_neg_edge_index, seed, train_ratio)
        data.test_pos_edge_index = sample_edge(data.test_pos_edge_index, seed, train_ratio)
        data.test_neg_edge_index = sample_edge(data.test_neg_edge_index, seed, train_ratio)

    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    
    return mask, split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, seed, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            
            np.random.seed(seed)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        
        # subsample for pos_edge
        np.random.seed(seed)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        
        # subsample for neg_edge
        np.random.seed(seed)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(seed)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results
        

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)
    
    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results


def set_all_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
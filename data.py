import numpy as np
import scipy
import os
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import networkx as nx
import torch.optim as optim
import dgl
from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset
from dgl.data import CoraFullDataset
from dgl.data import CoauthorCSDataset
from dgl.data import CoauthorPhysicsDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, WikiCSDataset

from torch_geometric.datasets import Actor, Twitch, LastFMAsia

from dgl.data.utils import split_dataset
import random

from sklearn.metrics.pairwise import cosine_similarity as cos


def load_data(dataset, os_path=None):
    citation_data = ['Cora', 'Citeseer', 'Pubmed']
    if dataset == 'Cora':
        data = CoraGraphDataset()
    elif dataset == 'Citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'Pubmed':
        data = PubmedGraphDataset()
    elif dataset == 'CoraFull':
        data = CoraFullDataset()
    elif dataset == 'CaCS':
        data = CoauthorCSDataset()
    elif dataset == 'CaPH':
        data = CoauthorPhysicsDataset()
    elif dataset == 'APh':
        data = AmazonCoBuyPhotoDataset()
    elif dataset == 'ACom':
        data = AmazonCoBuyComputerDataset()
    elif dataset == 'WikiCS':
        data = WikiCSDataset()
    elif dataset in ['Actor', 'LastFM', 'Twitch']:
        root = os.path.join('dataset', dataset)
        if dataset == 'Actor':
            data = pyg_to_dgl(Actor(root)[0])
        elif dataset == 'Twitch':
            data = pyg_to_dgl(Twitch(root, "PT")[0])
        elif dataset == 'LastFM':
            data = pyg_to_dgl(LastFMAsia(root)[0])
    else:
        raise ValueError('wrong dataset name.') 
    
    g = data[0]
    g = dgl.add_self_loop(g)
    features = g.ndata['feat']
    labels = g.ndata['label']

    if dataset in citation_data:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    else:
        train_mask, val_mask, test_mask = generate_mask(data)
    nxg = g.to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=float)
    oadj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return g, adj, features, labels, train_mask, val_mask, test_mask, oadj


def pyg_to_dgl(dataset):
    edge_index = dataset.edge_index
    num_nodes = dataset.num_nodes
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    node_features = dataset.x
    if node_features is not None:
        g.ndata['feat'] = node_features
    g.ndata['label'] = dataset.y
    return [g]


def graph2adj(g):
    nxg = g.cpu().to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)
    oadj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def generate_mask(dataset, labelrate=[0.05,0.15,0.8]):
    train_ratio,  val_ratio, _ = labelrate
    num_nodes = dataset[0].number_of_nodes()
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    # Generate the train/validation/test masks
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Generate random indices for each split
    train_indices = np.random.choice(num_nodes, num_train, replace=False)
    val_indices = np.random.choice(np.setdiff1d(np.arange(num_nodes), train_indices), num_val, replace=False)
    test_indices = np.setdiff1d(np.arange(num_nodes), np.concatenate((train_indices, val_indices)))

    # Set the corresponding mask values to True
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)

def preprocess_adj(adj, with_ego=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    if with_ego:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj_normalized = normalize_adj(adj)
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, GAE, MLP
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

import random
import numpy as np


def set_all_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# class GCNEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout):
#         super(GCNEncoder, self).__init__()
#         self.lin = nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.lin(x)
#         return x

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.gcn2(x, edge_index, edge_weight)

        # x = F.leaky_relu(self.gcn1(x, edge_index), 0.1)
        # x = F.leaky_relu(self.gcn2(x, edge_index), 0.1)
        return x


class LinearDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(LinearDecoder, self).__init__()
        # self.lin1 = nn.Linear(in_channels, hidden_channels)
        # self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(2*in_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index=None, sigmoid=True):
        # x = self.lin1(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        if edge_index is None:
            num = x.shape[0]
            pred = torch.zeros((num, num)).to(x.device)
            for i in num:
                for j in num:
                    pred[i, j] = self.lin1(torch.cat([x[i, :], x[j: ]], dim=1))
        else:
            num = edge_index.shape[1]
            pred = torch.zeros(num).to(x.device)
            for i in range(num):
                row, col = edge_index[0][i], edge_index[0][1]
                pred[i] = self.lin1(torch.cat([x[row, :], x[col, :]]))

        if sigmoid:    
            pred = torch.sigmoid(pred)

        return pred


class DeepGAE(GAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels, dropout):
        super(DeepGAE, self).__init__(encoder=GCNEncoder(enc_in_channels,
                                                          enc_hidden_channels,
                                                          enc_out_channels, dropout),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encode(x, edge_index, edge_weight)
        adj_pred = self.decoder.forward_all(z)
        # adj_pred = self.decode(z)
        return adj_pred

    def loss(self, x, pos_edge_index, train_negative_edges, seed, train_edge_weight=None):
        # Set seed for negative sampling
        set_all_seed(seed)
        
        z = self.encode(x, pos_edge_index, train_edge_weight)
        neg_edge_index = negative_sampling(train_negative_edges, z.size(0), pos_edge_index.size(1))

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        loss = pos_loss + neg_loss

        # pos_prob = self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15
        # neg_prob = self.decoder(z, neg_edge_index, sigmoid=True) - 1e-15
        # prob = torch.cat([pos_prob, neg_prob]).to(z.device)
        # label = torch.cat([torch.ones(pos_prob.shape[0]), torch.zeros(neg_prob.shape[0])]).to(z.device)
        # loss = F.binary_cross_entropy(prob, label)

        return loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
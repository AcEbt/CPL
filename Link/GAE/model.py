import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE, GAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

import random
import numpy as np


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class DeepGAE(GAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels, dropout, losstype):
        super(DeepGAE, self).__init__(encoder=GCNEncoder(enc_in_channels,
                                                          enc_hidden_channels,
                                                          enc_out_channels, dropout),
                                       decoder=InnerProductDecoder())
        self.losstype = losstype

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, train_negative_edges, seed):
        # Set seed for negative sampling
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        z = self.encode(x, pos_edge_index)
        neg_edge_index = negative_sampling(train_negative_edges, z.size(0), pos_edge_index.size(1))

        if self.losstype == "ce":
            # cross-entropy loss
            pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        elif self.losstype == "hinge":
            # hinge loss
            pos_loss = 2 - self.decoder(z, pos_edge_index, sigmoid=False)
            pos_loss = (pos_loss + torch.abs(pos_loss))/2
            pos_loss = pos_loss.mean()
            neg_loss = 1 + self.decoder(z, neg_edge_index, sigmoid=False)
            neg_loss = (neg_loss + torch.abs(neg_loss))/2
            neg_loss = neg_loss.mean()
        else:
            raise Exception("loss type error")

        return pos_loss + neg_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score

from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
import scipy.io as scio
import torch
import numpy as np


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio, Conv=ChebConv, non_linearity=torch.sigmoid):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None, negative_score=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x, edge_index).squeeze()
        score = self.non_linearity(score)
        if negative_score is None:
            negative_score = torch.zeros(np.shape(score), dtype=torch.float)
        score = self.non_linearity(score - negative_score)

        perm = topk(score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]

        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score



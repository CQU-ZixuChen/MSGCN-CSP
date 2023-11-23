import torch
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from CSPool_Model import CSPool
from Dataset import MyGraphDataset
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = ChebConv(self.num_features, self.nhid, 1)

        self.poolp1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.pooln1 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.conv2 = ChebConv(self.nhid, self.nhid, 1)

        self.poolp2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.pooln2 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.conv3 = ChebConv(self.nhid, self.nhid, 1)

        self.poolp3 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.pooln3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x_n1, edge_index_n1, _, batch_n1, perm_n1, score_n1 = self.pooln1(x, edge_index, None, batch, None)
        x_p1, edge_index_p1, _, batch_p1, perm_p1, score_p1 = self.poolp1(x, edge_index, None, batch, score_n1)
        x1 = torch.cat([gmp(x_p1, batch_p1), gap(x_p1, batch_p1)], dim=1)
        n1 = torch.cat([gmp(x_n1, batch_n1), gap(x_n1, batch_n1)], dim=1)

        x_p1 = F.relu(self.conv2(x_p1, edge_index_p1))
        x_n2, edge_index_n2, _, batch_n2, perm_n2, score_n2 = self.pooln2(x_p1, edge_index_p1, None, batch_p1, None)
        x_p2, edge_index_p2, _, batch_p2, perm_p2, score_p2 = self.poolp2(x_p1, edge_index_p1, None, batch_p1, score_n2)
        x2 = torch.cat([gmp(x_p2, batch_p2), gap(x_p2, batch_p2)], dim=1)
        n2 = torch.cat([gmp(x_n2, batch_n2), gap(x_n2, batch_n2)], dim=1)

        x_p2 = F.relu(self.conv3(x_p2, edge_index_p2))
        x_n3, edge_index_n3, _, batch_n3, perm_n3, score_n3 = self.pooln3(x_p2, edge_index_p2, None, batch_p2, None)
        x_p3, edge_index_p3, _, batch_p3, perm_p3, score_p3 = self.pooln3(x_p2, edge_index_p2, None, batch_p2, score_n3)
        x3 = torch.cat([gmp(x_p3, batch_p3), gap(x_p3, batch_p3)], dim=1)
        n3 = torch.cat([gmp(x_n3, batch_n3), gap(x_n3, batch_n3)], dim=1)

        x = x1 + x2 + x3

        n = n1 + n2 + n3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        n = F.relu(self.lin1(n))
        n = F.dropout(n, p=self.dropout_ratio, training=self.training)
        n = F.relu(self.lin2(n))
        n = F.log_softmax(self.lin3(n), dim=-1)

        return x, n, score_n1, score_p1, score_n2, score_p2, score_n3, score_p3


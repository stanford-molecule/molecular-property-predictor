from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GMN(nn.Module):
    """Dense version of GMN."""
    def __init__(
            self,
            alpha: float,
            e_out: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            pos_dim: int,
            num_centroids: List[int],
            max_nodes: int,
            cluster_heads: int,
            dropout: float,
            batchnorm: bool,
            c_heads_pool: str,
            p2p: bool,
            linear_block: bool,
            backward_period: int,
            prior_centroids=None,
            num_classes: int = 2
    ):
        super(GMN, self).__init__()
        self.e_out = e_out
        self.cluster_heads = cluster_heads
        self.hidden_dim = hidden_dim
        self.positional_hiddim = pos_dim
        self.num_centroids = num_centroids
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.c_heads_pool = c_heads_pool
        self.p2p = p2p
        self.linear_block = linear_block
        self.backward_period = backward_period

        self.fc = nn.Linear(hidden_dim, 2)
        self.bn2 = torch.nn.BatchNorm1d(output_dim + pos_dim)
        self.total_cluster_layers = len(num_centroids) - 1
        self.total_centroids = sum(num_centroids)

        w = torch.empty(1, output_dim + e_out)

        self.ref_points = nn.init.xavier_uniform_(w, gain=2)
        for i in range(1, hidden_dim):
            self.ref_points = torch.cat((self.ref_points, nn.init.xavier_uniform_(w, gain=2)), dim=0)

        # note: may need to fix this line to assign to multiple GPUs if they ever become available
        # TODO: match this with `Experiment`
        self.device = torch.device("cuda" if self.has_gpu else "cpu")
        if self.has_gpu:
            self.ref_points = self.ref_points.to(self.device)

        self.q = [0] * self.cluster_heads
        self.p = [0] * self.cluster_heads
        self.q_adj = [0] * self.cluster_heads
        self.new_adj = [0] * self.cluster_heads
        self.new_feat = [0] * self.cluster_heads

        if prior_centroids is None:
            self.centroids = \
                nn.Parameter(2 * torch.rand(
                    self.cluster_heads,
                    (self.total_centroids - 1) * (self.hidden_dim // 2 + self.positional_hiddim)) - 1)
        else:
            self.centroids = nn.Parameter(prior_centroids)

        self.centroids.requires_grad = True
        self.last_layer_dnn = nn.Linear(self.hidden_dim // 2 + pos_dim, num_classes)
        self.lower_dimension_last = nn.Linear(hidden_dim, output_dim)
        self.hard_loss = torch.Tensor([0])
        self.headConv = nn.Parameter(torch.zeros(size=(self.cluster_heads, 1)))
        nn.init.xavier_uniform_(self.headConv.data, gain=1.414)
        self.adjlayer = nn.Linear(max_nodes, pos_dim)
        self.wm1 = nn.Linear(input_dim, hidden_dim)
        self.wm2 = nn.Linear(hidden_dim, self.hidden_dim // 2)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.wm21 = nn.Linear(hidden_dim // 2 + pos_dim,
                              hidden_dim // 2 + pos_dim)
        self.xblocklinear = nn.Linear(input_dim + pos_dim, input_dim + pos_dim)

    @property
    def has_gpu(self) -> bool:
        return torch.cuda.is_available()

    def forward(self, x_node, adj, epoch, graph_sizes, c_layer, master_node_flag):
        self.master_node_flag = master_node_flag
        if self.master_node_flag:  # Creating the super node connected to every node
            master_adj, master_feat = adj.to(self.device), x_node.to(self.device)

        # we need it only for the first layer
        if c_layer == 0:
            graph_sizes = torch.LongTensor(graph_sizes)
            # size : same az p
            aranger = torch.arange(adj.shape[1]).view(1, 1, -1).repeat(adj.shape[0], self.num_centroids[c_layer], 1)
            # size: same az p
            graph_broad = graph_sizes.view(-1, 1, 1).repeat(1, self.num_centroids[c_layer], adj.shape[1])
            if self.has_gpu:
                aranger = aranger.to(self.device)
                graph_broad = graph_broad.to(self.device)
                self.centroids = self.centroids.to(self.device)
            self.mask = aranger < graph_broad
        else:
            self.mask = None

        if self.master_node_flag:
            new_adj, new_feat, hardening_loss, h_prime = self.query(master_feat, master_adj, c_layer)
        else:
            new_adj, new_feat, hardening_loss, points = self.query(x_node, adj, c_layer)

        if not master_node_flag:   # Updating the centroids as well
            self.centroids.requires_grad = True
            return self.centroids, hardening_loss, new_adj, new_feat, points
        else:
            self.centroids.requires_grad = False
            if (epoch + 1) % self.backward_period:
                self.centroids.requires_grad = True
            h_prime = self.last_layer_dnn(torch.mean(h_prime, 1))
            return self.centroids, hardening_loss, new_adj, new_feat, h_prime

    def query(self, x_node, adj, cluster_layer_num):
        if cluster_layer_num == 0:
            x_node = self.leakyrelu(F.dropout(self.wm1(x_node), p=self.dropout, training=self.training))
            x_node = self.leakyrelu(F.dropout(self.wm2(x_node), p=self.dropout, training=self.training))
            adj_feat = self.leakyrelu(F.dropout(self.adjlayer(adj), p=self.dropout, training=self.training))
            h_prime = torch.cat((x_node, adj_feat), 2)
        else:
            h_prime = self.leakyrelu(F.dropout(self.wm21(x_node), p=self.dropout, training=self.training))

        if self.master_node_flag:
            return adj, x_node, self.hard_loss, h_prime
        else:
            if self.batchnorm:
                h_prime = self.bn2(h_prime.transpose(1, 2)).transpose(1, 2)
            else:
                h_prime = torch.squeeze(h_prime)
            new_adj, __, new_feat = self.cluster_block(h_prime, adj, cluster_layer_num)
            return new_adj, new_feat, self.hard_loss, h_prime

    def cluster_block(self, x, adj, cluster_layer_num):
        """ This function calculates the assignment matrix for keys (batch_centroids) and queries (points) """
        cumsum = np.cumsum(self.num_centroids)
        cumsum = np.insert(cumsum, 0, 0)
        batch_centroids = \
            self.centroids[:,
            cumsum[cluster_layer_num] * (self.hidden_dim // 2 + self.positional_hiddim):
            cumsum[cluster_layer_num + 1] * (self.hidden_dim // 2 + self.positional_hiddim)]
        batch_centroids = torch.unsqueeze(
            batch_centroids.view(self.cluster_heads, -1,
                                 (self.hidden_dim // 2 + self.positional_hiddim)), 0).\
            repeat(x.shape[0], 1, 1, 1)

        # size: [batch_szie, centers, graphsize, feat]
        points = torch.unsqueeze(x, 1).repeat(1, batch_centroids.shape[1], 1, 1)
        # size: [batch_szie, nHeads, centers, graphsize, feat]
        points = torch.unsqueeze(points, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1)
        # same size az points
        batch_centroids_broad = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, points.shape[3], 1)
        if self.has_gpu:
            batch_centroids_broad = batch_centroids_broad.to(self.device)

        # size [batch_size, cHeads, centrs, graphsize]
        dist = torch.sum(torch.abs(points - batch_centroids_broad) ** 2, 4)
        if self.mask is not None:
            self.mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.cluster_heads, 1, 1)
            m = torch.tensor(self.mask_broad, dtype=torch.float32)
            dist = dist * m.to(self.device)

        nu = 1  # this is a hyperparameter, same as the one in the taxonomy paper
        q = torch.pow((1 + dist / nu), -(nu + 1) / 2)
        denominator = torch.unsqueeze(torch.sum(q, 2), 2)
        q = q / denominator  # size: [batch, nHeads, centers, graphsize

        if self.mask is not None:
            self.mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.cluster_heads, 1, 1)
            m = torch.tensor(self.mask_broad, dtype=torch.float32)
            q = q * m.to(self.device)

        if self.cluster_heads > 1:
            if self.c_heads_pool == 'mean':
                q = torch.mean(q, 1)
            elif self.c_heads_pool == 'max':
                q, _ = torch.max(q, 1)
            elif self.c_heads_pool == 'conv':
                q = q.permute(0, 3, 2, 1)
                q = torch.matmul(q, self.headConv)
                q = torch.squeeze(q.permute(0, 3, 2, 1))

            # Sums to one for all of the nodes
            q = torch.softmax(q, 1)
            if self.mask is not None:
                m = torch.tensor(self.mask, dtype=torch.float32).to(self.device)
                q = q * m
        else:
            q = torch.squeeze(q)

        # Hard loss after convolution
        p = torch.pow(q, 2) / torch.unsqueeze(torch.sum(q, 2), 2)
        denominator = torch.sum(p, 1)

        if self.mask is not None:
            m = torch.squeeze(m)
            denominator[~self.mask[:, 0, :]] = 1.

        denominator = torch.unsqueeze(denominator, 1)
        p = p / denominator

        if self.mask is not None:
            p = p + 1 - m.to(self.device)
            q = q + 1 - m.to(self.device)
            hard_loss2 = p * torch.log(p / q)
            hard_loss2[~self.mask] = 0
            self.hard_loss = 100 * torch.sum(hard_loss2)
            q = q - 1 + m.to(self.device)

        q_adj = torch.matmul(q, adj)
        new_adj = torch.matmul(q_adj, q.transpose(1, 2))

        if self.p2p:
            if self.master_node_flag:
                new_adj[:, 0:-1, :] = 0.
            else:
                dg = torch.diag(torch.ones(new_adj.shape[1]))
                new_adj = torch.unsqueeze(dg, 0).repeat(new_adj.shape[0], 1, 1).to(self.device)

        new_feat = torch.matmul(q, x)

        if self.linear_block:
            new_feat = torch.relu_(self.xblocklinear(new_feat))

        return new_adj, q, new_feat

    @staticmethod
    def loss(y_pred, label):
        return F.cross_entropy(y_pred, label, reduction='mean')

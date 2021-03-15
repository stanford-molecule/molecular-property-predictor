from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU
from torch.nn import Linear
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GraphConv
from torch_geometric.data.batch import Batch
from utils import GCNConv
from ogb.graphproppred.mol_encoder import AtomEncoder
import deeper


class MemConv(nn.Module):
    def __init__(
        self,
        num_features,
        heads,
        num_keys,
        dim_out,
        key_std=10,
        variant="gmn",
        max_queries=100,
    ):
        super(MemConv, self).__init__()
        self.heads = heads
        self.num_keys = num_keys
        self.num_features = num_features
        self.dim_out = dim_out

        self.variant = variant

        self.conv1x1 = nn.Conv2d(
            in_channels=heads,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.keys = torch.rand(heads, num_keys, num_features)
        if variant == "gmn":
            self.keys.requires_grad = True
        else:  # do not optimize keys
            self.keys = self.keys * key_std  # keys ~ N(0, scale**2 I)
            self.keys.requires_grad = False

        if variant == "random":
            self.rm = torch.rand(max_queries, num_keys)
            self.rm.requires_grad = False

        self.lin = Linear(self.num_features, self.dim_out)
        self.sigma = LeakyReLU()

    @staticmethod
    def KL_reg(C, mask=None):
        P = C ** 2
        cn = C.sum(dim=-2).unsqueeze(-2) + 1e-08
        P = P / cn.expand_as(P)
        pn = P.sum(dim=-1).unsqueeze(-1) + 1e-08
        P = P / pn.expand_as(P)

        kl = (P * ((P + 1e-08).log() - (C + 1e-08).log())).sum()

        return 100 * kl

    def forward(self, Q, mask, tau=1.0):

        if self.variant == "random":
            C = (
                self.rm[: Q.shape[1], :]
                .unsqueeze(0)
                .expand(Q.shape[0], -1, -1)
                .to(Q.device)
            )
            if mask is not None:
                ext_mask = mask.unsqueeze(2).repeat(1, 1, self.keys.size(1)).to(C.dtype)
        else:
            broad_Q = (
                torch.unsqueeze(Q, 1)
                .expand(-1, self.heads, -1, -1)
                .unsqueeze(3)
                .expand(-1, -1, -1, self.num_keys, -1)
                .to(Q.device)
            )
            broad_keys = (
                torch.unsqueeze(self.keys, 0)
                .expand(broad_Q.shape[0], -1, -1, -1)
                .unsqueeze(2)
                .expand(-1, -1, broad_Q.shape[-3], -1, -1)
                .to(Q.device)
            )
            C = torch.sum(torch.abs(broad_Q - broad_keys) ** 2, 4).sqrt()

            if mask is not None:
                ext_mask = (
                    mask.unsqueeze(1)
                    .unsqueeze(3)
                    .repeat(1, self.heads, 1, self.keys.size(1))
                    .to(C.dtype)
                )
                C = C * ext_mask

            if self.variant == "gmn":
                C = (C ** 2) / tau
                C = 1.0 + C
                C = C ** -(0.5 * tau + 0.5)
                C = C / (C.sum(dim=-1).unsqueeze(-1).expand_as(C))

        if mask is not None:
            C = C * ext_mask

        if self.heads > 1 and self.variant != "random":
            C = self.conv1x1(C)

        C = F.softmax(C.squeeze(1), dim=-1)

        if mask is not None:
            C = C * mask.unsqueeze(-1).expand(-1, -1, self.keys.size(1)).to(C.dtype)

        kl = self.KL_reg(C, mask)

        V = C.transpose(1, 2) @ Q
        return self.sigma(self.lin(V)), kl


class GMN(torch.nn.Module):
    def __init__(
        self,
        num_feats,
        max_nodes,
        num_classes,
        num_heads,
        hidden_dim,
        num_keys,
        mem_hidden_dim=100,
        variant="gmn",
        encode_edge: bool = False,
        use_deeper: bool = False,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        block: Optional[str] = None,
        conv_encode_edge: Optional[bool] = None,
        add_virtual_node: Optional[bool] = None,
        conv: Optional[str] = None,
        gcn_aggr: Optional[str] = None,
        t: Optional[float] = None,
        learn_t: Optional[bool] = None,
        p: Optional[float] = None,
        learn_p: Optional[bool] = None,
        y: Optional[float] = None,
        learn_y: Optional[bool] = None,
        msg_norm: Optional[bool] = None,
        learn_msg_scale: Optional[bool] = None,
        norm: Optional[str] = None,
        mlp_layers: Optional[int] = None,
    ):
        super(GMN, self).__init__()

        self.encode_edge = encode_edge
        self.use_deeper = use_deeper

        if encode_edge:
            if use_deeper:
                self.q0 = deeper.DeeperGCN(
                    num_layers=num_layers,
                    dropout=dropout,
                    block=block,
                    conv_encode_edge=conv_encode_edge,
                    add_virtual_node=add_virtual_node,
                    hidden_channels=hidden_dim,
                    num_tasks=None,
                    conv=conv,
                    gcn_aggr=gcn_aggr,
                    t=t,
                    learn_t=learn_t,
                    p=p,
                    learn_p=learn_p,
                    y=y,
                    learn_y=learn_y,
                    msg_norm=msg_norm,
                    learn_msg_scale=learn_msg_scale,
                    norm=norm,
                    mlp_layers=mlp_layers,
                    graph_pooling=None,
                    node_encoder=True,
                    encode_atom=False,
                )
            else:
                self.q0 = GCNConv(hidden_dim, aggr="add")
        else:
            self.q0 = GraphConv(num_feats, hidden_dim, aggr="add")

        self.num_features = num_feats
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_keys = num_keys
        self.variant = variant

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.q1 = GraphConv(hidden_dim, hidden_dim)

        self.mem_layers = nn.ModuleList()

        max_dims = [self.max_nodes]
        for i, n in enumerate(self.num_keys):
            max_dims.append(n)

            if i == 0:
                num_feats = hidden_dim
            else:
                num_feats = mem_hidden_dim
            mem = MemConv(
                num_features=num_feats,
                heads=self.num_heads,
                num_keys=n,
                dim_out=mem_hidden_dim,
                variant=variant,
                max_queries=max_dims[i],
            )
            self.mem_layers.append(mem)

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)

        self.MLP = nn.Sequential(
            Linear(mem_hidden_dim, 50), nn.LeakyReLU(), Linear(50, self.num_classes)
        )

    def initial_query(self, batch, x, edge_index, edge_attr=None, perturb=None):
        if self.encode_edge:
            x = self.atom_encoder(x)
            x = x + perturb if perturb is not None else x

            if not self.use_deeper:
                x = self.q0(x, edge_index, edge_attr)
            else:
                # this is a hack to avoid changing the API
                b = Batch()
                b.x = x
                b.batch = batch
                b.edge_index = edge_index
                b.edge_attr = edge_attr
                x = self.q0(b)
        else:
            x = self.q0(x, edge_index)
        x = F.relu(x)
        x = F.relu(self.q1(x, edge_index))
        return x

    def forward(self, x, edge_index, batch, edge_attr, perturb=None):
        q0 = self.initial_query(batch, x, edge_index, edge_attr, perturb)

        q0, mask = to_dense_batch(q0, batch=batch)

        batch_size, num_nodes, num_channels = q0.size()
        q0 = q0.view(-1, q0.shape[-1])
        q0 = self.bn(q0)
        q0 = q0.view(batch_size, num_nodes, num_channels)

        q = q0
        kl_total = 0
        for i, mem_layer in enumerate(self.mem_layers):
            if i != 0:
                mask = None
            q, kl = mem_layer(q, mask)
            kl_total += kl

        yh = self.MLP(q.mean(-2))

        return yh, kl_total / len(batch)

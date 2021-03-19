"""
GMN.
Inspired by:
- https://github.com/AaltoPML/Rethinking-pooling-in-GNNs
- https://github.com/amirkhas/GraphMemoryNet
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder
from torch.nn import LeakyReLU
from torch.nn import Linear
from torch_geometric.data.batch import Batch
from torch_geometric.nn import APPNP
from torch_geometric.nn import GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

import deeper

loss_fn = F.nll_loss
softmax = F.log_softmax


def _flag(model, data, device, y, step_size, m, hidden_dim):
    """
    Runs FLAG for a batch.
    this is borrowed from https://github.com/devnkong/FLAG/tree/main/ogb/graphproppred/mol
    """

    forward = lambda p: model(
        data.x, data.edge_index, data.batch, data.edge_attr, perturb=p
    )
    perturb_shape = (data.x.shape[0], hidden_dim)
    perturb = (
        torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    )
    perturb.requires_grad_()
    out, kl = forward(perturb)
    out = softmax(out, dim=-1)
    loss = loss_fn(out, y, reduction="mean")
    loss /= m
    kl /= m

    for _ in range(m - 1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out, kl = forward(perturb)
        out = softmax(out, dim=-1)
        loss = loss_fn(out, y)
        loss /= m
        kl /= m
    return loss, kl


def train(
    model,
    optimizer,
    loader,
    device,
    hidden_dim,
    epoch_stop=None,
    flag: bool = False,
    step_size: float = 1e-3,
    m: int = 3,
):
    """
    Train loop for a single epoch. Runs FLAG if `flag == True`.
    """
    model.train()
    total_ce_loss, total_kl_loss = 0, 0

    for idx, data in enumerate(tqdm(loader, desc="Training")):
        data.to(device)
        y = data.y.view(-1)

        optimizer.zero_grad()
        if flag:
            loss, kl = _flag(model, data, device, y, step_size, m, hidden_dim)
        else:
            out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
            out = softmax(out, dim=-1)
            loss = loss_fn(out, y, reduction="mean")

        loss.backward()
        optimizer.step()

        total_ce_loss += loss.item() * data.y.size(0)
        total_kl_loss += kl.item() * data.y.size(0)

        if epoch_stop and idx >= epoch_stop:
            break

    return total_ce_loss, total_kl_loss


def kl_train(
    model,
    optimizer,
    loader,
    device,
    hidden_dim,
    epoch_stop=None,
    flag: bool = False,
    step_size: float = 1e-3,
    m: int = 3,
):
    """
    Run KL train. Not using FLAG here. TODO: does that make sense?
    """
    total_kl_loss = 0.0
    total_ce_loss = 0.0

    optimizer.zero_grad()
    for idx, data in enumerate(tqdm(loader, desc="KL train")):
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
        out = softmax(out, dim=-1)
        loss = loss_fn(out, data.y.view(-1), reduction="mean")
        kl.backward()

        total_kl_loss += kl.item() * data.y.size(0)
        total_ce_loss += loss.item() * data.y.size(0)

        if epoch_stop and idx >= epoch_stop:
            break

    optimizer.step()

    return total_ce_loss, total_kl_loss


@torch.no_grad()
def evaluate(model, loader, device, evaluator=None, data_split="", epoch_stop=None):
    """
    Evluate GMN given a data split represented by `loader`.
    """
    model.eval()
    loss, kl_loss, correct = 0, 0, 0
    y_pred, y_true = [], []

    for idx, data in enumerate(tqdm(loader, desc=data_split)):
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)

        y_pred.append(out[:, 1])
        y_true.append(data.y)

        out = F.log_softmax(out, dim=-1)
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss += F.nll_loss(out, data.y.view(-1), reduction="mean").item() * data.y.size(
            0
        )
        kl_loss += kl.item() * data.y.size(0)

        if epoch_stop and idx >= epoch_stop:
            break

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if evaluator is None:
        acc = correct / len(loader.dataset)
    else:
        acc = evaluator.eval({"y_pred": y_pred.view(y_true.shape), "y_true": y_true})[
            evaluator.eval_metric
        ]

    return acc, loss, kl_loss


class GCNConv(MessagePassing):
    """
    Taken from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
    """

    def __init__(self, emb_dim, aggr):
        super(GCNConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding) + F.relu(
            x + self.root_emb.weight
        )

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class MemConv(nn.Module):
    """
    Memory layer used in GMN.
    """

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
    def _compute_kl(c, mask=None, eps=1e-8, const=100):
        p = c ** 2
        cn = c.sum(dim=-2).unsqueeze(dim=-2) + eps
        p = p / cn.expand_as(p)
        pn = p.sum(dim=-1).unsqueeze(dim=-1) + eps
        p = p / pn.expand_as(p)
        kl = (p * ((p + eps).log() - (c + eps).log())).sum()
        return kl * const

    def _random(self, q, mask, tau=None):
        c = (
            self.rm[: q.shape[1], :]
            .unsqueeze(0)
            .expand(q.shape[0], -1, -1)
            .to(q.device)
        )
        ext_mask = (
            mask.unsqueeze(2).repeat(1, 1, self.keys.size(1)).to(c.dtype)
            if mask is not None
            else None
        )
        return c, ext_mask

    def _distance(self, q, mask, tau=None):
        qs = (
            torch.unsqueeze(q, 1)
            .expand(-1, self.heads, -1, -1)
            .unsqueeze(dim=3)
            .expand(-1, -1, -1, self.num_keys, -1)
            .to(q.device)
        )
        keys = (
            torch.unsqueeze(self.keys, dim=0)
            .expand(qs.shape[0], -1, -1, -1)
            .unsqueeze(2)
            .expand(-1, -1, qs.shape[-3], -1, -1)
            .to(q.device)
        )
        c = torch.sum(torch.abs(qs - keys) ** 2, dim=4).sqrt()
        if mask is None:
            return c, None

        m = (
            mask.unsqueeze(dim=1)
            .unsqueeze(dim=3)
            .repeat(1, self.heads, 1, self.keys.size(1))
            .to(c.dtype)
        )
        c = c * m
        return c, m

    def _gmn(self, q, mask, tau):
        c, ext_mask = self._distance(q, mask)
        c = 1 + ((c ** 2) / tau)
        c = c ** -(0.5 * tau + 0.5)
        c = c / (c.sum(dim=-1).unsqueeze(dim=-1).expand_as(c))
        return c, ext_mask

    def forward(self, q, mask, tau=1.0):
        fns = {"random": self._random, "gmn": self._gmn, "distance": self._distance}
        c, ext_mask = fns[self.variant](q, mask, tau)
        c = c * ext_mask if mask is not None else c
        c = self.conv1x1(c) if self.heads > 1 and self.variant != "random" else c
        c = F.softmax(c.squeeze(1), dim=-1)
        c = (
            c * mask.unsqueeze(-1).expand(-1, -1, self.keys.size(1)).to(c.dtype)
            if mask is not None
            else c
        )
        kl = self._compute_kl(c, mask)
        v = c.transpose(1, 2) @ q
        return self.sigma(self.lin(v)), kl


class Q0LayerType(str, Enum):
    deeper = "deeper"
    with_edge = "with_edge"
    no_edge = "no_edge"


class GMN(torch.nn.Module):
    """
    GMN with the following added:
    - DeeperGCN node encoder
    - APPNP node encoder
    - FLAG adversarial data augmentation
    """

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
        use_appnp: bool = False,
        k: int = 10,
        alpha: float = 0.1,
        mlp_hidden_dim: int = 50,
    ):
        super(GMN, self).__init__()

        self.k = k
        self.alpha = alpha
        self.use_deeper = use_deeper
        self.num_features = num_feats
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_keys = num_keys
        self.variant = variant

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.q0s = torch.nn.ModuleDict()
        self.q0s[Q0LayerType.with_edge] = GCNConv(hidden_dim, aggr="add")

        if use_deeper:
            deeper_gcn = deeper.DeeperGCN(
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
            self.q0s[Q0LayerType.deeper] = deeper_gcn

        if use_appnp:
            self.q0s[Q0LayerType.no_edge] = APPNP(K=self.k, alpha=self.alpha)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.q0_second = GraphConv(hidden_dim * len(self.q0s), hidden_dim)
        self.mem_layers = nn.ModuleList()

        max_dims = [self.max_nodes]
        for idx, num_keys in enumerate(self.num_keys):
            max_dims.append(num_keys)
            num_feats = hidden_dim if idx == 0 else mem_hidden_dim
            self.mem_layers.append(
                MemConv(
                    num_features=num_feats,
                    heads=self.num_heads,
                    num_keys=num_keys,
                    dim_out=mem_hidden_dim,
                    variant=variant,
                    max_queries=max_dims[idx],
                )
            )

        self.mlp = nn.Sequential(
            Linear(mem_hidden_dim, mlp_hidden_dim),
            nn.LeakyReLU(),
            Linear(mlp_hidden_dim, self.num_classes),
        )

    def _get_q0(self, batch, x, edge_index, edge_attr=None, perturb=None):
        x = self.atom_encoder(x)
        x = x + perturb if perturb is not None else x

        xs = []
        for layer_type, layer in self.q0s.items():
            if layer_type == Q0LayerType.deeper:
                # this is a hack to avoid changing the API
                b = Batch()
                b.x = x
                b.batch = batch
                b.edge_index = edge_index
                b.edge_attr = edge_attr
                xs.append(layer(b))
            elif layer_type == Q0LayerType.with_edge:
                xs.append(layer(x, edge_index, edge_attr))
            elif layer_type == Q0LayerType.no_edge:
                xs.append(layer(x, edge_index))

        return F.relu(self.q0_second(F.relu(torch.cat(xs, dim=1)), edge_index))

    def forward(self, x, edge_index, batch, edge_attr, perturb=None):
        q0 = self._get_q0(batch, x, edge_index, edge_attr, perturb)
        q0, mask = to_dense_batch(q0, batch=batch)
        q0 = self.bn(q0.view(-1, q0.shape[-1])).view(*q0.size())

        q, kl_total = q0, 0
        for i, mem_layer in enumerate(self.mem_layers):
            q, kl = mem_layer(q, mask if i == 0 else None)
            kl_total += kl

        return self.mlp(q.mean(dim=-2)), kl_total / len(batch)

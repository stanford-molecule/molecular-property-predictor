import torch
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)

from conv import GNN_node, GNN_node_Virtualnode


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_tasks,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        virtual_node=True,
        residual=False,
        drop_ratio=0.5,
        jk="last",
        graph_pooling="mean",
    ):
        if num_layer <= 1:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.jk = jk
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        # GNN to generate node embeddings
        gnn_cls = GNN_node_Virtualnode if virtual_node else GNN_node
        self.gnn_node = gnn_cls(
            num_layer,
            emb_dim,
            jk=jk,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
        )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        adj = 2 if graph_pooling == "set2set" else 1
        self.graph_pred_linear = torch.nn.Linear(adj * self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)
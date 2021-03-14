import networkx as nx
import torch
import torch.utils.data
import torch.utils.data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx

from utils import compute_rwr


class DataLoaderGNN(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, max_nodes: int):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        self.max_nodes = max_nodes


class DataLoaderGMN(DataLoaderGNN):
    def __init__(self, dataset, batch_size, shuffle, num_workers, max_nodes: int):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            max_nodes=max_nodes,
        )
        self.num_features = dataset.num_features

    @staticmethod
    def _create_rwr(batch_list, max_nodes: int):
        out = torch.zeros(len(batch_list), max_nodes, max_nodes)
        for idx, g in enumerate(batch_list):
            rwr = torch.tensor(compute_rwr(to_networkx(g)))
            out[idx, : len(rwr), : len(rwr)] = rwr
        return out

    @staticmethod
    def _create_adj(batch_list, max_nodes: int):
        out = torch.zeros(len(batch_list), max_nodes, max_nodes)
        for idx, g in enumerate(batch_list):
            adj = torch.tensor(nx.to_numpy_matrix(to_networkx(g)))
            out[idx, : len(adj), : len(adj)] = adj
        return out

    @staticmethod
    def _create_feats(batch_list, max_nodes: int, num_features: int):
        out = torch.zeros(len(batch_list), max_nodes, num_features)
        for idx, g in enumerate(batch_list):
            out[idx, : len(g.x)] = g.x
        return out

    def __iter__(self):
        for batch in super().__iter__():
            batch_list = batch.to_data_list()
            yield {
                "adj": self._create_adj(batch_list, self.max_nodes),  # TODO: normalize
                "num_nodes": torch.tensor([len(g.x) for g in batch_list]),
                # "rwr": self._create_rwr(batch_list, self.max_nodes),  # TODO: add this back
                "feats": self._create_feats(
                    batch_list, self.max_nodes, self.num_features
                ),
                "label": batch.y.squeeze(),
            }

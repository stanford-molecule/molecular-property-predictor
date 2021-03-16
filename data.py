import os.path as osp

import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader


def get_data(name, batch_size):
    if name == "ogbg-molhiv":
        data_train, data_val, data_test, max_num_nodes = get_molhiv()
        num_classes = 2
    else:
        raise ValueError("dataset not supported")

    stats = dict()
    stats["num_features"] = data_train.num_node_features
    stats["num_classes"] = num_classes
    stats["max_num_nodes"] = max_num_nodes

    evaluator = Evaluator(name)
    encode_edge = True
    train_loader = DataLoader(data_train, batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, stats, evaluator, encode_edge


def get_molhiv():
    path = osp.dirname(osp.realpath(__file__))
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=path)
    split_idx = dataset.get_idx_split()
    max_num_nodes = torch.tensor(dataset.data.num_nodes).max().item()
    return (
        dataset[split_idx["train"]],
        dataset[split_idx["valid"]],
        dataset[split_idx["test"]],
        max_num_nodes,
    )

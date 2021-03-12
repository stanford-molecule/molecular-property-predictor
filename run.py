from typing import NamedTuple, Type

from experiment import GNNExperiment, GMNExperimentRethink


class Experiment(NamedTuple):
    exp_cls: Type[GNNExperiment]
    args: dict


experiments = [
    Experiment(
        exp_cls=GNNExperiment,
        args={
            'gnn_type': 'gcn',
            'dropout': .5,
            'num_layers': 5,
            'emb_dim': 300,
            'epochs': 100,
            'lr': 1e-3,
            'device': 0,
            'batch_size': 32,
            'num_workers': 0,
            'debug': False
        }
    ),
    Experiment(
        exp_cls=GMNExperimentRethink,
        args={
            'dropout': .5,
            'num_layers': 5,
            'emb_dim': 300,
            'epochs': 100,
            'lr': 1e-3,
            'device': 0,
            'batch_size': 32,
            'num_workers': 0,
            'num_heads': 5,
            'hidden_dim': 64,
            'num_keys': [32, 1],
            'mem_hidden_dim': 16,
            'variant': 'distance',
            'lr_decay_patience': 10,
            'kl_period': 5,
            'early_stop_patience': 50,
            'debug': False
        }
    )
]


if __name__ == '__main__':
    # run experiments sequentially
    for cls, args in experiments:
        cls(**args).run()

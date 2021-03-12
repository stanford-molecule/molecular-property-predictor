from typing import NamedTuple, Type

from experiment import GNNExperiment, GMNExperimentRethink


class Experiment(NamedTuple):
    exp_cls: Type[GNNExperiment]
    args: dict


epochs = 30  # run everything for the same number of epochs so the results are comparable

experiments = [
    # vanilla GCN
    Experiment(
        exp_cls=GNNExperiment,
        args={
            'gnn_type': 'gcn',
            'dropout': .5,
            'num_layers': 5,
            'emb_dim': 300,
            'epochs': epochs,
            'lr': 1e-3,
            'device': 0,
            'batch_size': 32,
            'num_workers': 0,
            'debug': False
        }
    ),
    # vanilla GIN
    Experiment(
        exp_cls=GNNExperiment,
        args={
            'gnn_type': 'gin',
            'dropout': .5,
            'num_layers': 5,
            'emb_dim': 300,
            'epochs': epochs,
            'lr': 1e-3,
            'device': 0,
            'batch_size': 32,
            'num_workers': 0,
            'debug': False
        }
    ),
    # distance GMN
    Experiment(
        exp_cls=GMNExperimentRethink,
        args={
            'dropout': .5,
            'num_layers': 5,
            'emb_dim': 300,
            'epochs': epochs,
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
    ),
    # random GMN
    Experiment(
        exp_cls=GMNExperimentRethink,
        args={
            'dropout': .5,
            'num_layers': 5,
            'emb_dim': 300,
            'epochs': epochs,
            'lr': 1e-3,
            'device': 0,
            'batch_size': 32,
            'num_workers': 0,
            'num_heads': 5,
            'hidden_dim': 64,
            'num_keys': [32, 1],
            'mem_hidden_dim': 16,
            'variant': 'random',
            'lr_decay_patience': 10,
            'kl_period': 5,
            'early_stop_patience': 50,
            'debug': False
        }
    ),
    # vanilla GMN
    Experiment(
        exp_cls=GMNExperimentRethink,
        args={
            'dropout': .5,
            'num_layers': 5,
            'emb_dim': 300,
            'epochs': epochs,
            'lr': 1e-3,
            'device': 0,
            'batch_size': 32,
            'num_workers': 0,
            'num_heads': 5,
            'hidden_dim': 64,
            'num_keys': [32, 1],
            'mem_hidden_dim': 16,
            'variant': 'gmn',
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

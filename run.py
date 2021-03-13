from typing import NamedTuple, Type

from experiment import GNNExperiment, GMNExperimentRethink


class Experiment(NamedTuple):
    exp_cls: Type[GNNExperiment]
    args: dict
    desc: str  # short description of the experiment that'll show up in W&B as the experiment name


epochs = (
    30  # run everything for the same number of epochs so the results are comparable
)
batch_size = 32
debug = False

experiments = [
    Experiment(
        exp_cls=GNNExperiment,
        args={
            "gnn_type": "gcn",
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 300,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": 32,
            "num_workers": 0,
            "debug": debug,
        },
        desc="vanilla GCN",
    ),
    Experiment(
        exp_cls=GNNExperiment,
        args={
            "gnn_type": "gin",
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 300,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": 32,
            "num_workers": 0,
            "debug": debug,
        },
        desc="vanilla GIN",
    ),
    Experiment(
        exp_cls=GMNExperimentRethink,
        args={
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 300,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": 32,
            "num_workers": 0,
            "num_heads": 5,
            "hidden_dim": 64,
            "num_keys": [32, 1],
            "mem_hidden_dim": 16,
            "variant": "distance",
            "lr_decay_patience": 10,
            "kl_period": 5,
            "early_stop_patience": 50,
            "debug": debug,
        },
        desc="distance GMN",
    ),
    Experiment(
        exp_cls=GMNExperimentRethink,
        args={
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 300,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": 32,
            "num_workers": 0,
            "num_heads": 5,
            "hidden_dim": 64,
            "num_keys": [32, 1],
            "mem_hidden_dim": 16,
            "variant": "random",
            "lr_decay_patience": 10,
            "kl_period": 5,
            "early_stop_patience": 50,
            "debug": debug,
        },
        desc="random GMN",
    ),
    Experiment(
        exp_cls=GMNExperimentRethink,
        args={
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 300,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": 32,
            "num_workers": 0,
            "num_heads": 5,
            "hidden_dim": 64,
            "num_keys": [32, 1],
            "mem_hidden_dim": 16,
            "variant": "gmn",
            "lr_decay_patience": 10,
            "kl_period": 5,
            "early_stop_patience": 50,
            "debug": debug,
        },
        desc="vanilla GMN",
    ),
    Experiment(
        exp_cls=GNNExperiment,
        args={
            "gnn_type": "gcn",
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 100,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": batch_size,
            "num_workers": 0,
            "debug": debug,
        },
        desc="vanilla GCN hidden dim=100",
    ),
    Experiment(
        exp_cls=GNNExperiment,
        args={
            "gnn_type": "gcn",
            "dropout": 0.5,
            "num_layers": 3,
            "emb_dim": 300,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": batch_size,
            "num_workers": 0,
            "debug": debug,
        },
        desc="vanilla GCN num_layers=3",
    ),
    Experiment(
        exp_cls=GNNExperiment,
        args={
            "gnn_type": "gcn",
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 500,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": batch_size,
            "num_workers": 0,
            "debug": debug,
        },
        desc="vanilla GCN hidden dim=500",
    ),
]


if __name__ == "__main__":
    # run experiments sequentially
    print(f'going to run {len(experiments)} experiments')
    for cls, args, desc in experiments:
        print(f'running experiment {desc}')
        cls(**args, desc=desc).run()

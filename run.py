"""
Drives all the experiments.

`experiments` is a list of `Experiment` objects that captures all the hyper-params for an experiment.

This file is meant to run all the declared experiments (except for experiments with `skip==True`).
"""

from typing import NamedTuple, Type

from models import (
    GNNBaseline,
    GraphMemoryNetwork,
    GraphNeuralNetwork,
    GNNFLAG,
    DeeperGCN,
)


class Experiment(NamedTuple):
    """
    Model class, arguments, description of the experiment, and whether we should skip running it.
    """

    model_cls: Type[GraphNeuralNetwork]
    args: dict
    desc: str  # short description of the experiment that'll show up in W&B as the experiment name
    skip: bool


# global params
epochs = (
    100  # run everything for the same number of epochs so the results are comparable
)
batch_size = 32
debug = False
# eventually needs to be 5 runs per experiment to compute mean/std
# see https://piazza.com/class/kjjj38qxifm2vx?cid=788
runs = 1

experiments = [
    Experiment(
        model_cls=GNNBaseline,
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
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
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
        skip=True,
    ),
    Experiment(
        model_cls=GraphMemoryNetwork,
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
        skip=True,
    ),
    Experiment(
        model_cls=GraphMemoryNetwork,
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
        skip=True,
    ),
    Experiment(
        model_cls=GraphMemoryNetwork,
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
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
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
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
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
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
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
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
        args={
            "gnn_type": "gin",
            "dropout": 0.5,
            "num_layers": 5,
            "emb_dim": 500,
            "epochs": 100,
            "lr": 1e-3,
            "device": 0,
            "batch_size": batch_size,
            "num_workers": 0,
            "debug": debug,
        },
        desc="gin dim=500 epoch=100",
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
        args={
            "gnn_type": "gin",
            "dropout": 0.5,
            "num_layers": 10,
            "emb_dim": 500,
            "epochs": 100,
            "lr": 1e-3,
            "device": 0,
            "batch_size": batch_size,
            "num_workers": 0,
            "debug": debug,
        },
        desc="gin dim=500 epoch=100 10 layers",
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
        args={
            "gnn_type": "gin",
            "dropout": 0.5,
            "num_layers": 3,
            "emb_dim": 500,
            "epochs": epochs,
            "lr": 1e-3,
            "device": 0,
            "batch_size": batch_size,
            "num_workers": 0,
            "debug": debug,
        },
        desc="gin dim=500 epoch=100 3 layers",
        skip=True,
    ),
    Experiment(
        model_cls=GNNFLAG,
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
            "m": 3,
            "step_size": 1e-3,
        },
        desc="gcn flag",
        skip=True,
    ),
    Experiment(
        model_cls=GNNFLAG,
        args={
            "gnn_type": "gin",
            "dropout": 0.5,
            "num_layers": 3,
            "emb_dim": 300,
            "epochs": 30,
            "lr": 1e-3,
            "device": 0,
            "batch_size": batch_size,
            "num_workers": 0,
            "debug": True,
            "m": 3,
            "step_size": 1e-3,
        },
        desc="gin flag",
        skip=True,
    ),
    Experiment(
        model_cls=GraphMemoryNetwork,
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
            "flag": True,
            "step_size": 1e-3,
            "m": 3,
            "debug": debug,
        },
        desc="GMN flag",
        skip=True,
    ),
    Experiment(
        model_cls=DeeperGCN,
        args={
            "dropout": 0.2,
            "num_layers": 7,
            "emb_dim": 300,
            "epochs": epochs,
            "lr": 1e-2,
            "device": 0,
            "batch_size": 32,
            "num_workers": 0,
            "block": "res+",
            "conv_encode_edge": True,
            "add_virtual_node": False,
            "hidden_channels": 256,
            "conv": "gen",
            "gcn_aggr": "softmax",
            "t": 1.0,
            "learn_t": True,
            "p": 1.0,
            "learn_p": False,
            "y": 0.0,
            "learn_y": False,
            "msg_norm": False,
            "learn_msg_scale": False,
            "norm": "batch",
            "mlp_layers": 1,
            "graph_pooling": "mean",
            "debug": debug,
        },
        desc="vanilla Deeper GCN",
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
        args={
            "gnn_type": "gin-virtual",
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
        desc="vanilla GIN virtual",
        skip=True,
    ),
    Experiment(
        model_cls=GNNBaseline,
        args={
            "gnn_type": "gcn-virtual",
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
        desc="vanilla GCN virtual",
        skip=True,
    ),
    Experiment(
        model_cls=GraphMemoryNetwork,
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
            "hidden_dim": 128,
            "num_keys": [32, 1],
            "mem_hidden_dim": 16,
            "variant": "gmn",
            "lr_decay_patience": 10,
            "kl_period": 5,
            "early_stop_patience": 50,
            "use_deeper": True,
            "block": "res+",
            "conv_encode_edge": True,
            "add_virtual_node": False,
            "conv": "gen",
            "gcn_aggr": "softmax",
            "t": 1.0,
            "learn_t": True,
            "p": 1.0,
            "learn_p": False,
            "y": 0.0,
            "learn_y": False,
            "msg_norm": False,
            "learn_msg_scale": False,
            "norm": "batch",
            "mlp_layers": 1,
            "debug": debug,
        },
        desc="deeper GMN",
        skip=False,
    ),
    Experiment(
        model_cls=GraphMemoryNetwork,
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
            "hidden_dim": 128,
            "num_keys": [32, 1],
            "mem_hidden_dim": 16,
            "variant": "gmn",
            "lr_decay_patience": 10,
            "kl_period": 5,
            "early_stop_patience": 50,
            "use_deeper": True,
            "block": "res+",
            "conv_encode_edge": True,
            "add_virtual_node": False,
            "conv": "gen",
            "gcn_aggr": "softmax",
            "t": 1.0,
            "learn_t": True,
            "p": 1.0,
            "learn_p": False,
            "y": 0.0,
            "learn_y": False,
            "msg_norm": False,
            "learn_msg_scale": False,
            "norm": "batch",
            "mlp_layers": 1,
            "flag": True,
            "step_size": 1e-3,
            "m": 3,
            "debug": debug,
        },
        desc="deeper GMN FLAG",
        skip=False,
    ),
]


if __name__ == "__main__":
    # run experiments sequentially
    experiments_to_run = [exp for exp in experiments if not exp.skip]
    assert len({e.desc for e in experiments}) == len(
        experiments
    ), "make sure there are no duplicate experiment descriptions"
    print(
        f"going to run {len(experiments_to_run)} experiment(s) out of a total of {len(experiments)}"
    )
    for cls, args, desc, _ in experiments_to_run:
        for idx in range(runs):
            print(f"running experiment {desc} run {idx + 1}")
            cls(**args, desc=f"{desc} run={idx + 1}").run()

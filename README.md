## Molecular Property Predictor

## Setup
```bash
conda create -n ogb python=3.7
conda activate ogb

# Install PyTorch
pip install torch

# Install PyTorch Geometric
bash install-pytorch-geometric.sh

# Install the rest of the dependencies
pip install -r requirements.txt
```

## Running experiments
All experiments are listed in `run.py`. You can run all the experiments sequentially with:
```bash
python run.py
``` 

Note that that files contains all the arguments that define an experiment. The experiment results are stored in `results/` as pickle files. Also we're using W&B so if you're logged in the experiment will be streamed to W&B.

For example you can run GMN with DeeperGCN, FLAG, and APPNP using the following:
```python
from models import GraphMemoryNetwork

exp = GraphMemoryNetwork(**{
    "dropout": 0.5,
    "num_layers": 7,
    "emb_dim": 300,
    "epochs": 100,
    "lr": 1e-3,
    "device": 0,
    "batch_size": 32,
    "num_workers": 0,
    "num_heads": 5,
    "hidden_dim": 256,
    "num_keys": [32, 1],
    "mem_hidden_dim": 16,
    "variant": "gmn",
    "lr_decay_patience": 10,
    "kl_period": 5,
    "early_stop_patience": 50,
    "use_deeper": True,
    "block": "res+",
    "conv_encode_edge": True,
    "add_virtual_node": True,
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
    "use_appnp": True,
    "flag": True,
    "step_size": 1e-3,
    "m": 3,
    "k": 10,
    "alpha": 0.1,
    "debug": False,
})
exp.run()
```

see the class docstring for a description of these parameters.

## References
Code is borrowed from or heavily inspired by:
- [GraphMemoryNet](https://github.com/amirkhas/GraphMemoryNet)
- [OGB](http://ogb.stanford.edu/)
- [https://github.com/AaltoPML/Rethinking-pooling-in-GNNs](https://github.com/AaltoPML/Rethinking-pooling-in-GNNs)
- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/)

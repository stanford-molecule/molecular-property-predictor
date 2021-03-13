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

## Analysis
see [analysis.ipynb](analysis.ipynb)

## References
Code is mostly borrowed or heavily inspired by the following:
- [GraphMemoryNet](https://github.com/amirkhas/GraphMemoryNet)
- [OGB](http://ogb.stanford.edu/)
- [https://github.com/AaltoPML/Rethinking-pooling-in-GNNs](https://github.com/AaltoPML/Rethinking-pooling-in-GNNs)
- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/)

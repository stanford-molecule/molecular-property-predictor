## Molecular Property Predictor

## Introduction

This repo was forked from [GraphMemoryNet](https://github.com/amirkhas/GraphMemoryNet).

Edits are made to predict molecule properties.

## Setup

```sh
conda create -n ogb python=3.6
conda activate ogb

# Install PyTorch
pip install torch

# Install PyTorch Geometric
bash install-pytorch-geometric.sh

# Install the rest of the dependencies
pip install -r requirements.txt
```

## Getting Started

If you have CUDA, run

```sh
python train.py --cuda --num_epochs 5 --dataset ogbg-mol-hiv_full
```

If not, run

```sh
python train.py --no-cuda --num_epochs 5 --dataset ogbg-mol-hiv_full
```

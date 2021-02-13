## Molecule Property Predictor

## Introduction

This repo was forked from [GraphMemoryNet](https://github.com/amirkhas/GraphMemoryNet).

Edits are made to predict molecule properties.

## Getting Started

```bash
conda create -n ogb python=3.6
conda activate ogb

# PyTorch Geometric often fails to install from pip if you have a CPU-only device 
# I found this work around works for me.
pip install torch
bash install.sh

# Let's install the rest of the dependencies here
pip install -r ogb-requirements.txt

# Great! Hopefully that worked. Let's see if we can run the model code using the molecular dataset
# Note: we aren't using OGB yet, but it's the same dataset (slightly different format/data loader).
# Note: training over this dataset takes considerably longer than the ENZYMES dataset used above.

# if you have CUDA/GPU
pyhon train.py --cuda --num_epochs 5 --dataset ogbg-mol-hiv_full

# if not
python train.py --no-cuda --num_epochs 5 --dataset ogbg-mol-hiv_full
```

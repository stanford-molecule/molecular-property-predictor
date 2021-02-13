## GraphMemoryNet Implementation

This repo was forked from the official published code base for the GraphMemoryNet. Edits are made to 
make the code work for our use case (and eventually the OGB dataset format).

## Getting Started

Here are some simple instructions to get the virtual environment working and to train the model on 
the default ENZYMES dataset. The code depends on an older version of PyTorch. I'll see if we can
run the code with a newer version so that it will be easier to integrate with OGB / PyTorch Geometric.

-Collin

#### Sanity check the code with original dataset

```bash
conda create -n gmn python=3.6
conda activate gmn
pip install -r requirements.txt

# if you have CUDA/GPU
pyhon train.py --cuda --num_epochs 5

# if not
python train.py --no-cuda --num_epochs 5
```

#### Let's use a more up-to-date version of PyTorch

For our future work using OGB and PyTorch Geometric, we'll probably want a more up-to-date version of PyTorch.
Luckily, it seems like the code might just work using the latest version of PyTorch. To install these dependencies,
follow these instructions.

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
# Note: traning over this dataset takes considerably longer than the ENZYMES dataset used above.

# if you have CUDA/GPU
pyhon train.py --cuda --num_epochs 5 --dataset ogbg-mol-hiv_full

# if not
python train.py --no-cuda --num_epochs 5 --dataset ogbg-mol-hiv_full
```



## Reference

```
@inproceedings{
Khasahmadi2020Memory-Based,
title={Memory-Based Graph Networks},
author={Amir Hosein Khasahmadi and Kaveh Hassani and Parsa Moradi and Leo Lee and Quaid Morris},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=r1laNeBYPB}
}
```

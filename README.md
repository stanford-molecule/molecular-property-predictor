## GraphMemoryNet Implementation

This repo was forked from the official published code base for the GraphMemoryNet. Edits are made to 
make the code work for our use case (and eventually the OGB dataset format).

## Getting Started

```bash
conda create -n gmn python=3.6
conda activate gmn
pip install -r requirements.txt

# if you have CUDA/GPU
pyhon train.py --cuda

# if not
python train.py --no-cuda
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

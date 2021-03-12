#!/bin/bash

TORCH="1.7.0"
CUDA="cpu"

echo "Installing PyTorch Geometric for PyTorch version ${TORCH} and ${CUDA}"

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

#e.g.
#pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
#pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
#pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
#pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
#pip install torch-geometric
#pip install ogb

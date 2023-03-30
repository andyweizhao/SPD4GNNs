# ECML

Implementation for `Modeling Graphs Beyond Hyperbolic: Graph Neural Networks in Symmetric Positive Definite Matrices`

## Available Features 

### Graph Neural Networks

- `GCNConv`: Graph Convolutional Network 

- `GATConv`: Graph Attentional Network

- `SGCConv`: Simplified Graph Convolutional Network

- `GINConv`: Graph Isomorphism Network

- `ChebConv`: Chebyshev-based Graph Convolutional Network

### Manifolds

- `SPD`: The space of symmetric positive definite matrices - a special class of symmetric spaces.

- `Euclidean`: The space of vectors over the real field.

### Datasets 

- `Disease`: Disease Propagation Tree

- `Airport`: Flight Network

- `Pubmed`: Citation Network

- `Citeseer`: Citation Network

- `Cora`: Citation Network


## Usage

Below are the instructions on how to run experiments on SPD and Euclidean spaces respectively.

`` python runner.py --dataset cora --model spdgcn --manifold spd --classifier linear``

`` python runner.py --dataset cora --model gcn --manifold euclidean --classifier linear``

Please note that the choice of datasets, models and manifolds remain open. Once chosen, the pipeline will load the optimal configuration of hyperparameters for the current setup in the 6-dimensional space, which can be found in the `json` folder (identified via grid-search).

## Requirements
- Python == 3.8
- scikit-learn == 1.0.1 
- torch == 1.12.1
- torch-geometric == 2.1.0


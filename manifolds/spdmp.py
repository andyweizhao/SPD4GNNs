import torch
from manifolds import linalg as lalg

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from manifolds import SPDManifold
from abc import ABC

class SPDMessagePassing(ABC):
    def __init__(self, input_dims, args):
        
        self.bias = Parameter(torch.Tensor(1, input_dims, input_dims))

        glorot(self.bias)

    def recaling_sym_matrices(self, x, maxnorm = 10):
        
        x = lalg.sym(x) 
        
        norm = torch.linalg.matrix_norm(x, keepdim=True)

        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def normalizating_SPD(self, x, dims, mode='matrix_norm'):
    
        if mode == 'matrix_norm':
            x = lalg.sym(x)
            x = x / torch.linalg.matrix_norm(x).reshape(-1, 1, 1)
            
        if mode == 'det_norm':
            x = lalg.sym(x)
            dets = torch.linalg.det(x)
            x = torch.sign(x) * x / torch.pow(abs(dets), 1/dims).reshape(-1, 1, 1)
            
        if mode == 'vector_norm':
            num_nodes, n = x.size()[:2]
            x = x.reshape(num_nodes, -1)
            x = F.normalize(x, p=2., dim=-1)
            x = x.reshape(num_nodes, n, n)
        return x
    
    def bias_addition(self, x):
        output = SPDManifold.addition_id(x, SPDManifold.expmap_id(self.bias))
        return output
    
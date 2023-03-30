import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import ChebConv
from manifolds import SPDManifold, SPDMessagePassing, SPDTransform

class SPDChebConv(ChebConv, SPDMessagePassing):

    def __init__(self, input_dims, args):
        
        ChebConv.__init__(self, input_dims, input_dims, K=2, normalization = 'sym', bias=False)
        SPDMessagePassing.__init__(self, input_dims, args)
        
        self.args = args
        
        self.transform1 = SPDTransform.get(args.transform, input_dims)
        
        self.transform2 = SPDTransform.get(args.transform, input_dims)
        
        self.scaling_matrix = Parameter(torch.Tensor(1, input_dims, input_dims))
        
        glorot(self.scaling_matrix)
        
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""

        # input and output are in SPD

        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=mat_feats.dtype, device=mat_feats.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=mat_feats.dtype,
                                      device=mat_feats.device)
        assert lambda_max is not None

        num_nodes, n, _ = mat_feats.size()

        edge_index, norm = self.__norm__(edge_index, num_nodes,
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=mat_feats.dtype,
                                         batch=batch)

        Tx_0 = mat_feats
        Tx_1 = mat_feats  # Dummy.
        
        Tx_0 = self.transform1(Tx_0)
        
        mat_feats = SPDManifold.logmap_id(mat_feats)
        mat_feats = mat_feats.reshape(num_nodes, -1)
            
        Tx_1 = self.propagate(edge_index, x=mat_feats, norm=norm, size=None)
        Tx_1 = Tx_1.reshape(num_nodes, n, n)        
                          
        Tx_1 = SPDManifold.expmap_id(Tx_1)
        out = SPDManifold.addition_id(self.transform2(Tx_1), 
            SPDManifold.expmap_id(self.scaling_matrix * SPDManifold.logmap_id(Tx_0)))                                     
        
        if self.args.has_bias:
            out = self.bias_addition(out)
            
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
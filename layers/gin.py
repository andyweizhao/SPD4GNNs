from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, Size

from torch_geometric.nn import GINConv
from manifolds import SPDManifold, SPDMessagePassing, SPDTransform

class SPDGINConv(GINConv, SPDMessagePassing):

    def __init__(self, input_dims, args):
        
        GINConv.__init__(self, nn=None, eps = 0, train_eps = False)
        SPDMessagePassing.__init__(self, input_dims, args)        
        
        self.eps.data.fill_(self.initial_eps)
        self.args = args
        self.transform = SPDTransform.get(args.transform, input_dims)
        
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                size: Size = None, lambda_max: OptTensor = None, batch: OptTensor = None) -> Tensor:

        """"""
        # input and output are in SPD

        num_nodes, n, _ = mat_feats.size() 
                  
        symmat_feats = SPDManifold.logmap_id(mat_feats)
        symmat_feats = symmat_feats.reshape(num_nodes, -1)
            
        x = (symmat_feats, symmat_feats)        
        
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        
        if x_r is not None:
            out += (1 + self.eps) * x_r        
        
        out = out.reshape(num_nodes, n, n)
        
        out = self.normalizating_SPD(out, self.args.hidden_dims, mode=self.args.spd_norm)
        
        out = SPDManifold.expmap_id(out) 
        
        out = self.transform(out)
        
        if self.args.has_bias:
            out = self.bias_addition(out)
            
        return out
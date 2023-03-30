import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, Size, Optional
from torch_geometric.nn import GATConv
from manifolds import SPDManifold, SPDMessagePassing, SPDTransform
import torch.nn.functional as F
from torch_geometric.utils import softmax

class SPDGATConv(GATConv, SPDMessagePassing):
    def __init__(self, input_dims, args):
        
        GATConv.__init__(self, input_dims, input_dims, heads=1, dropout=0, concat=False, 
                         add_self_loops=False, edge_dim=None, bias=False)
        
        SPDMessagePassing.__init__(self, input_dims, args)
        
        self.args = args
        self.transform = SPDTransform.get(args.transform, input_dims)
                
        self.att_src = Parameter(torch.Tensor(1, self.heads, input_dims * input_dims))
        self.att_dst = Parameter(torch.Tensor(1, self.heads, input_dims * input_dims))        
        
        glorot(self.att_src)
        glorot(self.att_dst)
    
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None) -> Tensor:
        # input and output are in SPD
        
        num_nodes, n, _ = mat_feats.size()
        
        mat_feats = self.transform(mat_feats)
        
        symmat_feats = SPDManifold.logmap_id(mat_feats)        
        
        x_src = x_dst = symmat_feats.reshape(num_nodes, self.heads, -1)      # b x n*n only suppors heads = 1
        x = (x_src, x_dst)
            
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=None, size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)        
                
        out = out.reshape(num_nodes, n, n)
        
        out = SPDManifold.expmap_id(out)         
        
        if self.args.has_bias:
            out = self.bias_addition(out)
            
        return out     

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)  
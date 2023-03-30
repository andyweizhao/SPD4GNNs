from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn import SGConv
from manifolds import SPDManifold, SPDMessagePassing, SPDTransform

class SPDSGConv(SGConv, SPDMessagePassing):

    def __init__(self, input_dims, args):

        SGConv.__init__(self, input_dims, input_dims, K=2, cached = False, add_self_loops=True, bias=False)
        SPDMessagePassing.__init__(self, input_dims, args) 
        self.args = args
        self.transform = SPDTransform.get(args.transform, input_dims)
                
    def forward(self, mat_feats: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        # input and output are in SPD

        num_nodes, n, _ = mat_feats.size()

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes, False,
                    self.add_self_loops, dtype=mat_feats.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes, False,
                    self.add_self_loops, dtype=mat_feats.dtype)

            symmat_feats = SPDManifold.logmap_id(mat_feats)
            symmat_feats = symmat_feats.reshape(num_nodes, -1)
            
            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                symmat_feats = self.propagate(edge_index, x=symmat_feats, edge_weight=edge_weight,
                                   size=None)
                
                
                if self.cached:
                    self._cached_x = symmat_feats
        else:
            symmat_feats = cache.detach()

        symmat_feats = symmat_feats.reshape(num_nodes, n, n)                
        
        mat_feats = SPDManifold.expmap_id(symmat_feats)
        out = self.transform(mat_feats)

        if self.args.has_bias:
            out = self.bias_addition(out)
            
        return out
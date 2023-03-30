from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv
from manifolds import SPDManifold, SPDMessagePassing, SPDTransform
    
class SPDGCNConv(GCNConv, SPDMessagePassing):
    def __init__(self, input_dims, args):

        GCNConv.__init__(self, in_channels=input_dims, out_channels=input_dims, add_self_loops=True, normalize=True, bias=True, improved=False, cached=False)
        SPDMessagePassing.__init__(self, input_dims, args) 
        
        self.transform = SPDTransform.get(args.transform, input_dims)
        self.args = args
        
    def forward(self, mat_feats: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        # input and output are in SPD
        
        num_nodes, n, _ = mat_feats.size()
        edge_index, edge_weight = self.get_edge_index_and_weights(edge_index, edge_weight, num_nodes)

        mat_feats = self.transform(mat_feats)
        
        symmat_feats = SPDManifold.logmap_id(mat_feats)        

        symmat_feats = symmat_feats.reshape(num_nodes, -1)

        out = self.propagate(edge_index, x=symmat_feats, edge_weight=edge_weight, size=None)
                
        out = out.reshape(num_nodes, n, n)        
        
        out = SPDManifold.expmap_id(out) 
        
        if self.args.has_bias:
            out = self.bias_addition(out)
              
        return out

    def get_edge_index_and_weights(self, edge_index, edge_weight, num_nodes):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        return edge_index, edge_weight
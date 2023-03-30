from mappers import Vec2SymMat
from layers import Classifier
from manifolds import linalg as lalg
from layers import SPDGCNConv, SPDGATConv, SPDChebConv, SPDSGConv, SPDGINConv
from layers import SPDNonLinearity
import torch.nn as nn
from manifolds import SPDManifold
from .base import GNNModel

class SPDGNNModel(GNNModel):
    def __init__(self, args):
        super().__init__()

        self.vec2sym = Vec2SymMat.get(args.vec2sym, args.num_node_features, args.hidden_dims)

        self.layer_one = self.get(args.model, args.hidden_dims, args)
        self.layer_two = self.get(args.model, args.hidden_dims, args)

        self.dropout = nn.Dropout(args.dropout)
        
        self.classifier = Classifier.get(args.classifier, args)
        
        self.nonlinear = SPDNonLinearity.get(args.nonlinear)
    
    @classmethod
    def get(cls, type_str: str, input_dims: int, args):
        classes = {
            "spdgcn": SPDGCNConv,
            "spdgat": SPDGATConv,
            "spdcheb": SPDChebConv,
            "spdsgc": SPDSGConv,
            "spdgin": SPDGINConv,
        }
        return classes[type_str](input_dims, args)
    
    def forward(self, node_feats, edge_index):

        mat_feats = self.vec2sym(node_feats)      
        
        mat_feats = lalg.sym(self.dropout(mat_feats))
        
        mat_feats = SPDManifold.expmap_id(mat_feats)        
        
        mat_feats = self.apply_gnn_layers(mat_feats, edge_index)    
        
        outputs = self.classifier(mat_feats)        
        
        return outputs

    def apply_gnn_layers(self, mat_feats, edge_index):
                        
        mat_feats = self.nonlinear(self.layer_one(mat_feats, edge_index))

        mat_feats = self.nonlinear(self.layer_two(mat_feats, edge_index))
        
        return mat_feats
    


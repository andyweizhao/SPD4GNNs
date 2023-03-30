import torch
from torch import nn
from torch_geometric.nn.inits import glorot

from manifolds import linalg as lalg
from torch.nn import Parameter
from manifolds import SPDManifold

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, node_feats):
        log_mats = SPDManifold.logmap_id(node_feats)    
        outputs = self.predict_node_classes(log_mats)    
        return outputs
        
    def predict_node_classes(self, log_mats):
        raise NotImplementedError("predict_node_classes is not implemented in Classifier")
    
    @classmethod
    def get(cls, type_str, args):
        classifs = {
            "linear": LinearClassifier,
            "nc": NCClassifier,
            "svm": SVMClassifier
        }
        return classifs[type_str](args)

class LinearClassifier(Classifier):
    def __init__(self, args):        
        super().__init__()
        self.rows, self.cols = torch.triu_indices(args.hidden_dims, args.hidden_dims, device=args.device)
        self.proj = nn.Linear(int(args.hidden_dims * (args.hidden_dims + 1) / 2), args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def predict_node_classes(self, log_mats):
        node_feats = log_mats[:, self.rows, self.cols]       
        node_feats = self.dropout(node_feats)
        predictions = self.proj(node_feats)
        return predictions    

class NCClassifier(Classifier):

    def __init__(self, args):
        super().__init__()
        
        self.dims = args.hidden_dims
        self.n_centroids = args.n_centroids
                
        self.upper_diag_n = int(self.dims * (self.dims + 1) / 2)         

        self.means = Parameter(torch.Tensor(self.n_centroids, self.upper_diag_n))
        
        self.sigma = Parameter(torch.Tensor(self.n_centroids, self.upper_diag_n, self.upper_diag_n))
        
        self.bias = Parameter(torch.ones(self.n_centroids))
                                
        self.rows, self.cols = torch.triu_indices(self.dims, self.dims, device=args.device)
          
        self.dropout = nn.Dropout(args.dropout)
        
        glorot(self.means)
        glorot(self.sigma)
        
    def predict_node_classes(self, log_mats):
        
        node_feats = log_mats[:, self.rows, self.cols]        
        
        v = node_feats.unsqueeze(1) - self.means 
        
        v = v.unsqueeze(-2)    
        
        distances = (v @ SPDManifold.expmap_id(lalg.sym(self.sigma)) @ v.transpose(-1, -2)).squeeze()
        
        distances = -0.5 * distances + self.bias
        
        return distances   

class SVMClassifier(Classifier):
    def __init__(self, args):
        super().__init__()
    
        self.args = args
        
        self.W = Parameter(torch.Tensor(args.num_classes, args.hidden_dims, args.hidden_dims))
        
        self.dropout = nn.Dropout(args.dropout)
        
        glorot(self.W)
        
    def predict_node_classes(self, log_mats):        
        
        proj = lalg.sym(self.W) @ log_mats.unsqueeze(dim=1)
        
        predictions = self.b_trace(proj)

        center = log_mats.mean(dim=0)
        proj = lalg.sym(self.W) @ center
        
        g_invariant = self.b_trace(proj @ proj)

        return predictions, g_invariant.mean(-1)
    
    def b_trace(self, x):    
        return x.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
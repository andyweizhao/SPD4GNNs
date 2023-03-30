import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, SGConv, GINConv
import torch.nn as nn
from models import GNNModel

class LinearClassifier(nn.Module):
     def __init__(self, args):
         super().__init__()
     
         self.proj = nn.Linear(args.hidden_dims, args.num_classes)
         self.dropout = nn.Dropout(args.dropout)
         
     def forward(self, node_feats): 
         
         node_feats = self.dropout(node_feats)
         predictions = self.proj(node_feats)
         
         return predictions
     
class EuclideanGNNModel(GNNModel):
    def __init__(self, args):
        super().__init__()
        
        self.layer_one, self.layer_two = self.create_conv_layers(args.num_node_features, args.hidden_dims, args.model)
        
        self.classifier = LinearClassifier(args)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def create_conv_layers(self, num_node_features, hidden_dims, model_type):
        if model_type == 'gcn':
            layer_one = GCNConv(num_node_features, hidden_dims)
            layer_two = GCNConv(hidden_dims, hidden_dims)
            
        elif model_type == 'gat':        
            layer_one = GATConv(num_node_features, hidden_dims)
            layer_two = GATConv(hidden_dims, hidden_dims)
            
        elif model_type == 'cheb':   
            layer_one = ChebConv(num_node_features, hidden_dims, K=2)
            layer_two = ChebConv(hidden_dims, hidden_dims, K=2)    

        elif model_type == 'sgc':   
            layer_one = SGConv(num_node_features, hidden_dims, K=1)
            layer_two = SGConv(hidden_dims, hidden_dims, K=1) 
            
        elif model_type == 'gin':  
            layer_one = GINConv(nn=nn.Linear(num_node_features, hidden_dims), eps = 0, train_eps = False)
            layer_two = GINConv(nn=nn.Linear(hidden_dims, hidden_dims), eps = 0, train_eps = False)
        
        else:
            raise ValueError("Invalid model_type provided.")
            
        return layer_one, layer_two
    
    def forward(self, node_feats, edge_index):
        
        node_feats = self.apply_gnn_layers(node_feats, edge_index) 
        
        outputs = self.classifier(node_feats)
        
        return outputs
    
    def apply_gnn_layers(self, node_feats, edge_index):
        
        node_feats = F.relu(self.layer_one(node_feats, edge_index))
        node_feats = F.relu(self.layer_two(node_feats, edge_index))

        return node_feats
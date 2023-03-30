import torch.nn as nn   
   
class GNNModel(nn.Module):
    
   def forward(self, node_feats, edge_index):

       raise NotImplementedError("forward is not implemented in GNNModel")

   def apply_gnn_layers(self, node_feats, edge_index):
                        
       raise NotImplementedError("apply_gnn_layers is not implemented in GNNModel")

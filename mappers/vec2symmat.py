import torch.nn
from torch import nn
from manifolds import linalg as lalg

class Vec2SymMat(nn.Module):
    def __init__(self, input_dims, output_dims):
        """
        :param input_dims: dimension of input vectors: b x input_dims
        :param output_dims: dimension of output symmetric matrices: b x output_dims x output_dims
        :param dimred: string indicating dimensionality reduction method
        :param seed: random seed for dim red
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, node_feats):
        """
        Maps the node features which are Euclidean vectors into Symmetric matrices
        :param node_feats: |V| x input_dim
        :return: |V| x output_dims x output_dims
        """
        raise NotImplementedError("forward is not implemented in Vec2SymMat")        

    @classmethod
    def get(cls, type_str: str, input_dims: int, output_dims: int):
        classes = {
            "squared": SquaredVec2SymMat,
            "triangular": TriangularVec2SymMat
        }
        return classes[type_str](input_dims, output_dims)


class SquaredVec2SymMat(Vec2SymMat):

    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
        self.dimred = torch.nn.Linear(in_features=input_dims, out_features=output_dims * output_dims)                        
        
    def forward(self, node_feats):

        node_feats = self.dimred(node_feats)                
        node_feats = node_feats.reshape(-1, self.output_dims, self.output_dims)
        node_feats = lalg.sym(node_feats)

        return node_feats

class TriangularVec2SymMat(Vec2SymMat):

    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
        
        proj_dims = output_dims * (output_dims + 1) // 2
        
        self.dimred = torch.nn.Linear(in_features=input_dims, out_features=proj_dims) 
       
    def forward(self, node_feats):
        """
        Reduces the dimensionality of the vector to n * (n + 1) / 2 (triangular matrix)
        Builds a triangular symmetric matrix of n x n (upper triangular) U.
        A = U + U^T
        """
        node_feats = self.dimred(node_feats)    # b x out * (out + 1) / 2
        triu_indices = torch.triu_indices(row=self.output_dims, col=self.output_dims)

        mat_feats = torch.zeros((len(node_feats), self.output_dims, self.output_dims),
                                device=node_feats.device, dtype=node_feats.dtype)
        
        mat_feats[:, triu_indices[0], triu_indices[1]] = node_feats
        
        mat_feats[:, triu_indices[1], triu_indices[0]] = node_feats        
        
        return mat_feats
import torch
from torch import nn
from torch.nn import Parameter

from abc import ABC
class SPDTransform(nn.Module, ABC):
    def __init__(self, input_dims):
        super().__init__()
        
        self.dims = input_dims
        self.n_isom = input_dims * (input_dims - 1) // 2

        self.isom_params = Parameter(torch.Tensor(self.dims, self.dims))
        
        self.isom_params.data.copy_(torch.eye(self.dims))

    def forward(self, node_feats):
        raise NotImplementedError("forward is not implemented in SPDTransform")        

    @classmethod
    def get(cls, type_str: str, input_dims: int):
        classes = {
            "qr": SPDIsometryQR,
            "cayley": SPDIsometryCayley
        }
        return classes[type_str](input_dims)    

class SPDIsometryQR(SPDTransform):
    def __init__(self, input_dims):
        
        SPDTransform.__init__(self, input_dims)        
                        
    def forward(self, mat_feats):        
        W = self.isom_params.data
        
        q, r = torch.linalg.qr(W)        
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        W.copy_(q)
        W = W.unsqueeze(0)
        
        mat_feats = W @ mat_feats @ W.transpose(-1, -2)
        
        return mat_feats
        
class SPDIsometryCayley(SPDTransform):
    def __init__(self, input_dims):
        SPDTransform.__init__(self, input_dims)  
        
    def forward(self, mat_feats):
        W = self.isom_params.data
        
        W = W - W.t() # skew-symmetric A^T = -A
        Id = torch.eye(W.size(0)).cuda()

#       See https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv        
        W.copy_(torch.linalg.solve(Id - W, Id + W).t())  # faster than (Id - W) @ torch.linalg.inv(Id + W)
        W = W.unsqueeze(0)        
        
        mat_feats = W @ mat_feats @ W.transpose(-1, -2)
        return mat_feats
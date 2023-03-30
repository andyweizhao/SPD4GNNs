import torch.nn as nn
import torch

class SPDNonLinearity(nn.Module):
    def __init__(self):

        super().__init__()
        self.delta = 0.1

    def forward(self, mat_feats):

        e_x, v_x = torch.linalg.eigh(mat_feats, UPLO='U')
        
        e_x = self.apply_nonlin_to_ev(e_x)
        
        out = v_x @ torch.diag_embed(e_x) @ v_x.transpose(-1, -2)
        
        return out

    def apply_nonlin_to_ev(self, x):
        raise NotImplementedError("apply_nonlin_to_ev is not implemented in Classifier")

    @classmethod
    def get(cls, type_str: str):
        classes = {
            "tgrelu": TgReLU,
            "reeig": ReEig,
        }
        return classes[type_str]()

class TgReLU(SPDNonLinearity):
    def __init__(self):        
        SPDNonLinearity.__init__(self)  
    
    def apply_nonlin_to_ev(self, x):
        dims = x.size(-1)
        x = torch.log(x)
        step_noise = self.delta * torch.arange(start=0, end=dims, step=1,
                                      dtype=x.dtype, device=x.device)            
        step_noise = step_noise.expand_as(x)            
        x = torch.where(x > 0, x, step_noise)        
        x = torch.exp(x)
        return x

class ReEig(SPDNonLinearity):
    def __init__(self):        
        SPDNonLinearity.__init__(self)

    def apply_nonlin_to_ev(self, x):            
        dims = x.size(-1)
        x = torch.clamp(x, min=0.5)    
        step_noise = self.delta * torch.arange(start=0, end=dims, step=1,
                                      dtype=x.dtype, device=x.device)                    
        x = x + step_noise
        return x

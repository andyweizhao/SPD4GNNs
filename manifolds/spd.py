from typing import Tuple
import torch
from manifolds import linalg as lalg

class SPDManifold:
    def __init__(self):
        super().__init__()

    @staticmethod
    def dist(self, a: torch.Tensor, b: torch.Tensor, keepdim=True) -> Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError("dist is not implemented in SPDManifold")        

    @staticmethod
    def expmap_id(x: torch.Tensor) -> torch.Tensor:
        """
        Performs an exponential map using the Identity as basepoint :math:`\operatorname{Exp}_{Id}(u)`.
        """
        x = lalg.sym(x)
        return lalg.sym_funcm(x, torch.exp)
        
    @staticmethod
    def logmap_id(x: torch.Tensor) -> torch.Tensor:
        """
        Perform an logarithmic map using the Identity as basepoint :math:`\operatorname{Log}_{Id}(y)`.
        """
        x = lalg.sym(x)
        return lalg.sym_funcm(x, torch.log)

    @staticmethod
    def addition_id(a: torch.Tensor, b: torch.Tensor):
        """
        Performs addition using the Identity as basepoint.

        The addition on SPD using the identity as basepoint is :math:`A \oplus_{Id} B = sqrt(A) B sqrt(A)`.
        """
        sqrt_a = lalg.sym_sqrtm(a)
        return sqrt_a @ b @ sqrt_a

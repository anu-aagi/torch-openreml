from torch_openreml.covariance.matrix import Matrix
import torch

class IdentityMatrix(Matrix):
    
    def __init__(self, n):
        super().__init__((n, n), [])
    
    def build(self, params, grad=True):
        if isinstance(params, dict):
            params = next(iter(params))
        return torch.eye(self.shape[0], device=params.device, dtype=params.dtype)
    
    @property
    def repr_dict(self):
        return {"shape": self.shape}

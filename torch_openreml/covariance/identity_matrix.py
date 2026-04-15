from torch_openreml.covariance.covariance_matrix import CovarianceMatrix
import torch

class IdentityMatrix(CovarianceMatrix):
    
    def __init__(self, n):
        super().__init__(n, [])
    
    def build(self, params, grad=True):
        if isinstance(params, dict):
            params = next(iter(params))
        return torch.eye(self.n, device=params.device, dtype=params.dtype)

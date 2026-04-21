from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformSqrtLog
import torch

class ScalarMatrix(Matrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__((n, n), ["sigma^2"], [TransformSqrtLog()], no_grad_index)
        
    def _compute_grad(self, v):
        self.reset_grad()
        if len(self.no_grad_index) == 0:
            self._grad = (2 * v).unsqueeze(0)
            self._grad_names = self.param_names
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        
        sigma2 = torch.exp(2 * params[0])
        v = sigma2 * torch.eye(self.shape[0], device=device, dtype=dtype)
        
        if grad:
            self._compute_grad(v=v)
        
        return v

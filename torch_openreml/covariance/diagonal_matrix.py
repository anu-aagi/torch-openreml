from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2
import torch

class DiagonalMatrix(Matrix):
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        param_names = param_names or [f"sigma^2_{i}" for i in range(n)]
        trans = trans or [TransformExpPow2()]
        super().__init__((n, n), param_names, trans, no_grad_index)
        
    def _compute_grad(self, params, device, dtype):
        self.reset_grad()
        
        if len(self.no_grad_index) == self.num_params:
            return
        
        self._grad = torch.zeros(self.shape[0], self.shape[0], self.shape[0], device=device, dtype=dtype)
        idx = torch.arange(self.shape[0], device=device)
        self._grad[idx, idx, idx] = self.trans_chain_rule_factor(params)
        
        mask = torch.ones(self.shape[0], dtype=torch.bool, device=device)
        mask[self.no_grad_index] = False
        
        self._grad = self._grad[mask]
        self._grad_names = [name for i, name in enumerate(self.param_names) if i not in self.no_grad_index]
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        sigma2 = self.trans_params(params)
        
        if grad:
            self._compute_grad(params=params, device=device, dtype=dtype)
        
        return torch.diag(sigma2)

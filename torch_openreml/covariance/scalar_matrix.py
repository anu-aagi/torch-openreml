from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2
import torch

class ScalarMatrix(Matrix):
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        param_names = param_names or ["sigma^2"]
        trans = trans or [TransformExpPow2()]
        super().__init__((n, n), param_names, trans, no_grad_index)
        
    def _compute_grad(self, params, i_n):
        self.reset_grad()
        if len(self.no_grad_index) == 0:
            self._grad = (self.trans_chain_rule_factor(params[0]) * i_n).unsqueeze(0)
            self._grad_names = self.param_names
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        sigma2 = self.trans_params(params)[0]

        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        v = sigma2 * i_n
        
        if grad:
            self._compute_grad(params=params, i_n=i_n)
        
        return v

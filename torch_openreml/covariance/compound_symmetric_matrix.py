from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformChain, TransformScaleShift, TransformSigmoid
import torch

class CompoundSymmetricMatrix(Matrix):
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        self.rho_min = -1/(n - 1)
        param_names = param_names or ["sigma^2", "rho"]
        trans = trans or [
            TransformExpPow2(),
            TransformChain([TransformSigmoid(), TransformScaleShift((1 - self.rho_min), self.rho_min)])
        ]
        super().__init__((n, n), param_names, trans, no_grad_index)
      
    def _compute_grad(self, params, sigma2, rho_mat, i_n, j_n):
        self.reset_grad()
        
        if len(self.no_grad_index) == self.num_params:
            return
        
        grad = []

        chain_rule_factor = self.trans_chain_rule_factor(params)

        if 0 not in self.no_grad_index:
            grad.append(chain_rule_factor[0] * rho_mat)
            self._grad_names.append(self.param_names[0])
        
        if 1 not in self.no_grad_index:
            grad.append(sigma2 * (j_n - i_n) * chain_rule_factor[1])
            self._grad_names.append(self.param_names[1])
        
        self._grad = torch.stack(grad)
      
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        sigma2, rho = self.trans_params(params)
        
        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        j_n = torch.ones((self.shape[0], self.shape[0]), device=device, dtype=dtype)

        rho_mat = ((1 - rho) * i_n + rho * j_n)
        v = sigma2 * rho_mat
        
        if grad:
            self._compute_grad(params=params, sigma2=sigma2, rho_mat=rho_mat, i_n=i_n, j_n=j_n)
            
        return v

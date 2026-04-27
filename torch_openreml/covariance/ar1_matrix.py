from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformChain, TransformScaleShift, TransformSigmoid
import torch

class AR1Matrix(Matrix):
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        param_names = param_names or ["sigma^2", "rho"]
        trans = trans or [TransformExpPow2(), TransformChain([TransformSigmoid(), TransformScaleShift(2.0, -1.0)])]
        super().__init__((n, n), param_names, trans, no_grad_index)
      
    def _compute_grad(self, params, rho, sigma2, diff, rho_power):
        self.reset_grad()
        
        if len(self.no_grad_index) == self.num_params:
            return
        
        grad = []

        chain_rule_factor = self.trans_chain_rule_factor(params)

        if 0 not in self.no_grad_index:
            grad.append(chain_rule_factor[0] * rho_power)
            self._grad_names.append(self.param_names[0])
        
        if 1 not in self.no_grad_index:
            rho = torch.sign(rho) * torch.clamp(rho.abs(), min=1e-6)
            d_rho = chain_rule_factor[1] * sigma2 * diff * rho_power / rho
            d_rho.fill_diagonal_(0.0)
            grad.append(d_rho)
            self._grad_names.append(self.param_names[1])
            
        self._grad = torch.stack(grad)
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        sigma2, rho = self.trans_params(params)
      
        idx = torch.arange(self.shape[0], device=params.device, dtype=params.dtype)
        diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        rho_power = rho ** diff
        
        v = sigma2 * rho_power
        
        if grad:
            self._compute_grad(params=params, rho=rho, sigma2=sigma2, diff=diff, rho_power=rho_power)
            
        return v

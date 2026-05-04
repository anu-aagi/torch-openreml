from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2
import torch

class ScalarMatrix(Matrix):
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        param_names = param_names or ["sigma^2"]
        trans = trans or [TransformExpPow2()]
        super().__init__((n, n), param_names, trans, no_grad_index)
        
    def __call__(self, params):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        sigma2 = self.trans_params(params)

        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        v = sigma2 * i_n
        
        return v

    def manual_grad(self, params):
        if len(self.no_grad_index) > 0:
            return None, []

        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)

        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        grad = (self.trans_grad(params) * i_n).unsqueeze(0)
        grad_names = self.param_names

        return grad, grad_names

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2
import torch

class DiagonalMatrix(Matrix):
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        param_names = param_names or [f"sigma^2_{i}" for i in range(n)]
        trans = trans or [TransformExpPow2()]
        super().__init__((n, n), param_names, trans, no_grad_index)

    def __call__(self, params):
        sigma2 = self.trans_params(params)
        
        return torch.diag(sigma2)

    def manual_grad(self, params):
        if len(self.no_grad_index) == self.num_params:
            return None, []

        device, dtype = self.check_params(params)

        grad = torch.zeros(self.shape[0], self.shape[0], self.shape[0], device=device, dtype=dtype)
        idx = torch.arange(self.shape[0], device=device)
        grad[idx, idx, idx] = self.trans_grad(params)

        mask = torch.ones(self.shape[0], dtype=torch.bool, device=device)
        mask[self.no_grad_index] = False

        grad = grad[mask]
        grad_names = [name for i, name in enumerate(self.param_names) if i not in self.no_grad_index]

        return grad, grad_names


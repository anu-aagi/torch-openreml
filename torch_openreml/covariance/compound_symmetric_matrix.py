from torch_openreml.covariance.matrix import Matrix
import torch

class CompoundSymmetricMatrix(Matrix):
  
    def __init__(self, n, no_grad_index=None):
        self.rho_min = -1/(n - 1)
        super().__init__((n, n), ["log_sigma", "scaled_rho"], no_grad_index)
      
    def trans_rho(self, scaled_rho):
        return self.rho_min + (1 - self.rho_min) * torch.sigmoid(scaled_rho)
      
    def _compute_grad(self, v, sigma2, sigmoid, i_n, j_n):
        self.reset_grad()
        
        if len(self.no_grad_index) == self.num_params:
            return
        
        grad = []
        
        if 0 not in self.no_grad_index:
            grad.append(2 * v)
            self._grad_names.append("log_sigma")
        
        if 1 not in self.no_grad_index:
            d_rho = (1 - self.rho_min) * sigmoid * (1 - sigmoid)
            grad.append(sigma2 * d_rho * (j_n - i_n))
            self._grad_names.append("scaled_rho")
        
        self._grad = torch.stack(grad)
      
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        
        sigma2 = torch.exp(2 * params[0])
        sigmoid = torch.sigmoid(params[1])
        rho = self.rho_min + (1 - self.rho_min) * sigmoid
        
        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        j_n = torch.ones((self.shape[0], self.shape[0]), device=device, dtype=dtype)
        
        v = sigma2 * ((1 - rho) * i_n + rho * j_n)
        
        if grad:
            self._compute_grad(v=v, sigma2=sigma2, sigmoid=sigmoid, i_n=i_n, j_n=j_n)
            
        return v

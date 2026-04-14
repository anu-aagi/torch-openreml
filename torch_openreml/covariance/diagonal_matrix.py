from torch_openreml.covariance.covariance_matrix import CovarianceMatrix

class DiagonalMatrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__(n, [f"log_sigma_{i}" for i in range(n)], no_grad_index)
        
    def _compute_grad(self, sigma2, device, dtype):
        self.reset_grad()
            
        for i in range(self.num_params):
            if i in self.no_grad_index:
                continue
            e = torch.zeros((self.n, self.n), device=device, dtype=dtype)
            e[i, i] = 2 * sigma2[i]
            self._grad.append(e)
            self._grad_names.append(self.param_names[i])
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        
        sigma2 = torch.exp(2 * params)
        
        if grad:
            self._compute_grad(sigma2=sigma2, device=device, dtype=dtype)
        
        return torch.diag(sigma2)

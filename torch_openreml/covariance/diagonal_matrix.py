from torch_openreml.covariance.covariance_matrix import CovarianceMatrix

class DiagonalMatrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__(n, [f"log_sigma_{i}" for i in range(n)], no_grad_index)
        
    def _compute_grad(self, sigma2, device, dtype):
        self.reset_grad()
        
        if len(self.no_grad_index) == self.num_params:
            return
        
        self._grad = torch.zeros(self.n, self.n, self.n, device=device, dtype=dtype)
        idx = torch.arange(self.n, device=device)
        self._grad[idx, idx, idx] = 2 * sigma2
        
        mask = torch.ones(self.n, dtype=torch.bool, device=device)
        mask[self.no_grad_index] = False
        
        self._grad = self._grad[mask]
        self._grad_names = [name for i, name in enumerate(self.param_names) if i not in self.no_grad_index]
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        
        sigma2 = torch.exp(2 * params)
        
        if grad:
            self._compute_grad(sigma2=sigma2, device=device, dtype=dtype)
        
        return torch.diag(sigma2)

from torch_openreml.covariance.covariance_matrix import CovarianceMatrix

class ScalarMatrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__(n, ["log_sigma"], no_grad_index)
        
    def _comptue_grad(self, v):
        self.reset_grad()
        if len(self.no_grad_index) == 0:
            self._grad = (2 * v).unsqueeze(0)
            self._grad_names = self.param_names
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        
        sigma2 = torch.exp(2 * params[0])
        v = sigma2 * torch.eye(self.n, device=device, dtype=dtype)
        
        if grad:
            self._compute_grad(v=v)
        
        return v

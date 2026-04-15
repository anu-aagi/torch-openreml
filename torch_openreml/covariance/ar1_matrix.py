from torch_openreml.covariance.covariance_matrix import CovarianceMatrix

class AR1Matrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__(n, ["log_sigma", "scaled_rho"], no_grad_index)
    
    def trans_rho(self, scaled_rho):
        return 2 * torch.sigmoid(scaled_rho) - 1
      
    def _compute_grad(self, sigma2, sigmoid, v, diff, rho_power):
        self.reset_grad()
        
        if len(self.no_grad_index) == self.num_params:
            return
        
        grad = []
        
        if 0 not in self.no_grad_index:
            grad.append(2 * v)
            self._grad_names.append("log_sigma")  
        
        if 1 not in self.no_grad_index:
            rho = torch.sign(rho) * torch.clamp(rho.abs(), min=1e-6)
            grad.append(sigma2 * 2 * sigmoid * (1 - sigmoid) * diff * rho_power / rho)
            self._grad_names.append("scaled_rho") 
            
        self._grad = torch.stack(grad)
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
      
        idx = torch.arange(self.n, device=params.device, dtype=params.dtype)
        diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        sigma2 = torch.exp(2 * params[0])
        sigmoid = torch.sigmoid(params[1])
        rho = 2 * sigmoid - 1
        
        rho_power = rho ** diff
        
        v = sigma2 * rho_power
        
        if grad:
            self._compute_grad(sigma2=sigma2, sigmoid=sigmoid, v=v, diff=diff, rho_power=rho_power)
            
        return v

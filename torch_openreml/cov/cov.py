import torch
from torch_openreml.utils import get_device, get_dtype
from abc import ABC, abstarctmethod

# Abstract Covariance Matrix Class ----------------------------------------

class CovarianceMatrix(ABC):
  
    def __init__(self, n, param_names, no_grad_index=None):
        self.check_n(n)
        self._n = n
        
        self.reset_grad()
        self.check_no_grad_index(no_grad_index)
        
        self.check_param_names(param_names)
        self._param_names = param_names
        self._num_params = len(param_names)
        self._no_grad_index = no_grad_index or []
    
    def reset_grad(self):
        self._grad = []
        self._grad_names = []
    
    def set_no_grad(index=None, param_name=None):
        if (index is None) == (param_name is None):
            raise RuntimeError("Provide exactly one of 'index' or 'param_name'!")
        
        if (param_name is None):
            if (not isinstance(index, list)):
                index = [index]
            self.check_no_grad_index(index)
            self._no_grad_index.append(index)
            self._no_grad_index = set(self._no_grad_index)
        
        if (index is None):
            if (not isinstance(param_name, list)):
                param_names = [param_name]
            index_map = {name: i for i, name in enumerate(self._param_names)}
            positions = [index_map[name] for n in param_names]
            self._no_grad_index.append(positions)
            self._no_grad_index = set(self._no_grad_index)

    @abstractmethod
    def build(self, params, grad=True):
        raise NotImplementedError
    
    def check_n(self, n):
        if not isinstance(n, int):
            raise TypeError("'n' must be an int!")
      
    def check_no_grad_index(self, no_grad_index):
        if no_grad_index is not None:
            if any(idx < 0 or idx >= self._num_params for idx in no_grad_index):
                raise RuntimeError("Parameter index outside range!")
              
    def check_params(self, params):
        if not torch.is_tensor(params):
            raise TypeError("Parameters must be a Torch tensor!")
    
        if params.dim() != 1:
            raise RuntimeError("Parameters must be a 1D tensor!")
    
        if params.shape[0] != self._num_params:
            raise RuntimeError(f"Parameters must have length {self.num_params}, got {params.shape[0]}!")
    
    def check_param_names(self, param_names):
        if len(param_names) != len(set(param_names)):
            raise RuntimeError(f"Parameter names must be unique!")
        
    @property(self):
    def n(self):
        return self._n

    @property
    def grad(self):
        return self._grad
    
    @property
    def grad_names(self):
        return self._grad_names
      
    @property  
    def param_names(self):
        return self._param_names
    
    @property
    def num_params(self):
        return self._num_params
    
    @property
    def no_grad_index(self):
        return self._no_grad_index

# Indentity Matrix Class --------------------------------------------------
      
class IdentityMatrix(CovarianceMatrix):
    
    def __init__(self, n, device, dtype):
        super().__init__(n, {})
        self.device = device
        self.dtype = dtype
    
    def build(self, *args, **kwargs):
        return torch.eye(self.n, device=self.device, dtype=self.dtype)
      
# Scalar Matrix Class -----------------------------------------------------      
      
class ScalarMatrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__(n, ["log_sigma"], no_grad_index)
        
    def _comptue_grad(self, v):
        self.reset_grad()
        if len(self.no_grad_index) == 0:
            self._grad = [2 * v]
            self._grad_names = self.param_names
        
    def build(self, params, grad=True):
        self.check_params(params)
        v = torch.exp(2 * params[0]) * torch.eye(self.n, device=params.device, dtype=params.dtype)
        
        if grad:
            self._compute_grad(v=v)
        
        return v

# Diagnoal Matrix Class ---------------------------------------------------

class DiagonalMatrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__(n, [f"log_sigma_{i}" for i in range(n)], no_grad_index)
        
    def _compute_grad(self, sigma2, params):
        self.reset_grad()
            
        for i in range(self.num_params):
            if i in self.no_grad_index:
                continue
            E = torch.zeros((self.n, self.n), device=params.device, dtype=params.dtype)
            E[i, i] = 2 * sigma2[i]
            self._grad.append(E)
            self._grad_names.append(self.param_names[i])
        
    def build(self, params, grad=True):
        self.check_params(params)
        sigma2 = torch.exp(2 * params)
        
        if grad:
            self._compute_grad(sigma2=sigma2, params=params)
        
        return torch.diag(sigma2)
      
# Compound Symmetric Matrix Class -----------------------------------------     

class CompoundSymmetricMatrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        self.rho_min = -1/(n - 1)
        super().__init__(n, ["log_sigma", "scaled_rho"], no_grad_index)
      
    def trans_rho(self, scaled_rho):
        return self.rho_min + (1 - self.rho_min) * torch.sigmoid(scaled_rho)
      
    def _compute_grad(self, v, sigma2, sigmoid, J, I):
        self.reset_grad()
            
        if 0 not in self.no_grad_index:
            self._grad.append(2 * v)
            self._grad_names.append("log_sigma")
        
        if 1 not in self.no_grad_index:
            d_rho = (1 - self.rho_min) * sigmoid * (1 - sigmoid)
            self._grad.append(sigma2 * d_rho * (J - I))
            self._grad_names.append("scaled_rho")
      
    def build(self, params, grad=True):
        self.check_params(params)
        
        sigma2 = torch.exp(2 * params[0])
        sigmoid = torch.sigmoid(params[1])
        rho = self.rho_min + (1 - self.rho_min) * sigmoid
        
        I = torch.eye(self.n, device=params.device, dtype=params.dtype)
        J = torch.ones((self.n, self.n), device=params.device, dtype=params.dtype)
        
        cor = (1 - rho) * I + rho * J
        v = sigma2 * cor
        
        if grad:
            self._compute_grad(v=v, sigma2=sigma2, sigmoid=sigmoid, J=J, I=I)
            
        return v

# AR1 Matrix Class --------------------------------------------------------

class AR1Matrix(CovarianceMatrix):
  
    def __init__(self, n, no_grad_index=None):
        super().__init__(n, ["log_sigma", "scaled_rho"], no_grad_index)
    
    def trans_rho(self, scaled_rho):
        return 2 * torch.sigmoid(scaled_rho) - 1
        
    def build(self, params, grad=True):
        self.check_params(params)
      
        idx = torch.arange(self.n, device=params.device, dtype=params.dtype)
        diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        sigma2 = torch.exp(2 * params[0])
        sigmoid = torch.sigmoid(params[1])
        rho = 2 * sigmoid - 1
        
        rho_power = rho ** diff
        
        v = sigma2 * rho_power
        
        if grad:
            self.reset_grad()
            
            if 0 not in self.no_grad_index:
                self._grad.append(2 * v)
                self._grad_names.append("log_sigma")
            
            if 1 not in self.no_grad_index:
                rho = rho.clamp(min=1e-6, max=1-1e-6)
                self._grad.append(2 * sigmoid * (1 - sigmoid) * sigma2 * diff * rho_power / rho)
                self._grad_names.append("scaled_rho")
            
        return v

# Unstructured Matrix Class -----------------------------------------------

class UnstructuredMatrix(CovarianceMatrix):
  
    def __init__(self, n, entries):
        if torch.is_tensor(n):
            self.n = n.item()
        else:
            self.n = n
            
        self.entries = entries
        self.device = entries.device
        self.dtype = entries.dtype
    
    def build(self, grad=True):
      
        if grad:
            self._grad = []
            for k in range(self.entries.shape[0]):
                i = k // n
                j = k % n
                E = torch.zeros(n, n)
                E[i, j] = 1.0
                self._grad.append(E)

        return self.entries.reshape(n, n)

class KroneckerProductMatrix(CovarianceMatrix):
    
    def __init__(self, A, B, theta_A, theta_B):
        self.A = A
        self.B = B
        self.theta_A = theta_A
        self.theta_B = theta_B
        self.device = get_device(A, B, theta_A, theta_B)
        self.dtype = get_dtype(A, B, theta_A, theta_B)
    
    def build(self, grad=True):
        return torch.kron(A, B)

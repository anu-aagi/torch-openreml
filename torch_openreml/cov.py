import torch
from torch_openreml.utils import get_device, get_dtype
from abc import ABC, abstarctmethod

class CovMatrix(ABC):

    @abstractmethod
    def build(self, grad=True):
        raise NotImplementedError

    @property
    def grad(self):
        if not hasattr(self, "_grad"):
            raise RuntimeError(f"Call `build()` with grad=True first.")
        return self._grad
      
class ScalarMatrix(CovMatrix):
  
    def __init__(self, n, sigma2):
        self.n = n
        self.sigma2 = sigma2
        self.device = self.sigma2.device
        self.dtype = self.sigma2.dtype
        
    def build(self, grad=True):
        if grad:
            self._grad = [torch.eye(self.n, device=self.device, dtype=self.dtype)]
        return self.sigma2 * torch.eye(self.n, device=self.device, dtype=self.dtype)

class DiagonalMatrix(CovMatrix):
  
    def __init__(self, sigma2):
        self.sigma2 = sigma2
        self.n = sigma2.shape[0]
        self.device = self.sigma2.device
        self.dtype = self.sigma2.dtype
        
    def build(self, grad=True):
        if grad:
            m = []
            for i in range(self.n):
                E = torch.zeros((self.n, self.n), device=self.device, dtype=self.dtype)
                E[i, i] = 1.0
                m.append(E)
            self._grad = m
        return torch.diag(self.sigma2)

class CompoundSymmetricMatrix(CovMatrix):
  
    def __init__(self, n, sigma2, rho):
        self.n = n
        self.sigma2 = sigma2
        self.rho = rho
        self.device = get_device(sigma2, rho)
        self.dtype = get_dtype(sigma2, rho)
        
    def build(self, grad=True):
        I = scalar_matrix(self.n, self.sigma2)
        J = torch.ones((self.n, self.n), device=self.device, dtype=self.dtype)
        cor = (1 - rho) * I + rho * J
        
        if grad:
            d_sigma2 = cor
            d_rho = self.sigma2 * (J - I)
            self._grad = [d_sigma2, d_rho]
            
        return self.sigma2 * cor

class AR1Matrix(CovMatrix):
  
    def __init__(self, n, sigma2, rho):
        self.n = n
        self.sigma2 = sigma2
        self.rho = rho
        self.device = get_device(sigma2, rho)
        self.dtype = get_dtype(sigma2, rho)
        
    def build(self, grad=True):
        idx = torch.arange(self.n, device=self.device, dtype=self.dtype)
        diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        rho_power = self.rho ** diff
        
        if grad:
            d_sigma2 = rho_power
            d_rho = self.sigma2 * diff * rho_power / (rho + 1e-6)

            self._grad = [d_sigma2, d_rho]
            
        return self.sigma2 * rho_power
      
class UnstructuredMatrix(CovMatrix):
  
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
        

class KroneckerProductMatrix(CovMatrix):
    
    def __init__(self, A, B, theta_A, theta_B):
        self.A = A
        self.B = B
        self.theta_A = theta_A
        self.theta_B = theta_B
        self.device = get_device(A, B, theta_A, theta_B)
        self.dtype = get_dtype(A, B, theta_A, theta_B)
    
    def build(self, grad=True):
        pass

import torch
from torch_openreml.utils import get_device, get_dtype
from abc import ABC, abstarctmethod

class CovMatrix(ABC):

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def grad(self):
        if not hasattr(self, "_grad"):
            raise RuntimeError(f"Call with grad=True first.")
        return self._grad
      
class ScalarMatrix(CovMatrix):
    def build(n, sigma2, grad=True):
        if grad:
            self._grad = [torch.eye(n, device=sigma2.device, dtype=sigma2.dtype)]
        return sigma2 * torch.eye(n, device=sigma2.device, dtype=sigma2.dtype)

class DiagMatrix(CovMatrix):
    def build(sigma2, grad=True):
        if grad:
            m = []
            for i in range(sigma2.shape[0]):
                E = torch.zeros((n, n), device=sigma2.device, dtype=sigma2.dtype)
                E[i, i] = 1.0
                m.append(E)
            self._grad = m
        return torch.diag(sigma2)

class CompoundSymmetricMatrix(CovMatrix):
    def build(n, sigma2, rho, grad=True):
        device = get_device(sigma2, rho)
        dtype = get_dtype(sigma2, rho)
        I = scalar_matrix(n, sigma2)
        J = torch.ones((n, n), device=device, dtype=dtype)
        cor = (1 - rho) * I + rho * J
        
        if grad:
            d_sigma2 = cor
            d_rho = sigma2 * (J - I)
            self._grad = [d_sigma2, d_rho]
            
        return sigma2 * cor

class AR1Matrix(CovMatrix):
    def build(n, sigma2, rho, grad=True):
        device = get_device(sigma2, rho)
        dtype = get_dtype(sigma2, rho)
        idx = torch.arange(n, device=device,dtype=dtype)
        diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        rho_power = rho ** diff
        
        if grad:
            d_sigma2 = rho_power
            d_rho = sigma2 * diff * rho_power / (rho + 1e-6)

            self._grad = [d_sigma2, d_rho]
            
        return sigma2 * rho_power
      
class BlockDiagMatrix(CovMatrix):
    def build(blocks, grad=True):
        torch.block_diag(*blocks)

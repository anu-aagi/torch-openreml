from torch_openreml.covariance.covariance_matrix import CovarianceMatrix
from torch_openreml.covariance.operator import Operator
import torch

class LinearPropagation(Operator):
    
    def __init__(self, operands):
        if (len(operands) != 2):
            raise ValueError("Two operands are required")
    
        n = list(operands.values())[1].n
          
        super().__init__(n, operands)
    
    def build(self, params, grad=True):
        v_groups, grad_groups, grad_name_groups = self.build_operands(params, grad)
        
        z = v_groups[0]
        g = v_groups[1]
        v = z @ g @ z.T
        
        if grad:
            self.reset_grad()
            
            grad = []
            dz = grad_groups[0]
            dg = grad_groups[1]
            
            if dz is not None:
                dz = dz @ g @ z.T + z @ g @ dz.mT
                grad.append(dz)
                self._grad_names.extend(grad_name_groups[0])
                
            if dg is not None:
                dg = z @ dg @ z.T
                grad.append(dg)
                self._grad_names.extend(grad_name_groups[1])
                
            if len(grad) > 0:
                self._grad = torch.cat(grad)
                
        return v

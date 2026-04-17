from torch_openreml.covariance.operator import Operator
import torch

class Sum(Operator):
    
    def __init__(self, operands):
        if (len(operands) < 2):
            raise ValueError("At least two operands are required")
          
        super().__init__(None, operands)
    
    def build(self, params, grad=True):
        v_groups, grad_groups, grad_name_groups = self.build_operands(params, grad)
        
        v = sum(v_groups)
        if grad:
            self.reset_grad()
            
            grad_groups = [grad for grad in grad_groups if grad is not None]
            
            if len(grad_groups) > 0:
                self._grad = torch.cat(grad_groups)
                self._grad_names = [name for group in grad_name_groups for name in group]
        
        self._shape = tuple(v.shape)
        return v

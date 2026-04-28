from torch_openreml.covariance.operator import Operator
import torch

class KroneckerProduct(Operator):
    
    def __init__(self, operands):
        if len(operands) != 2:
            raise ValueError("Two operands are required")
          
        super().__init__(None, operands)
    
    def build(self, params, grad=True):
        v_groups, grad_groups, grad_name_groups = self.build_operands(params, grad)
        
        a = v_groups[0]
        b = v_groups[1]
        v = torch.kron(a, b)
        
        if grad:
            self.reset_grad()
            
            grad = []
            da = grad_groups[0]
            db = grad_groups[1]
            
            if da is not None:
                da = torch.kron(da, b)
                grad.append(da)
                self._grad_names.extend(grad_name_groups[0])
                
            if db is not None:
                db = torch.kron(a, db)
                grad.append(db)
                self._grad_names.extend(grad_name_groups[1])
                
            if len(grad) > 0:
                self._grad = torch.cat(grad)
        
        self._shape = tuple(v.shape)
        return v

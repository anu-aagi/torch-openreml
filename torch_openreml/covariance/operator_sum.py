from torch_openreml.covariance.operator import Operator
import torch

class Sum(Operator):
    
    def __init__(self, *args, **kwargs):
          
        super().__init__(*args, **kwargs)

        if len(self.operands) < 2:
            raise ValueError("At least two operands are required")
    
    def __call__(self, params):
        v_groups = self.build_operands(params)
        v = sum(v_groups)
        self._shape = tuple(v.shape)

        return v

    def manual_grad(self, params):
        grad_groups, grad_name_groups = self.operands_grad(params)

        grad_groups = [grad for grad in grad_groups if grad is not None]

        if len(grad_groups) > 0:
            grad = torch.cat(grad_groups)
            grad_names = [name for group in grad_name_groups for name in group]

            return grad, grad_names
        else:
            return None, []
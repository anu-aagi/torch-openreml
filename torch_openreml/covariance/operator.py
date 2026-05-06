from torch_openreml.covariance.matrix import Matrix
import torch

class Operator(Matrix):
  
    _repr_single_line = False
    
    def __init__(self, *args, **kwargs):

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError('Operands must be provided either as keyword arguments or as positional arguments, but not both!')

        if len(args) > 0:
            if len(args) == 1 and isinstance(args[0], dict):
                operands = args[0]
            else:
                operands = {f"op_{i}": arg for i, arg in enumerate(args)}
        else:
            operands = kwargs

        self.check_operands(operands)
        self._operands = operands
        
        param_names = [
            f"{operand_name}/{name}"
            for operand_name, operand in operands.items()
            for name in getattr(operand, "param_names", [])
        ]
        
        super().__init__(None, param_names, [])
        
        del self._no_grad_index
        
    def check_operands(self, operands):
        if not isinstance(operands, dict):
            raise TypeError(f"operands must be a dict, got {type(operands).__name__}!")
        
        for key, value in operands.items():
    
            if not isinstance(key, str):
                raise TypeError(f"Operand name must be a string, got {type(key).__name__}!")
    
            if "/" in key:
                raise ValueError(f"Invalid operand name '{key}': '/' is not allowed!")
    
            if not isinstance(value, (Matrix, torch.Tensor)):
                raise TypeError(
                    f"Operand '{key}' must be a Matrix or torch.Tensor, "
                    f"got {type(value).__name__}!"
                )
                
        if not any(isinstance(v, Matrix) for v in operands.values()):
            raise TypeError("operands must include at least one Matrix!")
            
    def set_no_grad(self, index=None, param_name=None):
        raise RuntimeError(
            "This operator only provides a view of no_grad_index. "
            "Set it on the covariance matrix that owns the parameters instead!"
        )

    def trans_params(self, params):
        params = self.from_param_dict(params)
        self.check_params(params)

        result = []

        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                operand_params = params[0:operand.num_params]
                params = params[operand.num_params:]

                result.append(operand.trans_params(operand_params))

        return torch.cat(result)
    
    def build_operands(self, params):
        params = self.from_param_dict(params)
        self.check_params(params)
        
        v_groups = []
        
        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                operand_params = params[0:operand.num_params]
                params = params[operand.num_params:]

                v_groups.append(operand(operand_params))
            else:
                v_groups.append(operand)
        
        return v_groups

    def operands_grad(self, params):
        params = self.from_param_dict(params)
        self.check_params(params)

        grad_groups = []
        grad_name_groups = []

        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                operand_params = params[0:operand.num_params]
                params = params[operand.num_params:]

                grad, grad_names = operand.grad(operand_params)

                if grad is not None:
                    grad_groups.append(grad)
                    grad_name_groups.append([f"{name}/{grad_name}" for grad_name in grad_names])
                else:
                    grad_groups.append(None)
                    grad_name_groups.append([])
            else:
                grad_groups.append(None)
                grad_name_groups.append([])

        return grad_groups, grad_name_groups
        
    @property
    def operands(self):
        return self._operands
      
    @property
    def no_grad_index(self):
        result = []
        total_num_params = 0
        
        for name, operand in self._operands.items():
            if isinstance(operand, Matrix):
                result.extend([index + total_num_params for index in operand.no_grad_index])
                total_num_params = total_num_params + operand.num_params
                
        return result
    
    @property
    def repr_dict(self):
        return {"operands": self.operands}
        

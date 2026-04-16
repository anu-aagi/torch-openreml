import torch
from functools import partial
from abc import ABC, abstractmethod

class CovarianceMatrix(ABC):
  
    def __init__(self, n, param_names, no_grad_index=None):
        self.check_n(n)
        self._n = n
        
        self.reset_grad()
        self.check_no_grad_index(no_grad_index)
        
        self.check_param_names(param_names)
        self._param_names = param_names
        self._num_params = len(param_names)
        self._no_grad_index = list(set(no_grad_index or []))
    
    def reset_grad(self):
        self._grad = None
        self._grad_names = []
    
    def set_no_grad(self, index=None, param_name=None):
        if (index is None) == (param_name is None):
            raise ValueError("Provide exactly one of 'index' or 'param_name'!")
        
        if (param_name is None):
            if (not isinstance(index, list)):
                index = [index]
            self.check_no_grad_index(index)
            self._no_grad_index = list(set(index))
        
        if (index is None):
            if (not isinstance(param_name, list)):
                param_names = [param_name]
            index_map = {name: i for i, name in enumerate(self._param_names)}
            index = [index_map[name] for n in param_names]
            self._no_grad_index = list(set(index))
            
    def from_param_dict(self, param_dict):
        if not isinstance(param_dict, dict):
            return param_dict
        
        missing = set(self._param_names) - set(param_dict.keys())
        if missing:
            raise ValueError(f"Missing parameters: {missing}!")
        
        extra = set(param_dict.keys()) - set(self._param_names)
        if extra:
            raise ValueError(f"Unexpected parameters: {extra}!")
        
        return torch.cat([param_dict[name] for name in self._param_names])
      
    def to_param_dict(self, params):
        if isinstance(params, dict):
            return params
        
        if len(params) != len(self._param_names):
            raise ValueError(f"Expected {len(self._param_names)} parameters, got {len(params)}!")
        
        return {name: tensor for name, tensor in zip(self.param_names, params)}
      
    def auto_grad(self, params):
        jacobian = torch.func.jacrev(partial(self.build, grad=False))(params)
        jacobian = jacobian.permute(2, 0, 1)

    @abstractmethod
    def build(self, params, grad=True):
        raise NotImplementedError
    
    def check_n(self, n):
        if not isinstance(n, int):
            raise TypeError("'n' must be an int!")
      
    def check_no_grad_index(self, no_grad_index):
        if no_grad_index is not None:
            if any(idx < 0 or idx >= self._num_params for idx in no_grad_index):
                raise ValueError("Parameter index outside range!")
              
    def check_params(self, params):
        params = self.from_param_dict(params)
        
        if not torch.is_tensor(params):
            raise TypeError("Parameters must be a Torch tensor!")
    
        if params.dim() != 1:
            raise ValueError("Parameters must be a 1D tensor!")
    
        if params.shape[0] != self._num_params:
            raise ValueError(f"Parameters must have length {self.num_params}, got {params.shape[0]}!")
        
        return params.device, params.dtype
    
    def check_param_names(self, param_names):
        if len(param_names) != len(set(param_names)):
            raise ValueError(f"Parameter names must be unique!")
        
    @property
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


class OperatorMatrix(CovarianceMatrix):
    
    def __init__(self, n, operands):
        self.check_operands(operands)
        self._operands = operands
        
        param_names = [
            f"{operand_name}/{name}"
            for operand_name, operand in operands.items()
            for name in getattr(operand, "param_names", [])
        ]
        
        super().__init__(n, param_names)
        
        del self._no_grad_index
        
    def check_operands(self, operands):
        if not isinstance(operands, dict):
            raise TypeError(f"operands must be a dict, got {type(operands).__name__}!")
        
        for key, value in operands.items():
    
            if not isinstance(key, str):
                raise TypeError(f"Operand name must be a string, got {type(key).__name__}!")
    
            if "/" in key:
                raise ValueError(f"Invalid operand name '{key}': '/' is not allowed!")
    
            if not isinstance(value, (CovarianceMatrix, torch.Tensor)):
                raise TypeError(
                    f"Operand '{key}' must be a CovarianceMatrix or torch.Tensor, "
                    f"got {type(value).__name__}!"
                )
                
        if not any(isinstance(v, CovarianceMatrix) for v in operands.values()):
            raise TypeError("operands must include at least one CovarianceMatrix!")
            
    def set_no_grad(self, index=None, param_name=None):
        raise RuntimeError(
            "This operator only provides a view of no_grad_index. "
            "Set it on the covariance matrix that owns the parameters instead!"
        )
    
    def build_operands(self, params, grad=True):
        params = self.from_param_dict(params)
        self.check_params(params)
        
        v_groups = []
        grad_groups = []
        grad_name_groups = []
        
        for name, operand in self.operands.items():
            if isinstance(operand, CovarianceMatrix):
                operand_params = params[0:operand.num_params]
                params = params[operand.num_params:]
                
                v_groups.append(operand.build(operand_params, grad))
                
                if grad:
                    if operand.grad is not None:
                        grad_groups.append(operand.grad)
                        grad_name_groups.append(operand.grad_names)
        
        return v_groups, grad_groups, grad_name_groups
        
    @property
    def operands(self):
        return self._operands
      
    @property
    def no_grad_index(self):
        result = []
        total_num_params = 0
        
        for name, operand in self._operands.items():
            if isinstance(operand, CovarianceMatrix):
                result.extend([index + total_num_params for index in operand.no_grad_index])
                total_num_params = total_num_params + operand.num_params
                
        return result
        

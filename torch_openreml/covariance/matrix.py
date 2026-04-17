import torch
from functools import partial
from abc import ABC, abstractmethod

class Matrix(ABC):
  
    _repr_single_line = True
  
    def __init__(self, shape, param_names, no_grad_index=None):
        self.check_shape(shape)
        self._shape = shape
        
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
      
    def map_theta_to_v(self, theta):
        return self.build(theta, grad=True)
      
    def map_theta_to_dv(self, theta):
        if self.grad is None and len(self.no_grad_index) < len(self.num_params):
            self.build(grad=True)
        return self.grad
    
    def check_n(self, n):
        if not isinstance(n, int):
            raise TypeError("'n' must be an int!")
          
    def check_shape(self, shape):
        if shape is None:
            return
        if not isinstance(shape, (list, tuple, torch.Size)):
            raise TypeError("'shape' must be a list, a tuple or a torch.Size!")
      
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

    def __repr__(self):
        return self._repr_indented(0)
    
    def _repr_indented(self, level):
        indent = " " * 2
        
        if self._repr_single_line:
            args = ", ".join(f"{key}={repr(value)}" for key, value in self.repr_dict.items())
            return f"{self.__class__.__name__}({args})"
          
        inner = level + 1
        pad = indent * inner
        closing_pad = indent * level
        parts = []
        
        for key, value in self.repr_dict.items():
            key_str = f"{key}="
            value_pad = pad + " " * len(key_str)
            value_str = self._repr_value(value, inner, value_pad)
            parts.append(f"{pad}{key_str}{value_str}")
            
        args = ",\n".join(parts)
        return f"{self.__class__.__name__}(\n{args}\n{closing_pad})"
      
    def _repr_value(self, value, level, continuation_pad=""):
        indent = " " * 2
        
        if hasattr(value, "_repr_indented"):
            return value._repr_indented(level)
        elif isinstance(value, dict):
            return self._repr_dict(value, level)
        else:
            if not continuation_pad:
                continuation_pad = indent * level
            return repr(value).replace("\n", "\n" + continuation_pad)
      
    def _repr_dict(self, d, level):
        indent = " " * 2
        inner = level + 1
        pad = indent * inner
        closing_pad = indent * level
        parts = []
        
        for key, value in d.items():
            key_str = f"{key!r}: "
            value_pad = pad + " " * len(key_str)
            value_str = self._repr_value(value, inner, value_pad)
            parts.append(f"{pad}{key_str}{value_str}")
            
        args = ",\n".join(parts)
        return "{\n" + args + "\n" + closing_pad + "}"
        
    @property
    def shape(self):
        return self._shape

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
    
    @property
    def repr_dict(self):
        return {"shape": self._shape, "no_grad_index": self._no_grad_index}


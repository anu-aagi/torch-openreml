import torch
from abc import ABC, abstarctmethod

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
            
    def map_param_dict(self, param_dict):
        missing = set(self._param_names) - set(param_dict.keys())
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        
        extra = set(param_dict.keys()) - set(self._param_names)
        if extra:
            raise ValueError(f"Unexpected parameters: {extra}")
        
        return [param_dict[name] for name in self._param_names]

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
        
        return params.device, params.dtype
    
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

"""
Covariance matrix abstraction system.

This module defines a flexible base class for constructing covariance matrices
used in linear mixed-effects models. Implementations support both manual and
automatic differentiation via :meth:`auto_grad`, enabling gradient-based
optimization of model parameters.

Classes:
  Matrix:
      Base class providing parameter validation, transform application,
      and Jacobian computation utilities for all covariance matrix
      implementations.
"""

import torch
from abc import ABC, abstractmethod
from torch_openreml.covariance.transform import Transform


class Matrix(ABC):
    r"""
    Abstract base class for covariance matrices with parameterized structure.

    .. math::
        \symbf{V} = \symbf{V}(\symbf{\theta})

    where :math:`\symbf{\theta}` denotes the collection of variance component
    parameters that define the matrix entries.

    This class provides utilities for parameter validation, transform application,
    and Jacobian computation (both manual and automatic).
    Subclasses must implement :meth:`build` to construct their specific matrix
    structure from the provided parameters.
    """
  
    _repr_single_line = True

    def __init__(self, shape, param_names, trans=None, no_grad_index=None):
        r"""
        Initialize a covariance matrix with optional parameter transforms.

        Args:
            shape (tuple or None): Expected output dimensions of the constructed matrix.
                Used for validation; the actual shape may be set by subclasses.
            param_names (list of str): Ordered names of parameters in :attr:`params`.
                Empty list if no trainable parameters (e.g., fixed matrices).
            trans (list of Transform or None): List of transforms applied to each
                parameter before constructing the matrix. If None, no transforms are used.
                Typically used for variance (:math:`\exp(2\theta) > 0`) or correlation
                constraints (:math:`\rho \in (-1, 1)`).
            no_grad_index (list of int): Indices to exclude from gradient computation.
                Parameters at these indices will be omitted from :attr:`grad` and
                :attr:`grad_names`. Use :meth:`set_no_grad` instead for convenience.

        Note:
            The transform applies as

            .. math::
                \symbf{V} = \left[f_0(\theta_0), \ldots, f_{p-1}(\theta_{p-1}) \right]^\top,

            where each :math:`f_i` is the i-th transform in :attr:`trans`.
            If :attr:`trans` has length 1, the single transform is broadcast and applied elementwise to all parameters.

        Raises:
            TypeError: If ``param_names`` is not a list of strings, or if
                transforms contain non-Transform objects.
            ValueError: If parameter names are not unique, or if indices in
                ``no_grad_index`` are out of range.
        """

        self._check_shape(shape)
        self._shape = tuple(shape or ())

        self._check_no_grad_index(no_grad_index)
        
        self._check_param_names(param_names)
        self._param_names = param_names
        self._num_params = len(param_names)

        self._check_trans(trans)
        self._trans = trans

        self._no_grad_index = list(set(no_grad_index or []))

        #: Gradient computation mode: ``"manual"`` uses a class-defined manual gradient,
        # ``"auto"`` uses automatic differentiation, and ``"default"`` uses the manual
        # gradient if :meth:`manual_grad` is defined, otherwise automatic differentiation.
        self.grad_mode = "default"

        self.reset_intermediates()

    def set_intermediates(self, params, intermediates):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)

        if params.shape[0] == 0:
            return None

        h = torch.hash_tensor(params).item()
        self._intermediates["hash"] = h
        self._intermediates["dtype"] = dtype
        self._intermediates["device"] = device
        self._intermediates["intermediates"] = intermediates

    def get_intermediates(self, params):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)

        if params.shape[0] == 0:
            return None

        h = torch.hash_tensor(params).item()
        if self._intermediates["hash"] == h:
            if self._intermediates["dtype"] == dtype:
                if self._intermediates["device"] == device:
                    return self._intermediates["intermediates"]

        return None

    def reset_intermediates(self):
        self._intermediates = {"hash": None, "dtype": None, "device": None, "intermediates": None}
    
    def set_no_grad(self, index=None, param_name=None):
        if (index is None) == (param_name is None):
            raise ValueError("Provide exactly one of 'index' or 'param_name'!")
        
        if param_name is None:
            if not isinstance(index, list):
                index = [index]
            self._check_no_grad_index(index)
            self._no_grad_index = list(set(index))
        
        if index is None:
            if not isinstance(param_name, list):
                param_name = [param_name]
            index_map = {name: i for i, name in enumerate(self._param_names)}
            index = [index_map[name] for name in param_name]
            self._no_grad_index = list(set(index))

    def from_param_dict(self, param_dict):
        r"""
        Extract parameter tensors from a dictionary into a flat 1D tensor.

        Converts a parameter dictionary to a concatenated 1D tensor ordered
        according to :attr:`param_names`. The inverse operation is provided
        by :meth:`to_param_dict`.

        Args:
            param_dict (torch.Tensor or dict): Either a flat parameter tensor
                (returned as-is), or a dictionary mapping parameter names to
                tensors. All keys must exist in :attr:`param_names` and no
                extra keys are allowed.

        Returns:
            torch.Tensor: Concatenated 1D tensor containing all parameters
                in the order specified by :attr:`param_names`.

        Raises:
            ValueError: If ``param_dict`` is a dictionary missing required keys
                or containing unexpected keys, or if the tensor length does not
                match the number of parameters.
        """

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
        
        return {name: tensor for name, tensor in zip(self.param_names, params.unsqueeze(-1))}

    def trans_params(self, params):
        params = self.from_param_dict(params)
        _, _ = self.check_params(params)

        if self.trans is None or len(self.trans) == 0:
            return params

        if len(self.trans) == 1:
            return self.trans[0](params)
        else:
            return torch.cat([self.trans[i](x) for i, x in enumerate(params.unsqueeze(-1))])

    def trans_grad(self, params):
        params = self.from_param_dict(params)
        _, _ = self.check_params(params)

        if self.trans is None or len(self.trans) == 0:
            return torch.tensor([1.0], dtype=params.dtype, device=params.device)

        if len(self.trans) == 1:
            return self.trans[0].grad(params)
        else:
            return torch.cat([self.trans[i].grad(x) for i, x in enumerate(params.unsqueeze(-1))])
      
    def auto_grad(self, params):
        """
        Compute the Jacobian of :meth:`build` with respect to
        trainable parameters using automatic differentiation.

        Uses :func:`torch.func.jacrev` to compute the full Jacobian, then
        masks out parameters listed in :attr:`no_grad_index`.

        If all parameters are excluded via ``no_grad_index``, returns ``(None, [])``

        Args:
            params (torch.Tensor): Flat 1D parameter tensor.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(`` :attr:`num_params` ``-len(`` :attr:`no_grad_index` ``), *`` :attr:`shape` ``)``, and
            ``grad_names`` has the same length as ``grad``.
        """
        if len(self.no_grad_index) == self._num_params:
            return None, []

        self.reset_intermediates()

        jacobian = torch.func.jacrev(self.__call__)(params)
        jacobian = jacobian.permute(2, 0, 1)

        mask = torch.ones(self.num_params, dtype=torch.bool)
        mask[self.no_grad_index] = False
        grad = jacobian[mask]
        grad_names = [name for i, name in enumerate(self.param_names) if i not in self.no_grad_index]

        return grad, grad_names

    def manual_grad(self, params):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, params):
        """
        Construct the matrix from a flat parameter tensor.

        Must be implemented by subclasses. Implementations should convert
        ``params`` via :meth:`from_param_dict` or :meth:`to_param_dict`,
        then call :meth:`check_params` to validate and :meth:`trans_params`
        to apply transforms before any computation.

        Args:
            params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            torch.Tensor: Constructed matrix of shape :attr:`shape`.
        """
        raise NotImplementedError

    def grad(self, params):
        if self.grad_mode == "default":
            try:
                return self.manual_grad(params)
            except NotImplementedError:
                return self.auto_grad(params)
        elif self.grad_mode == "auto":
            return self.auto_grad(params)
        else:
            raise RuntimeError(f"Unknown grad mode '{self.grad_mode}'")
      
    def map_theta_to_v(self, theta):
        """
        An interface compatible with :class:`torch_openreml.REML` that maps
        parameters to a matrix.

        Invokes :meth:`__call__`.

        Args:
            theta (torch.Tensor): Flat 1D parameter tensor.

        Returns:
            torch.Tensor: Constructed matrix.
        """
        return self(theta)
      
    def map_theta_to_dv(self, theta):
        """
        An interface compatible with :class:`torch_openreml.REML` that maps parameters
        to the matrix Jacobian.

        Invokes :meth:`grad`.

        Args:
            theta (torch.Tensor): Flat 1D parameter tensor.

        Returns:
            torch.Tensor or None: Jacobian tensor of shape
            ``(num_grad_params, *shape)``, or ``None`` if all parameters
            are excluded from gradient computation.
        """
        grad, grad_name = self.grad(theta)
        return grad
    
    def _check_n(self, n):
        if not isinstance(n, int):
            raise TypeError("'n' must be an int!")
          
    def _check_shape(self, shape):
        if shape is None:
            return
        
        if not isinstance(shape, (list, tuple, torch.Size)):
            raise TypeError("'shape' must be a list, a tuple or a torch.Size!")
        
        shape = tuple(shape)
        
        if not all([isinstance(p, int) and p > 0 for p in shape]):
            raise TypeError("All elements of 'shape' must be positive int!")

    def _check_trans(self, trans):
        if trans is None or len(trans) == 0:
            return

        if isinstance(trans, (list, tuple)):
            for t in trans:
                if not isinstance(t, Transform):
                    raise TypeError("'trans' must be a list of Transform objects!")
        else:
            raise TypeError("'trans' must be a list of Transform objects!")

        if len(trans) not in (0, 1, self._num_params):
            raise ValueError(f"'trans' must be 0, 1, or 'num_params'!")
      
    def _check_no_grad_index(self, no_grad_index):
        if no_grad_index is not None:
            if any(idx < 0 or idx >= self._num_params for idx in no_grad_index):
                raise ValueError("Parameter index outside range!")
              
    def check_params(self, params):
        """
        Validate a parameter tensor and return its device and dtype.

        Accepts a parameter dictionary and converts it to a flat tensor
        via :meth:`from_param_dict` before validation.

        Args:
            params (torch.Tensor or dict): Parameters to validate.

        Returns:
            tuple: ``(device, dtype)`` of the parameter tensor.

        Raises:
            TypeError: If ``params`` is not a tensor.
            ValueError: If ``params`` is not 1D or has the wrong length.
        """
        params = self.from_param_dict(params)
        
        if not torch.is_tensor(params):
            raise TypeError("Parameters must be a Torch tensor!")
    
        if params.dim() != 1:
            raise ValueError("Parameters must be a 1D tensor!")
    
        if params.shape[0] != self._num_params:
            raise ValueError(f"Parameters must have length {self.num_params}, got {params.shape[0]}!")
        
        return params.device, params.dtype
    
    def _check_param_names(self, param_names):
        if not isinstance(param_names, (list, tuple)) or not all(isinstance(x, str) for x in param_names):
            raise TypeError("'param_names' must be a list or tuple of strings!")
        if len(param_names) != len(set(param_names)):
            raise ValueError(f"Parameter names must be unique!")

    def __repr__(self):
        return self._repr_indented(0)
    
    def _repr_indented(self, level):
        indent = " " * 2
        
        if self._repr_single_line:
            args = []
            for key, value in self.repr_dict.items():
                if value:
                    if key in ("param_names", "trans") and len(value) > 3:
                        args.append(f"{key}=[{value[0]}, ..., {value[-1]}]")
                    else:
                        args.append(f"{key}={repr(value)}")
            args = ", ".join(args)
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
        elif isinstance(value, torch.Tensor):
            return value.shape
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
        """tuple: Output matrix shape."""
        return self._shape
      
    @property  
    def param_names(self):
        """list of str: Ordered parameter names."""
        return self._param_names
    
    @property
    def num_params(self):
        """int: Total number of parameters."""
        return self._num_params

    @property
    def trans(self):
        """list of Transform: Parameter transforms."""
        return self._trans
    
    @property
    def no_grad_index(self):
        """list of int: Indices of parameters excluded from gradient computation."""
        return self._no_grad_index
    
    @property
    def repr_dict(self):
        """dict: Key-value pairs used to build the string representation."""
        return {"shape": self._shape, "param_names": self._param_names, "trans": self._trans, "no_grad_index": self._no_grad_index}


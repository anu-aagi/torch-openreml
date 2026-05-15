"""
Covariance matrix abstraction system.

This module defines a flexible base class for constructing covariance matrices
used in linear mixed-effects models. Implementations support both manual (:meth:`manual_grad`) and
automatic differentiation (:meth:`auto_grad`).

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

    def __init__(self, shape, param_spec):
        r"""
        Initialize a covariance matrix with parameter specifications.

        Args:
            shape (tuple or None): Expected output dimensions of the constructed matrix.
                Used for validation; the actual shape may be set by subclasses.
            param_spec (dict): Parameter specifications. Keys should be strings
                representing parameter names. Values should be dictionaries
                containing the specification for each parameter. Each specification
                dictionary should contain the keys "fixed", "default", and "trans",
                representing whether the parameter is fixed or free (bool), the
                default value (1D torch.Tensor), and the transform (Transform),
                respectively.

        Note:
            The transform applies as

            .. math::
                \symbf{V} = \left[f_0(\theta_0), \ldots, f_{p-1}(\theta_{p-1}) \right]^\top,

            where :math:`f_i` is the transform for `i`-th parameter.

        Raises:
            TypeError: If ``param_spec`` does not follow any of the requirements
                list in the argument description.
            ValueError: If parameter names are not unique, or if keys in
                ``fix_params`` are out of range.
        """

        self._check_shape(shape)
        self._shape = tuple(shape or ())

        self._check_param_spec(param_spec)
        self._param_spec = param_spec

        #: Gradient computation mode: ``"manual"`` uses a class-defined manual gradient,
        # ``"auto"`` uses automatic differentiation, and ``"default"`` uses the manual
        # gradient if :meth:`manual_grad` is defined, otherwise automatic differentiation.
        self.grad_mode = "default"

        self.reset_intermediates()

    def set_intermediates(self, params, intermediates):
        """
        Cache intermediate computation results keyed by parameter hash.

        Stores arbitrary intermediate values alongside a hash of the current
        parameter tensor, dtype, and device. Cached values can be retrieved
        via :meth:`get_intermediates` to avoid redundant computation across
        multiple calls with identical parameters.

        Args:
            params (torch.Tensor): Current parameter tensor.
            intermediates: Arbitrary object to cache (e.g. Cholesky factors,
                eigendecompositions, or any reusable computation).

        Note:
            If ``params`` has length 0 (no free parameters), this is a no-op.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(3)
            free_params = torch.tensor([0.0, 0.5, 1.0])
            sigma2 = mat.build_params(free_params)
            mat.set_intermediates(free_params, {"sigma2": sigma2})
            mat.get_intermediates(free_params)
        """
        device, dtype = self._check_param_tensor(params)

        if params.shape[0] == 0:
            return None

        h = torch.hash_tensor(params).item()
        self._intermediates["hash"] = h
        self._intermediates["dtype"] = dtype
        self._intermediates["device"] = device
        self._intermediates["intermediates"] = intermediates

    def get_intermediates(self, params):
        """
        Retrieve cached intermediate computation results if still valid.

        Compares the hash, dtype, and device of ``params`` against the stored
        cache from the last :meth:`set_intermediates` call. Returns the cached
        value only if all three match, ensuring stale results are never returned
        after a parameter update, device transfer, or dtype cast.

        Args:
            params (torch.Tensor): Current parameter tensor.

        Returns:
            The cached intermediate object if the cache is valid, or ``None`` if
            the cache is missing, stale, or ``params`` has length 0.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(3)
            free_params = torch.tensor([0.0, 0.5, 1.0])
            sigma2 = mat.build_params(free_params)
            mat.set_intermediates(free_params, {"sigma2": sigma2})
            mat.get_intermediates(free_params)
        """
        device, dtype = self._check_param_tensor(params)

        if params.shape[0] == 0:
            return None

        h = torch.hash_tensor(params).item()
        if self._intermediates["hash"] == h:
            if self._intermediates["dtype"] == dtype:
                if self._intermediates["device"] == device:
                    return self._intermediates["intermediates"]

        return None

    def reset_intermediates(self):
        """
        Clear the intermediate computation cache.

        Resets all cached values set by :meth:`set_intermediates` to ``None``,
        forcing subsequent calls to :meth:`get_intermediates` to return ``None``
        until the cache is repopulated. Called automatically in :meth:`__init__`
        and within :meth:`auto_grad` before triggering a fresh Jacobian computation.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(3)
            free_params = torch.tensor([0.0, 0.5, 1.0])
            sigma2 = mat.build_params(free_params)
            mat.set_intermediates(free_params, {"sigma2": sigma2})
            print(mat.get_intermediates(free_params))
            mat.reset_intermediates()
            print(mat.get_intermediates(free_params))
        """
        self._intermediates = {"hash": None, "dtype": None, "device": None, "intermediates": None}

    def build_params(self, free_params, include_fixed=True, trans=True, out_format="tensor"):

        free_params = self._from_free_param_dict(free_params)
        device, dtype = self._check_param_tensor(free_params, length=self.num_free_params)

        if include_fixed:
            params = free_params.new_empty(self.num_params)

            free_mask = torch.tensor([not spec["fixed"] for spec in self.param_spec.values()], dtype=torch.bool, device=device)
            params[free_mask] = free_params
            params[~free_mask] = torch.as_tensor([spec["default"] for spec in self.param_spec.values() if spec["fixed"]],
                                                 device=device, dtype=dtype)
        else:
            params = free_params

        if len(params) == 0:
            if out_format == "tensor":
                return torch.tensor([], device=device, dtype=dtype)
            elif out_format == "dict":
                return {}
            else:
                raise ValueError(f"Unexpected 'out_format': {out_format}!")

        if trans:
            if include_fixed:
                param_trans = list(self.param_trans.values())
            else:
                param_trans = list(self.free_param_trans.values())
            ref_dict = param_trans[0].__dict__
            ref_type = type(param_trans[0])
            if all(type(trans) is ref_type and trans.__dict__ == ref_dict for trans in param_trans):
                params = param_trans[0](params)
            else:
                params = torch.cat([trans(param) for trans, param in zip(param_trans, params.unsqueeze(-1))])

        if out_format == "tensor":
            return params
        elif out_format == "dict":
            if include_fixed:
                param_names = self.param_names
            else:
                param_names = self.free_param_names
            return dict(zip(param_names, params.unsqueeze(-1)))
        else:
            raise ValueError(f"Unexpected 'out_format': {out_format}!")

    def _from_free_param_dict(self, free_param_dict):
        if not isinstance(free_param_dict, dict):
            return free_param_dict
        
        missing = set(self.free_param_names) - set(free_param_dict.keys())
        if missing:
            raise ValueError(f"Missing free parameters: {missing}!")
        
        extra = set(free_param_dict.keys()) - set(self.free_param_names)
        if extra:
            raise ValueError(f"Unexpected free parameters: {extra}!")
        
        return torch.cat([free_param_dict[name] for name in self.free_param_names])

    def _to_free_param_dict(self, free_params):
        if isinstance(free_params, dict):
            return free_params
        
        if len(free_params) != len(self.free_param_names):
            raise ValueError(f"Expected {len(self.free_param_names)} parameters, got {len(free_params)}!")
        
        return {name: tensor for name, tensor in zip(self.param_names, free_params.unsqueeze(-1))}

    def trans_grad(self, free_params):
        """
        Compute the element-wise derivative of the transforms for the free parameters.

        Returns the Jacobian diagonal of :meth:`trans_free_params` with respect
        to the raw (untransformed) parameters. Used in the chain rule when
        computing gradients of the matrix with respect to the original
        parameterisation.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                dictionary.

        Returns:
            torch.Tensor: 1D tensor of element-wise transform derivatives,
            of the same length as ``free_params``.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(3)
            free_params = torch.tensor([0.0, 0.5, 1.0])
            mat.trans_grad(free_params)
        """
        free_params = self._from_free_param_dict(free_params)
        device, dtype = self._check_param_tensor(free_params, length=self.num_free_params)

        free_param_trans = list(self.free_param_trans.values())
        ref_dict = free_param_trans[0].__dict__
        ref_type = type(free_param_trans[0])
        if all(type(trans) is ref_type and trans.__dict__ == ref_dict for trans in free_param_trans):
            return free_param_trans[0].grad(free_params)
        else:
            return torch.cat([trans.grad(free_param) for trans, free_param in zip(free_param_trans, free_params.unsqueeze(-1))])

    def auto_grad(self, free_params):
        """
        Compute the Jacobian of :meth:`build` with respect to
        free parameters using automatic differentiation.

        Uses :func:`torch.func.jacrev` to compute the full Jacobian.

        If all parameters are fixed, returns ``(None, [])``

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or dict.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)``, and
            ``grad_names`` has the same length as ``grad``.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(2)
            free_params = torch.tensor([0.0, 0.5])
            grad, grad_names = mat.auto_grad(free_params)
            grad, grad_names
        """
        if len(free_params) == 0:
            return None, []

        free_params = self._from_free_param_dict(free_params)
        device, dtype = self._check_param_tensor(free_params, length=self.num_free_params)

        self.reset_intermediates()

        jacobian = torch.func.jacrev(self.__call__)(free_params)
        grad = jacobian.permute(2, 0, 1)
        grad_names = self.free_param_names

        return grad, grad_names

    def manual_grad(self, free_params):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        This method is optional. When implemented by a subclass, :meth:`grad`
        will invoke it in preference to :meth:`auto_grad` under the default
        grad mode. If not implemented, calling this method raises
        :class:`NotImplementedError` and :meth:`grad` falls back to automatic
        differentiation.

        Implementations must satisfy the following contract:

        - Return ``(None, [])`` if all parameters are fixed.
        - Return a 3D gradient tensor of shape
          ``(num_free_params, *shape)`` and a matching list
          of parameter names.
        - Apply transform derivatives from :meth:`trans_grad` via the chain
          rule so that gradients are with respect to the raw (untransformed)
          parameters.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` if all parameters are fixed.

        Raises:
            NotImplementedError: If the subclass does not provide an analytic
                gradient. :meth:`grad` catches this and falls back to
                :meth:`auto_grad`.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, free_params):
        """
        Construct the matrix from a flat parameter tensor.

        Must be implemented by subclasses. Implementations should convert
        ``free_params`` via :meth:`build_params` to validate,
        include fixed parameters, and apply transforms before any computation.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            torch.Tensor: Constructed matrix of shape :attr:`shape`.
        """
        raise NotImplementedError

    def grad(self, free_params):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters.

        Dispatches to :meth:`manual_grad` or :meth:`auto_grad` according to
        :attr:`grad_mode`:

        - ``"default"``: attempts :meth:`manual_grad`, falling back to
          :meth:`auto_grad` if not implemented.
        - ``"auto"``: always uses :meth:`auto_grad`.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            tuple: ``(grad, grad_names)`` as described in :meth:`manual_grad`
            and :meth:`auto_grad`.

        Raises:
            RuntimeError: If :attr:`grad_mode` is not a recognised value.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(2)
            free_params = torch.tensor([0.0, 0.5])
            grad, grad_names = mat.grad(free_params)
            grad, grad_names
        """
        if self.grad_mode == "default":
            try:
                return self.manual_grad(free_params)
            except NotImplementedError:
                return self.auto_grad(free_params)
        elif self.grad_mode == "auto":
            return self.auto_grad(free_params)
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
            ``(num_free_params, *shape)``, or ``None`` if all parameters
            are fixed.
        """
        grad, grad_name = self.grad(theta)
        return grad
          
    def _check_shape(self, shape):
        if shape is None:
            return
        
        if not isinstance(shape, (list, tuple, torch.Size)):
            raise TypeError("'shape' must be a list, a tuple or a torch.Size!")
        
        shape = tuple(shape)
        
        if not all([isinstance(p, int) and p > 0 for p in shape]):
            raise TypeError("All elements of 'shape' must be positive int!")

    def _check_param_spec(self, param_spec):
        if not isinstance(param_spec, dict):
            raise TypeError("'param_sepc' must be a dict!")

        for param_name, spec in param_spec.items():
            if not isinstance(param_name, str):
                raise TypeError(f"Parameter name must be a str, got {type(param_name).__name__}!")

            if not isinstance(spec, dict):
                raise TypeError(f"Individual parameter specification must be a dict, got {type(spec).__name__}!")

            if sorted(list(spec.keys())) != ["default", "fixed", "trans"]:
                raise TypeError(f"Parameter specification fields must be 'fixed', 'default', and 'trans', got {sorted(list(spec.keys()))}!")

            if not isinstance(spec["fixed"], bool):
                raise TypeError(f"Parameter specification field 'fixed' must be a bool, got {type(spec["fixed"]).__name__}!")

            if not torch.is_tensor(spec["default"]):
                raise TypeError(f"Parameter specification field 'default' must be a torch.Tensor, got {type(spec["default"]).__name__}!")

            if spec["default"].ndim != 1:
                raise TypeError(f"Parameter specification field 'default' must be a 1D torch.Tensor, got {spec["default"].shape}!")

            if not isinstance(spec["trans"], Transform):
                raise TypeError(f"Parameter specification field 'trans' must be a Transform, got {type(spec["trans"]).__name__}!")

    def _check_param_tensor(self, params, length=None):
        if not torch.is_tensor(params):
            raise TypeError("Parameters must be a Torch tensor!")

        if params.dim() != 1:
            raise ValueError("Parameters must be a 1D tensor!")

        if length:
            if params.shape[0] != length:
                raise ValueError(f"Parameters must have length {length}, got {params.shape[0]}!")

        return params.device, params.dtype

    def __repr__(self):
        return self._repr_indented(0)
    
    def _repr_indented(self, level):
        indent = " " * 2
        
        if self._repr_single_line:
            args = []
            for key, value in self.repr_dict.items():
                if value:
                    # if key == "param_spec":
                    #     if len(value) < 3:
                    #         list(value.keys())
                    #         args.append(f"{value}")
                    #     else:
                    #         items = list(value.items())
                    #         first = items[0]
                    #         last = items[-1]
                    #         args.append(f"{key}={{{first[0]!r}: {first[1]!r}, ..., {last[0]!r}: {last[1]!r}}}")
                    # else:
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
    def param_spec(self):
        """dict: Parameter specifications."""
        return self._param_spec
      
    @property  
    def param_names(self):
        """list of str: Parameter names."""
        return list(self.param_spec.keys())

    @property
    def free_param_names(self):
        """list of str: Free parameter names."""
        return [param_name for param_name, spec in self.param_spec.items() if not spec["fixed"]]

    @property
    def fixed_param_names(self):
        """list of str: Fixed parameter names."""
        return [param_name for param_name, spec in self.param_spec.items() if spec["fixed"]]

    @property
    def free_param_index(self):
        return [i for i, spec in enumerate(self.param_spec.values()) if not spec["fixed"]]

    @property
    def fixed_param_index(self):
        return [i for i, spec in enumerate(self.param_spec.values()) if spec["fixed"]]
    
    @property
    def num_params(self):
        """int: Total number of parameters."""
        return len(self.param_spec)
    
    @property
    def num_free_params(self):
        """int: Total number of free parameters."""
        return len(self.free_param_names)
    
    @property
    def num_fixed_params(self):
        """int: Total number of fixed parameters."""
        return len(self.fixed_param_names)
    
    @property
    def param_defaults(self):
        """Dict of torch.Tensor: Parameter defaults."""
        return {param_name: spec["default"] for param_name, spec in self.param_spec.items()}

    @property
    def free_param_defaults(self):
        """Dict of torch.Tensor: Free parameter defaults."""
        return {param_name: spec["default"] for param_name, spec in self.param_spec.items() if not spec["fixed"]}

    @property
    def fixed_param_defaults(self):
        """Dict of torch.Tensor: Fixed parameter defaults."""
        return {param_name: spec["default"] for param_name, spec in self.param_spec.items() if spec["fixed"]}

    @property
    def param_trans(self):
        """Dict of Transform: Parameter transforms."""
        return {param_name: spec["trans"] for param_name, spec in self.param_spec.items()}
    
    @property
    def free_param_trans(self):
        """Dict of Transform: Transforms for free parameters."""
        return {param_name: spec["trans"] for param_name, spec in self.param_spec.items() if not spec["fixed"]}

    @property
    def fixed_param_trans(self):
        """Dict of Transform: Transforms for fixed parameters."""
        return {param_name: spec["trans"] for param_name, spec in self.param_spec.items() if spec["fixed"]}

    @property
    def repr_dict(self):
        """dict: Key-value pairs used to build the string representation."""
        return {"shape": self._shape, "param_spec": self.param_spec}


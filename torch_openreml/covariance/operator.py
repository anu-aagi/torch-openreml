"""
Operator base class for composite covariance matrices.

This module provides an abstract base class for combining multiple
covariance matrices into a single composite structure. Operators
delegate parameter management, transforms, and gradient computation
to their constituent operands, presenting a unified
:class:`~torch_openreml.covariance.matrix.Matrix` interface to the rest
of the library.

Classes:
    Operator:
        Base class for composite covariance matrix operators.
"""

from functools import reduce
from torch_openreml.covariance.matrix import Matrix
import torch

class Operator(Matrix):
    r"""
    Abstract base class for composite covariance matrix operators.

    An operator combines one or more
    :class:`~torch_openreml.covariance.matrix.Matrix` instances and optional
    fixed :class:`torch.Tensor` operands into a single covariance matrix.
    Parameters and gradients are namespaced by operand — a parameter
    ``"sigma^2"`` belonging to operand ``"residual"`` is exposed as
    ``"residual/sigma^2"`` in the combined :attr:`param_names`.

    Operands may be passed as positional arguments, as keyword arguments,
    or as a single dictionary. Mixing positional and keyword arguments is
    not permitted. Positional operands are assigned names ``"op_0"``,
    ``"op_1"``, etc.

    At least one operand must be a
    :class:`~torch_openreml.covariance.matrix.Matrix` instance. Pure-tensor
    operands are treated as fixed matrices with no free parameters.
    """
  
    _repr_single_line = False
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the operator from positional or keyword operands.

        Args:
            *args: Operands as positional arguments, each a
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor`. A single dict argument is also accepted
                and treated as a named operand mapping.
            **kwargs: Operands as keyword arguments, mapping operand names to
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor` instances. Operand names must not contain
                ``"/"``.

        Raises:
            ValueError: If both positional and keyword arguments are provided,
                or if any operand name contains ``"/"``.
            TypeError: If ``operands`` is not a dict, if any operand name is not
                a string, if any operand is not a
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor`, or if no operand is a
                :class:`~torch_openreml.covariance.matrix.Matrix`.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import Sum, ScalarMatrix

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                print(x)

                x = Sum(A = ScalarMatrix(2), B = ScalarMatrix(2))
                print(x)

                print(x.param_names)

                print(x(torch.zeros(2)))
                print(x({"A/sigma^2": torch.zeros(1), "B/sigma^2": torch.zeros(1)}))

        """

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError('Operands must be provided either as keyword arguments or as positional arguments, but not both!')

        if len(args) > 0:
            if len(args) == 1 and isinstance(args[0], dict):
                operands = args[0]
            else:
                operands = {f"op_{i}": arg for i, arg in enumerate(args)}
        else:
            operands = kwargs

        self._check_operands(operands)
        self._operands = operands
        
        super().__init__(None, {})
        
    def _check_operands(self, operands):
        """
        Validate the operand dictionary.

        Ensures that all keys are strings without ``"/"``, all values are
        :class:`~torch_openreml.covariance.matrix.Matrix` or
        :class:`torch.Tensor` instances, and that at least one value is a
        :class:`~torch_openreml.covariance.matrix.Matrix`.

        Args:
            operands (dict): Mapping from operand names to operands.

        Raises:
            TypeError: If ``operands`` is not a dict, if any key is not a
                string, if any value is not a
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor`, or if no value is a
                :class:`~torch_openreml.covariance.matrix.Matrix`.
            ValueError: If any key contains ``"/"``.
        """
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

    def build_params(self, free_params, include_fixed=True, trans=True, out_format="tensor"):
        free_params = self._from_free_param_dict(free_params)
        self._check_param_tensor(free_params, length=self.num_free_params)

        result = []

        for name, operand in self._operands.items():
            if isinstance(operand, Matrix):
                operand_free_params = free_params[0:operand.num_free_params]
                free_params = free_params[operand.num_free_params:]

                result.append(operand.build_params(operand_free_params, include_fixed=include_fixed, trans=trans, out_format="tensor"))

        result = torch.cat(result)

        if out_format == "tensor":
            return result
        elif out_format == "dict":
            return dict(zip(self.free_param_names, result))
        else:
            raise ValueError(f"Unexpected 'out_format': {out_format}!")
    
    def build_operands(self, free_params):
        """
        Evaluate each operand at the current free parameters.

        Splits ``free_params`` into per-operand slices and calls each
        :class:`~torch_openreml.covariance.matrix.Matrix` operand to
        produce its matrix. Fixed tensor operands are included as-is.

        Args:
            free_params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_free_params`.

        Returns:
            list of torch.Tensor: Evaluated operand matrices in the same
            order as :attr:`operands`.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import Sum, ScalarMatrix

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                v_groups = x.build_operands(torch.tensor([1.0, 2.0]))
                print(v_groups[0])
                print(v_groups[1])
        """
        free_params = self._from_free_param_dict(free_params)
        self._check_param_tensor(free_params, length=self.num_free_params)
        
        v_groups = []
        
        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                operand_params = free_params[0:operand.num_free_params]
                free_params = free_params[operand.num_free_params:]

                v_groups.append(operand(operand_params))
            else:
                v_groups.append(operand)
        
        return v_groups

    def operands_grad(self, free_params):
        """
        Compute the Jacobian of each operand with respect to its parameters.

        Splits ``free_params`` into per-operand slices, calls
        :meth:`~torch_openreml.covariance.matrix.Matrix.grad` on each
        :class:`~torch_openreml.covariance.matrix.Matrix` operand, and
        prefixes the returned names with the operand name. Fixed tensor
        operands contribute ``None`` and an empty name list.

        Args:
            params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_params`.

        Returns:
            tuple: ``(grad_groups, grad_name_groups)``, where
            ``grad_groups`` is a list of per-operand Jacobian tensors or
            ``None`` for fixed operands, and ``grad_name_groups`` is a list
            of corresponding namespaced parameter name lists.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import Sum, ScalarMatrix

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                grad_groups, grad_name_groups = x.operands_grad(torch.tensor([1.0, 2.0]))
                print(grad_groups[0])
                print(grad_groups[1])
                print(grad_name_groups[0])
                print(grad_name_groups[1])
        """
        free_params = self._from_free_param_dict(free_params)
        self._check_param_tensor(free_params, length=self.num_free_params)

        grad_groups = []
        grad_name_groups = []

        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                operand_params = free_params[0:operand.num_free_params]
                free_params = free_params[operand.num_free_params:]

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
        """dict: Mapping from operand names to operand matrices or tensors."""
        return self._operands

    @property
    def param_spec(self):
        param_spec = {}
        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                this_param_spec = {f"{name}/{param_name}": spec for param_name, spec in operand.param_spec.items()}
                param_spec.update(this_param_spec)
        return param_spec
    
    @property
    def repr_dict(self):
        """dict: Key-value pairs used to build the string representation."""
        return {"operands": self.operands}
        

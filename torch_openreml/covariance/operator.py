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
    operands are treated as fixed matrices with no trainable parameters.

    :attr:`no_grad_index` is a read-only derived property — it aggregates
    the ``no_grad_index`` values of each constituent operand. To exclude a
    parameter from gradient computation, call
    :meth:`~torch_openreml.covariance.matrix.Matrix.set_no_grad` on the
    operand that owns it directly.
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
            
    def set_no_grad(self, index=None, param_name=None):
        """
        Disabled — ``no_grad_index`` is managed by each operand directly.

        Raises:
            RuntimeError: Always. Call
                :meth:`~torch_openreml.covariance.matrix.Matrix.set_no_grad`
                on the operand that owns the parameter instead.
        """
        raise RuntimeError(
            "This operator only provides a view of no_grad_index. "
            "Set it on the covariance matrix that owns the parameters instead!"
        )

    def trans_params(self, params):
        """
        Apply each operand's parameter transforms to the joint parameter tensor.

        Splits ``params`` into per-operand slices, delegates transformation
        to each :class:`~torch_openreml.covariance.matrix.Matrix` operand
        via :meth:`~torch_openreml.covariance.matrix.Matrix.trans_params`,
        and concatenates the results. Fixed tensor operands are skipped.

        Args:
            params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_params`.

        Returns:
            torch.Tensor: Concatenated transformed parameter tensor.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import Sum, ScalarMatrix

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                print(x.trans_params(torch.zeros(2)))
        """
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
        """
        Evaluate each operand at the current parameters.

        Splits ``params`` into per-operand slices and calls each
        :class:`~torch_openreml.covariance.matrix.Matrix` operand to
        produce its matrix. Fixed tensor operands are included as-is.

        Args:
            params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_params`.

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
        """
        Compute the Jacobian of each operand with respect to its parameters.

        Splits ``params`` into per-operand slices, calls
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
        """dict: Mapping from operand names to operand matrices or tensors."""
        return self._operands
      
    @property
    def no_grad_index(self):
        """
        list of int: Aggregated indices of parameters excluded from gradient
        computation, derived from each operand's
        :attr:`~torch_openreml.covariance.matrix.Matrix.no_grad_index` and
        offset by the cumulative parameter count of preceding operands.
        """
        result = []
        total_num_params = 0
        
        for name, operand in self._operands.items():
            if isinstance(operand, Matrix):
                result.extend([index + total_num_params for index in operand.no_grad_index])
                total_num_params = total_num_params + operand.num_params
                
        return result
    
    @property
    def repr_dict(self):
        """dict: Key-value pairs used to build the string representation."""
        return {"operands": self.operands}
        

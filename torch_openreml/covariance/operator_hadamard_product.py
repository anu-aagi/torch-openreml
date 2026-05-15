"""
Hadamard product covariance operator.

This module provides a Hadamard (element-wise) product operator for
combining two covariance matrices, for use in linear mixed-effects models.

Classes:
    HadamardProduct:
        A Hadamard product covariance operator :math:`V = A \\odot B`.
"""

from torch_openreml.covariance.operator import Operator
import torch


class HadamardProduct(Operator):
    r"""
    Hadamard (element-wise) product of two covariance matrices.

    .. math::
        \symbf{V} = \symbf{A} \odot \symbf{B}

    Both operands must have the same shape. Either or both may be trainable
    :class:`~torch_openreml.covariance.matrix.Matrix` instances or fixed
    :class:`torch.Tensor` values.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Hadamard product operator from exactly two operands.

        Args:
            *args: Exactly two operands as positional arguments or a single
                dict. The first is :math:`\\symbf{A}`, the second
                :math:`\\symbf{B}`.
            **kwargs: Exactly two operands as keyword arguments.

        Raises:
            ValueError: If the number of operands is not exactly two.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import EquicorrelationMatrix, HadamardProduct

            n = 4
            op = HadamardProduct(a=EquicorrelationMatrix(n), b=torch.tensor([5.0]))
            free_params = torch.tensor([1.0])
            op(free_params)
        """

        super().__init__(*args, **kwargs)

        if len(self.operands) != 2:
            raise ValueError("Two operands are required")

    def _get_or_build_intermediates(self, free_params):
        built_params = self.build_params(free_params)
        cache = self.get_intermediates(built_params)

        if cache is None:
            v_groups = self.build_operands(free_params)

            a = v_groups[0]
            b = v_groups[1]
            v = a * b

            cache = {"a": a, "b": b, "v": v}

            self.set_intermediates(built_params, cache)

        return cache

    def __call__(self, free_params):
        cache = self._get_or_build_intermediates(free_params)
        v = cache["v"]
        self._shape = tuple(v.shape)

        return v

    def manual_grad(self, free_params):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` if all parameters are fixed.
        """
        grad_groups, grad_name_groups = self.operands_grad(free_params)

        cache = self._get_or_build_intermediates(free_params)
        a = cache["a"]
        b = cache["b"]

        grad = []
        grad_names = []

        da = grad_groups[0]
        if da is not None:
            grad.append(da * b)
            grad_names.extend(grad_name_groups[0])

        db = grad_groups[1]
        if db is not None:
            grad.append(a * db)
            grad_names.extend(grad_name_groups[1])

        if len(grad) > 0:
            grad = torch.cat(grad)
            return grad, grad_names
        else:
            return None, []
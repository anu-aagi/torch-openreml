"""
Sum covariance operator.

This module provides a Sum operator for combining multiple covariance
matrices additively, typically used to represent multi-component variance
structures in linear mixed-effects models.

Classes:
    Sum:
        A covariance operator representing
        :math:`V = \\sum_i A_i`, where all operands share the same shape.
"""

from torch_openreml.covariance.operator import Operator
import torch

class Sum(Operator):
    r"""
    Sum of multiple covariance matrices.

    .. math::
        \symbf{V} = \sum_{i=1}^{k} \symbf{A}_i

    where each :math:`\symbf{A}_i` is a covariance matrix of the same
    shape. This operator represents additive covariance structures,
    commonly used in linear mixed-effects models to combine multiple
    variance components (e.g., genetic, environmental, and residual).

    All operands must evaluate to matrices of identical shape. Each
    operand may be a trainable
    :class:`~torch_openreml.covariance.matrix.Matrix` or a fixed
    :class:`torch.Tensor`.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a sum operator from two or more operands.

        Args:
            *args: Two or more operands as positional arguments or a single
                dict mapping names to operands.
            **kwargs: Two or more operands as keyword arguments.

        Raises:
            ValueError: If fewer than two operands are provided.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import AR1Matrix, ScalarMatrix, Sum

            op = Sum(time=AR1Matrix(4), noise=ScalarMatrix(4))
            params = torch.tensor([0.5, 1.0, 1.0])
            op(params)
        """
          
        super().__init__(*args, **kwargs)

        if len(self.operands) < 2:
            raise ValueError("At least two operands are required")
    
    def __call__(self, params):
        v_groups = self.build_operands(params)
        v = sum(v_groups)
        self._shape = tuple(v.shape)

        return v

    def manual_grad(self, params):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        Args:
            params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_params - len(no_grad_index), *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` if all parameters are excluded from
            gradient computation.
        """
        grad_groups, grad_name_groups = self.operands_grad(params)

        grad_groups = [grad for grad in grad_groups if grad is not None]

        if len(grad_groups) > 0:
            grad = torch.cat(grad_groups)
            grad_names = [name for group in grad_name_groups for name in group]

            return grad, grad_names
        else:
            return None, []
"""
Transpose covariance operator.

This module provides a transpose operator for transposing a covariance
matrix, for use in linear mixed-effects models.

Classes:
    Transpose:
        A transpose covariance operator :math:`V = A^\\top`.
"""

from torch_openreml.covariance.operator import Operator
import torch


class Transpose(Operator):
    r"""
    Transpose of a covariance matrix.

    .. math::
        \symbf{V} = \symbf{A}^\top

    If :math:`\symbf{A}` is :math:`m \times n`, the result is an
    :math:`n \times m` matrix. The operand may be a trainable
    :class:`~torch_openreml.covariance.matrix.Matrix` instance or a fixed
    :class:`torch.Tensor` value.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a transpose operator from exactly one operand.

        Args:
            *args: Exactly one operand as a positional argument or a single
                dict. The operand is :math:`\\symbf{A}`.
            **kwargs: Exactly one operand as a keyword argument.

        Raises:
            ValueError: If the number of operands is not exactly one.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DummyMatrix, Transpose

            op = Transpose(a=DummyMatrix(["a", "b", "c", "a"]))
            op(torch.tensor([]))
        """

        super().__init__(*args, **kwargs)

        if len(self.operands) != 1:
            raise ValueError("One operand is required")

    def _get_or_build_intermediates(self, free_params):
        built_params = self.build_params(free_params)
        cache = self.get_intermediates(built_params)

        if cache is None:
            v_groups = self.build_operands(free_params)

            a = v_groups[0]
            v = a.T

            cache = {"a": a, "v": v}

            self.set_intermediates(built_params, cache)

        return cache

    def __call__(self, free_params=None):
        cache = self._get_or_build_intermediates(free_params)
        v = cache["v"]
        self._shape = tuple(v.shape)

        return v

    def inverse(self, free_params=None):
        """
        Return the inverse of the composite matrix from its closed form.

        The composite is :math:`\\symbf{V} = \\symbf{A}^\\top`, so its inverse
        is
        :math:`\\symbf{V}^{-1} = (\\symbf{A}^{-1})^\\top`: the operand is
        inverted through its own
        :meth:`~torch_openreml.covariance.matrix.Matrix.inverse` and the
        result is transposed, so the composite is never inverted densely.

        .. warning::

            The operand must be square, otherwise the composite has no
            inverse and raises.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary. If omitted, default values are used.
                Default: ``None``.

        Returns:
            torch.Tensor: The ``n x n`` inverse of the composite matrix.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import AR1Matrix, Transpose

            op = Transpose(a=AR1Matrix(3))
            free_params = torch.tensor([0.0, 0.5])
            op.inverse(free_params)
        """
        inverses = self.operands_inverse(free_params)

        return inverses[0].T

    def manual_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        If :math:`\\symbf{V} = \\symbf{A}^\\top`, the gradient with respect
        to :math:`\\theta_{\\symbf{A}}` is the transpose of the operand's own
        Jacobian:

        .. math::
            \\frac{\\partial \\symbf{V}}{\\partial \\theta_{\\symbf{A}}}
            = \\left(\\frac{\\partial \\symbf{A}}{\\partial \\theta_{\\symbf{A}}}\\right)^\\top

        The per-operand Jacobian from
        :meth:`~torch_openreml.covariance.operator.Operator.operands_grad`
        is transposed over its last two dimensions.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.
                If omitted, default values are used. Default: ``None``.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and ``grad_names`` is a list
            of the corresponding parameter names. Returns ``(None, [])`` if
            the operand has no free parameters.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the
                wrong length, or if ``free_params`` is a dict with missing
                or unexpected keys.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import LowerTriangularMatrix, Transpose

            op = Transpose(a=LowerTriangularMatrix(3, 2))
            free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
            grad, grad_names = op.manual_grad(free_params)
            grad

        .. jupyter-execute::

            grad_names
        """
        grad_groups, grad_name_groups = self.operands_grad(free_params)

        grad = grad_groups[0]

        if grad is not None:
            return grad.mT, grad_name_groups[0]
        else:
            return None, []

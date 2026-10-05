"""
Inverse covariance operator.

This module provides an inverse operator for inverting a covariance
matrix, for use in linear mixed-effects models.

Classes:
    Inverse:
        An inverse covariance operator :math:`V = A^{-1}`.
"""

from torch_openreml.covariance.operator import Operator
import torch


class Inverse(Operator):
    r"""
    Inverse of a covariance matrix.

    .. math::
        \symbf{V} = \symbf{A}^{-1}

    The operand must be square. The operand may be a trainable
    :class:`~torch_openreml.covariance.matrix.Matrix` instance.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an inverse operator from exactly one operand.

        Args:
            *args: Exactly one operand as a positional argument or a single
                dict. The operand is :math:`\\symbf{A}`.
            **kwargs: Exactly one operand as a keyword argument.

        Raises:
            ValueError: If the number of operands is not exactly one.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import ScalarMatrix, Inverse

            op = Inverse(a=ScalarMatrix(3))
            free_params = torch.tensor([0.5])
            op(free_params)
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
            v = torch.linalg.inv(a)

            cache = {"a": a, "v": v}

            self.set_intermediates(built_params, cache)

        return cache

    def __call__(self, free_params=None):
        cache = self._get_or_build_intermediates(free_params)
        v = cache["v"]
        self._shape = tuple(v.shape)

        return v

    def manual_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        Differentiating :math:`\\symbf{A} \\symbf{A}^{-1} = \\symbf{I}` gives
        the gradient with respect to :math:`\\theta_{\\symbf{A}}`:

        .. math::
            \\frac{\\partial \\symbf{V}}{\\partial \\theta_{\\symbf{A}}}
            = -\\symbf{A}^{-1}
            \\frac{\\partial \\symbf{A}}{\\partial \\theta_{\\symbf{A}}}
            \\symbf{A}^{-1}

        The inverse from the forward pass is reused, so no further matrix
        inversion is performed.

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
            from torch_openreml.covariance import ScalarMatrix, Inverse

            op = Inverse(a=ScalarMatrix(3))
            free_params = torch.tensor([0.5])
            grad, grad_names = op.manual_grad(free_params)
            grad

        .. jupyter-execute::

            grad_names
        """
        grad_groups, grad_name_groups = self.operands_grad(free_params)

        v = self._get_or_build_intermediates(free_params)["v"]

        da = grad_groups[0]

        if da is not None:
            return -v @ da @ v, grad_name_groups[0]
        else:
            return None, []

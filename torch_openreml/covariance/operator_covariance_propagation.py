"""
Covariance propagation operator.

This module provides a covariance propagation operator that transforms
a covariance matrix through a design matrix, for use in linear
mixed-effects models.

Classes:
    CovariancePropagation:
        A covariance propagation operator :math:`V = Z G Z^\\top`.
"""

from torch_openreml.covariance.operator import Operator
import torch


class CovariancePropagation(Operator):
    r"""
    Covariance propagation operator.

    .. math::
        \symbf{V} = \symbf{Z} \symbf{G} \symbf{Z}^\top

    Propagates the covariance matrix :math:`\symbf{G}` through the design
    matrix :math:`\symbf{Z}`. The first operand is treated as
    :math:`\symbf{Z}` and the second as :math:`\symbf{G}`. Either or both
    may be trainable :class:`~torch_openreml.covariance.matrix.Matrix`
    instances or fixed :class:`torch.Tensor` values.

    This structure arises naturally in linear mixed-effects models where
    :math:`\symbf{Z}` is the random-effect design matrix and
    :math:`\symbf{G}` is the random-effect covariance matrix, giving the
    random-effect contribution :math:`\symbf{Z}\symbf{G}\symbf{Z}^\top`
    to the marginal covariance.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a covariance propagation operator from exactly two operands.

        Args:
            *args: Exactly two operands as positional arguments or a single
                dict. The first is :math:`\\symbf{Z}`, the second
                :math:`\\symbf{G}`.
            **kwargs: Exactly two operands as keyword arguments.

        Raises:
            ValueError: If the number of operands is not exactly two.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, CovariancePropagation

            z = DummyMatrix(["a", "b", "c", "a"])
            z()

        .. jupyter-execute::

            g = DiagonalMatrix(3)
            op = CovariancePropagation(z=z, g=g)
            free_params = torch.tensor([0.0, 0.5, 1.0])
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

            z = v_groups[0]
            g = v_groups[1]
            v = z @ g @ z.T

            cache = {"z": z, "g": g, "v": v}

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

        Applies the product rule to :math:`\\symbf{V} = \\symbf{Z} \\symbf{G} \\symbf{Z}^\\top`:

        - With respect to :math:`\\theta_{\\symbf{Z}}`:
          :math:`\\frac{\\partial \\symbf{V}}{\\partial \\theta} =
          \\frac{\\partial \\symbf{Z}}{\\partial \\theta} \\symbf{G} \\symbf{Z}^\\top +
          \\symbf{Z} \\symbf{G} \\frac{\\partial \\symbf{Z}^\\top}{\\partial \\theta}`
          (two terms because :math:`\\symbf{Z}` appears twice).
        - With respect to :math:`\\theta_{\\symbf{G}}`:
          :math:`\\frac{\\partial \\symbf{V}}{\\partial \\theta} =
          \\symbf{Z} \\frac{\\partial \\symbf{G}}{\\partial \\theta} \\symbf{Z}^\\top`
          (linear in :math:`\\symbf{G}`).

        Per-operand Jacobians from
        :meth:`~torch_openreml.covariance.operator.Operator.operands_grad`
        are propagated through the same structure.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and ``grad_names`` is a list
            of the corresponding parameter names. Returns ``(None, [])`` if
            all parameters are fixed.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the
                wrong length, or if ``free_params`` is a dict with missing
                or unexpected keys.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, CovariancePropagation

            z = DummyMatrix(["a", "b", "c", "a"])
            z()

        .. jupyter-execute::

            g = DiagonalMatrix(3)
            op = CovariancePropagation(z=z, g=g)
            free_params = torch.tensor([0.0, 0.5, 1.0])
            grad, grad_names = op.manual_grad(free_params)
            grad

        .. jupyter-execute::

            grad_names
        """
        grad_groups, grad_name_groups = self.operands_grad(free_params)

        cache = self._get_or_build_intermediates(free_params)
        z = cache["z"]
        g = cache["g"]

        grad_list = []
        grad_names = []

        dz = grad_groups[0]
        if dz is not None:
            grad_z = dz @ g @ z.T + z @ g @ dz.mT
            grad_list.append(grad_z)
            grad_names.extend(grad_name_groups[0])

        dg = grad_groups[1]
        if dg is not None:
            grad_g = z @ dg @ z.T
            grad_list.append(grad_g)
            grad_names.extend(grad_name_groups[1])

        if len(grad_list) > 0:
            grad = torch.cat(grad_list)
            return grad, grad_names
        else:
            return None, []
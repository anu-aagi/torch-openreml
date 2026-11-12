"""
Marginal covariance operator.

This module provides a covariance operator for the marginal covariance of a
linear mixed-effects model,

.. math::

    \\symbf{V} = \\symbf{R} + \\symbf{Z}\\symbf{G}\\symbf{Z}^{\\top}.

Classes:
    MarginalCovariance:
        Marginal covariance built from design, random-effect and residual
        operands.
"""

import torch

from torch_openreml.covariance.operator import Operator


class MarginalCovariance(Operator):
    r"""
    Marginal covariance operator.

    .. math::

        \symbf{V}
        = \symbf{R} + \symbf{Z}\symbf{G}\symbf{Z}^{\top}

    The operator has exactly three named operands:

    - ``Z``: the incidence matrix, an :math:`n \times q` tensor or
      :class:`~torch_openreml.covariance.matrix.Matrix`;
    - ``G``: the random-effect covariance, a :math:`q \times q`
      :class:`~torch_openreml.covariance.matrix.Matrix` or fixed tensor;
    - ``R``: the residual covariance, an :math:`n \times n`
      :class:`~torch_openreml.covariance.matrix.Matrix` or fixed tensor.

    Parameters are namespaced by these operand names. A parameter
    ``sigma^2`` of a term ``gen`` inside a block-diagonal ``G`` is exposed as
    ``G/gen/sigma^2``. If ``Z`` is itself parameterised, its parameters are
    exposed as ``Z/...`` and come before ``G`` and ``R``, preserving the
    parameter order :math:`[\\symbf{Z}, \\symbf{G}, \\symbf{R}]`.

    This is the operator used by
    :class:`~torch_openreml.mixed_model_reml.MixedModelREML`. It is not a
    general sum: it enforces the design/covariance shapes and evaluates
    :math:`\\symbf{R} + \\symbf{Z}\\symbf{G}\\symbf{Z}^{\\top}` directly.
    """

    def __init__(self, *args, **kwargs):
        r"""
        Initialize the marginal covariance.

        Args:
            *args: Either exactly three positional operands ``Z``, ``G`` and
                ``R``, or a single dictionary with keys ``"Z"``, ``"G"`` and
                ``"R"``.
            **kwargs: The named operands ``Z``, ``G`` and ``R``.

        Raises:
            ValueError: If positional and keyword operands are mixed, or if
                the operands are not exactly ``Z``, ``G`` and ``R``.
            TypeError: If an operand is not a
                :class:`~torch_openreml.covariance.matrix.Matrix` or tensor,
                or if no operand is a ``Matrix``.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, ScalarMatrix, MarginalCovariance

            z = DummyMatrix(["a", "b", "c", "a"])
            g = DiagonalMatrix(3)
            r = ScalarMatrix(4)
            op = MarginalCovariance(Z=z, G=g, R=r)
            op(torch.zeros(4))
        """
        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError("Operands must be positional or keyword, not both!")

        if len(args) == 1 and isinstance(args[0], dict):
            operands = args[0]
        elif len(args) == 3:
            operands = {"Z": args[0], "G": args[1], "R": args[2]}
        elif len(args) == 0:
            operands = kwargs
        else:
            raise ValueError("MarginalCovariance requires exactly Z, G and R!")

        if set(operands) != {"Z", "G", "R"}:
            raise ValueError("MarginalCovariance requires exactly the operands Z, G and R!")

        super().__init__(**operands)

    def _get_or_build_intermediates(self, free_params):
        built_params = self.build_params(free_params)
        cache = self.get_intermediates(built_params)

        if cache is None:
            z, g, r = self.build_operands(free_params)

            if z.ndim != 2 or g.ndim != 2 or r.ndim != 2:
                raise ValueError("Z, G and R must all be two-dimensional!")
            if z.shape[0] != r.shape[0] or r.shape[0] != r.shape[1]:
                raise ValueError("Z and R must have the same number of rows, and R must be square!")
            if z.shape[1] != g.shape[0] or g.shape[0] != g.shape[1]:
                raise ValueError("Z.shape[1] and both dimensions of G must agree!")

            cache = {"z": z, "g": g, "r": r, "v": r + z @ g @ z.T}
            self.set_intermediates(built_params, cache)

        return cache

    def __call__(self, free_params=None):
        r"""
        Evaluate :math:`\symbf{R} + \symbf{Z}\symbf{G}\symbf{Z}^{\top}`.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary. If omitted, default values are used.
                Default: ``None``.

        Returns:
            torch.Tensor: The :math:`n \times n` marginal covariance.

        Raises:
            ValueError: If ``Z``, ``G`` or ``R`` is not two-dimensional, if
                ``R`` is not square, or if ``Z.shape[1]`` and the dimensions
                of ``G`` disagree.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, ScalarMatrix, MarginalCovariance

            z = DummyMatrix(["a", "b", "c", "a"])
            g = DiagonalMatrix(3)
            r = ScalarMatrix(4)
            op = MarginalCovariance(Z=z, G=g, R=r)
            op(torch.zeros(4))
        """
        v = self._get_or_build_intermediates(free_params)["v"]
        self._shape = tuple(v.shape)
        return v

    def manual_grad(self, free_params=None):
        r"""
        Compute the Jacobian of :meth:`__call__` analytically.

        The derivatives are

        .. math::

            \frac{\partial \symbf{V}}{\partial \theta_{\symbf{Z}}}
            &= \dot{\symbf{Z}}\symbf{G}\symbf{Z}^{\top}
             + \symbf{Z}\symbf{G}\dot{\symbf{Z}}^{\top}, \\
            \frac{\partial \symbf{V}}{\partial \theta_{\symbf{G}}}
            &= \symbf{Z}\dot{\symbf{G}}\symbf{Z}^{\top}, \\
            \frac{\partial \symbf{V}}{\partial \theta_{\symbf{R}}}
            &= \dot{\symbf{R}}.

        Per-operand Jacobians from
        :meth:`~torch_openreml.covariance.operator.Operator.operands_grad`
        are propagated through these expressions.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary. If omitted, default values are used.
                Default: ``None``.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` has shape
            ``(num_free_params, n, n)`` and ``grad_names`` is the matching
            list of namespaced parameter names. Returns ``(None, [])`` when
            no operand has free parameters.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor or a dict.
            ValueError: If ``free_params`` has the wrong length or contains
                unexpected parameter names.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, ScalarMatrix, MarginalCovariance

            z = DummyMatrix(["a", "b", "c", "a"])
            g = DiagonalMatrix(3)
            r = ScalarMatrix(4)
            op = MarginalCovariance(Z=z, G=g, R=r)
            grad, grad_names = op.manual_grad(torch.zeros(4))
            grad

        .. jupyter-execute::

            grad_names
        """
        cache = self._get_or_build_intermediates(free_params)
        z = cache["z"]
        g = cache["g"]

        grad_groups, name_groups = self.operands_grad(free_params)

        grads = []
        names = []

        for (operand_name, _), jac, operand_names in zip(self.operands.items(), grad_groups, name_groups):
            if jac is None:
                continue

            if operand_name == "Z":
                grads.append(jac @ (g @ z.T) + (z @ g) @ jac.mT)
            elif operand_name == "G":
                grads.append(z @ jac @ z.T)
            elif operand_name == "R":
                grads.append(jac)

            names.extend(operand_names)

        if not grads:
            return None, []

        return torch.cat(grads), names

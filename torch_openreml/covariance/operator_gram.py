"""
Gram covariance operator.

This module provides a Gram operator that computes :math:`\symbf{X}^\top \symbf{X}`
or :math:`\symbf{X} \symbf{X}^\top` from a supplied matrix,
for use in linear mixed-effects models.

Classes:
    OperatorGram:
        A covariance operator representing a Gram matrix.
"""

from torch_openreml.covariance.operator import Operator
from torch_openreml.covariance.matrix import Matrix
import torch


class OperatorGram(Operator):
    r"""
    Gram matrix operator: :math:`\symbf{X}^\top \symbf{X}` or
    :math:`\symbf{X} \symbf{X}^\top`.

    Given a matrix :math:`\symbf{X}` (which may be a fixed
    :class:`torch.Tensor` or a trainable
    :class:`~torch_openreml.covariance.matrix.Matrix`), this operator
    computes its Gram product in the direction specified at construction:

    .. math::

        \symbf{V} = \begin{cases}
            \symbf{X}^\top \symbf{X} & \text{if } gram\_type = \texttt{"xtx"} \\
            \symbf{X} \symbf{X}^\top & \text{if } gram\_type = \texttt{"xxt"}
        \end{cases}

    If :math:`\symbf{X}` has shape ``(n, m)``, then ``"xtx"`` yields an
    ``(m, m)`` matrix and ``"xxt"`` yields an ``(n, n)`` matrix.

    When :math:`\symbf{X}` is a
    :class:`~torch_openreml.covariance.matrix.Matrix`, its parameters are
    exposed through this operator and gradients are computed analytically
    via the product rule.
    """

    def __init__(self, x, gram_type="xtx"):
        """
        Initialise a Gram operator.

        Args:
            x (:class:`torch.Tensor` or :class:`~torch_openreml.covariance.matrix.Matrix`):
                The input matrix of shape ``(n, m)``.
            gram_type (str): Which Gram product to compute. Must be one of
                ``"xtx"`` (:math:`\\symbf{X}^\\top \\symbf{X}`) or
                ``"xxt"`` (:math:`\\symbf{X} \\symbf{X}^\\top`).
                Default: ``"xtx"``.

        Raises:
            ValueError: If ``gram_type`` is not ``"xtx"`` or ``"xxt"``.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import OperatorGram, LowerTriangularMatrix

            x = LowerTriangularMatrix(3, 2)
            op = OperatorGram(x, gram_type="xtx")
            op()

        .. jupyter-execute::

            op_xxt = OperatorGram(x, gram_type="xxt")
            op_xxt()
        """
        if gram_type not in ("xtx", "xxt"):
            raise ValueError(
                f"gram_type must be 'xtx' or 'xxt', got '{gram_type}'"
            )
        self._gram_type = gram_type

        super().__init__(x=x)

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        v_groups = self.build_operands(free_params)
        x = v_groups[0]

        if self._gram_type == "xtx":
            v = x.T @ x
        else:
            v = x @ x.T

        self._shape = tuple(v.shape)
        return v

    def manual_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using the product rule.

        For :math:`\\symbf{V} = \\symbf{X}^\\top \\symbf{X}`:

        .. math::
            \\frac{\\partial \\symbf{V}}{\\partial \\theta_k}
            = \\symbf{X}^\\top \\frac{\\partial \\symbf{X}}{\\partial \\theta_k}
            + \\left(\\frac{\\partial \\symbf{X}}{\\partial \\theta_k}\\right)^\\top
            \\symbf{X}

        For :math:`\\symbf{V} = \\symbf{X} \\symbf{X}^\\top`:

        .. math::
            \\frac{\\partial \\symbf{V}}{\\partial \\theta_k}
            = \\frac{\\partial \\symbf{X}}{\\partial \\theta_k} \\symbf{X}^\\top
            + \\symbf{X} \\left(\\frac{\\partial \\symbf{X}}{\\partial \\theta_k}\\right)^\\top

        If :math:`\\symbf{X}` has no trainable parameters, returns
        ``(None, [])``.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.
                If omitted, default values are used. Default: ``None``.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` if all parameters are fixed.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import OperatorGram, LowerTriangularMatrix

            x = LowerTriangularMatrix(3, 2)
            op = OperatorGram(x, gram_type="xtx")
            free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
            grad, grad_names = op.manual_grad(free_params)
            grad

        .. jupyter-execute::

            grad_names

        .. jupyter-execute::

            op_xxt = OperatorGram(x, gram_type="xxt")
            grad_xxt, grad_names_xxt  = op_xxt.manual_grad(free_params)
            grad_xxt
        """
        if free_params is None:
            free_params = self.free_param_defaults

        v_groups = self.build_operands(free_params)
        x = v_groups[0]

        grad_groups, grad_name_groups = self.operands_grad(free_params)
        x_grad = grad_groups[0]

        if x_grad is None:
            return None, []

        if self._gram_type == "xtx":
            grad = torch.zeros(x_grad.shape[0], x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
            for k in range(len(grad)):
                grad[k] = x.T @ x_grad[k] + x_grad[k].T @ x
        else:
            grad = torch.zeros(x_grad.shape[0], x.shape[0], x.shape[0], device=x.device, dtype=x.dtype)
            for k in range(len(grad)):
                grad[k] = x_grad[k] @ x.T + x @ x_grad[k].T

        grad_names = [name for group in grad_name_groups for name in group]

        return grad, grad_names

    @property
    def gram_type(self):
        """str: The Gram product type (``"xtx"`` or ``"xxt"``)."""
        return self._gram_type

    @property
    def repr_dict(self):
        """dict: Key-value pairs used to build the string representation."""
        return {"operands": self.operands, "gram_type": self.gram_type}

"""
Augment covariance operator.

This module provides an Augment operator for binding multiple covariance
matrices side by side (column-wise), for use in linear mixed-effects models.

Classes:
    Augment:
        A column-wise augmented covariance operator
        :math:`V = [A_0 \\mid A_1 \\mid \\ldots]`.
"""

from torch_openreml.covariance.operator import Operator
import torch


class Augment(Operator):
    r"""
    Column-wise augmentation (horizontal concatenation) of covariance matrices.

    .. math::
        \symbf{V} = [\symbf{A}_0 \mid \symbf{A}_1 \mid \ldots]

    Each operand is placed side by side (column-wise). All operands must
    have the same number of rows. Each operand may be a trainable
    :class:`~torch_openreml.covariance.matrix.Matrix` or a fixed
    :class:`torch.Tensor`.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an augment operator from two or more operands.

        Args:
            *args: Two or more operands as positional arguments or a single
                dict mapping names to operands.
            **kwargs: Two or more operands as keyword arguments.

        Raises:
            ValueError: If fewer than two operands are provided.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import ScalarMatrix, Augment

            op = Augment(A=ScalarMatrix(3), B=ScalarMatrix(3))
            free_params = torch.tensor([0.5, 1.0])
            op(free_params)
        """

        super().__init__(*args, **kwargs)

        if len(self.operands) < 2:
            raise ValueError("At least two operands are required")

    def _get_or_build_intermediates(self, free_params):
        built_params = self.build_params(free_params)
        cache = self.get_intermediates(built_params)

        if cache is None:
            v_groups = self.build_operands(free_params)
            v = torch.cat(v_groups, dim=1)

            col_offsets = []
            c = 0
            for vg in v_groups:
                cols = vg.shape[1]
                col_offsets.append((c, c + cols))
                c += cols

            cache = {
                "v_groups": v_groups,
                "v": v,
                "col_offsets": col_offsets
            }

            self.set_intermediates(built_params, cache)

        return cache

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        cache = self._get_or_build_intermediates(free_params)
        v = cache["v"]
        self._shape = tuple(v.shape)
        return v

    def manual_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        Each parameter's gradient is a matrix of the same shape as
        :math:`\\symbf{V}`, with the operand's block placed in its
        corresponding columns and zeros elsewhere.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.
                If omitted, default values are used. Default: ``None``.

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
            from torch_openreml.covariance import ScalarMatrix, Augment

            op = Augment(A=ScalarMatrix(3), B=ScalarMatrix(3))
            free_params = torch.tensor([0.5, 1.0])
            grad, grad_names = op.manual_grad(free_params)
            grad

        .. jupyter-execute::

            grad_names
        """
        if free_params is None:
            free_params = self.free_param_defaults
        grad_groups, grad_name_groups = self.operands_grad(free_params)

        col_offsets = self._get_or_build_intermediates(free_params)["col_offsets"]
        total_cols = col_offsets[-1][1]

        grad_list = []
        grad_names = []

        for grad, names, (c0, c1) in zip(grad_groups, grad_name_groups, col_offsets):
            if grad is not None:
                tmp = torch.zeros(grad.shape[0], grad.shape[1], total_cols,
                                  dtype=free_params.dtype, device=free_params.device)
                tmp[:, :, c0:c1] = grad
                grad_list.append(tmp)
                grad_names.extend(names)

        if len(grad_list) > 0:
            return torch.cat(grad_list), grad_names
        else:
            return None, []

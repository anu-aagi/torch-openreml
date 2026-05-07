"""
Block diagonal covariance matrix operator.

This module provides a block diagonal covariance matrix formed from two
or more constituent covariance matrices, for use in linear mixed-effects
models.

Classes:
    BlockDiagonal:
        A block diagonal covariance matrix operator.
"""

from torch_openreml.covariance.operator import Operator
import torch


class BlockDiagonal(Operator):
    r"""
   Block diagonal covariance matrix formed from two or more operands.

   .. math::
       \symbf{V} = \mathrm{blockdiag}(\symbf{V}_0, \symbf{V}_1, \ldots)

   Each operand contributes a contiguous block along the diagonal.
   Parameters and gradients are namespaced by operand name and aggregated
   into a single joint parameter tensor, following the convention of
   :class:`~torch_openreml.covariance.operator.Operator`.
   """

    def __init__(self, *args, **kwargs):
        """
        Initialize a block diagonal operator from two or more operands.

        Args:
           \*args: Two or more operands as positional arguments, each a
               :class:`~torch_openreml.covariance.matrix.Matrix` or
               :class:`torch.Tensor`. A single dict argument is also accepted.

           \**kwargs: Two or more operands as keyword arguments.

        Raises:
           ValueError: If fewer than two operands are provided.

        Example:

        .. jupyter-execute::

           import torch
           from torch_openreml.covariance import ScalarMatrix, DiagonalMatrix, BlockDiagonal

           block = BlockDiagonal(
               residual=ScalarMatrix(3),
               random=DiagonalMatrix(2)
           )
           params = torch.tensor([0.5, 0.0, 1.0])
           block(params)
        """

        super().__init__(*args, **kwargs)

        if len(self.operands) < 2:
            raise ValueError("At least two operands are required")

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            v_groups = self.build_operands(params)
            v = torch.block_diag(*v_groups)

            row_offsets = []
            col_offsets = []

            n = 0
            m = 0
            for vg in v_groups:
                rows, cols = vg.shape
                row_offsets.append((n, n + rows))
                col_offsets.append((m, m + cols))
                n += rows
                m += cols

            cache = {
                "v_groups": v_groups,
                "v": v,
                "row_offsets": row_offsets,
                "col_offsets": col_offsets
            }

            self.set_intermediates(params, cache)

        return cache

    def __call__(self, params):
        cache = self._get_or_build_intermediates(params)
        v = cache["v"]
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

        cache = self._get_or_build_intermediates(params)
        v = cache["v"]
        v_groups = cache["v_groups"]
        row_offsets = cache["row_offsets"]
        col_offsets = cache["col_offsets"]

        grad_list = []
        grad_names = []

        for i, grad in enumerate(grad_groups):
            if grad is None:
                continue

            (r0, r1) = row_offsets[i]
            (c0, c1) = col_offsets[i]

            tmp = torch.zeros((grad.shape[0],) + tuple(v.shape),
                              dtype=params.dtype,
                              device=params.device)

            tmp[:, r0:r1, c0:c1] = grad

            grad_list.append(tmp)
            grad_names.extend(grad_name_groups[i])

        if len(grad_list) > 0:
            grad = torch.cat(grad_list)
            return grad, grad_names
        else:
            return None, []
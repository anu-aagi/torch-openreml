"""
Lower triangular matrix.

This module provides a lower triangular matrix parameterised by its
lower-triangular entries, for use in linear mixed-effects models.

All lower-triangular entries (including the diagonal) are unconstrained.
The matrix may be non-square (``n x m``).

Classes:
    LowerTriangularMatrix:
        A lower triangular matrix :math:`\\symbf{L}` where :math:`L_{ij} = 0`
        for :math:`j > i` and :math:`j \\ge m`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformIdentity
import torch


class LowerTriangularMatrix(Matrix):
    r"""
    Lower triangular matrix parameterised by its lower-triangular entries.

    The matrix has free parameters for all entries on or below the diagonal
    (i.e., :math:`j \\le i` and :math:`j < m`). Entries above the diagonal
    are fixed at zero:

    .. math::
        \symbf{L}_{ij} = \\begin{cases}
            \\theta_{ij} & i \\ge j \\;\\text{and}\\; j < m \\\\
            0 & i < j
        \\end{cases}

    All parameters (including diagonal entries) are unconstrained and use
    :class:`~torch_openreml.covariance.transform.TransformIdentity` by default.
    """

    def __init__(self, n, m, param_specs=None):
        """
        Initialize a lower triangular matrix of size ``n x m``.

        By default, the matrix has free parameters for all entries on or below
        the diagonal, all using :class:`~torch_openreml.covariance.transform.TransformIdentity`
        (unconstrained).

        Args:
            n (int): Number of rows.
            m (int): Number of columns.
            param_specs (dict): Parameter specifications. Keys should be strings
                representing parameter names. Values should be dictionaries
                containing the specification for each parameter. Each specification
                dictionary should contain the keys ``"fixed"``, ``"default"``, and ``"trans"``,
                representing whether the parameter is fixed or free (bool), the
                default value (1D torch.Tensor), and the transform (:class:`~torch_openreml.covariance.transform.Transform`),
                respectively.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import LowerTriangularMatrix

            mat = LowerTriangularMatrix(3, 2)
            mat

        .. jupyter-execute::

            free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
            mat(free_params)

        .. jupyter-execute::

            mat.grad(free_params)

        """
        param_specs = param_specs or {
            f"L_{i}_{j}": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformIdentity()
            } for i in range(n) for j in range(min(i + 1, m))
        }
        super().__init__((n, m), param_specs)

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        tril_entries = self.build_params(free_params)

        mat = torch.zeros(self.shape[0], self.shape[1], device=tril_entries.device, dtype=tril_entries.dtype)

        i, j = torch.tril_indices(self.shape[0], self.shape[1], device=tril_entries.device)
        mat[i, j] = tril_entries

        return mat

    def manual_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.
                If omitted, default values are used. Default: ``None``.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` if all parameters are fixed.
        """
        if free_params is None:
            free_params = self.free_param_defaults
        if len(free_params) == 0:
            return None, []

        free_params = self.build_params(free_params, include_fixed=False, trans=False)
        device = free_params.device
        dtype = free_params.dtype

        grad = torch.zeros(free_params.shape[0], self.shape[0], self.shape[1], device=device, dtype=dtype)
        i_idx, j_idx = torch.tril_indices(self.shape[0], self.shape[1], device=device)

        free_mask = torch.zeros(self.num_params, dtype=torch.bool, device=device)
        free_mask[self.free_param_index] = True

        i_idx = i_idx[free_mask]
        j_idx = j_idx[free_mask]

        trans_grad = self.trans_grad(free_params)

        for k in range(len(grad)):
            grad[k, i_idx[k], j_idx[k]] = trans_grad[k]

        return grad, self.free_param_names

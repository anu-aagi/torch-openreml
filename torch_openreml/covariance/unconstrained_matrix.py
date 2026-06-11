"""
Unconstrained symmetric covariance matrix.

This module provides a full symmetric covariance matrix parameterised by
its lower-triangular entries, for use in linear mixed-effects models.

Diagonal entries are transformed to positive values, while off-diagonal
entries are unconstrained. The resulting matrix is symmetric but not
guaranteed to be positive definite.

Classes:
    UnconstrainedMatrix:
        A symmetric matrix :math:`\\symbf{V}` built from its lower-triangular
        entries :math:`(L_{ij})_{j \\le i}` with :math:`\\symbf{V} = \\symbf{L} + \\symbf{L}^\\top - \\mathrm{diag}(\\symbf{L})`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformIdentity
import torch


class UnconstrainedMatrix(Matrix):
    r"""
    Full symmetric covariance matrix parameterised by lower-triangular entries.

    The matrix is built from the lower triangle (including the diagonal),
    then mirrored to the upper triangle to ensure symmetry:

    .. math::
        \symbf{V}_{ij} = \begin{cases}
            \theta_{ij} & i \ge j \\
            \theta_{ji} & i < j
        \end{cases}

    Diagonal entries (:math:`i = j`) are transformed to positive values via
    :class:`~torch_openreml.covariance.transform.TransformExpPow2` by default.
    Off-diagonal entries (:math:`i > j`) are unconstrained and use
    :class:`~torch_openreml.covariance.transform.TransformIdentity`.

    Note:
        This parameterisation ensures symmetry but does **not** guarantee
        positive definiteness. For a positive-definite covariance matrix,
        consider a Cholesky-based parameterisation.
    """

    def __init__(self, n, param_specs=None):
        """
        Initialize an unconstrained symmetric matrix of size ``n x n``.

        By default, the matrix has :math:`n(n+1)/2` free parameters: one for
        each lower-triangular entry. Diagonal entries are parameterised on the
        positive real line (via :class:`~torch_openreml.covariance.transform.TransformExpPow2`);
        off-diagonal entries are unconstrained (via
        :class:`~torch_openreml.covariance.transform.TransformIdentity`).

        Args:
            n (int): Matrix dimension.
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
            from torch_openreml.covariance import UnconstrainedMatrix

            mat = UnconstrainedMatrix(3)
            mat

        .. jupyter-execute::

            free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3, 0.4])
            mat(free_params)

        .. jupyter-execute::

            mat.grad(free_params)

        """
        param_specs = param_specs or {
            f"sigma^2_{i}_{j}": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformExpPow2() if i == j else TransformIdentity()
            } for i in range(n) for j in range(i + 1)
        }
        super().__init__((n, n), param_specs)

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        tril_entries = self.build_params(free_params)

        mat = torch.zeros(self.shape[0], self.shape[1], device=tril_entries.device, dtype=tril_entries.dtype)

        i, j = torch.tril_indices(self.shape[0], self.shape[1], device=tril_entries.device)
        mat[i, j] = tril_entries
        mat[j, i] = tril_entries

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
            grad[k, j_idx[k], i_idx[k]] = trans_grad[k]

        return grad, self.free_param_names
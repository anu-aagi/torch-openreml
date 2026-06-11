"""
Diagonal covariance matrix.

This module provides a diagonal covariance matrix with one variance
parameter per diagonal entry, for use in linear mixed-effects models.

Classes:
    DiagonalMatrix:
        A diagonal covariance matrix :math:`V = \\mathrm{diag}(\\sigma^2)`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2
import torch

class DiagonalMatrix(Matrix):
    r"""
    Diagonal covariance matrix with one variance parameter per entry.

    .. math::
        \symbf{V} = \mathrm{diag}(\sigma^2_0, \ldots, \sigma^2_{n-1})

    Each diagonal entry is parameterised by a single unconstrained scalar
    transformed to a positive variance via :class:`~torch_openreml.covariance.transform.TransformExpPow2`
    by default. Off-diagonal entries are always zero.
    """
  
    def __init__(self, n, param_specs=None):
        """
        Initialize a diagonal covariance matrix of size ``n x n``.

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
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(3)
            mat

        .. jupyter-execute::

            free_params = torch.tensor([0.0, 0.5, 1.0])
            mat(free_params)

        .. jupyter-execute::

            mat.grad(free_params)

        """
        param_specs = param_specs or {
            f"sigma^2_{i}": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformExpPow2()
            } for i in range(n)
        }
        super().__init__((n, n), param_specs)

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        sigma2 = self.build_params(free_params)
        
        return torch.diag(sigma2)

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

        grad = torch.zeros(free_params.shape[0], self.shape[0], self.shape[0], device=device, dtype=dtype)
        idx = torch.arange(self.shape[0], device=device)

        mask = torch.zeros(self.shape[0], dtype=torch.bool, device=device)
        mask[self.free_param_index] = True
        idx = idx[mask]

        grad[:, idx, idx] = self.trans_grad(free_params)

        return grad, self.free_param_names


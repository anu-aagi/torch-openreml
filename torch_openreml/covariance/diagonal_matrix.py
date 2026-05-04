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
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        """
        Initialize a diagonal covariance matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.
            param_names (list of str, optional): Names for the ``n`` variance
                parameters. Defaults to ``["sigma^2_0", ..., "sigma^2_{n-1}"]``.
            trans (list of Transform, optional): Transforms applied to each
                parameter. Defaults to ``[TransformExpPow2()]``, broadcast
                across all parameters.
            no_grad_index (list of int, optional): Indices of parameters to
                exclude from gradient computation.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix

            mat = DiagonalMatrix(3)
            params = torch.tensor([0.0, 0.5, 1.0])
            print(mat(params))

            print(mat.grad(params))
        """
        param_names = param_names or [f"sigma^2_{i}" for i in range(n)]
        trans = trans or [TransformExpPow2()]
        super().__init__((n, n), param_names, trans, no_grad_index)

    def __call__(self, params):
        sigma2 = self.trans_params(params)
        
        return torch.diag(sigma2)

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
        if len(self.no_grad_index) == self.num_params:
            return None, []

        device, dtype = self.check_params(params)

        grad = torch.zeros(self.shape[0], self.shape[0], self.shape[0], device=device, dtype=dtype)
        idx = torch.arange(self.shape[0], device=device)
        grad[idx, idx, idx] = self.trans_grad(params)

        mask = torch.ones(self.shape[0], dtype=torch.bool, device=device)
        mask[self.no_grad_index] = False

        grad = grad[mask]
        grad_names = [name for i, name in enumerate(self.param_names) if i not in self.no_grad_index]

        return grad, grad_names


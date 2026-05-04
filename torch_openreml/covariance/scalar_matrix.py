"""
Scalar covariance matrix.

This module provides a scaled identity covariance matrix with a single
shared variance parameter, for use in linear mixed-effects models.

Classes:
    ScalarMatrix:
        A scaled identity covariance matrix :math:`V = \\sigma^2 I_n`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2
import torch

class ScalarMatrix(Matrix):
    r"""
    Scaled identity covariance matrix with a single shared variance parameter.

    .. math::
        \symbf{V} = \sigma^2 \symbf{I}_n

    A single unconstrained scalar parameter is transformed to a positive
    variance via :class:`~torch_openreml.covariance.transform.TransformExpPow2`
    by default and then broadcast across all diagonal entries. This
    structure assumes equal, independent variances across all observations.
    """
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        """
        Initialize a scaled identity covariance matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.
            param_names (list of str, optional): Name for the single variance
                parameter. Defaults to ``["sigma^2"]``.
            trans (list of Transform, optional): Transform applied to the
                parameter. Defaults to ``[TransformExpPow2()]``.
            no_grad_index (list of int, optional): Indices of parameters to
                exclude from gradient computation.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import ScalarMatrix

            mat = ScalarMatrix(3)
            params = torch.tensor([0.5])
            print(mat(params))
            print(mat.grad(params))
        """
        param_names = param_names or ["sigma^2"]
        trans = trans or [TransformExpPow2()]
        super().__init__((n, n), param_names, trans, no_grad_index)
        
    def __call__(self, params):
        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)
        sigma2 = self.trans_params(params)

        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        v = sigma2 * i_n
        
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
        if len(self.no_grad_index) > 0:
            return None, []

        params = self.from_param_dict(params)
        device, dtype = self.check_params(params)

        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        grad = (self.trans_grad(params) * i_n).unsqueeze(0)
        grad_names = self.param_names

        return grad, grad_names

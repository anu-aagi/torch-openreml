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
  
    def __init__(self, n, param_spec=None):
        """
        Initialize a scaled identity covariance matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.
            param_spec (dict): Parameter specifications. Keys should be strings
                representing parameter names. Values should be dictionaries
                containing the specification for each parameter. Each specification
                dictionary should contain the keys "fixed", "default", and "trans",
                representing whether the parameter is fixed or free (bool), the
                default value (1D torch.Tensor), and the transform (Transform),
                respectively.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import ScalarMatrix

            mat = ScalarMatrix(3)
            free_params = torch.tensor([0.5])
            print(mat(free_params))
            print(mat.grad(free_params))
        """

        param_spec = param_spec or {
            "sigma^2": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformExpPow2()
            }
        }
        super().__init__((n, n), param_spec)
        
    def __call__(self, free_params):
        sigma2 = self.build_params(free_params)
        device = sigma2.device
        dtype = sigma2.dtype

        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        v = sigma2 * i_n
        
        return v

    def manual_grad(self, free_params):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` if all parameters are fixed.
        """
        if len(free_params) == 0:
            return None, []

        sigma2 = self.build_params(free_params)
        device = sigma2.device
        dtype = sigma2.dtype

        i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
        grad = (self.trans_grad(free_params) * i_n).unsqueeze(0)

        return grad, self.free_param_names

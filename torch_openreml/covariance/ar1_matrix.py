"""
AR(1) covariance matrix.

This module provides a first-order autoregressive covariance matrix with
a variance and a correlation parameter, for use in linear mixed-effects
models.

Classes:
    AR1Matrix:
        An AR(1) covariance matrix :math:`V_{ij} = \\sigma^2 \\rho^{|i-j|}`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformChain, TransformScaleShift, TransformSigmoid
import torch

class AR1Matrix(Matrix):
    r"""
    First-order autoregressive covariance matrix.

    .. math::
        \symbf{V}_{ij} = \sigma^2 \rho^{|i - j|}

    Covariance decays geometrically with the lag between observations.
    The variance :math:`\sigma^2 > 0` is enforced by
    :class:`~torch_openreml.covariance.transform.TransformExpPow2` and the
    correlation :math:`\rho \in (-1, 1)` is enforced by a sigmoid scaled
    to :math:`(-1, 1)` by default.
    """

  
    def __init__(self, n, param_spec=None):
        """
        Initialize an AR(1) covariance matrix of size ``n x n``.

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
            from torch_openreml.covariance import AR1Matrix

            mat = AR1Matrix(4)
            free_params = torch.tensor([0.5, 0.0])
            mat(free_params)
        """
        param_spec = param_spec or {
            "sigma^2": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformExpPow2()},
            "rho" : {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(2.0, -1.0)])
            }
        }
        super().__init__((n, n), param_spec)

    def _get_or_build_intermediates(self, free_params):
        built_params = self.build_params(free_params)
        cache = self.get_intermediates(built_params)

        if cache is None:
            device = built_params.device
            dtype = built_params.dtype
            sigma2, rho = built_params

            idx = torch.arange(self.shape[0], device=device, dtype=dtype)
            diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

            rho_power = rho ** diff

            cache = {"sigma2": sigma2, "rho": rho, "diff": diff, "rho_power": rho_power}
            self.set_intermediates(built_params, cache)

        return cache

    def __call__(self, free_params):
        cache = self._get_or_build_intermediates(free_params)
        v = cache["sigma2"] * cache["rho_power"]
            
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

        cache = self._get_or_build_intermediates(free_params)

        grad = []

        free_param_trans_grad = self.trans_grad(free_params)
        free_param_index = self.free_param_index

        i = 0
        if 0 in free_param_index:
            grad.append(free_param_trans_grad[i] * cache["rho_power"])
            i = i + 1

        if 1 in free_param_index:
            scaled_rho = torch.sign(cache["rho"]) * torch.clamp(cache["rho"].abs(), min=1e-6)
            d_rho = free_param_trans_grad[i] * cache["sigma2"] * cache["diff"] * cache["rho_power"] / scaled_rho
            d_rho.fill_diagonal_(0.0)
            grad.append(d_rho)

        grad = torch.stack(grad)

        return grad, self.free_param_names

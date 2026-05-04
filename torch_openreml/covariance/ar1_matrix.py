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

  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        """
        Initialize an AR(1) covariance matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.
            param_names (list of str, optional): Names for the variance and
                correlation parameters. Defaults to ``["sigma^2", "rho"]``.
            trans (list of Transform, optional): Transforms applied to each
                parameter. Defaults to
                :class:`~torch_openreml.covariance.transform.TransformExpPow2`
                for :math:`\\sigma^2` and a sigmoid scaled to :math:`(-1, 1)`
                for :math:`\\rho`.
            no_grad_index (list of int, optional): Indices of parameters to
                exclude from gradient computation.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import AR1Matrix

            mat = AR1Matrix(4)
            params = torch.tensor([0.5, 0.0])
            mat(params)
        """
        param_names = param_names or ["sigma^2", "rho"]
        trans = trans or [TransformExpPow2(), TransformChain([TransformSigmoid(), TransformScaleShift(2.0, -1.0)])]
        super().__init__((n, n), param_names, trans, no_grad_index)

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            device, dtype = self.check_params(params)
            sigma2, rho = self.trans_params(params)

            idx = torch.arange(self.shape[0], device=device, dtype=dtype)
            diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

            rho_power = rho ** diff

            cache = {"sigma2": sigma2, "rho": rho, "diff": diff, "rho_power": rho_power}
            self.set_intermediates(params, cache)

        return cache

    def __call__(self, params):
        cache = self._get_or_build_intermediates(params)
        v = cache["sigma2"] * cache["rho_power"]
            
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
        if len(self.no_grad_index) == self.num_params:
            return None, []

        cache = self._get_or_build_intermediates(params)

        grad = []
        grad_names = []

        trans_grad = self.trans_grad(params)

        if 0 not in self.no_grad_index:
            grad.append(trans_grad[0] * cache["rho_power"])
            grad_names.append(self.param_names[0])

        if 1 not in self.no_grad_index:
            scaled_rho = torch.sign(cache["rho"]) * torch.clamp(cache["rho"].abs(), min=1e-6)
            d_rho = trans_grad[1] * cache["sigma2"] * cache["diff"] * cache["rho_power"] / scaled_rho
            d_rho.fill_diagonal_(0.0)
            grad.append(d_rho)
            grad_names.append(self.param_names[1])

        grad = torch.stack(grad)

        return grad, grad_names

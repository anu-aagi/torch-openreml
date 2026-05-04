"""
Equicorrelation matrix.

This module provides an equicorrelation matrix with a single shared
correlation parameter, for use in linear mixed-effects models.

Classes:
    EquicorrelationMatrix:
        An equicorrelation matrix :math:`V = (1 - \\rho) I_n + \\rho J_n`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformChain, TransformSigmoid, TransformScaleShift
import torch


class EquicorrelationMatrix(Matrix):
    r"""
    Equicorrelation matrix with a single shared correlation parameter.

    .. math::
        \symbf{V} = (1 - \rho)\symbf{I}_n + \rho\symbf{J}_n

    where :math:`\symbf{I}_n` is the identity matrix and :math:`\symbf{J}_n`
    is the matrix of ones. All diagonal entries equal one and all
    off-diagonal entries equal :math:`\rho`.

    For :math:`\symbf{V}` to be positive definite, the correlation parameter
    must satisfy :math:`\rho > -1/(n-1)`. The default transform enforces this
    by mapping an unconstrained scalar through a sigmoid scaled to
    :math:`(-1/(n-1),\, 1)`.

    Unlike :class:`~torch_openreml.covariance.CompoundSymmetricMatrix`, this
    matrix has no variance parameter.
    """

    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        """
        Initialize an equicorrelation matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.
            param_names (list of str, optional): Name for the correlation
                parameter. Defaults to ``["rho"]``.
            trans (list of Transform, optional): Transform applied to the
                parameter. Defaults to a sigmoid scaled to
                :math:`(-1/(n-1),\\, 1)`.
            no_grad_index (list of int, optional): Indices of parameters to
                exclude from gradient computation.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import EquicorrelationMatrix

            mat = EquicorrelationMatrix(3)
            params = torch.tensor([0.0])
            print(mat(params))
            print(mat.grad(params))
        """
        self.rho_min = -1 / (n - 1)
        param_names = param_names or ["rho"]
        trans = trans or [TransformChain([TransformSigmoid(), TransformScaleShift((1 - self.rho_min), self.rho_min)])]
        super().__init__((n, n), param_names, trans, no_grad_index)

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            device, dtype = self.check_params(params)
            rho = self.trans_params(params)

            i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
            j_n = torch.ones((self.shape[0], self.shape[0]), device=device, dtype=dtype)
            v = ((1 - rho) * i_n + rho * j_n)

            cache = {"i_n": i_n, "j_n": j_n, "v": v}
            self.set_intermediates(params, cache)

        return cache

    def __call__(self, params):
        cache = self._get_or_build_intermediates(params)
        v = cache["v"]

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

        trans_grad = self.trans_grad(params)

        grad = ((cache["j_n"] - cache["i_n"]) * trans_grad[0]).unsqueeze(0)
        grad_names = [self.param_names[0]]

        return grad, grad_names




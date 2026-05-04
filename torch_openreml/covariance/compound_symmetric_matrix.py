"""
Compound symmetric covariance matrix.

This module provides a compound symmetric covariance matrix with a shared
variance and a shared correlation parameter, for use in linear mixed-effects
models.

Classes:
    CompoundSymmetricMatrix:
        A compound symmetric covariance matrix
        :math:`V = \\sigma^2 [(1 - \\rho) I_n + \\rho J_n]`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformChain, TransformScaleShift, TransformSigmoid
import torch

class CompoundSymmetricMatrix(Matrix):
    r"""
    Compound symmetric covariance matrix with shared variance and correlation.

    .. math::
        \symbf{V} = \sigma^2 \left[(1 - \rho)\symbf{I}_n + \rho \symbf{J}_n \right]

    where :math:`\symbf{I}_n` is the identity matrix and :math:`\symbf{J}_n`
    is the matrix of ones. All diagonal entries equal :math:`\sigma^2` and
    all off-diagonal entries equal :math:`\sigma^2 \rho`.

    For :math:`\symbf{V}` to be positive definite, the correlation parameter
    must satisfy :math:`\rho > -1/(n-1)`. The default transform enforces this
    by mapping an unconstrained scalar through a sigmoid scaled to
    :math:`(-1/(n-1),\, 1)`.
    """

    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        """
        Initialize a compound symmetric covariance matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.
            param_names (list of str, optional): Names for the variance and
                correlation parameters. Defaults to ``["sigma^2", "rho"]``.
            trans (list of Transform, optional): Transforms applied to each
                parameter. Defaults to :class:`~torch_openreml.covariance.transform.TransformExpPow2`
                for :math:`\\sigma^2` and a sigmoid scaled to
                :math:`(-1/(n-1),\\, 1)` for :math:`\\rho`.
            no_grad_index (list of int, optional): Indices of parameters to
                exclude from gradient computation.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import CompoundSymmetricMatrix

            mat = CompoundSymmetricMatrix(3)
            params = torch.tensor([0.5, 0.0])
            print(mat(params))
            print(mat.grad(params))
        """
        self.rho_min = -1/(n - 1)
        param_names = param_names or ["sigma^2", "rho"]
        trans = trans or [
            TransformExpPow2(),
            TransformChain([TransformSigmoid(), TransformScaleShift((1 - self.rho_min), self.rho_min)])
        ]
        super().__init__((n, n), param_names, trans, no_grad_index)

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            device, dtype = self.check_params(params)
            sigma2, rho = self.trans_params(params)

            i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
            j_n = torch.ones((self.shape[0], self.shape[0]), device=device, dtype=dtype)
            rho_mat = ((1 - rho) * i_n + rho * j_n)

            cache = {"sigma2": sigma2, "i_n": i_n, "j_n": j_n, "rho_mat": rho_mat}
            self.set_intermediates(params, cache)

        return cache


    def __call__(self, params):
        cache = self._get_or_build_intermediates(params)
        v = cache["sigma2"] * cache["rho_mat"]

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
            grad.append(trans_grad[0] * cache["rho_mat"])
            grad_names.append(self.param_names[0])

        if 1 not in self.no_grad_index:
            grad.append(cache["sigma2"] * (cache["j_n"] - cache["i_n"]) * trans_grad[1])
            grad_names.append(self.param_names[1])

        grad = torch.stack(grad)

        return grad, grad_names

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

    def __init__(self, n, param_spec=None):
        """
        Initialize an equicorrelation matrix of size ``n x n``.

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
            from torch_openreml.covariance import EquicorrelationMatrix

            mat = EquicorrelationMatrix(3)
            free_params = torch.tensor([0.0])
            print(mat(free_params))
            print(mat.grad(free_params))
        """
        self.rho_min = -1 / (n - 1)
        param_spec = param_spec or {
            "rho": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift((1 - self.rho_min), self.rho_min)])}
        }
        super().__init__((n, n), param_spec)

    def _get_or_build_intermediates(self, free_params):
        built_params = self.build_params(free_params)
        cache = self.get_intermediates(built_params)

        if cache is None:
            device = built_params.device
            dtype = built_params.dtype
            rho = built_params

            i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
            j_n = torch.ones((self.shape[0], self.shape[0]), device=device, dtype=dtype)
            v = ((1 - rho) * i_n + rho * j_n)

            cache = {"i_n": i_n, "j_n": j_n, "v": v}
            self.set_intermediates(built_params, cache)

        return cache

    def __call__(self, free_params):
        cache = self._get_or_build_intermediates(free_params)
        v = cache["v"]

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

        free_param_trans_grad = self.trans_grad(free_params)

        grad = ((cache["j_n"] - cache["i_n"]) * free_param_trans_grad).unsqueeze(0)

        return grad, self.free_param_names




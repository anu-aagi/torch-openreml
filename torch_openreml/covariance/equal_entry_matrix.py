"""
Equal-entry covariance matrix.

This module provides an equal-entry covariance matrix in which every
entry shares a single common value, for use in linear mixed-effects
models.

Classes:
    EqualEntryMatrix:
        An equal-entry matrix :math:`V = \\sigma^2 J_{n \\times m}`.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2
import torch

class EqualEntryMatrix(Matrix):
    r"""
    Equal-entry matrix of size ``n x m`` with a single shared value
    across all entries.

    .. math::
        \symbf{V} = \sigma^2 \symbf{J}_{n \times m}

    where :math:`\symbf{J}_{n \times m}` is the :math:`n \times m` matrix
    of ones. A single unconstrained scalar parameter is transformed to a
    positive variance via :class:`~torch_openreml.covariance.transform.TransformExpPow2`
    by default and is then replicated across every entry of the matrix.
    Consequently all entries equal :math:`\sigma^2`.

    When ``m`` is omitted it defaults to ``n``, giving the square
    all-ones matrix :math:`\sigma^2 \symbf{J}_n`. This is singular for
    ``n > 1`` and represents the contribution of a single random effect
    shared by all observations (e.g. a common-environment effect), to be
    combined with other components via
    :class:`~torch_openreml.covariance.Sum`. Non-square sizes
    (``m != n``) may instead be used as rectangular factors, e.g. inside
    :class:`~torch_openreml.covariance.Gram`.
    """

    def __init__(self, n, m=None, param_specs=None):
        """
        Initialize an equal-entry matrix of size ``n x m``.

        Args:
            n (int): Number of rows. Matrix dimension when ``m`` is omitted.
            m (int, optional): Number of columns. Defaults to ``n`` when
                omitted or ``None``. Default: ``None``.
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
            from torch_openreml.covariance import EqualEntryMatrix

            mat = EqualEntryMatrix(3, 2)
            mat

        .. jupyter-execute::

            mat = EqualEntryMatrix(3)
            mat

        .. jupyter-execute::

            free_params = torch.tensor([0.5])
            mat(free_params)

        .. jupyter-execute::

            mat.grad(free_params)
        """

        m = n if m is None else m

        param_specs = param_specs or {
            "sigma^2": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformExpPow2()
            }
        }
        super().__init__((n, m), param_specs)

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        sigma2 = self.build_params(free_params)
        device = sigma2.device
        dtype = sigma2.dtype

        j_n_m = torch.ones((self.shape[0], self.shape[1]), device=device, dtype=dtype)
        v = sigma2 * j_n_m

        return v

    def manual_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` with respect to trainable
        parameters using a closed-form analytic expression.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary. If omitted, default values are used.
                Default: ``None``.

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

        free_params = self.build_params(free_params, include_fixed=False, trans=False, out_format="tensor")
        device = free_params.device
        dtype = free_params.dtype

        j_n_m = torch.ones((self.shape[0], self.shape[1]), device=device, dtype=dtype)
        grad = (self.trans_grad(free_params) * j_n_m).unsqueeze(0)

        return grad, self.free_param_names

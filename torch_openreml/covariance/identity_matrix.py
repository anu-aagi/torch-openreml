"""
Identity covariance matrix.

This module provides a fixed identity matrix for use as a covariance
structure in linear mixed-effects models. It has no trainable parameters
and always returns the same matrix regardless of input.

Classes:
    IdentityMatrix:
        A fixed :math:`n \\times n` identity covariance matrix.
"""

from torch_openreml.covariance.matrix import Matrix
import torch


class IdentityMatrix(Matrix):
    r"""
    Fixed :math:`n \times n` identity covariance matrix.

    .. math::
        \symbf{V} = \symbf{I}_n

    This matrix has no trainable parameters, so :meth:`grad` always
    returns ``(None, [])``. It is typically used to represent independent,
    homoscedastic residuals.
    """

    def __init__(self, n, dtype=None, device=None):
        """
        Initialize a fixed identity matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.
            dtype (torch.dtype, optional): Desired dtype of the matrix.
                Defaults to the PyTorch default dtype.
            device (torch.device, optional): Desired device of the matrix.
                Defaults to the PyTorch default device.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import IdentityMatrix

            mat = IdentityMatrix(3)
            mat()
        """
        self._matrix = torch.eye(n, dtype=dtype, device=device)
        super().__init__((n, n), {})

    def __call__(self, *args, **kwargs):
        return self._matrix
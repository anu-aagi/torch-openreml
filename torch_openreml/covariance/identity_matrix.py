"""
Identity covariance matrix.

This module provides a fixed identity matrix for use as a covariance
structure in linear mixed-effects models. It has no trainable parameters,
so it always returns the same matrix; only the dtype and the device of the
input are followed.

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

    :meth:`__call__` follows the dtype and the device of its input; when it
    receives none, the PyTorch default dtype and device are used.
    """

    def __init__(self, n):
        """
        Initialize a fixed identity matrix of size ``n x n``.

        Args:
            n (int): Matrix dimension.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import IdentityMatrix

            mat = IdentityMatrix(3)
            mat()
        """
        self._matrix = torch.eye(n)
        super().__init__((n, n), {})

    def __call__(self, free_params=None):
        """
        Return the identity matrix, on the dtype and device of the input.

        The identity matrix has no parameters, so ``free_params`` carries no
        values: only its dtype and device are used. When it is omitted, the
        PyTorch default dtype and device are used.

        Args:
            free_params (torch.Tensor or dict, optional): Empty 1D parameter
                tensor whose dtype and device are followed, or an empty
                parameter dict. If omitted, the PyTorch default dtype and
                device are used. Default: ``None``.

        Returns:
            torch.Tensor: The ``n x n`` identity matrix.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor or a dict.
            ValueError: If ``free_params`` is not empty, since the matrix has
                no parameters to receive.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import IdentityMatrix

            mat = IdentityMatrix(3)
            mat()

        .. jupyter-execute::

            mat(torch.tensor([], dtype=torch.float64))
        """
        params = self.build_params(free_params, include_fixed=False, trans=False)
        return self._matrix.to(device=params.device, dtype=params.dtype)
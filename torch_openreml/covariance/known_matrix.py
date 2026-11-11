"""
Known covariance matrix.

This module provides a fixed covariance matrix backed by a user-supplied
tensor.  It has no trainable parameters, so it is useful when a covariance
structure is known in advance and should not be estimated.

Classes:
    KnownMatrix:
        A fixed covariance matrix backed by a tensor.
"""

import torch

from torch_openreml.covariance.matrix import Matrix


class KnownMatrix(Matrix):
    r"""
    Fixed covariance matrix backed by a known tensor.

    .. math::
        \symbf{V} = \symbf{K}

    The matrix has no free parameters.  Its values are the tensor supplied
    at construction; :meth:`__call__` follows the dtype and device of the
    parameter tensor when one is provided.

    Args:
        matrix (torch.Tensor): The known covariance matrix, a two-dimensional tensor.

    Raises:
        TypeError: If ``matrix`` is not a Torch tensor.
        ValueError: If ``matrix`` is not two-dimensional.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml.covariance import KnownMatrix

        k = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        mat = KnownMatrix(k)
        mat()
    """

    def __init__(self, matrix):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("'matrix' must be a Torch tensor!")
        if matrix.ndim != 2:
            raise ValueError("'matrix' must be a two-dimensional tensor!")

        self._matrix = matrix
        super().__init__(tuple(matrix.shape), {})

    @property
    def matrix(self):
        """torch.Tensor: The known covariance matrix."""
        return self._matrix

    def __call__(self, free_params=None):
        """
        Return the known covariance matrix.

        Args:
            free_params (torch.Tensor or dict, optional): Empty parameter
                tensor or dict.  A tensor controls the dtype and device of
                the returned matrix.  If omitted, the PyTorch default dtype
                and device are used.  Default: ``None``.

        Returns:
            torch.Tensor: A copy of the known covariance matrix on the
            resolved dtype and device.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor or a dict.
            ValueError: If ``free_params`` is not a 1D tensor of length zero,
                or is a dict with any entries, since the matrix has no free
                parameters.
        """
        params = self.build_params(free_params, include_fixed=False, trans=False)
        return self._matrix.to(device=params.device, dtype=params.dtype).clone()

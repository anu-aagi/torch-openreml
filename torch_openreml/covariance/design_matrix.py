"""
Design matrix.

This module provides a fixed design matrix for use  in linear
mixed-effects models. The matrix is constructed
from numeric or categorical input at initialisation and has no
trainable parameters.

Classes:
    DesignMatrix:
        A fixed design matrix constructed from numeric or categorical data.
"""

import torch
import pandas as pd
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.utils import numeric_to_design_matrix, categorical_to_design_matrix


class DesignMatrix(Matrix):
    r"""
    Fixed design matrix constructed from numeric or categorical input.

    .. math::
        \symbf{V} = \symbf{X}

    where :math:`\symbf{X}` is constructed from ``x`` at initialisation
    and remains fixed thereafter. This matrix has no trainable parameters,
    so :meth:`grad` always returns ``(None, [])``.

    Numeric input is passed to :func:`~torch_openreml.utils.numeric_to_design_matrix`
    and categorical string input to :func:`~torch_openreml.utils.categorical_to_design_matrix`.
    In both cases ``levels`` and ``drop_first`` control which columns are
    retained.
    """

    def __init__(self, x, levels=None, drop_first=False, dtype=None, device=None):
        """
        Initialize a fixed design matrix from numeric or categorical input.

        Args:
            x (torch.Tensor, list, tuple, or pandas.Series): Input data. Either a numeric
                tensor or list, or a list of strings for categorical data.
            levels (list, optional): Explicit level ordering for categorical
                input, or bin edges for numeric input.
            drop_first (bool, optional): Whether to drop the first column.
                Defaults to ``False``.
            dtype (torch.dtype, optional): Desired dtype of the matrix.
            device (torch.device, optional): Desired device of the matrix.

        Raises:
            TypeError: If ``x`` is not a :class:`torch.Tensor`, list, or tuple.


        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DesignMatrix

            mat = DesignMatrix(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            print(mat())

            mat = DesignMatrix(["a", "b", "a", "c"])
            print(mat())

            mat = DesignMatrix(["a", "b", "a", "c"], levels=["c", "b", "a"])
            print(mat())
        """
        if not isinstance(x, (torch.Tensor, list, tuple, pd.Series)):
            raise TypeError("'x' must be a torch.Tensor, a list,  a tuple or a pandas.Series!")

        if isinstance(x, pd.Series):
            x = x.to_list()

        if torch.is_tensor(x):
            self._matrix = numeric_to_design_matrix(x, dtype=dtype or x.dtype, device=device or x.device)
        elif isinstance(x[0], str):
            self._matrix = categorical_to_design_matrix(x, levels, drop_first, dtype, device)
        else:
            self._matrix = numeric_to_design_matrix(x, levels, drop_first, dtype, device)

        super().__init__((self._matrix.shape[0], self._matrix.shape[1]), [], [])

    def __call__(self, *args, **kwargs):
        return self._matrix

    @property
    def repr_dict(self):
        return {"shape": self._shape}

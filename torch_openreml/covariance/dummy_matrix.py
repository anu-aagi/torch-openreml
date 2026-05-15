"""
Dummy matrix.

This module provides a fixed dummy matrix for use in linear
mixed-effects models. The matrix is constructed
from categorical input at initialisation and has no
trainable parameters.

Classes:
    DummyMatrix:
        A fixed dummy matrix constructed from categorical data.
"""
import warnings
from itertools import product
import torch
import pandas as pd
from torch_openreml.covariance.matrix import Matrix


class DummyMatrix(Matrix):
    r"""
    Fixed dummy matrix constructed from categorical input.

    .. math::
        \symbf{V} = \symbf{X}

    where :math:`\symbf{X}` is constructed from ``*args`` at initialisation
    and remains fixed thereafter. This matrix has no trainable parameters,
    so :meth:`grad` always returns ``(None, [])``.
    """

    def __init__(self, *args, levels=None, lex_order=True, drop_first=False, dtype=None, device=None):
        """
        Initialize a fixed dummy matrix from numeric or categorical input.

        Args:
            *args (list, tuple, or pandas.Series): Input data. One or many lists
                of strings for categorical data.
            levels (list or tuple, optional): Levels of each list of strings.
                Defaults to a list of sorted unique elements in each list of
                strings.
            lex_order (bool, optional): If ``True``, the result columns are
                lexically ordered.
            drop_first (bool, optional): Whether to drop the first column.
                Defaults to ``False``.
            dtype (torch.dtype, optional): Desired dtype of the matrix.
            device (torch.device, optional): Desired device of the matrix.

        Raises:
            TypeError: If any ``args`` is not a :class: list or tuple.

        Example:

        .. jupyter-execute::

            from torch_openreml.covariance import DummyMatrix

            rep = ["rep1", "rep2", "rep2"]
            block = ["block1", "block2", "block1"]

            mat = DummyMatrix(rep, block)
            print(mat())
            print(mat.colnames)

            mat = DummyMatrix(rep, block, drop_first=True)
            print(mat())
            print(mat.colnames)

            mat = DummyMatrix(rep, block, levels=[["rep1", "rep2", "rep3"], ["block1", "block2"]])
            print(mat())
            print(mat.colnames)

            mat = DummyMatrix(rep, block, levels=[["rep3", "rep1"], ["block1", "block2"]], lex_order=False)
            print(mat())
            print(mat.colnames)
        """
        dtype = dtype or torch.get_default_dtype()
        device = device or torch.get_default_device()

        for i, arg in enumerate(args):
            if not isinstance(arg, (list, tuple, pd.Series)):
                raise TypeError(f"Argument {i} must be a list, a tuple or a pandas.Series!")

        args = [arg.to_list() if isinstance(arg, pd.Series) else arg for arg in args]

        n = len(args[0])

        for i, arg in enumerate(args):
            if len(arg) != n:
                raise ValueError(f"Argument {i} must have the same number of elements as other arguments!")

        levels = levels or [sorted(set(arg)) for arg in args]

        rows = list(zip(*args))
        combos = list(product(*levels))
        if lex_order:
            combos = sorted(combos)
        colnames = ["⋈".join(c) for c in combos]

        x = pd.DataFrame(0, index=range(n), columns=colnames)

        for i, r in enumerate(rows):
            key = "⋈".join(r)
            if key not in x.columns:
                warnings.warn(f"Unknown combination {r} dropped!", RuntimeWarning)
            else:
                x.loc[i, key] = 1

        if drop_first:
            x = x.iloc[:, 1:]

        self._colnames = x.columns.tolist()
        self._matrix = torch.tensor(x.to_numpy(), dtype=dtype, device=device)

        super().__init__((self._matrix.shape[0], self._matrix.shape[1]), {})

    def __call__(self, *args, **kwargs):
        return self._matrix

    @property
    def colnames(self):
        """list: Column names of the matrix."""
        return self._colnames

    @property
    def repr_dict(self):
        return {"shape": self._shape}

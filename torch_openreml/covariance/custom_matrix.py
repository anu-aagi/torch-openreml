"""
Custom covariance matrix.

This module provides a customizable covariance matrix where users can
supply their own forward and gradient implementations.

Classes:
    CustomMatrix:
        A user-defined covariance matrix with pluggable call and gradient.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.param import simple_param_specs


class CustomMatrix(Matrix):
    r"""
    A covariance matrix with user-supplied forward and gradient functions.

    Users provide the number of parameters (or a full ``param_specs`` dict)
    and a callable that constructs the matrix from parameters. An optional
    manual gradient callable can be supplied; otherwise automatic
    differentiation is used via :meth:`~Matrix.auto_grad`.

    Args:
        n (int, optional): Number of parameters. Ignored if ``param_specs``
            is provided.
        call (callable): Function with signature
            ``call(free_params) -> torch.Tensor`` that constructs
            the covariance matrix from the parameter tensor.
        manual_grad (callable, optional): Function with signature
            ``manual_grad(free_params) -> (grad, grad_names)``
            that computes the closed-form Jacobian. If ``None`` (default),
            automatic differentiation is used.
        param_specs (dict, optional): Full parameter specification dict.
            If provided, ``n`` is ignored.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml.covariance.custom_matrix import CustomMatrix

        def my_call(mat, free_params):
            params = mat.build_params(free_params)
            n = params.shape[0]
            return torch.diag(params)

        mat = CustomMatrix(n=3, call=my_call)
        mat(torch.tensor([0.0, 0.5, 1.0]))
    """

    def __init__(self, n=None, call=None, manual_grad=None, param_specs=None):
        if param_specs is None:
            if n is None:
                raise ValueError("Either 'n' or 'param_specs' must be provided.")
            param_specs = simple_param_specs(n)

        if call is None:
            raise ValueError("'call' must be provided.")

        super().__init__(None, param_specs)
        self._call = call
        self._manual_grad = manual_grad

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        result = self._call(self, free_params)
        self._shape = tuple(result.shape)
        return result

    def manual_grad(self, free_params=None):
        if self._manual_grad is None:
            raise NotImplementedError
        if free_params is None:
            free_params = self.free_param_defaults
        return self._manual_grad(self, free_params)

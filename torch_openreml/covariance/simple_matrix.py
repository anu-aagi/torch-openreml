"""
Simple covariance matrix from a user-supplied function.

This module provides a minimal adapter for wrapping a plain function as a
:class:`~torch_openreml.covariance.matrix.Matrix`, intended for simple cases
where all parameters are free with identity transforms — no manual parameter
specifications or transform logic required.

Classes:
    SimpleMatrix:
        A covariance matrix backed by a user-defined callable.
"""

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.param import simple_param_specs


class SimpleMatrix(Matrix):
    r"""
    A covariance matrix for simple, function-based parameterisations.

    This is the easiest way to use :class:`~torch_openreml.REML` with a custom
    covariance structure: provide the number of parameters and a function that
    maps a flat parameter tensor to the covariance matrix. All parameters are
    free and use an identity transform (unconstrained). The ``default``
    argument sets the value used for each free parameter when none are
    provided.

    For more advanced needs (custom transforms, fixed parameters, manual
    gradients), subclass :class:`~torch_openreml.covariance.matrix.Matrix`
    directly.

    Args:
        n (int): Number of free parameters.
        call (callable): Function with signature
            ``call(free_params) -> torch.Tensor`` that constructs the
            covariance matrix from a flat 1D parameter tensor.
        manual_grad (callable, optional): Function with signature
            ``manual_grad(free_params) -> (grad, grad_names)`` for a
            closed-form Jacobian. If ``None`` (default), automatic
            differentiation is used.
        default (float or torch.Tensor, optional): Default value for each
            parameter. Passed to :func:`simple_param_specs`. Defaults to
            ``0.0``.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml.covariance import SimpleMatrix

        def my_v(free_params):
            n = free_params.shape[0]
            return torch.diag(free_params)

        mat = SimpleMatrix(n=3, call=my_v)
        mat(torch.tensor([1.0, 2.0, 3.0]))

    .. jupyter-execute::

        mat.grad(torch.tensor([1.0, 2.0, 3.0]))
    """

    def __init__(self, n, call, manual_grad=None, default=0.0):
        if call is None:
            raise ValueError("'call' must be provided.")

        super().__init__(None, simple_param_specs(n, default=default))
        self._call = call
        self._manual_grad = manual_grad

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        result = self._call(free_params)
        self._shape = tuple(result.shape)
        return result

    def manual_grad(self, free_params=None):
        if self._manual_grad is None:
            raise NotImplementedError
        if free_params is None:
            free_params = self.free_param_defaults
        return self._manual_grad(free_params)

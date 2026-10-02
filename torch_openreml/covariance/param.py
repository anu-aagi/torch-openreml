"""
Parameter specification helpers.

Provides utility functions for creating parameter specification
dictionaries used by :class:`~torch_openreml.covariance.matrix.Matrix`.

Functions:
    simple_param_specs:
        Create a default parameter specification with identity transforms.
"""

import torch
from torch_openreml.covariance.transform import TransformIdentity


def simple_param_specs(n, default=None, trans=None):
    """
    Create a parameter specification dictionary with ``n`` parameters.

    Each parameter is named ``"theta_0"``, ``"theta_1"``, ..., is not fixed,
    and uses a common transform for all parameters. All parameters share the
    same default value.

    Args:
        n (int): Number of parameters to create.
        default (torch.Tensor, optional): Default value for each parameter,
            given as a 1D tensor of shape ``(1,)``. Defaults to
            ``torch.tensor([0.0])``.
        trans (Transform, optional): Transform to apply to all parameters.
            Defaults to :class:`TransformIdentity` (unconstrained).

    Returns:
        dict: A dictionary mapping parameter names to specification dicts
        of the form ``{"fixed": False, "default": tensor, "trans": trans}``.

    Raises:
        TypeError: If ``default`` is not a Torch tensor.
        ValueError: If ``default`` does not have shape ``(1,)``.

    Example:

    .. jupyter-execute::

        from torch_openreml.covariance.param import simple_param_specs

        simple_param_specs(3)

    .. jupyter-execute::

        import torch

        simple_param_specs(2, default=torch.tensor([1.5]))
    """
    if trans is None:
        trans = TransformIdentity()

    if default is None:
        default = torch.tensor([0.0])

    if not torch.is_tensor(default):
        raise TypeError("default must be a 1D torch tensor of shape (1).")

    if default.ndim != 1 or default.shape[0] != 1:
        raise ValueError("Default must be a 1D tensor with shape (1).")

    return {
        f"theta_{i}": {
            "fixed": False,
            "default": default.detach().clone(),
            "trans": trans
        }
        for i in range(n)
    }

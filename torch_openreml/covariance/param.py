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


def simple_param_specs(n, default=0.0, trans=None):
    """
    Create a parameter specification dictionary with ``n`` parameters.

    Each parameter is named ``"theta_0"``, ``"theta_1"``, ..., is not fixed,
    and uses a common transform for all parameters. An optional scalar default
    value can be provided.

    Args:
        n (int): Number of parameters to create.
        default (float or torch.Tensor, optional): Default value for each
            parameter. If a tensor, it must be 1D with shape ``(1,)``.
            Defaults to ``0.0``.
        trans (Transform, optional): Transform to apply to all parameters.
            Defaults to :class:`TransformIdentity` (unconstrained).

    Returns:
        dict: A dictionary mapping parameter names to specification dicts
        of the form ``{"fixed": False, "default": tensor, "trans": trans}``.

    Raises:
        ValueError: If ``default`` is a tensor without shape ``(1,)``.
        TypeError: If ``default`` is not a float, int, or 1D tensor.

    Example:

    .. jupyter-execute::

        from torch_openreml.covariance.param import simple_param_specs

        simple_param_specs(3)
    """
    if trans is None:
        trans = TransformIdentity()

    if torch.is_tensor(default):
        if default.ndim != 1 or default.shape[0] != 1:
            raise ValueError("Tensor default must be 1D with shape (1).")
        d = default
    elif isinstance(default, (float, int)):
        d = torch.tensor([float(default)])
    else:
        raise TypeError("default must be a float/int or a 1D torch tensor of shape (1).")

    return {
        f"theta_{i}": {
            "fixed": False,
            "default": d.detach().clone(),
            "trans": trans
        }
        for i in range(n)
    }

"""
Identity transform module for unconstrained parameter mappings.

Provides a trivial bijective transform from
:math:`\\mathbb{R} \\rightarrow \\mathbb{R}` that leaves inputs unchanged.

Classes:
    TransformIdentity:
        Identity transform (:math:`f(x) = x`)
"""

from torch_openreml.covariance.transform.transform import Transform
import torch

class TransformIdentity(Transform):
    r"""
    Identity transform.

    .. math::

        f(x) = x
    """

    domain = "\u211D"
    codomain = "\u211D"

    def __init__(self):
        r"""
        Initialize the identity transform.
        """
        pass

    def __call__(self, x):
        r"""
        Apply the identity transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Unchanged input :math:`x`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformIdentity

            t = TransformIdentity()
            x = torch.tensor([0.0, 1.0, -3.5])
            t(x)
        """
        return x

    def inverse(self, x):
        r"""
        Apply the inverse transform (identity).

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Unchanged input :math:`x`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformIdentity

            t = TransformIdentity()
            x = torch.tensor([2.0, -1.0])
            t.inverse(x)
        """
        return x

    def chain_rule_factor(self, x):
        r"""
        Compute derivative of :math:`f(x) = x` for chain rule propagation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: [1.0]

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformIdentity

            t = TransformIdentity()
            x = torch.tensor([0.0])
            t.chain_rule_factor(x)
        """

        return torch.tensor([1.0], dtype=x.dtype, device=x.device)
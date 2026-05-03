"""
Scale-shift transform module for affine parameter mappings.

Provides a differentiable bijective transform from
:math:`\\mathbb{R} \\rightarrow \\mathbb{R}` using a configurable
affine function.

Classes:
    TransformScaleShift:
        Affine transform (:math:`f(x) = ax + b`)
"""
import torch

from torch_openreml.covariance.transform.transform import Transform


class TransformScaleShift(Transform):
    r"""
    Affine transform with configurable scale and shift.

    .. math::

        f(x) = ax + b
    """

    domain = "\u211D"
    codomain = "\u211D"

    def __init__(self, a, b=0.0):
        r"""
        Initialize the affine transform.

        Args:
            a (float): Scale factor.
            b (float): Shift offset. Defaults to ``0.0``.
        """
        self.a = a
        self.b = b

    def __call__(self, x):
        r"""
        Apply the affine transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Element-wise :math:`ax + b`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformScaleShift

            t = TransformScaleShift(a=2.0, b=1.0)
            x = torch.tensor([0.0, 1.0, 2.0])
            t(x)
        """
        return self.a * x + self.b

    def inverse(self, x):
        r"""
        Apply the inverse transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Element-wise :math:`\frac{x - b}{a}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformScaleShift

            t = TransformScaleShift(a=2.0, b=1.0)
            x = torch.tensor([1.0, 3.0, 5.0])
            t.inverse(x)
        """
        return (x - self.b) / self.a

    def chain_rule_factor(self, x):
        r"""
        Compute derivative of :math:`ax + b` for chain rule propagation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: [:math:`a`].

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformScaleShift

            t = TransformScaleShift(a=2.0, b=1.0)
            x = torch.tensor([0.0])
            t.chain_rule_factor(x)
        """
        return torch.tensor([self.a], dtype=x.dtype, device=x.device)

    def __repr__(self):
        return f"{self.__class__.__name__}(a={self.a}, b={self.b})"
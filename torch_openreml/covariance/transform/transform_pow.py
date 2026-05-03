"""
Power transform module for parameter mappings.

Provides a differentiable transform from
:math:`\\mathbb{R} \\rightarrow \\mathbb{R}` using a configurable
power function.

Classes:
    TransformPow:
        Power transform (:math:`f(x) = x^p`)
"""

from torch_openreml.covariance.transform.transform import Transform
import torch


class TransformPow(Transform):
    r"""
    Power transform with configurable exponent.

    .. math::

        f(x) = x^p
    """

    domain = "\u211D"
    codomain = "\u211D"

    def __init__(self, factor=2.0):
        r"""
        Initialize the power transform.

        Args:
            factor (float): Exponent :math:`p`. Defaults to ``2.0``.
        """
        self.factor = factor

    def __call__(self, x):
        r"""
        Apply the power transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Element-wise :math:`x^p`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformPow

            t = TransformPow(factor=3.0)
            x = torch.tensor([1.0, 2.0, 3.0])
            t(x)
        """
        return torch.pow(x, self.factor)

    def inverse(self, x):
        r"""
        Apply the inverse transform (square root).

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Element-wise :math:`\sqrt{x}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformPow

            t = TransformPow(factor=2.0)
            x = torch.tensor([1.0, 4.0, 9.0])
            t.inverse(x)
        """
        return torch.sqrt(x)

    def chain_rule_factor(self, x):
        r"""
        Compute derivative of :math:`x^p` for chain rule propagation.

        Note:
            .. math::
                \frac{d}{dx} x^p = p x^{p-1}

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: :math:`p x^{p-1}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformPow

            t = TransformPow(factor=3.0)
            x = torch.tensor([2.0, 3.0])
            t.chain_rule_factor(x)
        """
        return self.factor * x

    def __repr__(self):
        return f"{self.__class__.__name__}(factor={self.factor})"
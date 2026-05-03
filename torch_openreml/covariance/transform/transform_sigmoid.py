"""
Sigmoid transform module for bounded parameter mappings.

Provides a differentiable bijective transform from
:math:`\\mathbb{R}_{0+} \\rightarrow (0, 1)` using the
logistic sigmoid function.

Classes:
    TransformSigmoid:
        Sigmoid transform (:math:`f(x) = \\frac{1}{1 + e^{-x}}`)
"""

from torch_openreml.covariance.transform.transform import Transform
import torch


class TransformSigmoid(Transform):
    r"""
    Sigmoid transform mapping reals to the open unit interval.

    .. math::

        f(x) = \frac{1}{1 + e^{-x}}
    """

    domain = "\u211D\u2080\u207A"
    codomain = "(0, 1)"

    def __init__(self):
        r"""
        Initialize the sigmoid transform.
        """
        pass

    def __call__(self, x):
        r"""
        Apply the sigmoid transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}_{0+}`.

        Returns:
            torch.Tensor: Element-wise :math:`\frac{1}{1 + e^{-x}}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformSigmoid

            t = TransformSigmoid()
            x = torch.tensor([-2.0, 0.0, 2.0])
            t(x)
        """
        return torch.sigmoid(x)

    def inverse(self, x):
        r"""
        Apply the inverse transform (logit).

        Args:
            x (torch.Tensor): Input tensor in :math:`(0, 1)`.

        Returns:
            torch.Tensor: Element-wise :math:`\log\frac{x}{1 - x}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformSigmoid

            t = TransformSigmoid()
            x = torch.tensor([0.1, 0.5, 0.9])
            t.inverse(x)
        """
        return torch.logit(x)

    def grad(self, x):
        r"""
        Compute derivative of :math:`\sigma(x)` for chain rule propagation.

        Note:
            .. math::
                \frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x))

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: :math:`\sigma(x)(1 - \sigma(x))`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformSigmoid

            t = TransformSigmoid()
            x = torch.tensor([0.0, 1.0])
            t.grad(x)
        """
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 - sigmoid)
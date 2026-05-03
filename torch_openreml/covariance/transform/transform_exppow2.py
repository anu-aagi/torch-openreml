"""
Scaled exponential transform module for constrained parameter mappings.

Provides a differentiable bijective transform from
:math:`\\mathbb{R} \\rightarrow \\mathbb{R}_{0+}` using a natural
exponential with exponent scaled by 2.

Classes:
    TransformExpPow2:
        Scaled exponential transform (:math:`e^{2x}`)
"""

from torch_openreml.covariance.transform.transform import Transform
import torch


class TransformExpPow2(Transform):
    r"""
    Exponential transform with exponent scaled by 2.

    .. math::

        f(x) = e^{2x}
    """

    domain = "\u211D"
    codomain = "\u211D\u2080\u207A"

    def __init__(self):
        r"""
        Initialize the scaled exponential transform.
        """
        pass

    def __call__(self, x):
        r"""
        Apply the scaled exponential transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Element-wise :math:`e^{2x}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExpPow2

            t = TransformExpPow2()
            x = torch.tensor([0.0, 1.0])
            t(x)
        """
        return torch.exp(2.0 * x)

    def inverse(self, x):
        r"""
        Apply the inverse transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}_{0+}`.

        Returns:
            torch.Tensor: :math:`\frac{\log(x)}{2}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExpPow2

            t = TransformExpPow2()
            x = torch.tensor([1.0])
            t.inverse(x)
        """
        return torch.log(x) / 2.0

    def grad(self, x):
        r"""
        Compute derivative of :math:`e^{2x}` for chain rule propagation.

        Note:
            .. math::
                \frac{d}{dx} e^{2x} = 2e^{2x}

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: :math:`2e^{2x}`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExpPow2

            t = TransformExpPow2()
            x = torch.tensor([0.0, 1.0])
            t.grad(x)
        """
        return 2 * torch.exp(2 * x)
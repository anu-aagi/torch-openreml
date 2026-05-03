"""
Exponential transform module for constrained parameter mappings.

Provides a set of differentiable bijective transforms from
:math:`\\mathbb{R} \\rightarrow \\mathbb{R}_{+}` using different
exponential bases.

Classes:
    TransformExp:
        Natural exponential transform (:math:`e^x`)

    TransformExp2:
        Base-2 exponential transform (:math:`2^x`)

    TransformExp10:
        Base-10 exponential transform (:math:`10^x`)
"""

from torch_openreml.covariance.transform.transform import Transform
import torch


class TransformExp(Transform):
    r"""
    Exponential transform using the natural exponential function.

    .. math::

        f(x) = e^x
    """

    domain = "\u211D"
    codomain = "\u211D\u207A"

    def __init__(self):
        r"""
        Initialize the exponential transform.
        """
        pass

    def __call__(self, x):
        r"""
        Apply the natural exponential transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: Element-wise :math:`e^x`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp

            t = TransformExp()
            x = torch.tensor([0.0, 1.0])
            t(x)
        """
        return torch.exp(x)

    def inverse(self, x):
        r"""
        Apply the inverse transform (natural logarithm).

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}_{+}`.

        Returns:
            torch.Tensor: :math:`\log(x)`.

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp

            t = TransformExp()
            x = torch.tensor([1.0])
            t.inverse(x)
        """
        return torch.log(x)

    def chain_rule_factor(self, x):
        r"""
        Compute derivative of :math:`e^x` for chain rule propagation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: :math:`e^x`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp

            t = TransformExp()
            x = torch.tensor([0.0])
            t.chain_rule_factor(x)
        """
        return torch.exp(x)


class TransformExp2(Transform):
    r"""
    Base-2 exponential transform.

    .. math::

        f(x) = 2^x
    """

    domain = "\u211D"
    codomain = "\u211D\u207A"

    def __init__(self):
        r"""
        Initialize base-2 exponential transform.
        """
        pass

    def __call__(self, x):
        r"""
        Apply base-2 exponential transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: :math:`2^x`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp2

            t = TransformExp2()
            x = torch.tensor([0.0, 1.0])
            t(x)
        """
        return torch.exp2(x)

    def inverse(self, x):
        r"""
        Apply inverse base-2 logarithm.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}_{+}`.

        Returns:
            torch.Tensor: :math:`\log_{2}(x)`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp2

            t = TransformExp2()
            x = torch.tensor([1.0, 2.0])
            t.inverse(x)
        """
        return torch.log2(x)

    def chain_rule_factor(self, x):
        r"""
        Compute derivative of :math:`2^x`.

        Note:
            .. math::
                \frac{d}{dx} 2^x = 2^x \ln 2

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: :math:`2^x \ln 2`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp2

            t = TransformExp2()
            x = torch.tensor([1.0])
            t.chain_rule_factor(x)
        """
        return torch.exp2(x) * torch.log(
            torch.tensor([2], dtype=x.dtype, device=x.device)
        )


class TransformExp10(Transform):
    r"""
    Base-10 exponential transform.

    .. math::

        f(x) = 10^x
    """

    domain = "\u211D"
    codomain = "\u211D\u207A"

    def __init__(self):
        r"""
        Initialize base-10 exponential transform.
        """
        pass

    def __call__(self, x):
        r"""
        Apply base-10 exponential transform.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}`.

        Returns:
            torch.Tensor: :math:`10^x`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp10

            t = TransformExp10()
            x = torch.tensor([0.0, 1.0])
            t(x)
        """
        return torch.pow(10.0, x)

    def inverse(self, x):
        r"""
        Apply inverse base-10 logarithm.

        Args:
            x (torch.Tensor): Input tensor in :math:`\mathbb{R}_{+}`.

        Returns:
            torch.Tensor: :math:`\log_{10}(x)`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp10

            t = TransformExp10()
            x = torch.tensor([1.0, 10.0])
            t.inverse(x)
        """
        return torch.log10(x)

    def chain_rule_factor(self, x):
        r"""
        Compute derivative of :math:`10^x`.

        Note:
            .. math::
                \frac{d}{dx} 10^x = 10^x \ln 10

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: :math:`10^x \ln 10`.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp10

            t = TransformExp10()
            x = torch.tensor([1.0])
            t.chain_rule_factor(x)
        """
        return torch.pow(10.0, x) * torch.log(
            torch.tensor([10], dtype=x.dtype, device=x.device)
        )
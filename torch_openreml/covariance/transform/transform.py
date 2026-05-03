"""
Abstract and composite transform system.

This module defines a base abstraction for mathematical transforms
and a composition mechanism for chaining multiple transforms together.

Classes:
    Transform:
        Abstract base class defining forward, inverse, and derivative behavior.

    TransformChain:
        Composition of multiple transforms applied sequentially.
"""

from abc import ABC, abstractmethod


class Transform(ABC):
    """
    Abstract base class for mathematical transforms.

    A transform maps inputs from a domain to a codomain, and defines
    a forward operation, its inverse, and a chain rule factor for
    differentiation through the transform.
    """

    #: Domain of the transform.
    domain = "\u211D"

    #: Codomain of the transform.
    codomain = "\u211D"

    @abstractmethod
    def __call__(self, x):
        """
        Apply the forward transformation.

        Args:
            x: Input value in the transform's domain.

        Returns:
            Transformed value in the codomain.
        """
        pass

    @abstractmethod
    def inverse(self, x):
        """
        Apply the inverse transformation.

        Args:
            x: Input value in the codomain.

        Returns:
            Value mapped back to the domain.
        """
        pass

    @abstractmethod
    def grad(self, x):
        """
        Compute the derivative factor for chain rule propagation.

        Args:
            x: Input value at which to evaluate the derivative factor.

        Returns:
            Scalar or tensor representing the local Jacobian/derivative.
        """
        pass

    def __repr__(self):
        """
        Return a developer-friendly representation of the transform.
        """
        return f"{self.__class__.__name__}()"

    def __str__(self):
        """
        Return a human-readable representation of the transform.
        """
        return f"{self.__class__.__name__}: {self.domain} \u21A6 {self.codomain}"


class TransformChain(Transform):
    """
    Composition of multiple Transform objects applied sequentially.

    The chain behaves as a single transform:
    forward pass applies transforms in order, while inverse applies them
    in reverse order.
    """

    def __init__(self, chain):
        """
        Initialize a transform chain.

        Args:
            chain (Transform or list/tuple of Transform): Sequence of
                transforms to compose.

        Raises:
            TypeError: If any element in chain is not a Transform instance.
        """
        if not isinstance(chain, (list, tuple)):
            chain = [chain]
        self.chain = chain

        for trans in chain:
            if not isinstance(trans, Transform):
                raise TypeError(
                    "Chain needs to be a list or a tuple of Transform objects!"
                )

    def __call__(self, x):
        """
        Apply the chained transformation forward.

        Args:
            x: Input value in the domain of the first transform.

        Returns:
            Output after applying all transforms sequentially.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp, TransformPow, TransformChain

            t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
            x = torch.tensor([1.0])
            t(x)
        """
        for trans in self.chain:
            x = trans(x)
        return x

    def inverse(self, x):
        """
        Apply the inverse of the chained transformation.

        Args:
            x: Input value in the codomain of the last transform.

        Returns:
            Value mapped back through the inverse chain to the original domain.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp, TransformPow, TransformChain

            t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
            x = torch.tensor([4.0])
            t.inverse(x)
        """
        for trans in reversed(self.chain):
            x = trans.inverse(x)
        return x

    def grad(self, x):
        """
        Compute chain rule factor for the full composed transform.

        Note:
            This assumes local derivatives are evaluated consistently
            along the forward pass.

        Args:
            x: Input value in the original domain.

        Returns:
            Combined derivative factor of all transforms in the chain.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance.transform import TransformExp, TransformPow, TransformChain

            t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
            x = torch.tensor([1.0])
            t.grad(x)
        """
        factor = None
        for trans in self.chain:
            if factor is None:
                factor = trans.grad(x)
            else:
                factor = factor * trans.grad(x)

            x = trans(x)

        return factor

    def __repr__(self):
        """
        Return a developer-friendly representation of the transform chain.
        """
        return f"{self.__class__.__name__}({self.chain!r})"

    def __str__(self):
        """
        Return a human-readable representation of the transform chain.
        """
        return f"{self.__class__.__name__}({self.chain!r})"
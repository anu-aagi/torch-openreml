"""
Global configuration.

Functions:
    set_default_jacobian_method:
        Select the Jacobian method used throughout the package.
    get_default_jacobian_method:
        Return the currently selected Jacobian method.
    get_default_chunk_size:
        Return the currently selected chunk size.
    jacobian_method:
        Temporarily select the Jacobian method used throughout the package.
"""

import contextlib
import torch

_jacobian_method = torch.func.jacfwd
_chunk_size = None

def set_default_jacobian_method(name, chunk_size=None):
    r"""
    Select the Jacobian method used throughout the package.

    The three options construct the Jacobian in different ways, which affects
    how they scale for the :math:`(n, n)` covariance matrices differentiated by
    this package:

    * `"jacobian"`: the most conservative option. It uses an eager loop over
      the outputs in reverse mode, processing one output at a time and releasing
      it before moving to the next. Memory usage stays low, but it is slow, with
      computational cost increasing steeply with :math:`n`.
    * `"jacrev"`: a more efficient reverse-mode approach. It batches the
      outputs, with `chunk_size` controlling how much of the batch is held in
      memory at once, allowing a trade-off between speed and memory usage.
      Without chunking, memory usage grows with :math:`n^2`.
    * `"jacfwd"`: the most appropriate option for the :math:`(n, n)` covariance
      matrices differentiated by this package. It requires one forward pass per
      parameter, so its computational cost depends on the number of parameters
      rather than the size of the covariance matrix. It is therefore the fastest
      of the three when the number of parameters is small. However, it still
      holds one :math:`(n, n)` tangent matrix per parameter, so it can run out of
      memory for very large covariance matrices.

    Args:
        name (str): Which Jacobian method to use, one of ``"jacrev"``,
            ``"jacfwd"``, or ``"jacobian"``.
        chunk_size (int or None): Number of outputs per batch when
            differentiating with ``"jacrev"``, or ``None`` to compute the
            Jacobian in a single batch. Chunking trades time for memory: a
            smaller chunk holds less at once and needs more passes, a larger
            chunk holds more and needs fewer. Ignored by ``"jacfwd"``, which
            takes no chunk size argument, and by the eager ``"jacobian"``,
            which already loops over the outputs one at a time.

    Raises:
        ValueError: If ``name`` is not a supported Jacobian method, or if
            ``chunk_size`` is neither ``None`` nor a positive integer.

    Returns:
        None

    Example:

    .. jupyter-execute::

        from torch_openreml.config import (
            set_default_jacobian_method,
            get_default_jacobian_method,
        )

        set_default_jacobian_method("jacfwd")
        get_default_jacobian_method()
    """
    global _jacobian_method, _chunk_size

    _chunked_jacrev = lambda func: torch.func.jacrev(func, chunk_size=_chunk_size)
    _eager_jacobian = lambda func: lambda *args: torch.autograd.functional.jacobian(func, args[0])

    methods = {
        "jacrev": _chunked_jacrev,
        "jacfwd": torch.func.jacfwd,
        "jacobian": _eager_jacobian,
    }

    if name not in methods:
        raise ValueError(f"Unknown Jacobian method {name!r}! Expected one of 'jacrev', 'jacfwd', 'jacobian'.")

    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError(f"Chunk size must be a positive integer or None, got {chunk_size!r}.")

    _jacobian_method = methods[name]
    _chunk_size = chunk_size


def get_default_jacobian_method():
    """
    Return the currently selected Jacobian method.

    Returns:
        Callable: The currently selected Jacobian method. Defaults to
        `torch.func.jacfwd` until :func:`set_default_jacobian_method` is
        called.

    Example:

    .. jupyter-execute::

        from torch_openreml.config import get_default_jacobian_method

        get_default_jacobian_method()
    """
    return _jacobian_method


def get_default_chunk_size():
    """
    Return the currently selected chunk size.

    Returns:
        int or None: The current chunk size, ``None`` until
        :func:`set_default_jacobian_method` sets one.

    Example:

    .. jupyter-execute::

        from torch_openreml.config import get_default_chunk_size

        print(get_default_chunk_size())
    """
    return _chunk_size


@contextlib.contextmanager
def jacobian_method(name, chunk_size=None):
    """
    Temporarily select the Jacobian method used throughout the package.

    The previous method and chunk size are restored on exit, whether the
    block returns normally or raises.

    Args:
        name (str): Which Jacobian method to use for the duration of the
            block, one of ``"jacrev"``, ``"jacfwd"``, or ``"jacobian"``.
        chunk_size (int or None): Number of outputs per batch for the
            duration of the block, or ``None`` to compute the Jacobian in a
            single batch. Ignored by ``"jacfwd"``, which takes no chunk size
            argument, and by the eager ``"jacobian"``, which already loops
            over the outputs one at a time.

    Raises:
        ValueError: If ``name`` is not a supported Jacobian method, or if
            ``chunk_size`` is neither ``None`` nor a positive integer.

    Yields:
        None

    Example:

    .. jupyter-execute::

        from torch_openreml.config import (
            jacobian_method,
            get_default_jacobian_method,
            get_default_chunk_size,
        )

        with jacobian_method("jacrev", chunk_size=16):
            print(get_default_jacobian_method(), get_default_chunk_size())
        print(get_default_jacobian_method(), get_default_chunk_size())
    """
    global _jacobian_method, _chunk_size

    previous_method = _jacobian_method
    previous_chunk_size = _chunk_size

    set_default_jacobian_method(name, chunk_size=chunk_size)
    try:
        yield
    finally:
        _jacobian_method = previous_method
        _chunk_size = previous_chunk_size

"""
General utilities.

Functions:
    get_device:
        Validate and return the shared device of a collection of tensors.
    get_dtype:
        Validate and return the shared dtype of a collection of tensors.
    numeric_to_design_matrix:
        Construct a design matrix from one or more numeric vectors or tensors.
    categorical_to_design_matrix:
        Construct a one-hot encoded design matrix from a categorical vector.
    augment:
        Horizontally concatenate design matrices.
    interaction:
        Construct an interaction term from two or more categorical vectors.
"""

import torch
import pandas as pd

def get_device(*args):
    """
    Validate and return the shared device of a collection of tensors.

    Args:
        *args (torch.Tensor): Zero or more tensors. All must reside on
            the same device.

    Returns:
        torch.device: The shared device of all tensors, the PyTorch
        default device if no tensors are provided.

    Raises:
        ValueError: If any tensor resides on a different device than the first.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml.utils import get_device

        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        get_device(x, y)
    """
    if len(args) == 0:
        return torch.get_default_device()
      
    device = args[0].device
    for i, t in enumerate(args):
        if t.device != device:
            raise ValueError(f"Device mismatch at arg {i}: {t.device} != {device}")
    
    return device

def get_dtype(*args):
    """
    Validate and return the shared dtype of a collection of tensors.

    Args:
        *args (torch.Tensor): Zero or more tensors. All must share the
            same dtype.

    Returns:
        torch.dtype: The shared dtype of all tensors, or the PyTorch
        default dtype if no tensors are provided.

    Raises:
        ValueError: If any tensor has a different dtype than the first.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml.utils import get_dtype

        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        get_dtype(x, y)
    """
    if len(args) == 0:
        return torch.get_default_dtype()
    
    dtype = args[0].dtype
    for i, t in enumerate(args):
        if t.dtype != dtype:
            raise ValueError(f"Dtype mismatch at arg {i}: {t.dtype} != {dtype}")
    
    return dtype

def numeric_to_design_matrix(*args, dtype=None, device=None):
    """
    Construct a design matrix from one or more numeric vectors or tensors.

    Each argument becomes one column of the resulting design matrix. All
    inputs must have the same length along the first dimension.

    Args:
        *args (torch.Tensor, list, tuple or pd.Series): One or more numeric vectors
            of equal length. Lists, tuples and pd.Series are converted to tensors.
        dtype (torch.dtype, optional): Desired dtype of the matrix.
                Defaults to the PyTorch default dtype.
        device (torch.device, optional): Desired device of the matrix.
            Defaults to the PyTorch default device.

    Returns:
        torch.Tensor: Design matrix of shape ``(n, num_args)``.

    Raises:
        ValueError: If no inputs are provided, or if inputs have
            inconsistent lengths.
        TypeError: If any input is not a tensor, list, or tuple.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml.utils import numeric_to_design_matrix

        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([4.0, 5.0, 6.0])
        numeric_to_design_matrix(x1, x2)
    """

    if len(args) == 0:
        raise ValueError("At least one input is required.")

    cols = []
    n = None

    dtype = dtype or torch.get_default_dtype()
    device = device or torch.get_default_device()

    for i, x in enumerate(args):

        if isinstance(x, pd.Series):
            x = x.to_list()

        if torch.is_tensor(x):
            x = x.to(dtype=dtype, device=device)
        elif isinstance(x, (list, tuple)):
            x = torch.tensor(x, dtype=dtype, device=device)
        else:
            raise TypeError("'x' must be either a tensor or a list/tuple of values!")

        if len(x.shape) == 2:
            x.unsqueeze_(0)

        if n is None:
            n = x.shape[0]
        elif x.shape[0] != n:
            raise ValueError(f"Inconsistent lengths at argument {i}: expected {n}, got {x.shape[0]}")

        cols.append(x)

    return torch.stack(cols, dim=1)

def augment(*args):
    """
    Horizontally concatenate two or more design matrices.

    A convenience wrapper around :func:`torch.cat` along the column
    dimension. Useful for combining numeric and categorical design
    matrices into a single matrix.

    Args:
        *args (torch.Tensor): Two or more design matrices with the same
            number of rows.

    Returns:
        torch.Tensor: Concatenated matrix of shape
        ``(n, sum of columns across all inputs)``.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml.utils import augment

        x1 = torch.ones(4, 2)
        x2 = torch.zeros(4, 3)
        augment(x1, x2)
    """
    return torch.cat(args, dim=1)

def interaction(*args, sep="\u22C8"):
    """
    Construct an interaction term from two or more categorical vectors.

    Joins the corresponding elements of each input vector into a single
    string separated by ``sep``, producing a new categorical vector whose
    levels represent combined factor combinations.

    Args:
        *args (list or tuple of str): Two or more categorical vectors of
            equal length.
        sep (str, optional): Separator inserted between joined levels.
            Defaults to ``"⋈"`` (U+22C8, the bowtie symbol).

    Returns:
        list of str: Interaction vector of the same length as the inputs.

    Raises:
        ValueError: If no inputs are provided.
        TypeError: If any input is not a list or tuple of strings.

    Example:

    .. jupyter-execute::

        from torch_openreml.utils import interaction

        a = ["control", "control", "treatment"]
        b = ["male", "female", "male"]
        interaction(a, b)
    """
    if len(args) == 0:
        raise ValueError("At least one input is required.")

    for i, arg in enumerate(args):
        if not isinstance(arg, (list, tuple)):
            raise TypeError(f"Argument {i} is not list/tuple of strings!")

    return [sep.join(parts) for parts in zip(*args)]

def n_distinct(x):
    """
    Count the unique elements of a list/tuple of strings.

    Args:
        x (list or tuple): Input.

    Returns:
        int: Number of unique elements.

    Example:

    .. jupyter-execute::

        from torch_openreml.utils import n_distinct

        n_distinct(["a", "b", "a", "c"])
    """
    if isinstance(x, pd.Series):
        x = x.to_list()

    if isinstance(x, (list, tuple)):
        return len(set(x))
    else:
        raise TypeError(f"Argument x is not a list/tuple of strings!")
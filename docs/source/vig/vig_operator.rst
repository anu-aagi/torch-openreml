.. _vig_operator:

The Operator Interface
======================

A single matrix class describes one covariance structure. An
:class:`~torch_openreml.covariance.Operator` builds a matrix from other
matrices. It is itself a
:class:`~torch_openreml.covariance.Matrix`, so it has the same
``shape``, parameters, ``build_params``, ``grad``, and inverse interface
as any other matrix.

This page assumes the interface introduced in :ref:`vig_matrix`. It is
about the generic behaviour of an operator, not a catalogue of the
concrete operators. ``Sum`` and ``BlockDiagonal`` appear below only as
vehicles for that behaviour; the same interface applies to every
operator.

A first composite
-----------------

``Sum`` adds operands of the same shape. Its result is easy to read,
which makes it a convenient first composite:

.. jupyter-execute::

    import torch

    from torch_openreml.covariance import (
        DiagonalMatrix,
        ScalarMatrix,
        Sum,
    )

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)

    op = Sum(
        genetic=ScalarMatrix(3),
        environment=DiagonalMatrix(3),
    )
    theta = torch.zeros(4)

    print("names:", op.param_names)
    print(op(theta))
    print("shape:", op.shape)

The operator has one parameter namespace per operand. ``genetic``
contributes ``genetic/sigma^2``; ``environment`` contributes
``environment/sigma^2_0``, ``environment/sigma^2_1``, and
``environment/sigma^2_2``. At the default values both matrices are
identity, so the sum is ``2 * I``.

The operator's shape is resolved when it is built. Calling it before
reading ``shape`` avoids an empty placeholder.

``build_params`` behaves exactly as it does for a single matrix, except
that the names are namespaced:

.. jupyter-execute::

    op.build_params(theta, out_format="dict")

The free parameters are concatenated in operand order. Inside ``Sum``,
that order is the order in which the operands were written.

Operands and names
------------------

Operands may be supplied as keyword arguments, as a single dictionary,
or positionally. Positional operands receive the names ``op_0``,
``op_1``, and so on:

.. jupyter-execute::

    positional = Sum(ScalarMatrix(2), DiagonalMatrix(2))
    print(positional.param_names)

Keyword names are the readable form, and a single dictionary is
equivalent to keywords:

.. jupyter-execute::

    named = Sum({"left": ScalarMatrix(2), "right": DiagonalMatrix(2)})
    print(named.param_names)

Operand names must be non-empty and must not contain ``/``. Positional
and keyword operands cannot be mixed in one call.

A fixed tensor is also a valid operand. It contributes no parameters and
is cast to the dtype and device of the matrix operands:

.. jupyter-execute::

    fixed = Sum(
        signal=ScalarMatrix(2),
        baseline=torch.eye(2),
    )

    print("names:", fixed.param_names)
    print(fixed(torch.tensor([0.0])))

Only ``signal/sigma^2`` is trainable. ``baseline`` is added unchanged.

One dtype and device
--------------------

The composite resolves a single dtype and device for all operands. Explicitly supplied input parameters take precedence. Otherwise, the matrix operands' defaults are used. Fixed tensors are then cast to match the resolved dtype and device:

.. jupyter-execute::

    mixed = Sum(
        signal=ScalarMatrix(2),
        baseline=torch.eye(2, dtype=torch.float32),
    )

    print("default build:", mixed(torch.tensor([0.0])).dtype)
    print(
        "float32 input:",
        mixed(torch.tensor([0.0], dtype=torch.float32)).dtype,
    )

The first build uses the matrix operands' default dtype, while the second follows the explicitly supplied parameter tensor.

The operator tree
-----------------

An operator can contain another operator. The operands then form a tree,
with ``/`` as the root. ``walk`` lists every node without evaluating the
matrices:

.. jupyter-execute::

    from torch_openreml.covariance import BlockDiagonal

    nested = Sum(
        inner=BlockDiagonal(
            DiagonalMatrix(2),
            ScalarMatrix(2),
        ),
        extra=ScalarMatrix(4),
    )
    theta = torch.tensor([0.0, 0.5, 1.0, 0.5])

    for path, node, params in nested.walk(theta):
        print(f"{path:16} {type(node).__name__:22} {params}")

The root receives all four parameters. Each child receives only its own
slice, in operand order. ``tree`` evaluates every node and uses the same
paths as keys:

.. jupyter-execute::

    results = nested.tree(theta)

    print("paths:", list(results))
    print(results["inner/op_0"])

Only nested operators are descended into. A plain matrix is a leaf, even
if it contains its own internal structure.

Gradients
---------

``grad`` works on the composite as a whole. The parameter names and the
first dimension of the Jacobian follow the same namespaced order as
``free_param_names``:

.. jupyter-execute::

    grad, names = op.grad(theta)

    print("gradient shape:", grad.shape)
    print("gradient names:", names)

The composite has four free parameters and a 3 × 3 matrix, so the
Jacobian has shape ``(4, 3, 3)``.

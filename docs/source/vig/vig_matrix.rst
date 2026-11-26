.. _vig_matrix:

The Matrix Interface
====================

In ``torch-openreml``, a covariance matrix is an object rather than a
plain tensor. The object knows its shape, which parameters are free or
fixed, how to build the matrix from those parameters, and how to differentiate
and invert it.

This page introduces that interface. It is not a tutorial on a specific
covariance structure.

A matrix is a function of parameters
------------------------------------

:class:`~torch_openreml.covariance.ScalarMatrix` represents
:math:`\sigma^2 \boldsymbol{I}`. It has one free parameter,
``sigma^2``:

.. jupyter-execute::

    import torch

    from torch_openreml.covariance import ScalarMatrix

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)

    mat = ScalarMatrix(3)

    print("shape:", mat.shape)
    print("parameters:", mat.param_names)
    print("free parameters:", mat.free_param_names)
    print("number of parameters:", mat.num_params)
    print("number of free parameters:", mat.num_free_params)

A parameter specification records whether the parameter is fixed, its
default value on the *unconstrained* scale, and the transform that maps
it to the constrained scale:

.. jupyter-execute::

    mat.param_specs["sigma^2"]

Calling the object builds the matrix. With no argument, the default
parameter values are used:

.. jupyter-execute::

    mat()

``ScalarMatrix`` uses the transform :math:`\sigma^2 = e^{2\theta}`.
The default free parameter is ``0``, so the default matrix is the
identity. Passing ``0.5`` gives :math:`e^{2(0.5)} = e` on the diagonal:

.. jupyter-execute::

    theta = torch.tensor([0.5])
    mat(theta)

The distinction between the unconstrained value passed in and the value
used to build the matrix is important:

.. jupyter-execute::

    print("transformed:", mat.build_params(theta))
    print("untransformed:", mat.build_params(theta, trans=False))
    print("named:", mat.build_params(theta, out_format="dict"))

``build_params`` is the bridge between the optimizer and the matrix.
It accepts a flat tensor or a dictionary keyed by parameter name, so the
same call path can be used for named parameters:

.. jupyter-execute::

    named_theta = {"sigma^2": torch.tensor([0.5])}

    print("matrix:")
    print(mat(named_theta))
    print("named values:", mat.build_params(named_theta))

    named_grad, _ = mat.grad(named_theta)

    print("named gradient shape:", named_grad.shape)

The dictionary must contain exactly the free parameter names. Its values
are concatenated in ``free_param_names`` order, so a named input is
equivalent to a flat tensor with the same values.

Fixed parameters
----------------

Parameters can be fixed without removing them from the specification.
Here the variance is fixed at its default value:

.. jupyter-execute::

    fixed = ScalarMatrix(3)
    fixed.set_param_specs("sigma^2", fixed=True)

    print("free parameters:", fixed.free_param_names)
    print("built:", fixed.build_params(out_format="dict"))
    print("without fixed:", fixed.build_params(include_fixed=False))
    print("gradient:", fixed.grad())

The gradient is ``(None, [])`` because there is nothing to optimize.
The fixed value is still available through ``build_params`` and is used
when the matrix is evaluated.

Changing several specifications
-------------------------------

``set_param_specs`` can update more than one parameter at once. A key of
``"/"`` addresses the whole matrix, and a list or tuple supplies one
value per resolved specification. Here the first and third variances are
fixed, while the middle one stays free:

.. jupyter-execute::

    from torch_openreml.covariance import DiagonalMatrix
    from torch_openreml.covariance.transform import TransformExp

    diag = DiagonalMatrix(3)

    diag.set_param_specs(
        "/",
        default=torch.tensor([0.0, 0.0, 0.5]),
        fixed=[True, False, True],
    )
    diag.set_param_specs("sigma^2_1", trans=TransformExp())

    print("free parameters:", diag.free_param_names)
    print("built:", diag.build_params(torch.tensor([0.5])))
    print(
        "named:",
        diag.build_params(torch.tensor([0.5]), out_format="dict"),
    )

The two calls show the two addressing modes: ``"/"`` selects every
parameter, while ``"sigma^2_1"`` selects one exact specification. The
second call replaces only the transform of the middle parameter, so it
uses :math:`e^{\theta}` rather than :math:`e^{2\theta}`. The last fixed
parameter keeps its raw default of ``0.5`` and is transformed with the
original :math:`e^{2\theta}`, giving a variance of :math:`e`. Both fixed
parameters are still included in the result.

Gradients
---------

``grad`` returns the Jacobian of the matrix with respect to the free
parameters, together with the parameter names:

.. jupyter-execute::

    grad, names = mat.grad(theta)

    print("shape:", grad.shape)
    print("names:", names)
    print("diagonal of the first Jacobian:", torch.diagonal(grad[0]))

The leading dimension is the parameter index, so the result has shape
``(num_free_params, rows, columns)``. For ``ScalarMatrix`` the Jacobian
of the variance with respect to the raw parameter is
:math:`2e^{2\theta}`, applied to the identity.

The default mode tries a closed-form gradient and falls back to automatic
differentiation. The two should agree:

.. jupyter-execute::

    mat.grad_mode = "auto"
    grad_auto, _ = mat.grad(theta)
    mat.grad_mode = "default"

    torch.allclose(grad, grad_auto)

The modes are ``"default"``, ``"manual"`` and ``"auto"``. A matrix that
does not implement ``manual_grad`` falls back to autodiff under
``"default"``; forcing ``"manual"`` raises instead.

Inverses
--------

Some matrices also provide a closed-form inverse:

.. jupyter-execute::

    inverse = mat.manual_inverse(theta)

    print(inverse)
    print(
        "matches numerical inverse:",
        torch.allclose(inverse, torch.linalg.inv(mat(theta))),
    )

``manual_inverse`` is optional. The base class raises
``NotImplementedError`` when a closed form is unavailable; in that case
use ``torch.linalg.inv(mat(theta))``.

A structured example
--------------------

:class:`~torch_openreml.covariance.AR1Matrix` has two parameters,
``sigma^2`` and ``rho``:

.. jupyter-execute::

    from torch_openreml.covariance import AR1Matrix

    ar = AR1Matrix(4)

    print("parameters:", ar.param_names)
    print("free parameters:", ar.free_param_names)
    print("default matrix:")
    print(ar())

The default ``rho`` is zero, so the default matrix is the identity.
Passing two free parameters changes both the variance and the
correlation:

.. jupyter-execute::

    ar_params = torch.tensor([0.0, 0.5])

    print(ar.build_params(ar_params, out_format="dict"))
    print(ar(ar_params))

The same interface works for the gradient:

.. jupyter-execute::

    ar_grad, ar_names = ar.grad(ar_params)

    print("shape:", ar_grad.shape)
    print("names:", ar_names)

Fixed matrices and design matrices
----------------------------------

Not every matrix has parameters.
:class:`~torch_openreml.covariance.DummyMatrix` is a fixed dummy design
matrix built from labels:

.. jupyter-execute::

    from torch_openreml.covariance import DummyMatrix

    design = DummyMatrix(["a", "b", "a", "c"])

    print("shape:", design.shape)
    print("columns:", design.colnames)
    print("free parameters:", design.free_param_names)
    print(design())

It has no free parameters, so its gradient is empty. ``transform`` maps
new labels onto the same columns; labels not seen at construction become
all-zero rows:

.. jupyter-execute::

    print(design.grad())
    print(design.transform(["a", "c", "z"]))

:class:`~torch_openreml.covariance.KnownMatrix` wraps a fixed tensor and
can still provide an inverse when the tensor is square and invertible:

.. jupyter-execute::

    from torch_openreml.covariance import KnownMatrix

    known = KnownMatrix(torch.tensor([[1.0, 0.25], [0.25, 1.0]]))

    print(known())
    print(known.manual_inverse())
    print(known.grad())

Custom matrices
---------------

:class:`~torch_openreml.covariance.SimpleMatrix` is the shortest route
to a custom matrix. Supply the number of free parameters and a callable
that returns the matrix:

.. jupyter-execute::

    from torch_openreml.covariance import SimpleMatrix

    def equicorrelation(theta):
        rho = torch.sigmoid(theta[0])
        return (1 - rho) * torch.eye(3) + rho * torch.ones(3, 3)

    custom = SimpleMatrix(1, equicorrelation)

    print("parameters:", custom.param_names)
    print(custom())

``SimpleMatrix`` infers the shape on the first call, uses identity
transforms, and falls back to automatic differentiation:

.. jupyter-execute::

    custom_grad, _ = custom.grad()

    print("gradient shape:", custom_grad.shape)

For a production matrix, subclass
:class:`~torch_openreml.covariance.Matrix` directly when you need fixed
parameters, custom transforms, analytic gradients, or a closed-form
inverse.

Method summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Purpose
   * - ``shape``
     - Output shape after construction.
   * - ``param_names``, ``free_param_names``
     - All parameter names and the trainable subset.
   * - ``num_params``, ``num_free_params``
     - Number of all and trainable parameters.
   * - ``param_specs``
     - Per-parameter ``fixed``, ``default`` and ``trans`` entries.
   * - ``__call__``
     - Build the matrix from free parameters.
   * - ``build_params``
     - Accept a flat tensor or name-keyed dictionary; merge free and
       fixed values, apply transforms, and return a tensor or dictionary.
   * - ``set_param_specs``
     - Change ``fixed``, ``default`` or ``trans`` in place; ``"/"``
       addresses every parameter and lists provide per-parameter values.
   * - ``grad``
     - Jacobian with respect to free parameters, shape
       ``(num_free, rows, cols)``.
   * - ``manual_grad``, ``auto_grad``
     - Force a closed-form or automatic gradient.
   * - ``manual_inverse``
     - Closed-form inverse when the class provides one.

The key idea is that the covariance object, not the optimizer, owns the
parameterisation. Once a class implements the interface, it can be used
anywhere the model expects a covariance matrix.

For a first model, continue with :ref:`vig_random_intercept`.

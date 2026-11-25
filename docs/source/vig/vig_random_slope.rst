.. _vig_random_slope:

Random-Slope Models
===================

A random slope allows the effect of a covariate to vary across groups.
Like a random intercept, it is represented by a ``(Z, G)`` pair. The
design matrix ``Z`` determines how the covariate contributes to each
group's random effects, while ``G`` specifies their covariance structure.

For example, to give each subject their own intercept and slope for
``day``, each subject has two random-effect columns in ``Z``: one for
the intercept and one for the slope. For subject ``i``, these columns
are

.. math::

    \boldsymbol{Z}_i = \begin{bmatrix} 1 & \text{day}_{i1} \\
                          1 & \text{day}_{i2} \\
                          \vdots & \vdots \\
                          1 & \text{day}_{in_i}
           \end{bmatrix},
    \qquad
    \boldsymbol{G}_{\text{subject}} =
    \begin{bmatrix}
        \tau_{00} & \tau_{01} \\
        \tau_{01} & \tau_{11}
    \end{bmatrix}.

Here, ``n_i`` is the number of observations for subject ``i``.
The first column represents the random intercept, and the second
represents the random slope for ``day``. The off-diagonal elements
of ``G`` allow the two random effects to be correlated.

A random-slope-only model is obtained by removing the intercept column
from each subject's design matrix and using a scalar ``G``.

This page is a quick reference. See :ref:`vig_random_intercept` for
the random-intercept version and :ref:`vig_mixed_model_reml` for the
full API.

Simulate a sleep study
----------------------

Each of 72 subjects is measured on eight days.

.. jupyter-execute::
    :hide-code:

    import pandas as pd
    import torch

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import (
        Augment,
        DummyMatrix,
        Gram,
        HadamardProduct,
        IdentityMatrix,
        KroneckerProduct,
        LowerTriangularMatrix,
        ScalarMatrix,
    )

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("display.precision", 3)

    generator = torch.Generator().manual_seed(2026)
    n_subjects, n_days = 72, 8

    subject = torch.arange(n_subjects).repeat_interleave(n_days)
    day = torch.arange(n_days).repeat(n_subjects)

    z_intercept = torch.randn(n_subjects, generator=generator)
    z_slope = torch.randn(n_subjects, generator=generator)
    b_intercept = 30.0 * z_intercept
    b_slope = 10.0 * (-0.5 * z_intercept + 0.75**0.5 * z_slope)

    y = (
        250.0
        + 10.0 * day
        + b_intercept[subject]
        + b_slope[subject] * day
        + 25.0 * torch.randn(len(day), generator=generator)
    )

    data = pd.DataFrame(
        {
            "subject": [f"subject_{i:02d}" for i in subject.tolist()],
            "day": day.numpy(),
            "y": y.numpy(),
        }
    )

    y = torch.tensor(data["y"].values)
    X = torch.stack(
        [
            torch.ones(len(data)),
            torch.tensor(data["day"].values, dtype=torch.float64),
        ],
        dim=1,
    )
    data.head()

A per-column summary:

.. jupyter-execute::
    :hide-code:

    data.describe(include="all").T[
        ["count", "unique", "top", "freq", "mean", "std", "min", "max"]
    ]

.. raw:: html

   <details>
   <summary><strong>Show simulation code</strong></summary>

.. code-block:: python

    # Same code as the hidden cell above.
    import pandas as pd
    import torch

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import (
        Augment,
        DummyMatrix,
        Gram,
        HadamardProduct,
        IdentityMatrix,
        KroneckerProduct,
        LowerTriangularMatrix,
        ScalarMatrix,
    )

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("display.precision", 3)

    generator = torch.Generator().manual_seed(2026)
    n_subjects, n_days = 72, 8

    subject = torch.arange(n_subjects).repeat_interleave(n_days)
    day = torch.arange(n_days).repeat(n_subjects)

    z_intercept = torch.randn(n_subjects, generator=generator)
    z_slope = torch.randn(n_subjects, generator=generator)
    b_intercept = 30.0 * z_intercept
    b_slope = 10.0 * (-0.5 * z_intercept + 0.75**0.5 * z_slope)

    y = (
        250.0
        + 10.0 * day
        + b_intercept[subject]
        + b_slope[subject] * day
        + 25.0 * torch.randn(len(day), generator=generator)
    )

    data = pd.DataFrame(
        {
            "subject": [f"subject_{i:02d}" for i in subject.tolist()],
            "day": day.numpy(),
            "y": y.numpy(),
        }
    )

    y = torch.tensor(data["y"].values)
    X = torch.stack(
        [
            torch.ones(len(data)),
            torch.tensor(data["day"].values, dtype=torch.float64),
        ],
        dim=1,
    )

.. raw:: html

   </details>

``X`` contains only the fixed effects: an intercept and ``day``.

A random-slope-only model
-------------------------

To model a subject-specific slope for ``day`` without a random intercept,
multiply the subject dummy matrix by ``day``. This gives one random-effect
column per subject, with the corresponding slope variance specified by
``ScalarMatrix``:

.. jupyter-execute::

    D = DummyMatrix(data["subject"])
    day_values = torch.tensor(data["day"].values).to(torch.float64).unsqueeze(-1)
    Z_slope = HadamardProduct(group=D, x=day_values)

    reml_slope = MixedModelREML(subject=(Z_slope, ScalarMatrix(data["subject"])))
    reml_slope.optimize(y, X)

    reml_slope.build_params(out_format="dict")

Each column of ``Z_slope`` corresponds to a subject. Its entries are the
observed ``day`` values for that subject and zero for all other subjects.
``ScalarMatrix`` uses the subject labels to determine the number of
random effects and specifies their common variance.

Because this model includes no random intercept, it does not directly
model between-subject variation in baseline responses. With a single
random term, the parameter names are ``G/sigma^2`` and ``R/sigma^2``.
Since ``R`` is omitted, it defaults to
:math:`\sigma^2 \boldsymbol{I}`, with its dimension determined by the
random-effects design.

Random intercept and slope
--------------------------

To allow both the intercept and slope to vary across subjects, each subject
needs two random-effect columns: one for the intercept and one for the
slope. ``DummyMatrix`` creates the intercept columns, while multiplying
it by ``day`` gives the slope columns. ``Augment`` combines them, placing
all intercept columns first, followed by all slope columns.

The corresponding covariance matrix is constructed as :math:`G_2 \otimes I`,
where :math:`G_2` is a shared :math:`2 \times 2` covariance matrix for the intercept and
slope:

.. jupyter-execute::

    Z_subject = Augment(
        intercept=D,
        slope=HadamardProduct(group=D, x=day_values),
    )

    G_subject = KroneckerProduct(
        covariance=Gram(
            LowerTriangularMatrix(2, 2),
            gram_type="xxt",
        ),
        subject=IdentityMatrix(n_subjects),
    )

    reml_subject = MixedModelREML(subject=(Z_subject, G_subject))
    reml_subject.optimize(y, X)

    reml_subject.build_params(out_format="dict")

Here, ``Z_subject`` has 144 columns: 72 intercept columns followed by
72 slope columns. ``KroneckerProduct`` takes the :math:`2 \times 2` covariance first
and the subject identity matrix second, producing
:math:`G_2 \otimes I`. The resulting matrix consists of four :math:`72 \times 72`
blocks. The covariance parameters are shared across subjects, while
random effects from different subjects are independent.

Alternatively, the columns of ``Z_subject`` could be interleaved by
subject, placing each subject's intercept and slope columns together.
In that case, the Kronecker product order would be reversed, giving
:math:`I \otimes G_2` directly. The important point is that the column
ordering of ``Z`` must match the row and column ordering of ``G``.

``Gram(..., gram_type="xxt")`` parameterises the :math:`2 \times 2` covariance block
as :math:`G_2 = LL^\top`, ensuring that it is positive semidefinite. The
estimated parameters correspond to the entries of the lower-triangular
factor:

.. code-block:: text

    G/covariance/op_0/L_0_0   L[0, 0]
    G/covariance/op_0/L_1_0   L[1, 0]
    G/covariance/op_0/L_1_1   L[1, 1]

Put the fitted block back together:

.. jupyter-execute::

    reml_subject.tree()["G/covariance"]

Independent intercept and slope
-------------------------------

If you do not want to estimate the covariance, use two scalar terms instead.
This is the same construction as multiple random intercepts:

.. jupyter-execute::

    reml_independent = MixedModelREML(
        subject_intercept=(D, ScalarMatrix(data["subject"])),
        subject_slope=(Z_slope, ScalarMatrix(data["subject"])),
    )
    reml_independent.optimize(y, X)

    reml_independent.build_params(out_format="dict")

With several terms, the names include the term name:
``G/subject_intercept/sigma^2`` and ``G/subject_slope/sigma^2``. This form
forces the intercept-slope covariance to zero.

After fitting
-------------

The fitted object keeps the data and parameters, so post-fit quantities are
one call away:

.. jupyter-execute::

    print("converged:", reml_subject.is_converged())
    print("loglik:", reml_subject.loglik()[0])
    print("fixed effects:", reml_subject.blue())
    print("fixed-effect SEs:", torch.sqrt(reml_subject.var_blue()))

    blups = reml_subject.blup_per_term()["subject"]
    print("first three subject BLUPs (intercept, slope):")
    print(blups.reshape(2, n_subjects).T[:3])

For a vector-valued term, ``blup_per_term()`` returns one entry per column of
``Z``, in the same order. Here that is all 72 intercept BLUPs followed by all
72 slope BLUPs, so reshaping to ``(2, n_subjects)`` and transposing gives one
intercept-slope pair per subject.

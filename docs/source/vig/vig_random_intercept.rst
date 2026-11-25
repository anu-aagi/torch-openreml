.. _vig_random_intercept:

Random-Intercept Models
=======================

A random intercept allows the baseline response to vary across groups.
It is represented by a ``(Z, G)`` pair, where ``Z`` is the random-effects
design matrix and ``G`` specifies the covariance structure.

For a grouping factor ``g`` with ``m`` groups, each observation belongs
to one group. The design matrix ``Z`` contains one column per group,
with an indicator identifying the group to which each observation
belongs. The corresponding covariance matrix is

.. math::

    \boldsymbol{G} = \sigma^2 \boldsymbol{I}_m,

where ``I_m`` is an ``m``-dimensional identity matrix. This specifies
a common random-intercept variance ``sigma^2`` across groups, with
independent random intercepts.

In ``torch-openreml``, construct this pair using
:class:`~torch_openreml.covariance.DummyMatrix` for ``Z`` and
:class:`~torch_openreml.covariance.ScalarMatrix` for ``G``:

.. code-block:: python

    Z = DummyMatrix(g)
    G = ScalarMatrix(g)

Add one ``(Z, G)`` pair per grouping factor. This page is a quick
reference. See :ref:`vig_mixed_model_reml` for the full API.

Simulate a small sleep study
----------------------------

Six sites, six subjects per site, and six days per subject.

.. jupyter-execute::
    :hide-code:

    import pandas as pd
    import torch

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import DummyMatrix, ScalarMatrix

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("display.precision", 3)

    generator = torch.Generator().manual_seed(2026)
    n_sites, n_per_site, n_days = 6, 6, 6
    n_subjects = n_sites * n_per_site

    site_of_subject = torch.arange(n_subjects) // n_per_site
    subject = torch.arange(n_subjects).repeat_interleave(n_days)
    site = site_of_subject[subject]
    day = torch.arange(n_days).repeat(n_subjects)

    b_site = 20.0 * torch.randn(n_sites, generator=generator)
    b_subject = 25.0 * torch.randn(n_subjects, generator=generator)
    y = (
        250.0
        + 10.0 * day
        + b_site[site]
        + b_subject[subject]
        + 25.0 * torch.randn(len(day), generator=generator)
    )

    data = pd.DataFrame(
        {
            "site": [f"site_{s:02d}" for s in site.tolist()],
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
    from torch_openreml.covariance import DummyMatrix, ScalarMatrix

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("display.precision", 3)

    generator = torch.Generator().manual_seed(2026)
    n_sites, n_per_site, n_days = 6, 6, 6
    n_subjects = n_sites * n_per_site

    site_of_subject = torch.arange(n_subjects) // n_per_site
    subject = torch.arange(n_subjects).repeat_interleave(n_days)
    site = site_of_subject[subject]
    day = torch.arange(n_days).repeat(n_subjects)

    b_site = 20.0 * torch.randn(n_sites, generator=generator)
    b_subject = 25.0 * torch.randn(n_subjects, generator=generator)
    y = (
        250.0
        + 10.0 * day
        + b_site[site]
        + b_subject[subject]
        + 25.0 * torch.randn(len(day), generator=generator)
    )

    data = pd.DataFrame(
        {
            "site": [f"site_{s:02d}" for s in site.tolist()],
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

One random intercept
--------------------

For a subject-level intercept, pass the same label column to both classes:
``DummyMatrix`` builds ``Z``, and ``ScalarMatrix`` uses the distinct labels to
size ``G``.

.. jupyter-execute::

    Z_subject = DummyMatrix(data["subject"])
    G_subject = ScalarMatrix(data["subject"])

    reml_subject = MixedModelREML(subject=(Z_subject, G_subject))
    reml_subject.optimize(y, X)

    reml_subject.build_params(out_format="dict")

The dictionary maps parameter names to variances. Because ``R`` is omitted,
it defaults to :math:`\sigma^2 \boldsymbol{I}`, sized from the random-effect design. With one
random term, the names are ``G/sigma^2`` and ``R/sigma^2``.

Multiple random intercepts
--------------------------

Add another pair. Here site and subject are both random intercepts:

.. jupyter-execute::

    Z_site = DummyMatrix(data["site"])
    G_site = ScalarMatrix(data["site"])

    reml_site_subject = MixedModelREML(
        site=(Z_site, G_site),
        subject=(Z_subject, G_subject),
    )
    reml_site_subject.optimize(y, X)

    reml_site_subject.build_params(out_format="dict")

With several terms, the names include the term name:
``G/site/sigma^2`` and ``G/subject/sigma^2``.

After fitting
-------------

The fitted object keeps the data and parameters, so post-fit quantities are
one call away:

.. jupyter-execute::

    print("converged:", reml_site_subject.is_converged())
    print("loglik:", reml_site_subject.loglik()[0])
    print("fixed effects:", reml_site_subject.blue())
    print("fixed-effect SEs:", torch.sqrt(reml_site_subject.var_blue()))

    blups = reml_site_subject.blup_per_term()
    print("site BLUPs:", blups["site"])
    print("subject BLUPs:", blups["subject"])

``blue()`` returns the fixed effects, ``var_blue()`` their variances, and
``blup_per_term()`` one prediction vector per random term.


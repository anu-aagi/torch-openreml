.. _vig_known_covariance:

Random Effects with a Known Covariance
======================================

In some applications, the covariance structure among random-effect levels
is known up to a scale factor. For example, a kinship, pedigree,
phylogenetic, or spatial relationship matrix may be available before
fitting the model. In this case, the random effects are modelled as

.. math::

    \boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta}
    + \boldsymbol{Z}\boldsymbol{u} + \boldsymbol{e},
    \qquad
    \boldsymbol{u} \sim N(\boldsymbol{0}, \sigma^2 \boldsymbol{A}),
    \qquad
    \boldsymbol{e} \sim N(\boldsymbol{0}, \sigma_e^2 \boldsymbol{I}),

where :math:`\boldsymbol{A}` is a fixed, known covariance structure and
only the scale parameter :math:`\sigma^2` is estimated. Thus, the
random-effect covariance matrix is

.. math::

    \boldsymbol{G} = \sigma^2 \boldsymbol{A}.

In ``torch-openreml``, this structure can be specified using
:class:`~torch_openreml.covariance.Product`:

.. code-block:: python

    G = Product(
        scale=ScalarMatrix(n_levels),
        known=A,
    )

Here, ``ScalarMatrix(n_levels)`` represents
:math:`\sigma^2 \boldsymbol{I}`. ``Product`` combines it with the
known matrix :math:`\boldsymbol{A}` through ordinary matrix
multiplication to obtain :math:`\sigma^2 \boldsymbol{A}`.

This page provides a quick reference. For the special case of
independent random intercepts, see :ref:`vig_random_intercept`.

Define the known matrix
-----------------------

Consider six sites as the random-effect levels, with the known
correlation between sites :math:`i` and :math:`j` given by
:math:`A_{ij} = 0.3^{|i-j|}`. In practice, ``A`` could instead
represent a kinship, pedigree, phylogenetic, or spatial correlation
matrix.

.. jupyter-execute::

    import pandas as pd
    import torch

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import DummyMatrix, Product, ScalarMatrix

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("display.precision", 3)

    n_sites = 6
    site_index = torch.arange(n_sites)
    A = 0.3 ** (site_index[:, None] - site_index[None, :]).abs()

    A

The known matrix ``A`` must be symmetric positive semidefinite, with
dimensions matching the number of columns in the random-effects design
matrix ``Z``.

Simulate the data
-----------------

Each site is observed on 20 days. Correlated site effects are drawn
as :math:`\boldsymbol{u} = 10 \boldsymbol{L}\boldsymbol{z}`, where
:math:`\boldsymbol{A} = \boldsymbol{L}\boldsymbol{L}^\top` and
:math:`\boldsymbol{z} \sim N(\boldsymbol{0}, \boldsymbol{I})`.

.. jupyter-execute::
    :hide-code:

    generator = torch.Generator().manual_seed(2026)
    n_per_site = 20
    n_obs = n_sites * n_per_site

    site = torch.arange(n_sites).repeat_interleave(n_per_site)
    day = torch.arange(n_per_site).repeat(n_sites)

    L = torch.linalg.cholesky(A)
    z = torch.randn(n_sites, generator=generator)
    u = 10.0 * (L @ z)
    e = 6.0 * torch.randn(n_obs, generator=generator)
    y = 50.0 + 3.0 * day + u[site] + e

    data = pd.DataFrame(
        {
            "site": [f"site_{s:02d}" for s in site.tolist()],
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

    # A is defined in the setup cell above.
    generator = torch.Generator().manual_seed(2026)
    n_per_site = 20
    n_obs = n_sites * n_per_site

    site = torch.arange(n_sites).repeat_interleave(n_per_site)
    day = torch.arange(n_per_site).repeat(n_sites)

    L = torch.linalg.cholesky(A)
    z = torch.randn(n_sites, generator=generator)
    u = 10.0 * (L @ z)
    e = 6.0 * torch.randn(n_obs, generator=generator)
    y = 50.0 + 3.0 * day + u[site] + e

    data = pd.DataFrame(
        {
            "site": [f"site_{s:02d}" for s in site.tolist()],
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

Fit the model
-------------

Use ``DummyMatrix`` to construct the random-intercept design matrix for
the sites. Specify the known covariance structure in ``G`` using
``Product``. The residual covariance is omitted, so it defaults to
:math:`\sigma_e^2 \boldsymbol{I}`.

.. jupyter-execute::

    Z = DummyMatrix(data["site"])
    G = Product(scale=ScalarMatrix(data["site"]), known=A)

    reml = MixedModelREML(site=(Z, G))
    reml.optimize(y, X)

    reml.build_params(out_format="dict")

Here, ``ScalarMatrix`` takes the site labels directly and creates a
diagonal scale matrix with one row and column per distinct site.

The fitted variance parameters are named ``G/scale/sigma^2`` and
``R/sigma^2``. The former estimates the scale parameter
:math:`\sigma^2` in :math:`\boldsymbol{G} = \sigma^2 \boldsymbol{A}`,
while the latter estimates the residual variance.


What the fitted covariance looks like
-------------------------------------

The fitted random-effect covariance matrix can be inspected directly
using ``tree()``:

.. jupyter-execute::

    reml.tree()["G"]

The diagonal entries give the estimated site variances, while the
off-diagonal entries represent the fitted covariances
:math:`\hat{\sigma}^2 A_{ij}`.


After fitting
-------------

.. jupyter-execute::

    print("converged:", reml.is_converged())
    print("loglik:", reml.loglik()[0])
    print("fixed effects:", reml.blue())
    print("fixed-effect SEs:", torch.sqrt(reml.var_blue()))
    print("first five site BLUPs:")
    print(reml.blup_per_term()["site"][:5])

``blup_per_term()`` returns one BLUP per level of the site term, in the
same order as the columns of ``Z``.


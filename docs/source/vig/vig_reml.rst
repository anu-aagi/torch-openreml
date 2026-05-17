.. _vig_reml:

Introduction to REML Estimation
===============================

The :class:`~torch_openreml.REML` class implements restricted maximum
likelihood (REML) estimation for linear mixed models using the average
information (AI) algorithm. The AI algorithm combines the stability of
the expected information matrix with the curvature information of the
observed information matrix, leading to efficient and robust estimation
of variance components.

This vignette introduces the REML model formulation, explains how to
construct covariance models, demonstrates optimisation workflows, and
covers post-estimation utilities such as BLUEs, BLUPs, predictions,
residuals, and convergence diagnostics.

The model
---------

The :class:`~torch_openreml.REML` class assumes the linear mixed model

.. math::

    \mathbf{y} =
    \mathbf{X}\boldsymbol{\beta} +
    \mathbf{Z}\mathbf{b} +
    \boldsymbol{\varepsilon},

where

- :math:`\mathbf{y}` is the response vector of length :math:`n`,
- :math:`\mathbf{X}` is the fixed-effects design matrix,
- :math:`\boldsymbol{\beta}` is the vector of fixed effects,
- :math:`\mathbf{Z}` is the random-effects design matrix,
- :math:`\mathbf{b}` is the vector of random effects,
- :math:`\boldsymbol{\varepsilon}` is the residual error vector.

The random effects and residuals are assumed independent with

.. math::

    \mathbf{b} \sim \mathcal{N}(\mathbf{0}, \mathbf{G}), \qquad
    \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}).

The marginal covariance of :math:`\mathbf{y}` is therefore

.. math::

    \mathbf{V}(\boldsymbol{\theta}) =
    \mathbf{Z}\mathbf{G}(\boldsymbol{\theta})\mathbf{Z}^\top +
    \mathbf{R}(\boldsymbol{\theta}),

where :math:`\boldsymbol{\theta}` denotes the variance-component
parameters.

The REML estimator maximises the restricted log-likelihood

.. math::

    \ell_R(\boldsymbol{\theta}) =
    -\frac{1}{2}
    \Bigl(
        \log |\mathbf{V}|
        +
        \log |\mathbf{X}^\top \mathbf{V}^{-1} \mathbf{X}|
        +
        \mathbf{y}^\top \mathbf{P} \mathbf{y}
    \Bigr),

where

.. math::

    \mathbf{P} =
    \mathbf{V}^{-1}
    -
    \mathbf{V}^{-1}\mathbf{X}
    \left(
        \mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}
    \right)^{-1}
    \mathbf{X}^\top \mathbf{V}^{-1}

is the residual projection matrix.

.. important::

    Internally, :class:`~torch_openreml.REML` repeatedly evaluates

    - the covariance matrix :math:`\mathbf{V}`,
    - its derivatives :math:`\partial \mathbf{V} / \partial \theta_k`,
    - the score vector,
    - and the average information matrix,

    until convergence criteria are satisfied.

Constructing the REML object
----------------------------

The :class:`~torch_openreml.REML` constructor supports multiple ways to
define the covariance structure. The appropriate approach depends on the
complexity of the model and the desired level of control.

The most important constructor arguments are:

``map_theta_to_v``
    Callable mapping :math:`\boldsymbol{\theta}` to the marginal
    covariance matrix :math:`\mathbf{V}`.

``map_theta_to_dv``
    Callable returning the derivatives
    :math:`\partial\mathbf{V}/\partial\theta_k`.

``v_builder``
    A covariance :class:`~torch_openreml.covariance.matrix.Matrix`
    object representing the covariance structure.

``map_theta_to_g``
    Optional mapping from parameters to the random-effects covariance
    matrix :math:`\mathbf{G}`. Required for BLUPs and conditional
    predictions.

``mask_theta_to_g``
    Boolean mask selecting which parameters are passed to
    ``map_theta_to_g``.

Exactly one of ``map_theta_to_v`` or ``v_builder`` must be supplied.

Via ``map_theta_to_v`` only
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The minimal interface requires only a function mapping parameters to the
marginal covariance matrix.

.. code-block:: python

    import torch
    from torch_openreml import REML

    n = 50

    def map_theta_to_v(theta):

        sigma2 = theta[0].exp()
        rho = torch.sigmoid(theta[1])

        I = torch.eye(n)
        J = torch.ones(n, n)

        return sigma2 * (I + rho * J)

    reml = REML(map_theta_to_v=map_theta_to_v)

In this example:

- ``theta[0]`` controls the variance scale,
- ``theta[1]`` controls a correlation parameter,
- exponential and sigmoid transforms enforce parameter constraints.

The covariance matrix is

.. math::

    \mathbf{V} =
    \sigma^2 (\mathbf{I} + \rho \mathbf{J}),

where :math:`\mathbf{J}` is the all-ones matrix.

When only ``map_theta_to_v`` is supplied,
:class:`~torch_openreml.REML` computes derivatives automatically using
:func:`torch.func.jacrev`.

This approach is convenient for prototyping, but automatic differentiation can become expensive for large covariance matrices. At the same time, it offers essentially unrestricted flexibility: the marginal covariance can be constructed through arbitrary differentiable PyTorch operations, ranging from simple parameterizations to highly sophisticated matrix algebra, iterative procedures, decompositions, simulation-based constructions, dynamically assembled covariance components, or even neural networks that directly output or parameterize covariance structure.

Because the covariance construction is defined directly in Python/PyTorch code rather than through a fixed covariance specification, users are free to incorporate conditional branching, stochastic generation, adaptive logic, external modules, or even entirely different covariance structures across optimization iterations (although such behavior is usually not statistically meaningful in practice). In effect, any covariance model that can be expressed as a differentiable computational graph in PyTorch can be used within this framework.

Via ``map_theta_to_v`` and ``map_theta_to_dv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For better performance, analytical derivatives can be supplied
explicitly.

.. code-block:: python

    import torch
    from torch_openreml import REML

    n = 50

    def map_theta_to_v(theta):

        sigma2 = theta[0].exp()
        rho = torch.sigmoid(theta[1])

        I = torch.eye(n)
        J = torch.ones(n, n)

        return sigma2 * (I + rho * J)

    def map_theta_to_dv(theta):

        sigma2 = theta[0].exp()
        rho = torch.sigmoid(theta[1])

        dsigma2 = sigma2
        drho = rho * (1 - rho)

        I = torch.eye(n)
        J = torch.ones(n, n)

        dV_dtheta0 = dsigma2 * (I + rho * J)
        dV_dtheta1 = sigma2 * drho * J

        return torch.stack([dV_dtheta0, dV_dtheta1])

    reml = REML(
        map_theta_to_v=map_theta_to_v,
        map_theta_to_dv=map_theta_to_dv,
    )

The derivative function must return a tensor of shape
``(num_free_params, n, n)``, where slice ``k`` corresponds to

.. math::

    \frac{\partial \mathbf{V}}
    {\partial \theta_k}.

Providing analytical derivatives is strongly recommended for large
models because derivative evaluation is typically one of the most
computationally expensive parts of REML optimisation.

Via a covariance Matrix builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For realistic mixed models, the recommended workflow is to construct the
covariance structure using the
:mod:`torch_openreml.covariance` matrix system.

The covariance builder API provides reusable covariance components such
as:

- :class:`~torch_openreml.covariance.ScalarMatrix`
- :class:`~torch_openreml.covariance.IdentityMatrix`
- :class:`~torch_openreml.covariance.DiagonalMatrix`
- :class:`~torch_openreml.covariance.AR1Matrix`
- :class:`~torch_openreml.covariance.CompoundSymmetricMatrix`

along with composition operators including:

- :class:`~torch_openreml.covariance.Sum`
- :class:`~torch_openreml.covariance.BlockDiagonal`
- :class:`~torch_openreml.covariance.KroneckerProduct`
- :class:`~torch_openreml.covariance.CovariancePropagation`

Example:

.. code-block:: python

    import torch

    from torch_openreml import REML
    from torch_openreml.covariance import (
        DummyMatrix,
        ScalarMatrix,
        CovariancePropagation,
        Sum,
    )

    n, p = 50, 2

    y = torch.randn(n)
    X = torch.randn(n, p)

    Z = DummyMatrix(["a", "b"] * 25)()

    V = Sum(
        CovariancePropagation(
            Z,
            ScalarMatrix(2),
        ),
        ScalarMatrix(n),
    )

    reml = REML(v_builder=V)

The builder automatically manages:

- parameter bookkeeping,
- parameter transforms,
- covariance assembly,
- covariance derivatives.

Internally, :class:`~torch_openreml.REML` calls

- :meth:`~torch_openreml.covariance.matrix.Matrix.map_theta_to_v`
- :meth:`~torch_openreml.covariance.matrix.Matrix.map_theta_to_dv`

provided by the builder.

The derivative calculation uses
:meth:`~torch_openreml.covariance.matrix.Matrix.grad`,
which attempts a closed-form
:meth:`~torch_openreml.covariance.matrix.Matrix.manual_grad`
implementation first and falls back to automatic differentiation when
necessary.

Providing ``map_theta_to_g``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some methods require access to the random-effects covariance matrix
:math:`\mathbf{G}` separately from the marginal covariance matrix
:math:`\mathbf{V}`.

These include:

- :meth:`~torch_openreml.REML.blup`
- :meth:`~torch_openreml.REML.predict`
- :meth:`~torch_openreml.REML.residual`

The ``map_theta_to_g`` argument may be:

- a callable,
- a covariance :class:`~torch_openreml.covariance.matrix.Matrix`,
- or ``None``.

Example:

.. code-block:: python

    G = ScalarMatrix(2)

    reml = REML(
        v_builder=V,
        map_theta_to_g=G,
        mask_theta_to_g=torch.tensor([True, False]),
    )

The mask specifies which elements of
:math:`\boldsymbol{\theta}` are passed into ``G``.

This is useful when the full parameter vector contains both random-effect
and residual variance parameters.

Running the optimiser
---------------------

Once the REML object is constructed, optimisation is performed using
:meth:`~torch_openreml.REML.optimize`.

.. jupyter-execute::


    import torch

    from torch_openreml import REML
    from torch_openreml.covariance import (
        DummyMatrix,
        ScalarMatrix,
        CovariancePropagation,
        Sum,
    )

    n, p = 50, 2

    y = torch.randn(n)
    X = torch.randn(n, p)

    Z = DummyMatrix(["a", "b"] * 25)()

    V = Sum(
        CovariancePropagation(
            Z,
            ScalarMatrix(2),
        ),
        ScalarMatrix(n),
    )

    reml = REML(v_builder=V)

    theta_start = torch.zeros(V.num_free_params)

    theta_hat, beta_hat, n_iter = reml.optimize(
        y,
        X,
        theta_start,
        verbose=2,
    )

Inputs
~~~~~~

``y``
    Response vector of shape ``(n,)``.

``X``
    Fixed-effects design matrix of shape ``(n, p)``.

``theta_start``
    Initial parameter vector.

Choosing sensible starting values can substantially improve convergence,
especially for complex covariance structures.

Returned values
~~~~~~~~~~~~~~~

``theta_hat``
    Final variance-component estimates.

``beta_hat``
    Estimated fixed effects evaluated at ``theta_hat``.

``n_iter``
    Number of completed optimisation iterations.

Optimisation controls
~~~~~~~~~~~~~~~~~~~~~

The optimisation routine supports several keyword arguments.

``max_iter`` (default ``200``)
    Maximum number of iterations.

``eta`` (default ``1.0``)
    Step-size multiplier applied to the AI update.

    The parameter update is

    .. math::

        \boldsymbol{\theta}^{(t+1)}
        =
        \boldsymbol{\theta}^{(t)}
        +
        \eta \,
        \mathbf{AI}^{-1}\mathbf{s}.

    Smaller values of ``eta`` can improve stability when the optimisation
    oscillates or diverges.

``lb`` and ``ub``
    Optional lower and upper parameter bounds applied after each update.

``verbose`` (default ``0``)
    Controls optimisation output.

    - ``0``: silent
    - ``1``: progress bar
    - ``2``: detailed iteration diagnostics

Detailed diagnostics include:

- score norm,
- update norm,
- log-likelihood,
- learning rate,
- iteration count.

Convergence criteria
--------------------

At each iteration, convergence is evaluated using three criteria.

Score norm
~~~~~~~~~~

The score vector must satisfy

.. math::

    \|\mathbf{s}\| < \texttt{tol\_score}.

This checks whether the gradient is close to zero.

Parameter update norm
~~~~~~~~~~~~~~~~~~~~~

The parameter update must satisfy

.. math::

    \|\Delta\| < \texttt{tol\_delta}.

This checks whether the parameters have stabilised.

Log-likelihood change
~~~~~~~~~~~~~~~~~~~~~

The restricted log-likelihood change must satisfy

.. math::

    |\ell_R^{(t)} - \ell_R^{(t-1)}|
    <
    \texttt{tol\_loglik}.

Default tolerances
~~~~~~~~~~~~~~~~~~

The default tolerances are:

- ``tol_score = 1e-4``
- ``tol_delta = 1e-4``
- ``tol_loglik = 1e-4``

All enabled criteria must be satisfied simultaneously.

Criteria may be individually disabled using:

- ``check_score=False``
- ``check_delta=False``
- ``check_loglik=False``

At least two iterations are required before convergence can be declared.

Post-estimation methods
-----------------------

After optimisation, the fitted REML object provides several utilities for
extracting estimates and predictions.

Retrieving parameter estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    theta_last = reml.get_theta(select="last")
    theta_best = reml.get_theta(select="best")

    beta_last = reml.get_beta(select="last")
    beta_best = reml.get_beta(select="best")

``select="last"``
    Returns the final optimisation iterate.

``select="best"``
    Returns the iterate with the highest restricted log-likelihood.

The ``"best"`` option is useful if optimisation temporarily overshoots
or oscillates near convergence.

BLUE — fixed effects
~~~~~~~~~~~~~~~~~~~~

The best linear unbiased estimator (BLUE) of the fixed effects is

.. math::

    \widehat{\boldsymbol{\beta}}
    =
    \left(
        \mathbf{X}^\top
        \mathbf{V}^{-1}
        \mathbf{X}
    \right)^{-1}
    \mathbf{X}^\top
    \mathbf{V}^{-1}
    \mathbf{y}.

Compute it using:

.. jupyter-execute::

    beta_hat = reml.blue(y, X, theta_hat)

BLUP — random effects
~~~~~~~~~~~~~~~~~~~~~

The best linear unbiased predictor (BLUP) of the random effects is

.. math::

    \widehat{\mathbf{b}}
    =
    \mathbf{G}
    \mathbf{Z}^\top
    \mathbf{V}^{-1}
    \left(
        \mathbf{y}
        -
        \mathbf{X}\widehat{\boldsymbol{\beta}}
    \right).

Compute it using:

.. jupyter-execute::

    b_hat = reml.blup(
        y,
        X,
        Z,
        theta_hat,
        map_theta_to_g=ScalarMatrix(2),
        mask_theta_to_g=torch.tensor([True, False]),
    )

This method requires ``map_theta_to_g``.

Predictions
~~~~~~~~~~~

Marginal predictions use only fixed effects:

.. math::

    \widehat{\mathbf{y}}_{\mathrm{marginal}}
    =
    \mathbf{X}\widehat{\boldsymbol{\beta}}.

Conditional predictions additionally include random effects:

.. math::

    \widehat{\mathbf{y}}_{\mathrm{conditional}}
    =
    \mathbf{X}\widehat{\boldsymbol{\beta}}
    +
    \mathbf{Z}\widehat{\mathbf{b}}.

Example:

.. jupyter-execute::

    y_hat_marginal = reml.marginal_predict(
        y,
        X,
        theta_hat,
    )

    y_hat_conditional = reml.predict(
        y,
        X,
        Z,
        theta_hat,
        map_theta_to_g=ScalarMatrix(2),
        mask_theta_to_g=torch.tensor([True, False]),
    )

Residuals
~~~~~~~~~

Marginal residuals:

.. math::

    \mathbf{e}_{\mathrm{marginal}}
    =
    \mathbf{y}
    -
    \widehat{\mathbf{y}}_{\mathrm{marginal}}.

Conditional residuals:

.. math::

    \mathbf{e}_{\mathrm{conditional}}
    =
    \mathbf{y}
    -
    \widehat{\mathbf{y}}_{\mathrm{conditional}}.

Example:

.. jupyter-execute::

    e_marginal = reml.marginal_residual(
        y,
        X,
        theta_hat,
    )

    e_conditional = reml.residual(
        y,
        X,
        Z,
        theta_hat,
        map_theta_to_g=ScalarMatrix(2),
        mask_theta_to_g=torch.tensor([True, False]),
    )

Evaluating the log-likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The REML log-likelihood can be evaluated directly without optimisation.

.. jupyter-execute::

    loglik = reml.loglik(y, X, theta_hat)

This is useful for:

- debugging,
- likelihood profiling,
- model comparison,
- monitoring optimisation trajectories.

Complete example
----------------

The following example demonstrates a mixed model with:

- genotype random effects,
- replicate-block random effects,
- residual variance.

.. jupyter-execute::

    import torch

    from torch_openreml import REML
    from torch_openreml.utils import augment, n_distinct

    from torch_openreml.covariance import (
        DummyMatrix,
        IdentityMatrix,
        ScalarMatrix,
        Sum,
        CovariancePropagation,
        KroneckerProduct,
    )

    from torch_openreml.example_data import john_alpha

    # --- response ---
    y = torch.tensor(john_alpha["yield"].values)

    # --- fixed effects ---
    X = augment(
        torch.ones(len(john_alpha), 1),
        DummyMatrix(john_alpha["rep"], drop_first=True)()
    )

    # --- random effect design matrices ---
    Z_gen = DummyMatrix(john_alpha["gen"])
    Z_rep_block = DummyMatrix(john_alpha["rep"], john_alpha["block"])

    # --- covariance components ---
    G_gen = ScalarMatrix(n_distinct(john_alpha["gen"]))
    G_rep = IdentityMatrix(n_distinct(john_alpha["rep"]))
    G_block = ScalarMatrix(n_distinct(john_alpha["block"]))

    R = ScalarMatrix(len(john_alpha))

    # --- marginal covariance ---
    V = Sum(
        CovariancePropagation(Z_gen, G_gen),
        CovariancePropagation(
            Z_rep_block,
            KroneckerProduct(G_rep, G_block)
        ),
        R
    )

    # --- REML fit ---
    reml = REML(v_builder=V)

    theta_start = torch.zeros(V.num_free_params)

    theta_hat, beta_hat, n_iter = reml.optimize(
        y,
        X,
        theta_start,
        verbose=2,
    )

    # --- results ---
    print("theta:", theta_hat)

    print("variance components:", V.build_params(theta_hat))

    print("fixed effects:", beta_hat)

    print("loglik:", reml.loglik(y, X, theta_hat))

Optimisation history
--------------------

After optimisation, the full iteration history is stored in
:attr:`reml.history <torch_openreml.REML.history>`.

This dictionary contains per-iteration records including:

- ``theta``
- ``beta``
- ``loglik``
- ``score``
- ``ai``
- ``delta``
- ``update``

This history is useful for:

- diagnosing convergence problems,
- inspecting optimisation trajectories,
- plotting likelihood curves,
- monitoring parameter stability.

Example:

.. jupyter-execute::

    scores = [
        torch.norm(s).item()
        for s in reml.history["score"]
    ]

    logliks = [
        ll.item()
        for ll in reml.history["loglik"]
    ]

    print(
        "Score norms:",
        [f"{s:.6f}" for s in scores],
    )

    print(
        "Log-likelihoods:",
        [f"{ll:.4f}" for ll in logliks],
    )

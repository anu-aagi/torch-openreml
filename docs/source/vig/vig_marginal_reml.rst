.. _vig_reml:

Introduction to Marginal REML Estimation
========================================

The :class:`~torch_openreml.MarginalREML` class implements restricted maximum
likelihood (REML) estimation for generalised least squares with a parametric
marginal covariance matrix, using the average information (AI) algorithm.
The AI algorithm combines the stability of the expected information matrix
with the curvature information of the observed information matrix, leading to
efficient and robust estimation of covariance parameters.

This vignette introduces the marginal REML model formulation, explains how to
construct covariance models, demonstrates optimisation workflows, and
covers post-estimation utilities such as BLUEs, marginal predictions,
residuals, and convergence diagnostics.

The model
---------

The :class:`~torch_openreml.MarginalREML` class assumes the generalised
least squares model

.. math::

    \mathbf{y} \sim \mathcal{N}(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}(\boldsymbol{\theta})),

where

- :math:`\mathbf{y}` is the response vector of length :math:`n`,
- :math:`\mathbf{X}` is the fixed-effects design matrix,
- :math:`\boldsymbol{\beta}` is the vector of fixed effects,
- :math:`\mathbf{V}(\boldsymbol{\theta})` is a parametric marginal covariance
  matrix parameterised by :math:`\boldsymbol{\theta}`.

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

    Internally, :class:`~torch_openreml.MarginalREML` repeatedly evaluates

    - the covariance matrix :math:`\mathbf{V}`,
    - its derivatives :math:`\partial \mathbf{V} / \partial \theta_k`,
    - the score vector,
    - and the average information matrix,

    until convergence criteria are satisfied.

Constructing the REML object
----------------------------

The :class:`~torch_openreml.MarginalREML` constructor takes a single
:class:`~torch_openreml.covariance.matrix.Matrix` instance ``v`` that
defines the marginal covariance structure :math:`\mathbf{V}`.

.. code-block:: python

    from torch_openreml import MarginalREML

    reml = MarginalREML(v)

There are two main ways to build ``v``, depending on complexity:

- :class:`~torch_openreml.covariance.SimpleMatrix` — wrap a plain
  function for quick prototyping.
- The :mod:`~torch_openreml.covariance` builder system — compose
  pre-built covariance components with operators.

The MarginalREML class works directly with the marginal covariance matrix
:math:`\mathbf{V}` and does not require separate :math:`\mathbf{G}` (random)
or :math:`\mathbf{R}` (error) components as in mixed model.

Via ``SimpleMatrix`` (function-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick prototyping, :class:`~torch_openreml.covariance.SimpleMatrix`
wraps a plain function. All parameters are free and unconstrained.

.. code-block:: python

    import torch
    from torch_openreml import MarginalREML
    from torch_openreml.covariance import SimpleMatrix

    n = 50

    def my_v(theta):

        sigma2 = theta[0].exp()
        rho = torch.sigmoid(theta[1])

        I = torch.eye(n)
        J = torch.ones(n, n)

        return sigma2 * (I + rho * J)

    v = SimpleMatrix(num_free_params=2, call=my_v)
    reml = MarginalREML(v)

In this example:

- ``theta[0]`` controls the variance scale,
- ``theta[1]`` controls a correlation parameter,
- exponential and sigmoid transforms enforce parameter constraints.

The covariance matrix is

.. math::

    \mathbf{V} =
    \sigma^2 (\mathbf{I} + \rho \mathbf{J}),

where :math:`\mathbf{J}` is the all-ones matrix.

When no ``manual_grad`` is supplied,
:class:`~torch_openreml.covariance.SimpleMatrix` computes derivatives
automatically via :meth:`~torch_openreml.covariance.matrix.Matrix.auto_grad`.

This approach is convenient for prototyping, but automatic differentiation can become expensive for large covariance matrices. At the same time, it offers essentially unrestricted flexibility: the marginal covariance can be constructed through arbitrary differentiable PyTorch operations, ranging from simple parameterizations to highly sophisticated matrix algebra, iterative procedures, decompositions, simulation-based constructions, dynamically assembled covariance components, or even neural networks that directly output or parameterize covariance structure.

Because the covariance construction is defined directly in Python/PyTorch code rather than through a fixed covariance specification, users are free to incorporate conditional branching, stochastic generation, adaptive logic, external modules, or even entirely different covariance structures across optimization iterations (although such behavior is usually not statistically meaningful in practice). In effect, any covariance model that can be expressed as a differentiable computational graph in PyTorch can be used within this framework.

Via ``SimpleMatrix`` with manual gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For better performance, analytical derivatives can be supplied via the
``manual_grad`` argument.

.. code-block:: python

    import torch
    from torch_openreml import MarginalREML
    from torch_openreml.covariance import SimpleMatrix

    n = 50

    def my_v(theta):

        sigma2 = theta[0].exp()
        rho = torch.sigmoid(theta[1])

        I = torch.eye(n)
        J = torch.ones(n, n)

        return sigma2 * (I + rho * J)

    def my_dv(theta):

        sigma2 = theta[0].exp()
        rho = torch.sigmoid(theta[1])

        dsigma2 = sigma2
        drho = rho * (1 - rho)

        I = torch.eye(n)
        J = torch.ones(n, n)

        dV_dtheta0 = dsigma2 * (I + rho * J)
        dV_dtheta1 = sigma2 * drho * J

        grad = torch.stack([dV_dtheta0, dV_dtheta1])
        return grad, ["sigma2", "rho"]

    v = SimpleMatrix(num_free_params=2, call=my_v, manual_grad=my_dv)
    reml = MarginalREML(v)

The gradient function must return a tuple ``(grad, grad_names)``, where
``grad`` is a tensor of shape ``(num_free_params, n, n)`` and
``grad_names`` is a list of parameter name strings. Slice ``k`` of
``grad`` corresponds to

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

along with composition operators such as:

- :class:`~torch_openreml.covariance.Sum`
- :class:`~torch_openreml.covariance.BlockDiagonal`
- :class:`~torch_openreml.covariance.KroneckerProduct`
- :class:`~torch_openreml.covariance.CovariancePropagation`

Example:

.. code-block:: python

    import torch

    from torch_openreml import MarginalREML
    from torch_openreml.covariance import DummyMatrix, ScalarMatrix, CovariancePropagation, Sum

    n, p = 50, 2

    y = torch.randn(n)
    X = torch.randn(n, p)

    Z = DummyMatrix(["a", "b"] * 25)

    V = Sum(
        CovariancePropagation(
            Z,
            ScalarMatrix(2),
        ),
        ScalarMatrix(n),
    )

    reml = MarginalREML(V)

The builder automatically manages:

- parameter bookkeeping,
- parameter transforms,
- covariance assembly,
- covariance derivatives.

Internally, :class:`~torch_openreml.MarginalREML` calls

- :meth:`~torch_openreml.covariance.matrix.Matrix.__call__` to build
  :math:`\mathbf{V}`,
- :meth:`~torch_openreml.covariance.matrix.Matrix.grad` for the
  Jacobian.

The derivative calculation uses
:meth:`~torch_openreml.covariance.matrix.Matrix.grad`,
which attempts a closed-form
:meth:`~torch_openreml.covariance.matrix.Matrix.manual_grad`
implementation first and falls back to automatic differentiation when
necessary.

Running the optimiser
---------------------

Once the REML object is constructed, optimisation is performed using
:meth:`~torch_openreml.MarginalREML.optimize`.

.. jupyter-execute::


    import torch

    from torch_openreml import MarginalREML
    from torch_openreml.covariance import DummyMatrix, ScalarMatrix, CovariancePropagation, Sum

    n, p = 50, 2

    y = torch.randn(n)
    X = torch.randn(n, p)

    Z = DummyMatrix(["a", "b"] * 25)

    V = Sum(
        CovariancePropagation(
            Z,
            ScalarMatrix(2),
        ),
        ScalarMatrix(n),
    )

    reml = MarginalREML(V)

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

``trace_approx`` (default ``False``)
    Whether to start the optimisation with the stochastic probe estimate for
    the score. The estimator is Hutch++ — Meyer, Musco, Musco & Woodruff,
    *Hutch++: Optimal stochastic trace estimation*, SOSA 2021, pp. 142--155,
    doi:`10.1137/1.9781611976496.16 <https://doi.org/10.1137/1.9781611976496.16>`_.

    With

    .. math::

        \mathbf{A}_k =
        \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k},
        \qquad
        \mathbf{B}_k =
        \mathbf{V}^{-1} \mathbf{X}
        \left(\mathbf{X}^\top \mathbf{V}^{-1} \mathbf{X}\right)^{-1}
        \mathbf{X}^\top \mathbf{V}^{-1}
        \frac{\partial \mathbf{V}}{\partial \theta_k}

    the trace term of the score splits as
    :math:`\mathrm{tr}(\mathbf{P} \, \partial \mathbf{V} / \partial
    \theta_k) = \mathrm{tr}(\mathbf{A}_k) - \mathrm{tr}(\mathbf{B}_k)`, and
    the two terms are estimated separately, so neither matrix is formed in
    full.

    The estimate of a matrix :math:`\mathbf{C}` draws two independent
    Rademacher probe matrices :math:`\mathbf{S}, \mathbf{G} \in \{\pm
    1\}^{n \times m}` with :math:`m = \lfloor n f \rfloor + 1` probes each,
    where :math:`f` is ``subspace_fraction``, and combines a QR sketch with
    Hutchinson's estimator on the deflated residual:

    .. math::

        \mathbf{Q} = \mathrm{qr}(\mathbf{C} \mathbf{S}),
        \qquad
        \hat{h}(\mathbf{C}) =
        \mathrm{tr}(\mathbf{Q}^\top \mathbf{C} \mathbf{Q})
        +
        \frac{1}{m}
        \mathrm{tr}\!\left(
            \mathbf{G}^\top
            (\mathbf{I} - \mathbf{Q}\mathbf{Q}^\top)
            \mathbf{C}
            (\mathbf{I} - \mathbf{Q}\mathbf{Q}^\top)
            \mathbf{G}
        \right).

    :math:`\mathbf{Q}` spans the dominant range of :math:`\mathbf{C}`, so
    the first term is exact there; the second applies Hutchinson's estimator
    to the deflated residual.

    The estimate has an inherent noise floor: its error remains roughly
    constant even as the optimisation approaches the optimum, while the true
    score norm tends to zero. Far from the optimum, the score is much larger
    than this error, so the estimate is sufficiently accurate to make
    progress. Once the true score norm becomes comparable to the error,
    however, the norm seen by the optimiser stops decreasing and plateaus. As
    a result, ``tol_score`` is never reached, and the optimisation continues
    until ``max_iter`` rather than stopping. The average information matrix
    and log-likelihood are exact at every iteration and are computed
    regardless, so these additional iterations provide no benefit. In fact, a
    run that uses the approximation throughout can cost more than one that
    uses the exact score from the beginning.

    To avoid this, the optimiser monitors the score norm and switches to the
    exact score once the norm falls below ``exact_score_threshold``. The
    exact score is then used for that iteration and all subsequent
    iterations. This is what makes the approximation worthwhile: it
    accelerates the early iterations, when the score is well above the noise
    floor, while leaving the final stage of optimisation to the exact score.
    By default, ``trace_approx=False``, so the entire optimisation uses the
    exact score. Set ``trace_approx=True`` to opt in to the approximation.

``subspace_fraction`` (default ``0.01``)
    Fraction of the ``n`` observations used as the dimension of the random
    probe subspace in the stochastic trace estimate that forms the score on
    iterations that are still far from the optimum.

    Each of the two probe matrices holds ``int(n * subspace_fraction) + 1``
    vectors, so the value must lie in ``[0, 1]`` and the total probe budget
    is about twice that. Larger values give a more accurate approximate
    score at a higher cost per iteration. Only used when
    ``trace_approx=True``. Only the score is affected: the average
    information matrix and the log-likelihood are always computed exactly.

``exact_score_threshold`` (default ``10.0``)
    Score norm below which the stochastic probe estimate is switched off for
    the remainder of the optimisation.

    Since the first score is always computed with the current setting,
    ``inf`` switches the approximation off after the first iteration while
    ``0.0`` keeps it on throughout. Use ``trace_approx=False`` instead to skip it
    entirely.

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

Predictions
~~~~~~~~~~~

Predictions use the fixed effects:

.. math::

    \widehat{\mathbf{y}}
    =
    \mathbf{X}\widehat{\boldsymbol{\beta}}.

Example:

.. jupyter-execute::

    y_hat = reml.predict(
        y,
        X,
        theta_hat,
    )

Residuals
~~~~~~~~~

Residuals:

.. math::

    \mathbf{e}
    =
    \mathbf{y}
    -
    \widehat{\mathbf{y}}.

Example:

.. jupyter-execute::

    e = reml.residual(
        y,
        X,
        theta_hat,
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

    from torch_openreml import MarginalREML
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
    reml = MarginalREML(V)

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
:attr:`reml.history <torch_openreml.MarginalREML.history>`.

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

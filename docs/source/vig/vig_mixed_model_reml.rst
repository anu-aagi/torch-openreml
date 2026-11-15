.. _vig_mixed_model_reml:

Introduction to Mixed Model REML Estimation
===========================================

The :class:`~torch_openreml.MixedModelREML` class implements restricted
maximum likelihood (REML) estimation for linear mixed models, using the
average information (AI) algorithm. The AI algorithm combines the stability of
the expected information matrix with the curvature information of the observed
information matrix, leading to efficient and robust estimation of covariance
parameters.

This vignette introduces the mixed model formulation, explains how to specify
random-effect and residual terms, demonstrates optimisation workflows, and
covers post-estimation utilities such as BLUEs, BLUPs, marginal predictions,
residuals, and convergence diagnostics.

The model
---------

The :class:`~torch_openreml.MixedModelREML` class assumes the linear mixed
model

.. math::

    \mathbf{y}
    =
    \mathbf{X}\boldsymbol{\beta}
    +
    \sum_{i=1}^{b}\mathbf{Z}_i(\boldsymbol{\theta})\mathbf{u}_i
    +
    \boldsymbol{\epsilon},
    \qquad
    \mathbf{u}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{G}_i(\boldsymbol{\theta})),
    \qquad
    \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}(\boldsymbol{\theta})),

where

- :math:`\mathbf{y}` is the response vector of length :math:`n`,
- :math:`\mathbf{X}` is the fixed-effects design matrix,
- :math:`\boldsymbol{\beta}` is the vector of fixed effects,
- :math:`b` is the number of random-effect terms,
- :math:`\mathbf{Z}_i(\boldsymbol{\theta})` is the incidence matrix of the
  :math:`i`-th random-effect term, which may be fixed or parameterised,
- :math:`\mathbf{u}_i` is the vector of random effects of the :math:`i`-th
  term,
- :math:`\mathbf{G}_i(\boldsymbol{\theta})` is the covariance matrix of
  :math:`\mathbf{u}_i`,
- :math:`\boldsymbol{\epsilon}` is the vector of residual errors, and
- :math:`\mathbf{R}(\boldsymbol{\theta})` is the residual covariance matrix.

The random effects and the residual errors are independent, so the response
has the marginal distribution

.. math::

    \mathbf{y} \sim \mathcal{N}(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}(\boldsymbol{\theta})),

where the marginal covariance matrix is

.. math::

    \mathbf{V}(\boldsymbol{\theta})
    =
    \mathbf{R}(\boldsymbol{\theta})
    +
    \sum_{i=1}^{b}
    \mathbf{Z}_i(\boldsymbol{\theta})
    \mathbf{G}_i(\boldsymbol{\theta})
    \mathbf{Z}_i(\boldsymbol{\theta})^\top,

with :math:`\boldsymbol{\theta}` the variance-component vector collecting the
parameters of the random-effect covariances, the residual covariance, and any
parameterised incidence matrices.

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

The C-form evaluates the same log-likelihood without forming
:math:`\mathbf{V}`.  Write :math:`\mathbf{Z}` for the combined incidence
matrix of all :math:`b` terms and :math:`\mathbf{G}` for the block-diagonal
covariance of their random effects, and let

.. math::

    \mathbf{W} = [\mathbf{X}, \mathbf{Z}],
    \qquad
    \mathbf{C}
    =
    \mathbf{W}^\top \mathbf{R}^{-1} \mathbf{W}
    +
    \operatorname{blockdiag}(\mathbf{0}, \mathbf{G}^{-1}),
    \qquad
    \mathbf{U} = \mathbf{R}^{-1}\mathbf{W}.

The projection matrix is then

.. math::

    \mathbf{P}
    =
    \mathbf{R}^{-1}
    -
    \mathbf{U}\mathbf{C}^{-1}\mathbf{U}^\top,

and the restricted log-likelihood becomes

.. math::

    \ell_R(\boldsymbol{\theta})
    =
    -\frac{1}{2}
    \Bigl(
        \log |\mathbf{C}|
        +
        \log |\mathbf{R}|
        +
        \log |\mathbf{G}|
        +
        \mathbf{y}^\top \mathbf{P} \mathbf{y}
    \Bigr),

which involves only :math:`\mathbf{C}`, :math:`\mathbf{R}` and
:math:`\mathbf{G}`.  This is the form used by
:class:`~torch_openreml.methods.CForm`.

.. important::

    Internally, :class:`~torch_openreml.MixedModelREML` repeatedly evaluates

    - the score vector,
    - and the average information matrix,

    until convergence criteria are satisfied.  The default
    :class:`~torch_openreml.methods.CForm` method derives these from the
    random-effect and residual covariances
    :math:`\mathbf{Z}_i`, :math:`\mathbf{G}_i` and :math:`\mathbf{R}` and
    their derivatives, forming the projection :math:`\mathbf{P}` from that
    block structure without ever assembling the :math:`n \times n` marginal
    covariance :math:`\mathbf{V}`.  The alternative
    :class:`~torch_openreml.methods.Marginal` method instead evaluates
    :math:`\mathbf{V}` and its derivatives
    :math:`\partial \mathbf{V} / \partial \theta_k` directly.

Constructing the REML object
----------------------------

The :class:`~torch_openreml.MixedModelREML` constructor accepts random-effect terms as ``(design, covariance)`` pairs, where ``design`` is the incidence matrix :math:`\mathbf{Z}_i` and ``covariance`` represents :math:`\mathbf{G}_i`. Terms can be supplied positionally or by keyword. Positional terms are named ``random_term_0``, ``random_term_1``, and so on, in the order supplied, and precede keyword terms. Keyword names must be unique and cannot be ``R`` or ``method``. Each term name defines a namespace for its parameters and serves as the key in post-estimation methods that report results by term, such as :meth:`~torch_openreml.MixedModelREML.blup_per_term`.

Both ``design`` and ``covariance`` can be specified as either a tensor or a :class:`~torch_openreml.covariance.matrix.Matrix`. Tensors are treated as fixed matrices, while :class:`~torch_openreml.covariance.matrix.Matrix` objects may contain free parameters that are estimated jointly with the other model parameters.

The residual covariance ``R`` accepts either a :class:`~torch_openreml.covariance.matrix.Matrix` or a two-dimensional tensor. If omitted, it defaults to :math:`\sigma^2 \mathbf{I}`, with its dimension inferred from the random-effect design. It must be supplied explicitly when the model has no random-effect terms.

.. code-block:: python

    from torch_openreml import MixedModelREML

    reml = MixedModelREML((Z, G), R=R)


Via ``SimpleMatrix`` (function-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick prototyping, :class:`~torch_openreml.covariance.SimpleMatrix` provides a convenient way to define a matrix through a user-supplied function. All parameters are free and unconstrained, so transformations can be applied within the function to enforce parameter constraints.

.. code-block:: python

    import torch
    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import SimpleMatrix

    n = 50

    def my_r(theta):
        sigma2 = theta[0].exp()
        rho = torch.sigmoid(theta[1])

        I = torch.eye(n)
        J = torch.ones(n, n)

        return sigma2 * (I + rho * J)

    R = SimpleMatrix(num_free_params=2, call=my_r)
    reml = MixedModelREML(R=R)

In this example:

- ``theta[0]`` controls the variance scale through an exponential transformation.
- ``theta[1]`` controls the correlation parameter through a sigmoid transformation.

The resulting residual covariance matrix is

.. math::

    \mathbf{R} = \sigma^2 (\mathbf{I} + \rho \mathbf{J}),

where :math:`\mathbf{J}` is the all-ones matrix.

The supplied function is automatically differentiated by PyTorch, so no derivative code is required.

:class:`~torch_openreml.covariance.SimpleMatrix` can also be used to define design matrices :math:`\mathbf{Z}_i` and random-effect covariance matrices :math:`\mathbf{G}_i`. For example, the following defines a diagonal random-effect covariance matrix with an estimated variance parameter:

.. code-block:: python

    def my_g(theta):
        sigma2 = theta[0].exp()
        return sigma2 * torch.eye(3)

    G = SimpleMatrix(num_free_params=1, call=my_g)

    Z = torch.eye(3)
    reml = MixedModelREML((Z, G))

The gradient can also be supplied manually through the ``manual_grad`` argument. This is useful when analytical derivatives are available and can be computed more efficiently than automatic differentiation, particularly for computationally expensive matrix constructions. For the diagonal covariance matrix above, the analytical gradient is:

.. code-block:: python

    def my_g_grad(theta):
        sigma2 = theta[0].exp()
        return (sigma2 * torch.eye(3)).unsqueeze(0)

    G = SimpleMatrix(num_free_params=1, call=my_g, manual_grad=my_g_grad)

    reml = MixedModelREML((Z, G))

The gradient function returns a tensor of shape ``(num_free_params, *matrix_shape)``, where each slice along the first dimension contains the derivative of the matrix with respect to the corresponding parameter. Here, the shape is ``(1, 3, 3)``.

Via a covariance Matrix builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a covariance structure matches one of the built-in structures, it is generally clearer to construct it using the covariance builders rather than defining it manually.

The covariance builder API provides reusable components, including:

- :class:`~torch_openreml.covariance.ScalarMatrix`
- :class:`~torch_openreml.covariance.IdentityMatrix`
- :class:`~torch_openreml.covariance.DiagonalMatrix`
- :class:`~torch_openreml.covariance.AR1Matrix`
- :class:`~torch_openreml.covariance.CompoundSymmetricMatrix`

These components can be combined using composition operators, including:

- :class:`~torch_openreml.covariance.Sum`
- :class:`~torch_openreml.covariance.BlockDiagonal`
- :class:`~torch_openreml.covariance.KroneckerProduct`
- :class:`~torch_openreml.covariance.Gram`

For example:

.. code-block:: python

    from torch_openreml.covariance import IdentityMatrix, KroneckerProduct
    from torch_openreml.example_data import john_alpha

    gen, rep, block = john_alpha["gen"], john_alpha["rep"], john_alpha["block"]

    reml = MixedModelREML(
        gen=(DummyMatrix(gen), ScalarMatrix(gen)),
        block=(
            DummyMatrix(rep, block),
            KroneckerProduct(IdentityMatrix(rep), ScalarMatrix(block)),
        )
    )

The covariance builders handle the underlying implementation, including:

- Parameter bookkeeping and transformations.
- Construction of the design matrices :math:`\mathbf{Z}_i`, random-effect covariance matrices :math:`\mathbf{G}_i`, and residual covariance matrix :math:`\mathbf{R}`.
- Computation of component derivatives.

.. note::

    Internally, :class:`~torch_openreml.MixedModelREML` calls
    :meth:`~torch_openreml.covariance.Matrix.__call__` to construct each
    matrix component and :meth:`~torch_openreml.covariance.Matrix.grad`
    to compute its Jacobian. The gradient implementation first attempts
    to use a closed-form :meth:`~torch_openreml.covariance.Matrix.manual_grad`
    implementation and falls back to automatic differentiation when one
    is not available.

Parameter names and ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free parameters in :math:`\boldsymbol{\theta}` are named according to their associated term and matrix. Since these are the same parameters used to construct the marginal covariance matrix, their names can be inspected before fitting:

.. code-block:: python

    n, n_env = 50, 5
    environment = [f"env{i % n_env}" for i in range(n)]
    genotype = ["a", "b"] * 25

    reml = MixedModelREML(
        env=(DummyMatrix(environment), CompoundSymmetricMatrix(environment)),
        gen=(DummyMatrix(genotype), ScalarMatrix(genotype))
    )
    reml.V.free_param_names

which returns:

.. code-block:: text

    ['G/env/sigma^2', 'G/env/rho', 'G/gen/sigma^2', 'R/sigma^2']

The parameters are ordered as
:math:`[\boldsymbol{\theta}_{Z}, \boldsymbol{\theta}_{G}, \boldsymbol{\theta}_{R}]`.

Choosing the numerical method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``method`` argument controls how the score vector and average information matrix are computed. By default, ``"auto"`` uses the C-form when applicable and falls back to the :math:`\mathbf{V}`-based marginal method otherwise. The ``"cform"`` and ``"marginal"`` options force the corresponding method:

.. code-block:: python

    reml = MixedModelREML(
        genotype=(DummyMatrix(genotype), ScalarMatrix(genotype)),
        method="marginal",
    )

You can also pass a configured method instance to customise its options, such as the trace approximation used by the marginal method:

.. code-block:: python

    from torch_openreml.methods import Marginal

    reml = MixedModelREML(
        genotype=(DummyMatrix(genotype), ScalarMatrix(genotype)),
        method=Marginal(trace_approx="auto"),
    )

Running the optimiser
---------------------

Once the REML object is constructed, optimisation is performed with
:meth:`~torch_openreml.MixedModelREML.optimize`:

.. jupyter-execute::

    import torch

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import DummyMatrix, ScalarMatrix

    n, p = 10, 2

    group = ["a", "b"] * 5
    effect = torch.tensor([1.0, -1.0]).repeat(n // 2)

    X = torch.randn(n, p)
    y = X @ torch.tensor([0.5, -0.5]) + effect + 0.5 * torch.randn(n)

    reml = MixedModelREML(
        genotype=(DummyMatrix(group), ScalarMatrix(group)),
    )

    names, free_params = reml.optimize(y, X, verbose=1)
    names, free_params

Inputs
~~~~~~

``y``
    Response vector of shape ``(n,)``.

``X``
    Fixed-effects design matrix of shape ``(n, p)``.

``init_values``
    Initial values for the covariance parameters, ordered according to
    ``reml.V.free_param_names``. If omitted, the model's default initial
    values are used. Providing explicit starting values can improve
    convergence, particularly for complex covariance structures.

Returned values
~~~~~~~~~~~~~~~

``names``
    Names of the covariance parameters.

``free_params``
    Final covariance-parameter estimates, in the same order as ``names``.

The update step
~~~~~~~~~~~~~~~
At iteration :math:`t`, the optimiser evaluates the score vector :math:`\mathbf{s}` and the average-information matrix :math:`\mathbf{AI}` at the current parameter estimate :math:`\boldsymbol{\theta}^{(t)}`:

.. math::

    s_k = \frac{1}{2}
    \left(
        \mathbf{y}^{\top} \mathbf{P}
        \frac{\partial \mathbf{V}}{\partial \theta_k}
        \mathbf{P} \mathbf{y}
        -
        \operatorname{tr}\!\left(
            \mathbf{P}
            \frac{\partial \mathbf{V}}{\partial \theta_k}
        \right)
    \right),

.. math::

    \mathrm{AI}_{kj} = \frac{1}{2}\,
    \mathbf{y}^{\top} \mathbf{P}
    \frac{\partial \mathbf{V}}{\partial \theta_k}
    \mathbf{P}
    \frac{\partial \mathbf{V}}{\partial \theta_j}
    \mathbf{P} \mathbf{y},

where

.. math::

    \mathbf{P} = \mathbf{V}^{-1} -
    \mathbf{V}^{-1}\mathbf{X}
    (\mathbf{X}^{\top}\mathbf{V}^{-1}\mathbf{X})^{-1}
    \mathbf{X}^{\top}\mathbf{V}^{-1}.

The :class:`~torch_openreml.methods.CForm` method computes these quantities without explicitly forming :math:`\mathbf{V}`.

The AI direction is obtained by solving :math:`\mathbf{AI}\,\boldsymbol{\delta} = \mathbf{s}` in the least-squares sense. Given a step length :math:`\eta`, the candidate parameter update is

.. math::

    \boldsymbol{\theta}^{c}
    =
    \mathrm{clip}\!\left(
        \boldsymbol{\theta}^{(t)} + \eta \, \boldsymbol{\delta},
        \mathrm{lb},
        \mathrm{ub}
    \right).

The candidate is accepted if it satisfies the Armijo sufficient-increase condition:

.. math::

    \ell_R\!\left(\boldsymbol{\theta}^{c}\right)
    \geq
    \ell_R\!\left(\boldsymbol{\theta}^{(t)}\right)
    +
    c \, \eta \, \mathbf{s}^{\top} \boldsymbol{\delta}
    -
    \epsilon \,
    \max\!\left(
        1,
        \left|\ell_R\!\left(\boldsymbol{\theta}^{(t)}\right)\right|
    \right),

where :math:`c` and :math:`\epsilon` correspond to ``armijo_c`` and ``armijo_slack``, respectively.

If the condition is not satisfied, the step length is reduced according to

.. math::

    \eta \leftarrow \texttt{eta\_shrink} \times \eta,

and the candidate is evaluated again. This backtracking continues for at most ``armijo_trials`` attempts and stops if :math:`\eta` falls below ``eta_min``.

If no candidate is accepted, the optimiser takes a single normalised gradient step as a fallback:

.. math::

    \boldsymbol{\theta}^{(t+1)}
    =
    \mathrm{clip}\!\left(
        \boldsymbol{\theta}^{(t)}
        +
        \texttt{fallback\_eta}\,
        \frac{\mathbf{s}}{\max(\|\mathbf{s}\|, 1)},
        \mathrm{lb},
        \mathrm{ub}
    \right).

This fallback step is recorded as not accepted. After each iteration, the step length is reset to ``eta_max``.

Optimisation controls
~~~~~~~~~~~~~~~~~~~~~

``max_iter`` (default ``200``)
    Maximum number of AI iterations.

``verbose`` (default ``0``)
    If positive, print one line per iteration showing the score norm
    :math:`\|\nabla\|`, the AI step norm :math:`\|\Delta\|`, the step length
    :math:`\eta`, the clipped step norm :math:`\|\Delta^{c}\|`, whether the
    step was accepted, and the restricted log-likelihood.  A legend, the
    convergence status and the total optimisation time are printed at the
    end.

``lb`` and ``ub`` (default ``-torch.inf`` and ``torch.inf``)
    Lower and upper bounds applied to the free parameters. Each can be a
    scalar float or a tensor broadcastable to the free-parameter vector,
    allowing per-parameter bounds. If either is specified as a tensor, both
    must be tensors. Bounds apply to the untransformed parameter values,
    before any covariance parameter transformations. For example,
    ``lb=0`` on a variance parameter constrains its underlying parameter
    to be non-negative, rather than constraining the variance itself to be
    non-negative. Each candidate update is clipped to ``[lb, ub]`` before
    evaluating the log-likelihood.

``eta_max`` (default ``1.0``)
    Initial and maximum step length.  Every iteration starts from
    ``eta_max`` and backtracks from there.

``eta_shrink`` (default ``0.5``)
    Multiplier applied to the step length after each rejected candidate.

``eta_min`` (default ``1e-10``)
    Backtracking stops once the step length falls below ``eta_min``.

``armijo_trials`` (default ``30``)
    Maximum number of backtracking candidates tried per iteration.

``armijo_c`` (default ``1e-4``) and ``armijo_slack`` (default ``1e-7``)
    Constants :math:`c` and :math:`\epsilon` in the acceptance test defined
    in the update step.  The last term is numerical slack; set
    ``armijo_slack=0`` for a strict Armijo test.

``fallback_eta`` (default ``1e-3``)
    Step length of the normalised gradient step taken when no candidate is
    accepted.


Convergence criteria
~~~~~~~~~~~~~~~~~~~~

Two checks are available.

Parameter update
    The largest absolute component of the clipped step must satisfy

    .. math::

        \max_i |\Delta_i^{c}| < \texttt{tol\_update}.

Log-likelihood change
    The restricted log-likelihood change must satisfy

    .. math::

        \left| \ell_R^{(t)} - \ell_R^{(t-1)} \right| < \texttt{tol\_loglik}.

The defaults are ``tol_update = 1e-4`` and ``tol_loglik = 1e-4``.  Both
enabled checks must pass simultaneously, and at least two iterations are
required before convergence can be declared.  Either check can be disabled
with ``check_update=False`` or ``check_loglik=False``.
:meth:`~torch_openreml.MixedModelREML.is_converged` reports whether the last
recorded iteration met the criteria.

Optimisation history
--------------------

Every iteration of :meth:`~torch_openreml.MixedModelREML.optimize` is
appended to ``reml.history``, a dict of lists that is reset when a new
optimisation starts; it is ``None`` until ``optimize`` has run.  Each list
holds one entry per iteration, where ``k`` is the number of free parameters:

``free_params``
    Covariance parameters after the step, shape ``(k,)``.
``loglik``
    Restricted log-likelihood at the new parameters, a scalar tensor.
``score``
    Score vector used to compute the step, shape ``(k,)``.
``ai``
    Average information matrix, shape ``(k, k)``.
``delta``
    AI direction, the least-squares solution of
    :math:`\mathbf{AI}\,\boldsymbol{\delta} = \mathbf{s}`, shape ``(k,)``.
``update``
    Clipped step actually taken, shape ``(k,)``.
``eta``
    Step length used, which is below ``eta_max`` when backtracking shortened
    it.
``accepted``
    ``True`` for an accepted Newton step and ``False`` for a gradient
    fallback step.
``method``
    ``"cform"`` or ``"marginal"`` for that iteration.

.. jupyter-execute::

    history = reml.history
    history["loglik"]

Stacking a list gives the trajectory, for example the log-likelihood:

.. jupyter-execute::

    torch.stack(history["loglik"])

or the covariance parameters:

.. jupyter-execute::

    torch.stack(history["free_params"])

Post-estimation methods
-----------------------

After optimisation, the fitted REML object provides several utilities for extracting estimates and predictions.

Retrieving parameter estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`get_free_params(select="last")`
    Return ``(names, free_params)``.  ``select="best"`` selects the iterate
    with the largest recorded log-likelihood, which is useful when
    optimisation overshoots or oscillates near convergence.

.. jupyter-execute::

    names, theta_hat = reml.get_free_params("last")
    names, theta_hat

`build_params(free_params=None, include_fixed=True, trans=True, out_format="tensor")`
    Build the covariance parameters from the free parameters. By default,
    ``trans=True`` applies each parameter's transformation, returning values
    on the model scale.
    ``trans=False`` returns the untransformed values. Set
    ``out_format="dict"`` to return a name-keyed mapping, or
    ``include_fixed=False`` to exclude fixed parameters.

.. jupyter-execute::

    params = reml.build_params("last", out_format="dict")
    params

`tree(free_params=None)`
    Evaluate the covariance structure and return the matrix at each node,
    keyed by its path in the operand tree. The root path ``"/"`` contains
    the marginal covariance :math:`\mathbf{V}`. Other paths identify the
    model components: ``"Z"`` contains the combined random-effect design,
    with individual terms under ``"Z/<term>"``; ``"G"`` contains the
    block-diagonal random-effect covariance, with individual terms under
    ``"G/<term>"``; and ``"R"`` contains the residual covariance.

    Nested builders contribute additional paths using their operand names.
    For example, a term constructed with
    :class:`~torch_openreml.covariance.KroneckerProduct` adds paths such as
    ``"G/block/op_0"`` and ``"G/block/op_1"``. Each entry contains the
    evaluated matrix at that level, allowing the dimensions of individual
    components and how they combine to be inspected. If the model has no
    random-effect terms, the tree contains only ``{"/": V}``.

.. jupyter-execute::

    tree = reml.tree("last")
    tree

Fixed effects
~~~~~~~~~~~~~

The BLUE of the fixed effects is

.. math::

    \widehat{\boldsymbol{\beta}}
    = \left(\mathbf{X}^\top \mathbf{V}^{-1} \mathbf{X}\right)^{-1}
      \mathbf{X}^\top \mathbf{V}^{-1} \mathbf{y}.

`blue(y=None, x=None, free_params=None)`
    Fixed-effect estimates of shape ``(p,)``.

.. jupyter-execute::

    beta_hat = reml.blue(y, X, theta_hat)
    beta_hat

`cov_blue(x=None, free_params=None)`
    Covariance matrix of the BLUE, shape ``(p, p)``.

.. jupyter-execute::

    cov_beta = reml.cov_blue(X, theta_hat)
    cov_beta

`var_blue(x=None, free_params=None)`
    Sampling variances, the diagonal of ``cov_blue``.

.. jupyter-execute::

    se_beta = torch.sqrt(reml.var_blue(X, theta_hat))
    se_beta

Random effects
~~~~~~~~~~~~~~

The BLUP of the random effects is

.. math::

    \widehat{\mathbf{b}}
    = \mathbf{G}\mathbf{Z}^\top\mathbf{V}^{-1}
      \left(\mathbf{y} - \mathbf{X}\widehat{\boldsymbol{\beta}}\right),

with covariance
:math:`\operatorname{Var}(\widehat{\mathbf{b}}) = \mathbf{G}\mathbf{Z}^\top\mathbf{P}\mathbf{Z}\mathbf{G}`.

`blup(y=None, x=None, free_params=None)`
    Random-effect predictions of shape ``(q,)``.

.. jupyter-execute::

    b_hat = reml.blup()
    b_hat

`cov_blup(x=None, free_params=None)`
    Covariance matrix of the BLUPs, shape ``(q, q)``.

.. jupyter-execute::

    b_cov = reml.cov_blup()
    b_cov

`var_blup(x=None, free_params=None)`
    BLUP sampling variances, the diagonal of ``cov_blup``.

.. jupyter-execute::

    b_var = reml.var_blup()
    b_var

`blup_per_term(y=None, x=None, free_params=None)`
    BLUPs split by random-effect term, as a dict keyed by term name.

.. jupyter-execute::

    b_by_term = reml.blup_per_term()
    b_by_term

`cov_blup_per_term(x=None, free_params=None)`
    Diagonal blocks of ``cov_blup``, one per term, as a dict.

.. jupyter-execute::

    b_cov_by_term = reml.cov_blup_per_term()
    b_cov_by_term

`var_blup_per_term(x=None, free_params=None)`
    Term-wise sampling variances, as a dict.

.. jupyter-execute::

    b_var_by_term = reml.var_blup_per_term()
    b_var_by_term


Predictions
~~~~~~~~~~~

`marginal_predict(y=None, x=None, free_params=None)`
    Marginal fitted values :math:`\mathbf{X}\widehat{\boldsymbol{\beta}}`.

.. jupyter-execute::

    y_marginal = reml.marginal_predict()
    y_marginal

`predict(y=None, x=None, free_params=None)`
    Conditional fitted values
    :math:`\mathbf{X}\widehat{\boldsymbol{\beta}} + \mathbf{Z}\widehat{\mathbf{b}}`.
    For a model with no random-effect terms this reduces to
    ``marginal_predict``.

.. jupyter-execute::

    y_conditional = reml.predict()
    y_conditional

`predict_new(x, z=None, free_params=None)`
    Predict responses for new observations using pre-built design matrices.
    The ``z`` argument accepts a full ``(n_new, q)`` tensor, a list of
    design blocks (one per random-effect term), or a dictionary keyed by
    term name. Terms omitted from a dictionary contribute zero. If
    ``z=None``, predictions use only the fixed effects, giving the
    population-level predictions.

    The random-effect predictions use BLUPs computed from the stored
    training data, so ``optimize`` must have been called. Construct the
    design matrices using the same column order as the training data.
    ``DummyMatrix.transform`` encodes new factor levels against the
    training columns. A single new observation should be supplied as a
    one-row batch.

.. jupyter-execute::

    y_new = reml.predict_new(
        X[:2],
        {"genotype": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
    )
    y_new

.. jupyter-execute::

    z_new = DummyMatrix(group).transform(["a", "b"])

    y_new = reml.predict_new(
        X[:2],
        z_new,
    )
    y_new



Residuals
~~~~~~~~~

`marginal_residual(y=None, x=None, free_params=None)`
    Marginal residuals
    :math:`\mathbf{y} - \mathbf{X}\widehat{\boldsymbol{\beta}}`.

.. jupyter-execute::

    e_marginal = reml.marginal_residual(y, X, theta_hat)
    e_marginal

`residual(y=None, x=None, free_params=None)`
    Conditional residuals
    :math:`\mathbf{y} - \mathbf{X}\widehat{\boldsymbol{\beta}} - \mathbf{Z}\widehat{\mathbf{b}}`.
    For a model with no random-effect terms this reduces to
    ``marginal_residual``.

.. jupyter-execute::

    e_conditional = reml.residual(y, X, theta_hat)
    e_conditional

Covariance blocks
~~~~~~~~~~~~~~~~~

`c_inverse_blocks(x=None, free_params=None)`
    Blocks ``(c11, c12, c21, c22)`` of the inverse coefficient matrix

    .. math::

        \mathbf{C} =
        \begin{pmatrix}
            \mathbf{X}^\top\mathbf{R}^{-1}\mathbf{X} &
            \mathbf{X}^\top\mathbf{R}^{-1}\mathbf{Z} \\
            \mathbf{Z}^\top\mathbf{R}^{-1}\mathbf{X} &
            \mathbf{Z}^\top\mathbf{R}^{-1}\mathbf{Z} + \mathbf{G}^{-1}
        \end{pmatrix},

    with shapes ``(p, p)``, ``(p, q)``, ``(q, p)`` and ``(q, q)``.
    :math:`\mathbf{C}_{11}` equals ``cov_blue`` and :math:`\mathbf{C}_{22}`
    is the prediction-error covariance of the random effects.  The model
    must have random-effect terms.

.. jupyter-execute::

    c11, c12, c21, c22 = reml.c_inverse_blocks(X, theta_hat)
    c11, c12, c21, c22

Log-likelihood and convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`loglik(y=None, x=None, free_params=None)`
    Evaluate the log-likelihood and return ``(loglik, method)``, where
    ``method`` indicates the evaluation method: ``"cform"`` or
    ``"marginal"``. With
    ``method="auto"``, the evaluation silently falls back to the marginal
    method when the C-form is not applicable.

.. jupyter-execute::

    loglik_value, loglik_method = reml.loglik(y, X, theta_hat)
    loglik_value, loglik_method

`is_converged()`
    Whether the last recorded iteration met the enabled convergence checks.

.. jupyter-execute::

    reml.is_converged()


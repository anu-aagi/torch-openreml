.. _vig_factor_analysis:

Factor Analysis of a Multi-Environment Trial
============================================
In a multi-environment trial (MET), the same genotypes are evaluated across several environments, resulting in correlated genotype-by-environment effects. An unstructured covariance matrix for :math:`m` environments requires :math:`m(m+1)/2` parameters (21 for :math:`m = 6`). A factor-analytic (FA) structure reduces the number of parameters by representing the covariance through a small number of latent factors and environment-specific uniquenesses.

This vignette fits an FA(2) model to a simulated MET dataset, examining parameter recovery against the known generating values. It also demonstrates how the factor-analytic covariance is constructed from covariance classes and how its parameters are named. The analysis is then repeated on an unbalanced version of the trial, where each genotype is grown in only a subset of the environments.

The model
---------

For genotype :math:`i = 1, \ldots, n`, environment :math:`j = 1, \ldots, m`
and replicate :math:`r = 1, \ldots, R`, write

.. math::

    y_{ijr}
    =
    \mu_j
    +
    g_{ij}
    +
    \varepsilon_{ijr},
    \qquad
    \mathbf{g}_i
    =
    (g_{i1}, \ldots, g_{im})^\top
    \sim
    \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}),
    \qquad
    \varepsilon_{ijr} \sim \mathcal{N}(0, \sigma^2).

The genotype-by-environment effects follow a factor-analytic covariance
structure with :math:`k` latent factors,

.. math::

    \boldsymbol{\Sigma}
    =
    \boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top
    +
    \boldsymbol{\Psi},
    \qquad
    \boldsymbol{\Psi} = \operatorname{diag}(\psi_1, \ldots, \psi_m),

where :math:`\boldsymbol{\Lambda}` is the :math:`m \times k` factor-loading
matrix and :math:`\psi_j` is the uniqueness of environment :math:`j`.
Stacking the observations gives the linear mixed model

.. math::

    \mathbf{y}
    =
    \mathbf{X}\boldsymbol{\beta}
    +
    \mathbf{Z}\mathbf{g}
    +
    \boldsymbol{\varepsilon},
    \qquad
    \mathbf{g} \sim \mathcal{N}(\mathbf{0}, \mathbf{G}),
    \qquad
    \mathbf{G} = \mathbf{I}_n \otimes \boldsymbol{\Sigma},
    \qquad
    \mathbf{R} = \sigma^2 \mathbf{I}_N,

where

- :math:`\mathbf{X}` has one column per environment, with
  :math:`\boldsymbol{\beta}` containing the environment means :math:`\mu_j`,
- :math:`\mathbf{Z}` maps each observation to its genotype-by-environment
  cell,
- :math:`\mathbf{g}` contains the cell effects,
- :math:`\mathbf{I}_n \otimes \boldsymbol{\Sigma}` specifies that the
  :math:`m`-vector of effects for each genotype has covariance
  :math:`\boldsymbol{\Sigma}`, and that genotypes are independent.

See :ref:`vig_mixed_model_reml` for the general formulation of the mixed
model and the REML estimator.

Identification
~~~~~~~~~~~~~~

The decomposition :math:`\boldsymbol{\Sigma} = \boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top + \boldsymbol{\Psi}` is not unique, since orthogonal rotations of :math:`\boldsymbol{\Lambda}` leave :math:`\boldsymbol{\Sigma}` unchanged. To identify the factor loadings, we impose a lower-triangular constraint, setting :math:`{\Lambda}_{ij} = 0` whenever :math:`j > i`.

This constraint fixes the rotational indeterminacy, but the sign of each factor remains arbitrary. Consequently, estimated loadings may differ in sign from their generating values. The trial below illustrates these differences.

The data
--------

The trial is shipped with the package as the dataset
``torch_openreml.example_data.fa_met``.  It records the yield of 100
genotypes in six environments, with two replicates per genotype-by-environment
cell:

.. jupyter-execute::

    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=4, sci_mode=False)

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import (
        DiagonalMatrix,
        DummyMatrix,
        Gram,
        IdentityMatrix,
        KroneckerProduct,
        LowerTriangularMatrix,
        ScalarMatrix,
        Sum,
    )
    from torch_openreml.example_data import fa_met, fa_met_params

    fa_met.head()

The dataset contains 1,200 rows, covering 100 genotypes across 6 environments, with 2 replicates per genotype-environment combination. The columns are ``genotype``, ``environment``, ``rep`` (replicate 1 or 2), ``yield`` (observed yield), and ``genetic`` (latent genotype-by-environment effect).

The data were simulated from the FA(2) structure described above.  The
generating values are kept next to the trial in ``fa_met_params.json`` and
exposed as ``torch_openreml.example_data.fa_met_params``:

.. jupyter-execute::

    means = torch.tensor(fa_met_params["means"])
    loadings = torch.tensor(fa_met_params["loadings"])
    uniqueness = torch.tensor(fa_met_params["uniqueness"])
    sigma2 = fa_met_params["sigma2"]

    Sigma_g = loadings @ loadings.T + torch.diag(uniqueness)
    print(f"{means=}")
    print(f"{loadings=}")
    print(f"{sigma2=}")
    print(f"{Sigma_g=}")

Set up the design matrix :math:`\mathbf{X}` and response vector :math:`\mathbf{y}`:

.. jupyter-execute::

    genotype = fa_met["genotype"]
    environment = fa_met["environment"]

    y = torch.tensor(fa_met["yield"].values)
    X = DummyMatrix(environment)()

Specifying the covariance
-------------------------

The covariance of the genotype-by-environment cell effects is a sum of a rank-:math:`k` term and a
diagonal term.  :class:`~torch_openreml.covariance.Gram` with
``gram_type="xxt"`` turns the free loading matrix into
:math:`\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top`, and
:class:`~torch_openreml.covariance.LowerTriangularMatrix` provides the
free entries of :math:`\boldsymbol{\Lambda}` with the upper triangle fixed at
zero:

.. jupyter-execute::

    Z = DummyMatrix(genotype, environment)

    k = 2

    G = KroneckerProduct(
        genotype=IdentityMatrix(genotype),
        environment=Sum(
            factors=Gram(
                loadings=LowerTriangularMatrix(environment, k),
                gram_type="xxt",
            ),
            uniqueness=DiagonalMatrix(environment),
        ),
    )

    reml = MixedModelREML(g_by_e=(Z, G), R=ScalarMatrix(len(y)))

- :class:`~torch_openreml.covariance.DummyMatrix` builds the
  :math:`1200 \times 600` incidence matrix with one column per
  genotype-by-environment cell.
- :class:`~torch_openreml.covariance.LowerTriangularMatrix` builds the :math:`6 \times 2` lower triangular matrix with
  11 free entries :math:`\lambda_{00}, \lambda_{10}, \lambda_{11}, \ldots, \lambda_{50},
  \lambda_{51}`, and a structural zero in the upper
  triangle.
- :class:`~torch_openreml.covariance.DiagonalMatrix` supplies the six
  uniquenesses :math:`\psi_j`.
- :class:`~torch_openreml.covariance.KroneckerProduct` forms
  :math:`\mathbf{I}_{100} \otimes \boldsymbol{\Sigma}`.  The genotype factor
  is an :class:`~torch_openreml.covariance.IdentityMatrix`, so it has no free
  parameters. All of the free parameters live in the environment factor.
- :class:`~torch_openreml.covariance.ScalarMatrix` supplies
  :math:`\mathbf{R} = \sigma^2 \mathbf{I}`.  It could be omitted, since it is
  the default residual covariance, but it is written out here for clarity.

Parameter names and ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free parameters are ordered
:math:`[\boldsymbol{\theta}_Z, \boldsymbol{\theta}_G,
\boldsymbol{\theta}_R]`. Here :math:`\boldsymbol{\theta}_Z` is empty, so the
first parameters belong to the random-effect covariance and the last one to
the residual.

.. jupyter-execute::

    names = reml.V.free_param_names
    names

The names follow the paths of the covariance tree:

- ``G/environment/factors/loadings/L_i_j`` is the loading of environment
  :math:`i` on factor :math:`j`.
- ``G/environment/uniqueness/sigma^2_j`` is the uniqueness
  :math:`\psi_j`.
- ``R/sigma^2`` is the residual variance.

The loadings are unconstrained.  The uniquenesses and the residual variance
are parameterised on the log-variance scale, :math:`\psi_j = \exp(2\theta)`
and :math:`\sigma^2 = \exp(2\theta)`, so a free value of zero corresponds to a
variance of one.  In total the model has :math:`11 + 6 + 1 = 18` free
parameters.

Fitting the model
-----------------

The model is fitted with the default ``"auto"`` method, which selects the
C-form for this covariance structure:

.. jupyter-execute::

    names, theta_hat = reml.optimize(y, X, verbose=1)

The columns of the iteration log are the norm of the score vector
:math:`\lVert\nabla\rVert`, the length of the average-information step
:math:`\lVert\Delta\rVert`, the step length :math:`\eta` after any
backtracking, the length of the clipped step actually taken
:math:`\lVert\Delta^{c}\rVert`, whether the step was accepted, and the
restricted log-likelihood.  The fit converges in 17 iterations, in about a
second and a half.

The fitted covariance-parameter vector is on the transformed scale: the first
11 entries are loadings, the next six are half-log uniquenesses, and the last
is the half-log residual variance.

.. jupyter-execute::

    theta_hat

For interpretation, :meth:`~torch_openreml.MixedModelREML.build_params`
applies the transforms and returns the parameters on the model scale.  It can
return them as a name-keyed dictionary, which is convenient for picking out
individual components:

.. jupyter-execute::

    params = reml.build_params(out_format="dict")
    params

The residual variance `R/sigma^2` is estimated at :math:`\hat{\sigma}^2 \approx 0.30`,
almost exactly the generating value of 0.3.

The covariance tree is often the most direct way to get at the matrices
themselves.  The fitted :math:`\boldsymbol{\Sigma}` is at
``"G/environment"``, the loadings at
``"G/environment/factors/loadings"``, and the uniqueness matrix at
``"G/environment/uniqueness"``:

.. jupyter-execute::

    tree = reml.tree()

    Sigma_hat = tree["G/environment"]
    Lambda_hat = tree["G/environment/factors/loadings"]
    Psi_hat = tree["G/environment/uniqueness"]

.. jupyter-execute::

    Psi_hat

The estimated uniquenesses are in the right range but do not exactly match the generating values :math:`(0.5, 0.4, 0.5, 0.6, 0.4, 0.5)`. This reflects sampling variability in estimating the genetic covariance structure from 100 genotypes, with the uniquenesses estimated jointly with the factor loadings. Such deviations are expected even when the model is correctly specified and successfully fitted.


The fitted loadings are a representative of the rotation orbit, and the sign
of each column is arbitrary.  Before comparing with the generating matrix, it
is therefore natural to flip each column so that its leading entry is
positive.  The flip does not change :math:`\boldsymbol{\Sigma}`:

.. jupyter-execute::

    sign = torch.sign(torch.diagonal(Lambda_hat))
    Lambda_aligned = Lambda_hat * sign

    Lambda_aligned

.. jupyter-execute::

    loadings

The first factor captures the same broad pattern as the generator, with loadings that decline from early to late environments. The second factor is similarly dominated by the later environments, although its smaller loadings differ from the generating values. This reflects the trade-off between factor loadings and uniquenesses in the likelihood, making individual parameter estimates less stable than the covariance structure they jointly determine. Therefore, the appropriate quantity to compare with the generating values is :math:`\boldsymbol{\Sigma}`, rather than the individual loadings.

.. jupyter-execute::

    (Sigma_hat - Sigma_g).abs().max()

.. jupyter-execute::

    torch.corrcoef(torch.stack([Sigma_hat.flatten(), Sigma_g.flatten()]))[0, 1]

The fitted covariance has a correlation of approximately 0.95 with the generating covariance, with a maximum element-wise absolute error of approximately 0.23.

Recovery of the genotype-by-environment effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The random-effect predictions can be obtained from
:meth:`~torch_openreml.MixedModelREML.blup_per_term`, which returns predictions
keyed by term name. Here, there is a single term, `"genotype"`. Comparing
these predictions with the generating effects in the `genetic` column
assesses the fit at the individual genotype-by-environment level:

.. jupyter-execute::

    genetic = torch.tensor(fa_met.drop_duplicates(["genotype", "environment"])["genetic"].values)

    blup = reml.blup_per_term()["g_by_e"]

    torch.corrcoef(torch.stack([genetic, blup]))[0, 1]

.. jupyter-execute::

    (blup - genetic).pow(2).mean().sqrt()

The correlation is high, but the root mean squared error remains nonzero because the BLUPs are shrunk towards zero relative to the generating effects and are affected by estimation error in the covariance parameters.

Recovery of the fixed effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fixed effects are the environment means.  Their estimates and standard
errors are:

.. jupyter-execute::

    beta_hat = reml.blue()
    se_beta = torch.sqrt(reml.var_blue())

    beta_hat, se_beta

Compared with the generating means :math:`(5.0, 5.2, 4.8, 5.1, 5.3, 4.9)`,
all estimates are within approximately 2 standard errors.

Choosing the number of factors
------------------------------

In practice, :math:`k` is unknown. A simple approach is to fit the model with :math:`k = 1, 2, 3` and compare the restricted log-likelihoods. Since the fixed-effects structure is identical across fits, the AIC is :math:`-2\ell_R + 2p`, where :math:`p` is the number of covariance parameters.

.. jupyter-execute::

    def fit_fa(k):
        G = KroneckerProduct(
            genotype=IdentityMatrix(genotype),
            environment=Sum(
                factors=Gram(
                    LowerTriangularMatrix(environment, k),
                    gram_type="xxt",
                ),
                uniqueness=DiagonalMatrix(environment),
            ),
        )
        model = MixedModelREML(g_by_e=(Z, G), R=ScalarMatrix(len(y)))
        names_k, _ = model.optimize(y, X, armijo_trials=5, fallback_eta=0.1)
        loglik_k = model.loglik()[0].item()
        p_k = len(names_k)
        return p_k, loglik_k, -2 * loglik_k + 2 * p_k

    for k_k in (1, 2, 3):
        p_k, loglik_k, aic_k = fit_fa(k_k)
        print(f"k = {k_k}: p = {p_k}, loglik = {loglik_k:.3f}, AIC = {aic_k:.3f}")

AIC prefers two factors here, which is consistent with the generating model.

An unbalanced design
--------------------

The trial above is balanced by construction, with every genotype appearing in every environment and two replicates per cell. Real multi-environment trials rarely have such a complete structure. Genotypes enter and leave testing at different sites, resulting in a ragged pattern of observed cells and different numbers of genotypes across environments. In this second part, cells are removed from the simulated trial while retaining both replicates in every remaining cell. Each genotype is grown in three of the six environments, selected at random using a fixed seed to ensure reproducibility.

Removing cells changes only the data. The model, parameterisation and number of free parameters remain the same as in the first part. The difference lies in how much information the data provide about those parameters.

The unbalanced trial
~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    env_levels = sorted(fa_met["environment"].unique())
    gen_levels = sorted(fa_met["genotype"].unique())

    generator = torch.Generator().manual_seed(20261120)
    missing = {}
    for genotype in gen_levels:
        picks = torch.randperm(len(env_levels), generator=generator)[:3]
        missing[genotype] = {env_levels[i] for i in picks.tolist()}

    keep = [
        environment not in missing[genotype]
        for genotype, environment in zip(fa_met["genotype"], fa_met["environment"])
    ]
    unbalanced = fa_met[keep].reset_index(drop=True)

    unbalanced.head()

Each genotype retains three randomly selected environments, with both
replicates in each. `missing` records the
environments omitted for each genotype.

The resulting subset contains 600 observations across 300 genotype–environment
cells, with between 42 and 55 genotypes per environment, down from 100:

.. jupyter-execute::

    unbalanced.drop_duplicates(["genotype", "environment"]).groupby("environment").size()

The fixed and random effects are declared exactly as before:

.. jupyter-execute::

    y_u = torch.tensor(unbalanced["yield"].values)
    X_u = DummyMatrix(unbalanced["environment"], levels=[env_levels])()
    Z_u = DummyMatrix(unbalanced["genotype"], unbalanced["environment"])

The covariance is built from the same classes as in the first part:

.. jupyter-execute::

    G_u = KroneckerProduct(
        genotype=IdentityMatrix(gen_levels),
        environment=Sum(
            factors=Gram(LowerTriangularMatrix(env_levels, 2), gram_type="xxt"),
            uniqueness=DiagonalMatrix(env_levels),
        ),
    )

    model_u = MixedModelREML(genotype=(Z_u, G_u), R=ScalarMatrix(len(y_u)))

Fitting the unbalanced trial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    names_u, theta_u = model_u.optimize(y_u, X_u, verbose=1)

The residual variance, against a generating value of 0.3, is

.. jupyter-execute::

    model_u.build_params(out_format="dict")["R/sigma^2"]

Recovery of the covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    tree_u = model_u.tree()
    Sigma_u = tree_u["G/environment"]
    Psi_u = tree_u["G/environment/uniqueness"]

.. jupyter-execute::

    Psi_u

The uniquenesses are noisier. The fifth is estimated to be zero, or rather, so close to zero that it prints as zero. Since uniquenesses are exponentiated, no finite parameter estimate can reach the boundary. However, the optimizer pushes the unconstrained parameter far into the negative region, causing :math:`e^{2\theta}` to underflow. This reflects a flat direction in the likelihood: the two factors can explain almost all of environment 5's variance.

.. jupyter-execute::

    torch.corrcoef(torch.stack([Sigma_u.flatten(), Sigma_g.flatten()]))[0, 1]

.. jupyter-execute::

    (Sigma_u - Sigma_g).abs().max()


The correlation with the generating covariance drops slightly, and the largest entry-wise error grows.

Recovery of the genotype-by-environment effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The practical reason for fitting the covariance is to predict the cells that
were not observed.  The BLUPs of the 600 cell effects are

.. jupyter-execute::

    blup_u = model_u.blup_per_term()["genotype"]
    seen = (Z_u() != 0).any(dim=0)

``seen`` marks the 300 cells that have observations.

.. jupyter-execute::

    def corr(x, y):
        return torch.corrcoef(torch.stack([x, y]))[0, 1]

    def rmse(x, y):
        return (x - y).pow(2).mean().sqrt()

Against the generating effects, the cells with observations are predicted as

.. jupyter-execute::

    corr(genetic[seen], blup_u[seen])

.. jupyter-execute::

    rmse(genetic[seen], blup_u[seen])

and the cells without observations as

.. jupyter-execute::

    corr(genetic[~seen], blup_u[~seen])

.. jupyter-execute::

    rmse(genetic[~seen], blup_u[~seen])

.. jupyter-execute::

    blup_u[~seen].std(), genetic[~seen].std()

The observed cells are predicted as well as before.  The unobserved cells are
not simply left at zero.  Their BLUPs are given by

.. math::

   \text{BLUP} = \mathbf{G}\mathbf{Z}^\top\mathbf{V}^{-1}\mathbf{r},

and the rows of :math:`\mathbf{G}` corresponding to unobserved cells are
correlated with those of the same genotype's observed cells.  Through
:math:`\boldsymbol{\Sigma}`, the model therefore borrows information from the
environments where the genotype *was* grown.  The resulting predictions have
a correlation of 0.50 with the generating effects and an RMSE of 0.89.

Two factors limit their accuracy.  First, the covariance is estimated, so the
prediction weights are estimated rather than known.  Second, the observed
effects are themselves shrunk, providing a shrunken basis for predicting the
missing cells.  The standard deviation of the unobserved BLUPs is 0.60,
compared with 1.00 for the generating values.  This shrinkage pulls
predictions towards the mean, under-predicting good genotypes and
over-predicting poor ones.

Choosing the number of factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The comparison of the previous section has to be repeated on the reduced
data:

.. jupyter-execute::

    def fit_fa_u(k):
        G = KroneckerProduct(
            genotype=IdentityMatrix(gen_levels),
            environment=Sum(
                factors=Gram(
                    LowerTriangularMatrix(env_levels, k),
                    gram_type="xxt",
                ),
                uniqueness=DiagonalMatrix(env_levels),
            ),
        )
        model = MixedModelREML(genotype=(Z_u, G), R=ScalarMatrix(len(y_u)))
        names_k, _ = model.optimize(y_u, X_u, armijo_trials=5, fallback_eta=0.1)
        loglik_k = model.loglik()[0].item()
        p_k = len(names_k)
        return p_k, loglik_k, -2 * loglik_k + 2 * p_k

    for k_k in (1, 2, 3):
        p_k, loglik_k, aic_k = fit_fa_u(k_k)
        print(f"k = {k_k}: p = {p_k}, loglik = {loglik_k:.3f}, AIC = {aic_k:.3f}")

AIC now favours one factor, 568.0 compared with 570.6 for two factors, reversing
the result from the balanced analysis.  Although the covariance is still
recovered, there is little evidence for the second factor.  The :math:`k=2`
fit also had to push one uniqueness to zero to accommodate it.

A difference of 2.6 provides only weak evidence, and this is a simulated
trial whose true number of factors is two.  The conclusion is not that the
model has one factor, but that 300 observed cells provide too little
information to distinguish between the two.

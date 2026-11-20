.. _vig_factor_analysis:

Factor Analysis of a Multi-Environment Trial
============================================
In a multi-environment trial (MET), the same genotypes are evaluated across several environments, resulting in correlated genotype-by-environment effects. An unstructured covariance matrix for :math:`m` environments requires :math:`m(m+1)/2` parameters, which is 21 for :math:`m = 6`. A factor-analytic (FA) structure reduces the number of parameters by representing the covariance through a small number of latent factors and environment-specific uniquenesses.

This vignette fits an FA(2) model to a simulated MET dataset included with the package. Using the known generating values, it examines which parameters the estimator can and cannot recover. It also demonstrates how the factor-analytic covariance is constructed from the covariance classes and how its parameters are named. The second part repeats the analysis on an unbalanced version of the same trial, in which each genotype is grown in only some of the environments.

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

The main advantage of the FA structure is its reduced parameter count. For an FA(:math:`k`) model, the loading matrix :math:`\boldsymbol{\Lambda}` has :math:`mk` entries, of which :math:`k(k-1)/2` are fixed at zero under the lower-triangular convention described below. Including the :math:`m` environment-specific uniquenesses, the total number of covariance parameters is

.. math::

    mk - \frac{k(k-1)}{2} + m.

For six environments, an FA(1) model requires :math:`6 + 6 = 12` parameters, compared with 21 for an unstructured covariance. An FA(2) model requires :math:`12 - 1 + 6 = 17` parameters, providing a more flexible structure while still reducing the parameter count. An FA(3) model requires 21 parameters, the same as an unstructured covariance.

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

The columns are ``genotype`` and ``environment``, both with zero-padded labels, ``rep`` indicating replicate 1 or 2, ``yield``, and ``genetic``, the latent genotype-by-environment effect used to generate the yields. The labels are zero-padded so that lexicographic sorting, as performed by :class:`~torch_openreml.covariance.DummyMatrix` by default, preserves their numeric order. Each genotype-environment cell appears twice, once per replicate, with the same ``genetic`` value in both rows. The incidence matrix constructed below maps both replicate observations to the same cell effect, reusing the 600 cell effects across the two replicates.

The data were simulated from the FA(2) structure described above.  The
generating values are kept next to the trial in ``fa_met_params.json`` and
exposed as ``torch_openreml.example_data.fa_met_params``:

.. jupyter-execute::

    means = torch.tensor(fa_met_params["means"])
    loadings = torch.tensor(fa_met_params["loadings"])
    uniqueness = torch.tensor(fa_met_params["uniqueness"])
    sigma2 = fa_met_params["sigma2"]

    sigma_g = loadings @ loadings.T + torch.diag(uniqueness)
    means, sigma2, sigma_g.round(decimals=4)

The first factor is a general gradient from the early to the late
environments, while the second factor distinguishes the later environments.
The loadings are lower triangular, so the generating covariance is already in
the form that the model will estimate.

The BLUPs can be compared with the values they are meant to estimate by
reducing the ``genetic`` column to one row per cell, in the order in which the
cells appear in the file (genotype by genotype, environments ascending), which
is also the order of the columns of the incidence matrix built below:

.. jupyter-execute::

    cells = fa_met.drop_duplicates(["genotype", "environment"])
    genetic = torch.tensor(cells["genetic"].to_numpy())

    genotype_labels = fa_met["genotype"]
    environment_labels = fa_met["environment"]

    y = torch.tensor(fa_met["yield"].to_numpy())
    X = DummyMatrix(environment_labels)()

    y.shape, X.shape, len(genotype_labels)

Specifying the covariance
-------------------------

The covariance of the cell effects is a sum of a rank-:math:`k` term and a
diagonal term.  :class:`~torch_openreml.covariance.Gram` with
``gram_type="xxt"`` turns the free loading matrix into
:math:`\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top`, and
:class:`~torch_openreml.covariance.LowerTriangularMatrix` provides the
free entries of :math:`\boldsymbol{\Lambda}` with the upper triangle fixed at
zero:

.. jupyter-execute::

    Z = DummyMatrix(genotype_labels, environment_labels)

    k = 2

    fa = Sum(
        factors=Gram(
            LowerTriangularMatrix(environment_labels, k),
            gram_type="xxt",
        ),
        uniqueness=DiagonalMatrix(environment_labels),
    )

    G = KroneckerProduct(
        genotype=IdentityMatrix(genotype_labels),
        environment=fa,
    )

    reml = MixedModelREML(genotype=(Z, G), R=ScalarMatrix(len(y)))

- :class:`~torch_openreml.covariance.DummyMatrix` builds the
  :math:`1200 \times 600` incidence matrix with one column per
  genotype-by-environment cell.  Because the labels sort genotype first, the
  columns are ordered genotype by genotype.
- :class:`~torch_openreml.covariance.LowerTriangularMatrix` with
  :math:`n = 6` and :math:`m = 2` has the free entries
  :math:`\lambda_{00}, \lambda_{10}, \lambda_{11}, \ldots, \lambda_{50},
  \lambda_{51}`, which is 11 parameters, and a structural zero in the upper
  triangle.
- :class:`~torch_openreml.covariance.DiagonalMatrix` supplies the six
  uniquenesses :math:`\psi_j`.
- :class:`~torch_openreml.covariance.KroneckerProduct` forms
  :math:`\mathbf{I}_{100} \otimes \boldsymbol{\Sigma}`.  The genotype factor
  is an :class:`~torch_openreml.covariance.IdentityMatrix`, so it has no free
  parameters; all of the free parameters live in the environment factor.
- :class:`~torch_openreml.covariance.ScalarMatrix` supplies
  :math:`\mathbf{R} = \sigma^2 \mathbf{I}`.  It could be omitted, since it is
  the default residual covariance, but it is written out here for clarity.

Parameter names and ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free parameters are ordered
:math:`[\boldsymbol{\theta}_Z, \boldsymbol{\theta}_G,
\boldsymbol{\theta}_R]`; here :math:`\boldsymbol{\theta}_Z` is empty, so the
first parameters belong to the random-effect covariance and the last one to
the residual.

.. jupyter-execute::

    names = reml.V.free_param_names
    len(names), names

The names follow the paths of the covariance tree:

- ``G/environment/factors/op_0/L_i_j`` is the loading of environment
  :math:`i + 1` on factor :math:`j + 1`.  The ``op_0`` segment appears
  because :class:`~torch_openreml.covariance.Gram` is itself an operator with
  one operand.
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

.. jupyter-execute::

    reml.is_converged(), reml.loglik()

The fitted covariance-parameter vector is on the transformed scale: the first
11 entries are loadings, the next six are half-log uniquenesses, and the last
is the half-log residual variance.

.. jupyter-execute::

    theta_hat.round(decimals=4)

For interpretation, :meth:`~torch_openreml.MixedModelREML.build_params`
applies the transforms and returns the parameters on the model scale.  It can
return them as a name-keyed dictionary, which is convenient for picking out
individual components:

.. jupyter-execute::

    params = reml.build_params(out_format="dict")
    params

The covariance tree is often the most direct way to get at the matrices
themselves.  The fitted :math:`\boldsymbol{\Sigma}` is at
``"G/environment"``, the loadings at
``"G/environment/factors/op_0"``, and the uniqueness matrix at
``"G/environment/uniqueness"``:

.. jupyter-execute::

    tree = reml.tree()

    Sigma_hat = tree["G/environment"]
    Lambda_hat = tree["G/environment/factors/op_0"]
    Psi_hat = tree["G/environment/uniqueness"]

    Sigma_hat.round(decimals=4)

.. jupyter-execute::

    Lambda_hat.round(decimals=4)

.. jupyter-execute::

    psi_hat = Psi_hat.diagonal()
    uniqueness.round(decimals=4), psi_hat.round(decimals=4)

.. jupyter-execute::

    sigma2_hat = params["R/sigma^2"]
    sigma2_hat

The residual variance is estimated at :math:`\hat{\sigma}^2 \approx 0.30`,
almost exactly the generating value of 0.3.  The uniquenesses are in the
right region but are not equal to the generating values
:math:`(0.5, 0.4, 0.5, 0.6, 0.4, 0.5)`: with 100 genotypes, 17 covariance
parameters are estimated from the between-genotype variation, and the
variance components are correspondingly noisy.

Signs and rotations in practice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fitted loadings are a representative of the rotation orbit, and the sign
of each column is arbitrary.  Before comparing with the generating matrix, it
is therefore natural to flip each column so that its leading entry is
positive.  The flip does not change :math:`\boldsymbol{\Sigma}`:

.. jupyter-execute::

    sign = torch.sign(torch.diagonal(Lambda_hat))
    Lambda_aligned = Lambda_hat * sign

    Lambda_aligned.round(decimals=4)

.. jupyter-execute::

    loadings

The first factor has the same broad pattern as the generator: loadings that
decline from the early to the late environments.  The second factor is
dominated by the later environments, as in the generator, but its smaller
entries differ; the estimates of the loadings and uniquenesses trade off
against each other in the likelihood, so the individual entries are much less
stable than the covariance that they generate.  The quantity to compare with
the generating values is :math:`\boldsymbol{\Sigma}`, not the individual
loadings:

.. jupyter-execute::

    (Sigma_hat - sigma_g).abs().max()

.. jupyter-execute::

    torch.corrcoef(torch.stack([Sigma_hat.flatten(), sigma_g.flatten()]))[0, 1]

The fitted covariance has correlation about 0.95 with the generating
covariance, and its largest element-wise error is about 0.23.  That is the
level of agreement to expect from this design; it is not exact, and it should
not be presented as exact recovery of :math:`\boldsymbol{\Lambda}` and
:math:`\boldsymbol{\Psi}`.

Recovery of the genetic effects
-------------------------------

The random-effect predictions are available from
:meth:`~torch_openreml.MixedModelREML.blup_per_term`, keyed by term name;
here there is a single term, ``"genotype"``.  Comparing them with the
generating effects in the ``genetic`` column checks the fit at the level of
individual genotype:

.. jupyter-execute::

    blup = reml.blup_per_term()["genotype"]

    torch.corrcoef(torch.stack([genetic, blup]))[0, 1]

.. jupyter-execute::

    (blup - genetic).pow(2).mean().sqrt()

The correlation is high, but the root mean squared error is not zero: the
BLUPs are conditional means and are shrunk towards zero relative to the
generating effects, and they inherit the estimation error in the covariance.

The fixed effects are the environment means.  Their estimates and standard
errors are:

.. jupyter-execute::

    beta_hat = reml.blue()
    se_beta = torch.sqrt(reml.var_blue())

    beta_hat.round(decimals=4), se_beta.round(decimals=4)

Compared with the generating means
:math:`(5.0, 5.2, 4.8, 5.1, 5.3, 4.9)`, all estimates are within about two and
a half standard errors.  The standard errors are around 0.1, driven mainly by
the genetic variance: each environment mean averages the 100 genotype
effects, whose variance is of order one, rather than the much smaller
residual variance.

Choosing the number of factors
------------------------------

In practice :math:`k` is unknown.  A simple comparison fits the model for
:math:`k = 1, 2` and compares the restricted log-likelihoods; the fixed
effects are the same in all fits, so AIC with :math:`p` covariance parameters
is :math:`-2\ell_R + 2p`.

.. jupyter-execute::

    def fit_fa(k):
        fa = Sum(
            factors=Gram(
                LowerTriangularMatrix(environment_labels, k),
                gram_type="xxt",
            ),
            uniqueness=DiagonalMatrix(environment_labels),
        )
        G = KroneckerProduct(
            genotype=IdentityMatrix(genotype_labels),
            environment=fa,
        )
        model = MixedModelREML(genotype=(Z, G), R=ScalarMatrix(len(y)))
        names_k, _ = model.optimize(y, X)
        loglik_k = model.loglik()[0].item()
        p_k = len(names_k)
        return p_k, loglik_k, -2 * loglik_k + 2 * p_k

    for k_k in (1, 2):
        p_k, loglik_k, aic_k = fit_fa(k_k)
        print(f"k = {k_k}: p = {p_k}, loglik = {loglik_k:.3f}, AIC = {aic_k:.3f}")

AIC prefers two factors here, which is consistent with the generating model.
For :math:`m = 6` environments there is no third factor to consider: the
FA(3) model is saturated.  The unbalanced trial below shows how quickly that
preference can disappear.

An unbalanced design
--------------------

The trial above is balanced by construction: every genotype appears in every
environment, twice.  Real multi-environment trials rarely look like that.
Genotypes enter and leave testing at different sites, so the observed cells
form a ragged pattern and the environments differ in size.  This second part
removes cells from the simulated trial, keeping both replicates in every cell
that remains: each genotype is grown in three of the six environments,
chosen at random with a fixed seed so that the document is reproducible.

Removing cells changes the data, and only the data.  The model, the
parameterisation and the number of free parameters are the same as in the
first part.  What changes is how much information the data carry about them.

The unbalanced trial
~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    env_levels = sorted(fa_met["environment"].unique())
    geno_levels = sorted(fa_met["genotype"].unique())

    generator = torch.Generator().manual_seed(20261120)
    missing = {}
    for genotype in geno_levels:
        picks = torch.randperm(len(env_levels), generator=generator)[:3]
        missing[genotype] = {env_levels[i] for i in picks.tolist()}

    keep = [
        environment not in missing[genotype]
        for genotype, environment in zip(fa_met["genotype"], fa_met["environment"])
    ]
    unbalanced = fa_met[keep].reset_index(drop=True)

    unbalanced.head()

Each genotype keeps the three environments it is grown in, with both
replicates, and loses the other three; ``missing`` records which three.

The subset has 600 plots in 300 cells, and the environments now hold between
42 and 55 genotypes instead of 100:

.. jupyter-execute::

    unbalanced_cells = unbalanced.drop_duplicates(["genotype", "environment"])
    len(unbalanced), len(unbalanced_cells), unbalanced_cells.groupby("environment").size()

The fixed and random effects are declared exactly as before.  The incidence
matrices, however, deserve a look, because they do not shrink with the data:

.. jupyter-execute::

    y_u = torch.tensor(unbalanced["yield"].to_numpy())
    X_u = DummyMatrix(unbalanced["environment"], levels=[env_levels])()
    Z_u = DummyMatrix(
        unbalanced["genotype"],
        unbalanced["environment"],
        levels=[geno_levels, env_levels],
    )

    Z_u().shape, (Z_u() == 0).all(dim=0).sum()

:class:`~torch_openreml.covariance.DummyMatrix` keeps one column per
combination of the levels it is given, so :math:`\mathbf{Z}_u` still has
:math:`100 \times 6 = 600` columns, 300 of which are all zero — one for each
cell that is no longer observed.  Those columns are not an accident and they
must not be removed: the genetic effect of the cell still exists, it simply
carries no observation.  ``drop_empty_cols=True`` would delete them and leave
a design that no longer matches a :math:`600 \times 600` covariance matrix
:math:`\mathbf{G}`.  The levels are passed explicitly to pin the grid to the
original trial; the default takes them from the data, which makes no
difference here but would silently drop a level that had lost all its plots.

The covariance is built from the same classes as in the first part:

.. jupyter-execute::

    fa_u = Sum(
        factors=Gram(LowerTriangularMatrix(env_levels, 2), gram_type="xxt"),
        uniqueness=DiagonalMatrix(env_levels),
    )
    G_u = KroneckerProduct(
        genotype=IdentityMatrix(geno_levels),
        environment=fa_u,
    )

    model_u = MixedModelREML(genotype=(Z_u, G_u), R=ScalarMatrix(len(y_u)))
    len(model_u.V.free_param_names)

There are still 18 free parameters.  The amount of data changed, not the size
of the model.

Fitting the unbalanced trial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    names_u, theta_u = model_u.optimize(y_u, X_u)

    model_u.is_converged(), model_u.loglik()

The fit converges in 13 iterations to a restricted log-likelihood of
:math:`-267.28`.  That number is *not* comparable with the :math:`-433.71` of
the balanced fit: the two fits see different data, so the difference measures
the 600 missing plots, not the quality of the fit.  Comparisons across
designs have to be made against the generating values.

The residual variance, against a generating value of 0.300, is

.. jupyter-execute::

    sigma2_u = model_u.build_params(out_format="dict")["R/sigma^2"]
    sigma2_u

Recovery of the covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    tree_u = model_u.tree()
    Sigma_u = tree_u["G/environment"]
    psi_u = tree_u["G/environment/uniqueness"].diagonal()

.. jupyter-execute::

    torch.corrcoef(torch.stack([Sigma_u.flatten(), sigma_g.flatten()]))[0, 1]

.. jupyter-execute::

    (Sigma_u - sigma_g).abs().max()

.. jupyter-execute::

    psi_u.round(decimals=4), uniqueness.round(decimals=4)

The correlation with the generating covariance drops from 0.949 to 0.905, and
the largest entry-wise error grows from 0.23 to 0.38.  For a third of the
data that is a modest loss: the covariance is still recognisable, and the
residual variance is 0.339 against a generating value of 0.300.

The uniquenesses are noisier.  The fifth one is estimated at zero, or rather
so close to it that it prints as zero: uniquenesses are exponentiated, so no
finite estimate can reach the boundary, but the optimizer walks the
unconstrained parameter far into the negative region and :math:`e^{2\theta}`
underflows.  What it has found is a flat direction: the two factors are
allowed to explain the whole of environment 5's variance, and with only 48
genotypes in that environment the data barely distinguish "all factor" from
"mostly factor".  A likelihood that is flat in a direction produces a large
standard error, not a large estimate, so the zero is a statement about the
information in the data rather than about the environment.  The other
uniquenesses — 0.231, 0.488, 0.167, 0.649 and 0.880 against generating values
of 0.5, 0.4, 0.5, 0.6 and 0.5 — show the same pattern as in the balanced
analysis: each is estimated with enough error that a single one is unreliable,
while the covariance they jointly produce is not.

Predicting untested cells
~~~~~~~~~~~~~~~~~~~~~~~~~

The practical reason for fitting the covariance is to predict the cells that
were not observed.  The BLUPs of the 600 cell effects are

.. jupyter-execute::

    blup_u = model_u.blup_per_term()["genotype"]
    seen = (Z_u() != 0).any(dim=0)
    seen.sum()

``seen`` marks the 300 cells that have plots; the other 300 are the empty
columns of :math:`\mathbf{Z}_u`.  Two helpers, and then the comparisons:

.. jupyter-execute::

    def corr(x, y):
        return torch.corrcoef(torch.stack([x, y]))[0, 1]

    def rmse(x, y):
        return (x - y).pow(2).mean().sqrt()

Against the generating effects, the cells with plots are predicted as

.. jupyter-execute::

    corr(genetic[seen], blup_u[seen])

.. jupyter-execute::

    rmse(genetic[seen], blup_u[seen])

and the cells without plots as

.. jupyter-execute::

    corr(genetic[~seen], blup_u[~seen])

.. jupyter-execute::

    rmse(genetic[~seen], blup_u[~seen])

.. jupyter-execute::

    blup_u[~seen].std(), genetic[~seen].std()

The observed cells are predicted as well as before (correlation 0.937 against
0.936 in the balanced fit).  The untested cells are not left at zero:
:math:`\text{BLUP} = \mathbf{G}\mathbf{Z}^\top\mathbf{V}^{-1}\mathbf{r}`, and
the row of :math:`\mathbf{G}` belonging to an unobserved cell is correlated
with the rows of the same genotype's observed cells.  Through
:math:`\boldsymbol{\Sigma}` the model therefore carries information from the
environments where the genotype *was* grown, and the correlation with the
generating effects is 0.50 with a root mean squared error of 0.89.

Two things limit the prediction.  The covariance is estimated, so the weights
are the estimated ones; and the observed cells of each genotype are
themselves shrunk, so the prediction starts from a shrunken base.  The
standard deviation of the untested BLUPs is 0.60 against 1.00 in the
generating values: the predictions are pulled towards the mean, which
under-predicts the good genotypes and over-predicts the poor ones.  These are
predictions conditional on the fitted covariance, useful for ranking and for
filling a table of trial results, but they are not measurements and should be
reported as such.

Choosing the number of factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The comparison of the previous section has to be repeated on the reduced
data:

.. jupyter-execute::

    def fit_fa_u(k):
        fa = Sum(
            factors=Gram(
                LowerTriangularMatrix(env_levels, k),
                gram_type="xxt",
            ),
            uniqueness=DiagonalMatrix(env_levels),
        )
        G = KroneckerProduct(
            genotype=IdentityMatrix(geno_levels),
            environment=fa,
        )
        model = MixedModelREML(genotype=(Z_u, G), R=ScalarMatrix(len(y_u)))
        names_k, _ = model.optimize(y_u, X_u)
        loglik_k = model.loglik()[0].item()
        p_k = len(names_k)
        return p_k, loglik_k, -2 * loglik_k + 2 * p_k

    p_1, loglik_1, aic_1 = fit_fa_u(1)
    print(f"k = 1: p = {p_1}, loglik = {loglik_1:.3f}, AIC = {aic_1:.3f}")
    print(f"k = 2: p = {len(names_u)}, loglik = {model_u.loglik()[0].item():.3f}, "
          f"AIC = {-2 * model_u.loglik()[0].item() + 2 * len(names_u):.3f}")

AIC now favours one factor, 568.0 against 570.6, a difference of 2.6 in the
opposite direction to the balanced analysis.  The covariance is still
recovered, but the evidence for the second factor is not there — and the
:math:`k = 2` fit had to push one uniqueness to zero to accommodate it.  A
difference of 2.6 is weak evidence by any standard, and this is a simulated
trial whose truth is :math:`k = 2`.  The lesson is not that the model has one
factor; it is that 300 cells carry too little information to choose.
Reporting the number of factors as settled on the strength of an unbalanced
trial would overstate what the data can support.

Summary
-------

- A factor-analytic covariance combines a low-rank term
  :math:`\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top` with a diagonal
  uniqueness matrix :math:`\boldsymbol{\Psi}`.  For six environments and two
  factors this uses 17 parameters instead of the 21 of an unstructured
  covariance.
- :class:`~torch_openreml.covariance.Gram` with
  ``gram_type="xxt"`` builds the low-rank term from a
  :class:`~torch_openreml.covariance.LowerTriangularMatrix`, and
  :class:`~torch_openreml.covariance.DiagonalMatrix` supplies the
  uniquenesses.  Wrapping the environment covariance in a
  :class:`~torch_openreml.covariance.KroneckerProduct` with an
  :class:`~torch_openreml.covariance.IdentityMatrix` for genotypes gives
  :math:`\mathbf{G} = \mathbf{I}_n \otimes \boldsymbol{\Sigma}`.
- The model is identified through the lower-triangular form of the loadings,
  but only up to the sign of each column.  The covariance
  :math:`\boldsymbol{\Sigma}` is the quantity that is identified, and it is
  the right target for a recovery check: individual loadings and uniquenesses
  are noisier and trade off against each other.
- In this trial the residual variance and the covariance are recovered
  well, and the BLUPs track the generating genetic effects, but the
  entry-wise loadings are not recovered exactly from 100 genotypes.
- The number of factors can be compared by fitting several values of
  :math:`k` and using the restricted log-likelihood; here AIC favours two
  factors over one.
- An unbalanced trial enters by dropping rows of the data, not columns of the
  design: :class:`~torch_openreml.covariance.DummyMatrix` keeps the full grid
  of cells, the empty columns stay zero, and the covariance has the same size
  and the same 18 parameters.  ``drop_empty_cols=True`` would break the
  correspondence between :math:`\mathbf{Z}` and :math:`\mathbf{G}`.
- Unbalance costs information, not just precision.  With a third of the cells
  missing the covariance is still recovered (correlation 0.905 against 0.949),
  but the loadings become harder to interpret, one uniqueness collapses to
  zero, and AIC no longer prefers the second factor that the balanced data
  supported.
- BLUPs for untested cells are not zero: :math:`\mathbf{G}\mathbf{Z}^\top
  \mathbf{V}^{-1}\mathbf{r}` borrows from the same genotype's observed
  environments through :math:`\boldsymbol{\Sigma}`.  They correlate about 0.5
  with the generating effects and are heavily shrunk, so they are
  predictions, not measurements.

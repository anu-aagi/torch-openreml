.. _tech:

Technical Documentation
========================================

Gaussian Linear Mixed Model Formulation
---------------------------------------

Consider the linear mixed model

.. math::
   :label: eq-main

   \mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}\mathbf{b} + \boldsymbol{\varepsilon},

where:

- :math:`\mathbf{Y} \in \mathbb{R}^{n \times 1}` is the response vector,
- :math:`\mathbf{X} \in \mathbb{R}^{n \times p}` is the fixed-effects design matrix,
- :math:`\boldsymbol{\beta} \in \mathbb{R}^{p \times 1}` is the vector of fixed-effect coefficients,
- :math:`\mathbf{Z} \in \mathbb{R}^{n \times q}` is the random-effects design matrix,
- :math:`\mathbf{b} \in \mathbb{R}^{q \times 1}` is the vector of random effects,
- :math:`\boldsymbol{\varepsilon} \in \mathbb{R}^{n \times 1}` is the vector of residual errors.

We assume

.. math::

   \mathbf{b} \sim \mathcal{N}(\mathbf{0}, \mathbf{G}), \quad
   \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}), \quad
   \mathbf{b} \perp \boldsymbol{\varepsilon},

where :math:`\mathbf{G} \in \mathbb{R}^{q \times q}` and :math:`\mathbf{R} \in \mathbb{R}^{n \times n}` are covariance matrices.

Let :math:`\boldsymbol{\theta}` denote the collection of variance components that parameterise :math:`\mathbf{G}` and :math:`\mathbf{R}`, i.e.,

.. math::

   \mathbf{G} = \mathbf{G}(\boldsymbol{\theta}), \quad
   \mathbf{R} = \mathbf{R}(\boldsymbol{\theta}).

Under these assumptions, the marginal distribution of :math:`\mathbf{Y}` is given by

.. math::

   \mathbf{Y} \sim \mathcal{N}\left(\mathbf{X}\boldsymbol{\beta},\; \mathbf{V}(\boldsymbol{\theta})\right),

where

.. math::
   :label: eq-sigma

   \mathbf{V}(\boldsymbol{\theta}) = \mathbf{Z}\mathbf{G}(\boldsymbol{\theta})\mathbf{Z}^\top + \mathbf{R}(\boldsymbol{\theta}).

BLUE
~~~~

Given :math:`\boldsymbol{\theta}`, the Best Linear Unbiased Estimator (BLUE) of :math:`\boldsymbol{\beta}` is the linear unbiased estimator that minimises the variance among all such estimators. Under the linear mixed model with covariance matrix :math:`\mathbf{V}(\boldsymbol{\theta})`, the BLUE of :math:`\boldsymbol{\beta}` is given by

.. math::

   \widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})
   =
   \left(\mathbf{X}^\top \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X}\right)^{-1}
   \mathbf{X}^\top \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{Y}.

This estimator is the generalised least squares (GLS) estimator of :math:`\boldsymbol{\beta}`.

BLUP
~~~~

Given :math:`\boldsymbol{\theta}`, we often wish to predict the random effects :math:`\mathbf{b}`. The Best Linear Unbiased Predictor (BLUP) of :math:`\mathbf{b}` is given by:

.. math::
   :label: eq-blup

   \widehat{\mathbf{b}}(\boldsymbol{\theta}) = \mathbf{G}(\boldsymbol{\theta})\mathbf{Z}^\top \mathbf{V}^{-1}(\boldsymbol{\theta}) (\mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}).

ML Estimation
~~~~~~~~~~~~~

Maximum likelihood (ML) estimation jointly estimates the fixed effects :math:`\boldsymbol{\beta}` and variance components :math:`\boldsymbol{\theta}` by maximising the marginal likelihood of the observed data. After integrating out the random effects :math:`\mathbf{b}` from the joint model, the marginal distribution of :math:`\mathbf{Y}` is

.. math::

   \mathbf{Y} \sim \mathcal{N}\!\left(\mathbf{X}\boldsymbol{\beta},\; \mathbf{V}(\boldsymbol{\theta})\right),

where :math:`\mathbf{V}(\boldsymbol{\theta}) = \mathbf{Z}\mathbf{G}(\boldsymbol{\theta})\mathbf{Z}^\top + \mathbf{R}(\boldsymbol{\theta})`. The corresponding log-likelihood is given by

.. math::

   \ell(\boldsymbol{\beta}, \boldsymbol{\theta})
   =
   -\frac{1}{2}\left[
   n\log(2\pi)
   + \log\left|\mathbf{V}(\boldsymbol{\theta})\right|
   + (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^\top
   \mathbf{V}(\boldsymbol{\theta})^{-1}
   (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})
   \right].

For fixed :math:`\boldsymbol{\theta}`, the log-likelihood is quadratic in :math:`\boldsymbol{\beta}`, and the maximiser can be obtained in closed form by setting :math:`\partial \ell / \partial \boldsymbol{\beta} = \mathbf{0}`. This yields the generalised least squares (GLS) estimator

.. math::

   \widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})
   =
   \left(
   \mathbf{X}^\top \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{Y}.

Substituting this expression back into the log-likelihood eliminates :math:`\boldsymbol{\beta}` and produces the profile log-likelihood

.. math::

   \ell_p(\boldsymbol{\theta})
   =
   -\frac{1}{2}\left[
   n\log(2\pi)
   + \log\left|\mathbf{V}(\boldsymbol{\theta})\right|
   + \mathbf{Y}^\top \mathbf{P}(\boldsymbol{\theta})\, \mathbf{Y}
   \right],

where

.. math::

   \mathbf{P}(\boldsymbol{\theta})
   =
   \mathbf{V}(\boldsymbol{\theta})^{-1}
   -
   \mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}
   \left(
   \mathbf{X}^\top\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top\mathbf{V}(\boldsymbol{\theta})^{-1}

is the residual projection matrix. The resulting optimisation problem depends only on :math:`\boldsymbol{\theta}`.

Since :math:`\ell_p(\boldsymbol{\theta})` is nonlinear in :math:`\boldsymbol{\theta}` and does not admit a closed-form maximiser, numerical optimisation methods are required. These methods rely on the score vector and a curvature approximation. Differentiating the profile log-likelihood with respect to a variance component :math:`\theta_k` gives

.. math::

   \frac{\partial \ell_p}{\partial \theta_k}
   =
   -\frac{1}{2}\left[
   \operatorname{tr}\!\left(
   \mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}
   \right)
   -
   \mathbf{Y}^\top \mathbf{P}
   \frac{\partial \mathbf{V}}{\partial \theta_k}
   \mathbf{P}\,\mathbf{Y}
   \right],

where the identity :math:`\partial \log|\mathbf{V}|/\partial \theta_k = \operatorname{tr}(\mathbf{V}^{-1}\partial \mathbf{V}/\partial \theta_k)` has been used, together with :math:`\partial \mathbf{P}/\partial \theta_k = -\mathbf{P}(\partial \mathbf{V}/\partial \theta_k)\mathbf{P}`.

A common curvature approximation is given by the Fisher information matrix, whose :math:`(k,j)` entry is

.. math::

   \mathcal{I}_{kj}(\boldsymbol{\theta})
   =
   \frac{1}{2}\operatorname{tr}\!\left(
   \mathbf{V}^{-1}\frac{\partial\mathbf{V}}{\partial\theta_k}
   \mathbf{V}^{-1}\frac{\partial\mathbf{V}}{\partial\theta_j}
   \right).

Starting from an initial value :math:`\boldsymbol{\theta}^{(0)}`, the Fisher scoring iteration updates :math:`\boldsymbol{\theta}` according to

.. math::

   \boldsymbol{\theta}^{(t+1)}
   =
   \boldsymbol{\theta}^{(t)}
   +
   \mathcal{I}\!\left(\boldsymbol{\theta}^{(t)}\right)^{-1}
   \mathbf{s}\!\left(\boldsymbol{\theta}^{(t)}\right),

where :math:`\mathbf{s}(\boldsymbol{\theta})` denotes the score vector. In practice, variance components are typically reparameterised (e.g. on a logarithmic scale) to ensure positivity during optimisation. Iterations are continued until convergence, for example when :math:`\|\boldsymbol{\theta}^{(t+1)} - \boldsymbol{\theta}^{(t)}\| < \varepsilon` for a prescribed tolerance :math:`\varepsilon > 0`.

Upon convergence, the ML estimate of :math:`\boldsymbol{\theta}` is :math:`\widehat{\boldsymbol{\theta}}_{ML}`, and the ML estimate of :math:`\boldsymbol{\beta}_{ML}` is recovered analytically as

.. math::

   \widehat{\boldsymbol{\beta}}_{ML}
   = \widehat{\boldsymbol{\beta}}\!\left(\widehat{\boldsymbol{\theta}}_{ML}\right)
   =
   \left(
   \mathbf{X}^\top
   \mathbf{V}(\widehat{\boldsymbol{\theta}}_{ML})^{-1}
   \mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top
   \mathbf{V}(\widehat{\boldsymbol{\theta}}_{ML})^{-1}
   \mathbf{Y}.

.. admonition:: Remark

   ML treats :math:`\boldsymbol{\beta}` and :math:`\boldsymbol{\theta}` symmetrically in the likelihood and is therefore consistent as :math:`n \to \infty`. However, in finite samples ML underestimates variance components because it does not account for the degrees of freedom consumed by estimating :math:`\boldsymbol{\beta}`. This bias motivates restricted maximum likelihood (REML), which maximises a likelihood formed from residual contrasts that are orthogonal to the column space of :math:`\mathbf{X}`, thereby adjusting for the estimation of fixed effects when recovering :math:`\boldsymbol{\theta}`.

REML Estimation
---------------

Restricted maximum likelihood (REML) constructs a likelihood based on linear combinations of :math:`\mathbf{Y}` that are free of :math:`\boldsymbol{\beta}`. Let :math:`\mathbf{A}` be an :math:`(n-p)\times n` matrix satisfying :math:`\mathbf{A}\mathbf{X}=\mathbf{0}` and :math:`\operatorname{rank}(\mathbf{A})=n-p`. Then

.. math::

   \mathbf{A}\mathbf{Y} \sim \mathcal{N}\big(\mathbf{0},\; \mathbf{A}\mathbf{V}(\boldsymbol{\theta})\mathbf{A}^\top\big),

and the corresponding log-likelihood is

.. math::

   \ell_R(\boldsymbol{\theta})
   =
   -\frac{1}{2}\left[
   (n-p)\log(2\pi)
   + \log\big|\mathbf{A}\mathbf{V}(\boldsymbol{\theta})\mathbf{A}^\top\big|
   + (\mathbf{A}\mathbf{Y})^\top \big(\mathbf{A}\mathbf{V}(\boldsymbol{\theta})\mathbf{A}^\top\big)^{-1} (\mathbf{A}\mathbf{Y})
   \right].

Using standard matrix identities, this expression can be rewritten as

.. math::

   \boxed{
   \ell_R(\boldsymbol{\theta})
   =
   -\frac{1}{2}\left[
   \log\big|\mathbf{V}(\boldsymbol{\theta})\big|
   + \log\big|\mathbf{X}^\top \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X}\big|
   + \mathbf{Y}^\top \mathbf{P}(\boldsymbol{\theta}) \mathbf{Y}
   \right]
   + C
   },

where

.. math::

   \boxed{
   \mathbf{P}(\boldsymbol{\theta})
   =
   \mathbf{V}(\boldsymbol{\theta})^{-1}
   -
   \mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}
   \left(
   \mathbf{X}^\top \mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top \mathbf{V}(\boldsymbol{\theta})^{-1}
   },

and :math:`C` is a constant independent of :math:`\boldsymbol{\theta}`. Since additive constants do not affect maximisation with respect to :math:`\boldsymbol{\theta}`, the REML log-likelihood is taken to be the above expression up to an additive constant.

Score Function
~~~~~~~~~~~~~~

Let :math:`\theta_k` denote the :math:`k`-th component of :math:`\boldsymbol{\theta}`, and define

.. math::

   \dot{\mathbf{V}}_k
   =
   \frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}.

Starting from

.. math::

   \ell_R(\boldsymbol{\theta})
   =
   -\frac{1}{2}\left[
   \log|\mathbf{V}|
   + \log\big|\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}\big|
   + \mathbf{Y}^\top \mathbf{P}\mathbf{Y}
   \right]
   + C,

we differentiate each term with respect to :math:`\theta_k`.

**Derivative of** :math:`\log|\mathbf{V}|`


.. math::

   \frac{\partial}{\partial \theta_k}\log|\mathbf{V}|
   =
   \operatorname{tr}\!\big(\mathbf{V}^{-1}\dot{\mathbf{V}}_k\big).

**Derivative of** :math:`\log|\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}|`

Set :math:`\mathbf{M} = \mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}`. Then

.. math::

   \frac{\partial}{\partial \theta_k}\log|\mathbf{M}|
   =
   \operatorname{tr}\!\big(\mathbf{M}^{-1} \dot{\mathbf{M}}_k\big),

where

.. math::

   \dot{\mathbf{M}}_k
   =
   \frac{\partial}{\partial \theta_k}
   \big(\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}\big)
   =
   -\,\mathbf{X}^\top \mathbf{V}^{-1}
   \dot{\mathbf{V}}_k
   \mathbf{V}^{-1}\mathbf{X}.

Hence,

.. math::

   \frac{\partial}{\partial \theta_k}
   \log\big|\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}\big|
   =
   -\,\operatorname{tr}\!\Big(
   \big(\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}\big)^{-1}
   \mathbf{X}^\top \mathbf{V}^{-1}
   \dot{\mathbf{V}}_k
   \mathbf{V}^{-1}\mathbf{X}
   \Big).

**Derivative of** :math:`\mathbf{Y}^\top \mathbf{P}\mathbf{Y}`

.. math::

   \frac{\partial}{\partial \theta_k}
   \big(\mathbf{Y}^\top \mathbf{P}\mathbf{Y}\big)
   =
   \mathbf{Y}^\top
   \frac{\partial \mathbf{P}}{\partial \theta_k}
   \mathbf{Y}.

Using

.. math::

   \frac{\partial \mathbf{P}}{\partial \theta_k}
   =
   -\,\mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P},

we obtain

.. math::

   \frac{\partial}{\partial \theta_k}
   \big(\mathbf{Y}^\top \mathbf{P}\mathbf{Y}\big)
   =
   -\,\mathbf{Y}^\top \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\mathbf{Y}.

**Compact Form**

Combining the three components and noting that :math:`C` does not depend on :math:`\boldsymbol{\theta}`, the score function is

.. math::

   S_k(\boldsymbol{\theta})
   =
   \frac{\partial \ell_R(\boldsymbol{\theta})}{\partial \theta_k}
   =
   -\frac{1}{2}
   \left[
   \operatorname{tr}\!\big(\mathbf{V}^{-1}\dot{\mathbf{V}}_k\big)
   -
   \operatorname{tr}\!\Big(
   \big(\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}\big)^{-1}
   \mathbf{X}^\top \mathbf{V}^{-1}
   \dot{\mathbf{V}}_k
   \mathbf{V}^{-1}\mathbf{X}
   \Big)
   -
   \mathbf{Y}^\top \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\mathbf{Y}
   \right].

To obtain a more compact form, recall the identity

.. math::

   \mathbf{P}
   =
   \mathbf{V}^{-1}
   -
   \mathbf{V}^{-1}\mathbf{X}
   \left(
   \mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top \mathbf{V}^{-1}.

Hence,

.. math::

   \operatorname{tr}\!\big(\mathbf{P}\,\dot{\mathbf{V}}_k\big)
   =
   \operatorname{tr}\!\big(\mathbf{V}^{-1}\dot{\mathbf{V}}_k\big)
   -
   \operatorname{tr}\!\Big(
   \mathbf{V}^{-1}\mathbf{X}
   \left(
   \mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top \mathbf{V}^{-1}
   \dot{\mathbf{V}}_k
   \Big).

Using the cyclic property of the trace, :math:`\operatorname{tr}(\mathbf{D}\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D})`, the second term becomes

.. math::

   \operatorname{tr}\!\Big(
   \mathbf{V}^{-1}\mathbf{X}
   \left(
   \mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top \mathbf{V}^{-1}
   \dot{\mathbf{V}}_k
   \Big)
   =
   \operatorname{tr}\!\Big(
   \left(
   \mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}
   \right)^{-1}
   \mathbf{X}^\top \mathbf{V}^{-1}
   \dot{\mathbf{V}}_k
   \mathbf{V}^{-1}\mathbf{X}
   \Big).

Therefore,

.. math::

   \operatorname{tr}\!\big(\mathbf{P}\,\dot{\mathbf{V}}_k\big)
   =
   \operatorname{tr}\!\big(\mathbf{V}^{-1}\dot{\mathbf{V}}_k\big)
   -
   \operatorname{tr}\!\Big(
   \big(\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X}\big)^{-1}
   \mathbf{X}^\top \mathbf{V}^{-1}
   \dot{\mathbf{V}}_k
   \mathbf{V}^{-1}\mathbf{X}
   \Big).

Substituting this identity into the previous expression for :math:`S_k(\boldsymbol{\theta})` yields

.. math::
   :label: eq-score

   \boxed{
   S_k(\boldsymbol{\theta})
   =
   \frac{1}{2}
   \left[
   \mathbf{Y}^\top \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\mathbf{Y}
   -
   \operatorname{tr}\!\big(\mathbf{P}\,\dot{\mathbf{V}}_k\big)
   \right]
   }.

Average Information Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~

The score function for :math:`\theta_k` is

.. math::

   S_k(\boldsymbol{\theta})
   =
   \frac{1}{2}
   \left[
   \mathbf{Y}^\top \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\mathbf{Y}
   -
   \operatorname{tr}\!\big(\mathbf{P}\,\dot{\mathbf{V}}_k\big)
   \right],

where

.. math::

   \dot{\mathbf{V}}_k
   =
   \frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}.

Observed Information
~~~~~~~~~~~~~~~~~~~~

Differentiating :math:`S_k` with respect to :math:`\theta_j` gives

.. math::

   -\frac{\partial^2 \ell_R}{\partial \theta_k \partial \theta_j}
   =
   -\frac{1}{2}
   \left[
   \mathbf{Y}^\top
   \frac{\partial}{\partial \theta_j}
   \big(\mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\big)
   \mathbf{Y}
   -
   \frac{\partial}{\partial \theta_j}
   \operatorname{tr}\!\big(\mathbf{P}\,\dot{\mathbf{V}}_k\big)
   \right].

For the quadratic term, using the product rule and :math:`\partial \mathbf{P}/\partial \theta_j = -\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}`, we obtain

.. math::

   \frac{\partial}{\partial \theta_j}
   \big(\mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\big)
   =
   -\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}
   -
   \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}
   +
   \mathbf{P}\,\ddot{\mathbf{V}}_{kj}\,\mathbf{P}.

For the trace term,

.. math::

   \frac{\partial}{\partial \theta_j}
   \operatorname{tr}\!\big(\mathbf{P}\,\dot{\mathbf{V}}_k\big)
   =
   -\,\operatorname{tr}\!\big(
   \mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\,\dot{\mathbf{V}}_k
   \big)
   +
   \operatorname{tr}\!\big(
   \mathbf{P}\,\ddot{\mathbf{V}}_{kj}
   \big).

Putting everything together yields the observed information:

.. math::

   \mathcal{J}_{kj}(\boldsymbol{\theta}) = -\frac{\partial^2 \ell_R}{\partial \theta_k \partial \theta_j}
   =
   \mathbf{Y}^\top
   \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}
   \mathbf{Y}
   -\frac{1}{2}
   \operatorname{tr}\!\big(
   \mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\,\dot{\mathbf{V}}_k
   \big)
   +\frac{1}{2}
   \operatorname{tr}\!\big(
   \mathbf{P}\,\ddot{\mathbf{V}}_{kj}
   \big)
   - \frac{1}{2} \mathbf{Y}^\top \mathbf{P}\,\ddot{\mathbf{V}}_{kj}\,\mathbf{P}\mathbf{Y}.

Expected Information
~~~~~~~~~~~~~~~~~~~~

We now derive the expected Fisher information matrix by taking expectations of the negative second derivatives. Recall that under the model, :math:`\mathbb{E}[\mathbf{Y}] = \mathbf{X}\boldsymbol{\beta}`. However, a fundamental property of the projection matrix :math:`\mathbf{P}` is that :math:`\mathbf{P}\mathbf{X} = \mathbf{0}`, which implies :math:`\mathbf{P}\mathbf{Y} = \mathbf{P}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})`. Consequently, when computing expectations of expressions involving :math:`\mathbf{P}`, we may without loss of generality treat :math:`\mathbf{Y}` as centered at zero. We therefore take :math:`\mathbb{E}[\mathbf{Y}] = \mathbf{0}` and :math:`\mathbb{E}[\mathbf{Y}\mathbf{Y}^\top] = \mathbf{V}` for the remainder of this derivation.

The observed information matrix is given by

.. math::

   -\frac{\partial^2 \ell_R}{\partial \theta_k \partial \theta_j}
   =
   \underbrace{\mathbf{Y}^\top \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P} \mathbf{Y}}_{T_1}
   -\frac12 \underbrace{\operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\,\dot{\mathbf{V}}_k \big)}_{T_2}
   +\frac12 \underbrace{\operatorname{tr}\!\big( \mathbf{P}\,\ddot{\mathbf{V}}_{kj} \big)}_{T_3}
   - \frac12 \underbrace{\mathbf{Y}^\top \mathbf{P}\,\ddot{\mathbf{V}}_{kj}\,\mathbf{P}\mathbf{Y}}_{T_4}.

We evaluate the expectation of each term in turn.

**Expectation of** :math:`T_1`

The quantity :math:`T_1` is a quadratic form in :math:`\mathbf{Y}`. For a zero-mean random vector with covariance :math:`\mathbf{V}`, the identity :math:`\mathbb{E}[\mathbf{Y}^\top \mathbf{A} \mathbf{Y}] = \operatorname{tr}(\mathbf{A} \mathbf{V})` holds for any conformable matrix :math:`\mathbf{A}`. Applying this with :math:`\mathbf{A} = \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}` yields

.. math::

   \mathbb{E}[T_1] = \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P} \, \mathbf{V} \big).

A key identity satisfied by :math:`\mathbf{P}` is :math:`\mathbf{P}\mathbf{V}\mathbf{P} = \mathbf{P}`, which can be verified by direct substitution of the definition :math:`\mathbf{P} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{V}^{-1}`. Using the cyclic property of the trace,

.. math::
   :nowrap:

   \begin{align*}
   \mathbb{E}[T_1] &= \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\, (\mathbf{P}\mathbf{V}) \big) \\
   &= \operatorname{tr}\!\big( \dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\mathbf{V}\mathbf{P} \big) \quad \text{(cyclic permutation)} \\
   &= \operatorname{tr}\!\big( \dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P} \big) \quad \text{(since $\mathbf{P}\mathbf{V}\mathbf{P} = \mathbf{P}$)} \\
   &= \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big) \quad \text{(cyclic permutation again)}.
   \end{align*}

Thus

.. math::

   \mathbb{E}[T_1] = \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big).

**Expectation of** :math:`T_2`

The term :math:`T_2` contains no random quantities; it is a deterministic function of the parameters. Hence

.. math::

   \mathbb{E}[T_2] = \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\,\dot{\mathbf{V}}_k \big).

**Expectation of** :math:`T_3`

Similarly, :math:`T_3` is non-random, giving

.. math::

   \mathbb{E}[T_3] = \operatorname{tr}\!\big( \mathbf{P}\,\ddot{\mathbf{V}}_{kj} \big).

**Expectation of** :math:`T_4`

The term :math:`T_4` is again a quadratic form. Applying the same identity as for :math:`T_1`,

.. math::

   \mathbb{E}[T_4] = \operatorname{tr}\!\big( \mathbf{P}\,\ddot{\mathbf{V}}_{kj}\,\mathbf{P} \, \mathbf{V} \big).

Using the cyclic property followed by :math:`\mathbf{P}\mathbf{V}\mathbf{P} = \mathbf{P}`,

.. math::
   :nowrap:

   \begin{align*}
   \mathbb{E}[T_4] &= \operatorname{tr}\!\big( \ddot{\mathbf{V}}_{kj}\,\mathbf{P} \mathbf{V} \mathbf{P} \big) \\
   &= \operatorname{tr}\!\big( \ddot{\mathbf{V}}_{kj}\,\mathbf{P} \big) \\
   &= \operatorname{tr}\!\big( \mathbf{P}\,\ddot{\mathbf{V}}_{kj} \big).
   \end{align*}

**Combining the expectations**

Assembling the four components,

.. math::
   :nowrap:

   \begin{align*}
   \mathbb{E}\!\left[ -\frac{\partial^2 \ell_R}{\partial \theta_k \partial \theta_j} \right]
   &= \mathbb{E}[T_1] - \frac12 \mathbb{E}[T_2] + \frac12 \mathbb{E}[T_3] - \frac12 \mathbb{E}[T_4] \\[4pt]
   &= \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big)
   - \frac12 \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\,\dot{\mathbf{V}}_k \big) \\
   &\quad + \frac12 \operatorname{tr}\!\big( \mathbf{P}\,\ddot{\mathbf{V}}_{kj} \big)
   - \frac12 \operatorname{tr}\!\big( \mathbf{P}\,\ddot{\mathbf{V}}_{kj} \big).
   \end{align*}

The two terms involving the second derivatives :math:`\ddot{\mathbf{V}}_{kj}` cancel identically. Moreover, the trace is symmetric in its matrix arguments, so

.. math::

   \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P}\,\dot{\mathbf{V}}_k \big)
   = \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big).

Substituting this into the expression above,

.. math::
   :nowrap:

   \begin{align*}
   \mathbb{E}\!\left[ -\frac{\partial^2 \ell_R}{\partial \theta_k \partial \theta_j} \right]
   &= \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big)
   - \frac12 \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big) \\[4pt]
   &= \frac12 \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big).
   \end{align*}

This final expression defines the :math:`(k,j)`-th entry of the REML Fisher information matrix,

.. math::

   \mathcal{I}_{kj}(\boldsymbol{\theta}) = \frac12 \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big).

Definition of the Average Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected information (Fisher information) is

.. math::

   \mathcal{I}_{kj}(\boldsymbol{\theta}) = \frac12 \operatorname{tr}\!\big( \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j \big).

The observed information is denoted :math:`\mathcal{J}_{kj}(\boldsymbol{\theta})`, and the *average information* matrix is defined as

.. math::

   \mathcal{A}_{kj}(\boldsymbol{\theta}) = \frac12 \Big( \mathcal{I}_{kj}(\boldsymbol{\theta}) + \mathcal{J}_{kj}(\boldsymbol{\theta}) \Big).

Substituting the expressions and simplifying, the trace terms cancel, yielding

.. math::
   :nowrap:

   \begin{align*}
   \mathcal{A}_{kj} &= \frac12 \mathbf{Y}^\top \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P} \mathbf{Y} \\
   &\quad + \frac14 \operatorname{tr}\!\big( \mathbf{P}\,\ddot{\mathbf{V}}_{kj} \big)
   - \frac14 \mathbf{Y}^\top \mathbf{P}\,\ddot{\mathbf{V}}_{kj}\,\mathbf{P}\mathbf{Y}.
   \end{align*}

Simplification for Linear Variance Component Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the common case where :math:`\mathbf{V}(\boldsymbol{\theta}) = \sum_{t=1}^p \theta_t \mathbf{M}_t` with known matrices :math:`\mathbf{M}_t`, the second derivatives vanish: :math:`\ddot{\mathbf{V}}_{kj} = \mathbf{0}` for all :math:`k,j`. The average information matrix then reduces to

.. math::
   :label: eq-AI

   \boxed{
   \mathcal{A}_{kj} = \frac12 \,\mathbf{Y}^\top \mathbf{P}\,\dot{\mathbf{V}}_k\,\mathbf{P}\,\dot{\mathbf{V}}_j\,\mathbf{P} \mathbf{Y}
   }.

This is the form commonly used in the AI-REML algorithm [Gilmour1995]_, where the average information matrix is computed directly from the data without requiring second derivatives. The algorithm proceeds by iteratively solving

.. math::

   \boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + \big( \mathcal{A}(\boldsymbol{\theta}^{(t)}) \big)^{-1} \mathbf{S}(\boldsymbol{\theta}^{(t)}).

.. rubric:: References

.. [Gilmour1995] Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995). Average information REML: An efficient algorithm for variance parameter estimation in linear mixed models. *Biometrics*, 51(4), 1440-1450.
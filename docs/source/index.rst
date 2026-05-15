.. _home:

.. raw:: html

   <div style="display:flex; align-items:center; gap:16px; margin-top: 20px;">
     <svg width="120" height="139" viewBox="0 0 120 139">
       <defs>
         <clipPath id="hex-clip">
           <polygon points="60,4 116,34 116,104 60,134 4,104 4,34"/>
         </clipPath>
       </defs>
       <polygon
         points="60,4 116,34 116,104 60,134 4,104 4,34"
         fill="white"
         stroke="black"
         stroke-width="3"
       />
       <image
         href="_static/hex-icon-cut.png"
         x="15" y="20"
         width="90" height="90"
         clip-path="url(#hex-clip)"
       />
       <polygon
         points="60,4 116,34 116,104 60,134 4,104 4,34"
         fill="none"
         stroke="black"
         stroke-width="3"
       />
     </svg>
     <div style="display:flex; flex-direction:column; gap:4px;">
       <h1 style="margin:0; padding:0;">torch-openreml</h1>
       <p style="margin:0; padding:0;">A PyTorch-based library for AI-REML estimation of linear mixed models.</p>
     </div>
   </div>

.. raw:: html

   <p>
     <img src="https://img.shields.io/badge/version-0.1.1--alpha-blue" alt="Version 0.1.1-alpha">
     <img src="https://img.shields.io/badge/license-GPLv3-blue" alt="GPL-3.0 License">
     <a href="https://www.python.org/">
       <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
     </a>
     <a href="https://pytorch.org/">
       <img src="https://img.shields.io/badge/pytorch-%3E%3D2.0-orange" alt="PyTorch">
     </a>
     <img src="https://img.shields.io/badge/status-experimental-yellow" alt="Experimental">
   </p>

.. container:: project-meta

   **Author & Maintainer:** Weihao (Patrick) Li (patrick.li@anu.edu.au)


Overview
--------

**torch-openreml** fits linear mixed-effects models using the Average Information REML
(AI-REML) algorithm on a PyTorch backend. It supports flexible specification of covariance
structures through a modular system of matrices and operators, along with automatic or manual
gradients and optional parameter transformations for constrained estimation.

Unlike traditional mixed-model software, it does not provide a formula interface. Instead,
users define the fixed- and random-effects design matrices and covariance
structures directly in code. The library is focused purely on the computational and optimization
backend rather than model specification syntax.

Features
--------

- **Torch-based backend**:

Built on PyTorch, supporting execution on CPU, GPU, and other available accelerators.

- **AI-REML estimation engine**

Variance component estimation using the Average Information REML (AI-REML) quasi-Newton optimization framework.

- **Extensible covariance structure**

Composable covariance structures and operators from built-in and user-defined components.

- **Hybrid differentiation support**

Support automatic differentiation and manually specified gradients.

- **Composable parameter transformations**
Configurable, chainable transformation pipelines for flexible parameterization.

Installation
------------

.. code-block:: bash

   # TODO: replace with actual install command when packaging is set up
   pip install torch-openreml

**Dependencies:** ``torch``, ``pandas``, ``tqdm`` (Python 3.12).

Getting Started
---------------

Dataset
~~~~~~~

To illustrate a quick start with the library, we begin by fitting a mixed-effects model using
the ``john_alpha`` dataset. This dataset contains field trial data from a resolvable alpha lattice design
conducted at Craibstone near Aberdeen.

It consists of 72 observations and 7 variables. In this example, we use ``yield`` (dry matter yield) as the
response variable, and ``rep`` (replicate identifier), ``block`` (incomplete block within replicate),
and ``gen`` (genotype or variety identifier) as covariates.


.. jupyter-execute::
    :hide-code:

    import torch_openreml
    print(torch_openreml.example_data.john_alpha)

Model Specification
~~~~~~~~~~~~~~~~~~~

The model includes an intercept, a single categorical fixed effect (``rep``), a random intercept for ``gen``, and a random interaction effect between ``rep`` and ``block``.

The model is specified as:

.. math::

    \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}\mathbf{b} + \boldsymbol{\varepsilon}

with marginal covariance structure:

.. math::

    \mathrm{Var}(\mathbf{y}) = \mathbf{V} = \mathbf{Z}\mathbf{G}\mathbf{Z}^\top + \mathbf{R}

and distributional assumptions:

.. math::

    \mathbf{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{G}), \quad
    \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})

For the present model, which includes two random intercept components and their interaction, the covariance contribution from the random effects is expressed as:

.. math::

    \mathbf{Z}\mathbf{G}\mathbf{Z}^\top =
    \mathbf{Z}_{gen}\mathbf{G}_{gen}\mathbf{Z}_{gen}^\top +
    \mathbf{Z}_{rep:block}
    \left(\mathbf{G}_{rep} \otimes \mathbf{G}_{block}\right)
    \mathbf{Z}_{rep:block}^\top

where :math:`\mathbf{G}_{rep} = \mathbf{I}` is fixed as the identity matrix for identifiability.

Import modules
~~~~~~~~~~~~~~

We begin by importing the required modules.

.. jupyter-execute::

    import torch
    from torch_openreml import REML
    from torch_openreml.utils import augment, n_distinct
    from torch_openreml.covariance import DummyMatrix, IdentityMatrix, ScalarMatrix, Sum, CovariancePropagation, KroneckerProduct
    from torch_openreml.example_data import john_alpha

Covariance Builder
~~~~~~~~~~~~~~~~~~

Next, we construct :math:`\mathbf{y}`, :math:`\mathbf{X}`, and the components required to define :math:`\mathbf{V}`. Both :math:`\mathbf{y}` and :math:`\mathbf{X}` are represented as torch tensors. The :py:class:`DummyMatrix <torch_openreml.covariance.DummyMatrix>` class serves as a matrix builder: it constructs the dummy matrix upon evaluation and accepts either `pandas.Series` or lists of strings as input. The argument ``drop_first=True`` removes the first column of the dummy matrix to avoid redundancy, as an intercept term is already included.

The classes :py:class:`ScalarMatrix <torch_openreml.covariance.ScalarMatrix>` and :py:class:`IdentityMatrix <torch_openreml.covariance.IdentityMatrix>` are also matrix builders, parameterized by the required matrix dimension.

We then assemble the covariance structure using composable operators. The :py:class:`CovariancePropagation <torch_openreml.covariance.CovariancePropagation>` operator represents the transformation :math:`\mathbf{Z}\mathbf{G}\mathbf{Z}^\top`. The :py:class:`KroneckerProduct <torch_openreml.covariance.KroneckerProduct>` operator computes the direct (Kronecker) product of two matrices, and :py:class:`Sum <torch_openreml.covariance.Sum>` aggregates multiple matrix components.

Altogether, the covariance structure can be written as:

.. math::

    \mathbf{V} =
    \mathbf{Z}_{gen}\mathbf{G}_{gen}\mathbf{Z}_{gen}^\top +
    \mathbf{Z}_{rep:block}
    \left(\mathbf{G}_{rep} \otimes \mathbf{G}_{block}\right)
    \mathbf{Z}_{rep:block}^\top +
    \sigma^2_{\varepsilon}\mathbf{I}

where :math:`\mathbf{G}_{rep} = \mathbf{I}`,
:math:`\mathbf{G}_{gen} = \sigma^2_{gen}\mathbf{I}`,
and :math:`\mathbf{G}_{block} = \sigma^2_{block}\mathbf{I}`.

The model parameters are defined as:

.. math::

    \boldsymbol{\theta} =
    \begin{bmatrix}
    \log(\sigma_{gen}) \\
    \log(\sigma_{block}) \\
    \log(\sigma_{\varepsilon})
    \end{bmatrix}

The logarithmic parameterization ensures that the variance components remain positive during optimization.

.. jupyter-execute::

    y = torch.tensor(john_alpha["yield"].values)
    X = augment(torch.ones(len(john_alpha), 1),
                DummyMatrix(john_alpha["rep"], drop_first=True)())

    Z_gen = DummyMatrix(john_alpha["gen"])
    Z_rep_block = DummyMatrix(john_alpha["rep"], john_alpha["block"])

    G_gen = ScalarMatrix(n_distinct(john_alpha["gen"]))
    G_rep = IdentityMatrix(n_distinct(john_alpha["rep"]))
    G_block = ScalarMatrix(n_distinct(john_alpha["block"]))

    R = ScalarMatrix(len(john_alpha))

    V = Sum(
        CovariancePropagation(Z_gen, G_gen),
        CovariancePropagation(
            Z_rep_block,
            KroneckerProduct(G_rep, G_block)
        ),
        R
    )

    print(V)

REML Optimizer
~~~~~~~~~~~~~~

Once the covariance structure has been defined, it is passed to :py:class:`REML <torch_openreml.REML>` to initialize the estimation procedure. The :py:meth:`optimize <torch_openreml.REML.optimize>` method is then called with :math:`\mathbf{y}`, :math:`\mathbf{X}`, and an initial value for :math:`\boldsymbol{\theta}` (set to zeros in this example). The `verbose` argument controls the level of diagnostic output.

Because the optimization is performed on the transformed parameter scale, the estimated parameters can be mapped back to variance components using :py:meth:`V.build_params <torch_openreml.covariance.Matrix.build_params>`. The resulting values correspond to the variance components associated with the parameter names stored in :py:attr:`V.param_names <torch_openreml.covariance.Matrix.param_names>`.

.. jupyter-execute::

    reml = REML(V)
    theta_hat, beta_hat, n_iter = reml.optimize(y, X, torch.zeros(3), verbose=2)
    print(theta_hat, V.build_params(theta_hat))
    print(V.free_param_names)
    print(beta_hat)



Documentation
-------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Section
     - Description

   * - :ref:`tech`
     - Model formulation, REML and ML theory, score and AI matrix derivations.

   * - :ref:`api`
     - Full documentation for ``REML``, covariance matrices, operators, transforms, and utilities.

Citing
------

.. code-block:: bibtex

   @software{torch_openreml,
     author = {Weihao Li},
     title  = {torch-openreml},
     year   = {2026},
     url    = {https://github.com/anu-aagi/torch-openreml/}
   }

----

.. toctree::
   :hidden:

   r_user
   tech
   api
   change_log
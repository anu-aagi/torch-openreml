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
     <img src="https://img.shields.io/badge/dev-0.4.0--alpha-blue" alt="Development version 0.4.0-alpha">
     <a href="https://pypi.org/project/torch-openreml/">
       <img src="https://img.shields.io/pypi/v/torch-openreml?include_prereleases" alt="PyPI version">
     </a>
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

    \mathbf{b} \sim \mathcal{N}(\mathbf{0}, \mathbf{G}), \quad
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
    import pandas as pd
    from torch_openreml import MarginalREML, blup
    from torch_openreml.covariance import DummyMatrix, IdentityMatrix, ScalarMatrix, Sum, CovariancePropagation, KroneckerProduct, Augment, BlockDiagonal
    from torch_openreml.example_data import john_alpha

Covariance Builder
~~~~~~~~~~~~~~~~~~

Next, we construct :math:`\mathbf{y}`, :math:`\mathbf{X}`, and the components required to define :math:`\mathbf{V}`. Both :math:`\mathbf{y}` and :math:`\mathbf{X}` are represented as torch tensors. The :py:class:`DummyMatrix <torch_openreml.covariance.DummyMatrix>` class serves as a matrix builder: it constructs the dummy matrix upon evaluation and accepts either `pandas.Series` or lists of strings as input. The argument ``drop_first=True`` removes the first column of the dummy matrix to avoid redundancy, as an intercept term is already included.

The classes :py:class:`ScalarMatrix <torch_openreml.covariance.ScalarMatrix>` and :py:class:`IdentityMatrix <torch_openreml.covariance.IdentityMatrix>` are also matrix builders, parameterized by the required matrix dimension. The dimension may be given as an integer, or directly as the grouping factor itself, in which case the number of distinct values defines the dimension.

We then assemble the covariance structure using composable operators. The :py:class:`Augment <torch_openreml.covariance.Augment>` operator binds its operands side by side, forming the joint design matrix :math:`\mathbf{Z} = [\mathbf{Z}_{gen}\ \mathbf{Z}_{rep:block}]`, and :py:class:`BlockDiagonal <torch_openreml.covariance.BlockDiagonal>` places :math:`\mathbf{G}_{gen}` and the Kronecker product :math:`\mathbf{G}_{rep} \otimes \mathbf{G}_{block}` on the diagonal of the joint :math:`\mathbf{G}`. The :py:class:`CovariancePropagation <torch_openreml.covariance.CovariancePropagation>` operator then represents the transformation :math:`\mathbf{Z}\mathbf{G}\mathbf{Z}^\top`. The :py:class:`KroneckerProduct <torch_openreml.covariance.KroneckerProduct>` operator computes the direct (Kronecker) product of two matrices, and :py:class:`Sum <torch_openreml.covariance.Sum>` adds the residual covariance :math:`\mathbf{R}`.

Altogether, the covariance structure can be written as:

.. math::

    \mathbf{V} = \mathbf{Z}\mathbf{G}\mathbf{Z}^\top + \mathbf{R},
    \qquad
    \mathbf{Z} =
    \begin{bmatrix}
    \mathbf{Z}_{gen} & \mathbf{Z}_{rep:block}
    \end{bmatrix},
    \qquad
    \mathbf{G} =
    \begin{bmatrix}
    \mathbf{G}_{gen} & \mathbf{0} \\
    \mathbf{0} & \mathbf{G}_{rep} \otimes \mathbf{G}_{block}
    \end{bmatrix}

with :math:`\mathbf{G}_{rep} = \mathbf{I}`,
:math:`\mathbf{G}_{gen} = \sigma^2_{gen}\mathbf{I}`,
:math:`\mathbf{G}_{block} = \sigma^2_{block}\mathbf{I}`,
and :math:`\mathbf{R} = \sigma^2_{\varepsilon}\mathbf{I}`.

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

    rep = john_alpha["rep"]
    block = john_alpha["block"]
    gen = john_alpha["gen"]
    n = len(john_alpha)

    y = torch.tensor(john_alpha["yield"].values)
    X = Augment(torch.ones(n, 1), DummyMatrix(rep, drop_first=True))()

    V = Sum(
        random = CovariancePropagation(
            Z = Augment(
                Z_gen = DummyMatrix(gen),
                Z_rep_block = DummyMatrix(rep, block),
            ),
            G = BlockDiagonal(
                G_gen = ScalarMatrix(gen),
                G_rep_block = KroneckerProduct(
                    G_rep = IdentityMatrix(rep),
                    G_block = ScalarMatrix(block),
                ),
            ),
        ),
        residual = ScalarMatrix(n),
    )

    print(V)

MarginalREML Optimizer
~~~~~~~~~~~~~~

Once the covariance structure has been defined, it is passed to :py:class:`MarginalREML <torch_openreml.MarginalREML>` to initialize the estimation procedure. The :py:meth:`optimize <torch_openreml.MarginalREML.optimize>` method is then called with :math:`\mathbf{y}`, :math:`\mathbf{X}`, and an initial value for :math:`\boldsymbol{\theta}` (set to zeros in this example). The `verbose` argument controls the level of diagnostic output.

Because the optimization is performed on the transformed parameter scale, the estimated parameters can be mapped back to variance components using :py:meth:`V.build_params <torch_openreml.covariance.Matrix.build_params>`. The resulting values correspond to the variance components associated with the parameter names stored in :py:attr:`V.param_names <torch_openreml.covariance.Matrix.param_names>`.

.. jupyter-execute::

    reml = MarginalREML(V)
    theta_hat, beta_hat, n_iter = reml.optimize(y, X, torch.zeros(3), verbose=2)
    print(theta_hat, V.build_params(theta_hat))
    print(V.free_param_names)
    print(beta_hat)

BLUP
~~~~

Once the variance components have been estimated, the random effects can be predicted using the best linear unbiased predictor (BLUP):

.. math::

    \hat{\mathbf{b}} = \mathbf{G}\mathbf{Z}^\top\mathbf{V}^{-1}\hat{\mathbf{e}},
    \qquad
    \hat{\mathbf{e}} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}

Here, :math:`\mathbf{Z}` and :math:`\mathbf{G}` are the joint design and covariance matrices defined above, represented by the :py:class:`Augment <torch_openreml.covariance.Augment>` and :py:class:`BlockDiagonal <torch_openreml.covariance.BlockDiagonal>` operands nested within the :py:class:`CovariancePropagation <torch_openreml.covariance.CovariancePropagation>` operator. These matrices can be retrieved directly from the covariance structure rather than reconstructed manually. Calling :py:meth:`V.tree <torch_openreml.covariance.Operator.tree>` at ``theta_hat`` evaluates the structure and returns the evaluated nodes, including the root ``"/"`` for :math:`\mathbf{V}` and ``"random/Z"`` and ``"random/G"`` for the random-effects design and covariance matrices.

.. jupyter-execute::

    tree, _ = V.tree(theta_hat)

The :py:func:`blup <torch_openreml.blup>` function takes :math:`\mathbf{y}`, :math:`\mathbf{X}`, :math:`\mathbf{Z}`, :math:`\mathbf{G}`, and :math:`\mathbf{V}` as inputs and returns a prediction for each random-effect level. In this example, the predictions correspond to the 24 genotypes followed by the 18 replicate-by-block combinations:

.. jupyter-execute::

    b_hat = blup(y, X, tree["random/Z"], tree["random/G"], tree["/"])
    b_hat

BLUPs for an individual random-effect component can be obtained in the same way by extracting its corresponding design and covariance matrices. For example, the random intercepts for ``gen`` are defined by ``"random/Z/Z_gen"`` and ``"random/G/G_gen"``:

.. jupyter-execute::

    gen_blup = blup(y, X, tree["random/Z/Z_gen"], tree["random/G/G_gen"], tree["/"])
    pd.Series(gen_blup, index=sorted(set(gen)))

Documentation
-------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Section
     - Description

   * - :ref:`vig`
     - Vignettes

   * - :ref:`tech`
     - Model formulation, REML and ML theory, score and AI matrix derivations.

   * - :ref:`api`
     - Full documentation for ``MarginalREML``, covariance matrices, operators, transforms, and utilities.

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

   vig
   r_user
   tech
   api
   change_log

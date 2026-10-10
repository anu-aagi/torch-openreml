For R Users
===========

torch-openreml can be used in R through the ``reticulate`` package.
We recommend installing torch-openreml inside a dedicated conda environment.

Installation
------------

1. Install Conda
~~~~~~~~~~~~~~~~

You can skip this step if conda is already installed on your system.

.. code-block:: R

    if (is.null(reticulate:::find_conda()[[1]])) {
        reticulate::install_miniconda()
    }

2. Create a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: R

    if (reticulate::condaenv_exists("torch-openreml")) {
        reticulate::conda_remove("torch-openreml")
    }

    reticulate::conda_create("torch-openreml",
                             python_version = "3.12.12")

3. Install torch-openreml
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: R

    reticulate::conda_install("torch-openreml",
                              pip = TRUE,
                              packages = c("git+https://github.com/anu-aagi/torch-openreml.git"))


Usage
-----

The following example mirrors the Getting Started example.
In R, it is often more convenient to construct design matrices directly
using ``model.matrix()``.
As in Python, a matrix dimension may be given as the grouping factor itself,
whose distinct values define the dimension.

.. code-block:: R

    library(reticulate)
    use_condaenv("torch-openreml")

    openreml <- import("torch_openreml", convert = FALSE)
    torch <- import("torch", convert = FALSE)

    BlockDiagonal <- openreml$covariance$BlockDiagonal
    ScalarMatrix <- openreml$covariance$ScalarMatrix
    IdentityMatrix <- openreml$covariance$IdentityMatrix
    KroneckerProduct <- openreml$covariance$KroneckerProduct
    CovariancePropagation <- openreml$covariance$CovariancePropagation
    Sum <- openreml$covariance$Sum
    MarginalREML <- openreml$MarginalREML

    data <- agridat::john.alpha
    n <- nrow(data)

    y <- torch$tensor(data$yield, dtype = torch$float32)
    X <- model.matrix(~ rep, data = data) |>
        torch$tensor(dtype = torch$float32)

    Z <- model.matrix(~ 0 + gen + block:rep, data = data) |>
        torch$tensor(dtype = torch$float32)

    V <- Sum(
        random = CovariancePropagation(
            Z = Z,
            G = BlockDiagonal(
                G_gen = ScalarMatrix(data$gen),
                G_rep_block = KroneckerProduct(
                    G_rep = IdentityMatrix(data$rep),
                    G_block = ScalarMatrix(data$block)
                )
            )
        ),
        residual = ScalarMatrix(n)
    )

    fit_openreml <- MarginalREML(V)
    result <- fit_openreml$optimize(y, X, torch$zeros(3L), verbose = 2L)

    print(py_to_r(fit_openreml$get_theta()$numpy()))
    print(py_to_r(V$build_params(fit_openreml$get_theta())$numpy()))
    print(py_to_r(fit_openreml$get_beta()$numpy()))

Random effects are predicted with the same best linear unbiased predictor (BLUP) as in the
Python example. The joint design and covariance matrices are read back from the fitted
covariance structure with :py:meth:`V.tree <torch_openreml.covariance.Operator.tree>`, which
evaluates the structure at the estimated parameters and returns each node under its path:
``"random/Z"`` and ``"random/G"`` hold :math:`\mathbf{Z}` and :math:`\mathbf{G}`, and the root
``"/"`` holds :math:`\mathbf{V}`. Note that reticulate indexes Python objects from zero, so
``[[0]]`` selects the results from the ``(results, free_params_by_path)`` pair that ``tree()``
returns.

.. code-block:: R

    tree <- V$tree(fit_openreml$get_theta())[[0]]

    b_hat <- openreml$blup(y, X, tree[["random/Z"]], tree[["random/G"]], tree[["/"]])
    print(py_to_r(b_hat$numpy()))

The BLUPs of a single random-effect component follow the same pattern. Here the design matrix is
a plain tensor built by ``model.matrix()``, so the columns of the ``gen`` term are built on their
own, while its covariance is the ``"random/G/G_gen"`` node of the tree.

.. code-block:: R

    Z_gen <- model.matrix(~ 0 + gen, data = data) |>
        torch$tensor(dtype = torch$float32)
    gen_blup <- openreml$blup(y, X, Z_gen, tree[["random/G/G_gen"]], tree[["/"]])
    print(setNames(py_to_r(gen_blup$numpy()), levels(data$gen)))

.. jupyter-execute::
    :hide-code:

    !Rscript source/r_user/code.R > source/r_user/output.txt 2>/dev/null

    !cat source/r_user/output.txt


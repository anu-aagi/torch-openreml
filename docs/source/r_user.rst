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
                              packages = c("torch-openreml")))


Usage
-----

The following example mirrors the Getting Started example.
In R, it is often more convenient to construct design matrices directly
using ``model.matrix()``.

.. code-block:: R

    library(reticulate)
    use_condaenv("torch-openreml")

    openreml <- import("torch_openreml", convert = FALSE)
    torch <- import("torch", convert = FALSE)

    ScalarMatrix <- openreml$covariance$ScalarMatrix
    IdentityMatrix <- openreml$covariance$IdentityMatrix
    KroneckerProduct <- openreml$covariance$KroneckerProduct
    CovariancePropagation <- openreml$covariance$CovariancePropagation
    Sum <- openreml$covariance$Sum
    REML <- openreml$REML

    data <- agridat::john.alpha

    y <- torch$tensor(data$yield, dtype = torch$float32)
    X <- model.matrix(~ rep, data = data) |>
        torch$tensor(dtype = torch$float32)

    Z_gen <- model.matrix(~ gen - 1, data = data) |>
        torch$tensor(dtype = torch$float32)
    Z_rep_block <- model.matrix(~ interaction(rep, block, lex.order = TRUE) - 1, data = data) |>
        torch$tensor(dtype = torch$float32)

    G_gen <- ScalarMatrix(length(unique(data$gen)))
    G_rep_block <- KroneckerProduct(IdentityMatrix(length(unique(data$rep))),
                                    ScalarMatrix(length(unique(data$block))))
    V <- Sum(CovariancePropagation(Z_gen, G_gen),
             CovariancePropagation(Z_rep_block, G_rep_block),
             ScalarMatrix(nrow(data)))

    fit_openreml <- REML(V)
    result <- fit_openreml$optimize(y, X, torch$zeros(3L), verbose = 2L)

    print(py_to_r(fit_openreml$get_theta()$numpy()))
    print(py_to_r(V$build_params(fit_openreml$get_theta())$numpy()))
    print(py_to_r(fit_openreml$get_beta()$numpy()))

.. jupyter-execute::
    :hide-code:

    !Rscript source/r_user/code.R > source/r_user/output.txt 2>/dev/null

    !cat source/r_user/output.txt


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

    ScalarMatrix <- openreml$covariance$ScalarMatrix
    IdentityMatrix <- openreml$covariance$IdentityMatrix
    KroneckerProduct <- openreml$covariance$KroneckerProduct
    MixedModelREML <- openreml$MixedModelREML

    data <- agridat::john.alpha
    n <- nrow(data)

    y <- torch$tensor(data$yield, dtype = torch$float32)
    X <- model.matrix(~ rep, data = data) |>
        torch$tensor(dtype = torch$float32)

    Z_gen <- model.matrix(~ 0 + gen, data = data) |>
        torch$tensor(dtype = torch$float32)
    Z_rep_block <- model.matrix(~ 0 + block:rep, data = data) |>
        torch$tensor(dtype = torch$float32)

    reml <- MixedModelREML(
        genotype = list(Z_gen, ScalarMatrix(data$gen)),
        replicate_block = list(
            Z_rep_block,
            KroneckerProduct(
                G_rep = IdentityMatrix(data$rep),
                G_block = ScalarMatrix(data$block)
            )
        ),
        R = ScalarMatrix(n)
    )
    print(reml)

Printing the model shows the marginal covariance it assembles from those terms.
The ``optimize`` method is then called with ``y``, ``X``, and an initial value for the
covariance parameters (zeros in this example), and the estimated parameters are mapped
back to variance components using ``build_params()``:

.. code-block:: R

    result <- reml$optimize(y, X, torch$zeros(3L), verbose = 1L)
    print(py_to_r(reml$build_params()$numpy()))

The fixed effects follow from generalized least squares, the best linear unbiased estimate (BLUE):

.. code-block:: R

    print(py_to_r(reml$blue(y, X)$numpy()))

The random effects are predicted with the best linear unbiased predictor (BLUP), one value per
random-effect level. In this example the predictions correspond to the 24 genotypes followed by
the 18 replicate-by-block combinations:

.. code-block:: R

    print(py_to_r(reml$blup(y, X)$numpy()))

:py:meth:`blup_per_term <torch_openreml.MixedModelREML.blup_per_term>` splits the same
predictions by random-effect term, returning a dict keyed by the term names supplied to the
constructor. The random intercepts for ``gen`` are then:

.. code-block:: R

    gen_blup <- reml$blup_per_term(y, X)[["genotype"]]
    print(setNames(py_to_r(gen_blup$numpy()), levels(data$gen)))

.. jupyter-execute::
    :hide-code:

    !Rscript source/r_user/code.R > source/r_user/output.txt 2>/dev/null

    !cat source/r_user/output.txt


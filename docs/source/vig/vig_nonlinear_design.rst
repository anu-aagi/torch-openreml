.. _vig_nonlinear_design:

Nonlinear Design Matrices
=========================

A random-effect design matrix is typically fixed. For example, a subject intercept uses a column of ones, while a random slope uses the observed covariate. Some effects, however, depend on model parameters and therefore change during estimation.

This page considers a model in which each subject's effect decays exponentially over time. The decay rate is estimated alongside the variance components, making the random-effect design matrix itself a function of an estimated parameter.

For the matrix and operator machinery behind this example, see
:ref:`vig_matrix` and :ref:`vig_operator`. For the model interface, see
:ref:`vig_mixed_model_reml`.

The model
---------

For observation :math:`j` belonging to subject :math:`i`, write

.. math::

    y_j = \boldsymbol{x}_j^\top \boldsymbol{\beta}
        + Z_{ji}(\tau) u_i + e_j,

with

.. math::

    Z_{ji}(\tau) =
    \begin{cases}
        \exp(-t_j / \tau), & \text{if } j \text{ belongs to } i, \\
        0, & \text{otherwise}.
    \end{cases}

The random effects and residuals are independent:

.. math::

    u_i \sim N(0, \sigma_u^2), \qquad
    e_j \sim N(0, \sigma_e^2).

The marginal covariance is still

.. math::

    \boldsymbol{V}(\tau, \sigma_u^2, \sigma_e^2)
    = \sigma_u^2 \boldsymbol{Z}(\tau)\boldsymbol{Z}(\tau)^\top
    + \sigma_e^2 \boldsymbol{I}.

The fixed effects enter linearly; the covariance is nonlinear because
:math:`\boldsymbol{Z}` depends on :math:`\tau`.

Simulating the data
-------------------

The simulation has 40 subjects observed at 10 time points. The true decay
rate is 2.5, the subject variance is 0.8, and the residual variance is
0.3:

.. jupyter-execute::
    :hide-code:

    import pandas as pd
    import torch

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import ScalarMatrix, SimpleMatrix

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("display.precision", 3)

    def simulate_nonlinear_design():
        n_subject = 40
        n_time = 10
        beta = torch.tensor([2.0, 0.4])
        sigma2_random = 0.8
        sigma2_residual = 0.3
        tau = 2.5

        generator = torch.Generator().manual_seed(2026)

        subject = torch.arange(n_subject).repeat_interleave(n_time)
        time = torch.arange(n_time).repeat(n_subject)

        u = torch.randn(n_subject, generator=generator)
        u = u * sigma2_random**0.5

        y = (
            beta[0]
            + beta[1] * time
            + u[subject] * torch.exp(-time / tau)
            + torch.randn(len(time), generator=generator)
            * sigma2_residual**0.5
        )

        return {
            "y": y,
            "subject": subject,
            "time": time,
            "u": u,
        }

    sim = simulate_nonlinear_design()
    data = pd.DataFrame(
        {
            "y": sim["y"],
            "subject": sim["subject"],
            "time": sim["time"],
        }
    )

    data.head()

.. raw:: html

   <details>
   <summary><strong>Show simulation code</strong></summary>

.. code-block:: python

    # Same code as the hidden cell above.
    import pandas as pd
    import torch

    from torch_openreml import MixedModelREML
    from torch_openreml.covariance import ScalarMatrix, SimpleMatrix

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("display.precision", 3)

    def simulate_nonlinear_design():
        n_subject = 40
        n_time = 10
        beta = torch.tensor([2.0, 0.4])
        sigma2_random = 0.8
        sigma2_residual = 0.3
        tau = 2.5

        generator = torch.Generator().manual_seed(2026)

        subject = torch.arange(n_subject).repeat_interleave(n_time)
        time = torch.arange(n_time).repeat(n_subject)

        u = torch.randn(n_subject, generator=generator)
        u = u * sigma2_random**0.5

        y = (
            beta[0]
            + beta[1] * time
            + u[subject] * torch.exp(-time / tau)
            + torch.randn(len(time), generator=generator)
            * sigma2_residual**0.5
        )

        return {
            "y": y,
            "subject": subject,
            "time": time,
            "u": u,
        }

    sim = simulate_nonlinear_design()
    data = pd.DataFrame(
        {
            "y": sim["y"],
            "subject": sim["subject"],
            "time": sim["time"],
        }
    )

    data.head()

.. raw:: html

   </details>

The response has a subject-specific decay around a linear time trend.
The simulated subject effects ``sim["u"]`` are kept for the comparison
at the end.

Building the design
-------------------

``SimpleMatrix`` wraps a callable that returns a matrix. The callable
receives the matrix's free parameters and can use any differentiable
torch operations:

.. jupyter-execute::

    y = sim["y"]
    subject = sim["subject"]
    time = sim["time"]
    n_subject = int(subject.max()) + 1

    X = torch.stack([torch.ones_like(time), time], dim=1)

    def decay_design(params):
        tau = params[0].exp()
        loadings = torch.exp(-time / tau)
        return torch.eye(n_subject)[subject] * loadings.unsqueeze(-1)

    Z = SimpleMatrix(num_free_params=1, call=decay_design)

    print("Z parameters:", Z.param_names)
    print("Z shape:", Z(torch.tensor([0.0])).shape)

The single parameter is ``theta_0``. The design uses
:math:`\tau = e^{\theta_0}`, so :math:`\tau` stays positive without a
constraint on the optimizer. At ``theta_0 = 0`` the decay rate is one.
The shape is resolved on the first call: 400 observations by 40 subjects.

``SimpleMatrix`` uses automatic differentiation, so the optimizer can
differentiate through ``torch.exp`` and the indexing operation.

Fitting the model
-----------------

The random term is the pair ``(Z, G)``. The covariance ``G`` is
:math:`\sigma_u^2 \boldsymbol{I}_{40}`. ``R`` is omitted, so it defaults
to :math:`\sigma_e^2 \boldsymbol{I}_{400}`:

.. jupyter-execute::

    model = MixedModelREML(random=(Z, ScalarMatrix(n_subject)))

    free_names, theta_hat = model.optimize(y, X)

    print("free parameters:", free_names)
    print("converged:", model.is_converged())

The free parameters are the decay parameter and the two variances.

Converting the estimates
------------------------

``build_params`` returns the natural-scale parameters. The decay
parameter needs one more transformation:

.. jupyter-execute::

    params = model.build_params(out_format="dict")

    tau_hat = params["Z/theta_0"].exp()
    sigma2_random_hat = params["G/sigma^2"]
    sigma2_residual_hat = params["R/sigma^2"]

    print("tau:", tau_hat)
    print("subject variance:", sigma2_random_hat)
    print("residual variance:", sigma2_residual_hat)

The fitted values are close to the simulation: about 2.42 for the decay
rate, 0.70 for the subject variance, and 0.32 for the residual variance.

The fixed effects are available in the usual way:

.. jupyter-execute::

    print("fixed effects:", model.blue())
    print("standard errors:", torch.sqrt(model.var_blue()))

Inspecting the fitted design
----------------------------

``tree`` returns every evaluated node. With a single random term the
paths are ``"/"``, ``"Z"``, ``"G"``, and ``"R"``:

.. jupyter-execute::

    tree = model.tree()

    print("tree paths:", list(tree))
    print("design shape:", tree["Z"].shape)
    print(tree["Z"][:3, :3])

The first three rows are the first subject at times 0, 1, and 2. The
first column is that subject's loading, which decays from 1.

Random-effect predictions
-------------------------

``blup_per_term`` returns the conditional random effects by term name.
For a single term the name is ``"random"``:

.. jupyter-execute::

    b_hat = model.blup_per_term()["random"]

    print("BLUP shape:", b_hat.shape)
    print(
        "BLUP correlation:",
        torch.corrcoef(torch.stack([sim["u"], b_hat]))[0, 1],
    )

The 40 predicted effects track the simulated ones with a correlation of
about 0.90.


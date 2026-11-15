Post-hoc estimation
===================

Post-hoc estimation for linear mixed models fitted by REML, operating on
tensors: best linear unbiased estimation (BLUE) of the fixed effects,
best linear unbiased prediction (BLUP) of the random effects, their
sampling covariance and variance matrices, and marginal and conditional
fitted values and residuals.

Routines that factorise ``V`` add a fixed numerical ridge ``1e-6 * I`` before
the factorisation. Pass an already-regularised ``V`` if a different ridge is
required.

.. autosummary::
   :toctree: ../generated

   torch_openreml.blue
   torch_openreml.cov_blue
   torch_openreml.var_blue
   torch_openreml.blup
   torch_openreml.cov_blup
   torch_openreml.var_blup
   torch_openreml.marginal_predict
   torch_openreml.marginal_residual
   torch_openreml.predict
   torch_openreml.residual
   torch_openreml.loglik

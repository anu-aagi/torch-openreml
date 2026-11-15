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

   torch_openreml.post.blue
   torch_openreml.post.cov_blue
   torch_openreml.post.var_blue
   torch_openreml.post.blup
   torch_openreml.post.cov_blup
   torch_openreml.post.var_blup
   torch_openreml.post.marginal_predict
   torch_openreml.post.marginal_residual
   torch_openreml.post.predict
   torch_openreml.post.residual
   torch_openreml.post.loglik

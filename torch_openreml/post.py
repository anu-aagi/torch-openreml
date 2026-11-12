r"""
Post-hoc estimation for linear mixed models fitted by REML.

Given fitted covariance parameters :math:`\boldsymbol{\theta}` and a
marginal covariance matrix :math:`\symbf{V}(\boldsymbol{\theta})`,
these functions compute quantities from a fitted model.
All inputs are ``torch.Tensor`` objects.

Functions:
    blue:
        Best linear unbiased estimator of the fixed effects
        :math:`\boldsymbol{\beta}`.
    cov_blue:
        Covariance matrix of the best linear unbiased estimator.
    var_blue:
        Variances of the best linear unbiased estimator, the diagonal of
        :func:`cov_blue`.
    blup:
        Best linear unbiased predictor of the random effects
        :math:`\symbf{b}`, requiring the random-effect covariance
        :math:`\symbf{G}` and design matrix :math:`\symbf{Z}`.
    cov_blup:
        Covariance matrix of the best linear unbiased predictor.
    var_blup:
        Variances of the best linear unbiased predictor, the diagonal of
        :func:`cov_blup`.
    loglik:
        Restricted log-likelihood evaluated at given covariance parameters.
    marginal_predict:
        Fitted values from fixed effects only.
    predict:
        Fitted values from fixed and random effects.
    marginal_residual:
        Residuals after removing the fixed effects.
    residual:
        Residuals after removing both fixed and random effects.
"""

import torch
from torch_openreml.utils import get_device, get_dtype

def blue(y, x, v):
    r"""
    Compute the best linear unbiased estimator (BLUE) of the fixed effects.

    Solves for :math:`\hat{\boldsymbol{\beta}}` via generalised least squares:

    .. math::
        \hat{\boldsymbol{\beta}} = (\symbf{X}^\top \symbf{V}^{-1} \symbf{X})^{-1}
        \symbf{X}^\top \symbf{V}^{-1} \symbf{y}

    Args:
        y (torch.Tensor): Response vector of shape ``(n,)``.
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: Coefficient estimate :math:`\hat{\boldsymbol{\beta}}`
            of shape ``(p,)``.
    """
    device = get_device(y, x, v)
    dtype = get_dtype(y, x, v)

    scalar = {}
    matrix = {}

    scalar["N"] = y.shape[0]

    matrix["V"] = v + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
    matrix["L"] = torch.linalg.cholesky(matrix["V"])

    matrix["Y"] = y.unsqueeze(-1)
    matrix["X"] = x

    matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
    matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])

    matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
    matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]

    matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])

    matrix[r"\hat{\beta}"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])

    return matrix[r"\hat{\beta}"].squeeze()

def cov_blue(x, v):
    r"""
    Compute the covariance matrix of the BLUE of the fixed effects.

    .. math::

        \operatorname{Var}(\hat{\boldsymbol{\beta}})
        = \left(\symbf{X}^\top\symbf{V}^{-1}\symbf{X}\right)^{-1},

    which is the ``C11`` block of the inverse C-form matrix (see
    :meth:`~torch_openreml.methods.CForm.c_inverse_blocks`). The covariance
    does not depend on the response :math:`\symbf{y}`, only on the design
    and the covariance parameters.

    Args:
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: BLUE covariance matrix of shape ``(p, p)``.
    """
    device = get_device(x, v)
    dtype = get_dtype(x, v)

    matrix = {}

    matrix["V"] = v + 1e-6 * torch.eye(x.shape[0], device=device, dtype=dtype)
    matrix["L"] = torch.linalg.cholesky(matrix["V"])

    matrix["V^{-1} X"] = torch.cholesky_solve(x, matrix["L"])
    matrix["X^T V^{-1} X"] = x.T @ matrix["V^{-1} X"]

    matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
    matrix["Var(beta)"] = torch.cholesky_inverse(matrix["L_{X^T V^{-1} X}"])
    matrix["Var(beta)"] = 0.5 * (matrix["Var(beta)"] + matrix["Var(beta)"].T)

    return matrix["Var(beta)"]

def var_blue(x, v):
    r"""
    Compute the variances of the BLUE of the fixed effects.

    This is the diagonal of :func:`cov_blue`,

    .. math::

        \operatorname{Var}(\hat{\beta}_j)
        = \left[\left(\symbf{X}^\top\symbf{V}^{-1}\symbf{X}\right)^{-1}
          \right]_{jj},

    the sampling variance of each fixed-effect estimate.

    Args:
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: BLUE variances of shape ``(p,)``.
    """
    return torch.diagonal(cov_blue(x, v))

def marginal_predict(y, x, v):
    r"""
    Compute marginal fitted values from fixed effects only.

    .. math::
        \hat{\symbf{y}} = \symbf{X}\hat{\boldsymbol{\beta}}

    where :math:`\hat{\boldsymbol{\beta}}` is the BLUE of the fixed
    effects (see :func:`blue`).

    Args:
        y (torch.Tensor): Response vector of shape ``(n,)``.
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: Marginal fitted values of shape ``(n,)``.
    """
    return x @ blue(y, x, v)

def marginal_residual(y, x, v):
    r"""
    Compute marginal residuals from fixed effects only.

    .. math::
        \hat{\symbf{e}} = \symbf{y} - \symbf{X}\hat{\boldsymbol{\beta}}

    where :math:`\hat{\boldsymbol{\beta}}` is the BLUE of the fixed
    effects (see :func:`marginal_predict`).

    Args:
        y (torch.Tensor): Response vector of shape ``(n,)``.
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: Marginal residuals of shape ``(n,)``.
    """
    return y - marginal_predict(y, x, v)

def blup(y, x, z, g, v):
    r"""
    Compute the best linear unbiased predictor (BLUP) of the random effects.

    .. math::
        \hat{\symbf{b}} = \symbf{G} \symbf{Z}^\top \symbf{V}^{-1} \hat{\symbf{e}}

    where :math:`\hat{\symbf{e}}` are the marginal residuals.

    Args:
        y (torch.Tensor): Response vector of shape ``(n,)``.
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
        g (torch.Tensor): Random-effect covariance matrix of shape ``(q, q)``,
            e.g. :math:`\symbf{G}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: Random-effect predictions of shape ``(q,)``.
    """
    device = get_device(y, x, z, g, v)
    dtype = get_dtype(y, x, z, g, v)

    scalar = {}
    matrix = {}

    scalar["N"] = y.shape[0]

    matrix["V"] = v + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
    matrix["L"] = torch.linalg.cholesky(matrix["V"])

    matrix["Y"] = y.unsqueeze(-1)

    matrix["e"] = marginal_residual(y, x, v).unsqueeze(-1)

    matrix["V^{-1} e"] = torch.cholesky_solve(matrix["e"], matrix["L"])

    matrix["G"] = g

    return (matrix["G"] @ (z.T @ matrix["V^{-1} e"])).squeeze(-1)

def cov_blup(x, z, g, v):
    r"""
    Compute the covariance matrix of the BLUPs.

    .. math::

        \operatorname{Var}(\hat{\symbf{b}})
        = \symbf{G}\symbf{Z}^\top\symbf{P}\symbf{Z}\symbf{G},
        \qquad
        \symbf{P}
        = \symbf{V}^{-1}
          - \symbf{V}^{-1}\symbf{X}
            (\symbf{X}^\top\symbf{V}^{-1}\symbf{X})^{-1}
            \symbf{X}^\top\symbf{V}^{-1}.

    Unlike the equivalent block form
    :math:`\symbf{G} - \symbf{C}_{22}`, this expression only requires
    :math:`\symbf{V}` to be invertible, not :math:`\symbf{G}`.

    Args:
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
        g (torch.Tensor): Random-effect covariance matrix of shape ``(q, q)``,
            e.g. :math:`\symbf{G}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: BLUP covariance matrix of shape ``(q, q)``.
    """
    device = get_device(x, z, g, v)
    dtype = get_dtype(x, z, g, v)

    scalar = {}
    matrix = {}

    scalar["N"] = v.shape[0]
    matrix["X"] = x.to(device=device, dtype=dtype)
    matrix["Z"] = z.to(device=device, dtype=dtype)
    matrix["G"] = g.to(device=device, dtype=dtype)

    matrix["V"] = v + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
    matrix["L"] = torch.linalg.cholesky(matrix["V"])

    matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])
    matrix["V^{-1} Z G"] = torch.cholesky_solve(matrix["Z"] @ matrix["G"], matrix["L"])

    matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
    matrix["L_x"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])

    matrix["Z^T V^{-1} X"] = matrix["Z"].T @ matrix["V^{-1} X"]
    matrix["X^T V^{-1} Z G"] = matrix["X"].T @ matrix["V^{-1} Z G"]
    matrix["Z^T V^{-1} Z G"] = matrix["Z"].T @ matrix["V^{-1} Z G"]

    correction = matrix["Z^T V^{-1} X"] @ torch.cholesky_solve(matrix["X^T V^{-1} Z G"], matrix["L_x"])

    matrix["Z^T P Z G"] = matrix["Z^T V^{-1} Z G"] - correction

    matrix["Var(b)"] = matrix["G"] @ matrix["Z^T P Z G"]
    matrix["Var(b)"] = 0.5 * (matrix["Var(b)"] + matrix["Var(b)"].T)
    return matrix["Var(b)"]

def var_blup(x, z, g, v):
    r"""
    Compute the variances of the BLUPs.

    This is the diagonal of :func:`cov_blup`,

    .. math::

        \operatorname{Var}(\hat{b}_i)
        = \left(\symbf{G}\symbf{Z}^\top\symbf{P}\symbf{Z}\symbf{G}\right)_{ii},

    the sampling variance of each random-effect prediction.

    Args:
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
        g (torch.Tensor): Random-effect covariance matrix of shape ``(q, q)``,
            e.g. :math:`\symbf{G}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: BLUP variances of shape ``(q,)``.
    """
    return torch.diagonal(cov_blup(x, z, g, v))

def predict(y, x, z, g, v):
    r"""
    Compute conditional fitted values including random effects.

    .. math::
        \hat{\symbf{y}} = \symbf{X}\hat{\boldsymbol{\beta}} +
        \symbf{Z}\hat{\symbf{b}}

    Args:
        y (torch.Tensor): Response vector of shape ``(n,)``.
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
        g (torch.Tensor): Random-effect covariance matrix of shape ``(q, q)``,
            e.g. :math:`\symbf{G}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: Conditional fitted values of shape ``(n,)``.
    """
    device = get_device(y, x, z, g, v)
    dtype = get_dtype(y, x, z, g, v)

    scalar = {}
    matrix = {}

    scalar["N"] = y.shape[0]
    matrix["V"] = v + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
    matrix["L"] = torch.linalg.cholesky(matrix["V"])

    matrix["Y"] = y.unsqueeze(-1)
    matrix["X"] = x

    matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
    matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])

    matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
    matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]

    matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])

    matrix[r"\beta"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])

    matrix[r"\hat{Y}"] = matrix["X"] @ matrix[r"\beta"]

    matrix["e"] = matrix["Y"] - matrix[r"\hat{Y}"]

    matrix["V^{-1} e"] = torch.cholesky_solve(matrix["e"], matrix["L"])

    matrix["G"] = g

    return (matrix[r"\hat{Y}"] + z @ (matrix["G"] @ (z.T @ matrix["V^{-1} e"]))).squeeze()

def residual(y, x, z, g, v):
    r"""
    Compute conditional residuals including random effects.

    .. math::
        \hat{\symbf{e}} = \symbf{y} - \hat{\symbf{y}}

    where :math:`\hat{\symbf{y}}` is the conditional prediction from
    :func:`predict`.

    Args:
        y (torch.Tensor): Response vector of shape ``(n,)``.
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
        g (torch.Tensor): Random-effect covariance matrix of shape ``(q, q)``,
            e.g. :math:`\symbf{G}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: Conditional residuals of shape ``(n,)``.
    """
    return y - predict(y, x, z, g, v)

def loglik(y, x, v):
    r"""
    Evaluate the REML log-likelihood.

    .. math::
        \ell_R = -\frac{1}{2} \left(
            \log |\symbf{V}| +
            \log |\symbf{X}^\top \symbf{V}^{-1} \symbf{X}| +
            \symbf{y}^\top \symbf{P} \symbf{y}
        \right)

    where :math:`\symbf{P} = \symbf{V}^{-1} - \symbf{V}^{-1}\symbf{X}
    (\symbf{X}^\top\symbf{V}^{-1}\symbf{X})^{-1}\symbf{X}^\top\symbf{V}^{-1}`
    is the projection matrix. The quadratic :math:`\symbf{y}^\top \symbf{P}
    \symbf{y}` is evaluated as :math:`\hat{\symbf{e}}^\top \symbf{V}^{-1}
    \hat{\symbf{e}}`, with :math:`\hat{\symbf{e}}` the marginal residuals.

    Args:
        y (torch.Tensor): Response vector of shape ``(n,)``.
        x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
        v (torch.Tensor): Marginal covariance matrix of shape ``(n, n)``,
            e.g. :math:`\symbf{V}(\boldsymbol{\theta})` evaluated at fitted
            covariance parameters.

    Returns:
        torch.Tensor: Scalar REML log-likelihood value.
    """
    device = get_device(y, x, v)
    dtype = get_dtype(y, x, v)

    scalar = {}
    matrix = {}

    scalar["N"] = y.shape[0]
    matrix["X"] = x

    matrix["V"] = v + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)

    matrix["L"] = torch.linalg.cholesky(matrix["V"])

    matrix["Y"] = y.unsqueeze(-1)

    matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
    matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])

    matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
    matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]

    matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])

    matrix[r"\hat{\beta}"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])

    matrix["e"] = matrix["Y"] - matrix["X"] @ matrix[r"\hat{\beta}"]

    matrix["V^{-1} e"] = torch.cholesky_solve(matrix["e"], matrix["L"])

    scalar["log |V|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L"])))
    scalar["log |X^T V^{-1} X|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L_{X^T V^{-1} X}"])))
    scalar["Y^T P Y"] = (matrix["e"].T @ matrix["V^{-1} e"]).squeeze()

    return -0.5 * (scalar["log |V|"] + scalar["log |X^T V^{-1} X|"] + scalar["Y^T P Y"])

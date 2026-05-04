"""
Restricted maximum likelihood (REML) estimation.

This module implements REML estimation for linear mixed-effects models
via the average information (AI) algorithm. The AI algorithm combines
the computational efficiency of expected and observed information matrices
to achieve fast, robust convergence for variance component estimation.

Classes:
    REML:
        REML estimator with AI-based optimisation, supporting fixed-effect
        estimation via BLUE, random-effect prediction via BLUP, and
        marginal and conditional prediction.
"""

import torch
from torch_openreml.utils import get_device, get_dtype
from torch_openreml.covariance.matrix import Matrix
from tqdm import tqdm

class REML:
    r"""
    REML estimator for linear mixed-effects models.

    Fits variance components :math:`\boldsymbol{\theta}` by maximising the
    restricted log-likelihood

    .. math::
        \ell_R(\boldsymbol{\theta}) = -\frac{1}{2} \left(
            \log |\symbf{V}(\boldsymbol{\theta})| +
            \log |\symbf{X}^\top \symbf{V}(\boldsymbol{\theta})^{-1} \symbf{X}| +
            \symbf{y}^\top \symbf{P} \symbf{y}
        \right)

    where :math:`\symbf{P} = \symbf{V}(\boldsymbol{\theta})^{-1} - \symbf{V}(\boldsymbol{\theta})^{-1}\symbf{X}
    (\symbf{X}^\top\symbf{V}(\boldsymbol{\theta})^{-1}\symbf{X})^{-1}\symbf{X}^\top\symbf{V}(\boldsymbol{\theta})^{-1}`
    is the projection matrix,
    and :math:`\symbf{V}(\boldsymbol{\theta}) = \symbf{Z}\symbf{G}(\boldsymbol{\theta})\symbf{Z}^\top + \symbf{R}(\boldsymbol{\theta})`
    is the mariginal covariance matrix.

    Optimisation uses the average information (AI)
    algorithm, which forms a quasi-Newton step
    :math:`\Delta = \symbf{AI}^{-1} \symbf{s}` from the score vector
    :math:`\symbf{s}` and the AI matrix at each iteration.

    The covariance model :math:`\symbf{V}(\boldsymbol{\theta})` is supplied
    either as a :class:`~torch_openreml.covariance.matrix.Matrix` instance
    via ``v_builder``, or as a pair of callables ``map_theta_to_v`` and
    ``map_theta_to_dv``. If ``map_theta_to_dv`` is omitted, the Jacobian
    is computed automatically via :func:`torch.func.jacrev`.
    """
    
    def __init__(self, v_builder=None, map_theta_to_v=None, map_theta_to_g=None, map_theta_to_dv=None):
        """
        Initialize a REML estimator.

        Args:
            v_builder (Matrix, optional): A :class:`~torch_openreml.covariance.matrix.Matrix`
                instance whose :meth:`~torch_openreml.covariance.matrix.Matrix.map_theta_to_v`
                and :meth:`~torch_openreml.covariance.matrix.Matrix.map_theta_to_dv`
                methods are used to construct :math:`\symbf{V}` and its Jacobian.
                Takes precedence over ``map_theta_to_v`` and ``map_theta_to_dv``
                if both are provided.
            map_theta_to_v (callable, optional): Maps a flat parameter tensor
                :math:`\\boldsymbol{\\theta}`  to the covariance matrix :math:`\symbf{V}`.
                Required if ``v_builder`` is not provided.
            map_theta_to_g (callable, optional): Maps :math:`\\boldsymbol{\\theta}` to
                the random-effect covariance matrix :math:`\symbf{G}`. Required
                for :meth:`blup`, :meth:`predict`, and :meth:`residual`.
            map_theta_to_dv (callable, optional): Maps :math:`\\boldsymbol{\\theta}` to
                the Jacobian of :math:`\symbf{V}`, a 3D tensor of shape
                ``(num_params, n, n)``. If ``None`` and ``v_builder`` is not
                provided, the Jacobian is computed via automatic differentiation.

        Raises:
            TypeError: If ``v_builder`` is not a
                :class:`~torch_openreml.covariance.matrix.Matrix` instance.
            ValueError: If neither ``v_builder`` nor ``map_theta_to_v`` is provided.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml import REML
            from torch_openreml.covariance import ScalarMatrix

            n, p = 50, 2
            y = torch.randn(n)
            x = torch.randn(n, p)
            theta = torch.tensor([0.0])

            mat = ScalarMatrix(n)
            reml = REML(v_builder=mat)
            theta_hat, beta_hat, n_iter = reml.optimize(y, x, theta, verbose=2)
            theta_hat, beta_hat
        """
        
        if v_builder is not None and not isinstance(v_builder, Matrix):
            raise TypeError("'v_builder' must be a Matrix instance or None!")
        
        if v_builder is None and map_theta_to_v is None:
            raise ValueError("At least one of 'v_builder' or 'map_theta_to_v' must be provided!")
        
        if v_builder is not None:
            self.map_theta_to_v = v_builder.map_theta_to_v
            self.map_theta_to_dv = v_builder.map_theta_to_dv
        else:
            self.map_theta_to_v = map_theta_to_v
            self.jacobian_func = torch.func.jacrev(map_theta_to_v)
            self.map_theta_to_dv = map_theta_to_dv
            
        self.map_theta_to_g = map_theta_to_g

    def blue(self, y, x, theta):
        r"""
        Compute the best linear unbiased estimator (BLUE) of fixed effects.

        Solves for :math:`\hat{\boldsymbol{\beta}}` via generalised least squares:

        .. math::
            \hat{\boldsymbol{\beta}} = (\symbf{X}^\top \symbf{V}(\boldsymbol{\theta})^{-1} \symbf{X})^{-1}
            \symbf{X}^\top \symbf{V}(\boldsymbol{\theta})^{-1} \symbf{y}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat variance component parameter tensor.

        Returns:
            torch.Tensor: Fixed-effect estimate :math:`\hat{\boldsymbol{\beta}}`
            of shape ``(p,)``.
        """
        device = get_device(y, x, theta)
        dtype = get_dtype(y, x, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        matrix["V"] = self.map_theta_to_v(theta)
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
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

    def marginal_predict(self, y, x, theta):
        r"""
        Compute marginal fitted values from fixed effects only.

        .. math::
            \hat{\symbf{y}} = \symbf{X} \hat{\boldsymbol{\beta}}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat variance component parameter tensor.

        Returns:
            torch.Tensor: Marginal fitted values of shape ``(n,)``.
        """
        vector = {}
        
        vector[r"\hat{\beta}"] = self.blue(y, x, theta)
        return x @ vector[r"\hat{\beta}"]

    def marginal_residual(self, y, x, theta):
        r"""
        Compute marginal residuals from fixed effects only.

        .. math::
            \hat{\symbf{e}} = \symbf{y} - \symbf{X}\hat{\boldsymbol{\beta}}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat variance component parameter tensor.

        Returns:
            torch.Tensor: Marginal residuals of shape ``(n,)``.
        """
        return y - self.marginal_predict(y, x, theta)

    def blup(self, y, x, z, theta, map_theta_to_g=None):
        r"""
        Compute the best linear unbiased predictor (BLUP) of random effects.

        .. math::
            \hat{\symbf{u}} = \symbf{G} \symbf{Z}^\top \symbf{V}^{-1}(\boldsymbol{\theta}) \hat{\symbf{e}}

        where :math:`\hat{\symbf{e}}` are the marginal residuals.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
            theta (torch.Tensor): Flat variance component parameter tensor.
            map_theta_to_g (callable, optional): Maps ``theta`` to the
                random-effect covariance matrix :math:`\symbf{G}`. Defaults
                to :attr:`map_theta_to_g` set at initialisation.

        Returns:
            torch.Tensor: Random-effect predictions of shape ``(q,)``.
        """
        if map_theta_to_g is None:
            map_theta_to_g = self.map_theta_to_g
      
        device = get_device(y, x, z, theta)
        dtype = get_dtype(y, x, z, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        
        matrix["V"] = self.map_theta_to_v(theta)
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
        matrix["L"] = torch.linalg.cholesky(matrix["V"])
        
        matrix["Y"] = y.unsqueeze(-1)
        
        matrix["e"] = self.marginal_residual(y, x, theta).unsqueeze(-1)
        
        matrix["V^{-1} e"] = torch.cholesky_solve(matrix["e"], matrix["L"])
        
        matrix["G"] = map_theta_to_g(theta)
        
        return (matrix["G"] @ (z.T @ matrix["V^{-1} e"])).squeeze()

    def predict(self, y, x, z, theta, map_theta_to_g=None):
        r"""
        Compute conditional fitted values including random effects.

        .. math::
            \hat{\symbf{y}} = \symbf{X}\hat{\boldsymbol{\beta}} +
            \symbf{Z}\hat{\symbf{u}}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
            theta (torch.Tensor): Flat variance component parameter tensor.
            map_theta_to_g (callable, optional): Maps ``theta`` to
                :math:`\symbf{G}`. Defaults to :attr:`map_theta_to_g` set
                at initialisation.

        Returns:
            torch.Tensor: Conditional fitted values of shape ``(n,)``.
        """
        if map_theta_to_g is None:
            map_theta_to_g = self.map_theta_to_g
            
        device = get_device(y, x, z, theta)
        dtype = get_dtype(y, x, z, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        matrix["V"] = self.map_theta_to_v(theta)
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
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
        
        matrix["G"] = map_theta_to_g(theta)
        
        return (matrix[r"\hat{Y}"] + (matrix["G"] @ (z.T @ matrix["V^{-1} e"]))).squeeze()

    def residual(self, y, x, z, theta, map_theta_to_g=None):
        r"""
        Compute conditional residuals including random effects.

        .. math::
            \hat{\symbf{e}} = \symbf{y} - \hat{\symbf{y}}

        where :math:`\hat{\symbf{y}}` is the conditional prediction from
        :meth:`predict`.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            z (torch.Tensor): Random-effect design matrix of shape ``(n, q)``.
            theta (torch.Tensor): Flat variance component parameter tensor.
            map_theta_to_g (callable, optional): Maps ``theta`` to
                :math:`\symbf{G}`. Defaults to :attr:`map_theta_to_g` set
                at initialisation.

        Returns:
            torch.Tensor: Conditional residuals of shape ``(n,)``.
        """
        if map_theta_to_g is None:
            map_theta_to_g = self.map_theta_to_g
            
        return y - self.predict(y, x, z, theta, map_theta_to_g)

    def loglik(self, y, x, theta):
        r"""
        Evaluate the REML log-likelihood.

        .. math::
            \ell_R(\boldsymbol{\theta}) = -\frac{1}{2} \left(
                \log |\symbf{V}(\boldsymbol{\theta})| +
                \log |\symbf{X}^\top \symbf{V}(\boldsymbol{\theta})^{-1} \symbf{X}| +
                \symbf{y}^\top \symbf{P} \symbf{y}
            \right)

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat variance component parameter tensor.

        Returns:
            torch.Tensor: Scalar REML log-likelihood value.
        """
        device = get_device(y, x, theta)
        dtype = get_dtype(y, x, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        
        matrix["V"] = self.map_theta_to_v(theta)
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
        
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

    def compute_v_dv(self, theta):
        r"""
        Compute the covariance matrix and its Jacobian.

        Calls :attr:`map_theta_to_v` to build :math:`\symbf{V}(\boldsymbol{\theta})` and either
        :attr:`map_theta_to_dv` or :func:`torch.func.jacrev` to build the
        Jacobian :math:`\partial\symbf{V}(\boldsymbol{\theta})/\partial\boldsymbol{\theta}`.

        Args:
            theta (torch.Tensor): Flat variance component parameter tensor.

        Returns:
            tuple: ``(v, dv)``, where ``v`` is the covariance matrix of
            shape ``(n, n)`` and ``dv`` is the Jacobian of shape
            ``(num_params, n, n)``.
        """
        
        v = self.map_theta_to_v(theta)
        
        if self.map_theta_to_dv is None:
            jacobian = self.jacobian_func(theta)
            dv = jacobian.permute(2, 0, 1)
        else:
            dv = self.map_theta_to_dv(theta)
    
        return v, dv

    def ai_step(self, y, x, theta, require_loglik=True, require_beta=True):
        r"""
        Perform a single average information (AI) algorithm step.

        Computes the score vector :math:`\symbf{s}`, AI matrix, and
        optionally the REML log-likelihood and fixed-effect estimate at the
        current :math:`\boldsymbol{\theta}`.

        The score vector and AI matrix are:

        .. math::
            s_k &= \frac{1}{2}\left(
                \symbf{y}^\top \symbf{P} \frac{\partial\symbf{V}(\boldsymbol{\theta})}{\partial\theta_k}
                \symbf{P} \symbf{y} -
                \mathrm{tr}\!\left(\symbf{P}
                \frac{\partial\symbf{V}(\boldsymbol{\theta})}{\partial\theta_k}\right)
            \right) \\
            \mathrm{AI}_{kj} &= \frac{1}{2} \symbf{y}^\top \symbf{P}
                \frac{\partial\symbf{V}(\boldsymbol{\theta})}{\partial\theta_k}
                \symbf{P}
                \frac{\partial\symbf{V}(\boldsymbol{\theta})}{\partial\theta_j}
                \symbf{P} \symbf{y}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat variance component parameter tensor.
            require_loglik (bool, optional): Whether to evaluate the REML
                log-likelihood. Defaults to ``True``.
            require_beta (bool, optional): Whether to compute the fixed-effect
                estimate. Defaults to ``True``.

        Returns:
            tuple: ``(beta, score, ai, loglik)``, where ``beta`` is of shape
            ``(p,)``, ``score`` is of shape ``(num_params,)``, ``ai`` is of
            shape ``(num_params, num_params)``, and ``loglik`` is a scalar
            tensor. ``beta`` and ``loglik`` are ``torch.nan`` if their
            respective ``require_*`` flag is ``False``.
        """
        device = get_device(y, x, theta)
        dtype = get_dtype(y, x, theta)
        
        matrix = {}
        vector = {}
        scalar = {}
        tensor3d = {}
        
        matrix["X"] = x
        matrix["Y"] = y.unsqueeze(-1)
        scalar["N"] = y.shape[0]
        
        matrix["V"], tensor3d["dV"] = self.compute_v_dv(theta)
        scalar["K"] = len(tensor3d["dV"])
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
        # print(theta)
        # print(matrix["V"])
        # eigvals = torch.linalg.eigvalsh(matrix["V"])
        # print(eigvals)
        # print("min eigenvalue:", eigvals.min().item())
        matrix["L"] = torch.linalg.cholesky(matrix["V"])
        
        matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
        matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])
        
        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        
        matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        
        matrix["(X^T V^{-1} X)^{-1} X^T V{-1}"] = torch.cholesky_solve(matrix["V^{-1} X"].T,
                                                                       matrix["L_{X^T V^{-1} X}"])
        
        matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] = matrix["V^{-1} X"] @ matrix["(X^T V^{-1} X)^{-1} X^T V{-1}"]
        
        matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} Y"] = matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ matrix["Y"]
        
        matrix["P Y"] = matrix["V^{-1} Y"] - matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} Y"]
        
        tensor3d["V^{-1} dV"] = torch.cholesky_solve(tensor3d["dV"], matrix["L"])
        
        tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV"] = matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ tensor3d["dV"]
        
        tensor3d["P dV"] = tensor3d["V^{-1} dV"] - tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV"]
        
        tensor3d["P dV P Y"] = tensor3d["P dV"] @ matrix["P Y"]
        
        vector["Y^T P dV P Y"] = (matrix["Y"].T @ tensor3d["P dV P Y"]).squeeze()
        
        vector["tr(P dV)"] = torch.vmap(torch.trace)(tensor3d["P dV"])
        
        # Score vector
        vector["score"] = 0.5 * (vector["Y^T P dV P Y"] - vector["tr(P dV)"])
        
        # AI matrix
        tensor3d["Y^T P dV"] = matrix["Y"].T @ tensor3d["P dV"]
        
        matrix["AI"] = 0.5 * (tensor3d["Y^T P dV"].squeeze() @ tensor3d["P dV P Y"].squeeze().T)

        if matrix["AI"].ndim != 2:
            if matrix["AI"].numel() == 1:
                matrix["AI"] = matrix["AI"].reshape(1, 1)
            else:
                raise RuntimeError("AI matrix is not a 2D tensor!")

        if matrix["AI"].shape[0] != matrix["AI"].shape[1]:
            raise RuntimeError("AI matrix is not a square matrix!")
                
        # REML log-likelihood
        if require_loglik:
            scalar["log |V|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L"])))
            scalar["log |X^T V^{-1} X|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L_{X^T V^{-1} X}"])))
            scalar["Y^T P Y"] = (matrix["Y"].T @ matrix["P Y"]).squeeze()
            scalar["loglik"] = -0.5 * (scalar["log |V|"] + scalar["log |X^T V^{-1} X|"] + scalar["Y^T P Y"])
        else:
            scalar["loglik"] = torch.nan
        
        # Beta
        if require_beta:
            vector["beta"] = (matrix["(X^T V^{-1} X)^{-1} X^T V{-1}"] @ matrix["Y"]).squeeze()
        else:
            vector["beta"] = torch.nan
    
        return vector["beta"], vector["score"], matrix["AI"], scalar["loglik"]

    def get_theta(self, select="last", history=None):
        """
        Retrieve a parameter estimate from the optimisation history.

        Args:
            select (str, optional): ``"last"`` returns the final iterate;
                any other value returns the iterate with the highest
                log-likelihood. Defaults to ``"last"``.
            history (dict, optional): History dictionary to query. Defaults
                to :attr:`history` populated by :meth:`optimize`.

        Returns:
            torch.Tensor: Selected parameter tensor :math:`\\boldsymbol{\\theta}`.
        """
        if history is None:
            history = self.history
            
        if select == "last":
            return self.history["theta"][-1]
        else:
            if torch.is_tensor(self.history["loglik"][-1]):
                index = torch.argmax(torch.stack(self.history["loglik"])).item()
                return self.history["theta"][index]
            else:
                return self.history["theta"][-1]

    def get_beta(self, select="last", history=None):
        """
        Retrieve a fixed-effect estimate from the optimisation history.

        Args:
            select (str, optional): ``"last"`` returns the final iterate;
                any other value returns the iterate with the highest
                log-likelihood. Defaults to ``"last"``.
            history (dict, optional): History dictionary to query. Defaults
                to :attr:`history` populated by :meth:`optimize`.

        Returns:
            torch.Tensor: Selected fixed-effect estimate
            :math:`\\hat{\\boldsymbol{\\beta}}`.
        """
        if history is None:
            history = self.history
            
        if select == "last":
            return self.history["beta"][-1]
        else:
            if torch.is_tensor(self.history["loglik"][-1]):
                index = torch.argmax(torch.stack(self.history["loglik"])).item()
                return self.history["beta"][index]
            else:
                return self.history["beta"][-1]

    def is_converged(self,
                     check_score=True,
                     check_delta=True,
                     check_loglik=True,
                     tol_score=1e-4,
                     tol_delta=1e-4,
                     tol_loglik=1e-4):
        """
        Check whether the optimisation has converged.

        Convergence is declared when all enabled criteria fall below their
        respective tolerances. At least two iterations must have completed
        before any criterion can be satisfied.

        Args:
            check_score (bool, optional): Check the norm of the score vector.
                Defaults to ``True``.
            check_delta (bool, optional): Check the norm of the parameter
                update :math:`\\Delta`. Defaults to ``True``.
            check_loglik (bool, optional): Check the absolute change in
                log-likelihood between successive iterates. Defaults to
                ``True``.
            tol_score (float, optional): Score norm tolerance. Defaults to
                ``1e-4``.
            tol_delta (float, optional): Parameter update norm tolerance.
                Defaults to ``1e-4``.
            tol_loglik (float, optional): Log-likelihood change tolerance.
                Defaults to ``1e-4``.

        Returns:
            bool: ``True`` if all enabled criteria are satisfied, ``False``
            otherwise.
        """
                       
        if len(self.history["score"]) < 2:
            return False
          
        if check_score:
            score_norm = torch.norm(self.history["score"][-1]).item()
            if score_norm >= tol_score:
                return False
        
        if check_delta:
            delta_norm = torch.norm(self.history["delta"][-1]).item()
            if delta_norm >= tol_delta:
                return False
              
        if check_loglik and torch.is_tensor(self.history["loglik"][-1]):
            loglik_diff = torch.abs(self.history["loglik"][-1] - self.history["loglik"][-2]).item()
            if loglik_diff >= tol_loglik:
                return False
        
        return True

    def update(self, theta, delta, eta, lb=-torch.inf, ub=torch.inf):
        r"""
        Apply a damped AI step and clip parameters to bounds.

        .. math::
            \boldsymbol{\theta} \leftarrow
            \mathrm{clip}(\boldsymbol{\theta} + \eta \Delta,\, \text{lb},\, \text{ub})

        Args:
            theta (torch.Tensor): Current parameter tensor.
            delta (torch.Tensor): AI step :math:`\Delta = \symbf{AI}^{-1}\symbf{s}`.
            eta (float): Step size (learning rate).
            lb (float, optional): Lower bound for clipping. Defaults to
                ``-inf``.
            ub (float, optional): Upper bound for clipping. Defaults to
                ``inf``.

        Returns:
            tuple: ``(theta, update)``, where ``theta`` is the updated
            parameter tensor and ``update`` is the actual change after
            clipping.
        """
        last_theta = theta
        theta = theta + delta * eta
        theta = torch.clamp(theta, min=lb, max=ub)
        return theta, theta - last_theta

    def optimize(self,
                 y,
                 x,
                 theta,
                 max_iter=200,
                 eta=1.0,
                 require_loglik=True,
                 lb=-torch.inf,
                 ub=torch.inf,
                 verbose=0,
                 check_score=True,
                 check_delta=True,
                 check_loglik=True,
                 tol_score=1e-4,
                 tol_delta=1e-4,
                 tol_loglik=1e-4):
        """
        Run the AI-REML optimisation loop.

        Iterates :meth:`ai_step` and :meth:`update` until convergence or
        ``max_iter`` is reached. Optimisation history is stored in
        :attr:`history` and can be queried afterwards via :meth:`get_theta`
        and :meth:`get_beta`.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Initial variance component parameter tensor.
            max_iter (int, optional): Maximum number of iterations. Defaults
                to ``200``.
            eta (float, optional): Step size applied to each AI update.
                Defaults to ``1.0``.
            require_loglik (bool, optional): Whether to evaluate the REML
                log-likelihood at each iteration. Defaults to ``True``.
            lb (float, optional): Lower bound for parameter clipping. Defaults
                to ``-inf``.
            ub (float, optional): Upper bound for parameter clipping. Defaults
                to ``inf``.
            verbose (int, optional): Verbosity level. ``0`` suppresses all
                output, ``1`` shows a progress bar, ``2`` additionally prints
                per-iteration diagnostics. Defaults to ``0``.
            check_score (bool, optional): Include score norm in convergence
                check. Defaults to ``True``.
            check_delta (bool, optional): Include parameter update norm in
                convergence check. Defaults to ``True``.
            check_loglik (bool, optional): Include log-likelihood change in
                convergence check. Defaults to ``True``.
            tol_score (float, optional): Score norm tolerance. Defaults to
                ``1e-4``.
            tol_delta (float, optional): Parameter update norm tolerance.
                Defaults to ``1e-4``.
            tol_loglik (float, optional): Log-likelihood change tolerance.
                Defaults to ``1e-4``.

        Returns:
            tuple: ``(theta, beta, n_iter)``, where ``theta`` is the final
            parameter estimate, ``beta`` is the corresponding fixed-effect
            estimate, and ``n_iter`` is the number of iterations completed.
        """
        self.history = {"theta": [], 
                        "beta": [], 
                        "loglik": [], 
                        "score": [], 
                        "ai": [], 
                        "delta": [],
                        "update": []}
        
        pb = tqdm(disable=not verbose, bar_format="{desc} \u23F1 {elapsed} | \u26A1 {rate_fmt}")
        
        with torch.no_grad():
            for i in range(max_iter):
                beta, score, ai, loglik = self.ai_step(y, x, theta, require_loglik=require_loglik)
                delta = torch.linalg.lstsq(ai, score.unsqueeze(-1)).solution.squeeze()
                theta, update = self.update(theta, delta, eta, lb, ub)
                
                self.history["theta"].append(theta)
                self.history["beta"].append(beta)
                self.history["loglik"].append(loglik)
                self.history["score"].append(score)
                self.history["ai"].append(ai)
                self.history["delta"].append(delta)
                self.history["update"].append(update)
                
                if verbose > 0:
                    pb.set_description(f"Iter {i + 1}")
                    pb.update(1)
                    
                    if verbose > 1:
                        write_str = f"\u2225\u2207\u2225: {torch.norm(score):12.4f}, \u2225\u0394\u2225: {torch.norm(delta):6.4f}, \u03B7: {eta:.2f}, \u2225\u0394\u1D9C\u2225: {torch.norm(update):6.4f}"
                      
                        if require_loglik:
                            if len(self.history["loglik"]) > 1:
                                delta_loglik = self.history["loglik"][-1].item() - self.history["loglik"][-2].item()
                                write_str += f", log \U0001D4DB: {loglik:8.4f} ({delta_loglik:+.4f})"
                            else:
                                write_str += f", log \U0001D4DB: {loglik:8.4f}"
                        
                        if i == 0:
                            tqdm.write("")
                        tqdm.write(write_str)
                
                if self.is_converged(check_score, check_delta, check_loglik, tol_score, tol_delta, tol_loglik):
                    if verbose > 0:
                        if verbose > 1:
                            tqdm.write(f"\n[\u2207: score, \u0394: \U0001D409\u207B\u00B9\u2207, \u03B7: learning rate, \u0394\u1D9C: clip(\U0001D6C9 + \u03B7\u0394, lb, ub) - \U0001D6C9, \U0001D4DB: restricted likelihood]")
                        tqdm.write(f"\n\u2713 Converged at iteration {i + 1}")
                    break
        
        pb.close()
        
        return theta, beta, i + 1
  

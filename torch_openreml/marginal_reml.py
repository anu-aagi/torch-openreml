"""
Restricted maximum likelihood (REML) estimation for parametric marginal covariance matrices.

This module implements REML estimation for generalised least squares (GLS)
with a parametric marginal covariance matrix :math:`\\symbf{V}(\\boldsymbol{\\theta})`,
estimated via the average information (AI) algorithm.

Classes:
    MarginalREML:
        MarginalREML estimator with AI-based optimisation, supporting
        coefficient estimation via BLUE and prediction.
"""

import torch
from torch_openreml.utils import get_device, get_dtype
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.post import blue, loglik, marginal_predict, marginal_residual
from tqdm import tqdm

class MarginalREML:
    r"""
    Marginal REML estimator for generalised least squares with a
    parametric marginal covariance matrix.

    Fits covariance parameters :math:`\boldsymbol{\theta}` by maximising the
    restricted log-likelihood

    .. math::
        \ell_R(\boldsymbol{\theta}) = -\frac{1}{2} \left(
            \log |\symbf{V}(\boldsymbol{\theta})| +
            \log |\symbf{X}^\top \symbf{V}(\boldsymbol{\theta})^{-1} \symbf{X}| +
            \symbf{y}^\top \symbf{P} \symbf{y}
        \right)

    where :math:`\symbf{P} = \symbf{V}(\boldsymbol{\theta})^{-1} - \symbf{V}(\boldsymbol{\theta})^{-1}\symbf{X}
    (\symbf{X}^\top\symbf{V}(\boldsymbol{\theta})^{-1}\symbf{X})^{-1}\symbf{X}^\top\symbf{V}(\boldsymbol{\theta})^{-1}`
    is the projection matrix and :math:`\symbf{V}(\boldsymbol{\theta})` is the
    marginal covariance matrix of :math:`\symbf{y}`.

    Optimisation uses the average information (AI)
    algorithm, which forms a quasi-Newton step
    :math:`\Delta = \symbf{AI}^{-1} \symbf{s}` from the score vector
    :math:`\symbf{s}` and the AI matrix at each iteration.

    The covariance model :math:`\symbf{V}(\boldsymbol{\theta})` is supplied
    as a :class:`~torch_openreml.covariance.matrix.Matrix` instance via
    ``v``. Gradients are handled internally by the matrix.
    """
    
    def __init__(self, v):
        """
        Initialize a MarginalREML estimator.

        Args:
            v (Matrix): A :class:`~torch_openreml.covariance.matrix.Matrix`
                instance that constructs :math:`\\symbf{V}(\\boldsymbol{\\theta})`
                and its Jacobian.

        Raises:
            TypeError: If ``v`` is not a
                :class:`~torch_openreml.covariance.matrix.Matrix` instance.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml import MarginalREML
            from torch_openreml.covariance import ScalarMatrix

            n, p = 50, 2
            y = torch.randn(n)
            x = torch.randn(n, p)
            theta = torch.tensor([0.0])

            mat = ScalarMatrix(n)
            reml = MarginalREML(mat)
            theta_hat, beta_hat, n_iter = reml.optimize(y, x, theta, verbose=2)
            theta_hat, beta_hat
        """

        if not isinstance(v, Matrix):
            raise TypeError("'v' must be a Matrix instance!")

        self.v = v

    def blue(self, y, x, theta):
        r"""
        Compute the best linear unbiased estimator (BLUE) of :math:`\boldsymbol{\beta}`.

        Solves for :math:`\hat{\boldsymbol{\beta}}` via generalised least squares:

        .. math::
            \hat{\boldsymbol{\beta}} = (\symbf{X}^\top \symbf{V}(\boldsymbol{\theta})^{-1} \symbf{X})^{-1}
            \symbf{X}^\top \symbf{V}(\boldsymbol{\theta})^{-1} \symbf{y}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat covariance parameter tensor.

        Returns:
            torch.Tensor: Coefficient estimate :math:`\hat{\boldsymbol{\beta}}`
            of shape ``(p,)``.
        """
        return blue(y, x, self.v(theta))

    def predict(self, y, x, theta):
        r"""
        Compute fitted values.

        .. math::
            \hat{\symbf{y}} = \symbf{X} \hat{\boldsymbol{\beta}}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat covariance parameter tensor.

        Returns:
            torch.Tensor: Fitted values of shape ``(n,)``.
        """
        return marginal_predict(y, x, self.v(theta))

    def residual(self, y, x, theta):
        r"""
        Compute residuals.

        .. math::
            \hat{\symbf{e}} = \symbf{y} - \symbf{X}\hat{\boldsymbol{\beta}}

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat covariance parameter tensor.

        Returns:
            torch.Tensor: Residuals of shape ``(n,)``.
        """
        return marginal_residual(y, x, self.v(theta))

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
            theta (torch.Tensor): Flat covariance parameter tensor.

        Returns:
            torch.Tensor: Scalar REML log-likelihood value.
        """
        return loglik(y, x, self.v(theta))

    def compute_v_dv(self, theta):
        r"""
        Compute the covariance matrix and its Jacobian.

        Calls :attr:`v` to build :math:`\symbf{V}(\boldsymbol{\theta})`
        and :meth:`~torch_openreml.covariance.matrix.Matrix.grad` for the
        Jacobian :math:`\partial\symbf{V}(\boldsymbol{\theta})/\partial\boldsymbol{\theta}`.

        Args:
            theta (torch.Tensor): Flat covariance parameter tensor.

        Returns:
            tuple: ``(v, dv)``, where ``v`` is the covariance matrix of
            shape ``(n, n)`` and ``dv`` is the Jacobian of shape
            ``(num_params, n, n)``.
        """
        
        v = self.v(theta)
        dv, _ = self.v.grad(theta)
        return v, dv

    def ai_step(self, y, x, theta, require_loglik=True, require_beta=True, trace_approx=False, subspace_fraction=0.01):
        r"""
        Perform a single average information (AI) algorithm step.

        Computes the score vector :math:`\symbf{s}`, AI matrix, and
        optionally the REML log-likelihood and coefficient estimate at the
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

        When ``trace_approx`` is ``True``, the trace term
        :math:`\mathrm{tr}(\symbf{P}\,\partial\symbf{V}/\partial\theta_k)`
        is replaced by a stochastic probe estimate, which avoids forming
        :math:`\symbf{V}^{-1}\partial\symbf{V}` in full. With

        .. math::
            \symbf{A}_k = \symbf{V}^{-1}
                \frac{\partial \symbf{V}}{\partial \theta_k},
            \qquad
            \symbf{B}_k = \symbf{V}^{-1} \symbf{X}
                (\symbf{X}^\top \symbf{V}^{-1} \symbf{X})^{-1}
                \symbf{X}^\top \symbf{V}^{-1}
                \frac{\partial \symbf{V}}{\partial \theta_k},

        the trace splits as

        .. math::
            \mathrm{tr}\!\left(\symbf{P}
                \frac{\partial \symbf{V}}{\partial \theta_k}\right)
            = \mathrm{tr}(\symbf{A}_k) - \mathrm{tr}(\symbf{B}_k),

        and each of the two terms is estimated on its own, so that neither
        :math:`\symbf{A}_k` nor :math:`\symbf{B}_k` is ever formed in full.

        The estimator is Hutch++ (Meyer, Musco, Musco & Woodruff,
        *Hutch++: Optimal stochastic trace estimation*, SOSA 2021,
        pp. 142--155, `doi:10.1137/1.9781611976496.16
        <https://doi.org/10.1137/1.9781611976496.16>`_). For a matrix
        :math:`\symbf{C}`, two independent Rademacher probe matrices
        :math:`\symbf{S}, \symbf{G} \in \{\pm 1\}^{n \times m}` are drawn
        once per call and shared across parameters, with
        :math:`m = \lfloor n f \rfloor + 1` probes each and :math:`f` the
        ``subspace_fraction``, and

        .. math::
            \symbf{Q} &= \mathrm{qr}(\symbf{C} \symbf{S}), \\
            \hat{h}(\symbf{C}) &= \mathrm{tr}(\symbf{Q}^\top \symbf{C}
                \symbf{Q})
                + \frac{1}{m} \mathrm{tr}\!\left( \symbf{G}^\top
                (\symbf{I} - \symbf{Q}\symbf{Q}^\top) \symbf{C}
                (\symbf{I} - \symbf{Q}\symbf{Q}^\top) \symbf{G} \right).

        :math:`\symbf{Q}` spans the dominant range of :math:`\symbf{C}`, so
        the first term is exact there; the second applies Hutchinson's
        estimator to the deflated residual.

        When ``trace_approx`` is ``False`` the traces are evaluated
        exactly. The average information matrix and the log-likelihood are
        computed exactly either way; only the trace term of the score is
        approximated.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Flat covariance parameter tensor.
            require_loglik (bool, optional): Whether to evaluate the REML
                log-likelihood. Defaults to ``True``.
            require_beta (bool, optional): Whether to compute the coefficient
                estimate. Defaults to ``True``.
            trace_approx (bool, optional): Whether to estimate the trace term of
                the score stochastically. ``False`` computes the exact
                score, which is more expensive per call. This is a plain
                switch with no internal state:
                :meth:`optimize` owns the decision of when to turn it off.
                Defaults to ``False``.
            subspace_fraction (float, optional): Fraction of the ``n``
                observations used as the dimension of the random probe
                subspace in the stochastic trace estimate. Each of the two
                probe matrices holds ``int(n * subspace_fraction) + 1``
                vectors, so the total probe budget is about twice that.
                Must lie in ``[0, 1]``; larger values trade time for
                accuracy of the approximate score. Only used when
                ``trace_approx`` is ``True``. Defaults to ``0.01``.

        Raises:
            ValueError: If ``subspace_fraction`` is outside ``[0, 1]``.

        Returns:
            tuple: ``(beta, score, ai, loglik)``, where ``beta`` is of shape
            ``(p,)``, ``score`` is of shape ``(num_params,)``, ``ai`` is of
            shape ``(num_params, num_params)``, and ``loglik`` is a scalar
            tensor. ``beta`` and ``loglik`` are ``torch.nan`` if their
            respective ``require_*`` flag is ``False``.
        """
        device = get_device(y, x, theta)
        dtype = get_dtype(y, x, theta)

        if not 0.0 <= subspace_fraction <= 1.0:
            raise ValueError(f"subspace_fraction must be between 0 and 1, got {subspace_fraction}.")
        
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

        if trace_approx:
            subspace_dim = int(scalar["N"] * subspace_fraction) + 1
            scalar["m / 3"] = subspace_dim

            matrix["S"] = torch.randint(0, 2, (scalar["N"], scalar["m / 3"]), dtype=dtype, device=device) * 2 - 1.0
            matrix["G"] = torch.randint(0, 2, (scalar["N"], scalar["m / 3"]), dtype=dtype, device=device) * 2 - 1.0

            tensor3d["dV S"] = tensor3d["dV"] @ matrix["S"]
            tensor3d["V^{-1} dV S"] = torch.cholesky_solve(tensor3d["dV S"], matrix["L"])
            tensor3d["Q"], tensor3d["R"] = torch.linalg.qr(tensor3d["V^{-1} dV S"])
            tensor3d["(I - Q Q^T) G"] = matrix["G"] - tensor3d["Q"] @ (tensor3d["Q"].mT @ matrix["G"])
            tensor3d["dV (I - Q Q^T) G"] = tensor3d["dV"] @ tensor3d["(I - Q Q^T) G"]
            tensor3d["V^{-1} dV (I - Q Q^T) G"] = torch.cholesky_solve(tensor3d["dV (I - Q Q^T) G"], matrix["L"])
            tensor3d["G^T (I - Q Q^T)"] = tensor3d["(I - Q Q^T) G"].mT
            tensor3d["G^T (I - Q Q^T) (V^{-1} dV) (I - Q Q^T) G"] = tensor3d["G^T (I - Q Q^T)"] @ tensor3d["V^{-1} dV (I - Q Q^T) G"]
            tensor3d["Q^T (V^{-1} dV) Q"] = tensor3d["Q"].mT @ torch.cholesky_solve(tensor3d["dV"] @ tensor3d["Q"], matrix["L"])
            vector["tr(V^{-1} dV)"] = torch.vmap(torch.trace)(tensor3d["Q^T (V^{-1} dV) Q"]) + 1.0 / scalar["m / 3"] * torch.vmap(torch.trace)(tensor3d["G^T (I - Q Q^T) (V^{-1} dV) (I - Q Q^T) G"])

            tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV S"] = matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ tensor3d["dV S"]
            tensor3d["Q"], tensor3d["R"] = torch.linalg.qr(tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV S"])
            tensor3d["(I - Q Q^T) G"] = matrix["G"] - tensor3d["Q"] @ (tensor3d["Q"].mT @ matrix["G"])
            tensor3d["dV (I - Q Q^T) G"] = tensor3d["dV"] @ tensor3d["(I - Q Q^T) G"]
            tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV (I - Q Q^T) G"] = matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ tensor3d["dV (I - Q Q^T) G"]
            tensor3d["G^T (I - Q Q^T)"] = matrix["G"].T - (matrix["G"].T @ tensor3d["Q"]) @ tensor3d["Q"].mT
            tensor3d["G^T (I - Q Q^T) (V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV) (I - Q Q^T) G"] = tensor3d["G^T (I - Q Q^T)"] @ tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV (I - Q Q^T) G"]
            tensor3d["Q^T (V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV) Q"] = (tensor3d["Q"].mT @ matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"]) @ (tensor3d["dV"] @ tensor3d["Q"])
            vector["tr(V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV)"] = torch.vmap(torch.trace)(tensor3d["Q^T (V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV) Q"]) + 1.0 / scalar["m / 3"] * torch.vmap(torch.trace)(tensor3d["G^T (I - Q Q^T) (V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV) (I - Q Q^T) G"])
            vector["tr(P dV)"] = vector["tr(V^{-1} dV)"] - vector["tr(V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV)"]

            tensor3d["dV P Y"] = tensor3d["dV"] @ matrix["P Y"]
            vector["Y^T P dV P Y"] = (matrix["P Y"].T @ tensor3d["dV P Y"]).squeeze()

            # Score vector
            vector["score"] = 0.5 * (vector["Y^T P dV P Y"] - vector["tr(P dV)"])

            # AI matrix
            tensor3d["Y^T P dV"] = matrix["P Y"].T @ tensor3d["dV"]

            tensor3d["V^{-1} dV P Y"] = torch.cholesky_solve(tensor3d["dV P Y"], matrix["L"])
            tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV P Y"] = matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ tensor3d["dV P Y"]
            tensor3d["P dV P Y"] = tensor3d["V^{-1} dV P Y"] - tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV P Y"]

            matrix["AI"] = 0.5 * (tensor3d["Y^T P dV"].squeeze() @ tensor3d["P dV P Y"].squeeze().T)
        else:
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
        Retrieve a covariance parameter estimate from the optimisation history.

        Args:
            select (str, optional): ``"last"`` returns the final iterate;
                any other value returns the iterate with the highest
                log-likelihood. Defaults to ``"last"``.
            history (dict, optional): History dictionary to query. Defaults
                to :attr:`history` populated by :meth:`optimize`.

        Returns:
            torch.Tensor: Selected covariance parameter tensor
            :math:`\\boldsymbol{\\theta}`.
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
        Retrieve a coefficient estimate from the optimisation history.

        Args:
            select (str, optional): ``"last"`` returns the final iterate;
                any other value returns the iterate with the highest
                log-likelihood. Defaults to ``"last"``.
            history (dict, optional): History dictionary to query. Defaults
                to :attr:`history` populated by :meth:`optimize`.

        Returns:
            torch.Tensor: Selected coefficient estimate
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
            check_delta (bool, optional): Check the norm of the covariance
                parameter update :math:`\\Delta`. Defaults to ``True``.
            check_loglik (bool, optional): Check the absolute change in
                log-likelihood between successive iterates. Defaults to
                ``True``.
            tol_score (float, optional): Score norm tolerance. Defaults to
                ``1e-4``.
            tol_delta (float, optional): Covariance parameter update norm
                tolerance. Defaults to ``1e-4``.
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
        Apply a damped AI step and clip covariance parameters to bounds.

        .. math::
            \boldsymbol{\theta} \leftarrow
            \mathrm{clip}(\boldsymbol{\theta} + \eta \Delta,\, \text{lb},\, \text{ub})

        Args:
            theta (torch.Tensor): Current covariance parameter tensor.
            delta (torch.Tensor): AI step :math:`\Delta = \symbf{AI}^{-1}\symbf{s}`.
            eta (float): Step size (learning rate).
            lb (float, optional): Lower bound for clipping. Defaults to
                ``-inf``.
            ub (float, optional): Upper bound for clipping. Defaults to
                ``inf``.

        Returns:
            tuple: ``(theta, update)``, where ``theta`` is the updated
            covariance parameter tensor and ``update`` is the actual change
            after clipping.
        """
        last_theta = theta
        theta = theta + delta * eta
        theta = torch.clamp(theta, min=lb, max=ub)
        return theta, theta - last_theta

    def optimize(self,
                 y,
                 x,
                 theta=None,
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
                 tol_loglik=1e-4,
                 trace_approx=False,
                 subspace_fraction=0.01,
                 exact_score_threshold=10.0):
        r"""
        Run the AI-REML optimisation loop.

        Iterates :meth:`ai_step` and :meth:`update` until convergence or
        ``max_iter`` is reached. Optimisation history is stored in
        :attr:`history` and can be queried afterwards via :meth:`get_theta`
        and :meth:`get_beta`.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Design matrix of shape ``(n, p)``.
            theta (torch.Tensor): Initial covariance parameter tensor.
            max_iter (int, optional): Maximum number of iterations. Defaults
                to ``200``.
            eta (float, optional): Step size applied to each AI update.
                Defaults to ``1.0``.
            require_loglik (bool, optional): Whether to evaluate the REML
                log-likelihood at each iteration. Defaults to ``True``.
            lb (float, optional): Lower bound for covariance parameter clipping. Defaults
                to ``-inf``.
            ub (float, optional): Upper bound for covariance parameter clipping. Defaults
                to ``inf``.
            verbose (int, optional): Verbosity level. ``0`` suppresses all
                output, ``1`` shows a progress bar, ``2`` additionally prints
                per-iteration diagnostics. Defaults to ``0``.
            check_score (bool, optional): Include score norm in convergence
                check. Defaults to ``True``.
            check_delta (bool, optional): Include covariance parameter update
                norm in convergence check. Defaults to ``True``.
            check_loglik (bool, optional): Include log-likelihood change in
                convergence check. Defaults to ``True``.
            tol_score (float, optional): Score norm tolerance. Defaults to
                ``1e-4``.
            tol_delta (float, optional): Covariance parameter update norm
                tolerance. Defaults to ``1e-4``.
            tol_loglik (float, optional): Log-likelihood change tolerance.
                Defaults to ``1e-4``.
            trace_approx (bool, optional): Whether to start the optimisation with
                the stochastic probe estimate of :meth:`ai_step` for the
                score, which follows Hutch++ (Meyer, Musco, Musco & Woodruff,
                *Hutch++: Optimal stochastic trace estimation*, SOSA 2021,
                pp. 142--155, `doi:10.1137/1.9781611976496.16
                <https://doi.org/10.1137/1.9781611976496.16>`_). While it is
                on, the optimiser monitors the score norm, and once that norm
                drops below ``exact_score_threshold`` the exact score is used
                for that and every remaining iteration. The probe estimate has
                a noise floor that does not decay, so an optimisation left on
                the approximation cannot satisfy ``tol_score`` and will run to
                ``max_iter``; the switch is what makes the approximation
                usable. Defaults to ``False``, which runs the whole
                optimisation with the exact score.
            subspace_fraction (float, optional): Fraction of the ``n``
                observations used as the dimension of the random probe
                subspace in the stochastic trace estimate of :meth:`ai_step`.
                Must lie in ``[0, 1]``; larger values trade time for accuracy
                of the score on iterations that are still far from the
                optimum. Defaults to ``0.01``.
            exact_score_threshold (float, optional): Score norm below which
                the stochastic probe estimate is switched off for the
                remainder of the optimisation. Since the first score is
                always computed with the current setting, ``inf`` switches off
                after the first iteration while ``0.0`` keeps the
                approximation on throughout. Defaults to ``10.0``.

        Returns:
            tuple: ``(theta, beta, n_iter)``, where ``theta`` is the final
            covariance parameter estimate, ``beta`` is the corresponding coefficient
            estimate, and ``n_iter`` is the number of iterations completed.
        """
        if theta is None:
            theta = self.v.build_params(include_fixed=False, trans=False)

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
                beta, score, ai, loglik = self.ai_step(y, x, theta,
                                                       require_loglik=require_loglik,
                                                       subspace_fraction=subspace_fraction,
                                                       trace_approx=trace_approx)

                if trace_approx and torch.norm(score) < exact_score_threshold:
                    trace_approx = False

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
  

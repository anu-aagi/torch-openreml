r"""Marginal REML score and average information for a supplied covariance.

This module implements a single average-information (AI) REML step when
the marginal covariance :math:`\symbf{V}` and its derivatives are already
available.  The model is

.. math::

    \symbf{y}
    = \symbf{X}\symbf{\beta} + \symbf{\epsilon},
    \qquad
    \symbf{\epsilon} \sim N(\symbf{0}, \symbf{V}),

where :math:`\symbf{V}` includes all random effects and the residual
covariance.  No random-effect incidence matrix is needed to evaluate the
score or the AI matrix.

The REML projection matrix is

.. math::

    \symbf{P}
    = \symbf{V}^{-1}
      - \symbf{V}^{-1}\symbf{X}
        \left(\symbf{X}^\top\symbf{V}^{-1}\symbf{X}\right)^{-1}
        \symbf{X}^\top\symbf{V}^{-1},

and the score and AI matrix are

.. math::

    s_k
    = \frac{1}{2}\left[
        \symbf{y}^\top\symbf{P}
        \frac{\partial\symbf{V}}{\partial\theta_k}
        \symbf{P}\symbf{y}
        - \operatorname{tr}
          \left(\symbf{P}
          \frac{\partial\symbf{V}}{\partial\theta_k}\right)
      \right],
    \qquad
    \symbf{I}_{kl}
    = \frac{1}{2}
      \symbf{q}_k^\top\symbf{P}\symbf{q}_l,

where
:math:`\symbf{q}_k = (\partial\symbf{V}/\partial\theta_k)\symbf{P}\symbf{y}`.

The restricted log-likelihood is

.. math::

    \ell
    = -\frac{1}{2}\left(
        \log|\symbf{V}|
        + \log|\symbf{X}^\top\symbf{V}^{-1}\symbf{X}|
        + \symbf{y}^\top\symbf{P}\symbf{y}
      \right).

The trace term of the score can be evaluated exactly
(``trace_approx=False``), estimated with Hutch++ (``trace_approx=True``),
or started approximately and switched to exact once the previous score
norm is small (``trace_approx="auto"``).  Only the trace is approximated;
the AI matrix and the log-likelihood remain exact.  The projection matrix
:math:`\symbf{P}` is still formed.

Classes:
    Marginal
"""

import torch


class Marginal:
    r"""Marginal REML score and AI for a supplied :math:`\symbf{V}`.

    :meth:`ai_step` is the model-aware entry point.  It extracts
    :math:`\symbf{V}` and its derivatives from the model, reuses a closed-form
    inverse when available, and otherwise applies the residual jitter rule.

    :meth:`ai_step_from_tensors` is the stateless tensor-level entry point.
    It takes the marginal covariance :math:`\symbf{V}` and its derivatives
    directly and dispatches to the no-inverse or with-inverse kernel.  The
    no-inverse kernel forms :math:`\symbf{V}` through Cholesky solves and
    never materialises :math:`\symbf{V}^{-1}`; the with-inverse kernel
    reuses a supplied :math:`\symbf{V}^{-1}` and forms
    :math:`\symbf{P}` explicitly.
    """

    #: Minimum ``n`` for ``trace_approx="auto"`` to start with the stochastic
    #: trace estimate.  Below this size exact traces are cheaper and more stable.
    MIN_N_FOR_TRACE = 100

    def __init__(self, trace_approx=False, subspace_fraction=0.01, exact_score_threshold=10.0):
        r"""Configure the marginal method.

        Args:
            trace_approx: ``False`` for exact traces, ``True`` to always
                estimate them, or ``"auto"`` to start with the estimate
                for large problems and switch to exact traces once the
                previous score norm falls below
                ``exact_score_threshold``.
            subspace_fraction: Probe-subspace fraction for Hutch++.
            exact_score_threshold: Score-norm threshold below which
                ``"auto"`` switches to exact traces.

        Raises:
            ValueError: If an option is outside its valid range.
        """
        if trace_approx not in (False, True, "auto"):
            raise ValueError("`trace_approx` must be False, True or 'auto'.")
        if not 0.0 <= subspace_fraction <= 1.0:
            raise ValueError("`subspace_fraction` must lie in [0, 1].")
        if exact_score_threshold < 0:
            raise ValueError("`exact_score_threshold` must be non-negative.")

        self.trace_approx = trace_approx
        self.subspace_fraction = subspace_fraction
        self.exact_score_threshold = exact_score_threshold
        self.reset()

    def reset(self):
        """Reset the score history used by ``trace_approx='auto'``."""
        self._last_score_norm = None

    def loglik(self, model, y, x, free_params):
        r"""Evaluate the restricted log-likelihood for ``model``.

        This evaluates

        .. math::

            \ell
            = -\frac{1}{2}\left(
                \log|\symbf{V}|
                + \log|\symbf{X}^\top\symbf{V}^{-1}\symbf{X}|
                + \symbf{y}^\top\symbf{P}\symbf{y}
              \right),

        without forming the full projection matrix :math:`\symbf{P}`.

        Args:
            model (MixedModelREML): Model supplying :math:`\symbf{V}`.
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Free parameters of the model.

        Returns:
            torch.Tensor: Scalar restricted log-likelihood.

        Raises:
            ValueError: If ``model`` is ``None``.
        """
        if model is None:
            raise ValueError("`model` must not be None.")

        data = self.preprocess_input(model, y, x, free_params, grad=False)
        return self.loglik_from_tensors(**data)

    def loglik_from_tensors(self, x, y, v, v_inv=None):
        r"""Evaluate the marginal restricted log-likelihood from tensors.

        Args:
            x: Fixed-effect design matrix of shape ``(n, p)``.
            y: Response vector of shape ``(n,)``.
            v: Marginal covariance matrix of shape ``(n, n)``.
            v_inv: Optional precomputed inverse of ``v``.  When supplied it is
            used instead of forming ``V^{-1}`` from a Cholesky factorisation.

        Returns:
            torch.Tensor: Scalar restricted log-likelihood.
        """
        matrix = {"X": x, "Y": y.unsqueeze(-1), "V": v}
        scalar = {}
        vector = {}

        matrix["L_V"] = torch.linalg.cholesky(matrix["V"])
        if v_inv is None:
            matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L_V"])
            matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L_V"])
        else:
            matrix["V^{-1} Y"] = v_inv @ matrix["Y"]
            matrix["V^{-1} X"] = v_inv @ matrix["X"]

        scalar["logdet_V"] = 2.0 * torch.log(torch.diagonal(matrix["L_V"])).sum()

        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        matrix["L_X"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        vector["X^T V^{-1} y"] = matrix["X"].T @ matrix["V^{-1} Y"]
        vector["beta"] = torch.cholesky_solve(vector["X^T V^{-1} y"], matrix["L_X"])

        scalar["quad"] = (matrix["Y"].T @ matrix["V^{-1} Y"]).squeeze() - (vector["X^T V^{-1} y"].T @ vector["beta"]).squeeze()
        scalar["logdet_X"] = 2.0 * torch.log(torch.diagonal(matrix["L_X"])).sum()
        scalar["loglik"] = -0.5 * (scalar["logdet_V"] + scalar["logdet_X"] + scalar["quad"])
        return scalar["loglik"]

    def preprocess_input(self, model, y, x, free_params, grad=True):
        r"""Prepare :math:`\symbf{V}` and its derivatives for ``model``.

        Args:
            model (MixedModelREML): Model supplying :math:`\symbf{V}`.
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Free parameters of the model.
            grad (bool, optional): Whether to compute ``dv``.  Set
                to ``False`` for loglik-only evaluation.  Default: ``True``.

        Returns:
            dict: Prepared tensors with keys ``x``, ``y``, ``v`` and
            ``v_inv``.  ``v_inv`` is ``None`` when the model has no
            closed-form inverse; in that case ``v`` already includes the
            ``1e-6 I`` jitter.  The derivative key ``dv`` is included only
            when ``grad`` is ``True``.

        Raises:
            ValueError: If ``model`` is ``None``.
        """

        y = y.to(device=free_params.device, dtype=free_params.dtype)
        x = x.to(device=free_params.device, dtype=free_params.dtype)

        v = model.V(free_params)

        v_inv = None
        try:
            v_inv = model.V.manual_inverse(free_params)
        except NotImplementedError:
            pass

        if v_inv is None:
            n = v.shape[0]
            v = v + 1e-6 * torch.eye(n, dtype=v.dtype, device=v.device)

        data = {"x": x, "y": y, "v": v, "v_inv": v_inv}
        if grad:
            data["dv"] = model.V.grad(free_params)[0]
        return data

    def ai_step(self, model, y, x, free_params, require_loglik=True):
        r"""Evaluate one marginal AI step for ``model``.

        The score and AI matrix are

        .. math::

            s_k
            = \frac{1}{2}\left[
                \symbf{q}_k^\top\symbf{P}\symbf{y}
                - \operatorname{tr}
                  \left(\symbf{P}\dot{\symbf{V}}_k\right)
              \right],
            \qquad
            \symbf{I}_{kl}
            = \frac{1}{2}\symbf{q}_k^\top\symbf{P}\symbf{q}_l,

        with :math:`\symbf{q}_k = \dot{\symbf{V}}_k\symbf{P}\symbf{y}`.

        ``trace_approx`` is resolved here:

        - ``True`` always uses the stochastic trace estimate;
        - ``False`` always uses the exact trace;
        - ``"auto"`` starts exact for small problems or after the previous
          score norm fell below ``exact_score_threshold``, and otherwise
          uses the estimate.

        Args:
            model (MixedModelREML): Model supplying :math:`\symbf{V}`.
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Free parameters of the model.
            require_loglik (bool, optional): Whether to evaluate the REML
                log-likelihood.  Default: ``True``.

        Returns:
            tuple: ``(score, ai, loglik)``.  ``loglik`` is ``torch.nan`` when
            ``require_loglik`` is ``False``.

        Raises:
            ValueError: If ``model`` is ``None``.
        """
        n = y.shape[0]
        tensor_dict = self.preprocess_input(model, y, x, free_params)

        if self.trace_approx is True:
            score, ai, loglik = self.ai_step_from_tensors(**tensor_dict, require_loglik=require_loglik, trace_approx=True, subspace_fraction=self.subspace_fraction)

        if self.trace_approx is False:
            score, ai, loglik = self.ai_step_from_tensors(**tensor_dict, require_loglik=require_loglik, trace_approx=False, subspace_fraction=self.subspace_fraction)

        if self.trace_approx == "auto":
            if n >= self.MIN_N_FOR_TRACE and (self._last_score_norm is None or self._last_score_norm >= self.exact_score_threshold):
                score, ai, loglik = self.ai_step_from_tensors(**tensor_dict, require_loglik=require_loglik, trace_approx=True, subspace_fraction=self.subspace_fraction)
            else:
                score, ai, loglik = self.ai_step_from_tensors(**tensor_dict, require_loglik=require_loglik, trace_approx=False, subspace_fraction=self.subspace_fraction)

        self._last_score_norm = torch.norm(score)

        return score, ai, loglik

    @torch.no_grad()
    def ai_step_from_tensors(self, x, y, v, dv, v_inv=None, require_loglik=True, trace_approx=False, subspace_fraction=0.01):
        r"""Dispatch one marginal REML score and AI step.

        If ``v_inv`` is ``None`` this delegates to
        :meth:`_ai_step_from_tensors_no_inverse`, which factors ``v`` and
        avoids forming :math:`\symbf{V}^{-1}`.  Otherwise it delegates to
        :meth:`_ai_step_from_tensors_with_inverse`, which reuses the supplied
        inverse and forms :math:`\symbf{P}` explicitly.  The two branches use
        different stochastic trace estimators, so their approximate scores
        differ slightly; their exact scores agree.

        Args:
            x: Fixed-effect design matrix of shape ``(n, p)``.
            y: Response vector of shape ``(n,)``.
            v: Marginal covariance matrix of shape ``(n, n)``.
            dv: Derivatives of ``v`` with shape ``(k, n, n)``.  The
                first dimension is the parameter dimension.
            v_inv: Optional precomputed inverse of ``v``.  If ``None``,
                the no-inverse branch factors ``v`` and uses Cholesky solves.
                A supplied value selects the with-inverse branch.
            require_loglik: If ``True``, return the restricted
                log-likelihood.  If ``False``, return ``torch.nan`` and skip
                the log-determinant calculations.
            trace_approx: If ``True``, estimate the trace term of the
                score with Hutch++.  The AI matrix and log-likelihood
                remain exact.  If ``False``, evaluate the trace exactly.
            subspace_fraction: Fraction of the ``n`` observations used as
                the probe-subspace dimension in the stochastic trace
                estimate.  Each of the two probe matrices holds
                ``int(n * subspace_fraction) + 1`` vectors.  Must lie in
                ``[0, 1]``; only used when ``trace_approx`` is ``True``.
                Default: ``0.01``.

        Raises:
            ValueError: If ``dv`` has a zero-length parameter dimension,
                or if ``subspace_fraction`` is outside ``[0, 1]``.

        Returns:
            tuple: ``(score, ai, loglik)``, where ``score`` has shape
            ``(k,)``, ``ai`` has shape ``(k, k)``, and ``loglik`` is a
            scalar tensor (``torch.nan`` when ``require_loglik`` is
            ``False``).
        """

        if v_inv is None:
            return self._ai_step_from_tensors_no_inverse(x, y, v, dv, require_loglik=require_loglik, trace_approx=trace_approx, subspace_fraction=subspace_fraction,)
        return self._ai_step_from_tensors_with_inverse(x, y, v, dv, v_inv, require_loglik=require_loglik, trace_approx=trace_approx, subspace_fraction=subspace_fraction,)

    def _ai_step_from_tensors_no_inverse(self, x, y, v, dv, require_loglik=True, trace_approx=False, subspace_fraction=0.01):
        r"""Compute one marginal REML score and AI step without forming ``v^{-1}``.

        This branch factors ``v`` once and applies :math:`\symbf{V}^{-1}`
        and the projection through Cholesky solves.  The exact trace uses
        :math:`\operatorname{tr}(\symbf{V}^{-1}\dot{\symbf{V}})
        - \operatorname{tr}(\symbf{H}\dot{\symbf{V}})`; the approximate
        branch estimates the two terms separately with the two-term Hutch++
        estimator used by the legacy marginal implementation.

        Args:
            x: Fixed-effect design matrix of shape ``(n, p)``.
            y: Response vector of shape ``(n,)``.
            v: Marginal covariance matrix of shape ``(n, n)``.
            dv: Derivatives of ``v`` with shape ``(k, n, n)``.  The
                first dimension is the parameter dimension.
            require_loglik: If ``True``, return the restricted
                log-likelihood.  If ``False``, return ``torch.nan`` and skip
                the log-determinant calculations.
            trace_approx: If ``True``, estimate the trace term with the
                two-term Hutch++ estimator.  The AI matrix and
                log-likelihood remain exact.
            subspace_fraction: Probe-subspace fraction for Hutch++.
                Must lie in ``[0, 1]``; only used when ``trace_approx`` is
                ``True``.  Default: ``0.01``.

        Raises:
            ValueError: If ``dv`` has a zero-length parameter dimension,
                or if ``subspace_fraction`` is outside ``[0, 1]``.

        Returns:
            tuple: ``(score, ai, loglik)``.  ``score`` has shape ``(k,)``,
            ``ai`` has shape ``(k, k)``, and ``loglik`` is a scalar tensor
            (``torch.nan`` when ``require_loglik`` is ``False``).
        """
        if dv.shape[0] == 0:
            raise ValueError("dv must have at least one parameter; got dv.shape[0] == 0.")
        if not 0.0 <= subspace_fraction <= 1.0:
            raise ValueError("subspace_fraction must lie in [0, 1].")

        matrix = {}
        scalar = {}
        vector = {}
        tensor3d = {}

        scalar["N"] = y.shape[0]

        matrix["L_V"] = torch.linalg.cholesky(v)
        if require_loglik:
            scalar["logdet_V"] = 2.0 * torch.log(torch.diagonal(matrix["L_V"])).sum()
        else:
            scalar["logdet_V"] = None

        matrix["X"] = x
        matrix["Y"] = y.unsqueeze(-1)
        matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L_V"])
        matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L_V"])
        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]
        matrix["(X^T V^{-1} X)^{-1} X^T V^{-1} Y"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])
        matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"] = torch.cholesky_solve(matrix["V^{-1} X"].T, matrix["L_{X^T V^{-1} X}"])

        matrix["P Y"] = matrix["V^{-1} Y"] - matrix["V^{-1} X"] @ (matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"] @ matrix["Y"])

        tensor3d["dV P Y"] = dv @ matrix["P Y"]
        vector["quad"] = (matrix["P Y"].T @ tensor3d["dV P Y"]).squeeze(-1).squeeze(-1)

        if trace_approx:
            m = max(1, int(subspace_fraction * scalar["N"]) + 1)

            matrix["S"] = torch.randint(0, 2, (scalar["N"], m), dtype=dv.dtype, device=dv.device) * 2.0 - 1.0
            matrix["G"] = torch.randint(0, 2, (scalar["N"], m), dtype=dv.dtype, device=dv.device) * 2.0 - 1.0

            tensor3d["dV S"] = dv @ matrix["S"]
            tensor3d["V^{-1} dV S"] = torch.cholesky_solve(tensor3d["dV S"], matrix["L_V"])
            tensor3d["Q"], tensor3d["R"] = torch.linalg.qr(tensor3d["V^{-1} dV S"])
            tensor3d["(I - Q Q^T) G"] = matrix["G"] - tensor3d["Q"] @ (tensor3d["Q"].mT @ matrix["G"])
            tensor3d["dV (I - Q Q^T) G"] = dv @ tensor3d["(I - Q Q^T) G"]
            tensor3d["V^{-1} dV (I - Q Q^T) G"] = torch.cholesky_solve(tensor3d["dV (I - Q Q^T) G"], matrix["L_V"])
            tensor3d["G^T (I - Q Q^T)"] = tensor3d["(I - Q Q^T) G"].mT
            tensor3d["G^T (I - Q Q^T) (V^{-1} dV) (I - Q Q^T) G"] = tensor3d["G^T (I - Q Q^T)"] @ tensor3d["V^{-1} dV (I - Q Q^T) G"]
            tensor3d["Q^T (V^{-1} dV) Q"] = tensor3d["Q"].mT @ torch.cholesky_solve(dv @ tensor3d["Q"], matrix["L_V"])
            vector["tr(V^{-1} dV)"] = torch.diagonal(tensor3d["Q^T (V^{-1} dV) Q"], dim1=-2, dim2=-1).sum(dim=-1) + 1.0 / m * torch.diagonal(tensor3d["G^T (I - Q Q^T) (V^{-1} dV) (I - Q Q^T) G"], dim1=-2, dim2=-1).sum(dim=-1)

            tensor3d["H dV S"] = torch.einsum(
                "np,kpm->knm",
                matrix["V^{-1} X"],
                torch.einsum("pn,knm->kpm", matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"], tensor3d["dV S"]),
            )
            tensor3d["Q"], tensor3d["R"] = torch.linalg.qr(tensor3d["H dV S"])
            tensor3d["(I - Q Q^T) G"] = matrix["G"] - tensor3d["Q"] @ (tensor3d["Q"].mT @ matrix["G"])
            tensor3d["dV (I - Q Q^T) G"] = dv @ tensor3d["(I - Q Q^T) G"]
            tensor3d["H dV (I - Q Q^T) G"] = torch.einsum(
                "np,kpm->knm",
                matrix["V^{-1} X"],
                torch.einsum("pn,knm->kpm", matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"], tensor3d["dV (I - Q Q^T) G"]),
            )
            tensor3d["G^T (I - Q Q^T)"] = tensor3d["(I - Q Q^T) G"].mT
            tensor3d["G^T (I - Q Q^T) (H dV) (I - Q Q^T) G"] = tensor3d["G^T (I - Q Q^T)"] @ tensor3d["H dV (I - Q Q^T) G"]
            vector["tr(Q^T (H dV) Q)"] = torch.einsum(
                "kmp,kpm->k",
                torch.einsum("kmn,np->kmp", tensor3d["Q"].mT, matrix["V^{-1} X"]),
                torch.einsum("pn,knm->kpm", matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"], dv @ tensor3d["Q"]),
            )
            vector["tr(H dV)"] = vector["tr(Q^T (H dV) Q)"] + 1.0 / m * torch.diagonal(tensor3d["G^T (I - Q Q^T) (H dV) (I - Q Q^T) G"], dim1=-2, dim2=-1).sum(dim=-1)

            vector["tr(P dV)"] = vector["tr(V^{-1} dV)"] - vector["tr(H dV)"]
        else:
            matrix["V^{-1} dV"] = torch.cholesky_solve(dv, matrix["L_V"])
            vector["tr(V^{-1} dV)"] = torch.diagonal(matrix["V^{-1} dV"], dim1=-2, dim2=-1).sum(dim=-1)
            vector["tr(H dV)"] = torch.einsum(
                "pn,knp->k",
                matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"],
                dv @ matrix["V^{-1} X"],
            )
            vector["tr(P dV)"] = vector["tr(V^{-1} dV)"] - vector["tr(H dV)"]

        vector["score"] = 0.5 * (vector["quad"] - vector["tr(P dV)"])

        tensor3d["Q"] = (dv @ matrix["P Y"]).squeeze(-1)
        matrix["V^{-1} Q"] = torch.cholesky_solve(tensor3d["Q"].T, matrix["L_V"])
        matrix["P Q"] = matrix["V^{-1} Q"] - matrix["V^{-1} X"] @ (matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"] @ tensor3d["Q"].T)
        matrix["AI"] = 0.5 * (tensor3d["Q"] @ matrix["P Q"])

        if require_loglik:
            scalar["loglik"] = -0.5 * (scalar["logdet_V"] + 2.0 * torch.log(torch.diagonal(matrix["L_{X^T V^{-1} X}"])).sum() + (matrix["Y"].T @ matrix["P Y"]).squeeze())
        else:
            scalar["loglik"] = torch.nan

        return vector["score"], matrix["AI"], scalar["loglik"]

    def _ai_step_from_tensors_with_inverse(self, x, y, v, dv, v_inv, require_loglik=True, trace_approx=False, subspace_fraction=0.01):
        r"""Compute one marginal REML score and AI step using a supplied inverse.

        This branch requires ``v_inv`` and forms :math:`\symbf{P}` explicitly.
        Its approximate trace uses the combined :math:`\symbf{P}\dot{\symbf{V}}`
        Hutch++ estimator, so it can differ from the no-inverse branch when
        ``trace_approx=True``.  Exact traces agree between the two branches.

        Args:
            x: Fixed-effect design matrix of shape ``(n, p)``.
            y: Response vector of shape ``(n,)``.
            v: Marginal covariance matrix of shape ``(n, n)``.
            dv: Derivatives of ``v`` with shape ``(k, n, n)``.  The
                first dimension is the parameter dimension.
            v_inv: Precomputed inverse of ``v``.
            require_loglik: If ``True``, return the restricted
                log-likelihood.  If ``False``, return ``torch.nan`` and skip
                the log-determinant calculations.
            trace_approx: If ``True``, estimate the trace term with Hutch++.
                The AI matrix and log-likelihood remain exact.
            subspace_fraction: Probe-subspace fraction for Hutch++.
                Must lie in ``[0, 1]``; only used when ``trace_approx`` is
                ``True``.  Default: ``0.01``.

        Raises:
            ValueError: If ``dv`` has a zero-length parameter dimension,
                or if ``subspace_fraction`` is outside ``[0, 1]``.

        Returns:
            tuple: ``(score, ai, loglik)``.  ``score`` has shape ``(k,)``,
            ``ai`` has shape ``(k, k)``, and ``loglik`` is a scalar tensor
            (``torch.nan`` when ``require_loglik`` is ``False``).
        """
        if dv.shape[0] == 0:
            raise ValueError("dv must have at least one parameter; got dv.shape[0] == 0.")
        if not 0.0 <= subspace_fraction <= 1.0:
            raise ValueError("subspace_fraction must lie in [0, 1].")

        matrix = {}
        scalar = {}
        vector = {}
        tensor3d = {}

        scalar["N"] = y.shape[0]

        matrix["V^{-1}"] = v_inv
        if require_loglik:
            matrix["L_V"] = torch.linalg.cholesky(v)
            scalar["logdet_V"] = 2.0 * torch.log(torch.diagonal(matrix["L_V"])).sum()
        else:
            matrix["L_V"] = None
            scalar["logdet_V"] = None

        matrix["V^{-1} X"] = matrix["V^{-1}"] @ x
        matrix["V^{-1} Y"] = matrix["V^{-1}"] @ y.unsqueeze(-1)

        matrix["X"] = x
        matrix["Y"] = y.unsqueeze(-1)
        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]
        matrix["(X^T V^{-1} X)^{-1} X^T V^{-1} Y"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])
        matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"] = torch.cholesky_solve(matrix["V^{-1} X"].T, matrix["L_{X^T V^{-1} X}"])

        matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1}"] = matrix["V^{-1} X"] @ matrix["(X^T V^{-1} X)^{-1} X^T V^{-1}"]
        matrix["P"] = matrix["V^{-1}"] - matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1}"]
        matrix["P Y"] = matrix["V^{-1} Y"] - matrix["V^{-1} X"] @ matrix["(X^T V^{-1} X)^{-1} X^T V^{-1} Y"]

        tensor3d["dV P Y"] = dv @ matrix["P Y"]
        vector["quad"] = (matrix["P Y"].T @ tensor3d["dV P Y"]).squeeze(-1).squeeze(-1)

        if trace_approx:
            m = max(1, int(subspace_fraction * scalar["N"]) + 1)

            matrix["S"] = torch.randint(0, 2, (scalar["N"], m), dtype=dv.dtype, device=dv.device) * 2.0 - 1.0
            matrix["G"] = torch.randint(0, 2, (scalar["N"], m), dtype=dv.dtype, device=dv.device) * 2.0 - 1.0

            # A = P dV, applied batchwise: a_s[k] = P dV[k] s.
            tensor3d["A S"] = matrix["P"] @ (dv @ matrix["S"])
            tensor3d["Q"], tensor3d["R"] = torch.linalg.qr(tensor3d["A S"])

            # Exact contribution on the dominant subspace: tr(Q' A Q).
            tensor3d["A Q"] = matrix["P"] @ (dv @ tensor3d["Q"])
            vector["tr(Q^T A Q)"] = (
                tensor3d["Q"].mT * tensor3d["A Q"].transpose(-1, -2)
            ).sum(dim=(-2, -1))

            # Hutchinson correction on the deflated residual:
            # tr(G' (I - Q Q') A (I - Q Q') G) / m.
            tensor3d["A G"] = matrix["P"] @ (dv @ matrix["G"])
            tensor3d["A Q Q^T G"] = tensor3d["A Q"] @ (tensor3d["Q"].mT @ matrix["G"])
            tensor3d["residual"] = tensor3d["A G"] - tensor3d["A Q Q^T G"]
            tensor3d["residual"] = tensor3d["residual"] - tensor3d["Q"] @ (tensor3d["Q"].mT @ tensor3d["residual"])
            vector["tr(G^T residual)"] = (
                matrix["G"].mT.unsqueeze(0)
                * tensor3d["residual"].transpose(-1, -2)
            ).sum(dim=(-2, -1))

            vector["tr(P dV)"] = vector["tr(Q^T A Q)"] + vector["tr(G^T residual)"] / m
        else:
            vector["tr(P dV)"] = (
                matrix["P"].unsqueeze(0) * dv.transpose(-1, -2)
            ).sum(dim=(-2, -1))

        vector["score"] = 0.5 * (vector["quad"] - vector["tr(P dV)"])

        tensor3d["Q"] = (dv @ matrix["P Y"]).squeeze(-1)
        matrix["AI"] = 0.5 * (tensor3d["Q"] @ matrix["P"] @ tensor3d["Q"].mT)

        if require_loglik:
            scalar["loglik"] = -0.5 * (scalar["logdet_V"] + 2.0 * torch.log(torch.diagonal(matrix["L_{X^T V^{-1} X}"])).sum() + (matrix["Y"].T @ matrix["P Y"]).squeeze())
        else:
            scalar["loglik"] = torch.nan

        return vector["score"], matrix["AI"], scalar["loglik"]

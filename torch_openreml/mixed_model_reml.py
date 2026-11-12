r"""Frontend for mixed-model REML estimation.

This module provides the public :class:`MixedModelREML` entry point for
linear mixed models

.. math::

    \symbf{y}
    = \symbf{X}\symbf{\beta}
      + \sum_i \symbf{Z}_i\symbf{u}_i
      + \symbf{\epsilon},
    \qquad
    \symbf{u}_i \sim N(\symbf{0}, \symbf{G}_i),
    \qquad
    \symbf{\epsilon} \sim N(\symbf{0}, \symbf{R}),

with marginal covariance

.. math::

    \symbf{V}(\boldsymbol{\theta})
    = \symbf{R}(\boldsymbol{\theta})
      + \sum_i \symbf{Z}_i\symbf{G}_i(\boldsymbol{\theta})
        \symbf{Z}_i^\top.

The frontend owns the structured model specification, the parameter
ordering, and the choice of numerical method.  It does not implement the
score or average-information algebra itself; that is delegated to the
methods in :mod:`torch_openreml.methods`.

Classes:
    MixedModelREML
"""

import time
import warnings

import torch

from torch_openreml.covariance import (
    Augment,
    BlockDiagonal,
    KnownMatrix,
    MarginalCovariance,
    Matrix,
    Operator,
)
from torch_openreml.methods import CForm, Marginal
from torch_openreml.post import blue, loglik, marginal_predict, marginal_residual


class MixedModelREML:
    r"""REML estimator for a structured linear mixed model.

    The model is specified by zero or more random-effect terms and one
    residual term.  A random-effect term is a ``(design, covariance)``
    pair, where *design* is the incidence matrix :math:`\symbf{Z}_i` and
    *covariance* is the :class:`~torch_openreml.covariance.matrix.Matrix`
    for :math:`\symbf{G}_i`, or a fixed two-dimensional tensor for a
    parameter-free :math:`\symbf{G}_i`.  The residual term is a single
    :class:`~torch_openreml.covariance.matrix.Matrix` for :math:`\symbf{R}`,
    or a fixed two-dimensional tensor, which is wrapped in a
    :class:`~torch_openreml.covariance.known_matrix.KnownMatrix`.

    The constructor builds the composite marginal covariance

    .. math::

        \symbf{V}(\boldsymbol{\theta})
        = \symbf{R}(\boldsymbol{\theta})
          + \symbf{Z}\symbf{G}(\boldsymbol{\theta})\symbf{Z}^\top,

    with :math:`\symbf{Z}` the column-wise augmentation of the
    :math:`\symbf{Z}_i` and :math:`\symbf{G}` their block-diagonal
    combination.  Score and average-information evaluation is then
    dispatched to a numerical method:

    * :class:`~torch_openreml.methods.cform.CForm` for one or more
      random-effect blocks, with fixed or parameterised designs and
      dense, invertible :math:`\symbf{G}` and :math:`\symbf{R}`;
    * :class:`~torch_openreml.methods.marginal.Marginal` for the general
      :math:`\symbf{V}`-based case, including multiple random-effect
      blocks and approximate traces.

    The method is selected by :attr:`method`, which is ``"auto"`` by
    default.  ``"cform"`` and ``"marginal"`` force a particular method.
    A configured :class:`~torch_openreml.methods.cform.CForm` or
    :class:`~torch_openreml.methods.marginal.Marginal` instance may also
    be supplied, for example ``Marginal(trace_approx="auto")``; the
    instance is stored on the model and reused across steps.

    Free covariance parameters are ordered as
    ``[free_params_Z, free_params_G, free_params_R]``.  A block with no free
    parameters occupies a zero-length slice.  The fitted model keeps the
    optimisation trace in :attr:`history`; use :meth:`optimize` for fitting
    and :meth:`blue`, :meth:`predict`, :meth:`residual` and :meth:`loglik`
    for post-fitting quantities.

    Args:
        *args: Random-effect ``(design, covariance)`` pairs and exactly one
            unnamed residual covariance matrix.  The residual covariance is
            the only positional argument that is not a ``list`` or
            ``tuple``; it may be a ``Matrix`` or a two-dimensional tensor.
        **kwargs: Named random-effect terms, each a ``(design, covariance)``
            pair.  Names must be unique.

    Raises:
        TypeError: If a design is not a tensor or
            :class:`~torch_openreml.covariance.matrix.Matrix`, or a
            covariance is not a
            :class:`~torch_openreml.covariance.matrix.Matrix` or a fixed
            two-dimensional tensor.
        ValueError: If the residual term is missing or duplicated, a term
            has the wrong length, a random-effect covariance tensor is not
            two-dimensional, a term name is duplicated, or the marginal
            covariance has no free parameters.

    Example:

    .. jupyter-execute::

        import torch
        from torch_openreml import MixedModelREML
        from torch_openreml.covariance import ScalarMatrix, DummyMatrix

        n, p = 100, 3
        y = torch.randn(n)
        x = torch.randn(n, p)
        group = [f"g{i % 20}" for i in range(n)]
        z = DummyMatrix(group)
        reml = MixedModelREML((z, ScalarMatrix(20)), ScalarMatrix(n))
        free_params_hat, n_iter = reml.optimize(
            y, x, torch.tensor([0.0, 0.0]), verbose=0
        )
    """

    def __init__(self, *args, method="auto", **kwargs):
        def as_matrix(covariance):
            if isinstance(covariance, torch.Tensor):
                return KnownMatrix(covariance)
            return covariance

        random_terms = []
        residual = None

        for arg in args:
            if isinstance(arg, (list, tuple)):
                self._validate_term(arg, len(random_terms))
                random_terms.append((f"random_term_{len(random_terms)}", arg[0], as_matrix(arg[1])))
            else:
                if residual is not None:
                    raise ValueError("Only a single unnamed residual term should be provided!")
                if not isinstance(arg, (Matrix, torch.Tensor)):
                    raise TypeError("The residual term must be a Matrix or a two-dimensional Tensor!")
                residual = as_matrix(arg)

        for name, term in kwargs.items():
            if not isinstance(term, (list, tuple)):
                raise ValueError(f"Keyword argument `{name}` must be a `(design, covariance)` pair!")
            self._validate_term(term, name)
            if any(name == existing for existing, _, _ in random_terms):
                raise ValueError(f"Term name `{name}` already exists!")
            random_terms.append((name, term[0], as_matrix(term[1])))

        if residual is None:
            raise ValueError("An unnamed residual term must be provided!")

        self._r = residual
        self._z_terms = [(name, design) for name, design, _ in random_terms]
        self._g_terms = [(name, cov) for name, _, cov in random_terms]

        if not random_terms:
            self._z = None
            self._g = None
            v = self._r
        else:
            if len(random_terms) == 1:
                self._z = random_terms[0][1]
                self._g = random_terms[0][2]
            else:
                if any(isinstance(design, Matrix) for _, design in self._z_terms):
                    self._z = Augment(**dict(self._z_terms))
                else:
                    self._z = torch.cat([design for _, design in self._z_terms], dim=1)
                self._g = BlockDiagonal(**dict(self._g_terms))

            v = MarginalCovariance(Z=self._z, G=self._g, R=self._r)

        if v.num_free_params == 0:
            raise ValueError("The marginal covariance must have at least one free parameter!")

        self._v = v
        self.history = None
        self.method = method

        if isinstance(method, CForm):
            self._cform = method
            self._marginal = None
        elif isinstance(method, Marginal):
            self._cform = None
            self._marginal = method
        elif method == "cform":
            self._cform = CForm()
            self._marginal = None
        elif method == "marginal":
            self._cform = None
            self._marginal = Marginal()
        else:  # "auto" and unknown strings behave like auto
            self._cform = CForm()
            self._marginal = Marginal()

    @staticmethod
    def _validate_term(term, label):
        if len(term) != 2:
            raise ValueError(f"Random term `{label}` must be a length-2 `(design, covariance)` pair!")
        design, covariance = term
        if not isinstance(design, (torch.Tensor, Matrix)):
            raise TypeError(f"The design of random term `{label}` must be a Tensor or Matrix!")
        if isinstance(covariance, torch.Tensor):
            if covariance.ndim != 2:
                raise ValueError(f"The covariance of random term `{label}` must be two-dimensional!")
        elif not isinstance(covariance, Matrix):
            raise TypeError(f"The covariance of random term `{label}` must be a Matrix or a fixed two-dimensional Tensor!")

    @property
    def v(self):
        r"""Marginal covariance :math:`\symbf{V}=\symbf{R}+\symbf{Z}\symbf{G}\symbf{Z}'`."""
        return self._v

    @property
    def r(self):
        r"""Residual covariance :math:`\symbf{R}`."""
        return self._r

    @property
    def g(self):
        r"""Composite random-effect covariance :math:`\symbf{G}`."""
        return self._g

    @property
    def z(self):
        r"""Composite incidence matrix :math:`\symbf{Z}`."""
        return self._z

    @property
    def g_terms(self):
        """Named ``(name, covariance)`` random-effect terms."""
        return self._g_terms

    @property
    def z_terms(self):
        """Named ``(name, design)`` random-effect terms."""
        return self._z_terms

    def ai_step(self, y, x, free_params, require_loglik=True):
        r"""Evaluate one AI step with the selected method.

        The score and AI matrix are computed by
        :class:`~torch_openreml.methods.cform.CForm` or
        :class:`~torch_openreml.methods.marginal.Marginal`.  With
        ``method="auto"`` CForm is tried first and Marginal is used if CForm
        cannot handle the model.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Flat covariance-parameter tensor,
                ordered as ``[free_params_Z, free_params_G, free_params_R]``.
            require_loglik (bool, optional): Whether to evaluate the
                restricted log-likelihood.  Default: ``True``.

        Returns:
            tuple: ``(score, ai, loglik)``.  ``score`` has shape ``(k,)``,
            ``ai`` has shape ``(k, k)``, and ``loglik`` is a scalar tensor,
            or ``torch.nan`` when ``require_loglik`` is ``False``.
        """
        kwargs = {"model": self, "y": y, "x": x, "free_params": free_params, "require_loglik": require_loglik}

        if isinstance(self.method, (CForm, Marginal)):
            return self.method.ai_step(**kwargs)

        if self.method == "marginal":
            if self._marginal is None:
                self._marginal = Marginal()
            return self._marginal.ai_step(**kwargs)

        if self.method == "cform":
            if self._cform is None:
                self._cform = CForm()
            return self._cform.ai_step(**kwargs)

        # "auto" (and unknown strings) dispatch on capability.
        if self._cform is None:
            self._cform = CForm()
        if self._marginal is None:
            self._marginal = Marginal()

        try:
            return self._cform.ai_step(**kwargs)
        except RuntimeError as exc:
            warnings.warn(f"CForm failed ({exc}); falling back to Marginal.", RuntimeWarning, stacklevel=2)
            return self._marginal.ai_step(**kwargs)

    def blue(self, y, x, free_params):
        r"""Best linear unbiased estimator of the fixed effects.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Covariance parameters.

        Returns:
            torch.Tensor: Estimated fixed effects of shape ``(p,)``.
        """
        return blue(y, x, self.v(free_params))

    def predict(self, y, x, free_params):
        r"""Fitted values at ``free_params``.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Covariance parameters.

        Returns:
            torch.Tensor: Fitted values of shape ``(n,)``.
        """
        return marginal_predict(y, x, self.v(free_params))

    def residual(self, y, x, free_params):
        r"""Residuals at ``free_params``.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Covariance parameters.

        Returns:
            torch.Tensor: Residuals of shape ``(n,)``.
        """
        return marginal_residual(y, x, self.v(free_params))

    def get_free_params(self, select="last"):
        r"""Return a covariance-parameter iterate from the history.

        Args:
            select (str, optional): ``"last"`` for the final iterate or
                ``"best"`` for the iterate with the highest recorded
                log-likelihood.  Default: ``"last"``.

        Returns:
            torch.Tensor: Selected covariance-parameter vector.
        """
        if select == "last":
            return self.history["free_params"][-1]
        if torch.is_tensor(self.history["loglik"][-1]):
            index = torch.argmax(torch.stack(self.history["loglik"])).item()
            return self.history["free_params"][index]
        return self.history["free_params"][-1]

    def tree(self, free_params=None, *, select="last"):
        r"""Evaluate the covariance structure and return its path-keyed tree.

        Args:
            free_params (torch.Tensor or dict, optional): Covariance
                parameters to evaluate.  When omitted, ``select`` chooses an
                iterate from :attr:`history`.
            select (str, optional): History selector used when
                ``free_params`` is omitted.  Default: ``"last"``.

        Returns:
            dict: Path-keyed covariance tree.

        Raises:
            TypeError: If ``free_params`` or ``select`` has the wrong type.
            ValueError: If no history exists and no parameters are supplied.
        """
        if isinstance(free_params, str):
            raise TypeError(
                "'free_params' must be a Torch tensor, a dict or None; pass a "
                "selector string as select='best' instead of positionally."
            )
        if select is not None and not isinstance(select, str):
            raise TypeError(
                f"'select' must be a string or None, got {type(select).__name__}!"
            )
        if free_params is None and select is not None:
            if self.history is None:
                raise ValueError(
                    f"No optimisation history to select '{select}' from; call "
                    "optimize() first, or pass free_params."
                )
            free_params = self.get_free_params(select)

        if isinstance(self.v, Operator):
            return self.v.tree(free_params)
        return {"/": self.v(free_params)}

    def is_converged(self):
        """Whether the log-likelihood and parameter update have settled.

        A step is accepted only when both enabled checks pass: the change in
        log-likelihood is below ``tol_loglik`` and the largest absolute
        parameter update is below ``tol_update``.
        """
        if len(self.history["loglik"]) < 2:
            return False

        if self.check_loglik:
            last = self.history["loglik"][-1]
            prev = self.history["loglik"][-2]
            if torch.is_tensor(last) and torch.is_tensor(prev):
                if torch.abs(last - prev).item() >= self.tol_loglik:
                    return False

        if self.check_update:
            step = self.history["update"][-1]
            if step.numel() > 0:
                if torch.max(torch.abs(step)).item() >= self.tol_update:
                    return False

        return True

    def update(self, free_params, delta, eta, lb=-torch.inf, ub=torch.inf):
        r"""Apply a damped AI step and clip to bounds.

        Args:
            free_params (torch.Tensor): Current parameters.
            delta (torch.Tensor): AI direction.
            eta (float): Step-length multiplier.
            lb (float, optional): Lower bound.  Default: ``-torch.inf``.
            ub (float, optional): Upper bound.  Default: ``torch.inf``.

        Returns:
            tuple: ``(updated, step)``, where ``updated = clip(free_params +
            eta * delta)`` and ``step = updated - free_params``.
        """
        updated = torch.clamp(free_params + delta * eta, min=lb, max=ub)
        return updated, updated - free_params

    def loglik(self, y, x, free_params):
        r"""Evaluate the restricted log-likelihood with the selected method.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Covariance parameters.

        Returns:
            torch.Tensor: Scalar restricted log-likelihood.

        Raises:
            RuntimeError: If ``method="cform"`` and CForm cannot handle the
                model.
        """
        if isinstance(self.method, CForm):
            return self.method.loglik(self, y, x, free_params)

        if isinstance(self.method, Marginal):
            return self.method.loglik(self, y, x, free_params)

        if self.method == "cform":
            if self._cform is None:
                self._cform = CForm()
            return self._cform.loglik(self, y, x, free_params)

        if self.method == "marginal":
            if self._marginal is None:
                self._marginal = Marginal()
            return self._marginal.loglik(self, y, x, free_params)

        if self._cform is None:
            self._cform = CForm()
        try:
            return self._cform.loglik(self, y, x, free_params)
        except RuntimeError:
            if self._marginal is None:
                self._marginal = Marginal()
            return self._marginal.loglik(self, y, x, free_params)

    def optimize(
        self,
        y,
        x,
        init_values=None,
        max_iter=200,
        lb=-torch.inf,
        ub=torch.inf,
        verbose=0,
        check_update=True,
        check_loglik=True,
        tol_update=1e-4,
        tol_loglik=1e-4,
        armijo_trials=30,
        armijo_c=1e-4,
        eta_shrink=0.5,
        eta_max=1.0,
        eta_min=1e-10,
        fallback_eta=1e-3,
        loglik_tol=1e-8,
    ):
        r"""Run the AI-REML optimisation loop with Armijo backtracking.

        The AI direction is always ``delta = AI^{-1} score``.  A backtracking
        line search starts from ``eta_max`` and multiplies it by
        ``eta_shrink`` after each rejected candidate.  ``eta_max`` is also
        the value restored after every accepted or fallback step, so it acts
        as the maximum step length.  If the line search cannot find a finite
        candidate that sufficiently increases the log-likelihood, one small
        gradient-ascent step is taken instead.

        A candidate is accepted when

        .. math::

            \ell_{\mathrm{new}}
            \geq \ell
            + c \, \eta \, \symbf{s}^\top\boldsymbol{\delta}
            - \epsilon \max(1, |\ell|),

        where ``c = armijo_c`` and ``epsilon = loglik_tol``.  The last term is
        a small numerical slack; set ``loglik_tol=0`` for strict Armijo.

        Any score history kept by the selected :class:`Marginal` instance
        for ``trace_approx="auto"`` is reset before the first iteration.

        Args:
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            init_values (torch.Tensor, optional): Initial covariance
                parameters.  When omitted, the model's default parameters are
                used.
            max_iter (int, optional): Maximum number of AI iterations.
                Default: ``200``.
            lb (float, optional): Lower bound for parameters.
                Default: ``-torch.inf``.
            ub (float, optional): Upper bound for parameters.
                Default: ``torch.inf``.
            verbose (int, optional): If positive, print one line per iteration.
                Default: ``0``.
            check_update (bool, optional): Whether the maximum absolute
                parameter update participates in convergence.  Default:
                ``True``.
            check_loglik (bool, optional): Whether the log-likelihood change
                participates in convergence.  Default: ``True``.
            tol_update (float, optional): Convergence tolerance for the
                maximum absolute parameter update.  Default: ``1e-4``.
            tol_loglik (float, optional): Convergence tolerance for the
                log-likelihood change.  Default: ``1e-4``.
            armijo_trials (int, optional): Maximum number of backtracking
                candidates.  Default: ``30``.
            armijo_c (float, optional): Armijo sufficient-increase constant.
                Default: ``1e-4``.
            eta_shrink (float, optional): Multiplier applied after each
                rejected candidate.  Default: ``0.5``.
            eta_max (float, optional): Initial and restored step length.
                Default: ``1.0``.
            eta_min (float, optional): Smallest step length tried.
                Default: ``1e-10``.
            fallback_eta (float, optional): Step length for the normalized
                gradient fallback.  Default: ``1e-3``.
            loglik_tol (float, optional): Relative numerical slack in the
                Armijo test.  Default: ``1e-8``.

        Returns:
            tuple: ``(free_params, n_iter)``, where ``free_params`` is the
            final covariance-parameter tensor and ``n_iter`` is the number of
            completed iterations.

        Notes:
            The optimisation trace is stored in :attr:`history` with keys
            ``"free_params"``, ``"loglik"``, ``"score"``, ``"ai"``,
            ``"delta"``, ``"update"``, ``"eta"`` and ``"accepted"``.
            ``"accepted"`` is ``True`` only for a line-search-accepted Newton
            step; gradient fallback steps are recorded with ``False``.
        """
        eta = eta_max
        free_params = self.v.build_params(init_values, include_fixed=False, trans=False).detach()
        y = y.to(device=free_params.device, dtype=free_params.dtype)
        x = x.to(device=free_params.device, dtype=free_params.dtype)

        if isinstance(self.method, Marginal):
            self.method.reset()
        elif self._marginal is not None:
            self._marginal.reset()

        self.history = {"free_params": [], "loglik": [], "score": [], "ai": [], "delta": [], "update": [], "eta": [], "accepted": []}
        self.check_update = check_update
        self.check_loglik = check_loglik
        self.tol_update = tol_update
        self.tol_loglik = tol_loglik

        start_time = time.perf_counter()
        n_iter = 0

        with torch.no_grad():
            for i in range(max_iter):
                score, ai, ll = self.ai_step(
                    y,
                    x,
                    free_params,
                    require_loglik=True,
                )

                delta = torch.linalg.lstsq(
                    ai, score.unsqueeze(-1)
                ).solution.squeeze()

                theta_old = free_params
                eta_i = eta
                accepted = False

                for _ in range(armijo_trials):
                    if eta_i < eta_min:
                        break

                    theta_new, _ = self.update(theta_old, delta, eta_i, lb, ub)

                    if not torch.isfinite(theta_new).all():
                        eta_i *= eta_shrink
                        continue

                    try:
                        ll_new = self.loglik(y, x, theta_new)
                    except RuntimeError:
                        eta_i *= eta_shrink
                        continue

                    threshold = ll + armijo_c * eta_i * torch.dot(score, delta) - loglik_tol * torch.clamp(
                        torch.abs(ll), min=1.0
                    )
                    if torch.isfinite(ll_new) and ll_new >= threshold:
                        accepted = True
                        break

                    eta_i *= eta_shrink

                if not accepted:
                    grad = score / torch.clamp(
                        torch.linalg.vector_norm(score), min=1.0
                    )
                    theta_new, step = self.update(
                        theta_old, grad, fallback_eta, lb, ub
                    )
                    free_params = theta_new
                    ll_new = self.loglik(y, x, free_params)
                    eta_used = float(fallback_eta)
                    eta = eta_max
                else:
                    step = theta_new - theta_old
                    free_params = theta_new
                    eta_used = float(eta_i)
                    eta = eta_max

                self.history["free_params"].append(free_params)
                self.history["loglik"].append(ll_new)
                self.history["score"].append(score)
                self.history["ai"].append(ai)
                self.history["delta"].append(delta)
                self.history["update"].append(step)
                self.history["eta"].append(eta_used)
                self.history["accepted"].append(accepted)

                n_iter = i + 1

                if verbose > 0:
                    print(
                        f"[{i + 1:3d}] ||score||: "
                        f"{torch.norm(score):12.4f}, ||delta||: "
                        f"{torch.norm(delta):6.4f}, eta: {eta_used:.2f}, "
                        f"||step||: {torch.norm(step):6.4f}, accepted: {accepted}, "
                        f"loglik: {ll_new:8.4f}"
                    )

                if self.is_converged():
                    if verbose > 0:
                        print(
                            f"Converged at iteration {i + 1}, total time: "
                            f"{time.perf_counter() - start_time:.2f}s"
                        )
                    break
            else:
                if verbose > 0:
                    print(
                        f"Reached maximum iterations ({max_iter}), total time: "
                        f"{time.perf_counter() - start_time:.2f}s"
                    )

        return free_params, n_iter

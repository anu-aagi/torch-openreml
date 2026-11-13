r"""Frontend for mixed-model REML estimation.

This module provides the public :class:`MixedModelREML` entry point for
linear mixed models

.. math::

    \symbf{y}
    = \symbf{X}\symbf{\beta}
      + \sum_i \symbf{Z}_i(\boldsymbol{\theta})\symbf{u}_i
      + \symbf{\epsilon},
    \qquad
    \symbf{u}_i \sim N(\symbf{0}, \symbf{G}_i(\boldsymbol{\theta})),
    \qquad
    \symbf{\epsilon} \sim N(\symbf{0}, \symbf{R}(\boldsymbol{\theta})),

with marginal covariance

.. math::

    \symbf{V}(\boldsymbol{\theta})
    = \symbf{R}(\boldsymbol{\theta})
      + \sum_i \symbf{Z}_i(\boldsymbol{\theta})
        \symbf{G}_i(\boldsymbol{\theta})
        \symbf{Z}_i(\boldsymbol{\theta})^\top.

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
    ScalarMatrix,
)
from torch_openreml.methods import CForm, Marginal
from torch_openreml.post import (
    blue,
    blup,
    cov_blue,
    cov_blup,
    loglik,
    marginal_predict,
    marginal_residual,
    predict as conditional_predict,
    residual as conditional_residual,
    var_blue,
)


class MixedModelREML:
    r"""REML estimator for a structured linear mixed model.

    The model is specified by zero or more random-effect terms and one
    residual term, supplied with the keyword ``R``.  A random-effect term
    is a ``(design, covariance)`` pair, where *design* is the incidence
    matrix :math:`\symbf{Z}_i(\boldsymbol{\theta})`, either a fixed tensor
    or a :class:`~torch_openreml.covariance.matrix.Matrix` — possibly
    parameterised, in which case its free parameters are part of
    :math:`\boldsymbol{\theta}` — and *covariance* is the
    :class:`~torch_openreml.covariance.matrix.Matrix` for
    :math:`\symbf{G}_i`, or a fixed two-dimensional tensor for a
    parameter-free :math:`\symbf{G}_i`.  The residual term is a single
    :class:`~torch_openreml.covariance.matrix.Matrix` for :math:`\symbf{R}`,
    or a fixed two-dimensional tensor, which is wrapped in a
    :class:`~torch_openreml.covariance.known_matrix.KnownMatrix`.  When
    ``R`` is omitted it defaults to :math:`\sigma^2\symbf{I}`, with the
    dimension inferred from the random-effect design.

    The constructor builds the composite marginal covariance

    .. math::

        \symbf{V}(\boldsymbol{\theta})
        = \symbf{R}(\boldsymbol{\theta})
          + \symbf{Z}(\boldsymbol{\theta})
            \symbf{G}(\boldsymbol{\theta})
            \symbf{Z}(\boldsymbol{\theta})^\top,

    with :math:`\symbf{Z}(\boldsymbol{\theta})` the column-wise
    augmentation of the :math:`\symbf{Z}_i(\boldsymbol{\theta})` and
    :math:`\symbf{G}(\boldsymbol{\theta})` their block-diagonal
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
    and :meth:`blue`, :meth:`blup`, :meth:`blup_per_term`, :meth:`build_params`, :meth:`marginal_predict`, :meth:`marginal_residual`, :meth:`predict`, :meth:`residual` and :meth:`loglik`
    for post-fitting quantities.

    Args:
        *args: Random-effect ``(design, covariance)`` pairs.
        R: Residual covariance :math:`\symbf{R}`, either a
            :class:`~torch_openreml.covariance.matrix.Matrix` or a fixed
            two-dimensional tensor, which is wrapped in a
            :class:`~torch_openreml.covariance.known_matrix.KnownMatrix`.
            Defaults to ``None``, in which case
            :math:`\sigma^2\symbf{I}` is used, i.e. a
            :class:`~torch_openreml.covariance.scalar_matrix.ScalarMatrix`
            whose dimension is inferred from the random-effect design.
            ``R`` must be supplied explicitly when the model has no
            random-effect terms, because the dimension cannot then be
            inferred.
        method: Numerical method used for score and average-information
            evaluation.  ``"auto"`` (the default) tries
            :class:`~torch_openreml.methods.cform.CForm` first and falls
            back to :class:`~torch_openreml.methods.marginal.Marginal` when
            CForm cannot handle the model; ``"cform"`` and ``"marginal"``
            force one method; a configured
            :class:`~torch_openreml.methods.cform.CForm` or
            :class:`~torch_openreml.methods.marginal.Marginal` instance is
            stored on the model and reused across steps.  Any other value
            behaves like ``"auto"``.
        **kwargs: Named random-effect terms, each a ``(design,
            covariance)`` pair.  Term names must be unique, and no term may
            be named ``R`` (or ``method``).

    Raises:
        TypeError: If a positional argument is not a ``(design,
            covariance)`` pair, a design is not a tensor or
            :class:`~torch_openreml.covariance.matrix.Matrix`, or a
            covariance is not a
            :class:`~torch_openreml.covariance.matrix.Matrix` or a fixed
            two-dimensional tensor.
        ValueError: If ``R`` is omitted and the model has no random-effect
            terms, a term has the wrong length, a random-effect covariance
            tensor is not two-dimensional, a term name is duplicated, or the
            marginal covariance has no free parameters.

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
        reml = MixedModelREML((z, ScalarMatrix(20)), R=ScalarMatrix(n))
        free_param_names, free_params_hat = reml.optimize(y, x, torch.tensor([0.0, 0.0]), verbose=1)
    """

    def __init__(self, *args, R=None, method="auto", **kwargs):
        def as_matrix(covariance):
            if isinstance(covariance, torch.Tensor):
                return KnownMatrix(covariance)
            return covariance

        random_terms = []

        for index, arg in enumerate(args):
            if not isinstance(arg, (list, tuple)):
                raise TypeError(f"Positional argument {index} must be a `(design, covariance)` pair.")
            self._validate_term(arg, index)
            random_terms.append((f"random_term_{index}", arg[0], as_matrix(arg[1])))

        for name, term in kwargs.items():
            if not isinstance(term, (list, tuple)):
                raise ValueError(f"Keyword argument `{name}` must be a `(design, covariance)` pair!")
            self._validate_term(term, name)
            if any(name == existing for existing, _, _ in random_terms):
                raise ValueError(f"Term name `{name}` already exists!")
            random_terms.append((name, term[0], as_matrix(term[1])))

        self._z_terms = [(name, design) for name, design, _ in random_terms]
        self._g_terms = [(name, cov) for name, _, cov in random_terms]

        if not random_terms:
            self._z = None
            self._g = None
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

        if R is not None and not isinstance(R, (Matrix, torch.Tensor)):
            raise TypeError("The residual term `R` must be a Matrix or a two-dimensional Tensor!")

        if R is None:
            if self._z is None:
                raise ValueError("A residual term must be provided as `R=` when the model has no random-effect terms!")
            if isinstance(self._z, torch.Tensor):
                n = self._z.shape[0]
            else:
                with torch.no_grad():
                    n = self._z().shape[0]
            residual = ScalarMatrix(n)
        else:
            residual = as_matrix(R)

        self._r = residual
        v = self._r if self._z is None else MarginalCovariance(Z=self._z, G=self._g, R=self._r)

        if v.num_free_params == 0:
            raise ValueError("The marginal covariance must have at least one free parameter!")

        self._v = v
        self.history = None
        self.method = method
        self._y = None
        self._x = None

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
    def V(self):
        r"""Marginal covariance :math:`\symbf{V}(\boldsymbol{\theta})=\symbf{R}(\boldsymbol{\theta})+\symbf{Z}(\boldsymbol{\theta})\symbf{G}(\boldsymbol{\theta})\symbf{Z}(\boldsymbol{\theta})^\top`."""
        return self._v

    @property
    def R(self):
        r"""Residual covariance :math:`\symbf{R}`."""
        return self._r

    @property
    def G(self):
        r"""Composite random-effect covariance :math:`\symbf{G}`."""
        return self._g

    @property
    def Z(self):
        r"""Composite incidence matrix :math:`\symbf{Z}(\boldsymbol{\theta})`; it may carry its own free parameters."""
        return self._z

    @property
    def G_terms(self):
        """Named ``(name, covariance)`` random-effect terms."""
        return self._g_terms

    @property
    def Z_terms(self):
        """Named ``(name, design)`` random-effect terms."""
        return self._z_terms

    def __repr__(self):
        return repr(self.V)

    def ai_step(self, y=None, x=None, free_params=None, require_loglik=True):
        r"""Evaluate one AI step with the selected method.

        The score and AI matrix are computed by
        :class:`~torch_openreml.methods.cform.CForm` or
        :class:`~torch_openreml.methods.marginal.Marginal`.  With
        ``method="auto"`` CForm is tried first and Marginal is used if CForm
        cannot handle the model.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor, optional): Flat covariance-parameter
                tensor, ordered as ``[free_params_Z, free_params_G,
                free_params_R]``. Default: the last fitted iterate.
            require_loglik (bool, optional): Whether to evaluate the
                restricted log-likelihood.  Default: ``True``.

        Returns:
            tuple: ``(score, ai, loglik)``.  ``score`` has shape ``(k,)``,
            ``ai`` has shape ``(k, k)``, and ``loglik`` is a scalar tensor,
            or ``torch.nan`` when ``require_loglik`` is ``False``.
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)

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

    def _post_free_params(self, free_params):
        if isinstance(free_params, str):
            if free_params not in {"last", "best"}:
                raise ValueError("free_params must be a tensor, 'last', 'best' or None!")
            if self.history is None or not self.history["free_params"]:
                raise ValueError(f"No optimisation history to select '{free_params}' from; call optimize() first.")
            _, selected = self.get_free_params(free_params)
            return selected

        if free_params is not None:
            return free_params

        if self.history is not None and self.history["free_params"]:
            _, last = self.get_free_params("last")
            return last

        raise ValueError("No fitted parameters available; pass free_params or call optimize() first.")

    def _post_xy(self, y, x):
        r"""Resolve the response and fixed-effect design of a post-fit method.

        Either both ``y`` and ``x`` are supplied, or both are omitted and the
        training data stored by :meth:`optimize` is used.

        Args:
            y (torch.Tensor or None): Response vector, or ``None`` to use the
                stored training response.
            x (torch.Tensor or None): Fixed-effect design matrix, or ``None``
                to use the stored training design.

        Returns:
            tuple: ``(y, x)``.

        Raises:
            ValueError: If only one of ``y`` and ``x`` is supplied, or when no
                training data has been stored yet.
        """
        if y is None and x is None:
            if self._y is None or self._x is None:
                raise ValueError("No training data available; pass y and x or call optimize() first.")
            return self._y, self._x
        if y is None or x is None:
            raise ValueError("Pass both `y` and `x`, or neither to use the training data stored by optimize().")
        return y, x

    def _post_x(self, x):
        r"""Resolve the fixed-effect design of a post-fit method.

        Args:
            x (torch.Tensor or None): Fixed-effect design matrix, or ``None``
                to use the stored training design.

        Returns:
            torch.Tensor: The design matrix.

        Raises:
            ValueError: When ``x`` is ``None`` and no training data has been
                stored yet.
        """
        if x is None:
            if self._x is None:
                raise ValueError("No training data available; pass x or call optimize() first.")
            return self._x
        return x

    def blue(self, y=None, x=None, free_params=None):
        r"""Best linear unbiased estimator of the fixed effects.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            torch.Tensor: Estimated fixed effects of shape ``(p,)``.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.blue(y, x, torch.zeros(2))
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)
        return blue(y, x, self.V(free_params))

    def cov_blue(self, x=None, free_params=None):
        r"""Covariance matrix of the fixed-effect BLUE.

        The BLUE is :math:`\hat{\boldsymbol{\beta}}
        = (\symbf{X}^{\top}\symbf{V}^{-1}\symbf{X})^{-1}
          \symbf{X}^{\top}\symbf{V}^{-1}\symbf{y}`, so its covariance is

        .. math::

            \operatorname{Var}(\hat{\boldsymbol{\beta}})
            = \left(\symbf{X}^{\top}\symbf{V}^{-1}\symbf{X}\right)^{-1},

        the ``C11`` block of the inverse C-form matrix. The covariance does
        not depend on the response :math:`\symbf{y}`, only on the design and
        the covariance parameters.

        Args:
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            torch.Tensor: BLUE covariance matrix of shape ``(p, p)``.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.cov_blue(x, torch.zeros(2))
        """
        x = self._post_x(x)
        free_params = self._post_free_params(free_params)
        return cov_blue(x, self.V(free_params))

    def var_blue(self, x=None, free_params=None):
        r"""Variances of the fixed-effect BLUE.

        This is the diagonal of :meth:`cov_blue`,

        .. math::

            \operatorname{Var}(\hat{\beta}_j)
            = \left[\left(\symbf{X}^{\top}\symbf{V}^{-1}\symbf{X}\right)^{-1}
              \right]_{jj},

        the sampling variance of each fixed-effect estimate.

        Args:
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            torch.Tensor: BLUE variances of shape ``(p,)``.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.var_blue(x, torch.zeros(2))
        """
        x = self._post_x(x)
        free_params = self._post_free_params(free_params)
        return var_blue(x, self.V(free_params))

    def c_inverse_blocks(self, x=None, free_params=None):
        r"""Blocks of the inverse of the mixed-model coefficient matrix.

        With the coefficient matrix

        .. math::

            \symbf{C}
            = \begin{pmatrix}
                  \symbf{X}^\top\symbf{R}^{-1}\symbf{X}
                  & \symbf{X}^\top\symbf{R}^{-1}\symbf{Z} \\
                  \symbf{Z}^\top\symbf{R}^{-1}\symbf{X}
                  & \symbf{Z}^\top\symbf{R}^{-1}\symbf{Z}
                    + \symbf{G}^{-1}
              \end{pmatrix},

        the method returns the blocks of :math:`\symbf{C}^{-1}`.  Block
        :math:`\symbf{C}_{11} = (\symbf{X}^\top\symbf{V}^{-1}\symbf{X})^{-1}`
        is the covariance of the fixed-effect estimates and
        :math:`\symbf{C}_{22}` is the prediction-error covariance of the
        random-effect predictions.

        Args:
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            tuple: ``(c11, c12, c21, c22)``, the blocks of
            :math:`\symbf{C}^{-1}`, with shapes ``(p, p)``, ``(p, q)``,
            ``(q, p)`` and ``(q, q)``.

        Raises:
            RuntimeError: If the model has no random-effect term or no
                free parameters.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                c11, c12, c21, c22 = reml.c_inverse_blocks(x, torch.zeros(2))
                print(c11.shape, c12.shape, c21.shape, c22.shape)
        """
        x = self._post_x(x)
        free_params = self._post_free_params(free_params)
        if self._cform is None:
            self._cform = CForm()
        return self._cform.c_inverse_blocks(self, x, free_params)

    def marginal_predict(self, y=None, x=None, free_params=None):
        r"""Marginal fitted values at ``free_params``.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            torch.Tensor: Marginal fitted values of shape ``(n,)``.
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)
        return marginal_predict(y, x, self.V(free_params))

    def marginal_residual(self, y=None, x=None, free_params=None):
        r"""Marginal residuals at ``free_params``.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            torch.Tensor: Marginal residuals of shape ``(n,)``.
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)
        return marginal_residual(y, x, self.V(free_params))

    def blup(self, y=None, x=None, free_params=None):
        r"""Best linear unbiased predictions of the random effects.

        .. math::

            \hat{\symbf{b}}
            = \symbf{G}\symbf{Z}^{\top}\symbf{V}^{-1}
              (\symbf{y} - \symbf{X}\hat{\boldsymbol{\beta}}).

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used. A ``ValueError`` is raised when no fit
                exists.

        Returns:
            torch.Tensor: Random-effect predictions of shape ``(q,)``.

        Raises:
            ValueError: If the model has no random-effect terms, so there is
                no random-effect vector to predict.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.blup(y, x, torch.zeros(2))
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)
        if self._z is None:
            raise ValueError("Cannot compute BLUP: the model has no random-effect terms.")

        kz = self._z.num_free_params if isinstance(self._z, Matrix) else 0
        kg = self._g.num_free_params if self._g is not None else 0
        theta_z = free_params[:kz]
        theta_g = free_params[kz:kz + kg]

        z = self._z(theta_z) if isinstance(self._z, Matrix) else self._z
        g = self._g(theta_g) if self._g is not None else None
        return blup(y, x, z, g, self.V(free_params))

    def cov_blup(self, x=None, free_params=None):
        r"""Covariance matrix of the BLUPs.

        The BLUP is :math:`\hat{\symbf{b}}
        = \symbf{G}\symbf{Z}^{\top}\symbf{P}\symbf{y}`, so its covariance is

        .. math::

            \operatorname{Var}(\hat{\symbf{b}})
            = \symbf{G}\symbf{Z}^{\top}\symbf{P}\symbf{Z}\symbf{G},

        with :math:`\symbf{P} = \symbf{V}^{-1}
        - \symbf{V}^{-1}\symbf{X}
          (\symbf{X}^{\top}\symbf{V}^{-1}\symbf{X})^{-1}
          \symbf{X}^{\top}\symbf{V}^{-1}`.
        This is equal to :math:`\symbf{G} - \symbf{C}_{22}`, but it only
        requires :math:`\symbf{V}` to be invertible, not :math:`\symbf{G}`.
        The covariance does not depend on the response :math:`\symbf{y}`,
        only on the design and the covariance parameters.

        Args:
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used. A ``ValueError`` is raised when no fit
                exists.

        Returns:
            torch.Tensor: BLUP covariance matrix of shape ``(q, q)``.

        Raises:
            ValueError: If the model has no random-effect terms, so there is
                no random-effect vector to predict.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.cov_blup(x, torch.zeros(2))
        """
        x = self._post_x(x)
        free_params = self._post_free_params(free_params)
        if self._z is None:
            raise ValueError("Cannot compute the BLUP covariance: the model has no random-effect terms.")

        kz = self._z.num_free_params if isinstance(self._z, Matrix) else 0
        kg = self._g.num_free_params if self._g is not None else 0
        theta_z = free_params[:kz]
        theta_g = free_params[kz:kz + kg]

        z = self._z(theta_z) if isinstance(self._z, Matrix) else self._z
        g = self._g(theta_g)
        return cov_blup(x, z, g, self.V(free_params))

    def var_blup(self, x=None, free_params=None):
        r"""Variances of the BLUPs.

        This is the diagonal of :meth:`cov_blup`, the sampling variance of
        each random-effect prediction.  The variance does not depend on the
        response :math:`\symbf{y}`, only on the design and the covariance
        parameters.

        Args:
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used. A ``ValueError`` is raised when no fit
                exists.

        Returns:
            torch.Tensor: BLUP variances of shape ``(q,)``.

        Raises:
            ValueError: If the model has no random-effect terms, so there is
                no random-effect vector to predict.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.var_blup(x, torch.zeros(2))
        """
        x = self._post_x(x)
        return torch.diagonal(self.cov_blup(x, free_params))

    def blup_per_term(self, y=None, x=None, free_params=None):
        r"""BLUPs split by random-effect term.

        The combined BLUP is computed once, then sliced into the
        random-effect term order used by the constructor. This shares one
        factorization and one residual solve across all terms.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            dict: Mapping from random-effect term name to its BLUP vector of
            shape ``(q_i,)``.

        Raises:
            ValueError: If the model has no random-effect terms, or if no
                fitted parameters are available.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z1 = DummyMatrix([f"a{i % 5}" for i in range(n)])
                z2 = DummyMatrix([f"b{i % 3}" for i in range(n)])
                reml = MixedModelREML(
                    (z1, ScalarMatrix(5)),
                    (z2, ScalarMatrix(3)),
                    R=ScalarMatrix(n),
                )
                reml.blup_per_term(y, x, torch.zeros(3))
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)

        if self._z is None:
            raise ValueError("Cannot compute BLUP: the model has no random-effect terms.")

        combined = self.blup(y, x, free_params)

        result = {}
        offset = 0
        kz_offset = 0

        for name, design in self._z_terms:
            kz = design.num_free_params if isinstance(design, Matrix) else 0
            theta_z = free_params[kz_offset:kz_offset + kz]
            z = design(theta_z) if isinstance(design, Matrix) else design
            q = z.shape[1]
            result[name] = combined[offset:offset + q]
            offset += q
            kz_offset += kz

        return result

    def cov_blup_per_term(self, x=None, free_params=None):
        r"""BLUP covariance matrices split by random-effect term.

        The combined BLUP covariance is computed once, then sliced into the
        random-effect term order used by the constructor. Each entry is the
        covariance of that term's BLUPs,

        .. math::

            \operatorname{Var}(\hat{\symbf{b}}_i)
            = \symbf{G}_i\symbf{Z}_i^\top\symbf{P}\symbf{Z}_i\symbf{G}_i,

        the corresponding diagonal block of :meth:`cov_blup`. The fit is
        shared across terms, so only one factorization is performed.
        Cross-covariances between terms can be sliced from the full
        :meth:`cov_blup` matrix.

        Args:
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used. A ``ValueError`` is raised when no fit
                exists.

        Returns:
            dict: Mapping from random-effect term name to its covariance
            matrix of shape ``(q_i, q_i)``.

        Raises:
            ValueError: If the model has no random-effect terms, or if no
                fitted parameters are available.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                x = torch.randn(n, 2)
                z1 = DummyMatrix([f"a{i % 5}" for i in range(n)])
                z2 = DummyMatrix([f"b{i % 3}" for i in range(n)])
                reml = MixedModelREML(
                    (z1, ScalarMatrix(5)),
                    (z2, ScalarMatrix(3)),
                    R=ScalarMatrix(n),
                )
                reml.cov_blup_per_term(x, torch.zeros(3))
        """
        x = self._post_x(x)
        free_params = self._post_free_params(free_params)

        if self._z is None:
            raise ValueError("Cannot compute the BLUP covariance: the model has no random-effect terms.")

        combined = self.cov_blup(x, free_params)

        result = {}
        offset = 0
        kz_offset = 0

        for name, design in self._z_terms:
            kz = design.num_free_params if isinstance(design, Matrix) else 0
            theta_z = free_params[kz_offset:kz_offset + kz]
            z = design(theta_z) if isinstance(design, Matrix) else design
            q = z.shape[1]
            result[name] = combined[offset:offset + q, offset:offset + q]
            offset += q
            kz_offset += kz

        return result

    def var_blup_per_term(self, x=None, free_params=None):
        r"""BLUP variances split by random-effect term.

        This is the diagonal of :meth:`cov_blup_per_term`,

        .. math::

            \operatorname{Var}(\hat{b}_{ij})
            = \left(\symbf{G}_i\symbf{Z}_i^\top\symbf{P}
              \symbf{Z}_i\symbf{G}_i\right)_{jj},

        the sampling variance of each random-effect prediction, grouped by
        the term that produced it. The combined covariance is computed once
        and shared across terms.

        Args:
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used. A ``ValueError`` is raised when no fit
                exists.

        Returns:
            dict: Mapping from random-effect term name to its variance vector
            of shape ``(q_i,)``.

        Raises:
            ValueError: If the model has no random-effect terms, or if no
                fitted parameters are available.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                x = torch.randn(n, 2)
                z1 = DummyMatrix([f"a{i % 5}" for i in range(n)])
                z2 = DummyMatrix([f"b{i % 3}" for i in range(n)])
                reml = MixedModelREML(
                    (z1, ScalarMatrix(5)),
                    (z2, ScalarMatrix(3)),
                    R=ScalarMatrix(n),
                )
                reml.var_blup_per_term(x, torch.zeros(3))
        """
        x = self._post_x(x)
        return {
            name: torch.diagonal(block)
            for name, block in self.cov_blup_per_term(x, free_params).items()
        }

    def predict(self, y=None, x=None, free_params=None):
        r"""Conditional fitted values including random effects.

        For a model with no random-effect terms this reduces to
        :meth:`marginal_predict`.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used. A ``ValueError`` is raised when no fit
                exists.

        Returns:
            torch.Tensor: Conditional fitted values of shape ``(n,)``.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.predict(y, x, torch.zeros(2))
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)
        if self._z is None:
            return self.marginal_predict(y, x, free_params)

        kz = self._z.num_free_params if isinstance(self._z, Matrix) else 0
        kg = self._g.num_free_params if self._g is not None else 0
        theta_z = free_params[:kz]
        theta_g = free_params[kz:kz + kg]

        z = self._z(theta_z) if isinstance(self._z, Matrix) else self._z
        g = self._g(theta_g) if self._g is not None else None
        return conditional_predict(y, x, z, g, self.V(free_params))

    def predict_new(self, x, z=None, free_params=None):
        r"""Predictions at new rows from already-built design matrices.

        This method only consumes materialised tensors: it does not encode
        factors or construct the incidence matrix of new data. Build ``x``
        and each ``z`` block in the training column order, then pass them
        here. A single new observation is just a one-row batch. The
        random-effect contribution uses the BLUPs computed from the stored
        training data by :meth:`blup`.

        Args:
            x (torch.Tensor): Fixed-effect design matrix of shape
                ``(n_new, p)`` with at least one row. It must have the same
                columns as the training design, in the same order.
            z (torch.Tensor, list, dict or None, optional): Incidence matrix
                of new data. Either a full tensor of shape ``(n_new, q)``, a
                list with one ``(n_new, q_i)`` block per random-effect term,
                or a dict keyed by term name. Terms omitted from a dict
                contribute zero, so the caller chooses the active terms by
                what it supplies. ``None`` gives the population prediction,
                ignoring the random effects. Default: ``None``.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used.

        Returns:
            torch.Tensor: Predictions of shape ``(n_new,)``.

        Raises:
            ValueError: If ``x`` is empty, the shapes disagree, a term name
                is unknown, or the model has no random-effect terms but
                ``z`` is given.
            TypeError: If ``x`` or ``z`` has an unsupported type.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.optimize(y, x, torch.zeros(2))

                x_new = torch.zeros(2, 2)
                z_new = torch.zeros(2, 5)
                z_new[0, 0] = 1.0
                reml.predict_new(x_new, z_new)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("`x` must be a torch.Tensor.")
        if x.dim() != 2:
            raise ValueError(f"`x` must be a two-dimensional tensor of shape (n_new, p); got {tuple(x.shape)}.")
        if x.shape[0] == 0:
            raise ValueError("`x` must contain at least one row.")

        free_params = self._post_free_params(free_params)
        beta = self.blue(free_params=free_params)
        if x.shape[1] != beta.shape[0]:
            raise ValueError(f"`x` must have {beta.shape[0]} columns; got {x.shape[1]}.")

        device, dtype = beta.device, beta.dtype
        predicted = x.to(device=device, dtype=dtype) @ beta

        if z is None:
            return predicted
        if not self._z_terms:
            raise ValueError("The model has no random-effect terms; pass `z=None`.")

        layout = []
        offset = 0
        num_params_offset = 0
        for name, design in self._z_terms:
            num_params = design.num_free_params if isinstance(design, Matrix) else 0
            if isinstance(design, Matrix):
                theta = free_params[num_params_offset:num_params_offset + num_params]
                block = design(theta)
            else:
                block = design
            q = block.shape[1]
            layout.append((name, q, num_params, offset))
            offset += q
            num_params_offset += num_params

        n_new = x.shape[0]
        total = layout[-1][3] + layout[-1][1]
        names = {name for name, _, _, _ in layout}
        blocks = {}
        if isinstance(z, torch.Tensor):
            if z.dim() != 2 or z.shape != (n_new, total):
                raise ValueError(f"`z` must have shape ({n_new}, {total}); got {tuple(z.shape)}.")
            for name, q, _, offset in layout:
                blocks[name] = z[:, offset:offset + q]
        elif isinstance(z, (list, tuple)):
            if len(z) != len(layout):
                raise ValueError(f"`z` must contain one block per random-effect term: expected {len(layout)}, got {len(z)}.")
            for (name, q, _, _), block in zip(layout, z):
                if not isinstance(block, torch.Tensor) or block.dim() != 2 or block.shape != (n_new, q):
                    raise ValueError(f"Term `{name}` expects a block of shape ({n_new}, {q}).")
                blocks[name] = block
        elif isinstance(z, dict):
            unknown = sorted(set(z) - names)
            if unknown:
                raise ValueError(f"Unknown random-effect term(s) in `z`: {unknown}; expected a subset of {sorted(names)}.")
            for name, q, _, _ in layout:
                if name in z:
                    block = z[name]
                    if not isinstance(block, torch.Tensor) or block.dim() != 2 or block.shape != (n_new, q):
                        raise ValueError(f"Term `{name}` expects a block of shape ({n_new}, {q}).")
                    blocks[name] = block
        else:
            raise TypeError("`z` must be a tensor, a list of tensors or a dict of tensors.")

        z_full = torch.zeros(n_new, total, device=device, dtype=dtype)
        for name, q, _, offset in layout:
            if name in blocks:
                z_full[:, offset:offset + q] = blocks[name].to(device=device, dtype=dtype)
        b_full = self.blup(free_params=free_params).to(device=device, dtype=dtype)
        return predicted + z_full @ b_full

    def residual(self, y=None, x=None, free_params=None):
        r"""Conditional residuals after removing fixed and random effects.

        For a model with no random-effect terms this reduces to
        :meth:`marginal_residual`.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters in model order, or ``"last"`` / ``"best"`` to
                select a fitted iterate. If omitted, the last fitted
                parameters are used. A ``ValueError`` is raised when no fit
                exists.

        Returns:
            torch.Tensor: Conditional residuals of shape ``(n,)``.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.predict(y, x, torch.zeros(2))
                reml.residual(y, x, torch.zeros(2))
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)
        if self._z is None:
            return self.marginal_residual(y, x, free_params)

        kz = self._z.num_free_params if isinstance(self._z, Matrix) else 0
        kg = self._g.num_free_params if self._g is not None else 0
        theta_z = free_params[:kz]
        theta_g = free_params[kz:kz + kg]

        z = self._z(theta_z) if isinstance(self._z, Matrix) else self._z
        g = self._g(theta_g) if self._g is not None else None
        return conditional_residual(y, x, z, g, self.V(free_params))

    def get_free_params(self, select="last"):
        r"""Return a covariance-parameter iterate from the history.

        Args:
            select (str, optional): ``"last"`` for the final iterate or
                ``"best"`` for the iterate with the highest recorded
                log-likelihood.  Default: ``"last"``.

        Returns:
            tuple: ``(free_param_names, free_params)``, where
            ``free_param_names`` is the list of namespaced covariance
            parameter names and ``free_params`` is the selected
            covariance-parameter tensor.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                reml.optimize(y, x, torch.zeros(2), max_iter=20)
                names, values = reml.get_free_params("best")
                names, values
        """
        if select == "last":
            index = -1
        elif torch.is_tensor(self.history["loglik"][-1]):
            index = torch.argmax(torch.stack(self.history["loglik"])).item()
        else:
            index = -1
        return self.V.free_param_names, self.history["free_params"][index]

    def build_params(self, free_params=None, include_fixed=True, trans=True, out_format="tensor"):
        r"""Build covariance parameters from free parameters.

        This exposes the covariance model's ``build_params`` on the fitted
        model, using the selected model parameters when ``free_params`` is
        omitted.

        Args:
            free_params (torch.Tensor, dict or str, optional): Free
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate.  If omitted, the last fitted parameters are used.
            include_fixed (bool, optional): Whether to include fixed
                parameters in the output.  Default: ``True``.
            trans (bool, optional): Whether to apply parameter transforms.
                Default: ``True``.
            out_format (str, optional): ``"tensor"`` for a flat tensor or
                ``"dict"`` for a name-keyed mapping.  Default: ``"tensor"``.

        Returns:
            torch.Tensor or dict: The requested covariance parameters.

        Raises:
            ValueError: If no fitted parameters are available, or if
                ``out_format`` is invalid.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 30
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 5}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
                raw = reml.build_params(torch.zeros(2), trans=False)
                model_scale = reml.build_params(torch.zeros(2), trans=True)
                raw, model_scale
        """
        free_params = self._post_free_params(free_params)
        return self.V.build_params(
            free_params,
            include_fixed=include_fixed,
            trans=trans,
            out_format=out_format,
        )

    def tree(self, free_params=None):
        r"""Evaluate the covariance structure and return its path-keyed tree.

        Args:
            free_params (torch.Tensor, dict or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate.  If omitted, the last fitted parameters are used.

        Returns:
            dict: Path-keyed covariance tree.

        Raises:
            ValueError: If no fitted parameters are available.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml import MixedModelREML
                from torch_openreml.covariance import DummyMatrix, ScalarMatrix

                torch.manual_seed(0)
                n = 8
                y = torch.randn(n)
                x = torch.randn(n, 2)
                z = DummyMatrix([f"g{i % 3}" for i in range(n)])
                reml = MixedModelREML((z, ScalarMatrix(3)), R=ScalarMatrix(n))
                reml.optimize(y, x, torch.zeros(2), max_iter=20)
                reml.tree()
        """
        free_params = self._post_free_params(free_params)

        if isinstance(self.V, Operator):
            return self.V.tree(free_params)
        return {"/": self.V(free_params)}

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

    def loglik(self, y=None, x=None, free_params=None):
        r"""Evaluate the restricted log-likelihood with the selected method.

        Args:
            y (torch.Tensor, optional): Response vector of shape ``(n,)``.
                Default: the response stored by :meth:`optimize`.
            x (torch.Tensor, optional): Fixed-effect design matrix of shape ``(n, p)``.
                Default: the design stored by :meth:`optimize`.
            free_params (torch.Tensor or str, optional): Covariance
                parameters, or ``"last"`` / ``"best"`` to select a fitted
                iterate. If omitted, the last fitted parameters are used.
                A ``ValueError`` is raised when no fit exists.

        Returns:
            torch.Tensor: Scalar restricted log-likelihood.

        Raises:
            RuntimeError: If ``method="cform"`` and CForm cannot handle the
                model.
        """
        y, x = self._post_xy(y, x)
        free_params = self._post_free_params(free_params)
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
        armijo_slack=1e-8,
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

        where ``c = armijo_c`` and ``epsilon = armijo_slack``.  The last term is
        a small numerical slack; set ``armijo_slack=0`` for strict Armijo.

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
            armijo_slack (float, optional): Relative numerical slack in the
                Armijo test.  Default: ``1e-8``.

        Returns:
            tuple: ``(free_param_names, free_params)``, where
            ``free_param_names`` is the list of namespaced covariance
            parameter names and ``free_params`` is the final
            covariance-parameter tensor in that order.

        Notes:
            The optimisation trace is stored in :attr:`history` with keys
            ``"free_params"``, ``"loglik"``, ``"score"``, ``"ai"``,
            ``"delta"``, ``"update"``, ``"eta"`` and ``"accepted"``.
            ``"accepted"`` is ``True`` only for a line-search-accepted Newton
            step; gradient fallback steps are recorded with ``False``.
            The response ``y`` and design ``x`` are stored on the instance and
            are reused by the post-estimation methods (:meth:`blue`,
            :meth:`blup`, :meth:`predict`, :meth:`residual`, :meth:`loglik`,
            and the covariance methods) when called without them.
        """
        eta = eta_max
        free_params = self.V.build_params(init_values, include_fixed=False, trans=False).detach()
        y = y.to(device=free_params.device, dtype=free_params.dtype)
        x = x.to(device=free_params.device, dtype=free_params.dtype)
        self._y = y.detach()
        self._x = x.detach()

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

        with torch.no_grad():
            for i in range(max_iter):
                score, ai, ll = self.ai_step(y, x, free_params, require_loglik=True)

                delta = torch.linalg.lstsq(ai, score.unsqueeze(-1)).solution.squeeze(-1)

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

                    threshold = ll + armijo_c * eta_i * torch.dot(score, delta) - armijo_slack * torch.clamp(torch.abs(ll), min=1.0)
                    if torch.isfinite(ll_new) and ll_new >= threshold:
                        accepted = True
                        break

                    eta_i *= eta_shrink

                if not accepted:
                    grad = score / torch.clamp(torch.linalg.vector_norm(score), min=1.0)
                    theta_new, step = self.update(theta_old, grad, fallback_eta, lb, ub)
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

                if verbose > 0:
                    print(
                        f"[{i + 1:3d}] \u2225\u2207\u2225: "
                        f"{torch.norm(score):12.4f}, \u2225\u0394\u2225: "
                        f"{torch.norm(delta):6.4f}, \u03B7: {eta_used:.2f}, "
                        f"\u2225\u0394\u1D9C\u2225: {torch.norm(step):6.4f}, "
                        f"accepted: {str(accepted):<5}, log \U0001D4DB: {ll_new:8.4f}"
                    )

                if self.is_converged():
                    if verbose > 0:
                        print(
                            "\n[\u2207: score, \u0394: \U0001D409\u207B\u00B9\u2207, "
                            "\u03B7: learning rate, \u0394\u1D9C: "
                            "clip(\U0001D6C9 + \u03B7\u0394, lb, ub) - \U0001D6C9, "
                            "\U0001D4DB: restricted likelihood]"
                        )
                        print(
                            f"\n\u2713 Converged at iteration {i + 1}, total time: "
                            f"{time.perf_counter() - start_time:.2f}s\n"
                        )
                    break
            else:
                if verbose > 0:
                    print(
                        f"Reached maximum iterations ({max_iter}), total time: "
                        f"{time.perf_counter() - start_time:.2f}s"
                    )

        return self.V.free_param_names, free_params

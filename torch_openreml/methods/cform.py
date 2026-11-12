r"""Dense C-form REML score and average information for one random effect.

This module implements the dense C-form of the average-information (AI)
REML algorithm for the linear mixed model

.. math::

    \symbf{y}
    = \symbf{X}\symbf{\beta} + \symbf{Z}\symbf{u} + \symbf{\epsilon},
    \qquad
    \symbf{u} \sim N(\symbf{0}, \symbf{G}),
    \qquad
    \symbf{\epsilon} \sim N(\symbf{0}, \symbf{R}),

where :math:`\symbf{X}` is the fixed-effect design matrix,
:math:`\symbf{Z}` is the incidence matrix for a single random-effect
block, and :math:`\symbf{G}` and :math:`\symbf{R}` are the corresponding
covariance matrices.

The C-form works with the combined design matrix and coefficient matrix

.. math::

    \symbf{W} = [\symbf{X}, \symbf{Z}],
    \qquad
    \symbf{C}
    = \symbf{W}^\top \symbf{R}^{-1} \symbf{W}
      + \operatorname{blockdiag}(\symbf{0}, \symbf{G}^{-1}),
    \qquad
    \symbf{U} = \symbf{R}^{-1}\symbf{W}.

The REML projection matrix is then obtained without forming the
:math:`n \times n` marginal covariance :math:`\symbf{V}`:

.. math::

    \symbf{P}
    = \symbf{R}^{-1}
      - \symbf{U}\symbf{C}^{-1}\symbf{U}^\top.

The restricted log-likelihood is

.. math::

    \ell
    = -\frac{1}{2}\left(
        \log|\symbf{C}|
        + \log|\symbf{R}|
        + \log|\symbf{G}|
        + \symbf{y}^\top\symbf{P}\symbf{y}
      \right),

and the score and AI matrix follow from the derivatives of
:math:`\symbf{P}`; see :meth:`CForm.ai_step`.

The inverse of the coefficient matrix is partitioned into the
fixed-effect and random-effect blocks,

.. math::

    \symbf{C}^{-1}
    = \begin{pmatrix}
          \symbf{C}_{11} & \symbf{C}_{12} \\
          \symbf{C}_{21} & \symbf{C}_{22}
      \end{pmatrix},

where :math:`\symbf{C}_{11} = (\symbf{X}^\top\symbf{V}^{-1}\symbf{X})^{-1}`
is the covariance of the fixed-effect estimates and :math:`\symbf{C}_{22}`
is the prediction-error covariance of the random-effect predictions.
:meth:`CForm.c_inverse_blocks` returns these blocks.

The first dimension of ``dz``, ``dg`` and ``dr`` is the parameter
dimension.  A block with no free parameters is represented by a
zero-length first dimension, for example ``dz.shape == (0, n, q)``.
At least one of ``dz``, ``dg`` and ``dr`` must contain a parameter.

Classes:
    CForm
"""

import torch

from torch_openreml.covariance import Matrix


class CForm:
    r"""Dense C-form REML score and AI for one random-effect block.

    The class is stateless.  :meth:`ai_step` is the model-aware entry point:
    it extracts :math:`\symbf{Z}`, :math:`\symbf{G}`, :math:`\symbf{R}` and
    their derivatives from the model, applies the residual jitter rule, and
    evaluates the score, AI matrix and optional restricted log-likelihood.

    :meth:`preprocess_input` exposes that extraction separately and returns a
    dictionary of tensors.  :meth:`ai_step_from_tensors` is the standalone
    tensor-level kernel; it builds :math:`\symbf{W}`,
    :math:`\symbf{C}` and :math:`\symbf{P}` and evaluates the score and AI
    matrix directly from ``x``, ``y``, ``z``, ``g``, ``r`` and their
    derivatives.

    :meth:`c_inverse_blocks` and :meth:`c_inverse_blocks_from_tensors` return
    the blocks of the inverse coefficient matrix :math:`\symbf{C}^{-1}`,
    which give the fixed-effect covariance and the random-effect
    prediction-error covariance.

    This implementation is for dense problems where the
    :math:`n \times n` matrices fit in memory.  It has no sparse
    factorisation or matrix-free path.
    """

    def eligible(self, model):
        """Whether ``model`` can use the dense C-form method."""
        return (
            model is not None
            and model.Z is not None
            and model.G is not None
            and model.V.num_free_params > 0
        )

    def loglik(self, model, y, x, free_params):
        r"""Evaluate the restricted log-likelihood for ``model``.

        This is the C-form log-likelihood

        .. math::

            \ell
            = -\frac{1}{2}\left(
                \log|\symbf{C}|
                + \log|\symbf{R}|
                + \log|\symbf{G}|
                + \symbf{y}^\top\symbf{P}\symbf{y}
              \right),

        evaluated without forming :math:`\symbf{V}` or :math:`\symbf{P}`.

        Args:
            model (MixedModelREML): Model supplying ``Z``, ``G`` and ``R``.
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Free parameters of the model, ordered as
                ``[free_params_Z, free_params_G, free_params_R]``.

        Returns:
            torch.Tensor: Scalar restricted log-likelihood.

        Raises:
            RuntimeError: If ``model`` has no random-effect term or no
                free parameters.
        """
        if not self.eligible(model):
            raise RuntimeError("CForm requires at least one random-effect term with free parameters.")
        tensor_dict = self.preprocess_input(model, y, x, free_params, grad=False)
        return self.loglik_from_tensors(**tensor_dict)

    def loglik_from_tensors(self, x, y, z, g, r, g_inv=None, r_inv=None):
        r"""Evaluate the C-form restricted log-likelihood from tensors.

        Args:
            x: Fixed-effect design matrix of shape ``(n, p)``.
            y: Response vector of shape ``(n,)``.
            z: Random-effect incidence matrix of shape ``(n, q)``.
            g: Random-effect covariance matrix of shape ``(q, q)``.
            r: Residual covariance matrix of shape ``(n, n)``.
            g_inv: Optional precomputed inverse of ``g``.  When supplied it
                is used for the C-form block instead of forming the inverse
                from a Cholesky factorisation.
            r_inv: Optional precomputed inverse of ``r``.  When supplied it
                is used for the C-form block instead of forming the inverse
                from a Cholesky factorisation.

        Returns:
            torch.Tensor: Scalar restricted log-likelihood.
        """
        matrix = {"X": x, "Y": y, "Z": z, "G": g, "R": r}
        scalar = {}
        vector = {}

        if r_inv is None:
            matrix["L_R"] = torch.linalg.cholesky(matrix["R"])
            matrix["R^{-1}"] = torch.cholesky_inverse(matrix["L_R"])
        else:
            matrix["R^{-1}"] = r_inv
            matrix["L_R"] = torch.linalg.cholesky(matrix["R"])
        scalar["logdet_R"] = 2.0 * torch.log(torch.diagonal(matrix["L_R"])).sum()

        if g_inv is None:
            matrix["L_G"] = torch.linalg.cholesky(matrix["G"])
            matrix["G^{-1}"] = torch.cholesky_inverse(matrix["L_G"])
        else:
            matrix["G^{-1}"] = g_inv
            matrix["L_G"] = torch.linalg.cholesky(matrix["G"])
        scalar["logdet_G"] = 2.0 * torch.log(torch.diagonal(matrix["L_G"])).sum()

        scalar["P"] = matrix["X"].shape[1]
        matrix["W"] = torch.cat([matrix["X"], matrix["Z"]], dim=1)
        matrix["U"] = matrix["R^{-1}"] @ matrix["W"]
        matrix["C"] = matrix["W"].T @ matrix["R^{-1}"] @ matrix["W"]
        matrix["C"][scalar["P"]:, scalar["P"]:] += matrix["G^{-1}"]
        matrix["L_C"] = torch.linalg.cholesky(matrix["C"])
        scalar["logdet_C"] = 2.0 * torch.log(torch.diagonal(matrix["L_C"])).sum()

        vector["W^T R^{-1} y"] = matrix["W"].T @ (matrix["R^{-1}"] @ matrix["Y"])
        scalar["quad"] = matrix["Y"] @ (matrix["R^{-1}"] @ matrix["Y"]) - vector["W^T R^{-1} y"] @ torch.cholesky_solve(vector["W^T R^{-1} y"].unsqueeze(-1), matrix["L_C"]).squeeze(-1)
        scalar["loglik"] = -0.5 * (scalar["logdet_C"] + scalar["logdet_R"] + scalar["logdet_G"] + scalar["quad"])
        return scalar["loglik"]

    def c_inverse_blocks(self, model, x, free_params):
        r"""Compute the blocks of the inverse coefficient matrix for ``model``.

        The coefficient matrix is

        .. math::

            \symbf{C}
            = \begin{pmatrix}
                  \symbf{X}^\top\symbf{R}^{-1}\symbf{X}
                  & \symbf{X}^\top\symbf{R}^{-1}\symbf{Z} \\
                  \symbf{Z}^\top\symbf{R}^{-1}\symbf{X}
                  & \symbf{Z}^\top\symbf{R}^{-1}\symbf{Z}
                    + \symbf{G}^{-1}
              \end{pmatrix}.

        This is the model-aware wrapper around
        :meth:`c_inverse_blocks_from_tensors`.  It extracts
        :math:`\symbf{Z}`, :math:`\symbf{G}` and :math:`\symbf{R}` from the
        model and applies the same residual jitter rule as
        :meth:`preprocess_input`, so that the blocks are consistent with the
        matrices used by :meth:`ai_step`.

        Args:
            model (MixedModelREML): Model supplying ``Z``, ``G`` and ``R``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Free parameters of the model, ordered as
                ``[free_params_Z, free_params_G, free_params_R]``.

        Returns:
            tuple: ``(c11, c12, c21, c22)``, the blocks of
            :math:`\symbf{C}^{-1}`, with shapes ``(p, p)``, ``(p, q)``,
            ``(q, p)`` and ``(q, q)``.

        Raises:
            RuntimeError: If ``model`` has no random-effect term or no
                free parameters.
        """
        if not self.eligible(model):
            raise RuntimeError("CForm requires at least one random-effect term with free parameters.")

        x = x.to(device=free_params.device, dtype=free_params.dtype)

        z_is_matrix = isinstance(model.Z, Matrix)
        kz = model.Z.num_free_params if z_is_matrix else 0
        kg = model.G.num_free_params
        free_params_g = free_params[kz:kz + kg]
        free_params_r = free_params[kz + kg:]

        z = model.Z(free_params[:kz]) if z_is_matrix else model.Z
        g = model.G(free_params_g)
        r = model.R(free_params_r)

        g_inv = None
        try:
            g_inv = model.G.manual_inverse(free_params_g)
        except NotImplementedError:
            pass

        r_inv = None
        try:
            r_inv = model.R.manual_inverse(free_params_r)
        except NotImplementedError:
            pass
        if r_inv is None:
            n = r.shape[0]
            r = r + 1e-6 * torch.eye(n, dtype=r.dtype, device=r.device)

        return self.c_inverse_blocks_from_tensors(x, z, g, r, g_inv=g_inv, r_inv=r_inv)

    @torch.no_grad()
    def c_inverse_blocks_from_tensors(self, x, z, g, r, g_inv=None, r_inv=None):
        r"""Compute the blocks of :math:`\symbf{C}^{-1}` from tensors.

        The coefficient matrix is

        .. math::

            \symbf{C}
            = \begin{pmatrix}
                  \symbf{X}^\top\symbf{R}^{-1}\symbf{X}
                  & \symbf{X}^\top\symbf{R}^{-1}\symbf{Z} \\
                  \symbf{Z}^\top\symbf{R}^{-1}\symbf{X}
                  & \symbf{Z}^\top\symbf{R}^{-1}\symbf{Z}
                    + \symbf{G}^{-1}
              \end{pmatrix},

        and the returned blocks satisfy

        .. math::

            \symbf{C}^{-1}
            = \begin{pmatrix}
                  \symbf{C}_{11} & \symbf{C}_{12} \\
                  \symbf{C}_{21} & \symbf{C}_{22}
              \end{pmatrix}.

        Args:
            x: Fixed-effect design matrix of shape ``(n, p)``.
            z: Random-effect incidence matrix of shape ``(n, q)``.
            g: Random-effect covariance matrix of shape ``(q, q)``.
            r: Residual covariance matrix of shape ``(n, n)``.
            g_inv: Optional precomputed inverse of ``g``.  If ``None``,
                it is computed from a Cholesky factorisation.  A
                supplied value is used as given.
            r_inv: Optional precomputed inverse of ``r``.  If ``None``,
                it is computed from a Cholesky factorisation.  A
                supplied value is used as given.

        Returns:
            tuple: ``(c11, c12, c21, c22)``, the blocks of
            :math:`\symbf{C}^{-1}`, with shapes ``(p, p)``, ``(p, q)``,
            ``(q, p)`` and ``(q, q)``.
        """
        if r_inv is None:
            r_inv = torch.cholesky_inverse(torch.linalg.cholesky(r))
        if g_inv is None:
            g_inv = torch.cholesky_inverse(torch.linalg.cholesky(g))

        p = x.shape[1]
        w = torch.cat([x, z], dim=1)
        c = w.T @ r_inv @ w
        c[p:, p:] += g_inv
        c_inv = torch.cholesky_inverse(torch.linalg.cholesky(c))

        c11 = c_inv[:p, :p]
        c12 = c_inv[:p, p:]
        c21 = c_inv[p:, :p]
        c22 = c_inv[p:, p:]
        return c11, c12, c21, c22

    def preprocess_input(self, model, y, x, free_params, grad=True):
        r"""Prepare the tensors needed by :meth:`ai_step`.

        The model supplies :math:`\symbf{Z}`, :math:`\symbf{G}` and
        :math:`\symbf{R}` together with their free-parameter counts.  The
        parameter vector ``free_params`` is ordered as
        ``[free_params_Z, free_params_G, free_params_R]``.

        Args:
            model (MixedModelREML): Model supplying ``Z``, ``G`` and ``R``.
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Free parameters of the model, ordered as
                ``[free_params_Z, free_params_G, free_params_R]``.
            grad (bool, optional): Whether to compute ``dz``, ``dg``
                and ``dr``.  Set to ``False`` for loglik-only evaluation.
                Default: ``True``.

        Returns:
            dict: Prepared tensors with keys ``x``, ``y``, ``z``, ``g``,
            ``r``, ``g_inv`` and ``r_inv``.  When ``grad`` is ``True`` the
            derivative keys ``dz``, ``dg`` and ``dr`` are included as well;
            when it is ``False`` they are omitted.

        Raises:
            RuntimeError: If ``model`` has no random-effect term or no
                free parameters.
        """
        if not self.eligible(model):
            raise RuntimeError("CForm requires at least one random-effect term with free parameters.")

        y = y.to(device=free_params.device, dtype=free_params.dtype)
        x = x.to(device=free_params.device, dtype=free_params.dtype)

        # free_params splitting
        z_is_matrix = isinstance(model.Z, Matrix)
        kz = model.Z.num_free_params if z_is_matrix else 0
        kg = model.G.num_free_params
        free_params_z = free_params[:kz]
        free_params_g = free_params[kz:kz + kg]
        free_params_r = free_params[kz + kg:]

        # z, g, r
        if z_is_matrix:
            z = model.Z(free_params_z)
        else:
            z = model.Z
        g = model.G(free_params_g)
        r = model.R(free_params_r)
        n = r.shape[0]

        if grad:
            q = g.shape[0]

            # dz, dg, dr
            if z_is_matrix and kz > 0:
                dz = model.Z.grad(free_params_z)[0]
            else:
                dz = z.new_zeros((0, n, q))

            if model.G.num_free_params > 0:
                dg = model.G.grad(free_params_g)[0]
            else:
                dg = free_params_g.new_zeros((0, q, q))

            if model.R.num_free_params > 0:
                dr = model.R.grad(free_params_r)[0]
            else:
                dr = free_params_r.new_zeros((0, n, n))

        # g_inv, r_inv
        g_inv = None
        try:
            g_inv = model.G.manual_inverse(free_params_g)
        except NotImplementedError:
            pass

        r_inv = None
        try:
            r_inv = model.R.manual_inverse(free_params_r)
        except NotImplementedError:
            pass
        if r_inv is None:
            r = r + 1e-6 * torch.eye(n, dtype=r.dtype, device=r.device)

        data = {"x": x, "y": y, "z": z, "g": g, "r": r, "g_inv": g_inv, "r_inv": r_inv}
        if grad:
            data.update({"dz": dz, "dg": dg, "dr": dr})
        return data

    def ai_step(self, model, y, x, free_params, require_loglik=True):
        r"""Evaluate one C-form AI step for ``model``.

        This is the model-aware wrapper around :meth:`ai_step_from_tensors`.
        The score and AI matrix are

        .. math::

            s_k
            = \frac{1}{2}\left[
                \symbf{q}_k^\top\symbf{P}\symbf{y}
                - \operatorname{tr}\left(\symbf{P}\dot{\symbf{V}}_k\right)
              \right],
            \qquad
            \symbf{I}_{kl}
            = \frac{1}{2}\symbf{q}_k^\top\symbf{P}\symbf{q}_l,

        with :math:`\symbf{q}_k = \dot{\symbf{V}}_k\symbf{P}\symbf{y}`.

        Args:
            model (MixedModelREML): Model supplying ``Z``, ``G`` and ``R``.
            y (torch.Tensor): Response vector of shape ``(n,)``.
            x (torch.Tensor): Fixed-effect design matrix of shape ``(n, p)``.
            free_params (torch.Tensor): Flat covariance-parameter tensor.
            require_loglik (bool, optional): Whether to evaluate the REML
                log-likelihood.  Default: ``True``.

        Returns:
            tuple: ``(score, ai, loglik)``.  ``loglik`` is ``torch.nan`` when
            ``require_loglik`` is ``False``.

        Raises:
            RuntimeError: If ``model`` has no random-effect term or no
                free parameters.
        """
        if not self.eligible(model):
            raise RuntimeError("CForm requires at least one random-effect term with free parameters.")
        tensor_dict = self.preprocess_input(model, y, x, free_params)
        score, ai, loglik = self.ai_step_from_tensors(**tensor_dict, require_loglik=require_loglik)
        return score, ai, loglik

    @torch.no_grad()
    def ai_step_from_tensors(self, x, y, z, g, r, dz, dg, dr, g_inv=None, r_inv=None, require_loglik=True):
        r"""Compute the REML score and average information for one AI step.

        The parameters are ordered as ``[Z, G, R]``.  With
        :math:`\symbf{z} = \symbf{P}\symbf{y}`,
        :math:`\symbf{b} = \symbf{Z}^\top\symbf{z}` and
        :math:`\symbf{A} = \symbf{Z}^\top\symbf{P}\symbf{Z}`, the score
        entries are

        .. math::

            s_{Z,k}
            &= \left(\dot{\symbf{Z}}_k^\top\symbf{z}\right)^\top
               \symbf{G}\symbf{b}
               - \operatorname{tr}
                 \left(\symbf{Z}^\top\symbf{P}
                 \dot{\symbf{Z}}_k\symbf{G}\right), \\
            s_{G,k}
            &= \frac{1}{2}\left[
                 \symbf{b}^\top\dot{\symbf{G}}_k\symbf{b}
                 - \operatorname{tr}
                   \left(\symbf{A}\dot{\symbf{G}}_k\right)
               \right], \\
            s_{R,k}
            &= \frac{1}{2}\left[
                 \symbf{z}^\top\dot{\symbf{R}}_k\symbf{z}
                 - \operatorname{tr}
                   \left(\symbf{P}\dot{\symbf{R}}_k\right)
               \right].

        The corresponding working columns are

        .. math::

            \symbf{q}_{Z,k}
            &= \dot{\symbf{Z}}_k\symbf{G}\symbf{b}
               + \symbf{Z}\symbf{G}
                 \dot{\symbf{Z}}_k^\top\symbf{z}, \\
            \symbf{q}_{G,k}
            &= \symbf{Z}\dot{\symbf{G}}_k\symbf{b}, \\
            \symbf{q}_{R,k}
            &= \dot{\symbf{R}}_k\symbf{z},

        and the AI matrix is

        .. math::

            \symbf{I}_{kl}
            = \frac{1}{2}\symbf{q}_k^\top
              \symbf{P}\symbf{q}_l.

        Args:
            x: Fixed-effect design matrix of shape ``(n, p)``.
            y: Response vector of shape ``(n,)``.
            z: Random-effect incidence matrix of shape ``(n, q)``.
            g: Random-effect covariance matrix of shape ``(q, q)``.
            r: Residual covariance matrix of shape ``(n, n)``.
            dz: Derivatives of ``z`` with shape ``(kz, n, q)``.  The
                first dimension is the parameter dimension.
            dg: Derivatives of ``g`` with shape ``(kg, q, q)``.
            dr: Derivatives of ``r`` with shape ``(kr, n, n)``.
            g_inv: Optional precomputed inverse of ``g``.  If ``None``,
                it is computed from a Cholesky factorisation.  A
                supplied value is used as given.
            r_inv: Optional precomputed inverse of ``r``.  If ``None``,
                it is computed from a Cholesky factorisation.  A
                supplied value is used as given.
            require_loglik: If ``True``, return the restricted
                log-likelihood.  If ``False``, return ``torch.nan`` and skip
                the log-determinant calculations.
        Raises:
            ValueError: If ``dz``, ``dg`` and ``dr`` all have a
                zero-length parameter dimension.

        Returns:
            tuple: ``(score, ai, loglik)``, where ``score`` has
            shape ``(kz + kg + kr,)`` in ``[Z, G, R]`` order, ``ai`` has
            shape ``(kz + kg + kr, kz + kg + kr)``, and ``loglik`` is a
            scalar tensor (``torch.nan`` when ``require_loglik`` is
            ``False``).

            Individual empty parameter blocks are supported by passing
            a zero-length first dimension, for example
            ``dz.shape == (0, n, q)``, provided at least one of ``dz``,
            ``dg`` and ``dr`` is non-empty.
        """
        if dz.shape[0] + dg.shape[0] + dr.shape[0] == 0:
            raise ValueError("at least one of dz, dg, or dr must contain a parameter")

        matrix = {}
        scalar = {}
        vector = {}
        tensor3d = {}

        scalar["P"] = x.shape[1]

        matrix["X"] = x
        matrix["Y"] = y.unsqueeze(-1)
        matrix["Z"] = z
        matrix["G"] = g
        matrix["R"] = r
        tensor3d["dZ"] = dz
        tensor3d["dG"] = dg
        tensor3d["dR"] = dr

        # P = R^{-1} - U C^{-1} U',  U = R^{-1} W,  W = [X, Z]
        if r_inv is None:
            matrix["L_R"] = torch.linalg.cholesky(matrix["R"])
            matrix["R^{-1}"] = torch.cholesky_inverse(matrix["L_R"])
            if require_loglik:
                scalar["logdet_R"] = 2.0 * torch.log(torch.diagonal(matrix["L_R"])).sum()
            else:
                scalar["logdet_R"] = None
        else:
            matrix["R^{-1}"] = r_inv
            if require_loglik:
                matrix["L_R"] = torch.linalg.cholesky(matrix["R"])
                scalar["logdet_R"] = 2.0 * torch.log(torch.diagonal(matrix["L_R"])).sum()
            else:
                scalar["logdet_R"] = None

        if g_inv is None:
            matrix["L_G"] = torch.linalg.cholesky(matrix["G"])
            matrix["G^{-1}"] = torch.cholesky_solve(torch.eye(matrix["G"].shape[0], dtype=matrix["G"].dtype, device=matrix["G"].device), matrix["L_G"])
            if require_loglik:
                scalar["logdet_G"] = 2.0 * torch.log(torch.diagonal(matrix["L_G"])).sum()
            else:
                scalar["logdet_G"] = None
        else:
            matrix["G^{-1}"] = g_inv
            if require_loglik:
                matrix["L_G"] = torch.linalg.cholesky(matrix["G"])
                scalar["logdet_G"] = 2.0 * torch.log(torch.diagonal(matrix["L_G"])).sum()
            else:
                scalar["logdet_G"] = None

        matrix["W"] = torch.cat([matrix["X"], matrix["Z"]], dim=1)
        matrix["U"] = matrix["R^{-1}"] @ matrix["W"]
        matrix["C"] = matrix["W"].T @ matrix["R^{-1}"] @ matrix["W"]
        matrix["C"][scalar["P"]:, scalar["P"]:] += matrix["G^{-1}"]
        matrix["L_C"] = torch.linalg.cholesky(matrix["C"])
        matrix["C^{-1} U^T"] = torch.cholesky_solve(matrix["U"].T, matrix["L_C"])
        if require_loglik:
            scalar["logdet_C"] = 2.0 * torch.log(torch.diagonal(matrix["L_C"])).sum()
        else:
            scalar["logdet_C"] = None
        matrix["P"] = matrix["R^{-1}"] - matrix["U"] @ matrix["C^{-1} U^T"]

        matrix["P Y"] = matrix["P"] @ matrix["Y"]

        matrix["Z^T P"] = matrix["Z"].T @ matrix["P"]
        matrix["Z^T P Y"] = matrix["Z^T P"] @ matrix["Y"]
        matrix["Z^T P Z"] = matrix["Z^T P"] @ matrix["Z"]
        matrix["G Z^T P Y"] = matrix["G"] @ matrix["Z^T P Y"]
        matrix["Z G"] = matrix["Z"] @ matrix["G"]

        # Z block
        tensor3d["dZ^T P Y"] = tensor3d["dZ"].mT @ matrix["P Y"]
        tensor3d["dZ G Z^T P Y"] = tensor3d["dZ"] @ matrix["G Z^T P Y"]
        tensor3d["Z G dZ^T P Y"] = matrix["Z G"] @ tensor3d["dZ^T P Y"]
        tensor3d["Q_Z"] = tensor3d["dZ G Z^T P Y"] + tensor3d["Z G dZ^T P Y"]

        vector["quad_Z"] = (tensor3d["dZ^T P Y"] * matrix["G Z^T P Y"]).sum(dim=(-2, -1))
        vector["tr(Z^T P dZ G)"] = (
            matrix["Z^T P"].unsqueeze(0)
            * (tensor3d["dZ"] @ matrix["G"]).transpose(-1, -2)
        ).sum(dim=(-2, -1))
        vector["score_Z"] = vector["quad_Z"] - vector["tr(Z^T P dZ G)"]

        # G block
        tensor3d["dG b"] = tensor3d["dG"] @ matrix["Z^T P Y"]
        tensor3d["Q_G"] = matrix["Z"] @ tensor3d["dG b"]

        vector["quad_G"] = (matrix["Z^T P Y"].mT @ tensor3d["dG b"]).squeeze(-1).squeeze(-1)
        vector["tr(A dG)"] = (
            matrix["Z^T P Z"].unsqueeze(0) * tensor3d["dG"].transpose(-1, -2)
        ).sum(dim=(-2, -1))
        vector["score_G"] = 0.5 * (vector["quad_G"] - vector["tr(A dG)"])

        # R block
        tensor3d["Q_R"] = tensor3d["dR"] @ matrix["P Y"]

        vector["tr(P dR)"] = (
            matrix["P"].unsqueeze(0) * tensor3d["dR"].transpose(-1, -2)
        ).sum(dim=(-2, -1))
        vector["quad_R"] = (matrix["P Y"].mT @ tensor3d["Q_R"]).squeeze(-1).squeeze(-1)
        vector["score_R"] = 0.5 * (vector["quad_R"] - vector["tr(P dR)"])

        # Stack in [Z, G, R] order
        matrix["Q"] = torch.cat(
            [tensor3d["Q_Z"].squeeze(-1), tensor3d["Q_G"].squeeze(-1), tensor3d["Q_R"].squeeze(-1)],
            dim=0,
        )

        vector["score"] = torch.cat([vector["score_Z"], vector["score_G"], vector["score_R"]])

        matrix["AI"] = 0.5 * (matrix["Q"] @ matrix["P"] @ matrix["Q"].T)

        if require_loglik:
            vector["W^T R^{-1} Y"] = matrix["W"].T @ (matrix["R^{-1}"] @ matrix["Y"])
            scalar["quad"] = (matrix["Y"].T @ matrix["R^{-1}"] @ matrix["Y"]).squeeze() - (
                vector["W^T R^{-1} Y"].T
                @ torch.cholesky_solve(vector["W^T R^{-1} Y"], matrix["L_C"])
            ).squeeze()
            scalar["loglik"] = -0.5 * (scalar["logdet_C"] + scalar["logdet_R"] + scalar["logdet_G"] + scalar["quad"])
        else:
            scalar["loglik"] = torch.nan

        return vector["score"], matrix["AI"], scalar["loglik"]

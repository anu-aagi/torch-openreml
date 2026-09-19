"""
Adapter for reparameterising an existing covariance matrix.

This module provides :class:`Adapter`, which wraps an existing
:class:`~torch_openreml.covariance.matrix.Matrix` and exposes a different
set of parameters.  A user-supplied mapping function translates the
adapter's parameters into the wrapped matrix's (adaptee's) parameters,
enabling custom reparameterisations (e.g. combining, splitting, or
constraining parameters).

Classes:
    Adapter:
        Wraps a matrix with a new parameterisation.
"""

import torch
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformIdentity


class Adapter(Matrix):
    r"""
    Reparameterise an existing covariance matrix with a new set of parameters.

    An :class:`Adapter` wraps an *adaptee* matrix and exposes a user-defined
    set of parameters.  A callable ``param_map`` translates the adapter's
    parameter tensor into the parameter tensor expected by the adaptee:

    .. math::
        \symbf{V}(\boldsymbol{\theta}) =
        \symbf{V}_{\text{adaptee}}\!\big(f(\boldsymbol{\theta})\big)

    where :math:`f` is ``param_map`` and :math:`\boldsymbol{\theta}` are the
    adapter's free parameters.

    Gradients are computed via the chain rule: the adaptee's gradient with
    respect to its own parameters is pulled back through the Jacobian of
    ``param_map``.
    """

    _repr_single_line = False

    def __init__(self, adaptee, param_specs, param_map):
        """
        Initialise an adapter wrapping an existing matrix.

        Args:
            adaptee (:class:`~torch_openreml.covariance.matrix.Matrix`):
                The matrix to reparameterise.
            param_specs (dict): Parameter specifications for the adapter's
                own parameters. Keys are parameter names; values are
                dictionaries with keys ``"fixed"``, ``"default"``, and
                ``"trans"``. All parameters should be free and use
                :class:`~torch_openreml.covariance.transform.TransformIdentity`.
            param_map (callable): A function ``f(params) -> adaptee_params``
                that maps the adapter's parameter tensor to the parameter
                tensor expected by ``adaptee``.  Must be differentiable
                (compatible with :attr:`~torch_openreml.covariance.matrix.Matrix.jacobian_method`).

        Raises:
            ValueError: If any parameter specification uses a non-identity
                transform.

        Example:

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import DiagonalMatrix, Adapter
            from torch_openreml.covariance.transform import TransformIdentity

            # Wrap a 2x2 diagonal matrix so that both variances share a
            # single parameter (sum-to-one constraint).
            adaptee = DiagonalMatrix(2)

            def param_map(params):
                # params[0] drives one variance; the other is 1 - params[0]
                p = torch.sigmoid(params[0])
                return torch.stack([p, 1 - p])

            param_specs = {
                "logit": {
                    "fixed": False,
                    "default": torch.tensor([0.0]),
                    "trans": TransformIdentity(),
                }
            }

            adapter = Adapter(adaptee, param_specs, param_map)
            adapter(torch.tensor([0.0]))
        """

        super().__init__(adaptee.shape, param_specs)

        if not all([isinstance(trans, TransformIdentity) for trans in self.param_trans.values()]):
            raise ValueError("Adapter preprocessing does not support parameter transformations other than TransformIdentity!")

        self._adaptee = adaptee
        self._param_map = param_map

    def __call__(self, free_params=None):
        """
        Build the covariance matrix from the adapter's parameters.

        ``free_params`` is checked by
        :meth:`~torch_openreml.covariance.matrix.Matrix.build_params`, called
        with ``include_fixed=False`` and ``trans=False``: only the free
        parameters are kept, and no transform is applied. Those free
        parameters are passed to ``param_map``, and its output builds the
        adaptee's matrix.

        Args:
            free_params (torch.Tensor or dict, optional): Flat 1D tensor of the
                adapter's free parameters, or a parameter dictionary. If
                omitted, default values are used. Default: ``None``.

        Returns:
            torch.Tensor: The covariance matrix built by the adaptee.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor or a dict.
            ValueError: If ``free_params`` is not a 1D tensor or has the wrong
                length, or if it is a dict with missing or unexpected keys.
        """
        if free_params is None:
            free_params = self.free_param_defaults
        params = self.build_params(free_params, include_fixed=False, trans=False)
        adaptee_free_params = self.param_map(params)
        return self.adaptee(adaptee_free_params)

    def auto_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` using automatic
        differentiation.

        Resets the adaptee's intermediate cache before calling the parent
        :meth:`~torch_openreml.covariance.matrix.Matrix.auto_grad`, which
        uses the configured Jacobian method.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.
                If omitted, default values are used. Default: ``None``.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` when the matrix has no free parameters.
        """
        self.adaptee.reset_intermediates()
        return super().auto_grad(free_params)

    def manual_grad(self, free_params=None):
        """
        Compute the Jacobian of :meth:`__call__` via the chain rule.

        The gradient is obtained by pulling back the adaptee's gradient
        through the Jacobian of ``param_map``:

        .. math::
            \\frac{\\partial \\symbf{V}}{\\partial \\boldsymbol{\\theta}}
            = \\sum_k
            \\frac{\\partial \\symbf{V}_{\\text{adaptee}}}{\\partial \\phi_k}
            \\cdot
            \\frac{\\partial \\phi_k}{\\partial \\boldsymbol{\\theta}}

        where :math:`\\boldsymbol{\\phi} = f(\\boldsymbol{\\theta})` are the
        adaptee's parameters. ``free_params`` is checked by
        :meth:`~torch_openreml.covariance.matrix.Matrix.build_params`, called
        with ``include_fixed=False`` and ``trans=False``, and the free
        parameters are then passed to ``param_map`` untransformed.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.
                If omitted, default values are used. Default: ``None``.

        Returns:
            tuple: ``(grad, grad_names)``, where ``grad`` is a 3D tensor of
            shape ``(num_free_params, *shape)`` and
            ``grad_names`` is a list of the corresponding parameter names.
            Returns ``(None, [])`` when the matrix has no free parameters,
            after ``free_params`` is validated via
            :meth:`~torch_openreml.covariance.matrix.Matrix.build_params`.
        """
        if free_params is None:
            free_params = self.free_param_defaults

        params = self.build_params(free_params, include_fixed=False, trans=False)
        if self.num_free_params == 0:
            return None, []

        adaptee_free_params = self.param_map(params)
        adaptee_grad, _ = self.adaptee.grad(adaptee_free_params)

        chunk_size = self.jacobian_chunk_size

        if self.jacobian_method == "jacrev":
            jacobian = torch.func.jacrev(self.param_map, chunk_size=chunk_size)(params)
        elif self.jacobian_method == "jacfwd":
            jacobian = torch.func.jacfwd(self.param_map)(params)
        elif self.jacobian_method == "jacobian":
            jacobian = torch.autograd.functional.jacobian(self.param_map, params)
        else:
            raise ValueError(f"Unknown Jacobian method {self.jacobian_method!r}! Expected one of 'jacrev', 'jacfwd', 'jacobian'.")

        grad = (jacobian[:, :, None, None] * adaptee_grad[:, None, :, :]).sum(dim=0)

        return grad, self.free_param_names

    @property
    def param_map(self):
        """
        callable: The mapping function ``f(params) -> adaptee_params`` that
        translates the adapter's free parameter tensor to the adaptee's
        parameter tensor. The parameters are untransformed.
        """
        return self._param_map

    @property
    def adaptee(self):
        """
        :class:`~torch_openreml.covariance.matrix.Matrix`: The wrapped matrix
        that this adapter reparameterises.
        """
        return self._adaptee

    @property
    def repr_dict(self):
        """
        dict: A dictionary representation for display, containing
        ``param_specs``, ``adaptee``, and a placeholder for ``param_map``.
        """
        return {"param_specs": self.param_specs, "adaptee": self.adaptee, "param_map": "<fn>"}
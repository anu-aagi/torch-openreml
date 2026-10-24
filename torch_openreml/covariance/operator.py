"""
Operator base class for composite covariance matrices.

This module provides an abstract base class for combining multiple
covariance matrices into a single composite structure. Operators
delegate parameter management, transforms, and gradient computation
to their constituent operands, presenting a unified
:class:`~torch_openreml.covariance.matrix.Matrix` interface to the rest
of the library.

Classes:
    Operator:
        Base class for composite covariance matrix operators.
"""

from functools import reduce
from torch_openreml.covariance.matrix import Matrix
import torch

class Operator(Matrix):
    r"""
    Abstract base class for composite covariance matrix operators.

    An operator combines one or more
    :class:`~torch_openreml.covariance.matrix.Matrix` instances and optional
    fixed :class:`torch.Tensor` operands into a single covariance matrix.
    Parameters and gradients are namespaced by operand — a parameter
    ``"sigma^2"`` belonging to operand ``"residual"`` is exposed as
    ``"residual/sigma^2"`` in the combined :attr:`param_names`.

    Operands may be passed as positional arguments, as keyword arguments,
    or as a single dictionary. Mixing positional and keyword arguments is
    not permitted. Positional operands are assigned names ``"op_0"``,
    ``"op_1"``, etc.

    At least one operand must be a
    :class:`~torch_openreml.covariance.matrix.Matrix` instance. Pure-tensor
    operands are treated as fixed matrices with no free parameters.

    Every operand is built on a single dtype and device, resolved once for the
    whole composite:

    1. the input parameters (``free_params``);
    2. the free-parameter defaults of the :class:`Matrix` operands, or all
       parameter defaults when no operand has a free parameter;
    3. the default dtype and device of Torch.

    In case 2 the dtype and the device must be equal across all
    :class:`Matrix` operands, including any nested inside nested operators, or
    a :class:`ValueError` is raised. :class:`Matrix` operands follow the
    resolved dtype and device through the parameter tensor they receive, and
    fixed tensor operands are cast to it.
    """
  
    _repr_single_line = False
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the operator from positional or keyword operands.

        Args:
            *args: Operands as positional arguments, each a
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor`. A single dict argument is also accepted
                and treated as a named operand mapping.
            **kwargs: Operands as keyword arguments, mapping operand names to
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor` instances. Operand names must be non-empty
                and must not contain ``"/"``.

        Raises:
            ValueError: If both positional and keyword arguments are provided,
                or if any operand name is empty or contains ``"/"``.
            TypeError: If ``operands`` is not a dict, if any operand name is not
                a string, if any operand is not a
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor`, or if no operand is a
                :class:`~torch_openreml.covariance.matrix.Matrix`.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import Sum, ScalarMatrix

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                x

            .. jupyter-execute::

                x = Sum(A = ScalarMatrix(2), B = ScalarMatrix(2))
                x

            .. jupyter-execute::

                x.param_names

            .. jupyter-execute::

                x(torch.zeros(2))

            .. jupyter-execute::

                x({"A/sigma^2": torch.zeros(1), "B/sigma^2": torch.zeros(1)})

        """

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError('Operands must be provided either as keyword arguments or as positional arguments, but not both!')

        if len(args) > 0:
            if len(args) == 1 and isinstance(args[0], dict):
                operands = args[0]
            else:
                operands = {f"op_{i}": arg for i, arg in enumerate(args)}
        else:
            operands = kwargs

        self._check_operands(operands)
        self._operands = operands
        
        super().__init__(None, {})
        
    def _check_operands(self, operands):
        """
        Validate the operand dictionary.

        Ensures that all keys are non-empty strings without ``"/"``, all values are
        :class:`~torch_openreml.covariance.matrix.Matrix` or
        :class:`torch.Tensor` instances, and that at least one value is a
        :class:`~torch_openreml.covariance.matrix.Matrix`.

        Args:
            operands (dict): Mapping from operand names to operands.

        Raises:
            TypeError: If ``operands`` is not a dict, if any key is not a
                string, if any value is not a
                :class:`~torch_openreml.covariance.matrix.Matrix` or
                :class:`torch.Tensor`, or if no value is a
                :class:`~torch_openreml.covariance.matrix.Matrix`.
            ValueError: If any key is empty or contains ``"/"``.
        """
        if not isinstance(operands, dict):
            raise TypeError(f"operands must be a dict, got {type(operands).__name__}!")
        
        for key, value in operands.items():
    
            if not isinstance(key, str):
                raise TypeError(f"Operand name must be a string, got {type(key).__name__}!")

            if key == "":
                raise ValueError("Operand name must not be empty!")

            if "/" in key:
                raise ValueError(f"Invalid operand name '{key}': '/' is not allowed!")
    
            if not isinstance(value, (Matrix, torch.Tensor)):
                raise TypeError(
                    f"Operand '{key}' must be a Matrix or torch.Tensor, "
                    f"got {type(value).__name__}!"
                )
                
        if not any(isinstance(v, Matrix) for v in operands.values()):
            raise TypeError("operands must include at least one Matrix!")

    def build_params(self, free_params=None, include_fixed=True, trans=True, out_format="tensor"):
        """
        Construct the full parameter tensor by delegating to each operand.

        Splits ``free_params`` into per-operand slices according to each
        operand's :attr:`~torch_openreml.covariance.matrix.Matrix.num_free_params`,
        calls :meth:`~torch_openreml.covariance.matrix.Matrix.build_params` on
        each :class:`~torch_openreml.covariance.matrix.Matrix` operand, and
        concatenates the results. Fixed tensor operands are skipped.

        Args:
            free_params (torch.Tensor or dict): Flat 1D tensor of free parameters
                or a dictionary mapping parameter names to tensors.
                If omitted, default values are used. Default: ``None``.
            include_fixed (bool, optional): Whether to include fixed parameters in
                the output. Passed through to each operand's
                :meth:`~torch_openreml.covariance.matrix.Matrix.build_params`.
                Default: ``True``.
            trans (bool, optional): Whether to apply parameter transforms to the
                output. Passed through to each operand's
                :meth:`~torch_openreml.covariance.matrix.Matrix.build_params`.
                Default: ``True``.
            out_format (str, optional): Output format. One of ``"tensor"`` or
                ``"dict"``. Default: ``"tensor"``.

        Returns:
            torch.Tensor or dict: Concatenated parameter tensor of length
            :attr:`num_params` (or :attr:`num_free_params` when
            ``include_fixed=False``), or a dictionary mapping namespaced
            parameter names to value tensors.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the wrong
                length, or if ``free_params`` is a dict with missing or
                unexpected keys.
            ValueError: If ``out_format`` is not ``"tensor"`` or ``"dict"``.

        .. jupyter-execute::

            import torch
            from torch_openreml.covariance import Sum, ScalarMatrix

            x = Sum(ScalarMatrix(2), ScalarMatrix(2))
            free_params = torch.tensor([0.0, 0.5])

            x.build_params(free_params)

        .. jupyter-execute::

            x.build_params()

        .. jupyter-execute::

            x.build_params(free_params, trans=False)

        .. jupyter-execute::

            x.build_params(free_params, out_format="dict")
        """
        if free_params is None:
            free_params = self.free_param_defaults
        free_params = self._from_free_param_dict(free_params)
        self._check_param_tensor(free_params, length=self.num_free_params)

        result = []

        for name, operand in self._operands.items():
            if isinstance(operand, Matrix):
                operand_free_params = free_params[0:operand.num_free_params]
                free_params = free_params[operand.num_free_params:]

                result.append(operand.build_params(operand_free_params, include_fixed=include_fixed, trans=trans, out_format="tensor"))

        result = torch.cat(result)

        if out_format == "tensor":
            return result
        elif out_format == "dict":
            if include_fixed:
                param_names = self.param_names
            else:
                param_names = self.free_param_names
            return dict(zip(param_names, result.unsqueeze(-1)))
        else:
            raise ValueError(f"Unexpected 'out_format': {out_format}!")
    
    def build_operands(self, free_params=None):
        """
        Evaluate each operand at the current free parameters.

        Splits ``free_params`` into per-operand slices and calls each
        :class:`~torch_openreml.covariance.matrix.Matrix` operand to
        produce its matrix. Fixed tensor operands are cast to the dtype and
        the device of ``free_params``.

        Args:
            free_params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_free_params`.
                If omitted, default values are used. Default: ``None``.

        Returns:
            list of torch.Tensor: Evaluated operand matrices in the same
            order as :attr:`operands`.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import Sum, ScalarMatrix

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                v_groups = x.build_operands(torch.tensor([1.0, 2.0]))
                print(v_groups[0])
                print(v_groups[1])
        """
        if free_params is None:
            free_params = self.free_param_defaults
        free_params = self._from_free_param_dict(free_params)
        device, dtype = self._check_param_tensor(free_params, length=self.num_free_params)

        v_groups = []

        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                operand_params = free_params[0:operand.num_free_params]
                free_params = free_params[operand.num_free_params:]

                v_groups.append(operand(operand_params))
            else:
                v_groups.append(operand.to(device=device, dtype=dtype))
        
        return v_groups

    def walk(self, free_params=None):
        """
        Validate ``free_params`` and list every node of the operand tree.

        Walks the structure once, in the order :meth:`tree`,
        :meth:`free_param_tree` and :meth:`grad_tree` report it: the root
        first, then depth first in operand order. Nothing is evaluated: the walk
        only validates the parameters, splits them, and reads the operand
        structure, so it costs no evaluation however deep the tree is.

        Args:
            free_params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_free_params`.
                If omitted, default values are used. Default: ``None``.

        Returns:
            list: ``(path, operand, free_params)`` triples — ``free_params``
            being the 1D slice the node received, or ``None`` for a fixed tensor
            operand. The first entry is always ``("/", self, free_params)``:
            the operator itself and the validated input tensor, whose dtype and
            device are the resolved ones.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the wrong
                length, or if ``free_params`` is a dict with missing or
                unexpected keys.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

                op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
                         extra=ScalarMatrix(4))
                nodes = op.walk(torch.tensor([0.0, 0.5, 1.0, 0.5]))

            .. jupyter-execute::

                [(path, operand, free_params) for path, operand, free_params in nodes]
        """
        if free_params is None:
            free_params = self.free_param_defaults
        free_params = self._from_free_param_dict(free_params)
        self._check_param_tensor(free_params, length=self.num_free_params)

        nodes = [("/", self, free_params)]

        def visit(operator, params, prefix):
            for name, operand in operator.operands.items():
                if isinstance(operand, Matrix):
                    operand_params = params[0:operand.num_free_params]
                    params = params[operand.num_free_params:]
                else:
                    operand_params = None

                path = f"{prefix}{name}"
                nodes.append((path, operand, operand_params))

                if isinstance(operand, Operator):
                    visit(operand, operand_params, f"{path}/")

        visit(self, free_params, "")

        return nodes

    def tree(self, free_params=None):
        """
        Evaluate this operator and every nested operand, returning each result
        keyed by its path.

        Descends the operand structure, reporting the root and every nested
        :class:`~torch_openreml.covariance.operator.Operator` operand's own
        evaluation. Paths join the operand names leading to a node with
        ``"/"``, and ``"/"`` is the root. ``results`` maps each path to the
        matrix the node evaluated to, and the root result is the same matrix
        :meth:`__call__` returns. :meth:`free_param_tree` reports the free
        parameters each node received, for the same paths in the same order, and
        :meth:`walk` lists the nodes themselves.

        Args:
            free_params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_free_params`.
                If omitted, default values are used. Default: ``None``.

        Returns:
            dict: Maps each path to the matrix the node evaluated to.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the wrong
                length, or if ``free_params`` is a dict with missing or
                unexpected keys.

        Note:
            Only nested :class:`~torch_openreml.covariance.operator.Operator`
            operands are descended into. Every other
            :class:`~torch_openreml.covariance.matrix.Matrix` appears as a leaf,
            including :class:`~torch_openreml.covariance.adapter.Adapter` with
            its adaptee.

        Note:
            Cost: a node's own result comes from its :meth:`__call__` while its
            operands are evaluated separately, so every node below the root is
            entered twice — once while its parent composes its result, once by
            the descent. The root is entered once. An operator that caches
            intermediates serves the second entry from its cache and
            re-evaluates nothing below it; one that does not cache propagates
            the re-evaluation down. A preceding :meth:`__call__` therefore
            lowers the leaf count under caching operators and changes no value.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

                op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
                         extra=ScalarMatrix(4))
                results = op.tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))

            .. jupyter-execute::

                results

            .. jupyter-execute::

                results["inner/op_0"]

            .. jupyter-execute::

                op.free_param_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        """
        nodes = self.walk(free_params)
        device, dtype = nodes[0][2].device, nodes[0][2].dtype

        return {
            path: operand(operand_params)
            if isinstance(operand, Matrix)
            else operand.to(device=device, dtype=dtype)
            for path, operand, operand_params in nodes
        }

    def free_param_tree(self, free_params=None):
        """
        Report the free parameters each node of the operand tree receives.

        Splits ``free_params`` exactly as :meth:`walk` splits it for
        :meth:`tree` and :meth:`grad_tree`, without evaluating anything, and
        returns one entry per path those methods produce — the same keys in the
        same order, the root ``"/"`` first.

        Args:
            free_params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_free_params`.
                If omitted, default values are used. Default: ``None``.

        Returns:
            dict: Maps each path to the 1D free parameter slice the node
            received, or to ``None`` for fixed tensor operands. The root entry
            is the validated ``free_params`` tensor and the remaining entries
            are views of it.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the wrong
                length, or if ``free_params`` is a dict with missing or
                unexpected keys.

        Note:
            Only nested :class:`~torch_openreml.covariance.operator.Operator`
            operands are descended into. Every other
            :class:`~torch_openreml.covariance.matrix.Matrix` appears as a leaf,
            including :class:`~torch_openreml.covariance.adapter.Adapter` with
            its adaptee.

        Note:
            Because nothing is evaluated, this is the cheap way to inspect how a
            composite distributes its parameters, and the slices are the very
            views each node is handed at evaluation time.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

                op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
                         extra=ScalarMatrix(4))
                free_params_by_path = op.free_param_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))

            .. jupyter-execute::

                free_params_by_path["inner"]
        """
        nodes = self.walk(free_params)

        return {path: operand_params for path, _, operand_params in nodes}

    def grad_tree(self, free_params=None):
        """
        Compute the gradient of this operator and of every nested operand,
        returning each keyed by its path.

        Descends the operand structure, computing the Jacobian of the root and
        of every nested :class:`~torch_openreml.covariance.operator.Operator`
        operand. Paths join the operand names leading to a node with ``"/"``,
        and ``"/"`` is the root — the same keys :meth:`tree` produces, in the
        same order. ``grads`` maps each path to the ``(grad, grad_names)`` pair
        :meth:`~torch_openreml.covariance.matrix.Matrix.grad` returns for that
        node, or to ``(None, [])`` when the node has no free parameters.

        Each node's ``grad`` has shape ``(num_free_params, *shape)`` for that
        node's own free parameters and its own matrix shape, so a nested node's
        Jacobian is not embedded in its parent's shape. The root entry is the
        same pair :meth:`~torch_openreml.covariance.matrix.Matrix.grad` returns
        for the whole composite.

        ``grad_names`` are namespaced from the node that produced them, as
        :meth:`~torch_openreml.covariance.matrix.Matrix.grad` returns them: at
        the root they are the composite's full names, and below it joining the
        path to a name with ``"/"`` gives the name the root exposes in
        :attr:`~torch_openreml.covariance.matrix.Matrix.free_param_names`.

        Args:
            free_params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_free_params`.
                If omitted, default values are used. Default: ``None``.

        Returns:
            dict: Maps each path to the node's ``(grad, grad_names)`` pair.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the wrong
                length, or if ``free_params`` is a dict with missing or
                unexpected keys.

        Note:
            Only nested :class:`~torch_openreml.covariance.operator.Operator`
            operands are descended into. Every other
            :class:`~torch_openreml.covariance.matrix.Matrix` appears as a leaf,
            including :class:`~torch_openreml.covariance.adapter.Adapter` with
            its adaptee.

        Note:
            Cost: every Jacobian is computed independently, so a node at depth
            ``d`` is evaluated ``d + 1`` times — once for the root's Jacobian and
            once for each intermediate ancestor's, plus once for its own entry.
            No Jacobian is cached, so a preceding
            :meth:`~torch_openreml.covariance.matrix.Matrix.grad` changes
            nothing. Viewing a deep composite in full is correspondingly
            expensive.

        Note:
            A node's Jacobian is in that node's own shape, so the tree does not
            show how a child's Jacobian is placed into its parent's — how
            ``BlockDiagonal`` pads it into a block, for instance. Only the root
            entry carries the composite's full shape.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

                op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
                         extra=ScalarMatrix(4))
                grads = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))

            .. jupyter-execute::

                grads["inner/op_0"]

            .. jupyter-execute::

                grads["inner"][1]
        """
        nodes = self.walk(free_params)

        return {
            path: operand.grad(operand_params)
            if isinstance(operand, Matrix)
            else (None, [])
            for path, operand, operand_params in nodes
        }

    def operands_grad(self, free_params=None):
        """
        Compute the Jacobian of each operand with respect to its parameters.

        Splits ``free_params`` into per-operand slices, calls
        :meth:`~torch_openreml.covariance.matrix.Matrix.grad` on each
        :class:`~torch_openreml.covariance.matrix.Matrix` operand, and
        prefixes the returned names with the operand name. Fixed tensor
        operands contribute ``None`` and an empty name list.

        Args:
            params (torch.Tensor or dict): Flat 1D joint parameter tensor or
                parameter dictionary of length :attr:`num_params`.
                If omitted, default values are used. Default: ``None``.

        Returns:
            tuple: ``(grad_groups, grad_name_groups)``, where
            ``grad_groups`` is a list of per-operand Jacobian tensors or
            ``None`` for fixed operands, and ``grad_name_groups`` is a list
            of corresponding namespaced parameter name lists.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import Sum, ScalarMatrix

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                grad_groups, grad_name_groups = x.operands_grad(torch.tensor([1.0, 2.0]))
                print(grad_groups[0])
                print(grad_groups[1])
                print(grad_name_groups[0])
                print(grad_name_groups[1])
        """
        if free_params is None:
            free_params = self.free_param_defaults
        free_params = self._from_free_param_dict(free_params)
        self._check_param_tensor(free_params, length=self.num_free_params)

        grad_groups = []
        grad_name_groups = []

        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                operand_params = free_params[0:operand.num_free_params]
                free_params = free_params[operand.num_free_params:]

                grad, grad_names = operand.grad(operand_params)

                if grad is not None:
                    grad_groups.append(grad)
                    grad_name_groups.append([f"{name}/{grad_name}" for grad_name in grad_names])
                else:
                    grad_groups.append(None)
                    grad_name_groups.append([])
            else:
                grad_groups.append(None)
                grad_name_groups.append([])

        return grad_groups, grad_name_groups

    def operands_inverse(self, free_params=None):
        """
        Compute the inverse of each operand at the current free parameters.

        Splits ``free_params`` into per-operand slices, calls
        :meth:`~torch_openreml.covariance.matrix.Matrix.inverse` on each
        :class:`~torch_openreml.covariance.matrix.Matrix` operand, and casts
        and inverts each fixed tensor operand. A
        :class:`~torch_openreml.covariance.matrix.Matrix` operand is never
        built here, so an analytic inverse — or a nested operator's own
        structured inverse — is used instead of inverting the operand
        densely.

        Args:
            free_params (torch.Tensor or dict): Flat 1D parameter tensor or
                parameter dictionary.
                If omitted, default values are used. Default: ``None``.

        Returns:
            list of torch.Tensor: Each operand's inverse, in the order of
            :attr:`operands`.

        Raises:
            TypeError: If ``free_params`` is not a Torch tensor.
            ValueError: If ``free_params`` is not a 1D tensor or has the
                wrong length, or if ``free_params`` is a dict with missing
                or unexpected keys.

        Example:
            .. jupyter-execute::

                import torch
                from torch_openreml.covariance import ScalarMatrix, Sum

                x = Sum(ScalarMatrix(2), ScalarMatrix(2))
                inverses = x.operands_inverse(torch.tensor([1.0, 2.0]))
                print(inverses[0])
                print(inverses[1])
        """
        if free_params is None:
            free_params = self.free_param_defaults
        free_params = self._from_free_param_dict(free_params)
        device, dtype = self._check_param_tensor(free_params, length=self.num_free_params)

        inverses = []

        for operand in self.operands.values():
            if isinstance(operand, Matrix):
                operand_params = free_params[0:operand.num_free_params]
                free_params = free_params[operand.num_free_params:]

                inverses.append(operand.inverse(operand_params))
            else:
                inverses.append(torch.linalg.inv(operand.to(device=device, dtype=dtype)))

        return inverses

    def reset_intermediates(self):
        """
        Clear the intermediate caches of this operator and its operands.

        Extends :meth:`~torch_openreml.covariance.matrix.Matrix.reset_intermediates`,
        which clears this operator's own cache, by recursively clearing the
        cache of every :class:`~torch_openreml.covariance.matrix.Matrix`
        operand. Nested operators continue the descent through this same
        override, so resetting a node resets everything below it.
        """
        super().reset_intermediates()

        for operand in self.operands.values():
            if isinstance(operand, Matrix):
                operand.reset_intermediates()

    def enable_cache(self):
        """
        Enable intermediate caching on this operator and its operands.

        Extends :meth:`~torch_openreml.covariance.matrix.Matrix.enable_cache`,
        which enables caching on this operator, by recursively enabling it on
        every :class:`~torch_openreml.covariance.matrix.Matrix` operand. Nested
        operators continue the descent through this same override, so enabling
        a node enables everything below it.
        """
        super().enable_cache()

        for operand in self.operands.values():
            if isinstance(operand, Matrix):
                operand.enable_cache()

    def disable_cache(self):
        """
        Disable intermediate caching on this operator and its operands.

        Extends :meth:`~torch_openreml.covariance.matrix.Matrix.disable_cache`,
        which disables caching on this operator and clears its cache, by
        recursively disabling caching on every
        :class:`~torch_openreml.covariance.matrix.Matrix` operand and clearing
        their caches. Nested operators continue the descent through this same
        override, so disabling a node disables everything below it.
        """
        super().disable_cache()

        for operand in self.operands.values():
            if isinstance(operand, Matrix):
                operand.disable_cache()

    @property
    def operands(self):
        """dict: Mapping from operand names to operand matrices or tensors."""
        return self._operands

    @property
    def param_specs(self):
        param_specs = {}
        for name, operand in self.operands.items():
            if isinstance(operand, Matrix):
                this_param_specs = {f"{name}/{param_name}": spec for param_name, spec in operand.param_specs.items()}
                param_specs.update(this_param_specs)
        return param_specs
    
    @property
    def repr_dict(self):
        """dict: Key-value pairs used to build the string representation."""
        return {"operands": self.operands}
        

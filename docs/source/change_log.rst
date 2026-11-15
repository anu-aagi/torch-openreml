Changelog
=========

0.4.1-alpha (unreleased)
------------------------

Breaking changes
~~~~~~~~~~~~~~~~

* ``MarginalREML`` has been removed, and ``torch_openreml.marginal_reml`` no
  longer exists. Its replacement is
  :class:`~torch_openreml.mixed_model_reml.MixedModelREML`, which takes one or
  more random-effect ``(design, covariance)`` terms and a keyword ``R`` for the
  residual covariance, builds :math:`V = R + Z G Z^\top` with
  :class:`~torch_openreml.covariance.operator_marginal_covariance.MarginalCovariance`,
  and selects a numerical backend instead of taking a single ``v`` matrix.
  Code that passed ``v`` to ``MarginalREML`` must be rewritten in this form;
  the post-estimation functions in ``torch_openreml.post`` are unchanged.
* ``Operator.call_tree`` is now ``Operator.tree``. It and
  ``Operator.grad_tree`` each return a single dict now, rather than a pair of
  dicts: ``tree`` maps a path to the matrix the node evaluated to, and
  ``grad_tree`` to that node's ``(grad, grad_names)`` pair. The per-node
  parameter slices they used to return alongside moved to
  ``free_param_tree``.

Changes
~~~~~~~

* ``Matrix.set_param_specs`` distributes each field it is given: a single value
  is broadcast to every resolved specification, while a sequence holding one
  value per specification is split element-wise — a ``bool`` or a list or tuple
  of bools for ``fixed``, a ``Transform`` or a list or tuple of transforms for
  ``trans``, and a length-1 or ``len(specs)`` tensor for ``default``. Any other
  length raises ``ValueError``, and a value of the wrong type raises
  ``TypeError``, before anything is written. Previously every specification was
  assigned the value as given, so a tensor longer than one parameter was
  duplicated across specifications instead of split between them.
* ``MixedModelREML.optimize`` names its starting covariance parameters
  ``init_values`` rather than ``theta``; positional calls are unaffected, but
  ``theta=`` must be renamed. ``update`` keeps its ``theta`` argument, where
  the name does mean an estimate.
* ``MixedModelREML.optimize`` converts ``y`` and ``x`` to the dtype and device
  of the initial covariance parameters once, then stores the converted tensors
  for the post-fit methods, instead of repeating the conversion inside every
  step.
* ``MixedModelREML.optimize`` uses a relative Armijo slack
  (``armijo_slack=1e-7``) in its sufficient-increase test, so a step within
  floating-point noise is accepted. Set ``armijo_slack=0`` for strict Armijo.
* ``MixedModelREML.history`` records the backend that produced each AI step
  under ``"method"``, alongside ``free_params``, ``loglik``, ``score``, ``ai``,
  ``delta``, ``update``, ``eta`` and ``accepted``. ``ai_step`` and ``loglik``
  return the method with their numerical result.

New features
~~~~~~~~~~~~

* ``MixedModelREML``, the new frontend for linear mixed models. Each
  random-effect term is a ``(design, covariance)`` pair; ``R=`` supplies the
  residual covariance and defaults to a ``ScalarMatrix`` whose dimension is
  inferred from a random-effect design. Free parameters are ordered as
  ``[Z, G, R]`` and exposed with namespaced names. ``method=`` accepts
  ``"auto"`` (the default), ``"cform"``, ``"marginal"``, or a configured
  ``CForm`` / ``Marginal`` instance; ``"auto"`` uses ``CForm`` when the model
  is eligible and falls back to ``Marginal`` with a warning if CForm fails.
  ``optimize`` runs AI-REML with Armijo backtracking, and ``build_params``,
  ``tree``, ``c_inverse_blocks`` and ``get_free_params`` expose the fitted
  structure.
* ``torch_openreml.methods`` with the two public numerical backends:
  :class:`~torch_openreml.methods.cform.CForm` for dense, invertible ``G`` and
  ``R`` and :class:`~torch_openreml.methods.marginal.Marginal` for the general
  ``V``-based case, including multiple random-effect blocks and approximate
  traces. Both expose score, average-information and restricted
  log-likelihood; ``Marginal`` adds the ``trace_approx`` controls.
* ``MixedModelREML`` post-fit quantities: ``blue``, ``cov_blue``, ``var_blue``,
  ``blup``, ``cov_blup``, ``var_blup``, ``blup_per_term``,
  ``cov_blup_per_term``, ``var_blup_per_term``, ``marginal_predict``,
  ``marginal_residual``, ``predict``, ``predict_new``, ``residual`` and
  ``loglik``. They reuse the response and design stored by ``optimize`` when
  ``y`` and ``x`` are omitted, and accept ``free_params="last"`` or
  ``"best"``. ``predict_new`` predicts at new rows from already-built design
  matrices, with ``z`` given as a full matrix, a list per term, or a dict
  keyed by term name.
* ``post.cov_blue`` and ``post.var_blue``, and ``post.cov_blup`` and
  ``post.var_blup``, the sampling covariance and variance of the fixed-effect
  estimates and random-effect predictions. ``cov_blue`` is the ``C11`` block
  of the inverse coefficient matrix and ``cov_blup`` uses
  :math:`G Z^\top P Z G`, which needs only ``V`` to be invertible.
* ``KnownMatrix``, wrapping a fixed two-dimensional tensor as a ``Matrix`` with
  no free parameters. It follows the parameter tensor's dtype and device when
  one is supplied, and precomputes the inverse for a square, invertible matrix.
* ``MarginalCovariance``, an operator with exactly three named operands,
  ``Z``, ``G`` and ``R``, evaluating :math:`R + Z G Z^\top`. It validates the
  design and covariance shapes and namespaces parameters as ``Z/...``,
  ``G/...`` and ``R/...``; it is the operator ``MixedModelREML`` builds.
* ``Matrix.manual_inverse``, the opt-in closed-form inversion hook. The base
  implementation raises ``NotImplementedError``, so a matrix without a closed
  form fails loudly; concrete implementations exist for ``ScalarMatrix``,
  ``IdentityMatrix``, ``DiagonalMatrix``, ``EquicorrelationMatrix``,
  ``AR1Matrix``, ``CompoundSymmetricMatrix``, ``Adapter`` and ``KnownMatrix``.
  ``Operator.operands_manual_inverse`` is the corresponding operator hook and
  propagates ``NotImplementedError`` when an operand has no closed form, with
  implementations for ``BlockDiagonal``, ``Product``, ``KroneckerProduct``,
  ``Transpose`` and ``Inverse``.
* ``Matrix.resolve_dim``, letting these matrix constructors take a list, tuple
  or ``pandas.Series`` of levels instead of an integer dimension; the dimension
  is the number of distinct levels. It is used by ``ScalarMatrix``,
  ``IdentityMatrix``, ``DiagonalMatrix``, ``AR1Matrix``,
  ``CompoundSymmetricMatrix``, ``EquicorrelationMatrix``, ``EqualEntryMatrix``,
  ``LowerTriangularMatrix`` and ``UnconstrainedMatrix``.
* ``Adapter.manual_inverse``, forwarding its mapped parameters to the
  adaptee's ``manual_inverse``, so a reparameterised matrix keeps the
  adaptee's closed form instead of falling back to a dense inversion.
* ``Operator.walk``, listing every node of the operand tree as one
  ``(path, operand, free_params)`` triple, in the order ``tree``,
  ``free_param_tree`` and ``grad_tree`` report. The first entry is always
  ``("/", self, free_params)``: the operator itself and the validated input
  tensor, whose dtype and device are the resolved ones, while a fixed tensor
  operand gets ``None``. The walk validates and splits the parameters and reads
  the operand structure, so it evaluates nothing — the other three methods are
  built on it.
* ``Operator.free_param_tree``, mapping the same paths as ``tree`` to the 1D
  free-parameter slice each node received: the root entry is the validated
  input tensor and the remaining entries are views of it, while fixed tensor
  operands get ``None``. It evaluates no operand, so it is the cheap way to
  inspect how a composite splits its parameters between operands.
* ``MixedModelREML.tree``, evaluating the fitted covariance structure at a
  parameter estimate and returning the same path-keyed dict as
  ``Operator.tree``, the root ``"/"`` holding :math:`V(\theta)` itself. It
  takes a ``free_params`` tensor or dict, or the selectors ``"last"`` and
  ``"best"``; ``tree()`` uses the last fitted iterate. When ``v`` is a plain
  matrix rather than an operator, the tree is the root entry alone.
* ``DummyMatrix.transform``, building the design matrix of new observations
  from the columns fixed at initialisation: each row is one-hot in the column
  named by the ``"⋈"`` join of its factor values, and a combination with no
  column — an unseen level, or the reference combination dropped by
  ``drop_first`` — is an all-zero row. New levels therefore cannot widen the
  matrix, which makes the result the ``z`` block that
  ``MixedModelREML.predict_new`` expects. It takes ``dtype`` and ``device``
  keywords, defaulting to the PyTorch defaults, and emits no warning, since it
  cannot tell a dropped reference from an unseen level.

Bug fixes
~~~~~~~~~

* ``post.blue``, ``post.blup`` and ``post.predict`` now keep at least one
  dimension: they end in ``.squeeze(-1)`` instead of ``.squeeze()``, so a
  single fixed effect, a single random effect or a single observation returns
  the documented ``(p,)`` / ``(q,)`` / ``(n,)`` shape rather than a 0-d
  tensor. This also fixes the downstream ``marginal_predict``,
  ``marginal_residual`` and ``blup`` failures for those cases.
* ``MixedModelREML.is_converged`` now raises ``ValueError`` with an explanatory
  message when called before ``optimize()``, instead of leaking a ``TypeError``
  from a ``None`` history.
* ``MixedModelREML.optimize`` keeps ``delta`` one-dimensional when the model
  has one covariance parameter, instead of a 0-d tensor.
* ``Operator.build_params(..., out_format="dict")`` now returns one-element
  tensors keyed by the names that match ``include_fixed``; previously it
  returned 0-d scalars and always used the free-parameter names.
* ``AR1Matrix`` computes the correlation derivative without dividing by a
  clamped ``rho``, which previously replaced ``|rho| < 1e-6`` and produced a
  wrong gradient at ``rho == 0``.
* ``Marginal`` now rejects ``trace_approx`` values other than ``True``,
  ``False`` or ``"auto"``. Previously ``0`` and ``1`` passed validation and
  then matched no dispatch branch, ending in ``UnboundLocalError``.
* ``CForm`` casts a fixed-tensor (or zero-parameter ``Matrix``) ``Z`` to the
  free parameters' dtype and device, so a float32 design with float64
  parameters no longer fails in matmul.
* ``EqualEntryMatrix``, ``LowerTriangularMatrix`` and ``UnconstrainedMatrix``
  now validate the size of a custom ``param_specs`` at construction.
  ``EqualEntryMatrix`` requires exactly one spec, so a multi-parameter surface
  is rejected instead of producing a broadcast ``(1, n, m)`` gradient. The two
  lower-triangular classes require one spec per slot: short custom surfaces
  previously broadcast silently and then failed inside ``manual_grad`` with a
  masked ``IndexError``, while long ones raised ``RuntimeError`` from the
  assignment.

Documentation
~~~~~~~~~~~~~

* Updated the API, README, R-user guide and examples for ``MixedModelREML``,
  the ``methods`` backends, ``manual_inverse`` and the path-keyed ``tree``
  output. The API reference gained the ``methods`` page and the
  ``KnownMatrix``, ``MarginalCovariance`` and post-fit covariance entries; the
  old ``MarginalREML`` vignette moved to ``docs/archive/``. The R examples now
  use ``MixedModelREML`` with ``build_params()``, ``blue()``, ``blup()`` and
  ``blup_per_term()``.
* ``Marginal.ai_step`` now documents the ``AttributeError`` raised when
  ``model`` is ``None``; the previous docstring claimed ``ValueError``. It also
  states that ``require_loglik=False`` returns the ``torch.nan`` sentinel.
* The ``post`` module and API page now document the fixed ``1e-6 * I`` ridge
  added before factorising ``V``; the behaviour itself is unchanged.

Testing
~~~~~~~

* Added ``tests/test_mixed_model_reml.py``, ``tests/methods/test_cform.py`` and
  ``tests/methods/test_marginal.py``, with matching planning documents, for the
  new estimator and both numerical backends.
* Added dedicated tests and plans for ``param``, ``post``, ``KnownMatrix``,
  ``Gram``, ``MarginalCovariance``, ``EqualEntryMatrix``,
  ``LowerTriangularMatrix`` and ``UnconstrainedMatrix``.
* Added dedicated tests and plans for the ``Inverse``, ``Product``, ``Stack``
  and ``Transpose`` operators, and plans for ``Adapter`` and ``SimpleMatrix``.
  Expanded the matrix and operator tests for ``manual_inverse``,
  ``resolve_dim``, ``tree``, ``walk``, ``free_param_tree`` and the distributed
  ``set_param_specs`` form, and removed the obsolete ``map_theta_to_v`` /
  ``map_theta_to_dv`` tests.
* Expanded the suite to cover every exported module: each test module now has
  a matching ``test_plan_<module>.md``, and ``tests/test_summary.md`` records
  the per-module status. The default pytest path is now the whole ``tests``
  tree rather than only ``tests/covariance``.

0.4.0-alpha (2026-10-05)
------------------------

Breaking changes
~~~~~~~~~~~~~~~~

* ``simple_param_specs`` and ``SimpleMatrix`` now take their default as a 1D
  tensor of shape ``(1,)``. A default passed as a float raises a ``TypeError``,
  and a tensor of another shape a ``ValueError``.
* Parameter names must be non-empty and must not contain ``"/"``, which the
  path syntax of ``set_param_specs`` reserves as its separator. Operand names
  in an operator follow the same rule.

New features
~~~~~~~~~~~~

* New operators: ``Product``, the matrix product :math:`V = A B` of exactly
  two operands; ``Transpose``, :math:`V = A^\top`; ``Stack``, binding two or
  more operands row-wise; and ``Inverse``, :math:`V = A^{-1}`, whose gradient
  reuses the inverse computed in the forward pass. As with the other
  operators, each operand may be a trainable matrix or a fixed tensor.
* ``Matrix.set_param_specs``, updating one or more parameter specifications in
  place. A key without a trailing ``"/"`` addresses one parameter exactly,
  while a key ending with ``"/"`` addresses a subtree: ``"A/"`` every parameter
  of the operand ``"A"``, nested operands below it included, and ``"/"`` every
  parameter of the matrix. Each field passed replaces that field of every
  resolved specification and fields left as ``None`` are kept, so a call
  without fields changes nothing. Values are assigned as given, without
  validation or copying, and take effect on the next call.
* ``Matrix.cache`` and the cache controls ``enable_cache``, ``disable_cache``
  and ``reset_intermediates``. Disabling the cache also clears it, so an entry
  stored while caching was on cannot come back when it is turned on again.
  ``Operator`` overrides all three to recurse into its matrix operands and
  ``Adapter`` into its adaptee, so one call covers a whole composite.

Changes
~~~~~~~

* ``MarginalREML.ai_step`` now resolves the dtype and device from ``theta``
  alone and converts ``x`` and ``y`` to them, rather than resolving a common
  dtype and device from the three.

Documentation
~~~~~~~~~~~~~

* Added API reference pages for ``Product``, ``Transpose``, ``Stack`` and
  ``Inverse``, listed them on the operators page, and regenerated the
  per-class pages.

Testing
~~~~~~~

* Added test modules and planning documents for ``Product``, ``Transpose``,
  ``Stack`` and ``Inverse``.

0.3.0-alpha (2026-09-20)
------------------------

Breaking changes
~~~~~~~~~~~~~~~~

* ``IdentityMatrix`` and ``DummyMatrix`` no longer accept ``dtype`` and
  ``device`` arguments. Both follow the dtype and the device of their input
  instead, falling back to the PyTorch defaults when they receive none.
* ``MarginalREML.get_theta`` and ``MarginalREML.get_beta`` no longer accept a
  ``history`` argument, always reading the history populated by ``optimize``.
* A composite covariance matrix is now built on a single dtype and device.
  Where the operands' own defaults disagree and no input parameters override
  them, a ``ValueError`` is raised instead of ``torch.cat`` silently promoting.

New features
~~~~~~~~~~~~

* ``Operator.call_tree``, a debugging view of a composite. It returns two
  dictionaries over one key set, keyed by the path to each node — the operand
  names leading to it joined with ``"/"``, rooted at ``"/"`` — holding the
  matrix every node evaluated to and the free parameter slice that node
  received. Nested operators are reached by recursing through ``call_tree``
  rather than by capturing the calls a forward pass makes, so the tree is
  complete whether or not an intermediate cache is warm.
* ``Operator.grad_tree``, the gradient counterpart over the same keys: each
  node's ``(grad, grad_names)`` pair, in that node's own shape, alongside the
  same parameter slices. A node's names are namespaced from that node, so
  joining the path to a name gives the name the composite exposes in
  ``free_param_names``.
* ``Matrix.get_default_dtype_device``, reporting the dtype and device that would
  be resolved from the current parameters.

Changes
~~~~~~~

* The dtype and device of a composite are resolved once for the whole
  structure: the input parameters first, then the free-parameter defaults of
  the ``Matrix`` operands, then Torch's defaults. Input parameters override
  every default. Fixed tensor operands are never a resolution source — they are
  cast to follow. The operators and the identity and dummy matrices now observe
  the resolved dtype and device rather than fixing their own, and the identity
  and dummy matrices return a clone so a cached result cannot be mutated by the
  caller.
* The intermediate cache compares parameter values elementwise rather than
  hashing them, removing the dependency on ``torch.hash_tensor`` and the
  limitation it imposed on MPS. A copy of the parameter tensor is stored, so a
  later in-place edit of the caller's tensor cannot change the key of an entry
  already cached.
* ``Adapter`` no longer forces its ``param_specs``; ``param_map`` now receives
  the free, untransformed parameters.

Bug fixes
~~~~~~~~~

* Fixed silent dtype promotion in the operators, where ``torch.cat`` would
  promote the results of the operands.
* Fixed the parameter conversions between dicts and tensors, including the empty
  case, which now yields an empty tensor rather than failing.
* ``Matrix.auto_grad`` returns ``(None, [])`` for a matrix without free
  parameters without building a Jacobian, and ``trans_grad`` returns an empty
  tensor where appropriate.
* The covariance matrices now validate ``free_params`` through ``build_params``
  before returning early for a matrix without free parameters, so invalid input
  raises even there.
* ``SimpleMatrix`` validates its input before passing it on.
* Corrected the documented shapes in ``MarginalREML``, whose score and AI matrix
  are sized by the number of free parameters.

Documentation
~~~~~~~~~~~~~

* Regenerated and extended the API reference, adding pages for ``Adapter``,
  ``Augment``, ``Gram`` and the post-estimation functions.
* Documented the single dtype and device contract on ``Operator``, and the cost
  of ``call_tree`` and ``grad_tree``.

Testing
~~~~~~~

* Added test modules for the adapter and for ``Augment``, and extended the
  operator, matrix, identity, dummy and scalar matrix suites to cover the dtype
  and device contract and the two new tree methods. Device tests follow the
  accelerator available at run time.
* Added a test plan document beside each test module under ``tests/covariance``,
  recording what must be covered and the risks worth testing.

0.2.0-alpha (2026-09-15)
------------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Renamed the ``REML`` estimator to ``MarginalREML``. The module
  ``torch_openreml.reml`` is now ``torch_openreml.marginal_reml``, and
  ``torch_openreml.REML`` is no longer available.
* ``MarginalREML`` now takes a single ``Matrix`` instance as its only
  argument. The previous functional interface (``v_builder``,
  ``map_theta_to_v``, ``map_theta_to_g`` and ``map_theta_to_dv``) has been
  removed.
* Reworked the ``Matrix`` parameter interface. The constructor now takes a
  single ``param_specs`` dictionary in place of ``param_names``, ``trans`` and
  ``no_grad_index``. Each parameter is specified by its own dictionary with the
  keys ``"fixed"``, ``"default"`` and ``"trans"``, so fixed and free parameters
  can be described together, each with an individual default and transform.
* Removed ``Matrix.from_param_dict``, ``Matrix.to_param_dict``,
  ``Matrix.map_theta_to_v``, ``Matrix.map_theta_to_dv``,
  ``Matrix.set_no_grad`` and ``Matrix.check_params``. Use ``build_params`` and
  the ``param_specs`` accessors instead.
* ``Matrix.__call__``, ``grad``, ``auto_grad``, ``manual_grad`` and
  ``trans_grad`` now take an optional ``free_params`` argument. When it is
  omitted, the default value of each free parameter is used, so a matrix can be
  evaluated without supplying parameters explicitly.
* Post-estimation methods ``marginal_predict``, ``marginal_residual`` and
  ``blup`` were removed from the estimator. All post-estimation quantities are
  now computed by the module-level functions in ``torch_openreml.post``.
  ``MarginalREML`` keeps ``blue``, ``predict``, ``residual`` and ``loglik`` as
  convenience methods.
* Removed ``utils.categorical_to_design_matrix``. Categorical encoding is
  handled by ``DummyMatrix``.

New features
~~~~~~~~~~~~

* New covariance matrices: ``EqualEntryMatrix``, ``UnconstrainedMatrix``,
  ``LowerTriangularMatrix`` and ``SimpleMatrix``.
* New ``Adapter`` matrix, which reparameterises an existing matrix through a
  user-supplied mapping from the adapter's parameters to the adaptee's
  parameters, allowing parameters to be combined, split or constrained.
* New operators: ``Gram``, computing :math:`X^\top X` or :math:`X X^\top`, and
  ``Augment``, binding operands column-wise.
* New ``torch_openreml.post`` module with the post-estimation functions
  ``blue``, ``blup``, ``marginal_predict``, ``marginal_residual``, ``predict``,
  ``residual`` and ``loglik``, operating directly on tensors.
* New ``simple_param_specs`` helper for building parameter specifications with a
  common transform and default.
* Trace approximation in ``MarginalREML``. ``ai_step`` and ``optimize`` accept
  ``trace_approx`` and ``subspace_fraction``, which estimate the trace term of
  the score from a random subspace instead of computing it exactly.
  ``optimize`` also accepts ``exact_score_threshold``, below which the exact
  score is used again so that convergence is not affected by the approximation.
* Configurable differentiation. ``Matrix`` exposes a ``grad_mode`` attribute
  (``"manual"``, ``"auto"`` or ``"default"``) and a ``jacobian_method``
  attribute, together with ``jacobian_chunk_size``.
* The package now exposes ``__version__``.

Changes
~~~~~~~

* The default Jacobian method for automatic differentiation is now
  ``"jacfwd"``, which avoids the memory blow-up that ``"jacrev"`` suffers as
  the matrix dimension grows.
* ``identity_matrix``, ``scalar_matrix`` and the operators were updated to the
  new ``param_specs`` parameter interface.
* ``DummyMatrix`` gained a ``drop_empty_cols`` option for dropping columns that
  contain no observations.
* ``TransformIdentity`` now returns a result of the same length as its input.
* ``CompoundSymmetricMatrix`` and ``EquicorrelationMatrix`` now raise an error
  for ``n = 1``.

Bug fixes
~~~~~~~~~

* Fixed incorrect gradient computation in ``DiagonalMatrix``, including the
  gradient masking and the selection of the manual gradient for all parameters.
* Fixed ``BlockDiagonal`` when ``free_params`` was passed as a dictionary.
* Fixed an empty-tensor concatenation bug in ``Matrix``.
* Fixed the gradient and the inverse of ``TransformPow``.
* Fixed ``Operator`` so that nested operators and matrices have their
  intermediates reset correctly during automatic differentiation.
* Fixed the operators so that free parameter defaults are used when
  ``free_params`` is ``None``.

Documentation
~~~~~~~~~~~~~

* Added this changelog, a vignette on marginal REML, and a guide for using
  ``torch-openreml`` from R via ``reticulate``.
* Restructured the API reference, adding pages for the adapter, post-estimation
  functions and the new matrices and operators.
* Rewrote the README with the package overview, features and installation
  instructions.

Testing
~~~~~~~

* Added a pytest suite under ``tests/covariance`` covering the matrices, the
  operators, the transforms, ``utils`` and the example data, and registered it
  as the default test path in ``pyproject.toml``.

0.1.0-alpha (2026-05-08)
------------------------

* First release of ``torch-openreml``

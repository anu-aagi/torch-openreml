Changelog
=========

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

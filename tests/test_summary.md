# Test Situation Summary

Generated 2026-11-15 from the working tree on top of `main` @ `15c07f2`.

Environment: conda env `torch-openreml`, Python 3.12.12, torch 2.14.0.
MPS is available; CUDA is not.

Command:

```bash
conda run -n torch-openreml python -m pytest tests -q
```

**Collected 3306 — 3297 passed, 9 skipped, 0 failed (~8 s).**

`pyproject.toml` now sets `testpaths = ["tests"]`, so bare `pytest` collects the whole
suite (3306 tests), not just `tests/covariance`.

Every module in `torch_openreml/` except `ao_marginal_reml.py` has a dedicated test
module and a neighbouring `test_plan_<module>.md`. `ao_marginal_reml.py` is a legacy,
unexported half-finished module and is intentionally left untested. All 41 plan files
carry an Expected Test Count Summary whose total matches the collected count. The 9
skips are CUDA-only cases in `tests/covariance/transform/test_transform_exp.py`
(`TestExponentialTransformsCUDA`, `skipif not torch.cuda.is_available()`); device tests
elsewhere use `cuda → mps → pytest.skip("no accelerator available")`.

No xfails remain. All defects found while writing the tests were fixed in the library
during this pass — the `blue`/`predict` squeezes, the `is_converged` pre-fit error,
loose `trace_approx` validation, the `ai_step(None)` documentation, CForm's fixed-`Z`
cast, and the spec-count guards on `EqualEntryMatrix`, `LowerTriangularMatrix` and
`UnconstrainedMatrix` — and their tests are now ordinary passing regressions.

## Status by module

Test counts are collected tests; every row passes unless noted.

### Matrix infrastructure

| Module | Test module | Tests | Coverage |
|---|---|---:|---|
| `covariance/matrix.py` | `tests/covariance/test_matrix.py` | 470 | `Matrix` ABC contract: parameter-spec validation and merging, fixed/free parameters, transforms, caching and cache toggles, dtype/device resolution, gradient dispatch (`default`/`manual`/`auto`) across nine concrete vehicles, `manual_inverse`, `repr`. |
| `covariance/operator.py` | `tests/covariance/test_operator.py` | 130 | `Operator` ABC: operand namespacing (`"operand/param"`), `build_params`/`build_operands`, call/free-parameter/gradient trees, `dtype`/`device`, `operands_grad`, inverse delegation, `repr`, cache control. |
| `covariance/param.py` | `tests/covariance/test_param.py` | 40 | `simple_param_specs`: names/order, spec fields, zero/custom/default tensors, clone independence, shared identity transform vs custom transforms, validation (`TypeError`/`ValueError`), and `SimpleMatrix` integration. |
| `covariance/adapter.py` | `tests/covariance/test_adapter.py` | 30 | Reparameterisation through a param mapping, forwarded `manual_inverse`, cache propagation, manual gradients. |
| `covariance/simple_matrix.py` | `tests/covariance/test_simple_matrix.py` | 21 | Callable-backed matrix, auto/manual gradients, default-parameter validation. |
| `covariance/known_matrix.py` | `tests/covariance/test_known_matrix.py` | 100 | Fixed 2-D tensor covariance: construction/validation, empty parameter surface, cast clone from `__call__`, precomputed inverse, `NotImplementedError` for rectangular/singular, dtype/device, `repr`. |
| `covariance/operator_gram.py` | `tests/covariance/test_operator_gram.py` | 56 | `Gram` (`XᵀX` / `XXᵀ`): one positional operand `op_0`, `gram_type` validation, shape `()` before first call, values, product-rule manual gradients, dtype/device resolution. |
| `covariance/operator_marginal_covariance.py` | `tests/covariance/test_operator_marginal_covariance.py` | 63 | `MarginalCovariance(Z, G, R)` = `R + ZGZᵀ`: three-operand construction and namespacing, shape validation, cached result, analytic gradients for Z/G/R, dtype/device. |

### Concrete covariance matrices

| Module | Test module | Tests | Coverage |
|---|---|---:|---|
| `covariance/identity_matrix.py` | `tests/covariance/test_identity_matrix.py` | 135 | Fixed `Iₙ`: empty parameter surface, dtype/device, `build_params`, `grad` returning `None`, cache no-op, inverse, `repr`. |
| `covariance/scalar_matrix.py` | `tests/covariance/test_scalar_matrix.py` | 248 | `σ²I`: single parameter, transforms, multi-parameter limitation, dtype/device, inverse, `repr`. |
| `covariance/diagonal_matrix.py` | `tests/covariance/test_diagonal_matrix.py` | 40 | Per-entry variances, mixed fixed/free parameters, per-parameter transforms, inverse, dtype/device. |
| `covariance/ar1_matrix.py` | `tests/covariance/test_ar1_matrix.py` | 45 | `σ²ρ^abs(i-j)` values and geometric decay, gradients, fixed configurations, caching, dtype, manual inverse. |
| `covariance/compound_symmetric_matrix.py` | `tests/covariance/test_compound_symmetric_matrix.py` | 44 | `σ²[(1-ρ)I+ρJ]` values, ρ bounds, gradients, fixed configurations, caching, inverse. |
| `covariance/equicorrelation_matrix.py` | `tests/covariance/test_equicorrelation_matrix.py` | 44 | Unit diagonal, ρ bounds, gradients, inverse. |
| `covariance/dummy_matrix.py` | `tests/covariance/test_dummy_matrix.py` | 170 | Constructor, factor levels, one-hot encoding, column names, transform (including the positional contract), dtype/device, validation, gradients, caching, `repr`. |
| `covariance/equal_entry_matrix.py` | `tests/covariance/test_equal_entry_matrix.py` | 237 | `σ²J`: construction from int/list/tuple/Series, exact values, single exp-pow2 parameter, fixed/free specs, `build_params`, manual-vs-auto gradients, dtype/device, cache, base inverse error, `repr`, single-spec enforcement. |
| `covariance/lower_triangular_matrix.py` | `tests/covariance/test_lower_triangular_matrix.py` | 233 | Free lower-triangular entries, upper triangle exactly zero, param names/order/defaults, fixed/free masks, custom transforms, `build_params`, manual-vs-auto gradients, dtype/device, cache, `repr`, spec-count enforcement. |
| `covariance/unconstrained_matrix.py` | `tests/covariance/test_unconstrained_matrix.py` | 226 | Symmetric fill from the lower triangle, diag `exp2`/identity off-diagonal, exact mirroring, fixed/free specs, gradients, dtype/device, cache, `repr`, spec-count enforcement. |

### Operators

| Module | Test module | Tests | Coverage |
|---|---|---:|---|
| `covariance/operator_augment.py` | `tests/covariance/test_operator_augment.py` | 17 | Column-wise concatenation and `col_offsets`, gradients, dtype/device. |
| `covariance/operator_block_diagonal.py` | `tests/covariance/test_operator_block_diagonal.py` | 30 | Blocks of unequal shape, per-block gradients, inverse, caching. |
| `covariance/operator_covariance_propagation.py` | `tests/covariance/test_operator_covariance_propagation.py` | 14 | `ZGZᵀ` values, gradients with respect to `Z` and `G`, shape validation. |
| `covariance/operator_hadamard_product.py` | `tests/covariance/test_operator_hadamard_product.py` | 14 | Elementwise product, scalar-tensor operand, gradients. |
| `covariance/operator_inverse.py` | `tests/covariance/test_operator_inverse.py` | 47 | Numeric inverse, manual-inverse composition, gradients, caching, dtype/device, tree. |
| `covariance/operator_kronecker_product.py` | `tests/covariance/test_operator_kronecker_product.py` | 26 | Kron values, gradients with respect to both operands, manual inverse. |
| `covariance/operator_product.py` | `tests/covariance/test_operator_product.py` | 56 | Matrix product, rectangular operands, reversed inverse, gradients, caching, dtype/device, tree. |
| `covariance/operator_stack.py` | `tests/covariance/test_operator_stack.py` | 23 | Row concatenation, the dual of `Augment`, gradients, composition. |
| `covariance/operator_sum.py` | `tests/covariance/test_operator_sum.py` | 19 | Addition, tensor operands, gradients. |
| `covariance/operator_transpose.py` | `tests/covariance/test_operator_transpose.py` | 45 | Rectangular transpose, manual inverse, gradients, caching, dtype/device, composition. |

### Transforms

| Module | Test module | Tests | Coverage |
|---|---|---:|---|
| `covariance/transform/transform.py` | `tests/covariance/transform/test_transform.py` | 19 | `Transform` ABC and `TransformChain` composition. |
| `covariance/transform/transform_exp.py` | `tests/covariance/transform/test_transform_exp.py` | 67 (58 run, 9 skipped) | `eˣ`, `2ˣ`, `10ˣ`: values, gradients, inverse, round trips; CUDA-only class skipped. |
| `covariance/transform/transform_exppow2.py` | `tests/covariance/transform/test_transform_exppow2.py` | 24 | `e^(2x)` values, gradients, inverse, round trips. |
| `covariance/transform/transform_identity.py` | `tests/covariance/transform/test_transform_identity.py` | 18 | Identity transform values, gradients, inverse. |
| `covariance/transform/transform_pow.py` | `tests/covariance/transform/test_transform_pow.py` | 38 | Power transform values, exponents, gradients, inverse, round trips. |
| `covariance/transform/transform_scale_shift.py` | `tests/covariance/transform/test_transform_scale_shift.py` | 34 | `ax+b` values, gradients, inverse, round trips. |
| `covariance/transform/transform_sigmoid.py` | `tests/covariance/transform/test_transform_sigmoid.py` | 27 | Sigmoid values, gradients, inverse, round trips, bounds. |

### Estimators and post-fit quantities

| Module | Test module | Tests | Coverage |
|---|---|---:|---|
| `mixed_model_reml.py` | `tests/test_mixed_model_reml.py` | 108 | `MixedModelREML`: construction and validation, method selection (`auto`/`marginal`/`cform`), `ai_step`, optimiser history and convergence, post-fit accessors, multi-term and fixed-only models, GLS/BLUP closed forms, parameter recovery, dtype/device, `john_alpha` fits. |
| `post.py` | `tests/test_post.py` | 175 | All 11 public callables (`blue`, `cov_blue`, `var_blue`, `blup`, `cov_blup`, `var_blup`, `loglik`, `marginal_predict`, `marginal_residual`, `predict`, `residual`) and re-exports: hand-computable values, explicit GLS/REML/P-matrix/C22 formulas, cross-function equivalences, shapes, validation, dtype/device, singular G, input immutability. |
| `ao_marginal_reml.py` | — | — | Intentionally untested legacy module, not exported from `torch_openreml/__init__.py`; its tests and plan were removed so the suite covers only supported modules. |
| `methods/cform.py` | `tests/methods/test_cform.py` | 61 | `CForm`: construction/defaults, eligibility and errors, loglik vs explicit C-form and P-form formulas, score/AI vs formulas and autograd, symmetry/PSD, known-G/known-R blocks, `C⁻¹` blocks, jitter path, `preprocess_input`, dtype/device, `john_alpha`. |
| `methods/marginal.py` | `tests/methods/test_marginal.py` | 69 | `Marginal`: construction/defaults, loglik, score/AI, `"auto"` trace resolution, `preprocess_input`, parameterised `Z`, dtype/device, CForm-vs-Marginal consistency, `john_alpha`. |

### Data and utilities

| Module | Test module | Tests | Coverage |
|---|---|---:|---|
| `utils.py` | `tests/test_utils.py` | 31 | `get_device`/`get_dtype`, `numeric_to_design_matrix`, `augment`, `interaction`, `n_distinct`. |
| `example_data/` | `tests/example_data/test_example_data.py` | 12 | `john_alpha` dataset shape, columns, factor coding, and design-matrix construction (72 observations, 3 reps, 6 blocks/rep, 24 genotypes). |

## Remaining defects

None. The last two spec-count gaps are now guarded; the guards and their tests are
listed below.

### Defects fixed during this pass

These defects were found by the tests, confirmed as bugs, and fixed in the library;
the former xfails are now passing regression tests.

- **`blue` / `predict` shape (1–2 of the old list).** `.squeeze()` became
  `.squeeze(-1)` at `torch_openreml/post.py:87` and `:387`, so one-column/one-row
  calls keep the documented `(p,)` / `(n,)` shapes.
- **`is_converged` before a fit.** `mixed_model_reml.py` now raises
  `ValueError("is_converged() requires a fitted model; call optimize() first.")`
  instead of leaking a `TypeError`; the precondition is documented.
- **`trace_approx` validation.** `methods/marginal.py` now accepts only `bool` or
  the exact string `"auto"`; `0`/`1` are rejected before the identity dispatch.
- **`ai_step(None)`.** The docstring now documents the `AttributeError` raised when
  `preprocess_input` dereferences `model`.
- **CForm fixed `Z`.** `methods/cform.py` casts a fixed-tensor (or zero-parameter
  `Matrix`) `Z` to the free parameters' dtype and device.
- **EqualEntryMatrix spec count.** The constructor raises `ValueError` unless
  `param_specs` contains exactly one parameter.
- **LowerTriangularMatrix / UnconstrainedMatrix spec count.** Their constructors now
  require exactly one spec per lower-triangular slot (`sum(min(i + 1, m) for i in
  range(n))` and `n(n+1)/2` respectively), so a short or long custom surface raises
  `ValueError` instead of broadcasting or failing inside `manual_grad`. As with every
  `Matrix`, mutating `param_specs` after construction bypasses the check.
- **Skipped-loglik `nan`.** `ai_step(require_loglik=False)` returns the documented
  `torch.nan` sentinel in both methods; because that is a Python `float`, the skipped
  path intentionally differs in type from the computed scalar. The tests assert
  `isinstance(loglik, float)` and `math.isnan(loglik)`.

## Notes and residual risks

- **Documented jitter.** Every `post.py` function factorises `V + 1e-6·I`, not `V`; the
  module docstring and API page now state this. `Marginal` uses the same jitter whenever no
  `manual_inverse` is available, which is always the case for `MarginalCovariance`;
  CForm/Marginal therefore agree only to ~1e-6 (cross-method tests use `atol=1e-5`).
- **No shape validation of `v` in `post.py`.** A 1-D `v` of length `n` broadcasts against
  `eye(n)` and silently produces an `(n, n)` matrix.
- **`Gram` docstring vs base class.** `Gram` says a fixed tensor is allowed, but
  `Operator` requires at least one `Matrix`, so `Gram(tensor)` raises `TypeError`;
  `Gram(KnownMatrix(x))` is the supported route.
- **Parameter dtype and device.** `MixedModelREML` does not promote `int64`
  parameters (raises `RuntimeError`), and `CForm` can emit a `RuntimeWarning` and fall
  back to `Marginal` at ill-conditioned intermediate iterates.
- **Base-class validation rough edges.** For `LowerTriangularMatrix` and
  `UnconstrainedMatrix`, mixed 1-D/2-D dict values fail inside `torch.cat` as
  `RuntimeError` rather than the documented `ValueError`; non-integer dimensions can
  raise `TypeError` from `range(n)` before the shape check. `EqualEntryMatrix` now
  rejects multi-spec input but still accepts `bool` dimensions at construction.
- **Post-construction mutation.** `param_specs` is a public dict; adding or removing
  entries after construction bypasses every constructor guard (EqualEntry, LowerTri and
  Unconstrained). The supported route is to build the matrix with the intended surface.
- **Plan audit.** `check_plan_totals.py` was used during the update to verify that every
  test file has a plan and that each plan's total matches `--collect-only`; it was
  removed after the final check.

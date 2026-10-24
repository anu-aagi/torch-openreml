# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.matrix`
**Purpose of Module:**
Defines the abstract base class `Matrix` for all covariance matrix types. It is the single place where
parameter specs are validated, free parameters are merged with fixed defaults, parameter transforms are
applied, intermediate results are cached, and gradient computation is dispatched between the closed-form
(`manual_grad`) and automatic (`auto_grad`) paths. Every concrete matrix and every operator inherits this
behaviour, so a defect here surfaces in all of them.

**Classes Covered:**

- `Matrix` (ABC) — abstract base class for parameterized covariance matrices

**Testing Goal:**
Test **every method defined in `matrix.py`** directly, through concrete subclasses used as vehicles.
Where a method has a documented contract (input validation, `(None, [])` for parameter-less matrices,
dtype/device propagation), assert the contract rather than the incidental behaviour. Where a method has a
fast path and a slow path, both must be exercised.

**Vehicle policy:** `Matrix` is abstract (`__call__`), so no test instantiates it. The module defines a
local `MinimalMatrix` subclass to reach base-class code that concrete matrices do not expose. Tests for a
concrete matrix's *own* `__call__` / `manual_grad` belong in that matrix's test module, not here.

---

## 2. Testing Scope

### 2.1 Method Inventory and Coverage Status

Every callable in `matrix.py`, against `tests/covariance/test_matrix.py` as it stood before this plan
was implemented. The "Gap" column is what the tests were missing; §5 records what was added.

| # | Method | Status | Gap |
|---|---|---|---|
| 1 | `__init__` | Partial | Only `grad_mode` default asserted; `jacobian_method`, `jacobian_chunk_size`, `_shape` normalisation and init-time cache reset not asserted |
| 2 | `_check_shape` | Partial | list/`torch.Size` acceptance, zero/float/bool elements, empty sequence, validation ordering |
| 3 | `_check_param_specs` | Partial | non-str name, non-dict spec, extra field, empty dict |
| 4 | `_check_param_tensor` | Covered | — |
| 5 | `get_default_dtype_device` | Covered | — |
| 6 | `build_params` | Partial | no-arg call, mixed free/fixed scatter, all four `include_fixed` × `out_format` combinations, fast vs slow transform path, invalid `out_format` on the empty branch |
| 7 | `_from_free_param_dict` | Covered | — |
| 8 | `_to_free_param_dict` | Covered | — (note: no library code calls it; only tests do) |
| 9 | `set_param_specs` | **New** | added after this plan was implemented; §H-bis records what it must cover |
| 10 | `trans_grad` | Partial | slow path (differing transforms), value checks beyond the single fast-path case, dtype follow |
| 11 | `auto_grad` | Partial | only the default `jacfwd`; `jacrev`, `jacobian`, `jacobian_chunk_size`, unknown method, cache-clearing side effect |
| 12 | `manual_grad` (base) | **Missing** | `NotImplementedError` never asserted; subclass contract is covered by `TestManualGradContract` |
| 13 | `__call__` (abstract) | **Missing** | direct instantiation of `Matrix` |
| 14 | `grad` | Partial | `grad_mode="manual"` branch untested, both with and without a subclass `manual_grad` |
| 15 | `set_intermediates` | Partial | overwrite, object identity, the cached values are a copy of the caller's tensor |
| 16 | `get_intermediates` | Partial | **dtype staleness** and device staleness — the two guards checked before the value comparison |
| 17 | `reset_intermediates` | Covered | — |
| 18 | Properties (17 of them) | Partial | non-empty `fixed_*` cases, mixed free/fixed indices, `param_specs` liveness |
| 19 | `__repr__` / `_repr_indented` | Partial | ≥3-param ellipsis, 0-param omission, multi-line mode |
| 20 | `_repr_value` | **Missing** | all four branches |
| 21 | `_repr_dict` | **Missing** | — |
| 22 | `map_theta_to_v` / `map_theta_to_dv` | **Stale** | removed from the class; `TestREMLInterface` fails |
| 23 | `cache` | **New** | added after this plan was implemented; §P-bis records what it must cover |
| 24 | `enable_cache` / `disable_cache` | **New** | added after this plan was implemented; §P-ter records what they must cover |
| 25 | `resolve_dim` | **New** | added after this plan was implemented; §S records what it must cover |
| 26 | `inverse` | **New** | added after this plan was implemented; §T records what it must cover |

`TestREMLInterface` (2 tests) must be **deleted**: `map_theta_to_v` / `map_theta_to_dv` no longer exist on
`Matrix` or anywhere in the package. Section I of the previous revision of this plan documented a removed
interface and is dropped below.

### 2.2 What Needs to Be Tested, Method by Method

Each entry states the observable behaviour to pin down. Expected error types and messages are taken from
the source, so tests can match on them.

#### A. `__init__(shape, param_specs)`

Assert the attributes the constructor establishes, since subclasses and operators read them:

- `grad_mode == "default"`, `jacobian_method == "jacfwd"`, `jacobian_chunk_size is None`
- `_shape` is a `tuple`, even when built from a list or `torch.Size`
- `_intermediates` is reset on construction: `get_intermediates(free_param_defaults)` returns `None`
- Validation order: `_check_shape` runs **before** `_check_param_specs`, so a bad shape and bad specs
  raise the shape error first
- `MinimalMatrix((2, 2), {})` is legal — a zero-parameter matrix is constructible

#### B. `_check_shape(shape)`

Constructor calls only, through `MinimalMatrix`:

- Accepts `list` → stored as `tuple`; accepts `tuple`; accepts `torch.Size`
- `None` → accepted, `_shape == ()`
- Empty sequence (`[]`, `()`) → accepted, `_shape == ()` (`tuple(shape or ())` treats it as falsy)
- Non-sequence (`42`, `"ab"`) → `TypeError("'shape' must be a list, a tuple or a torch.Size!")`
- Zero or negative element → `ValueError("All elements of 'shape' must be positive int!")`
- Float element (`2.0`) → `ValueError` with the same message (the `isinstance(p, int)` test fails, it is
  not raised as a `TypeError`)
- Nested sequence (`[[2], 2]`) → `ValueError`
- **Quirk to pin:** `True` is an `int` subclass, so `[True, 2]` is accepted and `_shape == (True, 2)`.
  Assert current behaviour so a future tightening is deliberate.

#### C. `_check_param_specs(param_specs)`

- Accepts `{}` (no parameters) and a well-formed dict
- Accepts any `Transform` subclass instance, including a custom subclass and a `TransformChain`
- Non-dict → `TypeError("'param_sepc' must be a dict!")` — **note the typo in the source**; the test
  should match the actual string so the message is not silently "fixed" without a plan update
- Non-`str` parameter name → `TypeError("Parameter name must be a str, got int!")`
- Parameter name containing `/` → `ValueError("Invalid parameter name '{name}': '/' is not allowed!")`,
  mirroring the operand-name rule in `Operator._check_operands`. `/` is reserved for the composite
  namespace, so it can never be part of a leaf name; any other character, `.` included, is accepted
- Empty parameter name → `ValueError("Parameter name must not be empty!")`. An empty leaf name makes
  the composite key `"A/"` — the same string a trailing-slash subtree query would use — so the name
  must be rejected for `/` to be unambiguous syntax at every position
- Non-dict spec → `TypeError("Individual parameter specification must be a dict, got str!")`
- Field set not exactly `{"fixed", "default", "trans"}` → `TypeError`, matching
  `"Parameter specification fields must be 'fixed', 'default', and 'trans'"`. Cover **missing** a field
  and **extra** a field (the check is a sorted-key equality)
- `fixed` not a `bool` (e.g. `1`, `"yes"`) → `TypeError`; `True`/`False` pass
- `default` not a tensor → `TypeError("... must be a torch.Tensor, got list!")`
- `default` not 1D → `TypeError("... must be a 1D torch.Tensor, got torch.Size([1, 1])!")` — raised as
  `TypeError`, not `ValueError`, despite being about shape
- `trans` not a `Transform` → `TypeError("... must be a Transform, got str!")`
- Parameter names are otherwise used verbatim: no normalisation, no uniqueness rule beyond the dict
  keys themselves

#### D. `_check_param_tensor(params, length=None)`

- Non-tensor → `TypeError("Parameters must be a Torch tensor!")`; include a `list` input
- 2D → `ValueError("Parameters must be a 1D tensor!")`; also cover 0-d (`torch.tensor(1.0)`)
- `length` mismatch → `ValueError("Parameters must have length {length}, got {n}!")`
- `length=0` is enforced, not skipped: an empty tensor passes, a non-empty one raises
- `length=None` skips the length check
- Returns `(device, dtype)` — assert both, including a non-default dtype and a non-CPU device

#### E. `get_default_dtype_device()`

- No parameters at all → `(torch.get_default_device(), torch.get_default_dtype())`, and it **tracks**
  `torch.set_default_dtype` changes (use `try`/`finally` to restore)
- Parameters present → the shared dtype/device of *all* defaults
- Disagreement among free defaults → `ValueError`; among fixed defaults → `ValueError`; mixed → `ValueError`
- Uses `param_defaults` (fixed + free), not just free
- Operators inherit this and resolve through their operands' defaults — already covered by a `Sum` test
  here; deeper coverage belongs to `test_plan_operator.md`

#### F. `build_params(free_params=None, include_fixed=True, trans=True, out_format="tensor")`

This is the widest method in the module; the four keyword combinations and two transform paths each need
a test.

**Input sources**

- `free_params=None` → falls back to `free_param_defaults` (a dict), so `ScalarMatrix(3).build_params()`
  returns `tensor([1.])` (transform applied to the default `0.0`) and `IdentityMatrix(3).build_params()`
  returns `tensor([])`
- Tensor input of length `num_free_params`
- Dict input, routed through `_from_free_param_dict` (missing/extra key and dtype/device errors are
  covered under §G)
- Validation is length-checked against `num_free_params`, even when the matrix has zero free parameters

**Scatter (the `include_fixed=True` path with both kinds of parameter)**

Use `DiagonalMatrix(3)` with `sigma^2_1` fixed at default `7.5`:

- Free values land at the positions of `free_param_names`, in order — `free_mask` is built from
  `param_specs` insertion order, so it must agree with `free_param_index`
- The fixed slot receives the fixed parameter's **default**
- Transforms are then applied to the **whole** vector, fixed entries included:
  `build_params([1.0, 3.0])` → `[e^2, e^15, e^6]`
- `trans=False` returns the raw merged vector → `[1.0, 7.5, 3.0]`
- All-fixed matrix with `include_fixed=True` → length `num_params`, values from the fixed defaults

**`include_fixed=False`**

- Returns only the free parameters, in `free_param_names` order, transformed with `free_param_trans`
- Zero free parameters → empty tensor (not the `num_params`-length fixed vector)

**Transform application: fast path vs slow path**

`build_params` and `trans_grad` share a dispatch: when every relevant transform has the same type **and**
an equal `__dict__`, the transform is called once on the whole vector; otherwise it is called per element
and concatenated. Both paths must produce identical results, and both must be reached:

- Fast path: `ScalarMatrix`, `DiagonalMatrix` (all `TransformExpPow2()`); two `TransformSigmoid()`
  instances (both have an empty `__dict__`, so they compare equal)
- Slow path by mixed type: `TransformExp()` next to `TransformExpPow2()`
- Slow path by differing `__dict__`: `TransformScaleShift(1.0)` next to `TransformScaleShift(2.0)` —
  same type, unequal instance dict, so the fast-path comparison fails. This is the case a
  type-only check would miss, and it is the sharpest test of the branch
- Equivalence: for a given parameter vector, fast-path output equals slow-path output equals the
  element-wise expectation

**`out_format`**

- `"tensor"` → flat 1D tensor
- `"dict"` → `dict(zip(names, params.unsqueeze(-1)))`, so every value is shape `(1,)`
- `"dict"` with `include_fixed=True` → keys are `param_names` (fixed included)
- `"dict"` with `include_fixed=False` → keys are `free_param_names`
- Unknown value → `ValueError("Unexpected 'out_format': {out_format}!")`, raised both on the
  empty-parameter early branch and on the normal path — cover a parameter-less matrix too
- Zero parameters: `"tensor"` → `torch.tensor([], device=device, dtype=dtype)`; `"dict"` → `{}`

**Value/dtype/device fidelity**

- Output dtype and device follow the input, including `float64` and a non-CPU device
- Output preserves values when `trans=False` is an identity for the free parameters

**Degenerate matrices** (no parameters at all, and all parameters fixed)

The `include_fixed` × `out_format` combinations must also hold when there is nothing to
include, where several branches collapse:

- `include_fixed=False` on a parameter-less matrix, and on an all-fixed one, returns the empty
  tensor — shape `(0,)`, not the `num_params`-length fixed vector
- The same with `out_format="dict"` returns `{}`, not an empty-keyed dict
- `free_params=None` reaches the same empty result on both
- The empty return still follows the input's dtype (`float64` in, `float64` out)
- `trans` is read only after the empty early-return, so `trans=False` and `trans=True` are
  indistinguishable here. Assert that rather than assume it

**Dict values that are not 1D tensors**

`build_params` documents `ValueError` for a dict "with missing or unexpected keys" and says
nothing about the *values*. Pin what actually happens, since none of it is a designed error:

- A 2D value is concatenated first and then caught by `_check_param_tensor` →
  `ValueError("Parameters must be a 1D tensor!")`
- A 0-d value — `torch.tensor(0.5)` where `torch.tensor([0.5])` was meant, the likeliest
  caller mistake — never reaches that check, and escapes as
  `RuntimeError("zero-dimensional tensor (at position 0) cannot be concatenated")` from
  `torch.cat`
- A non-tensor value (`float`, `list`) makes `_from_free_param_dict` read `.device` off it →
  `AttributeError("'float' object has no attribute 'device'")`
- Supplying a *fixed* parameter's name on an all-fixed matrix raises
  `ValueError("Unexpected free parameters: {'...'}!")` — "unexpected" for a name that is in
  `param_names`, merely not free

The last three are pinned as **current** behaviour, not desired behaviour (§4). If they are
fixed, these tests change with them.

#### G. `_from_free_param_dict(free_param_dict)`

Well covered; keep and extend only where the contract changed:

- Non-dict input passes through untouched (identity), including a tensor
- Dict → tensor in `free_param_names` order
- Missing key → `ValueError("Missing free parameters: {...}!")`
- Extra key → `ValueError("Unexpected free parameters: {...}!")`
- Values must share one dtype and device → `ValueError("All free parameters must share the same dtype
  and device!")`. Both checks are on the *values supplied*, not on the defaults, so a dict may override
  defaults that disagree
- Empty dict: key validation still runs first, then a `(0,)` tensor is returned with the dtype/device
  from `get_default_dtype_device()`, i.e. it follows the fixed defaults when there are any, and Torch's
  defaults when there are not
- Empty dict with inconsistent defaults → `ValueError` from `get_default_dtype_device`

#### H. `_to_free_param_dict(free_params)`

- Dict input passes through unchanged (same object)
- Tensor of the right length → `{name: tensor}` with each value shape `(1,)`
- Wrong length → `ValueError("Expected {n} parameters, got {m}!")`
- Skips fixed parameters; the mapping is over `free_param_names`
- Zero free parameters + empty tensor → `{}`
- Note in the plan: no library code calls this method — it is a caller-facing helper. Keep it tested as
  part of the public surface, but a change here breaks nothing internal.

#### H-bis. `set_param_specs(key, *, fixed=None, default=None, trans=None)`

- A `key` not ending in `"/"` is resolved against `param_specs` by exact lookup, so it is the key
  exactly as `param_specs` exposes it: `"sigma^2_1"`, and for composites the flat namespaced form
  `"A/sigma^2"`, `"inner/op_0/sigma^2_0"`
- A `key` ending in `"/"` addresses a **subtree**: every key whose leading segments are the query's,
  the node itself included. `"A/"` is every parameter under operand `A` (nested operands below it too),
  `"/"` is every parameter of the matrix, and `"sigma^2/"` on a leaf matrix is that one parameter
- The match is on segments, not on the raw string: `"A/"` must not reach an operand named `"AB"`. This
  is the test that separates a correct implementation from `key.startswith(prefix)`
- A subtree key matching nothing → `ValueError("Cannot resolve key '{key}'!")`, the same message as an
  unknown exact key; cover a tensor-only operand (no parameters below it) and a query with an empty
  segment (`"A//"`, which no key can match now that names are non-empty)
- Each field passed as non-`None` replaces that field of the resolved specification: `fixed` (including
  back to `False`), `default`, `trans`, and several in one call; the whole spec is replaced by passing
  all three
- Fields left as `None` are untouched — assert the `trans` object identity survives a `default`-only
  call — and a call with no fields at all is a no-op
- The write lands on the stored spec dict, so it takes effect on the next property read or
  `build_params` call; values are assigned as given, with no copy (identity check on the assigned
  `default`)
- Every field is distributed over the resolved specifications the same way: a single value is
  broadcast, the same object reaching every spec (identity check), while a `list`/`tuple` with one
  value per resolved specification is split element-wise. `default` is a tensor, so its single-value
  form is shape `(1,)` and its split form gives each spec a shape `(1,)` slice; `fixed` is a `bool` or
  a `list`/`tuple` of bools; `trans` is a `Transform` or a `list`/`tuple` of transforms (a
  `TransformChain` counts as one transform). Cover the split with three distinct transforms and check
  that `build_params` applies each to its own parameter, and that a length-1 sequence broadcasts like
  the single value
- A field of any other length → `ValueError` naming the field, its value count and the number of
  resolved parameters; a length other than 1 on a one-parameter key is the same error. The checks run
  before any field is written, so a rejected call leaves every spec untouched — including a valid
  `fixed` passed alongside a rejected `trans`
- Wrong field type → `TypeError`: `default` not a `torch.Tensor`; `fixed` neither a `bool` nor a
  `list`/`tuple` of bools, and a non-bool element; `trans` neither a `Transform` nor a `list`/`tuple`
  of transforms, and a non-`Transform` element. A 0-dim or 2-D `default` → `ValueError`
- Composites need no override: `Operator.param_specs` rebuilds a flat dict on every call, but its
  values are the live leaf spec dicts, so mutating the resolved spec persists. Pin this on an
  `Operator` and on a nested one, asserting through the operand's own `param_specs` that the leaf spec
  changed, and that the change reaches `build_params`
- Unknown key → `ValueError("Cannot resolve key '{key}'!")`, for a plain matrix (`"sigma^3"`), for an
  operator (`"A/sigma^3"`), and for a bare operand name (`"A"`, which is not a key)
- Non-str key → `TypeError("'key' must be a str, ...")`
- The fields are keyword-only: a second positional argument is a `TypeError`

#### I. `trans_grad(free_params=None)`

- Length equals `num_free_params`; `free_params=None` uses `free_param_defaults`
- Values are the element-wise `Transform.grad` at the free parameters — verify against a hand-computed
  derivative for the identity, `TransformExpPow2` and one scale-shift transform
- **Fast path**: all free transforms same type and equal `__dict__` → one call on the whole tensor
- **Slow path**: differing types, and same type with differing `__dict__` → per-element `torch.cat`;
  the values must still equal the fast-path result
- Zero free parameters → empty 1D tensor, dtype/device following the input (including `float64`), and
  an `IdentityMatrix` with no parameters at all still returns `(0,)`
- Wrong length → `ValueError("Parameters must have length {n}, got {m}!")`, including the
  `length=0` case
- Only *free* transforms participate: fix a parameter and confirm its transform drops out of the result
- Dict input means the same as the flat tensor: `trans_grad(_to_free_param_dict(p))` equals
  `trans_grad(p)` on a mixed free/fixed matrix, so the fixed names are dropped rather than
  rejected

#### J. `auto_grad(free_params=None)`

- Default `jacobian_method="jacfwd"` → shape `(num_free_params, *shape)`, names `free_param_names`
- `jacrev` → same values; `jacobian` (`torch.autograd.functional.jacobian`) → same values
- **All three methods agree** with each other and with `manual_grad` for a matrix that has both
  (`ScalarMatrix`, `DiagonalMatrix`). This is the cross-check that the `.permute(2, 0, 1)` axis order is
  right for each method
- `jacobian_chunk_size` is honoured by `jacrev` (set it to `1`, confirm identical output) and ignored by
  `jacfwd` / `jacobian`
- Unknown method → `ValueError("Unknown Jacobian method {m!r}! Expected one of 'jacrev', 'jacfwd',
  'jacobian'.")`
- Zero free parameters → `(None, [])`, for `None`, `{}` and an empty tensor
- Input validation still runs when there are no free parameters (a non-empty tensor raises
  `ValueError`)
- **Side effect:** `auto_grad` calls `reset_intermediates()` before differentiating — set a cached
  intermediate, call `auto_grad`, assert `get_intermediates` is now `None`
- Gradients are w.r.t. the raw (untransformed) parameters: `auto_grad` differentiates through
  `__call__`, which calls `build_params` with transforms on, so the chain rule is applied by autodiff.
  Comparing against `manual_grad` (which applies `trans_grad` by hand) pins this down
- Gradients are returned on the same dtype/device as the input
- Dict input means the same as the flat tensor, in values and in names — not just in shape

#### J-bis. `auto_grad` repeatability and call ordering

`auto_grad` clears the intermediate cache before differentiating, and eight subclasses return a
cached intermediate as their result, so the order of calls on one instance is part of the
contract rather than an implementation detail. Parametrized over every test vehicle, comparing
with `torch.equal` against the first result:

- `auto_grad(p)` twice on one instance
- `auto_grad(p)` three times on one instance with nothing in between, comparing every pair
  rather than only first against last, so a call that alternates is caught
- `auto_grad(p)`, then a plain `__call__(p)` that repopulates the cache, then `auto_grad(p)`
- `auto_grad(p)` on a used instance equals `auto_grad(p)` on a fresh one
- `manual_grad(p)` after `auto_grad(p)` equals `manual_grad(p)` on a clean instance — the
  hazard, since the traced `__call__` leaves whatever it cached behind and the closed form
  reads that cache
- `auto_grad(p)` after an earlier `auto_grad(p)` equals the fresh result

The last one is the check that the cache reset is load-bearing: without it, a matrix whose
`__call__` returns a cached object would serve the second call a stale matrix. Every existing
per-matrix `manual_grad` vs `auto_grad` test calls the closed form **first**, so none of them
reach this ordering.

#### K. `manual_grad(free_params=None)` — base implementation

- `Matrix.manual_grad` raises `NotImplementedError`; assert it through a subclass with no override
  (`MinimalMatrix` without its own `manual_grad`, or `IdentityMatrix`)
- The inherited implementation is reached only through `grad` in `"default"` mode, which catches it and
  falls back — see §M
- The **contract** the docstring imposes on subclasses is covered by the parametrized
  `TestManualGradContract`, which runs over every subclass implementing `manual_grad`:
  valid input returns `(grad, names)` with `grad.shape == (num_free_params, *shape)`; empty dict and
  empty tensor raise; an all-fixed matrix returns `(None, [])` for `None`, `{}` and `[]` while still
  rejecting a bad-length tensor. Keep that parametrization and add any newly added matrix to
  `_free_matrices()`
- The same parametrization pins **values**, not just shape: `manual_grad` and `auto_grad` must
  agree under `torch.allclose` for every vehicle, both with all parameters free and with all but
  the first fixed, so each subclass's free/fixed mask is on the path rather than the all-free
  shortcut. This is the only value-level cross-check for `EqualEntryMatrix`,
  `LowerTriangularMatrix` and `UnconstrainedMatrix`, which have no test module of their own
- Dict input is accepted and means the same as the flat tensor, which also gives
  `_to_free_param_dict` a consumer on a real model rather than only in its own unit tests

#### L. `__call__(free_params=None)` — abstract

- `Matrix((2, 2), {})` raises `TypeError` mentioning the abstract method — this is what keeps the ABC
  abstract and guarantees every subclass implements it
- Concrete `__call__` implementations must route through `build_params` (contract note for subclass
  plans, not testable here)

#### M. `grad(free_params=None)`

Four dispatch branches, plus the error path:

- `grad_mode="default"` on a subclass **with** `manual_grad` (e.g. `ScalarMatrix`) → equals
  `manual_grad` output
- `grad_mode="default"` on a subclass **without** `manual_grad` (e.g. `IdentityMatrix`) → falls back to
  `auto_grad`
- `grad_mode="auto"` → equals `auto_grad`, even when the subclass has a `manual_grad` (`ScalarMatrix`)
- `grad_mode="manual"` on a subclass with `manual_grad` → equals `manual_grad`
- **`grad_mode="manual"` on a subclass without `manual_grad` → `NotImplementedError` propagates.** Only
  `"default"` catches it, so this is the one mode where an unimplemented manual gradient is a hard
  error. Untested today
- `grad_mode` set to an unrecognised value → `RuntimeError("Unknown grad mode '{mode}'")`; cover a few
  values (`"invalid"`, `""`, `"MANUAL"`)
- Input validation is not bypassed by the dispatch: `grad({})` on a matrix with free parameters raises
  `ValueError` in every mode (covered by the contract test for `"default"`; extend to `"manual"`/`"auto"`)
- Zero free parameters → `(None, [])` in all three modes
- All three modes agree **per vehicle**, not just on `ScalarMatrix`: dispatch changes the route
  taken, never the values
- `grad` accepts a dict of free parameters and dispatches as it does for a flat tensor

#### N. `set_intermediates(params, intermediates)`

- Stores, and `get_intermediates` returns the **same object** (`is`, not `==`) — arbitrary objects such
  as a dict or a tensor are cached by reference
- Overwrites a previous entry: set twice with the same params, get back the second value
- Empty params (`shape[0] == 0`) → no-op; the previous cache is left untouched (assert a cache set on a
  non-empty vector survives a follow-up empty-set)
- Writes all four keys: `params`, `dtype`, `device`, `intermediates`
- Non-tensor → `TypeError("Parameters must be a Torch tensor!")`; 2D → `ValueError("... 1D tensor!")`
- The stored `params` is a **copy**: an in-place edit of the caller's tensor after the set cannot
  change the key of the entry already cached
- With `cache` off the method is a no-op that returns before validating or writing (§P-bis)

#### O. `get_intermediates(params)`

The three-part guard (values → dtype → device) is the whole point of the method; each part needs a
test that fails if its guard is removed:

- Same values, same dtype, same device → cached object returned
- Different values → `None`
- **Same values, different dtype → `None`.** The dtype is compared explicitly, so an equal-valued
  tensor of another dtype cannot hit the entry. This test is the only thing that pins the guard down
- **Same values, different device → `None`.** Same reasoning as dtype. Runs on `cuda` or `mps` via the
  standard device-pick pattern. The guard must be checked *before* the value comparison, because
  `torch.equal` raises `RuntimeError` across devices rather than returning `False`
- Different order, same values → `None`. Position is part of the key
- Repeated values → `None` when compared against a vector of distinct values
- Distinct values that are neither equal nor permutations → `None`
- `+0.0` and `-0.0` → **cache hit**. `torch.equal` compares numerically, so the key is numerical
  equality rather than bit identity; the sign of a zero cannot change the matrix these parameters build
- After `reset_intermediates()` → `None`
- Empty params → `None`, before any cache lookup
- Non-tensor → `TypeError`; 2D → `ValueError` (the check runs before the empty-params shortcut, so a
  malformed empty tensor still raises)
- Never returns a value that no `set_intermediates` produced: on a fresh matrix it is `None`
- With `cache` off it always misses, returning before validation (§P-bis)

#### P. `reset_intermediates()`

- Clears all four keys to `None`
- Called from `__init__` (fresh matrix caches nothing)
- Called from `auto_grad` (§J)
- Subsequent `set_intermediates` repopulates normally

#### P-bis. `cache`

`cache` (default `True`) gates the caching trio: with the flag off, `set_intermediates` returns
before validating or writing and `get_intermediates` returns `None` before validating, so both
completely ignore the request. `reset_intermediates` is unaffected.

- Default is `True`: an untouched matrix caches exactly as before
- With `cache = False`, `set_intermediates` writes nothing; an entry stored while the flag was on
  stays in `_intermediates`, it is simply unreachable
- With `cache = False`, `get_intermediates` always misses, including for params that hit while the
  flag was on
- Neither method validates while the cache is off: a non-tensor argument is silently ignored
  rather than raising. That is the point of "ignore the request", so pin it
- Re-enabling repopulates on the next `set_intermediates`; nothing cached during the off period
  comes back

#### P-ter. `enable_cache()` / `disable_cache()`

The toggles for `cache`. `enable_cache` sets the flag to `True`; `disable_cache` sets it to `False`
**and** calls `reset_intermediates`, so the blessed way to disable caching cannot leave an entry
behind for a later re-enable to return. `Operator` overrides `reset_intermediates` to recurse into
its operands, so disabling a composite clears its operands' caches as well.

- `disable_cache` clears all four keys, not just the flag: assert `_intermediates` after a set, not
  only `cache`
- Calling it twice, or on a fresh matrix, changes nothing further — `reset_intermediates` is
  idempotent
- `enable_cache` restores caching: a set after it hits, and an entry cleared by the disable does not
  come back

#### Q. Properties

Seventeen properties. Existing tests use one-parameter, all-free matrices, which leaves every
fixed-parameter branch unverified. Use a mixed matrix — `DiagonalMatrix(3)` with `sigma^2_1` fixed —
as the second vehicle throughout:

- `shape` → `_shape`; retains `()` for a shape-less matrix; a list-built shape is a `tuple`
- `param_specs` → the stored dict, **not a copy**; an in-place edit (setting `"fixed"` to `True`) is
  visible on the next call, and `num_free_params` drops accordingly. This is documented behaviour, so
  assert it deliberately
- `param_names` → insertion order of `param_specs`, not sorted
- `free_param_names` / `fixed_param_names` → partitioned, order-preserving, and they partition exactly:
  `free + fixed` as sets equals `param_names`, with no overlap
- `free_param_index` / `fixed_param_index` → indices into `param_specs`; mixed matrix gives
  `[0, 2]` / `[1]`. They must agree with the names lists and with the `free_mask` scatter in
  `build_params`. **The docstrings say "tuple" but a list is returned** — assert list, and note the
  docstring inconsistency in §4
- `num_params` / `num_free_params` / `num_fixed_params` → counts, including non-zero fixed and
  all-fixed (`num_free_params == 0`)
- `param_defaults` / `free_param_defaults` / `fixed_param_defaults` → keys and the exact tensors;
  `fixed_param_defaults` is non-empty in the mixed case; `param_defaults` is the union
- `param_trans` / `free_param_trans` / `fixed_param_trans` → the exact `Transform` **objects** from the
  specs (identity, not copies); `fixed_param_trans` non-empty in the mixed case
- `repr_dict` → `{"shape": ..., "param_specs": ...}`; a subclass overriding it drives `__repr__`
- All properties are read-only views over live state — mutating `param_specs` changes what the next
  call returns

#### R. `__repr__`, `_repr_indented(level)`, `_repr_value(value, level, continuation_pad)`, `_repr_dict(d, level)`

Three modes, all reachable in production (`Operator` and `Adapter` set `_repr_single_line = False`):

**Single-line (`_repr_single_line = True`)**

- Format is `ClassName(key=value, ...)`, and falsy values are **omitted**. Cover:
  - `IdentityMatrix(3)` — `param_specs` is `{}` (falsy) and omitted → `IdentityMatrix(shape=(3, 3))`
  - A shape-less matrix — `shape` is `()` (falsy) too → the args list is empty → `S()`
- `param_specs` is special-cased when it has **≥ 3** entries: rendered as
  `{first_key: first_val, ..., last_key: last_val}` with exactly one `...`. Boundary tests:
  - 1 and 2 parameters → full spec, **no** ellipsis
  - 3 parameters → ellipsis (`len(value) >= 3`)
- Other values go through `repr()` as-is; a `dict` remains on one line in this mode
- A subclass overriding `repr_dict` changes the rendered keys

**Multi-line (`_repr_single_line = False for Operator` / `Adapter`)**

- Header `ClassName(`, one `key=value` per line indented two spaces, closing paren at the parent level
- Each value renders through `_repr_value`:
  - an object with `_repr_indented` (a nested `Matrix` / `Operator`) → its own repr, inlined
  - a `dict` → `_repr_dict` (multi-line, keys `repr`'d, values via `_repr_value`)
  - a `torch.Tensor` → **its `.shape`**, not its contents. Reached through a `repr_dict` override that
    exposes a tensor
  - anything else → `repr(value)`, with embedded newlines re-indented by `continuation_pad` so the
    continuation lines line up under the opening value
- Nesting: a `Sum` inside a `Sum` indents one level deeper each time; assert on the indentation, not
  just the content
- `repr(op)` must not raise for a composite containing tensors, nested operators and empty
  `param_specs`

#### S. `resolve_dim(n)`

The dimension-resolution helper every concrete matrix calls on its `n` (and `m`) argument before
building parameter specs, computing `rho_min`, or sizing a tensor.

- A non-sequence passes through unchanged — `int`, `None`, a float and a string all come back
  identical — so validation stays with `_check_shape` and the `n <= 1` checks in the correlated
  matrices
- A `list`, `tuple` or `pandas.Series` is a vector of levels: the result is its number of distinct
  elements, via `n_distinct`
- Boundaries: duplicates collapse (`["a", "b", "a"]` → `2`), an empty sequence → `0`, and a
  sequence of unhashable elements (`[[1], [2]]`) raises `TypeError` from `set`

#### T. `inverse(free_params=None)`

The default implementation is `self(free_params).inverse()` — build the matrix, then invert it — and
it is what every matrix without a cheaper analytic inverse inherits. Six subclasses now override it
with a closed form (`ScalarMatrix`, `DiagonalMatrix`, `IdentityMatrix`, `EquicorrelationMatrix`,
`AR1Matrix`, `CompoundSymmetricMatrix`); each override has its own section in its own module plan,
and this section covers the default itself and the override point:

- The default equals `self(free_params).inverse()` and `mat(p) @ mat.inverse(p)` is the identity, on
  a vehicle that does not override it
- `None` means the defaults; a dict of free parameters means the same as the flat tensor
- A matrix with no parameters inverts (`IdentityMatrix` → the identity, through its own override)
- Validation is not bypassed: a non-tensor, a wrong-length vector and a bad dict raise from
  `build_params`, exactly as they do through `__call__`
- The result stays attached to `free_params`: `requires_grad` on the output, a non-zero `.grad` for
  every free parameter after `backward()`. Nothing in the default detaches or inverts under `no_grad`
- A subclass that overrides `inverse` is dispatched to, so the extension point the docstring
  promises is real
- `Operator` inherits the default — no operator defines its own `inverse`

**The vehicle decides what is tested.** Most of these cases use `DiagonalMatrix` and
`IdentityMatrix` as vehicles, following the convention that a concrete subclass carries base-class
behavior. Both now override `inverse`, so those cases hold the *contract* to account without
reaching the default implementation. `ThreeParamMatrix` is the vehicle that pins the default
itself — it defines `__call__` and nothing else, so `inverse` can only come from the base class.

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Primary vehicles: `ScalarMatrix(3)` (one free parameter, has `manual_grad`),
  `DiagonalMatrix(3)` (three parameters, mixed free/fixed variants), `IdentityMatrix(3)` (no
  parameters), `MinimalMatrix` (base-class access a concrete matrix cannot reach)
- Reach private validators (`_check_shape`, `_check_param_specs`) through the constructor, and
  `_repr_value` / `_repr_dict` through a `repr_dict` override — do not call them as free functions.
  `resolve_dim` is reached on a `ScalarMatrix` instance, since the base class is abstract
- For every fast-path/slow-path pair, assert **both** that the path was taken (by construction) and
  that the results agree
- Cache tests must isolate each guard: use the *same values* with a different dtype or device, so the
  value comparison would succeed and only that guard can reject the entry
- Device tests pick `cuda` → `mps` → `pytest.skip("no accelerator available")` inline, matching the
  house style

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**. Class-per-method, mirroring the source order:

```python
class TestCheckShape:
    """``_check_shape`` accepts a list, a tuple or a torch.Size, and rejects the rest."""

    @pytest.mark.parametrize("shape,expected", [
        ([2, 3], (2, 3)),
        ((2, 3), (2, 3)),
        (torch.Size([2, 3]), (2, 3)),
    ])
    def test_accepts_sequence_types(self, shape, expected):
        assert MinimalMatrix(shape, {}).shape == expected

    def test_empty_sequence_becomes_empty_tuple(self):
        assert MinimalMatrix([], {}).shape == ()

    def test_rejects_non_sequence(self):
        with pytest.raises(TypeError, match="must be a list, a tuple or a torch.Size"):
            MinimalMatrix(42, {})

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="must be positive int"):
            MinimalMatrix([0, 2], {})

    def test_rejects_float_element_as_value_error(self):
        with pytest.raises(ValueError, match="must be positive int"):
            MinimalMatrix([2.0, 2], {})

    def test_bool_element_is_accepted(self):
        """``bool`` is an ``int`` subclass, so it passes the element check."""
        assert MinimalMatrix([True, 2], {}).shape == (True, 2)
```

```python
class TestIntermediateCacheGuards:
    """Each part of the values/dtype/device key rejects the cache on its own."""

    def test_same_values_different_dtype_is_a_miss(self):
        """The key carries the dtype, so an equal-valued tensor of another dtype misses."""
        mat = ScalarMatrix(3)
        f32 = torch.tensor([1.0], dtype=torch.float32)
        f64 = torch.tensor([1.0], dtype=torch.float64)

        mat.set_intermediates(f32, "value")

        assert mat.get_intermediates(f64) is None
        assert mat.get_intermediates(f32) == "value"

    def test_reordering_free_params_returns_the_right_matrix(self):
        """Regression: a permuted parameter vector must not reuse another model's result."""
        op = BlockDiagonal(a=ScalarMatrix(3), b=ScalarMatrix(2))
        first = op(torch.tensor([0.5, 1.0]))
        second = op(torch.tensor([1.0, 0.5]))
        truth = BlockDiagonal(a=ScalarMatrix(3), b=ScalarMatrix(2))(torch.tensor([1.0, 0.5]))

        assert not torch.equal(second, first)
        assert torch.equal(second, truth)
```

```python
class TestGradDispatch:
    """``grad`` dispatch across the four branches."""

    def test_manual_mode_without_manual_grad_raises(self):
        """Only ``"default"`` mode catches ``NotImplementedError``."""
        mat = IdentityMatrix(3)
        mat.grad_mode = "manual"
        with pytest.raises(NotImplementedError):
            mat.grad(torch.tensor([]))

    def test_default_mode_falls_back_to_auto(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []
```

### 3.3 Test Execution

```bash
pytest tests/covariance/test_matrix.py -q
```

Run the whole covariance suite before finishing, since `Matrix` is the base class for every module the
other plans cover:

```bash
pytest tests/covariance -q
```

---

## 4. Key Risks / Notes

- **Two failing tests were stale, not regressions.** `TestREMLInterface::test_map_theta_to_v` and
  `test_map_theta_to_dv` failed with `AttributeError`: both methods were removed from `Matrix` and
  exist nowhere in the package. The class has been deleted. The same stale API still accounts for the
  24 failures in the other covariance modules noted in `CLAUDE.md`; those are outside this plan's
  scope, but they are the same root cause and the same one-line deletion.
- **The cache key is the parameter vector, and it must compare exactly.** Entries are keyed on the
  built parameter tensor, compared elementwise with `torch.equal` after the dtype and device guards.
  A key that summarises the values rather than comparing them is unsound here, in three ways: it
  cannot distinguish a permutation of a parameter vector, so `[a, b]` and `[b, a]` — genuinely
  different models, and different matrices — share one entry; it can merge a vector with repeated
  values against one without, so `[a, a]` and `[b, c]` are not reliably told apart; and unrelated
  vectors can coincide. The cache is not a pure optimisation — eight classes (`AR1Matrix`,
  `CompoundSymmetricMatrix`, `EquicorrelationMatrix`, and the `BlockDiagonal`, `KroneckerProduct`,
  `HadamardProduct`, `CovariancePropagation`, `Augment` operators) return the cached object as their
  result, so a false hit silently returns another model's covariance matrix. §O pins each of the
  three cases.
- **Guard order matters, and dtype is load-bearing.** Dtype must be compared explicitly, since two
  tensors of equal value but different dtype must not share an entry. Device must be checked *before*
  the value comparison: `torch.equal` raises `RuntimeError` across devices instead of returning
  `False`.
- **The value comparison makes the key numerical, not bitwise.** `+0.0` and `-0.0` share an entry.
  This is the intended contract: the sign of a zero cannot change the matrix those parameters build,
  so the two are interchangeable as model parameters.
- **The fast-path transform check compares `__dict__`, not just type.** Two `TransformScaleShift` with
  different `a` differ in type not at all but in behaviour entirely; a test that only mixes transform
  *types* would not catch a regression to a type-only check. Keep the differing-`__dict__` case.
- **`build_params` applies transforms to fixed parameters too.** The fixed slot is filled with the
  fixed *default* and then transformed along with the free values, so a fixed parameter's reported value
  is `trans(default)` rather than `default`. Tests asserting fixed values must apply the transform (or
  pass `trans=False`).
- **`_check_param_specs` raises `TypeError` for a non-1D `default`**, even though the problem is a
  shape, and its non-dict message has a typo (`'param_sepc'`). Match the real strings; if either is
  changed, update this plan.
- **`_check_shape` accepts `True`/`False` as dimensions** (`bool` is an `int` subclass) and reports a
  float element as `ValueError` rather than `TypeError`. Both are pinned above as current behaviour, not
  as desired behaviour.
- **`free_param_index` / `fixed_param_index` are documented as tuples but return lists.** Assert the
  list; fix the docstrings separately.
- **`_to_free_param_dict` has no in-library caller.** It is tested as public surface but nothing
  internal depends on it — worth remembering before treating its tests as load-bearing.
- **`auto_grad` mutates cache state.** It calls `reset_intermediates()`, so a caller holding a cached
  intermediate across an `auto_grad` call loses it. Covered as an explicit test rather than left as a
  surprise. §J-bis pins the consequences: repeated calls, a repopulated cache in between, and the
  trace-then-`manual_grad` ordering all agree with a clean instance.
- **Three bad-dict paths raise undocumented errors**, pinned in §F as current behaviour rather than
  assertions anyone would design: a 0-d value gives `RuntimeError` from `torch.cat`, a non-tensor value
  gives `AttributeError` from `_from_free_param_dict`, and a fixed parameter's name on an all-fixed
  matrix is reported as "unexpected". The class documents `ValueError`/`TypeError` for bad dict input,
  so these tests record a contract gap rather than endorse it. Fixing the library means updating them.
- **The three matrices with no test module of their own.** `EqualEntryMatrix`,
  `LowerTriangularMatrix` and `UnconstrainedMatrix` appear in `_free_matrices()` but have no
  `test_<module>.py` and no plan, so `TestManualGradContract` is the only thing exercising them at all.
  Its value cross-check (added here) is therefore their only check against autodiff. That cross-check
  was verified to have teeth: multiplying `EqualEntryMatrix.manual_grad`'s result by 2 makes both
  cross-check cases fail.
- **`auto_grad` cost.** `jacrev` allocates per-output; per project notes it OOMs at `n = 512` where
  `jacfwd` stays flat. Keep auto-grad tests at small `n` (`2`–`3`) and use `jacfwd`-friendly sizes when
  cross-checking methods.
- **`MinimalMatrix` must keep validating input.** Its `manual_grad` mirrors the base contract, so the
  base-class tests stay honest about `(None, [])` versus skipped validation.

---

## 5. Expected Test Count Summary

Counts are per collected test — parametrized cases multiply, which is why the
`TestManualGradContract` row (9 matrices × 9 tests) and `TestAutoGradRepeatability` (9 matrices × 6
tests) dominate. The file previously collected 157 tests, 2 of them failing; the first revision of this
plan took it to 280. A second revision, closing gaps found by review, took it to 397. A third revision
adds `set_param_specs` (§H-bis) and takes it to 414. A fourth revision adds the caching toggle — the
`cache` flag (§P-bis) and the `enable_cache` / `disable_cache` pair (§P-ter) — and takes it to 422. A
fifth revision bans `/` from leaf parameter names and rejects the empty name, replacing the test that
accepted `/` with a rejection test plus a `.`-is-still-legal test, and takes it to 424. A sixth
revision adds the trailing-slash subtree form to `set_param_specs` and takes it to 432. A seventh
revision adds `resolve_dim` (§S) — a matrix dimension may be given as a list, tuple or
`pandas.Series` of levels, resolved to its number of distinct elements — and takes it to 442. An
eighth revision adds `inverse` (§T) — the default builds the matrix via `__call__` and inverts the
result, and subclasses are expected to override it with a cheaper analytic inverse — and takes it to
**451**. A ninth revision adds the `ThreeParamMatrix` vehicle and two tests that pin the default
implementation after six subclasses began overriding it, and takes it to **453**. A tenth revision gives
`set_param_specs` broadcast and split semantics for all three fields, so one call sets a whole subtree
from a single value or from one value per parameter, and takes it to **474**.

| Area | Method(s) | Tests |
|---|---|---|
| Constructor & validation | `__init__`, `_check_shape`, `_check_param_specs`, `resolve_dim` | 47 |
| Parameter tensor checks | `_check_param_tensor` | 10 |
| Default resolution | `get_default_dtype_device` | 6 |
| Parameter construction | `build_params` | 47 |
| Dict conversion | `_from_free_param_dict`, `_to_free_param_dict` | 19 |
| Specification updates | `set_param_specs` | 46 |
| Transform derivatives | `trans_grad` | 17 |
| Automatic differentiation | `auto_grad` | 19 |
| Manual differentiation | `manual_grad` + subclass contract | 85 |
| `auto_grad` repeatability | §J-bis, over every vehicle | 54 |
| Abstractness | `__call__` | 2 |
| Dispatch | `grad` | 34 |
| Caching | `set_` / `get_` / `reset_intermediates`, `cache`, `enable_cache` / `disable_cache` | 31 |
| Properties | all 17 | 32 |
| Representation | `__repr__`, `_repr_indented`, `_repr_value`, `_repr_dict` | 14 |
| Inverse | `inverse` | 11 |
| **Total** | | **474** |

Result: `474 passed`. What the eighth revision added, and why:

- **`inverse`** (§T), added to the class after this plan: the default is `self(free_params).inverse()`,
  deliberately the slow generic path, with subclasses expected to override it. The load-bearing tests
  are the override dispatch — an override that stops being called would silently change
  `Operator.inverse`, which no operator defines — and the gradient-flow case, which fails if the
  default is ever wrapped in `no_grad` or switched to a detaching `numpy`-style path. The remaining
  tests only hold the default to the input contract the rest of the class already promises

What the ninth revision changed, and why:

- **The default lost its vehicle.** The eighth revision's tests used `DiagonalMatrix` and
  `IdentityMatrix`, which were plain subclasses at the time. Both have since gained an analytic
  `inverse`, so those cases stopped reaching `matrix.py` and started testing the overrides instead —
  the suite stayed green while the default quietly lost its coverage. `ThreeParamMatrix` restores it:
  a concrete subclass with a `__call__` and no `inverse`, so any pass it produces came from the base
  class. The two new cases are the value/product contract and the gradient flow, which are the two
  the eighth revision called load-bearing

What the tenth revision added, and why:

- **One dispatch rule for all three fields.** A single value is broadcast, a per-parameter sequence is
  split, and the type of the value is what says which: a `bool` vs a `list`/`tuple` of bools, a
  `Transform` vs a `list`/`tuple` of transforms, a length-1 tensor vs a tensor with one value per
  parameter. Before this revision `default` was distributed while `fixed` and `trans` were copied
  wholesale, so `set_param_specs("A/", fixed=[...])` silently handed the list to every spec and failed
  later inside `_check_param_specs`, or in `fixed`'s case aliased a mutable value into every
  specification. The split-with-three-distinct-transforms test is the load-bearing one: it pins that
  each spec gets *its own* transform and that `build_params` applies that one to its own parameter, not
  the last or the first of the list
- **Validation before mutation.** All three fields are checked up front, so a rejected call is atomic.
  The load-bearing case pairs a valid `fixed` with a rejected `trans`: an implementation that validated
  and wrote field by field would leave `fixed` changed by a call that raised, and the test that only
  checks a rejected `default` (the earlier revision's case) would not notice
- **Length-1 sequences broadcast.** A one-element `list`/`tuple` counts as the single-value form, which
  keeps `set_param_specs("/", fixed=[True])` meaning "everything fixed" rather than "this one parameter
  fixed", matching the length-1 tensor's meaning for `default`

What the sixth revision added, and why:

- **The trailing-slash subtree form** of `set_param_specs`, the payoff of the two name bans: with every
  key made of non-empty segments, `key.endswith("/")` can never collide with a stored key, and
  segment-prefix matching is well defined. The load-bearing tests are `"A/"` not reaching an operand
  named `"AB"` (which a `startswith(prefix)` implementation would fail) and `"/"` resolving to
  everything (which a raw `startswith` on the stripped prefix would resolve to nothing, since no key
  starts with `/`)
- **A subtree key that matches nothing** raises the same `ValueError` as an unknown exact key, so a
  typo'd subtree fails loudly rather than silently changing nothing

What the fifth revision added, and why:

- **The `/` ban on leaf parameter names**, matching the operand-name rule in `Operator._check_operands`.
  It replaces a test that pinned the opposite (`"a/b"` accepted, on the grounds that namespacing was the
  operators' concern); once a leaf name may contain `/`, a composite key like `"A/a/b"` is ambiguous
  between a two-level path and a one-level path with a slash in the name, so the old contract was the
  hole rather than the rule. The `.` test is the cheap guard on the other side: the ban is on `/` only,
  so R-style names (`"theta.1"`) keep working.
- **The empty-name ban**, closing the position the `/` ban alone leaves open: `""` contains no `/`, so a
  leaf named `""` under operand `A` produces the key `"A/"` — the exact string a subtree query would
  use. Rejecting it makes every composite key one or more non-empty segments, so `/` is unambiguous at
  every position and no stored key can equal `"A/"` or `"/"`.

What the fourth revision added, and why:

- **`cache`**, the caching switch added to the class after this plan was first written: the default,
  the no-op set, the always-miss get, the skipped validation, and re-enabling. The validation case is
  the one that pins "completely ignore the request": with the flag off, a non-tensor argument must
  return `None` rather than raise, so a test that only checked a valid argument would not notice the
  guards being reordered.
- **`enable_cache` / `disable_cache`**, the toggles added with it (§P-ter). The load-bearing test is
  that `disable_cache` clears `_intermediates` rather than only flipping the flag, because that is
  what stops a re-enable from returning an entry the caller asked to drop.

What the third revision added, and why:

- **`set_param_specs`**, the path-free setter added to the class after this plan was first written:
  field replacement, untargeted fields kept, the all-`None` no-op, exact-key resolution including the
  flat namespaced keys of operators and nested operators, and the keyword-only signature. The composite
  cases matter most: `Operator.param_specs` rebuilds its flat dict on every call, so the tests assert
  the write landed in the *operand's* stored spec and reached `build_params`, which is what pins the
  method to mutating the resolved spec rather than rebinding it.

What the second revision added, and why:

- **Value-level `manual_grad` vs `auto_grad` agreement**, all-free and partially-fixed, over every
  vehicle. The contract class previously asserted shape and names only, which left `EqualEntryMatrix`,
  `LowerTriangularMatrix` and `UnconstrainedMatrix` — none of which has a test module — never compared
  against autodiff anywhere in the suite.
- **§J-bis repeatability and ordering**, including the trace-then-`manual_grad` order that no existing
  cross-check reaches. Both revisions of this pass were checked against a mutation, not just a green
  run: deleting the `reset_intermediates()` call from `auto_grad` fails the consecutive-call tests on
  the three cache-returning subclasses (`AR1Matrix`, `CompoundSymmetricMatrix`,
  `EquicorrelationMatrix`). `test_fresh_instance_agrees` survives that mutation by design — each call
  gets a clean instance, which is exactly the property it is meant to isolate.
- **Degenerate `include_fixed=False` paths** on parameter-less and all-fixed matrices, in both
  `out_format`s.
- **Dict input on the remaining entry points** — `manual_grad`, `grad`, `trans_grad`, and `auto_grad` by
  value rather than by shape.
- **Bad-dict-value behaviour**, pinned as current-not-desired (§4).

Two earlier notes still hold. The device-staleness case no longer skips, and the cache-ordering `xfail`
is now a passing regression test — both followed from keying the cache on an exact comparison (§4). One
case could not be written as originally planned because the underlying behaviour differs from the
assumption: the newline re-indentation in `_repr_value` only applies to a value whose `repr` contains a
literal newline, so the test needed a vehicle whose `repr` does.

The 24 failures elsewhere in `pytest tests/covariance` are the stale `map_theta_to_v` /
`map_theta_to_dv` API documented in `CLAUDE.md`; none is in this module, and none was introduced here.

# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.scalar_matrix`
**Purpose of Module:**
Provides a scaled identity covariance matrix `V = σ² I_n` with a single shared variance parameter. This is the simplest *parameterized* covariance structure, assuming equal, independent variances across all observations.

**Classes Covered:**

- `ScalarMatrix` — scaled identity `V = σ² I_n`

**Testing Goal:**
Ensure correct construction, forward evaluation, and gradient computation of the one-parameter matrix, and pin the dtype/device resolution rules for a matrix whose only dtype and device source is the input parameter tensor. Because `ScalarMatrix` is the smallest concrete `Matrix` subclass with a live parameter, it also carries the shared single-parameter contract: `build_params` composition, the transform chain applied through `trans_grad`, and the intermediate-cache keying — all exercised here through this class as the vehicle for base-class behavior.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor takes `n` (positional or keyword) and an optional `param_specs`; `n` is required
- Instance is a subclass of `Matrix`
- Shape is `(n, n)`, stored as a tuple
- Default `param_specs`: one free parameter `"sigma^2"`, default `[0.0]`, transformed by `TransformExpPow2`
- Default parameter surface: `num_params == 1`, `num_free_params == 1`, `num_fixed_params == 0`; the name, index, defaults and transform properties all report the single `"sigma^2"` entry, and the fixed counterparts are empty
- A falsy `param_specs` (`{}` or `[]`) silently falls back to the defaults, because the constructor uses `param_specs or {...}`
- Two instances never share a `param_specs` dict or a transform instance, so editing one matrix's specs cannot affect another
- `grad_mode` defaults to `"default"`; `jacobian_method` to `"jacfwd"`; `jacobian_chunk_size` to `None`
- `get_default_dtype_device` returns the parameter default's dtype and device, and tracks a change to the Torch default dtype
- `n` validation: `0`, a negative, a float, a string, `None` and a 0-dimensional tensor all raise `ValueError` from the shape check; `True` is accepted because `bool` is an `int` subclass, and only fails later inside `torch.eye`
- `param_specs` validation: a non-dict, a non-str name, a non-dict spec, wrong or extra spec fields, a non-bool `fixed`, a non-tensor or non-1D `default`, and a non-`Transform` `trans` each raise `TypeError` with a specific message

#### B. `__call__(free_params=None)` Behavior

- Returns `σ² I_n` where `σ²` is the transformed parameter: diagonal equals `exp(2θ)`, off-diagonal is exactly zero, and the matrix is symmetric
- The result is positive definite for a wide range of `θ`, including large negative and large positive values, verified through the diagonal being positive and through a successful Cholesky factorisation
- Correct shape for a range of `n`, including the `n = 1` boundary
- Accepts the argument omitted, as a tensor, as a dict, and as the `free_params` keyword; omitting it is identical to passing `free_param_defaults`
- Returns a freshly allocated tensor: mutating a result, or holding two results from consecutive calls, must not affect the matrix or later calls, and call order must not matter
- Rejects a wrong-length tensor, a non-1D or 0-dimensional tensor, a non-tensor, a dict with an extra key, a dict with a missing key, and an unexpected keyword

#### C. Transform Behavior

- The default `TransformExpPow2` maps the raw parameter through `exp(2x)`, and `trans_grad` returns `2exp(2x)`
- Custom transforms are honoured end to end: `TransformExp`, `TransformExp2`, `TransformExp10` and `TransformIdentity` each drive `__call__`, `trans_grad` and `manual_grad` consistently, with `manual_grad` equal to `trans_grad * I_n`
- `TransformScaleShift` and `TransformPow` are accepted even though they are not variance-preserving by construction
- The parameter name is fully configurable and flows through `param_names`, the `grad` name list, and the `build_params` dict keys

#### D. Fixed Parameter Configuration

- A single fixed parameter: `num_free_params == 0`, `num_fixed_params == 1`, and the empty name/index/default/transform properties
- `__call__` uses the fixed default (transform applied) and is constant across calls, for the omitted, empty-tensor, empty-dict and `None` forms
- `build_params` includes the fixed parameter by default and returns an empty tensor when `include_fixed=False`
- Every gradient path (`grad`, `manual_grad`, `auto_grad`) returns `(None, [])`, under `grad_mode` `"default"`, `"manual"` and `"auto"`
- `trans_grad` is empty
- The fixed default's dtype and device become the matrix defaults, and the fixed default's own transform is used

#### E. Multi-Parameter Specifications (pinned limitation)

- A specification with two parameters (both free, or one free and one fixed) constructs successfully and reports the expected parameter surface
- `build_params` then returns one value per parameter, and `__call__` raises `RuntimeError` from the broadcast against the `n x n` identity — for a free-parameter tensor, for the defaults, and for a free-parameter dict
- This is a pinned limitation rather than intended behavior: `ScalarMatrix` defines exactly one shared variance, so no `n x n` result could be built from two parameter values. The tests exist so that a future change to the contract surfaces as a test change. `EqualEntryMatrix` has the same structure and the same limitation

#### F. dtype and Device Resolution

- The input parameter tensor is the single source of truth: the output follows its dtype (float32 and float64) and its device
- An integer input is promoted to the Torch default float dtype by the transform, so an `int64` parameter yields a `float32` matrix
- With no input, the Torch default dtype and device are used, resolved at call time rather than at construction time
- A call that casts the matrix does not change the dtype or device of later default calls, and the defaults stay on the default device after an accelerator call
- The input dtype wins when it disagrees with the Torch default dtype
- `build_params`, `trans_grad`, `grad`, `manual_grad` and `auto_grad` all follow the input dtype and device
- A dtype and a device that both differ from the defaults are followed together; the accelerator dtype used is `float16`, since MPS rejects `float64` outright, and that rejection is pinned by its own MPS-only test
- Device tests pick `cuda` → `mps` → `pytest.skip("no accelerator available")` inline

#### G. `build_params` Behavior

- Returns the transformed value by default, and the raw parameter with `trans=False`
- `include_fixed=False` keeps the free parameter, and with `trans=False` returns it raw
- `"dict"` output returns one entry keyed by the parameter name, whose value is a 1-D length-1 tensor — the `unsqueeze(-1)` in `build_params` is undone by `zip` iterating over dimension 0, so the value is *not* `(1, 1)`; the tensor and dict outputs carry the same values
- Dict input is accepted and matches the tensor input
- An unknown `out_format` raises `ValueError`, both on the parameterized branch and on the empty-output branch
- A dict with an extra key, or an empty dict, raises `ValueError`

#### H. Gradient Behavior

- `manual_grad` returns `(1, n, n)` and `["sigma^2"]`, with the gradient equal to `2exp(2θ) I_n`, non-zero only on the diagonal, and different at different `θ`
- `manual_grad` and `auto_grad` agree at several `θ`, for the default and for every custom transform tested
- `auto_grad` agrees with `manual_grad` under `"jacfwd"`, `"jacrev"` and `"jacobian"`, and with `jacrev` chunked
- `auto_grad` raises `ValueError` for an unknown `jacobian_method`
- `grad` dispatches per `grad_mode`: `"default"` and `"manual"` use `manual_grad`, `"auto"` uses `auto_grad`, and an unknown mode raises `RuntimeError`
- Omitted arguments and dict arguments are accepted by `manual_grad`, `auto_grad` and `grad`
- Every entry point that takes `free_params` enforces the same contract: wrong length, non-1D, non-tensor and bad dict input all raise, including on the zero-free-parameter path and on an accelerator

#### I. Intermediate Caching

- `get_intermediates` returns `None` before any `set_intermediates` call
- A value set for one parameter tensor is returned for that same tensor and only for it: different values, a different dtype, and a different device each invalidate the entry
- The stored key is a copy, so an in-place edit of the caller's tensor after `set_intermediates` does not corrupt the entry
- `reset_intermediates` clears the entry; overwriting replaces it
- Empty parameters are a no-op on both `set` and `get`
- `auto_grad` clears the cache before differentiating, while `manual_grad` and `__call__` leave it intact
- Both methods validate `params`: a non-tensor raises `TypeError`, a non-1D tensor raises `ValueError`

#### J. `repr`

- Contains the class name, the shape, the parameter name and the transform, is a single line, and exposes `repr_dict` with the shape and the live `param_specs` object

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Test with small `n` values (1, 2, 3, 5) for easy manual verification, plus the `n = 1` boundary
- Compare against the transform's own `__call__` and `grad` rather than hard-coded constants wherever a parametrized transform is involved, so the test tracks the transform rather than duplicating its formula; hard-code `exp(2x)` only for the default
- Use `pytest.mark.parametrize` over `θ`, over `n`, over the transform list, over the wrong-length inputs, and over the validation failures, rather than one test per case
- Use `torch.set_default_dtype` inside a `try`/`finally` so the global default is always restored
- Verify the error contract (type and message) at every entry point that accepts `free_params`: `__call__`, `build_params`, `grad`, `manual_grad`, `auto_grad`, `trans_grad`, `set_intermediates`, `get_intermediates`
- Assert positive definiteness structurally (Cholesky succeeds) as well as numerically, since `V = σ² I` must be a valid covariance matrix for every `θ`
- Confirm the suite bites with mutation testing: mutating the default transform, dropping `trans_grad` from `manual_grad`, replacing the identity with a ones matrix, hard-coding the dtype or the device, ignoring `include_fixed=False`, and ignoring `free_params` in `__call__` each fail the suite. The base-class cache guards (dtype, device, value comparison, key copy) were mutated in `matrix.py` and also caught from this file

### 3.2 Test Structure

Tests are grouped into classes by contract area, following **Arrange → Act → Assert**:

```python
class TestScalarMatrixMultiParamSpecs:
    """``ScalarMatrix`` represents one shared variance and builds exactly one value.

    A specification with more than one parameter still constructs, but
    ``build_params`` then returns ``num_params`` values, so ``__call__``
    cannot broadcast them against the ``n x n`` identity.
    """

    def test_call_raises_on_one_free_one_fixed_param(self):
        mat = self.mixed_param_matrix()
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat(torch.tensor([0.5]))
```

### 3.3 Test Execution

`pytest tests/covariance/test_scalar_matrix.py`

---

## 4. Key Risks / Notes

- **The single-parameter assumption is unchecked**: `__call__` reads `build_params(free_params)`, which returns `num_params` values including fixed ones. That happens to be a length-1 tensor for the one-parameter matrices every caller builds (README, `marginal_reml.py` and `mixed_model_reml.py` all construct `ScalarMatrix(n)`), but any user-supplied spec with two parameters reaches `sigma2 * i_n` with a length-2 tensor and dies on a broadcast `RuntimeError` rather than a contract error. The tests pin the failure rather than asserting a result; see section 2.1.E.
- **Dict output shape is not what the source suggests**: `build_params` writes `dict(zip(names, params.unsqueeze(-1)))`, but `zip` iterates the leading dimension, so the `unsqueeze` is undone and each value is a 1-D length-1 tensor. The `(1,)` shape is pinned, and a future change to, for example, a `params[:, None]`-style construction would surface here.
- **Integer parameters are promoted, not followed**: unlike an empty-parameter matrix, where an integer input dtype is followed verbatim, an `int64` parameter here is passed through `torch.exp`, which returns the Torch default float dtype. The "follow the input dtype" rule therefore holds for floating-point inputs only.
- **`True` is a valid `n` at construction**: `Matrix._check_shape` tests `isinstance(p, int) and p > 0`, and `bool` satisfies both, so `ScalarMatrix(True)` builds a `(True, True)` shape recorded as `(1, 1)` and only fails when `torch.eye(bool)` is called. Both halves are pinned.
- **A falsy `param_specs` is silently upgraded to the default spec**: `param_specs or {...}` means `{}` and `[]` both mean "use the defaults", so a caller cannot express "no parameters" this way. `IdentityMatrix` is the class for a matrix with no parameters.
- **`θ` is unbounded**: `TransformExpPow2` is `exp(2x)`, so a moderate `θ` overflows float32 quickly. Tests use `θ` up to `±20` and assert positivity and Cholesky success rather than exact values at the edge.
- **MPS rejects float64**: the accelerator dtype test uses `float16`, and the float64 rejection is pinned by a separate MPS-only test, since `torch.eye(..., dtype=torch.float64)` on MPS raises `TypeError`.
- **Removed REML interface**: `map_theta_to_v` and `map_theta_to_dv` no longer exist on `Matrix`, so this module no longer tests a REML-specific mapping; `__call__` and `grad` are the interface. The two stale tests that referenced them were dropped, which is why this file no longer appears in the `map_theta_to_*` failure list that the rest of the covariance suite still carries.
- **Cache tests exercise base-class code**: `set_intermediates` / `get_intermediates` live in `matrix.py`; they are covered here through `ScalarMatrix` as the concrete vehicle. One base-class mutation (`set_intermediates` storing instead of returning early on empty parameters) survives, because `get_intermediates` short-circuits on empty parameters identically — the difference is unobservable through the public API.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor, shape, default specs, `n` and `param_specs` validation | 35 |
| `__call__` values and argument forms | 28 |
| Free-parameter validation across entry points | 28 |
| `grad` / `manual_grad` / `auto_grad` / `trans_grad` | 27 |
| Transforms | 24 |
| dtype & device resolution | 21 |
| Fixed parameter configuration | 18 |
| Intermediate caching | 17 |
| `build_params` | 17 |
| Multi-parameter specifications (pinned limitation) | 7 |
| `repr` | 6 |
| **Total** | **228** |

# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.dummy_matrix`
**Purpose of Module:**
Provides a fixed dummy (design) matrix constructed from categorical input data at initialization. The matrix is a one-hot encoding of the Cartesian product of factor levels, with no trainable parameters. Used for fixed effects design matrices in linear mixed-effects models.

**Classes Covered:**

- `DummyMatrix` — fixed dummy matrix from categorical data

**Testing Goal:**
Ensure correct construction from categorical variables, proper handling of levels/ordering/drops, no-param behavior, and correct shape (n × p rather than square). Unlike other Matrix subclasses, `DummyMatrix` produces rectangular matrices (n rows × p columns), so the square-matrix assumptions of the rest of the hierarchy do not apply.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor accepts one or many `list`, `tuple` or `pandas.Series` arguments, including a mix of the two types
- All arguments must have the same length (`ValueError`); a non-sequence argument raises `TypeError` naming its position
- At least one argument is required; with none, construction fails with `IndexError`
- An empty argument list produces zero rows and is rejected by the shape check (`ValueError`)
- Levels must be strings: non-string values raise `TypeError` from the column-name join, and mutually incomparable types raise `TypeError` from the level sort
- Instance is a subclass of `Matrix`
- Shape is `(n, p)` — NOT `(n, n)` — this is a design matrix
- The whole parameter surface is empty: `param_specs`, `param_defaults`, `param_trans` and their free/fixed counterparts are `{}`; the name and index properties are `[]`; `num_params == num_free_params == num_fixed_params == 0`
- `grad_mode` defaults to `"default"`
- `repr_dict` carries `shape` only, unlike the base-class default which also carries `param_specs`
- Constructing under a changed default dtype yields a matrix in that dtype

#### B. Single and Multiple Categorical Variables

- One factor with k levels produces k columns, one-hot encoded per observation
- Multiple factors produce the Cartesian product of their levels, with one 1 per row
- Three factors (2 × 2 × 2) are exercised as the multi-factor expansion case
- Column names are the "⋈" join of the level tuple
- `colnames` length always matches the number of columns, including after drops

#### C. Levels and Ordering

- Default: sorted unique values of each argument
- Custom `levels`: explicit specification, including levels absent from the data (all-zero column) and out-of-order levels
- `lex_order=True` (default): combinations sorted lexicographically
- `lex_order=False`: combinations follow `itertools.product` order, which differs from lex order only when levels are supplied out of order
- Row order always follows the input order; column order is decided by the levels and `lex_order`

#### D. Drop Options

- `drop_first=True`: the first column of the product is removed — for one factor that is one level, for several factors it is a single combination, not one level per factor
- `drop_empty_cols=True`: all-zero columns removed, partial columns kept
- Both options combined
- Dropping every column leaves zero columns and raises `ValueError` from the shape check

#### E. Unknown Combinations

- An observation whose combination is not in the `levels` product triggers one `RuntimeWarning` per offending row
- That row is all zeros; the other rows are unaffected
- Covered for one and two factors

#### F. `__call__(free_params=None)` Behavior

- Returns the stored dummy matrix for every accepted argument form: omitted, `None`, an empty tensor, an empty dict, or the `free_params` keyword
- Result shape equals `shape`
- Returns a freshly allocated tensor: mutating the result, or holding two results from consecutive calls, must not affect the matrix or later calls
- Rejects any non-empty free-parameter input: a non-empty tensor (`ValueError`, length 0) or a dict with any key (`ValueError`, unexpected free parameters)
- Rejects a non-tensor (`TypeError`) and a non-1D or 0-dimensional tensor (`ValueError`)

#### G. dtype and Device Resolution (cross-cutting contract)

- Rule 1: the dtype and the device of the input free-parameter tensor are followed, including non-float dtypes (e.g. `int64`), which are followed verbatim
- Rule 2: with no input, the Torch default dtype and device are used, resolved at call time rather than at construction time, for the device as well as the dtype
- An empty dict input also follows the Torch defaults, since it carries no tensor to read from
- A call that casts the matrix does not change the dtype of later default calls, nor the stored matrix or its column names
- `build_params` and `trans_grad` follow the input dtype as well
- Device tests pick `cuda` → `mps` → `pytest.skip("no accelerator available")` inline

#### H. `build_params` Behavior

- Returns an empty 1D tensor by default, and `{}` in `"dict"` format
- `include_fixed` and `trans` do not change the empty result, in either output format
- Rejects an unknown `out_format` in both the empty-output branch and the parameterized branch
- Rejects non-empty tensor and dict input

#### I. Gradient Behavior

- `grad` and `auto_grad` return `(None, [])` for omitted, `None`, empty-tensor and empty-dict input
- `auto_grad` returns `(None, [])` even when `jacobian_method` is invalid, proving no Jacobian is dispatched for a parameterless matrix
- `manual_grad` raises `NotImplementedError`, with or without arguments
- `grad_mode = "manual"` surfaces that `NotImplementedError`; `grad_mode = "auto"` returns `(None, [])`; an unknown mode raises `RuntimeError`
- Gradient entry points still validate their input on the zero-parameter path: a non-empty tensor raises `ValueError`, a non-tensor raises `TypeError`
- `trans_grad` returns an empty tensor (also when called with no arguments) and validates its input
- `get_default_dtype_device` falls back to the Torch defaults and tracks changes to the default dtype

#### J. Intermediate Caching

- `get_intermediates` returns `None` before any `set_intermediates` call
- `set_intermediates` with empty parameters is a no-op returning `None`, so the cache can never be populated and a later `get_intermediates` is still `None`
- `reset_intermediates` leaves the cache empty
- Both methods validate `params`: a non-tensor raises `TypeError`, a non-1D tensor raises `ValueError`

#### K. `repr`

- Contains the class name and the shape, is a single line, and omits the empty `param_specs`

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use small categorical data (`["a", "b", "a"]`, `["x", "y", "x"]`) so every encoding can be checked against an explicitly written literal
- Pair every column-name assertion with an encoding assertion where the two could drift apart, so a wrong expectation fails loudly rather than passing on names alone
- Test each constructor option independently, then in combination
- Use `pytest.warns` for unknown combinations, and count the warnings to pin the per-row granularity
- Use `torch.set_default_dtype` inside a `try`/`finally` for the default-resolution tests, so the global default is always restored
- Verify the error contract (type and message) at every entry point that accepts `free_params`: `__call__`, `build_params`, `grad`, `auto_grad`, `trans_grad`, `set_intermediates`, `get_intermediates`

### 3.2 Test Structure

Tests are grouped into classes by contract area, following **Arrange → Act → Assert**:

```python
class TestDummyMatrixEncoding:
    """The one-hot encoding produced by the constructor options."""

    def test_two_factor_encoding(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "x", "y"])
        result = mat()
        expected = torch.tensor([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
        ])
        assert torch.equal(result, expected)
```

### 3.3 Test Execution

`pytest tests/covariance/test_dummy_matrix.py`

---

## 4. Key Risks / Notes

- **Aliasing of the returned tensor**: `__call__` resolves the dtype and device with `Tensor.to`, which returns the receiver when nothing needs converting. Returning that directly would hand the caller the internal `_matrix`, so an in-place edit such as `mat()[0, 0] = 99.0` would silently change every later call. `__call__` therefore clones on the no-conversion path, and `test_result_can_be_mutated_without_affecting_later_calls` and `test_consecutive_calls_are_distinct_tensors` pin that down.
- **`drop_first` is not per-factor contrast coding**: the option literally drops the first column of the level product. With one factor that removes one level, but with several factors it removes only the single first combination — a 3 × 3 product keeps 8 columns, not 6. The docstring says "drop the first column" and the code matches; it is easy to misread as dropping a reference level per factor, so `test_drop_first_drops_only_the_first_combination` states the real behavior explicitly.
- **Dropping every column fails the shape check**: with `drop_empty_cols=True` and no observation matching any level, the matrix has zero columns and construction raises `ValueError` from `Matrix._check_shape` rather than returning an empty matrix. The warning for the dropped rows is still emitted first.
- **No-argument construction raises `IndexError`**: `DummyMatrix()` fails at `args[0]` rather than with a validated message. Pinned as-is, so any future input validation will surface as a test change.
- **Levels must be strings and mutually comparable**: non-string levels reach the `"⋈".join` and raise `TypeError`; mixed comparable-incompatible types raise `TypeError` at the level sort. Neither is validated up front, so the error type is pinned but the message is not.
- **`levels` must nest one sequence per argument**: a flat list such as `levels=["a", "b"]` is read as a sequence of *characters* for one factor, silently producing a malformed matrix alongside an unknown-combination warning. This misuse is documented here rather than tested, since the garbage result is not a contract worth pinning.
- **Resolution happens at call time, not construction time**: with no input, the dtype and device come from `Matrix.get_default_dtype_device`, which returns the Torch defaults for a matrix with no parameters. Changing the default dtype after construction therefore changes the result of `mat()`, even though `_matrix` keeps the dtype it was built with.
- **Unused levels produce empty columns**: a level specified in `levels` but absent from the data yields an all-zero column unless `drop_empty_cols=True`.
- **The intermediate cache is dead weight here**: `set_intermediates` returns early when `params` has length 0, and an empty tensor is the only valid input, so the cache can never hold a value. The tests assert the no-op rather than an unreachable populated state.
- **Removed REML interface**: `map_theta_to_v` and `map_theta_to_dv` no longer exist on `Matrix`, so this module no longer tests a REML-specific mapping; `__call__` and `grad` are the interface.
- **Not covered here**: operator integration (`Sum`, `BlockDiagonal`, `Augment` with an identity or dummy operand) is covered by the operator test modules, and these tests do not exercise a REML fit.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor, argument validation, shape, empty parameter surface | 35 |
| Encoding, levels, ordering, drops, unknown combinations | 20 |
| `colnames` | 5 |
| `__call__` values, argument forms, freshness | 10 |
| dtype & device resolution | 14 |
| Free-parameter validation | 13 |
| `build_params` | 12 |
| `grad` / `auto_grad` / `manual_grad` / `trans_grad` | 20 |
| Intermediate caching | 8 |
| `repr` | 4 |
| **Total** | **141** |

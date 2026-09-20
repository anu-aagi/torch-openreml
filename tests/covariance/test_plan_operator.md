# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator`
**Purpose of Module:**
Defines the abstract base class `Operator` (extends `Matrix`) for composite covariance matrices. Operators combine multiple `Matrix` or `torch.Tensor` operands, manage parameter namespacing (`"operand/param"`), delegate `build_params` and `grad` to operands, and present a unified `Matrix` interface.

**Classes Covered:**

- `Operator` (extends `Matrix`, ABC via `__call__`) — base class for composite covariance operators

**Testing Goal:**
Ensure correct operand validation, parameter namespacing, per-operand parameter splitting, per-operand gradient delegation, and operand iteration ordering consistency. Tests use concrete subclasses (primarily `Sum`) as test vehicles.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Constructor & Operand Validation

- Positional args → auto-named `op_0`, `op_1`, ...
- Keyword args → user-defined names
- Single dict arg → treated as named operand mapping
- Rejects mixed positional + keyword
- `_check_operands`: rejects non-dict operands
- `_check_operands`: rejects non-string keys
- `_check_operands`: rejects keys containing `"/"`
- `_check_operands`: rejects non-Matrix/non-Tensor values
- `_check_operands`: requires at least one Matrix operand
- Subclass constraint: `Sum` requires ≥2 operands (tested in Sum plan)
- `_repr_single_line = False`

#### B. Parameter Namespacing

- `param_specs`: aggregated from all Matrix operands with `"operand/param"` keys
- Tensor operands contribute no param_specs
- `param_names` / `free_param_names` / `fixed_param_names` all namespaced
- `num_params` sums across Matrix operands
- `num_free_params` / `num_fixed_params` correct

#### C. `build_params` Delegation

- Splits `free_params` into per-operand slices by `num_free_params`
- Delegates to each Matrix operand's `build_params`
- Tensor operands skipped
- Concatenates results
- Supports `include_fixed`, `trans`, `out_format` passthrough
- `out_format="dict"` returns namespaced keys
- Dict input for `free_params`
- Validates total length

#### D. `build_operands` Method

- Evaluates each operand with its param slice
- Tensor operands cast to the resolved dtype and device (not called)
- Returns list in operand iteration order

#### E. Dtype & Device Resolution

- Resolution order: input params → free-param defaults → Torch defaults
- Input tensor dtype wins over inconsistent operand defaults
- Input dtype wins over a Matrix operand's own default dtype
- Input device wins over a Matrix operand's own default device
- Dict input dtype/device wins over inconsistent operand defaults
- No input: resolved from the operands' free-param defaults
- No free params anywhere: resolved from Torch's default dtype and device
- Tensor operand is cast to the input dtype, never promoted by it
- Tensor operand is cast to the input device
- Nested operator follows the input dtype/device, and its nested defaults are checked
- Raises `ValueError` when Matrix operands disagree on dtype or device in case 2
- Dict input does not raise on disagreement among the defaults it overrides
- Dict input raises when the values themselves disagree

#### F. `operands_grad` Method

- Computes per-operand gradient
- Namespaced gradient names: `"operand/param"`
- Fixed Tensor operands contribute `None` and `[]`
- Matrix operands with all-fixed params contribute `None` and `[]`

#### G. `operands` Property

- Returns `_operands` dict
- Preserves insertion order

#### H. `repr_dict` Property

- Returns `{"operands": self.operands}`

#### I. `call_tree` Method

- Returns `(results, free_params_by_path)`, two dictionaries over one key set
- Root key is `"/"`, holding the operator's own result and the input parameters
- Non-root keys are the operand names leading to a node, joined with `"/"`
- Positional operands keyed `"op_0"`, `"op_1"`, ...
- Nested operator operands recurse, giving keys such as `"inner/op_0"` and `"x/op_0/op_0"`
- Root result equals `__call__`
- Leaf results equal the corresponding `build_operands` entries
- Nested node results equal the nested operator called at its own slice
- `free_params_by_path` splits the input per operand and is `None` for fixed tensor operands
- Fixed tensor operands appear as leaves with `None` parameters
- A fixed tensor operand nested inside a nested operator is a leaf with `None` parameters
- Matrix operands with no free parameters appear as leaves with an empty parameter slice
- An operator with no free parameters anywhere is a valid root, holding an empty parameter tensor
- Accepts an explicit empty parameter tensor when there are no free parameters
- Tree is complete, and its values correct, after a warm-cache forward call
- Accepts a dict input and the default (`None`) input
- Raises `ValueError` on a wrong-length parameter tensor
- Results follow the resolved dtype and device
- A fixed tensor operand is cast to the input dtype, at the root and under a nested operator
- A fixed tensor operand is cast to the input device
- Does not descend into non-operator `Matrix` operands (e.g. `Adapter`)

#### J. `grad_tree` Method

- Returns `(grads, free_params_by_path)`, two dictionaries over one key set
- Key set and key order are the same as `call_tree`'s, `free_params_by_path` included
- Root key is `"/"`, holding the composite's own `(grad, grad_names)` pair
- Every entry is a `(grad, grad_names)` pair, with `grad.shape[0] == len(grad_names)`
- Root entry equals `grad`'s own return
- Leaf entries equal the corresponding operand's own `grad`
- Nested node entries equal the nested operator called at its own slice
- Each node's `grad` has shape `(num_free_params, *shape)` for that node — a nested node's Jacobian is not embedded in its parent's shape
- A node's names are namespaced from that node, and joining the path to them with `"/"` reproduces `free_param_names`
- Fixed tensor operands give `(None, [])` with `None` parameters
- Matrix operands with no free parameters give `(None, [])` with an empty parameter slice
- An operator with no free parameters anywhere is a valid root
- Accepts a dict input and the default (`None`) input
- Raises `ValueError` on a wrong-length parameter tensor
- Under `grad_mode="auto"` the root entry matches the manual one
- Does not descend into non-operator `Matrix` operands (e.g. `Adapter`)

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use `Sum` as the primary concrete test vehicle (simplest Operator)
- Test operand validation through both valid and invalid constructor calls
- Test parameter ordering consistency: the order of operands in build_params/build_operands/operands_grad must match
- Verify namespacing by checking param_specs keys

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance import Sum, ScalarMatrix, IdentityMatrix
from torch_openreml.covariance.operator import Operator
from torch_openreml.covariance.matrix import Matrix


class TestOperatorConstructor:
    """Tests for operand validation in Operator.__init__."""

    def test_positional_auto_names(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        assert list(op.operands.keys()) == ["op_0", "op_1"]

    def test_keyword_names(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert list(op.operands.keys()) == ["a", "b"]

    def test_dict_arg(self):
        op = Sum({"a": ScalarMatrix(3), "b": ScalarMatrix(3)})
        assert list(op.operands.keys()) == ["a", "b"]

    def test_rejects_mixed_args_kwargs(self):
        with pytest.raises(ValueError):
            Sum(ScalarMatrix(3), b=ScalarMatrix(3))

    def test_check_operands_rejects_non_dict(self):
        with pytest.raises(TypeError):
            Operator("not_a_dict")

    def test_check_operands_rejects_non_string_key(self):
        with pytest.raises(TypeError, match="Operand name must be a string"):
            Operator({1: ScalarMatrix(3), "b": ScalarMatrix(3)})

    def test_check_operands_rejects_slash_in_key(self):
        with pytest.raises(ValueError, match="'/' is not allowed"):
            Operator({"a/b": ScalarMatrix(3), "c": ScalarMatrix(3)})

    def test_check_operands_rejects_non_matrix_tensor(self):
        with pytest.raises(TypeError, match="must be a Matrix or torch.Tensor"):
            Operator({"a": "not_a_matrix", "b": ScalarMatrix(3)})

    def test_check_operands_requires_at_least_one_matrix(self):
        with pytest.raises(TypeError, match="at least one Matrix"):
            Operator(a=torch.eye(3), b=torch.ones(3, 3))


class TestOperatorNamespacing:
    """Tests for parameter namespacing."""

    def test_param_names_prefixed(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert sorted(op.param_names) == ["a/sigma^2", "b/sigma^2"]

    def test_free_param_names_prefixed(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert sorted(op.free_param_names) == ["a/sigma^2", "b/sigma^2"]

    def test_param_specs_prefixed(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert "a/sigma^2" in op.param_specs
        assert op.num_params == 1

    def test_num_params_sums_across_operands(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert op.num_params == 2
        assert op.num_free_params == 2

    def test_tensor_operand_no_params(self):
        op = Sum(a=ScalarMatrix(3), b=torch.eye(3))
        assert op.num_params == 1
        assert op.num_free_params == 1


class TestOperatorBuildParams:
    """Tests for per-operand parameter delegation."""

    def test_splits_params_by_operand(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        built = op.build_params(torch.tensor([0.0, 0.5]))
        assert built.numel() == 2

    def test_dict_format_namespaced(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        result = op.build_params(torch.tensor([0.0, 0.5]), out_format="dict")
        assert "a/sigma^2" in result
        assert "b/sigma^2" in result

    def test_dict_input_namespaced(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        result = op.build_params({
            "a/sigma^2": torch.tensor([0.0]),
            "b/sigma^2": torch.tensor([0.5]),
        })
        assert result.numel() == 2

    def test_dict_input_missing_key(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError, match="Missing"):
            op.build_params({"a/sigma^2": torch.tensor([0.0])})

    def test_wrong_length_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError):
            op.build_params(torch.tensor([0.0, 0.5, 1.0]))

    def test_dict_input_with_tensor_operand(self):
        op = Sum(a=ScalarMatrix(3), b=torch.eye(3))
        result = op.build_params({"a/sigma^2": torch.tensor([0.0])})
        assert result.numel() == 1


class TestOperatorBuildOperands:
    """Tests for build_operands."""

    def test_returns_list_in_order(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        v_groups = op.build_operands(torch.tensor([0.0, 0.5]))
        assert len(v_groups) == 2
        assert v_groups[0].shape == (3, 3)
        assert v_groups[1].shape == (3, 3)

    def test_includes_tensor_operand(self):
        fixed = torch.ones(3, 3)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        v_groups = op.build_operands(torch.tensor([0.0]))
        assert len(v_groups) == 2
        assert torch.equal(v_groups[1], fixed)


class TestOperatorDtypeDevice:
    """Tests for the single dtype/device resolved for the whole composite."""

    def test_input_dtype_wins(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_no_input_uses_free_param_defaults(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op().dtype == torch.float32

    def test_no_params_uses_torch_defaults(self):
        op = Sum(a=IdentityMatrix(3), b=IdentityMatrix(3))
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()

    def test_mixed_free_defaults_raise(self):
        b = ScalarMatrix(3)
        b.param_specs["sigma^2"]["default"] = torch.tensor([0.5], dtype=torch.float64)
        op = Sum(a=ScalarMatrix(3), b=b)
        with pytest.raises(ValueError, match="same dtype and device"):
            op()

    def test_dict_input_overrides_mixed_defaults(self):
        b = ScalarMatrix(3)
        b.param_specs["sigma^2"]["default"] = torch.tensor([0.5], dtype=torch.float64)
        op = Sum(a=ScalarMatrix(3), b=b)
        v = op({
            "a/sigma^2": torch.tensor([0.5], dtype=torch.float64),
            "b/sigma^2": torch.tensor([0.5], dtype=torch.float64),
        })
        assert v.dtype == torch.float64

    def test_dict_input_mixed_values_raise(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError, match="same dtype and device"):
            op({
                "a/sigma^2": torch.tensor([0.5], dtype=torch.float32),
                "b/sigma^2": torch.tensor([0.5], dtype=torch.float64),
            })

    def test_tensor_operand_is_cast_not_promoted(self):
        fixed = torch.eye(3, dtype=torch.float64)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        v_groups = op.build_operands(torch.tensor([0.5]))
        assert v_groups[1].dtype == torch.float32
        assert torch.equal(v_groups[1], fixed.to(torch.float32))


class TestOperatorOperandsGrad:
    """Tests for operands_grad."""

    def test_returns_per_operand_grads(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grad_groups, name_groups = op.operands_grad(torch.tensor([0.0, 0.5]))
        assert len(grad_groups) == 2
        assert grad_groups[0].shape == (1, 3, 3)
        assert grad_groups[1].shape == (1, 3, 3)
        assert name_groups[0] == ["a/sigma^2"]
        assert name_groups[1] == ["b/sigma^2"]

    def test_tensor_operand_none_grad(self):
        op = Sum(a=ScalarMatrix(3), b=torch.eye(3))
        grad_groups, name_groups = op.operands_grad(torch.tensor([0.5]))
        assert grad_groups[0] is not None
        assert grad_groups[1] is None
        assert name_groups[1] == []

    def test_fixed_matrix_operand_none_grad(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        grad_groups, name_groups = op.operands_grad(torch.tensor([0.5]))
        assert grad_groups[0] is not None
        assert grad_groups[1] is None
        assert name_groups[1] == []


class TestOperatorRepr:
    """Tests for Operator repr."""

    def test_operands_property(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert isinstance(op.operands, dict)
        assert len(op.operands) == 2

    def test_repr_dict(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert "operands" in op.repr_dict
        assert op.repr_dict["operands"]["a"] is not None

    def test_repr_multiline(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        r = repr(op)
        assert "\n" in r  # _repr_single_line = False


class TestOperatorCallTree:
    """Tests for call_tree."""

    def test_root_result_matches_call(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        results, _ = op.call_tree(free_params)
        assert torch.equal(results["/"], op(free_params))

    def test_nested_paths(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        results, _ = op.call_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        assert sorted(results) == ["/", "extra", "inner", "inner/op_0", "inner/op_1"]

    def test_params_split_per_operand(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        _, free_params_by_path = op.call_tree(free_params)
        assert torch.equal(free_params_by_path["/"], free_params)
        assert torch.equal(free_params_by_path["inner/op_0"], torch.tensor([0.0, 0.5]))
        assert torch.equal(free_params_by_path["extra"], torch.tensor([0.5]))

    def test_adapter_operand_is_leaf(self):
        param_specs = {
            "logit": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformIdentity(),
            }
        }

        def param_map(params):
            p = torch.sigmoid(params[0])
            return torch.stack([p, 1 - p])

        adapter = Adapter(DiagonalMatrix(2), param_specs, param_map)
        op = Sum(a=adapter, b=ScalarMatrix(2))
        results, free_params_by_path = op.call_tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "a", "b"]
        assert sorted(free_params_by_path) == ["/", "a", "b"]
        assert torch.equal(free_params_by_path["a"], torch.tensor([0.0]))
        assert torch.equal(results["a"], adapter(torch.tensor([0.0])))


class TestOperatorGradTree:
    """Tests for grad_tree."""

    def test_root_grad_matches_grad(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grads, _ = op.grad_tree(free_params)
        grad, grad_names = op.grad(free_params)
        assert torch.equal(grads["/"][0], grad)
        assert grads["/"][1] == grad_names

    def test_node_grad_shape_is_its_own(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        grads, _ = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        assert grads["/"][0].shape == (4, 4, 4)
        assert grads["inner"][0].shape == (3, 4, 4)
        assert grads["inner/op_0"][0].shape == (2, 2, 2)

    def test_names_join_to_free_param_names(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        grads, _ = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        joined = [f"{path}/{name}" for path in ["inner/op_0", "inner/op_1", "extra"] for name in grads[path][1]]
        assert joined == op.free_param_names
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Operand iteration order**: Python dict preserves insertion order (3.7+). All methods that iterate over `self.operands` (`build_params`, `build_operands`, `operands_grad`) must use the same order. Parameter splitting and gradient concatenation depend on this.
- **Namespacing convention**: `"operand_name/param_name"` with `/` as separator. This is why operand names must not contain `/`.
- **At least one Matrix**: The Operator requires at least one Matrix operand (to have params to optimize). Pure-tensor operators don't make sense since they'd have no free params and would just be a fixed matrix.
- **`_repr_single_line = False`**: Unlike base `Matrix`, operators use multi-line repr because they contain nested operand reprs.
- **One dtype and device for the whole composite**: `build_operands` resolves it once and casts every operand to it. `torch.cat` promotes silently on a dtype mismatch and raises on a device mismatch, so a disagreement among the operands' free-param defaults is caught up front with a `ValueError`; a disagreement the caller overrides with input parameters is not an error. Fixed tensor operands are never a resolution source — they are cast and follow.
- **`call_tree` paths**: The tree is flat, keyed by operand names joined with `/` and rooted at `"/"`. Those keys are unambiguous only because operand names cannot contain `/` — the same constraint the namespacing convention rests on. `call_tree` recurses by calling itself on nested operators instead of capturing the calls a forward pass makes, so the tree is complete whether or not an operator's intermediate cache is warm; a capture-based implementation would lose the subtree of any nested operator whose cache is warm, since `_get_or_build_intermediates` then skips `build_operands`. Because a node's composed result comes from `__call__` while its operands are evaluated separately, every node below the root is entered twice — once while its parent composes, once by the descent — and an operator that does not cache propagates that second entry to its own operands.
- **`call_tree` cost**: Measured per-node `__call__` counts for `Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))` are `{root: 1, inner: 2, extra: 2, inner/op_0: 2, inner/op_1: 2}` cold. With a preceding `op(fp)`, the entries under the caching `BlockDiagonal` drop to 1 (`inner/op_0` and `inner/op_1`), because `inner`'s second entry is served from its cache and never re-enters `build_operands`. `extra`, a direct child of the non-caching `Sum`, stays at 2. A plain `op(fp)` calls every node once. Warm and cold produce identical values.
- **`grad_tree` cost**: Gradients are not cached, so a preceding `grad()` on the root changes nothing — warm and cold counts are identical. A node at depth *d* is evaluated *d + 1* times: once for the root's Jacobian, once for each intermediate ancestor's own entry, and once for its own. Measured for the three-level `Sum(x=BlockDiagonal(BlockDiagonal(...), ...), extra=...)` composite: root 1, depth-1 nodes 2, depth-2 nodes 3, depth-3 nodes 4. Under a `BlockDiagonal`, the same tree built during the cold case also runs one `__call__` per node, all of which the warm case serves from the matrix cache.
- **`grad_tree` names are node-relative**: A node's `grad_names` are namespaced from that node, matching what `grad()` returns for it, so they are not directly comparable with `free_param_names` until the path is joined to them. This mirrors `free_params_by_path[path]` holding the node's own slice rather than the root's tensor. Tested by `test_names_join_to_free_param_names`.
- **`grad_tree` does not show the embedding**: A node's Jacobian is in that node's own shape, so how a child's Jacobian is placed into its parent's — `Sum` concatenating, `BlockDiagonal` padding into a block, the products expanding — is not represented. Only the root entry has the composite's full shape. Exposing the embedding would need each operator to describe how it places a child's Jacobian.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & validation | 9 |
| Parameter namespacing | 5 |
| `build_params` delegation | 6 |
| `build_operands` | 2 |
| Dtype & device resolution | 14 |
| `operands_grad` | 3 |
| repr & properties | 3 |
| `call_tree` | 23 |
| `grad_tree` | 17 |
| **Total** | **82** |

import torch
import pytest
from torch_openreml.covariance import (
    Adapter,
    BlockDiagonal,
    DiagonalMatrix,
    Sum,
    ScalarMatrix,
    IdentityMatrix,
)
from torch_openreml.covariance.operator import Operator
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformIdentity


class MinimalOperator(Operator):
    """Minimal concrete Operator subclass for testing the base class directly."""

    def __call__(self, free_params):
        v_groups = self.build_operands(free_params)
        v = sum(v_groups)
        self._shape = tuple(v.shape)
        return v

    def manual_grad(self, free_params):
        grad_groups, grad_name_groups = self.operands_grad(free_params)
        grad_groups = [g for g in grad_groups if g is not None]
        if len(grad_groups) > 0:
            grad = torch.cat(grad_groups)
            grad_names = [n for group in grad_name_groups for n in group]
            return grad, grad_names
        else:
            return None, []


class TestOperatorConstructor:
    """Tests for operand validation in Operator.__init__."""

    def test_positional_auto_names(self):
        op = MinimalOperator(ScalarMatrix(3), ScalarMatrix(3))
        assert list(op.operands.keys()) == ["op_0", "op_1"]

    def test_keyword_names(self):
        op = MinimalOperator(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert list(op.operands.keys()) == ["a", "b"]

    def test_dict_arg(self):
        op = MinimalOperator({"a": ScalarMatrix(3), "b": ScalarMatrix(3)})
        assert list(op.operands.keys()) == ["a", "b"]

    def test_rejects_mixed_args_kwargs(self):
        with pytest.raises(ValueError):
            MinimalOperator(ScalarMatrix(3), b=ScalarMatrix(3))

    def test_check_operands_rejects_non_dict(self):
        with pytest.raises(TypeError):
            MinimalOperator("not_a_dict")

    def test_check_operands_rejects_non_string_key(self):
        with pytest.raises(TypeError, match="Operand name must be a string"):
            MinimalOperator({1: ScalarMatrix(3), "b": ScalarMatrix(3)})

    def test_check_operands_rejects_slash_in_key(self):
        with pytest.raises(ValueError, match="'/' is not allowed"):
            MinimalOperator({"a/b": ScalarMatrix(3), "c": ScalarMatrix(3)})

    def test_check_operands_rejects_non_matrix_tensor(self):
        with pytest.raises(TypeError, match="must be a Matrix or torch.Tensor"):
            MinimalOperator({"a": "not_a_matrix", "b": ScalarMatrix(3)})

    def test_check_operands_requires_at_least_one_matrix(self):
        with pytest.raises(TypeError, match="at least one Matrix"):
            MinimalOperator(a=torch.eye(3), b=torch.ones(3, 3))


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


class TestOperatorCallTree:
    """Tests for call_tree."""

    def test_root_result_matches_call(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        results, _ = op.call_tree(free_params)
        assert torch.equal(results["/"], op(free_params))

    def test_paths_are_operand_names(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results, free_params_by_path = op.call_tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "a", "b"]
        assert sorted(free_params_by_path) == ["/", "a", "b"]

    def test_positional_operand_names(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        results, _ = op.call_tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "op_0", "op_1"]

    def test_leaf_results_match_build_operands(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        results, _ = op.call_tree(free_params)
        v_groups = op.build_operands(free_params)
        assert torch.equal(results["a"], v_groups[0])
        assert torch.equal(results["b"], v_groups[1])

    def test_nested_paths(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        results, _ = op.call_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        assert sorted(results) == ["/", "extra", "inner", "inner/op_0", "inner/op_1"]
        assert results["inner/op_0"].shape == (2, 2)
        assert results["inner/op_1"].shape == (2, 2)

    def test_nested_result_matches_operand_call(self):
        inner = BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2))
        op = Sum(inner=inner, extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        results, _ = op.call_tree(free_params)
        assert torch.equal(results["inner"], inner(free_params[:3]))

    def test_deeply_nested_path(self):
        deep = Sum(x=BlockDiagonal(BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), ScalarMatrix(2)), extra=ScalarMatrix(6))
        results, _ = deep.call_tree(torch.tensor([0.0, 0.5, 1.0, 0.5, 0.5]))
        assert results["x/op_0/op_0"].shape == (2, 2)

    def test_nested_tensor_operand_leaf(self):
        inner = Sum(x=ScalarMatrix(2), t=torch.eye(2))
        op = Sum(inner=inner, b=ScalarMatrix(2))
        results, free_params_by_path = op.call_tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "b", "inner", "inner/t", "inner/x"]
        assert free_params_by_path["inner/t"] is None
        assert torch.equal(results["inner/t"], torch.eye(2))

    def test_nested_tensor_operand_follows_input_dtype(self):
        inner = Sum(x=ScalarMatrix(2), t=torch.eye(2))
        op = Sum(inner=inner, b=ScalarMatrix(2))
        results, _ = op.call_tree(torch.tensor([0.0, 0.5], dtype=torch.float64))
        assert results["inner/t"].dtype == torch.float64
        assert results["inner/t"].device == results["/"].device

    def test_params_split_per_operand(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        _, free_params_by_path = op.call_tree(free_params)
        assert torch.equal(free_params_by_path["/"], free_params)
        assert torch.equal(free_params_by_path["inner"], torch.tensor([0.0, 0.5, 1.0]))
        assert torch.equal(free_params_by_path["inner/op_0"], torch.tensor([0.0, 0.5]))
        assert torch.equal(free_params_by_path["extra"], torch.tensor([0.5]))

    def test_zero_free_params_root(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        results, free_params_by_path = op.call_tree()
        assert sorted(results) == ["/", "a", "b"]
        assert results["/"].shape == (2, 2)
        assert free_params_by_path["/"].numel() == 0
        assert free_params_by_path["a"].numel() == 0

    def test_zero_free_params_root_explicit_empty_tensor(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        results, free_params_by_path = op.call_tree(torch.tensor([]))
        assert results["/"].shape == (2, 2)
        assert free_params_by_path["/"].numel() == 0

    def test_tensor_operand_leaf(self):
        fixed = torch.ones(3, 3)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        results, free_params_by_path = op.call_tree(torch.tensor([0.0]))
        assert torch.equal(results["fixed"], fixed)
        assert free_params_by_path["fixed"] is None

    def test_fixed_matrix_operand_leaf(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        results, free_params_by_path = op.call_tree(torch.tensor([0.5]))
        assert results["b"].shape == (3, 3)
        assert free_params_by_path["b"].numel() == 0

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

    def test_full_tree_after_warm_cache(self):
        inner = BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2))
        op = Sum(inner=inner, extra=ScalarMatrix(4))
        free_params = torch.tensor([1.0, 2.0, 3.0, 4.0])
        op(free_params)
        results, _ = op.call_tree(free_params)
        assert sorted(results) == ["/", "extra", "inner", "inner/op_0", "inner/op_1"]
        assert torch.equal(results["inner"], inner(free_params[:3]))
        assert torch.equal(results["extra"], op.operands["extra"](free_params[3:]))
        assert torch.equal(results["inner/op_0"], inner.operands["op_0"](free_params[:2]))
        assert torch.equal(results["/"], results["inner"] + results["extra"])

    def test_default_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results, _ = op.call_tree()
        assert torch.equal(results["/"], op())

    def test_dict_input(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results, free_params_by_path = op.call_tree({
            "a/sigma^2": torch.tensor([0.0]),
            "b/sigma^2": torch.tensor([0.5]),
        })
        assert torch.equal(results["/"], op(torch.tensor([0.0, 0.5])))
        assert torch.equal(free_params_by_path["a"], torch.tensor([0.0]))

    def test_wrong_length_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError):
            op.call_tree(torch.tensor([0.0, 0.5, 1.0]))

    def test_input_dtype_is_used(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        results, _ = op.call_tree(torch.tensor([0.0, 0.5, 1.0, 0.5], dtype=torch.float64))
        assert results["/"].dtype == torch.float64
        assert results["inner/op_0"].dtype == torch.float64

    def test_tensor_operand_follows_input_dtype(self):
        op = Sum(a=ScalarMatrix(2), fixed=torch.eye(2))
        results, _ = op.call_tree(torch.tensor([0.5], dtype=torch.float64))
        assert results["/"].dtype == torch.float64
        assert results["fixed"].dtype == torch.float64

    def test_input_device_is_used(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4), fixed=torch.eye(4))
        results, _ = op.call_tree(torch.tensor([0.0, 0.5, 1.0, 0.5], device=device))
        assert results["/"].device.type == device
        assert results["inner/op_0"].device.type == device
        assert results["fixed"].device.type == device

    def test_minimal_operator_vehicle(self):
        op = MinimalOperator(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results, free_params_by_path = op.call_tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "a", "b"]
        assert sorted(results) == sorted(free_params_by_path)


class TestOperatorGradTree:
    """Tests for grad_tree."""

    def test_root_grad_matches_grad(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grads, _ = op.grad_tree(free_params)
        grad, grad_names = op.grad(free_params)
        assert torch.equal(grads["/"][0], grad)
        assert grads["/"][1] == grad_names

    def test_grad_and_names_pair_per_path(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads, free_params_by_path = op.grad_tree(torch.tensor([0.0, 0.5]))
        assert sorted(grads) == ["/", "a", "b"]
        assert sorted(grads) == sorted(free_params_by_path)
        for grad, grad_names in grads.values():
            assert grad.shape[0] == len(grad_names)

    def test_keys_match_call_tree(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        grads, free_params_by_path = op.grad_tree(free_params)
        results, call_free_params_by_path = op.call_tree(free_params)
        assert sorted(grads) == sorted(results)
        assert sorted(free_params_by_path) == sorted(call_free_params_by_path)

    def test_leaf_grad_matches_operand_grad(self):
        op = Sum(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 1.5])
        grads, _ = op.grad_tree(free_params)
        assert torch.equal(grads["a"][0], op.operands["a"].grad(free_params[:1])[0])
        assert torch.equal(grads["b"][0], op.operands["b"].grad(free_params[1:])[0])

    def test_nested_grad_matches_operand_grad(self):
        inner = BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2))
        op = Sum(inner=inner, extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        grads, _ = op.grad_tree(free_params)
        assert torch.equal(grads["inner"][0], inner.grad(free_params[:3])[0])
        assert grads["inner"][1] == inner.free_param_names

    def test_node_grad_shape_is_its_own(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        grads, _ = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        assert grads["/"][0].shape == (4, 4, 4)
        assert grads["inner"][0].shape == (3, 4, 4)
        assert grads["inner/op_0"][0].shape == (2, 2, 2)
        assert grads["inner/op_1"][0].shape == (1, 2, 2)
        assert grads["extra"][0].shape == (1, 4, 4)

    def test_names_join_to_free_param_names(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        grads, _ = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        joined = [f"{path}/{name}" for path in ["inner/op_0", "inner/op_1", "extra"] for name in grads[path][1]]
        assert joined == op.free_param_names

    def test_deeply_nested_path(self):
        deep = Sum(x=BlockDiagonal(BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), ScalarMatrix(2)), extra=ScalarMatrix(6))
        grads, _ = deep.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5, 0.5]))
        assert sorted(grads) == ["/", "extra", "x", "x/op_0", "x/op_0/op_0", "x/op_0/op_1", "x/op_1"]
        assert grads["x/op_0/op_0"][0].shape == (2, 2, 2)
        assert grads["x/op_0"][0].shape == (3, 4, 4)

    def test_tensor_operand_has_no_grad(self):
        op = Sum(a=ScalarMatrix(2), fixed=torch.eye(2))
        grads, free_params_by_path = op.grad_tree(torch.tensor([0.5]))
        assert grads["fixed"] == (None, [])
        assert free_params_by_path["fixed"] is None

    def test_fixed_matrix_operand_has_no_grad(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        grads, free_params_by_path = op.grad_tree(torch.tensor([0.5]))
        assert grads["b"] == (None, [])
        assert free_params_by_path["b"].numel() == 0

    def test_zero_free_params_root(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        grads, free_params_by_path = op.grad_tree()
        assert sorted(grads) == ["/", "a", "b"]
        assert grads["/"] == (None, [])
        assert free_params_by_path["/"].numel() == 0

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
        free_params = torch.tensor([0.0, 0.5])
        grads, free_params_by_path = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a", "b"]
        assert torch.equal(grads["a"][0], adapter.grad(free_params[:1])[0])
        assert torch.equal(free_params_by_path["a"], free_params[:1])

    def test_default_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads, _ = op.grad_tree()
        assert torch.equal(grads["/"][0], op.grad()[0])

    def test_dict_input(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads, free_params_by_path = op.grad_tree({
            "a/sigma^2": torch.tensor([0.0]),
            "b/sigma^2": torch.tensor([0.5]),
        })
        assert torch.equal(grads["/"][0], op.grad(torch.tensor([0.0, 0.5]))[0])
        assert torch.equal(free_params_by_path["a"], torch.tensor([0.0]))

    def test_wrong_length_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError):
            op.grad_tree(torch.tensor([0.0, 0.5, 1.0]))

    def test_auto_grad_mode(self):
        op = Sum(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 1.5])
        manual, _ = op.grad_tree(free_params)
        op.grad_mode = "auto"
        auto, _ = op.grad_tree(free_params)
        assert torch.allclose(manual["/"][0], auto["/"][0])
        assert auto["/"][1] == op.free_param_names

    def test_minimal_operator_vehicle(self):
        op = MinimalOperator(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads, free_params_by_path = op.grad_tree(torch.tensor([0.0, 0.5]))
        assert sorted(grads) == ["/", "a", "b"]
        assert sorted(grads) == sorted(free_params_by_path)


class TestOperatorDtypeDevice:
    """Tests for dtype and device resolution across a composite."""

    def test_input_tensor_dtype_is_used(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_input_dict_dtype_is_used(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        v = op({
            "a/sigma^2": torch.tensor([0.5], dtype=torch.float64),
            "b/sigma^2": torch.tensor([0.5], dtype=torch.float64),
        })
        assert v.dtype == torch.float64

    def test_input_device_is_used(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.tensor([0.5], device=device)).device.type == device

    def test_no_input_uses_free_param_defaults(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op().dtype == torch.float32

    def test_no_params_uses_torch_defaults(self):
        op = Sum(a=IdentityMatrix(3), b=IdentityMatrix(3))
        assert op.num_params == 0
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()

    def test_tensor_operand_follows_input_dtype(self):
        op = Sum(a=ScalarMatrix(3), fixed=torch.eye(3, dtype=torch.float64))
        assert op(torch.tensor([0.5], dtype=torch.float32)).dtype == torch.float32

    def test_tensor_operand_casts_up_to_input_dtype(self):
        op = Sum(a=ScalarMatrix(3), fixed=torch.eye(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_tensor_operand_is_cast_not_promoted(self):
        fixed = torch.eye(3, dtype=torch.float64)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        v_groups = op.build_operands(torch.tensor([0.5]))
        assert v_groups[1].dtype == torch.float32
        assert torch.equal(v_groups[1], fixed.to(torch.float32))

    def test_tensor_operand_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        op = Sum(a=ScalarMatrix(3), fixed=torch.eye(3, device=device))
        assert op(torch.tensor([0.5])).device == torch.device("cpu")

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

    def test_nested_operator_follows_input_dtype(self):
        inner = BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2))
        op = Sum(inner=inner, fixed=torch.eye(4, dtype=torch.float64))
        assert op(torch.full((3,), 0.5, dtype=torch.float64)).dtype == torch.float64

    def test_nested_mixed_defaults_raise(self):
        b = ScalarMatrix(2)
        b.param_specs["sigma^2"]["default"] = torch.tensor([0.5], dtype=torch.float64)
        inner = BlockDiagonal(DiagonalMatrix(2), b)
        op = Sum(inner=inner, extra=ScalarMatrix(4))
        with pytest.raises(ValueError, match="same dtype and device"):
            op()


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
        assert "\n" in r

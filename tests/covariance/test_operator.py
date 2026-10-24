import torch
import pytest
from torch_openreml.covariance import (
    AR1Matrix,
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


class StubMatrix(Matrix):
    """Matrix operand stub recording how operands_inverse consumes it."""

    def __init__(self):
        super().__init__((2, 2), {})
        self.call_count = 0
        self.inverse_count = 0

    def __call__(self, free_params=None):
        self.call_count += 1
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def inverse(self, free_params=None):
        self.inverse_count += 1
        return torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])


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

    def test_check_operands_rejects_empty_key(self):
        """An empty name would put an empty segment in every path it prefixes."""
        with pytest.raises(ValueError, match="Operand name must not be empty!"):
            MinimalOperator({"": ScalarMatrix(3), "c": ScalarMatrix(3)})

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

    def test_dict_format_includes_fixed_params(self):
        op = BlockDiagonal(x=DiagonalMatrix(2), y=ScalarMatrix(2))
        op.set_param_specs("x/sigma^2_0", fixed=True, default=torch.tensor([99.0]))
        result = op.build_params(torch.tensor([1.0, 2.0]), trans=False, out_format="dict")
        assert set(result) == {"x/sigma^2_0", "x/sigma^2_1", "y/sigma^2"}
        assert torch.equal(result["x/sigma^2_0"], torch.tensor([99.0]))
        assert torch.equal(result["x/sigma^2_1"], torch.tensor([1.0]))
        assert torch.equal(result["y/sigma^2"], torch.tensor([2.0]))

    def test_dict_format_free_only(self):
        op = BlockDiagonal(x=DiagonalMatrix(2), y=ScalarMatrix(2))
        op.set_param_specs("x/sigma^2_0", fixed=True, default=torch.tensor([99.0]))
        result = op.build_params(
            torch.tensor([1.0, 2.0]), trans=False, include_fixed=False, out_format="dict"
        )
        assert set(result) == {"x/sigma^2_1", "y/sigma^2"}
        assert torch.equal(result["x/sigma^2_1"], torch.tensor([1.0]))
        assert torch.equal(result["y/sigma^2"], torch.tensor([2.0]))

    def test_dict_format_all_fixed(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        op.set_param_specs("/", fixed=True)
        result = op.build_params(out_format="dict")
        assert set(result) == {"a/sigma^2", "b/sigma^2"}

    def test_dict_format_values_align_with_flat_tensor(self):
        op = BlockDiagonal(x=DiagonalMatrix(2), y=ScalarMatrix(2))
        op.set_param_specs("x/sigma^2_0", fixed=True, default=torch.tensor([99.0]))
        free_params = torch.tensor([1.0, 2.0])
        built = op.build_params(free_params, trans=False)
        result = op.build_params(free_params, trans=False, out_format="dict")
        assert torch.equal(torch.cat([result[name] for name in op.param_names]), built)

    def test_dict_format_nested_alignment(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        op.set_param_specs("inner/op_0/sigma^2_0", fixed=True, default=torch.tensor([8.0]))
        free_params = torch.tensor([1.0, 2.0, 3.0])
        result = op.build_params(free_params, trans=False, out_format="dict")
        assert torch.equal(result["inner/op_0/sigma^2_0"], torch.tensor([8.0]))
        assert torch.equal(
            torch.cat([result[name] for name in op.param_names]),
            op.build_params(free_params, trans=False),
        )

    def test_dict_format_roundtrip(self):
        op = BlockDiagonal(x=DiagonalMatrix(2), y=ScalarMatrix(2))
        op.set_param_specs("x/sigma^2_0", fixed=True, default=torch.tensor([99.0]))
        free_params = torch.tensor([1.0, 2.0])
        as_dict = op.build_params(
            free_params, trans=False, include_fixed=False, out_format="dict"
        )
        assert torch.equal(op.build_params(as_dict, trans=False, include_fixed=False), free_params)

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
    """Tests for tree."""

    def test_root_result_matches_call(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        results = op.tree(free_params)
        assert torch.equal(results["/"], op(free_params))

    def test_paths_are_operand_names(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results = op.tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "a", "b"]

    def test_positional_operand_names(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        results = op.tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "op_0", "op_1"]

    def test_leaf_results_match_build_operands(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        results = op.tree(free_params)
        v_groups = op.build_operands(free_params)
        assert torch.equal(results["a"], v_groups[0])
        assert torch.equal(results["b"], v_groups[1])

    def test_nested_paths(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        results = op.tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        assert sorted(results) == ["/", "extra", "inner", "inner/op_0", "inner/op_1"]
        assert results["inner/op_0"].shape == (2, 2)
        assert results["inner/op_1"].shape == (2, 2)

    def test_nested_result_matches_operand_call(self):
        inner = BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2))
        op = Sum(inner=inner, extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        results = op.tree(free_params)
        assert torch.equal(results["inner"], inner(free_params[:3]))

    def test_deeply_nested_path(self):
        deep = Sum(x=BlockDiagonal(BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), ScalarMatrix(2)), extra=ScalarMatrix(6))
        results = deep.tree(torch.tensor([0.0, 0.5, 1.0, 0.5, 0.5]))
        assert results["x/op_0/op_0"].shape == (2, 2)

    def test_nested_tensor_operand_leaf(self):
        inner = Sum(x=ScalarMatrix(2), t=torch.eye(2))
        op = Sum(inner=inner, b=ScalarMatrix(2))
        results = op.tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "b", "inner", "inner/t", "inner/x"]
        assert torch.equal(results["inner/t"], torch.eye(2))

    def test_nested_tensor_operand_follows_input_dtype(self):
        inner = Sum(x=ScalarMatrix(2), t=torch.eye(2))
        op = Sum(inner=inner, b=ScalarMatrix(2))
        results = op.tree(torch.tensor([0.0, 0.5], dtype=torch.float64))
        assert results["inner/t"].dtype == torch.float64
        assert results["inner/t"].device == results["/"].device

    def test_zero_free_params_root(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        results = op.tree()
        assert sorted(results) == ["/", "a", "b"]
        assert results["/"].shape == (2, 2)

    def test_zero_free_params_root_explicit_empty_tensor(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        results = op.tree(torch.tensor([]))
        assert results["/"].shape == (2, 2)

    def test_tensor_operand_leaf(self):
        fixed = torch.ones(3, 3)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        results = op.tree(torch.tensor([0.0]))
        assert torch.equal(results["fixed"], fixed)

    def test_fixed_matrix_operand_leaf(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        results = op.tree(torch.tensor([0.5]))
        assert results["b"].shape == (3, 3)

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
        results = op.tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "a", "b"]
        assert torch.equal(results["a"], adapter(torch.tensor([0.0])))

    def test_full_tree_after_warm_cache(self):
        inner = BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2))
        op = Sum(inner=inner, extra=ScalarMatrix(4))
        free_params = torch.tensor([1.0, 2.0, 3.0, 4.0])
        op(free_params)
        results = op.tree(free_params)
        assert sorted(results) == ["/", "extra", "inner", "inner/op_0", "inner/op_1"]
        assert torch.equal(results["inner"], inner(free_params[:3]))
        assert torch.equal(results["extra"], op.operands["extra"](free_params[3:]))
        assert torch.equal(results["inner/op_0"], inner.operands["op_0"](free_params[:2]))
        assert torch.equal(results["/"], results["inner"] + results["extra"])

    def test_default_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results = op.tree()
        assert torch.equal(results["/"], op())

    def test_dict_input(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results = op.tree({
            "a/sigma^2": torch.tensor([0.0]),
            "b/sigma^2": torch.tensor([0.5]),
        })
        assert torch.equal(results["/"], op(torch.tensor([0.0, 0.5])))

    def test_wrong_length_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError):
            op.tree(torch.tensor([0.0, 0.5, 1.0]))

    def test_input_dtype_is_used(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        results = op.tree(torch.tensor([0.0, 0.5, 1.0, 0.5], dtype=torch.float64))
        assert results["/"].dtype == torch.float64
        assert results["inner/op_0"].dtype == torch.float64

    def test_tensor_operand_follows_input_dtype(self):
        op = Sum(a=ScalarMatrix(2), fixed=torch.eye(2))
        results = op.tree(torch.tensor([0.5], dtype=torch.float64))
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
        results = op.tree(torch.tensor([0.0, 0.5, 1.0, 0.5], device=device))
        assert results["/"].device.type == device
        assert results["inner/op_0"].device.type == device
        assert results["fixed"].device.type == device

    def test_minimal_operator_vehicle(self):
        op = MinimalOperator(a=ScalarMatrix(3), b=ScalarMatrix(3))
        results = op.tree(torch.tensor([0.0, 0.5]))
        assert sorted(results) == ["/", "a", "b"]


class TestOperatorWalk:
    """Tests for walk."""

    def test_root_entry_is_the_operator_and_the_validated_tensor(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        nodes = op.walk(free_params)
        path, operand, node_params = nodes[0]
        assert path == "/"
        assert operand is op
        assert node_params is free_params

    def test_order_matches_the_tree_methods(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        paths = [path for path, _, _ in op.walk(free_params)]
        assert paths == list(op.tree(free_params)) == list(op.free_param_tree(free_params))
        assert paths == list(op.grad_tree(free_params))

    def test_slices_follow_the_operands(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        nodes = {
            path: (operand, node_params)
            for path, operand, node_params in op.walk(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        }
        assert torch.equal(nodes["inner"][1], torch.tensor([0.0, 0.5, 1.0]))
        assert torch.equal(nodes["inner/op_0"][1], torch.tensor([0.0, 0.5]))
        assert torch.equal(nodes["extra"][1], torch.tensor([0.5]))

    def test_fixed_tensor_operand_gets_none(self):
        op = Sum(a=torch.eye(2), b=DiagonalMatrix(2))
        nodes = {path: node_params for path, _, node_params in op.walk(torch.tensor([0.0, 0.5]))}
        assert nodes["a"] is None
        assert torch.equal(nodes["b"], torch.tensor([0.0, 0.5]))

    def test_nested_tensor_operand_gets_none(self):
        op = Sum(inner=BlockDiagonal(torch.eye(2), ScalarMatrix(2)), extra=ScalarMatrix(2))
        nodes = {path: node_params for path, _, node_params in op.walk(torch.tensor([0.0, 0.5]))}
        assert nodes["inner/op_0"] is None
        assert torch.equal(nodes["inner/op_1"], torch.tensor([0.0]))

    def test_leaf_operands_are_the_operand_instances(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        nodes = {path: operand for path, operand, _ in op.walk(torch.tensor([0.0, 0.5, 1.0, 0.5]))}
        assert nodes["inner"] is op.operands["inner"]
        assert nodes["inner/op_0"] is op.operands["inner"].operands["op_0"]
        assert nodes["extra"] is op.operands["extra"]

    def test_default_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        nodes = dict((path, node_params) for path, _, node_params in op.walk())
        assert torch.equal(nodes["/"], op.build_params(include_fixed=False, trans=False))

    def test_dict_input_is_collapsed_to_one_tensor(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        nodes = dict((path, node_params) for path, _, node_params in op.walk({
            "a/sigma^2": torch.tensor([0.0]),
            "b/sigma^2": torch.tensor([0.5]),
        }))
        assert torch.equal(nodes["/"], torch.tensor([0.0, 0.5]))
        assert torch.equal(nodes["a"], torch.tensor([0.0]))

    def test_wrong_length_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError):
            op.walk(torch.tensor([0.0, 0.5, 1.0]))

    def test_non_tensor_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(TypeError):
            op.walk([0.0, 0.5])

    def test_does_not_evaluate_operands(self):
        stub = StubMatrix()
        op = Sum(a=stub, b=DiagonalMatrix(2))
        op.walk(torch.tensor([0.0, 0.5]))
        assert stub.call_count == 0

    def test_returns_a_list(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert isinstance(op.walk(torch.tensor([0.0, 0.5])), list)


class TestOperatorFreeParamTree:
    """Tests for free_param_tree."""

    def test_keys_and_order_match_tree_and_grad_tree(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        results = op.tree(free_params)
        grads = op.grad_tree(free_params)
        free_params_by_path = op.free_param_tree(free_params)
        assert list(free_params_by_path) == list(results) == list(grads)

    def test_root_entry_is_the_input_tensor(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        free_params_by_path = op.free_param_tree(free_params)
        assert free_params_by_path["/"] is free_params

    def test_params_split_per_operand(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        free_params_by_path = op.free_param_tree(free_params)
        assert torch.equal(free_params_by_path["/"], free_params)
        assert torch.equal(free_params_by_path["inner"], torch.tensor([0.0, 0.5, 1.0]))
        assert torch.equal(free_params_by_path["inner/op_0"], torch.tensor([0.0, 0.5]))
        assert torch.equal(free_params_by_path["extra"], torch.tensor([0.5]))

    def test_slices_are_views_of_the_input(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        free_params_by_path = op.free_param_tree(free_params)
        free_params[0] = 9.0
        assert torch.equal(free_params_by_path["inner/op_0"], torch.tensor([9.0, 0.5]))

    def test_nested_tensor_operand_is_none(self):
        inner = Sum(x=ScalarMatrix(2), t=torch.eye(2))
        op = Sum(inner=inner, b=ScalarMatrix(2))
        free_params_by_path = op.free_param_tree(torch.tensor([0.0, 0.5]))
        assert sorted(free_params_by_path) == ["/", "b", "inner", "inner/t", "inner/x"]
        assert free_params_by_path["inner/t"] is None

    def test_tensor_operand_leaf_is_none(self):
        fixed = torch.ones(3, 3)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        free_params_by_path = op.free_param_tree(torch.tensor([0.0]))
        assert free_params_by_path["fixed"] is None

    def test_zero_free_params_root(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        free_params_by_path = op.free_param_tree()
        assert sorted(free_params_by_path) == ["/", "a", "b"]
        assert free_params_by_path["/"].numel() == 0
        assert free_params_by_path["a"].numel() == 0

    def test_zero_free_params_root_explicit_empty_tensor(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        free_params_by_path = op.free_param_tree(torch.tensor([]))
        assert free_params_by_path["/"].numel() == 0

    def test_fixed_matrix_operand_leaf(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        free_params_by_path = op.free_param_tree(torch.tensor([0.5]))
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
        free_params_by_path = op.free_param_tree(torch.tensor([0.0, 0.5]))
        assert sorted(free_params_by_path) == ["/", "a", "b"]
        assert torch.equal(free_params_by_path["a"], torch.tensor([0.0]))

    def test_default_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params_by_path = op.free_param_tree()
        assert torch.equal(free_params_by_path["/"], op.build_params(include_fixed=False, trans=False))

    def test_dict_input(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params_by_path = op.free_param_tree({
            "a/sigma^2": torch.tensor([0.0]),
            "b/sigma^2": torch.tensor([0.5]),
        })
        assert torch.equal(free_params_by_path["/"], torch.tensor([0.0, 0.5]))
        assert torch.equal(free_params_by_path["a"], torch.tensor([0.0]))

    def test_wrong_length_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError):
            op.free_param_tree(torch.tensor([0.0, 0.5, 1.0]))

    def test_non_tensor_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(TypeError):
            op.free_param_tree([0.0, 0.5])

    def test_dtype_is_preserved(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params_by_path = op.free_param_tree(torch.tensor([0.0, 0.5, 1.0, 0.5], dtype=torch.float64))
        assert free_params_by_path["/"].dtype == torch.float64
        assert free_params_by_path["inner/op_0"].dtype == torch.float64

    def test_does_not_evaluate_operands(self):
        stub = StubMatrix()
        op = Sum(a=stub, b=DiagonalMatrix(2))
        op.free_param_tree(torch.tensor([0.0, 0.5]))
        assert stub.call_count == 0
        op.tree(torch.tensor([0.0, 0.5]))
        # Entered once while Sum composes its result and once for its own entry.
        assert stub.call_count == 2


class TestOperatorGradTree:
    """Tests for grad_tree."""

    def test_root_grad_matches_grad(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grads = op.grad_tree(free_params)
        grad, grad_names = op.grad(free_params)
        assert torch.equal(grads["/"][0], grad)
        assert grads["/"][1] == grad_names

    def test_grad_and_names_pair_per_path(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads = op.grad_tree(torch.tensor([0.0, 0.5]))
        assert sorted(grads) == ["/", "a", "b"]
        for grad, grad_names in grads.values():
            assert grad.shape[0] == len(grad_names)

    def test_keys_match_tree(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        grads = op.grad_tree(free_params)
        results = op.tree(free_params)
        assert sorted(grads) == sorted(results)

    def test_leaf_grad_matches_operand_grad(self):
        op = Sum(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 1.5])
        grads = op.grad_tree(free_params)
        assert torch.equal(grads["a"][0], op.operands["a"].grad(free_params[:1])[0])
        assert torch.equal(grads["b"][0], op.operands["b"].grad(free_params[1:])[0])

    def test_nested_grad_matches_operand_grad(self):
        inner = BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2))
        op = Sum(inner=inner, extra=ScalarMatrix(4))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.5])
        grads = op.grad_tree(free_params)
        assert torch.equal(grads["inner"][0], inner.grad(free_params[:3])[0])
        assert grads["inner"][1] == inner.free_param_names

    def test_node_grad_shape_is_its_own(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        grads = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        assert grads["/"][0].shape == (4, 4, 4)
        assert grads["inner"][0].shape == (3, 4, 4)
        assert grads["inner/op_0"][0].shape == (2, 2, 2)
        assert grads["inner/op_1"][0].shape == (1, 2, 2)
        assert grads["extra"][0].shape == (1, 4, 4)

    def test_names_join_to_free_param_names(self):
        op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), extra=ScalarMatrix(4))
        grads = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))
        joined = [f"{path}/{name}" for path in ["inner/op_0", "inner/op_1", "extra"] for name in grads[path][1]]
        assert joined == op.free_param_names

    def test_deeply_nested_path(self):
        deep = Sum(x=BlockDiagonal(BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)), ScalarMatrix(2)), extra=ScalarMatrix(6))
        grads = deep.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5, 0.5]))
        assert sorted(grads) == ["/", "extra", "x", "x/op_0", "x/op_0/op_0", "x/op_0/op_1", "x/op_1"]
        assert grads["x/op_0/op_0"][0].shape == (2, 2, 2)
        assert grads["x/op_0"][0].shape == (3, 4, 4)

    def test_tensor_operand_has_no_grad(self):
        op = Sum(a=ScalarMatrix(2), fixed=torch.eye(2))
        grads = op.grad_tree(torch.tensor([0.5]))
        assert grads["fixed"] == (None, [])

    def test_fixed_matrix_operand_has_no_grad(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        grads = op.grad_tree(torch.tensor([0.5]))
        assert grads["b"] == (None, [])

    def test_zero_free_params_root(self):
        op = Sum(a=IdentityMatrix(2), b=IdentityMatrix(2))
        grads = op.grad_tree()
        assert sorted(grads) == ["/", "a", "b"]
        assert grads["/"] == (None, [])

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
        grads = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a", "b"]
        assert torch.equal(grads["a"][0], adapter.grad(free_params[:1])[0])

    def test_default_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads = op.grad_tree()
        assert torch.equal(grads["/"][0], op.grad()[0])

    def test_dict_input(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads = op.grad_tree({
            "a/sigma^2": torch.tensor([0.0]),
            "b/sigma^2": torch.tensor([0.5]),
        })
        assert torch.equal(grads["/"][0], op.grad(torch.tensor([0.0, 0.5]))[0])

    def test_wrong_length_raises(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(ValueError):
            op.grad_tree(torch.tensor([0.0, 0.5, 1.0]))

    def test_auto_grad_mode(self):
        op = Sum(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 1.5])
        manual = op.grad_tree(free_params)
        op.grad_mode = "auto"
        auto = op.grad_tree(free_params)
        assert torch.allclose(manual["/"][0], auto["/"][0])
        assert auto["/"][1] == op.free_param_names

    def test_minimal_operator_vehicle(self):
        op = MinimalOperator(a=ScalarMatrix(3), b=ScalarMatrix(3))
        grads = op.grad_tree(torch.tensor([0.0, 0.5]))
        assert sorted(grads) == ["/", "a", "b"]


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


class TestOperatorOperandsInverse:
    """Tests for operands_inverse."""

    def test_returns_per_operand_inverses(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        inverses = op.operands_inverse(torch.tensor([0.0, 0.5]))
        assert len(inverses) == 2
        assert torch.allclose(inverses[0], torch.linalg.inv(op.operands["a"](torch.tensor([0.0]))))
        assert torch.allclose(inverses[1], torch.linalg.inv(op.operands["b"](torch.tensor([0.5]))))

    def test_uses_operand_own_inverse(self):
        stub = StubMatrix()
        op = Sum(a=stub, b=ScalarMatrix(2))
        inverses = op.operands_inverse(torch.tensor([0.5]))
        assert stub.inverse_count == 1
        assert torch.allclose(inverses[0], torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]))

    def test_does_not_build_matrix_operands(self):
        stub = StubMatrix()
        op = Sum(a=stub, b=ScalarMatrix(2))
        op.operands_inverse(torch.tensor([0.5]))
        assert stub.call_count == 0

    def test_tensor_operand_inverted_directly(self):
        op = Sum(a=ScalarMatrix(2), b=torch.eye(2) * 4.0)
        inverses = op.operands_inverse(torch.tensor([0.5]))
        assert torch.allclose(inverses[1], torch.eye(2) / 4.0)

    def test_tensor_operand_follows_input_dtype(self):
        op = Sum(a=ScalarMatrix(2), b=torch.eye(2) * 4.0)
        inverses = op.operands_inverse(torch.tensor([0.5], dtype=torch.float64))
        assert inverses[1].dtype == torch.float64
        assert inverses[0].dtype == torch.float64

    def test_uses_default_free_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert torch.allclose(op.operands_inverse()[1], op.operands_inverse(op.free_param_defaults)[1])

    def test_accepts_dict_of_free_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        as_dict = op.operands_inverse(op._to_free_param_dict(free_params))
        assert torch.allclose(as_dict[0], op.operands_inverse(free_params)[0])

    def test_validation_is_not_bypassed(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.operands_inverse([0.0, 0.5])
        with pytest.raises(ValueError, match="length 2"):
            op.operands_inverse(torch.tensor([0.0]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.operands_inverse({})


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


class TestOperatorCacheControl:
    """Tests for enable_cache and disable_cache across an operand tree."""

    def test_disable_cache_propagates_to_operands(self):
        a, b = AR1Matrix(3), AR1Matrix(3)
        op = MinimalOperator(a=a, b=b)

        op.disable_cache()

        assert op.cache is False
        assert a.cache is False
        assert b.cache is False

    def test_disable_cache_clears_operand_caches(self):
        a = AR1Matrix(3)
        op = MinimalOperator(a=a, b=ScalarMatrix(3))
        raw = torch.tensor([0.5, 1.0, 0.0])
        a_params = a.build_params(raw[:2])

        op(raw)
        assert a.get_intermediates(a_params) is not None

        op.disable_cache()

        assert a.get_intermediates(a_params) is None

    def test_enable_cache_propagates_to_operands(self):
        a = AR1Matrix(3)
        op = MinimalOperator(a=a, b=ScalarMatrix(3))
        op.disable_cache()

        op.enable_cache()

        assert op.cache is True
        assert a.cache is True

    def test_disable_cache_reaches_nested_operators(self):
        inner = Sum(a=AR1Matrix(3), b=ScalarMatrix(3))
        outer = MinimalOperator(inner=inner, c=ScalarMatrix(3))

        outer.disable_cache()

        assert inner.cache is False
        assert inner.operands["a"].cache is False

    def test_tensor_operands_are_skipped(self):
        op = MinimalOperator(a=AR1Matrix(3), fixed=torch.eye(3))

        op.disable_cache()
        assert op.cache is False

        op.enable_cache()
        assert op.cache is True

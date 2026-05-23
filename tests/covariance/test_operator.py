import torch
import pytest
from torch_openreml.covariance import Sum, ScalarMatrix, IdentityMatrix
from torch_openreml.covariance.operator import Operator
from torch_openreml.covariance.matrix import Matrix


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

import torch
import pytest
from torch_openreml.covariance import ScalarMatrix, IdentityMatrix, DiagonalMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformExp, TransformIdentity


class MinimalMatrix(Matrix):
    """Minimal concrete Matrix subclass for testing the ABC directly."""

    def __call__(self, free_params):
        params = self.build_params(free_params)
        return torch.eye(2, dtype=params.dtype, device=params.device)

    def manual_grad(self, free_params):
        if len(free_params) == 0:
            return None, []
        trans_grad = self.trans_grad(free_params)
        grad = torch.zeros(self.num_free_params, *self.shape, dtype=free_params.dtype, device=free_params.device)
        for i in range(self.num_free_params):
            grad[i, i, i] = trans_grad[i]
        return grad, self.free_param_names


class TestMatrixConstructor:
    """Tests for Matrix.__init__ validation."""

    def test_grad_mode_default(self):
        mat = ScalarMatrix(3)
        assert mat.grad_mode == "default"

    def test_check_shape_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            MinimalMatrix(42, param_specs={
                "p": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformIdentity()}
            })

    def test_check_shape_rejects_non_positive(self):
        with pytest.raises(ValueError):
            MinimalMatrix((-1, 2), param_specs={
                "p": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformIdentity()}
            })

    def test_check_shape_accepts_none(self):
        class NoneShapeMatrix(Matrix):
            def __call__(self, free_params):
                return torch.eye(2)

        mat = NoneShapeMatrix(None, {})
        assert mat.shape == ()

    def test_check_param_specs_rejects_non_dict(self):
        with pytest.raises(TypeError, match="param_sepc"):
            MinimalMatrix((2, 2), "not_a_dict")

    def test_check_param_specs_rejects_bad_keys(self):
        with pytest.raises(TypeError):
            MinimalMatrix((2, 2), {
                "p": {"wrong_key": True, "default": torch.tensor([0.0]), "trans": TransformIdentity()}
            })

    def test_check_param_specs_rejects_non_bool_fixed(self):
        with pytest.raises(TypeError):
            MinimalMatrix((2, 2), {
                "p": {"fixed": 1, "default": torch.tensor([0.0]), "trans": TransformIdentity()}
            })

    def test_check_param_specs_rejects_non_tensor_default(self):
        with pytest.raises(TypeError):
            MinimalMatrix((2, 2), {
                "p": {"fixed": False, "default": [0.0], "trans": TransformIdentity()}
            })

    def test_check_param_specs_rejects_non_1d_default(self):
        with pytest.raises(TypeError):
            MinimalMatrix((2, 2), {
                "p": {"fixed": False, "default": torch.tensor([[0.0]]), "trans": TransformIdentity()}
            })

    def test_check_param_specs_rejects_non_transform(self):
        with pytest.raises(TypeError):
            MinimalMatrix((2, 2), {
                "p": {"fixed": False, "default": torch.tensor([0.0]), "trans": "not_a_transform"}
            })


class TestMatrixProperties:
    """Tests for Matrix properties."""

    def test_shape(self):
        mat = ScalarMatrix(3)
        assert mat.shape == (3, 3)
        assert isinstance(mat.shape, tuple)

    def test_param_names(self):
        mat = ScalarMatrix(3)
        assert mat.param_names == ["sigma^2"]

    def test_free_param_names(self):
        mat = ScalarMatrix(3)
        assert mat.free_param_names == ["sigma^2"]

    def test_fixed_param_names(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_names == []

    def test_free_param_index(self):
        mat = ScalarMatrix(3)
        assert mat.free_param_index == [0]

    def test_fixed_param_index(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_index == []

    def test_num_params(self):
        mat = DiagonalMatrix(3)
        assert mat.num_params == 3

    def test_num_free_params(self):
        mat = DiagonalMatrix(3)
        assert mat.num_free_params == 3

    def test_num_fixed_params(self):
        mat = IdentityMatrix(3)
        assert mat.num_fixed_params == 0

    def test_param_defaults(self):
        mat = ScalarMatrix(3)
        defaults = mat.param_defaults
        assert "sigma^2" in defaults
        assert torch.equal(defaults["sigma^2"], torch.tensor([0.0]))

    def test_free_param_defaults(self):
        mat = ScalarMatrix(3)
        assert "sigma^2" in mat.free_param_defaults

    def test_fixed_param_defaults_empty(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_defaults == {}

    def test_param_trans(self):
        mat = ScalarMatrix(3)
        trans = mat.param_trans
        assert "sigma^2" in trans
        assert isinstance(trans["sigma^2"], TransformExpPow2)

    def test_free_param_trans(self):
        mat = ScalarMatrix(3)
        assert "sigma^2" in mat.free_param_trans

    def test_fixed_param_trans_empty(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_trans == {}

    def test_repr_dict(self):
        mat = ScalarMatrix(3)
        rd = mat.repr_dict
        assert rd["shape"] == (3, 3)
        assert "sigma^2" in rd["param_specs"]


class TestBuildParams:
    """Tests for Matrix.build_params."""

    def test_tensor_format(self):
        mat = ScalarMatrix(3)
        result = mat.build_params(torch.tensor([0.0]))
        assert result.ndim == 1
        assert result.numel() == 1

    def test_dict_format(self):
        mat = ScalarMatrix(3)
        result = mat.build_params(torch.tensor([0.0]), out_format="dict")
        assert isinstance(result, dict)
        assert "sigma^2" in result

    def test_include_fixed_false(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        result = mat.build_params(torch.tensor([]), include_fixed=False)
        assert result.numel() == 0

    def test_trans_false(self):
        mat = ScalarMatrix(3)
        result = mat.build_params(torch.tensor([0.5]), trans=False)
        assert torch.allclose(result, torch.tensor([0.5]))

    def test_empty_params(self):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]))
        assert torch.equal(result, torch.tensor([]))

    def test_empty_params_dict_format(self):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]), out_format="dict")
        assert result == {}

    def test_raises_on_bad_out_format(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="out_format"):
            mat.build_params(torch.tensor([0.0]), out_format="invalid")

    def test_dict_input(self):
        mat = ScalarMatrix(3)
        result = mat.build_params({"sigma^2": torch.tensor([0.0])})
        assert result.numel() == 1

    def test_dict_input_missing_key(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Missing"):
            mat.build_params({"wrong_key": torch.tensor([0.0])})

    def test_dict_input_extra_key(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Unexpected"):
            mat.build_params({"sigma^2": torch.tensor([0.0]), "extra": torch.tensor([1.0])})

    def test_wrong_length_raises(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError):
            mat.build_params(torch.tensor([0.0, 1.0]))


class TestTransGrad:
    """Tests for Matrix.trans_grad."""

    def test_trans_grad_length(self):
        mat = ScalarMatrix(3)
        tg = mat.trans_grad(torch.tensor([0.0]))
        assert tg.numel() == 1

    def test_trans_grad_value(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([1.0])
        tg = mat.trans_grad(free_params)
        expected = 2.0 * torch.exp(torch.tensor(2.0))
        assert torch.allclose(tg, expected)

    def test_trans_grad_dict_input(self):
        mat = ScalarMatrix(3)
        tg = mat.trans_grad({"sigma^2": torch.tensor([0.0])})
        assert tg.numel() == 1

    def test_trans_grad_multi_param(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.1, 0.2])
        tg = mat.trans_grad(free_params)
        assert tg.shape == (3,)


class TestAutoGrad:
    """Tests for Matrix.auto_grad."""

    def test_auto_grad_shape(self):
        mat = ScalarMatrix(3)
        grad, grad_names = mat.auto_grad(torch.tensor([0.5]))
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["sigma^2"]

    def test_auto_grad_empty_params(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.auto_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_auto_grad_dict_input(self):
        mat = ScalarMatrix(3)
        grad, grad_names = mat.auto_grad({"sigma^2": torch.tensor([0.5])})
        assert grad.shape == (1, 3, 3)


class TestGradDispatch:
    """Tests for Matrix.grad dispatch logic."""

    def test_default_mode_uses_manual(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        grad_default, _ = mat.grad(free_params)
        grad_manual, _ = mat.manual_grad(free_params)
        assert torch.allclose(grad_default, grad_manual)

    def test_auto_mode(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "auto"
        free_params = torch.tensor([0.5])
        grad, _ = mat.grad(free_params)
        auto, _ = mat.auto_grad(free_params)
        assert torch.allclose(grad, auto)

    def test_unknown_grad_mode_raises(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "invalid"
        with pytest.raises(RuntimeError, match="grad mode"):
            mat.grad(torch.tensor([0.5]))

    def test_default_falls_back_to_auto(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []


class TestIntermediateCaching:
    """Tests for set/get/reset_intermediates."""

    def test_set_and_get(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        mat.set_intermediates(params, {"key": "value"})
        assert mat.get_intermediates(params) == {"key": "value"}

    def test_cache_stale_on_different_params(self):
        mat = ScalarMatrix(3)
        params1 = mat.build_params(torch.tensor([0.5]))
        params2 = mat.build_params(torch.tensor([1.0]))
        mat.set_intermediates(params1, {"key": "value1"})
        assert mat.get_intermediates(params2) is None

    def test_cache_stale_after_reset(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        mat.set_intermediates(params, {"key": "value"})
        mat.reset_intermediates()
        assert mat.get_intermediates(params) is None

    def test_empty_params_no_op_set(self):
        mat = IdentityMatrix(3)
        result = mat.set_intermediates(torch.tensor([]), {"key": "value"})
        assert result is None

    def test_empty_params_no_op_get(self):
        mat = IdentityMatrix(3)
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_raises_on_non_tensor(self):
        mat = ScalarMatrix(3)
        with pytest.raises(TypeError):
            mat.set_intermediates([1.0, 2.0], {"key": "value"})

    def test_raises_on_non_1d(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError):
            mat.set_intermediates(torch.tensor([[1.0]]), {"key": "value"})

    def test_reset_on_init(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        assert mat.get_intermediates(params) is None


class TestREMLInterface:
    """Tests for map_theta_to_v and map_theta_to_dv."""

    def test_map_theta_to_v(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)


class TestRepr:
    """Tests for Matrix repr."""

    def test_repr_includes_class_name(self):
        mat = ScalarMatrix(3)
        assert "ScalarMatrix" in repr(mat)

    def test_repr_includes_shape(self):
        mat = ScalarMatrix(3)
        assert "(3, 3)" in repr(mat)

    def test_repr_single_line(self):
        mat = ScalarMatrix(3)
        r = repr(mat)
        assert "\n" not in r


class TestDictFreeParams:
    """Tests for _from_free_param_dict and _to_free_param_dict."""

    def test_from_free_param_dict_passthrough_tensor(self):
        mat = ScalarMatrix(3)
        result = mat._from_free_param_dict(torch.tensor([0.0]))
        assert torch.equal(result, torch.tensor([0.0]))

    def test_from_free_param_dict_from_dict(self):
        mat = ScalarMatrix(3)
        result = mat._from_free_param_dict({"sigma^2": torch.tensor([0.0])})
        assert torch.equal(result, torch.tensor([0.0]))

    def test_from_free_param_dict_missing_key(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Missing"):
            mat._from_free_param_dict({"wrong": torch.tensor([0.0])})

    def test_from_free_param_dict_extra_key(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Unexpected"):
            mat._from_free_param_dict({"sigma^2": torch.tensor([0.0]), "extra": torch.tensor([1.0])})

    def test_to_free_param_dict_passthrough_dict(self):
        mat = ScalarMatrix(3)
        d = {"sigma^2": torch.tensor([0.5])}
        result = mat._to_free_param_dict(d)
        assert result == d

    def test_to_free_param_dict_from_tensor(self):
        mat = ScalarMatrix(3)
        result = mat._to_free_param_dict(torch.tensor([0.5]))
        assert "sigma^2" in result
        assert torch.equal(result["sigma^2"], torch.tensor([0.5]))

import torch
import pytest
from torch_openreml.covariance import Adapter, DiagonalMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExp, TransformIdentity


def make_param_specs(fixed=False, trans=None):
    """A single free parameter 'logit', optionally fixed or with another transform."""
    return {
        "logit": {
            "fixed": fixed,
            "default": torch.tensor([0.0]),
            "trans": TransformIdentity() if trans is None else trans,
        }
    }


def make_adapter(param_specs=None, seen=None):
    """
    An Adapter wrapping DiagonalMatrix(2) so that both variances share one
    parameter: v1 = sigmoid(theta) and v2 = 1 - v1.
    """
    def param_map(params):
        if seen is not None:
            seen.append(params)
        p = torch.sigmoid(params[0])
        return torch.stack([p, 1 - p])

    if param_specs is None:
        param_specs = make_param_specs()
    return Adapter(DiagonalMatrix(2), param_specs, param_map)


class TestAdapter:
    """Tests for the reparameterising adapter matrix."""

    def test_constructor(self):
        assert isinstance(make_adapter(), Matrix)

    def test_shape(self):
        assert make_adapter().shape == (2, 2)

    def test_num_free_params(self):
        assert make_adapter().num_free_params == 1

    @pytest.mark.parametrize("theta", [0.0, 1.0, -1.0])
    def test_call_returns_reparameterised_matrix(self, theta):
        """The mapped parameters are handed to the adaptee, which builds the matrix."""
        mat = make_adapter()
        p = torch.sigmoid(torch.tensor(theta))
        expected = DiagonalMatrix(2)(torch.stack([p, 1 - p]))
        assert torch.allclose(mat(torch.tensor([theta])), expected)

    def test_call_depends_on_free_param(self):
        """The single free parameter drives the adaptee's two parameters."""
        mat = make_adapter()
        assert not torch.allclose(mat(torch.tensor([0.0])), mat(torch.tensor([1.0])))

    def test_call_default_params(self):
        mat = make_adapter()
        assert torch.allclose(mat(), mat(torch.tensor([0.0])))

    def test_call_dict_input(self):
        mat = make_adapter()
        assert torch.allclose(mat({"logit": torch.tensor([0.0])}), mat(torch.tensor([0.0])))

    def test_param_specs_returns_stored_specs(self):
        """param_specs is the stored dict, not a rebuild: mutations take effect."""
        mat = make_adapter()
        before = mat.param_specs["logit"]["trans"]
        assert isinstance(before, TransformIdentity)
        mat.param_specs["logit"]["fixed"] = True
        assert mat.num_free_params == 0
        assert mat.free_param_names == []

    def test_param_map_receives_free_params_only(self):
        """Fixed parameters are not passed to param_map."""
        seen = []
        param_specs = make_param_specs()
        param_specs["offset"] = {
            "fixed": True,
            "default": torch.tensor([0.5]),
            "trans": TransformIdentity(),
        }
        mat = make_adapter(param_specs, seen=seen)
        assert mat.num_free_params == 1
        mat(torch.tensor([0.0]))
        assert len(seen) == 1
        assert torch.equal(seen[0], torch.tensor([0.0]))

    def test_param_map_receives_untransformed_params(self):
        """The adapter never applies transforms: param_map sees the raw free params."""
        seen = []
        make_adapter(seen=seen)(torch.tensor([0.7]))
        assert torch.equal(seen[0], torch.tensor([0.7]))

    def test_constructor_rejects_non_identity_transform(self):
        with pytest.raises(ValueError, match="TransformIdentity"):
            make_adapter(make_param_specs(trans=TransformExp()))

    def test_manual_grad_shape_matches_names(self):
        """grad rows and grad_names both follow num_free_params."""
        mat = make_adapter()
        grad, grad_names = mat.manual_grad(torch.tensor([0.0]))
        assert grad.shape == (mat.num_free_params, *mat.shape)
        assert grad_names == mat.free_param_names
        assert grad.shape[0] == len(grad_names)

    def test_manual_grad_shape_matches_names_with_fixed_param(self):
        param_specs = make_param_specs()
        param_specs["offset"] = {
            "fixed": True,
            "default": torch.tensor([0.5]),
            "trans": TransformIdentity(),
        }
        mat = make_adapter(param_specs)
        grad, grad_names = mat.manual_grad(torch.tensor([0.0]))
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["logit"]

    def test_manual_grad_matches_auto_grad(self):
        mat = make_adapter()
        free_params = torch.tensor([0.3])
        manual, manual_names = mat.manual_grad(free_params)
        auto, auto_names = mat.auto_grad(free_params)
        assert manual_names == auto_names
        assert torch.allclose(manual, auto, atol=1e-6)

    def test_manual_grad_matches_auto_grad_with_fixed_param(self):
        param_specs = make_param_specs()
        param_specs["offset"] = {
            "fixed": True,
            "default": torch.tensor([0.5]),
            "trans": TransformIdentity(),
        }
        mat = make_adapter(param_specs)
        free_params = torch.tensor([0.3])
        manual, manual_names = mat.manual_grad(free_params)
        auto, auto_names = mat.auto_grad(free_params)
        assert manual.shape == auto.shape
        assert manual_names == auto_names
        assert torch.allclose(manual, auto, atol=1e-6)

    def test_grad_uses_manual(self):
        mat = make_adapter()
        free_params = torch.tensor([0.3])
        grad, grad_names = mat.grad(free_params)
        manual, manual_names = mat.manual_grad(free_params)
        assert grad_names == manual_names
        assert torch.allclose(grad, manual)

    def test_all_fixed_returns_none(self):
        mat = make_adapter(make_param_specs(fixed=True))
        grad, grad_names = mat.manual_grad()
        assert grad is None
        assert grad_names == []

    def test_repr(self):
        assert "Adapter" in repr(make_adapter())

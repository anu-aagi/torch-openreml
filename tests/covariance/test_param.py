import pytest
import torch

from torch_openreml.covariance import SimpleMatrix
from torch_openreml.covariance.param import simple_param_specs
from torch_openreml.covariance.transform import TransformExpPow2, TransformIdentity


class TestSimpleParamSpecsNames:
    """The generated keys and their order."""

    def test_zero_parameters_gives_empty_dict(self):
        assert simple_param_specs(0) == {}

    def test_single_parameter_name(self):
        assert list(simple_param_specs(1)) == ["theta_0"]

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_names_are_sequential(self, n):
        assert list(simple_param_specs(n)) == [f"theta_{i}" for i in range(n)]

    def test_insertion_order_matches_index_order(self):
        specs = simple_param_specs(4)
        assert list(specs) == ["theta_0", "theta_1", "theta_2", "theta_3"]

    def test_keys_are_strings(self):
        assert all(isinstance(key, str) for key in simple_param_specs(4))


class TestSimpleParamSpecsContents:
    """Every entry is a free specification with a shared transform."""

    @pytest.mark.parametrize("n", [1, 3])
    def test_every_parameter_is_free(self, n):
        assert all(spec["fixed"] is False for spec in simple_param_specs(n).values())

    def test_specification_has_exactly_the_three_fields(self):
        assert set(simple_param_specs(1)["theta_0"]) == {"fixed", "default", "trans"}

    def test_default_is_zero_by_default(self):
        specs = simple_param_specs(2)
        for spec in specs.values():
            assert torch.equal(spec["default"], torch.tensor([0.0]))

    def test_default_is_one_dimensional(self):
        for spec in simple_param_specs(3).values():
            assert spec["default"].shape == (1,)

    def test_custom_default_value(self):
        specs = simple_param_specs(2, default=torch.tensor([1.5]))
        for spec in specs.values():
            assert torch.equal(spec["default"], torch.tensor([1.5]))

    def test_custom_default_dtype_is_preserved(self):
        specs = simple_param_specs(2, default=torch.tensor([1.5], dtype=torch.float64))
        for spec in specs.values():
            assert spec["default"].dtype == torch.float64

    def test_custom_default_requires_grad_is_detached(self):
        default = torch.tensor([1.0], requires_grad=True)
        specs = simple_param_specs(1, default=default)
        assert specs["theta_0"]["default"].requires_grad is False

    def test_defaults_are_independent_tensors(self):
        specs = simple_param_specs(3)
        assert specs["theta_0"]["default"] is not specs["theta_1"]["default"]
        assert specs["theta_1"]["default"] is not specs["theta_2"]["default"]

    def test_mutating_one_default_does_not_change_another(self):
        specs = simple_param_specs(3)
        specs["theta_0"]["default"][0] = 99.0
        assert specs["theta_1"]["default"].item() == 0.0
        assert specs["theta_2"]["default"].item() == 0.0

    def test_mutating_the_source_tensor_does_not_change_specs(self):
        default = torch.tensor([2.0])
        specs = simple_param_specs(1, default=default)
        default[0] = 99.0
        assert specs["theta_0"]["default"].item() == 2.0

    def test_default_transform_is_identity(self):
        specs = simple_param_specs(2)
        for spec in specs.values():
            assert isinstance(spec["trans"], TransformIdentity)

    def test_transform_is_shared_across_parameters(self):
        specs = simple_param_specs(3)
        assert specs["theta_0"]["trans"] is specs["theta_1"]["trans"]
        assert specs["theta_1"]["trans"] is specs["theta_2"]["trans"]

    def test_custom_transform_is_used(self):
        trans = TransformExpPow2()
        specs = simple_param_specs(2, trans=trans)
        assert all(spec["trans"] is trans for spec in specs.values())

    def test_custom_transform_is_not_replaced_by_identity(self):
        specs = simple_param_specs(1, trans=TransformExpPow2())
        assert isinstance(specs["theta_0"]["trans"], TransformExpPow2)


class TestSimpleParamSpecsValidation:
    """Bad ``n`` and ``default`` arguments."""

    @pytest.mark.parametrize("default", [1.0, [1.0], "default", None])
    def test_rejects_non_tensor_default(self, default):
        if default is None:
            # None means "use the default default", not an error.
            assert simple_param_specs(1)["theta_0"]["default"].item() == 0.0
            return
        with pytest.raises(TypeError, match="1D torch tensor"):
            simple_param_specs(1, default=default)

    @pytest.mark.parametrize(
        "default",
        [torch.tensor(1.0), torch.zeros(2), torch.zeros(1, 1), torch.zeros(2, 1)],
    )
    def test_rejects_default_with_wrong_shape(self, default):
        with pytest.raises(ValueError, match=r"shape \(1\)"):
            simple_param_specs(1, default=default)

    def test_rejects_non_integer_n(self):
        with pytest.raises(TypeError):
            simple_param_specs(2.0)

    def test_rejects_string_n(self):
        with pytest.raises(TypeError):
            simple_param_specs("2")

    def test_negative_n_gives_no_parameters(self):
        assert simple_param_specs(-2) == {}

    def test_none_transform_falls_back_to_identity(self):
        specs = simple_param_specs(1, trans=None)
        assert isinstance(specs["theta_0"]["trans"], TransformIdentity)

    def test_accepts_keyword_arguments(self):
        specs = simple_param_specs(n=1, default=torch.tensor([3.0]), trans=None)
        assert specs["theta_0"]["default"].item() == 3.0


class TestSimpleParamSpecsIntegration:
    """The specs are directly usable by a Matrix-backed callable."""

    def test_simple_matrix_uses_the_generated_names(self):
        mat = SimpleMatrix(2, call=lambda p: torch.diag(p))
        assert mat.param_names == ["theta_0", "theta_1"]

    def test_simple_matrix_default_values(self):
        mat = SimpleMatrix(2, call=lambda p: torch.diag(p), default=torch.tensor([1.5]))
        assert torch.allclose(mat(), torch.eye(2) * 1.5)

    def test_simple_matrix_custom_parameters(self):
        mat = SimpleMatrix(2, call=lambda p: torch.diag(p))
        result = mat(torch.tensor([2.0, 3.0]))
        assert torch.equal(result, torch.diag(torch.tensor([2.0, 3.0])))

    def test_simple_matrix_dict_input(self):
        mat = SimpleMatrix(2, call=lambda p: torch.diag(p))
        result = mat({"theta_0": torch.tensor([2.0]), "theta_1": torch.tensor([3.0])})
        assert torch.equal(result, torch.diag(torch.tensor([2.0, 3.0])))

    def test_simple_matrix_gradient_names_are_the_generated_names(self):
        mat = SimpleMatrix(2, call=lambda p: torch.diag(p))
        _, names = mat.grad(torch.tensor([2.0, 3.0]))
        assert names == ["theta_0", "theta_1"]

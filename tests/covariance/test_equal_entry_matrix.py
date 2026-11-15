import pandas as pd
import pytest
import torch

from torch_openreml.covariance import EqualEntryMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExp,
    TransformExpPow2,
    TransformIdentity,
)

ENTRY_POINTS = ["__call__", "grad", "manual_grad", "auto_grad", "trans_grad", "build_params"]


def single_spec(name="sigma^2", *, fixed=False, default=0.0, trans=None):
    """A one-parameter specification dict for the class's single variance."""
    return {
        name: {
            "fixed": fixed,
            "default": torch.tensor([float(default)]),
            "trans": trans if trans is not None else TransformExpPow2(),
        }
    }


def two_specs():
    """Two free parameters: outside EqualEntryMatrix's single-variance model."""
    return {
        "a": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
        "b": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
    }


def mixed_specs():
    """One free and one fixed parameter: also a multi-parameter specification."""
    return {
        "a": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
        "b": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExpPow2()},
    }


class TestEqualEntryMatrixConstructor:
    """Construction, shape, default parameter surface, and input validation."""

    def test_is_matrix_subclass(self):
        mat = EqualEntryMatrix(3)
        assert isinstance(mat, Matrix)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_shape(self, n):
        mat = EqualEntryMatrix(n)
        assert mat.shape == (n, n)
        assert isinstance(mat.shape, tuple)

    def test_shape_rectangular(self):
        assert EqualEntryMatrix(3, 2).shape == (3, 2)
        assert EqualEntryMatrix(2, 5).shape == (2, 5)

    def test_m_omitted_defaults_to_n(self):
        mat = EqualEntryMatrix(3, m=None)
        assert mat.shape == (3, 3)

    def test_shape_from_levels(self):
        assert EqualEntryMatrix(["a", "b", "a", "c"]).shape == (3, 3)
        assert EqualEntryMatrix(("a", "b", "a", "c")).shape == (3, 3)
        assert EqualEntryMatrix(pd.Series(["a", "b", "a", "c"])).shape == (3, 3)

    def test_shape_from_levels_is_distinct_count(self):
        assert EqualEntryMatrix(["a", "b", "a"]).shape == (2, 2)

    def test_rectangular_shape_from_levels(self):
        mat = EqualEntryMatrix(["a", "b", "a"], ["x", "y", "z", "x"])
        assert mat.shape == (2, 3)

    def test_levels_equivalent_to_distinct_count(self):
        from_levels = EqualEntryMatrix(["a", "b", "a", "c"])
        from_int = EqualEntryMatrix(3)
        free_params = torch.tensor([0.5])
        assert from_levels.shape == from_int.shape
        assert from_levels.param_names == from_int.param_names
        assert torch.equal(from_levels(free_params), from_int(free_params))

    def test_accepts_keywords(self):
        mat = EqualEntryMatrix(n=3, m=2)
        assert mat.shape == (3, 2)

    def test_requires_n(self):
        with pytest.raises(TypeError):
            EqualEntryMatrix()

    @pytest.mark.parametrize("n", [0, -1, 3.0, "3", None, torch.tensor(3), torch.tensor([1, 2])])
    def test_rejects_bad_n(self, n):
        with pytest.raises(ValueError, match="positive int"):
            EqualEntryMatrix(n)

    @pytest.mark.parametrize("m", [0, -1, 2.5, "2", torch.tensor(2)])
    def test_rejects_bad_m(self, m):
        with pytest.raises(ValueError, match="positive int"):
            EqualEntryMatrix(3, m=m)

    def test_rejects_empty_levels(self):
        with pytest.raises(ValueError, match="positive int"):
            EqualEntryMatrix([])

    def test_rejects_empty_m_levels(self):
        with pytest.raises(ValueError, match="positive int"):
            EqualEntryMatrix(3, m=[])

    def test_accepts_bool_size_as_one(self):
        mat = EqualEntryMatrix(True)
        assert mat.shape == (1, 1)

    def test_bool_size_fails_on_call(self):
        mat = EqualEntryMatrix(True)
        with pytest.raises(TypeError):
            mat()

    def test_default_param_specs(self):
        mat = EqualEntryMatrix(3)
        assert mat.param_names == ["sigma^2"]
        assert mat.param_specs["sigma^2"]["fixed"] is False
        assert torch.equal(mat.param_specs["sigma^2"]["default"], torch.tensor([0.0]))
        assert isinstance(mat.param_specs["sigma^2"]["trans"], TransformExpPow2)

    def test_default_param_counts(self):
        mat = EqualEntryMatrix(3)
        assert mat.num_params == 1
        assert mat.num_free_params == 1
        assert mat.num_fixed_params == 0

    def test_default_param_name_properties(self):
        mat = EqualEntryMatrix(3)
        assert mat.free_param_names == ["sigma^2"]
        assert mat.fixed_param_names == []
        assert mat.free_param_index == [0]
        assert mat.fixed_param_index == []

    def test_default_param_dict_properties(self):
        mat = EqualEntryMatrix(3)
        assert mat.param_defaults.keys() == {"sigma^2"}
        assert mat.free_param_defaults.keys() == {"sigma^2"}
        assert mat.fixed_param_defaults == {}
        assert mat.param_trans.keys() == {"sigma^2"}
        assert mat.free_param_trans.keys() == {"sigma^2"}
        assert mat.fixed_param_trans == {}

    def test_empty_param_specs_fall_back_to_defaults(self):
        mat = EqualEntryMatrix(3, param_specs={})
        assert mat.param_names == ["sigma^2"]

    def test_falsy_non_dict_param_specs_fall_back_to_defaults(self):
        mat = EqualEntryMatrix(3, param_specs=[])
        assert mat.param_names == ["sigma^2"]

    def test_instances_do_not_share_param_specs(self):
        assert EqualEntryMatrix(3).param_specs is not EqualEntryMatrix(3).param_specs

    def test_instances_do_not_share_transforms(self):
        first = EqualEntryMatrix(3).param_specs["sigma^2"]["trans"]
        second = EqualEntryMatrix(3).param_specs["sigma^2"]["trans"]
        assert first is not second

    def test_grad_mode_default(self):
        assert EqualEntryMatrix(3).grad_mode == "default"

    def test_jacobian_defaults(self):
        mat = EqualEntryMatrix(3)
        assert mat.jacobian_method == "jacfwd"
        assert mat.jacobian_chunk_size is None

    def test_get_default_dtype_device(self):
        mat = EqualEntryMatrix(3)
        assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float32)

    def test_get_default_dtype_device_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = EqualEntryMatrix(3)
            assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float64)
        finally:
            torch.set_default_dtype(original)

    def test_rejects_non_dict_param_specs(self):
        with pytest.raises(TypeError, match="must be a dict"):
            EqualEntryMatrix(3, param_specs="sigma^2")

    def test_rejects_non_str_param_name(self):
        with pytest.raises(TypeError, match="must be a str"):
            EqualEntryMatrix(3, param_specs={
                0: {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            })

    def test_rejects_non_dict_spec(self):
        with pytest.raises(TypeError, match="Individual parameter specification must be a dict"):
            EqualEntryMatrix(3, param_specs={"sigma^2": [False, torch.tensor([0.0])]})

    @pytest.mark.parametrize(
        "spec",
        [
            {"fixed": False, "default": torch.tensor([0.0])},
            {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2(), "extra": 1},
        ],
    )
    def test_rejects_wrong_spec_fields(self, spec):
        with pytest.raises(TypeError, match="must be 'fixed', 'default', and 'trans'"):
            EqualEntryMatrix(3, param_specs={"sigma^2": spec})

    def test_rejects_non_bool_fixed(self):
        with pytest.raises(TypeError, match="'fixed' must be a bool"):
            EqualEntryMatrix(3, param_specs={
                "sigma^2": {"fixed": 0, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            })

    def test_rejects_non_tensor_default(self):
        with pytest.raises(TypeError, match="'default' must be a torch.Tensor"):
            EqualEntryMatrix(3, param_specs={
                "sigma^2": {"fixed": False, "default": 0.0, "trans": TransformExpPow2()},
            })

    def test_rejects_non_1d_default(self):
        with pytest.raises(TypeError, match="'default' must be a 1D torch.Tensor"):
            EqualEntryMatrix(3, param_specs={
                "sigma^2": {"fixed": False, "default": torch.zeros(1, 1), "trans": TransformExpPow2()},
            })

    def test_rejects_non_transform(self):
        with pytest.raises(TypeError, match="'trans' must be a Transform"):
            EqualEntryMatrix(3, param_specs={
                "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": "exp"},
            })

    def test_repr_dict(self):
        mat = EqualEntryMatrix(3, 2)
        assert set(mat.repr_dict.keys()) == {"shape", "param_specs"}
        assert mat.repr_dict["shape"] == (3, 2)
        assert mat.repr_dict["param_specs"] is mat.param_specs


class TestEqualEntryMatrixCallValues:
    """``__call__`` builds ``sigma^2 * J`` and follows its parameter input."""

    def test_default_call_is_all_ones(self):
        mat = EqualEntryMatrix(3)
        assert torch.equal(mat(), torch.ones(3, 3))

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, -2.0])
    def test_call_all_entries_equal_transformed_sigma2(self, theta):
        mat = EqualEntryMatrix(3)
        free_params = torch.tensor([theta])
        result = mat(free_params)
        expected = torch.exp(2.0 * torch.tensor(theta))
        assert torch.allclose(result, expected.expand(3, 3))
        assert torch.equal(result, result.T)

    def test_call_rectangular_values(self):
        mat = EqualEntryMatrix(3, 2)
        result = mat(torch.tensor([0.5]))
        assert result.shape == (3, 2)
        assert torch.allclose(result, torch.full((3, 2), torch.exp(torch.tensor(1.0))))

    def test_call_rectangular_m_greater_than_n(self):
        mat = EqualEntryMatrix(2, 5)
        result = mat(torch.tensor([0.0]))
        assert result.shape == (2, 5)
        assert torch.equal(result, torch.ones(2, 5))

    def test_call_n1(self):
        mat = EqualEntryMatrix(1)
        result = mat(torch.tensor([0.5]))
        assert result.shape == (1, 1)
        assert torch.allclose(result, torch.exp(torch.tensor(1.0)).reshape(1, 1))

    def test_call_negative_theta_stays_positive(self):
        mat = EqualEntryMatrix(3)
        assert (mat(torch.tensor([-5.0])) > 0).all()

    def test_call_without_arguments_uses_defaults(self):
        mat = EqualEntryMatrix(3)
        assert torch.equal(mat(), mat(torch.tensor([0.0])))

    def test_call_accepts_dict(self):
        mat = EqualEntryMatrix(3)
        assert torch.equal(
            mat({"sigma^2": torch.tensor([0.5])}),
            mat(torch.tensor([0.5])),
        )

    def test_call_custom_param_name_and_transform(self):
        mat = EqualEntryMatrix(2, param_specs=single_spec("v", default=0.5, trans=TransformExp()))
        result = mat(torch.tensor([0.5]))
        assert torch.allclose(result, torch.full((2, 2), torch.exp(torch.tensor(0.5))))
        assert mat.param_names == ["v"]

    def test_call_custom_default_value(self):
        mat = EqualEntryMatrix(2, param_specs=single_spec("v", default=0.25, trans=TransformIdentity()))
        assert torch.allclose(mat(), torch.full((2, 2), torch.tensor(0.25)))

    def test_call_uses_fixed_default(self):
        mat = EqualEntryMatrix(2, param_specs=single_spec("v", fixed=True, default=0.5, trans=TransformExp()))
        assert torch.allclose(mat(), torch.full((2, 2), torch.exp(torch.tensor(0.5))))

    def test_call_rejects_unexpected_keyword(self):
        mat = EqualEntryMatrix(2)
        with pytest.raises(TypeError):
            mat(theta=torch.tensor([0.0]))

    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (2, 5), (5, 2)])
    def test_call_shape_matches_construction(self, shape):
        n, m = shape
        assert EqualEntryMatrix(n, m)(torch.tensor([0.0])).shape == shape

    def test_call_does_not_change_later_default_calls(self):
        mat = EqualEntryMatrix(3)
        mat(torch.tensor([0.5], dtype=torch.float64))
        assert torch.equal(mat(), torch.ones(3, 3))


class TestEqualEntryMatrixFixedParams:
    """Fixed-only and custom single-parameter configurations."""

    def fixed_matrix(self):
        return EqualEntryMatrix(3, param_specs=single_spec("v", fixed=True, default=0.5, trans=TransformExp()))

    def test_all_fixed_param_counts(self):
        mat = self.fixed_matrix()
        assert mat.num_params == 1
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 1
        assert mat.free_param_names == []
        assert mat.fixed_param_names == ["v"]
        assert mat.free_param_index == []
        assert mat.fixed_param_index == [0]

    def test_fixed_defaults_dict(self):
        mat = self.fixed_matrix()
        assert mat.fixed_param_defaults.keys() == {"v"}
        assert mat.free_param_defaults == {}
        assert mat.param_trans.keys() == {"v"}
        assert mat.free_param_trans == {}

    def test_fixed_call(self):
        assert torch.allclose(self.fixed_matrix()(), torch.full((3, 3), torch.exp(torch.tensor(0.5))))

    def test_manual_grad_returns_none(self):
        grad, names = self.fixed_matrix().manual_grad()
        assert grad is None
        assert names == []

    def test_auto_grad_returns_none(self):
        grad, names = self.fixed_matrix().auto_grad()
        assert grad is None
        assert names == []

    def test_grad_returns_none_in_default_mode(self):
        grad, names = self.fixed_matrix().grad()
        assert grad is None
        assert names == []

    def test_grad_returns_none_in_auto_mode(self):
        mat = self.fixed_matrix()
        mat.grad_mode = "auto"
        grad, names = mat.grad()
        assert grad is None
        assert names == []

    def test_build_params_includes_fixed_default(self):
        mat = self.fixed_matrix()
        assert torch.allclose(mat.build_params(), torch.exp(torch.tensor(0.5)).reshape(1))

    def test_build_params_without_fixed_is_empty(self):
        mat = self.fixed_matrix()
        built = mat.build_params(include_fixed=False)
        assert built.shape == (0,)

    def test_build_params_without_fixed_dict_is_empty(self):
        mat = self.fixed_matrix()
        assert mat.build_params(include_fixed=False, out_format="dict") == {}

    def test_fixed_matrix_rejects_non_empty_free_params(self):
        mat = self.fixed_matrix()
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.tensor([1.0]))
        with pytest.raises(ValueError, match="length 0"):
            mat.grad(torch.tensor([1.0]))

    def test_fixed_matrix_trans_grad_is_empty(self):
        mat = self.fixed_matrix()
        grad = mat.trans_grad()
        assert grad.shape == (0,)
        assert grad.dtype == torch.float32


class TestEqualEntryMatrixBuildParams:
    """``build_params`` composition for a single-variance matrix."""

    def test_defaults_to_transformed_default(self):
        mat = EqualEntryMatrix(3)
        assert torch.allclose(mat.build_params(), torch.tensor([1.0]))

    def test_applies_transform(self):
        mat = EqualEntryMatrix(3)
        assert torch.allclose(mat.build_params(torch.tensor([0.5])), torch.exp(torch.tensor(1.0)).reshape(1))

    def test_trans_false_returns_raw_parameter(self):
        mat = EqualEntryMatrix(3)
        assert torch.allclose(mat.build_params(torch.tensor([0.5]), trans=False), torch.tensor([0.5]))

    def test_include_fixed_false_keeps_free_parameter(self):
        mat = EqualEntryMatrix(3)
        assert torch.allclose(mat.build_params(torch.tensor([0.5]), include_fixed=False), torch.exp(torch.tensor(1.0)).reshape(1))

    def test_include_fixed_false_trans_false(self):
        mat = EqualEntryMatrix(3)
        assert torch.allclose(
            mat.build_params(torch.tensor([0.5]), include_fixed=False, trans=False),
            torch.tensor([0.5]),
        )

    def test_dict_output_keys_and_values(self):
        mat = EqualEntryMatrix(3)
        result = mat.build_params(torch.tensor([0.5]), out_format="dict")
        assert list(result.keys()) == ["sigma^2"]
        assert result["sigma^2"].shape == (1,)
        assert torch.allclose(result["sigma^2"], torch.exp(torch.tensor(1.0)).reshape(1))

    def test_dict_output_contains_fixed_parameter(self):
        mat = EqualEntryMatrix(3, param_specs=mixed_specs())
        result = mat.build_params(torch.tensor([0.5]), out_format="dict")
        assert list(result.keys()) == ["a", "b"]
        assert torch.allclose(result["a"], torch.exp(torch.tensor(1.0)).reshape(1))
        assert torch.allclose(result["b"], torch.exp(torch.tensor(2.0)).reshape(1))

    def test_dict_output_without_fixed(self):
        mat = EqualEntryMatrix(3, param_specs=mixed_specs())
        result = mat.build_params(torch.tensor([0.5]), include_fixed=False, out_format="dict")
        assert list(result.keys()) == ["a"]

    def test_accepts_dict_input(self):
        mat = EqualEntryMatrix(3)
        assert torch.equal(
            mat.build_params({"sigma^2": torch.tensor([0.5])}),
            mat.build_params(torch.tensor([0.5])),
        )

    def test_accepts_free_param_default_dict(self):
        mat = EqualEntryMatrix(3)
        assert torch.allclose(mat.build_params(mat.free_param_defaults), torch.tensor([1.0]))

    def test_rejects_unknown_out_format(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Unexpected 'out_format'"):
            mat.build_params(torch.tensor([0.5]), out_format="list")

    def test_rejects_unknown_out_format_on_empty_output(self):
        mat = EqualEntryMatrix(3, param_specs=single_spec("v", fixed=True))
        with pytest.raises(ValueError, match="Unexpected 'out_format'"):
            mat.build_params(include_fixed=False, out_format="list")

    @pytest.mark.parametrize("length", [0, 2, 3])
    def test_rejects_wrong_length(self, length):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat.build_params(torch.zeros(length))

    def test_rejects_non_1d_tensor(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat.build_params(torch.zeros(1, 1))

    def test_rejects_non_tensor(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.build_params([0.5])

    def test_rejects_dict_with_missing_key(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.build_params({})

    def test_rejects_dict_with_extra_key(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({"sigma^2": torch.tensor([0.0]), "x": torch.tensor([1.0])})

    def test_rejects_dict_with_non_1d_value(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat.build_params({"sigma^2": torch.zeros(1, 1)})

    def test_rejects_dict_values_of_mixed_dtype(self):
        mat = EqualEntryMatrix(3, param_specs=two_specs())
        with pytest.raises(ValueError, match="All free parameters must share the same dtype and device"):
            mat.build_params({
                "a": torch.tensor([0.0], dtype=torch.float64),
                "b": torch.tensor([0.0], dtype=torch.float32),
            })


class TestEqualEntryMatrixFreeParamsValidation:
    """Every entry point that takes ``free_params`` validates it the same way."""

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    @pytest.mark.parametrize("length", [0, 2])
    def test_entry_point_rejects_wrong_length(self, entry_point, length):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            getattr(mat, entry_point)(torch.zeros(length))

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    def test_entry_point_rejects_non_1d_tensor(self, entry_point):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            getattr(mat, entry_point)(torch.zeros(1, 1))

    def test_call_rejects_zero_dim_tensor(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.tensor(0.5))

    @pytest.mark.parametrize("free_params", [[0.0], 0.0, "params", 1])
    def test_call_rejects_non_tensor(self, free_params):
        mat = EqualEntryMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat(free_params)

    @pytest.mark.parametrize("entry_point", ["grad", "manual_grad", "auto_grad", "trans_grad", "build_params"])
    def test_other_entry_points_reject_non_tensor(self, entry_point):
        mat = EqualEntryMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            getattr(mat, entry_point)(0.5)

    def test_call_rejects_dict_with_extra_key(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({"sigma^2": torch.tensor([0.0]), "x": torch.tensor([1.0])})

    def test_call_rejects_empty_dict(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat({})

    def test_call_rejects_wrong_dict_key(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat({"var": torch.tensor([0.0])})

    def test_manual_grad_rejects_extra_dict_key(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.manual_grad({"sigma^2": torch.tensor([0.0]), "x": torch.tensor([1.0])})

    @pytest.mark.parametrize("entry_point", ["grad", "auto_grad", "trans_grad", "build_params"])
    def test_entry_points_reject_missing_dict_key(self, entry_point):
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="Missing free parameters"):
            getattr(mat, entry_point)({})

    def test_call_on_accelerator_validates_length(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = EqualEntryMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat(torch.zeros(2, device=device))


class TestEqualEntryMatrixTransforms:
    """The default and custom transforms apply to the shared variance."""

    def test_default_transform_is_exp_pow2(self):
        mat = EqualEntryMatrix(2)
        assert isinstance(mat.free_param_trans["sigma^2"], TransformExpPow2)

    @pytest.mark.parametrize("theta", [0.0, 0.5, -1.5])
    def test_trans_grad_values(self, theta):
        mat = EqualEntryMatrix(2)
        expected = 2.0 * torch.exp(2.0 * torch.tensor(theta))
        assert torch.allclose(mat.trans_grad(torch.tensor([theta])), expected.reshape(1))

    def test_trans_grad_without_arguments(self):
        mat = EqualEntryMatrix(2)
        assert torch.allclose(mat.trans_grad(), torch.tensor([2.0]))

    def test_trans_grad_accepts_dict(self):
        mat = EqualEntryMatrix(2)
        assert torch.allclose(
            mat.trans_grad({"sigma^2": torch.tensor([0.5])}),
            mat.trans_grad(torch.tensor([0.5])),
        )

    def test_identity_transform_is_applied_verbatim(self):
        mat = EqualEntryMatrix(2, param_specs=single_spec("v", default=2.0, trans=TransformIdentity()))
        assert torch.allclose(mat(), torch.full((2, 2), torch.tensor(2.0)))
        assert torch.allclose(mat.trans_grad(torch.tensor([2.0])), torch.tensor([1.0]))

    def test_custom_transforms_per_spec(self):
        mat = EqualEntryMatrix(2, param_specs={"v": {
            "fixed": False,
            "default": torch.tensor([0.5]),
            "trans": TransformExp(),
        }})
        assert torch.allclose(mat(torch.tensor([0.5])), torch.full((2, 2), torch.exp(torch.tensor(0.5))))


class TestEqualEntryMatrixGrad:
    """Manual and automatic gradients of the shared variance."""

    @pytest.mark.parametrize("shape", [(3, 3), (3, 2), (2, 5)])
    def test_manual_grad_shape_and_names(self, shape):
        n, m = shape
        grad, names = EqualEntryMatrix(n, m).manual_grad(torch.tensor([0.5]))
        assert grad.shape == (1, n, m)
        assert names == ["sigma^2"]

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, -2.0])
    def test_manual_grad_is_trans_grad_times_ones(self, theta):
        mat = EqualEntryMatrix(3, 2)
        free_params = torch.tensor([theta])
        grad, _ = mat.manual_grad(free_params)
        expected = 2.0 * torch.exp(2.0 * torch.tensor(theta)) * torch.ones(3, 2)
        assert torch.allclose(grad[0], expected)

    def test_manual_grad_has_no_zero_entries(self):
        grad, _ = EqualEntryMatrix(3).manual_grad(torch.tensor([0.5]))
        assert (grad != 0.0).all()

    def test_manual_grad_without_arguments(self):
        mat = EqualEntryMatrix(3)
        grad, names = mat.manual_grad()
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.0]))[0])
        assert names == ["sigma^2"]

    def test_manual_grad_accepts_dict(self):
        mat = EqualEntryMatrix(3)
        assert torch.equal(
            mat.manual_grad({"sigma^2": torch.tensor([0.5])})[0],
            mat.manual_grad(torch.tensor([0.5]))[0],
        )

    @pytest.mark.parametrize("shape", [(3, 3), (3, 2), (2, 5)])
    def test_auto_grad_shape_and_names(self, shape):
        n, m = shape
        grad, names = EqualEntryMatrix(n, m).auto_grad(torch.tensor([0.5]))
        assert grad.shape == (1, n, m)
        assert names == ["sigma^2"]

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, -2.0])
    def test_manual_grad_matches_auto_grad(self, theta):
        mat = EqualEntryMatrix(3, 2)
        free_params = torch.tensor([theta])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    @pytest.mark.parametrize("method", ["jacfwd", "jacrev", "jacobian"])
    def test_auto_grad_methods_agree(self, method):
        mat = EqualEntryMatrix(3)
        mat.jacobian_method = method
        grad, _ = mat.auto_grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_auto_grad_jacrev_chunked(self):
        mat = EqualEntryMatrix(3)
        mat.jacobian_method = "jacrev"
        mat.jacobian_chunk_size = 1
        grad, _ = mat.auto_grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_auto_grad_rejects_unknown_method(self):
        mat = EqualEntryMatrix(3)
        mat.jacobian_method = "bogus"
        with pytest.raises(ValueError, match="Unknown Jacobian method"):
            mat.auto_grad(torch.tensor([0.5]))

    def test_auto_grad_without_arguments(self):
        mat = EqualEntryMatrix(3)
        grad, _ = mat.auto_grad()
        assert torch.allclose(grad, mat.auto_grad(torch.tensor([0.0]))[0])

    def test_grad_default_mode_uses_manual(self):
        mat = EqualEntryMatrix(3)
        grad, _ = mat.grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_grad_mode_manual(self):
        mat = EqualEntryMatrix(3)
        mat.grad_mode = "manual"
        grad, _ = mat.grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_grad_mode_auto(self):
        mat = EqualEntryMatrix(3)
        mat.grad_mode = "auto"
        grad, _ = mat.grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.auto_grad(torch.tensor([0.5]))[0])

    def test_grad_mode_unknown_raises(self):
        mat = EqualEntryMatrix(3)
        mat.grad_mode = "bogus"
        with pytest.raises(RuntimeError, match="Unknown grad mode"):
            mat.grad(torch.tensor([0.5]))

    def test_grad_without_arguments(self):
        mat = EqualEntryMatrix(3)
        grad, names = mat.grad()
        assert torch.allclose(grad, mat.grad(torch.tensor([0.0]))[0])
        assert names == ["sigma^2"]

    def test_grad_accepts_dict(self):
        mat = EqualEntryMatrix(3)
        assert torch.equal(
            mat.grad({"sigma^2": torch.tensor([0.5])})[0],
            mat.grad(torch.tensor([0.5]))[0],
        )

    def test_grad_varies_with_theta(self):
        mat = EqualEntryMatrix(3)
        assert not torch.allclose(mat.grad(torch.tensor([0.0]))[0], mat.grad(torch.tensor([1.0]))[0])

    def test_grad_names_follow_custom_param_name(self):
        mat = EqualEntryMatrix(3, param_specs=single_spec("var"))
        assert mat.grad(torch.tensor([0.5]))[1] == ["var"]


class TestEqualEntryMatrixDtypeAndDevice:
    """The parameter tensor is the single source of dtype and device."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        result = EqualEntryMatrix(3)(torch.tensor([0.5], dtype=dtype))
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_build_params_follows_input_dtype(self, dtype):
        assert EqualEntryMatrix(3).build_params(torch.tensor([0.5], dtype=dtype)).dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_grad_follows_input_dtype(self, dtype):
        grad, _ = EqualEntryMatrix(3).grad(torch.tensor([0.5], dtype=dtype))
        assert grad.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_auto_grad_follows_input_dtype(self, dtype):
        grad, _ = EqualEntryMatrix(3).auto_grad(torch.tensor([0.5], dtype=dtype))
        assert grad.dtype == dtype

    def test_trans_grad_follows_input_dtype(self):
        mat = EqualEntryMatrix(3)
        assert mat.trans_grad(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_int_input_is_promoted_by_the_transform(self):
        mat = EqualEntryMatrix(3)
        result = mat(torch.tensor([0], dtype=torch.int64))
        assert result.dtype == torch.float32
        assert torch.equal(result, torch.ones(3, 3))

    def test_default_dtype_when_omitted(self):
        assert EqualEntryMatrix(3)().dtype == torch.get_default_dtype()

    def test_default_dtype_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = EqualEntryMatrix(3)
            assert mat().dtype == torch.float64
            assert mat.build_params().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_cast_call_does_not_change_later_default_calls(self):
        mat = EqualEntryMatrix(3)
        mat(torch.tensor([0.5], dtype=torch.float64))
        assert mat().dtype == torch.get_default_dtype()

    def test_input_dtype_wins_over_default_dtype(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = EqualEntryMatrix(3)
            assert mat(torch.tensor([0.5], dtype=torch.float32)).dtype == torch.float32
        finally:
            torch.set_default_dtype(original)

    def test_default_device_when_omitted(self):
        assert EqualEntryMatrix(3)().device == torch.get_default_device()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = EqualEntryMatrix(3)
        result = mat(torch.tensor([0.5], device=device))
        assert result.device.type == device
        assert torch.allclose(result.cpu(), torch.full((3, 3), torch.exp(torch.tensor(1.0))))

    def test_build_params_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert EqualEntryMatrix(3).build_params(torch.tensor([0.5], device=device)).device.type == device

    def test_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = EqualEntryMatrix(3).grad(torch.tensor([0.5], device=device))
        assert grad.device.type == device

    def test_manual_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = EqualEntryMatrix(3).manual_grad(torch.tensor([0.5], device=device))
        assert grad.device.type == device

    def test_auto_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = EqualEntryMatrix(3).auto_grad(torch.tensor([0.5], device=device))
        assert grad.device.type == device

    def test_trans_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert EqualEntryMatrix(3).trans_grad(torch.tensor([0.5], device=device)).device.type == device

    def test_follows_dtype_and_device_together(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        result = EqualEntryMatrix(3)(torch.tensor([0.5], dtype=torch.float16, device=device))
        assert result.dtype == torch.float16
        assert result.device.type == device

    def test_accelerator_call_does_not_move_the_default_call(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = EqualEntryMatrix(3)
        mat(torch.tensor([0.5], device=device))
        assert mat().device == torch.get_default_device()


class TestEqualEntryMatrixIntermediates:
    """The intermediate cache is keyed by the built parameter values."""

    def built(self, mat, theta=0.5):
        return mat.build_params(torch.tensor([theta]))

    def test_get_before_set_returns_none(self):
        mat = EqualEntryMatrix(3)
        assert mat.get_intermediates(self.built(mat)) is None

    def test_set_then_get_returns_value(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        assert mat.set_intermediates(params, {"tag": "cached"}) is None
        assert mat.get_intermediates(params) == {"tag": "cached"}

    def test_cache_is_invalidated_by_different_params(self):
        mat = EqualEntryMatrix(3)
        mat.set_intermediates(self.built(mat, 0.5), "cached")
        assert mat.get_intermediates(self.built(mat, 1.5)) is None

    def test_cache_is_invalidated_by_dtype(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        assert mat.get_intermediates(params.double()) is None

    def test_cache_is_invalidated_by_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        assert mat.get_intermediates(params.to(device)) is None

    def test_stored_key_is_copied(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        params[0] = 99.0
        assert mat.get_intermediates(self.built(mat)) == "cached"

    def test_reset_clears_cache(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.reset_intermediates()
        assert mat.get_intermediates(params) is None

    def test_set_with_empty_params_is_a_noop(self):
        mat = EqualEntryMatrix(3)
        assert mat.set_intermediates(torch.tensor([]), "cached") is None
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_overwriting_replaces_the_entry(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "first")
        mat.set_intermediates(params, "second")
        assert mat.get_intermediates(params) == "second"

    def test_auto_grad_clears_cache(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.auto_grad(torch.tensor([0.5]))
        assert mat.get_intermediates(params) is None

    def test_manual_grad_preserves_cache(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.manual_grad(torch.tensor([0.5]))
        assert mat.get_intermediates(params) == "cached"

    def test_call_preserves_cache(self):
        mat = EqualEntryMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat(torch.tensor([0.5]))
        assert mat.get_intermediates(params) == "cached"

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_set_validates_params(self, params, error):
        mat = EqualEntryMatrix(3)
        with pytest.raises(error):
            mat.set_intermediates(params, "cached")

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_get_validates_params(self, params, error):
        mat = EqualEntryMatrix(3)
        with pytest.raises(error):
            mat.get_intermediates(params)


class TestEqualEntryMatrixInverse:
    """``EqualEntryMatrix`` does not provide a closed-form inverse."""

    def test_does_not_override_manual_inverse(self):
        assert "manual_inverse" not in EqualEntryMatrix.__dict__

    def test_raises_not_implemented_with_defaults(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(NotImplementedError):
            mat.manual_inverse()

    def test_raises_not_implemented_with_free_params(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(NotImplementedError):
            mat.manual_inverse(torch.tensor([0.5]))

    def test_raises_not_implemented_with_dict(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(NotImplementedError):
            mat.manual_inverse({"sigma^2": torch.tensor([0.5])})

    def test_raises_when_all_fixed(self):
        mat = EqualEntryMatrix(3, param_specs=single_spec("v", fixed=True))
        with pytest.raises(NotImplementedError):
            mat.manual_inverse()

    def test_error_message_names_the_class_and_points_to_linalg(self):
        mat = EqualEntryMatrix(3)
        with pytest.raises(NotImplementedError, match=r"EqualEntryMatrix does not provide a closed-form inverse"):
            mat.manual_inverse()
        with pytest.raises(NotImplementedError, match=r"torch\.linalg\.inv"):
            mat.manual_inverse()


class TestEqualEntryMatrixRepr:
    """The single-line repr shows the shape and the parameter specification."""

    def test_repr_contains_class_name(self):
        assert "EqualEntryMatrix" in repr(EqualEntryMatrix(3))

    def test_repr_contains_shape(self):
        assert "(3, 2)" in repr(EqualEntryMatrix(3, 2))

    def test_repr_contains_param_name(self):
        assert "sigma^2" in repr(EqualEntryMatrix(3))

    def test_repr_is_single_line(self):
        assert "\n" not in repr(EqualEntryMatrix(3))

    def test_repr_does_not_abbreviate_a_single_param(self):
        assert "..." not in repr(EqualEntryMatrix(3))

    def test_repr_contains_custom_param_name(self):
        mat = EqualEntryMatrix(2, param_specs=single_spec("v"))
        assert "v" in repr(mat)

    def test_repr_starts_with_class_name(self):
        assert repr(EqualEntryMatrix(2)).startswith("EqualEntryMatrix(")


class TestEqualEntryMatrixMultiParamSpecLimitation:
    """A multi-parameter spec constructs, but the class builds one shared value.

    These tests pin the limitation (and one latent gradient-shape bug) rather
    than a supported configuration: ``EqualEntryMatrix`` models a single
    variance replicated across all entries, so a two-parameter spec has no
    well-defined ``sigma^2 * J`` result.
    """

    def test_two_free_params_construct(self):
        mat = EqualEntryMatrix(3, param_specs=two_specs())
        assert mat.num_params == 2
        assert mat.num_free_params == 2
        assert mat.param_names == ["a", "b"]

    def test_mixed_params_construct(self):
        mat = EqualEntryMatrix(3, param_specs=mixed_specs())
        assert mat.num_free_params == 1
        assert mat.num_fixed_params == 1
        assert mat.free_param_names == ["a"]
        assert mat.fixed_param_names == ["b"]

    def test_build_params_returns_one_value_per_parameter(self):
        mat = EqualEntryMatrix(3, param_specs=two_specs())
        built = mat.build_params(torch.tensor([0.0, 0.5]))
        assert built.shape == (2,)
        assert torch.allclose(built, torch.tensor([1.0, torch.exp(torch.tensor(1.0))]))

    def test_build_params_dict_keys_both_parameters(self):
        mat = EqualEntryMatrix(3, param_specs=two_specs())
        assert list(mat.build_params(torch.tensor([0.0, 0.5]), out_format="dict").keys()) == ["a", "b"]

    def test_call_raises_on_two_free_params(self):
        mat = EqualEntryMatrix(3, param_specs=two_specs())
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat(torch.tensor([0.0, 0.5]))

    def test_call_raises_on_defaults(self):
        mat = EqualEntryMatrix(3, param_specs=two_specs())
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat()

    def test_call_raises_on_free_param_dict(self):
        mat = EqualEntryMatrix(3, param_specs=two_specs())
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat(mat.free_param_defaults)

    def test_call_raises_on_one_free_one_fixed_param(self):
        mat = EqualEntryMatrix(3, param_specs=mixed_specs())
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat(torch.tensor([0.5]))

    def test_auto_grad_is_defined_when_the_call_is(self):
        mat = EqualEntryMatrix(2, param_specs=two_specs())
        grad, names = mat.auto_grad(torch.tensor([0.0, 1.0]))
        assert grad.shape == (2, 2, 2)
        assert names == ["a", "b"]

    def test_call_on_2x2_is_not_an_equal_entry_matrix(self):
        # At this shape the (num_free_params,) vector broadcasts along the
        # columns of J instead of raising: each transformed variance is
        # repeated down its column and the matrix is no longer equal-entry.
        mat = EqualEntryMatrix(2, param_specs=two_specs())
        result = mat(torch.tensor([0.0, 1.0]))
        expected = torch.full((2, 2), torch.exp(torch.tensor(2.0)))
        expected[:, 0] = 1.0
        assert torch.allclose(result, expected)
        assert not torch.allclose(result, result[0, 0].expand(2, 2))

    @pytest.mark.xfail(
        reason=(
            "EqualEntryMatrix.manual_grad multiplies the (num_free_params,) transform "
            "gradient by the (n, m) all-ones matrix and only then unsqueezes, so a "
            "two-parameter spec on a 2x2 matrix returns (1, 2, 2) while auto_grad "
            "returns (2, 2, 2); the multi-parameter spec is outside the class's "
            "single-variance model, so this is a latent shape-contract violation"
        ),
        strict=False,
    )
    def test_two_param_manual_grad_shape_matches_auto_grad(self):
        mat = EqualEntryMatrix(2, param_specs=two_specs())
        free_params = torch.tensor([0.0, 1.0])
        manual, manual_names = mat.manual_grad(free_params)
        auto, auto_names = mat.auto_grad(free_params)
        assert manual.shape == (2, 2, 2)
        assert torch.allclose(manual, auto)
        assert manual_names == auto_names

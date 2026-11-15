import pandas as pd
import pytest
import torch

from torch_openreml.covariance import LowerTriangularMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExpPow2,
    TransformIdentity,
)

ENTRY_POINTS = ["__call__", "grad", "manual_grad", "auto_grad", "trans_grad", "build_params"]


def spec(*, fixed=False, default=1.0, trans=None):
    """One parameter specification for a lower-triangular entry."""
    return {
        "fixed": fixed,
        "default": torch.tensor([float(default)]),
        "trans": trans if trans is not None else TransformIdentity(),
    }


def custom_specs(names, *, defaults=None, transforms=None):
    """Build a full specification dict in the requested parameter order."""
    defaults = defaults if defaults is not None else [1.0] * len(names)
    transforms = transforms if transforms is not None else [TransformIdentity()] * len(names)
    return {
        name: {"fixed": False, "default": torch.tensor([float(default)]), "trans": trans}
        for name, default, trans in zip(names, defaults, transforms)
    }


def mixed_3x2_specs(fixed_name="L_1_0", fixed_value=9.0):
    """The default (3, 2) parameter surface with one entry fixed."""
    specs = {}
    for i in range(3):
        for j in range(min(i + 1, 2)):
            name = f"L_{i}_{j}"
            specs[name] = spec(fixed=name == fixed_name, default=1.0)
    specs[fixed_name]["default"] = torch.tensor([float(fixed_value)])
    return specs


class TestLowerTriangularMatrixConstructor:
    """Construction, shape, default parameter surface, and input validation."""

    def test_is_matrix_subclass(self):
        assert isinstance(LowerTriangularMatrix(3, 2), Matrix)

    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 2), (2, 5), (5, 1)])
    def test_shape(self, shape):
        n, m = shape
        mat = LowerTriangularMatrix(n, m)
        assert mat.shape == shape
        assert isinstance(mat.shape, tuple)

    def test_shape_from_levels(self):
        assert LowerTriangularMatrix(["a", "b", "a", "c"], ["x", "y"]).shape == (3, 2)
        assert LowerTriangularMatrix(("a", "b", "a"), ("x", "y", "x")).shape == (2, 2)
        assert LowerTriangularMatrix(pd.Series(["a", "b", "a"]), pd.Series(["x", "y"])).shape == (2, 2)

    def test_levels_equivalent_to_distinct_count(self):
        from_levels = LowerTriangularMatrix(["a", "b", "a"], ["x", "y", "x", "z"])
        from_int = LowerTriangularMatrix(2, 3)
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert from_levels.shape == from_int.shape
        assert from_levels.param_names == from_int.param_names
        assert torch.equal(from_levels(free_params), from_int(free_params))

    def test_accepts_keywords(self):
        assert LowerTriangularMatrix(n=3, m=2).shape == (3, 2)

    def test_requires_n(self):
        with pytest.raises(TypeError):
            LowerTriangularMatrix()

    def test_requires_m(self):
        with pytest.raises(TypeError):
            LowerTriangularMatrix(3)

    @pytest.mark.parametrize("n", [0, -1])
    def test_rejects_non_positive_n(self, n):
        with pytest.raises(ValueError, match="positive int"):
            LowerTriangularMatrix(n, 2)

    @pytest.mark.parametrize("m", [0, -1])
    def test_rejects_non_positive_m(self, m):
        with pytest.raises(ValueError, match="positive int"):
            LowerTriangularMatrix(3, m)

    @pytest.mark.parametrize("n", [3.0, "3", None, torch.tensor(3)])
    def test_rejects_non_int_n(self, n):
        # A non-integer n fails before shape validation, while building the
        # default parameter specs with range(n).
        with pytest.raises((TypeError, ValueError)):
            LowerTriangularMatrix(n, 2)

    @pytest.mark.parametrize("m", [2.5, "2", None, torch.tensor(2)])
    def test_rejects_non_int_m(self, m):
        with pytest.raises((TypeError, ValueError)):
            LowerTriangularMatrix(3, m)

    def test_rejects_empty_levels(self):
        with pytest.raises(ValueError, match="positive int"):
            LowerTriangularMatrix([], 2)

    def test_rejects_empty_m_levels(self):
        with pytest.raises(ValueError, match="positive int"):
            LowerTriangularMatrix(3, [])

    def test_accepts_bool_dims_as_one(self):
        mat = LowerTriangularMatrix(True, True)
        assert mat.shape == (1, 1)

    def test_bool_dims_fail_on_call(self):
        mat = LowerTriangularMatrix(True, True)
        with pytest.raises(TypeError):
            mat()

    def test_default_param_names_3x2(self):
        mat = LowerTriangularMatrix(3, 2)
        assert mat.param_names == ["L_0_0", "L_1_0", "L_1_1", "L_2_0", "L_2_1"]

    def test_default_param_names_3x3(self):
        mat = LowerTriangularMatrix(3, 3)
        assert mat.param_names == [
            "L_0_0",
            "L_1_0", "L_1_1",
            "L_2_0", "L_2_1", "L_2_2",
        ]

    def test_default_param_names_2x5(self):
        # Column indices are bounded by the row index, so an extra-wide matrix
        # has no parameters beyond the diagonal of the first min(n, m) rows.
        mat = LowerTriangularMatrix(2, 5)
        assert mat.param_names == ["L_0_0", "L_1_0", "L_1_1"]

    def test_default_param_counts(self):
        mat = LowerTriangularMatrix(4, 2)
        assert mat.num_params == 7
        assert mat.num_free_params == 7
        assert mat.num_fixed_params == 0
        assert mat.free_param_names == mat.param_names
        assert mat.free_param_index == list(range(7))
        assert mat.fixed_param_index == []

    def test_default_param_specs(self):
        mat = LowerTriangularMatrix(2, 2)
        for name in mat.param_names:
            assert mat.param_specs[name]["fixed"] is False
            assert torch.equal(mat.param_specs[name]["default"], torch.tensor([1.0]))
            assert isinstance(mat.param_specs[name]["trans"], TransformIdentity)

    def test_default_param_dict_properties(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat.param_defaults.keys() == {"L_0_0", "L_1_0", "L_1_1"}
        assert mat.free_param_defaults.keys() == {"L_0_0", "L_1_0", "L_1_1"}
        assert mat.fixed_param_defaults == {}
        assert mat.param_trans.keys() == {"L_0_0", "L_1_0", "L_1_1"}
        assert mat.free_param_trans.keys() == {"L_0_0", "L_1_0", "L_1_1"}
        assert mat.fixed_param_trans == {}

    def test_empty_param_specs_fall_back_to_defaults(self):
        mat = LowerTriangularMatrix(2, 2, param_specs={})
        assert mat.param_names == ["L_0_0", "L_1_0", "L_1_1"]

    def test_falsy_non_dict_param_specs_fall_back_to_defaults(self):
        mat = LowerTriangularMatrix(2, 2, param_specs=[])
        assert mat.param_names == ["L_0_0", "L_1_0", "L_1_1"]

    def test_instances_do_not_share_param_specs(self):
        assert LowerTriangularMatrix(2, 2).param_specs is not LowerTriangularMatrix(2, 2).param_specs

    def test_instances_do_not_share_transforms(self):
        first = LowerTriangularMatrix(2, 2).param_specs["L_0_0"]["trans"]
        second = LowerTriangularMatrix(2, 2).param_specs["L_0_0"]["trans"]
        assert first is not second

    def test_grad_mode_default(self):
        assert LowerTriangularMatrix(2, 2).grad_mode == "default"

    def test_jacobian_defaults(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat.jacobian_method == "jacfwd"
        assert mat.jacobian_chunk_size is None

    def test_get_default_dtype_device(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float32)

    def test_get_default_dtype_device_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = LowerTriangularMatrix(2, 2)
            assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float64)
        finally:
            torch.set_default_dtype(original)

    def test_rejects_non_dict_param_specs(self):
        with pytest.raises(TypeError, match="must be a dict"):
            LowerTriangularMatrix(2, 2, param_specs="L_0_0")

    def test_rejects_non_str_param_name(self):
        with pytest.raises(TypeError, match="must be a str"):
            LowerTriangularMatrix(2, 2, param_specs={
                0: {"fixed": False, "default": torch.tensor([1.0]), "trans": TransformIdentity()},
            })

    def test_rejects_non_dict_spec(self):
        with pytest.raises(TypeError, match="Individual parameter specification must be a dict"):
            LowerTriangularMatrix(2, 2, param_specs={"L_0_0": [False, torch.tensor([1.0])]})

    @pytest.mark.parametrize(
        "bad_spec",
        [
            {"fixed": False, "default": torch.tensor([1.0])},
            {"fixed": False, "default": torch.tensor([1.0]), "trans": TransformIdentity(), "extra": 1},
        ],
    )
    def test_rejects_wrong_spec_fields(self, bad_spec):
        with pytest.raises(TypeError, match="must be 'fixed', 'default', and 'trans'"):
            LowerTriangularMatrix(2, 2, param_specs={"L_0_0": bad_spec})

    def test_rejects_non_bool_fixed(self):
        with pytest.raises(TypeError, match="'fixed' must be a bool"):
            LowerTriangularMatrix(2, 2, param_specs={
                "L_0_0": {"fixed": 0, "default": torch.tensor([1.0]), "trans": TransformIdentity()},
            })

    def test_rejects_non_tensor_default(self):
        with pytest.raises(TypeError, match="'default' must be a torch.Tensor"):
            LowerTriangularMatrix(2, 2, param_specs={
                "L_0_0": {"fixed": False, "default": 1.0, "trans": TransformIdentity()},
            })

    def test_rejects_non_1d_default(self):
        with pytest.raises(TypeError, match="'default' must be a 1D torch.Tensor"):
            LowerTriangularMatrix(2, 2, param_specs={
                "L_0_0": {"fixed": False, "default": torch.zeros(1, 1), "trans": TransformIdentity()},
            })

    def test_rejects_non_transform(self):
        with pytest.raises(TypeError, match="'trans' must be a Transform"):
            LowerTriangularMatrix(2, 2, param_specs={
                "L_0_0": {"fixed": False, "default": torch.tensor([1.0]), "trans": "identity"},
            })

    def test_repr_dict(self):
        mat = LowerTriangularMatrix(3, 2)
        assert set(mat.repr_dict.keys()) == {"shape", "param_specs"}
        assert mat.repr_dict["shape"] == (3, 2)
        assert mat.repr_dict["param_specs"] is mat.param_specs


class TestLowerTriangularMatrixCallValues:
    """``__call__`` scatters the parameters onto the lower triangle."""

    def test_default_call_has_unit_lower_triangle(self):
        mat = LowerTriangularMatrix(3, 2)
        expected = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        assert torch.equal(mat(), expected)

    def test_call_places_each_parameter_in_tril_order(self):
        mat = LowerTriangularMatrix(3, 2)
        result = mat(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
        expected = torch.tensor([[0.1, 0.0], [0.2, 0.3], [0.4, 0.5]])
        assert torch.allclose(result, expected)

    def test_call_zeros_are_exactly_zero(self):
        mat = LowerTriangularMatrix(4, 3)
        result = mat(torch.arange(1.0, 10.0))
        assert (result[torch.triu(torch.ones(4, 3), diagonal=1).bool()] == 0.0).all()

    def test_call_above_diagonal_zero_on_the_diagonal_row(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat(torch.tensor([2.0, 3.0, 4.0]))[0, 1] == 0.0

    @pytest.mark.parametrize(
        "shape, expected",
        [
            ((1, 1), [[2.0]]),
            ((1, 4), [[2.0, 0.0, 0.0, 0.0]]),
            ((3, 1), [[2.0], [3.0], [4.0]]),
            ((2, 4), [[2.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]),
        ],
    )
    def test_call_rectangular_shapes(self, shape, expected):
        n, m = shape
        mat = LowerTriangularMatrix(n, m)
        result = mat(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0][:mat.num_free_params]))
        assert result.shape == shape
        assert torch.equal(result, torch.tensor(expected))

    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 2), (4, 2), (2, 5)])
    def test_call_shape_matches_construction(self, shape):
        n, m = shape
        mat = LowerTriangularMatrix(n, m)
        assert mat(torch.ones(mat.num_free_params)).shape == shape

    def test_call_without_arguments_uses_defaults(self):
        mat = LowerTriangularMatrix(2, 2)
        assert torch.equal(mat(), mat(torch.tensor([1.0, 1.0, 1.0])))

    def test_call_accepts_dict(self):
        mat = LowerTriangularMatrix(2, 2)
        assert torch.equal(
            mat({"L_0_0": torch.tensor([2.0]), "L_1_0": torch.tensor([3.0]), "L_1_1": torch.tensor([4.0])}),
            mat(torch.tensor([2.0, 3.0, 4.0])),
        )

    def test_call_custom_param_specs(self):
        mat = LowerTriangularMatrix(2, 2, param_specs=custom_specs(
            ["a", "b", "c"], defaults=[2.0, 3.0, 4.0]
        ))
        assert torch.equal(mat(), torch.tensor([[2.0, 0.0], [3.0, 4.0]]))
        assert mat.param_names == ["a", "b", "c"]

    def test_call_uses_fixed_default(self):
        mat = LowerTriangularMatrix(2, 2, param_specs={
            "L_0_0": spec(default=2.0),
            "L_1_0": spec(fixed=True, default=7.0),
            "L_1_1": spec(default=4.0),
        })
        assert torch.equal(mat(torch.tensor([2.0, 4.0])), torch.tensor([[2.0, 0.0], [7.0, 4.0]]))

    def test_call_rejects_unexpected_keyword(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(TypeError):
            mat(theta=torch.ones(3))

    def test_call_does_not_change_later_default_calls(self):
        mat = LowerTriangularMatrix(2, 2)
        mat(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
        assert torch.equal(mat(), mat(torch.tensor([1.0, 1.0, 1.0])))


class TestLowerTriangularMatrixFixedParams:
    """All-fixed and mixed fixed/free configurations."""

    def all_fixed_matrix(self):
        return LowerTriangularMatrix(2, 2, param_specs={
            "L_0_0": spec(fixed=True, default=2.0),
            "L_1_0": spec(fixed=True, default=3.0),
            "L_1_1": spec(fixed=True, default=4.0),
        })

    def test_all_fixed_param_counts(self):
        mat = self.all_fixed_matrix()
        assert mat.num_params == 3
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 3
        assert mat.free_param_names == []
        assert mat.fixed_param_names == ["L_0_0", "L_1_0", "L_1_1"]
        assert mat.free_param_index == []
        assert mat.fixed_param_index == [0, 1, 2]

    def test_all_fixed_call(self):
        assert torch.equal(self.all_fixed_matrix()(), torch.tensor([[2.0, 0.0], [3.0, 4.0]]))

    def test_all_fixed_manual_grad_returns_none(self):
        grad, names = self.all_fixed_matrix().manual_grad()
        assert grad is None
        assert names == []

    def test_all_fixed_auto_grad_returns_none(self):
        grad, names = self.all_fixed_matrix().auto_grad()
        assert grad is None
        assert names == []

    def test_all_fixed_grad_returns_none(self):
        grad, names = self.all_fixed_matrix().grad()
        assert grad is None
        assert names == []

    def test_all_fixed_build_params_includes_fixed_defaults(self):
        assert torch.equal(self.all_fixed_matrix().build_params(), torch.tensor([2.0, 3.0, 4.0]))

    def test_all_fixed_build_params_without_fixed_is_empty(self):
        assert self.all_fixed_matrix().build_params(include_fixed=False).shape == (0,)

    def test_all_fixed_trans_grad_is_empty(self):
        grad = self.all_fixed_matrix().trans_grad()
        assert grad.shape == (0,)
        assert grad.dtype == torch.float32

    def test_all_fixed_matrix_rejects_non_empty_free_params(self):
        mat = self.all_fixed_matrix()
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.tensor([1.0]))
        with pytest.raises(ValueError, match="length 0"):
            mat.grad(torch.tensor([1.0]))

    def test_mixed_param_counts(self):
        mat = LowerTriangularMatrix(3, 2, param_specs=mixed_3x2_specs())
        assert mat.num_params == 5
        assert mat.num_free_params == 4
        assert mat.num_fixed_params == 1
        assert mat.free_param_names == ["L_0_0", "L_1_1", "L_2_0", "L_2_1"]
        assert mat.fixed_param_names == ["L_1_0"]
        assert mat.free_param_index == [0, 2, 3, 4]
        assert mat.fixed_param_index == [1]

    def test_mixed_call(self):
        mat = LowerTriangularMatrix(3, 2, param_specs=mixed_3x2_specs())
        free_params = torch.tensor([2.0, 3.0, 4.0, 5.0])
        expected = torch.tensor([[2.0, 0.0], [9.0, 3.0], [4.0, 5.0]])
        assert torch.equal(mat(free_params), expected)

    def test_mixed_manual_grad_uses_only_free_slices(self):
        mat = LowerTriangularMatrix(3, 2, param_specs=mixed_3x2_specs())
        free_params = torch.tensor([2.0, 3.0, 4.0, 5.0])
        grad, names = mat.manual_grad(free_params)
        assert grad.shape == (4, 3, 2)
        assert names == ["L_0_0", "L_1_1", "L_2_0", "L_2_1"]
        positions = [(0, 0), (1, 1), (2, 0), (2, 1)]
        for k, (i, j) in enumerate(positions):
            assert grad[k, i, j] == 1.0
            assert grad[k].sum() == 1.0

    def test_mixed_manual_grad_matches_auto_grad(self):
        mat = LowerTriangularMatrix(3, 2, param_specs=mixed_3x2_specs())
        free_params = torch.tensor([2.0, 3.0, 4.0, 5.0])
        assert torch.allclose(mat.manual_grad(free_params)[0], mat.auto_grad(free_params)[0])

    def test_mixed_build_params_includes_fixed_value(self):
        mat = LowerTriangularMatrix(3, 2, param_specs=mixed_3x2_specs())
        built = mat.build_params(torch.tensor([2.0, 3.0, 4.0, 5.0]))
        assert torch.equal(built, torch.tensor([2.0, 9.0, 3.0, 4.0, 5.0]))

    def test_mixed_build_params_dict_keys_all_parameters(self):
        mat = LowerTriangularMatrix(3, 2, param_specs=mixed_3x2_specs())
        result = mat.build_params(torch.tensor([2.0, 3.0, 4.0, 5.0]), out_format="dict")
        assert list(result.keys()) == ["L_0_0", "L_1_0", "L_1_1", "L_2_0", "L_2_1"]
        assert torch.equal(result["L_1_0"], torch.tensor([9.0]))

    def test_fixed_last_param_manual_grad_matches_auto_grad(self):
        mat = LowerTriangularMatrix(2, 2, param_specs={
            "L_0_0": spec(default=1.0),
            "L_1_0": spec(default=1.0),
            "L_1_1": spec(fixed=True, default=7.0),
        })
        free_params = torch.tensor([2.0, 3.0])
        assert torch.equal(mat(free_params), torch.tensor([[2.0, 0.0], [3.0, 7.0]]))
        manual, names = mat.manual_grad(free_params)
        auto, _ = mat.auto_grad(free_params)
        assert names == ["L_0_0", "L_1_0"]
        assert torch.allclose(manual, auto)


class TestLowerTriangularMatrixCustomTransforms:
    """Mixed transforms exercise the chain rule entry by entry."""

    def mixed_transform_matrix(self):
        return LowerTriangularMatrix(2, 2, param_specs={
            "L_0_0": spec(default=0.0, trans=TransformExpPow2()),
            "L_1_0": spec(default=3.0),
            "L_1_1": spec(default=4.0),
        })

    def test_call_applies_each_transform(self):
        mat = self.mixed_transform_matrix()
        result = mat(torch.tensor([0.5, 3.0, 4.0]))
        expected = torch.tensor([[torch.exp(torch.tensor(1.0)), 0.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_manual_grad_applies_each_transform_derivative(self):
        mat = self.mixed_transform_matrix()
        free_params = torch.tensor([0.5, 3.0, 4.0])
        grad, _ = mat.manual_grad(free_params)
        expected_00 = 2.0 * torch.exp(torch.tensor(1.0))
        assert torch.allclose(grad[0, 0, 0], expected_00)
        assert grad[0].sum() == expected_00
        assert grad[1, 1, 0] == 1.0
        assert grad[1].sum() == 1.0
        assert grad[2, 1, 1] == 1.0

    def test_manual_grad_matches_auto_grad_with_mixed_transforms(self):
        mat = self.mixed_transform_matrix()
        free_params = torch.tensor([0.5, 3.0, 4.0])
        assert torch.allclose(mat.manual_grad(free_params)[0], mat.auto_grad(free_params)[0])

    def test_trans_grad_combines_transforms(self):
        mat = self.mixed_transform_matrix()
        expected = torch.tensor([2.0 * torch.exp(torch.tensor(1.0)), 1.0, 1.0])
        assert torch.allclose(mat.trans_grad(torch.tensor([0.5, 3.0, 4.0])), expected)


class TestLowerTriangularMatrixGrad:
    """Manual and automatic gradients of the lower-triangular entries."""

    @pytest.mark.parametrize("shape", [(3, 2), (4, 2), (2, 4), (1, 1)])
    def test_manual_grad_shape_and_names(self, shape):
        n, m = shape
        mat = LowerTriangularMatrix(n, m)
        grad, names = mat.manual_grad(torch.ones(mat.num_free_params))
        assert grad.shape == (mat.num_free_params, n, m)
        assert names == mat.free_param_names

    def test_manual_grad_one_nonzero_per_slice(self):
        mat = LowerTriangularMatrix(3, 2)
        grad, _ = mat.manual_grad(torch.ones(5))
        positions = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
        for k, (i, j) in enumerate(positions):
            assert grad[k, i, j] == 1.0
            assert grad[k].sum() == 1.0

    def test_manual_grad_values_with_identity_transform(self):
        mat = LowerTriangularMatrix(2, 2)
        grad, _ = mat.manual_grad(torch.tensor([2.0, 3.0, 4.0]))
        assert torch.equal(grad, torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]))

    def test_manual_grad_rectangular_n_greater_than_m(self):
        mat = LowerTriangularMatrix(4, 2)
        grad, _ = mat.manual_grad(torch.linspace(0.1, 0.7, 7))
        expected_positions = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        for k, (i, j) in enumerate(expected_positions):
            assert grad[k, i, j] == 1.0
            assert grad[k].sum() == 1.0
        assert grad.sum() == 7.0

    def test_manual_grad_matches_auto_grad(self):
        mat = LowerTriangularMatrix(4, 2)
        free_params = torch.linspace(0.1, 0.7, 7)
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_without_arguments(self):
        mat = LowerTriangularMatrix(2, 2)
        grad, names = mat.manual_grad()
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([1.0, 1.0, 1.0]))[0])
        assert names == ["L_0_0", "L_1_0", "L_1_1"]

    def test_manual_grad_accepts_dict(self):
        mat = LowerTriangularMatrix(2, 2)
        assert torch.equal(
            mat.manual_grad({"L_0_0": torch.tensor([2.0]), "L_1_0": torch.tensor([3.0]), "L_1_1": torch.tensor([4.0])})[0],
            mat.manual_grad(torch.tensor([2.0, 3.0, 4.0]))[0],
        )

    def test_auto_grad_shape_and_names(self):
        mat = LowerTriangularMatrix(3, 2)
        grad, names = mat.auto_grad(torch.ones(5))
        assert grad.shape == (5, 3, 2)
        assert names == mat.param_names

    @pytest.mark.parametrize("method", ["jacfwd", "jacrev", "jacobian"])
    def test_auto_grad_methods_agree(self, method):
        mat = LowerTriangularMatrix(3, 2)
        mat.jacobian_method = method
        grad, _ = mat.auto_grad(torch.linspace(0.1, 0.5, 5))
        assert torch.allclose(grad, mat.manual_grad(torch.linspace(0.1, 0.5, 5))[0])

    def test_auto_grad_jacrev_chunked(self):
        mat = LowerTriangularMatrix(3, 2)
        mat.jacobian_method = "jacrev"
        mat.jacobian_chunk_size = 2
        grad, _ = mat.auto_grad(torch.linspace(0.1, 0.5, 5))
        assert torch.allclose(grad, mat.manual_grad(torch.linspace(0.1, 0.5, 5))[0])

    def test_auto_grad_rejects_unknown_method(self):
        mat = LowerTriangularMatrix(2, 2)
        mat.jacobian_method = "bogus"
        with pytest.raises(ValueError, match="Unknown Jacobian method"):
            mat.auto_grad(torch.ones(3))

    def test_auto_grad_without_arguments(self):
        mat = LowerTriangularMatrix(2, 2)
        grad, _ = mat.auto_grad()
        assert torch.allclose(grad, mat.auto_grad(torch.ones(3))[0])

    def test_grad_default_mode_uses_manual(self):
        mat = LowerTriangularMatrix(2, 2)
        grad, _ = mat.grad(torch.tensor([2.0, 3.0, 4.0]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([2.0, 3.0, 4.0]))[0])

    def test_grad_mode_manual(self):
        mat = LowerTriangularMatrix(2, 2)
        mat.grad_mode = "manual"
        grad, _ = mat.grad(torch.tensor([2.0, 3.0, 4.0]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([2.0, 3.0, 4.0]))[0])

    def test_grad_mode_auto(self):
        mat = LowerTriangularMatrix(2, 2)
        mat.grad_mode = "auto"
        grad, _ = mat.grad(torch.tensor([2.0, 3.0, 4.0]))
        assert torch.allclose(grad, mat.auto_grad(torch.tensor([2.0, 3.0, 4.0]))[0])

    def test_grad_mode_unknown_raises(self):
        mat = LowerTriangularMatrix(2, 2)
        mat.grad_mode = "bogus"
        with pytest.raises(RuntimeError, match="Unknown grad mode"):
            mat.grad(torch.ones(3))

    def test_grad_without_arguments(self):
        mat = LowerTriangularMatrix(2, 2)
        grad, names = mat.grad()
        assert torch.allclose(grad, mat.grad(torch.ones(3))[0])
        assert names == ["L_0_0", "L_1_0", "L_1_1"]

    def test_grad_accepts_dict(self):
        mat = LowerTriangularMatrix(2, 2)
        assert torch.equal(
            mat.grad({"L_0_0": torch.tensor([2.0]), "L_1_0": torch.tensor([3.0]), "L_1_1": torch.tensor([4.0])})[0],
            mat.grad(torch.tensor([2.0, 3.0, 4.0]))[0],
        )

    def test_grad_varies_with_params_only_through_diagonal_transforms(self):
        mat = LowerTriangularMatrix(2, 2)
        assert torch.equal(mat.grad(torch.tensor([2.0, 3.0, 4.0]))[0], mat.grad(torch.tensor([9.0, 8.0, 7.0]))[0])

    def test_grad_names_follow_custom_param_names(self):
        mat = LowerTriangularMatrix(2, 2, param_specs=custom_specs(["a", "b", "c"]))
        assert mat.grad(torch.ones(3))[1] == ["a", "b", "c"]


class TestLowerTriangularMatrixBuildParams:
    """``build_params`` composition for a lower-triangular matrix."""

    def test_defaults_to_unit_values(self):
        mat = LowerTriangularMatrix(3, 2)
        assert torch.equal(mat.build_params(), torch.ones(5))

    def test_trans_false_returns_raw_parameters(self):
        mat = LowerTriangularMatrix(2, 2)
        free_params = torch.tensor([2.0, 3.0, 4.0])
        assert torch.equal(mat.build_params(free_params, trans=False), free_params)

    def test_include_fixed_false_keeps_free_parameters(self):
        mat = LowerTriangularMatrix(2, 2, param_specs=mixed_specs_2x2())
        assert torch.equal(mat.build_params(torch.tensor([2.0, 4.0]), include_fixed=False), torch.tensor([2.0, 4.0]))

    def test_dict_output_keys_and_values(self):
        mat = LowerTriangularMatrix(2, 2)
        result = mat.build_params(torch.tensor([2.0, 3.0, 4.0]), out_format="dict")
        assert list(result.keys()) == ["L_0_0", "L_1_0", "L_1_1"]
        for value in result.values():
            assert value.shape == (1,)
        assert torch.equal(result["L_1_0"], torch.tensor([3.0]))

    def test_dict_output_without_fixed(self):
        mat = LowerTriangularMatrix(2, 2, param_specs=mixed_specs_2x2())
        result = mat.build_params(torch.tensor([2.0, 4.0]), include_fixed=False, out_format="dict")
        assert list(result.keys()) == ["L_0_0", "L_1_1"]

    def test_accepts_dict_input(self):
        mat = LowerTriangularMatrix(2, 2)
        dict_input = {"L_0_0": torch.tensor([2.0]), "L_1_0": torch.tensor([3.0]), "L_1_1": torch.tensor([4.0])}
        assert torch.equal(mat.build_params(dict_input), mat.build_params(torch.tensor([2.0, 3.0, 4.0])))

    def test_accepts_free_param_default_dict(self):
        mat = LowerTriangularMatrix(2, 2)
        assert torch.equal(mat.build_params(mat.free_param_defaults), torch.ones(3))

    def test_rejects_unknown_out_format(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="Unexpected 'out_format'"):
            mat.build_params(torch.ones(3), out_format="list")

    def test_rejects_unknown_out_format_on_empty_output(self):
        mat = LowerTriangularMatrix(2, 2, param_specs=all_fixed_specs_2x2())
        with pytest.raises(ValueError, match="Unexpected 'out_format'"):
            mat.build_params(include_fixed=False, out_format="list")

    @pytest.mark.parametrize("length", [0, 2, 4, 6])
    def test_rejects_wrong_length(self, length):
        mat = LowerTriangularMatrix(3, 2)
        with pytest.raises(ValueError, match="length 5"):
            mat.build_params(torch.zeros(length))

    def test_rejects_non_1d_tensor(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="1D tensor"):
            mat.build_params(torch.zeros(1, 1))

    def test_rejects_non_tensor(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.build_params([1.0, 1.0, 1.0])

    def test_rejects_dict_with_missing_key(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.build_params({"L_0_0": torch.tensor([1.0])})

    def test_rejects_dict_with_extra_key(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({
                "L_0_0": torch.tensor([1.0]),
                "L_1_0": torch.tensor([1.0]),
                "L_1_1": torch.tensor([1.0]),
                "x": torch.tensor([1.0]),
            })

    def test_rejects_dict_with_non_1d_values(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="1D tensor"):
            mat.build_params({
                "L_0_0": torch.zeros(1, 1),
                "L_1_0": torch.zeros(1, 1),
                "L_1_1": torch.zeros(1, 1),
            })

    def test_rejects_dict_values_of_mixed_dtype(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="All free parameters must share the same dtype and device"):
            mat.build_params({
                "L_0_0": torch.tensor([1.0], dtype=torch.float64),
                "L_1_0": torch.tensor([1.0], dtype=torch.float32),
                "L_1_1": torch.tensor([1.0], dtype=torch.float32),
            })


def mixed_specs_2x2():
    return {
        "L_0_0": spec(default=1.0),
        "L_1_0": spec(fixed=True, default=7.0),
        "L_1_1": spec(default=1.0),
    }


def all_fixed_specs_2x2():
    return {
        "L_0_0": spec(fixed=True, default=2.0),
        "L_1_0": spec(fixed=True, default=3.0),
        "L_1_1": spec(fixed=True, default=4.0),
    }


class TestLowerTriangularMatrixFreeParamsValidation:
    """Every entry point that takes ``free_params`` validates it the same way."""

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    @pytest.mark.parametrize("length", [0, 4, 6])
    def test_entry_point_rejects_wrong_length(self, entry_point, length):
        mat = LowerTriangularMatrix(3, 2)
        with pytest.raises(ValueError, match="length 5"):
            getattr(mat, entry_point)(torch.zeros(length))

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    def test_entry_point_rejects_non_1d_tensor(self, entry_point):
        mat = LowerTriangularMatrix(3, 2)
        with pytest.raises(ValueError, match="1D tensor"):
            getattr(mat, entry_point)(torch.zeros(1, 1))

    def test_call_rejects_zero_dim_tensor(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.tensor(1.0))

    @pytest.mark.parametrize("free_params", [[1.0, 1.0, 1.0], 1.0, "params", 1])
    def test_call_rejects_non_tensor(self, free_params):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat(free_params)

    @pytest.mark.parametrize("entry_point", ["grad", "manual_grad", "auto_grad", "trans_grad", "build_params"])
    def test_other_entry_points_reject_non_tensor(self, entry_point):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(TypeError, match="Torch tensor"):
            getattr(mat, entry_point)(1.0)

    def test_call_rejects_dict_with_extra_key(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({
                "L_0_0": torch.tensor([1.0]),
                "L_1_0": torch.tensor([1.0]),
                "L_1_1": torch.tensor([1.0]),
                "x": torch.tensor([1.0]),
            })

    def test_call_rejects_empty_dict(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat({})

    def test_manual_grad_rejects_extra_dict_key(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.manual_grad({
                "L_0_0": torch.tensor([1.0]),
                "L_1_0": torch.tensor([1.0]),
                "L_1_1": torch.tensor([1.0]),
                "x": torch.tensor([1.0]),
            })

    @pytest.mark.parametrize("entry_point", ["grad", "auto_grad", "trans_grad", "build_params"])
    def test_entry_points_reject_missing_dict_key(self, entry_point):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="Missing free parameters"):
            getattr(mat, entry_point)({})

    def test_call_on_accelerator_validates_length(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(ValueError, match="length 3"):
            mat(torch.zeros(2, device=device))


class TestLowerTriangularMatrixDtypeAndDevice:
    """The parameter tensor is the single source of dtype and device."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        result = LowerTriangularMatrix(2, 2)(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_build_params_follows_input_dtype(self, dtype):
        assert LowerTriangularMatrix(2, 2).build_params(torch.ones(3, dtype=dtype)).dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_manual_grad_follows_input_dtype(self, dtype):
        grad, _ = LowerTriangularMatrix(2, 2).manual_grad(torch.ones(3, dtype=dtype))
        assert grad.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_auto_grad_follows_input_dtype(self, dtype):
        grad, _ = LowerTriangularMatrix(2, 2).auto_grad(torch.ones(3, dtype=dtype))
        assert grad.dtype == dtype

    def test_trans_grad_follows_input_dtype(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat.trans_grad(torch.ones(3, dtype=torch.float64)).dtype == torch.float64

    def test_int_input_is_followed_by_the_identity_transform(self):
        mat = LowerTriangularMatrix(2, 2)
        result = mat(torch.tensor([1, 2, 3], dtype=torch.int64))
        assert result.dtype == torch.int64
        assert torch.equal(result, torch.tensor([[1, 0], [2, 3]]))

    def test_build_params_follows_int_dtype(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat.build_params(torch.tensor([1, 2, 3], dtype=torch.int64)).dtype == torch.int64

    def test_default_dtype_when_omitted(self):
        assert LowerTriangularMatrix(2, 2)().dtype == torch.get_default_dtype()

    def test_default_dtype_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = LowerTriangularMatrix(2, 2)
            assert mat().dtype == torch.float64
            assert mat.build_params().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_cast_call_does_not_change_later_default_calls(self):
        mat = LowerTriangularMatrix(2, 2)
        mat(torch.ones(3, dtype=torch.float64))
        assert mat().dtype == torch.get_default_dtype()

    def test_input_dtype_wins_over_default_dtype(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = LowerTriangularMatrix(2, 2)
            assert mat(torch.ones(3, dtype=torch.float32)).dtype == torch.float32
        finally:
            torch.set_default_dtype(original)

    def test_default_device_when_omitted(self):
        assert LowerTriangularMatrix(2, 2)().device == torch.get_default_device()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = LowerTriangularMatrix(2, 2)
        result = mat(torch.tensor([1.0, 2.0, 3.0], device=device))
        assert result.device.type == device
        assert torch.equal(result.cpu(), torch.tensor([[1.0, 0.0], [2.0, 3.0]]))

    def test_build_params_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert LowerTriangularMatrix(2, 2).build_params(torch.ones(3, device=device)).device.type == device

    def test_manual_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = LowerTriangularMatrix(2, 2).manual_grad(torch.ones(3, device=device))
        assert grad.device.type == device

    def test_auto_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = LowerTriangularMatrix(2, 2).auto_grad(torch.ones(3, device=device))
        assert grad.device.type == device

    def test_trans_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert LowerTriangularMatrix(2, 2).trans_grad(torch.ones(3, device=device)).device.type == device

    def test_follows_dtype_and_device_together(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        result = LowerTriangularMatrix(2, 2)(torch.ones(3, dtype=torch.float16, device=device))
        assert result.dtype == torch.float16
        assert result.device.type == device

    def test_accelerator_call_does_not_move_the_default_call(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = LowerTriangularMatrix(2, 2)
        mat(torch.ones(3, device=device))
        assert mat().device == torch.get_default_device()


class TestLowerTriangularMatrixIntermediates:
    """The intermediate cache is keyed by the built parameter values."""

    def built(self, mat, value=1.0):
        return mat.build_params(torch.full((mat.num_free_params,), value))

    def test_get_before_set_returns_none(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat.get_intermediates(self.built(mat)) is None

    def test_set_then_get_returns_value(self):
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        assert mat.set_intermediates(params, {"tag": "cached"}) is None
        assert mat.get_intermediates(params) == {"tag": "cached"}

    def test_cache_is_invalidated_by_different_params(self):
        mat = LowerTriangularMatrix(2, 2)
        mat.set_intermediates(self.built(mat, 1.0), "cached")
        assert mat.get_intermediates(self.built(mat, 2.0)) is None

    def test_cache_is_invalidated_by_dtype(self):
        mat = LowerTriangularMatrix(2, 2)
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
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        assert mat.get_intermediates(params.to(device)) is None

    def test_stored_key_is_copied(self):
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        params[0] = 99.0
        assert mat.get_intermediates(self.built(mat)) == "cached"

    def test_reset_clears_cache(self):
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.reset_intermediates()
        assert mat.get_intermediates(params) is None

    def test_set_with_empty_params_is_a_noop(self):
        mat = LowerTriangularMatrix(2, 2)
        assert mat.set_intermediates(torch.tensor([]), "cached") is None
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_overwriting_replaces_the_entry(self):
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        mat.set_intermediates(params, "first")
        mat.set_intermediates(params, "second")
        assert mat.get_intermediates(params) == "second"

    def test_auto_grad_clears_cache(self):
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.auto_grad(torch.ones(3))
        assert mat.get_intermediates(params) is None

    def test_manual_grad_preserves_cache(self):
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.manual_grad(torch.ones(3))
        assert mat.get_intermediates(params) == "cached"

    def test_call_preserves_cache(self):
        mat = LowerTriangularMatrix(2, 2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat(torch.ones(3))
        assert mat.get_intermediates(params) == "cached"

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_set_validates_params(self, params, error):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(error):
            mat.set_intermediates(params, "cached")

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_get_validates_params(self, params, error):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(error):
            mat.get_intermediates(params)


class TestLowerTriangularMatrixInverse:
    """``LowerTriangularMatrix`` does not provide a closed-form inverse."""

    def test_does_not_override_manual_inverse(self):
        assert "manual_inverse" not in LowerTriangularMatrix.__dict__

    def test_raises_not_implemented_with_defaults(self):
        with pytest.raises(NotImplementedError):
            LowerTriangularMatrix(2, 2).manual_inverse()

    def test_raises_not_implemented_with_free_params(self):
        with pytest.raises(NotImplementedError):
            LowerTriangularMatrix(2, 2).manual_inverse(torch.ones(3))

    def test_raises_not_implemented_with_dict(self):
        with pytest.raises(NotImplementedError):
            LowerTriangularMatrix(2, 2).manual_inverse({
                "L_0_0": torch.tensor([1.0]),
                "L_1_0": torch.tensor([1.0]),
                "L_1_1": torch.tensor([1.0]),
            })

    def test_raises_when_all_fixed(self):
        with pytest.raises(NotImplementedError):
            LowerTriangularMatrix(2, 2, param_specs=all_fixed_specs_2x2()).manual_inverse()

    def test_error_message_names_the_class_and_points_to_linalg(self):
        mat = LowerTriangularMatrix(2, 2)
        with pytest.raises(NotImplementedError, match=r"LowerTriangularMatrix does not provide a closed-form inverse"):
            mat.manual_inverse()
        with pytest.raises(NotImplementedError, match=r"torch\.linalg\.inv"):
            mat.manual_inverse()


class TestLowerTriangularMatrixRepr:
    """The single-line repr shows the shape and the abbreviated parameter specs."""

    def test_repr_contains_class_name(self):
        assert "LowerTriangularMatrix" in repr(LowerTriangularMatrix(2, 2))

    def test_repr_contains_shape(self):
        assert "(3, 2)" in repr(LowerTriangularMatrix(3, 2))

    def test_repr_contains_first_and_last_param_names(self):
        text = repr(LowerTriangularMatrix(3, 2))
        assert "L_0_0" in text
        assert "L_2_1" in text

    def test_repr_abbreviates_when_there_are_three_or_more_params(self):
        assert "..." in repr(LowerTriangularMatrix(3, 2))

    def test_repr_does_not_abbreviate_a_single_param(self):
        assert "..." not in repr(LowerTriangularMatrix(1, 1))

    def test_repr_is_single_line(self):
        assert "\n" not in repr(LowerTriangularMatrix(3, 2))

    def test_repr_starts_with_class_name(self):
        assert repr(LowerTriangularMatrix(2, 2)).startswith("LowerTriangularMatrix(")


class TestLowerTriangularMatrixSpecCount:
    """A custom spec must have one entry per lower-triangular slot.

    ``LowerTriangularMatrix`` builds its parameter vector positionally and
    assigns it to ``torch.tril_indices``. A mismatched count would otherwise
    broadcast silently for one entry, or fail deep inside ``__call__`` /
    ``manual_grad``. The constructor enforces the invariant; as with any
    ``Matrix``, mutating ``param_specs`` after construction is unsupported.
    """

    def test_rejects_too_few_specs(self):
        with pytest.raises(ValueError, match=r"expected 3 parameter specs, got 1"):
            LowerTriangularMatrix(2, 2, param_specs={"a": spec(default=1.0)})

    def test_rejects_too_many_specs(self):
        with pytest.raises(ValueError, match=r"expected 3 parameter specs, got 4"):
            LowerTriangularMatrix(2, 2, param_specs=custom_specs(["a", "b", "c", "d"]))

    def test_rectangular_slot_count(self):
        # (4, 2) has rows of 1, 2, 2, 2 lower entries = 7 slots.
        with pytest.raises(ValueError, match=r"expected 7 parameter specs, got 6"):
            LowerTriangularMatrix(4, 2, param_specs=custom_specs(list("abcdef")))
        mat = LowerTriangularMatrix(4, 2, param_specs=custom_specs(list("abcdefg")))
        assert mat.num_params == 7

    def test_accepts_a_full_custom_surface(self):
        mat = LowerTriangularMatrix(2, 2, param_specs=custom_specs(["a", "b", "c"]))
        assert mat.param_names == ["a", "b", "c"]
        assert torch.equal(
            mat(torch.tensor([1.0, 2.0, 3.0])),
            torch.tensor([[1.0, 0.0], [2.0, 3.0]]),
        )

    def test_fixed_specs_count_toward_the_total(self):
        specs = custom_specs(["a", "b", "c"])
        specs["b"]["fixed"] = True
        mat = LowerTriangularMatrix(2, 2, param_specs=specs)
        assert mat.num_params == 3
        assert mat.num_free_params == 2

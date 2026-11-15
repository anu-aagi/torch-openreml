import pandas as pd
import pytest
import torch

from torch_openreml.covariance import UnconstrainedMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExp,
    TransformExpPow2,
    TransformIdentity,
)

ENTRY_POINTS = ["__call__", "grad", "manual_grad", "auto_grad", "trans_grad", "build_params"]


def spec(*, fixed=False, default=0.0, trans=None):
    """One parameter specification for an unconstrained-entry matrix."""
    return {
        "fixed": fixed,
        "default": torch.tensor([float(default)]),
        "trans": trans if trans is not None else TransformIdentity(),
    }


def custom_specs(names, *, defaults=None, transforms=None, fixed=None):
    """Build a full specification dict in the requested parameter order."""
    defaults = defaults if defaults is not None else [0.0] * len(names)
    transforms = transforms if transforms is not None else [TransformIdentity()] * len(names)
    fixed = fixed if fixed is not None else [False] * len(names)
    return {
        name: {
            "fixed": is_fixed,
            "default": torch.tensor([float(default)]),
            "trans": trans,
        }
        for name, default, trans, is_fixed in zip(names, defaults, transforms, fixed)
    }


def lower_triangle_names(n):
    return [f"sigma^2_{i}_{j}" for i in range(n) for j in range(i + 1)]


def default_specs(n):
    return {
        f"sigma^2_{i}_{j}": spec(
            default=0.0,
            trans=TransformExpPow2() if i == j else TransformIdentity(),
        )
        for i in range(n)
        for j in range(i + 1)
    }


def mixed_3_specs(fixed_name="sigma^2_0_0", fixed_value=9.0):
    """The default 3x3 parameter surface with one entry fixed."""
    specs = default_specs(3)
    specs[fixed_name]["fixed"] = True
    specs[fixed_name]["default"] = torch.tensor([float(fixed_value)])
    return specs


class TestUnconstrainedMatrixConstructor:
    """Construction, shape, default parameter surface, and input validation."""

    def test_is_matrix_subclass(self):
        assert isinstance(UnconstrainedMatrix(3), Matrix)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_shape(self, n):
        mat = UnconstrainedMatrix(n)
        assert mat.shape == (n, n)
        assert isinstance(mat.shape, tuple)

    def test_shape_from_levels(self):
        assert UnconstrainedMatrix(["a", "b", "a", "c"]).shape == (3, 3)
        assert UnconstrainedMatrix(("a", "b", "a")).shape == (2, 2)
        assert UnconstrainedMatrix(pd.Series(["a", "b", "a", "b"])).shape == (2, 2)

    def test_levels_equivalent_to_distinct_count(self):
        from_levels = UnconstrainedMatrix(["a", "b", "a"])
        from_int = UnconstrainedMatrix(2)
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert from_levels.shape == from_int.shape
        assert from_levels.param_names == from_int.param_names
        assert torch.allclose(from_levels(free_params), from_int(free_params))

    def test_accepts_keyword(self):
        assert UnconstrainedMatrix(n=3).shape == (3, 3)

    def test_requires_n(self):
        with pytest.raises(TypeError):
            UnconstrainedMatrix()

    @pytest.mark.parametrize("n", [0, -1])
    def test_rejects_non_positive_n(self, n):
        with pytest.raises(ValueError, match="positive int"):
            UnconstrainedMatrix(n)

    @pytest.mark.parametrize("n", [3.0, "3", None, torch.tensor(3)])
    def test_rejects_non_int_n(self, n):
        # A non-integer n fails before shape validation, while building the
        # default parameter specs with range(n).
        with pytest.raises((TypeError, ValueError)):
            UnconstrainedMatrix(n)

    def test_rejects_empty_levels(self):
        with pytest.raises(ValueError, match="positive int"):
            UnconstrainedMatrix([])

    def test_accepts_bool_dim_as_one(self):
        mat = UnconstrainedMatrix(True)
        assert mat.shape == (1, 1)

    def test_bool_dim_fails_on_call(self):
        mat = UnconstrainedMatrix(True)
        with pytest.raises(TypeError):
            mat()

    def test_default_param_names_2(self):
        assert UnconstrainedMatrix(2).param_names == ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"]

    def test_default_param_names_3(self):
        assert UnconstrainedMatrix(3).param_names == [
            "sigma^2_0_0",
            "sigma^2_1_0", "sigma^2_1_1",
            "sigma^2_2_0", "sigma^2_2_1", "sigma^2_2_2",
        ]

    def test_default_param_names_1(self):
        assert UnconstrainedMatrix(1).param_names == ["sigma^2_0_0"]

    @pytest.mark.parametrize("n, expected", [(1, 1), (2, 3), (3, 6), (4, 10)])
    def test_default_param_counts(self, n, expected):
        mat = UnconstrainedMatrix(n)
        assert mat.num_params == expected
        assert mat.num_free_params == expected
        assert mat.num_fixed_params == 0
        assert mat.free_param_names == mat.param_names
        assert mat.free_param_index == list(range(expected))
        assert mat.fixed_param_index == []

    def test_default_specs_transform_diagonal_and_off_diagonal(self):
        mat = UnconstrainedMatrix(2)
        assert isinstance(mat.param_specs["sigma^2_0_0"]["trans"], TransformExpPow2)
        assert isinstance(mat.param_specs["sigma^2_1_1"]["trans"], TransformExpPow2)
        assert isinstance(mat.param_specs["sigma^2_1_0"]["trans"], TransformIdentity)
        for name in mat.param_names:
            assert mat.param_specs[name]["fixed"] is False
            assert torch.equal(mat.param_specs[name]["default"], torch.tensor([0.0]))

    def test_default_param_dict_properties(self):
        mat = UnconstrainedMatrix(2)
        names = {"sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"}
        assert mat.param_defaults.keys() == names
        assert mat.free_param_defaults.keys() == names
        assert mat.fixed_param_defaults == {}
        assert mat.param_trans.keys() == names
        assert mat.free_param_trans.keys() == names
        assert mat.fixed_param_trans == {}

    def test_empty_param_specs_fall_back_to_defaults(self):
        mat = UnconstrainedMatrix(2, param_specs={})
        assert mat.param_names == ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"]

    def test_falsy_non_dict_param_specs_fall_back_to_defaults(self):
        mat = UnconstrainedMatrix(2, param_specs=[])
        assert mat.param_names == ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"]

    def test_instances_do_not_share_param_specs(self):
        assert UnconstrainedMatrix(2).param_specs is not UnconstrainedMatrix(2).param_specs

    def test_instances_do_not_share_transforms(self):
        first = UnconstrainedMatrix(2).param_specs["sigma^2_0_0"]["trans"]
        second = UnconstrainedMatrix(2).param_specs["sigma^2_0_0"]["trans"]
        assert first is not second

    def test_grad_mode_default(self):
        assert UnconstrainedMatrix(2).grad_mode == "default"

    def test_jacobian_defaults(self):
        mat = UnconstrainedMatrix(2)
        assert mat.jacobian_method == "jacfwd"
        assert mat.jacobian_chunk_size is None

    def test_get_default_dtype_device(self):
        mat = UnconstrainedMatrix(2)
        assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float32)

    def test_get_default_dtype_device_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = UnconstrainedMatrix(2)
            assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float64)
        finally:
            torch.set_default_dtype(original)

    def test_rejects_non_dict_param_specs(self):
        with pytest.raises(TypeError, match="must be a dict"):
            UnconstrainedMatrix(2, param_specs="sigma^2_0_0")

    def test_rejects_non_str_param_name(self):
        with pytest.raises(TypeError, match="must be a str"):
            UnconstrainedMatrix(2, param_specs={0: spec()})

    def test_rejects_non_dict_spec(self):
        with pytest.raises(TypeError, match="Individual parameter specification must be a dict"):
            UnconstrainedMatrix(2, param_specs={"a": [False, torch.tensor([0.0])]})

    @pytest.mark.parametrize(
        "bad_spec",
        [
            {"fixed": False, "default": torch.tensor([0.0])},
            {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformIdentity(), "extra": 1},
        ],
    )
    def test_rejects_wrong_spec_fields(self, bad_spec):
        with pytest.raises(TypeError, match="must be 'fixed', 'default', and 'trans'"):
            UnconstrainedMatrix(2, param_specs={"a": bad_spec})

    def test_rejects_non_bool_fixed(self):
        with pytest.raises(TypeError, match="'fixed' must be a bool"):
            UnconstrainedMatrix(2, param_specs={"a": {"fixed": 0, "default": torch.tensor([0.0]), "trans": TransformIdentity()}})

    def test_rejects_non_tensor_default(self):
        with pytest.raises(TypeError, match="'default' must be a torch.Tensor"):
            UnconstrainedMatrix(2, param_specs={"a": {"fixed": False, "default": 0.0, "trans": TransformIdentity()}})

    def test_rejects_non_1d_default(self):
        with pytest.raises(TypeError, match="'default' must be a 1D torch.Tensor"):
            UnconstrainedMatrix(2, param_specs={"a": {"fixed": False, "default": torch.zeros(1, 1), "trans": TransformIdentity()}})

    def test_rejects_non_transform(self):
        with pytest.raises(TypeError, match="'trans' must be a Transform"):
            UnconstrainedMatrix(2, param_specs={"a": {"fixed": False, "default": torch.tensor([0.0]), "trans": "identity"}})

    def test_repr_dict(self):
        mat = UnconstrainedMatrix(2)
        assert set(mat.repr_dict.keys()) == {"shape", "param_specs"}
        assert mat.repr_dict["shape"] == (2, 2)
        assert mat.repr_dict["param_specs"] is mat.param_specs


class TestUnconstrainedMatrixCallValues:
    """``__call__`` builds a symmetric matrix from the lower-triangular entries."""

    def test_default_call_is_identity(self):
        assert torch.equal(UnconstrainedMatrix(3)(), torch.eye(3))

    def test_default_call_1x1(self):
        assert torch.equal(UnconstrainedMatrix(1)(), torch.ones(1, 1))

    def test_call_places_each_parameter_in_tril_order(self):
        mat = UnconstrainedMatrix(3, param_specs=custom_specs(
            lower_triangle_names(3), transforms=[TransformIdentity()] * 6
        ))
        result = mat(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        expected = torch.tensor([
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 5.0],
            [4.0, 5.0, 6.0],
        ])
        assert torch.equal(result, expected)

    def test_call_is_symmetric(self):
        mat = UnconstrainedMatrix(4)
        result = mat(torch.linspace(-0.3, 0.9, 10))
        assert torch.equal(result, result.t())

    def test_call_mirrors_off_diagonal_entries(self):
        mat = UnconstrainedMatrix(3)
        result = mat(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert result[0, 1] == result[1, 0] == 2.0
        assert result[0, 2] == result[2, 0] == 4.0
        assert result[1, 2] == result[2, 1] == 5.0

    def test_call_uses_supplied_diagonal_entries(self):
        mat = UnconstrainedMatrix(2, param_specs=custom_specs(
            lower_triangle_names(2), transforms=[TransformIdentity()] * 3
        ))
        result = mat(torch.tensor([1.0, 2.0, 3.0]))
        assert result[0, 0] == 1.0
        assert result[1, 1] == 3.0
        assert result[0, 1] == 2.0

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_call_shape_matches_construction(self, n):
        mat = UnconstrainedMatrix(n)
        assert mat(torch.zeros(mat.num_free_params)).shape == (n, n)

    def test_call_without_arguments_uses_defaults(self):
        mat = UnconstrainedMatrix(2)
        assert torch.equal(mat(), mat(torch.zeros(3)))

    def test_call_accepts_dict(self):
        mat = UnconstrainedMatrix(2)
        dict_input = {
            "sigma^2_0_0": torch.tensor([1.0]),
            "sigma^2_1_0": torch.tensor([2.0]),
            "sigma^2_1_1": torch.tensor([3.0]),
        }
        assert torch.equal(mat(dict_input), mat(torch.tensor([1.0, 2.0, 3.0])))

    def test_call_custom_param_specs(self):
        mat = UnconstrainedMatrix(2, param_specs=custom_specs(
            ["a", "b", "c"],
            defaults=[1.0, 2.0, 3.0],
        ))
        assert torch.equal(mat(), torch.tensor([[1.0, 2.0], [2.0, 3.0]]))
        assert mat.param_names == ["a", "b", "c"]

    def test_call_uses_fixed_default(self):
        mat = UnconstrainedMatrix(2, param_specs={
            "sigma^2_0_0": spec(default=0.0, trans=TransformExpPow2()),
            "sigma^2_1_0": spec(fixed=True, default=7.0),
            "sigma^2_1_1": spec(default=0.0, trans=TransformExpPow2()),
        })
        assert torch.equal(mat(torch.tensor([0.0, 0.0])), torch.tensor([[1.0, 7.0], [7.0, 1.0]]))

    def test_call_rejects_unexpected_keyword(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(TypeError):
            mat(theta=torch.zeros(3))

    def test_call_does_not_change_later_default_calls(self):
        mat = UnconstrainedMatrix(2)
        mat(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert torch.equal(mat(), torch.eye(2))


class TestUnconstrainedMatrixFixedParams:
    """All-fixed and mixed fixed/free configurations."""

    def all_fixed_matrix(self):
        return UnconstrainedMatrix(2, param_specs=custom_specs(
            ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"],
            defaults=[1.0, 2.0, 3.0],
            fixed=[True, True, True],
        ))

    def test_all_fixed_param_counts(self):
        mat = self.all_fixed_matrix()
        assert mat.num_params == 3
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 3
        assert mat.free_param_names == []
        assert mat.fixed_param_names == ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"]
        assert mat.free_param_index == []
        assert mat.fixed_param_index == [0, 1, 2]

    def test_all_fixed_call(self):
        assert torch.equal(self.all_fixed_matrix()(), torch.tensor([[1.0, 2.0], [2.0, 3.0]]))

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
        assert torch.equal(self.all_fixed_matrix().build_params(), torch.tensor([1.0, 2.0, 3.0]))

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
        mat = UnconstrainedMatrix(3, param_specs=mixed_3_specs())
        assert mat.num_params == 6
        assert mat.num_free_params == 5
        assert mat.num_fixed_params == 1
        assert mat.free_param_names == [
            "sigma^2_1_0", "sigma^2_1_1", "sigma^2_2_0", "sigma^2_2_1", "sigma^2_2_2",
        ]
        assert mat.fixed_param_names == ["sigma^2_0_0"]
        assert mat.free_param_index == [1, 2, 3, 4, 5]
        assert mat.fixed_param_index == [0]

    def test_mixed_call(self):
        mat = UnconstrainedMatrix(3, param_specs=mixed_3_specs())
        result = mat(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
        expected = torch.tensor([
            [torch.exp(torch.tensor(18.0)), 2.0, 4.0],
            [2.0, torch.exp(torch.tensor(6.0)), 5.0],
            [4.0, 5.0, torch.exp(torch.tensor(12.0))],
        ])
        assert torch.allclose(result, expected)

    def test_mixed_manual_grad_shape_and_names(self):
        mat = UnconstrainedMatrix(3, param_specs=mixed_3_specs())
        grad, names = mat.manual_grad(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
        assert grad.shape == (5, 3, 3)
        assert names == ["sigma^2_1_0", "sigma^2_1_1", "sigma^2_2_0", "sigma^2_2_1", "sigma^2_2_2"]

    def test_mixed_manual_grad_marks_both_off_diagonal_cells(self):
        mat = UnconstrainedMatrix(3, param_specs=mixed_3_specs())
        grad, _ = mat.manual_grad(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
        positions = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
        diagonal_values = {1: 3.0, 4: 6.0}
        for k, (i, j) in enumerate(positions):
            # Off-diagonal slices mark their mirrored pair; diagonal slices mark
            # only the diagonal cell.
            if i != j:
                assert grad[k, i, j] == 1.0
                assert grad[k, j, i] == 1.0
                assert grad[k].sum() == 2.0
            else:
                expected = 2.0 * torch.exp(torch.tensor(2.0 * diagonal_values[k]))
                assert torch.allclose(grad[k, i, j], expected)
                assert torch.allclose(grad[k].sum(), expected)

    def test_mixed_manual_grad_matches_auto_grad(self):
        mat = UnconstrainedMatrix(3, param_specs=mixed_3_specs())
        free_params = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        manual, _ = mat.manual_grad(free_params)
        auto, _ = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)

    def test_mixed_build_params_includes_fixed_value(self):
        mat = UnconstrainedMatrix(3, param_specs=mixed_3_specs())
        built = mat.build_params(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
        expected = torch.tensor([
            torch.exp(torch.tensor(18.0)), 2.0, torch.exp(torch.tensor(6.0)), 4.0, 5.0,
            torch.exp(torch.tensor(12.0)),
        ])
        assert torch.allclose(built, expected)

    def test_mixed_build_params_dict_keys_all_parameters(self):
        mat = UnconstrainedMatrix(3, param_specs=mixed_3_specs())
        result = mat.build_params(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]), out_format="dict")
        assert list(result.keys()) == [
            "sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1",
            "sigma^2_2_0", "sigma^2_2_1", "sigma^2_2_2",
        ]
        assert torch.allclose(result["sigma^2_0_0"], torch.exp(torch.tensor([18.0])))

    def test_fixed_off_diagonal_manual_grad_matches_auto_grad(self):
        mat = UnconstrainedMatrix(2, param_specs={
            "sigma^2_0_0": spec(default=0.0, trans=TransformExpPow2()),
            "sigma^2_1_0": spec(fixed=True, default=5.0),
            "sigma^2_1_1": spec(default=0.0, trans=TransformExpPow2()),
        })
        free_params = torch.tensor([0.0, 0.0])
        assert torch.equal(mat(free_params), torch.tensor([[1.0, 5.0], [5.0, 1.0]]))
        manual, names = mat.manual_grad(free_params)
        auto, _ = mat.auto_grad(free_params)
        assert names == ["sigma^2_0_0", "sigma^2_1_1"]
        assert torch.allclose(manual, auto)


class TestUnconstrainedMatrixCustomTransforms:
    """Mixed transforms exercise the chain rule entry by entry."""

    def mixed_transform_matrix(self):
        return UnconstrainedMatrix(2, param_specs={
            "a": spec(default=0.0, trans=TransformExp()),
            "b": spec(default=2.0, trans=TransformIdentity()),
            "c": spec(default=0.0, trans=TransformExpPow2()),
        })

    def test_call_applies_each_transform(self):
        mat = self.mixed_transform_matrix()
        result = mat(torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(result, torch.tensor([
            [torch.exp(torch.tensor(1.0)), 2.0],
            [2.0, torch.exp(torch.tensor(6.0))],
        ]))

    def test_manual_grad_applies_each_transform_derivative(self):
        mat = self.mixed_transform_matrix()
        grad, names = mat.manual_grad(torch.tensor([1.0, 2.0, 3.0]))
        assert names == ["a", "b", "c"]
        assert torch.allclose(grad[0, 0, 0], torch.exp(torch.tensor(1.0)))
        assert grad[0].sum() == torch.exp(torch.tensor(1.0))
        assert grad[1, 0, 1] == 1.0
        assert grad[1, 1, 0] == 1.0
        assert grad[1].sum() == 2.0
        assert torch.allclose(grad[2, 1, 1], 2.0 * torch.exp(torch.tensor(6.0)))

    def test_manual_grad_matches_auto_grad_with_mixed_transforms(self):
        mat = self.mixed_transform_matrix()
        free_params = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(mat.manual_grad(free_params)[0], mat.auto_grad(free_params)[0])

    def test_trans_grad_combines_transforms(self):
        mat = self.mixed_transform_matrix()
        expected = torch.tensor([
            torch.exp(torch.tensor(1.0)),
            1.0,
            2.0 * torch.exp(torch.tensor(6.0)),
        ])
        assert torch.allclose(mat.trans_grad(torch.tensor([1.0, 2.0, 3.0])), expected)


class TestUnconstrainedMatrixGrad:
    """Manual and automatic gradients of the unconstrained entries."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_manual_grad_shape_and_names(self, n):
        mat = UnconstrainedMatrix(n)
        grad, names = mat.manual_grad(torch.zeros(mat.num_free_params))
        assert grad.shape == (mat.num_free_params, n, n)
        assert names == mat.free_param_names

    def test_manual_grad_values_default_transforms(self):
        mat = UnconstrainedMatrix(2)
        grad, _ = mat.manual_grad(torch.tensor([0.0, 1.0, 0.0]))
        expected = torch.tensor([
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ])
        assert torch.equal(grad, expected)

    def test_manual_grad_diagonal_slices_are_doubled(self):
        mat = UnconstrainedMatrix(2)
        grad, _ = mat.manual_grad(torch.zeros(3))
        assert grad[0, 0, 0] == 2.0
        assert grad[2, 1, 1] == 2.0

    def test_manual_grad_off_diagonal_slices_are_symmetric(self):
        mat = UnconstrainedMatrix(3)
        grad, _ = mat.manual_grad(torch.linspace(-0.5, 0.5, 6))
        assert torch.equal(grad, grad.transpose(1, 2))

    def test_manual_grad_matches_auto_grad(self):
        mat = UnconstrainedMatrix(4)
        free_params = torch.linspace(-0.4, 0.6, 10)
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_off_diagonal_derivative_is_one(self):
        mat = UnconstrainedMatrix(2)
        grad, _ = mat.manual_grad(torch.tensor([0.3, 0.7, 0.2]))
        assert grad[1, 0, 1] == 1.0
        assert grad[1, 1, 0] == 1.0

    def test_manual_grad_without_arguments(self):
        mat = UnconstrainedMatrix(2)
        grad, names = mat.manual_grad()
        assert torch.allclose(grad, mat.manual_grad(torch.zeros(3))[0])
        assert names == ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"]

    def test_manual_grad_accepts_dict(self):
        mat = UnconstrainedMatrix(2)
        assert torch.equal(
            mat.manual_grad({
                "sigma^2_0_0": torch.tensor([0.0]),
                "sigma^2_1_0": torch.tensor([1.0]),
                "sigma^2_1_1": torch.tensor([0.0]),
            })[0],
            mat.manual_grad(torch.tensor([0.0, 1.0, 0.0]))[0],
        )

    def test_auto_grad_shape_and_names(self):
        mat = UnconstrainedMatrix(3)
        grad, names = mat.auto_grad(torch.zeros(6))
        assert grad.shape == (6, 3, 3)
        assert names == mat.param_names

    @pytest.mark.parametrize("method", ["jacfwd", "jacrev", "jacobian"])
    def test_auto_grad_methods_agree(self, method):
        mat = UnconstrainedMatrix(3)
        mat.jacobian_method = method
        free_params = torch.linspace(0.1, 0.6, 6)
        assert torch.allclose(mat.auto_grad(free_params)[0], mat.manual_grad(free_params)[0])

    def test_auto_grad_jacrev_chunked(self):
        mat = UnconstrainedMatrix(3)
        mat.jacobian_method = "jacrev"
        mat.jacobian_chunk_size = 2
        free_params = torch.linspace(0.1, 0.6, 6)
        assert torch.allclose(mat.auto_grad(free_params)[0], mat.manual_grad(free_params)[0])

    def test_auto_grad_rejects_unknown_method(self):
        mat = UnconstrainedMatrix(2)
        mat.jacobian_method = "bogus"
        with pytest.raises(ValueError, match="Unknown Jacobian method"):
            mat.auto_grad(torch.zeros(3))

    def test_auto_grad_without_arguments(self):
        mat = UnconstrainedMatrix(2)
        grad, _ = mat.auto_grad()
        assert torch.allclose(grad, mat.auto_grad(torch.zeros(3))[0])

    def test_grad_default_mode_uses_manual(self):
        mat = UnconstrainedMatrix(2)
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(mat.grad(free_params)[0], mat.manual_grad(free_params)[0])

    def test_grad_mode_manual(self):
        mat = UnconstrainedMatrix(2)
        mat.grad_mode = "manual"
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(mat.grad(free_params)[0], mat.manual_grad(free_params)[0])

    def test_grad_mode_auto(self):
        mat = UnconstrainedMatrix(2)
        mat.grad_mode = "auto"
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(mat.grad(free_params)[0], mat.auto_grad(free_params)[0])

    def test_grad_mode_unknown_raises(self):
        mat = UnconstrainedMatrix(2)
        mat.grad_mode = "bogus"
        with pytest.raises(RuntimeError, match="Unknown grad mode"):
            mat.grad(torch.zeros(3))

    def test_grad_without_arguments(self):
        mat = UnconstrainedMatrix(2)
        grad, names = mat.grad()
        assert torch.allclose(grad, mat.grad(torch.zeros(3))[0])
        assert names == ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"]

    def test_grad_accepts_dict(self):
        mat = UnconstrainedMatrix(2)
        assert torch.equal(
            mat.grad({
                "sigma^2_0_0": torch.tensor([0.0]),
                "sigma^2_1_0": torch.tensor([1.0]),
                "sigma^2_1_1": torch.tensor([0.0]),
            })[0],
            mat.grad(torch.tensor([0.0, 1.0, 0.0]))[0],
        )

    def test_grad_names_follow_custom_param_names(self):
        mat = UnconstrainedMatrix(2, param_specs=custom_specs(["a", "b", "c"]))
        assert mat.grad(torch.zeros(3))[1] == ["a", "b", "c"]


class TestUnconstrainedMatrixBuildParams:
    """``build_params`` composition for an unconstrained matrix."""

    def test_defaults_transform_diagonal_to_one(self):
        mat = UnconstrainedMatrix(3)
        assert torch.allclose(mat.build_params(), torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]))

    def test_raw_defaults_are_zero(self):
        mat = UnconstrainedMatrix(3)
        assert torch.equal(mat.build_params(trans=False), torch.zeros(6))

    def test_trans_false_returns_raw_parameters(self):
        mat = UnconstrainedMatrix(2)
        free_params = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(mat.build_params(free_params, trans=False), free_params)

    def test_transforms_diagonal_to_exp_2x(self):
        mat = UnconstrainedMatrix(2)
        built = mat.build_params(torch.tensor([0.0, 2.0, 1.0]))
        expected = torch.tensor([1.0, 2.0, torch.exp(torch.tensor(2.0))])
        assert torch.allclose(built, expected)

    def test_include_fixed_false_keeps_free_parameters(self):
        mat = UnconstrainedMatrix(2, param_specs=custom_specs(
            ["a", "b", "c"], defaults=[0.0, 0.0, 0.0], fixed=[True, False, False]
        ))
        assert torch.equal(mat.build_params(torch.tensor([2.0, 3.0]), include_fixed=False), torch.tensor([2.0, 3.0]))

    def test_dict_output_keys_and_values(self):
        mat = UnconstrainedMatrix(2)
        result = mat.build_params(torch.tensor([1.0, 2.0, 3.0]), out_format="dict")
        assert list(result.keys()) == ["sigma^2_0_0", "sigma^2_1_0", "sigma^2_1_1"]
        for value in result.values():
            assert value.shape == (1,)
        assert torch.equal(result["sigma^2_1_0"], torch.tensor([2.0]))

    def test_dict_output_without_fixed(self):
        mat = UnconstrainedMatrix(2, param_specs=custom_specs(
            ["a", "b", "c"], defaults=[0.0, 0.0, 0.0], fixed=[True, False, False]
        ))
        result = mat.build_params(torch.tensor([0.0, 0.0]), include_fixed=False, out_format="dict")
        assert list(result.keys()) == ["b", "c"]

    def test_accepts_dict_input(self):
        mat = UnconstrainedMatrix(2)
        dict_input = {
            "sigma^2_0_0": torch.tensor([1.0]),
            "sigma^2_1_0": torch.tensor([2.0]),
            "sigma^2_1_1": torch.tensor([3.0]),
        }
        assert torch.equal(mat.build_params(dict_input), mat.build_params(torch.tensor([1.0, 2.0, 3.0])))

    def test_accepts_free_param_default_dict(self):
        mat = UnconstrainedMatrix(2)
        assert torch.allclose(mat.build_params(mat.free_param_defaults), torch.tensor([1.0, 0.0, 1.0]))

    def test_rejects_unknown_out_format(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="Unexpected 'out_format'"):
            mat.build_params(torch.zeros(3), out_format="list")

    def test_rejects_unknown_out_format_on_empty_output(self):
        mat = UnconstrainedMatrix(2, param_specs=custom_specs(
            ["a", "b", "c"], defaults=[0.0, 0.0, 0.0], fixed=[True, True, True]
        ))
        with pytest.raises(ValueError, match="Unexpected 'out_format'"):
            mat.build_params(include_fixed=False, out_format="list")

    @pytest.mark.parametrize("length", [0, 5, 7])
    def test_rejects_wrong_length(self, length):
        mat = UnconstrainedMatrix(3)
        with pytest.raises(ValueError, match="length 6"):
            mat.build_params(torch.zeros(length))

    def test_rejects_non_1d_tensor(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="1D tensor"):
            mat.build_params(torch.zeros(1, 1))

    def test_rejects_non_tensor(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.build_params([1.0, 2.0, 3.0])

    def test_rejects_dict_with_missing_key(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.build_params({"sigma^2_0_0": torch.tensor([0.0])})

    def test_rejects_dict_with_extra_key(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({
                "sigma^2_0_0": torch.tensor([0.0]),
                "sigma^2_1_0": torch.tensor([0.0]),
                "sigma^2_1_1": torch.tensor([0.0]),
                "x": torch.tensor([0.0]),
            })

    def test_rejects_dict_with_non_1d_values(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="1D tensor"):
            mat.build_params({
                "sigma^2_0_0": torch.zeros(1, 1),
                "sigma^2_1_0": torch.zeros(1, 1),
                "sigma^2_1_1": torch.zeros(1, 1),
            })

    def test_rejects_dict_values_of_mixed_dtype(self):
        mat = UnconstrainedMatrix(3)
        with pytest.raises(ValueError, match="All free parameters must share the same dtype and device"):
            mat.build_params({
                "sigma^2_0_0": torch.tensor([0.0], dtype=torch.float64),
                "sigma^2_1_0": torch.tensor([0.0], dtype=torch.float32),
                "sigma^2_1_1": torch.tensor([0.0], dtype=torch.float32),
                "sigma^2_2_0": torch.tensor([0.0], dtype=torch.float32),
                "sigma^2_2_1": torch.tensor([0.0], dtype=torch.float32),
                "sigma^2_2_2": torch.tensor([0.0], dtype=torch.float32),
            })


class TestUnconstrainedMatrixFreeParamsValidation:
    """Every entry point that takes ``free_params`` validates it the same way."""

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    @pytest.mark.parametrize("length", [0, 5, 7])
    def test_entry_point_rejects_wrong_length(self, entry_point, length):
        mat = UnconstrainedMatrix(3)
        with pytest.raises(ValueError, match="length 6"):
            getattr(mat, entry_point)(torch.zeros(length))

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    def test_entry_point_rejects_non_1d_tensor(self, entry_point):
        mat = UnconstrainedMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            getattr(mat, entry_point)(torch.zeros(1, 1))

    def test_call_rejects_zero_dim_tensor(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.tensor(1.0))

    @pytest.mark.parametrize("free_params", [[0.0, 0.0, 0.0], 1.0, "params", 1])
    def test_call_rejects_non_tensor(self, free_params):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat(free_params)

    @pytest.mark.parametrize("entry_point", ["grad", "manual_grad", "auto_grad", "trans_grad", "build_params"])
    def test_other_entry_points_reject_non_tensor(self, entry_point):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(TypeError, match="Torch tensor"):
            getattr(mat, entry_point)(1.0)

    def test_call_rejects_dict_with_extra_key(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({
                "sigma^2_0_0": torch.tensor([0.0]),
                "sigma^2_1_0": torch.tensor([0.0]),
                "sigma^2_1_1": torch.tensor([0.0]),
                "x": torch.tensor([0.0]),
            })

    def test_call_rejects_empty_dict(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat({})

    def test_manual_grad_rejects_extra_dict_key(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.manual_grad({
                "sigma^2_0_0": torch.tensor([0.0]),
                "sigma^2_1_0": torch.tensor([0.0]),
                "sigma^2_1_1": torch.tensor([0.0]),
                "x": torch.tensor([0.0]),
            })

    @pytest.mark.parametrize("entry_point", ["grad", "auto_grad", "trans_grad", "build_params"])
    def test_entry_points_reject_missing_dict_key(self, entry_point):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="Missing free parameters"):
            getattr(mat, entry_point)({})

    def test_call_on_accelerator_validates_length(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = UnconstrainedMatrix(2)
        with pytest.raises(ValueError, match="length 3"):
            mat(torch.zeros(2, device=device))


class TestUnconstrainedMatrixDtypeAndDevice:
    """The parameter tensor is the single source of dtype and device."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        result = UnconstrainedMatrix(2)(torch.tensor([0.0, 1.0, 0.5], dtype=dtype))
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_build_params_follows_input_dtype(self, dtype):
        assert UnconstrainedMatrix(2).build_params(torch.zeros(3, dtype=dtype)).dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_manual_grad_follows_input_dtype(self, dtype):
        grad, _ = UnconstrainedMatrix(2).manual_grad(torch.zeros(3, dtype=dtype))
        assert grad.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_auto_grad_follows_input_dtype(self, dtype):
        grad, _ = UnconstrainedMatrix(2).auto_grad(torch.zeros(3, dtype=dtype))
        assert grad.dtype == dtype

    def test_trans_grad_follows_input_dtype(self):
        mat = UnconstrainedMatrix(2)
        assert mat.trans_grad(torch.zeros(3, dtype=torch.float64)).dtype == torch.float64

    def test_int_input_is_promoted_by_the_exp_transform(self):
        mat = UnconstrainedMatrix(2)
        result = mat(torch.tensor([1, 2, 3], dtype=torch.int64))
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor([
            [torch.exp(torch.tensor(2.0)), 2.0],
            [2.0, torch.exp(torch.tensor(6.0))],
        ]))

    def test_build_params_promotes_int_input(self):
        mat = UnconstrainedMatrix(2)
        assert mat.build_params(torch.tensor([1, 2, 3], dtype=torch.int64)).dtype == torch.float32

    def test_default_dtype_when_omitted(self):
        assert UnconstrainedMatrix(2)().dtype == torch.get_default_dtype()

    def test_default_dtype_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = UnconstrainedMatrix(2)
            assert mat().dtype == torch.float64
            assert mat.build_params().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_cast_call_does_not_change_later_default_calls(self):
        mat = UnconstrainedMatrix(2)
        mat(torch.zeros(3, dtype=torch.float64))
        assert mat().dtype == torch.get_default_dtype()

    def test_input_dtype_wins_over_default_dtype(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = UnconstrainedMatrix(2)
            assert mat(torch.zeros(3, dtype=torch.float32)).dtype == torch.float32
        finally:
            torch.set_default_dtype(original)

    def test_default_device_when_omitted(self):
        assert UnconstrainedMatrix(2)().device == torch.get_default_device()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = UnconstrainedMatrix(2)
        result = mat(torch.tensor([0.0, 1.0, 0.5], device=device))
        assert result.device.type == device
        assert torch.allclose(result.cpu(), torch.tensor([[1.0, 1.0], [1.0, torch.exp(torch.tensor(1.0))]]))

    def test_build_params_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert UnconstrainedMatrix(2).build_params(torch.zeros(3, device=device)).device.type == device

    def test_manual_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = UnconstrainedMatrix(2).manual_grad(torch.zeros(3, device=device))
        assert grad.device.type == device

    def test_auto_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = UnconstrainedMatrix(2).auto_grad(torch.zeros(3, device=device))
        assert grad.device.type == device

    def test_trans_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert UnconstrainedMatrix(2).trans_grad(torch.zeros(3, device=device)).device.type == device

    def test_follows_dtype_and_device_together(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        result = UnconstrainedMatrix(2)(torch.zeros(3, dtype=torch.float16, device=device))
        assert result.dtype == torch.float16
        assert result.device.type == device

    def test_accelerator_call_does_not_move_the_default_call(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = UnconstrainedMatrix(2)
        mat(torch.zeros(3, device=device))
        assert mat().device == torch.get_default_device()


class TestUnconstrainedMatrixIntermediates:
    """The intermediate cache is keyed by the built parameter values."""

    def built(self, mat, value=0.0):
        return mat.build_params(torch.full((mat.num_free_params,), value))

    def test_get_before_set_returns_none(self):
        mat = UnconstrainedMatrix(2)
        assert mat.get_intermediates(self.built(mat)) is None

    def test_set_then_get_returns_value(self):
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        assert mat.set_intermediates(params, {"tag": "cached"}) is None
        assert mat.get_intermediates(params) == {"tag": "cached"}

    def test_cache_is_invalidated_by_different_params(self):
        mat = UnconstrainedMatrix(2)
        mat.set_intermediates(self.built(mat, 0.0), "cached")
        assert mat.get_intermediates(self.built(mat, 1.0)) is None

    def test_cache_is_invalidated_by_dtype(self):
        mat = UnconstrainedMatrix(2)
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
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        assert mat.get_intermediates(params.to(device)) is None

    def test_stored_key_is_copied(self):
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        params[0] = 99.0
        assert mat.get_intermediates(self.built(mat)) == "cached"

    def test_reset_clears_cache(self):
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.reset_intermediates()
        assert mat.get_intermediates(params) is None

    def test_set_with_empty_params_is_a_noop(self):
        mat = UnconstrainedMatrix(2)
        assert mat.set_intermediates(torch.tensor([]), "cached") is None
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_overwriting_replaces_the_entry(self):
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        mat.set_intermediates(params, "first")
        mat.set_intermediates(params, "second")
        assert mat.get_intermediates(params) == "second"

    def test_auto_grad_clears_cache(self):
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.auto_grad(torch.zeros(3))
        assert mat.get_intermediates(params) is None

    def test_manual_grad_preserves_cache(self):
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.manual_grad(torch.zeros(3))
        assert mat.get_intermediates(params) == "cached"

    def test_call_preserves_cache(self):
        mat = UnconstrainedMatrix(2)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat(torch.zeros(3))
        assert mat.get_intermediates(params) == "cached"

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_set_validates_params(self, params, error):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(error):
            mat.set_intermediates(params, "cached")

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_get_validates_params(self, params, error):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(error):
            mat.get_intermediates(params)


class TestUnconstrainedMatrixInverse:
    """``UnconstrainedMatrix`` does not provide a closed-form inverse."""

    def test_does_not_override_manual_inverse(self):
        assert "manual_inverse" not in UnconstrainedMatrix.__dict__

    def test_raises_not_implemented_with_defaults(self):
        with pytest.raises(NotImplementedError):
            UnconstrainedMatrix(2).manual_inverse()

    def test_raises_not_implemented_with_free_params(self):
        with pytest.raises(NotImplementedError):
            UnconstrainedMatrix(2).manual_inverse(torch.zeros(3))

    def test_raises_not_implemented_with_dict(self):
        with pytest.raises(NotImplementedError):
            UnconstrainedMatrix(2).manual_inverse({
                "sigma^2_0_0": torch.tensor([0.0]),
                "sigma^2_1_0": torch.tensor([0.0]),
                "sigma^2_1_1": torch.tensor([0.0]),
            })

    def test_raises_when_all_fixed(self):
        mat = UnconstrainedMatrix(2, param_specs=custom_specs(
            ["a", "b", "c"], defaults=[0.0, 0.0, 0.0], fixed=[True, True, True]
        ))
        with pytest.raises(NotImplementedError):
            mat.manual_inverse()

    def test_error_message_names_the_class_and_points_to_linalg(self):
        mat = UnconstrainedMatrix(2)
        with pytest.raises(NotImplementedError, match=r"UnconstrainedMatrix does not provide a closed-form inverse"):
            mat.manual_inverse()
        with pytest.raises(NotImplementedError, match=r"torch\.linalg\.inv"):
            mat.manual_inverse()


class TestUnconstrainedMatrixRepr:
    """The single-line repr shows the shape and the abbreviated parameter specs."""

    def test_repr_contains_class_name(self):
        assert "UnconstrainedMatrix" in repr(UnconstrainedMatrix(2))

    def test_repr_contains_shape(self):
        assert "(3, 3)" in repr(UnconstrainedMatrix(3))

    def test_repr_contains_first_and_last_param_names(self):
        text = repr(UnconstrainedMatrix(2))
        assert "sigma^2_0_0" in text
        assert "sigma^2_1_1" in text

    def test_repr_abbreviates_when_there_are_three_or_more_params(self):
        assert "..." in repr(UnconstrainedMatrix(2))

    def test_repr_does_not_abbreviate_a_single_param(self):
        assert "..." not in repr(UnconstrainedMatrix(1))

    def test_repr_is_single_line(self):
        assert "\n" not in repr(UnconstrainedMatrix(2))

    def test_repr_starts_with_class_name(self):
        assert repr(UnconstrainedMatrix(2)).startswith("UnconstrainedMatrix(")


class TestUnconstrainedMatrixSpecCountLimitation:
    """A custom spec whose parameter count differs from the lower-triangle size.

    ``UnconstrainedMatrix`` models one parameter per lower-triangular entry,
    but nothing validates that a custom ``param_specs`` dict has one entry per
    position. These tests pin the resulting behaviour.
    """

    def one_param_matrix(self):
        return UnconstrainedMatrix(2, param_specs={"a": spec(default=1.0)})

    def four_param_matrix(self):
        return UnconstrainedMatrix(2, param_specs=custom_specs(["a", "b", "c", "d"]))

    def test_too_few_params_constructs(self):
        mat = self.one_param_matrix()
        assert mat.num_params == 1

    def test_too_few_params_call_broadcasts_the_single_value(self):
        mat = self.one_param_matrix()
        assert torch.equal(mat(torch.tensor([1.0])), torch.ones(2, 2))

    def test_too_many_params_raise_on_call(self):
        mat = self.four_param_matrix()
        with pytest.raises(RuntimeError, match="shape mismatch"):
            mat(torch.ones(4))

    @pytest.mark.xfail(
        reason=(
            "manual_grad indexes the full tril index tensors with a free-parameter "
            "mask of length num_params, so a spec with fewer parameters than "
            "lower-triangular entries raises IndexError instead of returning the "
            "documented (num_free_params, n, n) gradient that auto_grad produces"
        ),
        strict=False,
    )
    def test_too_few_params_manual_grad_shape_matches_auto_grad(self):
        mat = self.one_param_matrix()
        manual, manual_names = mat.manual_grad(torch.tensor([1.0]))
        auto, auto_names = mat.auto_grad(torch.tensor([1.0]))
        assert manual.shape == auto.shape
        assert manual_names == auto_names

import torch
import pytest
from torch_openreml.covariance import ScalarMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    Transform,
    TransformExp,
    TransformExp2,
    TransformExp10,
    TransformExpPow2,
    TransformIdentity,
    TransformPow,
    TransformScaleShift,
)


class TestScalarMatrixConstructor:
    """Construction, shape, default parameter specs, and input validation."""

    def test_is_matrix_subclass(self):
        mat = ScalarMatrix(3)
        assert isinstance(mat, Matrix)

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_shape(self, n):
        mat = ScalarMatrix(n)
        assert mat.shape == (n, n)
        assert isinstance(mat.shape, tuple)

    def test_accepts_size_keyword(self):
        mat = ScalarMatrix(n=3)
        assert mat.shape == (3, 3)

    def test_requires_size(self):
        with pytest.raises(TypeError):
            ScalarMatrix()

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="positive int"):
            ScalarMatrix(0)

    def test_rejects_negative_size(self):
        with pytest.raises(ValueError, match="positive int"):
            ScalarMatrix(-1)

    @pytest.mark.parametrize("n", [3.0, "3", None, torch.tensor(3)])
    def test_rejects_non_int_size(self, n):
        with pytest.raises(ValueError, match="positive int"):
            ScalarMatrix(n)

    def test_accepts_bool_size_as_one(self):
        mat = ScalarMatrix(True)
        assert mat.shape == (1, 1)

    def test_bool_size_fails_on_call(self):
        mat = ScalarMatrix(True)
        with pytest.raises(TypeError):
            mat()

    def test_default_param_specs(self):
        mat = ScalarMatrix(3)
        assert mat.param_names == ["sigma^2"]
        assert mat.param_specs["sigma^2"]["fixed"] is False
        assert torch.equal(mat.param_specs["sigma^2"]["default"], torch.tensor([0.0]))
        assert isinstance(mat.param_specs["sigma^2"]["trans"], TransformExpPow2)

    def test_default_param_counts(self):
        mat = ScalarMatrix(3)
        assert mat.num_params == 1
        assert mat.num_free_params == 1
        assert mat.num_fixed_params == 0

    def test_default_param_name_properties(self):
        mat = ScalarMatrix(3)
        assert mat.free_param_names == ["sigma^2"]
        assert mat.fixed_param_names == []
        assert mat.free_param_index == [0]
        assert mat.fixed_param_index == []

    def test_default_param_dict_properties(self):
        mat = ScalarMatrix(3)
        assert mat.param_defaults.keys() == {"sigma^2"}
        assert mat.free_param_defaults.keys() == {"sigma^2"}
        assert mat.fixed_param_defaults == {}
        assert mat.param_trans.keys() == {"sigma^2"}
        assert mat.free_param_trans.keys() == {"sigma^2"}
        assert mat.fixed_param_trans == {}

    def test_empty_param_specs_fall_back_to_defaults(self):
        mat = ScalarMatrix(3, param_specs={})
        assert mat.param_names == ["sigma^2"]

    def test_falsy_non_dict_param_specs_fall_back_to_defaults(self):
        mat = ScalarMatrix(3, param_specs=[])
        assert mat.param_names == ["sigma^2"]

    def test_instances_do_not_share_param_specs(self):
        assert ScalarMatrix(3).param_specs is not ScalarMatrix(3).param_specs

    def test_instances_do_not_share_transforms(self):
        first = ScalarMatrix(3).param_specs["sigma^2"]["trans"]
        second = ScalarMatrix(3).param_specs["sigma^2"]["trans"]
        assert first is not second

    def test_grad_mode_default(self):
        mat = ScalarMatrix(3)
        assert mat.grad_mode == "default"

    def test_jacobian_defaults(self):
        mat = ScalarMatrix(3)
        assert mat.jacobian_method == "jacfwd"
        assert mat.jacobian_chunk_size is None

    def test_get_default_dtype_device(self):
        mat = ScalarMatrix(3)
        assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float32)

    def test_get_default_dtype_device_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = ScalarMatrix(3)
            assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float64)
        finally:
            torch.set_default_dtype(original)

    def test_rejects_non_dict_param_specs(self):
        with pytest.raises(TypeError, match="must be a dict"):
            ScalarMatrix(3, param_specs="sigma^2")

    def test_rejects_non_str_param_name(self):
        with pytest.raises(TypeError, match="must be a str"):
            ScalarMatrix(3, param_specs={0: {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()}})

    def test_rejects_non_dict_spec(self):
        with pytest.raises(TypeError, match="Individual parameter specification must be a dict"):
            ScalarMatrix(3, param_specs={"sigma^2": [False, torch.tensor([0.0])]})

    @pytest.mark.parametrize(
        "spec",
        [
            {"fixed": False, "default": torch.tensor([0.0])},
            {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2(), "extra": 1},
        ],
    )
    def test_rejects_wrong_spec_fields(self, spec):
        with pytest.raises(TypeError, match="must be 'fixed', 'default', and 'trans'"):
            ScalarMatrix(3, param_specs={"sigma^2": spec})

    def test_rejects_non_bool_fixed(self):
        with pytest.raises(TypeError, match="'fixed' must be a bool"):
            ScalarMatrix(3, param_specs={"sigma^2": {"fixed": 0, "default": torch.tensor([0.0]), "trans": TransformExpPow2()}})

    def test_rejects_non_tensor_default(self):
        with pytest.raises(TypeError, match="'default' must be a torch.Tensor"):
            ScalarMatrix(3, param_specs={"sigma^2": {"fixed": False, "default": 0.0, "trans": TransformExpPow2()}})

    def test_rejects_non_1d_default(self):
        with pytest.raises(TypeError, match="'default' must be a 1D torch.Tensor"):
            ScalarMatrix(3, param_specs={"sigma^2": {"fixed": False, "default": torch.zeros(1, 1), "trans": TransformExpPow2()}})

    def test_rejects_non_transform(self):
        with pytest.raises(TypeError, match="'trans' must be a Transform"):
            ScalarMatrix(3, param_specs={"sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": "exp"}})


class TestScalarMatrixCallValues:
    """The matrix returned by ``__call__``."""

    def test_default_call_returns_identity(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat(), torch.eye(3))

    def test_defaults_match_default_call(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat(), mat(mat.free_param_defaults))

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, -2.0])
    def test_diagonal_is_transformed_param(self, theta):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([theta])
        result = mat(free_params)
        expected = torch.exp(2.0 * torch.tensor(theta))
        assert torch.allclose(result.diag(), expected.reshape(1).expand(3))
        assert torch.allclose(result, expected * torch.eye(3))

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, -2.0])
    def test_off_diagonal_is_zero(self, theta):
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([theta]))
        assert (result[~torch.eye(3, dtype=torch.bool)] == 0.0).all()

    def test_is_symmetric(self):
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([0.5]))
        assert torch.equal(result, result.T)

    @pytest.mark.parametrize("theta", [-20.0, -5.0, 0.0, 5.0, 20.0])
    def test_is_positive_definite(self, theta):
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([theta]))
        assert (result.diag() > 0).all()
        assert torch.linalg.cholesky(result).shape == (3, 3)

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_shape_matches_size(self, n):
        mat = ScalarMatrix(n)
        assert mat(torch.tensor([0.0])).shape == (n, n)

    def test_n1(self):
        mat = ScalarMatrix(1)
        assert torch.allclose(mat(torch.tensor([0.5])), torch.tensor([[torch.exp(torch.tensor(1.0))]]))

    def test_zero_theta_gives_ones(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat(torch.tensor([0.0])).diag(), torch.ones(3))

    def test_accepts_free_params_keyword(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat(free_params=torch.tensor([0.5])), mat(torch.tensor([0.5])))

    def test_accepts_dict_input(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat({"sigma^2": torch.tensor([0.5])}), mat(torch.tensor([0.5])))

    def test_accepts_default_dict_input(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat({"sigma^2": torch.tensor([0.0])}), mat())

    def test_consecutive_calls_are_distinct_tensors(self):
        mat = ScalarMatrix(3)
        assert mat(torch.tensor([0.5])) is not mat(torch.tensor([0.5]))

    def test_result_can_be_mutated_without_affecting_later_calls(self):
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([0.5]))
        result[0, 0] = 99.0
        result[0, 1] = 99.0
        assert torch.allclose(mat(torch.tensor([0.5])), torch.exp(torch.tensor(1.0)) * torch.eye(3))

    def test_is_stable_across_repeated_calls(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat(torch.tensor([0.5])), mat(torch.tensor([0.5])))

    def test_calls_are_independent_of_order(self):
        mat = ScalarMatrix(3)
        first = mat(torch.tensor([1.0]))
        mat(torch.tensor([-3.0]))
        assert torch.equal(mat(torch.tensor([1.0])), first)


class TestScalarMatrixTransforms:
    """The transform maps the unconstrained parameter onto the variance."""

    @pytest.mark.parametrize(
        "transform",
        [TransformExpPow2(), TransformExp(), TransformExp2(), TransformExp10(), TransformIdentity()],
    )
    def test_call_applies_the_transform(self, transform):
        mat = ScalarMatrix(2, param_specs={"v": {"fixed": False, "default": torch.tensor([0.0]), "trans": transform}})
        free_params = torch.tensor([0.5])
        assert torch.allclose(mat(free_params)[0, 0], transform(free_params).squeeze())

    @pytest.mark.parametrize(
        "transform",
        [TransformExpPow2(), TransformExp(), TransformExp2(), TransformExp10(), TransformIdentity()],
    )
    def test_trans_grad_matches_transform_grad(self, transform):
        mat = ScalarMatrix(2, param_specs={"v": {"fixed": False, "default": torch.tensor([0.0]), "trans": transform}})
        free_params = torch.tensor([0.5])
        assert torch.allclose(mat.trans_grad(free_params), transform.grad(free_params))

    @pytest.mark.parametrize(
        "transform",
        [TransformExpPow2(), TransformExp(), TransformExp2(), TransformExp10(), TransformIdentity()],
    )
    def test_manual_grad_is_transform_grad_times_eye(self, transform):
        mat = ScalarMatrix(2, param_specs={"v": {"fixed": False, "default": torch.tensor([0.0]), "trans": transform}})
        free_params = torch.tensor([0.5])
        grad, _ = mat.manual_grad(free_params)
        expected = mat.trans_grad(free_params).item() * torch.eye(2)
        assert grad.shape == (1, 2, 2)
        assert torch.allclose(grad[0], expected)

    def test_default_transform_value(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat(torch.tensor([0.5])).diag(), torch.exp(torch.tensor(1.0)))

    def test_default_trans_grad_value(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat.trans_grad(torch.tensor([0.0])), torch.tensor([2.0]))
        assert torch.allclose(mat.trans_grad(torch.tensor([0.5])), 2.0 * torch.exp(torch.tensor(1.0)))

    def test_default_trans_grad_without_arguments(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat.trans_grad(), torch.tensor([2.0]))

    def test_custom_param_name(self):
        mat = ScalarMatrix(3, param_specs={"var": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()}})
        assert mat.param_names == ["var"]
        assert mat.free_param_names == ["var"]
        assert mat({"var": torch.tensor([0.5])}).shape == (3, 3)

    def test_custom_param_name_in_grad_names(self):
        mat = ScalarMatrix(3, param_specs={"var": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()}})
        _, grad_names = mat.grad(torch.tensor([0.5]))
        assert grad_names == ["var"]

    def test_custom_param_name_in_dict_output(self):
        mat = ScalarMatrix(3, param_specs={"var": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()}})
        assert mat.build_params(torch.tensor([0.5]), out_format="dict").keys() == {"var"}

    def test_scale_shift_transform(self):
        mat = ScalarMatrix(2, param_specs={"v": {"fixed": False, "default": torch.tensor([1.0]), "trans": TransformScaleShift(a=2.0, b=1.0)}})
        free_params = torch.tensor([1.0])
        assert torch.allclose(mat(free_params)[0, 0], torch.tensor(3.0))
        assert torch.allclose(mat.manual_grad(free_params)[0][0], 2.0 * torch.eye(2))

    def test_pow_transform(self):
        mat = ScalarMatrix(2, param_specs={"v": {"fixed": False, "default": torch.tensor([1.0]), "trans": TransformPow(factor=2.0)}})
        free_params = torch.tensor([2.0])
        assert torch.allclose(mat(free_params)[0, 0], torch.tensor(4.0))
        assert torch.allclose(mat.manual_grad(free_params)[0][0], 4.0 * torch.eye(2))

    def test_transform_instance_is_used_directly(self):
        mat = ScalarMatrix(3)
        assert isinstance(mat.param_specs["sigma^2"]["trans"], Transform)


class TestScalarMatrixFixedParams:
    """A single fixed variance parameter."""

    def fixed_matrix(self):
        return ScalarMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExpPow2()},
        })

    def test_param_counts(self):
        mat = self.fixed_matrix()
        assert mat.num_params == 1
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 1

    def test_param_name_properties(self):
        mat = self.fixed_matrix()
        assert mat.free_param_names == []
        assert mat.fixed_param_names == ["sigma^2"]
        assert mat.free_param_index == []
        assert mat.fixed_param_index == [0]
        assert mat.free_param_defaults == {}
        assert mat.fixed_param_defaults.keys() == {"sigma^2"}

    def test_call_uses_default(self):
        mat = self.fixed_matrix()
        expected = torch.exp(torch.tensor(2.0)) * torch.eye(3)
        assert torch.allclose(mat(), expected)

    @pytest.mark.parametrize("free_params", [torch.tensor([]), {}, None])
    def test_call_accepts_empty_free_params_of_every_form(self, free_params):
        mat = self.fixed_matrix()
        assert torch.allclose(mat(free_params), torch.exp(torch.tensor(2.0)) * torch.eye(3))

    def test_call_is_constant_across_calls(self):
        mat = self.fixed_matrix()
        assert torch.equal(mat(), mat())

    def test_build_params_includes_fixed(self):
        mat = self.fixed_matrix()
        assert torch.allclose(mat.build_params(), torch.exp(torch.tensor(2.0)).reshape(1))

    def test_build_params_can_exclude_fixed(self):
        mat = self.fixed_matrix()
        result = mat.build_params(include_fixed=False)
        assert result.shape == (0,)

    def test_build_params_dict_output(self):
        mat = self.fixed_matrix()
        assert mat.build_params(out_format="dict").keys() == {"sigma^2"}

    def test_every_grad_path_returns_none(self):
        mat = self.fixed_matrix()
        assert mat.grad() == (None, [])
        assert mat.manual_grad() == (None, [])
        assert mat.auto_grad() == (None, [])

    @pytest.mark.parametrize("free_params", [torch.tensor([]), {}, None])
    def test_every_grad_path_returns_none_for_empty_input(self, free_params):
        mat = self.fixed_matrix()
        assert mat.grad(free_params) == (None, [])
        assert mat.manual_grad(free_params) == (None, [])
        assert mat.auto_grad(free_params) == (None, [])

    def test_grad_mode_manual_and_auto_still_return_none(self):
        mat = self.fixed_matrix()
        mat.grad_mode = "manual"
        assert mat.grad() == (None, [])
        mat.grad_mode = "auto"
        assert mat.grad() == (None, [])

    def test_trans_grad_is_empty(self):
        mat = self.fixed_matrix()
        assert mat.trans_grad().shape == (0,)

    def test_defaults_drive_dtype_and_device(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([1.0], dtype=torch.float64), "trans": TransformExpPow2()},
        })
        assert mat.get_default_dtype_device() == (torch.device("cpu"), torch.float64)
        assert mat().dtype == torch.float64

    def test_fixed_default_follows_its_own_transform(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExp()},
        })
        assert torch.allclose(mat(), torch.exp(torch.tensor(1.0)) * torch.eye(3))


class TestScalarMatrixMultiParamSpecs:
    """``ScalarMatrix`` represents one shared variance and builds exactly one value.

    A specification with more than one parameter still constructs, but
    ``build_params`` then returns ``num_params`` values, so ``__call__``
    cannot broadcast them against the ``n x n`` identity.
    """

    def two_param_matrix(self):
        return ScalarMatrix(3, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "tau^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
        })

    def mixed_param_matrix(self):
        return ScalarMatrix(3, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "tau^2": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExpPow2()},
        })

    def test_construction_is_allowed(self):
        mat = self.two_param_matrix()
        assert mat.num_params == 2
        assert mat.num_free_params == 2
        assert mat.param_names == ["sigma^2", "tau^2"]

    def test_mixed_construction_is_allowed(self):
        mat = self.mixed_param_matrix()
        assert mat.num_free_params == 1
        assert mat.num_fixed_params == 1

    def test_build_params_returns_one_value_per_parameter(self):
        mat = self.mixed_param_matrix()
        built = mat.build_params(torch.tensor([0.5]))
        assert built.shape == (2,)
        assert torch.allclose(built, torch.tensor([torch.exp(torch.tensor(1.0)), torch.exp(torch.tensor(2.0))]))

    def test_call_raises_on_two_free_params(self):
        mat = self.two_param_matrix()
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat(torch.tensor([0.0, 0.0]))

    def test_call_raises_on_one_free_one_fixed_param(self):
        mat = self.mixed_param_matrix()
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat(torch.tensor([0.5]))

    def test_call_raises_on_defaults(self):
        mat = self.mixed_param_matrix()
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat()

    def test_call_raises_on_free_param_dict(self):
        mat = self.mixed_param_matrix()
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            mat(mat.free_param_defaults)


class TestScalarMatrixFreeParamsValidation:
    """Every entry point that takes ``free_params`` validates it the same way."""

    @pytest.mark.parametrize("length", [0, 2, 3])
    def test_call_rejects_wrong_length(self, length):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat(torch.zeros(length))

    @pytest.mark.parametrize("shape", [(1, 1), (1, 2, 2)])
    def test_call_rejects_non_1d_tensor(self, shape):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.zeros(shape))

    def test_call_rejects_zero_dim_tensor(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.tensor(0.5))

    @pytest.mark.parametrize("free_params", [[0.0], 0.0, "params", 1])
    def test_call_rejects_non_tensor(self, free_params):
        mat = ScalarMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat(free_params)

    def test_call_rejects_dict_with_extra_key(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({"sigma^2": torch.tensor([0.0]), "x": torch.tensor([1.0])})

    def test_call_rejects_empty_dict(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat({})

    def test_call_rejects_wrong_dict_key(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat({"var": torch.tensor([0.0])})

    def test_call_rejects_unexpected_keyword(self):
        mat = ScalarMatrix(3)
        with pytest.raises(TypeError):
            mat(theta=torch.tensor([0.0]))

    @pytest.mark.parametrize("length", [0, 2])
    def test_grad_rejects_wrong_length(self, length):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat.grad(torch.zeros(length))

    def test_grad_rejects_non_tensor(self):
        mat = ScalarMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.grad(0.5)

    def test_grad_rejects_non_1d_tensor(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat.grad(torch.zeros(1, 1))

    @pytest.mark.parametrize("length", [0, 2])
    def test_manual_grad_rejects_wrong_length(self, length):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat.manual_grad(torch.zeros(length))

    @pytest.mark.parametrize("length", [0, 2])
    def test_auto_grad_rejects_wrong_length(self, length):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat.auto_grad(torch.zeros(length))

    @pytest.mark.parametrize("length", [0, 2])
    def test_trans_grad_rejects_wrong_length(self, length):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat.trans_grad(torch.zeros(length))

    @pytest.mark.parametrize("length", [0, 2])
    def test_build_params_rejects_wrong_length(self, length):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat.build_params(torch.zeros(length))

    def test_fixed_matrix_rejects_non_empty_free_params(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExpPow2()},
        })
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.tensor([1.0]))
        with pytest.raises(ValueError, match="length 0"):
            mat.grad(torch.tensor([1.0]))

    def test_call_on_accelerator_validates_length(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="length 1"):
            mat(torch.zeros(2, device=device))


class TestScalarMatrixDtypeAndDevice:
    """The input tensor is the single source of dtype and device."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([0.5], dtype=dtype))
        assert result.dtype == dtype
        assert result.diag().dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_build_params_follows_input_dtype(self, dtype):
        mat = ScalarMatrix(3)
        assert mat.build_params(torch.tensor([0.5], dtype=dtype)).dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_grad_follows_input_dtype(self, dtype):
        mat = ScalarMatrix(3)
        grad, _ = mat.grad(torch.tensor([0.5], dtype=dtype))
        assert grad.dtype == dtype

    def test_trans_grad_follows_input_dtype(self):
        mat = ScalarMatrix(3)
        assert mat.trans_grad(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_int_input_is_promoted_by_the_transform(self):
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([0], dtype=torch.int64))
        assert result.dtype == torch.float32
        assert torch.equal(result, torch.eye(3))

    def test_default_dtype_when_omitted(self):
        mat = ScalarMatrix(3)
        assert mat().dtype == torch.get_default_dtype()

    def test_default_dtype_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = ScalarMatrix(3)
            assert mat().dtype == torch.float64
            assert mat.build_params().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_cast_call_does_not_change_later_default_calls(self):
        mat = ScalarMatrix(3)
        mat(torch.tensor([0.5], dtype=torch.float64))
        assert mat().dtype == torch.get_default_dtype()

    def test_input_dtype_wins_over_default_dtype(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = ScalarMatrix(3)
            assert mat(torch.tensor([0.5], dtype=torch.float32)).dtype == torch.float32
        finally:
            torch.set_default_dtype(original)

    def test_default_device_when_omitted(self):
        mat = ScalarMatrix(3)
        assert mat().device == torch.get_default_device()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([0.5], device=device))
        assert result.device.type == device
        assert torch.allclose(result.cpu(), torch.exp(torch.tensor(1.0)) * torch.eye(3))

    def test_build_params_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        assert mat.build_params(torch.tensor([0.5], device=device)).device.type == device

    def test_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        grad, _ = mat.grad(torch.tensor([0.5], device=device))
        assert grad.device.type == device

    def test_trans_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        assert mat.trans_grad(torch.tensor([0.5], device=device)).device.type == device

    def test_manual_grad_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        grad, _ = mat.manual_grad(torch.tensor([0.5], device=device))
        assert grad.device.type == device

    def test_accelerator_call_does_not_move_the_default_call(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        mat(torch.tensor([0.5], device=device))
        assert mat().device == torch.get_default_device()

    def test_follows_dtype_and_device_together(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = ScalarMatrix(3)
        result = mat(torch.tensor([0.5], dtype=torch.float16, device=device))
        assert result.dtype == torch.float16
        assert result.device.type == device

    def test_rejects_float64_on_mps(self):
        if not torch.backends.mps.is_available():
            pytest.skip("requires MPS")
        mat = ScalarMatrix(3)
        with pytest.raises(TypeError):
            mat(torch.tensor([0.5], dtype=torch.float64, device="mps"))


class TestScalarMatrixBuildParams:
    """``build_params`` composition on a single-parameter matrix."""

    def test_defaults_to_default_value(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat.build_params(), torch.tensor([1.0]))

    def test_applies_transform(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat.build_params(torch.tensor([0.5])), torch.exp(torch.tensor(1.0)).reshape(1))

    def test_trans_false_returns_raw_parameter(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat.build_params(torch.tensor([0.5]), trans=False), torch.tensor([0.5]))

    def test_include_fixed_false_keeps_free_params(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat.build_params(torch.tensor([0.5]), include_fixed=False), torch.exp(torch.tensor(1.0)).reshape(1))

    def test_include_fixed_false_trans_false_returns_raw_free_params(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat.build_params(torch.tensor([0.5]), include_fixed=False, trans=False), torch.tensor([0.5]))

    def test_output_length_is_one(self):
        mat = ScalarMatrix(3)
        assert mat.build_params(torch.tensor([0.5])).shape == (1,)

    def test_dict_output(self):
        mat = ScalarMatrix(3)
        result = mat.build_params(torch.tensor([0.5]), out_format="dict")
        assert isinstance(result, dict)
        assert list(result.keys()) == ["sigma^2"]
        assert result["sigma^2"].shape == (1,)
        assert torch.allclose(result["sigma^2"], torch.exp(torch.tensor(1.0)).reshape(1))

    def test_dict_output_matches_tensor_output(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        assert torch.equal(mat.build_params(free_params, out_format="dict")["sigma^2"], mat.build_params(free_params))

    def test_dict_input(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat.build_params({"sigma^2": torch.tensor([0.5])}), mat.build_params(torch.tensor([0.5])))

    def test_dict_input_with_flags(self):
        mat = ScalarMatrix(3)
        result = mat.build_params({"sigma^2": torch.tensor([0.5])}, include_fixed=False, trans=False)
        assert torch.equal(result, torch.tensor([0.5]))

    @pytest.mark.parametrize("out_format", ["matrix", "tensor ", "DICT"])
    def test_rejects_bad_out_format(self, out_format):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="out_format"):
            mat.build_params(torch.tensor([0.5]), out_format=out_format)

    def test_rejects_bad_out_format_on_empty_output(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExpPow2()},
        })
        with pytest.raises(ValueError, match="out_format"):
            mat.build_params(include_fixed=False, out_format="matrix")

    def test_rejects_dict_with_extra_key(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({"sigma^2": torch.tensor([0.5]), "x": torch.tensor([1.0])})

    def test_rejects_empty_dict(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.build_params({})

    def test_result_is_transformed_value_not_raw(self):
        mat = ScalarMatrix(3)
        assert not torch.allclose(mat.build_params(torch.tensor([0.5])), torch.tensor([0.5]))


class TestScalarMatrixGrad:
    """Manual and automatic gradients of the single variance parameter."""

    def test_manual_grad_shape_and_names(self):
        mat = ScalarMatrix(3)
        grad, grad_names = mat.manual_grad(torch.tensor([0.5]))
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["sigma^2"]

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, -2.0])
    def test_manual_grad_is_trans_grad_times_eye(self, theta):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([theta])
        grad, _ = mat.manual_grad(free_params)
        expected = 2.0 * torch.exp(2.0 * torch.tensor(theta)) * torch.eye(3)
        assert torch.allclose(grad[0], expected)

    def test_manual_grad_is_diagonal_only(self):
        mat = ScalarMatrix(3)
        grad, _ = mat.manual_grad(torch.tensor([0.5]))
        assert (grad[0][~torch.eye(3, dtype=torch.bool)] == 0.0).all()

    def test_manual_grad_without_arguments(self):
        mat = ScalarMatrix(3)
        grad, grad_names = mat.manual_grad()
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.0]))[0])
        assert grad_names == ["sigma^2"]

    def test_manual_grad_accepts_dict(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat.manual_grad({"sigma^2": torch.tensor([0.5])})[0], mat.manual_grad(torch.tensor([0.5]))[0])

    def test_auto_grad_shape_and_names(self):
        mat = ScalarMatrix(3)
        grad, grad_names = mat.auto_grad(torch.tensor([0.5]))
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["sigma^2"]

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, -2.0])
    def test_manual_grad_matches_auto_grad(self, theta):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([theta])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    @pytest.mark.parametrize("method", ["jacfwd", "jacrev", "jacobian"])
    def test_auto_grad_methods_agree(self, method):
        mat = ScalarMatrix(3)
        mat.jacobian_method = method
        grad, _ = mat.auto_grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_auto_grad_jacrev_chunked(self):
        mat = ScalarMatrix(3)
        mat.jacobian_method = "jacrev"
        mat.jacobian_chunk_size = 1
        grad, _ = mat.auto_grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_auto_grad_rejects_unknown_method(self):
        mat = ScalarMatrix(3)
        mat.jacobian_method = "bogus"
        with pytest.raises(ValueError, match="Unknown Jacobian method"):
            mat.auto_grad(torch.tensor([0.5]))

    def test_auto_grad_without_arguments(self):
        mat = ScalarMatrix(3)
        grad, _ = mat.auto_grad()
        assert torch.allclose(grad, mat.auto_grad(torch.tensor([0.0]))[0])

    def test_grad_default_mode_uses_manual(self):
        mat = ScalarMatrix(3)
        grad, _ = mat.grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_grad_mode_manual(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "manual"
        grad, _ = mat.grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.manual_grad(torch.tensor([0.5]))[0])

    def test_grad_mode_auto(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "auto"
        grad, _ = mat.grad(torch.tensor([0.5]))
        assert torch.allclose(grad, mat.auto_grad(torch.tensor([0.5]))[0])

    def test_grad_mode_unknown_raises(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "bogus"
        with pytest.raises(RuntimeError, match="Unknown grad mode"):
            mat.grad(torch.tensor([0.5]))

    def test_grad_without_arguments(self):
        mat = ScalarMatrix(3)
        grad, grad_names = mat.grad()
        assert torch.allclose(grad, mat.grad(torch.tensor([0.0]))[0])
        assert grad_names == ["sigma^2"]

    def test_grad_accepts_dict(self):
        mat = ScalarMatrix(3)
        assert torch.equal(mat.grad({"sigma^2": torch.tensor([0.5])})[0], mat.grad(torch.tensor([0.5]))[0])

    def test_grad_is_never_constant_in_theta(self):
        mat = ScalarMatrix(3)
        first, _ = mat.grad(torch.tensor([0.0]))
        second, _ = mat.grad(torch.tensor([1.0]))
        assert not torch.allclose(first, second)

    def test_grad_names_follow_custom_param_name(self):
        mat = ScalarMatrix(3, param_specs={"var": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()}})
        assert mat.grad(torch.tensor([0.5]))[1] == ["var"]


class TestScalarMatrixIntermediates:
    """The intermediate cache is keyed by the built parameter values."""

    def built(self, mat, theta=0.5):
        return mat.build_params(torch.tensor([theta]))

    def test_get_before_set_returns_none(self):
        mat = ScalarMatrix(3)
        assert mat.get_intermediates(self.built(mat)) is None

    def test_set_then_get_returns_value(self):
        mat = ScalarMatrix(3)
        params = self.built(mat)
        assert mat.set_intermediates(params, {"tag": "cached"}) is None
        assert mat.get_intermediates(params) == {"tag": "cached"}

    def test_cache_is_invalidated_by_different_params(self):
        mat = ScalarMatrix(3)
        mat.set_intermediates(self.built(mat, 0.5), "cached")
        assert mat.get_intermediates(self.built(mat, 1.5)) is None

    def test_cache_is_invalidated_by_dtype(self):
        mat = ScalarMatrix(3)
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
        mat = ScalarMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        assert mat.get_intermediates(params.to(device)) is None

    def test_stored_key_is_copied(self):
        mat = ScalarMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        params[0] = 99.0
        assert mat.get_intermediates(self.built(mat)) == "cached"

    def test_reset_clears_cache(self):
        mat = ScalarMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.reset_intermediates()
        assert mat.get_intermediates(params) is None

    def test_set_with_empty_params_is_a_noop(self):
        mat = ScalarMatrix(3)
        assert mat.set_intermediates(torch.tensor([]), "cached") is None
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_get_with_empty_params_returns_none(self):
        mat = ScalarMatrix(3)
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_overwriting_replaces_the_entry(self):
        mat = ScalarMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "first")
        mat.set_intermediates(params, "second")
        assert mat.get_intermediates(params) == "second"

    def test_auto_grad_clears_cache(self):
        mat = ScalarMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.auto_grad(torch.tensor([0.5]))
        assert mat.get_intermediates(params) is None

    def test_manual_grad_preserves_cache(self):
        mat = ScalarMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat.manual_grad(torch.tensor([0.5]))
        assert mat.get_intermediates(params) == "cached"

    def test_call_preserves_cache(self):
        mat = ScalarMatrix(3)
        params = self.built(mat)
        mat.set_intermediates(params, "cached")
        mat(torch.tensor([0.5]))
        assert mat.get_intermediates(params) == "cached"

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_set_validates_params(self, params, error):
        mat = ScalarMatrix(3)
        with pytest.raises(error):
            mat.set_intermediates(params, "cached")

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_get_validates_params(self, params, error):
        mat = ScalarMatrix(3)
        with pytest.raises(error):
            mat.get_intermediates(params)


class TestScalarMatrixRepr:
    def test_contains_class_name(self):
        assert "ScalarMatrix" in repr(ScalarMatrix(3))

    def test_contains_shape(self):
        assert "(3, 3)" in repr(ScalarMatrix(3))

    def test_contains_param_name(self):
        assert "sigma^2" in repr(ScalarMatrix(3))

    def test_contains_transform(self):
        assert "TransformExpPow2" in repr(ScalarMatrix(3))

    def test_is_single_line(self):
        assert "\n" not in repr(ScalarMatrix(3))

    def test_repr_dict(self):
        mat = ScalarMatrix(3)
        assert mat.repr_dict["shape"] == (3, 3)
        assert mat.repr_dict["param_specs"] is mat.param_specs

import torch
import pytest
from torch_openreml.covariance import (
    Adapter,
    AR1Matrix,
    BlockDiagonal,
    CompoundSymmetricMatrix,
    DiagonalMatrix,
    EqualEntryMatrix,
    EquicorrelationMatrix,
    IdentityMatrix,
    LowerTriangularMatrix,
    ScalarMatrix,
    Sum,
    UnconstrainedMatrix,
)
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    Transform,
    TransformChain,
    TransformExp,
    TransformExpPow2,
    TransformIdentity,
    TransformScaleShift,
    TransformSigmoid,
)


# --------------------------------------------------------------------------- #
# Test vehicles and helpers
# --------------------------------------------------------------------------- #

class MinimalMatrix(Matrix):
    """Minimal concrete Matrix subclass for testing the ABC directly.

    ``__call__`` returns the identity scaled by the first free parameter, which
    is enough to exercise the base class while keeping ``manual_grad`` checkable
    against ``auto_grad``.
    """

    def __call__(self, free_params=None):
        params = self.build_params(free_params)
        n = self.shape[0]
        scale = params[0] if len(params) else 1.0
        return torch.eye(n, dtype=params.dtype, device=params.device) * scale

    def manual_grad(self, free_params=None):
        params = self.build_params(free_params, include_fixed=False, trans=False)
        if len(params) == 0:
            return None, []
        n = self.shape[0]
        eye = torch.eye(n, dtype=params.dtype, device=params.device)
        grad = torch.zeros(self.num_free_params, n, n, dtype=params.dtype, device=params.device)
        grad[0] = eye * self.trans_grad(params)[0]
        return grad, self.free_param_names


class NoManualGradMatrix(Matrix):
    """A Matrix that deliberately leaves ``manual_grad`` to the base class."""

    def __init__(self):
        super().__init__((2, 2), {
            "p": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExp()}
        })

    def __call__(self, free_params=None):
        params = self.build_params(free_params)
        return torch.eye(2, dtype=params.dtype, device=params.device) * params[0]


class MultiLineMatrix(Matrix):
    """A Matrix with the multi-line repr and a controllable ``repr_dict``."""

    _repr_single_line = False

    def __init__(self, entries=None, shape=(2, 2), param_specs=None):
        super().__init__(shape, {} if param_specs is None else param_specs)
        self._entries = {} if entries is None else entries

    def __call__(self, free_params=None):
        return torch.eye(self.shape[0])

    @property
    def repr_dict(self):
        return self._entries


class MultilineValue:
    """A plain (non-Matrix, non-dict, non-tensor) value whose repr spans lines."""

    def __repr__(self):
        return "line1\nline2"


class RecordingTransform(TransformIdentity):
    """An identity transform that records the inputs it was called with.

    Used to observe which branch of the fast/slow transform dispatch in
    ``build_params`` and ``trans_grad`` was taken: the fast path calls a single
    transform once on the whole parameter vector, the slow path calls every
    transform on its own one-element slice.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, x):
        self.calls.append(x.clone())
        return x


class PassthroughTransform(Transform):
    """A minimal custom Transform, to check that subclass instances are accepted."""

    def __call__(self, x):
        return x

    def inverse(self, x):
        return x

    def grad(self, x):
        return torch.ones_like(x)


def _spec(**overrides):
    """A well-formed parameter specification, with fields overridable."""
    spec = {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformIdentity()}
    spec.update(overrides)
    return spec


def _mixed_matrix():
    """``DiagonalMatrix(3)`` with the middle parameter fixed at a distinctive default."""
    mat = DiagonalMatrix(3)
    mat.param_specs["sigma^2_1"]["fixed"] = True
    mat.param_specs["sigma^2_1"]["default"] = torch.tensor([7.5])
    return mat


def _two_param_matrix(trans_a, trans_b):
    """A two-parameter DiagonalMatrix, with one transform per parameter."""
    return DiagonalMatrix(2, param_specs={
        "a": {"fixed": False, "default": torch.tensor([0.0]), "trans": trans_a},
        "b": {"fixed": False, "default": torch.tensor([0.0]), "trans": trans_b},
    })


# --------------------------------------------------------------------------- #
# A. Constructor
# --------------------------------------------------------------------------- #

class TestMatrixConstructor:
    """Tests for Matrix.__init__ validation."""

    def test_grad_mode_default(self):
        mat = ScalarMatrix(3)
        assert mat.grad_mode == "default"

    def test_jacobian_method_default(self):
        mat = ScalarMatrix(3)
        assert mat.jacobian_method == "jacfwd"

    def test_jacobian_chunk_size_default(self):
        mat = ScalarMatrix(3)
        assert mat.jacobian_chunk_size is None

    def test_shape_stored_as_tuple(self):
        mat = MinimalMatrix([2, 3], {})
        assert mat.shape == (2, 3)
        assert isinstance(mat.shape, tuple)

    def test_intermediates_reset_on_init(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        assert mat.get_intermediates(params) is None

    def test_zero_param_matrix_is_constructible(self):
        mat = MinimalMatrix((2, 2), {})
        assert mat.num_params == 0
        assert mat.num_free_params == 0

    def test_shape_checked_before_param_specs(self):
        """A bad shape and bad specs report the shape error, not the specs error."""
        with pytest.raises(TypeError, match="must be a list, a tuple or a torch.Size"):
            MinimalMatrix(42, "not_a_dict")


# --------------------------------------------------------------------------- #
# B. _check_shape
# --------------------------------------------------------------------------- #

class TestCheckShape:
    """``_check_shape`` accepts a list, a tuple or a torch.Size, and rejects the rest."""

    @pytest.mark.parametrize("shape,expected", [
        ([2, 3], (2, 3)),
        ((2, 3), (2, 3)),
        (torch.Size([2, 3]), (2, 3)),
    ])
    def test_accepts_sequence_types(self, shape, expected):
        assert MinimalMatrix(shape, {}).shape == expected

    def test_accepts_none(self):
        assert MinimalMatrix(None, {}).shape == ()

    @pytest.mark.parametrize("shape", [[], ()])
    def test_empty_sequence_becomes_empty_tuple(self, shape):
        """``tuple(shape or ())`` treats an empty sequence as absent."""
        assert MinimalMatrix(shape, {}).shape == ()

    @pytest.mark.parametrize("shape", [42, "ab"])
    def test_rejects_non_sequence(self, shape):
        with pytest.raises(TypeError, match="must be a list, a tuple or a torch.Size"):
            MinimalMatrix(shape, {})

    @pytest.mark.parametrize("shape", [[0, 2], [-1, 2]])
    def test_rejects_non_positive(self, shape):
        with pytest.raises(ValueError, match="must be positive int"):
            MinimalMatrix(shape, {})

    def test_rejects_float_element_as_value_error(self):
        """A float fails the ``isinstance(p, int)`` test, reported as a ValueError."""
        with pytest.raises(ValueError, match="must be positive int"):
            MinimalMatrix([2.0, 2], {})

    def test_rejects_nested_sequence(self):
        with pytest.raises(ValueError, match="must be positive int"):
            MinimalMatrix([[2], 2], {})

    def test_bool_element_is_accepted(self):
        """``bool`` is an ``int`` subclass, so it passes the element check."""
        assert MinimalMatrix([True, 2], {}).shape == (True, 2)


# --------------------------------------------------------------------------- #
# C. _check_param_specs
# --------------------------------------------------------------------------- #

class TestCheckParamSpecs:
    """``_check_param_specs`` validates the shape of the specification dictionary."""

    def test_accepts_empty_dict(self):
        mat = MinimalMatrix((2, 2), {})
        assert mat.param_specs == {}

    def test_accepts_custom_transform_subclass(self):
        mat = MinimalMatrix((2, 2), {"p": _spec(trans=PassthroughTransform())})
        assert isinstance(mat.param_trans["p"], PassthroughTransform)

    def test_accepts_transform_chain(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        mat = MinimalMatrix((2, 2), {"p": _spec(trans=chain)})
        assert mat.param_trans["p"] is chain

    def test_accepts_slash_in_parameter_name(self):
        """Namespacing is the operators' concern; the base class allows any str key."""
        mat = MinimalMatrix((2, 2), {"a/b": _spec()})
        assert mat.param_names == ["a/b"]

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError, match="param_sepc"):
            MinimalMatrix((2, 2), "not_a_dict")

    def test_rejects_non_string_name(self):
        with pytest.raises(TypeError, match="Parameter name must be a str, got int"):
            MinimalMatrix((2, 2), {1: _spec()})

    def test_rejects_non_dict_spec(self):
        with pytest.raises(TypeError, match="specification must be a dict, got str"):
            MinimalMatrix((2, 2), {"p": "not_a_spec"})

    def test_rejects_missing_field(self):
        spec = _spec()
        del spec["default"]
        with pytest.raises(TypeError, match="fields must be 'fixed', 'default', and 'trans'"):
            MinimalMatrix((2, 2), {"p": spec})

    def test_rejects_extra_field(self):
        spec = _spec(extra="unexpected")
        with pytest.raises(TypeError, match="fields must be 'fixed', 'default', and 'trans'"):
            MinimalMatrix((2, 2), {"p": spec})

    @pytest.mark.parametrize("fixed", [1, "yes", None])
    def test_rejects_non_bool_fixed(self, fixed):
        with pytest.raises(TypeError, match="'fixed' must be a bool"):
            MinimalMatrix((2, 2), {"p": _spec(fixed=fixed)})

    def test_rejects_non_tensor_default(self):
        with pytest.raises(TypeError, match="'default' must be a torch.Tensor, got list"):
            MinimalMatrix((2, 2), {"p": _spec(default=[0.0])})

    def test_rejects_non_1d_default(self):
        """A shape problem, but reported as a TypeError by the specification check."""
        with pytest.raises(TypeError, match="'default' must be a 1D torch.Tensor"):
            MinimalMatrix((2, 2), {"p": _spec(default=torch.tensor([[0.0]]))})

    def test_rejects_non_transform(self):
        with pytest.raises(TypeError, match="'trans' must be a Transform, got str"):
            MinimalMatrix((2, 2), {"p": _spec(trans="not_a_transform")})


# --------------------------------------------------------------------------- #
# D. _check_param_tensor
# --------------------------------------------------------------------------- #

class TestCheckParamTensor:
    """Tests for _check_param_tensor length validation."""

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="Torch tensor"):
            ScalarMatrix(3)._check_param_tensor([0.0])

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            ScalarMatrix(3)._check_param_tensor(torch.zeros(2, 2))

    def test_rejects_zero_dimensional(self):
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            ScalarMatrix(3)._check_param_tensor(torch.tensor(1.0))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="must have length 1, got 2"):
            ScalarMatrix(3)._check_param_tensor(torch.tensor([0.0, 1.0]), length=1)

    def test_zero_length_is_enforced(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        assert mat.num_free_params == 0
        with pytest.raises(ValueError, match="must have length 0, got 1"):
            mat._check_param_tensor(torch.tensor([0.0]), length=0)
        with pytest.raises(ValueError, match="must have length 0, got 5"):
            mat._check_param_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), length=0)

    def test_zero_length_accepts_empty_tensor(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        device, dtype = mat._check_param_tensor(torch.tensor([]), length=0)
        assert device == torch.device("cpu")
        assert dtype == torch.float32

    def test_no_length_skips_check(self):
        device, dtype = ScalarMatrix(3)._check_param_tensor(torch.tensor([]))
        assert device == torch.device("cpu")
        assert dtype == torch.float32

    def test_returns_input_device_and_dtype(self):
        device, dtype = ScalarMatrix(3)._check_param_tensor(torch.tensor([0.0], dtype=torch.float64))
        assert device == torch.device("cpu")
        assert dtype == torch.float64

    def test_build_params_rejects_extra_params_on_fixed_matrix(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        with pytest.raises(ValueError, match="must have length 0, got 1"):
            mat.build_params(torch.tensor([9.9]))

    def test_build_params_accepts_empty_tensor_on_fixed_matrix(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        assert mat.build_params(torch.tensor([])).shape == (1,)


# --------------------------------------------------------------------------- #
# E. get_default_dtype_device
# --------------------------------------------------------------------------- #

class TestGetDefaultDtypeDevice:
    """Tests for get_default_dtype_device."""

    def test_returns_defaults_dtype_device(self):
        mat = DiagonalMatrix(2, param_specs={
            "a": {"fixed": True, "default": torch.tensor([1.0], dtype=torch.float64), "trans": TransformIdentity()},
            "b": {"fixed": True, "default": torch.tensor([2.0], dtype=torch.float64), "trans": TransformIdentity()},
        })
        device, dtype = mat.get_default_dtype_device()
        assert device == torch.device("cpu")
        assert dtype == torch.float64

    def test_no_params_falls_back_to_torch_defaults(self):
        mat = IdentityMatrix(3)
        assert mat.num_params == 0
        device, dtype = mat.get_default_dtype_device()
        assert device == torch.get_default_device()
        assert dtype == torch.get_default_dtype()

    def test_no_params_tracks_torch_default_dtype(self):
        mat = IdentityMatrix(3)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            _, dtype = mat.get_default_dtype_device()
            assert dtype == torch.float64
        finally:
            torch.set_default_dtype(default_dtype)

    def test_free_params_are_checked_too(self):
        mat = DiagonalMatrix(2, param_specs={
            "a": {"fixed": False, "default": torch.tensor([1.0], dtype=torch.float32), "trans": TransformIdentity()},
            "b": {"fixed": False, "default": torch.tensor([2.0], dtype=torch.float64), "trans": TransformIdentity()},
        })
        with pytest.raises(ValueError, match="same dtype and device"):
            mat.get_default_dtype_device()

    def test_inconsistent_defaults_raise(self):
        mat = DiagonalMatrix(2, param_specs={
            "a": {"fixed": True, "default": torch.tensor([1.0], dtype=torch.float32), "trans": TransformIdentity()},
            "b": {"fixed": True, "default": torch.tensor([2.0], dtype=torch.float64), "trans": TransformIdentity()},
        })
        with pytest.raises(ValueError, match="same dtype and device"):
            mat.get_default_dtype_device()

    def test_operator_uses_operand_defaults(self):
        op = Sum(ScalarMatrix(3), IdentityMatrix(3))
        device, dtype = op.get_default_dtype_device()
        assert device == torch.device("cpu")
        assert dtype == torch.float32


# --------------------------------------------------------------------------- #
# F. build_params
# --------------------------------------------------------------------------- #

class TestBuildParams:
    """Tests for Matrix.build_params."""

    # -- input sources -------------------------------------------------------

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

    def test_defaults_used_when_omitted(self):
        """``free_params=None`` falls back to the free parameter defaults, transformed."""
        result = ScalarMatrix(3).build_params()
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_defaults_used_when_omitted_no_params(self):
        assert IdentityMatrix(3).build_params().shape == (0,)

    def test_raises_on_bad_out_format(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="out_format"):
            mat.build_params(torch.tensor([0.0]), out_format="invalid")

    def test_raises_on_bad_out_format_on_empty_matrix(self):
        """The empty-parameter branch raises before reaching the main format logic."""
        with pytest.raises(ValueError, match="out_format"):
            IdentityMatrix(3).build_params(torch.tensor([]), out_format="invalid")

    def test_empty_params(self):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]))
        assert torch.equal(result, torch.tensor([]))

    def test_empty_params_dict_format(self):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]), out_format="dict")
        assert result == {}

    # -- scatter of free values around fixed defaults ------------------------

    def test_scatter_matches_free_param_index(self):
        mat = _mixed_matrix()
        free_params = torch.tensor([1.0, 3.0])
        result = mat.build_params(free_params, trans=False)

        for index, value in zip(mat.free_param_index, free_params):
            assert result[index] == value
        for index in mat.fixed_param_index:
            assert result[index] == mat.param_defaults[mat.param_names[index]]

    def test_fixed_slot_keeps_its_default(self):
        result = _mixed_matrix().build_params(torch.tensor([1.0, 3.0]), trans=False)
        assert torch.equal(result, torch.tensor([1.0, 7.5, 3.0]))

    def test_fixed_defaults_are_transformed_too(self):
        """Transforms are applied to the merged vector, fixed entries included."""
        result = _mixed_matrix().build_params(torch.tensor([1.0, 3.0]))
        expected = torch.tensor([1.0, 7.5, 3.0]).mul(2).exp()
        assert torch.allclose(result, expected)

    def test_trans_false_is_identity_for_free_params(self):
        mat = ScalarMatrix(3)
        result = mat.build_params(torch.tensor([0.5]), trans=False)
        assert torch.allclose(result, torch.tensor([0.5]))

    def test_include_fixed_false_returns_only_free_params(self):
        result = _mixed_matrix().build_params(torch.tensor([1.0, 3.0]), include_fixed=False, trans=False)
        assert torch.equal(result, torch.tensor([1.0, 3.0]))

    def test_include_fixed_false_with_zero_free_params_returns_empty(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        result = mat.build_params(torch.tensor([]), include_fixed=False)
        assert result.numel() == 0

    def test_all_fixed_matrix_returns_defaults(self):
        mat = _mixed_matrix()
        mat.param_specs["sigma^2_0"]["fixed"] = True
        mat.param_specs["sigma^2_2"]["fixed"] = True
        assert mat.num_free_params == 0
        assert mat.build_params(torch.tensor([]), trans=False).shape == (3,)

    # -- transform dispatch: fast path vs slow path --------------------------

    @pytest.mark.parametrize("trans_a,trans_b", [
        pytest.param(TransformExpPow2(), TransformExpPow2(), id="fast-same-type-equal-dict"),
        pytest.param(TransformSigmoid(), TransformSigmoid(), id="fast-stateless"),
        pytest.param(TransformExp(), TransformExpPow2(), id="slow-mixed-types"),
        pytest.param(TransformScaleShift(2.0), TransformScaleShift(3.0), id="slow-same-type-different-dict"),
    ])
    def test_transforms_match_elementwise_expectation(self, trans_a, trans_b):
        mat = _two_param_matrix(trans_a, trans_b)
        free_params = torch.tensor([0.3, -0.4])
        expected = torch.cat([trans_a(free_params[0:1]), trans_b(free_params[1:2])])
        assert torch.allclose(mat.build_params(free_params), expected)

    def test_fast_path_calls_one_transform_on_whole_vector(self):
        trans_a, trans_b = RecordingTransform(), RecordingTransform()
        mat = _two_param_matrix(trans_a, trans_b)

        mat.build_params(torch.tensor([0.3, -0.4]))

        assert len(trans_a.calls) == 1
        assert trans_a.calls[0].shape == (2,)
        assert trans_b.calls == []

    def test_slow_path_calls_each_transform_on_its_own_element(self):
        """Differing types force the per-element path, one slice per transform."""
        trans_a, trans_b = RecordingTransform(), TransformExpPow2()
        mat = _two_param_matrix(trans_a, trans_b)

        mat.build_params(torch.tensor([0.3, -0.4]))

        assert len(trans_a.calls) == 1
        assert trans_a.calls[0].shape == (1,)

    def test_slow_path_triggered_by_differing_dicts(self):
        """Same type, different ``__dict__``: the fast-path comparison must fail."""
        trans_a, trans_b = TransformScaleShift(2.0, 0.0), TransformScaleShift(2.0, 1.0)
        mat = _two_param_matrix(trans_a, trans_b)
        assert trans_a.__dict__ != trans_b.__dict__

        result = mat.build_params(torch.tensor([1.0, 1.0]))

        assert torch.allclose(result, torch.tensor([2.0, 3.0]))

    # -- out_format ----------------------------------------------------------

    def test_dict_values_are_one_dimensional(self):
        mat = ScalarMatrix(3)
        result = mat.build_params(torch.tensor([0.5]), out_format="dict")
        assert result["sigma^2"].shape == (1,)

    def test_dict_keys_include_fixed_params(self):
        mat = _mixed_matrix()
        result = mat.build_params(torch.tensor([1.0, 3.0]), out_format="dict")
        assert list(result.keys()) == mat.param_names

    def test_dict_keys_are_free_only_when_include_fixed_false(self):
        mat = _mixed_matrix()
        result = mat.build_params(torch.tensor([1.0, 3.0]), include_fixed=False, out_format="dict")
        assert list(result.keys()) == mat.free_param_names

    # -- value fidelity ------------------------------------------------------

    def test_follows_input_dtype(self):
        result = ScalarMatrix(3).build_params(torch.tensor([0.5], dtype=torch.float64))
        assert result.dtype == torch.float64

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        result = ScalarMatrix(3).build_params(torch.tensor([0.5], device=device))
        assert result.device.type == device

    def test_wrong_length_raises(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError):
            mat.build_params(torch.tensor([0.0, 1.0]))

    # -- dict input ----------------------------------------------------------

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

    def test_dict_input_matches_tensor_input(self):
        """The two input forms are interchangeable for a well-formed dict."""
        mat = DiagonalMatrix(2)
        as_tensor = mat.build_params(torch.tensor([0.1, 0.2]), trans=False)
        as_dict = mat.build_params(
            {"sigma^2_0": torch.tensor([0.1]), "sigma^2_1": torch.tensor([0.2])}, trans=False
        )
        assert torch.equal(as_tensor, as_dict)

    def test_dict_input_missing_one_of_two_keys(self):
        with pytest.raises(ValueError, match="Missing free parameters"):
            DiagonalMatrix(2).build_params({"sigma^2_0": torch.tensor([0.1])})

    def test_dict_input_free_keys_on_partially_fixed_matrix(self):
        """Only the free names are accepted; the fixed slot takes its default."""
        mat = _mixed_matrix()
        result = mat.build_params(
            {"sigma^2_0": torch.tensor([1.0]), "sigma^2_2": torch.tensor([3.0])}, trans=False
        )
        assert torch.equal(result, torch.tensor([1.0, 7.5, 3.0]))

    def test_dict_input_all_param_names_on_partially_fixed_matrix(self):
        """Naming a fixed parameter is rejected, so "all the names" is not a valid input."""
        mat = _mixed_matrix()
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({
                "sigma^2_0": torch.tensor([1.0]),
                "sigma^2_1": torch.tensor([2.0]),
                "sigma^2_2": torch.tensor([3.0]),
            })

    def test_dict_input_correct_key_on_all_fixed_matrix(self):
        """Current behaviour, not desired: the name is "unexpected" although it is in
        ``param_names`` — it is simply not free. Pin the message so a future wording fix
        is deliberate.
        """
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({"sigma^2": torch.tensor([0.5])})

    def test_dict_input_non_1d_value_raises(self):
        """A 2D value is concatenated first and then caught by the rank check."""
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            ScalarMatrix(3).build_params({"sigma^2": torch.tensor([[0.5]])})

    def test_dict_input_zero_dim_value_raises_from_torch_cat(self):
        """Current behaviour, not desired: a 0-d value never reaches the rank check, and
        escapes as a bare ``RuntimeError`` from ``torch.cat`` rather than the ``ValueError``
        the docstring promises. ``torch.tensor(0.5)`` instead of ``torch.tensor([0.5])`` is
        the likeliest way a caller hits this.
        """
        with pytest.raises(RuntimeError, match="zero-dimensional tensor"):
            ScalarMatrix(3).build_params({"sigma^2": torch.tensor(0.5)})

    @pytest.mark.parametrize("value", [0.5, [0.5]])
    def test_dict_input_non_tensor_value_raises_attribute_error(self, value):
        """Current behaviour, not desired: ``_from_free_param_dict`` reads ``.device`` off
        each value, so a non-tensor leaks an ``AttributeError`` rather than the ``TypeError``
        the docstring promises for a non-tensor ``free_params``.
        """
        with pytest.raises(AttributeError, match="has no attribute 'device'"):
            ScalarMatrix(3).build_params({"sigma^2": value})

    # -- degenerate matrices: nothing to include ------------------------------

    @pytest.mark.parametrize("out_format", ["tensor", "dict"])
    def test_include_fixed_false_no_params(self, out_format):
        """A parameter-less matrix has no fixed entries to drop either way."""
        result = IdentityMatrix(3).build_params(
            torch.tensor([]), include_fixed=False, out_format=out_format
        )
        if out_format == "tensor":
            assert result.shape == (0,)
        else:
            assert result == {}

    @pytest.mark.parametrize("out_format", ["tensor", "dict"])
    def test_include_fixed_false_all_fixed(self, out_format):
        """An all-fixed matrix is empty on this path, not ``num_params`` long."""
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        result = mat.build_params(torch.tensor([]), include_fixed=False, out_format=out_format)
        if out_format == "tensor":
            assert result.shape == (0,)
        else:
            assert result == {}

    def test_include_fixed_false_defaults_when_omitted(self):
        """``free_params=None`` reaches the same empty result on both degenerate matrices."""
        all_fixed = ScalarMatrix(3)
        all_fixed.param_specs["sigma^2"]["fixed"] = True

        for mat in (IdentityMatrix(3), all_fixed):
            result = mat.build_params(include_fixed=False)
            assert result.shape == (0,)
            assert result.dtype == torch.get_default_dtype()

    def test_include_fixed_false_empty_follows_input_dtype(self):
        """The empty return is still built on the input's dtype, not the default one."""
        all_fixed = ScalarMatrix(3)
        all_fixed.param_specs["sigma^2"]["fixed"] = True

        for mat in (IdentityMatrix(3), all_fixed):
            result = mat.build_params(torch.tensor([], dtype=torch.float64), include_fixed=False)
            assert result.dtype == torch.float64

    def test_include_fixed_false_empty_ignores_trans(self):
        """There is no free transform to apply, so ``trans`` cannot change the result."""
        all_fixed = ScalarMatrix(3)
        all_fixed.param_specs["sigma^2"]["fixed"] = True

        for mat in (IdentityMatrix(3), all_fixed):
            assert torch.equal(
                mat.build_params(torch.tensor([]), include_fixed=False, trans=False),
                mat.build_params(torch.tensor([]), include_fixed=False, trans=True),
            )


# --------------------------------------------------------------------------- #
# G, H. Dict conversion
# --------------------------------------------------------------------------- #

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

    def test_from_free_param_dict_raises_on_mixed_dtype_values(self):
        mat = DiagonalMatrix(2)
        with pytest.raises(ValueError, match="same dtype and device"):
            mat._from_free_param_dict({
                "sigma^2_0": torch.tensor([1.0], dtype=torch.float32),
                "sigma^2_1": torch.tensor([2.0], dtype=torch.float64),
            })

    def test_from_free_param_dict_raises_on_mixed_device_values(self):
        mat = DiagonalMatrix(2)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        with pytest.raises(ValueError, match="same dtype and device"):
            mat._from_free_param_dict({
                "sigma^2_0": torch.tensor([1.0]),
                "sigma^2_1": torch.tensor([2.0], device=device),
            })

    def test_from_free_param_dict_accepts_consistent_values(self):
        mat = DiagonalMatrix(2)
        result = mat._from_free_param_dict({
            "sigma^2_0": torch.tensor([1.0], dtype=torch.float64),
            "sigma^2_1": torch.tensor([2.0], dtype=torch.float64),
        })
        assert result.dtype == torch.float64

    def test_from_free_param_dict_values_override_mixed_defaults(self):
        mat = DiagonalMatrix(2, param_specs={
            "a": {"fixed": False, "default": torch.tensor([1.0], dtype=torch.float32), "trans": TransformIdentity()},
            "b": {"fixed": False, "default": torch.tensor([2.0], dtype=torch.float64), "trans": TransformIdentity()},
        })
        result = mat._from_free_param_dict({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
        assert result.dtype == torch.float32

    def test_from_free_param_dict_empty_no_free_params(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        result = mat._from_free_param_dict({})
        assert torch.is_tensor(result)
        assert result.shape == (0,)

    def test_from_free_param_dict_empty_no_params(self):
        mat = IdentityMatrix(3)
        assert mat.num_params == 0
        result = mat._from_free_param_dict({})
        assert result.shape == (0,)
        assert result.dtype == torch.get_default_dtype()
        assert result.device == torch.get_default_device()

    def test_from_free_param_dict_empty_follows_fixed_defaults(self):
        mat = DiagonalMatrix(2, param_specs={
            "a": {"fixed": True, "default": torch.tensor([1.0], dtype=torch.float64), "trans": TransformIdentity()},
            "b": {"fixed": True, "default": torch.tensor([2.0], dtype=torch.float64), "trans": TransformIdentity()},
        })
        result = mat._from_free_param_dict({})
        assert result.shape == (0,)
        assert result.dtype == torch.float64
        assert result.device == torch.device("cpu")

    def test_from_free_param_dict_empty_uses_torch_default_when_no_defaults(self):
        mat = IdentityMatrix(3)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            result = mat._from_free_param_dict({})
            assert result.dtype == torch.float64
        finally:
            torch.set_default_dtype(default_dtype)

    def test_from_free_param_dict_empty_raises_on_inconsistent_defaults(self):
        mat = DiagonalMatrix(2, param_specs={
            "a": {"fixed": True, "default": torch.tensor([1.0], dtype=torch.float32), "trans": TransformIdentity()},
            "b": {"fixed": True, "default": torch.tensor([2.0], dtype=torch.float64), "trans": TransformIdentity()},
        })
        with pytest.raises(ValueError, match="same dtype and device"):
            mat._from_free_param_dict({})

    def test_from_free_param_dict_empty_still_validates_keys(self):
        with pytest.raises(ValueError, match="Missing"):
            ScalarMatrix(3)._from_free_param_dict({})
        with pytest.raises(ValueError, match="Unexpected"):
            IdentityMatrix(3)._from_free_param_dict({"extra": torch.tensor([1.0])})

    def test_to_free_param_dict_passthrough_dict(self):
        mat = ScalarMatrix(3)
        d = {"sigma^2": torch.tensor([0.5])}
        result = mat._to_free_param_dict(d)
        assert result is d

    def test_to_free_param_dict_from_tensor(self):
        mat = ScalarMatrix(3)
        result = mat._to_free_param_dict(torch.tensor([0.5]))
        assert "sigma^2" in result
        assert result["sigma^2"].shape == (1,)
        assert torch.equal(result["sigma^2"], torch.tensor([0.5]))

    def test_to_free_param_dict_wrong_length(self):
        with pytest.raises(ValueError, match="Expected 1 parameters, got 2"):
            ScalarMatrix(3)._to_free_param_dict(torch.tensor([0.5, 1.0]))

    def test_to_free_param_dict_skips_fixed_params(self):
        mat = DiagonalMatrix(3)
        mat.param_specs["sigma^2_0"]["fixed"] = True
        result = mat._to_free_param_dict(torch.tensor([0.5, 1.5]))
        assert list(result.keys()) == mat.free_param_names
        assert torch.equal(result["sigma^2_1"], torch.tensor([0.5]))
        assert torch.equal(result["sigma^2_2"], torch.tensor([1.5]))

    def test_to_free_param_dict_no_free_params(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        assert mat._to_free_param_dict(torch.tensor([])) == {}


# --------------------------------------------------------------------------- #
# I. trans_grad
# --------------------------------------------------------------------------- #

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

    def test_trans_grad_uses_defaults_when_omitted(self):
        mat = ScalarMatrix(3)
        assert torch.allclose(mat.trans_grad(), mat.trans_grad(torch.tensor([0.0])))

    def test_trans_grad_dict_input(self):
        mat = ScalarMatrix(3)
        tg = mat.trans_grad({"sigma^2": torch.tensor([0.0])})
        assert tg.numel() == 1

    def test_trans_grad_dict_matches_tensor(self):
        """A dict of free parameters means the same as the flat tensor, fixed names dropped."""
        mat = _mixed_matrix()
        free_params = torch.tensor([0.3, -0.4])
        assert torch.allclose(
            mat.trans_grad(mat._to_free_param_dict(free_params)),
            mat.trans_grad(free_params),
        )

    def test_trans_grad_multi_param(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.1, 0.2])
        tg = mat.trans_grad(free_params)
        assert tg.shape == (3,)

    @pytest.mark.parametrize("trans_a,trans_b", [
        pytest.param(TransformExpPow2(), TransformExpPow2(), id="fast-same-type-equal-dict"),
        pytest.param(TransformScaleShift(2.0), TransformScaleShift(3.0), id="slow-same-type-different-dict"),
        pytest.param(TransformExp(), TransformExpPow2(), id="slow-mixed-types"),
    ])
    def test_trans_grad_matches_elementwise_expectation(self, trans_a, trans_b):
        mat = _two_param_matrix(trans_a, trans_b)
        free_params = torch.tensor([0.3, -0.4])
        expected = torch.cat([trans_a.grad(free_params[0:1]), trans_b.grad(free_params[1:2])])
        assert torch.allclose(mat.trans_grad(free_params), expected)

    def test_trans_grad_excludes_fixed_params(self):
        mat = _mixed_matrix()
        tg = mat.trans_grad(torch.tensor([0.0, 0.0]))
        assert tg.shape == (mat.num_free_params,)

    def test_trans_grad_follows_input_dtype(self):
        mat = ScalarMatrix(3)
        tg = mat.trans_grad(torch.tensor([0.5], dtype=torch.float64))
        assert tg.dtype == torch.float64

    def test_trans_grad_no_params_returns_empty(self):
        """A parameter-less matrix has no free transforms to differentiate."""
        mat = IdentityMatrix(3)
        tg = mat.trans_grad()
        assert tg.shape == (0,)
        assert tg.numel() == 0

    def test_trans_grad_all_fixed_returns_empty(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        assert mat.num_free_params == 0
        tg = mat.trans_grad()
        assert tg.shape == (0,)
        assert tg.numel() == 0

    def test_trans_grad_empty_dict_input(self):
        mat = IdentityMatrix(3)
        assert mat.trans_grad({}).shape == (0,)

    def test_trans_grad_empty_follows_input_dtype(self):
        mat = IdentityMatrix(3)
        tg = mat.trans_grad(torch.tensor([], dtype=torch.float64))
        assert tg.dtype == torch.float64

    def test_trans_grad_empty_rejects_extra_params(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="must have length 0, got 2"):
            mat.trans_grad(torch.tensor([1.0, 2.0]))

    def test_trans_grad_wrong_length(self):
        mat = ScalarMatrix(3)
        with pytest.raises(ValueError, match="must have length 1, got 3"):
            mat.trans_grad(torch.tensor([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# J. auto_grad
# --------------------------------------------------------------------------- #

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

    def test_auto_grad_dict_matches_tensor(self):
        mat = ScalarMatrix(3)
        from_dict, dict_names = mat.auto_grad({"sigma^2": torch.tensor([0.5])})
        from_tensor, tensor_names = mat.auto_grad(torch.tensor([0.5]))
        assert dict_names == tensor_names
        assert torch.equal(from_dict, from_tensor)

    @pytest.mark.parametrize("method", ["jacfwd", "jacrev", "jacobian"])
    def test_jacobian_methods_agree(self, method):
        """Every method must produce the same axis order and the same values."""
        reference, _ = DiagonalMatrix(3).auto_grad(torch.tensor([0.1, 0.2, 0.3]))

        mat = DiagonalMatrix(3)
        mat.jacobian_method = method
        grad, grad_names = mat.auto_grad(torch.tensor([0.1, 0.2, 0.3]))

        assert grad.shape == (3, 3, 3)
        assert grad_names == mat.free_param_names
        assert torch.allclose(grad, reference)

    def test_jacobian_methods_agree_with_manual_grad(self):
        free_params = torch.tensor([0.1, 0.2, 0.3])
        manual, _ = DiagonalMatrix(3).manual_grad(free_params)
        for method in ("jacfwd", "jacrev", "jacobian"):
            mat = DiagonalMatrix(3)
            mat.jacobian_method = method
            grad, _ = mat.auto_grad(free_params)
            assert torch.allclose(grad, manual), method

    @pytest.mark.parametrize("chunk_size", [None, 1, 2])
    def test_jacrev_honours_chunk_size(self, chunk_size):
        expected, _ = DiagonalMatrix(3).auto_grad(torch.tensor([0.1, 0.2, 0.3]))

        mat = DiagonalMatrix(3)
        mat.jacobian_method = "jacrev"
        mat.jacobian_chunk_size = chunk_size
        grad, _ = mat.auto_grad(torch.tensor([0.1, 0.2, 0.3]))

        assert torch.allclose(grad, expected)

    def test_unknown_jacobian_method_raises(self):
        mat = ScalarMatrix(3)
        mat.jacobian_method = "bogus"
        with pytest.raises(ValueError, match="Unknown Jacobian method 'bogus'"):
            mat.auto_grad(torch.tensor([0.5]))

    def test_clears_cached_intermediates(self):
        """auto_grad resets the cache before differentiating."""
        mat = DiagonalMatrix(2)
        free_params = torch.tensor([0.0, 0.1])
        params = mat.build_params(free_params)
        mat.set_intermediates(params, "cached")
        assert mat.get_intermediates(params) == "cached"

        mat.auto_grad(free_params)

        assert mat.get_intermediates(params) is None

    def test_follows_input_dtype(self):
        grad, _ = ScalarMatrix(3).auto_grad(torch.tensor([0.5], dtype=torch.float64))
        assert grad.dtype == torch.float64

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = ScalarMatrix(3).auto_grad(torch.tensor([0.5], device=device))
        assert grad.device.type == device

    def test_auto_grad_rejects_empty_dict_with_free_params(self):
        with pytest.raises(ValueError, match="Missing free parameters"):
            ScalarMatrix(3).auto_grad({})

    def test_auto_grad_rejects_empty_tensor_with_free_params(self):
        with pytest.raises(ValueError, match="length 1"):
            ScalarMatrix(3).auto_grad(torch.tensor([]))

    def test_auto_grad_all_fixed_returns_none(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        for free_params in (None, {}, torch.tensor([])):
            grad, grad_names = mat.auto_grad(free_params)
            assert grad is None
            assert grad_names == []

    def test_auto_grad_all_fixed_validates_input(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        with pytest.raises(ValueError, match="length 0"):
            mat.auto_grad(torch.tensor([1.0]))


# --------------------------------------------------------------------------- #
# K. manual_grad (base implementation)
# --------------------------------------------------------------------------- #

class TestManualGradBase:
    """The base ``manual_grad`` is optional; subclasses may leave it unimplemented."""

    def test_base_implementation_raises(self):
        with pytest.raises(NotImplementedError):
            NoManualGradMatrix().manual_grad(torch.tensor([0.5]))

    def test_base_implementation_raises_before_validation(self):
        """The base class raises unconditionally, even for a no-parameter matrix."""
        with pytest.raises(NotImplementedError):
            IdentityMatrix(3).manual_grad(torch.tensor([]))

    def test_default_mode_falls_back_to_auto(self):
        grad, grad_names = NoManualGradMatrix().grad(torch.tensor([0.5]))
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["p"]

    def test_fallback_matches_auto_grad(self):
        mat = NoManualGradMatrix()
        grad, _ = mat.grad(torch.tensor([0.5]))
        auto, _ = mat.auto_grad(torch.tensor([0.5]))
        assert torch.allclose(grad, auto)


def _adapter(fixed):
    """An Adapter wrapping DiagonalMatrix(2) with a single logit parameter."""
    adaptee = DiagonalMatrix(2)

    def param_map(params):
        p = torch.sigmoid(params[0])
        return torch.stack([p, 1 - p])

    param_specs = {
        "logit": {"fixed": fixed, "default": torch.tensor([0.0]), "trans": TransformIdentity()}
    }
    return Adapter(adaptee, param_specs, param_map)


def _free_matrices():
    """One instance per Matrix subclass that implements manual_grad, with all params free."""
    return {
        "DiagonalMatrix": DiagonalMatrix(3),
        "ScalarMatrix": ScalarMatrix(3),
        "EqualEntryMatrix": EqualEntryMatrix(3, 3),
        "LowerTriangularMatrix": LowerTriangularMatrix(3, 3),
        "UnconstrainedMatrix": UnconstrainedMatrix(3),
        "AR1Matrix": AR1Matrix(3),
        "CompoundSymmetricMatrix": CompoundSymmetricMatrix(3),
        "EquicorrelationMatrix": EquicorrelationMatrix(3),
        "Adapter": _adapter(fixed=False),
    }


def _all_fixed_matrix(name):
    """The same matrices with every parameter fixed."""
    if name == "Adapter":
        return _adapter(fixed=True)
    mat = _free_matrices()[name]
    for spec in mat.param_specs.values():
        spec["fixed"] = True
    return mat


def _partially_fixed_matrix(name):
    """The same matrices with every parameter but the first fixed.

    A single free parameter is enough to exercise the free/fixed mask each
    ``manual_grad`` builds; the single-parameter matrices come back unchanged.
    """
    mat = _free_matrices()[name]
    for spec in list(mat.param_specs.values())[1:]:
        spec["fixed"] = True
    return mat


@pytest.mark.parametrize("name", list(_free_matrices()))
class TestManualGradContract:
    """``(None, [])`` is decided by the matrix, never by skipping input validation."""

    def test_manual_grad_accepts_valid_input(self, name):
        mat = _free_matrices()[name]
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (mat.num_free_params, *mat.shape)
        assert grad_names == mat.free_param_names

    def test_manual_grad_rejects_empty_dict(self, name):
        with pytest.raises(ValueError, match="Missing free parameters"):
            _free_matrices()[name].manual_grad({})

    def test_manual_grad_rejects_empty_tensor(self, name):
        with pytest.raises(ValueError, match="length"):
            _free_matrices()[name].manual_grad(torch.tensor([]))

    def test_manual_grad_all_fixed_returns_none(self, name):
        mat = _all_fixed_matrix(name)
        for free_params in (None, {}, torch.tensor([])):
            grad, grad_names = mat.manual_grad(free_params)
            assert grad is None
            assert grad_names == []

    def test_manual_grad_all_fixed_validates_input(self, name):
        with pytest.raises(ValueError, match="length 0"):
            _all_fixed_matrix(name).manual_grad(torch.tensor([1.0]))

    def test_grad_default_mode_validates_input(self, name):
        with pytest.raises(ValueError, match="Missing free parameters"):
            _free_matrices()[name].grad({})

    def test_manual_grad_matches_auto_grad(self, name):
        """Shape agreement is not enough: the closed form must get the values right.

        This is the only value-level cross-check for ``EqualEntryMatrix``,
        ``LowerTriangularMatrix`` and ``UnconstrainedMatrix``, which have no test
        module of their own.
        """
        mat = _free_matrices()[name]
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)

        manual, manual_names = mat.manual_grad(free_params)
        auto, auto_names = mat.auto_grad(free_params)

        assert manual_names == auto_names == mat.free_param_names
        assert torch.allclose(manual, auto), (manual - auto).abs().max().item()

    def test_manual_grad_matches_auto_grad_with_fixed_params(self, name):
        """The same check with all but the first parameter fixed, so each subclass's
        free/fixed mask is on the path rather than the all-free shortcut.
        """
        mat = _partially_fixed_matrix(name)
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)

        manual, manual_names = mat.manual_grad(free_params)
        auto, auto_names = mat.auto_grad(free_params)

        assert manual_names == auto_names == mat.free_param_names
        assert torch.allclose(manual, auto), (manual - auto).abs().max().item()

    def test_manual_grad_dict_input_matches_tensor(self, name):
        """A dict is accepted, and means the same as the flat tensor.

        Also the only consumer of ``_to_free_param_dict``'s output on a real model.
        """
        mat = _free_matrices()[name]
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)

        from_dict, dict_names = mat.manual_grad(mat._to_free_param_dict(free_params))
        from_tensor, tensor_names = mat.manual_grad(free_params)

        assert dict_names == tensor_names
        assert torch.allclose(from_dict, from_tensor)


# --------------------------------------------------------------------------- #
# K-bis. auto_grad repeatability and ordering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(_free_matrices()))
class TestAutoGradRepeatability:
    """``auto_grad`` is a pure function of its parameters, whatever the instance state.

    It clears the intermediate cache before differentiating, and eight subclasses return
    a cached intermediate as their result, so the order in which calls are made on one
    instance is part of the contract rather than an implementation detail.
    """

    def test_repeated_calls_agree(self, name):
        mat = _free_matrices()[name]
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)

        first, first_names = mat.auto_grad(free_params)
        second, second_names = mat.auto_grad(free_params.clone())

        assert first_names == second_names
        assert torch.equal(first, second)

    def test_three_consecutive_calls_agree(self, name):
        """Three calls in a row on one instance, with nothing in between.

        Every pair is compared rather than only first against last, so a call that
        returns the right answer on alternate invocations is caught too.
        """
        mat = _free_matrices()[name]
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)

        results = [mat.auto_grad(free_params)[0] for _ in range(3)]

        for left, right in ((0, 1), (0, 2), (1, 2)):
            assert torch.equal(results[left], results[right]), (left, right)

    def test_repeat_with_a_cache_populated_between(self, name):
        """A plain ``__call__`` in between must not change the next gradient."""
        mat = _free_matrices()[name]
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)

        first, _ = mat.auto_grad(free_params)
        mat(free_params)
        second, _ = mat.auto_grad(free_params)

        assert torch.equal(first, second)

    def test_fresh_instance_agrees(self, name):
        """Nothing leaks in from the instance that ran the earlier gradient."""
        free_params = torch.linspace(-0.3, 0.4, _free_matrices()[name].num_free_params)

        first, _ = _free_matrices()[name].auto_grad(free_params)
        fresh, _ = _free_matrices()[name].auto_grad(free_params)

        assert torch.equal(first, fresh)

    def test_trace_does_not_poison_a_later_manual_grad(self, name):
        """Whatever the traced ``__call__`` cached must not be handed to the closed form.

        This is the ordering the per-matrix ``manual_grad`` vs ``auto_grad`` tests miss,
        since they always call the closed form first.
        """
        free_params = torch.linspace(-0.3, 0.4, _free_matrices()[name].num_free_params)
        reference, _ = _free_matrices()[name].manual_grad(free_params)

        mat = _free_matrices()[name]
        mat.auto_grad(free_params)
        manual, _ = mat.manual_grad(free_params)

        assert torch.allclose(manual, reference), (manual - reference).abs().max().item()

    def test_matches_a_fresh_instance_after_the_trace(self, name):
        """The gradient after tracing equals the one a clean instance produces."""
        free_params = torch.linspace(-0.3, 0.4, _free_matrices()[name].num_free_params)

        mat = _free_matrices()[name]
        mat.auto_grad(free_params)
        traced, _ = mat.auto_grad(free_params)
        clean, _ = _free_matrices()[name].auto_grad(free_params)

        assert torch.equal(traced, clean)


# --------------------------------------------------------------------------- #
# L. __call__ (abstract)
# --------------------------------------------------------------------------- #

class TestCallIsAbstract:
    """``Matrix`` cannot be instantiated while ``__call__`` is unimplemented."""

    def test_matrix_is_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Matrix((2, 2), {})

    def test_subclass_without_call_is_abstract(self):
        class Incomplete(Matrix):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete((2, 2), {})


# --------------------------------------------------------------------------- #
# M. grad dispatch
# --------------------------------------------------------------------------- #

class TestGradDispatch:
    """Tests for Matrix.grad dispatch logic."""

    def test_default_mode_uses_manual(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        grad_default, _ = mat.grad(free_params)
        grad_manual, _ = mat.manual_grad(free_params)
        assert torch.allclose(grad_default, grad_manual)

    def test_manual_mode_uses_manual(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "manual"
        free_params = torch.tensor([0.5])
        grad, grad_names = mat.grad(free_params)
        expected, _ = mat.manual_grad(free_params)
        assert grad_names == mat.free_param_names
        assert torch.allclose(grad, expected)

    def test_manual_mode_does_not_fall_back_for_param_less_matrix(self):
        """``"default"`` returns ``(None, [])`` here; ``"manual"`` must not fall back instead."""
        assert IdentityMatrix(3).grad(torch.tensor([])) == (None, [])

        mat = IdentityMatrix(3)
        mat.grad_mode = "manual"
        with pytest.raises(NotImplementedError):
            mat.grad(torch.tensor([]))

    def test_manual_mode_without_manual_grad_raises(self):
        """Only ``"default"`` mode catches ``NotImplementedError``."""
        mat = NoManualGradMatrix()
        mat.grad_mode = "manual"
        with pytest.raises(NotImplementedError):
            mat.grad(torch.tensor([0.5]))

    def test_auto_mode(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "auto"
        free_params = torch.tensor([0.5])
        grad, _ = mat.grad(free_params)
        auto, _ = mat.auto_grad(free_params)
        assert torch.allclose(grad, auto)

    def test_auto_mode_ignores_manual_grad(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "auto"
        assert hasattr(mat, "manual_grad")
        grad, _ = mat.grad(torch.tensor([0.5]))
        auto, _ = mat.auto_grad(torch.tensor([0.5]))
        assert torch.allclose(grad, auto)

    @pytest.mark.parametrize("mode", ["invalid", "", "MANUAL"])
    def test_unknown_grad_mode_raises(self, mode):
        mat = ScalarMatrix(3)
        mat.grad_mode = mode
        with pytest.raises(RuntimeError, match="grad mode"):
            mat.grad(torch.tensor([0.5]))

    def test_default_falls_back_to_auto(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    @pytest.mark.parametrize("mode", ["default", "manual", "auto"])
    def test_zero_free_params_returns_none(self, mode):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        mat.grad_mode = mode
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    @pytest.mark.parametrize("mode", ["default", "manual", "auto"])
    def test_validation_is_not_bypassed(self, mode):
        mat = ScalarMatrix(3)
        mat.grad_mode = mode
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.grad({})

    @pytest.mark.parametrize("name", list(_free_matrices()))
    def test_all_three_modes_agree(self, name):
        """Dispatch changes the route, never the values."""
        free_params = torch.linspace(-0.3, 0.4, _free_matrices()[name].num_free_params)

        gradients = {}
        for mode in ("default", "manual", "auto"):
            mat = _free_matrices()[name]
            mat.grad_mode = mode
            gradients[mode], names = mat.grad(free_params)
            assert names == mat.free_param_names

        assert torch.allclose(gradients["default"], gradients["manual"])
        assert torch.allclose(gradients["default"], gradients["auto"])

    @pytest.mark.parametrize("name", list(_free_matrices()))
    def test_grad_accepts_dict_input(self, name):
        """A dict of free parameters dispatches like the flat tensor."""
        mat = _free_matrices()[name]
        free_params = torch.linspace(-0.3, 0.4, mat.num_free_params)

        from_dict, dict_names = mat.grad(mat._to_free_param_dict(free_params))
        from_tensor, tensor_names = mat.grad(free_params)

        assert dict_names == tensor_names
        assert torch.allclose(from_dict, from_tensor)


# --------------------------------------------------------------------------- #
# N, O, P. Intermediate caching
# --------------------------------------------------------------------------- #

class TestIntermediateCaching:
    """Tests for set/get/reset_intermediates."""

    def test_set_and_get(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        mat.set_intermediates(params, {"key": "value"})
        assert mat.get_intermediates(params) == {"key": "value"}

    def test_returns_the_cached_object(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        cached = {"key": "value"}
        mat.set_intermediates(params, cached)
        assert mat.get_intermediates(params) is cached

    def test_overwrites_previous_entry(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        mat.set_intermediates(params, "first")
        mat.set_intermediates(params, "second")
        assert mat.get_intermediates(params) == "second"

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

    def test_set_after_reset_repopulates(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        mat.set_intermediates(params, "value")
        mat.reset_intermediates()
        mat.set_intermediates(params, "value again")
        assert mat.get_intermediates(params) == "value again"

    def test_same_values_different_dtype_is_a_miss(self):
        """The key carries the dtype, so an equal-valued tensor of another dtype misses."""
        mat = ScalarMatrix(3)
        f32 = torch.tensor([1.0], dtype=torch.float32)
        f64 = torch.tensor([1.0], dtype=torch.float64)

        mat.set_intermediates(f32, "value")

        assert mat.get_intermediates(f64) is None
        assert mat.get_intermediates(f32) == "value"

    def test_same_values_different_device_is_a_miss(self):
        """Device is checked before the comparison, which would raise across devices."""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")

        mat = ScalarMatrix(3)
        cpu = torch.tensor([1.0])
        accelerated = torch.tensor([1.0], device=device)

        mat.set_intermediates(cpu, "value")

        assert mat.get_intermediates(accelerated) is None
        assert mat.get_intermediates(cpu) == "value"

    def test_permuted_params_are_a_miss(self):
        """Position is part of the key: the same values reordered is a different model."""

        mat = DiagonalMatrix(2)
        first = mat.build_params(torch.tensor([1.0, 2.0]))
        swapped = mat.build_params(torch.tensor([2.0, 1.0]))
        assert torch.equal(first.flip(0), swapped)  # a genuine permutation, not a rounding artifact

        mat.set_intermediates(first, "value")

        assert mat.get_intermediates(swapped) is None

    def test_repeated_values_do_not_cancel(self):
        """Two equal parameters and two distinct ones are different models."""

        mat = DiagonalMatrix(2)
        doubled = mat.build_params(torch.tensor([1.0, 1.0]))
        distinct = mat.build_params(torch.tensor([1.0, 2.0]))

        mat.set_intermediates(doubled, "value")

        assert mat.get_intermediates(distinct) is None

    def test_distinct_values_are_a_miss(self):
        """``(0.5, 1.0, 3.0)`` and ``(0.5, 1.5, 2.0)`` are neither equal nor permutations
        of one another, and must not share a cache entry.
        """
        specs = {
            f"sigma^2_{i}": {
                "fixed": False,
                "default": torch.tensor([1.0]),
                "trans": TransformIdentity(),
            }
            for i in range(3)
        }
        mat = DiagonalMatrix(3, param_specs=specs)
        mat.set_intermediates(torch.tensor([0.5, 1.0, 3.0]), "value")

        assert mat.get_intermediates(torch.tensor([0.5, 1.5, 2.0])) is None
        assert mat.get_intermediates(torch.tensor([0.5, 1.0, 3.0])) == "value"

    def test_reordering_free_params_returns_the_right_matrix(self):
        """Regression: a permuted parameter vector must not reuse another model's result."""
        op = BlockDiagonal(a=ScalarMatrix(3), b=ScalarMatrix(2))
        first = op(torch.tensor([0.5, 1.0]))
        second = op(torch.tensor([1.0, 0.5]))
        truth = BlockDiagonal(a=ScalarMatrix(3), b=ScalarMatrix(2))(torch.tensor([1.0, 0.5]))

        assert not torch.equal(second, first)
        assert torch.equal(second, truth)

    def test_key_survives_in_place_edit_of_callers_tensor(self):
        """The key holds a copy, so mutating the caller's tensor cannot retarget it."""
        mat = ScalarMatrix(3)
        params = torch.tensor([0.5])
        mat.set_intermediates(params, "value")

        params[0] = 99.0

        assert mat.get_intermediates(torch.tensor([0.5])) == "value"
        assert mat.get_intermediates(torch.tensor([99.0])) is None

    def test_signed_zero_is_a_hit(self):
        """The key is numerical equality, not bit identity."""
        mat = ScalarMatrix(3)
        mat.set_intermediates(torch.tensor([0.0]), "value")

        assert mat.get_intermediates(torch.tensor([-0.0])) == "value"

    def test_empty_params_no_op_set(self):
        mat = IdentityMatrix(3)
        result = mat.set_intermediates(torch.tensor([]), {"key": "value"})
        assert result is None

    def test_empty_set_leaves_cache_untouched(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        mat.set_intermediates(params, "value")

        mat.set_intermediates(torch.tensor([]), "ignored")

        assert mat.get_intermediates(params) == "value"

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

    def test_get_raises_on_non_tensor(self):
        with pytest.raises(TypeError, match="Torch tensor"):
            ScalarMatrix(3).get_intermediates([1.0, 2.0])

    def test_get_raises_on_non_1d(self):
        with pytest.raises(ValueError, match="1D tensor"):
            ScalarMatrix(3).get_intermediates(torch.zeros(2, 2))

    def test_reset_on_init(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        assert mat.get_intermediates(params) is None

    def test_reset_clears_every_key(self):
        mat = ScalarMatrix(3)
        params = mat.build_params(torch.tensor([0.5]))
        mat.set_intermediates(params, "value")
        mat.reset_intermediates()
        assert mat._intermediates == {
            "params": None, "dtype": None, "device": None, "intermediates": None
        }


# --------------------------------------------------------------------------- #
# Q. Properties
# --------------------------------------------------------------------------- #

class TestMatrixProperties:
    """Tests for Matrix properties."""

    def test_shape(self):
        mat = ScalarMatrix(3)
        assert mat.shape == (3, 3)
        assert isinstance(mat.shape, tuple)

    def test_shape_retains_empty_tuple(self):
        assert MinimalMatrix(None, {}).shape == ()

    def test_param_specs_is_not_a_copy(self):
        """The specification is returned as stored, so in-place edits take effect."""
        mat = ScalarMatrix(3)
        stored = mat.param_specs

        assert mat.param_specs is stored

        mat.param_specs["sigma^2"]["fixed"] = True

        assert stored["sigma^2"]["fixed"] is True
        assert mat.free_param_names == []

    def test_param_specs_edit_updates_counts(self):
        mat = ScalarMatrix(3)
        assert mat.num_free_params == 1
        mat.param_specs["sigma^2"]["fixed"] = True
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 1

    def test_param_names(self):
        mat = ScalarMatrix(3)
        assert mat.param_names == ["sigma^2"]

    def test_param_names_preserve_insertion_order(self):
        mat = DiagonalMatrix(3, param_specs={
            "c": _spec(), "a": _spec(), "b": _spec(),
        })
        assert mat.param_names == ["c", "a", "b"]

    def test_free_param_names(self):
        mat = ScalarMatrix(3)
        assert mat.free_param_names == ["sigma^2"]

    def test_free_param_names_mixed(self):
        assert _mixed_matrix().free_param_names == ["sigma^2_0", "sigma^2_2"]

    def test_fixed_param_names(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_names == []

    def test_fixed_param_names_mixed(self):
        assert _mixed_matrix().fixed_param_names == ["sigma^2_1"]

    def test_free_and_fixed_partition_param_names(self):
        mat = _mixed_matrix()
        assert set(mat.free_param_names) | set(mat.fixed_param_names) == set(mat.param_names)
        assert set(mat.free_param_names) & set(mat.fixed_param_names) == set()
        assert len(mat.free_param_names) + len(mat.fixed_param_names) == mat.num_params

    def test_free_param_index(self):
        mat = ScalarMatrix(3)
        assert mat.free_param_index == [0]

    def test_free_param_index_mixed(self):
        assert _mixed_matrix().free_param_index == [0, 2]

    def test_fixed_param_index(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_index == []

    def test_fixed_param_index_mixed(self):
        assert _mixed_matrix().fixed_param_index == [1]

    def test_indices_agree_with_names(self):
        mat = _mixed_matrix()
        names = mat.param_names
        assert [names[i] for i in mat.free_param_index] == mat.free_param_names
        assert [names[i] for i in mat.fixed_param_index] == mat.fixed_param_names

    def test_num_params(self):
        mat = DiagonalMatrix(3)
        assert mat.num_params == 3

    def test_num_free_params(self):
        mat = DiagonalMatrix(3)
        assert mat.num_free_params == 3

    def test_num_fixed_params(self):
        mat = IdentityMatrix(3)
        assert mat.num_fixed_params == 0

    def test_num_fixed_params_mixed(self):
        assert _mixed_matrix().num_fixed_params == 1

    def test_num_free_params_zero_when_all_fixed(self):
        mat = ScalarMatrix(3)
        mat.param_specs["sigma^2"]["fixed"] = True
        assert mat.num_free_params == 0

    def test_param_defaults(self):
        mat = ScalarMatrix(3)
        defaults = mat.param_defaults
        assert "sigma^2" in defaults
        assert torch.equal(defaults["sigma^2"], torch.tensor([0.0]))

    def test_param_defaults_is_the_union(self):
        mat = _mixed_matrix()
        assert set(mat.param_defaults) == set(mat.param_names)

    def test_free_param_defaults(self):
        mat = ScalarMatrix(3)
        assert "sigma^2" in mat.free_param_defaults

    def test_fixed_param_defaults_empty(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_defaults == {}

    def test_fixed_param_defaults_nonempty(self):
        mat = _mixed_matrix()
        assert list(mat.fixed_param_defaults) == ["sigma^2_1"]
        assert torch.equal(mat.fixed_param_defaults["sigma^2_1"], torch.tensor([7.5]))

    def test_param_trans(self):
        mat = ScalarMatrix(3)
        trans = mat.param_trans
        assert "sigma^2" in trans
        assert isinstance(trans["sigma^2"], TransformExpPow2)

    def test_param_trans_returns_the_spec_objects(self):
        mat = ScalarMatrix(3)
        assert mat.param_trans["sigma^2"] is mat.param_specs["sigma^2"]["trans"]

    def test_free_param_trans(self):
        mat = ScalarMatrix(3)
        assert "sigma^2" in mat.free_param_trans

    def test_fixed_param_trans_empty(self):
        mat = ScalarMatrix(3)
        assert mat.fixed_param_trans == {}

    def test_fixed_param_trans_nonempty(self):
        mat = _mixed_matrix()
        assert list(mat.fixed_param_trans) == ["sigma^2_1"]

    def test_repr_dict(self):
        mat = ScalarMatrix(3)
        rd = mat.repr_dict
        assert rd["shape"] == (3, 3)
        assert "sigma^2" in rd["param_specs"]


# --------------------------------------------------------------------------- #
# R. Representation
# --------------------------------------------------------------------------- #

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

    def test_single_line_omits_empty_param_specs(self):
        """``param_specs`` is falsy when empty, so the argument is dropped."""
        assert repr(IdentityMatrix(3)) == "IdentityMatrix(shape=(3, 3))"

    def test_single_line_omits_falsy_shape(self):
        """Every argument is falsy, so the argument list is empty."""
        assert repr(MinimalMatrix(None, {})) == "MinimalMatrix()"

    def test_single_line_no_ellipsis_for_two_params(self):
        assert "..." not in repr(DiagonalMatrix(2))

    def test_single_line_ellipsis_for_three_params(self):
        r = repr(DiagonalMatrix(3))
        assert r.count("...") == 1
        assert "'sigma^2_0'" in r
        assert "'sigma^2_2'" in r
        assert "'sigma^2_1'" not in r

    def test_multi_line_layout(self):
        mat = MultiLineMatrix({"shape": (2, 2), "n": 1})
        assert repr(mat) == "MultiLineMatrix(\n  shape=(2, 2),\n  n=1\n)"

    def test_multi_line_renders_nested_matrix(self):
        mat = MultiLineMatrix({"inner": ScalarMatrix(2)})
        assert "inner=ScalarMatrix(shape=(2, 2)" in repr(mat)

    def test_multi_line_renders_tensor_as_shape(self):
        mat = MultiLineMatrix({"t": torch.zeros(3, 4)})
        assert "t=torch.Size([3, 4])" in repr(mat)

    def test_multi_line_renders_dict(self):
        mat = MultiLineMatrix({"d": {"k": 1}})
        r = repr(mat)
        assert "d={" in r
        assert "'k': 1" in r

    def test_multi_line_indents_continuation_lines(self):
        """Newlines inside a plain value are re-indented to line up under the value."""
        mat = MultiLineMatrix({"v": MultilineValue()})
        assert repr(mat).split("\n") == [
            "MultiLineMatrix(",
            "  v=line1",
            "    line2",
            ")",
        ]

    def test_repr_does_not_raise_for_nested_operators(self):
        op = Sum(Sum(ScalarMatrix(2), IdentityMatrix(2)), IdentityMatrix(2))
        assert "Sum(" in repr(op)

    def test_repr_dict_can_be_overridden(self):
        mat = MultiLineMatrix({"custom": "value"})
        assert mat.repr_dict == {"custom": "value"}
        assert "custom='value'" in repr(mat)

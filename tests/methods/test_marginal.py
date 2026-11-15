"""Tests for :class:`torch_openreml.methods.marginal.Marginal`.

The marginal method evaluates the REML score and average-information matrix
from ``V`` directly.  The tiny models below are small enough that the exact
score, AI matrix and log-likelihood can be checked against explicit matrix
formulas, while the stochastic trace path is exercised with fixed seeds.  The
model-aware entry point is also checked against ``torch.autograd`` through the
differentiable log-likelihood.
"""

import math

import pytest
import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix
from torch_openreml.example_data import john_alpha
from torch_openreml.methods import CForm, Marginal

DTYPE = torch.float64
JITTER = 1e-6

TINY_Z = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
TINY_X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
TINY_Y = [1.0, 2.0, 0.5, 1.5]
TINY_FREE = [0.1, -0.2]


def _tiny_model(dtype=DTYPE):
    """A 4-observation model whose composite V has no closed-form inverse."""
    z = torch.tensor(TINY_Z, dtype=dtype)
    x = torch.tensor(TINY_X, dtype=dtype)
    y = torch.tensor(TINY_Y, dtype=dtype)
    model = MixedModelREML((z, ScalarMatrix(2)), R=ScalarMatrix(4))
    free = torch.tensor(TINY_FREE, dtype=dtype)
    return model, y, x, free


def _no_random_model(dtype=DTYPE):
    """A model whose V is a plain ScalarMatrix, so a closed-form inverse exists."""
    x = torch.tensor(TINY_X, dtype=dtype)
    y = torch.tensor(TINY_Y, dtype=dtype)
    model = MixedModelREML(R=ScalarMatrix(4))
    free = torch.tensor([0.1], dtype=dtype)
    return model, y, x, free


def _dummy_design_model(dtype=DTYPE):
    """A model whose composite V follows a Matrix design's parameters."""
    z = DummyMatrix(["a", "a", "b", "b"])
    model = MixedModelREML((z, ScalarMatrix(2)), R=ScalarMatrix(4))
    x = torch.tensor(TINY_X, dtype=dtype)
    y = torch.tensor(TINY_Y, dtype=dtype)
    free = torch.tensor(TINY_FREE, dtype=dtype)
    return model, y, x, free


def _jittered_v(model, free):
    """The covariance the marginal path actually uses for a composite V."""
    v = model.V(free)
    return v + JITTER * torch.eye(v.shape[0], dtype=v.dtype, device=v.device)


def _reference_reml(x, y, v, dv):
    """Explicit marginal REML score, AI matrix and log-likelihood."""
    v_inv = torch.linalg.inv(v)
    x_v_inv_x = x.T @ v_inv @ x
    p = v_inv - v_inv @ x @ torch.linalg.inv(x_v_inv_x) @ x.T @ v_inv
    p_y = p @ y
    q = dv @ p_y
    score = 0.5 * (q @ p_y - (p.unsqueeze(0) * dv.transpose(-1, -2)).sum(dim=(-2, -1)))
    ai = 0.5 * (q @ p @ q.T)
    loglik = -0.5 * (torch.logdet(v) + torch.logdet(x_v_inv_x) + y @ p_y)
    return score, ai, loglik


def _expected(model, y, x, free):
    """The exact reference values for the covariance the marginal path uses."""
    v = _jittered_v(model, free)
    dv = model.V.grad(free)[0]
    return _reference_reml(x, y, v, dv)


def _accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("no accelerator available")


def _john_alpha_model(dtype=DTYPE):
    """John's alpha design with a single genotype random effect."""
    n = len(john_alpha)
    y = torch.tensor(john_alpha["yield"].to_numpy(), dtype=dtype)
    ones = torch.ones(n, 1, dtype=dtype)
    rep = DummyMatrix(john_alpha["rep"], drop_first=True)(torch.zeros(0, dtype=dtype))
    x = torch.cat([ones, rep], dim=1)
    z = DummyMatrix(john_alpha["gen"])
    model = MixedModelREML((z, ScalarMatrix(24)), R=ScalarMatrix(n))
    free = torch.zeros(2, dtype=dtype)
    return model, y, x, free


class TestMarginalConstruction:
    """Construction, defaults, validation and the reset contract."""

    def test_defaults(self):
        method = Marginal()
        assert method.trace_approx is False
        assert method.subspace_fraction == 0.01
        assert method.exact_score_threshold == 10.0
        assert method._last_score_norm is None

    def test_min_n_for_trace_is_100(self):
        assert Marginal.MIN_N_FOR_TRACE == 100

    def test_accepts_custom_options(self):
        method = Marginal(trace_approx="auto", subspace_fraction=0.25, exact_score_threshold=2.5)
        assert method.trace_approx == "auto"
        assert method.subspace_fraction == 0.25
        assert method.exact_score_threshold == 2.5
        assert method._last_score_norm is None

    @pytest.mark.parametrize("flag", [None, "exact", "approx", 2, -1])
    def test_rejects_unknown_trace_mode(self, flag):
        with pytest.raises(ValueError, match="`trace_approx` must be False, True or 'auto'"):
            Marginal(trace_approx=flag)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, 2.0])
    def test_rejects_subspace_fraction_outside_unit_interval(self, fraction):
        with pytest.raises(ValueError, match="`subspace_fraction` must lie in \\[0, 1\\]"):
            Marginal(subspace_fraction=fraction)

    @pytest.mark.parametrize("threshold", [-1.0, -1e-9])
    def test_rejects_negative_score_threshold(self, threshold):
        with pytest.raises(
            ValueError, match="`exact_score_threshold` must be non-negative"
        ):
            Marginal(exact_score_threshold=threshold)

    @pytest.mark.parametrize("boundary", [0.0, 1.0])
    def test_accepts_subspace_fraction_boundaries(self, boundary):
        assert Marginal(subspace_fraction=boundary).subspace_fraction == boundary

    def test_reset_clears_the_score_history(self):
        method = Marginal()
        method._last_score_norm = torch.tensor(3.0)
        method.reset()
        assert method._last_score_norm is None

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "trace_approx is validated with `not in (False, True, 'auto')`, so the integers "
            "0 and 1 pass as the booleans; the documented contract says non-boolean values "
            "are rejected"
        ),
    )
    def test_rejects_integer_trace_mode(self):
        with pytest.raises(ValueError):
            Marginal(trace_approx=1)

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "with trace_approx=1 (accepted by the loose validation) none of the three `if` "
            "branches in ai_step runs, so it raises UnboundLocalError instead of a ValueError"
        ),
    )
    def test_integer_trace_mode_does_not_crash_ai_step(self):
        model, y, x, free = _tiny_model()
        method = Marginal(trace_approx=1)
        score, _, _ = method.ai_step(model, y, x, free)
        assert score.shape == (2,)


class TestMarginalLoglik:
    """The restricted log-likelihood, model-aware and tensor-level."""

    def test_none_model_raises_value_error(self):
        _, y, x, free = _tiny_model()
        with pytest.raises(ValueError, match="`model` must not be None"):
            Marginal().loglik(None, y, x, free)

    def test_matches_explicit_reference(self):
        model, y, x, free = _tiny_model()
        _, _, expected = _expected(model, y, x, free)
        actual = Marginal().loglik(model, y, x, free)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_without_random_effect_matches_exact_reference(self):
        model, y, x, free = _no_random_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        _, _, expected = _reference_reml(x, y, v, dv)
        actual = Marginal().loglik(model, y, x, free)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_tensor_kernel_matches_explicit_reference(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        _, _, expected = _reference_reml(x, y, v, dv)
        actual = Marginal().loglik_from_tensors(x, y, v)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_supplied_inverse_matches_explicit_reference(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        _, _, expected = _reference_reml(x, y, v, dv)
        actual = Marginal().loglik_from_tensors(x, y, v, v_inv=torch.linalg.inv(v))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_loglik_agrees_with_cform(self):
        model, y, x, free = _tiny_model()
        assert math.isclose(
            Marginal().loglik(model, y, x, free).item(),
            CForm().loglik(model, y, x, free).item(),
            rel_tol=0.0,
            abs_tol=1e-5,
        )


class TestMarginalScoreAndInformation:
    """Score vector and AI matrix values, invariants and trace modes."""

    def test_score_shape(self):
        model, y, x, free = _tiny_model()
        score, _, _ = Marginal().ai_step(model, y, x, free)
        assert score.shape == (2,)

    def test_score_matches_explicit_reference(self):
        model, y, x, free = _tiny_model()
        expected, _, _ = _expected(model, y, x, free)
        score, _, _ = Marginal().ai_step(model, y, x, free)
        torch.testing.assert_close(score, expected, rtol=0.0, atol=1e-12)

    def test_ai_matches_explicit_reference(self):
        model, y, x, free = _tiny_model()
        _, expected, _ = _expected(model, y, x, free)
        _, ai, _ = Marginal().ai_step(model, y, x, free)
        torch.testing.assert_close(ai, expected, rtol=0.0, atol=1e-12)

    def test_score_matches_autograd_of_loglik(self):
        model, y, x, _ = _tiny_model()
        free = torch.tensor(TINY_FREE, dtype=DTYPE, requires_grad=True)
        # The composite covariance caches its value; clear it so the cached
        # tensor is not the detached one built by ai_step.
        model.V.reset_intermediates()
        loglik = Marginal().loglik(model, y, x, free)
        (autograd_score,) = torch.autograd.grad(loglik, free)
        score, _, _ = Marginal().ai_step(model, y, x, free.detach())
        torch.testing.assert_close(score, autograd_score, rtol=0.0, atol=1e-12)

    def test_without_random_effect_uses_exact_inverse_branch(self):
        model, y, x, free = _no_random_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        expected_score, expected_ai, expected_loglik = _reference_reml(x, y, v, dv)
        score, ai, loglik = Marginal().ai_step(model, y, x, free)
        torch.testing.assert_close(score, expected_score, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(ai, expected_ai, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(loglik, expected_loglik, rtol=0.0, atol=1e-12)

    def test_ai_is_symmetric(self):
        model, y, x, free = _tiny_model()
        _, ai, _ = Marginal().ai_step(model, y, x, free)
        torch.testing.assert_close(ai, ai.T, rtol=0.0, atol=1e-14)

    def test_ai_is_positive_semidefinite(self):
        model, y, x, free = _tiny_model()
        _, ai, _ = Marginal().ai_step(model, y, x, free)
        eigenvalues = torch.linalg.eigvalsh((ai + ai.T) / 2.0)
        assert torch.all(eigenvalues >= -1e-12)

    def test_require_loglik_false_skips_loglik_but_keeps_estimates(self):
        model, y, x, free = _tiny_model()
        score, ai, _ = Marginal().ai_step(model, y, x, free)
        short_score, short_ai, short_loglik = Marginal().ai_step(
            model, y, x, free, require_loglik=False
        )
        assert torch.isnan(torch.as_tensor(short_loglik))
        torch.testing.assert_close(short_score, score, rtol=0.0, atol=0.0)
        torch.testing.assert_close(short_ai, ai, rtol=0.0, atol=0.0)

    def test_exact_branches_agree(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        no_inverse = Marginal().ai_step_from_tensors(x, y, v, dv)
        with_inverse = Marginal().ai_step_from_tensors(
            x, y, v, dv, v_inv=torch.linalg.inv(v)
        )
        for got, want in zip(no_inverse, with_inverse):
            torch.testing.assert_close(got, want, rtol=0.0, atol=1e-12)

    def test_full_subspace_approximation_reproduces_exact_score(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        expected = Marginal().ai_step_from_tensors(x, y, v, dv)
        actual = Marginal().ai_step_from_tensors(
            x, y, v, dv, trace_approx=True, subspace_fraction=1.0
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=1e-12)
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)

    def test_approximate_trace_changes_score_but_not_ai_or_loglik(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        expected = Marginal().ai_step_from_tensors(x, y, v, dv)
        torch.manual_seed(0)
        actual = Marginal().ai_step_from_tensors(
            x, y, v, dv, trace_approx=True, subspace_fraction=0.5
        )
        assert not torch.allclose(actual[0], expected[0], rtol=0.0, atol=1e-12)
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)

    def test_approximate_trace_is_reproducible_for_a_fixed_seed(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        torch.manual_seed(0)
        first = Marginal().ai_step_from_tensors(
            x, y, v, dv, trace_approx=True, subspace_fraction=0.5
        )
        torch.manual_seed(0)
        second = Marginal().ai_step_from_tensors(
            x, y, v, dv, trace_approx=True, subspace_fraction=0.5
        )
        assert torch.equal(first[0], second[0])

    def test_different_seeds_give_different_trace_estimates(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        torch.manual_seed(0)
        first = Marginal().ai_step_from_tensors(
            x, y, v, dv, trace_approx=True, subspace_fraction=0.5
        )
        torch.manual_seed(1)
        second = Marginal().ai_step_from_tensors(
            x, y, v, dv, trace_approx=True, subspace_fraction=0.5
        )
        assert not torch.equal(first[0], second[0])

    def test_approximate_trace_with_supplied_inverse_runs(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        expected = Marginal().ai_step_from_tensors(
            x, y, v, dv, v_inv=torch.linalg.inv(v)
        )
        torch.manual_seed(0)
        actual = Marginal().ai_step_from_tensors(
            x,
            y,
            v,
            dv,
            v_inv=torch.linalg.inv(v),
            trace_approx=True,
            subspace_fraction=0.5,
        )
        assert actual[0].shape == expected[0].shape
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("supply_inverse", [False, True])
    def test_empty_derivative_raises(self, supply_inverse):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = v.new_zeros((0, 4, 4))
        kwargs = {"v_inv": torch.linalg.inv(v)} if supply_inverse else {}
        with pytest.raises(ValueError, match="dv must have at least one parameter"):
            Marginal().ai_step_from_tensors(x, y, v, dv, **kwargs)

    @pytest.mark.parametrize("supply_inverse", [False, True])
    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_bad_subspace_fraction_raises(self, supply_inverse, fraction):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        kwargs = {"v_inv": torch.linalg.inv(v)} if supply_inverse else {}
        with pytest.raises(ValueError, match="subspace_fraction must lie in \\[0, 1\\]"):
            Marginal().ai_step_from_tensors(
                x, y, v, dv, subspace_fraction=fraction, **kwargs
            )

    def test_tensor_kernel_rejects_mismatched_shapes(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = v.new_zeros((1, 3, 3))
        with pytest.raises(RuntimeError):
            Marginal().ai_step_from_tensors(x, y, v, dv)

    def test_model_aware_step_matches_tensor_kernel(self):
        model, y, x, free = _tiny_model()
        score, ai, loglik = Marginal().ai_step(model, y, x, free)
        expected = Marginal().ai_step_from_tensors(
            x, y, _jittered_v(model, free), model.V.grad(free)[0]
        )
        assert torch.equal(score, expected[0])
        assert torch.equal(ai, expected[1])
        assert torch.equal(loglik, expected[2])


class TestMarginalAutoTrace:
    """Resolution of ``trace_approx='auto'`` through the score history."""

    def test_first_auto_call_uses_approximation_for_large_problem(self):
        model, y, x, free = _tiny_model()
        method = Marginal(trace_approx="auto", subspace_fraction=0.5)
        method.MIN_N_FOR_TRACE = 4
        exact_score, _, _ = method.ai_step_from_tensors(
            x, y, _jittered_v(model, free), model.V.grad(free)[0]
        )
        torch.manual_seed(0)
        score, _, _ = method.ai_step(model, y, x, free)
        assert not torch.allclose(score, exact_score, rtol=0.0, atol=1e-12)

    def test_auto_records_the_score_norm(self):
        model, y, x, free = _tiny_model()
        method = Marginal(trace_approx="auto")
        method.MIN_N_FOR_TRACE = 4
        score, _, _ = method.ai_step(model, y, x, free)
        torch.testing.assert_close(method._last_score_norm, torch.norm(score))

    def test_auto_switches_to_exact_after_small_score(self):
        model, y, x, free = _tiny_model()
        method = Marginal(
            trace_approx="auto", exact_score_threshold=1e9, subspace_fraction=0.5
        )
        method.MIN_N_FOR_TRACE = 4
        torch.manual_seed(0)
        method.ai_step(model, y, x, free)  # first call: approximate
        score, _, _ = method.ai_step(model, y, x, free)  # norm < threshold: exact
        exact, _, _ = method.ai_step_from_tensors(
            x, y, _jittered_v(model, free), model.V.grad(free)[0]
        )
        torch.testing.assert_close(score, exact, rtol=0.0, atol=0.0)

    def test_auto_keeps_approximating_while_the_score_is_large(self):
        model, y, x, free = _tiny_model()
        method = Marginal(
            trace_approx="auto", exact_score_threshold=0.0, subspace_fraction=0.5
        )
        method.MIN_N_FOR_TRACE = 4
        method.ai_step(model, y, x, free)
        torch.manual_seed(0)
        score, _, _ = method.ai_step(model, y, x, free)
        exact, _, _ = method.ai_step_from_tensors(
            x, y, _jittered_v(model, free), model.V.grad(free)[0]
        )
        assert not torch.allclose(score, exact, rtol=0.0, atol=1e-12)

    def test_auto_stays_exact_below_min_n(self):
        model, y, x, free = _tiny_model()
        method = Marginal(trace_approx="auto", subspace_fraction=0.5)
        score, _, _ = method.ai_step(model, y, x, free)
        exact, _, _ = method.ai_step_from_tensors(
            x, y, _jittered_v(model, free), model.V.grad(free)[0]
        )
        torch.testing.assert_close(score, exact, rtol=0.0, atol=0.0)

    def test_reset_restarts_the_auto_history(self):
        model, y, x, free = _tiny_model()
        method = Marginal(trace_approx="auto")
        method.MIN_N_FOR_TRACE = 4
        method.ai_step(model, y, x, free)
        method.reset()
        assert method._last_score_norm is None


class TestMarginalPreprocessInput:
    """The tensor-extraction contract behind ``ai_step``."""

    def test_keys_with_grad(self):
        model, y, x, free = _tiny_model()
        data = Marginal().preprocess_input(model, y, x, free)
        assert set(data) == {"x", "y", "v", "v_inv", "dv"}

    def test_keys_without_grad(self):
        model, y, x, free = _tiny_model()
        data = Marginal().preprocess_input(model, y, x, free, grad=False)
        assert set(data) == {"x", "y", "v", "v_inv"}

    def test_composite_covariance_has_no_inverse_and_is_jittered(self):
        model, y, x, free = _tiny_model()
        data = Marginal().preprocess_input(model, y, x, free)
        assert data["v_inv"] is None
        torch.testing.assert_close(
            data["v"], _jittered_v(model, free), rtol=0.0, atol=0.0
        )

    def test_scalar_covariance_keeps_exact_inverse_and_no_jitter(self):
        model, y, x, free = _no_random_model()
        data = Marginal().preprocess_input(model, y, x, free)
        assert data["v_inv"] is not None
        torch.testing.assert_close(data["v"], model.V(free), rtol=0.0, atol=0.0)

    def test_derivative_shape(self):
        model, y, x, free = _tiny_model()
        data = Marginal().preprocess_input(model, y, x, free)
        assert data["dv"].shape == (2, 4, 4)

    def test_casts_response_and_design_to_free_parameter_dtype(self):
        model, _, x, _ = _dummy_design_model()
        y = torch.tensor(TINY_Y, dtype=torch.float64)
        free = torch.tensor(TINY_FREE, dtype=torch.float32)
        data = Marginal().preprocess_input(model, y, x, free)
        assert data["y"].dtype == torch.float32
        assert data["x"].dtype == torch.float32

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "ai_step documents ValueError for model=None, but only loglik checks for it; "
            "preprocess_input dereferences model and raises AttributeError"
        ),
    )
    def test_ai_step_with_none_model_raises_documented_value_error(self):
        _, y, x, free = _tiny_model()
        with pytest.raises(ValueError):
            Marginal().ai_step(None, y, x, free)


class TestMarginalDtypeAndDevice:
    """dtype and device propagation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_estimates_follow_free_parameter_dtype(self, dtype):
        model, y, x, _ = _dummy_design_model()
        free = torch.tensor(TINY_FREE, dtype=dtype)
        score, ai, loglik = Marginal().ai_step(model, y, x, free)
        assert score.dtype == dtype
        assert ai.dtype == dtype
        assert loglik.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_tensor_kernel_follows_input_dtype(self, dtype):
        z = torch.tensor(TINY_Z, dtype=dtype)
        x = torch.tensor(TINY_X, dtype=dtype)
        y = torch.tensor(TINY_Y, dtype=dtype)
        v = torch.eye(4, dtype=dtype)
        dv = torch.eye(4, dtype=dtype).unsqueeze(0)
        score, ai, loglik = Marginal().ai_step_from_tensors(x, y, v, dv)
        assert score.dtype == dtype
        assert ai.dtype == dtype
        assert loglik.dtype == dtype

    def test_model_aware_step_follows_accelerator(self):
        device = _accelerator()
        model, y, x, _ = _dummy_design_model(dtype=torch.float32)
        free = torch.tensor(TINY_FREE, dtype=torch.float32, device=device)
        score, ai, loglik = Marginal().ai_step(model, y, x, free)
        assert score.device.type == device.type
        assert ai.device.type == device.type
        assert loglik.device.type == device.type


class TestMarginalJohnAlpha:
    """A realistic case on John's alpha design."""

    def test_loglik_matches_explicit_reference(self):
        model, y, x, free = _john_alpha_model()
        _, _, expected = _expected(model, y, x, free)
        actual = Marginal().loglik(model, y, x, free)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-9)

    def test_score_and_ai_match_explicit_reference(self):
        model, y, x, free = _john_alpha_model()
        expected_score, expected_ai, _ = _expected(model, y, x, free)
        score, ai, _ = Marginal().ai_step(model, y, x, free)
        assert score.shape == (2,)
        torch.testing.assert_close(score, expected_score, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(ai, expected_ai, rtol=1e-10, atol=1e-10)

    def test_agrees_with_cform(self):
        model, y, x, free = _john_alpha_model()
        marginal_score, marginal_ai, marginal_loglik = Marginal().ai_step(
            model, y, x, free
        )
        cform_score, cform_ai, cform_loglik = CForm().ai_step(model, y, x, free)
        torch.testing.assert_close(marginal_score, cform_score, rtol=0.0, atol=1e-4)
        torch.testing.assert_close(marginal_ai, cform_ai, rtol=0.0, atol=1e-4)
        torch.testing.assert_close(marginal_loglik, cform_loglik, rtol=0.0, atol=1e-4)

    def test_ai_is_symmetric_and_positive_semidefinite(self):
        model, y, x, free = _john_alpha_model()
        _, ai, _ = Marginal().ai_step(model, y, x, free)
        torch.testing.assert_close(ai, ai.T, rtol=0.0, atol=1e-10)
        eigenvalues = torch.linalg.eigvalsh((ai + ai.T) / 2.0)
        assert torch.all(eigenvalues >= -1e-10)

    def test_loglik_is_finite(self):
        model, y, x, free = _john_alpha_model()
        loglik = Marginal().loglik(model, y, x, free)
        assert torch.isfinite(loglik)
        assert not math.isnan(loglik.item())

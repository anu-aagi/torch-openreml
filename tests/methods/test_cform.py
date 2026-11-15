"""Tests for :class:`torch_openreml.methods.cform.CForm`.

The C-form evaluates the REML score, average-information matrix and
restricted log-likelihood from the coefficient matrix
``C = W' R^{-1} W + blockdiag(0, G^{-1})`` with ``W = [X, Z]``.  The tiny
models below are small enough that every reported quantity can be checked
against an explicit matrix formula, and the model-aware entry points are
checked against ``torch.autograd`` through the differentiable log-likelihood.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import (
    DummyMatrix,
    KnownMatrix,
    ScalarMatrix,
    SimpleMatrix,
)
from torch_openreml.example_data import john_alpha
from torch_openreml.methods import CForm, Marginal

DTYPE = torch.float64

TINY_Z = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
TINY_X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
TINY_Y = [1.0, 2.0, 0.5, 1.5]
TINY_FREE = [0.1, -0.2]


def _tiny_model(dtype=DTYPE):
    """A 4-observation model with a fixed 4x2 design, G = g I_2, R = r I_4."""
    z = torch.tensor(TINY_Z, dtype=dtype)
    x = torch.tensor(TINY_X, dtype=dtype)
    y = torch.tensor(TINY_Y, dtype=dtype)
    model = MixedModelREML((z, ScalarMatrix(2)), R=ScalarMatrix(4))
    free = torch.tensor(TINY_FREE, dtype=dtype)
    return model, y, x, free


def _parameterised_design(params):
    """A 4x2 design whose last row is parameterised by the two free parameters."""
    z = torch.tensor(TINY_Z, dtype=params.dtype, device=params.device)
    z[3, 0] = params[0]
    z[3, 1] = params[1]
    return z


def _parameterised_z_model(dtype=DTYPE):
    """A 4-observation model whose design carries two free parameters."""
    z = SimpleMatrix(2, _parameterised_design)
    model = MixedModelREML((z, ScalarMatrix(2)), R=ScalarMatrix(4))
    free = torch.tensor([0.2, -0.1, 0.1, -0.2], dtype=dtype)
    return model, free


def _dummy_design_model(dtype=DTYPE):
    """A model whose design is a ``Matrix`` and therefore follows the parameters."""
    z = DummyMatrix(["a", "a", "b", "b"])
    model = MixedModelREML((z, ScalarMatrix(2)), R=ScalarMatrix(4))
    x = torch.tensor(TINY_X, dtype=dtype)
    y = torch.tensor(TINY_Y, dtype=dtype)
    free = torch.tensor(TINY_FREE, dtype=dtype)
    return model, y, x, free


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


def _reference_c_form_loglik(x, y, z, g, r):
    """The C-form log-likelihood written out with an explicit ``C``."""
    w = torch.cat([x, z], dim=1)
    p = x.shape[1]
    r_inv = torch.linalg.inv(r)
    c = w.T @ r_inv @ w
    c[p:, p:] += torch.linalg.inv(g)
    w_ry = w.T @ (r_inv @ y)
    quad = y @ (r_inv @ y) - w_ry @ torch.linalg.solve(c, w_ry)
    return -0.5 * (torch.logdet(c) + torch.logdet(r) + torch.logdet(g) + quad)


def _explicit_c_inverse(x, z, g, r):
    """Explicit inverse of the coefficient matrix ``C``."""
    w = torch.cat([x, z], dim=1)
    p = x.shape[1]
    c = w.T @ torch.linalg.inv(r) @ w
    c[p:, p:] += torch.linalg.inv(g)
    return torch.linalg.inv(c)


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


class TestCFormConstructionAndEligibility:
    """Construction, statelessness and the eligibility contract."""

    def test_is_stateless(self):
        cform = CForm()
        assert vars(cform) == {}

    def test_eligible_for_random_effect_model(self):
        model, _, _, _ = _tiny_model()
        assert CForm().eligible(model) is True

    def test_not_eligible_for_none(self):
        assert CForm().eligible(None) is False

    def test_not_eligible_without_random_effect(self):
        model = MixedModelREML(R=ScalarMatrix(4))
        assert CForm().eligible(model) is False

    def test_not_eligible_when_marginal_covariance_has_no_free_parameters(self):
        stub = SimpleNamespace(Z=object(), G=object(), V=SimpleNamespace(num_free_params=0))
        assert CForm().eligible(stub) is False

    def test_eligible_needs_only_one_free_parameter(self):
        stub = SimpleNamespace(Z=object(), G=object(), V=SimpleNamespace(num_free_params=1))
        assert CForm().eligible(stub) is True

    @pytest.mark.parametrize(
        "call",
        [
            lambda cform, model, y, x, free: cform.loglik(model, y, x, free),
            lambda cform, model, y, x, free: cform.ai_step(model, y, x, free),
            lambda cform, model, y, x, free: cform.c_inverse_blocks(model, x, free),
        ],
        ids=["loglik", "ai_step", "c_inverse_blocks"],
    )
    def test_ineligible_model_raises_runtime_error(self, call):
        model, y, x, free = _tiny_model()
        ineligible = MixedModelREML(R=ScalarMatrix(4))
        with pytest.raises(RuntimeError, match="CForm requires at least one random-effect term"):
            call(CForm(), ineligible, y, x, free)

    @pytest.mark.parametrize(
        "call",
        [
            lambda cform, y, x, free: cform.loglik(None, y, x, free),
            lambda cform, y, x, free: cform.ai_step(None, y, x, free),
            lambda cform, y, x, free: cform.c_inverse_blocks(None, x, free),
        ],
        ids=["loglik", "ai_step", "c_inverse_blocks"],
    )
    def test_none_model_raises_runtime_error(self, call):
        _, y, x, free = _tiny_model()
        with pytest.raises(RuntimeError, match="CForm requires at least one random-effect term"):
            call(CForm(), y, x, free)


class TestCFormLoglik:
    """The restricted log-likelihood, model-aware and tensor-level."""

    def test_matches_explicit_c_form_expression(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        expected = _reference_c_form_loglik(x, y, z, g, r)
        actual = CForm().loglik(model, y, x, free)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_matches_explicit_marginal_reml_expression(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        dv = model.V.grad(free)[0]
        _, _, expected = _reference_reml(x, y, v, dv)
        actual = CForm().loglik(model, y, x, free)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_agrees_with_marginal_method(self):
        model, y, x, free = _tiny_model()
        cform_value = CForm().loglik(model, y, x, free)
        marginal_value = Marginal().loglik(model, y, x, free)
        # The marginal path jitters V by 1e-6 I because the composite covariance
        # has no closed-form inverse.
        torch.testing.assert_close(cform_value, marginal_value, rtol=0.0, atol=1e-5)

    def test_supplied_residual_inverse_matches_computed_inverse(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        expected = CForm().loglik_from_tensors(x, y, z, g, r)
        actual = CForm().loglik_from_tensors(x, y, z, g, r, r_inv=torch.linalg.inv(r))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_supplied_random_inverse_matches_computed_inverse(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        expected = CForm().loglik_from_tensors(x, y, z, g, r)
        actual = CForm().loglik_from_tensors(x, y, z, g, r, g_inv=torch.linalg.inv(g))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_supplied_both_inverses_match_computed_inverses(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        expected = CForm().loglik_from_tensors(x, y, z, g, r)
        actual = CForm().loglik_from_tensors(
            x,
            y,
            z,
            g,
            r,
            g_inv=torch.linalg.inv(g),
            r_inv=torch.linalg.inv(r),
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_rejects_mismatched_design_rows(self):
        model, y, x, free = _tiny_model()
        bad_z = torch.zeros(3, 2, dtype=DTYPE)
        g = model.G(free[:1])
        r = model.R(free[1:])
        with pytest.raises(RuntimeError):
            CForm().loglik_from_tensors(x, y, bad_z, g, r)

    def test_rejects_non_positive_definite_residual(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = -torch.eye(4, dtype=DTYPE)
        with pytest.raises(RuntimeError):
            CForm().loglik_from_tensors(x, y, z, g, r)


class TestCFormScoreAndInformation:
    """Score vector and average-information matrix values and invariants."""

    def test_score_shape(self):
        model, y, x, free = _tiny_model()
        score, _, _ = CForm().ai_step(model, y, x, free)
        assert score.shape == (2,)

    @pytest.mark.parametrize("k", [0, 1])
    def test_score_entry_matches_explicit_reference(self, k):
        model, y, x, free = _tiny_model()
        dv = model.V.grad(free)[0]
        expected, _, _ = _reference_reml(x, y, model.V(free), dv)
        score, _, _ = CForm().ai_step(model, y, x, free)
        torch.testing.assert_close(score[k], expected[k], rtol=0.0, atol=1e-12)

    def test_score_matches_autograd_of_loglik(self):
        model, y, x, _ = _tiny_model()
        free = torch.tensor(TINY_FREE, dtype=DTYPE, requires_grad=True)
        loglik = CForm().loglik(model, y, x, free)
        (autograd_score,) = torch.autograd.grad(loglik, free)
        score, _, _ = CForm().ai_step(model, y, x, free.detach())
        torch.testing.assert_close(score, autograd_score, rtol=0.0, atol=1e-12)

    def test_ai_matches_explicit_reference(self):
        model, y, x, free = _tiny_model()
        dv = model.V.grad(free)[0]
        _, expected, _ = _reference_reml(x, y, model.V(free), dv)
        _, ai, _ = CForm().ai_step(model, y, x, free)
        torch.testing.assert_close(ai, expected, rtol=0.0, atol=1e-12)

    def test_ai_is_symmetric(self):
        model, y, x, free = _tiny_model()
        _, ai, _ = CForm().ai_step(model, y, x, free)
        torch.testing.assert_close(ai, ai.T, rtol=0.0, atol=1e-14)

    def test_ai_is_positive_semidefinite(self):
        model, y, x, free = _tiny_model()
        _, ai, _ = CForm().ai_step(model, y, x, free)
        eigenvalues = torch.linalg.eigvalsh((ai + ai.T) / 2.0)
        assert torch.all(eigenvalues >= -1e-12)

    def test_require_loglik_false_skips_loglik_but_keeps_estimates(self):
        model, y, x, free = _tiny_model()
        score, ai, loglik = CForm().ai_step(model, y, x, free)
        short_score, short_ai, short_loglik = CForm().ai_step(
            model, y, x, free, require_loglik=False
        )
        assert torch.isnan(torch.as_tensor(short_loglik))
        torch.testing.assert_close(short_score, score, rtol=0.0, atol=0.0)
        torch.testing.assert_close(short_ai, ai, rtol=0.0, atol=0.0)

    def test_parameter_order_is_z_g_r(self):
        model, free = _parameterised_z_model()
        assert model.V.free_param_names == [
            "Z/theta_0",
            "Z/theta_1",
            "G/sigma^2",
            "R/sigma^2",
        ]
        _, y, x, _ = _tiny_model()
        dv = model.V.grad(free)[0]
        expected_score, expected_ai, _ = _reference_reml(x, y, model.V(free), dv)
        score, ai, _ = CForm().ai_step(model, y, x, free)
        torch.testing.assert_close(score, expected_score, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(ai, expected_ai, rtol=0.0, atol=1e-12)

    def test_all_empty_derivative_blocks_raise(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        dz = z.new_zeros((0, 4, 2))
        dg = g.new_zeros((0, 2, 2))
        dr = r.new_zeros((0, 4, 4))
        with pytest.raises(ValueError, match="at least one of dz, dg, or dr"):
            CForm().ai_step_from_tensors(x, y, z, g, r, dz, dg, dr)

    def test_mismatched_derivative_shape_raises(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        dz = z.new_zeros((0, 4, 2))
        dg = g.new_zeros((1, 3, 3))
        dr = r.new_zeros((0, 4, 4))
        with pytest.raises(RuntimeError):
            CForm().ai_step_from_tensors(x, y, z, g, r, dz, dg, dr)

    def test_known_random_effect_contributes_no_score_entry(self):
        z = torch.tensor(TINY_Z, dtype=DTYPE)
        model = MixedModelREML((z, KnownMatrix(torch.eye(2, dtype=DTYPE))), R=ScalarMatrix(4))
        free = torch.tensor([0.1], dtype=DTYPE)
        _, y, x, _ = _tiny_model()
        dv = model.V.grad(free)[0]
        expected_score, expected_ai, _ = _reference_reml(x, y, model.V(free), dv)
        score, ai, _ = CForm().ai_step(model, y, x, free)
        assert score.shape == (1,)
        torch.testing.assert_close(score, expected_score, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(ai, expected_ai, rtol=0.0, atol=1e-12)

    def test_known_residual_contributes_no_score_entry(self):
        z = torch.tensor(TINY_Z, dtype=DTYPE)
        model = MixedModelREML(
            (z, ScalarMatrix(2)), R=KnownMatrix(torch.eye(4, dtype=DTYPE))
        )
        free = torch.tensor([0.1], dtype=DTYPE)
        _, y, x, _ = _tiny_model()
        dv = model.V.grad(free)[0]
        expected_score, expected_ai, _ = _reference_reml(x, y, model.V(free), dv)
        score, ai, _ = CForm().ai_step(model, y, x, free)
        assert score.shape == (1,)
        torch.testing.assert_close(score, expected_score, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(ai, expected_ai, rtol=0.0, atol=1e-12)

    def test_design_only_parameters_contribute_score_and_ai(self):
        model = MixedModelREML(
            (SimpleMatrix(2, _parameterised_design), KnownMatrix(torch.eye(2, dtype=DTYPE))),
            R=KnownMatrix(torch.eye(4, dtype=DTYPE)),
        )
        free = torch.tensor([0.2, -0.1], dtype=DTYPE)
        _, y, x, _ = _tiny_model()
        dv = model.V.grad(free)[0]
        expected_score, expected_ai, _ = _reference_reml(x, y, model.V(free), dv)
        score, ai, _ = CForm().ai_step(model, y, x, free)
        assert score.shape == (2,)
        torch.testing.assert_close(score, expected_score, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(ai, expected_ai, rtol=0.0, atol=1e-12)


class TestCFormInverseBlocks:
    """The blocks of the inverse coefficient matrix."""

    def test_block_shapes(self):
        model, _, x, free = _tiny_model()
        blocks = CForm().c_inverse_blocks(model, x, free)
        assert [block.shape for block in blocks] == [(2, 2), (2, 2), (2, 2), (2, 2)]

    def test_blocks_match_explicit_inverse(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        expected = _explicit_c_inverse(x, z, g, r)
        c11, c12, c21, c22 = CForm().c_inverse_blocks(model, x, free)
        actual = torch.cat(
            [torch.cat([c11, c12], dim=1), torch.cat([c21, c22], dim=1)], dim=0
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_top_left_block_is_fixed_effect_covariance(self):
        model, y, x, free = _tiny_model()
        v = model.V(free)
        expected = torch.linalg.inv(x.T @ torch.linalg.inv(v) @ x)
        c11 = CForm().c_inverse_blocks(model, x, free)[0]
        torch.testing.assert_close(c11, expected, rtol=0.0, atol=1e-12)

    def test_cross_blocks_are_transposes(self):
        model, y, x, free = _tiny_model()
        _, c12, c21, _ = CForm().c_inverse_blocks(model, x, free)
        torch.testing.assert_close(c12, c21.T, rtol=0.0, atol=1e-14)

    def test_supplied_inverses_match_computed_inverses(self):
        model, y, x, free = _tiny_model()
        z = model.Z
        g = model.G(free[:1])
        r = model.R(free[1:])
        expected = CForm().c_inverse_blocks_from_tensors(x, z, g, r)
        actual = CForm().c_inverse_blocks_from_tensors(
            x,
            z,
            g,
            r,
            g_inv=torch.linalg.inv(g),
            r_inv=torch.linalg.inv(r),
        )
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0.0, atol=1e-12)

    def test_jitter_applied_when_residual_has_no_closed_form_inverse(self):
        def r_call(params):
            return torch.eye(4, dtype=params.dtype, device=params.device) * torch.exp(
                2.0 * params[0]
            )

        z = torch.tensor(TINY_Z, dtype=DTYPE)
        model = MixedModelREML((z, ScalarMatrix(2)), R=SimpleMatrix(1, r_call))
        free = torch.tensor([0.1, -0.2], dtype=DTYPE)
        g = model.G(free[:1])
        r = model.R(free[1:]) + 1e-6 * torch.eye(4, dtype=DTYPE)
        x = torch.tensor(TINY_X, dtype=DTYPE)
        expected = _explicit_c_inverse(x, z, g, r)
        c11, c12, c21, c22 = CForm().c_inverse_blocks(model, x, free)
        actual = torch.cat(
            [torch.cat([c11, c12], dim=1), torch.cat([c21, c22], dim=1)], dim=0
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_dtype_follows_free_params(self):
        model, y, x, _ = _dummy_design_model()
        free = torch.tensor(TINY_FREE, dtype=torch.float32)
        blocks = CForm().c_inverse_blocks(model, x, free)
        assert all(block.dtype == torch.float32 for block in blocks)


class TestCFormPreprocessInput:
    """The tensor-extraction contract behind ``ai_step``."""

    def test_keys_with_grad(self):
        model, y, x, free = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free)
        assert set(data) == {"x", "y", "z", "g", "r", "g_inv", "r_inv", "dz", "dg", "dr"}

    def test_keys_without_grad(self):
        model, y, x, free = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free, grad=False)
        assert set(data) == {"x", "y", "z", "g", "r", "g_inv", "r_inv"}

    def test_derivative_shapes_for_tiny_model(self):
        model, y, x, free = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free)
        assert data["dz"].shape == (0, 4, 2)
        assert data["dg"].shape == (1, 2, 2)
        assert data["dr"].shape == (1, 4, 4)

    def test_derivative_shapes_for_parameterised_design(self):
        model, free = _parameterised_z_model()
        _, y, x, _ = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free)
        assert data["dz"].shape == (2, 4, 2)
        assert data["dg"].shape == (1, 2, 2)
        assert data["dr"].shape == (1, 4, 4)

    def test_casts_response_and_design_to_free_parameter_dtype(self):
        model, _, x, _ = _dummy_design_model()
        y = torch.tensor(TINY_Y, dtype=torch.float64)
        free = torch.tensor(TINY_FREE, dtype=torch.float32)
        data = CForm().preprocess_input(model, y, x, free)
        assert data["y"].dtype == torch.float32
        assert data["x"].dtype == torch.float32

    def test_jitters_residual_only_when_inverse_missing(self):
        def r_call(params):
            return torch.eye(4, dtype=params.dtype, device=params.device) * torch.exp(
                2.0 * params[0]
            )

        z = torch.tensor(TINY_Z, dtype=DTYPE)
        model = MixedModelREML((z, ScalarMatrix(2)), R=SimpleMatrix(1, r_call))
        free = torch.tensor([0.1, -0.2], dtype=DTYPE)
        _, y, x, _ = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free)
        assert data["r_inv"] is None
        expected = model.R(free[1:]) + 1e-6 * torch.eye(4, dtype=DTYPE)
        torch.testing.assert_close(data["r"], expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(data["g"], model.G(free[:1]), rtol=0.0, atol=0.0)

    def test_does_not_jitter_when_residual_inverse_available(self):
        model, y, x, free = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free)
        assert data["r_inv"] is not None
        torch.testing.assert_close(data["r"], model.R(free[1:]), rtol=0.0, atol=0.0)

    def test_missing_random_effect_inverse_yields_none(self):
        def g_call(params):
            return torch.eye(2, dtype=params.dtype, device=params.device) * torch.exp(
                2.0 * params[0]
            )

        z = torch.tensor(TINY_Z, dtype=DTYPE)
        model = MixedModelREML((z, SimpleMatrix(1, g_call)), R=ScalarMatrix(4))
        free = torch.tensor([0.1, -0.2], dtype=DTYPE)
        _, y, x, _ = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free)
        assert data["g_inv"] is None
        torch.testing.assert_close(data["g"], model.G(free[:1]), rtol=0.0, atol=0.0)

    def test_splits_parameters_in_z_g_r_order(self):
        model, y, x, free = _tiny_model()
        data = CForm().preprocess_input(model, y, x, free)
        torch.testing.assert_close(data["z"], model.Z, rtol=0.0, atol=0.0)
        torch.testing.assert_close(data["g"], model.G(free[:1]), rtol=0.0, atol=0.0)
        torch.testing.assert_close(data["r"], model.R(free[1:]), rtol=0.0, atol=0.0)


class TestCFormDtypeAndDevice:
    """dtype and device propagation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_estimates_follow_free_parameter_dtype(self, dtype):
        model, y, x, _ = _dummy_design_model()
        free = torch.tensor(TINY_FREE, dtype=dtype)
        score, ai, loglik = CForm().ai_step(model, y, x, free)
        assert score.dtype == dtype
        assert ai.dtype == dtype
        assert loglik.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_tensor_kernel_follows_input_dtype(self, dtype):
        z = torch.tensor(TINY_Z, dtype=dtype)
        x = torch.tensor(TINY_X, dtype=dtype)
        y = torch.tensor(TINY_Y, dtype=dtype)
        g = torch.eye(2, dtype=dtype)
        r = torch.eye(4, dtype=dtype)
        dz = z.new_zeros((0, 4, 2))
        dg = g.new_zeros((1, 2, 2))
        dr = r.new_zeros((1, 4, 4))
        score, ai, loglik = CForm().ai_step_from_tensors(
            x, y, z, g, r, dz, dg, dr, require_loglik=True
        )
        assert score.dtype == dtype
        assert ai.dtype == dtype
        assert loglik.dtype == dtype

    def test_model_aware_step_follows_accelerator(self):
        device = _accelerator()
        model, y, x, _ = _dummy_design_model(dtype=torch.float32)
        free = torch.tensor(TINY_FREE, dtype=torch.float32, device=device)
        score, ai, loglik = CForm().ai_step(model, y, x, free)
        assert score.device.type == device.type
        assert ai.device.type == device.type
        assert loglik.device.type == device.type

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "CForm casts x and y to the free-parameter dtype but leaves a fixed-tensor Z "
            "alone, so a float32 parameter tensor with a float64 design raises instead of "
            "following the parameters"
        ),
    )
    def test_fixed_tensor_design_follows_free_parameter_dtype(self):
        model, y, x, _ = _tiny_model()
        free = torch.tensor(TINY_FREE, dtype=torch.float32)
        score, _, _ = CForm().ai_step(model, y, x, free)
        assert score.dtype == torch.float32


class TestCFormJohnAlpha:
    """A realistic case on John's alpha design."""

    def test_loglik_matches_explicit_formula(self):
        model, y, x, free = _john_alpha_model()
        z = model.Z(free[:0])
        g = model.G(free[:1])
        r = model.R(free[1:])
        expected = _reference_c_form_loglik(x, y, z, g, r)
        actual = CForm().loglik(model, y, x, free)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-9)

    def test_score_and_ai_match_explicit_formula(self):
        model, y, x, free = _john_alpha_model()
        dv = model.V.grad(free)[0]
        expected_score, expected_ai, _ = _reference_reml(x, y, model.V(free), dv)
        score, ai, _ = CForm().ai_step(model, y, x, free)
        assert score.shape == (2,)
        torch.testing.assert_close(score, expected_score, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(ai, expected_ai, rtol=1e-10, atol=1e-10)

    def test_agrees_with_marginal_method(self):
        model, y, x, free = _john_alpha_model()
        cform_score, cform_ai, cform_loglik = CForm().ai_step(model, y, x, free)
        marginal_score, marginal_ai, marginal_loglik = Marginal().ai_step(
            model, y, x, free
        )
        torch.testing.assert_close(cform_score, marginal_score, rtol=0.0, atol=1e-4)
        torch.testing.assert_close(cform_ai, marginal_ai, rtol=0.0, atol=1e-4)
        torch.testing.assert_close(cform_loglik, marginal_loglik, rtol=0.0, atol=1e-4)

    def test_ai_is_symmetric_and_positive_semidefinite(self):
        model, y, x, free = _john_alpha_model()
        _, ai, _ = CForm().ai_step(model, y, x, free)
        torch.testing.assert_close(ai, ai.T, rtol=0.0, atol=1e-10)
        eigenvalues = torch.linalg.eigvalsh((ai + ai.T) / 2.0)
        assert torch.all(eigenvalues >= -1e-10)

    def test_loglik_is_finite(self):
        model, y, x, free = _john_alpha_model()
        loglik = CForm().loglik(model, y, x, free)
        assert torch.isfinite(loglik)
        assert not math.isnan(loglik.item())

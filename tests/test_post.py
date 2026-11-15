"""Tests for :mod:`torch_openreml.post`.

The functions in ``post.py`` are pure tensor computations: they take a response
``y``, design matrices ``X`` and ``Z`` and covariance matrices ``G`` and ``V``,
and return BLUEs, BLUPs, predictions, residuals, their covariances, and the
REML log-likelihood.  The tests below pin the documented formulas on small
hand-computable fixtures, the cross-function identities, the dtype/device
contract, and the error paths that the implementations inherit from
``torch.linalg`` and ``torch_openreml.utils``.

Two implementation details shape the assertions:

* every function adds a fixed ``1e-6 * I`` ridge to ``V`` before factorising
  it (``v + 1e-6 * torch.eye(...)``).  Formula-level tests therefore compare
  against the textbook expression in ``V`` with a ``1e-5`` tolerance, while
  identity tests that exercise two functions at once use ``1e-10`` because
  both sides see the same ridge.
* ``blue`` and ``predict`` end in ``Tensor.squeeze(-1)``, so a single fixed
  effect (``p == 1``) or a single observation (``n == 1``) keeps the documented
  ``(p,)`` / ``(n,)`` shape instead of collapsing to 0-d.
"""

import inspect
import math

import pytest
import torch

import torch_openreml
import torch_openreml.post as post
from torch_openreml.post import (
    blue,
    blup,
    cov_blue,
    cov_blup,
    loglik,
    marginal_predict,
    marginal_residual,
    predict,
    residual,
    var_blue,
    var_blup,
)

# Tight tolerance for identities where both sides share the same ridge.
EXACT = {"rtol": 1e-10, "atol": 1e-10}
# Formula-level tolerance: accommodates the undocumented 1e-6 ridge on V.
FORMULA = {"rtol": 1e-5, "atol": 1e-5}
# Accelerator runs use float32 comparisons against CPU float32.
ACCEL = {"rtol": 1e-3, "atol": 1e-3}

ALL_FUNCS = (
    "blue",
    "cov_blue",
    "var_blue",
    "blup",
    "cov_blup",
    "var_blup",
    "loglik",
    "marginal_predict",
    "marginal_residual",
    "predict",
    "residual",
)

Y_FUNCS = (
    "blue",
    "blup",
    "loglik",
    "marginal_predict",
    "marginal_residual",
    "predict",
    "residual",
)

ZG_FUNCS = (
    "blup",
    "cov_blup",
    "var_blup",
    "predict",
    "residual",
)


# --------------------------------------------------------------------------- #
# Fixtures and reference implementations
# --------------------------------------------------------------------------- #


def _lmm_fixture(dtype=torch.float64):
    """A small, well-conditioned linear mixed model with n=6, p=2, q=2."""
    y = torch.tensor([1.2, 2.1, 1.7, 3.0, 4.2, 4.8], dtype=dtype)
    x = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
        ],
        dtype=dtype,
    )
    z = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=dtype,
    )
    g = torch.tensor([[0.6, 0.2], [0.2, 0.4]], dtype=dtype)
    r = torch.diag(torch.tensor([0.5, 0.7, 0.6, 0.8, 0.9, 0.7], dtype=dtype))
    v = z @ g @ z.T + r
    return y, x, z, g, r, v


def _exact_fixture(dtype=torch.float64):
    """n=4, p=2, V=I, with X'X diagonal and residuals orthogonal to X.

    The columns of ``X`` are orthogonal (intercept and a centred linear
    contrast), and ``e`` is orthogonal to both, so the GLS estimate is exactly
    ``beta_true`` and ``e'e = 1``.  This yields hand-checkable values for
    ``blue`` (``[2, 0.5]``), ``marginal_residual`` (``e``) and ``loglik``
    (``-0.5 * (log(20) + 1)``).
    """
    x = torch.tensor(
        [[1.0, -1.5], [1.0, -0.5], [1.0, 0.5], [1.0, 1.5]], dtype=dtype
    )
    beta_true = torch.tensor([2.0, 0.5], dtype=dtype)
    e = torch.tensor([0.5, -0.5, -0.5, 0.5], dtype=dtype)
    y = x @ beta_true + e
    v = torch.eye(4, dtype=dtype)
    return y, x, beta_true, e, v


def _gls_beta(y, x, v):
    """Textbook GLS solution, used as the reference for :func:`blue`."""
    v_inv = torch.linalg.inv(v)
    return torch.linalg.solve(x.T @ v_inv @ x, x.T @ v_inv @ y)


def _p_matrix(x, v):
    """Projection matrix P = V^{-1} - V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}."""
    v_inv = torch.linalg.inv(v)
    x_v_inv_x_inv = torch.linalg.inv(x.T @ v_inv @ x)
    return v_inv - v_inv @ x @ x_v_inv_x_inv @ x.T @ v_inv


def _explicit_loglik(y, x, v):
    """REML log-likelihood from the textbook expression, in the given V."""
    v_inv = torch.linalg.inv(v)
    beta = _gls_beta(y, x, v)
    e = y - x @ beta
    _, logdet_v = torch.linalg.slogdet(v)
    _, logdet_x = torch.linalg.slogdet(x.T @ v_inv @ x)
    return -0.5 * (logdet_v + logdet_x + e @ v_inv @ e)


def _ridge(v):
    """The exact matrix the library factorises: V plus its 1e-6 ridge."""
    n = v.shape[0]
    return v + 1e-6 * torch.eye(n, dtype=v.dtype, device=v.device)


def _call(name, y, x, z, g, v):
    """Dispatch by function name so dtype/device tests can be parametrised."""
    if name == "blue":
        return blue(y, x, v)
    if name == "cov_blue":
        return cov_blue(x, v)
    if name == "var_blue":
        return var_blue(x, v)
    if name == "blup":
        return blup(y, x, z, g, v)
    if name == "cov_blup":
        return cov_blup(x, z, g, v)
    if name == "var_blup":
        return var_blup(x, z, g, v)
    if name == "loglik":
        return loglik(y, x, v)
    if name == "marginal_predict":
        return marginal_predict(y, x, v)
    if name == "marginal_residual":
        return marginal_residual(y, x, v)
    if name == "predict":
        return predict(y, x, z, g, v)
    if name == "residual":
        return residual(y, x, z, g, v)
    raise AssertionError(f"unknown post function: {name}")


def _accelerator():
    """Pick an accelerator in the suite's documented order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("no accelerator available")


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


class TestPublicSurface:
    """The module's callables and their exact positional signatures."""

    def test_package_reexports_the_documented_post_functions(self):
        # Arrange
        documented = (
            "blue",
            "blup",
            "loglik",
            "marginal_predict",
            "marginal_residual",
            "predict",
            "residual",
        )

        # Act / Assert
        for name in documented:
            assert callable(getattr(torch_openreml, name)), name

    def test_post_module_exposes_all_public_functions(self):
        # Arrange
        documented = ALL_FUNCS

        # Act / Assert
        for name in documented:
            assert callable(getattr(post, name)), name

    @pytest.mark.parametrize(
        "func_name, expected_params",
        [
            ("blue", ["y", "x", "v"]),
            ("cov_blue", ["x", "v"]),
            ("var_blue", ["x", "v"]),
            ("blup", ["y", "x", "z", "g", "v"]),
            ("cov_blup", ["x", "z", "g", "v"]),
            ("var_blup", ["x", "z", "g", "v"]),
            ("loglik", ["y", "x", "v"]),
            ("marginal_predict", ["y", "x", "v"]),
            ("marginal_residual", ["y", "x", "v"]),
            ("predict", ["y", "x", "z", "g", "v"]),
            ("residual", ["y", "x", "z", "g", "v"]),
        ],
    )
    def test_exact_signature(self, func_name, expected_params):
        # Arrange
        func = getattr(post, func_name)

        # Act
        parameters = inspect.signature(func).parameters

        # Assert
        assert list(parameters) == expected_params
        assert all(p.default is inspect.Parameter.empty for p in parameters.values())


# --------------------------------------------------------------------------- #
# blue
# --------------------------------------------------------------------------- #


class TestBlueValues:
    """BLUE values, invariance properties and the single-coefficient edge."""

    def test_exact_orthogonal_fixture(self):
        # Arrange
        y, x, beta_true, _, v = _exact_fixture()

        # Act
        beta = blue(y, x, v)

        # Assert
        assert beta.shape == (2,)
        assert beta.dtype == torch.float64
        assert torch.allclose(beta, beta_true, **FORMULA)

    def test_matches_explicit_gls_formula(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        expected = _gls_beta(y, x, v)

        # Act
        beta = blue(y, x, v)

        # Assert
        assert torch.allclose(beta, expected, **FORMULA)

    def test_weighted_least_squares_with_diagonal_v(self):
        # Arrange
        x = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=torch.float64
        )
        y = torch.tensor([1.0, 3.0, 2.0, 5.0], dtype=torch.float64)
        v = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        w = 1.0 / torch.diagonal(v)
        slope_x = x[:, 1]
        sw = w.sum()
        swx = (w * slope_x).sum()
        swy = (w * y).sum()
        swxy = (w * slope_x * y).sum()
        swxx = (w * slope_x * slope_x).sum()
        det = sw * swxx - swx * swx
        expected = torch.stack(
            [(swxx * swy - swx * swxy) / det, (sw * swxy - swx * swy) / det]
        )

        # Act
        beta = blue(y, x, v)

        # Assert
        assert torch.allclose(beta, expected, **FORMULA)

    def test_scale_invariance_in_v(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        beta_plain = blue(y, x, v)

        # Act
        beta_scaled = blue(y, x, 3.5 * v)

        # Assert
        assert torch.allclose(beta_plain, beta_scaled, **FORMULA)

    def test_constant_response_recovers_intercept(self):
        # Arrange
        _, x, _, _, v = _exact_fixture()
        y = torch.full((4,), 3.0, dtype=torch.float64)
        expected = torch.tensor([3.0, 0.0], dtype=torch.float64)

        # Act
        beta = blue(y, x, v)

        # Assert
        assert torch.allclose(beta, expected, **FORMULA)

    def test_single_fixed_effect_value(self):
        # Arrange
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones(4, 1, dtype=torch.float64)
        v = torch.eye(4, dtype=torch.float64)

        # Act
        beta = blue(y, x, v)

        # Assert: the value is the sample mean; only the documented shape is wrong.
        assert torch.allclose(beta.reshape(1), torch.tensor([2.5], dtype=torch.float64))

    def test_single_fixed_effect_returns_1d(self):
        # Arrange
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones(4, 1, dtype=torch.float64)
        v = torch.eye(4, dtype=torch.float64)

        # Act
        beta = blue(y, x, v)

        # Assert
        assert beta.shape == (1,)

    def test_no_fixed_effects_returns_empty(self):
        # Arrange
        y, _, _, _, _, v = _lmm_fixture()
        x = torch.zeros(6, 0, dtype=torch.float64)

        # Act
        beta = blue(y, x, v)

        # Assert
        assert beta.shape == (0,)
        assert beta.dtype == torch.float64

    def test_row_permutation_invariance(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        permutation = torch.tensor([2, 4, 0, 5, 1, 3])
        beta = blue(y, x, v)

        # Act
        beta_permuted = blue(y[permutation], x[permutation], v[permutation][:, permutation])

        # Assert
        assert torch.allclose(beta, beta_permuted, **EXACT)


# --------------------------------------------------------------------------- #
# cov_blue / var_blue
# --------------------------------------------------------------------------- #


class TestCovBlueValues:
    """(X'V^{-1}X)^{-1}, its diagonal and its structural properties."""

    def test_matches_explicit_information_inverse(self):
        # Arrange
        _, x, _, _, _, v = _lmm_fixture()
        expected = torch.linalg.inv(x.T @ torch.linalg.inv(v) @ x)

        # Act
        cov = cov_blue(x, v)

        # Assert
        assert cov.shape == (2, 2)
        assert torch.allclose(cov, expected, **FORMULA)

    def test_is_symmetric(self):
        # Arrange
        _, x, _, _, _, v = _lmm_fixture()

        # Act
        cov = cov_blue(x, v)

        # Assert
        assert torch.allclose(cov, cov.T, **EXACT)

    def test_var_blue_is_the_diagonal(self):
        # Arrange
        _, x, _, _, _, v = _lmm_fixture()

        # Act
        cov = cov_blue(x, v)
        variances = var_blue(x, v)

        # Assert
        assert variances.shape == (2,)
        assert torch.allclose(variances, torch.diagonal(cov), **EXACT)

    def test_var_blue_matches_explicit_formula(self):
        # Arrange
        _, x, _, _, _, v = _lmm_fixture()
        expected = torch.diagonal(torch.linalg.inv(x.T @ torch.linalg.inv(v) @ x))

        # Act
        variances = var_blue(x, v)

        # Assert
        assert torch.allclose(variances, expected, **FORMULA)

    def test_diagonal_is_positive(self):
        # Arrange
        _, x, _, _, _, v = _lmm_fixture()

        # Act
        variances = var_blue(x, v)

        # Assert
        assert torch.all(variances > 0)

    def test_single_fixed_effect_shapes(self):
        # Arrange
        x = torch.ones(4, 1, dtype=torch.float64)
        v = torch.eye(4, dtype=torch.float64)

        # Act
        cov = cov_blue(x, v)
        variances = var_blue(x, v)

        # Assert
        assert cov.shape == (1, 1)
        assert variances.shape == (1,)
        assert torch.allclose(variances, torch.tensor([0.25], dtype=torch.float64), **FORMULA)

    def test_no_fixed_effects_shapes(self):
        # Arrange
        _, _, _, _, _, v = _lmm_fixture()
        x = torch.zeros(6, 0, dtype=torch.float64)

        # Act
        cov = cov_blue(x, v)
        variances = var_blue(x, v)

        # Assert
        assert cov.shape == (0, 0)
        assert variances.shape == (0,)

    def test_scales_linearly_with_v(self):
        # Arrange
        _, x, _, _, _, v = _lmm_fixture()
        cov_plain = cov_blue(x, v)

        # Act
        cov_scaled = cov_blue(x, 3.5 * v)

        # Assert
        assert torch.allclose(cov_scaled, 3.5 * cov_plain, **FORMULA)


# --------------------------------------------------------------------------- #
# loglik
# --------------------------------------------------------------------------- #


class TestLoglikValues:
    """REML log-likelihood values and its quadratic-form decomposition."""

    def test_exact_orthogonal_fixture(self):
        # Arrange
        y, x, _, _, v = _exact_fixture()
        expected = -0.5 * (math.log(20.0) + 1.0)

        # Act
        value = loglik(y, x, v)

        # Assert
        assert value.shape == ()
        assert value.dtype == torch.float64
        assert torch.allclose(value, torch.tensor(expected, dtype=torch.float64), **FORMULA)

    def test_matches_explicit_reml_formula(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        expected = _explicit_loglik(y, x, v)

        # Act
        value = loglik(y, x, v)

        # Assert
        assert torch.allclose(value, expected, **FORMULA)

    def test_matches_explicit_quadratic_form_with_p_matrix(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        p_matrix = _p_matrix(x, v)
        _, logdet_v = torch.linalg.slogdet(v)
        _, logdet_x = torch.linalg.slogdet(x.T @ torch.linalg.inv(v) @ x)
        expected = -0.5 * (logdet_v + logdet_x + y @ p_matrix @ y)

        # Act
        value = loglik(y, x, v)

        # Assert
        assert torch.allclose(value, expected, **FORMULA)

    def test_single_fixed_effect_hand_value(self):
        # Arrange
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones(4, 1, dtype=torch.float64)
        v = torch.eye(4, dtype=torch.float64)
        # beta = 2.5, e = (-1.5, -0.5, 0.5, 1.5), e'e = 5, |X'X| = 4.
        expected = -0.5 * (math.log(4.0) + 5.0)

        # Act
        value = loglik(y, x, v)

        # Assert
        assert torch.allclose(value, torch.tensor(expected, dtype=torch.float64), **FORMULA)

    def test_no_fixed_effects_reduces_to_marginal_likelihood(self):
        # Arrange
        y, _, _, _, _, v = _lmm_fixture()
        x = torch.zeros(6, 0, dtype=torch.float64)
        v_inv = torch.linalg.inv(v)
        expected = -0.5 * (torch.linalg.slogdet(v)[1] + y @ v_inv @ y)

        # Act
        value = loglik(y, x, v)

        # Assert
        assert torch.allclose(value, expected, **FORMULA)

    def test_quadratic_form_equals_marginal_residual_inner_product(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        v_ridge = _ridge(v)
        v_inv = torch.linalg.inv(v_ridge)
        e = marginal_residual(y, x, v)
        expected = -0.5 * (
            torch.linalg.slogdet(v_ridge)[1]
            + torch.linalg.slogdet(x.T @ v_inv @ x)[1]
            + e @ v_inv @ e
        )

        # Act
        value = loglik(y, x, v)

        # Assert
        assert torch.allclose(value, expected, **EXACT)


# --------------------------------------------------------------------------- #
# marginal_predict / marginal_residual
# --------------------------------------------------------------------------- #


class TestMarginalValues:
    """Fixed-effect predictions and residuals."""

    def test_marginal_predict_is_x_times_blue(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        beta = blue(y, x, v)

        # Act
        fitted = marginal_predict(y, x, v)

        # Assert
        assert fitted.shape == (6,)
        assert torch.allclose(fitted, x @ beta, **EXACT)

    def test_exact_orthogonal_fixture(self):
        # Arrange
        y, x, beta, e, v = _exact_fixture()
        expected_fitted = x @ beta

        # Act
        fitted = marginal_predict(y, x, v)
        residuals = marginal_residual(y, x, v)

        # Assert
        assert torch.allclose(fitted, expected_fitted, **FORMULA)
        assert torch.allclose(residuals, e, **FORMULA)

    def test_marginal_residual_is_y_minus_fitted(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()

        # Act
        residuals = marginal_residual(y, x, v)

        # Assert
        assert residuals.shape == (6,)
        assert torch.allclose(residuals, y - marginal_predict(y, x, v), **EXACT)

    def test_residual_is_orthogonal_to_design_in_v_metric(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        v_inv = torch.linalg.inv(_ridge(v))
        residuals = marginal_residual(y, x, v)

        # Act
        normal_equations = x.T @ v_inv @ residuals

        # Assert
        assert torch.allclose(
            normal_equations, torch.zeros(2, dtype=torch.float64), rtol=0.0, atol=1e-10
        )

    def test_no_fixed_effects(self):
        # Arrange
        y, _, _, _, _, v = _lmm_fixture()
        x = torch.zeros(6, 0, dtype=torch.float64)

        # Act
        fitted = marginal_predict(y, x, v)
        residuals = marginal_residual(y, x, v)

        # Assert
        assert fitted.shape == (6,)
        assert torch.allclose(fitted, torch.zeros(6, dtype=torch.float64), **EXACT)
        assert torch.allclose(residuals, y, **EXACT)

    def test_single_fixed_effect_predict(self):
        # Arrange
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones(4, 1, dtype=torch.float64)
        v = torch.eye(4, dtype=torch.float64)

        # Act
        fitted = marginal_predict(y, x, v)

        # Assert
        assert fitted.shape == (4,)

    def test_single_fixed_effect_residual(self):
        # Arrange
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones(4, 1, dtype=torch.float64)
        v = torch.eye(4, dtype=torch.float64)

        # Act
        residuals = marginal_residual(y, x, v)

        # Assert
        assert residuals.shape == (4,)


# --------------------------------------------------------------------------- #
# blup
# --------------------------------------------------------------------------- #


class TestBlupValues:
    """BLUP values G Z' V^{-1} e and the zero/singular edge cases."""

    def test_matches_explicit_formula(self):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()
        v_inv = torch.linalg.inv(v)
        beta = _gls_beta(y, x, v)
        expected = g @ z.T @ v_inv @ (y - x @ beta)

        # Act
        random_effects = blup(y, x, z, g, v)

        # Assert
        assert random_effects.shape == (2,)
        assert random_effects.dtype == torch.float64
        assert torch.allclose(random_effects, expected, **FORMULA)

    def test_matches_formula_using_marginal_residual(self):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()
        v_inv = torch.linalg.inv(_ridge(v))
        expected = g @ z.T @ v_inv @ (y - marginal_predict(y, x, v))

        # Act
        random_effects = blup(y, x, z, g, v)

        # Assert
        assert torch.allclose(random_effects, expected, **EXACT)

    def test_zero_g_returns_zeros(self):
        # Arrange
        y, x, z, _, _, v = _lmm_fixture()
        g = torch.zeros(2, 2, dtype=torch.float64)

        # Act
        random_effects = blup(y, x, z, g, v)

        # Assert
        assert random_effects.shape == (2,)
        assert torch.allclose(random_effects, torch.zeros(2, dtype=torch.float64), **EXACT)

    def test_zero_random_effects_returns_empty(self):
        # Arrange
        y, x, _, _, r, _ = _lmm_fixture()
        z = torch.zeros(6, 0, dtype=torch.float64)
        g = torch.zeros(0, 0, dtype=torch.float64)

        # Act
        random_effects = blup(y, x, z, g, r)

        # Assert
        assert random_effects.shape == (0,)

    def test_no_fixed_effects(self):
        # Arrange
        y, _, z, g, _, v = _lmm_fixture()
        x = torch.zeros(6, 0, dtype=torch.float64)
        expected = g @ z.T @ torch.linalg.inv(v) @ y

        # Act
        random_effects = blup(y, x, z, g, v)

        # Assert
        assert torch.allclose(random_effects, expected, **FORMULA)

    def test_singular_g_is_allowed_and_matches_formula(self):
        # Arrange
        y, x, z, _, _, v = _lmm_fixture()
        g = torch.ones(2, 2, dtype=torch.float64)  # rank one
        expected = g @ z.T @ torch.linalg.inv(v) @ (y - x @ _gls_beta(y, x, v))

        # Act
        random_effects = blup(y, x, z, g, v)

        # Assert
        assert torch.allclose(random_effects, expected, **FORMULA)

    def test_single_random_effect_shape_and_value(self):
        # Arrange
        y, x, _, _, r, _ = _lmm_fixture()
        z = torch.tensor(
            [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], dtype=torch.float64
        )
        g = torch.tensor([[0.6]], dtype=torch.float64)
        v = z @ g @ z.T + r
        expected = g @ z.T @ torch.linalg.inv(v) @ (y - x @ _gls_beta(y, x, v))

        # Act
        random_effects = blup(y, x, z, g, v)

        # Assert
        assert random_effects.shape == (1,)
        assert torch.allclose(random_effects, expected, **FORMULA)

    def test_single_fixed_effect(self):
        # Arrange
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones(4, 1, dtype=torch.float64)
        z = torch.ones(4, 1, dtype=torch.float64)
        g = torch.tensor([[0.5]], dtype=torch.float64)
        v = z @ g @ z.T + torch.eye(4, dtype=torch.float64)

        # Act
        random_effects = blup(y, x, z, g, v)

        # Assert
        assert random_effects.shape == (1,)


# --------------------------------------------------------------------------- #
# cov_blup / var_blup
# --------------------------------------------------------------------------- #


class TestCovBlupValues:
    """G Z' P Z G, its diagonal, and the G - C22 block identity."""

    def test_matches_explicit_formula(self):
        # Arrange
        _, x, z, g, _, v = _lmm_fixture()
        p_matrix = _p_matrix(x, v)
        expected = g @ z.T @ p_matrix @ z @ g

        # Act
        cov = cov_blup(x, z, g, v)

        # Assert
        assert cov.shape == (2, 2)
        assert cov.dtype == torch.float64
        assert torch.allclose(cov, expected, **FORMULA)

    def test_is_symmetric(self):
        # Arrange
        _, x, z, g, _, v = _lmm_fixture()

        # Act
        cov = cov_blup(x, z, g, v)

        # Assert
        assert torch.allclose(cov, cov.T, **EXACT)

    def test_var_blup_is_the_diagonal(self):
        # Arrange
        _, x, z, g, _, v = _lmm_fixture()

        # Act
        cov = cov_blup(x, z, g, v)
        variances = var_blup(x, z, g, v)

        # Assert
        assert variances.shape == (2,)
        assert torch.allclose(variances, torch.diagonal(cov), **EXACT)

    def test_equals_g_minus_c22_block(self):
        # Arrange: the C-form blocks use R^{-1}, where R = V - Z G Z'.
        _, x, z, g, r, v = _lmm_fixture()
        r_inv = torch.linalg.inv(r)
        g_inv = torch.linalg.inv(g)
        top = torch.cat([x.T @ r_inv @ x, x.T @ r_inv @ z], dim=1)
        bottom = torch.cat(
            [z.T @ r_inv @ x, z.T @ r_inv @ z + g_inv], dim=1
        )
        coefficient_matrix = torch.cat([top, bottom], dim=0)
        c_matrix = torch.linalg.inv(coefficient_matrix)
        p = x.shape[1]
        expected = g - c_matrix[p:, p:]

        # Act
        cov = cov_blup(x, z, g, v)

        # Assert
        assert torch.allclose(cov, expected, **FORMULA)

    def test_zero_g_returns_zeros(self):
        # Arrange
        _, x, z, _, _, v = _lmm_fixture()
        g = torch.zeros(2, 2, dtype=torch.float64)

        # Act
        cov = cov_blup(x, z, g, v)
        variances = var_blup(x, z, g, v)

        # Assert
        assert torch.allclose(cov, torch.zeros(2, 2, dtype=torch.float64), **EXACT)
        assert torch.allclose(variances, torch.zeros(2, dtype=torch.float64), **EXACT)

    def test_singular_g_is_allowed_and_matches_formula(self):
        # Arrange
        _, x, z, _, _, v = _lmm_fixture()
        g = torch.ones(2, 2, dtype=torch.float64)  # rank one
        p_matrix = _p_matrix(x, v)
        expected = g @ z.T @ p_matrix @ z @ g

        # Act
        cov = cov_blup(x, z, g, v)

        # Assert
        assert torch.allclose(cov, expected, **FORMULA)

    def test_zero_random_effects_returns_empty(self):
        # Arrange
        _, x, _, _, r, _ = _lmm_fixture()
        z = torch.zeros(6, 0, dtype=torch.float64)
        g = torch.zeros(0, 0, dtype=torch.float64)

        # Act
        cov = cov_blup(x, z, g, r)
        variances = var_blup(x, z, g, r)

        # Assert
        assert cov.shape == (0, 0)
        assert variances.shape == (0,)

    def test_no_fixed_effects(self):
        # Arrange
        _, _, z, g, _, v = _lmm_fixture()
        x = torch.zeros(6, 0, dtype=torch.float64)
        expected = g @ z.T @ torch.linalg.inv(v) @ z @ g

        # Act
        cov = cov_blup(x, z, g, v)

        # Assert
        assert torch.allclose(cov, expected, **FORMULA)

    def test_is_positive_semidefinite(self):
        # Arrange
        _, x, z, g, _, v = _lmm_fixture()

        # Act
        cov = cov_blup(x, z, g, v)

        # Assert
        assert torch.linalg.eigvalsh(cov).min() >= -1e-10

    def test_single_random_effect_shape_and_value(self):
        # Arrange
        _, x, _, _, r, _ = _lmm_fixture()
        z = torch.ones(6, 1, dtype=torch.float64)
        g = torch.tensor([[0.6]], dtype=torch.float64)
        v = z @ g @ z.T + r
        expected = g @ z.T @ _p_matrix(x, v) @ z @ g

        # Act
        cov = cov_blup(x, z, g, v)

        # Assert
        assert cov.shape == (1, 1)
        assert torch.allclose(cov, expected, **FORMULA)


# --------------------------------------------------------------------------- #
# predict / residual
# --------------------------------------------------------------------------- #


class TestPredictResidualValues:
    """Conditional predictions and residuals and their equivalences."""

    def test_predict_is_marginal_plus_random(self):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()
        marginal = marginal_predict(y, x, v)
        random_effects = blup(y, x, z, g, v)

        # Act
        fitted = predict(y, x, z, g, v)

        # Assert
        assert fitted.shape == (6,)
        assert fitted.dtype == torch.float64
        assert torch.allclose(fitted, marginal + z @ random_effects, **EXACT)

    def test_predict_is_x_blue_plus_z_blup(self):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()

        # Act
        fitted = predict(y, x, z, g, v)

        # Assert
        assert torch.allclose(
            fitted, x @ blue(y, x, v) + z @ blup(y, x, z, g, v), **EXACT
        )

    def test_residual_is_y_minus_predict(self):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()

        # Act
        residuals = residual(y, x, z, g, v)

        # Assert
        assert residuals.shape == (6,)
        assert torch.allclose(residuals, y - predict(y, x, z, g, v), **EXACT)

    def test_zero_random_effects_equals_marginal_fit(self):
        # Arrange
        y, x, _, _, r, _ = _lmm_fixture()
        z = torch.zeros(6, 0, dtype=torch.float64)
        g = torch.zeros(0, 0, dtype=torch.float64)

        # Act
        fitted = predict(y, x, z, g, r)
        residuals = residual(y, x, z, g, r)

        # Assert
        assert torch.allclose(fitted, marginal_predict(y, x, r), **EXACT)
        assert torch.allclose(residuals, marginal_residual(y, x, r), **EXACT)

    def test_zero_g_equals_marginal_fit(self):
        # Arrange
        y, x, z, _, _, v = _lmm_fixture()
        g = torch.zeros(2, 2, dtype=torch.float64)

        # Act
        fitted = predict(y, x, z, g, v)

        # Assert
        assert torch.allclose(fitted, marginal_predict(y, x, v), **EXACT)

    def test_no_fixed_effects_equals_z_blup(self):
        # Arrange
        y, _, z, g, _, v = _lmm_fixture()
        x = torch.zeros(6, 0, dtype=torch.float64)

        # Act
        fitted = predict(y, x, z, g, v)
        expected = z @ blup(y, x, z, g, v)

        # Assert
        assert torch.allclose(fitted, expected, **EXACT)

    def test_single_fixed_effect_predicts(self):
        # Arrange
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
        z = torch.zeros(4, 0, dtype=torch.float64)
        g = torch.zeros(0, 0, dtype=torch.float64)
        v = torch.eye(4, dtype=torch.float64)
        beta = blue(y, x, v)
        expected = x.squeeze(-1) * beta

        # Act
        fitted = predict(y, x, z, g, v)

        # Assert: predict() does not go through blue(), so p == 1 works here.
        assert fitted.shape == (4,)
        assert torch.allclose(fitted, expected, **EXACT)

    def test_single_observation_residual_shape(self):
        # Arrange
        y = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([[1.0]], dtype=torch.float64)
        z = torch.zeros(1, 1, dtype=torch.float64)
        g = torch.zeros(1, 1, dtype=torch.float64)
        v = torch.tensor([[1.0]], dtype=torch.float64)

        # Act
        residuals = residual(y, x, z, g, v)

        # Assert: broadcasting y - scalar restores the (n,) shape.
        assert residuals.shape == (1,)
        assert torch.allclose(residuals, torch.zeros(1, dtype=torch.float64), **EXACT)

    def test_single_observation_predict_shape(self):
        # Arrange
        y = torch.tensor([2.0], dtype=torch.float64)
        x = torch.tensor([[1.0]], dtype=torch.float64)
        z = torch.zeros(1, 1, dtype=torch.float64)
        g = torch.zeros(1, 1, dtype=torch.float64)
        v = torch.tensor([[1.0]], dtype=torch.float64)

        # Act
        fitted = predict(y, x, z, g, v)

        # Assert
        assert fitted.shape == (1,)


# --------------------------------------------------------------------------- #
# Input validation and error paths
# --------------------------------------------------------------------------- #


class TestInputValidation:
    """Errors inherited from get_dtype / get_device and torch.linalg."""

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_dtype_mismatch_raises(self, func_name):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()
        x = x.to(torch.float32)

        # Act / Assert
        with pytest.raises(ValueError, match="Dtype mismatch"):
            _call(func_name, y, x, z, g, v)

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_non_positive_definite_v_raises(self, func_name):
        # Arrange
        y, x, z, g, _, _ = _lmm_fixture()
        v = torch.diag(
            torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        )

        # Act / Assert
        with pytest.raises(torch.linalg.LinAlgError):
            _call(func_name, y, x, z, g, v)

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_singular_design_raises(self, func_name):
        # Arrange
        y, _, z, g, _, v = _lmm_fixture()
        x = torch.zeros(6, 2, dtype=torch.float64)
        x[:, 0] = 1.0  # a zero column is a reliably rank-deficient design

        # Act / Assert
        with pytest.raises(torch.linalg.LinAlgError):
            _call(func_name, y, x, z, g, v)

    @pytest.mark.parametrize("func_name", Y_FUNCS)
    def test_wrong_y_length_raises(self, func_name):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()
        y = y[:-1]

        # Act / Assert
        with pytest.raises(RuntimeError):
            _call(func_name, y, x, z, g, v)

    @pytest.mark.parametrize("func_name", ZG_FUNCS)
    def test_wrong_z_row_count_raises(self, func_name):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()
        z = z[:-1]

        # Act / Assert
        with pytest.raises(RuntimeError):
            _call(func_name, y, x, z, g, v)

    @pytest.mark.parametrize("func_name", ZG_FUNCS)
    def test_wrong_g_shape_raises(self, func_name):
        # Arrange
        y, x, z, _, _, v = _lmm_fixture()
        g = torch.eye(3, dtype=torch.float64)

        # Act / Assert
        with pytest.raises(RuntimeError):
            _call(func_name, y, x, z, g, v)

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_wrong_v_shape_raises(self, func_name):
        # Arrange
        y, x, z, g, _, _ = _lmm_fixture()
        v = torch.eye(5, dtype=torch.float64)

        # Act / Assert
        with pytest.raises(RuntimeError):
            _call(func_name, y, x, z, g, v)


# --------------------------------------------------------------------------- #
# Dtype and device contract
# --------------------------------------------------------------------------- #


class TestDtypeDevice:
    """float32 preservation, accelerator execution and input immutability."""

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_preserves_float32_dtype(self, func_name):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture(torch.float32)

        # Act
        result = _call(func_name, y, x, z, g, v)

        # Assert
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_runs_on_accelerator(self, func_name):
        # Arrange
        accelerator = _accelerator()
        y, x, z, g, _, v = _lmm_fixture(torch.float32)
        expected = _call(func_name, y, x, z, g, v)
        device_args = [t.to(accelerator) for t in (y, x, z, g, v)]

        # Act
        result = _call(func_name, *device_args)

        # Assert
        assert result.device.type == accelerator.type
        assert result.dtype == torch.float32
        assert torch.allclose(result.cpu(), expected, **ACCEL)

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_device_mismatch_raises(self, func_name):
        # Arrange
        accelerator = _accelerator()
        y, x, z, g, _, v = _lmm_fixture(torch.float32)
        x = x.to(accelerator)

        # Act / Assert
        with pytest.raises(ValueError, match="Device mismatch"):
            _call(func_name, y, x, z, g, v)

    @pytest.mark.parametrize("func_name", ALL_FUNCS)
    def test_does_not_mutate_inputs(self, func_name):
        # Arrange
        y, x, z, g, _, v = _lmm_fixture()
        inputs = (y, x, z, g, v)
        before = [t.clone() for t in inputs]

        # Act
        _call(func_name, y, x, z, g, v)

        # Assert
        for original, current in zip(before, inputs):
            assert torch.equal(original, current)

import math

import pytest
import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import (
    Augment,
    BlockDiagonal,
    DummyMatrix,
    KnownMatrix,
    MarginalCovariance,
    ScalarMatrix,
)
from torch_openreml.example_data.read_example_data import john_alpha
from torch_openreml.methods import CForm, Marginal

N = 40
GROUPS = 5


def _toy_data(n=N, groups=GROUPS, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    labels = [f"g{i % groups}" for i in range(n)]
    y = torch.randn(n, dtype=dtype)
    x = torch.randn(n, 2, dtype=dtype)
    return y, x, labels


def _toy_model(labels, n=N, groups=GROUPS, method="auto"):
    return MixedModelREML(
        (DummyMatrix(labels), ScalarMatrix(groups)),
        R=ScalarMatrix(n),
        method=method,
    )


def _fit_toy(n=N, groups=GROUPS, seed=0, max_iter=100):
    y, x, labels = _toy_data(n=n, groups=groups, seed=seed)
    model = _toy_model(labels, n=n, groups=groups)
    model.optimize(y, x, torch.zeros(2), max_iter=max_iter)
    return model, y, x


def _multi_term_model(n=60):
    torch.manual_seed(7)
    labels_rep = [f"r{i % 3}" for i in range(n)]
    labels_block = [f"b{i % 6}" for i in range(n)]
    model = MixedModelREML(
        rep=(DummyMatrix(labels_rep), ScalarMatrix(3)),
        block=(DummyMatrix(labels_block), ScalarMatrix(6)),
        R=ScalarMatrix(n),
    )
    return model, labels_rep, labels_block


def _recovery_data(seed=0):
    torch.manual_seed(seed)
    n = 50
    groups = 5
    labels = [f"g{i % groups}" for i in range(n)]
    x = torch.stack([torch.ones(n), torch.randn(n)], dim=1)
    u = torch.randn(groups)
    e = 0.25 * torch.randn(n)
    z = torch.zeros(n, groups)
    z[torch.arange(n), torch.arange(n) % groups] = 1.0
    y = x @ torch.tensor([2.0, -1.0]) + z @ u + e
    return y, x, labels, u, z


class TestConstruction:
    """Default configuration and design/covariance assembly."""

    def test_default_residual_is_scalar_matrix(self):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        assert isinstance(model.V, MarginalCovariance)
        assert isinstance(model.R, ScalarMatrix)
        assert model.R().shape == (N, N)
        assert model.V.num_free_params == 2
        assert model.V.free_param_names == ["G/sigma^2", "R/sigma^2"]
        assert model.V().dtype == torch.float32

    def test_single_term_preserves_design_and_covariance(self):
        _, _, labels = _toy_data()
        z = DummyMatrix(labels)
        g = ScalarMatrix(GROUPS)

        model = MixedModelREML((z, g), R=ScalarMatrix(N))

        assert model.Z is z
        assert model.G is g
        assert model.Z_terms == [("random_term_0", z)]
        assert model.G_terms == [("random_term_0", g)]

    def test_positional_terms_are_named_by_index(self):
        _, _, labels = _toy_data()
        z = DummyMatrix(labels)
        model = MixedModelREML(
            (z, ScalarMatrix(GROUPS)),
            (z, ScalarMatrix(GROUPS)),
            R=ScalarMatrix(N),
        )

        assert [name for name, _ in model.Z_terms] == [
            "random_term_0",
            "random_term_1",
        ]
        assert isinstance(model.Z, Augment)
        assert isinstance(model.G, BlockDiagonal)
        assert model.V.num_free_params == 3

    def test_keyword_terms_keep_their_names(self):
        model, _, _ = _multi_term_model()

        assert [name for name, _ in model.Z_terms] == ["rep", "block"]
        assert model.V.free_param_names == [
            "G/rep/sigma^2",
            "G/block/sigma^2",
            "R/sigma^2",
        ]

    def test_residual_can_be_given_explicitly(self):
        _, _, labels = _toy_data()
        residual = ScalarMatrix(N)

        model = MixedModelREML((DummyMatrix(labels), ScalarMatrix(GROUPS)), R=residual)

        assert model.R is residual
        assert model.V.free_param_names == ["G/sigma^2", "R/sigma^2"]

    def test_residual_tensor_is_wrapped_in_known_matrix(self):
        _, _, labels = _toy_data()
        fixed_residual = 2.0 * torch.eye(N)

        model = MixedModelREML(
            (DummyMatrix(labels), ScalarMatrix(GROUPS)),
            R=fixed_residual,
        )

        assert isinstance(model.R, KnownMatrix)
        assert model.V.free_param_names == ["G/sigma^2"]
        assert model.V.num_free_params == 1

    def test_repr_matches_the_covariance(self):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        assert repr(model) == repr(model.V)

    def test_default_method_is_auto_with_both_backends(self):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        assert model.method == "auto"
        assert isinstance(model._cform, CForm)
        assert isinstance(model._marginal, Marginal)


def _validation_cases():
    z = DummyMatrix([f"g{i % 3}" for i in range(12)])
    return [
        (
            lambda: MixedModelREML(z),
            TypeError,
            "Positional argument 0 must be a",
        ),
        (
            lambda: MixedModelREML(
                (z, ScalarMatrix(3)), (z, ScalarMatrix(3)), z
            ),
            TypeError,
            "Positional argument 2 must be a",
        ),
        (
            lambda: MixedModelREML(rep=z),
            ValueError,
            "Keyword argument `rep` must be a",
        ),
        (
            lambda: MixedModelREML(rep=(z, ScalarMatrix(3), 1)),
            ValueError,
            "Random term `rep` must be a length-2",
        ),
        (
            lambda: MixedModelREML(rep=([1, 2], ScalarMatrix(3))),
            TypeError,
            "design of random term `rep` must be a Tensor or Matrix",
        ),
        (
            lambda: MixedModelREML(rep=(z, torch.ones(3))),
            ValueError,
            "covariance of random term `rep` must be two-dimensional",
        ),
        (
            lambda: MixedModelREML(rep=(z, 3.0)),
            TypeError,
            "covariance of random term `rep` must be a Matrix",
        ),
        (
            lambda: MixedModelREML(
                (z, ScalarMatrix(3)), random_term_0=(z, ScalarMatrix(3))
            ),
            ValueError,
            "Term name `random_term_0` already exists",
        ),
        (
            lambda: MixedModelREML((z, ScalarMatrix(3)), R="not a covariance"),
            TypeError,
            "residual term `R` must be a Matrix",
        ),
        (
            lambda: MixedModelREML((z, ScalarMatrix(3)), R=torch.ones(12)),
            ValueError,
            "must be a two-dimensional tensor",
        ),
        (
            lambda: MixedModelREML(),
            ValueError,
            "A residual term must be provided as `R=`",
        ),
        (
            lambda: MixedModelREML(R=torch.eye(12)),
            ValueError,
            "must have at least one free parameter",
        ),
    ]


class TestConstructorValidation:
    """Every rejected construction form raises with a useful message."""

    @pytest.mark.parametrize("factory,exc,match", _validation_cases())
    def test_invalid_model_raises(self, factory, exc, match):
        with pytest.raises(exc, match=match):
            factory()


class TestMethodSelection:
    """``method`` chooses the C-form or marginal backend."""

    def test_auto_uses_cform_for_eligible_model(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        _, _, _, method = model.ai_step(y, x, torch.zeros(2))

        assert method == "cform"

    def test_marginal_selection(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels, method="marginal")

        loglik, method = model.loglik(y, x, torch.zeros(2))
        _, _, _, ai_method = model.ai_step(y, x, torch.zeros(2))

        assert torch.isfinite(loglik)
        assert method == "marginal"
        assert ai_method == "marginal"

    def test_cform_selection(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels, method="cform")

        _, method = model.loglik(y, x, torch.zeros(2))

        assert method == "cform"

    def test_unknown_method_behaves_like_auto(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels, method="not-a-method")

        _, _, _, method = model.ai_step(y, x, torch.zeros(2))

        assert method == "cform"
        assert isinstance(model._cform, CForm)
        assert isinstance(model._marginal, Marginal)

    def test_marginal_instance_is_reused(self):
        y, x, labels = _toy_data()
        marginal = Marginal(trace_approx=False)
        model = _toy_model(labels, method=marginal)

        _, _, _, method = model.ai_step(y, x, torch.zeros(2))

        assert model.method is marginal
        assert model._marginal is marginal
        assert method == "marginal"

    def test_cform_instance_is_reused(self):
        y, x, labels = _toy_data()
        cform = CForm()
        model = _toy_model(labels, method=cform)

        _, _, _, method = model.ai_step(y, x, torch.zeros(2))

        assert model.method is cform
        assert model._cform is cform
        assert method == "cform"

    def test_backends_agree_on_loglik(self):
        y, x, labels = _toy_data()
        theta = torch.tensor([-0.5, 0.2])
        via_cform = _toy_model(labels, method="cform")
        via_marginal = _toy_model(labels, method="marginal")

        cform_value, cform_method = via_cform.loglik(y, x, theta)
        marginal_value, marginal_method = via_marginal.loglik(y, x, theta)

        assert cform_method == "cform"
        assert marginal_method == "marginal"
        assert torch.allclose(cform_value, marginal_value, atol=1e-4)

    def test_cform_rejects_fixed_only_model(self):
        y, x, _ = _toy_data()
        model = MixedModelREML(R=ScalarMatrix(N), method="cform")

        with pytest.raises(RuntimeError, match="CForm requires"):
            model.loglik(y, x, torch.zeros(1))
        with pytest.raises(RuntimeError, match="CForm requires"):
            model.ai_step(y, x, torch.zeros(1))


class TestAiStep:
    """Score vector, average-information matrix and log-likelihood."""

    def test_returns_shapes_and_method(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        score, ai, loglik, method = model.ai_step(y, x, torch.tensor([-0.5, 0.2]))

        assert score.shape == (2,)
        assert ai.shape == (2, 2)
        assert loglik.ndim == 0
        assert torch.isfinite(loglik)
        assert method == "cform"

    def test_ai_matrix_is_symmetric(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        _, ai, _, _ = model.ai_step(y, x, torch.tensor([-0.5, 0.2]))

        assert torch.allclose(ai, ai.T)

    def test_ai_matrix_is_positive_semidefinite(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        _, ai, _, _ = model.ai_step(y, x, torch.tensor([-0.5, 0.2]))

        assert torch.linalg.eigvalsh(ai).min() > -1e-6

    def test_score_matches_finite_difference_of_loglik(self):
        y, x, labels = _toy_data(dtype=torch.float64)
        theta = torch.tensor([-0.5, 0.2], dtype=torch.float64)
        model = _toy_model(labels)

        score, _, _, _ = model.ai_step(y, x, theta)
        step = 1e-5
        numerical = []
        for k in range(2):
            delta = torch.zeros(2, dtype=torch.float64)
            delta[k] = step
            plus, _ = model.loglik(y, x, theta + delta)
            minus, _ = model.loglik(y, x, theta - delta)
            numerical.append((plus - minus) / (2.0 * step))

        assert torch.allclose(score, torch.stack(numerical), rtol=1e-5, atol=1e-7)

    def test_without_loglik_keeps_score_and_ai(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)
        theta = torch.tensor([-0.5, 0.2])

        score, ai, loglik, _ = model.ai_step(y, x, theta)
        score_skip, ai_skip, loglik_skip, _ = model.ai_step(
            y, x, theta, require_loglik=False
        )

        assert torch.allclose(score, score_skip)
        assert torch.allclose(ai, ai_skip)
        assert math.isnan(loglik_skip)

    def test_invalid_parameter_length_raises(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        with pytest.raises(ValueError, match="Parameters must have length"):
            model.ai_step(y, x, torch.zeros(3))


class TestOptimize:
    """The AI-REML optimizer on a small synthetic data set."""

    def test_returns_names_and_parameters(self):
        model, _, _ = _fit_toy()

        names, free_params = model.get_free_params()

        assert names == model.V.free_param_names
        assert free_params.shape == (2,)
        assert torch.isfinite(free_params).all()

    def test_records_history(self):
        model, _, _ = _fit_toy()

        expected = {
            "free_params",
            "loglik",
            "score",
            "ai",
            "delta",
            "update",
            "eta",
            "accepted",
            "method",
        }
        assert set(model.history) == expected
        length = len(model.history["loglik"])
        assert length >= 2
        assert len(model.history["free_params"]) == length
        assert len(model.history["score"]) == length
        assert len(model.history["ai"]) == length
        assert len(model.history["update"]) == length
        assert len(model.history["accepted"]) == length
        assert model.history["ai"][0].shape == (2, 2)
        assert model.history["method"][0] == "cform"

    def test_converges_and_improves_loglik(self):
        model, _, _ = _fit_toy()

        assert model.is_converged()
        first = model.history["loglik"][0]
        last = model.history["loglik"][-1]
        assert torch.isfinite(last)
        assert last >= first

    def test_fitted_variance_components_are_positive(self):
        model, _, _ = _fit_toy()

        values = model.build_params(trans=True)
        by_name = model.build_params(trans=True, out_format="dict")

        assert values.shape == (2,)
        assert (values > 0).all()
        assert list(by_name) == model.V.free_param_names
        for value in by_name.values():
            assert value.shape == (1,)
            assert value.item() > 0

    def test_best_iterate_has_the_highest_loglik(self):
        model, _, _ = _fit_toy()

        _, last = model.get_free_params("last")
        _, best = model.get_free_params("best")
        logliks = torch.stack(model.history["loglik"])
        index = int(torch.argmax(logliks).item())

        assert torch.equal(last, model.history["free_params"][-1])
        assert torch.equal(best, model.history["free_params"][index])
        assert logliks[index] >= logliks[-1]

    def test_max_iter_one_stops_early(self):
        model, y, x = _fit_toy(max_iter=1)

        assert len(model.history["loglik"]) == 1
        assert not model.is_converged()
        assert torch.isfinite(model.loglik()[0])

    def test_stores_training_data_for_post_fit_accessors(self):
        model, y, x = _fit_toy()

        assert model.blue().shape == (2,)
        assert model.predict().shape == (N,)
        assert torch.allclose(model.loglik()[0], model.history["loglik"][-1])

    def test_invalid_initial_values_raise(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        with pytest.raises(ValueError, match="Parameters must have length 2, got 3"):
            model.optimize(y, x, torch.zeros(3))


class TestPostFitShapes:
    """Shapes and internal consistency of the post-fit quantities."""

    def test_basic_accessor_shapes(self):
        model, y, x = _fit_toy()

        assert model.blue().shape == (2,)
        assert model.blup().shape == (GROUPS,)
        assert model.predict().shape == (N,)
        assert model.residual().shape == (N,)
        assert model.loglik()[0].ndim == 0

    def test_covariance_accessor_shapes(self):
        model, _, _ = _fit_toy()

        assert model.cov_blue().shape == (2, 2)
        assert model.var_blue().shape == (2,)
        assert model.cov_blup().shape == (GROUPS, GROUPS)
        assert model.var_blup().shape == (GROUPS,)
        assert torch.allclose(model.cov_blue(), model.cov_blue().T)
        assert torch.allclose(model.cov_blup(), model.cov_blup().T)

    def test_inverse_c_form_blocks(self):
        model, _, _ = _fit_toy()

        c11, c12, c21, c22 = model.c_inverse_blocks()

        assert c11.shape == (2, 2)
        assert c12.shape == (2, GROUPS)
        assert c21.shape == (GROUPS, 2)
        assert c22.shape == (GROUPS, GROUPS)
        assert torch.allclose(c12, c21.T)
        assert torch.allclose(c11, model.cov_blue(), atol=1e-4)
        assert torch.allclose(c22, c22.T)

    def test_predict_and_residual_decompose(self):
        model, y, x = _fit_toy()
        beta = model.blue()
        blup = model.blup()
        z = model.Z()

        assert torch.allclose(model.predict(), x @ beta + z @ blup, atol=1e-5)
        assert torch.allclose(model.residual(), y - model.predict(), atol=1e-5)

    def test_marginal_accessors_match_the_blue(self):
        model, y, x = _fit_toy()
        beta = model.blue()

        assert torch.allclose(model.marginal_predict(), x @ beta, atol=1e-5)
        assert torch.allclose(
            model.marginal_residual(), y - x @ beta, atol=1e-5
        )

    def test_per_term_accessors_partition_the_joint_quantities(self):
        model, _, _ = _fit_toy()

        blup = model.blup_per_term()
        cov = model.cov_blup_per_term()
        var = model.var_blup_per_term()

        assert list(blup) == ["random_term_0"]
        assert list(cov) == ["random_term_0"]
        assert list(var) == ["random_term_0"]
        assert torch.allclose(blup["random_term_0"], model.blup())
        assert torch.allclose(cov["random_term_0"], model.cov_blup())
        assert torch.allclose(var["random_term_0"], model.var_blup())

    def test_tree_keys_and_values(self):
        model, _, _ = _fit_toy()
        free_params = model.get_free_params()[1]

        tree = model.tree()

        assert set(tree) == {"/", "Z", "G", "R"}
        assert torch.allclose(tree["/"], model.V(free_params))
        assert torch.allclose(tree["G"], model.G(free_params[:1]))
        assert torch.allclose(tree["R"], model.R(free_params[1:]))
        assert torch.allclose(tree["Z"], model.Z())

    def test_build_params_applies_transforms(self):
        model, _, _ = _fit_toy()

        raw = model.build_params(trans=False)
        transformed = model.build_params(trans=True)

        assert torch.allclose(transformed, torch.exp(2.0 * raw), atol=1e-6)

    def test_build_params_dict_output(self):
        model, _, _ = _fit_toy()

        values = model.build_params(out_format="dict")

        assert list(values) == model.V.free_param_names
        assert all(value.shape == (1,) for value in values.values())

    def test_loglik_matches_the_last_history_entry(self):
        model, _, _ = _fit_toy()

        value, method = model.loglik()

        assert method == "cform"
        assert torch.allclose(value, model.history["loglik"][-1])


class TestMultiTerm:
    """Two random-effect terms composed through ``Augment``/``BlockDiagonal``."""

    def _fit(self):
        model, _, _ = _multi_term_model()
        n = 60
        y, x, _ = _toy_data(n=n, groups=3, seed=7)
        model.optimize(y, x, torch.zeros(3), max_iter=100)
        return model, y, x

    def test_structure_and_free_parameter_names(self):
        model, _, _ = _multi_term_model()

        assert isinstance(model.Z, Augment)
        assert isinstance(model.G, BlockDiagonal)
        assert [name for name, _ in model.Z_terms] == ["rep", "block"]
        assert model.V.free_param_names == [
            "G/rep/sigma^2",
            "G/block/sigma^2",
            "R/sigma^2",
        ]

    def test_per_term_blups_concatenate_to_the_joint_blup(self):
        model, _, _ = self._fit()

        per_term = model.blup_per_term()

        assert list(per_term) == ["rep", "block"]
        assert torch.allclose(
            torch.cat([per_term["rep"], per_term["block"]]), model.blup()
        )

    def test_tree_contains_per_term_nodes(self):
        model, _, _ = self._fit()

        tree = model.tree()

        assert {"Z", "Z/rep", "Z/block", "G", "G/rep", "G/block", "R"} <= set(tree)
        assert tree["Z/rep"].shape == (60, 3)
        assert tree["Z/block"].shape == (60, 6)
        assert model.Z().shape == (60, 9)

    def test_cov_blup_per_term_matches_the_joint_diagonal_blocks(self):
        model, _, _ = self._fit()

        joint = model.cov_blup()
        per_term = model.cov_blup_per_term()

        assert torch.allclose(per_term["rep"], joint[:3, :3])
        assert torch.allclose(per_term["block"], joint[3:, 3:])


class TestClosedForm:
    """Compare the fitted quantities with the GLS closed forms."""

    def test_blue_matches_generalised_least_squares(self):
        model, y, x = _fit_toy()
        free_params = model.get_free_params()[1]
        v = model.V(free_params) + 1e-6 * torch.eye(N)

        expected = torch.linalg.solve(
            x.T @ torch.linalg.inv(v) @ x, x.T @ torch.linalg.inv(v) @ y
        )

        assert torch.allclose(model.blue(), expected, atol=1e-5)

    def test_blup_matches_the_closed_form(self):
        model, y, x = _fit_toy()
        free_params = model.get_free_params()[1]
        v = model.V(free_params) + 1e-6 * torch.eye(N)
        g = model.G(free_params[:1])
        z = model.Z()
        beta = model.blue()

        expected = g @ (z.T @ torch.linalg.inv(v) @ (y - x @ beta))

        assert torch.allclose(model.blup(), expected, atol=1e-5)


class TestRecovery:
    """Recover the fixed and random effects of a known small model."""

    def _fit(self):
        y, x, labels, u, z = _recovery_data()
        model = MixedModelREML(
            (DummyMatrix(labels), ScalarMatrix(5)), R=ScalarMatrix(50)
        )
        model.optimize(y, x, torch.zeros(2), max_iter=300)
        return model, y, x, u, z

    def test_fixed_effects_are_recovered(self):
        model, _, _, _, _ = self._fit()

        beta = model.blue()

        assert model.is_converged()
        assert torch.allclose(beta, torch.tensor([2.0, -1.0]), atol=0.3)

    def test_random_effects_correlate_with_the_truth(self):
        model, _, _, u, _ = self._fit()

        correlation = torch.corrcoef(torch.stack([model.blup(), u]))[0, 1]

        assert correlation > 0.9

    def test_variance_components_are_plausible(self):
        model, _, _, _, _ = self._fit()

        values = model.build_params(trans=True)
        group_variance, residual_variance = values[0].item(), values[1].item()

        assert group_variance > 0.01
        assert 0.0 < residual_variance < 0.5

    def test_predictions_track_the_response(self):
        model, y, _, _, _ = self._fit()

        correlation = torch.corrcoef(torch.stack([model.predict(), y]))[0, 1]

        assert correlation > 0.9


class TestFixedOnly:
    """A model with no random-effect terms reduces to a marginal model."""

    def _fit(self):
        y, x, _ = _toy_data()
        model = MixedModelREML(R=ScalarMatrix(N))
        model.optimize(y, x, None, max_iter=100)
        return model, y, x

    def test_structure(self):
        model = MixedModelREML(R=ScalarMatrix(N))

        assert model.Z is None
        assert model.G is None
        assert isinstance(model.V, ScalarMatrix)
        assert model.V.free_param_names == ["sigma^2"]
        assert model.V.num_free_params == 1

    def test_loglik_uses_the_marginal_backend(self):
        y, x, _ = _toy_data()
        model = MixedModelREML(R=ScalarMatrix(N))

        value, method = model.loglik(y, x, torch.zeros(1))

        assert method == "marginal"
        assert torch.isfinite(value)

    def test_optimize_estimates_the_residual_variance(self):
        model, y, x = self._fit()

        assert model.is_converged()
        assert model.get_free_params()[0] == ["sigma^2"]
        # V = sigma^2 I, so the REML estimate is the OLS residual variance.
        beta = torch.linalg.solve(x.T @ x, x.T @ y)
        residual = y - x @ beta
        expected = (residual @ residual) / (N - 2)
        assert torch.allclose(model.build_params(), expected, atol=1e-3)

    def test_blue_matches_ordinary_least_squares(self):
        model, y, x = self._fit()

        expected = torch.linalg.solve(x.T @ x, x.T @ y)

        assert torch.allclose(model.blue(), expected, atol=1e-5)

    def test_predict_and_residual_reduce_to_the_marginal_quantities(self):
        model, y, x = self._fit()

        assert torch.allclose(model.predict(), model.marginal_predict())
        assert torch.allclose(model.residual(), model.marginal_residual())
        assert torch.allclose(model.residual(), y - model.predict())

    @pytest.mark.parametrize(
        "accessor",
        [
            "blup",
            "cov_blup",
            "var_blup",
            "blup_per_term",
            "cov_blup_per_term",
            "var_blup_per_term",
        ],
    )
    def test_blup_accessors_reject_models_without_random_effects(self, accessor):
        model, y, x = self._fit()

        with pytest.raises(ValueError, match="no random-effect terms"):
            getattr(model, accessor)()

    def test_c_inverse_blocks_requires_random_effects(self):
        model, _, _ = self._fit()

        with pytest.raises(RuntimeError, match="CForm requires"):
            model.c_inverse_blocks()

    def test_predict_new_rejects_random_blocks(self):
        model, y, x = self._fit()

        with pytest.raises(ValueError, match="no random-effect terms"):
            model.predict_new(x[:2], torch.zeros(2, 5))

    def test_tree_has_a_single_root(self):
        model, _, _ = self._fit()

        tree = model.tree()

        assert set(tree) == {"/"}
        assert tree["/"].shape == (N, N)


class TestAccessorValidation:
    """Argument validation shared by the post-fit accessors."""

    def test_response_and_design_must_be_given_together(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        with pytest.raises(ValueError, match="Pass both"):
            model.loglik(y, None, torch.zeros(2))
        with pytest.raises(ValueError, match="Pass both"):
            model.loglik(None, x, torch.zeros(2))

    @pytest.mark.parametrize(
        "accessor",
        ["blue", "blup", "predict", "residual", "loglik", "cov_blue", "tree"],
    )
    def test_accessors_require_a_fit(self, accessor):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        with pytest.raises(ValueError, match=r"call optimize\(\) first"):
            getattr(model, accessor)()

    def test_invalid_free_parameter_string_raises(self):
        model, _, _ = _fit_toy()

        with pytest.raises(ValueError, match="free_params must be a tensor"):
            model.blue(free_params="median")

    def test_unknown_history_selection_raises(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        with pytest.raises(ValueError, match="No optimisation history"):
            model.blue(y, x, free_params="best")

    def _predict_new_cases(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)
        model.optimize(y, x, torch.zeros(2), max_iter=20)
        z = model.Z()
        return model, x, z

    def test_predict_new_rejects_bad_designs(self):
        model, x, _ = self._predict_new_cases()

        with pytest.raises(TypeError, match="`x` must be a torch.Tensor"):
            model.predict_new([[0.0, 0.0]])
        with pytest.raises(ValueError, match="two-dimensional tensor"):
            model.predict_new(torch.zeros(2))
        with pytest.raises(ValueError, match="at least one row"):
            model.predict_new(torch.zeros(0, 2))
        with pytest.raises(ValueError, match="must have 2 columns"):
            model.predict_new(x[:, :1])

    def test_predict_new_rejects_bad_random_designs(self):
        model, _, z = self._predict_new_cases()

        with pytest.raises(ValueError, match="Unknown random-effect term"):
            model.predict_new(torch.zeros(2, 2), {"nope": torch.zeros(2, 5)})
        with pytest.raises(ValueError, match="one block per random-effect term"):
            model.predict_new(torch.zeros(2, 2), [z[:2], z[:2]])
        with pytest.raises(ValueError, match="`z` must have shape"):
            model.predict_new(torch.zeros(2, 2), z[:2, :4])
        with pytest.raises(TypeError, match="`z` must be a tensor"):
            model.predict_new(torch.zeros(2, 2), 3.0)

    def test_predict_new_population_predictions_match_the_marginal_predictor(self):
        model, x, _ = self._predict_new_cases()

        assert torch.allclose(
            model.predict_new(x, None), model.marginal_predict(), atol=1e-5
        )

    def test_predict_new_with_random_blocks_matches_the_training_predictions(self):
        model, x, z = self._predict_new_cases()

        assert torch.allclose(model.predict_new(x, z), model.predict(), atol=1e-5)

    def test_update_clips_to_the_bounds(self):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        updated, step = model.update(
            torch.zeros(2), torch.tensor([10.0, -10.0]), 1.0, -1.0, 1.0
        )

        assert torch.equal(updated, torch.tensor([1.0, -1.0]))
        assert torch.equal(step, updated)

    def test_update_without_bounds_applies_the_damping(self):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        updated, step = model.update(
            torch.zeros(2), torch.tensor([0.1, -0.2]), 0.5
        )

        assert torch.allclose(updated, torch.tensor([0.05, -0.1]))
        assert torch.allclose(step, updated)


class TestDtypeDevice:
    """Precision and device resolution of the fitted quantities."""

    def test_covariance_follows_the_parameter_dtype(self):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        assert model.V().dtype == torch.float32
        assert model.V(torch.zeros(2, dtype=torch.float64)).dtype == torch.float64

    def test_full_fit_follows_a_float64_initialisation(self):
        y, x, labels = _toy_data(dtype=torch.float64)
        model = _toy_model(labels)

        model.optimize(y, x, torch.zeros(2, dtype=torch.float64), max_iter=50)

        for value in (
            model.blue(),
            model.blup(),
            model.predict(),
            model.cov_blue(),
            model.loglik()[0],
            model.history["loglik"][-1],
        ):
            assert value.dtype == torch.float64

    def test_default_initialisation_follows_the_covariance_default(self):
        y, x, labels = _toy_data(dtype=torch.float64)
        model = _toy_model(labels)

        model.optimize(y, x, None, max_iter=50)

        assert model.get_free_params()[1].dtype == torch.float32
        assert model.loglik()[0].dtype == torch.float32

    def test_fit_runs_on_an_accelerator(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")

        n = 24
        groups = 4
        with torch.device(device):
            y, x, labels = _toy_data(n=n, groups=groups, seed=1)
            model = _toy_model(labels, n=n, groups=groups)
            model.optimize(y, x, torch.zeros(2), max_iter=30)

            assert model.loglik()[0].device.type == torch.device(device).type
            assert model.blue().device.type == torch.device(device).type
            assert model.blup().device.type == torch.device(device).type


@pytest.fixture(scope="module")
def john_alpha_fit():
    frame = john_alpha
    n = len(frame)
    y = torch.tensor(frame["yield"].to_numpy(), dtype=torch.float32)
    x = torch.stack(
        [
            torch.ones(n),
            torch.tensor(frame["row"].to_numpy(), dtype=torch.float32),
        ],
        dim=1,
    )
    model = MixedModelREML(
        rep=(DummyMatrix(frame["rep"]), ScalarMatrix(3)),
        block=(DummyMatrix(frame["block"]), ScalarMatrix(6)),
        gen=(DummyMatrix(frame["gen"]), ScalarMatrix(24)),
        R=ScalarMatrix(n),
    )
    model.optimize(y, x, None, max_iter=200)
    return model, y, x, n


class TestJohnAlpha:
    """The John's alpha example data set end to end."""

    def test_fit_converges_with_finite_loglik(self, john_alpha_fit):
        model, _, _, _ = john_alpha_fit

        value, method = model.loglik()

        assert model.is_converged()
        assert 2 <= len(model.history["loglik"]) <= 200
        assert method == "cform"
        assert torch.isfinite(value)

    def test_variance_components_are_plausible(self, john_alpha_fit):
        model, _, _, _ = john_alpha_fit

        values = model.build_params(trans=True, out_format="dict")
        rep = values["G/rep/sigma^2"].item()
        block = values["G/block/sigma^2"].item()
        gen = values["G/gen/sigma^2"].item()
        residual = values["R/sigma^2"].item()

        assert all(math.isfinite(value) for value in (rep, block, gen, residual))
        assert rep > 0.01
        assert gen > 0.01
        assert 0.0 <= block < 1.0
        assert 0.02 < residual < 0.5
        assert abs(rep + block + gen) < 10.0 * residual

    def test_post_fit_accessors_have_the_expected_shapes(self, john_alpha_fit):
        model, _, _, n = john_alpha_fit

        assert model.blue().shape == (2,)
        assert model.blup().shape == (33,)
        assert model.predict().shape == (n,)
        assert model.residual().shape == (n,)
        assert model.cov_blue().shape == (2, 2)
        assert model.cov_blup().shape == (33, 33)

    def test_per_term_blups_match_the_random_effects(self, john_alpha_fit):
        model, _, _, _ = john_alpha_fit

        per_term = model.blup_per_term()

        assert list(per_term) == ["rep", "block", "gen"]
        assert per_term["rep"].shape == (3,)
        assert per_term["block"].shape == (6,)
        assert per_term["gen"].shape == (24,)
        assert torch.allclose(
            torch.cat([per_term["rep"], per_term["block"], per_term["gen"]]),
            model.blup(),
        )

    def test_predictions_decompose_into_fixed_and_random_parts(self, john_alpha_fit):
        model, _, x, _ = john_alpha_fit
        beta = model.blue()
        blup = model.blup()
        z = model.Z()

        assert torch.allclose(model.predict(), x @ beta + z @ blup, atol=1e-4)

    def test_predict_new_supports_named_random_blocks(self, john_alpha_fit):
        model, _, x, _ = john_alpha_fit
        new_x = x[:2]
        new_z = {
            "rep": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "gen": torch.zeros(2, 24),
        }

        prediction = model.predict_new(new_x, new_z)

        assert prediction.shape == (2,)
        assert torch.isfinite(prediction).all()

    def test_canonical_intercept_only_model_fits(self):
        frame = john_alpha
        n = len(frame)
        y = torch.tensor(frame["yield"].to_numpy(), dtype=torch.float32)
        x = torch.ones(n, 1)
        model = MixedModelREML(
            rep=(DummyMatrix(frame["rep"]), ScalarMatrix(3)),
            block=(DummyMatrix(frame["block"]), ScalarMatrix(6)),
            gen=(DummyMatrix(frame["gen"]), ScalarMatrix(24)),
            R=ScalarMatrix(n),
        )

        model.optimize(y, x, None, max_iter=200)
        value, method = model.loglik()
        values = model.build_params(trans=True)

        assert model.is_converged()
        assert method == "cform"
        assert torch.isfinite(value)
        assert (values > 0).all()
        assert values[0].item() > 0.01  # rep
        assert values[2].item() > 0.01  # gen


class TestKnownApiGaps:
    """Contracts the current implementation does not satisfy (see plan)."""

    @pytest.mark.xfail(
        strict=False,
        reason="post.blue squeezes the (p, 1) result, so a one-column X yields a 0-dim tensor",
    )
    def test_blue_returns_a_vector_with_one_fixed_effect(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        assert model.blue(y, x[:, :1], torch.zeros(2)).shape == (1,)

    @pytest.mark.xfail(
        strict=False,
        reason="blup calls marginal_residual, which fails on the 0-dim BLUE of a one-column X",
    )
    def test_blup_works_with_one_fixed_effect(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        assert model.blup(y, x[:, :1], torch.zeros(2)).shape == (GROUPS,)

    @pytest.mark.xfail(
        strict=False,
        reason="predict_new indexes beta.shape[0], which fails on the 0-dim BLUE",
    )
    def test_predict_new_works_with_one_fixed_effect(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)
        model.optimize(y, x[:, :1], torch.zeros(2), max_iter=20)

        assert model.predict_new(x[:2, :1]).shape == (2,)

    @pytest.mark.xfail(
        strict=False,
        reason="is_converged dereferences history before a fit; raises TypeError instead of returning False",
    )
    def test_is_converged_before_a_fit_is_false(self):
        _, _, labels = _toy_data()
        model = _toy_model(labels)

        assert model.is_converged() is False

    @pytest.mark.xfail(
        strict=False,
        reason="ai_step returns a Python float nan, not torch.nan as documented",
    )
    def test_skipped_loglik_is_a_tensor_nan(self):
        y, x, labels = _toy_data()
        model = _toy_model(labels)

        _, _, loglik, _ = model.ai_step(y, x, torch.zeros(2), require_loglik=False)

        assert torch.is_tensor(loglik)

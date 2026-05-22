import torch
from torch_openreml.covariance.transform import (
    Transform,
    TransformExpPow2,
    TransformExp,
    TransformChain,
)


class TestTransformExpPow2:
    """Tests for the scaled exponential transform."""

    def test_constructor(self):
        t = TransformExpPow2()
        assert isinstance(t, Transform)

    def test_domain_codomain(self):
        t = TransformExpPow2()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ₀⁺"

    def test_forward_matches_formula(self):
        t = TransformExpPow2()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.exp(2.0 * x))

    def test_forward_zero_is_one(self):
        t = TransformExpPow2()
        assert torch.allclose(t(torch.tensor(0.0)), torch.tensor(1.0))

    def test_forward_positive_is_larger_than_exp(self):
        t = TransformExpPow2()
        x = torch.tensor([1.0, 2.0])
        assert (t(x) > torch.exp(x)).all()

    def test_forward_always_positive(self):
        t = TransformExpPow2()
        x = torch.randn(100, dtype=torch.float64) * 10
        assert (t(x) > 0).all()

    def test_forward_shape_preserved(self):
        t = TransformExpPow2()
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    def test_forward_equivalent_to_exp_of_2x(self):
        t = TransformExpPow2()
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), exp(2.0 * x))

    def test_inverse_roundtrip(self):
        t = TransformExpPow2()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_matches_formula(self):
        t = TransformExpPow2()
        x = torch.tensor([1.0, 4.0, 10.0], dtype=torch.float64)
        assert torch.allclose(t.inverse(x), torch.log(x) / 2.0)

    def test_inverse_of_one_is_zero(self):
        t = TransformExpPow2()
        assert torch.allclose(t.inverse(torch.tensor(1.0)), torch.tensor(0.0))

    def test_inverse_non_positive(self):
        t = TransformExpPow2()
        assert torch.isneginf(t.inverse(torch.tensor(0.0)))
        assert torch.isnan(t.inverse(torch.tensor(-1.0)))

    def test_grad_matches_formula(self):
        t = TransformExpPow2()
        x = torch.randn(10, dtype=torch.float64)
        expected = 2.0 * torch.exp(2.0 * x)
        assert torch.allclose(t.grad(x), expected)

    def test_grad_is_twice_forward(self):
        t = TransformExpPow2()
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), 2.0 * t(x))

    def test_grad_at_zero(self):
        t = TransformExpPow2()
        assert torch.allclose(t.grad(torch.tensor(0.0)), torch.tensor(2.0))

    def test_grad_positive(self):
        t = TransformExpPow2()
        x = torch.randn(5, dtype=torch.float64)
        assert (t.grad(x) > 0).all()

    def test_grad_matches_autograd(self):
        t = TransformExpPow2()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformExpPow2()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformExpPow2()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformExpPow2()
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformExpPow2()
        x = torch.randn(5, requires_grad=True).exp()
        assert t.inverse(x).requires_grad


class TestTransformChainWithExpPow2:
    """Integration tests with TransformChain."""

    def test_chain_exppow2_then_exp_forward(self):
        t = TransformChain([TransformExpPow2(), TransformExp()])
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        expected = torch.exp(torch.exp(2.0 * x))
        assert torch.allclose(t(x), expected)

    def test_chain_exppow2_then_exp_inverse(self):
        t = TransformChain([TransformExpPow2(), TransformExp()])
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_chain_exppow2_then_exp_grad(self):
        t = TransformChain([TransformExpPow2(), TransformExp()])
        x = torch.randn(3, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

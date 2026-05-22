import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformPow,
    TransformExp,
    TransformChain,
)


class TestTransformPow:
    """Tests for the power transform."""

    def test_constructor_default(self):
        t = TransformPow()
        assert isinstance(t, Transform)
        assert t.factor == 2.0

    def test_constructor_custom_factor(self):
        t = TransformPow(factor=3.0)
        assert t.factor == 3.0

    def test_constructor_integer_factor(self):
        t = TransformPow(factor=3)
        assert t.factor == 3

    def test_repr(self):
        t = TransformPow(factor=3.0)
        assert repr(t) == "TransformPow(factor=3.0)"

    def test_repr_default(self):
        t = TransformPow()
        assert repr(t) == "TransformPow(factor=2.0)"

    def test_domain_codomain(self):
        t = TransformPow()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    def test_forward_default_factor(self):
        t = TransformPow()
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(t(x), x**2)

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0, 0.5])
    def test_forward_matches_power(self, factor):
        t = TransformPow(factor=factor)
        x = torch.abs(torch.randn(10, dtype=torch.float64))
        assert torch.allclose(t(x), torch.pow(x, factor))

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0, 0.5])
    def test_forward_shape_preserved(self, factor):
        t = TransformPow(factor=factor)
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    def test_forward_scalar(self):
        t = TransformPow(factor=2.0)
        x = torch.tensor(3.0)
        assert torch.allclose(t(x), torch.tensor(9.0))

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
    def test_inverse_roundtrip(self, factor):
        t = TransformPow(factor=factor)
        x = torch.abs(torch.randn(10, dtype=torch.float64))
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_default_known_values(self):
        t = TransformPow(factor=2.0)
        assert torch.allclose(t.inverse(torch.tensor(4.0)), torch.tensor(2.0))
        assert torch.allclose(t.inverse(torch.tensor(9.0)), torch.tensor(3.0))
        assert torch.allclose(t.inverse(torch.tensor(0.0)), torch.tensor(0.0))

    def test_grad_default_factor(self):
        t = TransformPow()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        expected = 2.0 * x
        assert torch.allclose(t.grad(x), expected)

    def test_grad_factor_one(self):
        t = TransformPow(factor=1.0)
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), torch.ones_like(x))

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0, 0.5])
    def test_grad_matches_analytic(self, factor):
        t = TransformPow(factor=factor)
        if factor == 2.0 or factor == 0.5:
            x = torch.abs(torch.randn(5, dtype=torch.float64))
        else:
            x = torch.randn(5, dtype=torch.float64)
        expected = factor * torch.pow(x, factor - 1.0)
        assert torch.allclose(t.grad(x), expected)

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
    def test_grad_matches_autograd(self, factor):
        t = TransformPow(factor=factor)
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformPow(factor=2.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformPow(factor=2.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformPow(factor=2.0)
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformPow(factor=2.0)
        x = torch.randn(5, requires_grad=True) + 1.0
        assert t.inverse(x).requires_grad

    def test_forward_negative_input_even_factor(self):
        t = TransformPow(factor=2.0)
        x = torch.tensor([-2.0, -3.0])
        assert torch.allclose(t(x), x**2)

    def test_forward_zero_input(self):
        t = TransformPow(factor=3.0)
        assert torch.allclose(t(torch.tensor(0.0)), torch.tensor(0.0))


class TestTransformChainWithPow:
    """Integration tests with TransformChain."""

    def test_chain_pow_then_exp_forward(self):
        t = TransformChain([TransformPow(factor=2.0), TransformExp()])
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        expected = torch.exp(x**2)
        assert torch.allclose(t(x), expected)

    def test_chain_pow_then_exp_inverse(self):
        t = TransformChain([TransformPow(factor=2.0), TransformExp()])
        x = torch.abs(torch.randn(5, dtype=torch.float64))
        assert torch.allclose(t.inverse(t(x)), x)

    def test_chain_pow_then_exp_grad(self):
        t = TransformChain([TransformPow(factor=2.0), TransformExp()])
        x = torch.randn(3, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

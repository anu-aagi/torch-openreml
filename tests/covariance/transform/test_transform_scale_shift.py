import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformScaleShift,
    TransformExp,
    TransformChain,
)


class TestTransformScaleShift:
    """Tests for the affine transform."""

    def test_constructor(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        assert isinstance(t, Transform)
        assert t.a == 2.0
        assert t.b == 1.0

    def test_constructor_b_defaults_to_zero(self):
        t = TransformScaleShift(a=3.0)
        assert t.b == 0.0

    def test_repr(self):
        t = TransformScaleShift(a=2.0, b=-1.5)
        assert repr(t) == "TransformScaleShift(a=2.0, b=-1.5)"

    def test_domain_codomain(self):
        t = TransformScaleShift(a=2.0)
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, 3.0),
        (-3.0, -2.0),
    ])
    def test_forward_matches_formula(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), a * x + b)

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, 3.0),
        (-3.0, -2.0),
    ])
    def test_forward_shape_preserved(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 5.0),
        (-1.0, 3.0),
        (0.5, -2.0),
        (-3.0, 0.0),
    ])
    def test_inverse_roundtrip(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_known_values(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        assert torch.allclose(t.inverse(torch.tensor(1.0)), torch.tensor(0.0))
        assert torch.allclose(t.inverse(torch.tensor(3.0)), torch.tensor(1.0))

    def test_grad_constant(self):
        t = TransformScaleShift(a=2.5, b=10.0)
        x = torch.randn(5, dtype=torch.float64)
        expected = torch.full_like(x, 2.5)
        assert torch.allclose(t.grad(x).expand_as(x), expected)

    def test_grad_negative_a(self):
        t = TransformScaleShift(a=-3.0, b=1.0)
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), torch.tensor([-3.0], dtype=x.dtype))

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 0.0),
        (-1.0, 5.0),
        (0.5, -2.0),
    ])
    def test_grad_matches_autograd(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical.expand_as(x.grad), x.grad)

    def test_dtype_float32(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformScaleShift(a=2.0)
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformScaleShift(a=2.0)
        x = torch.randn(5, requires_grad=True)
        assert t.inverse(x).requires_grad

    def test_inverse_zero_a(self):
        t = TransformScaleShift(a=0.0, b=1.0)
        x = torch.tensor([1.0, 2.0])
        result = t.inverse(x)
        assert torch.isinf(result).any() or torch.isnan(result).any()


class TestTransformChainWithScaleShift:
    """Integration tests with TransformChain."""

    def test_chain_scale_shift_then_exp_forward(self):
        t = TransformChain([TransformScaleShift(a=2.0, b=1.0), TransformExp()])
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        expected = torch.exp(2.0 * x + 1.0)
        assert torch.allclose(t(x), expected)

    def test_chain_scale_shift_then_exp_inverse(self):
        t = TransformChain([TransformScaleShift(a=2.0, b=1.0), TransformExp()])
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_chain_scale_shift_then_exp_grad(self):
        t = TransformChain([TransformScaleShift(a=2.0, b=1.0), TransformExp()])
        x = torch.randn(3, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

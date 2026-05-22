import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformSigmoid,
    TransformScaleShift,
    TransformChain,
)


class TestTransformSigmoid:
    """Tests for the sigmoid transform."""

    def test_constructor(self):
        t = TransformSigmoid()
        assert isinstance(t, Transform)

    def test_domain_codomain(self):
        t = TransformSigmoid()
        assert t.domain == "ℝ₀⁺"
        assert t.codomain == "(0, 1)"

    def test_forward_matches_torch_sigmoid(self):
        t = TransformSigmoid()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.sigmoid(x))

    def test_forward_zero_is_half(self):
        t = TransformSigmoid()
        assert torch.allclose(t(torch.tensor(0.0)), torch.tensor(0.5))

    def test_forward_large_positive_approaches_one(self):
        t = TransformSigmoid()
        result = t(torch.tensor(100.0))
        assert torch.allclose(result, torch.tensor(1.0))

    def test_forward_large_negative_approaches_zero(self):
        t = TransformSigmoid()
        result = t(torch.tensor(-100.0))
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_output_bounded(self):
        t = TransformSigmoid()
        x = torch.randn(100, dtype=torch.float64) * 10
        result = t(x)
        assert (result > 0).all() and (result < 1).all()

    def test_forward_shape_preserved(self):
        t = TransformSigmoid()
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    def test_inverse_roundtrip(self):
        t = TransformSigmoid()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_matches_logit(self):
        t = TransformSigmoid()
        x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        assert torch.allclose(t.inverse(x), torch.logit(x))

    def test_inverse_of_half_is_zero(self):
        t = TransformSigmoid()
        assert torch.allclose(t.inverse(torch.tensor(0.5)), torch.tensor(0.0))

    def test_inverse_at_zero(self):
        t = TransformSigmoid()
        assert torch.isneginf(t.inverse(torch.tensor(0.0)))

    def test_inverse_at_one(self):
        t = TransformSigmoid()
        assert torch.isposinf(t.inverse(torch.tensor(1.0)))

    def test_inverse_outside_domain(self):
        t = TransformSigmoid()
        result = t.inverse(torch.tensor(-0.5))
        assert torch.isnan(result)

    def test_grad_matches_analytic(self):
        t = TransformSigmoid()
        x = torch.randn(10, dtype=torch.float64)
        s = torch.sigmoid(x)
        expected = s * (1 - s)
        assert torch.allclose(t.grad(x), expected)

    def test_grad_at_zero(self):
        t = TransformSigmoid()
        assert torch.allclose(t.grad(torch.tensor(0.0)), torch.tensor(0.25))

    def test_grad_symmetric(self):
        t = TransformSigmoid()
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), t.grad(-x))

    def test_grad_positive(self):
        t = TransformSigmoid()
        x = torch.randn(5, dtype=torch.float64)
        assert (t.grad(x) > 0).all()

    def test_grad_approaches_zero_for_large_inputs(self):
        t = TransformSigmoid()
        g = t.grad(torch.tensor(50.0))
        assert torch.allclose(g, torch.tensor(0.0), atol=1e-10)

    def test_grad_matches_autograd(self):
        t = TransformSigmoid()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformSigmoid()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformSigmoid()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformSigmoid()
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformSigmoid()
        x = torch.randn(5, requires_grad=True).sigmoid()
        assert t.inverse(x).requires_grad


class TestTransformChainWithSigmoid:
    """Integration tests with TransformChain."""

    def test_chain_sigmoid_then_scale_shift_forward(self):
        t = TransformChain([TransformSigmoid(), TransformScaleShift(a=2.0, b=-1.0)])
        x = torch.tensor([0.0], dtype=torch.float64)
        expected = 2.0 * torch.sigmoid(x) - 1.0
        assert torch.allclose(t(x), expected)

    def test_chain_sigmoid_then_scale_shift_inverse(self):
        t = TransformChain([TransformSigmoid(), TransformScaleShift(a=2.0, b=-1.0)])
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_chain_sigmoid_then_scale_shift_grad(self):
        t = TransformChain([TransformSigmoid(), TransformScaleShift(a=2.0, b=-1.0)])
        x = torch.randn(3, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

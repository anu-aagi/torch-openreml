import torch
from torch_openreml.covariance.transform import (
    Transform,
    TransformIdentity,
    TransformExp,
    TransformChain,
)


class TestTransformIdentity:
    """Tests for the identity transform."""

    def test_constructor(self):
        t = TransformIdentity()
        assert isinstance(t, Transform)

    def test_domain_codomain(self):
        t = TransformIdentity()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    def test_forward_returns_input(self):
        t = TransformIdentity()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(t(x), x)

    def test_forward_scalar(self):
        t = TransformIdentity()
        x = torch.tensor(3.5)
        assert torch.equal(t(x), x)

    def test_forward_2d_preserves_shape(self):
        t = TransformIdentity()
        x = torch.randn(3, 4, dtype=torch.float64)
        result = t(x)
        assert result.shape == x.shape
        assert torch.equal(result, x)

    def test_inverse_returns_input(self):
        t = TransformIdentity()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(t.inverse(x), x)

    def test_inverse_roundtrip(self):
        t = TransformIdentity()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(t.inverse(t(x)), x)

    def test_grad_returns_ones(self):
        t = TransformIdentity()
        x = torch.randn(5, dtype=torch.float64)
        expected = torch.ones_like(x)
        assert torch.equal(t.grad(x), expected)

    def test_grad_scalar(self):
        t = TransformIdentity()
        x = torch.tensor(5.0)
        assert torch.equal(t.grad(x), torch.tensor(1.0))

    def test_grad_matches_autograd(self):
        t = TransformIdentity()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformIdentity()
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(x).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformIdentity()
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(x).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformIdentity()
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformIdentity()
        x = torch.randn(5, requires_grad=True)
        assert t.inverse(x).requires_grad


class TestTransformChainWithIdentity:
    """Integration tests with TransformChain — identity as neutral element."""

    def test_identity_then_exp_equals_exp(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain(x), exp(x))

    def test_exp_then_identity_equals_exp(self):
        chain = TransformChain([TransformExp(), TransformIdentity()])
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain(x), exp(x))

    def test_identity_then_identity_is_identity(self):
        chain = TransformChain([TransformIdentity(), TransformIdentity()])
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(chain(x), x)

    def test_chain_with_identity_inverse_roundtrip(self):
        chain = TransformChain([TransformIdentity(), TransformExp(), TransformIdentity()])
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain.inverse(chain(x)), x)

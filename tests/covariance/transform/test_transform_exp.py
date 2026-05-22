import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformExp,
    TransformExp2,
    TransformExp10,
    TransformChain,
    TransformPow,
)


@pytest.mark.parametrize(
    "transform_cls, base",
    [
        (TransformExp, None),
        (TransformExp2, 2.0),
        (TransformExp10, 10.0),
    ],
)
class TestExponentialTransforms:
    """Shared test suite for all three exponential transform classes."""

    def test_constructor(self, transform_cls, base):
        t = transform_cls()
        assert isinstance(t, Transform)

    def test_forward_zero(self, transform_cls, base):
        t = transform_cls()
        result = t(torch.tensor(0.0))
        assert torch.allclose(result, torch.tensor(1.0))

    def test_forward_positive(self, transform_cls, base):
        t = transform_cls()
        x = torch.tensor([1.0, 2.0])
        expected = base**x if base is not None else torch.exp(x)
        assert torch.allclose(t(x), expected)

    def test_forward_shape_preserved(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    def test_forward_negative_input(self, transform_cls, base):
        t = transform_cls()
        x = torch.tensor([-1.0, -5.0, -10.0])
        result = t(x)
        assert (result > 0).all()
        assert (result < 1).all()

    def test_inverse_roundtrip(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_known_values(self, transform_cls, base):
        t = transform_cls()
        assert torch.allclose(t.inverse(torch.tensor(1.0)), torch.tensor(0.0))

    def test_inverse_non_positive(self, transform_cls, base):
        t = transform_cls()
        result_zero = t.inverse(torch.tensor(0.0))
        assert torch.isneginf(result_zero) or torch.isnan(result_zero)
        result_neg = t.inverse(torch.tensor(-1.0))
        assert torch.isnan(result_neg)

    def test_grad_matches_autograd(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_grad_positive(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(5, dtype=torch.float64)
        assert (t.grad(x) > 0).all()

    def test_dtype_float32(self, transform_cls, base):
        t = transform_cls()
        x = torch.tensor([1.0, -2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self, transform_cls, base):
        t = transform_cls()
        x = torch.tensor([1.0, -2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_domain_codomain_attributes(self, transform_cls, base):
        t = transform_cls()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ⁺"

    def test_repr(self, transform_cls, base):
        t = transform_cls()
        assert transform_cls.__name__ in repr(t)

    def test_forward_preserves_requires_grad(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(5, requires_grad=True) + 1.0  # positive to stay in domain
        assert t.inverse(x).requires_grad


class TestTransformExpSpecific:
    """Tests specific to the natural exponential transform."""

    def test_forward_match_torch_exp(self):
        t = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.exp(x))

    def test_grad_is_self(self):
        t = TransformExp()
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), t(x))


class TestTransformExp2Specific:
    """Tests specific to the base-2 exponential transform."""

    def test_forward_match_torch_exp2(self):
        t = TransformExp2()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.exp2(x))

    def test_grad_includes_ln2(self):
        t = TransformExp2()
        x = torch.randn(5, dtype=torch.float64)
        ln2 = torch.log(torch.tensor(2.0, dtype=x.dtype))
        expected = torch.exp2(x) * ln2
        assert torch.allclose(t.grad(x), expected)


class TestTransformExp10Specific:
    """Tests specific to the base-10 exponential transform."""

    def test_forward_known_values(self):
        t = TransformExp10()
        assert torch.allclose(t(torch.tensor(0.0)), torch.tensor(1.0))
        assert torch.allclose(t(torch.tensor(1.0)), torch.tensor(10.0))
        assert torch.allclose(t(torch.tensor(2.0)), torch.tensor(100.0))

    def test_inverse_known_values(self):
        t = TransformExp10()
        assert torch.allclose(t.inverse(torch.tensor(1.0)), torch.tensor(0.0))
        assert torch.allclose(t.inverse(torch.tensor(10.0)), torch.tensor(1.0))
        assert torch.allclose(t.inverse(torch.tensor(100.0)), torch.tensor(2.0))

    def test_grad_includes_ln10(self):
        t = TransformExp10()
        x = torch.randn(5, dtype=torch.float64)
        ln10 = torch.log(torch.tensor(10.0, dtype=x.dtype))
        expected = torch.pow(10.0, x) * ln10
        assert torch.allclose(t.grad(x), expected)


class TestTransformChainWithExp:
    """Integration tests with TransformChain."""

    def test_chain_exp_then_pow_forward(self):
        t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
        x = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(x) ** 2.0
        assert torch.allclose(t(x), expected)

    def test_chain_exp_then_pow_inverse(self):
        t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x, atol=1e-7)

    def test_chain_exp2_then_pow_grad(self):
        t = TransformChain([TransformExp2(), TransformPow(factor=3.0)])
        x = torch.randn(3, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestExponentialTransformsCUDA:
    """CUDA-specific tests for all three transform classes."""

    @pytest.mark.parametrize("transform_cls", [TransformExp, TransformExp2, TransformExp10])
    def test_forward_cpu_vs_cuda(self, transform_cls):
        t = transform_cls()
        x_cpu = torch.randn(10, dtype=torch.float64)
        x_cuda = x_cpu.to("cuda")
        assert torch.allclose(t(x_cpu), t(x_cuda).cpu())

    @pytest.mark.parametrize("transform_cls", [TransformExp, TransformExp2, TransformExp10])
    def test_inverse_cpu_vs_cuda(self, transform_cls):
        t = transform_cls()
        x_cpu = torch.randn(10, dtype=torch.float64).exp()  # positive values
        x_cuda = x_cpu.to("cuda")
        assert torch.allclose(t.inverse(x_cpu), t.inverse(x_cuda).cpu())

    @pytest.mark.parametrize("transform_cls", [TransformExp, TransformExp2, TransformExp10])
    def test_grad_cpu_vs_cuda(self, transform_cls):
        t = transform_cls()
        x_cpu = torch.randn(10, dtype=torch.float64)
        x_cuda = x_cpu.to("cuda")
        assert torch.allclose(t.grad(x_cpu), t.grad(x_cuda).cpu())

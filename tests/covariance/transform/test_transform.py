import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformChain,
    TransformIdentity,
    TransformExp,
    TransformPow,
    TransformScaleShift,
)


class TestTransformABC:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Transform()

    def test_concrete_subclass_instantiates(self):
        t = TransformIdentity()
        assert isinstance(t, Transform)

    def test_default_domain_codomain(self):
        class MinimalTransform(Transform):
            def __call__(self, x):
                return x

            def inverse(self, x):
                return x

            def grad(self, x):
                return torch.ones_like(x)

        t = MinimalTransform()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    def test_repr(self):
        assert repr(TransformIdentity()) == "TransformIdentity()"

    def test_str(self):
        s = str(TransformIdentity())
        assert "TransformIdentity" in s
        assert "ℝ" in s


class TestTransformChain:
    """Tests for the TransformChain composite."""

    def test_constructor_with_list(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        assert len(chain.chain) == 2

    def test_constructor_with_single_transform(self):
        chain = TransformChain(TransformIdentity())
        assert len(chain.chain) == 1
        assert isinstance(chain.chain, list)

    def test_constructor_raises_on_non_transform(self):
        with pytest.raises(TypeError):
            TransformChain([TransformIdentity(), "not_a_transform"])

    def test_call_single_is_equivalent(self):
        chain = TransformChain(TransformExp())
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain(x), exp(x))

    def test_call_applies_in_order(self):
        chain = TransformChain([TransformPow(factor=2.0), TransformScaleShift(a=2.0, b=1.0)])
        x = torch.randn(10, dtype=torch.float64)
        expected = 2.0 * (x ** 2) + 1.0
        assert torch.allclose(chain(x), expected)

    def test_call_shape_preserved(self):
        chain = TransformChain([TransformExp(), TransformPow(factor=3.0)])
        x = torch.randn(3, 4, dtype=torch.float64)
        assert chain(x).shape == x.shape

    def test_call_preserves_requires_grad(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        x = torch.randn(5, requires_grad=True)
        assert chain(x).requires_grad

    def test_inverse_applies_in_reverse_order(self):
        chain = TransformChain([TransformPow(factor=2.0), TransformExp()])
        x = torch.abs(torch.randn(10, dtype=torch.float64))
        assert torch.allclose(chain.inverse(chain(x)), x)

    def test_inverse_single_is_equivalent(self):
        chain = TransformChain(TransformExp())
        exp = TransformExp()
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(chain.inverse(x), exp.inverse(x))

    def test_grad_matches_autograd_two_transform(self):
        chain = TransformChain([TransformExp(), TransformPow(factor=2.0)])
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = chain.grad(x.detach())
        y = chain(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_grad_matches_autograd_three_transform(self):
        chain = TransformChain([
            TransformScaleShift(a=2.0, b=1.0),
            TransformExp(),
            TransformPow(factor=3.0),
        ])
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = chain.grad(x.detach())
        y = chain(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_grad_single_is_equivalent(self):
        chain = TransformChain(TransformExp())
        exp = TransformExp()
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(chain.grad(x), exp.grad(x))

    def test_repr(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        r = repr(chain)
        assert "TransformChain" in r
        assert "TransformIdentity" in r
        assert "TransformExp" in r

    def test_dtype_preserved(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert chain(x).dtype == torch.float64
        assert chain.inverse(chain(x)).dtype == torch.float64
        assert chain.grad(x).dtype == torch.float64

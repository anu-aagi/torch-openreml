from torch_openreml.covariance.operator import Operator
import torch


class HadamardProduct(Operator):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if len(self.operands) != 2:
            raise ValueError("Two operands are required")

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            v_groups = self.build_operands(params)

            a = v_groups[0]
            b = v_groups[1]
            v = a * b

            cache = {"a": a, "b": b, "v": v}

            self.set_intermediates(params, cache)

        return cache

    def __call__(self, params):
        cache = self._get_or_build_intermediates(params)
        v = cache["v"]
        self._shape = tuple(v.shape)

        return v

    def manual_grad(self, params):
        grad_groups, grad_name_groups = self.operands_grad(params)

        cache = self._get_or_build_intermediates(params)
        a = cache["a"]
        b = cache["b"]

        grad = []
        grad_names = []

        da = grad_groups[0]
        if da is not None:
            grad.append(da * b)
            grad_names.extend(grad_name_groups[0])

        db = grad_groups[1]
        if db is not None:
            grad.append(a * db)
            grad_names.extend(grad_name_groups[1])

        if len(grad) > 0:
            grad = torch.cat(grad)
            return grad, grad_names
        else:
            return None, []
from torch_openreml.covariance.operator import Operator
import torch


class LinearPropagation(Operator):

    def __init__(self, operands):
        if len(operands) != 2:
            raise ValueError("Two operands are required")

        super().__init__(None, operands)

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            v_groups = self.build_operands(params)

            z = v_groups[0]
            g = v_groups[1]
            v = z @ g @ z.T

            cache = {"z": z, "g": g, "v": v}

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
        z = cache["z"]
        g = cache["g"]

        grad_list = []
        grad_names = []

        dz = grad_groups[0]
        if dz is not None:
            grad_z = dz @ g @ z.T + z @ g @ dz.mT
            grad_list.append(grad_z)
            grad_names.extend(grad_name_groups[0])

        dg = grad_groups[1]
        if dg is not None:
            grad_g = z @ dg @ z.T
            grad_list.append(grad_g)
            grad_names.extend(grad_name_groups[1])

        if len(grad_list) > 0:
            grad = torch.cat(grad_list)
            return grad, grad_names
        else:
            return None, []
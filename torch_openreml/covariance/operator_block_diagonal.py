from torch_openreml.covariance.operator import Operator
import torch


class BlockDiagonal(Operator):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if len(self.operands) < 2:
            raise ValueError("At least two operands are required")

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            v_groups = self.build_operands(params)
            v = torch.block_diag(*v_groups)

            row_offsets = []
            col_offsets = []

            n = 0
            m = 0
            for vg in v_groups:
                rows, cols = vg.shape
                row_offsets.append((n, n + rows))
                col_offsets.append((m, m + cols))
                n += rows
                m += cols

            cache = {
                "v_groups": v_groups,
                "v": v,
                "row_offsets": row_offsets,
                "col_offsets": col_offsets
            }

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
        v = cache["v"]
        v_groups = cache["v_groups"]
        row_offsets = cache["row_offsets"]
        col_offsets = cache["col_offsets"]

        grad_list = []
        grad_names = []

        for i, grad in enumerate(grad_groups):
            if grad is None:
                continue

            (r0, r1) = row_offsets[i]
            (c0, c1) = col_offsets[i]

            tmp = torch.zeros((grad.shape[0],) + tuple(v.shape),
                              dtype=params.dtype,
                              device=params.device)

            tmp[:, r0:r1, c0:c1] = grad

            grad_list.append(tmp)
            grad_names.extend(grad_name_groups[i])

        if len(grad_list) > 0:
            grad = torch.cat(grad_list)
            return grad, grad_names
        else:
            return None, []
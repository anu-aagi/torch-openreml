from torch_openreml.covariance.operator import Operator
import torch

class HadamardProduct(Operator):

    def __init__(self, operands):
        if len(operands) != 2:
            raise ValueError("Two operands are required")

        super().__init__(None, operands)

    def build(self, params, grad=True):
        v_groups, grad_groups, grad_name_groups = self.build_operands(params, grad)

        a = v_groups[0]
        b = v_groups[1]
        v = a * b

        if grad:
            self.reset_grad()

            grad_list = []
            da = grad_groups[0]
            db = grad_groups[1]

            if da is not None:
                da = da * b
                grad_list.append(da)
                self._grad_names.extend(grad_name_groups[0])

            if db is not None:
                db = a * db
                grad_list.append(db)
                self._grad_names.extend(grad_name_groups[1])

            if len(grad_list) > 0:
                self._grad = torch.cat(grad_list)

        self._shape = tuple(v.shape)
        return v

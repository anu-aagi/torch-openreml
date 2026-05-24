import torch
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformIdentity


class Adapter(Matrix):

    _repr_single_line = False

    def __init__(self, adaptee, param_specs, param_map):

        super().__init__(adaptee.shape, param_specs)

        if not all([isinstance(trans, TransformIdentity) for trans in self.param_trans.values()]):
            raise ValueError("Adapter preprocessing does not support parameter transformations other than TransformIdentity!")

        self._adaptee = adaptee
        self._param_map = param_map

    def __call__(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults
        params = self.build_params(free_params)
        adaptee_free_params = self.param_map(params)
        return self.adaptee(adaptee_free_params)

    def auto_grad(self, free_params=None):
        self.adaptee.reset_intermediates()
        return super().auto_grad(free_params)

    def manual_grad(self, free_params=None):
        if free_params is None:
            free_params = self.free_param_defaults

        if self.num_free_params == 0:
            return None, []

        params = self.build_params(free_params)
        adaptee_free_params = self.param_map(params)
        adaptee_grad, _ = self.adaptee.grad(adaptee_free_params)

        jacobian = torch.func.jacrev(self.param_map)(params)

        grad = (jacobian[:, :, None, None] * adaptee_grad[:, None, :, :]).sum(dim=0)

        return grad, self.free_param_names

    @property
    def param_specs(self):
        return {
            param_name: {
                "fixed": param_spec["fixed"],
                "default": param_spec["default"],
                "trans": TransformIdentity()
            } for param_name, param_spec in self._param_specs.items()
        }

    @property
    def param_map(self):
        return self._param_map

    @property
    def adaptee(self):
        return self._adaptee

    @property
    def repr_dict(self):
        return {"param_specs": self.param_specs, "adaptee": self.adaptee, "param_map": "..."}
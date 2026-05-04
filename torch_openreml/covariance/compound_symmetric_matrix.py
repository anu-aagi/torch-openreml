from pandas.core.dtypes.cast import can_hold_element

from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformChain, TransformScaleShift, TransformSigmoid
import torch

class CompoundSymmetricMatrix(Matrix):
  
    def __init__(self, n, param_names=None, trans=None, no_grad_index=None):
        self.rho_min = -1/(n - 1)
        param_names = param_names or ["sigma^2", "rho"]
        trans = trans or [
            TransformExpPow2(),
            TransformChain([TransformSigmoid(), TransformScaleShift((1 - self.rho_min), self.rho_min)])
        ]
        super().__init__((n, n), param_names, trans, no_grad_index)

    def _get_or_build_intermediates(self, params):
        cache = self.get_intermediates(params)

        if cache is None:
            device, dtype = self.check_params(params)
            sigma2, rho = self.trans_params(params)

            i_n = torch.eye(self.shape[0], device=device, dtype=dtype)
            j_n = torch.ones((self.shape[0], self.shape[0]), device=device, dtype=dtype)
            rho_mat = ((1 - rho) * i_n + rho * j_n)

            cache = {"sigma2": sigma2, "i_n": i_n, "j_n": j_n, "rho_mat": rho_mat}
            self.set_intermediates(params, cache)

        return cache


    def __call__(self, params):
        cache = self._get_or_build_intermediates(params)
        v = cache["sigma2"] * cache["rho_mat"]

        return v

    def manual_grad(self, params):
        if len(self.no_grad_index) == self.num_params:
            return None, []

        cache = self._get_or_build_intermediates(params)

        grad = []
        grad_names = []

        trans_grad = self.trans_grad(params)

        if 0 not in self.no_grad_index:
            grad.append(trans_grad[0] * cache["rho_mat"])
            grad_names.append(self.param_names[0])

        if 1 not in self.no_grad_index:
            grad.append(cache["sigma2"] * (cache["j_n"] - cache["i_n"]) * trans_grad[1])
            grad_names.append(self.param_names[1])

        grad = torch.stack(grad)

        return grad, grad_names

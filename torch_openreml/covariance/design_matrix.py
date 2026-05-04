import torch
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.utils import numeric_to_design_matrix, categorical_to_design_matrix

class DesignMatrix(Matrix):

    def __init__(self, x, levels=None, drop_first=False, dtype=None, device=None):
        if not isinstance(x, (torch.Tensor, list, tuple)):
            raise TypeError("'x' must be a torch.Tensor, a list or a tuple!")

        if isinstance(x, torch.Tensor):
            self._matrix = numeric_to_design_matrix(x, levels, drop_first, dtype or x.dtype, device or x.device)
        elif isinstance(x[0], str):
            self._matrix = categorical_to_design_matrix(x, levels, drop_first, dtype, device)
        else:
            self._matrix = numeric_to_design_matrix(x, levels, drop_first, dtype, device)

        super().__init__((self._matrix.shape[0], self._matrix.shape[1]), [], [])

    def __call__(self, *args, **kwargs):
        return self._matrix

    @property
    def repr_dict(self):
        return {"shape": self._shape}

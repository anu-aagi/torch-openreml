from torch_openreml.covariance.matrix import Matrix
import torch

class IdentityMatrix(Matrix):

    def __init__(self, n, dtype=None, device=None):
        self._matrix = torch.eye(n, dtype=dtype, device=device)
        super().__init__((n, n), [], [])

    def build(self, *args, **kwargs):
        return self._matrix

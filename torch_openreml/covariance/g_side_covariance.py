from torch_openreml.covariance import DesignMatrix, LinearPropagation

class GSideCovariance(LinearPropagation):

    def __init__(self, x, g_builder, levels=None, drop_first=False, dtype=None, device=None):

        self._z = DesignMatrix(x, levels, drop_first, dtype, device)
        self._g = g_builder(self._z.shape[1])

        super().__init__({"z": self._z, "g": self._g})